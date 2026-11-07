"""Train FastSpeech 2.

Usage:
    python train.py config.yaml                    # start from scratch
    python train.py config.yaml --restore_step N   # continue from checkpoint N
    python train.py config.yaml --resume           # continue from the newest checkpoint
Losses are written to <log_path>/train_loss.csv and <log_path>/val_loss.csv
(use plot_losses.py to turn them into training_loss.png / validation_loss.png).
"""
import argparse
import csv
import glob
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.config import load_config
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_COLUMNS = ["step", "total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def latest_step(ckpt_dir):
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, "*.pth.tar")):
        m = re.match(r"(\d+)\.pth\.tar$", os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else 0


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOSS_COLUMNS)
        w.writerow(row)


def prune_checkpoints(ckpt_dir, keep_last, keep_every):
    steps = sorted(
        int(re.match(r"(\d+)", os.path.basename(f)).group(1))
        for f in glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
    )
    for s in steps[:-keep_last]:
        if keep_every and s % keep_every == 0:
            continue
        os.remove(os.path.join(ckpt_dir, "{}.pth.tar".format(s)))


def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config = configs

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    ckpt_dir = train_config["path"]["ckpt_path"]
    if args.resume:
        args.restore_step = latest_step(ckpt_dir)
        print("Resuming from step", args.restore_step)

    # Dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # larger than 1 enables sorting inside the Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", get_param_num(model))

    # Vocoder is only used to listen to samples in TensorBoard
    try:
        vocoder = get_vocoder(model_config, device)
    except FileNotFoundError as e:
        print("WARNING:", e, "\nTraining continues without audio samples.")
        vocoder = None

    # Loggers
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)
    except Exception as e:  # tensorboard missing: keep training anyway
        print("TensorBoard not available:", e)
        train_logger = val_logger = None
    train_csv = os.path.join(train_config["path"]["log_path"], "train_loss.csv")
    val_csv = os.path.join(train_config["path"]["log_path"], "val_loss.csv")

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    keep_last = train_config["step"].get("keep_last", 5)
    keep_every = train_config["step"].get("keep_every", 10000)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    done = step > total_step
    while not done:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                total_loss = losses[0] / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    vals = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *vals
                    )
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)
                    append_csv(train_csv, [step] + vals)
                    if train_logger is not None:
                        log(train_logger, step, losses=vals)

                if step % synth_step == 0 and train_logger is not None:
                    try:
                        fig, wav_rec, wav_pred, tag = synth_one_sample(
                            batch, output, vocoder, model_config, preprocess_config
                        )
                        log(train_logger, fig=fig, tag="Training/step_{}_{}".format(step, tag))
                        if wav_rec is not None:
                            log(train_logger, audio=wav_rec, sampling_rate=sampling_rate,
                                tag="Training/step_{}_{}_reconstructed".format(step, tag))
                            log(train_logger, audio=wav_pred, sampling_rate=sampling_rate,
                                tag="Training/step_{}_{}_synthesized".format(step, tag))
                    except Exception as e:
                        print("synth_one_sample failed (training continues):", e)

                if step % val_step == 0:
                    model.eval()
                    message, vals = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    append_csv(val_csv, [step] + vals)
                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(ckpt_dir, "{}.pth.tar".format(step)),
                    )
                    prune_checkpoints(ckpt_dir, keep_last, keep_every)

                if step == total_step:
                    done = True
                    break
                step += 1
                outer_bar.update(1)
            if done:
                break
            inner_bar.update(1)
        epoch += 1
    print("Training finished at step", step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config.yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the newest checkpoint in ckpt_path")
    args = parser.parse_args()

    configs = load_config(args.config)
    main(args, configs)
