"""Generate the test sentences and the variance-adaptor plots.

Usage (run inside English/ or Telugu/):
    python inference.py config.yaml --restore_step 30000 --out_dir ../Results/English

Writes
    <out_dir>/test_generated_audios/<ID>.wav
    <out_dir>/variance_adaptor_plots/duration/<ID>.png
    <out_dir>/variance_adaptor_plots/pitch/<ID>.png
    <out_dir>/variance_adaptor_plots/energy/<ID>.png
    <out_dir>/variance_adaptor_values.json   (numbers behind the plots, for the report)
"""
import argparse
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

from frontend import text_to_phones
from text import _id_to_symbol
from utils.config import load_config
from utils.model import get_model, get_vocoder, vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_speaker(preprocess_config, requested):
    """Returns (name, id). 'auto' = speaker with the most training utterances."""
    pre = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(pre, "speakers.json")) as f:
        speakers = json.load(f)
    if requested and requested != "auto":
        return requested, speakers[requested]
    with open(os.path.join(pre, "train.txt"), encoding="utf-8") as f:
        counts = Counter(line.split("|")[1] for line in f if line.strip())
    name = counts.most_common(1)[0][0]
    return name, speakers[name]


def _bar_labels(ax, labels):
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlim(-0.8, n - 0.2)


def plot_duration(sid, labels, frames, ms, path, speaker):
    fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(labels)), 4))
    colors = ["#bbbbbb" if l in ("sp", "spn", "sil") else "#3b6fb6" for l in labels]
    ax.bar(range(len(labels)), ms, color=colors)
    ax.set_ylabel("Predicted duration (ms)")
    ax.set_title("{} - predicted phoneme duration (speaker {}, grey = pause)".format(sid, speaker))
    _bar_labels(ax, labels)
    sec = ax.secondary_yaxis("right", functions=(lambda x: x / (ms.sum() / max(frames.sum(), 1)),
                                                 lambda x: x * (ms.sum() / max(frames.sum(), 1))))
    sec.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_contour(sid, kind, ylabel, color, labels, per_phone, frames, path, speaker):
    """Top: one value per phoneme. Bottom: same values stretched to frame level."""
    n = len(labels)
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(max(10, 0.28 * n), 6.5), gridspec_kw={"height_ratios": [1, 1]}
    )
    a1.plot(range(n), per_phone, marker="o", ms=3, color=color)
    a1.set_ylabel(ylabel + " per phoneme")
    a1.set_title("{} - predicted {} (speaker {})".format(sid, kind, speaker))
    _bar_labels(a1, labels)
    a1.grid(alpha=0.3)
    expanded = np.repeat(per_phone, np.maximum(frames.astype(int), 0))
    a2.plot(expanded, color=color)
    a2.set_ylabel(ylabel + " per frame")
    a2.set_xlabel("mel frame")
    a2.set_xlim(0, max(len(expanded), 1))
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config.yaml")
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True, help="e.g. ../Results/English")
    parser.add_argument("--sentences", type=str, default="test_sentences.json")
    parser.add_argument("--speaker", type=str, default=None,
                        help="speaker name from speakers.json, or 'auto'")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    configs = load_config(args.config)
    preprocess_config, model_config, train_config = configs
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    speaker_name, speaker_id = pick_speaker(
        preprocess_config, args.speaker or preprocess_config.get("inference_speaker", "auto")
    )
    print("Speaker: {} (id {})".format(speaker_name, speaker_id))

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    with open(args.sentences, encoding="utf-8") as f:
        sentences = json.load(f)

    audio_dir = os.path.join(args.out_dir, "test_generated_audios")
    plot_root = os.path.join(args.out_dir, "variance_adaptor_plots")
    for sub in ["duration", "pitch", "energy"]:
        os.makedirs(os.path.join(plot_root, sub), exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    all_values = {}
    for item in sentences:
        sid, text = item["id"], item["text"]
        phones, seq = text_to_phones(text, preprocess_config)
        labels = [_id_to_symbol[i].lstrip("@") for i in seq]

        texts = torch.from_numpy(np.array([seq])).long().to(device)
        lens = torch.tensor([len(seq)]).to(device)
        spk = torch.tensor([speaker_id]).long().to(device)
        with torch.no_grad():
            out = model(
                spk, texts, lens, len(seq),
                p_control=args.pitch_control,
                e_control=args.energy_control,
                d_control=args.duration_control,
            )
            mel, p_pred, e_pred, log_d, d_round, mel_len = out[1], out[2], out[3], out[4], out[5], out[9]
            wav = vocoder_infer(
                mel.transpose(1, 2), vocoder, model_config, preprocess_config,
                lengths=mel_len * hop,
            )[0]

        wavfile.write(os.path.join(audio_dir, "{}.wav".format(sid)), sr, wav)

        frames = d_round[0].cpu().numpy().astype(int)
        ms = frames * hop / sr * 1000.0
        pitch_hz = p_pred[0].cpu().numpy() * p_std + p_mean
        energy = e_pred[0].cpu().numpy() * e_std + e_mean

        plot_duration(sid, labels, frames, ms, os.path.join(plot_root, "duration", sid + ".png"), speaker_name)
        plot_contour(sid, "pitch (F0)", "F0 (Hz)", "tomato", labels, pitch_hz, frames,
                     os.path.join(plot_root, "pitch", sid + ".png"), speaker_name)
        plot_contour(sid, "energy", "Energy", "darkviolet", labels, energy, frames,
                     os.path.join(plot_root, "energy", sid + ".png"), speaker_name)

        all_values[sid] = {
            "text": text,
            "phonemes": labels,
            "duration_frames": frames.tolist(),
            "duration_ms": [round(float(x), 1) for x in ms],
            "pitch_hz": [round(float(x), 1) for x in pitch_hz],
            "energy": [round(float(x), 3) for x in energy],
            "total_duration_s": round(float(ms.sum()) / 1000.0, 2),
        }
        print("{}: {} phonemes, {:.2f} s".format(sid, len(labels), ms.sum() / 1000.0))

    with open(os.path.join(args.out_dir, "variance_adaptor_values.json"), "w", encoding="utf-8") as f:
        json.dump({"speaker": speaker_name, "step": args.restore_step, "sentences": all_values},
                  f, ensure_ascii=False, indent=1)
    print("Done. Results in", args.out_dir)


if __name__ == "__main__":
    main()
