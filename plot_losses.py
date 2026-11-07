"""Turn the CSV logs written by train.py into training_loss.png and validation_loss.png.

Usage:
    python plot_losses.py --log_dir English/output/log --out_dir Results/English
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLS = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def smooth(s, w):
    return s.rolling(w, min_periods=1).mean() if w > 1 else s


def plot(df, title, path, window=1):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(df["step"], smooth(df["total"], window), color="black")
    axes[0].set_title(title + " - total loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    for c in COLS[1:]:
        axes[1].plot(df["step"], smooth(df[c], window), label=c)
    axes[1].set_title(title + " - loss components")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_smooth", type=int, default=10, help="moving-average window for training curve")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tr = pd.read_csv(os.path.join(args.log_dir, "train_loss.csv")).drop_duplicates("step", keep="last")
    va = pd.read_csv(os.path.join(args.log_dir, "val_loss.csv")).drop_duplicates("step", keep="last")
    plot(tr, "Training", os.path.join(args.out_dir, "training_loss.png"), args.train_smooth)
    plot(va, "Validation", os.path.join(args.out_dir, "validation_loss.png"))

    best = va.loc[va["total"].idxmin()]
    print("Final train total loss : {:.4f} (step {})".format(tr["total"].iloc[-1], int(tr["step"].iloc[-1])))
    print("Final val total loss   : {:.4f} (step {})".format(va["total"].iloc[-1], int(va["step"].iloc[-1])))
    print("Best val total loss    : {:.4f} at step {}".format(best["total"], int(best["step"])))
    print("Saved plots to", args.out_dir)
