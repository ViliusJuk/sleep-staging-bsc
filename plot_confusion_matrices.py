#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["W", "N1", "N2", "N3", "REM"]

def plot_cm(cm: np.ndarray, title: str, outpath: Path, normalize: bool):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)

    # numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize:
                txt = f"{val:.2f}"
            else:
                txt = str(int(val))
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    report_dir = Path("test_reports_pp")
    out_dir = report_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("cnn_raw", "CNN RAW"),
        ("cnn_bias_smooth", "CNN BIAS+SMOOTH"),
        ("bilstm_raw", "BiLSTM RAW"),
        ("bilstm_bias_smooth", "BiLSTM BIAS+SMOOTH"),
        ("transformer_raw", "Transformer RAW"),
        ("transformer_bias_smooth", "Transformer BIAS+SMOOTH"),
    ]

    for prefix, title in specs:
        cm_path = report_dir / f"{prefix}_cm.npy"
        cmn_path = report_dir / f"{prefix}_cm_norm.npy"
        if not cm_path.exists() or not cmn_path.exists():
            print(f"SKIP (missing): {prefix}")
            continue

        cm = np.load(cm_path)
        cmn = np.load(cmn_path)

        plot_cm(cm, f"{title} (counts)", out_dir / f"{prefix}_counts.png", normalize=False)
        plot_cm(cmn, f"{title} (normalized)", out_dir / f"{prefix}_norm.png", normalize=True)

        print("Saved:", out_dir / f"{prefix}_counts.png")
        print("Saved:", out_dir / f"{prefix}_norm.png")

if __name__ == "__main__":
    main()

