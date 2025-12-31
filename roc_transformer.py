#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.sleepstaging.dataset import SleepEDFNPZDataset
from src.sleepstaging.labels import CLASSES
from src.sleepstaging.paths import MODELS
from src.sleepstaging.model_transformer import SleepTransformer

def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)

@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs_list = []
    y_true = []
    for X, y in loader:
        X = zscore(X.to(device, dtype=torch.float32))
        logits = model(X)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs_list.append(probs)
        y_true.extend(y.numpy())
    return np.vstack(probs_list), np.asarray(y_true)

def plot_roc_ovr(probs: np.ndarray, y_true: np.ndarray, outpath: Path, title: str):
    n_classes = probs.shape[1]
    Y = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = {}

    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, c], probs[:, c])
        roc_auc = auc(fpr, tpr)
        aucs[CLASSES[c]] = roc_auc
        ax.plot(fpr, tpr, label=f"{CLASSES[c]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    # save AUCs
    (outpath.with_suffix(".auc.txt")).write_text(
        "\n".join([f"{k}: {v:.6f}" for k, v in aucs.items()]) + "\n",
        encoding="utf-8"
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    out_dir = Path("test_reports_pp") / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ds = SleepEDFNPZDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    model = SleepTransformer(n_classes=len(CLASSES)).to(device, dtype=torch.float32)
    state = torch.load(MODELS / "best_transformer.pth", map_location=device)
    model.load_state_dict(state, strict=True)

    probs, y_true = collect_probs(model, test_loader, device)
    plot_roc_ovr(probs, y_true, out_dir / "transformer_roc_ovr.png", "Transformer ROC (One-vs-Rest) — TEST")

    print("Saved:", out_dir / "transformer_roc_ovr.png")
    print("Saved:", out_dir / "transformer_roc_ovr.auc.txt")

if __name__ == "__main__":
    main()

