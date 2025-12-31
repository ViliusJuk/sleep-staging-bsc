import os
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from .dataset import SleepEDFNPZDataset
from .labels import CLASSES
from .model_transformer import SleepTransformer
from .paths import MODELS


def zscore(x, eps=1e-6):
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)


def smooth_mode(y_pred: np.ndarray, k: int = 5) -> np.ndarray:
    y = np.asarray(y_pred)
    n = len(y)
    h = k // 2
    out = np.empty(n, dtype=y.dtype)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        w = y[a:b]
        out[i] = np.bincount(w, minlength=len(CLASSES)).argmax()
    return out


def softmax_np(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    """Stabili softmax numpy'je."""
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def plot_roc_ovr(y_true: np.ndarray, y_proba: np.ndarray, out_path: str, title: str):
    """
    Multi-class ROC (One-vs-Rest): kiekvienai klasei ROC vs rest.
    """
    n_classes = y_proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(6, 5))
    aucs = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        name = CLASSES[i] if i < len(CLASSES) else str(i)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return aucs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== DEVICE:", device, "===")

    ds = SleepEDFNPZDataset(split="val")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    model = SleepTransformer(n_classes=len(CLASSES)).to(device, dtype=torch.float32)
    state = torch.load(MODELS / "best_transformer.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    logits_list, y_true_list = [], []
    with torch.no_grad():
        for X, y in loader:
            X = zscore(X.to(device, dtype=torch.float32))
            L = model(X).cpu().numpy()  # (B,5) raw logits
            logits_list.append(L)
            y_true_list.append(y.numpy())

    logits = np.vstack(logits_list)                 # (N,5)
    y_true = np.concatenate(y_true_list, axis=0)    # (N,)

    # =========================
    # ROC (RAW softmax probs)
    # =========================
    y_proba = softmax_np(logits, axis=1)  # (N,5)

    os.makedirs("results", exist_ok=True)
    np.save("results/transformer_val_y.npy", y_true)
    np.save("results/transformer_val_proba.npy", y_proba)

    aucs = plot_roc_ovr(
        y_true=y_true,
        y_proba=y_proba,
        out_path="figures/roc_transformer_ovr.png",
        title="Transformer ROC (One-vs-Rest) [VAL]",
    )
    print("[ROC] Saved: figures/roc_transformer_ovr.png | AUCs:", [float(a) for a in aucs])

    # =========================
    # RAW metrics (argmax)
    # =========================
    y_raw = logits.argmax(1)
    acc_raw = accuracy_score(y_true, y_raw)
    f1_raw  = f1_score(y_true, y_raw, average="macro", zero_division=0)
    kap_raw = cohen_kappa_score(y_true, y_raw)
    print(f"RAW: acc={acc_raw:.3f} | macro-F1={f1_raw:.3f} | kappa={kap_raw:.3f}")

    # =========================
    # BIAS grid N1/N3 (postproc)
    # =========================
    best = (None, -1e9)
    for b1, b3 in itertools.product(np.linspace(0.0, -1.0, 11), np.linspace(0.0, -1.0, 11)):
        bias = np.array([0, b1, 0, b3, 0], dtype=np.float32)
        y_b = (logits + bias).argmax(1)
        f1 = f1_score(y_true, y_b, average="macro", zero_division=0)
        if f1 > best[1]:
            best = ((b1, b3), f1)

    (b1, b3), f1_best = best
    print(f"Best bias (N1,N3)=({b1:.2f},{b3:.2f}) | macro-F1={f1_best:.3f}")

    # =========================
    # BIAS + SMOOTH final metrics
    # =========================
    bias_vec = np.array([0, b1, 0, b3, 0], dtype=np.float32)
    y_bias   = (logits + bias_vec).argmax(1)
    y_bs     = smooth_mode(y_bias, k=5)

    acc = accuracy_score(y_true, y_bs)
    f1  = f1_score(y_true, y_bs, average="macro", zero_division=0)
    kap = cohen_kappa_score(y_true, y_bs)

    print(f"BIAS+SMOOTH: acc={acc:.3f} | macro-F1={f1:.3f} | kappa={kap:.3f}")
    print("\nPer-klasinė metrika (BIAS+SMOOTH):")
    print(classification_report(y_true, y_bs, target_names=CLASSES, digits=3))
    print("Confusion matrix (BIAS+SMOOTH):")
    print(confusion_matrix(y_true, y_bs))


if __name__ == "__main__":
    main()

