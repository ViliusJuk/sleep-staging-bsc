import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .dataset_seq import SleepEDFSequenceDataset
from .model_transformer import SleepTransformer
from .paths import MODELS, RESULTS
from .labels import CLASSES

def plot_metrics_bar(acc, f1, kappa, outpath):
    metrics = {"Accuracy": acc, "Macro-F1": f1, "Cohen's κ": kappa}
    plt.figure(figsize=(6, 4))
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.title("SleepTransformer Metrics")
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, min(v + 0.03, 0.98), f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, labels, outpath):
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    test_ds = SleepEDFSequenceDataset(split="test", seq_len=20)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SleepTransformer(n_classes=len(CLASSES)).to(device)
    ckpt = MODELS / "best_transformer.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = torch.argmax(model(X), dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Cohen's κ: {kappa:.4f}")

    plot_metrics_bar(acc, f1, kappa, RESULTS / "transformer_metrics.png")
    plot_confusion_matrix(cm, CLASSES, RESULTS / "transformer_confusion.png")

    summary = f"""
SleepTransformer – Test Results
-------------------------------
ACC={acc:.4f}
Macro-F1={f1:.4f}
Cohen's Kappa={kappa:.4f}

Notes:
- Sequence length = 20 epochs (last-step classification).
- Figures saved: transformer_metrics.png, transformer_confusion.png
- Model: models/best_transformer.pth
"""
    (RESULTS / "transformer_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

if __name__ == "__main__":
    main()

