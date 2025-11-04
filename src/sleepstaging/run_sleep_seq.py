# run_sleep_seq.py (naujas failas projekte)
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

from src.sleepstaging.paths import CFG, RESULTS, MODELS
from src.sleepstaging.utils import set_seed
from src.sleepstaging.dataset_seq import SleepEDFSeqDataset
from src.sleepstaging.model_seq import CNNGRU
from src.sleepstaging.labels import CLASSES

def plot_cm(cm, labels, outpath):
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

def main():
    set_seed(CFG["seed"]); RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = SleepEDFSeqDataset(split="test", window_size=7)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = CNNGRU(n_classes=len(CLASSES)).to(device)
    ckpt = MODELS / "best_cnngru.pth"
    assert ckpt.exists(), f"Nerastas checkpoint: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy()); y_pred.extend(preds)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
    kap = cohen_kappa_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    print(f"ACC={acc:.4f}  Macro-F1={f1:.4f}  Kappa={kap:.4f}")

    plot_cm(cm, CLASSES, RESULTS / "seq_confusion_matrix.png")

    prec, rec, f1c, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0.0)
    import csv
    with open(RESULTS / "seq_per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class","precision","recall","f1","support"])
        for i, name in enumerate(CLASSES):
            w.writerow([name, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1c[i]:.4f}", int(sup[i])])
    np.savetxt(RESULTS / "seq_confusion_matrix_raw.csv", cm, fmt="%d", delimiter=",")

if __name__ == "__main__":
    main()
