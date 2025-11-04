"""
run_sleep_staging.py
EEG sleep stage classification – testing & analysis pipeline (real NPZ)
Author: Domantas Bitaitis
Description:
  - Įkrauna test duomenis iš data/processed/*.npz (pagal config.yaml)
  - Įkelia treniruotą modelį (jei yra models/best_cnn1d.pth)
  - Skaičiuoja Accuracy, Macro-F1, Cohen's κ
  - Išsaugo metrikų bar chart ir Confusion Matrix heatmap
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt

from src.sleepstaging.paths import CFG, RESULTS, MODELS
from src.sleepstaging.utils import set_seed
from src.sleepstaging.dataset import SleepEDFNPZDataset
from src.sleepstaging.model_baseline import CNN1DBaseline
from src.sleepstaging.labels import CLASSES

def plot_metrics_bar(acc, f1, kappa, outpath):
    metrics = {"Accuracy": acc, "Macro-F1": f1, "Cohen's κ": kappa}
    plt.figure(figsize=(6, 4))
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.title("Sleep Stage Classification Metrics")
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, min(v + 0.03, 0.98), f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, labels, outpath):
    # eilutėmis normalizuota (pagal true klasę)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # užrašome ne-normalizuotas reikšmes langeliuose
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    set_seed(CFG["seed"])
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1) Test duomenys iš .npz (pagal config.yaml -> split.test_subjects)
    test_ds = SleepEDFNPZDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # 2) Modelis (bandome įkelti checkpoint, jei yra)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DBaseline(n_classes=len(CLASSES)).to(device)

    ckpt = MODELS / "best_cnn1d.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[INFO] Loaded checkpoint: {ckpt}")
    else:
        print("[WARN] Checkpoint not found. Using randomly initialized model (metrics will be poor).")

    model.eval()

    # 3) Inference
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4) Metrikos
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Cohen's κ: {kappa:.4f}")

    # 5) Grafikai
    plot_metrics_bar(acc, f1, kappa, RESULTS / "metrics_bar_chart.png")
    plot_confusion_matrix(cm, CLASSES, RESULTS / "confusion_matrix.png")

    # 6) Tekstinė santrauka
    used_ckpt = str(ckpt) if ckpt.exists() else "NO_CHECKPOINT (random weights)"
    summary = f"""
Sleep Stage Classification – Test Results
-----------------------------------------
Subjects (test): {CFG['split'].get('test_subjects', [])}
Checkpoint: {used_ckpt}

Accuracy: {acc:.4f}
Macro-F1: {f1:.4f}
Cohen's Kappa: {kappa:.4f}

Notes:
- Macro-F1 vertina vidurkį per klases ir yra atsparus disbalansui.
- Cohen's κ rodo sutapimą virš atsitiktinumo (≈0.6 „geras“, ≈0.8 „labai geras“).
- Confusion Matrix išsaugota kaip 'results/confusion_matrix.png'.
"""
    (RESULTS / "test_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

if __name__ == "__main__":
    main()
