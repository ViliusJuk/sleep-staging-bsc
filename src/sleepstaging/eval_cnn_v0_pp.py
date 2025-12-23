import torch, numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import SleepEDFNPZDataset
from .labels import CLASSES
from .model_baseline_v0 import CNN1DBaselineV0
from .paths import MODELS
from .postprocess import smooth_mode, apply_class_biases

def zscore(x, eps=1e-6):
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== DEVICE:", device, "===")

    val_ds = SleepEDFNPZDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = CNN1DBaselineV0(n_classes=len(CLASSES)).to(device)
    state = torch.load(MODELS / "best_cnn1d.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    y_true = []
    logits_all = []

    with torch.no_grad():
        for X, y in val_loader:
            X = zscore(X.to(device, dtype=torch.float32))
            L = model(X).cpu().numpy()          # RAW logits
            logits_all.append(L)
            y_true.extend(y.numpy())

    logits = np.vstack(logits_all)
    y_true = np.array(y_true)

    # ---- 1) raw ----
    y_pred_raw = logits.argmax(1)
    f1_raw = f1_score(y_true, y_pred_raw, average="macro", zero_division=0)
    print("Macro-F1 RAW:", round(f1_raw, 3))

    # ---- 2) bias -> argmax ----
    # iš tavo grid paieškos: N1=-0.9, N3=-1.0, kitos 0
    biases = np.array([0.0, -0.9, 0.0, -1.0, 0.0], dtype=np.float32)
    y_pred_bias = apply_class_biases(logits, biases).argmax(1)
    f1_bias = f1_score(y_true, y_pred_bias, average="macro", zero_division=0)
    print("Macro-F1 with BIAS:", round(f1_bias, 3))

    # ---- 3) bias + smoothing(k=5) ----
    y_pred_bias_s = smooth_mode(y_pred_bias, k=5, n_classes=len(CLASSES))
    f1_bias_s = f1_score(y_true, y_pred_bias_s, average="macro", zero_division=0)
    print("Macro-F1 with BIAS+SMOOTH:", round(f1_bias_s, 3))

    # pilnas reportas su bias+smooth
    print("\nPer-klasinė metrika (BIAS+SMOOTH):")
    print(classification_report(y_true, y_pred_bias_s, target_names=CLASSES, digits=3))
    print("Confusion matrix (BIAS+SMOOTH):")
    print(confusion_matrix(y_true, y_pred_bias_s))

if __name__ == "__main__":
    main()

