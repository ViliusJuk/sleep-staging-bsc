import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from .dataset import SleepEDFNPZDataset
from .labels import CLASSES
from .model_baseline_v0 import CNN1DBaselineV0
from .paths import MODELS

def zscore(x, eps=1e-6):
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== DEVICE:", device, "===")

    val_ds = SleepEDFNPZDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    # --- kraunam SENĄ modelį ---
    model = CNN1DBaselineV0(n_classes=len(CLASSES)).to(device)
    state = torch.load(MODELS / "best_cnn1d.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X = zscore(X.to(device, dtype=torch.float32))
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    print("Per-klasinė metrika:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()

