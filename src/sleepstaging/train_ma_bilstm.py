import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

from .paths import PROCESSED, MODELS, RESULTS, CFG
from .dataset import SleepEDFNPZDataset
from .utils import set_seed
from .labels import CLASSES

# Paprastas MA-CNN-BiLSTM modelis:
class MACNNBiLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # CNN dalis
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # LSTM dalis
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        # Klasifikatorius
        self.fc = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (B, 1, 3000)
        feat = self.conv(x)  # -> (B, 64, ~750)
        # LSTM laukia (B, T, F)
        feat = feat.transpose(1, 2)  # (B, 750, 64)
        out, _ = self.lstm(feat)     # (B, 750, 256)
        # paimam paskutinį laiko išėjimą
        out = out[:, -1, :]          # (B, 256)
        return self.fc(out)

def train():
    set_seed(CFG["seed"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_ds = SleepEDFNPZDataset(split="train")
    val_ds   = SleepEDFNPZDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MACNNBiLSTM(n_classes=len(CLASSES)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0
    EPOCHS = 12

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for batch_i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item()

        if batch_i % 1 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] Batch {batch_i} Loss={loss.item():.4f}")

 # VALIDACIJA
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(device), yv.to(device)
            logits = model(Xv)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yv.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[Epoch {epoch}] Loss={total_loss:.4f} | Val Acc={acc:.4f} | Val F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        ckpt = MODELS / "best_bilstm.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"✅ BEST UPDATED – saved to {ckpt}")


if __name__ == "__main__":
    train()

