import time, csv
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .paths import CFG, RESULTS, MODELS
from .utils import set_seed
from .labels import CLASSES
from .dataset import SleepEDFNPZDataset
from .model_baseline import CNN1DBaseline

def make_loader(split, batch_size=128, shuffle=False, weighted=False):
    ds = SleepEDFNPZDataset(split=split)
    if weighted:
        # klasių svoriai pagal dažnius (rečiau pasitaikančioms didesnis svoris)
        counts = np.bincount(ds.y, minlength=len(CLASSES)).astype(np.float64)
        probs = counts / counts.sum()
        class_weights = (1.0 / (probs + 1e-9))
        sample_weights = class_weights[ds.y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        return ds, loader, torch.tensor(class_weights, dtype=torch.float32)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return ds, loader, None

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
    kap = cohen_kappa_score(y_true, y_pred)
    return acc, f1, kap

def train():
    set_seed(CFG["seed"])
    RESULTS.mkdir(exist_ok=True, parents=True)
    MODELS.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds, train_loader, class_w = make_loader("train", batch_size=128, weighted=True)
    val_ds,   val_loader,   _       = make_loader("val",   batch_size=256, shuffle=False)
    n_classes = len(CLASSES)

    # Model
    model = CNN1DBaseline(n_classes=n_classes).to(device)

    # Loss (su class weights)
    if class_w is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_w.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Optimizer / scheduler
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1 = -1.0
    patience = 8
    wait = 0

    # log
    log_path = RESULTS / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_acc","val_macro_f1","val_kappa","lr"])

    max_epochs = 30
    for epoch in range(1, max_epochs+1):
        model.train()
        t0 = time.time()
        run_loss, n_batches = 0.0, 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            run_loss += loss.item()
            n_batches += 1

        train_loss = run_loss / max(1, n_batches)
        val_acc, val_f1, val_kap = eval_epoch(model, val_loader, device)
        sched.step(val_f1)

        # log
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{val_acc:.4f}", f"{val_f1:.4f}", f"{val_kap:.4f}", opt.param_groups[0]["lr"]])

        # early stopping & checkpoint
        improved = val_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = val_f1
            wait = 0
            ckpt_path = MODELS / "best_cnn1d.pth"
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1

        dur = time.time() - t0
        print(f"[{epoch:02d}/{max_epochs}] loss={train_loss:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_kappa={val_kap:.4f} | lr={opt.param_groups[0]['lr']:.1e} | {dur:.1f}s")

        if wait >= patience:
            print(f"Early stopping (no val_f1 improve {patience} epochs). Best val_f1={best_f1:.4f}")
            break

    # paprasta kreivių vizualizacija
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.read_csv(log_path)
        plt.figure(figsize=(7,4))
        plt.plot(df["epoch"], df["train_loss"], label="train loss")
        plt.plot(df["epoch"], df["val_macro_f1"], label="val macro-F1")
        plt.plot(df["epoch"], df["val_acc"], label="val acc")
        plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS / "training_curves.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Plot fail:", e)

if __name__ == "__main__":
    train()
