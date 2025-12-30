# src/sleepstaging/train_bilstm_cv.py
"""
Subject-wise k-fold cross-validation for BiLSTM model.

Run (example):
    python -m src.sleepstaging.train_bilstm_cv --k 10 --fold 3

This script mirrors train_cnn_cv.py logic, but trains the BiLSTM model.
It saves per-fold results to:
    results/cv_bilstm_fold{fold}.json
and the best model weights to:
    models/best_bilstm_fold{fold}.pth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .paths import MODELS, RESULTS, CFG
from .labels import CLASSES
from .utils import set_seed
from .cv_splits import get_fold_subjects


# --------------------------
# helpers
# --------------------------
def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-epoch z-score normalization over time axis T: (B, 1, T) or (B, L, 1, T)."""
    if x.ndim == 3:
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True)
        return (x - m) / (s + eps)
    if x.ndim == 4:
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True)
        return (x - m) / (s + eps)
    return x


def _import_bilstm_model():
    """
    Tries to import your BiLSTM model class from common names.
    Adjust this if your class name differs.
    """
    # Most likely in your project: src/sleepstaging/model_bilstm.py
    from . import model_bilstm as mb

    candidates = [
        "CNNBiLSTM",         # common
        "SleepBiLSTM",       # common
        "BiLSTMModel",       # generic
        "SleepStagingBiLSTM", # generic
	"SleepEDFSequenceDataset",
    ]
    for name in candidates:
        if hasattr(mb, name):
            return getattr(mb, name)

    # Fallback: if your file exports exactly one nn.Module subclass
    for v in mb.__dict__.values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            return v

    raise ImportError(
        "Neradau BiLSTM modelio klasės src/sleepstaging/model_bilstm.py.\n"
        "Pataisyk _import_bilstm_model() ir įrašyk tikslų klasės pavadinimą."
    )


def _import_seq_dataset():
    """
    Tries to import a sequence dataset from dataset_seq.py.
    Your BiLSTM should train on sequences/windows (not independent epochs).
    """
    from . import dataset_seq as dsq

    candidates = [
        "SleepEDFSeqDataset",
        "SleepEDFNPZSeqDataset",
        "SleepEDFNPZSequenceDataset",
        "SleepEDFWindowDataset",
        "SleepEDFSeqWindowDataset",
    ]
    for name in candidates:
        if hasattr(dsq, name):
            return getattr(dsq, name)

    # Fallback: any Dataset subclass in dataset_seq module
    from torch.utils.data import Dataset
    for v in dsq.__dict__.values():
        if isinstance(v, type) and issubclass(v, Dataset) and v is not Dataset:
            return v

    raise ImportError(
        "Neradau sequence dataset klasės src/sleepstaging/dataset_seq.py.\n"
        "Atidaryk dataset_seq.py ir pažiūrėk klasės pavadinimą, tada įrašyk jį į candidates."
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """
    Returns: (acc, macro_f1, kappa) on flattened epoch-level predictions.
    Assumes loader yields either:
      - X: (B, L, 1, T), y: (B, L)
    or:
      - X: (B, 1, T), y: (B,)   (fallback)
    Model output assumed:
      - logits: (B, L, C) OR (B, C)
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    model.eval()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for X, y in loader:
        X = X.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)

        X = zscore(X)

        logits = model(X)

        # Sequence case: (B, L, C)
        if logits.ndim == 3 and y.ndim == 2:
            pred = torch.argmax(logits, dim=-1)  # (B, L)
            y_true_all.extend(y.reshape(-1).detach().cpu().numpy().tolist())
            y_pred_all.extend(pred.reshape(-1).detach().cpu().numpy().tolist())
        else:
            # Epoch case: (B, C)
            pred = torch.argmax(logits, dim=-1)  # (B,)
            y_true_all.extend(y.detach().cpu().numpy().tolist())
            y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    if y_true.size == 0:
        return 0.0, 0.0, 0.0

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, f1, kappa


def run_fold(k: int, fold: int) -> None:
    set_seed(CFG["seed"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # --- split subjects ---
    train_subjects, val_subjects, test_subjects = get_fold_subjects(
        k=k, fold=fold, seed=CFG["seed"]
    )
    print(f"[CV] k={k} fold={fold}")
    print(f"[CV] train={len(train_subjects)} | val={len(val_subjects)} | test={len(test_subjects)}")

    # --- dataset/model imports ---
    SeqDataset = _import_seq_dataset()
    BiLSTMCls = _import_bilstm_model()

    # --- DATA ---
    # IMPORTANT:
    # Your SeqDataset signature might differ. Most common patterns:
    #   SeqDataset(split="train", subjects=[...], seq_len=..., stride=...)
    # or SeqDataset(subjects=[...], seq_len=..., stride=...)
    #
    # If this crashes, open dataset_seq.py and adjust these constructors.
    ds_kwargs = {}
    # Try to pass seq_len/stride from config if you have them
    if isinstance(CFG, dict):
        seq_cfg = CFG.get("seq", {}) if "seq" in CFG else {}
        if "seq_len" in seq_cfg:
            ds_kwargs["seq_len"] = seq_cfg["seq_len"]
        if "stride" in seq_cfg:
            ds_kwargs["stride"] = seq_cfg["stride"]

    try:
        train_ds = SeqDataset(subjects=train_subjects, **ds_kwargs)
        val_ds   = SeqDataset(subjects=val_subjects, **ds_kwargs)
        test_ds  = SeqDataset(subjects=test_subjects, **ds_kwargs)
    except TypeError:
        # fallback if dataset expects split=...
        train_ds = SeqDataset(split="train", subjects=train_subjects, **ds_kwargs)
        val_ds   = SeqDataset(split="val",   subjects=val_subjects,   **ds_kwargs)
        test_ds  = SeqDataset(split="test",  subjects=test_subjects,  **ds_kwargs)

    # loaders
    train_loader = DataLoader(train_ds, batch_size=CFG.get("batch_size_bilstm", 64), shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.get("batch_size_bilstm_eval", 128), shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=CFG.get("batch_size_bilstm_eval", 128), shuffle=False,
                              num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DEVICE: {device} ===")

    # --- MODEL ---
    # Common constructor patterns:
    #   BiLSTMCls(n_classes=5)
    #   BiLSTMCls(n_classes=5, ...)
    # Adjust if your model expects different args.
    try:
        model = BiLSTMCls(n_classes=len(CLASSES)).to(device=device, dtype=torch.float32)
    except TypeError:
        model = BiLSTMCls(n_classes=len(CLASSES), num_classes=len(CLASSES)).to(device=device, dtype=torch.float32)

    # --- LOSS / OPT ---
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05) if "label_smoothing" in nn.CrossEntropyLoss.__init__.__code__.co_varnames else nn.CrossEntropyLoss()

    lr = float(CFG.get("lr_bilstm", 5e-4))
    wd = float(CFG.get("weight_decay_bilstm", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    # --- TRAIN LOOP ---
    best_f1 = -1.0
    patience = int(CFG.get("patience_bilstm", 10))
    no_improve = 0
    epochs = int(CFG.get("epochs_bilstm", 30))

    t0 = time.time()
    print("=== START TRAIN (BiLSTM CV fold) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for bi, (X, y) in enumerate(train_loader):
            X = X.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)

            X = zscore(X)

            opt.zero_grad(set_to_none=True)
            logits = model(X)

            # Sequence case: logits (B,L,C), y (B,L)
            if logits.ndim == 3 and y.ndim == 2:
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=-1)
                    acc_b = (pred == y).float().mean().item()
            else:
                loss = loss_fn(logits, y)
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=-1)
                    acc_b = (pred == y).float().mean().item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().item())
            if bi % 10 == 0:
                print(f"[F{fold}] [E{epoch:02d}] batch {bi:04d} | loss={loss.item():.4f} | acc={acc_b:.3f}")

        val_acc, val_f1, val_kap = evaluate(model, val_loader, device)
        sched.step(val_f1)

        print(
            f"[F{fold}] [E{epoch:02d}] total_loss={total_loss:.3f} | "
            f"VAL acc={val_acc:.3f} | F1={val_f1:.3f} | kappa={val_kap:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            best_path = MODELS / f"best_bilstm_fold{fold}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"[SAVE] best fold={fold} (by VAL F1) -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[STOP] Early stopping fold={fold}")
                break

    # --- TEST with best ---
    best_path = MODELS / f"best_bilstm_fold{fold}.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)

    test_acc, test_f1, test_kap = evaluate(model, test_loader, device)
    elapsed = time.time() - t0
    print(f"[F{fold}] TEST acc={test_acc:.3f} | macro-F1={test_f1:.3f} | kappa={test_kap:.3f}")
    print(f"[F{fold}] time={elapsed/60:.1f} min")

    out = {
        "k": int(k),
        "fold": int(fold),
        "train_subjects": list(train_subjects),
        "val_subjects": list(val_subjects),
        "test_subjects": list(test_subjects),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_kappa": float(test_kap),
        "elapsed_sec": float(elapsed),
	"time_sec": float(elapsed),
	"classes": CLASSES,
    }
    out_path = RESULTS / f"cv_bilstm_fold{fold}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()

    run_fold(k=args.k, fold=args.fold)


if __name__ == "__main__":
    main()

