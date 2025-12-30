# src/sleepstaging/train_seq_transformer_cv.py
from __future__ import annotations

import argparse
import json
import time
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

from .paths import MODELS, RESULTS, CFG
from .labels import CLASSES
from .utils import set_seed
from .cv_splits import get_fold_subjects
from .dataset_seq import SleepEDFSequenceDataset
from .model_seq_transformer import SeqTransformer


def zscore_epochwise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B, L, T) OR (B, 1, T)
    Normalize per-epoch over last axis (T).
    """
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)


@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, np.ndarray]:
    model.eval()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for X, y in loader:
        # Expect X: (B, L, 3000), y: (B,)
        X = X.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)
        X = zscore_epochwise(X)

        logits = model(X)  # (B, C)
        pred = torch.argmax(logits, dim=-1)

        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kap = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    return float(acc), float(f1), float(kap), cm


def run_fold(
    k: int,
    fold: int,
    seq_len: int,
    stride: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    d_model: int,
    n_heads: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> None:
    set_seed(CFG.get("seed", 1337))
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train_subjects, val_subjects, test_subjects = get_fold_subjects(k=k, fold=fold, seed=CFG.get("seed", 1337))

    print(f"[CV] k={k} fold={fold}")
    print(f"[CV] TRAIN subjects ({len(train_subjects)}): {train_subjects[:5]} ...")
    print(f"[CV] VAL   subjects ({len(val_subjects)}):   {val_subjects}")
    print(f"[CV] TEST  subjects ({len(test_subjects)}):  {test_subjects[:5]} ...")

    train_ds = SleepEDFSequenceDataset(split="train", subjects=train_subjects, seq_len=seq_len, stride=stride)
    val_ds   = SleepEDFSequenceDataset(split="val",   subjects=val_subjects,   seq_len=seq_len, stride=stride)
    test_ds  = SleepEDFSequenceDataset(split="test",  subjects=test_subjects,  seq_len=seq_len, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DEVICE: {device} ===")
    print(f"=== seq_len={seq_len} stride={stride} batch_size={batch_size} ===")
    print(f"=== Transformer: d_model={d_model} heads={n_heads} layers={num_layers} ff={dim_feedforward} dropout={dropout} ===")

    model = SeqTransformer(
        n_classes=len(CLASSES),
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max(64, seq_len),
    ).to(device=device, dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1 = -1.0
    patience = int(CFG.get("patience_transformer", 10))
    no_improve = 0

    t0 = time.time()
    print("=== START TRAIN (SeqTransformer CV fold) ===")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for bi, (X, y) in enumerate(train_loader):
            X = X.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.long,   non_blocking=True)

            X = zscore_epochwise(X)

            opt.zero_grad(set_to_none=True)
            logits = model(X)  # (B,C)
            loss = loss_fn(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().item())

            if bi % 20 == 0:
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=-1)
                    acc_b = (pred == y).float().mean().item()
                print(f"[F{fold}] [E{epoch:02d}] batch {bi:04d} | loss={loss.item():.4f} | acc={acc_b:.3f}")

        val_acc, val_f1, val_kap, _ = evaluate_full(model, val_loader, device)
        sched.step(val_f1)

        print(
            f"[F{fold}] [E{epoch:02d}] total_loss={total_loss:.3f} | "
            f"VAL acc={val_acc:.3f} | F1={val_f1:.3f} | kappa={val_kap:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            best_path = MODELS / f"best_seqtransformer_fold{fold}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"[SAVE] best fold={fold} (by VAL F1) -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[STOP] Early stopping fold={fold}")
                break

    # TEST with best model
    best_path = MODELS / f"best_seqtransformer_fold{fold}.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device), strict=True)

    test_acc, test_f1, test_kap, test_cm = evaluate_full(model, test_loader, device)
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
        "time_sec": float(elapsed),
        "confusion_matrix": test_cm.tolist(),
        "classes": CLASSES,
        "seq_len": int(seq_len),
        "stride": int(stride),
        "model": {
            "name": "SeqTransformer",
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
        },
    }

    out_path = RESULTS / f"cv_seqtransformer_fold{fold}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fold", type=int, required=True)

    ap.add_argument("--seq_len", type=int, default=21)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dim_feedforward", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    run_fold(
        k=args.k,
        fold=args.fold,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()

