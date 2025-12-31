#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TEST evaluation (Sleep-EDF) for:
  - CNN1D baseline
  - CNN+BiLSTM sequence model
  - Transformer sequence model

Metrics:
  - Accuracy
  - Macro F1
  - Cohen's kappa
  - Confusion matrix (raw + normalized)

Assumes:
  - subject-level split in config.yaml (paths.py -> CFG)
  - prepared NPZ data
  - saved checkpoints in ./models/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

# =========================
# 1) IMPORT YOUR DATASETS
# =========================
# TODO: pakeisk import'us į savo tikrus
# Pvz.:
# from src.sleepstaging.data import SleepEDFDataset, SleepEDFSequenceDataset
from src.sleepstaging.dataset import SleepEDFNPZDataset
  # <-- PAKEISK jei pas tave kitaip

# =========================
# 2) IMPORT YOUR MODELS
# =========================
# TODO: pakeisk import'us į savo tikrus
# Pvz.:
# from src.sleepstaging.models.cnn1d import CNN1DBaseline
# from src.sleepstaging.models.bilstm import CNNBiLSTM
# from src.sleepstaging.models.transformer import SeqTransformer
from src.sleepstaging.models import CNN1DBaseline, CNNBiLSTM, SeqTransformer  # <-- PAKEISK jei pas tave kitaip


CLASSES = ["W", "N1", "N2", "N3", "REM"]


@dataclass
class EvalResult:
    name: str
    acc: float
    f1_macro: float
    kappa: float
    cm: np.ndarray
    cm_norm: np.ndarray
    report: str


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def predict_epoch_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)  # x: (B, 1, 3000)
        logits = model(x)                    # (B, 5)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


@torch.no_grad()
def predict_seq_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for x_seq, y_center in loader:
        # x_seq: (B, seq_len, 3000) or (B, seq_len, 1, 3000) depending on your Dataset
        x_seq = x_seq.to(device, non_blocking=True)
        logits = model(x_seq)  # (B, 5)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y_center.numpy())
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    kap = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    return EvalResult(name=name, acc=acc, f1_macro=f1m, kappa=kap, cm=cm, cm_norm=cm_norm, report=report)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)

    # Dažniausi formatai:
    # 1) state_dict tiesiogiai
    # 2) dict su raktu 'state_dict' arba 'model'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    # kartais state_dict turi "module." prefiksą (DDP)
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=True)
    return model


def build_loaders(batch_size: int, num_workers: int, seq_len: int, stride: int) -> Dict[str, DataLoader]:
    # CNN dataset (epoch-level)
    ds_cnn = SleepEDFNPZDataset(split="test", normalize=True)

    # sequence datasets (center label)
    ds_seq = SleepEDFSequenceDataset(split="test", seq_len=seq_len, stride=stride, normalize=True)

    loaders = {
        "cnn": DataLoader(ds_cnn, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "seq": DataLoader(ds_seq, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--cnn-ckpt", type=str, default="models/best_cnn1d.pth")
    p.add_argument("--bilstm-ckpt", type=str, default="models/best_bilstm.pth")
    p.add_argument("--tr-ckpt", type=str, default="models/best_transformer.pth")

    p.add_argument("--outdir", type=str, default="test_reports")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(args.batch_size, args.num_workers, args.seq_len, args.stride)

    results: List[EvalResult] = []

    # -------------------------
    # CNN1D
    # -------------------------
    cnn = CNN1DBaseline(num_classes=len(CLASSES))
    cnn = load_checkpoint(cnn, Path(args.cnn_ckpt), device).to(device)
    y_true, y_pred = predict_epoch_model(cnn, loaders["cnn"], device)
    results.append(compute_metrics("CNN1D", y_true, y_pred))

    # -------------------------
    # CNN + BiLSTM (sequence)
    # -------------------------
    bilstm = CNNBiLSTM(num_classes=len(CLASSES), seq_len=args.seq_len)
    bilstm = load_checkpoint(bilstm, Path(args.bilstm_ckpt), device).to(device)
    y_true, y_pred = predict_seq_model(bilstm, loaders["seq"], device)
    results.append(compute_metrics("CNN+BiLSTM", y_true, y_pred))

    # -------------------------
    # Transformer (sequence)
    # -------------------------
    tr = SeqTransformer(num_classes=len(CLASSES), seq_len=args.seq_len)
    tr = load_checkpoint(tr, Path(args.tr_ckpt), device).to(device)
    y_true, y_pred = predict_seq_model(tr, loaders["seq"], device)
    results.append(compute_metrics("Transformer", y_true, y_pred))

    # -------------------------
    # Save reports
    # -------------------------
    # Summary table
    lines = []
    lines.append("model,accuracy,f1_macro,kappa")
    for r in results:
        lines.append(f"{r.name},{r.acc:.6f},{r.f1_macro:.6f},{r.kappa:.6f}")
    (outdir / "summary.csv").write_text("\n".join(lines), encoding="utf-8")

    # Per-model reports + confusion matrices
    for r in results:
        (outdir / f"{r.name}_report.txt").write_text(r.report, encoding="utf-8")
        np.save(outdir / f"{r.name}_cm.npy", r.cm)
        np.save(outdir / f"{r.name}_cm_norm.npy", r.cm_norm)

    # Print to stdout
    print("\n=== TEST RESULTS ===")
    for r in results:
        print(f"\n[{r.name}] acc={r.acc:.4f}  f1_macro={r.f1_macro:.4f}  kappa={r.kappa:.4f}")
        print(r.report)

    print(f"\nSaved: {outdir}/summary.csv + per-model reports + cm arrays")


if __name__ == "__main__":
    main()

