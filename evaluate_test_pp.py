#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TEST evaluation su postprocess (pp) logika:

- CNN:  naudoja SleepEDFNPZDataset, per-epoch z-score, fiksuotas bias (N1=-0.9, N3=-1.0) + smooth k=5
- BiLSTM: bias (N1, N3) parenkamas GRID'u ant VAL, tada taikomas TEST + smooth k=5
- Transformer: bias (N1, N3) parenkamas GRID'u ant VAL, tada taikomas TEST + smooth k=5

Išsaugo:
  test_reports_pp/summary.csv
  *_report.txt
  *_cm.npy, *_cm_norm.npy

Paleidimas:
  python evaluate_test_pp.py --device auto
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

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

from src.sleepstaging.dataset import SleepEDFNPZDataset
from src.sleepstaging.labels import CLASSES
from src.sleepstaging.paths import MODELS

# CNN turi 2 variantus tavo rep'e, o checkpointas best_cnn1d.pth pasirodė nesuderinamas su V0,
# todėl darom auto-detect.
from src.sleepstaging.model_baseline import CNN1DBaseline
from src.sleepstaging.model_baseline_v0 import CNN1DBaselineV0

from src.sleepstaging.model_bilstm import CNNBiLSTM
from src.sleepstaging.model_transformer import SleepTransformer

# CNN pp funkcijos (kaip tavo eval_cnn_v0_pp)
from src.sleepstaging.postprocess import smooth_mode as smooth_mode_pp, apply_class_biases


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)


def smooth_mode_local(y_pred: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Majority-vote smoothing (kaip BiLSTM/Transformer pp eval'ai).
    """
    y = np.asarray(y_pred)
    n = len(y)
    h = k // 2
    out = np.empty(n, dtype=y.dtype)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        w = y[a:b]
        out[i] = np.bincount(w, minlength=len(CLASSES)).argmax()
    return out


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    y_true = []
    for X, y in loader:
        X = zscore(X.to(device, dtype=torch.float32))
        L = model(X).detach().cpu().numpy()
        logits_list.append(L)
        y_true.extend(y.numpy())
    return np.vstack(logits_list), np.asarray(y_true)


def pick_best_bias_on_val(logits_val: np.ndarray, y_val: np.ndarray) -> tuple[tuple[float, float], float]:
    """
    Grid search bias tik N1 ir N3:
      bias = [0, b1, 0, b3, 0], kur b1, b3 ∈ {0.0, -0.1, ..., -1.0}
    """
    best_bias = (0.0, 0.0)
    best_f1 = -1e9
    for b1, b3 in itertools.product(np.linspace(0.0, -1.0, 11), np.linspace(0.0, -1.0, 11)):
        bias = np.array([0.0, b1, 0.0, b3, 0.0], dtype=np.float32)
        y_pred = (logits_val + bias).argmax(1)
        f1m = f1_score(y_val, y_pred, average="macro", zero_division=0)
        if f1m > best_f1:
            best_f1 = float(f1m)
            best_bias = (float(b1), float(b3))
    return best_bias, best_f1


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray):
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kap = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cmn = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)
    return acc, f1m, kap, cm, cmn, rep


def build_and_load_cnn(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Auto-detect CNN architektūrą pagal checkpointą.
    Svarbu: nenaudojam strict=False, kad neliktų random sluoksnių.
    """
    state = torch.load(ckpt_path, map_location=device)

    candidates = [
        ("CNN1DBaseline", CNN1DBaseline(n_classes=len(CLASSES)).to(device)),
        ("CNN1DBaselineV0", CNN1DBaselineV0(n_classes=len(CLASSES)).to(device)),
    ]

    last_err = None
    for name, model in candidates:
        try:
            model.load_state_dict(state, strict=True)
            print(f"[CNN] Loaded checkpoint with: {name}")
            return model
        except RuntimeError as e:
            last_err = e

    raise RuntimeError(
        f"Could not load CNN checkpoint with known architectures.\n"
        f"Checkpoint: {ckpt_path}\n"
        f"Last error:\n{last_err}"
    )


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="test_reports_pp")
    ap.add_argument("--k-smooth", type=int, default=5)

    ap.add_argument("--cnn-ckpt", type=str, default="best_cnn1d.pth")
    ap.add_argument("--bilstm-ckpt", type=str, default="best_bilstm.pth")
    ap.add_argument("--tr-ckpt", type=str, default="best_transformer.pth")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("=== DEVICE:", device, "===")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Loaders
    val_ds = SleepEDFNPZDataset(split="val")
    test_ds = SleepEDFNPZDataset(split="test")
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    rows = ["model,mode,accuracy,f1_macro,kappa,bias_N1,bias_N3,k_smooth"]

    # =========================
    # 1) CNN (epoch-level)
    # =========================
    cnn = build_and_load_cnn(MODELS / args.cnn_ckpt, device)

    logits_test, y_test = collect_logits(cnn, test_loader, device)

    # RAW
    y_raw = logits_test.argmax(1)
    acc, f1m, kap, cm, cmn, rep = metrics_block(y_test, y_raw)
    print(f"\n[CNN RAW] acc={acc:.4f} f1={f1m:.4f} kappa={kap:.4f}")
    (outdir / "cnn_raw_report.txt").write_text(rep, encoding="utf-8")
    np.save(outdir / "cnn_raw_cm.npy", cm)
    np.save(outdir / "cnn_raw_cm_norm.npy", cmn)
    rows.append(f"CNN,raw,{acc:.6f},{f1m:.6f},{kap:.6f},,,{args.k_smooth}")

    # BIAS + SMOOTH (fiksuotas kaip tavo CNN pp)
    biases = np.array([0.0, -0.9, 0.0, -1.0, 0.0], dtype=np.float32)
    y_bias = apply_class_biases(logits_test, biases).argmax(1)
    y_bs = smooth_mode_pp(y_bias, k=args.k_smooth, n_classes=len(CLASSES))

    acc, f1m, kap, cm, cmn, rep = metrics_block(y_test, y_bs)
    print(f"[CNN BIAS+SMOOTH] acc={acc:.4f} f1={f1m:.4f} kappa={kap:.4f} | bias(N1,N3)=(-0.9,-1.0)")
    (outdir / "cnn_bias_smooth_report.txt").write_text(rep, encoding="utf-8")
    np.save(outdir / "cnn_bias_smooth_cm.npy", cm)
    np.save(outdir / "cnn_bias_smooth_cm_norm.npy", cmn)
    rows.append(f"CNN,bias_smooth,{acc:.6f},{f1m:.6f},{kap:.6f},-0.9,-1.0,{args.k_smooth}")

    # =========================
    # 2) BiLSTM (epoch-level input, sequence-aware inside model)
    # =========================
    bilstm = CNNBiLSTM(n_classes=len(CLASSES)).to(device, dtype=torch.float32)
    bilstm_state = torch.load(MODELS / args.bilstm_ckpt, map_location=device)
    bilstm.load_state_dict(bilstm_state, strict=True)

    logits_val, y_val = collect_logits(bilstm, val_loader, device)
    (b1, b3), f1_best = pick_best_bias_on_val(logits_val, y_val)
    print(f"\n[BiLSTM] best VAL bias (N1,N3)=({b1:.2f},{b3:.2f}) | VAL macro-F1={f1_best:.4f}")

    logits_test, y_test = collect_logits(bilstm, test_loader, device)

    # RAW
    y_raw = logits_test.argmax(1)
    acc, f1m, kap, cm, cmn, rep = metrics_block(y_test, y_raw)
    print(f"[BiLSTM RAW] acc={acc:.4f} f1={f1m:.4f} kappa={kap:.4f}")
    (outdir / "bilstm_raw_report.txt").write_text(rep, encoding="utf-8")
    np.save(outdir / "bilstm_raw_cm.npy", cm)
    np.save(outdir / "bilstm_raw_cm_norm.npy", cmn)
    rows.append(f"BiLSTM,raw,{acc:.6f},{f1m:.6f},{kap:.6f},,,{args.k_smooth}")

    # BIAS + SMOOTH (VAL-picked bias)
    bias_vec = np.array([0.0, b1, 0.0, b3, 0.0], dtype=np.float32)
    y_bias = (logits_test + bias_vec).argmax(1)
    y_bs = smooth_mode_local(y_bias, k=args.k_smooth)

    acc, f1m, kap, cm, cmn, rep = metrics_block(y_test, y_bs)
    print(f"[BiLSTM BIAS+SMOOTH] acc={acc:.4f} f1={f1m:.4f} kappa={kap:.4f} | bias(N1,N3)=({b1:.2f},{b3:.2f})")
    (outdir / "bilstm_bias_smooth_report.txt").write_text(rep, encoding="utf-8")
    np.save(outdir / "bilstm_bias_smooth_cm.npy", cm)
    np.save(outdir / "bilstm_bias_smooth_cm_norm.npy", cmn)
    rows.append(f"BiLSTM,bias_smooth,{acc:.6f},{f1m:.6f},{kap:.6f},{b1:.2f},{b3:.2f},{args.k_smooth}")

    # =========================
    # 3) Transformer
    # =========================
    tr = SleepTransformer(n_classes=len(CLASSES)).to(device, dtype=torch.float32)
    tr_state = torch.load(MODELS / args.tr_ckpt, map_location=device)
    tr.load_state_dict(tr_state, strict=True)

    logits_val, y_val = collect_logits(tr, val_loader, device)
    (b1, b3), f1_best = pick_best_bias_on_val(logits_val, y_val)
    print(f"\n[Transformer] best VAL bias (N1,N3)=({b1:.2f},{b3:.2f}) | VAL macro-F1={f1_best:.4f}")

    logits_test, y_test = collect_logits(tr, test_loader, device)

    # RAW
    y_raw = logits_test.argmax(1)
    acc, f1m, kap, cm, cmn, rep = metrics_block(y_test, y_raw)
    print(f"[Transformer RAW] acc={acc:.4f} f1={f1m:.4f} kappa={kap:.4f}")
    (outdir / "transformer_raw_report.txt").write_text(rep, encoding="utf-8")
    np.save(outdir / "transformer_raw_cm.npy", cm)
    np.save(outdir / "transformer_raw_cm_norm.npy", cmn)
    rows.append(f"Transformer,raw,{acc:.6f},{f1m:.6f},{kap:.6f},,,{args.k_smooth}")

    # BIAS + SMOOTH (VAL-picked bias)
    bias_vec = np.array([0.0, b1, 0.0, b3, 0.0], dtype=np.float32)
    y_bias = (logits_test + bias_vec).argmax(1)
    y_bs = smooth_mode_local(y_bias, k=args.k_smooth)

    acc, f1m, kap, cm, cmn, rep = metrics_block(y_test, y_bs)
    print(f"[Transformer BIAS+SMOOTH] acc={acc:.4f} f1={f1m:.4f} kappa={kap:.4f} | bias(N1,N3)=({b1:.2f},{b3:.2f})")
    (outdir / "transformer_bias_smooth_report.txt").write_text(rep, encoding="utf-8")
    np.save(outdir / "transformer_bias_smooth_cm.npy", cm)
    np.save(outdir / "transformer_bias_smooth_cm_norm.npy", cmn)
    rows.append(f"Transformer,bias_smooth,{acc:.6f},{f1m:.6f},{kap:.6f},{b1:.2f},{b3:.2f},{args.k_smooth}")

    # =========================
    # Summary
    # =========================
    (outdir / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved reports to: {outdir}/")
    print(f"Summary: {outdir}/summary.csv")


if __name__ == "__main__":
    main()

