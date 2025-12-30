# src/sleepstaging/dataset_seq.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .paths import PROCESSED, CFG


class SleepEDFSequenceDataset(Dataset):
    """
    Builds sliding-window sequences from per-subject .npz files.

    Expected npz contents:
      - X: (E, 1, 3000) or (E, 3000)
      - y: (E,)

    Returns:
      - x_seq: (seq_len, 3000) float32
      - y_center: () int64  (label of center epoch)

    Notes:
      - Sequences never cross subject boundaries (each subject processed separately).
      - If subjects are provided -> CV-friendly: ignores CFG["split"].
      - If subjects is None -> uses CFG["split"][train/val/test] (optional non-CV mode).
    """

    def __init__(
        self,
        split: str = "train",
        subjects=None,
        seq_len: int = 21,
        stride: int = 1,
        normalize: bool = False,
    ):
        assert split in {"train", "val", "test"}
        assert seq_len >= 2, "seq_len must be >= 2"
        assert stride >= 1, "stride must be >= 1"

        self.split = split
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.normalize = bool(normalize)
        self.center = self.seq_len // 2

        # ---- choose subjects ----
        if subjects is not None:
            self.subjects = list(subjects)
        else:
            # fallback non-CV split from CFG
            if split == "test":
                self.subjects = list(CFG["split"]["test_subjects"])
            elif split == "val":
                self.subjects = list(CFG["split"]["val_subjects"])
            else:
                all_npz = sorted([p.stem for p in PROCESSED.glob("*.npz")])
                exclude = set(CFG["split"]["test_subjects"] + CFG["split"]["val_subjects"])
                self.subjects = [s for s in all_npz if s not in exclude]

        if not self.subjects:
            raise RuntimeError(f"[dataset_seq] No subjects for split='{split}'.")

        # ---- materialize windows ----
        self.X_list = []
        self.y_list = []

        for subj in self.subjects:
            fpath = PROCESSED / f"{subj}.npz"
            if not fpath.exists():
                raise FileNotFoundError(f"[dataset_seq] Missing: {fpath}")

            data = np.load(fpath, allow_pickle=False)
            if "X" not in data or "y" not in data:
                raise KeyError(f"[dataset_seq] {fpath} must contain keys 'X' and 'y'")

            X = data["X"]  # (E,1,3000) or (E,3000)
            y = data["y"]  # (E,)

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64)

            # squeeze channel if present
            if X.ndim == 3 and X.shape[1] == 1:
                X = X[:, 0, :]  # (E,3000)

            if X.ndim != 2:
                raise ValueError(f"[dataset_seq] {subj}: expected X 2D after squeeze, got shape={X.shape}")
            if y.ndim != 1:
                raise ValueError(f"[dataset_seq] {subj}: expected y 1D, got shape={y.shape}")
            if len(X) != len(y):
                raise ValueError(f"[dataset_seq] {subj}: len(X) != len(y)")

            E = len(y)
            if E < self.seq_len:
                continue

            for start in range(0, E - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                x_seq = X[start:end]                 # (seq_len, 3000)
                y_center = int(y[start + self.center])

                self.X_list.append(x_seq)
                self.y_list.append(y_center)

        if len(self.y_list) == 0:
            raise RuntimeError(
                f"[dataset_seq] No windows generated (subjects={len(self.subjects)}, seq_len={self.seq_len})."
            )

        self.y = np.asarray(self.y_list, dtype=np.int64)  # helpful for samplers

    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, idx):
        x = self.X_list[idx]  # (seq_len, 3000)
        if self.normalize:
            m = x.mean(axis=-1, keepdims=True)
            s = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - m) / s
        x = torch.from_numpy(x).float()
        y = torch.tensor(self.y_list[idx], dtype=torch.long)
        return x, y

