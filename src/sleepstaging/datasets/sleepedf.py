# src/sleepstaging/datasets/sleepedf.py
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# --------- Pagalbinės funkcijos ---------

def _zscore_epoch(x: torch.Tensor) -> torch.Tensor:
    """
    Z-score normalizacija vienai 30 s epochai.
    x: (1, 3000) arba (3000,)
    """
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - m) / s


def _load_npz_dir(root: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Krauna visus *.npz, grąžina sąrašą (subject_id, X, y).
    subject_id imamas iš failo vardo (pvz., SC4001E0.npz → SC4001E0).
    """
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    out = []
    for p in files:
        z = np.load(p)
        X = z["X"]  # (N,1,3000) float32/64
        y = z["y"]  # (N,) int
        sid = os.path.splitext(os.path.basename(p))[0]
        out.append((sid, X, y))
    return out


def _make_indices_per_subject(
    y: np.ndarray, seq_len: int, stride: int
) -> List[int]:
    """
    Sudaro pradžios indeksus sekų langams (L) vienam subjektui.
    Jei likutis mažesnis už L — praleidžiam.
    """
    n = len(y)
    if n < seq_len:
        return []
    idxs = list(range(0, n - seq_len + 1, stride))
    return idxs


# --------- Dataset'ai ---------

@dataclass
class SeqConfig:
    seq_len: int = 12
    stride: int = 1


class SleepEDFSeqDataset(Dataset):
    """
    Sekų dataset'as per kelis subjektus.
    Kiekvienas įrašas: x_seq: (L,1,3000) [torch.float32, z-scored per epoch],
                        y_label: int (centrinės epochos klasė)
    """
    def __init__(
        self,
        subjects: List[Tuple[str, np.ndarray, np.ndarray]],
        seq_cfg: SeqConfig,
    ):
        self.seq_len = int(seq_cfg.seq_len)
        self.stride = int(seq_cfg.stride)

        # Saugo bazinius masyvus ir indeksus (subj_id, start_idx)
        self._store: List[Tuple[int, int]] = []  # (subj_idx, start)
        self._subs: List[Tuple[str, np.ndarray, np.ndarray]] = subjects

        for si, (_sid, X, y) in enumerate(self._subs):
            starts = _make_indices_per_subject(y, self.seq_len, self.stride)
            for st in starts:
                self._store.append((si, st))

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, idx: int):
        si, st = self._store[idx]
        sid, X_np, y_np = self._subs[si]

        L = self.seq_len
        x_np = X_np[st: st + L]   # (L,1,3000)
        y_win = y_np[st: st + L]  # (L,)

        # -> torch
        x_seq = torch.from_numpy(x_np).float()             # (L,1,3000)
        # z-score kiekvienai epochai
        x_seq = torch.stack([_zscore_epoch(xi) for xi in x_seq], dim=0)  # (L,1,3000)

        # Label = centrinė epocha
        y_label = int(y_win[L // 2])
        y_label = torch.tensor(y_label, dtype=torch.long)

        return x_seq, y_label


# --------- API, kurią kviečia treniravimo kodas ---------

def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Sukuria train/val DataLoader'ius:
    - surenka visus subject NPZ iš data_root
    - paskutinį subjektą skiria val, likusius – train
    - L=12, stride=1 (atitinka tavo treniruotės prielaidą)
    Grąžina (train_loader, val_loader, info)
    """
    seq_cfg = SeqConfig(seq_len=12, stride=1)

    subs = _load_npz_dir(data_root)
    if len(subs) == 0:
        raise FileNotFoundError(f"No NPZ files under {data_root}")

    # Subjektų dalinimas: paskutinis → val
    train_subs = subs[:-1] if len(subs) > 1 else subs
    val_subs = subs[-1:] if len(subs) > 1 else subs

    train_ds = SleepEDFSeqDataset(train_subs, seq_cfg)
    val_ds = SleepEDFSeqDataset(val_subs, seq_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    info = {
        "seq_len": seq_cfg.seq_len,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    return train_loader, val_loader, info


def get_class_counts(data_root: str) -> List[int]:
    """
    Skaičiuoja centrinių epohų klases sekų lange (L=12, stride=1)
    per VISUS subjektus (train+val), kad atitiktų treniravimo target'ą.
    Grąžina sąrašą [c0, c1, c2, c3, c4].
    """
    seq_cfg = SeqConfig(seq_len=12, stride=1)

    subs = _load_npz_dir(data_root)
    if len(subs) == 0:
        raise FileNotFoundError(f"No NPZ files under {data_root}")

    # Sukaupiam centrinių label'ų pasiskirstymą
    counts = [0, 0, 0, 0, 0]
    for (_sid, X, y) in subs:
        starts = _make_indices_per_subject(y, seq_cfg.seq_len, seq_cfg.stride)
        for st in starts:
            lab = int(y[st + seq_cfg.seq_len // 2])
            if 0 <= lab < len(counts):
                counts[lab] += 1
            else:
                # jei atsirastų netikėta klasė – praplėsti masyvą
                need = lab + 1
                if need > len(counts):
                    counts.extend([0] * (need - len(counts)))
                counts[lab] += 1
    return counts

