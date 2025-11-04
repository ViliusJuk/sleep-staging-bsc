# src/sleepstaging/dataset_seq.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .paths import CFG, PROC
from .labels import CLASSES

class SleepEDFSeqDataset(Dataset):
    """
    Sudaro slankius langus per kiekvieno subjekto epochas:
      - window_size (n_episodu) turi būti nelyginis (pvz., 7)
      - etiketė = vidurinės epochos y[mid]
    Klaidos neperžengia subjekto ribų.
    """
    def __init__(self, split="train", subjects=None, window_size=7, step=1):
        assert window_size % 2 == 1, "window_size turi būti nelyginis (pvz., 5,7,9)"
        self.window_size = window_size
        self.half = window_size // 2
        self.step = step

        if subjects is None:
            if split == "test":
                subjects = CFG["split"]["test_subjects"]
            elif split == "val":
                subjects = CFG["split"]["val_subjects"]
            else:
                all_npz = sorted([p.stem for p in PROC.glob("*.npz")])
                exclude = set(CFG["split"]["test_subjects"] + CFG["split"]["val_subjects"])
                subjects = [s for s in all_npz if s not in exclude]

        files = []
        for sid in subjects:
            p = PROC / f"{sid}.npz"
            if p.exists():
                files.append(p)
            else:
                print(f"[WARN] NPZ nerastas: {p.name}")

        self.windows = []  # sąrašas (failo_indeksas, centro_indeksas)
        self.X_list, self.y_list = [], []
        for f in files:
            d = np.load(f)
            X = d["X"]  # (E,1,T)
            y = d["y"]  # (E,)
            self.X_list.append(X)
            self.y_list.append(y)
            E = X.shape[0]
            for center in range(self.half, E - self.half, self.step):
                self.windows.append((len(self.X_list) - 1, center))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fi, c = self.windows[idx]
        X = self.X_list[fi]
        y = self.y_list[fi]
        sl = slice(c - self.half, c + self.half + 1)  # langas [c-half : c+half]
        seq = X[sl]                                   # (W,1,T)
        target = y[c].astype(np.int64)
        # išdėstom kaip (W, C=1, T) → grąžinam torch tensor
        return torch.from_numpy(seq.astype(np.float32)), torch.tensor(target, dtype=torch.long)
