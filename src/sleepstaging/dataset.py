import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .paths import CFG, PROCESSED
from .labels import CLASSES

class SleepEDFNPZDataset(Dataset):
    """
    Krauna .npz (X, y) iš data/processed. Galima rinktis split: 'train'/'val'/'test'.
    Split parenkamas pagal subject ID iš config.yaml.
    """
    def __init__(self, split="test", subjects=None):
        self.files = []
        if subjects is None:
            if split == "test":
                subjects = CFG["split"]["test_subjects"]
            elif split == "val":
                subjects = CFG["split"]["val_subjects"]
            else:
                all_npz = sorted([p.stem for p in PROCESSED.glob("*.npz")])
                exclude = set(CFG["split"]["test_subjects"] + CFG["split"]["val_subjects"])
                subjects = [s for s in all_npz if s not in exclude]

        for sid in subjects:
            p = PROCESSED / f"{sid}.npz"
            if p.exists():
                self.files.append(p)
            else:
                print(f"[WARN] NPZ nerastas: {p.name}")

        Xs, ys = [], []
        for f in self.files:
            d = np.load(f)
            Xs.append(d["X"])  # (E, 1, T)
            ys.append(d["y"])  # (E,)
        if not Xs:
            raise RuntimeError("Nerasta .npz duomenų pasirinktai aibei.")

        self.X = np.concatenate(Xs, axis=0).astype(np.float32)
        self.y = np.concatenate(ys, axis=0).astype(np.int64)

        assert self.X.ndim == 3 and self.X.shape[1] == 1, "Tikimasi (E,1,T)"
        assert self.X.shape[0] == self.y.shape[0], "X ir y turi sutapti pagal E"

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class SleepEDFTestDataset(Dataset):
    def __init__(self, n_samples=1024, epoch_samples=3000):
        self.X = np.random.randn(n_samples, 1, epoch_samples).astype(np.float32)
        probs = np.array([0.2, 0.2, 0.35, 0.1, 0.15])
        self.y = np.random.choice(len(CLASSES), size=n_samples, p=probs).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

