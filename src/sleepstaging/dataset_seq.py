import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .paths import PROCESSED, CFG


class SleepEDFSequenceDataset(Dataset):
    """
    Sudaro sekas iš .npz (X,y) failų:
      - X: (E, 1, 3000), y: (E,)
      - Grąžina (x_seq, y_center), kur x_seq.shape = (seq_len, 3000)
      - Sekos formuojamos slankiu langu su 'stride' (default 1)
      - Etiketė imama iš CENTRO epochos (seq_len//2)

    split ∈ {"train","val","test"}
    """
    def __init__(self, split: str = "train", seq_len: int = 20, stride: int = 1, normalize: bool = True):
        assert split in {"train", "val", "test"}
        assert seq_len >= 2, "seq_len turi būti ≥ 2"
        assert stride >= 1, "stride turi būti ≥ 1"

        self.seq_len = seq_len
        self.stride = stride
        self.normalize = normalize

        # 1) Išsirenkam subjektus pagal CFG split
        if split == "test":
            subjects = CFG["split"]["test_subjects"]
        elif split == "val":
            subjects = CFG["split"]["val_subjects"]
        else:
            # train = visi .npz minus val ir test
            all_npz = sorted([p.stem for p in PROCESSED.glob("*.npz")])
            exclude = set(CFG["split"]["test_subjects"] + CFG["split"]["val_subjects"])
            subjects = [s for s in all_npz if s not in exclude]

        if not subjects:
            raise RuntimeError(f"[dataset_seq] Nėra subjektų split='{split}'. Patikrink CFG['split'].")

        self.X_list, self.y_list = [], []

        # 2) Kraunam kiekvieno subjekto .npz ir formuojam sekas
        for sid in subjects:
            p = PROCESSED / f"{sid}.npz"
            if not p.exists():
                print(f"[WARN] NPZ nerastas: {p}")
                continue

            d = np.load(p)
            X = d["X"]              # (E, 1, 3000)
            y = d["y"]              # (E,)

            # saugiklis
            if X.ndim != 3 or X.shape[1] != 1:
                raise ValueError(f"[dataset_seq] Tikimasi X.shape=(E,1,3000), gauta {X.shape} faile {p.name}")

            # per-epoch normalizacija (z-score)
            if self.normalize:
                # (E,1,T) -> normalizuojam per T kiekvienam E
                mu = X.mean(axis=-1, keepdims=True)
                sigma = X.std(axis=-1, keepdims=True) + 1e-6
                X = (X - mu) / sigma

            # (E,1,T) -> (E,T)
            X = X[:, 0, :]          # (E, 3000)

            # Sekos su overlap (stride)
            E = X.shape[0]
            if E < self.seq_len:
                # per mažai epochų šiam subjektui – praleidžiam
                print(f"[WARN] {sid}: per mažai epochų ({E}) sekai {self.seq_len}, praleidžiu.")
                continue

            for i in range(0, E - self.seq_len + 1, self.stride):
                seg = X[i:i+self.seq_len]     # (seq_len, 3000)
                lab = y[i:i+self.seq_len]     # (seq_len,)
                self.X_list.append(seg)
                self.y_list.append(lab)

        if not self.X_list:
            raise RuntimeError("[dataset_seq] Nesusiformavo nė viena seka. Patikrink .npz failus ir split'us.")

        self.X = np.stack(self.X_list).astype(np.float32)    # (N, seq_len, 3000)
        self.y = np.stack(self.y_list).astype(np.int64)    # (N, seq_len)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        x = self.X[i]                                    # (seq_len, 3000)
        if self.normalize:
            # papildoma on-the-fly normalizacija (saugiai)
            mu = x.mean(axis=-1, keepdims=True)
            sigma = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mu) / sigma

        center = self.seq_len // 2
        y_center = int(self.y[i][center])

        # į Tensor
        x = torch.from_numpy(x).float()                  # (seq_len, 3000)
        y_t = torch.tensor(y_center, dtype=torch.long)   # ()
        return x, y_t

