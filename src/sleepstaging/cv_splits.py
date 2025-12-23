# src/sleepstaging/cv_splits.py
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from .paths import PROCESSED

def list_subject_ids():
    """
    Surenka visus NPZ stem'us iš data/processed.
    Pvz: SC4001E0, SC4002E0, ...
    """
    subs = sorted([p.stem for p in PROCESSED.glob("*.npz")])
    if not subs:
        raise RuntimeError(f"Nerasta .npz failų kataloge: {PROCESSED}")
    return subs

def get_fold_subjects(k: int, fold: int, seed: int = 42):
    """
    Subject-wise KFold.
    Grąžina (train_subjects, val_subjects, test_subjects).
    Val = 1 subjektas paimamas iš train deterministiškai.
    """
    subs = np.array(list_subject_ids())
    if fold < 0 or fold >= k:
        raise ValueError(f"fold turi būti 0..{k-1}, gauta: {fold}")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(kf.split(subs))
    train_idx, test_idx = splits[fold]

    train_subs = subs[train_idx].tolist()
    test_subs  = subs[test_idx].tolist()

    # Val paimame 1 subjektą iš train (deterministiškai)
    val_subs = [train_subs[0]]
    train_subs = train_subs[1:]

    return train_subs, val_subs, test_subs

