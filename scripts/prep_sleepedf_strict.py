#!/usr/bin/env python3
from __future__ import annotations
import os, glob, argparse, numpy as np, mne

try:
    import pyedflib
except Exception:
    pyedflib = None

# žemėlapiai
STAGE_MAP_FULL = {
    "Sleep stage W": 0, "W": 0, "N0": 0,
    "Sleep stage 1": 1, "N1": 1, "1": 1,
    "Sleep stage 2": 2, "N2": 2, "2": 2,
    "Sleep stage 3": 3, "N3": 3, "3": 3,  # N3 apima buvusią 4
    "Sleep stage 4": 3, "N4": 3, "4": 3,
    "Sleep stage R": 4, "R": 4,
}
IGNORE = {"Movement time", "Sleep stage ?", "UNKNOWN", "?", ""}

def to_float(val, default):
    try:
        # bytes -> str
        if isinstance(val, (bytes, bytearray)):
            val = val.decode(errors="ignore")
        return float(val)
    except Exception:
        return float(default)

def normalize_desc(desc):
    # bytes -> str
    if isinstance(desc, (bytes, bytearray)):
        desc = desc.decode(errors="ignore")
    desc = str(desc).strip()
    # dažni variantai „Sleep stage X“ -> X
    if desc.startswith("Sleep stage "):
        key = desc
    else:
        key = desc.upper()
        # NREM žymėjimai
        if key in {"W","N1","N2","N3","N4","R"}:
            pass
        elif key in {"0","1","2","3","4"}:
            pass
        else:
            # paliekam originalą, gal pateks į STAGE_MAP_FULL tiesiogiai
            key = desc
    return key

def find_hyp(psg_path: str) -> str | None:
    d = os.path.dirname(psg_path)
    stem = os.path.basename(psg_path).split("-")[0]     # SC4051E0
    pref7 = stem[:7]                                    # SC4051E
    cands = sorted(glob.glob(os.path.join(d, f"{pref7}*-Hypnogram.edf")))
    return cands[0] if cands else None

def pick_fpz_cz(raw: mne.io.BaseRaw) -> np.ndarray:
    # 1) Fpz-Cz
    for name in raw.ch_names:
        if "Fpz" in name and "Cz" in name:
            x, _ = raw.copy().pick([name])[:, :]
            return x.ravel()
    # 2) pirmas EEG
    eeg = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
    if eeg:
        x, _ = raw.copy().pick([eeg[0]])[:, :]
        return x.ravel()
    # 3) fallback – pirmas kanalas
    x, _ = raw.copy().pick([raw.ch_names[0]])[:, :]
    return x.ravel()

def to_epochs(signal: np.ndarray, sfreq: float, epoch_sec: int = 30) -> np.ndarray:
    L = int(epoch_sec * sfreq)
    N = len(signal) // L
    return signal[:N*L].reshape(N, L)

def read_hyp_pyedf(hpath: str):
    """Grąžina (onsets, durations, descriptions) iš pyedflib, suvalgius edge-case'us."""
    if pyedflib is None:
        raise RuntimeError("pyedflib nėra įdiegta")
    with pyedflib.EdfReader(hpath) as f:
        ons, durs, desc = f.readAnnotations()
    ons = np.asarray([to_float(o, 0.0) for o in ons], dtype=float)
    durs = np.asarray([to_float(d, 30.0) for d in durs], dtype=float)
    durs[~np.isfinite(durs) | (durs <= 0)] = 30.0
    ons[~np.isfinite(ons)] = 0.0
    desc = [normalize_desc(d) for d in desc]
    return ons, durs, desc

def hyp_to_epoch_labels(ons, durs, desc, n_epochs: int):
    """Konstruoja y pagal 30 s duraciją; trukmės suapvalinamos iki epohų skaičiaus."""
    stages = []
    for o, d, lab in zip(ons, durs, desc):
        if lab in IGNORE:
            continue
        if lab not in STAGE_MAP_FULL:
            continue
        n = int(round(d / 30.0))
        n = max(n, 1)
        stages.extend([STAGE_MAP_FULL[lab]] * n)
    y = np.asarray(stages, dtype=np.int64)
    if len(y) < n_epochs:
        if len(y) > 0:
            pad = np.full(n_epochs - len(y), y[-1], dtype=np.int64)
            y = np.concatenate([y, pad], axis=0)
        else:
            y = np.zeros(n_epochs, dtype=np.int64)
    elif len(y) > n_epochs:
        y = y[:n_epochs]
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--sfreq", type=float, default=100.0)
    ap.add_argument("--only", nargs="*", default=[], help="Filtras, pvz.: SC4051E0 SC4052E0 ...")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    psgs = sorted(glob.glob(os.path.join(args.root, "*-PSG.edf")))

    # filtras
    if args.only:
        psgs = [p for p in psgs if os.path.basename(p).split("-")[0] in set(args.only)]
        if not psgs:
            print("[ERR] Nerasta PSG pagal --only filtra")
            return

    for psg in psgs:
        base = os.path.basename(psg).split("-")[0]  # SCxxxxE0
        outp = os.path.join(args.out, f"{base}.npz")
        if os.path.exists(outp) and not args.overwrite:
            print(f"[SKIP] {base}.npz jau yra")
            continue

        hyp = find_hyp(psg)
        if hyp is None:
            print(f"[WARN] {base}: neradau Hypnogram")
            continue

        try:
            # PSG per MNE (tik signalui)
            raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
            raw.set_montage(None)
            raw.resample(args.sfreq)
            sig = pick_fpz_cz(raw)                      # (T,)
            X = to_epochs(sig, args.sfreq, 30)          # (N,3000)

            # Hypno griežtai per pyedflib
            ons, durs, desc = read_hyp_pyedf(hyp)
            y = hyp_to_epoch_labels(ons, durs, desc, X.shape[0])

            X = X.astype(np.float32)[:, None, :]        # (N,1,3000)
            assert X.shape[0] == y.shape[0]

            np.savez_compressed(outp, X=X, y=y)
            print(f"[SAVE] {base}.npz: X={X.shape} classes={sorted(set(y.tolist()))}")
        except Exception as e:
            print(f"[WARN] {base}: praleidžiu dėl klaidos: {e}")

if __name__ == "__main__":
    main()

