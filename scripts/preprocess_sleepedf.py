#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def find_pairs(root: Path):
    root = Path(root)
    psgs = sorted(root.glob("*-PSG.edf"))
    hyps = sorted(root.glob("*-Hypnogram.edf"))
    hyp_map = {}
    for h in hyps:
        hyp_map.setdefault(h.name[:6], []).append(h)
    pairs = []
    for p in psgs:
        key = p.name[:6]
        cand = hyp_map.get(key, [])
        if not cand:
            print(f"[WARN] No hypnogram for {p.name}")
            continue
        h = sorted(cand, key=lambda x: len(x.name))[0]
        pairs.append((p, h))
    return pairs

def pick_fpz_cz_channel(raw):
    names = [ch.lower().replace(" ", "") for ch in raw.ch_names]
    for i, nm in enumerate(names):
        if "fpz" in nm and "cz" in nm:
            return i
    eeg = [i for i, nm in enumerate(names) if "eeg" in nm]
    return eeg[0] if eeg else 0

# ----- nauja: hipnogramų skaitymas per pyEDFlib -----
def read_hypnogram_annotations(hyp_path: Path):
    """Grąžina (onsets, durations, descriptions) iš EDF per pyEDFlib."""
    import pyedflib
    f = pyedflib.EdfReader(str(hyp_path))
    try:
        ann = f.readAnnotations()  # (onsets, durations, desc)
        # pyEDFlib grąžina tuple of lists
        onsets, durations, descs = ann
        return onsets, durations, descs
    finally:
        f.close()

def load_with_mne(psg_path: Path, hyp_path: Path, sfreq=100.0):
    import mne
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - sfreq) > 1e-6:
        raw.resample(sfreq, npad="auto")

    ch_idx = pick_fpz_cz_channel(raw)
    data = raw.get_data(picks=[ch_idx])[0]  # (samples,)

    # Hipnogramą skaitom per pyEDFlib (ne per mne)
    onsets, durations, descs = read_hypnogram_annotations(hyp_path)

    # leidžiami aprašai -> klasė (0..4), palaikomi abu formatai (N1/N2/N3/N4 ir 1/2/3/4)
    allow = {
        "sleep stage w": 0, "w": 0,
        "sleep stage r": 4, "r": 4,
        "sleep stage n1": 1, "n1": 1, "sleep stage 1": 1, "1": 1,
        "sleep stage n2": 2, "n2": 2, "sleep stage 2": 2, "2": 2,
        "sleep stage n3": 3, "n3": 3, "sleep stage 3": 3, "3": 3,
        "sleep stage n4": 3, "n4": 3, "sleep stage 4": 3, "4": 3,
    }
    skip = {"movement time", "sleep stage ?", "?", "mt", "", None}

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _norm_desc(s):
        return (s if isinstance(s, str) else str(s)).strip().lower()

    epoch_len = int(round(sfreq * 30.0))
    X, Y = [], []

    # Rankinis ir labai saugus iteravimas
    for onset, dur, desc in zip(onsets, durations, descs):
        d = _norm_desc(desc)
        if d in skip or d not in allow:
            continue

        onset_f = _safe_float(onset)
        dur_f   = _safe_float(dur)
        if onset_f is None or dur_f is None:
            continue

        start = int(round(onset_f * sfreq))
        end   = start + epoch_len
        if start < 0 or end > len(data):
            continue

        seg = data[start:end].astype(np.float32)
        X.append(seg[None, :])
        Y.append(allow[d])

    if not X:
        raise RuntimeError(f"No valid epochs for {psg_path.name}")

    X = np.stack(X, axis=0)          # (N,1,3000)
    Y = np.array(Y, dtype=np.int64)  # (N,)
    return X, Y

def save_npz(out_dir: Path, base: str, X, y):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{base}.npz", X=X, y=y)
    classes = sorted(set(int(c) for c in y.tolist()))
    print(f"[SAVE] {base}.npz: X={X.shape} y={y.shape} classes={classes}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/raw")
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--sfreq", type=float, default=100.0)
    args = ap.parse_args()

    try:
        import mne  # signalui
        import pyedflib  # anotacijoms
        _ = (mne, pyedflib)
    except Exception:
        print("[ERR] reikia paketų: pip install mne pyEDFlib")
        return

    pairs = find_pairs(Path(args.root))
    if not pairs:
        print(f"[ERR] Nerasta porų {args.root}")
        return

    for psg, hyp in pairs:
        base = psg.stem.split("-")[0]  # SCxxxxE0
        try:
            X, y = load_with_mne(psg, hyp, sfreq=args.sfreq)
            save_npz(Path(args.out), base, X, y)
        except Exception as e:
            print(f"[WARN] {base}: praleidžiu dėl klaidos: {e}")

if __name__ == "__main__":
    main()

