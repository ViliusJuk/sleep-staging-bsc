#!/usr/bin/env python3
from __future__ import annotations
import os, glob, argparse, numpy as np, mne
try:
    import pyedflib
except Exception:
    pyedflib = None

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,   # vėliau sujungsim 3 ir 4 -> 3
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
IGNORE = {"Movement time", "Sleep stage ?"}  # praleidžiam

def find_hypno_for(psg_path: str) -> str | None:
    """Surandam Hypnogram pagal pirmus 7 simbolius (pvz., SC4001E)."""
    d = os.path.dirname(psg_path)
    stem = os.path.basename(psg_path).split("-")[0]  # SC4001E0
    pref7 = stem[:7]  # SC4001E
    cand = sorted(glob.glob(os.path.join(d, f"{pref7}*-Hypnogram.edf")))
    return cand[0] if cand else None

def read_hyp_annotations(hyp_path: str) -> mne.Annotations:
    """
    Pirma bandom per mne.read_annotations; jei krenta (ar trukmės keistos),
    skaitom per pyedflib ir patys sukonstruojam Annotations.
    Visi tušti/Ne-float duration/onset -> 30.0/0.0
    """
    # 1) MNE kelias
    try:
        ann = mne.read_annotations(hyp_path, verbose="ERROR")
        # bandome suvienodinti trukmes į float+
        durs = np.array(ann.duration, dtype=float)
        bad = ~np.isfinite(durs) | (durs <= 0)
        if bad.any(): durs[bad] = 30.0
        ann.duration = durs
        # onsets irgi normalizuojam, jei reikia
        ons = np.array(ann.onset, dtype=float)
        bad_o = ~np.isfinite(ons)
        if bad_o.any(): ons[bad_o] = 0.0
        ann.onset = ons
        return ann
    except Exception:
        pass

    # 2) pyedflib kelias
    if pyedflib is None:
        raise RuntimeError("pyedflib neįdiegta, o MNE skaitymas nepavyko")

    with pyedflib.EdfReader(hyp_path) as f:
        ons, durs, desc = f.readAnnotations()

    def _to_float_list(arr, default):
        out = []
        for v in arr:
            try:
                out.append(float(v))
            except Exception:
                out.append(default)
        return np.asarray(out, dtype=float)

    ons = _to_float_list(ons, 0.0)
    durs = _to_float_list(durs, 30.0)
    durs[~np.isfinite(durs) | (durs <= 0)] = 30.0
    ons[~np.isfinite(ons)] = 0.0
    desc = [str(x) for x in desc]

    return mne.Annotations(onset=ons, duration=durs, description=desc)


    # 2) fallback per pyedflib
    if pyedflib is None:
        raise RuntimeError("pyedflib nėra įdiegta, o MNE skaitymas nepavyko")

    with pyedflib.EdfReader(hyp_path) as f:
        # pyedflib readAnnotations -> (onsets, durations, descriptions)
        onsets, durations, desc = f.readAnnotations()
    # Konvertuojam tvarkingai
    onsets = np.asarray(onsets, dtype=float)
    clean_durs = []
    for d in durations:
        try:
            clean_durs.append(float(d))
        except Exception:
            clean_durs.append(30.0)
    clean_durs = np.asarray(clean_durs, dtype=float)
    bad = ~np.isfinite(clean_durs) | (clean_durs <= 0)
    if bad.any():
        clean_durs[bad] = 30.0
    # desc į str
    desc = [str(x) for x in desc]
    return mne.Annotations(onset=onsets, duration=clean_durs, description=desc)


def pick_fpz_cz(raw: mne.io.BaseRaw) -> np.ndarray:
    """Grąžina vieną EEG kanalą (Fpz-Cz jei yra), formos (T,)."""
    picks = None
    # 1) Fpz-Cz
    for name in raw.ch_names:
        if "Fpz" in name and "Cz" in name:
            picks = raw.copy().pick([name])
            break
    # 2) bet koks 'EEG'
    if picks is None:
        eeg_names = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
        if eeg_names:
            picks = raw.copy().pick([eeg_names[0]])
    # 3) pirmas kanalas (fallback)
    if picks is None:
        picks = raw.copy().pick([raw.ch_names[0]])
    x, _ = picks[:, :]
    return x.ravel()

def to_epochs(signal: np.ndarray, sfreq: float, epoch_sec: int = 30) -> np.ndarray:
    """pjausto į 30 s epochas → (N, 3000) kai sfreq=100."""
    L = int(epoch_sec * sfreq)
    N = len(signal) // L
    sig = signal[: N * L].reshape(N, L)
    return sig

def parse_stages(ann: mne.Annotations, n_epochs: int) -> np.ndarray:
    """
    Hipnogramo anotacijas verčiam į epochines etiketes (N,).
    Jei nėra duration arba ji ne pars'inama -> laikom 30 s.
    Nežinomus labelius praleidžiam.
    """
    onsets = getattr(ann, "onset", None)
    durations = getattr(ann, "duration", None)  # mne>=1.3
    labels = getattr(ann, "description", None)

    if labels is None:
        return np.zeros(n_epochs, dtype=np.int64)

    # Saugi trukmių konversija -> float sekundėmis
    if durations is None:
        durs = np.full(len(labels), 30.0, dtype=float)
    else:
        # Kai kuriuose EDF hypnogramuose duration būna kaip str arba ''.
        durs = []
        for d in durations:
            try:
                # jei jau float – ok; jei str '' – kris -> except
                durs.append(float(d))
            except Exception:
                durs.append(30.0)
        durs = np.asarray(durs, dtype=float)
        # suvaldom 0, NaN ar neigiama
        bad = ~np.isfinite(durs) | (durs <= 0)
        if bad.any():
            durs[bad] = 30.0

    stages = []
    for desc, dur in zip(labels, durs):
        if desc in IGNORE:
            continue
        if desc not in STAGE_MAP:
            continue
        n = int(round(dur / 30.0))
        n = max(n, 1)  # bent 1 epocha
        stages.extend([STAGE_MAP[desc]] * n)

    y = np.asarray(stages, dtype=np.int64)

    # Sulyginam ilgį su PSG epohų skaičiumi
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
    ap.add_argument("--root", default="data/raw", help="Kur guli EDF failai")
    ap.add_argument("--out",  default="data/processed", help="Kur saugoti NPZ")
    ap.add_argument("--sfreq", type=float, default=100.0, help="Resampling dažnis (Hz)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    psgs = sorted(glob.glob(os.path.join(args.root, "*-PSG.edf")))
    if not psgs:
        print(f"[ERR] Neradau PSG EDF po {args.root}")
        return

    for psg in psgs:
        base = os.path.basename(psg).split("-")[0]  # SCxxxxE0
        out_path = os.path.join(args.out, f"{base}.npz")
        if os.path.exists(out_path):
            print(f"[SKIP] {base}.npz jau yra")
            continue

        hyp = find_hypno_for(psg)
        if hyp is None:
            print(f"[WARN] {base}: neradau hypnogram (pagal {os.path.basename(psg)[:7]}*)")
            continue

        try:
            # PSG
            raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
            raw.set_montage(None)
            raw.resample(args.sfreq)
            x = pick_fpz_cz(raw)                  # (T,)
            X = to_epochs(x, args.sfreq, 30)      # (N, 3000)

            # Hypnogram
            ann = read_hyp_annotations(hyp)
            # kai kuriems įrašams onsets būna už ribų – apkarpom
            raw.set_annotations(ann, emit_warning=False)
            y = parse_stages(ann, X.shape[0])  # (N,)

            # atmesti IGNORE epochas jau padarėm parse_stages (tiesiog nesukuriam joms įrašų)
            # bet ilgį suvienodinom, tad X ir y sutampa
            # pridėkim "kanalą" (1)
            X = X.astype(np.float32)
            X = X[:, None, :]  # (N,1,3000)

            # sanity
            assert len(X) == len(y), f"Ilgio neatitikimas {base}: X={X.shape[0]} y={len(y)}"

            np.savez_compressed(out_path, X=X, y=y)
            uniq = sorted(set(y.tolist()))
            print(f"[SAVE] {base}.npz: X={X.shape} classes={uniq}")
        except Exception as e:
            print(f"[WARN] {base}: praleidžiu dėl klaidos: {e}")

if __name__ == "__main__":
    main()

