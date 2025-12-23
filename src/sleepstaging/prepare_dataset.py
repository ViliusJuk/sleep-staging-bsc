import re
from pathlib import Path
import numpy as np
import mne

from .paths import RAW, PROCESSED

# --- Konfigai ---
FS_TARGET = 100                # Hz
EP_LEN = FS_TARGET * 30        # 30 s epochas
N_CLASSES = 5

# Sleep-EDF anotacijų žemėlapis (N4 -> N3)
SLEEPEDF_MAP = {
    "SLEEP STAGE W": 0,
    "SLEEP STAGE 1": 1,
    "SLEEP STAGE 2": 2,
    "SLEEP STAGE 3": 3,
    "SLEEP STAGE 4": 3,  # N4 -> N3
    "SLEEP STAGE R": 4,
}
DROP_DESCRIPTIONS = {"MOVEMENT TIME", "SLEEP STAGE ?"}


# ---------- Pagalbinės ----------
def _norm_desc(desc: str) -> str:
    return re.sub(r"\s+", " ", desc.strip().upper())


def _pair_psg_hypno(raw_dir: Path):
    """Suporuoja *-PSG.edf su atitinkamais *Hypnogram*.edf, normalizuodama core SCxxxxE0/E1."""
    edfs = sorted(raw_dir.glob("*.edf"))
    psgs = [p for p in edfs if re.search(r'(?i)-PSG\.edf$', p.name)]
    hyps = [p for p in edfs if re.search(r'(?i)Hypnogram.*\.edf$', p.name)]

    def core_from_psg(name: str):
        # SC4001E0-PSG.edf -> SC4001E0
        m = re.match(r'^(SC\d{4}E[01])\-PSG\.edf$', name, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    def core_from_hyp(name: str):
        """
        SC4011EH-Hypnogram.edf  -> SC4011E0
        SC4001EC-Hypnogram.edf  -> SC4001E0
        SC4022EJ-Hypnogram.edf  -> SC4022E0
        SC4032EP-Hypnogram.edf  -> SC4032E0
        SC4001E0-Hypnogram.edf  -> SC4001E0
        """
        m = re.match(r'^(SC\d{4}E)([01]?)([A-Z]{0,2})\-Hypnogram.*\.edf$',
                     name, flags=re.IGNORECASE)
        if not m:
            return None
        prefix, digit, _ = m.groups()
        if not digit:
            digit = '0'
        return f"{prefix}{digit}".upper()

    def score(h):
        n = h.name.upper()
        if re.search(r'E[01]EC\-HYP', n):
            return 0
        if re.search(r'E[01][A-Z]{1,2}\-HYP', n):
            return 1
        return 2

    hyp_by_core = {}
    for h in hyps:
        core = core_from_hyp(h.name)
        if core:
            hyp_by_core.setdefault(core, []).append(h)

    pairs = []
    for psg in psgs:
        core = core_from_psg(psg.name)
        if not core:
            continue
        cands = hyp_by_core.get(core, [])
        if not cands:
            print(f"[WARN] Nerasta hipnograma PSG failui: {psg.name}")
            continue
        hyp = sorted(cands, key=score)[0]
        pairs.append((core, psg, hyp))
    return pairs


def _extract_epochs_from_annotations(ann: mne.Annotations, data_len_samples: int):
    """
    Pagal kiekvieną anotaciją (onset/duration) išskaido į 30 s epochas
    ir grąžina (start_idx, stop_idx, class_id) sąrašą. Naudoja realų laiką,
    ne i-indekso daugybas, kad X ir y neslystų.
    """
    epochs = []
    for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
        label = _norm_desc(desc)
        if label in DROP_DESCRIPTIONS:
            continue
        if label not in SLEEPEDF_MAP:
            # nežinoma anotacija — praleidžiam, bet nesulaužom pipeline
            continue

        cls = SLEEPEDF_MAP[label]
        start_samp = int(round(onset * FS_TARGET))
        dur_samp   = int(round(dur   * FS_TARGET))
        if dur_samp <= 0:
            continue

        n_full = dur_samp // EP_LEN
        for k in range(n_full):
            s = start_samp + k * EP_LEN
            e = s + EP_LEN
            if e <= data_len_samples:
                epochs.append((s, e, cls))
    return epochs


# ---------- Vykdymas ----------
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    pairs = _pair_psg_hypno(RAW)
    if not pairs:
        print("[ERR] Nerasta PSG/Hypnogram porų kataloge:", RAW)
        return

    print(f"[INFO] Rastos poros: {len(pairs)}")

    for core, psg_path, hyp_path in pairs:
        try:
            print(f"[INFO] {core}: skaitau PSG {psg_path.name}")
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

            # kanalų vardų normalizacija (Fpz-Cz)
            chs = [ch.upper().replace(" ", "") for ch in raw.ch_names]
            if "EEGFPZ-CZ" in chs:
                idx = chs.index("EEGFPZ-CZ")
                raw.rename_channels({raw.ch_names[idx]: "Fpz-Cz"})
            if "FPZ-CZ" not in raw.ch_names and "EEG Fpz-Cz" in raw.ch_names:
                raw.rename_channels({"EEG Fpz-Cz": "Fpz-Cz"})
            raw.pick_channels(["Fpz-Cz"])

            # resample į 100 Hz
            raw = raw.resample(FS_TARGET)


            raw = raw.notch_filter(freqs=[50], picks=["Fpz-Cz"], verbose=False)
	raw = raw.filter(l_freq=0.3, h_freq=35.0, picks=["Fpz-Cz"], verbose=False)
	raw = raw.resample(FS_TARGET)


            print(f"[INFO] {core}: skaitau anotacijas {hyp_path.name}")
            ann = mne.read_annotations(hyp_path)
            raw.set_annotations(ann)

            data = raw.get_data()[0]  # (n_samples,)
            ep_specs = _extract_epochs_from_annotations(ann, len(data))
            if not ep_specs:
                print(f"[WARN] {core}: nerasta tinkamų epochų – praleidžiam.")
                continue

            X_list, y_list = [], []
            for s, e, cls in ep_specs:
                X_list.append(data[s:e])
                y_list.append(cls)

            X = np.asarray(X_list, dtype=np.float32)   # (E, 3000)
            y = np.asarray(y_list, dtype=np.int64)     # (E,)
            X = X[:, None, :]                           # (E, 1, 3000)

            cnt = np.bincount(y, minlength=N_CLASSES)
            print(f"[INFO] {core}: class counts {cnt.tolist()} | X={X.shape}, y={y.shape}")

            out = PROCESSED / f"{core}.npz"
            np.savez(out, X=X, y=y)
            print(f"[OK] {core}: išsaugota {out.name}")

        except Exception as e:
            print(f"[FAIL] {core}: {e}")


if __name__ == "__main__":
    main()

