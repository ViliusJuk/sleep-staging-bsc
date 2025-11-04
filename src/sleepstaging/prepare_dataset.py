import re
from pathlib import Path
import numpy as np
import mne
from mne.io import read_raw_edf

from .paths import CFG, RAW, PROC
from .labels import CLASSES, LABEL_MAP, CLASS2IDX

def _pair_psg_hypno(raw_dir: Path):
    """
    Suranda PSG ir Hypnogram poras:
    - PSG: SC####E0-PSG.edf, SC####E1-PSG.edf
    - Hyp: SC####E0-Hypnogram.edf, SC####EC-Hypnogram.edf, SC####E1-Hypnogram.edf, SC####E1C-Hypnogram.edf
    Taisyklė: branduoliai (core) lyginami kaip SC####E0 arba SC####E1.
    Ekvivalencijos:
      EC -> E0
      E1C -> E1
    Jei randamos kelios hipnogramos, prioritetas corrected („*C“) versijai.
    """
    edfs = sorted(raw_dir.glob("*.edf"))
    psgs = [p for p in edfs if re.search(r'(?i)[\-\_\s]PSG\.edf$', p.name)]
    hyps = [p for p in edfs if re.search(r'(?i)Hypnogram.*\.edf$', p.name)]

    def core_from_psg(name: str):
        # SC4001E0-PSG.edf -> SC4001E0 ; SC4001E1-PSG.edf -> SC4001E1
        m = re.match(r'^(SC\d{4}E[01])[\-\_\s]PSG\.edf$', name, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    def core_from_hyp(name: str):
        # Leidžiame EC/E0/E1/E1C ir normalizuojame į E0/E1
        # Pvz.:
        #  SC4001EC-Hypnogram.edf -> SC4001E0
        #  SC4001E0-Hypnogram.edf -> SC4001E0
        #  SC4001E1-Hypnogram.edf -> SC4001E1
        #  SC4001E1C-Hypnogram.edf -> SC4001E1
        m = re.match(r'^(SC\d{4}E(0|1|C|1C))[\-\_\s]Hypnogram.*\.edf$', name, flags=re.IGNORECASE)
        if not m:
            return None
        base = m.group(1).upper()  # pvz., SC4001E0 / SC4001EC / SC4001E1 / SC4001E1C
        # normalizuojam „C“:
        base = base.replace('EC', 'E0').replace('E1C', 'E1')
        return base

    # surenkam hyp pagal core
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
        # teik teikti corrected („C“) jei yra
        def is_corrected(hname: str) -> bool:
            return bool(re.search(r'(?i)E(C|1C)[\-\_\s]Hypnogram', hname))
        # rūšiuojam: pirma C versijos, paskui trumpesnis vardas
        hyp = sorted(cands, key=lambda h: (0 if is_corrected(h.name) else 1, len(h.name)))[0]
        base = core  # pvz., SC4001E0
        pairs.append((base, psg, hyp))

    return pairs


def _select_channel(raw: mne.io.BaseRaw, want: str):
    """
    Pasirenka kanalą (pvz., 'Fpz-Cz'), leidžiant pavadinimo variacijas:
    - 'EEG Fpz-Cz', 'Fpz-Cz', 'FPZ-CZ', su/ be tarpu/ brūkšniu.
    """
    import re

    def norm(s: str) -> str:
        # paliekam tik raides/skaičius/brūkšnį, uppercase
        return re.sub(r"[^A-Z0-9\-]", "", s.upper())

    target = norm(want)

    # 1) tikslus atitikimas (case-insensitive, su tarpu/underscore/dash variacijomis)
    for ch in raw.ch_names:
        if norm(ch) == target:
            return raw.pick_channels([ch])

    # 2) leidžiam 'EEG ' prefiksą ar kitus prefiksus – tikrinam ar baigiasi tiksliniu kanalu
    for ch in raw.ch_names:
        if norm(ch).endswith(target):
            return raw.pick_channels([ch])

    # 3) bandome su keliomis "kandidatų" formomis (dažniausios)
    candidates = [
        want,
        f"EEG {want}",
        want.replace("Fpz", "FPz").replace("Cz", "CZ"),
        f"EEG {want.replace('Fpz', 'FPz').replace('Cz','CZ')}",
    ]
    low = [c.strip().lower() for c in candidates]
    for ch in raw.ch_names:
        if ch.strip().lower() in low:
            return raw.pick_channels([ch])

    raise RuntimeError(f"Kanalas '{want}' nerastas. Rasti: {raw.ch_names}")


def _desc_to_stage(desc: str):
    """Konvertuoja MNE anotacijų aprašą į {W,1,2,3,4,R,M,?} vienženklę etiketę."""
    d = desc.strip().upper()
    # Tipiniai Sleep-EDF aprašai: 'Sleep stage W', 'Sleep stage 1', ..., 'Sleep stage R', 'Movement time'
    if "SLEEP STAGE" in d:
        last = d.split()[-1]  # W/1/2/3/4/R/?
        return last if last in {"W","1","2","3","4","R","?"} else "?"
    if "MOVEMENT" in d:
        return "M"
    # jei neatpažinta
    return "?"

def _labels_from_annotations(ann: mne.Annotations, sfreq: float, n_epochs: int, epoch_sec: int):
    """Grąžina label vienženklę seką per 30s epochas pagal anotacijas."""
    onset = ann.onset
    dur = ann.duration
    desc = ann.description
    out = []
    for i in range(n_epochs):
        t0 = i * epoch_sec  # sekundėmis nuo įrašo pradžios
        label = "?"
        # surandame anotaciją, kuri dengia t0
        for o, d, de in zip(onset, dur, desc):
            if o <= t0 < o + d:
                label = _desc_to_stage(de)
                break
        out.append(label)
    return np.array(out, dtype=object)

def _zscore_per_record(X: np.ndarray, eps: float = 1e-7):
    """Z-score normalizacija per įrašą (per kanalą), X shape: (E, 1, T)."""
    mu = X.mean(axis=(0,2), keepdims=True)
    sd = X.std(axis=(0,2), keepdims=True)
    return (X - mu) / (sd + eps)

def process_one(base_id: str, psg_path: Path, hyp_path: Path, cfg):
    fs_target = cfg["fs_target"]
    epoch_sec = cfg["epoch_sec"]
    channel = cfg["channel"]

    print(f"[INFO] {base_id}: skaitau PSG {psg_path.name}")
    raw = read_raw_edf(psg_path, preload=True, verbose=False)
    _select_channel(raw, channel)  # paliks tik vieną kanalą
    # Resample
    raw.resample(fs_target, npad="auto")

    print(f"[INFO] {base_id}: skaitau anotacijas {hyp_path.name}")
    ann = mne.read_annotations(str(hyp_path))
    raw.set_annotations(ann)

    data = raw.get_data()  # shape (1, n_times)
    sf = raw.info["sfreq"]
    assert int(sf) == int(fs_target), f"Resampling nepavyko: {sf} != {fs_target}"

    epoch_samples = int(epoch_sec * sf)
    n_full_epochs = data.shape[1] // epoch_samples
    if n_full_epochs == 0:
        raise RuntimeError(f"{base_id}: per trumpas įrašas epochoms.")

    data = data[:, : n_full_epochs * epoch_samples]  # nukerpam uodegą
    X = data.reshape(1, n_full_epochs, epoch_samples)  # (1, E, T)
    X = np.transpose(X, (1, 0, 2))                    # (E, 1, T)

    labels_1char = _labels_from_annotations(ann, sfreq=sf, n_epochs=n_full_epochs, epoch_sec=epoch_sec)

    # M ir ? – pašalinam
    keep_mask = np.isin(labels_1char, ["M","?"], invert=True)
    X = X[keep_mask]
    labels_1char = labels_1char[keep_mask]

    # Map 3/4 -> N3, o po to -> indeksai 0..4
    mapped = []
    for s in labels_1char:
        s2 = LABEL_MAP.get(s, None)
        if s2 is None:
            # netikėtas labelis – praleidžiam
            continue
        mapped.append(s2)
    mapped = np.array(mapped, dtype=object)
    y = np.array([CLASS2IDX[m] for m in mapped], dtype=np.int64)

    # Normalizacija per įrašą
    X = _zscore_per_record(X)

    # Išsaugom
    PROC.mkdir(parents=True, exist_ok=True)
    out_path = PROC / f"{base_id}.npz"
    np.savez_compressed(out_path, X=X.astype(np.float32), y=y.astype(np.int64),
                        fs=int(sf), epoch_sec=int(epoch_sec), channel=channel)
    print(f"[OK] {base_id}: išsaugota {out_path.name} | X={X.shape}, y={y.shape}")

def main():
    pairs = _pair_psg_hypno(RAW)
    if not pairs:
        print("[ERR] Nerasta PSG/Hypnogram porų kataloge data/raw/")
        return
    for base, psg, hyp in pairs:
        try:
            process_one(base, psg, hyp, CFG)
        except Exception as e:
            print(f"[FAIL] {base}: {e}")

if __name__ == "__main__":
    main()
