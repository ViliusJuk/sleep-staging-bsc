from pathlib import Path
import yaml

# Project root
ROOT = Path("/scratch/lustre/home/viju8335/projects/sleep-staging-bsc")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CFG_PATH = ROOT / "src/sleepstaging/config.yaml"

def _load_cfg():
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found: {CFG_PATH}")

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "seed" not in cfg:
        cfg["seed"] = 42

    if "split" not in cfg:
        raise ValueError("config.yaml missing 'split' section")

    if "val_subjects" not in cfg["split"] or "test_subjects" not in cfg["split"]:
        raise ValueError("split must define val_subjects and test_subjects")

    return cfg

CFG = _load_cfg()

