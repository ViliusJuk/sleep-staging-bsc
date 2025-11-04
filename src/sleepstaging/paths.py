from pathlib import Path
import yaml

def load_config(path="src/sleepstaging/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()

RAW = Path(CFG["paths"]["raw_dir"])
PROC = Path(CFG["paths"]["processed_dir"])
MODELS = Path(CFG["paths"]["models_dir"])
RESULTS = Path(CFG["paths"]["results_dir"])

for p in [RAW, PROC, MODELS, RESULTS]:
    p.mkdir(parents=True, exist_ok=True)
