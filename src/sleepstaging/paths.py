from pathlib import Path

ROOT = Path("/scratch/lustre/home/viju8335/projects/sleep-staging-bsc")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CFG = {
    "seed": 42,
    "split": {
        "test_subjects": ["SC4031E0", "SC4032E0"],       # pvz. 2 test
        "val_subjects":  ["SC4021E0"],                   # pvz. 1 validation
    }
}


