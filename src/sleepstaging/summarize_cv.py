# src/sleepstaging/summarize_cv.py
import json
import numpy as np
from pathlib import Path

RES = Path("results")
files = sorted(RES.glob("cv_cnn_fold*.json"))

acc, f1, kappa, time = [], [], [], []

for f in files:
    d = json.load(open(f))
    acc.append(d["test_acc"])
    f1.append(d["test_macro_f1"])
    kappa.append(d["test_kappa"])
    time.append(d["time_sec"])

acc = np.array(acc)
f1 = np.array(f1)
kappa = np.array(kappa)
time = np.array(time)

# ---- PRINT (kaip iki šiol) ----
print("CNN k-fold summary (test):")
print("folds:", len(files))
print(f"acc  : {acc.mean():.4f} ± {acc.std():.4f}")
print(f"f1   : {f1.mean():.4f} ± {f1.std():.4f}")
print(f"kappa: {kappa.mean():.4f} ± {kappa.std():.4f}")
print(f"time : {time.mean():.1f} ± {time.std():.1f} sec")

# ---- SAVE TXT ----
txt_path = RES / "cnn_cv10_summary.txt"
with open(txt_path, "w") as ftxt:
    ftxt.write("CNN 10-fold cross-validation summary (TEST set)\n")
    ftxt.write(f"Folds: {len(files)}\n\n")
    ftxt.write(f"Accuracy : {acc.mean():.4f} ± {acc.std():.4f}\n")
    ftxt.write(f"Macro-F1 : {f1.mean():.4f} ± {f1.std():.4f}\n")
    ftxt.write(f"Cohen kappa : {kappa.mean():.4f} ± {kappa.std():.4f}\n")
    ftxt.write(f"Time per fold : {time.mean():.1f} ± {time.std():.1f} sec\n")

# ---- SAVE CSV (per fold) ----
csv_path = RES / "cnn_cv10_folds.csv"
with open(csv_path, "w") as fcsv:
    fcsv.write("fold,accuracy,macro_f1,kappa,time_sec\n")
    for i, (a, f_, k, t) in enumerate(zip(acc, f1, kappa, time)):
        fcsv.write(f"{i},{a:.4f},{f_:.4f},{k:.4f},{t:.1f}\n")

print(f"\n[SAVED]")
print(f" - {txt_path}")
print(f" - {csv_path}")

