# src/sleepstaging/summarize_cv.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize k-fold CV JSON results into TXT + CSV.")
    ap.add_argument("--results_dir", type=str, default="results", help="Directory containing cv_*.json files")
    ap.add_argument("--pattern", type=str, required=True, help='Glob pattern, e.g. "cv_cnn_fold*.json"')
    ap.add_argument("--name", type=str, default="model", help='Model name for printing/saving, e.g. "CNN" or "BiLSTM"')
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    if not res_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {res_dir}")

    files = sorted(res_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{args.pattern}' in {res_dir}")

    model_name = args.name.strip()
    safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_")
    if not safe_name:
        safe_name = "model"

    # collect metrics
    acc_list: List[float] = []
    f1_list: List[float] = []
    kappa_list: List[float] = []
    time_list: List[float] = []
    fold_ids: List[int] = []

    missing_keys: List[Tuple[Path, str]] = []

    for fp in files:
        d = load_json(fp)

        # fold id (prefer JSON, fallback parse from filename)
        fold = None
        if "fold" in d:
            try:
                fold = int(d["fold"])
            except Exception:
                fold = None
        if fold is None:
            # try parse "...fold7.json"
            name = fp.stem
            if "fold" in name:
                try:
                    fold = int(name.split("fold")[-1])
                except Exception:
                    fold = None

        # required keys
        for k in ("test_acc", "test_macro_f1", "test_kappa"):
            if k not in d:
                missing_keys.append((fp, k))

        if missing_keys:
            # don't crash immediately; we'll report after scanning all files
            continue

        # time key might differ (time_sec vs elapsed_sec)
        if "time_sec" in d:
            t = float(d["time_sec"])
        elif "elapsed_sec" in d:
            t = float(d["elapsed_sec"])
        else:
            t = float("nan")

        acc_list.append(float(d["test_acc"]))
        f1_list.append(float(d["test_macro_f1"]))
        kappa_list.append(float(d["test_kappa"]))
        time_list.append(t)
        fold_ids.append(-1 if fold is None else fold)

    if missing_keys:
        # show unique problems
        uniq = {}
        for fp, k in missing_keys:
            uniq.setdefault(str(fp), set()).add(k)
        msg_lines = ["Some JSON files are missing required keys:"]
        for fp, ks in uniq.items():
            msg_lines.append(f" - {fp}: missing {sorted(list(ks))}")
        raise KeyError("\n".join(msg_lines))

    # Convert to numpy
    acc = np.array(acc_list, dtype=np.float64)
    f1 = np.array(f1_list, dtype=np.float64)
    kappa = np.array(kappa_list, dtype=np.float64)
    time = np.array(time_list, dtype=np.float64)

    n = len(acc)
    # ---- PRINT ----
    print(f"{model_name} k-fold summary (test):")
    print("folds:", n)
    print(f"acc  : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"f1   : {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"kappa: {kappa.mean():.4f} ± {kappa.std():.4f}")

    if np.isfinite(time).any():
        # ignore NaNs when reporting time
        t_mean = np.nanmean(time)
        t_std = np.nanstd(time)
        print(f"time : {t_mean:.1f} ± {t_std:.1f} sec")
    else:
        print("time : (not available)")

    # ---- SAVE TXT ----
    txt_path = res_dir / f"{safe_name}_cv{n}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as ftxt:
        ftxt.write(f"{model_name} {n}-fold cross-validation summary (TEST set)\n")
        ftxt.write(f"Files pattern: {args.pattern}\n")
        ftxt.write(f"Folds: {n}\n\n")
        ftxt.write(f"Accuracy    : {acc.mean():.4f} ± {acc.std():.4f}\n")
        ftxt.write(f"Macro-F1    : {f1.mean():.4f} ± {f1.std():.4f}\n")
        ftxt.write(f"Cohen kappa : {kappa.mean():.4f} ± {kappa.std():.4f}\n")
        if np.isfinite(time).any():
            ftxt.write(f"Time per fold : {np.nanmean(time):.1f} ± {np.nanstd(time):.1f} sec\n")
        else:
            ftxt.write("Time per fold : (not available)\n")

    # ---- SAVE CSV (per fold) ----
    csv_path = res_dir / f"{safe_name}_cv{n}_folds.csv"
    # sort rows by fold id if we have them
    rows = list(zip(fold_ids, acc, f1, kappa, time, [str(p.name) for p in files]))
    rows.sort(key=lambda r: r[0])

    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("fold,accuracy,macro_f1,kappa,time_sec,file\n")
        for fold, a, f_, k, t, fname in rows:
            fold_out = "" if fold == -1 else str(fold)
            t_out = "" if not np.isfinite(t) else f"{t:.1f}"
            fcsv.write(f"{fold_out},{a:.4f},{f_:.4f},{k:.4f},{t_out},{fname}\n")

    print(f"\n[SAVED]")
    print(f" - {txt_path}")
    print(f" - {csv_path}")


if __name__ == "__main__":
    main()

