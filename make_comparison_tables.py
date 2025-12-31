#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

def main():
    report_dir = Path("test_reports_pp")
    df = pd.read_csv(report_dir / "summary.csv")

    # 1) Pivot: model x mode
    pivot = df.pivot(index="model", columns="mode", values=["accuracy", "f1_macro", "kappa"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    # 2) Best per metric (raw vs pp)
    best_rows = []
    for metric in ["accuracy", "f1_macro", "kappa"]:
        for mode in ["raw", "bias_smooth"]:
            sub = df[df["mode"] == mode].copy()
            i = sub[metric].idxmax()
            best_rows.append(sub.loc[i, ["model", "mode", metric]])
    best = pd.DataFrame(best_rows)

    out1 = report_dir / "comparison_pivot.csv"
    out2 = report_dir / "best_by_metric.csv"
    pivot.to_csv(out1, index=False)
    best.to_csv(out2, index=False)

    # also output a markdown table for easy paste into thesis
    md = pivot.to_markdown(index=False)
    (report_dir / "comparison_pivot.md").write_text(md + "\n", encoding="utf-8")

    print("Saved:", out1)
    print("Saved:", out2)
    print("Saved:", report_dir / "comparison_pivot.md")

if __name__ == "__main__":
    main()

