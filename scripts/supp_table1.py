#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Supplementary Table S1 (threshold sensitivity).

Input (relative to project root):
  data/hippocampal_threshold_sensitivity_pooled_80_240_summary.csv

Output (relative to project root):
  results/tables/Table_S1.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# -------------------------
# Project root (auto-detect)
# -------------------------
def find_root() -> Path:
    start = Path.cwd().resolve()
    try:
        start = Path(__file__).resolve()
    except NameError:
        pass
    for p in [start] + list(start.parents):
        if (p / "run_all.py").exists():
            return p
    for p in [start] + list(start.parents):
        if (p / "data").exists():
            return p
    return Path.cwd().resolve()


def main() -> None:
    project_root = find_root()
    data_dir = project_root / "data"
    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = project_root / "results"
    out_dir = results_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = data_dir / "hippocampal_threshold_sensitivity_pooled_80_240_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(
            "Input summary file not found.\n"
            f"Expected: {summary_csv}\n"
            "Place the CSV under: <project_root>/data/"
        )

    df = pd.read_csv(summary_csv)

    required = [
        "threshold", "n_HC", "n_SZ",
        "mean_HC", "mean_SZ",
        "sem_HC", "sem_SZ",
        "d_SZ_minus_HC", "p_perm_two_sided",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. existing={list(df.columns)}")

    df = df.sort_values("threshold").copy()

    out_csv = out_dir / "Table_S1.csv"
    cols = required + [c for c in ["seed", "n_perm"] if c in df.columns]
    df[cols].to_csv(out_csv, index=False)

    print("[ROOT]", project_root)
    print("[SAVED]", out_csv)


if __name__ == "__main__":
    main()