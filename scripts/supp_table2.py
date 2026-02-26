#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Supplementary Table S2: Peak-based vs ratio-based hippocampal event definitions.

Input (relative to project root):
  data/hippocampal_threshold_sensitivity_pooled_80_240_with_group.csv
  data/peak_based_counts_pooled_80_240.csv

Output (relative to project root):
  results/tables/Table_S2.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


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


RATIO_THR = 2.0


def cohens_d(hc, sz) -> float:
    hc = np.asarray(hc, float)
    sz = np.asarray(sz, float)
    n1, n2 = len(hc), len(sz)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = np.var(hc, ddof=1), np.var(sz, ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if not np.isfinite(sp) or sp == 0:
        return 0.0
    return float((np.mean(sz) - np.mean(hc)) / sp)


def analyze(df: pd.DataFrame, value_col: str) -> dict:
    hc = pd.to_numeric(df.loc[df["group"] == "HC", value_col], errors="coerce").to_numpy(float)
    sz = pd.to_numeric(df.loc[df["group"] == "SZ", value_col], errors="coerce").to_numpy(float)
    hc = hc[np.isfinite(hc)]
    sz = sz[np.isfinite(sz)]

    d = cohens_d(hc, sz)
    p = float(ttest_ind(sz, hc, equal_var=False).pvalue) if (len(hc) > 1 and len(sz) > 1) else np.nan

    return dict(
        n_HC=int(len(hc)),
        n_SZ=int(len(sz)),
        mean_HC=float(np.mean(hc)) if len(hc) else np.nan,
        mean_SZ=float(np.mean(sz)) if len(sz) else np.nan,
        d_SZ_minus_HC=float(d),
        p_value=float(p),
    )


def main():
    project_root = find_root()
    data_dir = project_root / "data"
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = project_root / "results"
    out_dir = results_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_with_group = data_dir / "hippocampal_threshold_sensitivity_pooled_80_240_with_group.csv"
    peak_counts = data_dir / "peak_based_counts_pooled_80_240.csv"

    if not ratio_with_group.exists():
        raise FileNotFoundError(f"Missing input: {ratio_with_group}")
    if not peak_counts.exists():
        raise FileNotFoundError(f"Missing input: {peak_counts}")

    df_ratio = pd.read_csv(ratio_with_group, dtype={"subject": str})
    df_peak = pd.read_csv(peak_counts, dtype={"subject": str})

    # ratio: use threshold=2.0 (robust to float encoding)
    df_ratio["threshold"] = pd.to_numeric(df_ratio["threshold"], errors="coerce")
    df_ratio = df_ratio[np.isclose(df_ratio["threshold"], float(RATIO_THR))].copy()

    # attach group to peak
    if "group" not in df_ratio.columns:
        raise RuntimeError(f"'group' column not found in {ratio_with_group}.")
    grp = df_ratio[["subject", "group"]].drop_duplicates()
    df_peak = df_peak.merge(grp, on="subject", how="inner")

    # required columns
    peak_col = "pooled_hip_peak_events"
    ratio_col = "pooled_hippocampal_events"

    if peak_col not in df_peak.columns:
        raise RuntimeError(f"Expected column '{peak_col}' not found in {peak_counts}. columns={list(df_peak.columns)}")
    if ratio_col not in df_ratio.columns:
        raise RuntimeError(f"Expected column '{ratio_col}' not found in {ratio_with_group}. columns={list(df_ratio.columns)}")

    res_ratio = analyze(df_ratio, ratio_col)
    res_peak = analyze(df_peak, peak_col)

    out = pd.DataFrame([
        dict(method=f"Ratio-based (ratio_thr={RATIO_THR})", **res_ratio),
        dict(method="Peak-based (peak vertex within hippocampal mask)", **res_peak),
    ])

    out_csv = out_dir / "Table_S2.csv"
    out.to_csv(out_csv, index=False)

    print("[ROOT]", project_root)
    print("[SAVED]", out_csv)


if __name__ == "__main__":
    main()