#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 4a (SZ only): PANSS positive vs pooled ripple counts
=========================================================

This script reproduces the Fig. 4a-style panel showing the association between
PANSS positive symptom severity and pooled ripple counts (80–240 Hz) in
individuals with schizophrenia (SZ).

Input (derived / anonymized; no raw MEG):
  <root>/data/public_clinical_subject_level.csv

Required columns:
  site_public, events_sum, minutes_sum,
  PANSS_positive,
  age, sex, JART, sleepiness_pre, antipsychotics
Optional columns:
  group  (if present, rows will be restricted to group == "SZ")

Output:
  <root>/results/figures/Fig4a_PANSS_positive_vs_ripple_counts.pdf

Model (covariate-adjusted curve):
  Negative Binomial GLM (log link)
    events_sum ~ PANSS_positive + age + sex + JART + sleepiness_pre + antipsychotics + C(site_public)
  Offset:
    log(minutes_sum)

Notes on the NB model:
- statsmodels GLM NegativeBinomial uses a fixed dispersion parameter by default.
  Here, the model is used primarily to obtain a covariate-adjusted mean curve
  and its 95% CI for visualization, consistent with the manuscript pipeline.

Plot:
- Scatter: raw subject points (PANSS_positive vs events_sum)
- Line: covariate-adjusted predicted mean (minutes fixed at median/mode)
- Shaded band: 95% confidence interval

"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf


# -------------------------
# Helpers
# -------------------------
def mode_or_median(s: pd.Series):
    """Use mode for low-cardinality variables, otherwise median."""
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return np.nan
    if s2.nunique() <= 3:
        try:
            return s2.mode().iloc[0]
        except Exception:
            return float(s2.median())
    return float(s2.median())


def get_repo_root(user_root: str | None) -> Path:
    """Resolve repository root robustly for Code Ocean / local runs."""
    if user_root:
        return Path(user_root).expanduser().resolve()
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd().resolve()


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

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Repository root (default: inferred).")
    ap.add_argument("--input", type=str, default=None,
                    help="Input CSV (default: <root>/data/public_clinical_subject_level.csv)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PDF (default: <root>/results/figures/Fig4a_PANSS_positive_vs_ripple_counts.pdf)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    in_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "public_clinical_subject_level.csv")
    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_pdf = Path(args.out).expanduser().resolve() if args.out else (results_root  / "figures" / "Fig4a_PANSS_positive_vs_ripple_counts.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}")

    df = pd.read_csv(in_csv)

    required = {
        "site_public", "events_sum", "minutes_sum",
        "PANSS_positive",
        "age", "sex", "JART", "sleepiness_pre", "antipsychotics",
    }
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Input is missing required columns: {sorted(miss)}")

    # numeric coercion
    num_cols = ["events_sum", "minutes_sum", "PANSS_positive", "age", "sex", "JART", "sleepiness_pre", "antipsychotics"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["site_public"] = df["site_public"].astype(str)

    # SZ-only (if group exists)
    if "group" in df.columns:
        g = df["group"].astype(str).str.upper().replace({"SC": "SZ", "SCZ": "SZ", "SCHIZOPHRENIA": "SZ"})
        df = df[g == "SZ"].copy()

    # drop NA / invalid
    cols_need = ["site_public"] + num_cols
    D = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_need).copy()
    D = D[D["minutes_sum"] > 0].copy()
    if D.empty:
        raise RuntimeError("No usable rows after NA filtering and minutes_sum > 0 filter.")

    # Fit NB-GLM (for visualization)
    fam = sm.families.NegativeBinomial(alpha=1.0, link=sm.families.links.Log())
    formula = "events_sum ~ PANSS_positive + age + sex + JART + sleepiness_pre + antipsychotics + C(site_public)"
    offset = np.log(D["minutes_sum"].astype(float))
    fit = smf.glm(formula=formula, data=D, family=fam, offset=offset).fit(cov_type="HC3")

    # Prediction grid
    x_min, x_max = float(D["PANSS_positive"].min()), float(D["PANSS_positive"].max())
    x_grid = np.linspace(x_min, x_max, 200) if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max else np.array([x_min])

    # Fix covariates at representative values
    site_mode = D["site_public"].mode().iloc[0]
    minutes_fixed = float(mode_or_median(D["minutes_sum"]))

    P = pd.DataFrame({
        "PANSS_positive": x_grid,
        "site_public": site_mode,
        "age": mode_or_median(D["age"]),
        "sex": mode_or_median(D["sex"]),
        "JART": mode_or_median(D["JART"]),
        "sleepiness_pre": mode_or_median(D["sleepiness_pre"]),
        "antipsychotics": mode_or_median(D["antipsychotics"]),
    })

    sf = fit.get_prediction(P, offset=np.log(np.full_like(x_grid, minutes_fixed))).summary_frame(alpha=0.05)
    yhat = sf["mean"].to_numpy(float)
    ylo = sf["mean_ci_lower"].to_numpy(float)
    yhi = sf["mean_ci_upper"].to_numpy(float)

    # Plot
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=160)
    ax.scatter(D["PANSS_positive"], D["events_sum"], s=18, alpha=0.75)

    ax.plot(x_grid, yhat, lw=2)
    ax.fill_between(x_grid, ylo, yhi, alpha=0.2, linewidth=0)

    ax.set_xlabel("PANSS positive")
    ax.set_ylabel("Pooled ripple count (80–240 Hz, per 5 min)")
    ax.set_title("SZ only: ripple count vs PANSS positive", loc="left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Input :", in_csv)
    print("[OK] N (SZ):", len(D))
    print("[OK] Saved :", out_pdf)


if __name__ == "__main__":
    main()
