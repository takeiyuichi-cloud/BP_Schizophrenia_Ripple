#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Table S10:
Associations between total ripple load and clinical measures (SZ only)
=====================================================================

This script fits Negative Binomial GLMs (one model per clinical predictor):
  events_sum ~ predictor + covariates + C(site_public)
  offset = log(minutes_sum)

Effect size:
  IRR = exp(beta_predictor), with 95% CI

Multiple comparisons:
  BH-FDR across predictors within this table.

Inputs (derived / anonymized; no raw MEG):
  <root>/data/public_clinical_subject_level.csv
    Required columns:
      - site_public, events_sum, minutes_sum
      - covariates: age, sex, JART, sleepiness_pre, antipsychotics
    Predictors (any subset; missing ones are skipped):
      - PANSS_positive, PANSS_negative, PANSS_general (preferred), GAF
    Optional:
      - group (if present, rows are restricted to SZ)

Notes:
  - Legacy project column "PANSS_pasological" is supported as alias for PANSS_general.

Outputs:
  <root>/results/tables/Table_S10_total_ripple_load_vs_clinical.csv


Template (optional):
  templates/supp_table8.xlsx  (or template/supp_table8.xlsx, supp_tables/supp_table8.xlsx)

Code Ocean / reproducibility:
  - Use --root or env var PROJECT_ROOT (optional)
  - Otherwise auto-detect root from CWD
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -------------------------
# Settings
# -------------------------
COVARS = ["age", "sex", "JART", "sleepiness_pre", "antipsychotics"]

PRED_MAP = [
    ("PANSS_positive", "PANSS POS"),
    ("PANSS_negative", "PANSS NEG"),
    ("PANSS_general",  "PANSS GEN"),  # preferred
    ("GAF",            "GAF"),
]


# -------------------------
# Root/path helpers
# -------------------------
def _guess_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "run_all.py").exists():
            return p
    for p in [start] + list(start.parents):
        if (p / "data").exists():
            return p
    return start

def get_root(cli_root: str | None = None) -> Path:
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    env = os.environ.get("PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _guess_repo_root(Path.cwd())


def _find_template(root: Path, name: str) -> Path | None:
    candidates = [
        root / "templates" / name,
        root / "template" / name,
        root / "supp_tables" / name,
        Path(__file__).resolve().parent / name,
        Path(__file__).resolve().parent.parent / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# -------------------------
# Formatting helpers
# -------------------------
def _stars(p: float) -> str:
    if p is None or (isinstance(p, float) and (not np.isfinite(p))):
        return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""

def _fmt_q(q: float) -> str:
    if q is None or (isinstance(q, float) and (not np.isfinite(q))):
        return ""
    return f"{q:.4g}{_stars(q)}"

def _resolve_panss_general_alias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Support legacy column name PANSS_pasological as alias for PANSS_general.
    """
    df = df.copy()
    if "PANSS_general" not in df.columns and "PANSS_pasological" in df.columns:
        df["PANSS_general"] = df["PANSS_pasological"]
    return df


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
    ap.add_argument("--root", type=str, default=None, help="Project root (default: env PROJECT_ROOT or auto-detect from CWD)")
    ap.add_argument("--input", type=str, default=None, help="Clinical CSV (default: <root>/data/public_clinical_subject_level.csv)")
    ap.add_argument("--template", type=str, default=None, help="Optional template xlsx (default: auto-find supp_table8.xlsx)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    data_dir = root / "data"
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_dir = results_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)



    in_csv = Path(args.input).expanduser().resolve() if args.input else (data_dir / "public_clinical_subject_level.csv")
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input: {in_csv}")

    df = pd.read_csv(in_csv)
    df = _resolve_panss_general_alias(df)

    # required base columns
    required = {"site_public", "events_sum", "minutes_sum"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"public_clinical_subject_level.csv missing required columns: {sorted(miss)}")

    # numeric coercion
    num_cols = ["events_sum", "minutes_sum"] + COVARS + [c for c, _ in PRED_MAP]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["site_public"] = df["site_public"].astype(str)

    # SZ-only filter if group exists
    if "group" in df.columns:
        g = df["group"].astype(str).str.upper().replace({"SC": "SZ", "SCZ": "SZ", "SCHIZOPHRENIA": "SZ"})
        df = df[g == "SZ"].copy()

    fam = sm.families.NegativeBinomial(alpha=1.0, link=sm.families.links.Log())

    rows = []
    for pred_col, pred_label in PRED_MAP:
        if pred_col not in df.columns:
            continue

        cols_need = ["site_public", "events_sum", "minutes_sum", pred_col] + COVARS
        D = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_need).copy()
        D = D[D["minutes_sum"] > 0].copy()
        if D.empty:
            continue

        formula = "events_sum ~ " + pred_col + " + " + " + ".join(COVARS) + " + C(site_public)"
        offset = np.log(D["minutes_sum"].astype(float))

        fit = smf.glm(formula=formula, data=D, family=fam, offset=offset).fit(cov_type="HC3")

        beta = float(fit.params[pred_col])
        se = float(fit.bse[pred_col])
        p = float(fit.pvalues[pred_col])

        irr = float(np.exp(beta))
        ci_low = float(np.exp(beta - 1.96 * se))
        ci_high = float(np.exp(beta + 1.96 * se))

        rows.append(dict(Predictor=pred_label, IRR=irr, CI_low=ci_low, CI_high=ci_high, p_value=p))

    if not rows:
        raise RuntimeError("No models were fit. Check that predictors exist and missingness is not excessive.")

    out = pd.DataFrame(rows)
    out["q_fdr"] = multipletests(out["p_value"].values, method="fdr_bh")[1]

    out_csv = out_dir / "Table_S10_total_ripple_load_vs_clinical.csv"
    out.to_csv(out_csv, index=False)

    print("[OK] root:", root)
    print("[OK] input:", in_csv)
    print("[OK] wrote:", out_csv)

if __name__ == "__main__":
    main()
