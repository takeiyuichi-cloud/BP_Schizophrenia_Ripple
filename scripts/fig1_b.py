#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 1B: Fold-change in ripple counts (SZ / HC) by frequency
===========================================================

Purpose
- Compute fold-change in ripple event counts (SZ / HC) across frequencies,
  separately for Hippocampus and Cortex, and plot one PDF per recording site.

Input (from repository root)
- data/df_clean_expanded.csv
  required columns:
    site, subject, group, cond, freq, event_count

Outputs (to repository root)
- results/figures/Fig1B_fold_change_counts_<Site>.pdf
- results/tables/Fig1B_fold_change_summary_means.csv
- results/tables/Fig1B_fold_change_sz_over_hc.csv

Notes
- "cond" is normalized to {Hippocampus, Cortex}. Extrahippocampal -> Cortex.
- Fold-change SE is approximated via a delta-method using group mean SEs.
- This script contains no absolute paths and is suitable for Code Ocean.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Normalization (match Fig1 A/C script)
# -------------------------
def norm_site(x: str) -> str:
    return str(x).strip().lower()

def display_site(site_raw: str) -> str:
    s = str(site_raw).lower()
    if "gundai" in s or "gunma" in s:
        return "Gunma"
    if "kumasou" in s or "kumagaya" in s:
        return "Kumagaya"
    return str(site_raw)

def norm_group(x: str) -> str:
    s = str(x).strip().upper()
    if s in ("HC", "CONTROL", "HEALTHY"):
        return "HC"
    if s in ("SZ", "SCHIZOPHRENIA", "SC", "SCZ"):
        return "SZ"
    return s

def norm_cond(x: str) -> str:
    """
    Force mapping to Hippocampus / Cortex.
    Extrahippocampal -> Cortex.
    """
    s = str(x).strip()
    s0 = s.lower()

    if s0 in ("hippocampus", "hippo", "hip"):
        return "Hippocampus"

    if s0 in (
        "cortex", "cortical", "ctx",
        "extrahippocampal", "extrahippocampus", "extrahippocapus",
        "extrahippocampal (cortex)", "extrahippocampal cortex",
        "extrahippocampal_cortex",
    ):
        return "Cortex"

    if s == "Extrahippocampal":
        return "Cortex"
    if s in ("Hippocampus", "Cortex"):
        return s

    return s


# -------------------------
# Utilities
# -------------------------
def _require_columns(df: pd.DataFrame, required: set[str], name: str):
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"[{name}] missing columns: {sorted(miss)}")

def summarize_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean±SE per (site, cond, freq, group).
    """
    rows = []
    for (site, cond, freq, group), sub in df.groupby(["site", "cond", "freq", "group"], dropna=False):
        vals = pd.to_numeric(sub["event_count"], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size >= 2 else np.nan
        rows.append(dict(site=site, cond=cond, freq=float(freq), group=group, n=int(vals.size), mean=mean, se=se))
    return pd.DataFrame(rows)

def compute_fold_change_sz_over_hc(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    fold = mean_SZ / mean_HC
    se_fold: delta-method approximation using SEs of group means
    """
    rows = []
    for (site, cond, freq), g in df_summary.groupby(["site", "cond", "freq"], dropna=False):
        hc = g[g["group"] == "HC"]
        sz = g[g["group"] == "SZ"]
        if hc.empty or sz.empty:
            continue

        mu_hc = float(hc["mean"].iloc[0])
        se_hc = float(hc["se"].iloc[0]) if np.isfinite(hc["se"].iloc[0]) else np.nan

        mu_sz = float(sz["mean"].iloc[0])
        se_sz = float(sz["se"].iloc[0]) if np.isfinite(sz["se"].iloc[0]) else np.nan

        # fold-change requires positive denominator
        if (not np.isfinite(mu_hc)) or (mu_hc <= 0) or (not np.isfinite(mu_sz)):
            fold, se_fold = np.nan, np.nan
        else:
            fold = mu_sz / mu_hc

            var_hc = (se_hc ** 2) if np.isfinite(se_hc) else 0.0
            var_sz = (se_sz ** 2) if np.isfinite(se_sz) else 0.0

            # delta method for ratio r = A/B
            # Var(r) ≈ (1/B)^2 Var(A) + (A/B^2)^2 Var(B)
            var_fold = (1.0 / mu_hc) ** 2 * var_sz + (mu_sz / (mu_hc ** 2)) ** 2 * var_hc
            se_fold = float(np.sqrt(var_fold)) if np.isfinite(var_fold) and var_fold > 0 else np.nan

        rows.append(dict(
            site=site,
            cond=cond,
            freq=float(freq),
            fold=float(fold) if np.isfinite(fold) else np.nan,
            se_fold=float(se_fold) if np.isfinite(se_fold) else np.nan,
            mean_HC=mu_hc,
            mean_SZ=mu_sz,
        ))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["site", "cond", "freq"]).reset_index(drop=True)


# -------------------------
# Plot
# -------------------------
def plot_site_foldchange(df_fold: pd.DataFrame, site: str, out_pdf: Path):
    """
    One figure per site:
      two lines: Hippocampus (yellow), Cortex (purple)
    """
    color_map = {
        "Hippocampus": "#FFD700",  # yellow (gold)
        "Cortex": "#7B2CBF",       # purple
    }

    plt.figure(figsize=(6.8, 4.6))

    for cond in ["Hippocampus", "Cortex"]:
        sub = df_fold[(df_fold["site"] == site) & (df_fold["cond"] == cond)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("freq")
        x = sub["freq"].to_numpy(float)
        y = sub["fold"].to_numpy(float)
        yerr = sub["se_fold"].to_numpy(float)

        plt.errorbar(
            x, y, yerr=yerr,
            fmt="-o",
            capsize=3,
            label=cond,
            color=color_map[cond],
            ecolor=color_map[cond],
        )

    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fold-change (SZ / HC)")
    plt.title(f"Fig. 1B — Fold-change in ripple counts ({display_site(site)})")
    plt.legend(frameon=False)
    plt.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

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
    ap.add_argument("--root", type=str, default=None, help="Repository root (default: auto-detect).")
    ap.add_argument("--input", type=str, default=None, help="Input CSV (default: <root>/data/df_clean_expanded.csv).")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    input_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "df_clean_expanded.csv")

    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_tab = results_root / "tables"
    out_fig = results_root / "figures"

    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)


    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    df = pd.read_csv(input_csv)
    _require_columns(df, {"site", "subject", "group", "cond", "freq", "event_count"}, "df_clean_expanded.csv")

    # normalize
    df = df.copy()
    df["site"] = df["site"].map(norm_site)
    df["subject"] = df["subject"].astype(str).str.strip()
    df["group"] = df["group"].map(norm_group)
    df["cond"] = df["cond"].map(norm_cond)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df["event_count"] = pd.to_numeric(df["event_count"], errors="coerce")

    # keep target
    df = df.dropna(subset=["site", "cond", "freq", "group", "event_count"])
    df = df[df["group"].isin(["HC", "SZ"])].copy()
    df = df[df["cond"].isin(["Hippocampus", "Cortex"])].copy()

    # summarize + fold
    summary = summarize_event_counts(df)
    df_fold = compute_fold_change_sz_over_hc(summary)

    # write tables
    out_summary = out_tab / "Fig1B_fold_change_summary_means.csv"
    out_fold = out_tab / "Fig1B_fold_change_sz_over_hc.csv"
    summary.to_csv(out_summary, index=False)
    df_fold.to_csv(out_fold, index=False)

    # one figure per known site
    for site in sorted(df_fold["site"].unique()):
        if not (("gundai" in site) or ("kumasou" in site) or (site in ("gunma", "kumagaya"))):
            continue
        out_pdf = out_fig / f"Fig1B_fold_change_counts_{display_site(site)}.pdf"
        plot_site_foldchange(df_fold, site, out_pdf)

    print("[OK] Written:")
    print(" -", out_summary)
    print(" -", out_fold)
    print(" -", out_fig)

if __name__ == "__main__":
    main()
