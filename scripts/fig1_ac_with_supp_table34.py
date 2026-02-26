#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.1 (Panels A/C) + Supplementary Tables (counts & durations)
==============================================================

Purpose
- Build frequency-resolved group statistics for ripple event counts and durations
  using the SAME permutation-test framework as the figure panels.
- Generate:
    (1) Table s3: ripple event counts (HC vs SZ)
    (2) Table s4: ripple event durations (HC vs SZ)
    (3) Fig.1A: counts (per site; Cortex & Hippocampus panels)
    (4) Fig.1C: durations (per site; Cortex & Hippocampus panels)

Inputs (from repository root)
- data/df_clean_expanded.csv
  required columns:
    site, subject, group, cond, freq, event_count, mean_event_length

resultss (to repository root)
- results/tables/Table_s3_ripple_event_counts.csv
- results/tables/Table_s4_ripple_event_durations.csv
- results/figures/Fig1A_counts_<Site>.pdf
- results/figures/Fig1C_durations_<Site>.pdf

Statistics
- Statistic: mean(HC) − mean(SZ) (raw scale; no log transform)
- Permutation test: two-sided label shuffle
- p-value: proportion(|perm| >= |obs|)  (no +1 correction)
- Effect size: Cohen's d (HC − SZ) / pooled SD
- Multiple comparisons: BH-FDR across frequencies within each (Site × Condition × Outcome)

Notes for public reproducibility
- No absolute paths.
- Reproducible randomness via --seed, and deterministic per-cell seeding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests


# -------------------------
# Constants
# -------------------------
GROUPS = ("HC", "SZ")
COND_ORDER = ("Cortex", "Hippocampus")

COLOR = {"HC": "#4C72B0", "SZ": "#C44E52"}  # blue/red
STAR_THRESH = [(0.001, "***"), (0.01, "**"), (0.05, "*")]   # for figure annotation


# -------------------------
# Helpers: normalization
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
    Map to: Hippocampus / Cortex.
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

    if s in ("Hippocampus", "Cortex"):
        return s
    if s == "Extrahippocampal":
        return "Cortex"

    return s


# -------------------------
# Stats: permutation + Cohen's d
# -------------------------
def _cell_seed(base_seed: int, *parts: str) -> int:
    """
    Deterministic seed per cell to avoid dependence on iteration order.
    """
    key = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    h = hashlib.sha256(key).hexdigest()[:8]
    return int(h, 16)

def permutation_test_mean_diff(x_hc: np.ndarray, y_sz: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    """
    Two-sided permutation test for mean difference:
      stat = mean(HC) − mean(SZ)
      p = proportion(|perm| >= |obs|)
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x_hc, float)
    y = np.asarray(y_sz, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return (np.nan, np.nan)

    obs = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    nx = x.size

    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        rng.shuffle(pooled)
        perm_stats[i] = pooled[:nx].mean() - pooled[nx:].mean()

    p = float(np.mean(np.abs(perm_stats) >= abs(obs)))
    p = max(p, 1.0 / float(n_perm))
    return (obs, p)

def cohens_d(x_hc: np.ndarray, y_sz: np.ndarray) -> float:
    x = np.asarray(x_hc, float)
    y = np.asarray(y_sz, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan

    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    df = x.size + y.size - 2
    if df <= 0:
        return np.nan

    pooled = ((x.size - 1) * vx + (y.size - 1) * vy) / df
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan

    return float((x.mean() - y.mean()) / np.sqrt(pooled))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    if p.size == 0:
        return p
    return multipletests(p, method="fdr_bh")[1]


# -------------------------
# Table builder
# -------------------------
def build_table(df: pd.DataFrame, metric: str, *, n_perm: int, seed: int, outcome_label: str) -> pd.DataFrame:
    """
    Build one long table for a single outcome metric.
    FDR is applied across frequencies within each (Site × Condition).
    """
    rows = []

    for (site, cond, freq), sub in df.groupby(["site", "cond", "freq"], dropna=False):
        hc = sub[sub["group"] == "HC"]
        sz = sub[sub["group"] == "SZ"]

        x = pd.to_numeric(hc[metric], errors="coerce").dropna().to_numpy(float)
        y = pd.to_numeric(sz[metric], errors="coerce").dropna().to_numpy(float)
        if x.size < 2 or y.size < 2:
            continue

        cell_seed = _cell_seed(seed, outcome_label, site, cond, str(int(freq)))
        stat, p = permutation_test_mean_diff(x, y, n_perm=n_perm, seed=cell_seed)
        d = cohens_d(x, y)

        rows.append(dict(
            Site=display_site(site),
            Condition=cond,
            Frequency_Hz=int(freq),
            Statistic_meanHC_minus_meanSZ=float(stat),
            Mean_HC=float(x.mean()),
            Mean_SZ=float(y.mean()),
            SD_HC=float(x.std(ddof=1)),
            SD_SZ=float(y.std(ddof=1)),
            Cohens_d=float(d),
            p_value=float(p),
        ))

    out = pd.DataFrame(rows)
    if out.empty:
        out["q_fdr"] = []
        return out

    out["q_fdr"] = np.nan
    for (s, c), g in out.groupby(["Site", "Condition"], sort=False):
        out.loc[g.index, "q_fdr"] = bh_fdr(g["p_value"].values)

    out = out.sort_values(["Site", "Condition", "Frequency_Hz"], kind="mergesort").reset_index(drop=True)
    return out


# -------------------------
# Figure builder
# -------------------------
def _stars_from_q(q: float) -> str:
    if q is None or (isinstance(q, float) and not np.isfinite(q)):
        return ""
    for thr, st in STAR_THRESH:
        if q < thr:
            return st
    return ""

def plot_fig1_metric(df_site: pd.DataFrame, table: pd.DataFrame, metric: str, ylabel: str, out_pdf: Path, title: str):
    site_raw = df_site["site"].iloc[0]
    site_name = display_site(site_raw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)

    for ax, cond in zip(axes, COND_ORDER):
        sub = df_site[df_site["cond"] == cond].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{cond} (no data)")
            continue

        freqs = sorted(sub["freq"].unique())
        x = np.arange(len(freqs))
        w = 0.36

        for gi, group in enumerate(GROUPS):
            means = []
            ses = []
            for f in freqs:
                vals = pd.to_numeric(
                    sub[(sub["freq"] == f) & (sub["group"] == group)][metric],
                    errors="coerce"
                ).dropna().to_numpy(float)
                means.append(float(np.mean(vals)) if vals.size else np.nan)
                ses.append(float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size >= 2 else np.nan)

            offset = (-w / 2) if gi == 0 else (w / 2)
            ax.bar(
                x + offset, means, width=w, yerr=ses, capsize=3,
                color=COLOR[group], edgecolor="black", linewidth=0.6, label=group
            )

        # annotate q (FDR) stars
        tsub = table[(table["Site"] == site_name) & (table["Condition"] == cond)]
        for i, f in enumerate(freqs):
            hit = tsub[tsub["Frequency_Hz"] == int(f)]
            if len(hit) == 1:
                q = float(hit["q_fdr"].iloc[0]) if pd.notna(hit["q_fdr"].iloc[0]) else np.nan
                st = _stars_from_q(q)
                if st:
                    ymax = ax.get_ylim()[1]
                    ax.text(i, ymax * 1.02, st, ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(f)) for f in freqs])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{site_name} · {cond}")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# I/O + main
# -------------------------
def _require_columns(df: pd.DataFrame, required: set[str], name: str):
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"[{name}] missing columns: {sorted(miss)}")

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
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Repository root (default: auto-detect).")
    ap.add_argument("--in-csv", type=str, default=None, help="Input CSV (default: <root>/data/df_clean_expanded.csv).")
    ap.add_argument("--n-perm", type=int, default=10000, help="Permutation count per cell.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed for deterministic per-cell seeds.")
    args = ap.parse_args()

    # root resolution
    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    in_csv = Path(args.in_csv).expanduser().resolve() if args.in_csv else (root / "data" / "df_clean_expanded.csv")

    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_tables = results_root / "tables"
    out_figs = results_root / "figures"

    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}")

    df = pd.read_csv(in_csv)

    _require_columns(
        df,
        {"site", "subject", "group", "cond", "freq", "event_count", "mean_event_length"},
        "df_clean_expanded.csv"
    )

    # normalize
    df = df.copy()
    df["site"] = df["site"].map(norm_site)
    df["subject"] = df["subject"].astype(str).str.strip()
    df["group"] = df["group"].map(norm_group)
    df["cond"] = df["cond"].map(norm_cond)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df["event_count"] = pd.to_numeric(df["event_count"], errors="coerce")
    df["mean_event_length"] = pd.to_numeric(df["mean_event_length"], errors="coerce")

    # keep target rows
    df = df[
        df["cond"].isin(COND_ORDER) &
        df["group"].isin(GROUPS)
    ].dropna(subset=["site", "cond", "freq"])

    # tables
    tab_s3 = build_table(df, metric="event_count", n_perm=args.n_perm, seed=args.seed, outcome_label="counts")
    tab_s4 = build_table(df, metric="mean_event_length", n_perm=args.n_perm, seed=args.seed, outcome_label="durations")

    out_s3 = out_tables / "Table_s3_ripple_event_counts.csv"
    out_s4 = out_tables / "Table_s4_ripple_event_durations.csv"
    tab_s3.to_csv(out_s3, index=False)
    tab_s4.to_csv(out_s4, index=False)

    # figures per site
    for site in sorted(df["site"].unique()):
        df_site = df[df["site"] == site].copy()
        if df_site.empty:
            continue

        plot_fig1_metric(
            df_site, tab_s3, metric="event_count",
            ylabel="Ripple events (per 5 min)",
            out_pdf=out_figs / f"Fig1A_counts_{display_site(site)}.pdf",
            title="Fig. 1A: Ripple event counts"
        )
        plot_fig1_metric(
            df_site, tab_s4, metric="mean_event_length",
            ylabel="Event duration (ms)",
            out_pdf=out_figs / f"Fig1C_durations_{display_site(site)}.pdf",
            title="Fig. 1C: Ripple event durations"
        )

    print("[OK] Written:")
    print(" -", out_s3)
    print(" -", out_s4)
    print(" -", out_figs)

if __name__ == "__main__":
    main()
