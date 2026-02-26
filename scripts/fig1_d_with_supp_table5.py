#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 1D: Event-controlled spectral power (PSD) and Supplementary Table S5
========================================================================

Purpose
- Compare ripple-band spectral power between HC and SZ, before and after
  controlling for event count / concatenated duration ("event-controlled PSD").

Input (default)
- <root>/data/df_PSD.csv
  required columns:
    site, cond, freq, group, subject,
    log10_psd, n_events_used, concat_duration_s
  optional:
    has_power_csv (boolean; if present, only rows with True are kept)

Outputs
- <root>/results/tables/Fig1D_cell_summary_log10_adj.csv
- <root>/results/tables/Supplementary_Table_S5_psd_raw_vs_event_normalized.csv
- <root>/results/figures/Fig1D_power_adj_<Site>.pdf

Definitions
- log10_psd_adj is computed within each (site, cond, freq) by residualizing:
    log10_psd ~ log10(n_events_used) + log10(concat_duration_s)
  and re-centering to the intercept (resid + const).

Statistics (Table S5)
- Two-sided permutation test on mean difference (SZ - HC) per site×cond×freq.
- Cohen's d (SZ - HC) using pooled SD.
- BH-FDR across frequencies within each Type×Site×Condition.
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
from statsmodels.stats.multitest import multipletests


# -------------------------
# Normalization (match other Fig1 scripts)
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
    if s in ("SZ", "SCHIZOPHRENIA", "SC", "SCZ"):
        return "SZ"
    if s in ("HC", "CONTROL", "HEALTHY"):
        return "HC"
    return s

def norm_cond(x: str) -> str:
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
# Helpers
# -------------------------
def _require_columns(df: pd.DataFrame, required: set[str], name: str):
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"[{name}] missing columns: {sorted(miss)}")

def _safe_log10_pos(x) -> np.ndarray:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return np.where(x > 0, np.log10(x), np.nan)

def cohens_d_sz_minus_hc(x_hc: np.ndarray, x_sz: np.ndarray) -> float:
    x_hc = np.asarray(x_hc, float); x_hc = x_hc[np.isfinite(x_hc)]
    x_sz = np.asarray(x_sz, float); x_sz = x_sz[np.isfinite(x_sz)]
    if x_hc.size < 2 or x_sz.size < 2:
        return np.nan
    v1 = np.var(x_hc, ddof=1)
    v2 = np.var(x_sz, ddof=1)
    denom = x_hc.size + x_sz.size - 2
    if denom <= 0:
        return np.nan
    pooled = ((x_hc.size - 1) * v1 + (x_sz.size - 1) * v2) / denom
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return float((np.mean(x_sz) - np.mean(x_hc)) / np.sqrt(pooled))

def perm_mean_diff_sz_minus_hc(x_hc: np.ndarray, x_sz: np.ndarray, n_perm: int = 10000, seed: int = 0):
    """
    Two-sided permutation test on mean difference (SZ - HC).
    p is floored at 1/n_perm.
    """
    rng = np.random.default_rng(seed)
    x_hc = np.asarray(x_hc, float); x_hc = x_hc[np.isfinite(x_hc)]
    x_sz = np.asarray(x_sz, float); x_sz = x_sz[np.isfinite(x_sz)]
    if x_hc.size < 2 or x_sz.size < 2:
        return (np.nan, np.nan)

    obs = float(np.mean(x_sz) - np.mean(x_hc))
    pooled = np.concatenate([x_hc, x_sz])
    n_hc = x_hc.size

    diffs = np.empty(n_perm, float)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diffs[i] = float(np.mean(pooled[n_hc:]) - np.mean(pooled[:n_hc]))

    p = float(np.mean(np.abs(diffs) >= abs(obs)))
    p = max(p, 1.0 / float(n_perm))
    return (obs, p)


# -------------------------
# Core: event-controlled PSD
# -------------------------
def recompute_log10_psd_adj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log10_psd_adj by residualizing within each (site, cond, freq):
      log10_psd ~ log10(n_events_used) + log10(concat_duration_s)
    and re-centering to intercept (resid + const).
    """
    out = df.copy()
    out["log10_n_events"] = _safe_log10_pos(out["n_events_used"])
    out["log10_dur_s"] = _safe_log10_pos(out["concat_duration_s"])
    out["log10_psd_adj"] = np.nan

    for (site, cond, freq), sub in out.groupby(["site", "cond", "freq"], dropna=False):
        tmp = sub[["log10_psd", "log10_n_events", "log10_dur_s"]].copy()
        tmp = tmp.dropna(subset=["log10_psd", "log10_n_events", "log10_dur_s"])
        if tmp.shape[0] < 4:
            continue

        X = pd.DataFrame({
            "log10_n_events": tmp["log10_n_events"],
            "log10_dur_s": tmp["log10_dur_s"],
        })
        X = sm.add_constant(X, has_constant="add")
        y = tmp["log10_psd"]

        try:
            model = sm.OLS(y, X, missing="drop").fit()
            y_adj = model.resid + float(model.params.get("const", 0.0))
        except Exception:
            continue

        idx = tmp.index
        out.loc[idx, "log10_psd_adj"] = y_adj

    return out


# -------------------------
# Tables
# -------------------------
def build_table_s5(df: pd.DataFrame, value_col: str, table_type: str, n_perm: int = 10000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for (site, cond, freq), sub in df.groupby(["site", "cond", "freq"], dropna=False):
        x_hc = pd.to_numeric(sub.loc[sub["group"] == "HC", value_col], errors="coerce").to_numpy(float)
        x_sz = pd.to_numeric(sub.loc[sub["group"] == "SZ", value_col], errors="coerce").to_numpy(float)
        x_hc = x_hc[np.isfinite(x_hc)]
        x_sz = x_sz[np.isfinite(x_sz)]

        n_hc = int(x_hc.size)
        n_sz = int(x_sz.size)

        mean_hc = float(np.mean(x_hc)) if n_hc else np.nan
        mean_sz = float(np.mean(x_sz)) if n_sz else np.nan
        sd_hc = float(np.std(x_hc, ddof=1)) if n_hc >= 2 else np.nan
        sd_sz = float(np.std(x_sz, ddof=1)) if n_sz >= 2 else np.nan

        diff, p = perm_mean_diff_sz_minus_hc(
            x_hc, x_sz, n_perm=n_perm,
            seed=seed + (abs(hash((table_type, site, cond, int(freq)))) % 100000)
        )
        d = cohens_d_sz_minus_hc(x_hc, x_sz)

        rows.append(dict(
            Type=table_type,
            Site=display_site(site),
            Condition=str(cond),
            Frequency_Hz=int(freq),
            N_HC=n_hc,
            N_SZ=n_sz,
            Mean_HC=mean_hc,
            Mean_SZ=mean_sz,
            SD_HC=sd_hc,
            SD_SZ=sd_sz,
            Diff_SZ_minus_HC=diff,
            Cohens_d=d,
            p_value=p,
        ))

    out = pd.DataFrame(rows)
    if out.empty:
        out["q_FDR"] = []
        return out

    out["q_FDR"] = np.nan
    for (typ, site, cond), g in out.groupby(["Type", "Site", "Condition"], sort=False):
        m = g["p_value"].notna()
        if m.any():
            out.loc[g.index[m], "q_FDR"] = multipletests(g.loc[m, "p_value"].values, method="fdr_bh")[1]

    return out.sort_values(["Type", "Site", "Condition", "Frequency_Hz"]).reset_index(drop=True)

def summarize_cell_means(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    mean±SE per (site, cond, freq, group) for plotting.
    """
    rows = []
    for (site, cond, freq, group), sub in df.groupby(["site", "cond", "freq", "group"], dropna=False):
        vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n == 0:
            continue
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
        rows.append(dict(site=site, cond=cond, freq=int(freq), group=group, n=n, mean=mean, se=se))
    return pd.DataFrame(rows)


# -------------------------
# Plot
# -------------------------
def plot_site_two_panels(cell_df: pd.DataFrame, site: str, value_label: str, out_pdf: Path):
    """
    One figure per site (two panels: Cortex, Hippocampus).
    Bars: HC vs SZ with mean±SE per frequency.
    """
    group_order = ["HC", "SZ"]
    color_map = {"HC": "#4C72B0", "SZ": "#C44E52"}  # blue / red

    site_df = cell_df[cell_df["site"] == site].copy()
    if site_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, cond in zip(axes, ["Cortex", "Hippocampus"]):
        sub = site_df[site_df["cond"] == cond].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{cond} (no data)")
            continue

        freqs = sorted(sub["freq"].unique())
        x = np.arange(len(freqs))
        w = 0.36

        for gi, g in enumerate(group_order):
            gsub = sub[sub["group"] == g]
            mean_map = {int(r["freq"]): float(r["mean"]) for _, r in gsub.iterrows()}
            se_map = {int(r["freq"]): float(r["se"]) for _, r in gsub.iterrows()}
            means = np.array([mean_map.get(int(f), np.nan) for f in freqs], float)
            ses = np.array([se_map.get(int(f), np.nan) for f in freqs], float)

            offset = (-w / 2) if gi == 0 else (w / 2)
            ax.bar(
                x + offset, means, width=w,
                yerr=ses, capsize=3,
                color=color_map[g],
                edgecolor="black", linewidth=0.6,
                label=g
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(f)) for f in freqs])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(value_label)
        ax.set_title(f"{display_site(site)} · {cond}", loc="left")
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    ap.add_argument("--input", type=str, default=None, help="Input CSV (default: <root>/data/df_PSD.csv).")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    in_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "df_PSD.csv")

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
        {"site", "cond", "freq", "group", "subject", "log10_psd", "n_events_used", "concat_duration_s"},
        "df_PSD.csv"
    )

    # normalize
    df = df.copy()
    df["site"] = df["site"].map(norm_site)
    df["cond"] = df["cond"].map(norm_cond)
    df["group"] = df["group"].map(norm_group)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df["subject"] = df["subject"].astype(str).str.strip()
    df["log10_psd"] = pd.to_numeric(df["log10_psd"], errors="coerce")
    df["n_events_used"] = pd.to_numeric(df["n_events_used"], errors="coerce")
    df["concat_duration_s"] = pd.to_numeric(df["concat_duration_s"], errors="coerce")

    # optional flag
    if "has_power_csv" in df.columns:
        df = df[df["has_power_csv"].astype(bool)]

    # keep only target conditions/groups/freq rows
    df = df[df["cond"].isin(["Hippocampus", "Cortex"])].copy()
    df = df[df["group"].isin(["HC", "SZ"])].copy()
    df = df.dropna(subset=["site", "cond", "freq", "group", "subject", "log10_psd"])

    # event-controlled PSD
    df_adj = recompute_log10_psd_adj(df)

    # summary for plotting
    cell_adj = summarize_cell_means(df_adj.dropna(subset=["log10_psd_adj"]), value_col="log10_psd_adj")
    out_cell = out_tables / "Fig1D_cell_summary_log10_adj.csv"
    cell_adj.to_csv(out_cell, index=False)

    # Table S5: raw vs event-controlled
    t_raw = build_table_s5(df_adj, value_col="log10_psd", table_type="raw", n_perm=args.n_perm, seed=args.seed)
    t_adj = build_table_s5(df_adj, value_col="log10_psd_adj", table_type="event_controlled", n_perm=args.n_perm, seed=args.seed)

    t_s5 = pd.concat([t_raw, t_adj], ignore_index=True)
    out_s5 = out_tables / "Supplementary_Table_S5_psd_raw_vs_event_normalized.csv"
    t_s5.to_csv(out_s5, index=False)

    # figures per site
    for site in sorted(df_adj["site"].unique()):
        out_pdf = out_figs / f"Fig1D_power_adj_{display_site(site)}.pdf"
        plot_site_two_panels(
            cell_df=cell_adj, site=site,
            value_label="log10 PSD (event-controlled)",
            out_pdf=out_pdf
        )

    print("[OK] Written:")
    print(" -", out_cell)
    print(" -", out_s5)
    print(" -", out_figs)

if __name__ == "__main__":
    main()
