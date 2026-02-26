#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 2b: Network composition of ripple events (HC vs SZ)
=======================================================

This script computes subject-level network composition (% of ripple events)
for each frequency band, and generates:

(1) Fig2b (PDF): mean ± SEM of network composition across frequencies for HC vs SZ
(2) Table S6 (CSV): network × frequency summary with HC/SZ mean±SD, SZ-HC difference,
    permutation p-values, and BH-FDR correction across frequencies within each network.

Input (default):
  <root>/data/events_by_network_subject_level.csv

Required columns:
  - group, subject, freq, network, n_events
Optional columns:
  - site (ignored; values are pooled across sites by default)

Outputs (default):
  <root>/results/figures/Fig2b_group_network_freq_percent_HC_vs_SZ.pdf
  <root>/results/tables/Table_S6_network_composition.csv

Statistics:
  - Subject-level composition: pct = 100 * n_events(network) / sum_n_events(all networks) per subject×freq
  - Group comparison per network×freq: two-sided permutation test on mean(SZ) - mean(HC)
  - Multiple comparison correction: BH-FDR across frequencies WITHIN each network block

Notes:
- This script uses only derived/anonymized tables.
- Random seed is fixed by default for reproducibility.

"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests


# -------------------------
# Defaults
# -------------------------
FREQ_ORDER_DEFAULT = [80, 120, 160, 200, 240]
NET_ORDER = [
    "Default",
    "Frontoparietal",
    "Dorsal Attention",
    "Ventral Attention",
    "Somatomotor",
    "Visual",
    "Limbic",
    "Hippocampus",
]
GROUP_ORDER = ["HC", "SZ"]

GROUP_COLOR = {"HC": "#4C72B0", "SZ": "#C44E52"}  # blue / red

# Table S6 uses manuscript-friendly abbreviations
NET_ABBREV = {
    "Default": "DMN",
    "Frontoparietal": "FPN",
    "Dorsal Attention": "DAN",
    "Ventral Attention": "VAN",
    "Somatomotor": "SMN",
    "Visual": "VN",
    "Limbic": "LM",
    "Hippocampus": "Hippocampus",
}


# -------------------------
# Normalizers
# -------------------------
def normalize_group(g: str) -> str:
    s = str(g).strip().upper()
    if s in ("SC", "SCZ", "SCHIZOPHRENIA"):
        return "SZ"
    if s in ("HEALTHY", "CONTROL"):
        return "HC"
    return s

def normalize_network(name: str) -> str:
    s = str(name).strip().replace("-", " ").replace("_", " ")
    s = s.replace("Network", "").replace("Mode", "")
    s = " ".join(s.split())
    key = s.lower().replace(" ", "")

    lut = {
        "default": "Default",
        "defaultmode": "Default",
        "dmn": "Default",
        "frontoparietal": "Frontoparietal",
        "fpn": "Frontoparietal",
        "dorsalattention": "Dorsal Attention",
        "dan": "Dorsal Attention",
        "ventralattention": "Ventral Attention",
        "van": "Ventral Attention",
        "somatomotor": "Somatomotor",
        "smn": "Somatomotor",
        "visual": "Visual",
        "vn": "Visual",
        "limbic": "Limbic",
        "ln": "Limbic",
        "hippocampus": "Hippocampus",
        "hpc": "Hippocampus",
    }
    if key in lut:
        return lut[key]
    if s in NET_ORDER:
        return s
    return s


# -------------------------
# Permutation test
# -------------------------
def perm_test_mean_diff(hc_vals, sz_vals, *, n_perm=10000, seed=0):
    """
    Two-sided permutation test:
      obs = mean(SZ) - mean(HC)
      p = proportion(|perm| >= |obs|)
    p is floored at 1/n_perm to avoid exact zeros.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(hc_vals, float); x = x[np.isfinite(x)]
    y = np.asarray(sz_vals, float); y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan

    obs = float(np.mean(y) - np.mean(x))
    cat = np.concatenate([x, y])
    nx = x.size

    diffs = np.empty(n_perm, float)
    for i in range(n_perm):
        rng.shuffle(cat)
        diffs[i] = float(np.mean(cat[nx:]) - np.mean(cat[:nx]))

    p = float(np.mean(np.abs(diffs) >= np.abs(obs)))
    p = max(p, 1.0 / float(n_perm))
    return obs, p

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
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--input", type=str, default=None,
                    help="CSV path (default: <root>/data/events_by_network_subject_level.csv)")
    ap.add_argument("--freqs", type=str, default="80,120,160,200,240")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    in_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "events_by_network_subject_level.csv")

    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_tab = results_root / "tables"
    out_fig = results_root / "figures"

    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)


    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}")

    freqs = [int(x.strip()) for x in args.freqs.split(",") if x.strip()]
    freq_order = [f for f in FREQ_ORDER_DEFAULT if f in freqs] + [f for f in freqs if f not in FREQ_ORDER_DEFAULT]

    df = pd.read_csv(in_csv)

    need = {"group", "subject", "freq", "network", "n_events"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    # normalize + filter
    df = df.copy()
    df["group"] = df["group"].map(normalize_group)
    df["network"] = df["network"].map(normalize_network)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").astype("Int64")
    df["n_events"] = pd.to_numeric(df["n_events"], errors="coerce")

    df = df[df["freq"].isin(freq_order)].copy()
    df = df[df["group"].isin(GROUP_ORDER)].copy()
    df = df[df["network"].isin(NET_ORDER)].copy()
    df = df.dropna(subset=["n_events", "freq", "subject"]).copy()

    # Pool across sites if site exists (safe even if missing)
    # group×subject×freq×network
    df = df.groupby(["group", "subject", "freq", "network"], as_index=False)["n_events"].sum()

    # subject×freq totals -> percent
    tot = df.groupby(["group", "subject", "freq"], as_index=False)["n_events"].sum().rename(columns={"n_events": "sum_events"})
    dfp = df.merge(tot, on=["group", "subject", "freq"], how="left")
    dfp.loc[dfp["sum_events"] <= 0, "sum_events"] = np.nan
    dfp["pct"] = (dfp["n_events"] / dfp["sum_events"]) * 100.0

    # Mean±SEM for plotting
    mean_rows = []
    for (grp, freq, net), sub in dfp.groupby(["group", "freq", "network"], dropna=False):
        vals = pd.to_numeric(sub["pct"], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if vals.size >= 2 else np.nan
        se = float(sd / np.sqrt(vals.size)) if vals.size >= 2 else np.nan
        mean_rows.append(dict(group=grp, freq=int(freq), network=net, n=int(vals.size), mean=mean, se=se))
    df_meanse = pd.DataFrame(mean_rows)

    if df_meanse.empty:
        raise RuntimeError("No data after filtering. Check input network labels and frequency bands.")

    # Permutation tests per network×freq
    perm_rows = []
    for (net, freq), sub in dfp.groupby(["network", "freq"], dropna=False):
        hc = pd.to_numeric(sub.loc[sub["group"] == "HC", "pct"], errors="coerce").dropna().to_numpy(float)
        sz = pd.to_numeric(sub.loc[sub["group"] == "SZ", "pct"], errors="coerce").dropna().to_numpy(float)

        mean_hc = float(np.mean(hc)) if hc.size else np.nan
        mean_sz = float(np.mean(sz)) if sz.size else np.nan
        sd_hc = float(np.std(hc, ddof=1)) if hc.size >= 2 else (0.0 if hc.size == 1 else np.nan)
        sd_sz = float(np.std(sz, ddof=1)) if sz.size >= 2 else (0.0 if sz.size == 1 else np.nan)

        obs, p = perm_test_mean_diff(
            hc, sz,
            n_perm=args.n_perm,
            seed=args.seed + (abs(hash((str(net), int(freq)))) % 100000)
        ) if (hc.size >= 2 and sz.size >= 2) else (np.nan, np.nan)
        perm_rows.append(dict(
            network=str(net),
            freq=int(freq),
            n_HC=int(hc.size),
            n_SZ=int(sz.size),
            mean_HC=mean_hc,
            mean_SZ=mean_sz,
            sd_HC=sd_hc,
            sd_SZ=sd_sz,
            diff_SZ_minus_HC=float(obs) if np.isfinite(obs) else np.nan,
            p_value=float(p) if np.isfinite(p) else np.nan,
        ))


    df_s6 = pd.DataFrame(perm_rows)

    # BH-FDR across frequencies WITHIN each network
    df_s6["q_fdr"] = np.nan
    for net, g in df_s6.groupby("network", sort=False):
        m = g["p_value"].notna()
        if m.any():
            df_s6.loc[g.index[m], "q_fdr"] = multipletests(g.loc[m, "p_value"].values, method="fdr_bh")[1]

    df_s6 = df_s6.sort_values(["network", "freq"]).reset_index(drop=True)

    # Save CSV
    out_csv_s6 = out_tab / "Table_S6_network_composition.csv"
    df_s6.to_csv(out_csv_s6, index=False)


    # -------------------------
    # Plot Fig2b
    # -------------------------
    # shared y max
    y_max = float((df_meanse["mean"] + df_meanse["se"]).max() * 1.15) if np.isfinite((df_meanse["mean"] + df_meanse["se"]).max()) else 100.0

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    width = 0.36
    x = np.arange(len(freq_order))

    for i, net in enumerate(NET_ORDER):
        ax = axes[i]
        sub = df_meanse[df_meanse["network"] == net].copy()
        if sub.empty:
            ax.axis("off")
            continue

        for gi, grp in enumerate(GROUP_ORDER):
            gsub = sub[sub["group"] == grp].set_index("freq").reindex(freq_order)
            means = gsub["mean"].to_numpy(float)
            ses = gsub["se"].to_numpy(float)

            offset = (-width/2) if gi == 0 else (width/2)
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=ses,
                capsize=2,
                color=GROUP_COLOR.get(grp),
                edgecolor="black",
                linewidth=0.5,
                label=grp if i == 0 else None,
            )

        ax.set_title(net, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(f)) for f in freq_order])
        ax.set_ylim(0, y_max)
        if i % 4 == 0:
            ax.set_ylabel("Percent of ripples (%), mean ± SEM")
        ax.grid(alpha=0.25, axis="y")

    # shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g], ec="black")
        for g in GROUP_ORDER
    ]
    fig.legend(handles, GROUP_ORDER, loc="upper right", frameon=False)

    fig.suptitle("Fig2b: Network composition of ripple events (HC vs SZ)", y=1.02, fontsize=13)
    fig.tight_layout()

    out_pdf = out_fig / "Fig2b_group_network_freq_percent_HC_vs_SZ.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Written:")
    print(" -", out_csv_s6)

    print(" -", out_pdf)


if __name__ == "__main__":
    main()
