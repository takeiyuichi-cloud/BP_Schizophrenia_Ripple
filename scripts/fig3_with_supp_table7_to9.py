#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 3b–e + Supplementary Tables S7–S9 (and optional S8)
========================================================

This script reproduces Fig. 3 temporal clustering panels and supporting
supplementary tables using *derived* (anonymized) subject-level tables.

Key definition (IMPORTANT)
--------------------------
Within-epoch ripple rate ("within-state density") is ALWAYS defined as:
    within_epoch_event_rate = n_events_in_epochs / sum_epoch_dur_s
(duration-weighted; NOT mean-of-epoch-densities; NOT max/peak).

Primary inputs
--------------
(1) Subject-level table (required):
    <root>/data/epoch_metrics_subject_freq.csv
    Required columns:
      subject, freq, n_epochs, n_events, n_events_in_epochs, sum_epoch_dur_s, recording_len_s
    Optional columns:
      site (only needed if you want epoch-level duration merge by site)
      group (HC/SZ); if absent, group will be attached via --group-csv or df_clean_expanded.csv

(2) Epoch-level table (optional; used to compute mean/median epoch durations more faithfully):
    <root>/data/epoch_metrics_epoch_level.csv
    Required columns (if used):
      subject, freq, duration_s
    Optional:
      site (recommended if subject-level has site)

(3) Surrogate-aware table for Table S8 (optional):
    Provide ONE of:
      A) <root>/data/high_rate_epoch_debug_merged.csv
         - If it contains surr_*_mean columns, those are used directly.
         - If it contains surr_within_epoch_event_rate_mean, we use it.
         - If it contains BOTH surr_n_events_in_epochs_mean and surr_sum_epoch_dur_s_mean,
           we compute surr_within_epoch_event_rate_mean = num/den.
      B) <root>/data/rate_epoch_surrogate_subject_level.csv  (legacy name; accepted)
      C) a surrogate-long table with multiple rows per subject×freq and a surrogate-id column
         (the script will compute surrogate means per subject×freq).

Outputs
-------
<root>/results/tables/
  - Table_S7_temporal_clustering_metrics.csv
  - Table_S9_correlations.csv
  - Table_S8_surrogate_excess.csv            (only if surrogate input is found/provided)
<root>/results/figures/
  - Fig3bcde_high_rate_epoch_group_comparison.pdf

Statistics
----------
- Group comparisons (HC vs SZ): permutation test on mean difference (SZ - HC), two-sided.
- Effect size: Cohen's d (SZ - HC), pooled SD.
- Multiple comparisons: BH-FDR across frequencies WITHIN each metric.

"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests
from scipy import stats as spstats


# -------------------------
# Settings
# -------------------------
FREQ_ORDER = [80, 120, 160, 200, 240]
GROUPS = ["HC", "SZ"]
COLOR = {"HC": "#4C72B0", "SZ": "#C44E52"}

N_PERM_DEFAULT = 10000
SEED_DEFAULT = 0


# -------------------------
# Helpers
# -------------------------
def canon_subject(x) -> str:
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return s.strip()
    return f"NB_subject_{int(m.group(1))}"

def normalize_group_labels(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({"SCHIZOPHRENIA": "SZ", "SCZ": "SZ", "SC": "SZ"})
    x = x.replace({"HEALTHY": "HC", "CONTROL": "HC"})
    return x

def permutation_test_diff_means(x_hc: np.ndarray, x_sz: np.ndarray, n_perm: int, seed: int = 0) -> tuple[float, float]:
    """
    Two-sided permutation test:
      obs = mean(SZ) - mean(HC)
      p = proportion(|perm| >= |obs|), floored at 1/n_perm.
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
    return obs, p

def cohens_d(x_hc: np.ndarray, x_sz: np.ndarray) -> float:
    """
    Cohen's d (SZ - HC), pooled SD.
    """
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
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.mean(x_sz) - np.mean(x_hc)) / np.sqrt(pooled))

def mean_sem(x: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return (np.nan, np.nan, 0)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
    return (m, se, n)

def stars(p: float) -> str:
    if p is None or (isinstance(p, float) and (not np.isfinite(p))):
        return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""

def fmt_p_or_q(pv: float) -> str:
    if pv is None or (isinstance(pv, float) and (not np.isfinite(pv))):
        return ""
    if pv < 1e-4:
        return "<0.0001" + stars(pv)
    return f"{pv:.4g}{stars(pv)}"


# -------------------------
# Path discovery
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

def _resolve_path(root: Path, p: str | None) -> Path | None:
    if not p:
        return None
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = (root / pp).resolve()
    return pp if pp.exists() else None

def find_subject_csv(root: Path, user_path: str | None) -> Path:
    p = _resolve_path(root, user_path)
    if p is not None:
        return p
    for c in [root/"data"/"epoch_metrics_subject_freq.csv", root/"results"/"tables"/"epoch_metrics_subject_freq.csv"]:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find epoch_metrics_subject_freq.csv. Provide --subject-csv.")

def find_epoch_csv(root: Path, user_path: str | None) -> Path | None:
    p = _resolve_path(root, user_path)
    if p is not None:
        return p
    for c in [root/"data"/"epoch_metrics_epoch_level.csv", root/"results"/"tables"/"epoch_metrics_epoch_level.csv"]:
        if c.exists():
            return c
    return None

def find_S8_input_csv(root: Path, user_path: str | None) -> Path | None:
    p = _resolve_path(root, user_path)
    if p is not None:
        return p
    cands = [
        root/"data"/"high_rate_epoch_debug_merged.csv",
        root/"results"/"tables"/"high_rate_epoch_debug_merged.csv",
        # legacy / alternative names (accepted)
        root/"data"/"rate_epoch_surrogate_subject_level.csv",
        root/"results"/"tables"/"rate_epoch_surrogate_subject_level.csv",
    ]
    for c in cands:
        if c.exists():
            return c
    return None

def find_template(root: Path, name: str) -> Path | None:
    cands = [root/"templates"/name, root/"template"/name, root/"supp_tables"/name, root/"data"/name]
    try:
        here = Path(__file__).resolve()
        cands += [here.parent/name, here.parent.parent/name, here.parents[2]/name]
    except NameError:
        pass
    for c in cands:
        if c.exists():
            return c
    return None


# -------------------------
# Group attachment
# -------------------------
def attach_group(subject_df: pd.DataFrame, root: Path, group_csv: Path | None) -> pd.DataFrame:
    out = subject_df.copy()

    if "group" in out.columns:
        out["group"] = normalize_group_labels(out["group"])
        return out

    if "S_ID" in out.columns:
        sid = pd.to_numeric(out["S_ID"], errors="coerce")
        out["group"] = sid.map({1: "HC", 2: "SZ"})
        out["group"] = normalize_group_labels(out["group"])
        return out

    if group_csv is not None and group_csv.exists():
        gdf = pd.read_csv(group_csv).copy()
        if "subject" not in gdf.columns:
            for cand in ["ID", "id", "Subject", "SUBJECT"]:
                if cand in gdf.columns:
                    gdf["subject"] = gdf[cand]
                    break
        if "subject" not in gdf.columns:
            raise ValueError("group_csv must have 'subject' (or ID-like) column.")
        gdf["subject"] = gdf["subject"].map(canon_subject).astype(str)

        if "group" not in gdf.columns:
            if "S_ID" in gdf.columns:
                sid = pd.to_numeric(gdf["S_ID"], errors="coerce")
                gdf["group"] = sid.map({1: "HC", 2: "SZ"})
            else:
                raise ValueError("group_csv must have 'group' or 'S_ID'.")
        gdf["group"] = normalize_group_labels(gdf["group"])
        gdf = gdf[["subject", "group"]].drop_duplicates("subject")
        return out.merge(gdf, on="subject", how="left")

    # fallback to df_clean_expanded.csv
    dfc = root / "data" / "df_clean_expanded.csv"
    if dfc.exists():
        gdf = pd.read_csv(dfc).copy()
        if {"subject", "group"}.issubset(gdf.columns):
            gdf["subject"] = gdf["subject"].map(canon_subject).astype(str)
            gdf["group"] = normalize_group_labels(gdf["group"])
            gdf = gdf[gdf["group"].isin(GROUPS)][["subject", "group"]].drop_duplicates("subject")
            return out.merge(gdf, on="subject", how="left")

    raise RuntimeError("Could not attach group labels. Provide --group-csv or include group/S_ID in subject-level csv.")


# -------------------------
# Derived metrics (S7/S9 base)
# -------------------------
def add_subject_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - n_events_outside_epochs
      - outside_dur_s
      - outside_event_rate
      - within_epoch_event_rate  (duration-weighted)
    """
    out = df.copy()
    for c in ["freq","n_epochs","sum_epoch_dur_s","n_events","n_events_in_epochs","recording_len_s"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["n_events_outside_epochs"] = out["n_events"] - out["n_events_in_epochs"]
    out["outside_dur_s"] = (out["recording_len_s"] - out["sum_epoch_dur_s"]).clip(lower=0.0)

    out["outside_event_rate"] = np.where(
        out["outside_dur_s"] > 0,
        out["n_events_outside_epochs"] / out["outside_dur_s"],
        np.where((out["outside_dur_s"] == 0) & (out["n_events_outside_epochs"] == 0), 0.0, np.nan),
    )

    out["within_epoch_event_rate"] = np.where(
        out["sum_epoch_dur_s"] > 0,
        out["n_events_in_epochs"] / out["sum_epoch_dur_s"],
        np.where((out["sum_epoch_dur_s"] == 0) & (out["n_events_in_epochs"] == 0), 0.0, np.nan),
    )

    return out

def add_epoch_duration_stats(df: pd.DataFrame, epoch_csv: Path | None) -> pd.DataFrame:
    """
    Adds/overwrites:
      - mean_epoch_dur_s
      - median_epoch_dur_s
    Prefer epoch-level aggregation if epoch_csv exists and includes duration_s.
    Fallback: mean = sum_epoch_dur_s / n_epochs; median = NaN.
    """
    out = df.copy()
    out["mean_epoch_dur_s"] = pd.to_numeric(out.get("mean_epoch_dur_s", np.nan), errors="coerce")
    out["median_epoch_dur_s"] = pd.to_numeric(out.get("median_epoch_dur_s", np.nan), errors="coerce")

    if epoch_csv is not None and epoch_csv.exists():
        ep = pd.read_csv(epoch_csv).copy()
        need = {"subject","freq","duration_s"}
        if need.issubset(ep.columns):
            if "site" in ep.columns:
                ep["site"] = ep["site"].astype(str).str.lower()
            ep["subject"] = ep["subject"].map(canon_subject).astype(str)
            ep["freq"] = pd.to_numeric(ep["freq"], errors="coerce")
            ep["duration_s"] = pd.to_numeric(ep["duration_s"], errors="coerce")
            ep = ep[ep["freq"].isin(FREQ_ORDER)].dropna(subset=["duration_s"]).copy()

            grp_cols = ["subject","freq"] + (["site"] if "site" in ep.columns else [])
            g = (ep.groupby(grp_cols, as_index=False)["duration_s"]
                   .agg(mean_epoch_dur_s="mean", median_epoch_dur_s="median"))

            # merge using the most specific keys available
            if "site" in out.columns and "site" in g.columns:
                out["site"] = out["site"].astype(str).str.lower()
                out = out.merge(g, on=["site","subject","freq"], how="left", suffixes=("", "_ep"))
            else:
                out = out.merge(g.drop(columns=["site"], errors="ignore"),
                                on=["subject","freq"], how="left", suffixes=("", "_ep"))

            out["mean_epoch_dur_s"] = out.get("mean_epoch_dur_s_ep").combine_first(out["mean_epoch_dur_s"])
            out["median_epoch_dur_s"] = out.get("median_epoch_dur_s_ep").combine_first(out["median_epoch_dur_s"])
            out = out.drop(columns=["mean_epoch_dur_s_ep","median_epoch_dur_s_ep"], errors="ignore")

    fallback_mean = np.where(
        pd.to_numeric(out["n_epochs"], errors="coerce") > 0,
        pd.to_numeric(out["sum_epoch_dur_s"], errors="coerce") / pd.to_numeric(out["n_epochs"], errors="coerce"),
        np.nan
    )
    out["mean_epoch_dur_s"] = out["mean_epoch_dur_s"].where(np.isfinite(out["mean_epoch_dur_s"]), fallback_mean)
    out.loc[pd.to_numeric(out["n_epochs"], errors="coerce") <= 0, ["mean_epoch_dur_s","median_epoch_dur_s"]] = np.nan
    return out


# -------------------------
# Stats (generic)
# -------------------------
def compute_group_stats(df: pd.DataFrame, metric_col: str, metric_key: str, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    for f in FREQ_ORDER:
        sub = df[df["freq"] == f].copy()
        g = normalize_group_labels(sub["group"])
        x = pd.to_numeric(sub[metric_col], errors="coerce")

        x_hc = x[g == "HC"].to_numpy(float)
        x_sz = x[g == "SZ"].to_numpy(float)

        n_hc = int(np.isfinite(x_hc).sum())
        n_sz = int(np.isfinite(x_sz).sum())

        m_hc, se_hc, _ = mean_sem(x_hc)
        m_sz, se_sz, _ = mean_sem(x_sz)

        diff, p = permutation_test_diff_means(
            x_hc, x_sz, n_perm=n_perm,
            seed=seed + int(f) + (abs(hash(metric_key)) % 100000)
        )
        d = cohens_d(x_hc, x_sz)

        rows.append(dict(
            metric=metric_key, freq=int(f),
            n_HC=n_hc, n_SZ=n_sz,
            mean_HC=m_hc, sem_HC=se_hc,
            mean_SZ=m_sz, sem_SZ=se_sz,
            diff_SZ_minus_HC=diff,
            cohens_d=d,
            p_perm=p
        ))

    out = pd.DataFrame(rows)
    out["q_fdr"] = np.nan
    m = out["p_perm"].notna()
    if m.any():
        out.loc[m, "q_fdr"] = multipletests(out.loc[m, "p_perm"].values, method="fdr_bh")[1]
    return out


# -------------------------
# S8 (surrogate-excess)
# -------------------------
S8_SPECS = [
    dict(label="Excess clustered-epoch count",
         real_col="n_epochs",
         surr_mean_col="surr_n_epochs_mean",
         round=3),
    dict(label="Excess within-epoch ripple rate (events/s)",
         real_col="within_epoch_event_rate",
         surr_mean_col="surr_within_epoch_event_rate_mean",
         round=3),
    dict(label="Excess median epoch duration (s)",
         real_col="median_epoch_dur_s",
         surr_mean_col="surr_median_epoch_dur_s_mean",
         round=3),
]

def _guess_surr_id_col(df: pd.DataFrame) -> str | None:
    cands = ["surrogate_id","surr_id","surr_idx","iter","iteration","rep","repeat","draw","sim","sample"]
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in low:
            return low[k]
    return None

def build_S8_input_table(df_real: pd.DataFrame, S8_input_path: Path) -> pd.DataFrame:
    """
    Returns a table that contains (subject, freq, group) and surr_*_mean columns required by S8_SPECS.
    Supports:
      A) debug/merged table already containing surr_*_mean columns
      B) surrogate subject-level means table
      C) surrogate-long table with surrogate-id column (computes means per subject×freq)
    """
    df = pd.read_csv(S8_input_path).copy()
    if "subject" not in df.columns or "freq" not in df.columns:
        raise ValueError(f"S8 input must contain subject,freq. Got cols={list(df.columns)}")

    df["subject"] = df["subject"].map(canon_subject).astype(str)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df = df[df["freq"].isin(FREQ_ORDER)].copy()

    # attach group if missing
    if "group" in df.columns:
        df["group"] = normalize_group_labels(df["group"])
    else:
        gm = df_real[["subject","freq","group"]].drop_duplicates()
        df = df.merge(gm, on=["subject","freq"], how="left")

    # Case A: already has surr_*_mean columns
    has_any_surr_mean = any(c.startswith("surr_") and c.endswith("_mean") for c in df.columns)
    if has_any_surr_mean:
        # ensure surr_within_epoch_event_rate_mean exists if possible
        if "surr_within_epoch_event_rate_mean" not in df.columns:
            # (A1) build from components if both exist
            if {"surr_n_events_in_epochs_mean","surr_sum_epoch_dur_s_mean"}.issubset(df.columns):
                num = pd.to_numeric(df["surr_n_events_in_epochs_mean"], errors="coerce")
                den = pd.to_numeric(df["surr_sum_epoch_dur_s_mean"], errors="coerce")
                df["surr_within_epoch_event_rate_mean"] = np.where(den > 0, num / den, np.nan)
            # (A2) accept legacy surrogate column if present
            elif "surr_epoch_event_rate_in_epochs_mean" in df.columns:
                df["surr_within_epoch_event_rate_mean"] = pd.to_numeric(df["surr_epoch_event_rate_in_epochs_mean"], errors="coerce")
        return df

    # Case C: surrogate-long
    surr_id = _guess_surr_id_col(df)
    if surr_id is None:
        raise RuntimeError(
            "S8 input does not contain any surr_*_mean columns, and no surrogate-id column was found.\n"
            "Provide high_rate_epoch_debug_merged.csv / rate_epoch_surrogate_subject_level.csv, or a surrogate-long table."
        )

    key = ["subject","freq","group"]
    agg_map = {}
    # compute means for usable columns
    for rc in ["n_epochs","median_epoch_dur_s","n_events_in_epochs","sum_epoch_dur_s","within_epoch_event_rate"]:
        if rc in df.columns:
            agg_map[rc] = "mean"

    if not agg_map:
        raise RuntimeError("Surrogate-long table found, but no usable metric columns were found.")

    sur_mean = df.groupby(key, as_index=False).agg(agg_map)

    # construct surr_* means
    if "n_epochs" in sur_mean.columns:
        sur_mean["surr_n_epochs_mean"] = pd.to_numeric(sur_mean["n_epochs"], errors="coerce")
    if "median_epoch_dur_s" in sur_mean.columns:
        sur_mean["surr_median_epoch_dur_s_mean"] = pd.to_numeric(sur_mean["median_epoch_dur_s"], errors="coerce")

    if {"n_events_in_epochs","sum_epoch_dur_s"}.issubset(sur_mean.columns):
        num = pd.to_numeric(sur_mean["n_events_in_epochs"], errors="coerce")
        den = pd.to_numeric(sur_mean["sum_epoch_dur_s"], errors="coerce")
        sur_mean["surr_within_epoch_event_rate_mean"] = np.where(den > 0, num / den, np.nan)
    elif "within_epoch_event_rate" in sur_mean.columns:
        sur_mean["surr_within_epoch_event_rate_mean"] = pd.to_numeric(sur_mean["within_epoch_event_rate"], errors="coerce")

    keep = ["subject","freq","group","surr_n_epochs_mean","surr_median_epoch_dur_s_mean","surr_within_epoch_event_rate_mean"]
    sur_mean = sur_mean[[c for c in keep if c in sur_mean.columns]]
    return sur_mean

def compute_S8_surrogate_excess_stats(df_real: pd.DataFrame, df_S8in: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    """
    For each spec, compute excess = real - surr_mean and compare SZ vs HC.
    BH-FDR across frequencies WITHIN metric_label.
    """
    rows = []
    for spec in S8_SPECS:
        rc = spec["real_col"]
        sc = spec["surr_mean_col"]
        if rc not in df_real.columns or sc not in df_S8in.columns:
            continue

        merged = df_real[["subject","freq","group",rc]].merge(
            df_S8in[["subject","freq","group",sc]], on=["subject","freq","group"], how="left"
        )
        merged["excess"] = pd.to_numeric(merged[rc], errors="coerce") - pd.to_numeric(merged[sc], errors="coerce")

        for f in FREQ_ORDER:
            sub = merged[merged["freq"] == int(f)].copy()
            g = normalize_group_labels(sub["group"])
            x = pd.to_numeric(sub["excess"], errors="coerce")

            x_hc = x[g == "HC"].to_numpy(float)
            x_sz = x[g == "SZ"].to_numpy(float)

            n_hc = int(np.isfinite(x_hc).sum())
            n_sz = int(np.isfinite(x_sz).sum())

            mean_hc = float(np.nanmean(x_hc)) if n_hc else np.nan
            mean_sz = float(np.nanmean(x_sz)) if n_sz else np.nan
            sd_hc = float(np.nanstd(x_hc, ddof=1)) if n_hc >= 2 else np.nan
            sd_sz = float(np.nanstd(x_sz, ddof=1)) if n_sz >= 2 else np.nan

            diff, p = permutation_test_diff_means(
                x_hc, x_sz, n_perm=n_perm,
                seed=seed + int(f) + (abs(hash(spec["label"])) % 100000)
            )
            d = cohens_d(x_hc, x_sz)

            rows.append(dict(
                metric_label=spec["label"],
                freq=int(f),
                n_HC=n_hc, n_SZ=n_sz,
                mean_excess_HC=mean_hc,
                mean_excess_SZ=mean_sz,
                sd_excess_HC=sd_hc,
                sd_excess_SZ=sd_sz,
                diff_excess_SZ_minus_HC=diff,
                cohens_d=d,
                p_perm=p
            ))

    out = pd.DataFrame(rows)
    if out.empty:
        out["q_fdr"] = []
        return out

    out["q_fdr"] = np.nan
    for m, subm in out.groupby("metric_label", dropna=False):
        mask = subm["p_perm"].notna()
        if mask.any():
            out.loc[subm.index[mask], "q_fdr"] = multipletests(subm.loc[mask, "p_perm"].values, method="fdr_bh")[1]
    return out


# -------------------------
# Table S9 (correlations)
# -------------------------
def build_table_S9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two relationship blocks:
      1) within_epoch_event_rate vs outside_event_rate
      2) n_epochs vs n_events_outside_epochs
    Spearman correlations computed per freq × group.
    BH-FDR across frequencies WITHIN block × group.
    """
    corr_rows = []
    for freq in FREQ_ORDER:
        subf = df[df["freq"] == int(freq)].copy()
        for grp in GROUPS:
            subg = subf[subf["group"] == grp].copy()

            # block1
            x = pd.to_numeric(subg["within_epoch_event_rate"], errors="coerce")
            y = pd.to_numeric(subg["outside_event_rate"], errors="coerce")
            xy = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(xy) >= 3:
                rho, p = spstats.spearmanr(xy["x"].to_numpy(float), xy["y"].to_numpy(float))
            else:
                rho, p = np.nan, np.nan
            corr_rows.append(dict(
                block="rho_withinRate_vs_outsideRate",
                freq=int(freq), group=grp,
                n=int(len(xy)),
                rho=float(rho) if np.isfinite(rho) else np.nan,
                p=float(p) if np.isfinite(p) else np.nan
            ))

            # block2
            x2 = pd.to_numeric(subg["n_epochs"], errors="coerce")
            y2 = pd.to_numeric(subg["n_events_outside_epochs"], errors="coerce")
            xy2 = pd.DataFrame({"x": x2, "y": y2}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(xy2) >= 3:
                rho2, p2 = spstats.spearmanr(xy2["x"].to_numpy(float), xy2["y"].to_numpy(float))
            else:
                rho2, p2 = np.nan, np.nan
            corr_rows.append(dict(
                block="rho_nEpochs_vs_outsideCount",
                freq=int(freq), group=grp,
                n=int(len(xy2)),
                rho=float(rho2) if np.isfinite(rho2) else np.nan,
                p=float(p2) if np.isfinite(p2) else np.nan
            ))

    df_S9 = pd.DataFrame(corr_rows)
    df_S9["q_fdr"] = np.nan
    for block in df_S9["block"].unique():
        for grp in GROUPS:
            msk = (df_S9["block"] == block) & (df_S9["group"] == grp) & df_S9["p"].notna()
            if msk.any():
                df_S9.loc[msk, "q_fdr"] = multipletests(df_S9.loc[msk, "p"], method="fdr_bh")[1]

    df_S9["p_str"] = df_S9["p"].apply(fmt_p_or_q)
    df_S9["q_str"] = df_S9["q_fdr"].apply(lambda q: fmt_p_or_q(float(q)) if q is not None and np.isfinite(q) else "")
    return df_S9


# -------------------------
# Plot (Fig3bcde)
# -------------------------
def plot_bcde(df_plot: pd.DataFrame, out_pdf: Path):
    plot_metrics = [
        ("n_epochs", "Fig3b: Clustered epoch count (per 300 s)"),
        ("within_epoch_event_rate", "Fig3c: Within-epoch ripple rate (events/s)"),
        ("mean_epoch_dur_s", "Fig3d: Mean epoch duration (s)"),
        ("outside_event_rate", "Fig3e: Outside-epoch ripple rate (events/s)"),
    ]
    fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(7.4, 3.0 * len(plot_metrics)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (mkey, ylab) in zip(axes, plot_metrics):
        sub = df_plot[df_plot["metric"] == mkey].set_index("freq").reindex(FREQ_ORDER).reset_index()
        x = np.arange(len(FREQ_ORDER))
        ax.errorbar(x, sub["mean_HC"], yerr=sub["sem_HC"], fmt="-o", capsize=3, label="HC", color=COLOR["HC"])
        ax.errorbar(x, sub["mean_SZ"], yerr=sub["sem_SZ"], fmt="-o", capsize=3, label="SZ", color=COLOR["SZ"])
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)

        y_top = np.nanmax(np.r_[sub["mean_HC"] + sub["sem_HC"], sub["mean_SZ"] + sub["sem_SZ"]])
        if not np.isfinite(y_top):
            y_top = ax.get_ylim()[1]
        for i, f in enumerate(FREQ_ORDER):
            q = sub.loc[sub["freq"] == int(f), "q_fdr"]
            if len(q) == 1 and np.isfinite(q.iloc[0]) and float(q.iloc[0]) < 0.05:
                star = "***" if float(q.iloc[0]) < 0.001 else ("**" if float(q.iloc[0]) < 0.01 else "*")
                ax.text(i, y_top * 1.05, star, ha="center", va="bottom", fontsize=12, fontweight="bold")

    axes[-1].set_xticks(np.arange(len(FREQ_ORDER)))
    axes[-1].set_xticklabels([str(f) for f in FREQ_ORDER])
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].legend(frameon=False, loc="upper right")

    fig.suptitle("Fig3bcde: Temporal clustering metrics (HC vs SZ)", y=1.01, fontsize=13)
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
# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--subject-csv", type=str, default=None)
    ap.add_argument("--epoch-csv", type=str, default=None)
    ap.add_argument("--group-csv", type=str, default=None)

    # S8
    ap.add_argument("--S8-input-csv", type=str, default=None)

    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()

    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_tab = results_root / "tables"
    out_fig = results_root / "figures"

    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    
    subject_csv = find_subject_csv(root, args.subject_csv)
    epoch_csv = find_epoch_csv(root, args.epoch_csv)

    # ---- Load REAL subject-level ----
    df = pd.read_csv(subject_csv).copy()
    need = {"subject", "freq", "n_epochs", "sum_epoch_dur_s", "n_events", "n_events_in_epochs", "recording_len_s"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"epoch_metrics_subject_freq.csv missing columns: {sorted(miss)}")

    df["subject"] = df["subject"].map(canon_subject).astype(str)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df = df[df["freq"].isin(FREQ_ORDER)].copy()

    group_csv = _resolve_path(root, args.group_csv)
    df = attach_group(df, root, group_csv)
    df["group"] = normalize_group_labels(df["group"])
    df = df[df["group"].isin(GROUPS)].copy()

    # ---- Derived metrics ----
    df = add_subject_level_metrics(df)
    df = add_epoch_duration_stats(df, epoch_csv)

    # =========================
    # Fig3b/c/d/e stats + combined figure
    # =========================
    df_b = compute_group_stats(df, "n_epochs", "n_epochs", args.n_perm, args.seed)
    df_c = compute_group_stats(df, "within_epoch_event_rate", "within_epoch_event_rate", args.n_perm, args.seed)
    df_d = compute_group_stats(df, "mean_epoch_dur_s", "mean_epoch_dur_s", args.n_perm, args.seed)
    df_e = compute_group_stats(df, "outside_event_rate", "outside_event_rate", args.n_perm, args.seed)

    df_plot = pd.concat([df_b, df_c, df_d, df_e], ignore_index=True)
    plot_bcde(df_plot, out_fig / "Fig3bcde_high_rate_epoch_group_comparison.pdf")

    # =========================
    # Table S7
    # =========================
    df_med = compute_group_stats(df, "median_epoch_dur_s", "median_epoch_dur_s", args.n_perm, args.seed)

    df_S7 = pd.concat([
        df_b,
        df_d,     # ← ここが置換点
        df_med,
        df_c,
        df_e,
    ], ignore_index=True)
    out_S7 = out_tab / "Table_S7_temporal_clustering_metrics.csv"
    df_S7.to_csv(out_S7, index=False)

    # =========================
    # Table S9
    # =========================
    df_S9 = build_table_S9(df)
    out_S9 = out_tab / "Table_S9_correlations.csv"
    df_S9.to_csv(out_S9, index=False)

  

    # =========================
    # Table S8 (optional)
    # =========================
    S8_in_path = find_S8_input_csv(root, args.S8_input_csv)
    if S8_in_path is None:
        df_S8 = None
        print("[INFO] S8 input not found; Table S8 will be skipped.")
    else:
        df_S8in = build_S8_input_table(df_real=df, S8_input_path=S8_in_path)
        df_S8 = compute_S8_surrogate_excess_stats(df_real=df, df_S8in=df_S8in, n_perm=args.n_perm, seed=args.seed)
        out_S8 = out_tab / "Table_S8_surrogate_excess.csv"
        df_S8.to_csv(out_S8, index=False)

    

    print("[OK] Written:")
    print(" -", out_S7)
    print(" -", out_S9)
    print(" -", out_fig / "Fig3bcde_high_rate_epoch_group_comparison.pdf")
    if S8_in_path is not None:
        print(" -", out_tab / "Table_S8_surrogate_excess.csv")
    print("[INFO] Subject-level input:", subject_csv)
    print("[INFO] Epoch-level input:", epoch_csv if epoch_csv else "(not provided)")

if __name__ == "__main__":
    main()
