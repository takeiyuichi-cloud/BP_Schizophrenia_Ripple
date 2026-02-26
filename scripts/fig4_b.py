#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 4b (SZ only): Clustered ripple dynamics and positive symptoms
==================================================================

This script reproduces the Fig. 4b-style panel set (SZ only), testing whether
positive symptom severity is associated with:
  (i)   number of temporally clustered (high-rate) ripple epochs
  (ii)  number of ripple events occurring outside clustered epochs
  (iii) within-epoch ripple rate (duration-weighted)

Input tables (derived / anonymized; no raw MEG):
  <root>/data/public_clinical_subject_level.csv
  <root>/data/anon_id_map.csv   (or legacy: private_anon_id_map.csv)
  <root>/data/epoch_metrics_subject_freq.csv

Outputs:
  <root>/results/figures/fig4b_clustered_SWRS_and_positive_symptoms.pdf

Reproducibility notes:
- No absolute paths.
- Repo root is detected from --root / env PROJECT_ROOT / CWD heuristics.
- Avoids pandas groupby.apply deprecation by using groupby.agg.
- Explicitly sets NB dispersion alpha=1.0 to avoid statsmodels warnings.

Diagnostics (added):
- Prints a compact "N flow" to explain Fig4a vs Fig4b N differences.
- Lists reason counts for why subjects drop (ID map / epoch metrics / missing outcomes).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -------------------------
# Analysis settings
# -------------------------
FREQS_USE = [80, 120, 160, 200, 240]
COVARS = ["age", "sex", "JART", "sleepiness_pre", "antipsychotics"]

# Clinical predictors (one model per predictor).
# NOTE: "PANSS_pasological" is a legacy project column name.
PRED_MAP = [
    ("PANSS_positive", "PANSS POS"),
    ("PANSS_negative", "PANSS NEG"),
    ("PANSS_pasological", "PANSS GEN"),
    ("GAF", "GAF"),
]

PLOT_PRED = "PANSS_positive"
OUTFIG_NAME = "fig4b_clustered_SWRS_and_positive_symptoms.pdf"


# -------------------------
# Root/path helpers
# -------------------------
def _guess_repo_root(start: Path) -> Path:
    """
    Heuristic repo-root detection for Code Ocean and local runs.

    Priority:
      1) directory containing run_all.py
      2) directory containing both data/ and scripts/
      3) directory containing data/
      4) fallback: start
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "run_all.py").exists():
            return p
    for p in [start] + list(start.parents):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    for p in [start] + list(start.parents):
        if (p / "data").exists():
            return p
    return start

def get_root(cli_root: str | None = None) -> Path:
    """
    Priority:
      1) CLI --root
      2) env PROJECT_ROOT
      3) auto-detect from CWD
    """
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    env = os.environ.get("PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _guess_repo_root(Path.cwd())


# -------------------------
# General helpers
# -------------------------
def canon_subject(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return f"NB_subject_{int(m.group(1))}" if m else s

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _dropna(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols, how="any").copy()

def mode_or_median(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return np.nan
    if s2.nunique() <= 3:
        try:
            return s2.mode().iloc[0]
        except Exception:
            return float(s2.median())
    return float(s2.median())

def _stars(q: float) -> str:
    if q is None or (isinstance(q, float) and (not np.isfinite(q))):
        return ""
    if q < 1e-4: return "****"
    if q < 1e-3: return "***"
    if q < 1e-2: return "**"
    if q < 5e-2: return "*"
    return ""

def _resolve_panss_gen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Support both:
      - PANSS_pasological (legacy)
      - PANSS_general (preferred name)
    """
    df = df.copy()
    if "PANSS_pasological" not in df.columns and "PANSS_general" in df.columns:
        df["PANSS_pasological"] = df["PANSS_general"]
    return df


# -------------------------
# Model fits
# -------------------------
def fit_nb_one_pred(D: pd.DataFrame, ycol: str, pred_col: str, covars: list[str], site_col: str, offset_col: str):
    """
    Negative Binomial GLM with log link, alpha fixed to 1.0 for reproducibility.
    """
    fam = sm.families.NegativeBinomial(alpha=1.0, link=sm.families.links.Log())
    formula = f"{ycol} ~ {pred_col} + events_sum"
    if covars:
        formula += " + " + " + ".join(covars)
    formula += f" + C({site_col})"
    off = np.log(pd.to_numeric(D[offset_col], errors="coerce").astype(float))
    fit = smf.glm(formula=formula, data=D, family=fam, offset=off).fit(cov_type="HC3")

    beta = float(fit.params[pred_col])
    p = float(fit.pvalues[pred_col])
    se = float(fit.bse[pred_col])

    irr = float(np.exp(beta))
    irr_lo = float(np.exp(beta - 1.96 * se))
    irr_hi = float(np.exp(beta + 1.96 * se))
    return fit, dict(p=p, IRR=irr, IRR_lo=irr_lo, IRR_hi=irr_hi)

def fit_ols_one_pred(D: pd.DataFrame, ycol: str, pred_col: str, covars: list[str], site_col: str):
    formula = f"{ycol} ~ {pred_col} + events_sum"
    if covars:
        formula += " + " + " + ".join(covars)
    formula += f" + C({site_col})"
    fit = smf.ols(formula=formula, data=D).fit(cov_type="HC3")

    beta = float(fit.params[pred_col])
    p = float(fit.pvalues[pred_col])
    se = float(fit.bse[pred_col])
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    return fit, dict(p=p, beta=beta, beta_lo=ci_lo, beta_hi=ci_hi)

def pred_count(fit, xname: str, xgrid: np.ndarray, fixed: dict, minutes_val: float):
    P = pd.DataFrame({xname: xgrid})
    for k, v in fixed.items():
        P[k] = v
    sf = fit.get_prediction(P, offset=np.log(np.full_like(xgrid, minutes_val))).summary_frame(alpha=0.05)
    return sf["mean"].to_numpy(float), sf["mean_ci_lower"].to_numpy(float), sf["mean_ci_upper"].to_numpy(float)

def pred_ols(fit, xname: str, xgrid: np.ndarray, fixed: dict):
    P = pd.DataFrame({xname: xgrid})
    for k, v in fixed.items():
        P[k] = v
    sf = fit.get_prediction(P).summary_frame(alpha=0.05)
    return sf["mean"].to_numpy(float), sf["mean_ci_lower"].to_numpy(float), sf["mean_ci_upper"].to_numpy(float)


# -------------------------
# Diagnostics: N flow
# -------------------------
def _print_n_flow(
    *,
    clin_raw: pd.DataFrame,
    clin_sz: pd.DataFrame,
    clin_fig4a_usable: pd.DataFrame,
    mp: pd.DataFrame,
    rate: pd.DataFrame,
    agg: pd.DataFrame,
    df_merged: pd.DataFrame,
):
    """
    Print a compact inclusion flow to explain Fig4a vs Fig4b N differences.
    """
    def _nuniq(x: pd.Series) -> int:
        return int(pd.Series(x).dropna().astype(str).nunique())

    # sets over anon_id for Fig4a-like usable cohort vs Fig4b merged cohort
    set_fig4a = set(clin_fig4a_usable["anon_id"].astype(str))
    set_fig4b = set(df_merged["anon_id"].astype(str))

    dropped = set_fig4a - set_fig4b

    mp_anon = set(mp["anon_id"].astype(str))
    # map anon_id -> subject (if possible)
    mp_idx = mp.set_index("anon_id")["subject"].to_dict()

    # subjects present in epoch metrics
    rate_subjects = set(rate["subject"].map(canon_subject).astype(str))

    # reasons
    missing_in_map = {a for a in dropped if a not in mp_anon}

    missing_epoch = set()
    for a in dropped:
        if a in missing_in_map:
            continue
        s = mp_idx.get(str(a), None)
        if s is None:
            missing_epoch.add(a)
            continue
        if canon_subject(s) not in rate_subjects:
            missing_epoch.add(a)

    # subjects that reach merge but have NA outcomes
    # (this is evaluated within the merged table itself)
    # Note: this reason applies to those IN fig4b set; for dropped, it is "not merged",
    # so outcome-wise NA is tracked separately.
    na_within = int(df_merged["within_epoch_rate"].isna().sum())
    na_clustered = int(df_merged["clustered_epochs"].isna().sum())
    na_outside = int(df_merged["outside_ripple_count"].isna().sum())

    print("\n=== Fig4a vs Fig4b inclusion diagnostics ===")
    print(f"[A] clinical raw rows: {len(clin_raw)} (anon_id unique={_nuniq(clin_raw.get('anon_id', []))})")
    print(f"[B] clinical SZ-only rows: {len(clin_sz)} (anon_id unique={_nuniq(clin_sz['anon_id'])})")
    print(f"[C] Fig4a-usable (PANSS_positive model) rows: {len(clin_fig4a_usable)} (anon_id unique={_nuniq(clin_fig4a_usable['anon_id'])})")
    print(f"[D] ID map rows: {len(mp)} (anon_id unique={_nuniq(mp['anon_id'])}, subject unique={_nuniq(mp['subject'])})")
    print(f"[E] epoch metrics rows: {len(rate)} (subject unique={_nuniq(rate['subject'])})")
    print(f"[F] pooled epoch metrics (agg) rows: {len(agg)} (anon_id unique={_nuniq(agg['anon_id'])})")
    print(f"[G] merged analysis table rows: {len(df_merged)} (anon_id unique={_nuniq(df_merged['anon_id'])})")

    print("\nDropped from Fig4a-usable to Fig4b-merged:")
    print(f" - dropped anon_id count: {len(dropped)}")
    print(f"   * missing in anon_id_map: {len(missing_in_map)}")
    print(f"   * missing epoch metrics:  {len(missing_epoch)}")
    other = len(dropped) - len(missing_in_map) - len(missing_epoch)
    print(f"   * other merge mismatches: {max(0, other)}")

    print("\nWithin Fig4b merged table: NA outcome counts (these reduce N per outcome/model):")
    print(f" - clustered_epochs NA:      {na_clustered}")
    print(f" - outside_ripple_count NA:  {na_outside}")
    print(f" - within_epoch_rate NA:     {na_within}")
    print("================================================\n")


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
    ap.add_argument("--outfig", type=str, default=None, help=f"Output filename (default: {OUTFIG_NAME})")
    ap.add_argument("--id-map", type=str, default=None,
                    help="ID map filename (default: data/anon_id_map.csv; fallback: data/private_anon_id_map.csv)")
    ap.add_argument("--print-n-flow", action="store_true",
                    help="Print inclusion diagnostics (Fig4a vs Fig4b) to stdout")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    data_dir = root / "data"

    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_fig_dir = results_root / "figures"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    
    p_clin = data_dir / "public_clinical_subject_level.csv"

    # ID map: prefer anon_id_map.csv, fallback to legacy private_anon_id_map.csv
    if args.id_map:
        p_map = Path(args.id_map).expanduser()
        if not p_map.is_absolute():
            p_map = (data_dir / p_map).resolve()
        else:
            p_map = p_map.resolve()
    else:
        p_map = data_dir / "anon_id_map.csv"
        if not p_map.exists():
            p_map = data_dir / "private_anon_id_map.csv"

    p_rate = data_dir / "epoch_metrics_subject_freq.csv"

    for p in [p_clin, p_map, p_rate]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    # --- clinical ---
    clin_raw = pd.read_csv(p_clin)
    clin = _resolve_panss_gen(clin_raw)

    req = {"anon_id", "site_public", "minutes_sum", "events_sum"}
    miss = req - set(clin.columns)
    if miss:
        raise ValueError(f"{p_clin.name} missing required columns: {sorted(miss)}")

    num_cols = ["minutes_sum", "events_sum"] + COVARS + [c for c, _ in PRED_MAP]
    clin = _safe_numeric(clin, [c for c in num_cols if c in clin.columns])
    clin["site_public"] = clin["site_public"].astype(str)
    clin["anon_id"] = clin["anon_id"].astype(str)

    # SZ-only filter if group is present
    clin_sz = clin.copy()
    if "group" in clin_sz.columns:
        clin_sz["group"] = clin_sz["group"].astype(str).str.upper().replace({"SC": "SZ", "SCZ": "SZ", "SCHIZOPHRENIA": "SZ"})
        clin_sz = clin_sz[clin_sz["group"].isin(["SZ"])].copy()

    # Define Fig4a-like usable cohort (PANSS_positive model spec)
    fig4a_need = ["site_public", "events_sum", "minutes_sum", "PANSS_positive"] + [c for c in COVARS if c in clin_sz.columns]
    clin_fig4a_usable = clin_sz.replace([np.inf, -np.inf], np.nan).dropna(subset=fig4a_need).copy()
    clin_fig4a_usable = clin_fig4a_usable[pd.to_numeric(clin_fig4a_usable["minutes_sum"], errors="coerce") > 0].copy()

    # --- subject-to-anon_id map (anonymized linkage) ---
    mp = pd.read_csv(p_map)
    if not {"subject", "anon_id"}.issubset(mp.columns):
        raise ValueError("ID map CSV must have columns: subject, anon_id")
    mp = mp.copy()
    mp["subject"] = mp["subject"].map(canon_subject)
    mp["anon_id"] = mp["anon_id"].astype(str)

    # --- epoch metrics (subject×freq) ---
    rate = pd.read_csv(p_rate)
    need_rate = {
        "subject", "freq",
        "n_epochs", "n_events", "n_events_in_epochs",
        "sum_epoch_dur_s",
        "within_epoch_event_rate",
    }
    miss = need_rate - set(rate.columns)
    if miss:
        raise ValueError(f"{p_rate.name} missing columns: {sorted(miss)}")

    rate = rate.copy()
    rate["subject"] = rate["subject"].map(canon_subject)
    rate["freq"] = pd.to_numeric(rate["freq"], errors="coerce").astype(int)

    for c in ["n_epochs", "n_events", "n_events_in_epochs", "sum_epoch_dur_s", "within_epoch_event_rate"]:
        rate[c] = pd.to_numeric(rate[c], errors="coerce")

    rate = rate[rate["freq"].isin(FREQS_USE)].copy()
    rate["n_events_outside_epochs"] = rate["n_events"] - rate["n_events_in_epochs"]

    # ---------------------------------------------------------
    # Pool outcomes across frequencies WITHOUT groupby.apply
    # ---------------------------------------------------------
    w = pd.to_numeric(rate["sum_epoch_dur_s"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(rate["within_epoch_event_rate"], errors="coerce").to_numpy(float)
    good = np.isfinite(w) & np.isfinite(y) & (w > 0)

    rate["_w"] = np.where(good, w, 0.0)
    rate["_wy"] = np.where(good, w * y, 0.0)

    agg_subject = (rate.groupby("subject", as_index=False)
                     .agg(
                         clustered_epochs=("n_epochs", "sum"),
                         outside_ripple_count=("n_events_outside_epochs", "sum"),
                         _w_sum=("_w", "sum"),
                         _wy_sum=("_wy", "sum"),
                     ))

    agg_subject["within_epoch_rate"] = np.where(
        agg_subject["_w_sum"] > 0,
        agg_subject["_wy_sum"] / agg_subject["_w_sum"],
        np.nan
    )
    agg_subject = agg_subject.drop(columns=["_w_sum", "_wy_sum"])

    # attach anon_id
    agg = agg_subject.merge(mp, on="subject", how="left").dropna(subset=["anon_id"]).copy()
    agg["anon_id"] = agg["anon_id"].astype(str)

    # merge to clinical
    df = clin_sz.merge(agg, on="anon_id", how="inner")
    df = df[pd.to_numeric(df["minutes_sum"], errors="coerce") > 0].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # diagnostics (optional)
    if args.print_n_flow:
        _print_n_flow(
            clin_raw=clin_raw,
            clin_sz=clin_sz,
            clin_fig4a_usable=clin_fig4a_usable,
            mp=mp,
            rate=rate,
            agg=agg,
            df_merged=df,
        )

    covars = [c for c in COVARS if c in df.columns]
    preds = [(c, lab) for c, lab in PRED_MAP if c in df.columns]
    if PLOT_PRED not in df.columns:
        raise ValueError(f"Required predictor '{PLOT_PRED}' not found in clinical table.")

    outcomes = [
        ("clustered_epochs", "High-rate epochs", "count"),
        ("outside_ripple_count", "Outside-epoch events", "count"),
        ("within_epoch_rate", "Within-epoch rate", "cont"),
    ]

    # Fit models for each outcome × predictor; BH-FDR across predictors within each outcome
    fitted = {}
    per_outcome_n = {}

    for out_col, out_title, out_type in outcomes:
        tmp_infos, tmp_fits = [], []
        for pred_col, pred_label in preds:
            cols_need = (["site_public", "minutes_sum", "events_sum", out_col, pred_col] + covars) if out_type == "count" \
                        else (["site_public", "events_sum", out_col, pred_col] + covars)

            Dm = _dropna(df, cols_need)
            if out_type == "count":
                Dm = Dm[Dm["minutes_sum"] > 0].copy()
            if Dm.empty:
                continue

            if out_type == "count":
                fit, info = fit_nb_one_pred(Dm, out_col, pred_col, covars, "site_public", "minutes_sum")
                info["label"] = f"IRR={info['IRR']:.3f}"
            else:
                fit, info = fit_ols_one_pred(Dm, out_col, pred_col, covars, "site_public")
                info["label"] = f"β={info['beta']:.3f}"

            tmp_infos.append((pred_col, info))
            tmp_fits.append((pred_col, fit, Dm, pred_label))

        if not tmp_infos:
            continue

        pvals = np.array([info["p"] for _, info in tmp_infos], float)
        q = multipletests(pvals, method="fdr_bh")[1]

        fitted[out_col] = {}
        for (pred_col, info), qi, (_pred_col2, fit, Dm, pred_label) in zip(tmp_infos, q, tmp_fits):
            info2 = dict(info)
            info2["q_FDR"] = float(qi)
            info2["pred_label"] = pred_label
            fitted[out_col][pred_col] = dict(fit=fit, D=Dm, info=info2)

        # outcome-wise usable N for the plotted predictor (PANSS_positive), if available
        if PLOT_PRED in fitted[out_col]:
            per_outcome_n[out_col] = int(fitted[out_col][PLOT_PRED]["D"]["anon_id"].nunique())

    for out_col, _, _ in outcomes:
        if out_col not in fitted or PLOT_PRED not in fitted[out_col]:
            raise RuntimeError(f"Could not fit outcome='{out_col}' with predictor='{PLOT_PRED}'. Check missingness.")

    # Plot (PANSS_positive only)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), dpi=160)

    for ax, (out_col, out_title, out_type) in zip(axes, outcomes):
        pack = fitted[out_col][PLOT_PRED]
        fit, Dm, info = pack["fit"], pack["D"], pack["info"]

        x = pd.to_numeric(Dm[PLOT_PRED], errors="coerce").to_numpy(float)
        yv = pd.to_numeric(Dm[out_col], errors="coerce").to_numpy(float)
        ax.scatter(x, yv, s=18, alpha=0.85, color="black")

        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        x_grid = np.linspace(x_min, x_max, 180) if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max else np.array([x_min])

        site_mode = Dm["site_public"].mode().iloc[0]
        fixed = {"site_public": site_mode, "events_sum": float(mode_or_median(Dm["events_sum"]))}
        for c in covars:
            fixed[c] = mode_or_median(Dm[c])

        if out_type == "count":
            minutes_fixed = float(mode_or_median(Dm["minutes_sum"]))
            yhat, ylo, yhi = pred_count(fit, PLOT_PRED, x_grid, fixed, minutes_val=minutes_fixed)
        else:
            yhat, ylo, yhi = pred_ols(fit, PLOT_PRED, x_grid, fixed)

        ax.plot(x_grid, yhat, lw=2, color="black")
        ax.fill_between(x_grid, ylo, yhi, color="black", alpha=0.12, linewidth=0)

        qv = info.get("q_FDR", np.nan)
        qtxt = f"FDR-p={qv:.4g}{_stars(qv)}" if np.isfinite(qv) else "FDR-p=NA"
        n_txt = f"N={per_outcome_n.get(out_col, 'NA')}"
        ax.text(0.02, 0.98, f"{info['label']}, {qtxt}\n{n_txt}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10)

        ax.set_title(out_title, fontsize=11)
        ax.set_xlabel("PANSS positive")

        if out_col == "within_epoch_rate":
            ax.set_ylabel("Within-epoch rate (events/s)")
        elif out_col == "clustered_epochs":
            ax.set_ylabel("Epoch count")
        else:
            ax.set_ylabel("Event count")
        ax.grid(alpha=0.25)

    fig.suptitle("(b) Clustered ripple dynamics and positive symptoms (SZ only)", y=1.04, fontsize=14)

    out_name = args.outfig if args.outfig else OUTFIG_NAME
    out_pdf = out_fig_dir / out_name
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] root :", root)
    print("[OK] data :", data_dir)
    print("[OK] N rows (merged table):", int(df.shape[0]))
    print("[OK] N subjects (merged anon_id unique):", int(df["anon_id"].nunique()))
    print("[OK] N subjects per outcome (PANSS_positive model):", {k: int(v) for k, v in per_outcome_n.items()})
    print("[OK] saved:", out_pdf)
    print("[HINT] To print Fig4a vs Fig4b inclusion diagnostics, run with: --print-n-flow")


if __name__ == "__main__":
    main()
