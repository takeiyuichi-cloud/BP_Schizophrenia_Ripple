#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 2a: Spatial distribution maps of voxel-normalized ripple events (NIfTI)
============================================================================

This script renders a grid figure from group-averaged NIfTI maps:
  - Regions: Hippocampus, Cortex
  - Groups: HC, SZ
  - Frequencies: 80, 120, 160, 200, 240 Hz
Each cell shows two views (sagittal x + axial z).

Inputs (default):
  <root>/data/fig2a/
    Hippocampus_HC_80Hz.nii.gz
    Hippocampus_SZ_80Hz.nii.gz
    ...
    Cortex_HC_240Hz.nii.gz
    Cortex_SZ_240Hz.nii.gz

Outputs (default):
  <root>/results/figures/Fig2a_spatial_distribution.pdf

Notes for public / reproducible use:
- This script does NOT require raw MEG data.
- It visualizes group-level maps only.
- nilearn is required. Install via: pip install nilearn

Robustness:
- Background MNI template is optional. If nilearn cannot load a local template
  (e.g., offline environment), the script will plot without background.

"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# nilearn is used for neuroimaging plots
from nilearn import plotting
from nilearn import image as nimg


# -------------------------
# Defaults
# -------------------------
FREQS_DEFAULT = [80, 120, 160, 200, 240]
REGIONS_DEFAULT = ["Hippocampus", "Cortex"]
GROUPS_DEFAULT = ["HC", "SZ"]

# Absolute display thresholds (visual only; does not affect any statistics)
THRESH_ABS_DEFAULT = {"Hippocampus": 1.0, "Cortex": 0.6}

# vmax estimation percentile per region (shared scaling within region)
VMAX_PCTL_DEFAULT = 99.5

# colormap (visual only)
CMAP_DEFAULT = "hot"


# -------------------------
# Helpers
# -------------------------
def expected_path(in_dir: Path, region: str, group: str, freq: int) -> Path:
    return in_dir / f"{region}_{group}_{freq}Hz.nii.gz"


def try_load_mni_template() -> object | None:
    """
    Try to load an MNI background image.
    In some environments, nilearn may attempt to download templates if not present.
    To keep Code Ocean / offline runs robust, fall back to bg_img=None.
    """
    try:
        from nilearn.datasets import load_mni152_template
        return load_mni152_template()
    except Exception:
        return None


def region_shared_vmax(
    in_dir: Path,
    region: str,
    groups: list[str],
    freqs: list[int],
    pctl: float,
) -> float | None:
    """
    Compute a shared vmax for a region across all groups+freqs
    using an upper percentile over positive finite voxels.
    """
    vals = []
    for g in groups:
        for f in freqs:
            p = expected_path(in_dir, region, g, f)
            img = nimg.load_img(str(p))
            data = img.get_fdata()
            data = data[np.isfinite(data)]
            data = data[data > 0]
            if data.size:
                vals.append(data)

    if not vals:
        return None

    cat = np.concatenate(vals)
    return float(np.percentile(cat, pctl))


def plot_one_view(ax, img, *, bg_img, display_mode: str, threshold: float, vmin: float, vmax: float, cmap: str):
    """
    Plot into a specific matplotlib axes.
    """
    plotting.plot_stat_map(
        img,
        bg_img=bg_img,
        display_mode=display_mode,
        cut_coords=1,
        black_bg=True,
        cmap=cmap,
        threshold=threshold,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        annotate=False,
        axes=ax,
    )


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
    ap.add_argument("--in-dir", type=str, default=None, help="Input directory for Fig2a NIfTI (default: <root>/data/fig2a).")
    ap.add_argument("--out", type=str, default=None, help="Output PDF path (default: <root>/results/figures/Fig2a_spatial_distribution.pdf).")
    ap.add_argument("--freqs", type=str, default="80,120,160,200,240", help="Comma-separated frequencies (Hz).")
    ap.add_argument("--vmax-pctl", type=float, default=VMAX_PCTL_DEFAULT, help="Percentile for shared vmax within each region.")
    ap.add_argument("--thr-hipp", type=float, default=THRESH_ABS_DEFAULT["Hippocampus"], help="Display threshold for Hippocampus maps.")
    ap.add_argument("--thr-ctx", type=float, default=THRESH_ABS_DEFAULT["Cortex"], help="Display threshold for Cortex maps.")
    ap.add_argument("--cmap", type=str, default=CMAP_DEFAULT, help="Matplotlib colormap name.")
    args = ap.parse_args()

    # root resolution (Code Ocean friendly)
    root = Path(args.root).expanduser().resolve() if args.root else find_root()

    in_dir = Path(args.in_dir).expanduser().resolve() if args.in_dir else (root / "data" / "fig2a")
    # Code Ocean環境なら /results/ を使い、そうでなければ root/results/
    if Path("/results").exists():
        results_root = Path("/results")
    else:
        results_root = root / "results"
    out_pdf = Path(args.out).expanduser().resolve() if args.out else (results_root / "figures" / "Fig2a_spatial_distribution.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    freqs = [int(x.strip()) for x in args.freqs.split(",") if x.strip()]
    regions = list(REGIONS_DEFAULT)
    groups = list(GROUPS_DEFAULT)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    # check inputs
    missing = []
    for region in regions:
        for group in groups:
            for f in freqs:
                p = expected_path(in_dir, region, group, f)
                if not p.exists():
                    missing.append(str(p))
    if missing:
        show = "\n".join(missing[:30])
        raise FileNotFoundError(
            "Missing Fig2a input NIfTI files.\n"
            f"Expected under: {in_dir}\n"
            "Example: Hippocampus_HC_80Hz.nii.gz\n\n"
            f"First missing entries:\n{show}\n"
            f"... ({len(missing)} missing total)"
        )

    # background template (optional)
    bg = try_load_mni_template()

    # region-wise shared vmax
    region_vmax = {}
    for region in regions:
        vmax = region_shared_vmax(in_dir, region, groups, freqs, pctl=float(args.vmax_pctl))
        region_vmax[region] = vmax

    thr_abs = {"Hippocampus": float(args.thr_hipp), "Cortex": float(args.thr_ctx)}

    # Layout:
    # Rows: Hippocampus HC, Hippocampus SZ, Cortex HC, Cortex SZ  (4 rows)
    # Cols: frequencies (len(freqs))
    # Each cell: two sub-axes (x-view and z-view)
    row_specs = [
        ("Hippocampus", "HC"),
        ("Hippocampus", "SZ"),
        ("Cortex", "HC"),
        ("Cortex", "SZ"),
    ]

    n_freq = len(freqs)
    fig = plt.figure(figsize=(3.2 * n_freq, 10.5))

    outer = fig.add_gridspec(
        nrows=len(row_specs), ncols=n_freq,
        height_ratios=[1, 1, 1, 1],
        wspace=0.05, hspace=0.22
    )

    # top titles (frequency)
    for j, f in enumerate(freqs):
        ax_t = fig.add_subplot(outer[0, j])
        ax_t.axis("off")
        ax_t.set_title(f"{int(f)} Hz", fontsize=12, pad=12)

    # plot cells
    for i, (region, group) in enumerate(row_specs):
        for j, f in enumerate(freqs):
            inner = outer[i, j].subgridspec(1, 2, wspace=0.02)

            img_path = expected_path(in_dir, region, group, int(f))
            img = nimg.load_img(str(img_path))

            vmax = region_vmax.get(region, None)
            if vmax is None or not np.isfinite(vmax) or vmax <= 0:
                # fallback vmax if something weird happens
                vmax = 1.0

            thr = thr_abs[region]

            ax_x = fig.add_subplot(inner[0, 0])
            ax_z = fig.add_subplot(inner[0, 1])

            plot_one_view(ax_x, img, bg_img=bg, display_mode="x", threshold=thr, vmin=0, vmax=vmax, cmap=args.cmap)
            plot_one_view(ax_z, img, bg_img=bg, display_mode="z", threshold=thr, vmin=0, vmax=vmax, cmap=args.cmap)

            # left labels
            if j == 0:
                ax_x.text(
                    -0.18, 0.5, f"{group}",
                    transform=ax_x.transAxes,
                    va="center", ha="right",
                    fontsize=12, fontweight="bold"
                )
                if group == "HC":
                    ax_x.text(
                        -0.18, 1.18,
                        f"(1) {region}" if region == "Hippocampus" else f"(2) {region}",
                        transform=ax_x.transAxes,
                        va="bottom", ha="right",
                        fontsize=12
                    )

            # clean frames
            for ax in (ax_x, ax_z):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # region-wise colorbars (shared scaling per region)
    # positions tuned for 4-row layout; adjust if needed
    for region, y0 in [("Hippocampus", 0.535), ("Cortex", 0.085)]:
        vmax = region_vmax.get(region, None)
        if vmax is None or not np.isfinite(vmax) or vmax <= 0:
            continue
        cax = fig.add_axes([0.92, y0, 0.015, 0.33])  # [left, bottom, width, height]
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=args.cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Voxel-normalized ripple events (%)", fontsize=9)

    fig.suptitle("(a) Spatial distribution of ripple events in hippocampus and cerebral cortex", fontsize=14, y=0.995)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Input dir:", in_dir)
    print("[OK] Saved:", out_pdf)
    if bg is None:
        print("[WARN] MNI background template could not be loaded; plotted without bg_img.")

if __name__ == "__main__":
    main()
