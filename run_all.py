#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py
==========
Entry point for reproducing all main figures and supplementary tables from the
derived (anonymized) data included in this repository.

Typical usage (from repository root):
  python run_all.py
  python run_all.py --dry-run
  python run_all.py --only fig3 fig4
  python run_all.py --skip fig2a_nifti
  python run_all.py --continue-on-error   # debugging only

What this script does
---------------------
- Runs a predefined sequence of analysis scripts under ./scripts/.
- Each script reads derived input tables under ./data/ and writes outputs under ./outputs/.
- No raw MEG data are required or accessed by this reproduction pipeline.

Outputs
-------
- Figures: ./outputs/figures/
- Tables:  ./outputs/tables/
- Logs:    ./outputs/logs/run_all_log_*.txt

Notes on optional assets
------------------------
- Fig.2a group NIfTI maps (if provided) are treated as pre-generated assets shipped with the repository.
  They are not recomputed in this pipeline unless you later add an explicit generator script.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# -------------------------
# Repo root
# -------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


# -------------------------
# Optional: container-friendly data mount
# -------------------------
def ensure_data_dir(repo_root: Path) -> None:
    """
    If ./data is missing but /data exists (common in containerized environments),
    create a symlink ./data -> /data.

    Notes:
    - Only acts when ./data does NOT exist.
    - If symlink already exists, it is left as-is.
    """
    data_dir = repo_root / "data"
    if data_dir.exists():
        return

    host_data = Path("/data")
    if host_data.exists():
        try:
            # Create symlink repo_root/data -> /data
            os.symlink(str(host_data), str(data_dir))
            print(f"[INFO] Created symlink: {data_dir} -> {host_data}", flush=True)
        except FileExistsError:
            pass
        except OSError as e:
            print(f"[WARN] Failed to create symlink {data_dir} -> {host_data}: {e}", flush=True)


# -------------------------
# Pipeline definition (UPDATED)
# -------------------------
PIPELINE = [
    # ---- Fig 1 ----
    ("fig1", "Fig. 1: Ripple counts/durations/power + Supplementary Tables (S3–S5)", [
        "fig1_ac_with_supp_table34.py",
        "fig1_b.py",
        "fig1_d_with_supp_table5.py",
    ]),

    # ---- Fig 2 ----
    ("fig2", "Fig. 2: Spatial/network distribution analyses + Supplementary Table S6", [
        "fig2_a.py",
        "fig2_b_with_supp_table6.py",
    ]),

    # ---- Fig 3 ----
    ("fig3", "Fig. 3: Seconds-long temporal clustering (high-rate epochs) + Supplementary Tables S7–S9", [
        "fig3_with_supp_table7_to9.py",
    ]),

    # ---- Fig 4 ----
    ("fig4", "Fig. 4: Symptom associations (SZ only)", [
        "fig4_a.py",
        "fig4_b.py",
    ]),

    # ---- Supplementary tables (standalone) ----
    ("supp", "Supplementary tables (standalone scripts)", [
        "supp_table1.py",
        "supp_table2.py",
        "supp_table10.py",
        "supp_table11.py",
    ]),
]

# Optional extra step: Fig2a NIfTI outputs may be shipped as files (no script required).
OPTIONAL_STEPS = {
    "fig2a_nifti": "Fig. 2a NIfTI group maps are shipped as pre-generated assets (no computation in this pipeline).",
}


# -------------------------
# Helpers
# -------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def run_one(script_path: Path, *, python_exe: str, cwd: Path, env: dict, log_fp, dry_run: bool) -> int:
    cmd = [python_exe, str(script_path)]
    print(f"\n>>> RUN: {' '.join(cmd)}", flush=True)
    log_fp.write(f"\n>>> RUN: {' '.join(cmd)}\n")
    log_fp.flush()

    if dry_run:
        print(">>> DRY-RUN: skipped execution.", flush=True)
        log_fp.write(">>> DRY-RUN: skipped execution.\n")
        log_fp.flush()
        return 0

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if p.stdout:
        print(p.stdout, flush=True)
        log_fp.write(p.stdout)
        log_fp.flush()

    return int(p.returncode)


def normalize_tags(tag_list: list[str] | None, valid: set[str]) -> set[str]:
    if not tag_list:
        return set(valid)
    out = set()
    for t in tag_list:
        t2 = str(t).strip()
        if t2 in valid:
            out.add(t2)
        else:
            raise ValueError(f"Unknown tag: '{t2}'. Valid tags: {sorted(valid)}")
    return out


# -------------------------
# Main
# -------------------------
def main():
    ensure_data_dir(REPO_ROOT)

    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", type=str, default="scripts",
                    help="Directory containing analysis scripts (default: scripts)")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python executable to use (default: current interpreter)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only selected tags (e.g., --only fig3 fig4)")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="Skip selected tags or optional steps (e.g., --skip fig2 fig2a_nifti)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned commands without executing")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Debugging only: continue even if a script fails (not recommended for formal reproduction)")
    args = ap.parse_args()

    repo_root = REPO_ROOT
    scripts_dir = (repo_root / args.scripts_dir).resolve()
    if not scripts_dir.exists():
        raise FileNotFoundError(f"scripts directory not found: {scripts_dir}")

    valid_tags = {t for (t, _desc, _lst) in PIPELINE}
    valid_optional = set(OPTIONAL_STEPS.keys())
    valid_all = valid_tags | valid_optional

    only_set = normalize_tags(args.only, valid_tags)

    skip_set = set()
    if args.skip:
        for t in args.skip:
            t2 = str(t).strip()
            if t2 not in valid_all:
                raise ValueError(f"Unknown skip target: '{t2}'. Valid: {sorted(valid_all)}")
            skip_set.add(t2)

    # Ensure output directories
    results_root = repo_root / "results"
    (results_root / "figures").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    (results_root / "logs").mkdir(parents=True, exist_ok=True)

    log_path = results_root / "logs" / f"run_all_log_{now_str()}.txt"


    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print("\n=== Reproduction pipeline (run_all.py) ===")
    print("Repository root:", repo_root)
    print("Scripts dir:", scripts_dir)
    print("Python:", args.python)
    print("Only tags:", sorted(only_set))
    print("Skip:", sorted(skip_set))
    print("Dry-run:", args.dry_run)
    print("Log file:", log_path)
    print("=========================================\n")

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write("=== Reproduction pipeline (run_all.py) ===\n")
        log_fp.write(f"Repository root: {repo_root}\n")
        log_fp.write(f"Scripts dir: {scripts_dir}\n")
        log_fp.write(f"Python: {args.python}\n")
        log_fp.write(f"Only tags: {sorted(only_set)}\n")
        log_fp.write(f"Skip: {sorted(skip_set)}\n")
        log_fp.write(f"Dry-run: {args.dry_run}\n")
        log_fp.write(f"Started: {datetime.now().isoformat()}\n")
        log_fp.write("=========================================\n\n")
        log_fp.flush()

        failures = []

        # optional steps (informational)
        for opt, desc in OPTIONAL_STEPS.items():
            if opt in skip_set:
                continue
            log_fp.write(f"[INFO] Optional asset: {opt} — {desc}\n")
            log_fp.flush()

        # main pipeline
        for tag, desc, script_list in PIPELINE:
            if tag not in only_set:
                continue
            if tag in skip_set:
                print(f"--- SKIP TAG: {tag} ---", flush=True)
                log_fp.write(f"--- SKIP TAG: {tag} ---\n")
                log_fp.flush()
                continue

            print(f"\n=== [{tag}] {desc} ===", flush=True)
            log_fp.write(f"\n=== [{tag}] {desc} ===\n")
            log_fp.flush()

            for script_name in script_list:
                script_path = scripts_dir / script_name
                if not script_path.exists():
                    msg = f"[ERROR] Missing script: {script_path}"
                    print(msg, flush=True)
                    log_fp.write(msg + "\n")
                    log_fp.flush()
                    failures.append((tag, script_name, "missing"))
                    if not args.continue_on_error:
                        print(f"\nStopped due to missing script. See log: {log_path}", flush=True)
                        sys.exit(1)
                    continue

                rc = run_one(
                    script_path,
                    python_exe=args.python,
                    cwd=repo_root,
                    env=env,
                    log_fp=log_fp,
                    dry_run=args.dry_run,
                )
                if rc != 0:
                    msg = f"[FAIL] {script_name} (exit code {rc})"
                    print(msg, flush=True)
                    log_fp.write(msg + "\n")
                    log_fp.flush()
                    failures.append((tag, script_name, f"exit{rc}"))
                    if not args.continue_on_error:
                        print(f"\nStopped due to error. See log: {log_path}", flush=True)
                        sys.exit(rc)

        # summary
        if failures:
            print("\n=== PIPELINE FINISHED WITH FAILURES ===", flush=True)
            for f in failures:
                print(" -", f, flush=True)
            print("Log:", log_path, flush=True)

            log_fp.write("\n=== PIPELINE FINISHED WITH FAILURES ===\n")
            for f in failures:
                log_fp.write(f" - {f}\n")
            log_fp.write(f"Finished: {datetime.now().isoformat()}\n")
            log_fp.flush()
            sys.exit(2)

        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===", flush=True)
        print("Log:", log_path, flush=True)
        log_fp.write("\n=== PIPELINE COMPLETED SUCCESSFULLY ===\n")
        log_fp.write(f"Finished: {datetime.now().isoformat()}\n")
        log_fp.flush()


if __name__ == "__main__":
    main()
