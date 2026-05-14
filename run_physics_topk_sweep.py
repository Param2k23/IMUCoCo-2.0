#!/usr/bin/env python
"""
run_physics_topk_sweep.py
=========================
Overnight grid sweep over physics + top-k rerank hyperparameters,
optimising for both exact-match and per-sensor accuracy.

Three stages (each fully resumable — skips a config if its
eval_summary.json already exists):
  A. Coarse:  weights × physics_k × hard_threshold  (~180 configs)
  B. Refine:  best stage-A config × n_windows       (4 configs)
  C. Final:   best overall × N_TRIALS_FINAL         (1 config, tight CIs)

Each config launches `evaluate.py --physics_rerank` as a subprocess.
CUDA is used automatically by evaluate.py if available.

Outputs (under --out_base/):
  stageA/<tag>/eval_summary.json     — per-config raw results
  stageA/<tag>/run.log               — per-config stdout/stderr
  stageA_summary.csv                 — one row per config (sortable)
  stageA_winners.json                — best by per_sensor / exact / combined
  stageB/<tag>/...                   — n_windows refinement
  stageB_summary.csv
  stageC/<tag>/...                   — high-trial final eval
  final_report.json                  — top configs + selected hyperparameters

Usage (overnight)
-----------------
  nohup python run_physics_topk_sweep.py \\
    --checkpoint checkpoints/smoke_single_subject/best_model.pt \\
    --data       data/single_subject_test.npz \\
    --out_base   results/physics_topk_sweep \\
    --n_trials   500 --n_trials_final 3000 \\
    > sweep.log 2>&1 &

Resume after interruption: rerun the same command — finished configs
are skipped automatically.

Skip stages with --skip_stages B,C  (e.g. just rerun stage A).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

# Each entry: (tag, "w_kin w_grav w_acc w_joint")
# w_grav stays at 0 (linear-acc data has no gravity signal — see PHYSICS §9.1)
WEIGHTS_PRESETS = [
    ("classifier_only",     "0.0  0.0 0.0  0.0"),   # baseline (P_phys=0.5 const)
    ("pure_kin",            "1.0  0.0 0.0  0.0"),
    ("pure_acc",            "0.0  0.0 1.0  0.0"),
    ("pure_jointlim",       "0.0  0.0 0.0  1.0"),
    ("kin_acc_50_50",       "0.5  0.0 0.5  0.0"),
    ("default_4_4_2",       "0.4  0.0 0.4  0.2"),
    ("light_joint",         "0.45 0.0 0.45 0.1"),
    ("balanced_3_3_4",      "0.3  0.0 0.3  0.4"),
    ("very_heavy_joint",    "0.25 0.0 0.25 0.5"),
    ("heavy_kin",           "0.6  0.0 0.2  0.2"),
    ("heavy_acc",           "0.2  0.0 0.6  0.2"),
    ("balanced_35_35_3",    "0.35 0.0 0.35 0.3"),
]

PHYSICS_K_LIST = [3, 5, 7]                       # combo enumeration breadth
HARD_THRESH_LIST = [None, 0.1, 0.2, 0.3, 0.5]    # None = soft mode

# Stage B: n_windows refinement on the stage-A winner
N_WINDOWS_LIST = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def thr_tag(thr: Optional[float]) -> str:
    return "soft" if thr is None else f"hard{str(thr).replace('.', 'p')}"


def w_tag(name: str) -> str:
    return name


def cfg_tag(weights_name: str, k: int, thr: Optional[float],
            n_windows: int = 1) -> str:
    parts = [f"w-{w_tag(weights_name)}", f"k{k}", thr_tag(thr)]
    if n_windows != 1:
        parts.append(f"nw{n_windows}")
    return "__".join(parts)


def run_one_config(
    tag: str,
    out_dir: Path,
    args,
    weights_str: str,
    physics_k: int,
    hard_threshold: Optional[float],
    n_windows: int,
    n_trials: int,
) -> tuple[bool, Optional[float]]:
    """
    Returns (success, elapsed_sec). Skips if eval_summary.json already exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_summary.json"
    if summary_path.exists():
        return True, None  # cached / resumed

    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", args.checkpoint,
        "--data", args.data,
        "--out_dir", str(out_dir),
        "--rerank_windows", "0",                 # skip temporal rerank
        "--physics_rerank",
        "--physics_calibration", args.calibration,
        "--physics_stats", args.stats,
        "--rerank_n_sensors", args.n_sensors,
        "--rerank_n_windows", str(n_windows),
        "--rerank_n_trials", str(n_trials),
        "--physics_k", str(physics_k),
        "--rerank_seed", str(args.seed),
        "--physics_weights", *weights_str.split(),
        "--base_filters", str(args.base_filters),
    ]
    if hard_threshold is not None:
        cmd += ["--joint_limits_hard", str(hard_threshold)]

    log_path = out_dir / "run.log"
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n# started: {ts()}\n\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - t0

    if rc != 0:
        print(f"[{ts()}]   FAILED (rc={rc})  log: {log_path}",
              flush=True)
        return False, elapsed
    return True, elapsed


def read_summary(summary_path: Path) -> Optional[dict]:
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(summary: dict, hard_mode: bool) -> dict:
    """
    Pull per-sensor and exact-match accuracy across n_sensors from a summary.
    Returns dict with mean_per_sensor, mean_exact, per_x dict, top1.
    """
    block_key = "physics_rerank_summary_hard" if hard_mode else "physics_rerank_summary"
    block = summary.get(block_key, {}) or {}

    per_sensor_vals: list[float] = []
    exact_vals: list[float] = []
    per_x: dict = {}
    n_eliminated_total = 0.0

    for x_str, row in block.items():
        ps = float(row.get("per_sensor_acc", 0.0))
        em = float(row.get("exact_match_acc", 0.0))
        per_sensor_vals.append(ps)
        exact_vals.append(em)
        per_x[x_str] = {
            "per_sensor": round(ps, 4),
            "exact":      round(em, 4),
            "n_elim":     row.get("n_eliminated_mean", 0.0),
            "n_combos":   row.get("n_combos_total", 0),
        }
        n_eliminated_total += float(row.get("n_eliminated_mean", 0.0))

    top1 = float(summary.get("topk_accuracy", {}).get("top1", 0.0))
    return {
        "top1":              round(top1, 4),
        "mean_per_sensor":   round(float(sum(per_sensor_vals) / max(1, len(per_sensor_vals))), 4),
        "mean_exact":        round(float(sum(exact_vals) / max(1, len(exact_vals))), 4),
        "per_x":             per_x,
        "n_elim_sum":        round(n_eliminated_total, 2),
    }


def aggregate(out_root: Path, configs: list[dict], stage_name: str) -> Path:
    """
    Read each config's eval_summary.json, write {stage_name}_summary.csv with
    one row per config plus a JSON ranking. Returns CSV path.
    """
    rows = []
    for cfg in configs:
        summary = read_summary(cfg["out_dir"] / "eval_summary.json")
        if summary is None:
            continue
        m = extract_metrics(summary, hard_mode=cfg["hard_threshold"] is not None)
        rows.append({
            "tag":              cfg["tag"],
            "weights_preset":   cfg["weights_name"],
            "weights_str":      cfg["weights_str"],
            "physics_k":        cfg["physics_k"],
            "hard_threshold":   "" if cfg["hard_threshold"] is None else cfg["hard_threshold"],
            "n_windows":        cfg["n_windows"],
            "n_trials":         cfg["n_trials"],
            "top1":             m["top1"],
            "mean_per_sensor":  m["mean_per_sensor"],
            "mean_exact":       m["mean_exact"],
            "combined_score":   round(0.5 * m["mean_per_sensor"] + 0.5 * m["mean_exact"], 4),
            **{f"x{x}_per_sensor": m["per_x"].get(x, {}).get("per_sensor", 0)
               for x in ["2", "3", "4", "5"]},
            **{f"x{x}_exact":      m["per_x"].get(x, {}).get("exact", 0)
               for x in ["2", "3", "4", "5"]},
            "n_elim_sum":       m["n_elim_sum"],
        })

    if not rows:
        print(f"[{ts()}] {stage_name}: no completed configs to aggregate.",
              flush=True)
        return out_root / f"{stage_name}_summary.csv"

    csv_path = out_root / f"{stage_name}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Winners by metric
    winners = {
        "by_per_sensor":    sorted(rows, key=lambda r: -r["mean_per_sensor"])[:5],
        "by_exact_match":   sorted(rows, key=lambda r: -r["mean_exact"])[:5],
        "by_combined":      sorted(rows, key=lambda r: -r["combined_score"])[:5],
    }
    with open(out_root / f"{stage_name}_winners.json", "w") as f:
        json.dump(winners, f, indent=2)

    print(f"\n[{ts()}] {stage_name}: aggregated {len(rows)} configs -> {csv_path}")
    print(f"[{ts()}] Top-3 by combined (0.5*per_sensor + 0.5*exact):")
    for r in winners["by_combined"][:3]:
        print(f"  {r['combined_score']:.4f}   ps={r['mean_per_sensor']:.4f}  "
              f"em={r['mean_exact']:.4f}   {r['tag']}")
    return csv_path


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------

def stage_A(args, out_root: Path) -> list[dict]:
    """
    Coarse sweep: weights × physics_k × hard_threshold.
    """
    print(f"\n[{ts()}] === STAGE A: coarse sweep ===")
    stage_dir = out_root / "stageA"
    stage_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict] = []
    grid = list(itertools.product(WEIGHTS_PRESETS, PHYSICS_K_LIST, HARD_THRESH_LIST))
    print(f"[{ts()}] {len(grid)} configurations to run "
          f"(skipping any with existing eval_summary.json)")

    for i, ((wname, wstr), pk, thr) in enumerate(grid, 1):
        # classifier_only with hard mode is degenerate (eliminator-only),
        # but keep it — it's a useful ablation.
        tag = cfg_tag(wname, pk, thr)
        out_dir = stage_dir / tag
        cfg = {
            "tag":            tag,
            "out_dir":        out_dir,
            "weights_name":   wname,
            "weights_str":    wstr,
            "physics_k":      pk,
            "hard_threshold": thr,
            "n_windows":      1,
            "n_trials":       args.n_trials,
        }
        configs.append(cfg)

        cached = (out_dir / "eval_summary.json").exists()
        marker = "CACHED" if cached else "RUNNING"
        print(f"[{ts()}] [{i:3d}/{len(grid)}] {marker:7s}  {tag}", flush=True)
        if cached:
            continue

        ok, elapsed = run_one_config(
            tag=tag, out_dir=out_dir, args=args,
            weights_str=wstr, physics_k=pk,
            hard_threshold=thr, n_windows=1,
            n_trials=args.n_trials,
        )
        if ok:
            print(f"[{ts()}]              done   ({elapsed:.1f}s)", flush=True)

    aggregate(out_root, configs, "stageA")
    return configs


def stage_B(args, out_root: Path, stageA_configs: list[dict]) -> list[dict]:
    """
    Refine the stage-A winner (by combined score) across n_windows.
    """
    print(f"\n[{ts()}] === STAGE B: n_windows refinement on stage-A winner ===")
    winners_path = out_root / "stageA_winners.json"
    if not winners_path.exists():
        print(f"[{ts()}] No stageA_winners.json — cannot start stage B.")
        return []
    with open(winners_path) as f:
        winners = json.load(f)
    if not winners.get("by_combined"):
        print(f"[{ts()}] No combined-best in stage A — skipping stage B.")
        return []

    best = winners["by_combined"][0]
    print(f"[{ts()}] Stage-A winner (combined={best['combined_score']:.4f}): {best['tag']}")
    print(f"[{ts()}]   weights={best['weights_str']}  k={best['physics_k']}  "
          f"hard={best['hard_threshold']}")

    stage_dir = out_root / "stageB"
    stage_dir.mkdir(parents=True, exist_ok=True)

    weights_str = best["weights_str"]
    pk          = int(best["physics_k"])
    thr_raw     = best["hard_threshold"]
    thr         = None if thr_raw == "" else float(thr_raw)

    configs: list[dict] = []
    for nw in N_WINDOWS_LIST:
        tag = cfg_tag(best["weights_preset"], pk, thr, n_windows=nw)
        out_dir = stage_dir / tag
        cfg = {
            "tag":            tag,
            "out_dir":        out_dir,
            "weights_name":   best["weights_preset"],
            "weights_str":    weights_str,
            "physics_k":      pk,
            "hard_threshold": thr,
            "n_windows":      nw,
            "n_trials":       args.n_trials,
        }
        configs.append(cfg)

        cached = (out_dir / "eval_summary.json").exists()
        marker = "CACHED" if cached else "RUNNING"
        print(f"[{ts()}] {marker}  {tag}", flush=True)
        if cached:
            continue

        ok, elapsed = run_one_config(
            tag=tag, out_dir=out_dir, args=args,
            weights_str=weights_str, physics_k=pk,
            hard_threshold=thr, n_windows=nw,
            n_trials=args.n_trials,
        )
        if ok:
            print(f"[{ts()}]   done   ({elapsed:.1f}s)", flush=True)

    aggregate(out_root, configs, "stageB")
    return configs


def stage_C(args, out_root: Path,
            all_configs: list[dict]) -> Optional[dict]:
    """
    Final eval at the best overall (stage A + B) config with N_TRIALS_FINAL
    trials for tight error bars.
    """
    print(f"\n[{ts()}] === STAGE C: final high-trial eval on best overall ===")

    # Read stage A + B summaries to find the absolute best by combined.
    rows = []
    for cfg in all_configs:
        summary = read_summary(cfg["out_dir"] / "eval_summary.json")
        if summary is None:
            continue
        m = extract_metrics(summary, hard_mode=cfg["hard_threshold"] is not None)
        combined = 0.5 * m["mean_per_sensor"] + 0.5 * m["mean_exact"]
        rows.append((combined, cfg, m))
    if not rows:
        print(f"[{ts()}] No completed configs — skipping stage C.")
        return None
    rows.sort(key=lambda x: -x[0])
    best_combined, best_cfg, best_m = rows[0]
    print(f"[{ts()}] Best overall (combined={best_combined:.4f}): {best_cfg['tag']}")

    stage_dir = out_root / "stageC"
    tag = best_cfg["tag"] + f"__final{args.n_trials_final}"
    out_dir = stage_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = (out_dir / "eval_summary.json").exists()
    marker = "CACHED" if cached else "RUNNING"
    print(f"[{ts()}] {marker}  {tag}", flush=True)
    if not cached:
        ok, elapsed = run_one_config(
            tag=tag, out_dir=out_dir, args=args,
            weights_str=best_cfg["weights_str"],
            physics_k=best_cfg["physics_k"],
            hard_threshold=best_cfg["hard_threshold"],
            n_windows=best_cfg["n_windows"],
            n_trials=args.n_trials_final,
        )
        if ok:
            print(f"[{ts()}]   done   ({elapsed:.1f}s)", flush=True)

    summary = read_summary(out_dir / "eval_summary.json")
    if summary is None:
        return None
    m_final = extract_metrics(summary,
                               hard_mode=best_cfg["hard_threshold"] is not None)

    final_report = {
        "selected_config": {
            "tag":              best_cfg["tag"],
            "weights_preset":   best_cfg["weights_name"],
            "weights_str":      best_cfg["weights_str"],
            "physics_k":        best_cfg["physics_k"],
            "hard_threshold":   best_cfg["hard_threshold"],
            "n_windows":        best_cfg["n_windows"],
        },
        "metrics_at_final_n_trials": {
            "n_trials":         args.n_trials_final,
            **m_final,
        },
        "metrics_at_search_n_trials": {
            "n_trials":         best_cfg["n_trials"],
            **best_m,
        },
        "checkpoint":           args.checkpoint,
        "data":                 args.data,
        "timestamp":            ts(),
    }
    report_path = out_root / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[{ts()}] Final report: {report_path}")
    print(f"   per_sensor (final, n={args.n_trials_final}): {m_final['mean_per_sensor']:.4f}")
    print(f"   exact_match (final): {m_final['mean_exact']:.4f}")
    print(f"   per-x: {m_final['per_x']}")
    return final_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Trained model checkpoint (.pt)")
    p.add_argument("--data", required=True, help="Test .npz dataset")
    p.add_argument("--out_base", default="results/physics_topk_sweep",
                   help="Base output directory")
    p.add_argument("--calibration", default="calibration",
                   help="Directory with region_sensor_rotmats.npy")
    p.add_argument("--stats", default="stats",
                   help="Directory with joint_angle_limits.npy etc.")
    p.add_argument("--n_sensors", default="2,3,4,5",
                   help="Comma-separated x values for combo synthesis")
    p.add_argument("--n_trials", type=int, default=500,
                   help="Trials per config in stage A and B")
    p.add_argument("--n_trials_final", type=int, default=3000,
                   help="Trials for stage C (high-precision final eval)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_stages", default="",
                   help="Comma-separated stages to skip: A,B,C")
    p.add_argument("--base_filters", type=int, default=128,
                   help="base_filters for ResNet1D architecture. Must match the "
                        "checkpoint (default: 128 for penalty_sweep models). "
                        "Pass 64 for older checkpoints trained without --base_filters 128.")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_base)
    out_root.mkdir(parents=True, exist_ok=True)

    skip = set(s.strip().upper() for s in args.skip_stages.split(",") if s.strip())

    print(f"[{ts()}] === Physics + Top-k Sweep ===")
    print(f"  checkpoint:    {args.checkpoint}")
    print(f"  data:          {args.data}")
    print(f"  out_base:      {out_root}")
    print(f"  n_sensors:     {args.n_sensors}")
    print(f"  n_trials:      {args.n_trials}  (final: {args.n_trials_final})")
    print(f"  skip_stages:   {sorted(skip) if skip else 'none'}")

    all_configs: list[dict] = []

    if "A" not in skip:
        all_configs += stage_A(args, out_root)
    else:
        # Still register stage A configs so stage C can rank against them.
        for (wname, wstr), pk, thr in itertools.product(
                WEIGHTS_PRESETS, PHYSICS_K_LIST, HARD_THRESH_LIST):
            tag = cfg_tag(wname, pk, thr)
            all_configs.append({
                "tag": tag, "out_dir": out_root / "stageA" / tag,
                "weights_name": wname, "weights_str": wstr,
                "physics_k": pk, "hard_threshold": thr, "n_windows": 1,
                "n_trials": args.n_trials,
            })

    if "B" not in skip:
        all_configs += stage_B(args, out_root, all_configs)

    if "C" not in skip:
        stage_C(args, out_root, all_configs)

    print(f"\n[{ts()}] === Sweep complete ===")
    print(f"  Aggregated CSVs:")
    for fn in ("stageA_summary.csv", "stageB_summary.csv"):
        path = out_root / fn
        if path.exists():
            print(f"    {path}")
    final_report = out_root / "final_report.json"
    if final_report.exists():
        print(f"  Final report: {final_report}")


if __name__ == "__main__":
    main()
