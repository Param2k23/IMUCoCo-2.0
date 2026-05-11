"""
physics_sweep.py
================
Diagnostics + sweep for the physics verification scorers.

Run via run_physics_sweep.sh. Designed to surface whether the current
implementation is worth pursuing, and where to invest next.

What this script answers:
  1. Are the rotation utilities numerically correct (orthonormal R, det=+1)?
  2. Does the IMU acceleration channel actually contain gravity? (drives whether
     the gravity scorer is even possible in principle on this dataset)
  3. Does the kinematic scorer actually consult the joint-angle limits it loads,
     or is it scoring "deviation from identity"?
  4. For multi-sensor combos, do TRUE region assignments score higher than
     - 1-region swaps with off-chain region (hard negatives)?
     - left/right swaps (symmetry-confused negatives)?
     - random combos (easy negatives)?
     Reported as ROC-AUC and mean-gap per scorer (kinematic / gravity / accel).
  5. Per-region accel/gravity score variance — which regions are actually
     discriminable?
  6. Sweep across n_sensors in {2,3,4,5} and num combos {50,200} — does the
     gap grow with combo size (kinematic should benefit) and stabilize with N?

Outputs JSON to stats/sweep_report.json and a human-readable summary to stdout.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotation_utils import r6d_to_rotmat, rotmat_to_r6d, _random_rotation_matrix  # noqa: E402
from smpl_regions import REGION_NAMES, REGION_PARENTS, SYMMETRY_PAIRS  # noqa: E402
from verify_combo import (  # noqa: E402
    extract_imu_components,
    load_calibration,
    load_data,
    load_stats,
    score_acceleration_profile,
    score_gravity_alignment,
    score_kinematic_chain,
)


# ---------------------------------------------------------------------------
# 1. Rotation correctness
# ---------------------------------------------------------------------------
def diag_rotation_utils(seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)

    # Orthonormality on real input
    a = rng.standard_normal(3)
    b = rng.standard_normal(3)  # NOT orthogonal to a
    r6d = np.concatenate([a, b]).astype(np.float32)
    R = r6d_to_rotmat(r6d)
    err_orth = float(np.linalg.norm(R.T @ R - np.eye(3)))
    err_det = float(abs(np.linalg.det(R) - 1.0))

    # Roundtrip on valid rotations
    Rs = np.stack([_random_rotation_matrix() for _ in range(64)])
    r6d_valid = rotmat_to_r6d(Rs)
    R_back = r6d_to_rotmat(r6d_valid)
    err_rt = float(np.mean(np.linalg.norm(Rs - R_back, axis=(1, 2))))

    return {
        "non_orthogonal_input_R^TR_minus_I_frob": err_orth,
        "non_orthogonal_input_det_minus_1_abs": err_det,
        "valid_input_roundtrip_mean_err": err_rt,
        "verdict_robust_to_non_orthogonal": err_orth < 1e-4,
    }


# ---------------------------------------------------------------------------
# 2. Does acceleration include gravity?
# ---------------------------------------------------------------------------
def diag_acceleration(X: np.ndarray, y: np.ndarray) -> Dict:
    acc = X[:, 6:9, :]  # (N, 3, T)
    per_axis_mean = acc.mean(axis=(0, 2)).tolist()
    overall_mag = float(np.linalg.norm(acc, axis=1).mean())

    # If acc includes gravity, expect mean magnitude ~9.81 m/s^2 across
    # static periods. If linear-only (DIP-IMU/TransPose convention), mean ~ 0.
    # We heuristically diagnose:
    has_gravity = overall_mag > 7.0  # would need to be >> 7 to hold gravity

    pelvis_mag = float(np.linalg.norm(acc[y == 0], axis=1).mean()) if (y == 0).any() else float("nan")

    per_region_mean_mag = {}
    for r in range(24):
        mask = y == r
        if mask.sum() == 0:
            continue
        per_region_mean_mag[REGION_NAMES[r]] = float(np.linalg.norm(acc[mask], axis=1).mean())

    return {
        "global_mean_per_axis": per_axis_mean,
        "global_mean_magnitude": overall_mag,
        "pelvis_mean_magnitude": pelvis_mag,
        "per_region_mean_magnitude": per_region_mean_mag,
        "verdict_acc_contains_gravity": bool(has_gravity),
        "implication": (
            "Linear acceleration (no gravity) — gravity scorer cannot recover"
            " gravity direction. Treat gravity scorer as inert on this dataset."
            if not has_gravity
            else "Raw acceleration with gravity — gravity scorer is meaningful."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Does the kinematic scorer actually use joint angle limits?
# ---------------------------------------------------------------------------
def diag_kinematic_uses_limits(stats: dict) -> Dict:
    src = inspect.getsource(score_kinematic_chain)
    # We require *indexing into* the limits dict by (parent, child) pair.
    has_limit_lookup = any(
        token in src for token in [
            "loose_limits",
            "strict_limits",
            "joint_angle_limits[",
            'limits["loose"',
            "limits['loose']",
            'limits["strict"',
            "limits['strict']",
            "['loose']",
            "['strict']",
        ]
    )
    # The scorer must do MORE than a None check — it must index the dict.
    # Look for any subscript/getitem on joint_angle_limits or a renamed local.
    indexes_limits_dict = any(tok in src for tok in [
        "joint_angle_limits[",
        "joint_angle_limits.get(",
        "joint_angle_limits.keys(",
    ])
    uses_pair_lookup = has_limit_lookup or indexes_limits_dict

    jal = stats.get("joint_angle_limits") or {}
    n_pairs = sum(len(v) for v in jal.values()) if isinstance(jal, dict) else 0

    return {
        "joint_angle_limits_consulted_in_scorer": has_limit_lookup or uses_pair_lookup,
        "joint_angle_limits_loaded_n_pairs": n_pairs,
        "verdict": (
            "BUG: kinematic scorer ignores precomputed joint angle limits. "
            "It scores ||R_parent^T @ R_child - I||_F, which has no biomech meaning."
            if not (has_limit_lookup or uses_pair_lookup)
            else "OK: scorer consults joint angle limits."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Combo-ranking AUC: true vs negative combos
# ---------------------------------------------------------------------------
def _build_segment_index(y: np.ndarray) -> Tuple[List[Dict[int, int]], int]:
    """
    The current dataset construction guarantees that for each motion segment,
    all 24 regions appear consecutively. We exploit that: the k-th occurrence
    of region r in y belongs to segment k. Returns one dict per segment
    mapping region_id -> sample index. A segment is dropped if any region is
    missing. Also returns the number of segments found.
    """
    occurrences: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(y.tolist()):
        occurrences[int(r)].append(i)
    n_segs = min(len(v) for v in occurrences.values()) if occurrences else 0
    segments: List[Dict[int, int]] = []
    for k in range(n_segs):
        segments.append({r: occurrences[r][k] for r in occurrences})
    return segments, n_segs


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / ROC AUC."""
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    rsum = ranks[:n_pos].sum() + n_pos  # +n_pos because argsort ranks are 0-indexed
    u = rsum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _score_all(combo: List[int], X_samples: np.ndarray, calibration, stats) -> Dict[str, float]:
    sk, _ = score_kinematic_chain(combo, X_samples, calibration, stats.get("joint_angle_limits"))
    sg, _ = score_gravity_alignment(combo, X_samples, calibration, stats.get("gravity_distributions"))
    sa, _ = score_acceleration_profile(combo, X_samples, stats.get("accel_distributions"), None)
    return {"kinematic": sk, "gravity": sg, "accel": sa}


def diag_combo_ranking(
    X: np.ndarray,
    y: np.ndarray,
    calibration: np.ndarray,
    stats: dict,
    n_sensors: int,
    n_segments_used: int,
    seed: int = 7,
) -> Dict:
    """
    For each held-out segment, build a TRUE n-sensor combo (regions match data)
    and three NEGATIVE variants:
      - random: pick n_sensors regions and shuffle the assignment so labels
        do not match the underlying samples
      - swap_offchain: take TRUE combo, replace one region with a region
        that is NOT on the same kinematic chain (hard for kinematic scorer)
      - swap_lr: take TRUE combo, swap one left↔right region (symmetry test)
    """
    rng = np.random.default_rng(seed)
    segments, n_segs = _build_segment_index(y)
    n_segs_eval = min(n_segments_used, n_segs)
    if n_segs_eval == 0:
        return {"error": "No complete segments available"}

    # Helper: kinematic chain membership
    chain = {r: set() for r in range(24)}
    for r in range(24):
        cur = r
        while cur != -1:
            chain[r].add(cur)
            cur = REGION_PARENTS.get(cur, -1)

    def on_same_chain(r1: int, r2: int) -> bool:
        return r2 in chain[r1] or r1 in chain[r2]

    lr_map = {}
    for l, r in SYMMETRY_PAIRS:
        lr_map[l] = r
        lr_map[r] = l

    pos = {"kinematic": [], "gravity": [], "accel": []}
    neg_random = {k: [] for k in pos}
    neg_offchain = {k: [] for k in pos}
    neg_lr = {k: [] for k in pos}

    all_regions = list(range(24))

    for seg in segments[:n_segs_eval]:
        # TRUE combo: pick n_sensors regions, take their samples from this seg
        true_regions = list(rng.choice(all_regions, size=n_sensors, replace=False))
        Xs_true = np.stack([X[seg[r]] for r in true_regions])
        s = _score_all(true_regions, Xs_true, calibration, stats)
        for k in pos:
            pos[k].append(s[k])

        # NEG: random — shuffle the labels so the data and labels do not match
        shuffled_labels = list(rng.permutation(true_regions))
        # Make sure permutation is a derangement-ish; if it accidentally matches, try again
        attempts = 0
        while shuffled_labels == true_regions and attempts < 10:
            shuffled_labels = list(rng.permutation(true_regions))
            attempts += 1
        s = _score_all(shuffled_labels, Xs_true, calibration, stats)
        for k in pos:
            neg_random[k].append(s[k])

        # NEG: swap one region with off-chain region
        bad_regions = list(true_regions)
        idx = int(rng.integers(0, n_sensors))
        candidates = [
            r for r in all_regions
            if r not in true_regions and not on_same_chain(r, bad_regions[idx])
        ]
        if candidates:
            bad_regions[idx] = int(rng.choice(candidates))
            s = _score_all(bad_regions, Xs_true, calibration, stats)
            for k in pos:
                neg_offchain[k].append(s[k])

        # NEG: swap one region with its L/R counterpart
        bad_regions = list(true_regions)
        lr_candidates = [i for i, r in enumerate(bad_regions) if r in lr_map]
        if lr_candidates:
            i = int(rng.choice(lr_candidates))
            bad_regions[i] = lr_map[bad_regions[i]]
            s = _score_all(bad_regions, Xs_true, calibration, stats)
            for k in pos:
                neg_lr[k].append(s[k])

    out = {
        "n_sensors": n_sensors,
        "n_segments_eval": n_segs_eval,
        "segment_total": n_segs,
    }
    for scorer in ("kinematic", "gravity", "accel"):
        p = np.asarray(pos[scorer])
        out[scorer] = {
            "true_mean": float(p.mean()),
            "true_std": float(p.std()),
            "vs_random": {
                "neg_mean": float(np.mean(neg_random[scorer])),
                "auc": _auc(p, np.asarray(neg_random[scorer])),
                "gap": float(p.mean() - np.mean(neg_random[scorer])),
            },
            "vs_offchain_swap": {
                "neg_mean": float(np.mean(neg_offchain[scorer])) if neg_offchain[scorer] else float("nan"),
                "auc": _auc(p, np.asarray(neg_offchain[scorer])) if neg_offchain[scorer] else float("nan"),
                "gap": (float(p.mean() - np.mean(neg_offchain[scorer])) if neg_offchain[scorer] else float("nan")),
            },
            "vs_lr_swap": {
                "neg_mean": float(np.mean(neg_lr[scorer])) if neg_lr[scorer] else float("nan"),
                "auc": _auc(p, np.asarray(neg_lr[scorer])) if neg_lr[scorer] else float("nan"),
                "gap": (float(p.mean() - np.mean(neg_lr[scorer])) if neg_lr[scorer] else float("nan")),
            },
        }
    return out


# ---------------------------------------------------------------------------
# 5. Per-region single-sensor scorer separability
# ---------------------------------------------------------------------------
def diag_per_region_separability(
    X: np.ndarray, y: np.ndarray, calibration: np.ndarray, stats: dict, seed: int = 11
) -> Dict:
    """
    For each region, take its samples and score them under the CORRECT label
    vs under every WRONG label (using accel scorer only — the only scorer that
    is purely per-sensor). Returns per-region mean-gap and AUC.
    """
    rng = np.random.default_rng(seed)
    per_region = {}
    for r in range(24):
        mask = y == r
        if mask.sum() == 0:
            continue
        idxs = np.where(mask)[0]
        # cap for speed
        idxs = rng.choice(idxs, size=min(len(idxs), 30), replace=False)
        true_scores = []
        wrong_scores = []
        for i in idxs:
            Xs = X[i:i + 1]
            s_true, _ = score_acceleration_profile([r], Xs, stats.get("accel_distributions"))
            true_scores.append(s_true)
            # try all other regions as the wrong label
            for r_wrong in range(24):
                if r_wrong == r:
                    continue
                s_w, _ = score_acceleration_profile([r_wrong], Xs, stats.get("accel_distributions"))
                wrong_scores.append(s_w)
        per_region[REGION_NAMES[r]] = {
            "n_samples": int(len(idxs)),
            "true_mean": float(np.mean(true_scores)),
            "wrong_mean": float(np.mean(wrong_scores)),
            "gap": float(np.mean(true_scores) - np.mean(wrong_scores)),
            "auc": _auc(np.asarray(true_scores), np.asarray(wrong_scores)),
        }
    # Aggregate
    aucs = [v["auc"] for v in per_region.values()]
    gaps = [v["gap"] for v in per_region.values()]
    return {
        "per_region": per_region,
        "summary": {
            "median_auc": float(np.median(aucs)),
            "mean_auc": float(np.mean(aucs)),
            "n_regions_auc_above_0p7": int(sum(a > 0.7 for a in aucs)),
            "n_regions_auc_above_0p9": int(sum(a > 0.9 for a in aucs)),
            "median_gap": float(np.median(gaps)),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--calibration", default="calibration")
    p.add_argument("--stats_dir", default="stats")
    p.add_argument("--n_sensors_sweep", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--n_segments_used", type=int, default=200)
    p.add_argument("--out_json", default="stats/sweep_report.json")
    args = p.parse_args()

    print("=" * 72)
    print("PHYSICS VERIFICATION SWEEP / DIAGNOSTICS")
    print("=" * 72)

    X, y, _, _ = load_data(args.data)
    calibration = load_calibration(args.calibration)
    stats = load_stats(args.stats_dir)

    report = {"data_path": args.data, "timestamp": time.time()}

    print("\n[1] Rotation-utility correctness")
    report["rotation_utils"] = diag_rotation_utils()
    for k, v in report["rotation_utils"].items():
        print(f"    {k}: {v}")

    print("\n[2] Acceleration channel diagnosis (gravity present?)")
    report["acceleration"] = diag_acceleration(X, y)
    print(f"    global_mean_magnitude: {report['acceleration']['global_mean_magnitude']:.3f}")
    print(f"    pelvis_mean_magnitude: {report['acceleration']['pelvis_mean_magnitude']:.3f}")
    print(f"    verdict_acc_contains_gravity: {report['acceleration']['verdict_acc_contains_gravity']}")
    print(f"    implication: {report['acceleration']['implication']}")

    print("\n[3] Does kinematic scorer use joint angle limits?")
    report["kinematic_uses_limits"] = diag_kinematic_uses_limits(stats)
    for k, v in report["kinematic_uses_limits"].items():
        print(f"    {k}: {v}")

    print("\n[4] Combo-ranking sweep across n_sensors")
    report["combo_ranking"] = {}
    for n in args.n_sensors_sweep:
        print(f"\n    --- n_sensors = {n} ---")
        r = diag_combo_ranking(X, y, calibration, stats, n, args.n_segments_used)
        report["combo_ranking"][str(n)] = r
        if "error" in r:
            print(f"    {r['error']}")
            continue
        print(f"    n_segments_eval: {r['n_segments_eval']}")
        for scorer in ("kinematic", "gravity", "accel"):
            s = r[scorer]
            print(f"    [{scorer:9s}] true_mean={s['true_mean']:.3f}"
                  f"  vs_random AUC={s['vs_random']['auc']:.3f} gap={s['vs_random']['gap']:+.3f}"
                  f"  | offchain AUC={s['vs_offchain_swap']['auc']:.3f} gap={s['vs_offchain_swap']['gap']:+.3f}"
                  f"  | LR AUC={s['vs_lr_swap']['auc']:.3f} gap={s['vs_lr_swap']['gap']:+.3f}")

    print("\n[5] Per-region accel-scorer separability (single-sensor classification)")
    report["per_region"] = diag_per_region_separability(X, y, calibration, stats)
    s = report["per_region"]["summary"]
    print(f"    median AUC: {s['median_auc']:.3f}  mean AUC: {s['mean_auc']:.3f}")
    print(f"    regions with AUC > 0.7: {s['n_regions_auc_above_0p7']}/24")
    print(f"    regions with AUC > 0.9: {s['n_regions_auc_above_0p9']}/24")
    print(f"    median gap: {s['median_gap']:.3f}")
    # Top and bottom 5 regions
    pr = report["per_region"]["per_region"]
    sorted_regions = sorted(pr.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    print("    BEST 5 regions (by AUC):")
    for name, m in sorted_regions[:5]:
        print(f"      {name:14s} AUC={m['auc']:.3f}  gap={m['gap']:+.3f}  n={m['n_samples']}")
    print("    WORST 5 regions (by AUC):")
    for name, m in sorted_regions[-5:]:
        print(f"      {name:14s} AUC={m['auc']:.3f}  gap={m['gap']:+.3f}  n={m['n_samples']}")

    # Save
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report -> {args.out_json}")

    # Verdict
    print("\n" + "=" * 72)
    print("OVERALL VERDICT")
    print("=" * 72)
    issues = []
    if not report["rotation_utils"]["verdict_robust_to_non_orthogonal"]:
        issues.append("r6d_to_rotmat does not orthogonalize (Gram-Schmidt missing); "
                      "ok for SMPL-derived input but fragile.")
    if not report["acceleration"]["verdict_acc_contains_gravity"]:
        issues.append("Acceleration is gravity-removed -> gravity scorer is INERT on this dataset.")
    if not report["kinematic_uses_limits"]["joint_angle_limits_consulted_in_scorer"]:
        issues.append("Kinematic scorer DOES NOT consult precomputed joint angle limits "
                      "(scores |R_rel - I|, which has no biomech meaning).")

    # Pull mean accel AUC across n_sensors
    accel_aucs = []
    for k, v in report["combo_ranking"].items():
        if "accel" in v:
            for kind in ("vs_random", "vs_offchain_swap", "vs_lr_swap"):
                a = v["accel"][kind]["auc"]
                if not (isinstance(a, float) and (a != a)):  # not NaN
                    accel_aucs.append(a)
    if accel_aucs:
        print(f"  Acceleration scorer mean AUC across negatives: {np.mean(accel_aucs):.3f}")
    if issues:
        print("\n  REMAINING ISSUES:")
        for it in issues:
            print(f"   - {it}")
        print("\n  RECOMMENDATION:")
        print("   - Gravity scorer is INERT on vimu data; default weight = 0 in combined_scorer.")
        print("     Re-enable only if total acceleration (with gravity) is later available.")
        print("   - Kinematic + accel are the two load-bearing scorers; tune their relative weight.")
    else:
        print("  All scorers structurally sound. Tune weights and integrate.")


if __name__ == "__main__":
    main()
