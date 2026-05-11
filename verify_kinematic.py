"""
verify_kinematic.py
===================
Targeted audit of the kinematic scorer. Five tests:

  T1. Envelope sanity. Print per-pair Euler envelope. Tight pairs
      (knees, elbows) should have small range on one axis.
  T2. Self-test. Score the TRUE combo on its own training data; the
      kinematic score per pair should be high (close to 1).
  T3. Monotonicity under perturbation. Apply a global rotation R(theta)
      to one child sensor; the pair score should fall monotonically as
      theta grows from 0 to pi.
  T4. Frame-level breakdown. For one (parent,child) pair, print per-frame
      score on TRUE data vs L/R-swapped data.
  T5. Pair-level swap audit. For a single 5-sensor combo, show every pair
      score under TRUE assignment, L/R-swap, off-chain swap, and
      data-shuffle.

Run:
    .venv/bin/python verify_kinematic.py --data data/single_subject_train.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotation_utils import r6d_to_rotmat, rotmat_to_r6d  # noqa: E402
from smpl_regions import REGION_NAMES, REGION_PARENTS, SYMMETRY_PAIRS  # noqa: E402
from verify_combo import (  # noqa: E402
    extract_imu_components,
    load_calibration,
    load_data,
    load_stats,
    score_kinematic_chain,
    _rotmat_to_euler_zyx_batch,
)


def _segment_index(y: np.ndarray) -> dict:
    occ: dict[int, list[int]] = {}
    for i, r in enumerate(y.tolist()):
        occ.setdefault(int(r), []).append(i)
    return occ


def t1_envelope_sanity(stats: dict) -> None:
    print("\n=== T1. Envelope sanity ===")
    jal = stats["joint_angle_limits"]
    pairs = [
        (4, 7),    # l_thigh -> l_shin (knee, ~hinge)
        (18, 20),  # l_upper_arm -> l_forearm (elbow, ~hinge)
        (1, 4),    # l_hip -> l_thigh (ball joint)
        (12, 15),  # neck -> head
        (0, 3),    # pelvis -> spine_lower
    ]
    print(f"{'pair':30s} | {'loose min (rad)':>30s} | {'loose max (rad)':>30s} | {'span':>20s}")
    for (p, c) in pairs:
        if (p, c) not in jal["loose"]:
            print(f"{REGION_NAMES[p]+'->'+REGION_NAMES[c]:30s} | (no envelope)")
            continue
        lo = jal["loose"][(p, c)]["min"]
        hi = jal["loose"][(p, c)]["max"]
        span = hi - lo
        print(f"{REGION_NAMES[p]+'->'+REGION_NAMES[c]:30s} | "
              f"{np.array2string(lo, precision=2, suppress_small=True):>30s} | "
              f"{np.array2string(hi, precision=2, suppress_small=True):>30s} | "
              f"{np.array2string(span, precision=2):>20s}")
    print("Expectation: knee/elbow should have one axis with much larger span than the other two.")


def t2_self_test(X, y, calibration, stats) -> None:
    print("\n=== T2. Self-test on TRUE combo ===")
    occ = _segment_index(y)
    n_segs = min(len(v) for v in occ.values())
    # Use first segment, full 24-region combo
    combo = list(range(24))
    Xs = np.stack([X[occ[r][0]] for r in combo])
    score, diag = score_kinematic_chain(combo, Xs, calibration, stats["joint_angle_limits"])
    print(f"Combined kinematic score (segment 0, all 24 regions): {score:.3f}")
    print(f"  valid_pairs evaluated: {diag['valid_pairs']}")
    pair_scores = diag["pair_scores"]
    arr = np.array(list(pair_scores.values()))
    print(f"  per-pair score: mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
          f"min={arr.min():.3f}  max={arr.max():.3f}")
    print(f"  worst 5 pairs:")
    for k, v in sorted(pair_scores.items(), key=lambda kv: kv[1])[:5]:
        p, c = map(int, k.split("->"))
        print(f"    {REGION_NAMES[p]:>14s} -> {REGION_NAMES[c]:<14s}  score={v:.3f}")


def t3_monotonicity(X, y, calibration, stats) -> None:
    print("\n=== T3. Monotonicity under controlled perturbation ===")
    print("Pair = l_thigh (parent) -> l_shin (child). Perturb child by R_x(theta).")
    occ = _segment_index(y)
    pid, cid = 4, 7
    # Take one segment with both regions
    Xp = X[occ[pid][0]].copy()  # (9, T)
    Xc = X[occ[cid][0]].copy()
    thetas = np.linspace(0, np.pi, 9)
    print(f"{'theta(rad)':>10s} | {'pair_score':>10s}")
    for theta in thetas:
        Xc_pert = Xc.copy()
        # Build R_x(theta)
        Rp = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]], dtype=np.float32)
        # Apply to every frame's r6d: R_new = Rp @ R_old
        r6d_c = Xc_pert[:6, :].T  # (T, 6)
        Rmat = r6d_to_rotmat(r6d_c)  # (T, 3, 3)
        Rmat_new = np.einsum("ij,tjk->tik", Rp, Rmat)
        r6d_new = rotmat_to_r6d(Rmat_new)  # (T, 6)
        Xc_pert[:6, :] = r6d_new.T
        Xs = np.stack([Xp, Xc_pert])
        combo = [pid, cid]
        sc, diag = score_kinematic_chain(combo, Xs, calibration, stats["joint_angle_limits"])
        print(f"{theta:>10.3f} | {sc:>10.3f}")
    print("Expectation: monotone decrease (small bumps OK) as theta grows away from 0.")


def t4_frame_breakdown(X, y, calibration, stats) -> None:
    print("\n=== T4. Frame-level score breakdown ===")
    print("Pair = l_thigh -> l_shin (real envelope). Compare:")
    print("  (a) correct data: l_thigh + l_shin")
    print("  (b) data swapped: l_thigh + r_shin   (envelope lookup unchanged, data mismatched)")
    occ = _segment_index(y)
    pid, cid = 4, 7  # l_thigh, l_shin
    Xp = X[occ[pid][0]]
    Xc_true = X[occ[cid][0]]
    Xc_swap = X[occ[8][0]]  # r_shin data, but label still l_shin

    def per_frame_score(combo, Xs):
        r6d_p, _ = extract_imu_components(Xs[0])
        r6d_c, _ = extract_imu_components(Xs[1])
        Rp = r6d_to_rotmat(r6d_p) @ calibration[combo[0]]
        Rc = r6d_to_rotmat(r6d_c) @ calibration[combo[1]]
        Rrel = np.einsum("...ij,...jk->...ik", Rp.transpose(0, 2, 1), Rc)
        euler = _rotmat_to_euler_zyx_batch(Rrel)
        jal = stats["joint_angle_limits"]
        key = (combo[0], combo[1])
        envelope = jal["strict"] if key in jal["strict"] else jal["loose"]
        if key not in envelope:
            return None
        lo = envelope[key]["min"]
        hi = envelope[key]["max"]
        mid = 0.5 * (lo + hi)
        half = np.maximum(0.5 * (hi - lo), 1e-3)
        outside = np.maximum(np.abs(euler - mid) - half, 0.0) / half
        return np.exp(-outside.sum(axis=-1))

    pf_true = per_frame_score([pid, cid], np.stack([Xp, Xc_true]))
    pf_swap = per_frame_score([pid, cid], np.stack([Xp, Xc_swap]))
    if pf_true is None or pf_swap is None:
        print("  envelope missing; skipping")
        return
    print(f"  TRUE  data: per-frame mean={pf_true.mean():.3f} "
          f"frac_within={(pf_true>0.99).mean():.3f}  min={pf_true.min():.3f}")
    print(f"  SWAP  data: per-frame mean={pf_swap.mean():.3f} "
          f"frac_within={(pf_swap>0.99).mean():.3f}  min={pf_swap.min():.3f}")
    print("Expectation: TRUE mean >> SWAP mean; TRUE frac_within close to 1.")


def t5_pair_swap_audit(X, y, calibration, stats) -> None:
    print("\n=== T5. Pair-level audit on a 5-sensor combo ===")
    occ = _segment_index(y)
    combo_true = [0, 1, 4, 7, 10]   # pelvis, l_hip, l_thigh, l_shin, l_foot
    Xs = np.stack([X[occ[r][0]] for r in combo_true])

    def kin_pairs(combo, Xs):
        sc, diag = score_kinematic_chain(combo, Xs, calibration, stats["joint_angle_limits"])
        return sc, diag.get("pair_scores", {})

    s_true, p_true = kin_pairs(combo_true, Xs)
    # L/R swap: l_thigh <-> r_thigh
    combo_lr = [0, 1, 5, 7, 10]
    s_lr, p_lr = kin_pairs(combo_lr, Xs)  # data still l_thigh, label says r_thigh
    # Off-chain: replace l_foot with r_collar
    combo_off = [0, 1, 4, 7, 14]
    s_off, p_off = kin_pairs(combo_off, Xs)
    # Random shuffle of labels
    rng = np.random.default_rng(0)
    combo_shuf = list(rng.permutation(combo_true))
    s_shuf, p_shuf = kin_pairs(combo_shuf, Xs)

    print(f"TRUE   combined={s_true:.3f}  pairs={ {k: round(v,3) for k,v in p_true.items()} }")
    print(f"L/R    combined={s_lr:.3f}    pairs={ {k: round(v,3) for k,v in p_lr.items()} }")
    print(f"OFFCH  combined={s_off:.3f}   pairs={ {k: round(v,3) for k,v in p_off.items()} }")
    print(f"SHUF   combined={s_shuf:.3f}  pairs={ {k: round(v,3) for k,v in p_shuf.items()} }")
    print("Expectation: TRUE > {L/R, OFFCH, SHUF}, with the gap concentrated on the swapped pair.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/single_subject_train.npz")
    p.add_argument("--stats_dir", default="stats")
    p.add_argument("--calibration", default="calibration")
    args = p.parse_args()

    X, y, _, _ = load_data(args.data)
    calibration = load_calibration(args.calibration)
    stats = load_stats(args.stats_dir)

    t1_envelope_sanity(stats)
    t2_self_test(X, y, calibration, stats)
    t3_monotonicity(X, y, calibration, stats)
    t4_frame_breakdown(X, y, calibration, stats)
    t5_pair_swap_audit(X, y, calibration, stats)


if __name__ == "__main__":
    main()
