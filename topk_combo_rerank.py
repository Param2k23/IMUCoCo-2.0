"""
topk_combo_rerank.py
====================
Bridge between the classifier's top-k output (from `evaluate.predict_all_topk`)
and the per-combo physics scorer (`verify_combo.combined_scorer`).

Pipeline (per the user's vision):
    n IMU streams
        -> classifier emits top-k per stream
        -> aggregate top-k across the n_windows of each stream
        -> Cartesian-product enumerate candidate combos
        -> physics eliminator (anatomical hard-mode optional)
        -> physics re-rank by  P(classifier) * P(physics)
        -> argmax = predicted region tuple

This file provides three reusable functions and one one-shot wrapper. It is
imported by `evaluate.py --physics_rerank` and is also runnable as a smoke
test (`python topk_combo_rerank.py`).
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from temporal_rerank import _build_prob_matrix
from verify_combo import combined_scorer
from smpl_regions import NUM_REGIONS, REGION_NAMES


# ---------------------------------------------------------------------------
# Step 1: aggregate per-window top-k into per-sensor top-k
# ---------------------------------------------------------------------------
def aggregate_topk_per_sensor(
    topk_indices: np.ndarray,   # (n_sensors, n_windows, k_max)
    topk_probs:   np.ndarray,   # (n_sensors, n_windows, k_max)
    k: int = 3,
    n_classes: int = NUM_REGIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse n_windows of top-k per sensor into a single per-sensor top-k.

    Strategy
    --------
    For each sensor, build the dense (n_windows, n_classes) probability
    matrix using `temporal_rerank._build_prob_matrix`, mean over the
    n_windows axis, then read off the top-k. This matches the `prob_sum`
    rerank strategy from temporal_rerank.py but on a per-sensor scope.

    Returns
    -------
    sensor_topk_idx   : (n_sensors, k) int64
    sensor_topk_probs : (n_sensors, k) float32  (re-normalised so that the
                        per-sensor top-k probs sum to 1, mirroring how
                        evaluate.predict_all_topk already returns
                        un-normalised topk slices of softmax outputs)
    """
    n_sensors, n_windows, _ = topk_indices.shape
    out_idx = np.zeros((n_sensors, k), dtype=np.int64)
    out_p = np.zeros((n_sensors, k), dtype=np.float32)

    for s in range(n_sensors):
        prob_matrix = _build_prob_matrix(
            topk_indices[s], topk_probs[s], n_classes
        )                                                     # (n_windows, C)
        mean_probs = prob_matrix.mean(axis=0)                 # (C,)
        # Top-k by descending probability.
        top_idx = np.argsort(-mean_probs)[:k]
        out_idx[s] = top_idx
        out_p[s] = mean_probs[top_idx]

    return out_idx, out_p


# ---------------------------------------------------------------------------
# Step 2: enumerate candidate combos by Cartesian product
# ---------------------------------------------------------------------------
def enumerate_combos(
    per_sensor_topk_idx:   np.ndarray,   # (n_sensors, k)
    per_sensor_topk_probs: np.ndarray,   # (n_sensors, k)
    distinct_regions: bool = True,
) -> list[tuple[tuple[int, ...], float]]:
    """
    Cartesian product of per-sensor top-k -> candidate combo list.

    Each entry is `((region_id_for_sensor_0, ..., region_id_for_sensor_{n-1}),
    classifier_prob)` where `classifier_prob` is the product of per-sensor
    softmax probabilities. With k=3 and n_sensors=5 this is 243 candidates
    worst case (PHYSICS_VERIFICATION_PLAN.md Appendix B).

    `distinct_regions=True` filters combos in which two physical sensors are
    assigned to the same region (impossible in practice).
    """
    n_sensors, k = per_sensor_topk_idx.shape
    combos: list[tuple[tuple[int, ...], float]] = []
    for choices in itertools.product(range(k), repeat=n_sensors):
        regions = tuple(int(per_sensor_topk_idx[s, c]) for s, c in enumerate(choices))
        if distinct_regions and len(set(regions)) != n_sensors:
            continue
        prob = float(np.prod([
            per_sensor_topk_probs[s, c] for s, c in enumerate(choices)
        ]))
        combos.append((regions, prob))
    # Pre-sort by classifier prob descending (helpful for early exit / debug).
    combos.sort(key=lambda x: x[1], reverse=True)
    return combos


# ---------------------------------------------------------------------------
# Step 3: re-rank candidate combos with physics
# ---------------------------------------------------------------------------
def rerank_combos(
    combos: list[tuple[tuple[int, ...], float]],
    X_per_sensor: np.ndarray,            # (n_sensors, 9, T)
    calibration: np.ndarray,             # (24, 3, 3)
    stats: dict,
    weights: tuple = (0.4, 0.0, 0.4, 0.2),
    joint_limits_hard: Optional[float] = None,
    activity_id: Optional[int] = None,
    eliminator_only: bool = False,
) -> list[dict]:
    """
    For each candidate combo:
        1. Run combined_scorer (with optional anatomical hard eliminator).
        2. final_score = classifier_prob * physics_score.
        3. If eliminator_only=True, skip the physics multiplicative rerank
           (final_score = classifier_prob if not eliminated else 0). Useful
           to isolate the eliminator's contribution from the soft re-ranking.

    Returns sorted (best first) list of dicts. n_eliminated is reported so
    callers can log it.
    """
    n_sensors = X_per_sensor.shape[0]
    results: list[dict] = []

    for regions, cls_prob in combos:
        if len(regions) != n_sensors:
            raise ValueError(
                f"combo length {len(regions)} != n_sensors {n_sensors}"
            )

        physics_score, diag = combined_scorer(
            list(regions),
            X_per_sensor,
            calibration,
            stats,
            weights=weights,
            activity_id=activity_id,
            joint_limits_hard=joint_limits_hard,
        )
        eliminated = bool(diag.get("hard_eliminated", False))

        if eliminator_only:
            final_score = 0.0 if eliminated else cls_prob
        else:
            final_score = cls_prob * physics_score

        results.append({
            "combo": regions,
            "classifier_prob": cls_prob,
            "physics_score": physics_score,
            "joint_limits_score": diag.get("joint_limits_score"),
            "hard_eliminated": eliminated,
            "violator": diag.get("violator"),
            "final_score": final_score,
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# One-shot wrapper
# ---------------------------------------------------------------------------
def predict_with_topk_physics(
    topk_indices: np.ndarray,             # (n_sensors, n_windows, k_max)
    topk_probs:   np.ndarray,             # (n_sensors, n_windows, k_max)
    X_per_sensor: np.ndarray,             # (n_sensors, 9, T)
    calibration:  np.ndarray,             # (24, 3, 3)
    stats: dict,
    k: int = 3,
    weights: tuple = (0.4, 0.0, 0.4, 0.2),
    joint_limits_hard: Optional[float] = None,
    activity_id: Optional[int] = None,
    distinct_regions: bool = True,
    eliminator_only: bool = False,
) -> list[dict]:
    """
    Aggregate top-k -> enumerate -> rerank in one call. Returns the sorted
    list of candidate dicts (best first). The top of the list is the
    final prediction.
    """
    sensor_idx, sensor_p = aggregate_topk_per_sensor(
        topk_indices, topk_probs, k=k
    )
    combos = enumerate_combos(sensor_idx, sensor_p, distinct_regions=distinct_regions)
    return rerank_combos(
        combos, X_per_sensor, calibration, stats,
        weights=weights, joint_limits_hard=joint_limits_hard,
        activity_id=activity_id, eliminator_only=eliminator_only,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Tiny synthetic check (no model, no data files required)."""
    rng = np.random.default_rng(0)
    n_sensors, n_windows, k_max = 3, 4, 5
    n_classes = NUM_REGIONS

    # True regions: pelvis(0), l_shoulder(16), l_forearm(20)
    true = [0, 16, 20]
    indices = np.zeros((n_sensors, n_windows, k_max), dtype=np.int64)
    probs = np.zeros((n_sensors, n_windows, k_max), dtype=np.float32)
    for s in range(n_sensors):
        for w in range(n_windows):
            # 70% chance the true region is top-1, 30% it's top-2
            if rng.random() < 0.7:
                top = true[s]; alt = (true[s] + 1) % n_classes
            else:
                top = (true[s] + 1) % n_classes; alt = true[s]
            others = [c for c in range(n_classes)
                      if c not in (top, alt)]
            extras = rng.choice(others, size=k_max - 2, replace=False)
            indices[s, w, 0] = top
            indices[s, w, 1] = alt
            indices[s, w, 2:] = extras
            raw = rng.dirichlet(np.ones(k_max))
            probs[s, w] = np.sort(raw)[::-1]

    # Aggregate -> per-sensor top-k
    sensor_idx, sensor_p = aggregate_topk_per_sensor(indices, probs, k=3)
    print("Per-sensor top-3:")
    for s in range(n_sensors):
        labels = [REGION_NAMES[i] for i in sensor_idx[s]]
        print(f"  sensor {s} (true={REGION_NAMES[true[s]]}):"
              f" {list(zip(labels, sensor_p[s].tolist()))}")

    # Enumerate combos
    combos = enumerate_combos(sensor_idx, sensor_p, distinct_regions=True)
    print(f"\nEnumerated {len(combos)} distinct combos (top 5 by classifier):")
    for c, p in combos[:5]:
        labels = str([REGION_NAMES[r] for r in c])
        print(f"  {labels:80s}  cls={p:.4g}")

    # Rerank without stats (combined_scorer falls back to neutral 0.5 — this
    # only checks the wiring, not the physics signal).
    fake_X = rng.standard_normal(
        (n_sensors, 9, 64), dtype=np.float32
    ).astype(np.float32)
    fake_cal = np.tile(np.eye(3, dtype=np.float32)[None], (24, 1, 1))
    fake_stats = {
        "joint_angle_limits": None,
        "accel_distributions": None,
        "gravity_distributions": None,
        "per_region_orientation_limits": None,
    }
    ranked = rerank_combos(
        combos, fake_X, fake_cal, fake_stats,
        weights=(0.4, 0.0, 0.4, 0.2),
    )
    print("\nTop-3 after physics rerank (with no stats — wiring only):")
    for r in ranked[:3]:
        labels = [REGION_NAMES[c] for c in r["combo"]]
        print(f"  {labels}  final={r['final_score']:.4g}"
              f"  cls={r['classifier_prob']:.4g}"
              f"  phys={r['physics_score']:.4g}"
              f"  elim={r['hard_eliminated']}")
    print("OK")
