"""
tune_weights.py
==============
Tune weights for the combined physics verification scorer.

Tests multi-sensor combos to properly evaluate the kinematic scorer.

Usage:
    python tune_weights.py --data <test.npz>
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from smpl_regions import REGION_NAMES
from verify_combo import (
    load_data,
    load_calibration,
    load_stats,
    combined_scorer,
)


# ---------------------------------------------------------------------------
# Generate test combos with multiple sensors
# ---------------------------------------------------------------------------
def generate_test_combos(
    y: np.ndarray,
    n_sensors: int = 3,
    n_samples: int = 50,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate test combos: (true_combo, X_samples).
    Each combo is a set of n_sensors consecutive samples from the data.
    """
    np.random.seed(seed)
    combos = []

    # Group samples by region
    region_indices = {}
    for idx, region in enumerate(y.tolist()):
        if region not in region_indices:
            region_indices[region] = []
        region_indices[region].append(idx)

    # Generate combos
    used = set()
    for _ in range(n_samples * 2):  # Try more to get valid combos
        if len(combos) >= n_samples:
            break

        # Pick n_sensors random regions
        available_regions = [r for r in region_indices if len(region_indices[r]) > 0]
        if len(available_regions) < n_sensors:
            continue

        regions = np.random.choice(available_regions, size=n_sensors, replace=False).tolist()

        # Check if we've used this combo
        key = tuple(sorted(regions))
        if key in used:
            continue
        used.add(key)

        # Get one sample per region
        indices = [region_indices[r].pop() for r in regions]
        combos.append((regions, indices))

    return combos


# ---------------------------------------------------------------------------
# Main: Compare weight combinations
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tune physics scorer weights")
    parser.add_argument("--data", required=True, help="Test .npz file")
    parser.add_argument("--calibration", default="calibration", help="Calibration directory")
    parser.add_argument("--stats_dir", default="stats", help="Statistics directory")
    parser.add_argument("--n_sensors", type=int, default=3, help="Sensors per combo")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of combos to test")
    args = parser.parse_args()

    print("=" * 60)
    print("Tuning Physics Scorer Weights")
    print("=" * 60)

    # Load data
    X, y, subject_ids, _ = load_data(args.data)
    print(f"\nData: X={X.shape}, y={y.shape}")

    # Load calibration and stats
    calibration = load_calibration(args.calibration)
    stats = load_stats(args.stats_dir)

    # Generate test combos
    test_combos = generate_test_combos(y, args.n_sensors, args.n_samples)
    print(f"Generated {len(test_combos)} test combos")

    # Weight combinations to test.
    # Gravity component is included for completeness but is INERT on
    # vimu_joints data (linear acceleration; no gravity to recover).
    weight_combos = [
        ("Kinematic only", (1.0, 0.0, 0.0)),
        ("Gravity only (inert on vimu)", (0.0, 1.0, 0.0)),
        ("Acceleration only", (0.0, 0.0, 1.0)),
        ("Default (0.5/0.0/0.5)", (0.5, 0.0, 0.5)),
        ("Kinematic-heavy (0.7/0.0/0.3)", (0.7, 0.0, 0.3)),
        ("Accel-heavy (0.3/0.0/0.7)", (0.3, 0.0, 0.7)),
        ("Legacy (0.4/0.3/0.3)", (0.4, 0.3, 0.3)),
    ]

    # Evaluate each weight combination
    results = {name: {"true_scores": [], "false_scores": []} for name, _ in weight_combos}

    for true_combo, indices in test_combos:
        X_samples = X[indices]  # (n_sensors, 9, T)

        for name, weights in weight_combos:
            # Score true combo
            true_score, _ = combined_scorer(
                true_combo, X_samples, calibration, stats, weights=weights
            )
            results[name]["true_scores"].append(true_score)

            # Score a random wrong combo (perturb one region)
            wrong_combo = true_combo.copy()
            wrong_idx = np.random.randint(args.n_sensors)
            available = [r for r in range(24) if r not in true_combo]
            wrong_combo[wrong_idx] = np.random.choice(available)
            wrong_score, _ = combined_scorer(
                wrong_combo, X_samples, calibration, stats, weights=weights
            )
            results[name]["false_scores"].append(wrong_score)

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    best_name = None
    best_gap = -1.0

    for name, _ in weight_combos:
        true_avg = np.mean(results[name]["true_scores"])
        false_avg = np.mean(results[name]["false_scores"])
        gap = true_avg - false_avg

        print(f"\n{name}:")
        print(f"  True combo avg score: {true_avg:.4f}")
        print(f"  False combo avg score: {false_avg:.4f}")
        print(f"  Gap (true - false): {gap:.4f}")

        if gap > best_gap:
            best_gap = gap
            best_name = name

    print("\n" + "=" * 60)
    print(f"Best weights: {best_name}")
    print(f"  Gap: {best_gap:.4f}")
    print("=" * 60)

    # Save best weights
    for name, weights in weight_combos:
        if name == best_name:
            output_path = os.path.join(args.stats_dir, "best_weights.npy")
            np.save(output_path, {"name": name, "weights": weights, "gap": best_gap})
            print(f"\nSaved best weights to {output_path}")
            break


if __name__ == "__main__":
    main()
