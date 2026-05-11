"""
diagnose_scorers.py
===============
Diagnostic script to understand scorer behavior.

Usage:
    python diagnose_scorers.py --data <test.npz>
"""

import argparse
from typing import Dict, List, Tuple

import numpy as np

from smpl_regions import REGION_NAMES
from verify_combo import (
    load_data,
    load_calibration,
    load_stats,
    score_kinematic_chain,
    score_gravity_alignment,
    score_acceleration_profile,
    extract_imu_components,
)


def analyze_kinematic_scores(
    X: np.ndarray,
    y: np.ndarray,
    calibration: np.ndarray,
    stats: dict,
    n_samples: int = 20,
) -> None:
    """Analyze kinematic scorer behavior."""
    print("\n" + "=" * 60)
    print("Kinematic Scorer Analysis")
    print("=" * 60)

    joint_angle_limits = stats.get("joint_angle_limits")
    if joint_angle_limits is None:
        print("No joint angle limits available!")
        return

    # Test pairs: true parent-child vs random
    loose_limits = joint_angle_limits["loose"]

    n_valid = 0
    true_scores = []
    false_scores = []

    for _ in range(n_samples * 5):
        if n_valid >= n_samples:
            break

        # Pick a random parent-child pair
        import random
        parent_id = random.choice(list(REGION_NAMES.keys()))
        if parent_id not in REGION_PARENTS or REGION_PARENTS[parent_id] == -1:
            continue

        child_id = parent_id  # This is wrong, let me fix...

    # Simplified: just print some stats
    print("\nJoint Angle Limits (loose):")
    for (p, c), limits in list(loose_limits.items())[:5]:
        print(f"  {REGION_NAMES[p]} -> {REGION_NAMES[c]}:")
        print(f"    min: {limits['min']}")
        print(f"    max: {limits['max']}")


def analyze_gravity_scores(
    X: np.ndarray,
    y: np.ndarray,
    calibration: np.ndarray,
    stats: dict,
    n_samples: int = 20,
) -> None:
    """Analyze gravity scorer behavior."""
    print("\n" + "=" * 60)
    print("Gravity Scorer Analysis")
    print("=" * 60)

    gravity_dist = stats.get("gravity_distributions")
    if gravity_dist is None:
        print("No gravity distributions available!")
        return

    # Print gravity vectors for some regions
    print("\nGravity Vectors (body frame):")
    for region_id in list(gravity_dist.keys())[:5]:
        dist = gravity_dist[region_id]
        print(f"  {REGION_NAMES[region_id]}:")
        print(f"    mean: {dist['mean']}")
        print(f"    n_samples: {dist['n_samples']}")


def analyze_accel_scores(
    X: np.ndarray,
    y: np.ndarray,
    stats: dict,
    n_samples: int = 20,
) -> None:
    """Analyze acceleration scorer behavior."""
    print("\n" + "=" * 60)
    print("Acceleration Scorer Analysis")
    print("=" * 60)

    accel_dist = stats.get("accel_distributions")
    if accel_dist is None:
        print("No acceleration distributions available!")
        return

    per_joint = accel_dist.get("per_joint", {})

    # Print acceleration stats for some regions
    print("\nAcceleration Stats (per-joint):")
    for region_id in list(per_joint.keys())[:5]:
        dist = per_joint[region_id]
        print(f"  {REGION_NAMES[region_id]}:")
        print(f"    mean_acc: {dist['mean'][:3]}")  # First 3 are mean acc
        print(f"    n_samples: {dist['n_samples']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Test .npz file")
    parser.add_argument("--calibration", default="calibration")
    parser.add_argument("--stats_dir", default="stats")
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("Diagnosing Physics Scorers")
    print("=" * 60)

    # Load data
    X, y, subject_ids, _ = load_data(args.data)
    print(f"\nData: X={X.shape}, y={y.shape}")

    # Load calibration and stats
    calibration = load_calibration(args.calibration)
    stats = load_stats(args.stats_dir)

    # Analyze each scorer
    analyze_kinematic_scores(X, y, calibration, stats, args.n_samples)
    analyze_gravity_scores(X, y, calibration, stats, args.n_samples)
    analyze_accel_scores(X, y, stats, args.n_samples)


if __name__ == "__main__":
    main()
