"""
compute_training_stats.py
=======================
Compute training statistics for physics verification.

Computes:
1. Joint angle limits (from kinematic chain analysis)
2. Per-joint and per-activity acceleration distributions
3. Per-region gravity vectors (from static periods)

Usage:
    python compute_training_stats.py --train_data <train.npz> --output_dir stats/
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from smpl_regions import REGION_NAMES, REGION_PARENTS, get_kinematic_chain
from rotation_utils import r6d_to_rotmat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_imu_components(X_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract r6d and acceleration from sample.
    Handles both (6, T) and (9, T) inputs.
    """
    n_features = X_sample.shape[0]
    if n_features >= 9:
        r6d = X_sample[:6, :].T  # (T, 6)
        acc = X_sample[6:9, :].T   # (T, 3)
    else:  # Only r6d (6 features)
        r6d = X_sample[:6, :].T  # (T, 6)
        acc = np.zeros((r6d.shape[0], 3), dtype=np.float32)  # Dummy acceleration
    return r6d, acc


def detect_static_periods(
    r6d: np.ndarray,
    acc: np.ndarray,
    window: int = 10,
    ang_vel_thresh: float = 0.1,
    acc_var_thresh: float = 0.5,
) -> np.ndarray:
    """Detect static periods (low motion). Returns (T,) bool mask."""
    T = r6d.shape[0]
    static = np.zeros(T, dtype=bool)

    if T < 2:
        return static

    # Angular velocity from r6d - compute rotation matrix differences
    rotmat = r6d_to_rotmat(r6d)  # (T, 3, 3)
    dR = np.diff(rotmat, axis=0)  # (T-1, 3, 3)
    # Frobenius norm of dR as proxy for angular velocity
    ang_vel_norm = np.linalg.norm(dR, axis=(1, 2))  # (T-1,)
    ang_vel_norm = np.concatenate([[0], ang_vel_norm])  # (T,)

    # Acceleration variance in sliding window
    for t in range(T):
        start = max(0, t - window // 2)
        end = min(T, t + window // 2 + 1)
        acc_var = np.var(acc[start:end], axis=0).sum()
        if ang_vel_norm[t] < ang_vel_thresh and acc_var < acc_var_thresh:
            static[t] = True

    return static


# ---------------------------------------------------------------------------
# 1. Joint Angle Limits
# ---------------------------------------------------------------------------
def compute_joint_angle_limits(
    X: np.ndarray,
    y: np.ndarray,
    calibration: np.ndarray,
) -> Dict:
    """
    Compute joint angle limits from training data.

    For each parent-child pair in the kinematic tree:
    - Compute relative rotation R_rel = R_parent^T @ R_child
    - Store 5th/95th percentile as limits

    Returns:
        dict with 'loose' and 'strict' limits per joint pair
    """
    print("\n[compute_joint_angle_limits] Computing...")
    loose_limits = {}  # 5th-95th percentile
    strict_limits = {}  # 10th-90th percentile

    # Group samples by parent-child pairs
    parent_child_pairs = []
    for region_id, parent_id in REGION_PARENTS.items():
        if parent_id != -1:
            parent_child_pairs.append((parent_id, region_id))

    for parent_id, child_id in parent_child_pairs:
        # Get samples for parent and child
        parent_mask = y == parent_id
        child_mask = y == child_id

        if parent_mask.sum() == 0 or child_mask.sum() == 0:
            continue

        # Sample alignment: dataset construction guarantees that the i-th
        # masked sample of region r comes from the i-th source segment, so
        # parent_X[i] and child_X[i] are temporally aligned (same pose, same
        # frames). See regenerate_npz.convert_to_samples.
        n_samples = min(parent_mask.sum(), child_mask.sum())
        parent_X = X[parent_mask][:n_samples]
        child_X = X[child_mask][:n_samples]

        cal_p = calibration[parent_id]
        cal_c = calibration[child_id]

        rel_rotations: list[np.ndarray] = []
        for i in range(n_samples):
            r6d_p, _ = extract_imu_components(parent_X[i])  # (T, 6)
            r6d_c, _ = extract_imu_components(child_X[i])

            # Use ALL T frames, not just t=0.
            rot_p = r6d_to_rotmat(r6d_p)  # (T, 3, 3)
            rot_c = r6d_to_rotmat(r6d_c)
            rot_body_p = rot_p @ cal_p
            rot_body_c = rot_c @ cal_c
            # Relative rotation per frame: R_p^T @ R_c
            rel = np.einsum("...ij,...jk->...ik",
                            rot_body_p.transpose(0, 2, 1), rot_body_c)  # (T,3,3)
            for t in range(rel.shape[0]):
                rel_rotations.append(rotmat_to_euler_angles(rel[t]))

        if not rel_rotations:
            continue

        rel_rotations = np.array(rel_rotations)  # (N*T, 3)

        # Compute percentiles for each Euler angle
        loose_limits[(parent_id, child_id)] = {
            "min": np.percentile(rel_rotations, 5, axis=0),
            "max": np.percentile(rel_rotations, 95, axis=0),
        }
        strict_limits[(parent_id, child_id)] = {
            "min": np.percentile(rel_rotations, 10, axis=0),
            "max": np.percentile(rel_rotations, 90, axis=0),
        }

        print(f"  Pair ({REGION_NAMES[parent_id]}, {REGION_NAMES[child_id]}): "
              f"n={len(rel_rotations)}")

    return {"loose": loose_limits, "strict": strict_limits}


def rotmat_to_euler_angles(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles (ZYX order).
    Returns (3,) array [roll, pitch, yaw].
    """
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2 + 1e-8)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z], dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Acceleration Distributions
# ---------------------------------------------------------------------------
def compute_accel_distributions(
    X: np.ndarray,
    y: np.ndarray,
    activity_ids: Optional[np.ndarray],
) -> Dict:
    """
    Compute per-joint and per-activity acceleration distributions.

    For each region (and activity), compute:
    - Mean and covariance of acceleration statistics vector:
      [mean_acc_x, mean_acc_y, mean_acc_z, std_acc_x, std_acc_y, std_acc_z, peak_acc]

    Returns:
        dict with 'per_joint' and 'per_activity' distributions
    """
    print("\n[compute_accel_distributions] Computing...")

    per_joint = {}
    per_activity = {}

    # Per-joint distributions
    for region_id in range(24):
        mask = y == region_id
        if mask.sum() == 0:
            continue

        region_X = X[mask]
        stats_list = []

        for i in range(len(region_X)):
            _, acc = extract_imu_components(region_X[i])  # (T, 3)
            stats = np.concatenate([
                np.mean(acc, axis=0),  # (3,)
                np.std(acc, axis=0),    # (3,)
                [np.max(np.linalg.norm(acc, axis=1))],  # peak
            ])  # (7,)
            stats_list.append(stats)

        if not stats_list:
            continue

        stats_array = np.array(stats_list)  # (N, 7)
        per_joint[region_id] = {
            "mean": np.mean(stats_array, axis=0),
            "cov": np.cov(stats_array.T) + 1e-6 * np.eye(7),
            "n_samples": len(stats_list),
        }
        print(f"  {REGION_NAMES[region_id]}: n={len(stats_list)}")

    # Per-activity distributions
    if activity_ids is not None:
        activities = np.unique(activity_ids)
        for act_id in activities:
            act_mask = activity_ids == act_id
            if act_mask.sum() == 0:
                continue

            # For each region in this activity
            for region_id in range(24):
                mask = act_mask & (y == region_id)
                if mask.sum() == 0:
                    continue

                region_X = X[mask]
                stats_list = []

                for i in range(len(region_X)):
                    _, acc = extract_imu_components(region_X[i])
                    stats = np.concatenate([
                        np.mean(acc, axis=0),
                        np.std(acc, axis=0),
                        [np.max(np.linalg.norm(acc, axis=1))],
                    ])
                    stats_list.append(stats)

                if not stats_list:
                    continue

                stats_array = np.array(stats_list)
                per_activity[(region_id, int(act_id))] = {
                    "mean": np.mean(stats_array, axis=0),
                    "cov": np.cov(stats_array.T) + 1e-6 * np.eye(7),
                    "n_samples": len(stats_list),
                }

        print(f"  Per-activity: {len(per_activity)} region-activity pairs")

    return {"per_joint": per_joint, "per_activity": per_activity}


# ---------------------------------------------------------------------------
# 3. Gravity Distributions
# ---------------------------------------------------------------------------
def compute_gravity_distributions(
    X: np.ndarray,
    y: np.ndarray,
    calibration: np.ndarray,
) -> Dict:
    """
    Compute per-region gravity vectors from static periods.

    Returns:
        dict mapping region_id -> {"mean": (3,), "cov": (3,3)}
    """
    print("\n[compute_gravity_distributions] Computing...")

    gravity_dist = {}

    for region_id in range(24):
        mask = y == region_id
        if mask.sum() == 0:
            continue

        region_X = X[mask]
        gravity_vectors = []

        for i in range(len(region_X)):
            r6d, acc = extract_imu_components(region_X[i])

            # Detect static periods
            static = detect_static_periods(r6d, acc)
            if static.sum() < 10:  # Need at least 10 static samples
                continue

            # Average gravity in sensor frame
            grav_sensor = np.mean(acc[static], axis=0)  # (3,)

            # Transform to body frame
            cal = calibration[region_id]  # (3, 3)
            grav_body = cal.T @ grav_sensor

            gravity_vectors.append(grav_body)

        if len(gravity_vectors) < 3:
            continue

        gravity_array = np.array(gravity_vectors)  # (N, 3)
        gravity_dist[region_id] = {
            "mean": np.mean(gravity_array, axis=0),
            "cov": np.cov(gravity_array.T) + 1e-6 * np.eye(3),
            "n_samples": len(gravity_vectors),
        }
        print(f"  {REGION_NAMES[region_id]}: n={len(gravity_vectors)}")

    return gravity_dist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute training statistics")
    parser.add_argument("--train_data", required=True, help="Training .npz file")
    parser.add_argument("--output_dir", default="stats", help="Output directory")
    parser.add_argument("--calibration", default="calibration", help="Calibration directory")
    parser.add_argument("--no_activity", action="store_true", help="Skip per-activity stats")
    args = parser.parse_args()

    print("=" * 60)
    print("Computing Training Statistics for Physics Verification")
    print("=" * 60)

    # Load training data
    data = np.load(args.train_data, allow_pickle=False)
    X = data["X"]  # (N, 9, T)
    y = data["y"]  # (N,)
    activity_ids = data.get("activity_ids", None)

    print(f"\nTraining data: X={X.shape}, y={y.shape}")
    if activity_ids is not None:
        print(f"  activity_ids: {activity_ids.shape}")

    # Load calibration
    calib_path = os.path.join(args.calibration, "region_sensor_rotmats.npy")
    if os.path.exists(calib_path):
        calibration = np.load(calib_path)
        print(f"  Loaded calibration shape: {calibration.shape}")
    else:
        print("Warning: No calibration found, using identity")
        calibration = np.zeros((24, 3, 3), dtype=np.float32)
        for i in range(24):
            calibration[i] = np.eye(3, dtype=np.float32)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Joint angle limits
    joint_limits = compute_joint_angle_limits(X, y, calibration)
    np.save(os.path.join(args.output_dir, "joint_angle_limits.npy"), joint_limits)
    print(f"\nSaved joint_angle_limits.npy")

    # 2. Acceleration distributions
    accel_dist = compute_accel_distributions(
        X, y, None if args.no_activity else activity_ids
    )
    np.save(os.path.join(args.output_dir, "accel_distributions.npy"), accel_dist)
    print(f"Saved accel_distributions.npy")

    # 3. Gravity distributions
    grav_dist = compute_gravity_distributions(X, y, calibration)
    np.save(os.path.join(args.output_dir, "gravity_distributions.npy"), grav_dist)
    print(f"Saved gravity_distributions.npy")

    print("\n" + "=" * 60)
    print("Done! Statistics saved to", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
