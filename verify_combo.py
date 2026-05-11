"""
verify_combo.py
==============
Physics verification scorer for IMU sensor placement classification.

Implements the Combined Multi-Signal Scorer (Option 4 from PHYSICS_VERIFICATION_PLAN.md):
- Option 1: Kinematic Chain Consistency Scorer
- Option 2: Gravity Alignment Scorer
- Option 3: Acceleration Profile Scorer
- Option 4: Combined weighted scorer

Usage:
    python verify_combo.py --data <test.npz> --stats <stats_dir> --calibration <calibration_dir>
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Local imports
from smpl_regions import REGION_NAMES, REGION_PARENTS, get_kinematic_chain
from rotation_utils import r6d_to_rotmat


def _rotmat_to_euler_zyx_batch(R: np.ndarray) -> np.ndarray:
    """Batched ZYX Euler conversion. R: (..., 3, 3) -> (..., 3) [x, y, z]."""
    R = np.asarray(R, dtype=np.float32)
    sy = np.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2 + 1e-8)
    x = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    y = np.arctan2(-R[..., 2, 0], sy)
    z = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SMOOTH_WINDOW = 10  # frames for static detection
STATIC_ANG_VEL_THRESH = 0.1  # rad/s
STATIC_ACC_VAR_THRESH = 0.5  # m/s^2 variance


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load preprocessed .npz file.

    Returns:
        X: (N, 9, T) float32 - IMU data (r6d + acc)
        y: (N,) int64 - true region labels
        subject_ids: (N,) int64
        activity_ids: (N,) int64 or None
    """
    data = np.load(data_path, allow_pickle=False)
    X = data["X"]  # (N, 9, T)
    y = data["y"]  # (N,)
    subject_ids = data["subject_ids"]  # (N,)

    activity_ids = data.get("activity_ids", None)
    if activity_ids is not None:
        activity_ids = activity_ids.astype(np.int64)

    print(f"[load_data] Loaded {data_path}")
    print(f"  X: {X.shape} ({X.dtype})")
    print(f"  y: {y.shape} ({y.dtype}), classes={len(np.unique(y))}")
    print(f"  subject_ids: {subject_ids.shape}")
    print(f"  activity_ids: {activity_ids.shape if activity_ids is not None else None}")

    return X, y, subject_ids, activity_ids


def load_calibration(calib_path: str) -> np.ndarray:
    """Load sensor-to-body calibration matrices."""
    path = os.path.join(calib_path, "region_sensor_rotmats.npy")
    if not os.path.exists(path):
        print(f"[load_calibration] WARNING: {path} not found, using identity")
        return np.tile(np.eye(3, dtype=np.float32)[None, :, :], (24, 1, 1))
    rotmats = np.load(path)
    print(f"[load_calibration] Loaded {path} shape={rotmats.shape}")
    return rotmats


def load_stats(stats_dir: str) -> dict:
    """
    Load precomputed statistics.

    Expected files:
        - joint_angle_limits.npy: dict with 'loose' and 'strict' limits
        - accel_distributions.npy: dict with per-joint and per-activity stats
        - gravity_distributions.npy: dict with per-region gravity vectors
    """
    stats = {}

    # Joint angle limits
    angle_path = os.path.join(stats_dir, "joint_angle_limits.npy")
    if os.path.exists(angle_path):
        stats["joint_angle_limits"] = np.load(angle_path, allow_pickle=True).item()
        print(f"[load_stats] Loaded joint_angle_limits")
    else:
        print(f"[load_stats] WARNING: {angle_path} not found")
        stats["joint_angle_limits"] = None

    # Acceleration distributions
    accel_path = os.path.join(stats_dir, "accel_distributions.npy")
    if os.path.exists(accel_path):
        stats["accel_distributions"] = np.load(accel_path, allow_pickle=True).item()
        print(f"[load_stats] Loaded accel_distributions")
    else:
        print(f"[load_stats] WARNING: {accel_path} not found")
        stats["accel_distributions"] = None

    # Gravity distributions
    grav_path = os.path.join(stats_dir, "gravity_distributions.npy")
    if os.path.exists(grav_path):
        stats["gravity_distributions"] = np.load(grav_path, allow_pickle=True).item()
        print(f"[load_stats] Loaded gravity_distributions")
    else:
        print(f"[load_stats] WARNING: {grav_path} not found")
        stats["gravity_distributions"] = None

    return stats


# ---------------------------------------------------------------------------
# Helper: Extract sensor orientation and acceleration
# ---------------------------------------------------------------------------
def extract_imu_components(X_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract r6d orientation and acceleration from X sample.
    Handles both (6, T) and (9, T) inputs.

    Args:
        X_sample: (9, T) or (6, T) - IMU data for one sample

    Returns:
        r6d: (T, 6) - 6D rotation
        acc: (T, 3) - acceleration in sensor frame
    """
    n_features = X_sample.shape[0]
    if n_features >= 9:
        r6d = X_sample[:6, :].T  # (T, 6)
        acc = X_sample[6:9, :].T  # (T, 3)
    else:  # Only r6d (6 features)
        r6d = X_sample[:6, :].T  # (T, 6)
        acc = np.zeros((r6d.shape[0], 3), dtype=np.float32)  # Dummy acceleration
    return r6d, acc


# ---------------------------------------------------------------------------
# Option 1: Kinematic Chain Consistency Scorer
# ---------------------------------------------------------------------------
def compute_relative_rotation(
    rotmat_child: np.ndarray,
    rotmat_parent: np.ndarray,
) -> np.ndarray:
    """
    Compute relative rotation from parent to child.

    Args:
        rotmat_child: (3, 3) or (T, 3, 3)
        rotmat_parent: (3, 3) or (T, 3, 3)

    Returns:
        rotmat_rel: (3, 3) or (T, 3, 3)
    """
    if rotmat_child.ndim == 2:
        return rotmat_parent.T @ rotmat_child
    else:
        return np.einsum("...ij,...jk->...ik", rotmat_parent.transpose(0, 2, 1), rotmat_child)


def score_kinematic_chain(
    combo: List[int],
    X_samples: np.ndarray,
    calibration: np.ndarray,
    joint_angle_limits: Optional[dict],
) -> Tuple[float, dict]:
    """
    Score a candidate combo based on kinematic chain consistency.

    Args:
        combo: List of region_ids assigned to sensors
        X_samples: (N_sensors, 9, T) - IMU data for each sensor
        calibration: (24, 3, 3) - sensor-to-body calibration
        joint_angle_limits: dict with 'loose' and 'strict' limits

    Returns:
        score: float in [0, 1] (higher = more plausible)
        diagnostics: dict with detailed info
    """
    if joint_angle_limits is None:
        return 0.5, {"error": "No joint angle limits available"}

    n_sensors = len(combo)
    valid_pairs = 0
    total_score = 0.0
    pair_scores = {}

    # Find all valid parent-child pairs in the combo
    for i, region_i in enumerate(combo):
        parent_id = REGION_PARENTS.get(region_i, -1)
        if parent_id == -1:
            continue  # Root has no parent

        # Check if parent is also in the combo
        if parent_id not in combo:
            continue

        j = combo.index(parent_id)
        valid_pairs += 1

        # Get orientations
        r6d_i, _ = extract_imu_components(X_samples[i])
        r6d_j, _ = extract_imu_components(X_samples[j])

        # Convert to rotation matrices
        rot_i = r6d_to_rotmat(r6d_i)  # (T, 3, 3)
        rot_j = r6d_to_rotmat(r6d_j)  # (T, 3, 3)

        # Apply calibration to get body frame rotations
        cal_i = calibration[region_i]  # (3, 3)
        cal_j = calibration[parent_id]  # (3, 3)
        rot_body_i = rot_i @ cal_i  # (T, 3, 3)
        rot_body_j = rot_j @ cal_j  # (T, 3, 3)

        # Relative rotation per frame (R_parent^T @ R_child)
        rot_rel = compute_relative_rotation(rot_body_i, rot_body_j)
        if rot_rel.ndim == 2:
            rot_rel = rot_rel[None, :, :]  # (1, 3, 3)
        euler_rel = _rotmat_to_euler_zyx_batch(rot_rel)  # (T, 3)

        # Look up the precomputed envelope for this (parent, child) pair.
        loose = joint_angle_limits.get("loose", {}) if isinstance(joint_angle_limits, dict) else {}
        strict = joint_angle_limits.get("strict", {}) if isinstance(joint_angle_limits, dict) else {}
        key = (parent_id, region_i)
        if key in strict:
            lo = np.asarray(strict[key]["min"], dtype=np.float32)
            hi = np.asarray(strict[key]["max"], dtype=np.float32)
        elif key in loose:
            lo = np.asarray(loose[key]["min"], dtype=np.float32)
            hi = np.asarray(loose[key]["max"], dtype=np.float32)
        else:
            # No envelope known for this pair: fall back to a neutral 0.5 score
            valid_pairs -= 1  # don't credit this pair as evaluable
            continue

        # Distance outside the envelope, normalized by envelope half-width.
        midpoint = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # Avoid divide-by-zero on degenerate axes
        half = np.maximum(half, 1e-3)
        # Per-frame, per-axis: 0 inside, grows linearly outside
        outside = np.maximum(np.abs(euler_rel - midpoint) - half, 0.0) / half
        # Aggregate axes (sum) then exponentiate; mean over time.
        per_frame = np.exp(-outside.sum(axis=-1))  # (T,)
        pair_score = float(per_frame.mean())
        pair_scores[f"{parent_id}->{region_i}"] = pair_score
        total_score += pair_score

    if valid_pairs == 0:
        return 0.5, {"valid_pairs": 0, "message": "No valid kinematic pairs in combo"}

    avg_score = total_score / valid_pairs
    diagnostics = {
        "valid_pairs": valid_pairs,
        "pair_scores": pair_scores,
        "avg_score": float(avg_score),
    }
    return float(avg_score), diagnostics


# ---------------------------------------------------------------------------
# Option 2: Gravity Alignment Scorer
# ---------------------------------------------------------------------------
def detect_static_periods(
    r6d: np.ndarray,
    acc: np.ndarray,
    window: int = SMOOTH_WINDOW,
    ang_vel_thresh: float = STATIC_ANG_VEL_THRESH,
    acc_var_thresh: float = STATIC_ACC_VAR_THRESH,
) -> np.ndarray:
    """
    Detect static/quasi-static periods.

    Args:
        r6d: (T, 6) - 6D rotation
        acc: (T, 3) - acceleration
        window: smoothing window
        ang_vel_thresh: angular velocity threshold (rad/s)
        acc_var_thresh: acceleration variance threshold

    Returns:
        static_mask: (T,) bool - True for static periods
    """
    T = r6d.shape[0]
    static_mask = np.zeros(T, dtype=bool)

    # Compute angular velocity from r6d
    rotmat = r6d_to_rotmat(r6d)  # (T, 3, 3)
    if T < 2:
        return static_mask

    # Angular velocity: dR/dt * R^T
    dR = np.diff(rotmat, axis=0)  # (T-1, 3, 3)
    R_mid = rotmat[:-1]  # (T-1, 3, 3)
    ang_vel = np.cross(dR[:, :, 0], dR[:, :, 1])  # Simplified
    ang_vel_norm = np.linalg.norm(ang_vel, axis=1)  # (T-1,)

    # Pad angular velocity
    ang_vel_norm = np.concatenate([[0], ang_vel_norm])

    # Acceleration variance in sliding window
    for t in range(T):
        start = max(0, t - window // 2)
        end = min(T, t + window // 2 + 1)
        acc_window = acc[start:end]
        acc_var = np.var(acc_window, axis=0).sum()

        if ang_vel_norm[t] < ang_vel_thresh and acc_var < acc_var_thresh:
            static_mask[t] = True

    return static_mask


def score_gravity_alignment(
    combo: List[int],
    X_samples: np.ndarray,
    calibration: np.ndarray,
    gravity_distributions: Optional[dict],
) -> Tuple[float, dict]:
    """
    Score a candidate combo based on gravity alignment in static periods.

    Args:
        combo: List of region_ids
        X_samples: (N_sensors, 9, T)
        calibration: (24, 3, 3)
        gravity_distributions: dict with per-region gravity vectors

    Returns:
        score: float in [0, 1]
        diagnostics: dict
    """
    n_sensors = len(combo)
    total_score = 0.0
    sensor_scores = {}
    static_coverages = {}

    for i, region_id in enumerate(combo):
        r6d, acc = extract_imu_components(X_samples[i])
        T = r6d.shape[0]

        # Detect static periods
        static_mask = detect_static_periods(r6d, acc)
        static_coverage = static_mask.sum() / T if T > 0 else 0
        static_coverages[region_id] = float(static_coverage)

        if static_coverage < 0.05:  # Less than 5% static
            sensor_scores[region_id] = 0.5
            continue

        # Get gravity vector in static periods
        static_acc = acc[static_mask]
        gravity_sensor = np.mean(static_acc, axis=0)  # (3,) - gravity in sensor frame

        # Transform to body frame using calibration
        cal = calibration[region_id]  # (3, 3)
        gravity_body = cal.T @ gravity_sensor  # Transform to body frame

        # Expected gravity for this region (from training data)
        if gravity_distributions is not None and region_id in gravity_distributions:
            grav_expected = gravity_distributions[region_id]["mean"]  # (3,)
            grav_cov = gravity_distributions[region_id].get("cov", np.eye(3))

            # Mahalanobis distance
            diff = gravity_body - grav_expected
            mahal = diff @ np.linalg.inv(grav_cov + 1e-6 * np.eye(3)) @ diff
            score = 1.0 / (1.0 + mahal)
        else:
            # No training data: just check if gravity is pointing down
            # Assuming SMPL y-up coordinate system
            grav_magnitude = np.linalg.norm(gravity_body)
            if grav_magnitude < 1e-6:
                score = 0.5
            else:
                gravity_normalized = gravity_body / grav_magnitude
                # Should be [0, -1, 0] in SMPL (y-up, gravity down)
                expected = np.array([0, -1, 0], dtype=np.float32)
                cosine_sim = np.dot(gravity_normalized, expected)
                score = (cosine_sim + 1) / 2  # Map from [-1, 1] to [0, 1]

        sensor_scores[region_id] = float(score)

    total_score = np.mean(list(sensor_scores.values()))
    diagnostics = {
        "sensor_scores": sensor_scores,
        "static_coverages": static_coverages,
        "avg_score": float(total_score),
    }
    return float(total_score), diagnostics


# ---------------------------------------------------------------------------
# Option 3: Acceleration Profile Scorer
# ---------------------------------------------------------------------------
def score_acceleration_profile(
    combo: List[int],
    X_samples: np.ndarray,
    accel_distributions: Optional[dict],
    activity_id: Optional[int] = None,
) -> Tuple[float, dict]:
    """
    Score based on acceleration profile matching.

    Args:
        combo: List of region_ids
        X_samples: (N_sensors, 9, T)
        accel_distributions: dict with per-joint and per-activity stats
        activity_id: optional activity label

    Returns:
        score: float in [0, 1]
        diagnostics: dict
    """
    n_sensors = len(combo)
    total_score = 0.0
    sensor_scores = {}

    for i, region_id in enumerate(combo):
        _, acc = extract_imu_components(X_samples[i])

        # Compute acceleration statistics
        acc_mean = np.mean(acc, axis=0)  # (3,)
        acc_std = np.std(acc, axis=0)  # (3,)
        acc_peak = np.max(np.linalg.norm(acc, axis=1))  # scalar

        stats_vector = np.concatenate([acc_mean, acc_std, [acc_peak]])  # (7,)

        # Compare to expected distribution
        if accel_distributions is not None:
            # Try per-activity first, then per-joint
            key = None
            if activity_id is not None and "per_activity" in accel_distributions:
                key = (region_id, activity_id)
                if key in accel_distributions["per_activity"]:
                    dist = accel_distributions["per_activity"][key]
                else:
                    key = None

            if key is None and "per_joint" in accel_distributions:
                if region_id in accel_distributions["per_joint"]:
                    dist = accel_distributions["per_joint"][region_id]
                else:
                    sensor_scores[region_id] = 0.5
                    continue

            # Compute Mahalanobis distance
            mean = dist["mean"]  # (7,)
            cov = dist.get("cov", np.eye(7))  # (7, 7)

            diff = stats_vector - mean
            mahal = diff @ np.linalg.inv(cov + 1e-6 * np.eye(7)) @ diff
            score = 1.0 / (1.0 + mahal)
        else:
            # No training data: use heuristic
            # Lower acceleration = more plausible for most body parts
            acc_mag = np.linalg.norm(acc_mean)
            score = 1.0 / (1.0 + acc_mag)

        sensor_scores[region_id] = float(score)

    total_score = np.mean(list(sensor_scores.values())) if sensor_scores else 0.5
    diagnostics = {
        "sensor_scores": sensor_scores,
        "avg_score": float(total_score),
    }
    return float(total_score), diagnostics


# ---------------------------------------------------------------------------
# Option 4: Combined Multi-Signal Scorer
# ---------------------------------------------------------------------------
def combined_scorer(
    combo: List[int],
    X_samples: np.ndarray,
    calibration: np.ndarray,
    stats: dict,
    weights: Tuple[float, float, float] = (0.5, 0.0, 0.5),
    activity_id: Optional[int] = None,
) -> Tuple[float, dict]:
    """
    Combined weighted scorer.

    Default weights: (kinematic=0.5, gravity=0.0, accel=0.5).

    Gravity is weighted 0 by default because the vimu_joints acceleration
    channel in this dataset is *linear* acceleration (gravity removed), so the
    gravity scorer cannot recover gravity direction. The component is kept
    here so it can be re-enabled if total acceleration is later available
    (e.g. raw TotalCapture IMUs); see PHYSICS_VERIFICATION_PLAN.md §10.

    Args:
        combo: List of region_ids
        X_samples: (N_sensors, 9, T)
        calibration: (24, 3, 3)
        stats: dict of precomputed statistics
        weights: (w1, w2, w3) for kinematic, gravity, acceleration
        activity_id: optional activity label

    Returns:
        total_score: float in [0, 1]
        diagnostics: dict with all sub-scores
    """
    w1, w2, w3 = weights

    # Kinematic score
    kinematic_score, kin_diag = score_kinematic_chain(
        combo, X_samples, calibration, stats.get("joint_angle_limits")
    )

    # Gravity score
    gravity_score, grav_diag = score_gravity_alignment(
        combo, X_samples, calibration, stats.get("gravity_distributions")
    )

    # Acceleration score
    accel_score, acc_diag = score_acceleration_profile(
        combo, X_samples, stats.get("accel_distributions"), activity_id
    )

    # Weighted combination
    total = w1 * kinematic_score + w2 * gravity_score + w3 * accel_score

    diagnostics = {
        "kinematic_score": kinematic_score,
        "gravity_score": gravity_score,
        "accel_score": accel_score,
        "weights": {"w1": w1, "w2": w2, "w3": w3},
        "total_score": float(total),
        "kinematics": kin_diag,
        "gravity": grav_diag,
        "acceleration": acc_diag,
    }

    return float(total), diagnostics


# ---------------------------------------------------------------------------
# Main: CLI for testing
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Physics Verification Scorer")
    parser.add_argument("--data", required=True, help="Path to test .npz file")
    parser.add_argument("--calibration", default="calibration", help="Calibration directory")
    parser.add_argument("--stats", default="stats", help="Statistics directory")
    parser.add_argument("--combo", type=int, nargs="+", help="Combo to test (list of region IDs)")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.4, 0.3, 0.3],
                        help="Weights for kinematic, gravity, acceleration")

    args = parser.parse_args()

    print("=" * 60)
    print("Physics Verification Scorer")
    print("=" * 60)

    # Load data
    X, y, subject_ids, activity_ids = load_data(args.data)

    # Load calibration
    calibration = load_calibration(args.calibration)

    # Load stats
    stats = load_stats(args.stats)

    # Test a combo
    if args.combo:
        combo = args.combo
        combo_names = [REGION_NAMES[r] for r in combo]
        print(f"\nTesting combo: {combo_names}")

        # Get first sample for each sensor in combo
        n_features = X.shape[1]  # 6 or 9
        X_samples = np.zeros((len(combo), n_features, X.shape[2]), dtype=np.float32)
        for i, r in enumerate(combo):
            mask = y == r
            if mask.sum() > 0:
                X_samples[i] = X[mask][0]

        score, diag = combined_scorer(
            combo, X_samples, calibration, stats, weights=args.weights
        )

        print(f"\nCombined Score: {score:.4f}")
        print(f"  Kinematic:   {diag['kinematic_score']:.4f}")
        print(f"  Gravity:     {diag['gravity_score']:.4f}")
        print(f"  Acceleration: {diag['accel_score']:.4f}")

        print("\nDiagnostics:")
        for k, v in diag.items():
            if k not in ["kinematics", "gravity", "acceleration"]:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
