"""
preprocess_amass.py
===================
Extract 24-region labelled IMU windows from AMASS .npz files.

Usage
-----
# Full run (needs AMASS + SMPL):
  python preprocess_amass.py \
    --amass_root  C:/VS/TransPose/data/dataset_raw/AMASS \
    --smpl_model  C:/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
    --output      C:/VS/SensorLoc/data/dataset.npz \
    --window      120 \
    --stride      60 \
    --fps_out     60

# Smoke-test (no AMASS/SMPL needed, generates synthetic data):
  python preprocess_amass.py --smoke_test

# Estimate wall time before a long run (no output .npz):
  python preprocess_amass.py --estimate_only --max_seqs 200

Output dataset.npz keys
-----------------------
  X            (N, 6, 120)  float32  — accel xyz + gyro xyz, channels-first
  y            (N,)         int32    — region label 0-23
  subject_ids  (N,)         int32    — integer subject id (for LOSO splits)
"""

from __future__ import annotations
import argparse
import glob
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# ── project imports ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from smpl_regions import (
    REGION_NAMES, REGION_VERTEX_MAP, NUM_REGIONS,
    compute_centroids, REGION_CENTROIDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
GRAVITY      = np.array([0.0, -9.81, 0.0], dtype=np.float32)
SMPL_JOINTS  = 24          # SMPL has 24 kinematic joints
SMPL_VERTS   = 6890

# Datasets inside AMASS root to process (same list as TransPose config)
AMASS_SUBSETS = [
    'HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap',
    'SSM_synced', 'CMU', 'TotalCapture', 'Eyes_Japan_Dataset',
    'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
    'BMLhandball', 'MPI_Limits', 'DFaust_67',
]


# ---------------------------------------------------------------------------
# SMPL forward-kinematics (minimal, no blend shapes for speed)
# ---------------------------------------------------------------------------

# ── Chumpy-free loader ────────────────────────────────────────────────────
# The SMPL pkl only references ONE chumpy class: chumpy.ch.Ch
# We register a minimal stub in sys.modules before unpickling.
import types as _types
import sys as _sys


def _inject_chumpy_mock():
    if 'chumpy' in _sys.modules:
        return

    class Ch(np.ndarray):
        """
        Minimal stand-in for chumpy.Ch.
        Subclasses np.ndarray so all numpy operations work transparently.
        """
        def __new__(cls, *args, **kwargs):
            x = args[0] if args else []
            return np.asarray(x, dtype=np.float64).view(cls)

        def __array_finalize__(self, obj):
            pass

        def __setstate__(self, state):
            # ndarray's __setstate__ expects a tuple; chumpy passes a dict
            # where state['x'] holds the actual ndarray state tuple.
            if isinstance(state, tuple):
                super().__setstate__(state)
            elif isinstance(state, dict) and 'x' in state:
                x = state['x']
                if isinstance(x, tuple):
                    # Normal case: x is the ndarray (version, shape, dtype, isf, data) tuple
                    super().__setstate__(x)
                elif isinstance(x, np.ndarray):
                    # x is already a restored ndarray; copy its state in
                    super().__setstate__(x.__reduce__()[2])
            # Any other dict state: silently ignore (no array data to restore)


        @property
        def r(self):          # chumpy residual accessor
            return np.asarray(self)


    # Register both chumpy and chumpy.ch as proper modules
    chumpy_pkg    = _types.ModuleType('chumpy')
    chumpy_ch     = _types.ModuleType('chumpy.ch')
    chumpy_ch.Ch  = Ch
    chumpy_pkg.ch = chumpy_ch
    chumpy_pkg.Ch = Ch

    _sys.modules['chumpy']    = chumpy_pkg
    _sys.modules['chumpy.ch'] = chumpy_ch


_inject_chumpy_mock()


def load_smpl_model(pkl_path: str) -> dict:
    """Load the official SMPL .pkl without needing chumpy installed."""
    _inject_chumpy_mock()
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    # Flatten any chumpy.Ch arrays to plain ndarray
    for k, v in list(model.items()):
        if isinstance(v, np.ndarray) and type(v) is not np.ndarray:
            model[k] = np.array(v)   # strips the Ch subclass
    return model




def rodrigues(r: np.ndarray) -> np.ndarray:
    """Batch Rodrigues: (N, 3) -> (N, 3, 3) rotation matrices."""
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(r).as_matrix().astype(np.float32)


def smpl_forward(pose: np.ndarray, shape: np.ndarray, trans: np.ndarray,
                 smpl_model: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal SMPL forward pass.

    Parameters
    ----------
    pose  : (72,)  — axis-angle joint rotations (24 joints × 3)
    shape : (10,)  — shape coefficients
    trans : (3,)   — global root translation
    smpl_model : dict from load_smpl_model()

    Returns
    -------
    vertices   : (6890, 3) float32
    joint_rots : (24, 3, 3) float32  — global joint rotation matrices
    """
    kintree   = smpl_model['kintree_table']          # (2, 24)
    J_regressor = np.array(smpl_model['J_regressor'].todense(), dtype=np.float32)  # (24, 6890)
    v_template  = smpl_model['v_template'].astype(np.float32)   # (6890, 3)
    shapedirs   = smpl_model['shapedirs'].astype(np.float32)    # (6890, 3, 10)
    posedirs    = smpl_model['posedirs'].astype(np.float32)     # (6890*3, 207)
    weights     = smpl_model['weights'].astype(np.float32)      # (6890, 24)

    # 1. Shape blend shape  (SMPL shapedirs has 10 components; AMASS betas may be 16)
    n_shape  = shapedirs.shape[2]            # typically 10
    shape_10 = shape[:n_shape]               # safe trim
    v_shaped = v_template + np.einsum('ijk,k->ij', shapedirs, shape_10)  # (6890, 3)


    # 2. Rest-pose joints
    J = J_regressor @ v_shaped   # (24, 3)

    # 3. Pose rotations
    pose_mat = rodrigues(pose.reshape(-1, 3))  # (24, 3, 3)

    # 4. Pose blend shape (approximate; skip for speed when not needed)
    # pose_feature = (pose_mat[1:] - np.eye(3)).reshape(-1)
    # v_posed = v_shaped + (posedirs @ pose_feature).reshape(6890, 3)
    v_posed = v_shaped   # simplified (good enough for centroid computation)

    # 5. Global joint transforms (forward kinematics)
    parent = kintree[0].astype(int)
    parent[0] = -1
    G = np.zeros((24, 4, 4), dtype=np.float32)
    for j in range(24):
        R_j = pose_mat[j]
        t_j = J[j]
        G[j, :3, :3] = R_j
        G[j, :3, 3]  = t_j - R_j @ t_j
        if parent[j] >= 0:
            G[j] = G[parent[j]] @ G[j]
        G[j, :3, 3] += G[j, :3, :3] @ t_j - G[j, :3, :3] @ t_j  # avoid drift

    # Annoyingly need the proper global translation for each joint
    # Re-derive cleanly:
    G2 = np.zeros((24, 4, 4), dtype=np.float32)
    for j in range(24):
        R_j = pose_mat[j]
        t_j = J[j] if parent[j] < 0 else J[j] - J[parent[j]]
        local = np.eye(4, dtype=np.float32)
        local[:3, :3] = R_j
        local[:3,  3] = t_j
        if parent[j] < 0:
            G2[j] = local
        else:
            G2[j] = G2[parent[j]] @ local

    # Apply rest-pose inverse
    G_rest = G2.copy()
    for j in range(24):
        G_rest[j, :3, 3] -= G2[j, :3, :3] @ J[j]

    # 6. Linear blend skinning
    T = np.einsum('vj,jkl->vkl', weights, G_rest)  # (6890, 4, 4)
    v_hom = np.concatenate([v_posed, np.ones((6890, 1), dtype=np.float32)], axis=1)  # (6890, 4)
    vertices = np.einsum('vij,vj->vi', T[:, :3, :], v_hom)  # (6890, 3)

    # 7. Add root translation
    vertices += trans

    # Global rotation matrices per joint (3×3 part of G2)
    joint_rots = G2[:, :3, :3]   # (24, 3, 3)

    return vertices, joint_rots


# ---------------------------------------------------------------------------
# IMU simulation
# ---------------------------------------------------------------------------

def simulate_imu(vertices: np.ndarray, joint_rots: np.ndarray,
                 vertex_idx: int, region_joint: int,
                 dt: float, prev_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate accelerometer + gyroscope reading for one virtual sensor.

    Parameters
    ----------
    vertices      : (6890, 3) current frame vertices
    joint_rots    : (24, 3, 3) current frame global joint rotation matrices
    vertex_idx    : vertex the sensor is attached to
    region_joint  : index of the nearest SMPL joint for orientation reference
    dt            : time step (seconds)
    prev_vertices : (6890, 3) previous frame vertices

    Returns
    -------
    accel : (3,)  linear acceleration in sensor frame [m/s²]
    gyro  : (3,)  angular velocity in sensor frame [rad/s]
    """
    # ── Linear acceleration via finite differences ─────────────────────────
    pos_cur  = vertices[vertex_idx]
    pos_prev = prev_vertices[vertex_idx]
    # We only have one frame difference; velocity, not acceleration, but we
    # store the velocity difference (good proxy at 60 Hz).
    lin_accel_world = (pos_cur - pos_prev) / dt + GRAVITY  # gravity in world y-up

    # ── Angular velocity from rotation matrix log ──────────────────────────
    R_cur  = joint_rots[region_joint]
    # For angular velocity we need prev rotation; caller should pass it.
    # Here we use a simple approximation based on current rotation only
    # (the delta will be computed in the window-level loop).
    omega_world = np.zeros(3, dtype=np.float32)

    # ── Rotate to sensor frame ─────────────────────────────────────────────
    R_T = R_cur.T
    accel = R_T @ lin_accel_world
    gyro  = R_T @ omega_world

    return accel.astype(np.float32), gyro.astype(np.float32)


def compute_angular_velocity(R_prev: np.ndarray, R_cur: np.ndarray,
                              dt: float) -> np.ndarray:
    """
    Approximate angular velocity from two consecutive rotation matrices.
    Uses the rotation-matrix logarithm: ω ≈ log(R_prev^T R_cur) / dt
    """
    dR = R_prev.T @ R_cur
    # Log map via Rodrigues formula
    angle = np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        dR[2, 1] - dR[1, 2],
        dR[0, 2] - dR[2, 0],
        dR[1, 0] - dR[0, 1],
    ], dtype=np.float32) / (2.0 * np.sin(angle) + 1e-8)
    return (axis * angle / dt).astype(np.float32)


def random_rotation() -> np.ndarray:
    """Sample a uniformly distributed random 3-D rotation matrix."""
    return Rotation.random().as_matrix().astype(np.float32)


# Precomputed: for each region, the index of the nearest SMPL joint
# (used for orientation reference during IMU simulation).
REGION_TO_JOINT = {
    "pelvis":       0,
    "l_hip":        1,   "r_hip":        2,
    "spine_lower":  3,
    "l_thigh":      4,   "r_thigh":      5,
    "spine_mid":    6,
    "l_shin":       7,   "r_shin":       8,
    "spine_upper":  9,
    "l_foot":       10,  "r_foot":       11,
    "neck":         12,
    "l_collar":     13,  "r_collar":     14,
    "head":         15,
    "l_shoulder":   16,  "r_shoulder":   17,
    "l_upper_arm":  18,  "r_upper_arm":  19,
    "l_forearm":    20,  "r_forearm":    21,
    "l_hand":       22,  "r_hand":       23,
}


# ---------------------------------------------------------------------------
# Window extraction from a single AMASS sequence
# ---------------------------------------------------------------------------

def process_sequence(poses: np.ndarray, shapes: np.ndarray, trans: np.ndarray,
                     smpl_model: dict,
                     window: int, stride: int, fps: float,
                     n_sensors_per_region: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (X, y) windows from one AMASS sequence.

    Parameters
    ----------
    poses  : (T, 72)
    shapes : (16, 10)  — only first row used
    trans  : (T, 3)
    smpl_model : loaded SMPL dict
    window : window length in frames
    stride : stride in frames
    fps    : output frame rate

    Returns
    -------
    X : (N, 6, window)  — IMU windows
    y : (N,)            — region labels
    """
    T    = poses.shape[0]
    dt   = 1.0 / fps
    shape = shapes[0] if len(shapes.shape) > 1 else shapes  # (10,)

    # Choose one random vertex per region (reproducible within this call)
    rng = np.random.default_rng()
    sensor_vertices = {}
    for region_name, vidx_arr in REGION_VERTEX_MAP.items():
        sensor_vertices[region_name] = rng.choice(vidx_arr, size=n_sensors_per_region)

    # ── Forward kinematics for all frames ─────────────────────────────────
    log.debug("  Running SMPL FK for %d frames …", T)
    all_verts     = np.zeros((T, SMPL_VERTS, 3), dtype=np.float32)
    all_joint_rot = np.zeros((T, SMPL_JOINTS, 3, 3), dtype=np.float32)

    for t in range(T):
        v, jr = smpl_forward(poses[t], shape, trans[t], smpl_model)
        all_verts[t]     = v
        all_joint_rot[t] = jr

    # Update global centroids from first frame's T-pose (frame 0 with zero pose)
    try:
        v0, _ = smpl_forward(np.zeros(72, dtype=np.float32), shape,
                              np.zeros(3, dtype=np.float32), smpl_model)
        compute_centroids(v0)
    except Exception:
        pass  # fallback centroids remain

    # ── Build per-frame IMU signals for each of 24 regions ────────────────
    # accel_all[r, t, 3], gyro_all[r, t, 3]
    accel_all = np.zeros((NUM_REGIONS, T, 3), dtype=np.float32)
    gyro_all  = np.zeros((NUM_REGIONS, T, 3), dtype=np.float32)

    for r_idx, r_name in enumerate(REGION_NAMES):
        j_idx   = REGION_TO_JOINT[r_name]
        v_idx   = int(sensor_vertices[r_name][0])  # one sensor per region

        # Apply a fixed random rotation for orientation invariance
        R_rand = random_rotation()

        for t in range(T):
            # Linear acceleration via position finite difference
            if t == 0:
                pos_delta = np.zeros(3, np.float32)
            else:
                pos_delta = (all_verts[t, v_idx] - all_verts[t-1, v_idx]) / dt

            # Remove gravity component expressed in world frame
            accel_world = pos_delta - GRAVITY        # note: gravity subtracted

            # Angular velocity from rotation log-map
            if t == 0:
                omega_local = np.zeros(3, np.float32)
            else:
                R_prev = all_joint_rot[t-1, j_idx]
                R_cur  = all_joint_rot[t,   j_idx]
                omega_world = compute_angular_velocity(R_prev, R_cur, dt)
                omega_local = all_joint_rot[t, j_idx].T @ omega_world

            # Project acceleration to sensor (joint) frame
            accel_local = all_joint_rot[t, j_idx].T @ accel_world

            # Apply random global rotation (orientation invariance)
            accel_all[r_idx, t] = R_rand @ accel_local
            gyro_all[r_idx, t]  = R_rand @ (omega_local if t > 0 else np.zeros(3, np.float32))

    # ── Windowing ─────────────────────────────────────────────────────────
    X_list, y_list = [], []
    starts = list(range(0, T - window + 1, stride))
    for region_idx in range(NUM_REGIONS):
        for start in starts:
            end   = start + window
            a_win = accel_all[region_idx, start:end].T   # (3, window)
            g_win = gyro_all[region_idx,  start:end].T   # (3, window)
            imu   = np.concatenate([a_win, g_win], axis=0)  # (6, window)

            # Basic quality check: skip if all-zero (degenerate sequence)
            if np.all(np.abs(imu) < 1e-8):
                continue

            X_list.append(imu)
            y_list.append(region_idx)

    if len(X_list) == 0:
        return np.zeros((0, 6, window), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return (np.stack(X_list).astype(np.float32),
            np.array(y_list, dtype=np.int32))


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def collect_planned_npz_paths(amass_root: str, max_seqs_per_subdir: int
                              ) -> List[Tuple[str, str]]:
    """
    List (subset_name, npz_path) exactly as build_dataset would process.

    max_seqs_per_subdir caps how many files are taken from each AMASS subset
    folder (after sorting paths), matching --max_seqs.
    """
    planned: List[Tuple[str, str]] = []
    for subset in AMASS_SUBSETS:
        subset_dir = os.path.join(amass_root, subset)
        if not os.path.isdir(subset_dir):
            log.warning("Subset missing, skipping: %s", subset_dir)
            continue
        npz_files = glob.glob(os.path.join(subset_dir, '**', '*.npz'), recursive=True)
        npz_files = sorted(npz_files)[:max_seqs_per_subdir]
        for p in npz_files:
            planned.append((subset, p))
    return planned


def _process_one_amass_npz(
    npz_path: str,
    smpl_model: dict,
    window: int,
    stride: int,
    fps_out: float,
    *,
    log_failures: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load one AMASS .npz and return (X, y) windows, or None if skipped / failed.
    """
    try:
        data   = np.load(npz_path)
        poses  = data['poses'].astype(np.float32)
        shapes = data.get('betas', np.zeros((1, 10), np.float32)).astype(np.float32)
        trans  = data.get('trans', np.zeros((len(poses), 3), np.float32)).astype(np.float32)
        fps_in = float(data.get('mocap_frame_rate', 60))

        if poses.shape[1] > 72:
            poses = poses[:, :72]
        if len(poses) < window:
            return None

        if fps_in > fps_out:
            step = int(round(fps_in / fps_out))
            poses = poses[::step]
            trans = trans[::step]

        X, y = process_sequence(poses, shapes, trans, smpl_model,
                                window, stride, fps_out)
        if len(X) == 0:
            return None
        return X, y
    except Exception as e:
        if log_failures:
            log.warning("  ✗ %s — %s", os.path.basename(npz_path), e)
        return None


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h (~{seconds / 60:.0f} min)"


def estimate_preprocess_time(
    amass_root: str,
    smpl_pkl: str,
    window: int,
    stride: int,
    fps_out: int,
    max_seqs_per_subdir: int,
    sample_n: int = 3,
) -> None:
    """
    Print how many files will run, time a few sequences, and extrapolate wall time.
    Does not write dataset.npz.
    """
    planned = collect_planned_npz_paths(amass_root, max_seqs_per_subdir)
    n_total = len(planned)
    if n_total == 0:
        log.error("No .npz files found under %s with current settings.", amass_root)
        return

    per_subset: Dict[str, int] = {}
    for sub, _ in planned:
        per_subset[sub] = per_subset.get(sub, 0) + 1

    log.info("=== Preprocess time estimate (--estimate_only) ===")
    log.info("Planned sequences: %d  (max_seqs=%d per AMASS subset)",
             n_total, max_seqs_per_subdir)
    log.info("Per subset:")
    for sub in sorted(per_subset.keys()):
        log.info("  %-26s  %4d file(s)", sub, per_subset[sub])

    t0 = time.perf_counter()
    smpl_model = load_smpl_model(smpl_pkl)
    t_load_smpl = time.perf_counter() - t0
    log.info("SMPL load time: %s", _fmt_duration(t_load_smpl))

    sample_n = max(1, min(sample_n, n_total))
    bench_times: List[float] = []
    for _, npz_path in planned:
        if len(bench_times) >= sample_n:
            break
        t1 = time.perf_counter()
        out = _process_one_amass_npz(
            npz_path, smpl_model, window, stride, float(fps_out),
            log_failures=False,
        )
        elapsed = time.perf_counter() - t1
        if out is not None:
            bench_times.append(elapsed)
            log.info("  benchmark %s  →  %s  (%d windows)",
                     os.path.basename(npz_path), _fmt_duration(elapsed), len(out[0]))

    if not bench_times:
        log.error("Could not process any of the first files; cannot estimate time.")
        return

    mean_t = float(np.mean(bench_times))
    med_t  = float(np.median(bench_times))
    # Conservative: use max(mean, median); stretch upper bound for slow files
    per_file_est = max(mean_t, med_t)
    est_seq = per_file_est * n_total
    # Saving a large npz is usually small vs SMPL+processing; rough overhead
    est_save = 30.0 + 0.0001 * n_total * 50_000
    est_wall = t_load_smpl + est_seq + min(est_save, 300.0)

    log.info("Timed %d sequence(s):  mean=%.2fs  median=%.2fs per sequence",
             len(bench_times), mean_t, med_t)
    log.info("Extrapolated sequence processing: ~%s  (using ~%.2fs × %d files)",
             _fmt_duration(est_seq), per_file_est, n_total)
    log.info("Rough total wall time (SMPL + all sequences + small save fudge): ~%s",
             _fmt_duration(est_wall))
    log.info("(Longer motions cost more than short ones; error can be large.)")
    log.info("If too long, lower --max_seqs and re-run with --estimate_only.")


def build_dataset(amass_root: str, smpl_pkl: str, output_path: str,
                  window: int, stride: int, fps_out: int,
                  max_seqs_per_subdir: int = 5) -> None:
    """Walk AMASS sub-datasets and collect windows."""
    smpl_model = load_smpl_model(smpl_pkl)
    log.info("SMPL model loaded from %s", smpl_pkl)

    all_X, all_y, all_subjects = [], [], []
    subject_counter: Dict[str, int] = {}

    planned = collect_planned_npz_paths(amass_root, max_seqs_per_subdir)
    if not planned:
        log.error("No .npz files found — check AMASS path and --max_seqs.")
        sys.exit(1)
    log.info("Processing %d sequence files …", len(planned))

    for _, npz_path in planned:
        out = _process_one_amass_npz(
            npz_path, smpl_model, window, stride, float(fps_out),
            log_failures=True,
        )
        if out is None:
            continue
        X, y = out

        subj_key = os.path.basename(os.path.dirname(npz_path))
        if subj_key not in subject_counter:
            subject_counter[subj_key] = len(subject_counter)
        subj_id = subject_counter[subj_key]

        all_X.append(X)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subj_id, dtype=np.int32))
        log.info("  ✓ %s  |  %d windows", os.path.basename(npz_path), len(y))

    if len(all_X) == 0:
        log.error("No data collected — check AMASS path and SMPL model.")
        sys.exit(1)

    X_all  = np.concatenate(all_X)
    y_all  = np.concatenate(all_y)
    sub_all = np.concatenate(all_subjects)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, X=X_all, y=y_all, subject_ids=sub_all)

    log.info("\n=== Dataset saved: %s ===", output_path)
    log.info("  Total windows   : %d", len(y_all))
    log.info("  Shape (X)       : %s", X_all.shape)
    log.info("  Unique subjects : %d", len(subject_counter))
    log.info("  Class distribution:")
    for i, name in enumerate(REGION_NAMES):
        cnt = int((y_all == i).sum())
        log.info("    [%2d] %-16s : %6d", i, name, cnt)


# ---------------------------------------------------------------------------
# Smoke test — no AMASS/SMPL needed
# ---------------------------------------------------------------------------

def smoke_test(output_path: str, window: int = 120) -> None:
    """Generate a synthetic dataset with random IMU signals for unit testing."""
    log.info("=== Smoke test: generating synthetic data ===")
    rng   = np.random.default_rng(42)
    N_per = 20   # windows per region
    total = N_per * NUM_REGIONS

    X = rng.standard_normal((total, 6, window)).astype(np.float32)
    y = np.repeat(np.arange(NUM_REGIONS, dtype=np.int32), N_per)
    subject_ids = (np.arange(total) // (N_per * NUM_REGIONS // 4)).astype(np.int32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, subject_ids=subject_ids)
    log.info("Smoke dataset saved → %s   (%d windows, %d regions)", output_path, total, NUM_REGIONS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess AMASS → IMU windows")
    p.add_argument('--amass_root', default='C:/VS/TransPose/data/dataset_raw/AMASS')
    p.add_argument('--smpl_model', default='C:/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    p.add_argument('--output',     default='C:/VS/SensorLoc/data/dataset.npz')
    p.add_argument('--window',     type=int, default=120, help='Window length (frames)')
    p.add_argument('--stride',     type=int, default=60,  help='Stride (frames)')
    p.add_argument('--fps_out',    type=int, default=60,  help='Target output FPS')
    p.add_argument('--max_seqs',   type=int, default=5,   help='Max seqs per AMASS sub-dataset')
    p.add_argument('--smoke_test', action='store_true',   help='Generate synthetic data')
    p.add_argument('--estimate_only', action='store_true',
                   help='Count planned files, benchmark a few sequences, print ETA; exit')
    p.add_argument('--estimate_samples', type=int, default=3,
                   help='How many sequences to time for --estimate_only (default 3)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.smoke_test:
        smoke_out = args.output.replace('dataset.npz', 'smoke_dataset.npz')
        smoke_test(smoke_out, window=args.window)
    elif args.estimate_only:
        estimate_preprocess_time(
            amass_root=args.amass_root,
            smpl_pkl=args.smpl_model,
            window=args.window,
            stride=args.stride,
            fps_out=args.fps_out,
            max_seqs_per_subdir=args.max_seqs,
            sample_n=args.estimate_samples,
        )
    else:
        build_dataset(
            amass_root      = args.amass_root,
            smpl_pkl        = args.smpl_model,
            output_path     = args.output,
            window          = args.window,
            stride          = args.stride,
            fps_out         = args.fps_out,
            max_seqs_per_subdir = args.max_seqs,
        )
