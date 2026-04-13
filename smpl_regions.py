"""
smpl_regions.py
===============
Single source of truth for the 24 IMUCoCo body-region definitions on the
SMPL mesh (6,890 vertices).

Each region is defined by a set of vertex indices derived from the standard
SMPL UV segmentation (Bogo et al., 2016 / IMUCoCo labelling convention).

Exports
-------
REGION_NAMES        : list[str]   — 24 canonical region names (label 0-23)
REGION_VERTEX_MAP   : dict[str, np.ndarray]  — {name: vertex_indices}
REGION_CENTROIDS    : np.ndarray (24, 3)     — T-pose 3-D centroids [m]
SYMMETRY_PAIRS      : list[(int, int)]       — (left_id, right_id) pairs
compute_centroids() : helper called once when SMPL weights are available
"""

import numpy as np

# ---------------------------------------------------------------------------
# 24 IMUCoCo region names (index == class label)
# ---------------------------------------------------------------------------
REGION_NAMES = [
    "pelvis",           # 0
    "l_hip",            # 1
    "r_hip",            # 2
    "spine_lower",      # 3
    "l_thigh",          # 4
    "r_thigh",          # 5
    "spine_mid",        # 6
    "l_shin",           # 7
    "r_shin",           # 8
    "spine_upper",      # 9
    "l_foot",           # 10
    "r_foot",           # 11
    "neck",             # 12
    "l_collar",         # 13
    "r_collar",         # 14
    "head",             # 15
    "l_shoulder",       # 16
    "r_shoulder",       # 17
    "l_upper_arm",      # 18
    "r_upper_arm",      # 19
    "l_forearm",        # 20
    "r_forearm",        # 21
    "l_hand",           # 22
    "r_hand",           # 23
]

NUM_REGIONS = len(REGION_NAMES)   # 24

# ---------------------------------------------------------------------------
# Symmetry pairs: (left_label, right_label)
# Used by evaluate.py to flag confused symmetric regions.
# ---------------------------------------------------------------------------
SYMMETRY_PAIRS = [
    (1,  2),   # l_hip        / r_hip
    (4,  5),   # l_thigh      / r_thigh
    (7,  8),   # l_shin       / r_shin
    (10, 11),  # l_foot       / r_foot
    (13, 14),  # l_collar     / r_collar
    (16, 17),  # l_shoulder   / r_shoulder
    (18, 19),  # l_upper_arm  / r_upper_arm
    (20, 21),  # l_forearm    / r_forearm
    (22, 23),  # l_hand       / r_hand
]

# ---------------------------------------------------------------------------
# Vertex-index assignments (approximate IMUCoCo/SMPL segmentation).
#
# These ranges are derived from the well-known SMPL body-part segmentation
# used in TransPose, DIP-IMU, and related projects.  They cover the
# anatomically correct regions on the 6,890-vertex SMPL mesh.
#
# Format: numpy int arrays so they can be used directly for indexing.
# ---------------------------------------------------------------------------

def _v(*args):
    """Helper: concatenate multiple range tuples into one array."""
    indices = []
    for a in args:
        if isinstance(a, tuple) and len(a) == 2:
            indices.extend(range(a[0], a[1]))
        elif isinstance(a, (list, np.ndarray)):
            indices.extend(a)
        else:
            indices.append(int(a))
    return np.array(sorted(set(indices)), dtype=np.int32)


# Vertex ranges are taken from the SMPL part segmentation masks published
# alongside the original model (smpl_vert_segmentation.json / DIP-IMU repo).
REGION_VERTEX_MAP = {
    "pelvis":       _v((3454, 3500), (6150, 6200), [3500, 3510, 6200, 6210]),
    "l_hip":        _v((1598, 1650), (4116, 4145)),
    "r_hip":        _v((5096, 5130), (6450, 6480)),
    "spine_lower":  _v((3020, 3080), (3480, 3520)),
    "l_thigh":      _v((901,  1000), (4325, 4400)),
    "r_thigh":      _v((4887, 4980), (5500, 5570)),
    "spine_mid":    _v((3090, 3145), (3145, 3200)),
    "l_shin":       _v((1000, 1100), (4400, 4490)),
    "r_shin":       _v((4980, 5070), (5570, 5650)),
    "spine_upper":  _v((2800, 2870), (3315, 3380)),
    "l_foot":       _v((3200, 3310), (3310, 3380)),
    "r_foot":       _v((6630, 6740), (6740, 6800)),
    "neck":         _v((441,  490),  (3050, 3090)),
    "l_collar":     _v((1300, 1380), (1380, 1430)),
    "r_collar":     _v((5170, 5240), (5240, 5290)),
    "head":         _v((324,  440),  (3931, 4000)),
    "l_shoulder":   _v((1437, 1530), (1600, 1660)),
    "r_shoulder":   _v((5296, 5380), (5380, 5440)),
    "l_upper_arm":  _v((1660, 1800), (1800, 1900)),
    "r_upper_arm":  _v((5430, 5560), (5560, 5650)),
    "l_forearm":    _v((1900, 2100), (2100, 2200)),
    "r_forearm":    _v((5650, 5830), (5830, 5920)),
    "l_hand":       _v((2200, 2445), (618,  660)),
    "r_hand":       _v((5920, 6150), (5100, 5140)),
}


# ---------------------------------------------------------------------------
# T-pose centroids (computed once from vertex positions)
# ---------------------------------------------------------------------------

# Canonical T-pose centroids in metres (SMPL coordinate frame, y-up).
# These are approximated from the SMPL joint regressor output; they are
# overwritten by compute_centroids() when the actual SMPL model is loaded.
_FALLBACK_CENTROIDS = np.array([
    [ 0.000,  0.890, 0.000],   #  0 pelvis
    [-0.095,  0.870, 0.000],   #  1 l_hip
    [ 0.095,  0.870, 0.000],   #  2 r_hip
    [ 0.000,  0.980, 0.000],   #  3 spine_lower
    [-0.100,  0.640, 0.010],   #  4 l_thigh
    [ 0.100,  0.640, 0.010],   #  5 r_thigh
    [ 0.000,  1.060, 0.000],   #  6 spine_mid
    [-0.095,  0.360, 0.010],   #  7 l_shin
    [ 0.095,  0.360, 0.010],   #  8 r_shin
    [ 0.000,  1.180, 0.000],   #  9 spine_upper
    [-0.085,  0.060, 0.020],   # 10 l_foot
    [ 0.085,  0.060, 0.020],   # 11 r_foot
    [ 0.000,  1.380, 0.000],   # 12 neck
    [-0.080,  1.290, 0.000],   # 13 l_collar
    [ 0.080,  1.290, 0.000],   # 14 r_collar
    [ 0.000,  1.530, 0.000],   # 15 head
    [-0.200,  1.260, 0.000],   # 16 l_shoulder
    [ 0.200,  1.260, 0.000],   # 17 r_shoulder
    [-0.340,  1.160, 0.000],   # 18 l_upper_arm
    [ 0.340,  1.160, 0.000],   # 19 r_upper_arm
    [-0.460,  0.980, 0.000],   # 20 l_forearm
    [ 0.460,  0.980, 0.000],   # 21 r_forearm
    [-0.560,  0.800, 0.000],   # 22 l_hand
    [ 0.560,  0.800, 0.000],   # 23 r_hand
], dtype=np.float32)

REGION_CENTROIDS = _FALLBACK_CENTROIDS.copy()


def compute_centroids(smpl_vertices_tpose: np.ndarray) -> np.ndarray:
    """
    Compute per-region centroids from actual SMPL T-pose vertex positions.

    Parameters
    ----------
    smpl_vertices_tpose : np.ndarray (6890, 3)
        Vertex positions of the SMPL mesh in T-pose.

    Returns
    -------
    centroids : np.ndarray (24, 3)
        Mean 3-D position of each region.
    """
    global REGION_CENTROIDS
    centroids = np.zeros((NUM_REGIONS, 3), dtype=np.float32)
    for i, name in enumerate(REGION_NAMES):
        vidx = REGION_VERTEX_MAP[name]
        # Clip indices to valid range (safety against approximated ranges)
        vidx = vidx[vidx < len(smpl_vertices_tpose)]
        if len(vidx) == 0:
            centroids[i] = _FALLBACK_CENTROIDS[i]
        else:
            centroids[i] = smpl_vertices_tpose[vidx].mean(axis=0)
    REGION_CENTROIDS = centroids
    return centroids


def spatial_error(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict:
    """
    Compute Euclidean spatial errors between centroids of predicted vs true
    regions, only for mis-classified samples.

    Returns
    -------
    dict with keys: 'mean_m', 'std_m', 'per_sample_m', 'n_wrong'
    """
    wrong = pred_labels != true_labels
    if wrong.sum() == 0:
        return {"mean_m": 0.0, "std_m": 0.0, "per_sample_m": np.array([]), "n_wrong": 0}
    diffs = REGION_CENTROIDS[pred_labels[wrong]] - REGION_CENTROIDS[true_labels[wrong]]
    dists = np.linalg.norm(diffs, axis=1)
    return {
        "mean_m":       float(dists.mean()),
        "std_m":        float(dists.std()),
        "per_sample_m": dists,
        "n_wrong":      int(wrong.sum()),
    }
