"""
rotation_utils.py
==================
Rotation representation conversion utilities for IMUCoCo-2.0.

Supports:
- 6D rotation (r6d) <-> 3x3 rotation matrix (Zhou et al. 2019)
- Diagnostic outputs for verification
"""

import numpy as np


def r6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al. 2019, "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        r6d: (6,) or (N, 6) array of 6D rotation vectors

    Returns:
        (3, 3) or (N, 3, 3) rotation matrix.
        For batched input, output is (N, 3, 3).
    """
    r6d = np.asarray(r6d, dtype=np.float32)
    original_shape = r6d.shape
    if r6d.ndim == 1:
        r6d = r6d[None, :]  # (1, 6)

    # First two columns of rotation matrix (Zhou et al. 2019, Gram-Schmidt)
    a = r6d[:, :3]  # (N, 3)
    b = r6d[:, 3:6]  # (N, 3)

    # First column: normalize a
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    # Second column: orthogonalize b against a_norm, then normalize
    proj = np.sum(b * a_norm, axis=1, keepdims=True) * a_norm
    b_orth = b - proj
    b_norm = b_orth / (np.linalg.norm(b_orth, axis=1, keepdims=True) + 1e-8)
    # Third column: cross product (right-handed)
    c_norm = np.cross(a_norm, b_norm)

    # Stack into rotation matrix: (N, 3, 3)
    rotmat = np.stack([a_norm, b_norm, c_norm], axis=2)

    # Only squeeze if original input was 1D (not batched)
    if len(original_shape) == 1:
        return rotmat[0]  # Return (3, 3)
    return rotmat  # Return (N, 3, 3)


def rotmat_to_r6d(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    Takes first two columns of the rotation matrix.

    Args:
        rotmat: (3, 3) or (N, 3, 3) rotation matrix

    Returns:
        (6,) or (N, 6) 6D rotation vector
    """
    rotmat = np.asarray(rotmat, dtype=np.float32)
    original_shape = rotmat.shape
    if rotmat.ndim == 2:
        rotmat = rotmat[None, :]  # (1, 3, 3)

    # Take first two columns
    r6d = np.concatenate([rotmat[:, :, 0], rotmat[:, :, 1]], axis=1)  # (N, 6)

    if len(original_shape) == 2:
        return r6d[0]  # Return (6,)
    return r6d  # Return (N, 6)


def verify_r6d_roundtrip(r6d: np.ndarray, verbose: bool = True) -> dict:
    """
    Verify r6d -> rotmat -> r6d roundtrip.
    Useful for diagnostics during development.

    Returns dict with 'input', 'rotmat', 'recovered_r6d', 'error', 'success'
    """
    r6d = np.asarray(r6d, dtype=np.float32)
    single = r6d.ndim == 1
    rotmat = r6d_to_rotmat(r6d)
    r6d_recovered = rotmat_to_r6d(rotmat)

    if single:
        err = np.linalg.norm(r6d - r6d_recovered)
    else:
        err = np.mean(np.linalg.norm(r6d - r6d_recovered, axis=1))

    result = {
        "input_shape": r6d.shape,
        "rotmat_shape": rotmat.shape,
        "recovered_shape": r6d_recovered.shape,
        "mean_error": float(err),
        "success": err < 1e-5,
    }

    if verbose:
        print(f"[r6d_verify] Input: {r6d.shape} -> Rotmat: {rotmat.shape} -> Recovered: {r6d_recovered.shape}")
        print(f"[r6d_verify] Mean roundtrip error: {err:.2e} (success={result['success']})")

    return result


def _random_rotation_matrix() -> np.ndarray:
    """Generate a random 3x3 rotation matrix."""
    # Generate random axis-angle
    axis = np.random.randn(3).astype(np.float32)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(0, 2 * np.pi)
    # Rodrigues formula
    K = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]], dtype=np.float32)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def test_r6d_conversion() -> dict:
    """
    Run basic tests on r6d conversion.
    Returns test results for diagnostics.
    """
    results = {}

    # Test 1: Identity rotation
    r6d_id = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rot_id = r6d_to_rotmat(r6d_id)
    results["identity"] = {
        "input": r6d_id,
        "rotmat": rot_id,
        "is_identity": np.allclose(rot_id, np.eye(3), atol=1e-6),
    }

    # Test 2: Random VALID rotations (generate from rotation matrices)
    np.random.seed(42)
    n_test = 100
    errors = []
    for _ in range(n_test):
        # Generate valid rotation matrix, convert to r6d
        R = _random_rotation_matrix()
        r6d_valid = rotmat_to_r6d(R)
        rt = verify_r6d_roundtrip(r6d_valid, verbose=False)
        errors.append(rt["mean_error"])

    results["random_batch"] = {
        "n_tests": n_test,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "all_pass": np.max(errors) < 1e-5,
    }

    # Test 3: Batched input with valid rotations
    r6d_batch = np.zeros((n_test, 6), dtype=np.float32)
    for i in range(n_test):
        R = _random_rotation_matrix()
        r6d_batch[i] = rotmat_to_r6d(R)
    rotmat_batch = r6d_to_rotmat(r6d_batch)
    results["batched"] = {
        "input_shape": r6d_batch.shape,
        "output_shape": rotmat_batch.shape,
        "correct_shape": rotmat_batch.shape == (n_test, 3, 3),
    }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Testing rotation_utils.py")
    print("=" * 60)
    results = test_r6d_conversion()
    print(f"\nIdentity test: {results['identity']['is_identity']}")
    print(f"Random batch: mean_error={results['random_batch']['mean_error']:.2e}, pass={results['random_batch']['all_pass']}")
    print(f"Batched test: {results['batched']['correct_shape']}")
    print("=" * 60)
