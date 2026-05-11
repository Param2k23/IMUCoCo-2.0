"""
extract_calibration.py
=====================
Extract sensor-to-body calibration matrices from the data generation pipeline.

For virtual IMUs (DIP-IMU style), the sensor frame IS the joint frame,
so calibration matrices are identity (24, 3, 3).

For real IMUs (TotalCapture), calibration is read from the dataset's
calibration files.

Output: calibration/region_sensor_rotmats.npy  (24, 3, 3) float32
"""

import os
import sys
import numpy as np

# Add data_gen to path so we can import if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_gen"))


def extract_virtual_imu_calibration() -> np.ndarray:
    """
    For virtual IMUs, sensor frame = joint frame.
    Calibration matrix is identity for all 24 regions.

    Returns:
        rotmats: (24, 3, 3) identity matrices
    """
    rotmats = np.zeros((24, 3, 3), dtype=np.float32)
    for i in range(24):
        rotmats[i] = np.eye(3, dtype=np.float32)
    return rotmats


def extract_totalcapture_calibration(
    calib_dir: str,
    subject_name: str,
    action_name: str,
) -> dict[int, np.ndarray]:
    """
    Extract calibration matrices from TotalCapture calibration files.

    Args:
        calib_dir: Path to TotalCapture/imu directory
        subject_name: e.g., 's1'
        action_name: e.g., 'acting1'

    Returns:
        dict mapping joint_index -> (3, 3) calibration matrix
    """
    import torch
    from articulate import math as art_math

    joint_names = ['Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm',
                   'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg',
                   'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot']
    n_extracted_imus = len(joint_names)

    name = subject_name + '_' + action_name.split('_')[0].lower()
    RSB = torch.zeros(n_extracted_imus, 3, 3)
    RIM = torch.zeros(n_extracted_imus, 3, 3)

    calib_bone_path = os.path.join(calib_dir, subject_name, name + '_calib_imu_bone.txt')
    if not os.path.exists(calib_bone_path):
        print(f"Warning: Calibration file not found: {calib_bone_path}")
        return {}

    with open(calib_bone_path, 'r') as f:
        n_sensors = int(f.readline())
        for _ in range(n_sensors):
            line = f.readline().split()
            if line[0] in joint_names:
                j = joint_names.index(line[0])
                q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])
                RSB[j] = art_math.quaternion_to_rotation_matrix(q)[0].t()

    calib_ref_path = os.path.join(calib_dir, subject_name, name + '_calib_imu_ref.txt')
    if not os.path.exists(calib_ref_path):
        print(f"Warning: Calibration file not found: {calib_ref_path}")
        return {}

    with open(calib_ref_path, 'r') as f:
        n_sensors = int(f.readline())
        for _ in range(n_sensors):
            line = f.readline().split()
            if line[0] in joint_names:
                j = joint_names.index(line[0])
                q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])
                RIM[j] = art_math.quaternion_to_rotation_matrix(q)[0].t()

    # Convert to numpy
    result = {}
    for j in range(n_extracted_imus):
        result[j] = RSB[j].numpy()  # Sensor-to-bone matrix

    return result


def save_calibration(rotmats: np.ndarray, output_path: str) -> None:
    """Save calibration matrices to .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, rotmats)
    print(f"Saved calibration matrices to {output_path}")
    print(f"  Shape: {rotmats.shape}")
    print(f"  Dtype: {rotmats.dtype}")
    # Verify: all should be identity for virtual IMUs
    for i in range(24):
        if not np.allclose(rotmats[i], np.eye(3), atol=1e-6):
            print(f"  Warning: Region {i} is not identity!")


def main():
    output_path = os.path.join(os.path.dirname(__file__), "calibration", "region_sensor_rotmats.npy")

    print("=" * 60)
    print("Extracting Virtual IMU Calibration (Identity Matrices)")
    print("=" * 60)

    rotmats = extract_virtual_imu_calibration()
    save_calibration(rotmats, output_path)

    # Diagnostic: print first few matrices
    print("\nDiagnostic: First 3 regions' calibration matrices:")
    for i in range(min(3, 24)):
        print(f"  Region {i}:")
        print(f"    {rotmats[i]}")

    print("\n" + "=" * 60)
    print("To extract TotalCapture real IMU calibration, use:")
    print("  python extract_calibration.py --source totalcapture --calib_dir <path>")
    print("=" * 60)


if __name__ == "__main__":
    main()
