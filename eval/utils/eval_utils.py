import math
import sys

import numpy as np
from evo.core.trajectory import PosePath3D, PoseTrajectory3D


def save_kitti_poses(poses, save_path):
    with open(save_path, "w") as f:
        for pose in poses:  # pose: 4x4 numpy array
            pose_line = pose[:3].reshape(-1)  # flatten first 3 rows
            f.write(" ".join(map(str, pose_line)) + "\n")


def save_tum_poses(poses, timestamps, save_path):
    """
    Save poses in TUM RGB-D format.
    Args:
        poses: list or array of 4x4 numpy arrays (T_w_c)
        timestamps: list or array of float timestamps (same length as poses)
        save_path: output file path
    """
    assert len(poses) == len(timestamps), "poses and timestamps length mismatch"

    with open(save_path, "w") as f:
        for ts, pose in zip(timestamps, poses):
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]

            R = pose[:3, :3]
            qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
            qx = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
            qy = np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
            qz = np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
            qx = math.copysign(qx, R[2, 1] - R[1, 2])
            qy = math.copysign(qy, R[0, 2] - R[2, 0])
            qz = math.copysign(qz, R[1, 0] - R[0, 1])

            f.write(
                f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )


def align_gt_pred(gt_views, poses_c2w_estimated):
    poses_c2w_gt = [view["camera_pose"][0] for view in gt_views]
    gt = PosePath3D(poses_se3=poses_c2w_gt)
    pred = PosePath3D(poses_se3=poses_c2w_estimated[0])
    r_a, t_a, s = pred.align(gt, correct_scale=True)

    return pred.poses_se3, gt.poses_se3


def align_gt_pred_2(poses_c2w_gt, poses_c2w_estimated):
    # poses_c2w_gt = [view["camera_pose"][0] for view in gt_views]
    gt = PosePath3D(poses_se3=poses_c2w_gt)
    pred = PosePath3D(poses_se3=poses_c2w_estimated)
    r_a, t_a, s = pred.align(gt, correct_scale=True)

    return pred.poses_se3, gt.poses_se3


def save_all_intrinsics_to_txt(result, filename="all_intrinsics.txt"):
    with open(filename, "w") as f:
        for i, res in enumerate(result):
            intrinsic = res["intrinsic"].squeeze(0).reshape(-1).cpu().numpy()  # (9,)
            line = "\t".join([f"{v:.6f}" for v in intrinsic])
            f.write(line + "\n")
    print(f"[TXT] Saved {len(result)} intrinsics to {filename}")


def uniform_sample(total: int, select: int) -> list:
    if select > total:
        raise ValueError("select cannot be greater than total")
    step = total / select
    return [int(i * step) for i in range(select)]
