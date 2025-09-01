import logging
import math
import random
from collections import namedtuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def kabsch(pts1, pts2, estimate_scale=False):
    c_pts1 = pts1 - pts1.mean(axis=0)
    c_pts2 = pts2 - pts2.mean(axis=0)

    covariance = np.matmul(c_pts1.T, c_pts2) / c_pts1.shape[0]

    U, S, VT = np.linalg.svd(covariance)

    d = np.sign(np.linalg.det(np.matmul(VT.T, U.T)))
    correction = np.eye(3)
    correction[2, 2] = d

    if estimate_scale:
        pts_var = np.mean(np.linalg.norm(c_pts2, axis=1) ** 2)
        scale_factor = pts_var / np.trace(S * correction)
    else:
        scale_factor = 1.0

    R = scale_factor * np.matmul(np.matmul(VT.T, correction), U.T)
    t = pts2.mean(axis=0) - np.matmul(R, pts1.mean(axis=0))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, scale_factor


def get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r):
    # h_T aligns ground truth poses with estimates poses
    poses_gt_transformed = h_T @ poses_gt

    # calculate differences in position and rotations
    translations_delta = poses_gt_transformed[:, :3, 3] - poses_est[:, :3, 3]
    rotations_delta = poses_gt_transformed[:, :3, :3] @ poses_est[:, :3, :3].transpose(
        [0, 2, 1]
    )

    # translation inliers
    inliers_t = np.linalg.norm(translations_delta, axis=1) < inlier_threshold_t
    # rotation inliers
    inliers_r = Rotation.from_matrix(rotations_delta).magnitude() < (
        inlier_threshold_r / 180 * math.pi
    )
    # intersection of both
    return np.logical_and(inliers_r, inliers_t)


def print_hyp(hypothesis, hyp_name):
    h_translation = np.linalg.norm(hypothesis["transformation"][:3, 3])
    h_angle = (
        np.linalg.norm(
            Rotation.from_matrix(hypothesis["transformation"][:3, :3]).as_rotvec()
        )
        * 180
        / math.pi
    )
    print(
        f"{hyp_name}: score={hypothesis['score']}, translation={h_translation:.2f}m, "
        f"rotation={h_angle:.1f}deg."
    )


def estimated_alignment(
    pose_est,
    pose_gt,
    inlier_threshold_t=0.05,
    inlier_threshold_r=5,
    ransac_iterations=1000,
    refinement_max_hyp=12,
    refinement_max_it=8,
    estimate_scale=False,
):
    n_pose = len(pose_est)
    ransac_hypotheses = []
    for i in range(ransac_iterations):
        min_sample_size = 3
        samples = random.sample(range(n_pose), min_sample_size)
        h_pts1 = pose_gt[samples, :3, 3]
        h_pts2 = pose_est[samples, :3, 3]

        h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale=estimate_scale)

        inliers = get_inliers(
            h_T, pose_gt, pose_est, inlier_threshold_t, inlier_threshold_r
        )

        if inliers[samples].sum() >= 3:
            # only keep hypotheses if minimal sample is all inliers
            ransac_hypotheses.append(
                {
                    "transformation": h_T,
                    "inliers": inliers,
                    "score": inliers.sum(),
                    "scale": h_scale,
                }
            )
    if len(ransac_hypotheses) == 0:
        print(
            f"Did not fine a single valid RANSAC hypothesis, abort alignment estimation."
        )
        return None, 1

    # sort according to score
    ransac_hypotheses = sorted(
        ransac_hypotheses, key=lambda x: x["score"], reverse=True
    )

    # for hyp_idx, hyp in enumerate(ransac_hypotheses):
    #     print_hyp(hyp, f"Hypothesis {hyp_idx}")

    # create shortlist of best hypotheses for refinement
    # print(f"Starting refinement of {refinement_max_hyp} best hypotheses.")
    ransac_hypotheses = ransac_hypotheses[:refinement_max_hyp]

    # refine all hypotheses in the short list
    for ref_hyp in ransac_hypotheses:
        # print_hyp(ref_hyp, "Pre-Refinement")

        # refinement loop
        for ref_it in range(refinement_max_it):
            # re-solve alignment on all inliers
            h_pts1 = pose_gt[ref_hyp["inliers"], :3, 3]
            h_pts2 = pose_est[ref_hyp["inliers"], :3, 3]

            h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

            # calculate new inliers
            inliers = get_inliers(
                h_T, pose_gt, pose_est, inlier_threshold_t, inlier_threshold_r
            )

            # check whether hypothesis score improved
            refined_score = inliers.sum()

            if refined_score > ref_hyp["score"]:
                ref_hyp["transformation"] = h_T
                ref_hyp["inliers"] = inliers
                ref_hyp["score"] = refined_score
                ref_hyp["scale"] = h_scale

                # print_hyp(ref_hyp, f"Refinement interation {ref_it}")

            else:
                # print(f"Stopping refinement. Score did not improve: New score={refined_score}, "
                #              f"Old score={ref_hyp['score']}")
                break

    # re-sort refined hyotheses
    ransac_hypotheses = sorted(
        ransac_hypotheses, key=lambda x: x["score"], reverse=True
    )

    # for hyp_idx, hyp in enumerate(ransac_hypotheses):
    # print_hyp(hyp, f"Hypothesis {hyp_idx}")

    return ransac_hypotheses[0]["transformation"], ransac_hypotheses[0]["scale"]


def eval_pose_ransac(gt, est, t_thres=0.05, r_thres=5, aligned=True, save_dir=None):
    if aligned:
        alignment_transformation, alignment_scale = estimated_alignment(
            est,
            gt,
            inlier_threshold_t=0.05,
            inlier_threshold_r=5,
            ransac_iterations=1000,
            refinement_max_hyp=12,
            refinement_max_it=8,
            estimate_scale=True,
        )
        if alignment_transformation is None:
            _logger.info(
                f"Alignment requested but failed. Setting all pose errors to {math.inf}."
            )
    else:
        alignment_transformation = np.eye(4)
        alignment_scale = 1.0
    # Evaluation Loop

    rErrs = []
    tErrs = []
    accuracy = 0
    r_acc_5 = 0
    r_acc_2 = 0
    r_acc_1 = 0
    t_acc_15 = 0
    t_acc_10 = 0
    t_acc_5 = 0
    t_acc_2 = 0
    t_acc_1 = 0
    acc_10 = 0
    acc_5 = 0
    acc_2 = 0
    acc_1 = 0

    for pred_pose, gt_pose in zip(est, gt):
        if alignment_transformation is not None:
            # Apply alignment transformation to GT pose
            gt_pose = alignment_transformation @ gt_pose

            # Calculate translation error.
            t_err = float(np.linalg.norm(gt_pose[0:3, 3] - pred_pose[0:3, 3]))

            # Correct translation scale with the inverse alignment scale (since we align GT with estimates)
            t_err = t_err / alignment_scale

            # Rotation error.
            gt_R = gt_pose[0:3, 0:3]
            out_R = pred_pose[0:3, 0:3]

            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi
        else:
            pose_gt = None
            t_err, r_err = math.inf, math.inf

        # _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

        # Save the errors.
        rErrs.append(r_err)
        tErrs.append(t_err * 100)  # in cm

        # Check various thresholds.
        if r_err < r_thres and t_err < t_thres:
            accuracy += 1
        if r_err < 5:
            r_acc_5 += 1
        if r_err < 2:
            r_acc_2 += 1
        if r_err < 1:
            r_acc_1 += 1
        if t_err < 0.15:
            t_acc_15 += 1
        if t_err < 0.10:
            t_acc_10 += 1
        if t_err < 0.05:
            t_acc_5 += 1
        if t_err < 0.02:
            t_acc_2 += 1
        if t_err < 0.01:
            t_acc_1 += 1
        if r_err < 10 and t_err < 0.10:
            acc_10 += 1
        if r_err < 5 and t_err < 0.05:
            acc_5 += 1
        if r_err < 2 and t_err < 0.02:
            acc_2 += 1
        if r_err < 1 and t_err < 0.01:
            acc_1 += 1

    total_frames = len(rErrs)
    assert total_frames == len(est)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute final precision.
    accuracy = accuracy / total_frames * 100
    r_acc_5 = r_acc_5 / total_frames * 100
    r_acc_2 = r_acc_2 / total_frames * 100
    r_acc_1 = r_acc_1 / total_frames * 100
    t_acc_15 = t_acc_15 / total_frames * 100
    t_acc_10 = t_acc_10 / total_frames * 100
    t_acc_5 = t_acc_5 / total_frames * 100
    t_acc_2 = t_acc_2 / total_frames * 100
    t_acc_1 = t_acc_1 / total_frames * 100
    acc_10 = acc_10 / total_frames * 100
    acc_5 = acc_5 / total_frames * 100
    acc_2 = acc_2 / total_frames * 100
    acc_1 = acc_1 / total_frames * 100

    # _logger.info("===================================================")
    # _logger.info("Test complete.")

    # _logger.info(f'Accuracy: {accuracy:.1f}%')
    # _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    # print("===================================================")
    # print("Test complete.")

    with open(save_dir, "w") as f:
        f.write(f"Accuracy: {accuracy:.1f}%\n\n")
        f.write(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm\n")
        f.write(f"R acc 5: {r_acc_5:.1f}%\n")
        f.write(f"R acc 2: {r_acc_2:.1f}%\n")
        f.write(f"R acc 1: {r_acc_1:.1f}%\n")
        f.write(f"T acc 15: {t_acc_15:.1f}%\n")
        f.write(f"T acc 10: {t_acc_10:.1f}%\n")
        f.write(f"T acc 5: {t_acc_5:.1f}%\n")
        f.write(f"T acc 2: {t_acc_2:.1f}%\n")
        f.write(f"T acc 1: {t_acc_1:.1f}%\n")
        f.write(f"Acc 10: {acc_10:.1f}%\n")
        f.write(f"Acc 5: {acc_5:.1f}%\n")
        f.write(f"Acc 2: {acc_2:.1f}%\n")
        f.write(f"Acc 1: {acc_1:.1f}%\n")
