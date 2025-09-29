import numpy as np

from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D

from .. import logger
from ..structs.trajectory import Trajectory


def valid_estimate(
    est_pose_traj: PoseTrajectory3D,
    gt_pose_traj: PoseTrajectory3D,
    min_duration_ratio: float = 0.5,
) -> bool:
    """" Check if the estimated trajectory is at least 
    half as long as the ground-truth trajectory in terms of
    timestamps overlap.
    Args:
        est_pose_traj (PoseTrajectory3D): Estimated trajectory.
        gt_pose_traj (PoseTrajectory3D): Ground-truth trajectory.
    Returns:
        bool: True if valid, False otherwise.
    """
    
    est_timestamps = est_pose_traj.timestamps
    gt_timestamps = gt_pose_traj.timestamps

    gt_length = (gt_timestamps[-1] - gt_timestamps[0]) / 1e9
    est_length = (est_timestamps[-1] - est_timestamps[0]) / 1e9

    if est_length < min_duration_ratio * gt_length:
        return False
    
    return True


def convert_trajectory_to_evo_posetrajectory(
    traj: Trajectory
) -> PoseTrajectory3D:
    """Convert a Trajectory to an evo PoseTrajectory3D.
    Args:
        traj (Trajectory): The trajectory to convert.
    Returns:
        PoseTrajectory3D: The converted evo trajectory.
    """
    assert traj.is_loaded(), "Trajectory is not loaded"
    if not traj.invert_poses:
        poses = traj.poses
    else:
        # store in COLMAP format (i.e., rig_from_world)
        poses = [pose.inverse() for pose in traj.poses]
    
    timestamps = traj.timestamps
    assert len(poses) == len(timestamps), "Poses and timestamps length mismatch"

    evo_traj = PoseTrajectory3D(
        positions_xyz=np.array([pose.translation for pose in poses]),
        orientations_xyzw=np.array([pose.rotation.quat for pose in poses]),
        timestamps=np.array(timestamps),
    )
    return evo_traj


def evaluate_wrt_mps(
    est_traj: Trajectory,
    gt_traj: Trajectory,
) -> float | None:
    """
    Evaluate an estimated trajectory with respect to 
    MPS pGT. Alignment performed using Umeyama via evo package.
    Important:
        This function assumes that invert_poses=False for both trajectories.
    Args:
        est_traj (Trajectory): Estimated trajectory.
        gt_traj (Trajectory): Pseudo ground-truth trajectory.
        sim3d (pycolmap.Sim3d): Similarity transformation from the
            estimated trajectory to the control points obtained from
            sparse evaluation.
        output_path (Path): Path to save the evaluation results.

    Estimate file format (space-separated columns):
        ```
        timestamp(ns) tx ty tz qx qy qz qw
        ```
    """

    est_pose_traj = convert_trajectory_to_evo_posetrajectory(est_traj)
    gt_pose_traj = convert_trajectory_to_evo_posetrajectory(gt_traj)

    if not valid_estimate(est_pose_traj, gt_pose_traj):
        logger.error("Estimated trajectory is too short compared to pGT.")
        return None
    
    gt_pose_traj_sync, est_pose_traj_sync = sync.associate_trajectories(
        gt_pose_traj, 
        est_pose_traj,
        max_diff=1e6 # 1 ms in ns
    )

    try:
        est_pose_traj_sync.align(
            gt_pose_traj_sync,
            correct_scale=True
        )
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return None
    
    pose_relation = metrics.PoseRelation.translation_part
    trajectories = (gt_pose_traj_sync, est_pose_traj_sync)
    ate_metric = metrics.APE(pose_relation)
    ate_metric.process_data(trajectories)

    ate_stats = ate_metric.get_all_statistics()

    return ate_stats["rmse"]
    

