
from pathlib import Path
import numpy as np
import csv
import pycolmap

from .. import logger
from ..structs.trajectory import Trajectory, associate_trajectories


def evaluate_wrt_pgt(
    est_traj: Trajectory,
    gt_traj: Trajectory,
    sim3d: pycolmap.Sim3d,
    output_path: Path,
) -> None:
    """
    Evaluate an estimated trajectory with respect to a pseudo 
    ground-truth trajectory that observes control points.
    Important:
        This function assumes that invert_poses=False for both trajectories.
    Args:
        est_traj (Trajectory): Estimated trajectory.
        gt_traj (Trajectory): Pseudo ground-truth trajectory.
        sim3d (pycolmap.Sim3d): Similarity transformation from the
            estimated trajectory to the control points obtained from 
            sparse evaluation.
        output_path (Path): Path to save the evaluation results.
    """
    
    est_traj.transform(sim3d)
    est_traj, gt_traj = associate_trajectories(est_traj, gt_traj)
    if est_traj is None or gt_traj is None:
        logger.error("Trajectory association failed.")
        return None
    
    E = est_traj.positions - gt_traj.positions
    error = np.array([np.linalg.norm(e[:2]) for e in E])

    error_file = output_path / "error_per_kf.txt"
    with open(error_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "error_m"])
        for ts, e in zip(est_traj.timestamps, error):
            writer.writerow([ts, e])

    logger.info("pGT evaluation completed successfully!")
    return error_file
    
