import argparse
from pathlib import Path

from lamaria import logger
from lamaria.eval.evo_evaluation import evaluate_wrt_mps
from lamaria.structs.trajectory import Trajectory


def run(
    estimate: Path,
    gt_estimate: Path,
) -> bool:
    """Evaluate an estimated trajectory with respect to
    MPS pGT. There are no control points involved when
    evaluating using this function. Resorting to evo
    package for the evaluation.

    Args:
        estimate (Path): Path to the pose estimate file.
        gt_estimate (Path): Path to the pGT pose estimate file.
        output_path (Path): Path to save the evaluation results.
    Returns:
        bool: True if the evaluation was successful, False otherwise.
    """

    est_traj = Trajectory.load_from_file(estimate, invert_poses=False)
    if not est_traj.is_loaded():
        logger.error("Estimate could not be loaded")
        return False

    gt_traj = Trajectory.load_from_file(gt_estimate, invert_poses=False)
    if not gt_traj.is_loaded():
        logger.error("Ground-truth estimate could not be loaded")
        return False

    result = evaluate_wrt_mps(est_traj, gt_traj)

    if result is None:
        logger.error("Evaluation w.r.t MPS failed")
        return False

    logger.info(f"ATE RMSE: {result:.4f} m")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an estimated trajectory with respect to MPS pGT."
    )
    parser.add_argument(
        "--estimate",
        type=Path,
        required=True,
        help="Path to the pose estimate file.",
    )
    parser.add_argument(
        "--gt_estimate",
        type=Path,
        required=True,
        help="Path to the MPS pGT file.",
    )
    args = parser.parse_args()

    run(
        args.estimate,
        args.gt_estimate,
    )
