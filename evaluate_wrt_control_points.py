import argparse
from pathlib import Path

from lamaria import logger
from lamaria.eval.sparse_evaluation import evaluate_wrt_control_points
from lamaria.structs.control_point import (
    load_cp_json,
    run_control_point_triangulation,
)
from lamaria.structs.trajectory import Trajectory
from lamaria.utils.aria import initialize_reconstruction_from_calibration_file


def run(
    estimate: Path,
    cp_json_file: Path,
    device_calibration_json: Path,
    output_path: Path,
    corresponding_sensor: str = "imu",
) -> bool:
    """Run sparse evaluation for sequences that observe control points.

    Args:
        estimate (Path): Path to the pose estimate file.
        cp_json_file (Path): Path to the sparse ground-truth JSON file
            containing control point data.
        device_calibration_json (Path): Path to the device calibration JSON.
        output_path (Path): Directory where intermediate data and
        evaluation results will be saved.
        corresponding_sensor (str): The reference sensor to use
        ("imu" or "cam0").

    Returns:
        bool: True if the evaluation was successful, False otherwise.

    Notes:
        Expected estimate file format (space-separated columns):
        ```
        timestamp tx ty tz qx qy qz qw
        ```
    """
    traj = Trajectory.load_from_file(
        estimate, invert_poses=False, corresponding_sensor=corresponding_sensor
    )
    if not traj.is_loaded():
        logger.error("Estimate could not be loaded")
        return False

    init_reconstruction = initialize_reconstruction_from_calibration_file(
        device_calibration_json
    )
    control_points, timestamp_to_images = load_cp_json(cp_json_file)

    reconstruction = traj.add_estimate_poses_to_reconstruction(
        init_reconstruction, timestamp_to_images
    )

    run_control_point_triangulation(
        reconstruction,
        control_points,
    )

    result = evaluate_wrt_control_points(
        reconstruction,
        control_points,
    )

    result_path = output_path / "sparse_eval_result.npy"
    if result is None:
        logger.error("Sparse evaluation failed.")
        return False

    result.save_as_npy(result_path)

    # TODO: Add metrics here?


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline sparse evaluation"
    )
    parser.add_argument(
        "--estimate",
        type=Path,
        required=True,
        help="Path to the pose estimate file produced by a specific method",
    )
    parser.add_argument(
        "--cp_json_file",
        type=Path,
        required=True,
        help="Path to the sparse GT JSON file",
    )
    parser.add_argument(
        "--device_calibration_json",
        type=Path,
        required=True,
        help="Path to the Aria device calibration JSON file, "
        "found on the Lamaria website",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to the output folder where intermediate data "
        "and evaluation results will be saved",
    )
    parser.add_argument(
        "--corresponding_sensor",
        type=str,
        default="imu",
        choices=["imu", "cam0"],
        help="The sensor in which the poses are expressed.",
    )
    args = parser.parse_args()

    _ = run(
        args.estimate,
        args.cp_json_file,
        args.device_calibration_json,
        args.output_path,
        args.corresponding_sensor,
    )
