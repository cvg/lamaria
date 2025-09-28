import argparse
import copy
import os
from pathlib import Path

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

from .. import logger
from ..structs.estimate import Estimate
from ..utils.control_point import (
    construct_control_points_from_json,
    get_cps_for_initial_alignment,
    run_control_point_triangulation_from_json,
)
from ..utils.aria import get_t_imu_camera_from_json, initialize_reconstruction_from_calibration_file


def update_sim3d_scale(variables: dict) -> None:
    """Update the scale of the sim3d in the variables dictionary
    from the log_scale variable. Occurs in place."""
    if "log_scale" not in variables:
        raise ValueError("log_scale not found in variables")

    log_scale = copy.deepcopy(variables["log_scale"])
    variables["sim3d"].scale = np.exp(log_scale)


def create_variables_for_sparse_evaluation(
    control_points: dict, sim3d: pycolmap.Sim3d, cp_reproj_std: float = 1.0
) -> dict:
    """Create variables dictionary for sparse evaluation.

    Variables consists of -
    - ```control_points```: Control points dictionary
    - ```sim3d```: pycolmap.sim3d transformation
    - ```cp_reproj_std```: Control point reprojection
    - ```log scale``` (np.ndarray): Logarithm of the scale factor
        (from sim3d.scale)
    """

    variables = {}
    variables["control_points"] = copy.deepcopy(control_points)
    variables["sim3d"] = copy.deepcopy(sim3d)
    variables["cp_reproj_std"] = cp_reproj_std
    scale = copy.deepcopy(variables["sim3d"].scale)
    variables["log_scale"] = np.array(np.log(scale), dtype=np.float64)

    return variables


def get_problem_for_sparse_alignment(
    reconstruction: pycolmap.Reconstruction,
    variables: dict,
) -> tuple:
    """Create a Ceres problem for sparse alignment.
    Args:
        reconstruction (pycolmap.Reconstruction): The COLMAP reconstruction.
        variables (dict): The variables dictionary from
            create_variables_for_sparse_evaluation.

    Returns:
        problem (pyceres.Problem): The Ceres problem.
        solver_options (pyceres.SolverOptions): The solver options.
        summary (pyceres.SolverSummary): The solver summary.
    """
    problem = pyceres.Problem()
    problem = add_alignment_residuals(
        problem,
        reconstruction,
        variables,
    )
    solver_options = pyceres.SolverOptions()
    solver_options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()

    return problem, solver_options, summary


def add_alignment_residuals(
    problem,
    reconstruction: pycolmap.Reconstruction,
    variables: dict,
) -> pyceres.Problem:
    """Add alignment residuals to the Ceres problem.

    Variables consists of -
    - ReprojErrorCost for each observation of each control point
    - Point3DAlignmentCost for each control point
    Args:
        problem (pyceres.Problem): The Ceres problem.
        reconstruction (pycolmap.Reconstruction): The COLMAP reconstruction.
        variables (dict): The variables dictionary from
            create_variables_for_sparse_evaluation.
    """
    if (
        variables["control_points"] is not None
        and variables["sim3d"] is not None
    ):
        loss = pyceres.TrivialLoss()
        for tag_id, cp in variables["control_points"].items():
            if cp["triangulated"] is None:
                logger.info(f"Control point {tag_id} not triangulated")
                continue
            for image_id_and_point2d in cp["image_id_and_point2d"]:
                image_id, point2d = image_id_and_point2d
                image = reconstruction.images[image_id]
                pose = image.cam_from_world()
                camera = reconstruction.cameras[image.camera_id]

                point2d = np.asarray(point2d, dtype=np.float64).reshape(2, 1)
                point2d_cov = np.eye(2) * pow(variables["cp_reproj_std"], 2)
                cost = pycolmap.cost_functions.ReprojErrorCost(
                    camera.model,
                    point2d_cov,
                    point2d,
                    pose,
                )
                problem.add_residual_block(
                    cost, loss, [cp["triangulated"], camera.params]
                )
                problem.set_parameter_block_constant(camera.params)

        for tag_id, cp in variables["control_points"].items():
            if cp["triangulated"] is None:
                logger.info(f"Control point {tag_id} not triangulated")
                continue
            cost = pycolmap.cost_functions.Point3DAlignmentCost(
                cp["covariance"],
                cp["topo"],
                use_log_scale=True,
            )
            problem.add_residual_block(
                cost,
                loss,
                [
                    cp["triangulated"],
                    variables["sim3d"].rotation.quat,
                    variables["sim3d"].translation,
                    variables["log_scale"],
                ],
            )

        logger.info("Added Point3dAlignmentCost and ReprojErrorCost costs")

        problem.set_manifold(
            variables["sim3d"].rotation.quat,
            pyceres.EigenQuaternionManifold(),
        )

    return problem


def run(
    estimate: Path,
    cp_json_file: Path,
    device_calibration_json: Path,
    output_path: Path,
    reference_sensor: str = "imu",
    cp_reproj_std=1.0,
):
    """Run sparse evaluation for sequences that observe control points.

    Args:
        estimate (Path): Path to the pose estimate file.
        cp_json_file (Path): Path to the sparse ground-truth JSON file
            containing control point data.
        device_calibration_json (Path): Path to the device calibration JSON.
        output_path (Path): Directory where intermediate data and
        evaluation results will be saved.
        reference_sensor (str): The reference sensor to use ("imu" or "cam0").
        cp_reproj_std (float, optional): Control point reprojection standard
            deviation. Defaults to 1.0.

    Returns:
        bool: True if the evaluation was successful, False otherwise.

    Notes:
        Expected estimate file format (space-separated columns):
        ```
        timestamp tx ty tz qx qy qz qw
        ```
    """
    est = Estimate()
    est.load_from_file(estimate, invert_poses=False, reference_sensor=reference_sensor)
    if not est.is_loaded():
        logger.error("Estimate could not be loaded")
        return False

    reconstruction = initialize_reconstruction_from_calibration_file(
        device_calibration_json
    )
    
    if reference_sensor == "imu":
        rig_from_sensor = get_t_imu_camera_from_json(
            device_calibration_json, camera_label="cam0"
        )
        sensor_from_rig = rig_from_sensor.inverse()
    else:
        sensor_from_rig = pycolmap.Rigid3d()

    reconstruction_path = est.create_baseline_reconstruction(
        cp_json_file,
        sensor_from_rig,
        output_path,
    )
    reconstruction_path = est.reconstruction_path

    aligned_transformed_folder = output_path / "aligned_transformed"
    aligned_transformed_folder.mkdir(parents=True, exist_ok=True)

    control_points = construct_control_points_from_json(cp_json_file)
    assert control_points is not None, (
        "Control points could not be constructed from JSON"
    )

    run_control_point_triangulation_from_json(
        reconstruction_path,
        cp_json_file,
        control_points,
    )

    triangulated_cp_alignment, topo_cp_alignment = (
        get_cps_for_initial_alignment(control_points)
    )

    if triangulated_cp_alignment is None or topo_cp_alignment is None:
        logger.error("Not enough control points for initial alignment")
        return False

    ret = pycolmap.estimate_sim3d_robust(
        triangulated_cp_alignment, topo_cp_alignment, {"max_error": 5}
    )

    if ret is None:
        logger.error("Failed to estimate robust sim3d")
        return False

    robust_sim3d = ret["tgt_from_src"]

    #########################################################################
    output_data = {}
    output_data["robust_sim3d"] = copy.deepcopy(robust_sim3d)

    variables = create_variables_for_sparse_evaluation(
        control_points,
        robust_sim3d,
        cp_reproj_std,
    )

    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    problem, solver_options, summary = get_problem_for_sparse_alignment(
        reconstruction, variables
    )
    pyceres.solve(solver_options, problem, summary)
    print(summary.BriefReport())

    update_sim3d_scale(variables)

    output_data["full_alignment"] = {}
    output_data["full_alignment"]["sim3"] = copy.deepcopy(variables["sim3d"])
    output_data["full_alignment"]["error_3d"] = {}
    output_data["full_alignment"]["points"] = {}

    for tag_id, cp in control_points.items():
        output_data["full_alignment"]["points"][tag_id] = {}

        if cp["triangulated"] is None:
            output_data["full_alignment"]["points"][tag_id]["triangulated"] = (
                None
            )
            output_data["full_alignment"]["error_3d"][tag_id] = [
                np.nan,
                np.nan,
                np.nan,
            ]
            continue

        original_triangulated_point = cp["triangulated"]
        topo_cp = cp["topo"]
        output_data["full_alignment"]["points"][tag_id]["triangulated"] = (
            original_triangulated_point
        )
        output_data["full_alignment"]["points"][tag_id]["topo"] = topo_cp
        output_data["full_alignment"]["points"][tag_id]["transformed"] = (
            variables["sim3d"] * original_triangulated_point
        )
        error3d = variables["sim3d"] * original_triangulated_point - topo_cp

        output_data["full_alignment"]["error_3d"][tag_id] = error3d

    recon = pycolmap.Reconstruction(reconstruction)
    recon.transform(output_data["full_alignment"]["sim3"])
    recon.write(aligned_transformed_folder)

    output_data["control_points"] = {}
    for tag_id, cp in control_points.items():
        output_data["control_points"][tag_id] = {}
        output_data["control_points"][tag_id]["covariance"] = cp["covariance"]
        output_data["control_points"][tag_id]["geo_id"] = cp["control_point"]
        output_data["control_points"][tag_id]["topo"] = cp["topo"]

    np.save(os.path.join(output_path, "sparse_evaluation.npy"), output_data)

    logger.info("Sparse evaluation completed successfully")
    return True


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
        "--reference_sensor",
        type=str,
        default="imu",
        choices=["imu", "cam0"],
        help="The reference sensor in which the poses are expressed.",
    )
    parser.add_argument(
        "--cp_reproj_std",
        type=float,
        default=1.0,
        help="Control point reprojection standard deviation",
    )
    args = parser.parse_args()

    _ = run(
        args.estimate,
        args.cp_json_file,
        args.device_calibration_json,
        args.output_path,
        args.reference_sensor,
        args.cp_reproj_std,
    )
