import copy

import numpy as np
import pycolmap
import pyceres
import pycolmap.cost_functions
from pathlib import Path
from typing import Dict, List

from lamaria import logger

def update_sim3d_scale(variables):
    if "log_scale" not in variables:
        raise ValueError("log_scale not found in variables")

    log_scale = copy.deepcopy(variables["log_scale"])
    variables["sim3d"].scale = np.exp(log_scale)

def create_variables_for_sparse_evaluation(
    control_points: Dict,
    sim3d: pycolmap.Sim3d,
    cp_reproj_std: float = 1.0
) -> Dict:
    
    variables = {}
    variables["control_points"] = copy.deepcopy(control_points)
    variables["sim3d"] = copy.deepcopy(sim3d)
    variables["cp_reproj_std"] = cp_reproj_std
    scale = copy.deepcopy(variables["sim3d"].scale)
    variables["log_scale"] = np.array(np.log(scale), dtype=np.float64)
    
    return variables


def get_problem_for_sparse_alignment(
    reconstruction: pycolmap.Reconstruction,
    variables: Dict,
):
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
    variables: Dict,
) -> pyceres.Problem:
    
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
        
        logger.info(f"Added Point3dAlignmentCost and ReprojErrorCost \
                    for {len(variables['control_points'])} control points")

        problem.set_manifold(
            variables["sim3d"].rotation.quat,
            pyceres.EigenQuaternionManifold(),
        )

    return problem
