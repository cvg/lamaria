from dataclasses import dataclass, field
import copy
import pycolmap
import pyceres
import numpy as np

from .. import logger
from .control_point import ControlPoint


@dataclass(slots=True)
class SparseEvalVariables:
    """Container for sparse-eval optimization state."""
    control_points: dict[int, "ControlPoint"]       # tag_id to ControlPoint
    sim3d: pycolmap.Sim3d
    cp_reproj_std: float = 1.0
    log_scale: np.ndarray = field(default_factory=lambda: np.array(0.0, dtype=np.float64))

    @classmethod
    def create_from_inputs(
        cls,
        control_points: dict,
        sim3d: pycolmap.Sim3d,
        cp_reproj_std: float = 1.0,
    ) -> "SparseEvalVariables":
        
        scale = copy.deepcopy(sim3d.scale)
        v = cls(
            control_points=copy.deepcopy(control_points),
            sim3d=copy.deepcopy(sim3d),
            cp_reproj_std=float(cp_reproj_std),
            log_scale=np.array(np.log(scale), dtype=np.float64),
        )
        
        return v
    
    def update_sim3d_scale_from_log(self) -> None:
        """Propagate optimized log_scale back into sim3d.scale."""
        log_scale = copy.deepcopy(self.log_scale)
        self.sim3d.scale = np.exp(log_scale)
    
    def reproj_cov(self) -> np.ndarray:
        """2x2 covariance matrix for CP reprojection noise."""
        return np.eye(2) * pow(self.cp_reproj_std, 2)
    

def get_problem_for_sparse_alignment(
    reconstruction: pycolmap.Reconstruction,
    variables: SparseEvalVariables,
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
    variables: SparseEvalVariables,
) -> pyceres.Problem:
    """Add alignment residuals to the Ceres problem.

    Variables consists of -
    - ReprojErrorCost for each observation of each control point
    - Point3DAlignmentCost for each control point
    Args:
        problem (pyceres.Problem): The Ceres problem.
        reconstruction (pycolmap.Reconstruction): The COLMAP reconstruction.
        variables (SparseEvalVariables): The sparse evaluation variables.
    """
    if variables.control_points is None or variables.sim3d is None:
        return problem

    loss = pyceres.TrivialLoss()
    point2d_cov = variables.reproj_cov()
    for tag_id, cp in variables.control_points.items():
        tri = cp.triangulated
        if tri is None:
            logger.info(f"Control point {tag_id} not triangulated")
            continue
        
        obs = cp.image_id_and_point2d
        for image_id, point2d in obs:
            image = reconstruction.images[image_id]
            pose = image.cam_from_world()
            camera = reconstruction.cameras[image.camera_id]

            point2d = np.asarray(point2d, dtype=np.float64).reshape(2, 1)

            cost = pycolmap.cost_functions.ReprojErrorCost(
                camera.model,
                point2d_cov,
                point2d,
                pose,
            )
            problem.add_residual_block(
                cost, loss, [tri, camera.params]
            )
            problem.set_parameter_block_constant(camera.params)

        cost = pycolmap.cost_functions.Point3DAlignmentCost(
            cp.covariance,
            cp.topo,
            use_log_scale=True,
        )
        problem.add_residual_block(
            cost,
            loss,
            [
                cp.triangulated,
                variables.sim3d.rotation.quat,
                variables.sim3d.translation,
                variables.log_scale,
            ],
        )

    problem.set_manifold(
        variables.sim3d.rotation.quat,
        pyceres.EigenQuaternionManifold(),
    )

    logger.info("Added Point3dAlignmentCost and ReprojErrorCost costs")

    return problem