import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

from .. import logger
from .control_point import ControlPoint, ControlPointSummary


@dataclass(slots=True)
class SparseEvalVariables:
    """Container for sparse evaluation optimization variables."""

    control_points: dict[int, "ControlPoint"]  # tag_id to ControlPoint
    sim3d: pycolmap.Sim3d
    log_scale: np.ndarray = field(
        default_factory=lambda: np.array(0.0, dtype=np.float64)
    )

    @classmethod
    def create_from_inputs(
        cls,
        control_points: dict,
        sim3d: pycolmap.Sim3d,
    ) -> "SparseEvalVariables":
        scale = copy.deepcopy(sim3d.scale)
        v = cls(
            control_points=copy.deepcopy(control_points),
            sim3d=copy.deepcopy(sim3d),
            log_scale=np.array(np.log(scale), dtype=np.float64),
        )

        return v

    def update_sim3d_scale(self) -> None:
        """Propagate optimized log_scale back into sim3d.scale."""
        log_scale = copy.deepcopy(self.log_scale)
        self.sim3d.scale = np.exp(log_scale)

    def get_cp_summary(self) -> dict:
        """Get a brief summary of control points."""
        summary: dict[int, ControlPointSummary] = {}
        for tag_id, cp in self.control_points.items():
            summary[tag_id] = cp.summary()

        return summary


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
    problem = add_residuals_for_sparse_eval(
        problem,
        reconstruction,
        variables,
    )
    solver_options = pyceres.SolverOptions()
    solver_options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()

    return problem, solver_options, summary


def add_residuals_for_sparse_eval(
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

    for tag_id, cp in variables.control_points.items():
        tri = cp.triangulated
        if tri is None:
            logger.info(f"Control point {tag_id} not triangulated")
            continue

        point2d_cov = np.eye(2)

        obs = cp.inlier_detections
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
            problem.add_residual_block(cost, loss, [tri, camera.params])
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



@dataclass(slots=True)
class SparseEvalResult:
    """Container for sparse evaluation results."""
    alignment: pycolmap.Sim3d
    cp_summary: dict[int, ControlPointSummary]

    @staticmethod
    def from_variables(
        variables: SparseEvalVariables,
    ) -> "SparseEvalResult":
        alignment = copy.deepcopy(variables.sim3d)
        cp_summary = variables.get_cp_summary()

        return SparseEvalResult(
            alignment=alignment,
            cp_summary=cp_summary,
        )

    @classmethod
    def load_from_npy(cls, path: Path) -> "SparseEvalResult" | None:
        if not path.exists():
            logger.error(f"Result file not found: {path}")
            return None

        data = np.load(path, allow_pickle=True).item()
        if not isinstance(data, dict):
            logger.error(f"Invalid data format in: {path}")
            return None
        
        try:
            cp_summary = data["cp_summary"]
            alignment = data["alignment"]
        except KeyError as e:
            logger.error(f"Missing key in data: {e}")
            return None
        
        return cls(cp_summary=cp_summary, alignment=alignment)

    def save_as_npy(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)  # just in case
        np.save(path, asdict(self))
