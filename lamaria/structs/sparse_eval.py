import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

from .. import logger
from .control_point import ControlPoint


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
        summary = {}
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

        point2d_cov = np.eye(2) * pow(cp.cp_reproj_std, 2)

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
class AlignedPoint:
    triangulated: np.ndarray | None
    topo: np.ndarray
    transformed: np.ndarray | None = None
    error_3d: np.ndarray | None = None


@dataclass(slots=True)
class AlignmentResult:
    optimized_sim3d: pycolmap.Sim3d
    points: dict[int, AlignedPoint]

    @staticmethod
    def calculate(
        variables: SparseEvalVariables,
    ) -> "AlignmentResult":
        points = {}
        sim3d = copy.deepcopy(variables.sim3d)

        for tag_id, cp in variables.control_points.items():
            tri = cp.triangulated
            if tri is None:
                points[tag_id] = AlignedPoint(
                    triangulated=None,
                    topo=cp.topo,
                    transformed=None,
                    error_3d=None,
                )
                continue

            transformed = sim3d * tri
            error_3d = transformed - cp.topo
            points[tag_id] = AlignedPoint(
                triangulated=tri,
                topo=cp.topo,
                transformed=transformed,
                error_3d=error_3d,
            )

        return AlignmentResult(
            optimized_sim3d=sim3d,
            points=points,
        )


@dataclass(slots=True)
class SparseEvalResult:
    alignment: AlignmentResult
    cp_summary: dict | None = None

    @staticmethod
    def from_variables(
        variables: SparseEvalVariables,
    ) -> "SparseEvalResult":
        alignment = AlignmentResult.calculate(variables)
        cp_summary = variables.get_cp_summary()

        return SparseEvalResult(
            alignment=alignment,
            cp_summary=cp_summary,
        )

    @classmethod
    def load_from_npy(cls, path: Path) -> "SparseEvalResult":
        if not path.exists():
            logger.error(f"Result file not found: {path}")
            return None
        
        data = np.load(path, allow_pickle=True).item()
        alignment_data = data["alignment"]

        opt = alignment_data["optimized_sim3d"]
        if isinstance(opt, pycolmap.Sim3d):
            sim3d = opt

        alignment = AlignmentResult(
            optimized_sim3d=sim3d,
            points={
                int(tag_id): AlignedPoint(
                    triangulated=np.asarray(point["triangulated"])
                    if point["triangulated"] is not None
                    else None,
                    topo=np.asarray(point["topo"]),
                    transformed=np.asarray(point["transformed"])
                    if point["transformed"] is not None
                    else None,
                    error_3d=np.asarray(point["error_3d"])
                    if point["error_3d"] is not None
                    else None,
                )
                for tag_id, point in alignment_data["points"].items()
            },
        )
        cp_summary = data.get("cp_summary", None)

        return cls(
            alignment=alignment,
            cp_summary=cp_summary,
        )
    
    def save_as_npy(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)  # just in case
        np.save(path, asdict(self)) 
