import pyceres
import pycolmap

from .. import logger
from ..structs.control_point import (
    ControlPoints,
    get_cps_for_initial_alignment,
)
from ..structs.sparse_eval import (
    SparseEvalResult,
    SparseEvalVariables,
    get_problem_for_sparse_alignment,
)


def estimate_initial_alignment_from_control_points(
    control_points: ControlPoints,
) -> pycolmap.Sim3d | None:
    """Estimate a robust Sim3d from control points.

    Args:
        control_points (ControlPoints): Control points dictionary.
    Returns:
        sim3d (pycolmap.Sim3d | None): The estimated Sim3d or None if
            estimation failed.
    """
    triangulated_cp_alignment, topo_cp_alignment = (
        get_cps_for_initial_alignment(control_points)
    )

    if triangulated_cp_alignment is None or topo_cp_alignment is None:
        logger.error("Not enough control points for initial alignment")
        return None

    ret = pycolmap.estimate_sim3d_robust(
        triangulated_cp_alignment, topo_cp_alignment, {"max_error": 5}
    )

    if ret is None:
        return None

    robust_sim3d = ret["tgt_from_src"]

    return robust_sim3d


def evaluate_wrt_control_points(
    reconstruction: pycolmap.Reconstruction,
    control_points: ControlPoints,
) -> SparseEvalResult | None:
    """
    Evaluate the trajectory with respect to control points.

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object
        which contains the estimated poses.
        control_points (ControlPoints): Control points dictionary.
        output_path (Path): Directory where results will be saved.

    Returns:
        sparse_npy_path (Path): Path to the saved SparseEvalResult .npy file.
    """

    robust_sim3d = estimate_initial_alignment_from_control_points(
        control_points
    )

    if robust_sim3d is None:
        logger.error("Robust Sim3d estimation failed")
        return None

    variables = SparseEvalVariables.create_from_inputs(
        control_points,
        robust_sim3d,
    )

    problem, solver_options, summary = get_problem_for_sparse_alignment(
        reconstruction, variables
    )
    pyceres.solve(solver_options, problem, summary)
    print(summary.BriefReport())

    variables.update_sim3d_scale()

    result = SparseEvalResult.from_variables(
        variables,
    )

    logger.info("Sparse evaluation completed successfully!")
    return result
