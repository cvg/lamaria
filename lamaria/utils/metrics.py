import numpy as np
import pycolmap
from scipy.interpolate import interp1d

from .. import logger
from ..structs.sparse_eval import SparseEvalResult


def piecewise_linear_scoring():
    """
    Error             Score
    0   - 0.05 m       20
    0.05 - 0.20 m      18
    0.20 - 0.50 m      15
    0.50 - 1.00 m      12
    1.00 - 2.00 m       8
    2.00 - 5.00 m       4
    5.00 - 10.00 m      0
    > 10.00 m           0
    """
    x_vals = [0.00, 0.05, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00]
    y_vals = [20, 20, 18, 15, 12, 8, 4, 0]

    scoring = interp1d(
        x_vals,
        y_vals,
        kind="linear",
        fill_value=(y_vals[0], 0),  # left=20, right=0
        bounds_error=False,
    )
    return scoring


def calculate_control_point_score(result: SparseEvalResult) -> float:
    """Calculate CP score from SparseEvalResult object."""
    error_2d = calculate_error(result)
    if any(e < 0 for e in error_2d):
        logger.error("Negative errors found, norm error cannot be negative.")
        return 0.0

    errors = np.nan_to_num(error_2d, nan=1e6)  # Large error for NaNs
    scoring = piecewise_linear_scoring()
    scores = scoring(errors)
    score_sum = np.sum(scores)
    max_score = 20 * len(errors)

    S_j = (score_sum / max_score) * 100
    return S_j


def calculate_control_point_recall(
    result: SparseEvalResult,
    threshold: float = 1.0,  # meters
) -> float:
    """Calculate CP recall from SparseEvalResult object."""
    error_2d = calculate_error(result)
    if len(error_2d) == 0:
        return 0.0

    if any(e < 0 for e in error_2d):
        logger.error("Negative errors found, norm error cannot be negative.")
        return 0.0

    errors = np.nan_to_num(error_2d, nan=1e6)  # Large error for NaNs
    inliers = np.sum(errors <= threshold)
    recall = inliers / len(errors) * 100
    return recall


def calculate_error(result: SparseEvalResult) -> np.ndarray:
    """Calculate 2D errors from SparseEvalResult"""
    sim3d = result.alignment  # sim3d cannot be None here
    if not isinstance(sim3d, pycolmap.Sim3d):
        logger.error("No valid Sim3d found in SparseEvalResult")
        return np.array([])

    error_2d = []
    for _, cp in result.cp_summary.items():
        if not cp.is_triangulated():
            error_2d.append(np.nan)
            continue

        transformed_point = sim3d * cp.triangulated
        e = np.linalg.norm(transformed_point[:2] - cp.topo[:2])
        error_2d.append(e)

    assert len(error_2d) == len(result.cp_summary), "Error length mismatch"

    return np.array(error_2d)
