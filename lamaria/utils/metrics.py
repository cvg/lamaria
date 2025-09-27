from pathlib import Path

import numpy as np
import pycolmap
from scipy.interpolate import interp1d


def get_sim3_from_sparse_evaluation(
    sparse_evaluation_npy: Path,
) -> pycolmap.Sim3d:
    data_sparse = np.load(sparse_evaluation_npy, allow_pickle=True).item()
    sim3 = data_sparse["full_alignment"]["sim3"]
    return sim3


def get_all_tag_ids(sparse_evaluation_npy: Path) -> list[str]:
    data_sparse = np.load(sparse_evaluation_npy, allow_pickle=True).item()
    tag_ids = sorted(list(data_sparse["control_points"].keys()))
    return tag_ids


def calculate_2d_alignment_errors(sparse_evaluation_npy: Path) -> list[float]:
    data_sparse = np.load(sparse_evaluation_npy, allow_pickle=True).item()
    tag_ids = get_all_tag_ids(sparse_evaluation_npy)
    err_2d = [
        np.linalg.norm(data_sparse["full_alignment"]["error_3d"][i][:2])
        for i in tag_ids
    ]

    return err_2d


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


def calculate_score_piecewise(errors):
    if len(errors) == 0 or errors is None:
        return 0.0

    if any(e < 0 for e in errors):
        raise ValueError("Norm error cannot be negative.")

    errors = np.nan_to_num(errors, nan=1000)  # 1km error for NaNs

    scoring = piecewise_linear_scoring()
    scores = scoring(errors)
    score_sum = np.sum(scores)
    max_score = 20 * len(errors)

    S_j = (score_sum / max_score) * 100
    return S_j


def calculate_score_from_alignment_data(sparse_evaluation_npy: Path) -> float:
    tag_ids = get_all_tag_ids(sparse_evaluation_npy)
    err_2d = calculate_2d_alignment_errors(sparse_evaluation_npy)

    if len(err_2d) == 0:
        raise ValueError("No valid control points found.")

    assert len(err_2d) == len(tag_ids), (
        "Mismatch in number of 2D CPs and errors."
    )

    S_2d = calculate_score_piecewise(err_2d)

    return S_2d
