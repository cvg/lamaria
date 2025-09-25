import copy
import os

from pathlib import Path
import numpy as np
import pyceres
import pycolmap

from ..utils.control_point import (
    construct_control_points_from_json,
    get_cps_for_initial_alignment,
    run_control_point_triangulation_from_json,
)
from ..utils.sparse_eval import (
    create_variables_for_sparse_evaluation,
    get_problem_for_sparse_alignment,
    update_sim3d_scale,
)


def run_baseline_evaluation(
    reconstruction_path: Path,
    cp_json_file: str,
    output_path: Path,
    cp_reproj_std=1.0,
):
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    aligned_transformed_folder = output_path / "aligned_transformed"
    aligned_transformed_folder.mkdir(parents=True, exist_ok=True)
    
    control_points = construct_control_points_from_json(cp_json_file)
    assert control_points is not None, "Control points could not be constructed from JSON"

    run_control_point_triangulation_from_json(
        reconstruction_path,
        cp_json_file,
        control_points,
    )

    triangulated_cp_alignment, topo_cp_alignment = (
        get_cps_for_initial_alignment(control_points)
    )

    if triangulated_cp_alignment is None or topo_cp_alignment is None:
        return (False, "Not enough control points for INITIAL alignment")

    ret = pycolmap.estimate_sim3d_robust(
        triangulated_cp_alignment, topo_cp_alignment, {"max_error": 5}
    )

    if ret is None:
        return (False, "Failed to estimate robust sim3d")

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
        reconstruction,
        variables
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

    return (True, "Sparse evaluation completed successfully")