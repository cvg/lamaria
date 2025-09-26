import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from tqdm import tqdm
from typing import Dict, List

from .general import (
    get_image_names_to_ids,
    CUSTOM_ORIGIN_COORDINATES,
)


def construct_control_points_from_json(
    cp_json_file: Path,
) -> Dict:
    """Construct control points dict from JSON file
    Args:
        cp_json_file (Path): Path to the sparse GT JSON file
    """
    
    cp_data = json.load(open(cp_json_file, 'r'))
    control_points = {}
    for geo_id, data in cp_data["control_points"].items():
        tag_ids = data["tag_id"]
        measurement = data["measurement"]
        unc = data["uncertainty"]

        if measurement[2] == None:
            assert unc[2] == None, "Uncertainty for z coordinate should be None if measurement is None"
            measurement[2] = CUSTOM_ORIGIN_COORDINATES[2]
            unc[2] = 1e9 # some large number
        
        translated_measurement = np.array(measurement) - np.array(CUSTOM_ORIGIN_COORDINATES)
        # those without height will have 0 height and large uncertainty in z

        for tag_id in tag_ids:
            control_points[tag_id] = {
                "control_point": geo_id,
                "topo": translated_measurement,
                "covariance": np.diag(np.square(unc))
            }
            
    return control_points


def check_3d_error_bet_triang_and_topo(triangulated: List, topo: List):
    """Calculate L2 error between triangulated and topo points
    Args:
        triangulated (list): list of triangulated points
        topo (list): list of topocentric points

    Returns:
        errors (list): list of L2 errors per point"""
    errors = []
    for i in range(len(triangulated)):
        errors.append(np.linalg.norm(triangulated[i] - topo[i]))

    return errors


def check_2d_error_bet_triang_and_topo(triangulated: List, topo: List):
    """Calculate L2 error between triangulated and topo points
    Args:
        triangulated (list): list of 3d triangulated points
        topo (list): list of 3d topocentric points

    Returns:
        errors (list): list of 2d L2 errors per point"""

    errors = []
    for i in range(len(triangulated)):
        errors.append(np.linalg.norm(triangulated[i][:2] - topo[i][:2]))

    return errors


def transform_points(points, r, t, scale):
    transformed_points = []
    for point in points:
        transformed_point = scale * r.apply(point) + t
        transformed_points.append(transformed_point)

    return transformed_points


def transform_triangulated_control_points(control_points: Dict, r, t, scale):
    for tag_id, cp in control_points.items():
        triangulated_point = cp["triangulated"]
        triangulated_point = scale * r.apply(triangulated_point) + t
        cp["triangulated"] = triangulated_point

    return control_points


def run_control_point_triangulation_from_json(
    reconstruction_path: Path,
    cp_json_file: Path,
    control_points: Dict, # edits control_points in place
) -> None:
    rec = pycolmap.Reconstruction(reconstruction_path)

    image_names_to_ids = get_image_names_to_ids(reconstruction_path=reconstruction_path)

    with open(cp_json_file, "r") as file:
        data = json.load(file)

    image_data = data["images"]
    control_point_data = data["control_points"]

    for tag_id, cp in tqdm(
        control_points.items(), desc="Triangulating control points"
    ):
        geo_id = cp["control_point"]
        images_observing_cp = control_point_data[geo_id]["image_names"]

        pixel_points = []
        cam_from_worlds = []
        cameras = []

        image_ids_and_centers = []

        for image_name in images_observing_cp:
            if image_name not in image_names_to_ids:
                continue
            id = image_names_to_ids[image_name]
            image = rec.images[id]
            detection = image_data[image_name]["detection"]
            pixel_points.append(detection)
            cam_from_worlds.append(image.cam_from_world())
            cameras.append(rec.cameras[image.camera_id])
            image_ids_and_centers.append((id, detection))

        # HANDLING THE CASE WHERE NO IMAGES OBSERVE THE CONTROL POINT
        if pixel_points == []:
            cp["triangulated"] = None
            cp["inlier_ratio"] = 0
            cp["image_id_and_point2d"] = []
            continue

        pixel_points = np.array(pixel_points)
        try:
            output = pycolmap.estimate_triangulation(
                pixel_points,
                cam_from_worlds,
                cameras,
                options={
                    "residual_type": "REPROJECTION_ERROR",
                    "ransac": {"max_error": 4.0},
                },
            )
        except Exception as e:
            print(
                f"Error in triangulating control point {geo_id}: {e}"
            )
            output = None

        if output is None:
            cp["triangulated"] = None
            cp["inlier_ratio"] = 0
            cp["image_id_and_point2d"] = []
            continue

        control_points[tag_id]["triangulated"] = output["xyz"]
        inliers = output["inliers"]
        control_points[tag_id]["inlier_ratio"] = np.sum(inliers) / len(inliers)
        image_ids_and_centers = [
            image_ids_and_centers[i] for i in range(len(inliers)) if inliers[i]
        ]
        control_points[tag_id]["image_id_and_point2d"] = image_ids_and_centers


def get_cps_for_initial_alignment(control_points: Dict):
    triangulated_cp_alignment = []
    topo_cp_alignment = []
    for tag_id, cp in control_points.items():
        if cp["triangulated"] is None:
            continue

        if cp["topo"][2] != 0:
            print(
                "Control point %s, %s has z != 0", tag_id, cp["control_point"]
            )
            triangulated_cp_alignment.append(cp["triangulated"])
            topo_cp_alignment.append(cp["topo"])

    if len(topo_cp_alignment) < 3:
        print("Not enough control points with z != 0")
        return None, None

    triangulated_cp_alignment = np.array(triangulated_cp_alignment)
    topo_cp_alignment = np.array(topo_cp_alignment)

    return triangulated_cp_alignment, topo_cp_alignment
