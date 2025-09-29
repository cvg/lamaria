import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from .. import logger
from ..utils.constants import (
    CUSTOM_ORIGIN_COORDINATES,
)
from ..utils.timestamps import get_timestamp_to_images

ControlPoints = dict[int, "ControlPoint"]


@dataclass(slots=True)
class ControlPointSummary:
    name: str  # geo_id
    triangulated: np.ndarray
    topo: np.ndarray
    covariance: np.ndarray


@dataclass(slots=True)
class ControlPoint:
    name: str  # geo_id
    topo: np.ndarray
    covariance: np.ndarray
    image_name_to_detection: dict[str, np.ndarray] | None = field(
        default_factory=dict
    )  # image_name to [x, y] detection

    triangulated: np.ndarray | None = None  # None if triangulation fails
    inlier_ratio: float = 0.0
    # inlier list of (image_id, [x, y]) tuples
    inlier_detections: list[(int, np.ndarray)] = field(default_factory=list)

    @staticmethod
    def from_measurement(
        name: str,
        measurement_xyz: list[float | None],
        unc_xyz: list[float | None],
        origin_xyz: tuple[float, float, float],
    ) -> "ControlPoint":
        m = list(measurement_xyz)
        u = list(unc_xyz)
        if m[2] is None:
            assert u[2] is None, "If z is None, its uncertainty must be None."
            m[2] = origin_xyz[2]
            u[2] = 1e9  # very uncertain height

        # translated measurement to have smaller numerical values
        topo = np.asarray(m, dtype=np.float64) - np.asarray(
            origin_xyz, dtype=np.float64
        )
        cov = np.diag(np.square(np.asarray(u, dtype=np.float64)))

        return ControlPoint(name=name, topo=topo, covariance=cov)

    def has_height(self) -> bool:
        return bool(self.topo[2] != 0)

    def is_triangulated(self) -> bool:
        return self.triangulated is not None

    def summary(self) -> ControlPointSummary:
        return ControlPointSummary(
            name=self.name,
            triangulated=self.triangulated.tolist()
            if self.triangulated is not None
            else None,
            topo=self.topo.tolist(),
            covariance=self.covariance.tolist(),
        )


def load_cp_json(
    cp_json_file: Path,
    skip_detections: bool = False,
) -> tuple[ControlPoints, dict]:
    """Load control points and timestamp→images mapping from a JSON file.

    Args:
        cp_json_file (Path): Path to the sparse ground-truth JSON file.
        skip_detections (bool): If True, skip per-image detections and only
            load control point measurement data.

    Returns:
        Tuple:
            - control_points: Parsed control points collection.
            - timestamp_to_images: Mapping from timestamps (ns) to image names.
    """
    with open(cp_json_file) as file:
        cp_data = json.load(file)

    control_points = construct_control_points(cp_data, skip_detections)
    timestamp_to_images = get_timestamp_to_images(cp_data)

    return control_points, timestamp_to_images


def construct_control_points(
    cp_data: dict,
    skip_detections: bool = False,
) -> ControlPoints:
    """
    Construct ControlPoints dict from JSON data.

    Args:
        cp_data (dict): Loaded JSON data containing control point information.
        skip_detections (bool): If True, skip per-image detections and only
            load control point measurement data.

    Returns:
        control_points (ControlPoints): Control points dictionary.
    """

    control_points: ControlPoints = {}
    for geo_id, data in cp_data["control_points"].items():
        tag_ids = data["tag_id"]
        measurement = data["measurement"]
        unc = data["uncertainty"]
        images_observing_cp = data["image_names"]

        for tag_id in tag_ids:
            control_points[tag_id] = ControlPoint.from_measurement(
                geo_id,
                measurement,
                unc,
                CUSTOM_ORIGIN_COORDINATES,
            )

            if not skip_detections:
                for image_name in images_observing_cp:
                    detection = cp_data["images"][image_name]["detection"]
                    control_points[tag_id].image_name_to_detection[
                        image_name
                    ] = np.array(detection)

    return control_points


def transform_points(points, r, t, scale):
    transformed_points = []
    for point in points:
        transformed_point = scale * r.apply(point) + t
        transformed_points.append(transformed_point)

    return transformed_points


def transform_triangulated_control_points(
    control_points: ControlPoints, r, t, scale
) -> ControlPoints:
    """Apply similarity transform to triangulated control points in place."""
    for cp in control_points.values():
        if cp.triangulated is None:
            continue
        cp.triangulated = scale * r.apply(cp.triangulated) + t
    return control_points


def run_control_point_triangulation_from_json(
    reconstruction: pycolmap.Reconstruction,
    control_points: ControlPoints,
) -> None:
    """
    Triangulate control points from JSON file and add to control_points dict.
    Updates `control_points` in place to add:

    - ``triangulated``: np.ndarray(3,) or None if triangulation fails
    - ``inlier_ratio``: float
    - ``inlier_detections``: list of observations used for triangulation
    Args:
        reconstruction_path (Path): Path to the reconstruction folder
        cp_json_file (Path): Path to the sparse GT JSON file
        control_points (dict): Control points dictionary to be updated
    """

    image_names_to_ids = {
        image.name: image_id
        for image_id, image in reconstruction.images.items()
    }

    for _, cp in tqdm(
        control_points.items(), desc="Triangulating control points"
    ):
        geo_id = cp.name
        images_observing_cp = list(cp.image_name_to_detection.keys())

        pixel_points = []
        cam_from_worlds = []
        cameras = []

        image_ids_and_centers = []

        for image_name in images_observing_cp:
            if image_name not in image_names_to_ids:
                continue
            id = image_names_to_ids[image_name]
            image = reconstruction.images[id]
            detection = cp.image_name_to_detection[image_name]
            pixel_points.append(detection)
            cam_from_worlds.append(image.cam_from_world())
            cameras.append(reconstruction.cameras[image.camera_id])
            image_ids_and_centers.append((id, detection))

        # HANDLING THE CASE WHERE NO IMAGES OBSERVE THE CONTROL POINT
        if pixel_points == []:
            cp.triangulated = None
            cp.inlier_ratio = 0.0
            cp.inlier_detections = []
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
            print(f"Error in triangulating control point {geo_id}: {e}")
            output = None

        if output is None:
            cp.triangulated = None
            cp.inlier_ratio = 0.0
            cp.inlier_detections = []
            continue

        cp.triangulated = output["xyz"]
        inliers = output["inliers"]
        cp.inlier_ratio = np.sum(inliers) / len(inliers)
        image_ids_and_centers = [
            image_ids_and_centers[i] for i in range(len(inliers)) if inliers[i]
        ]
        cp.inlier_detections = image_ids_and_centers


def get_cps_for_initial_alignment(control_points: ControlPoints):
    """Get control points with z != 0 for initial alignment"""
    triangulated_cp_alignment = []
    topo_cp_alignment = []
    for _, cp in control_points.items():
        if not cp.is_triangulated():
            continue

        if cp.has_height():
            triangulated_cp_alignment.append(cp.triangulated)
            topo_cp_alignment.append(cp.topo)

    if len(topo_cp_alignment) < 3:
        logger.error(
            "Not enough control points with height for initial alignment. "
            "At least 3 control points with z != 0 are required."
        )
        return None, None

    triangulated_cp_alignment = np.array(triangulated_cp_alignment)
    topo_cp_alignment = np.array(topo_cp_alignment)

    return triangulated_cp_alignment, topo_cp_alignment
