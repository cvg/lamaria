"""Synthetic scene shared by the evaluation tests.

A rig with two fisheye cameras mounted on an IMU drives past a wall of
control points. Ground-truth poses are expressed in the frame of the
control-point measurements (the "topo" frame). The estimate under evaluation
is the same trajectory expressed in a different similarity frame, so the
evaluation has to recover a known Sim(3), including its scale.

Control point detections are rendered from the estimate's own reconstruction
(same rig calibration the evaluation uses), which makes the triangulation
exact and lets the tests assert tight tolerances. Every image carries at most
one detection, matching the sparse ground-truth JSON format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from lamaria.structs.control_point import ControlPoints, load_cp_json
from lamaria.structs.sparse_eval import SparseEvalResult
from lamaria.structs.trajectory import Trajectory
from lamaria.utils.aria import initialize_reconstruction_from_calibration_file
from lamaria.utils.constants import (
    CUSTOM_ORIGIN_COORDINATES,
    LEFT_CAMERA_STREAM_LABEL,
    RIGHT_CAMERA_STREAM_LABEL,
)
from lamaria.utils.metrics import calculate_error

FIRST_TIMESTAMP_NS = 1_000_000_000
FRAME_PERIOD_NS = 50_000_000  # 20 Hz
NUM_FRAMES = 40

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_MODEL = "RAD_TAN_THIN_PRISM_FISHEYE"
# fx, fy, cx, cy, k0..k5, p0, p1, s0..s3 (distortion-free)
CAMERA_PARAMS = [300.0, 300.0, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2] + [0.0] * 12

CAMERA_LABELS = {
    "cam0": LEFT_CAMERA_STREAM_LABEL,
    "cam1": RIGHT_CAMERA_STREAM_LABEL,
}
CAMERA_IDS = {"cam0": 2, "cam1": 3}  # camera id 1 is the dummy IMU

CP_UNCERTAINTY_M = 0.02


def rotation_from_euler(yaw: float, pitch: float, roll: float):
    """Rotation3d from z-y-x Euler angles in degrees."""
    quat_xyzw = Rotation.from_euler(
        "zyx", [yaw, pitch, roll], degrees=True
    ).as_quat()
    return pycolmap.Rotation3d(quat_xyzw)


# imu_from_camera for both SLAM cameras (T_b_s in the calibration file).
# The cameras look along +z of the rig and are toed out a little.
RIG_FROM_CAMERA = {
    "cam0": pycolmap.Rigid3d(
        rotation_from_euler(0.0, 8.0, 0.0), np.array([-0.06, 0.01, 0.0])
    ),
    "cam1": pycolmap.Rigid3d(
        rotation_from_euler(0.0, -8.0, 0.0), np.array([0.06, -0.01, 0.0])
    ),
}


@dataclass(frozen=True)
class ControlPointSpec:
    geo_id: str
    tag_id: int
    topo: np.ndarray  # metres, relative to CUSTOM_ORIGIN_COORDINATES
    has_height: bool


# Control points on a wall ~3 m in front of the trajectory, spaced along it.
CONTROL_POINTS = [
    ControlPointSpec("CP_A", 11, np.array([0.3, 0.35, 3.0]), True),
    ControlPointSpec("CP_B", 12, np.array([1.4, -0.35, 3.15]), True),
    ControlPointSpec("CP_C", 13, np.array([2.5, 0.35, 2.85]), False),
    ControlPointSpec("CP_D", 14, np.array([3.6, -0.35, 3.0]), True),
    ControlPointSpec("CP_E", 15, np.array([4.7, 0.35, 3.15]), True),
    ControlPointSpec("CP_F", 16, np.array([5.8, -0.35, 2.85]), False),
    ControlPointSpec("CP_G", 17, np.array([6.9, 0.35, 3.0]), True),
    ControlPointSpec("CP_H", 18, np.array([8.0, -0.35, 3.15]), True),
]
TAGS_WITH_HEIGHT = [cp.tag_id for cp in CONTROL_POINTS if cp.has_height]


def _format_pose(timestamp: int, world_from_sensor: pycolmap.Rigid3d) -> str:
    values = [
        *world_from_sensor.translation,
        *world_from_sensor.rotation.quat,
    ]
    return f"{timestamp} " + " ".join(f"{v:.15g}" for v in values) + "\n"


def _rigid_part(sim3d: pycolmap.Sim3d) -> pycolmap.Rigid3d:
    return pycolmap.Rigid3d(sim3d.rotation, sim3d.translation)


class SyntheticScene:
    """Ground truth, estimate and control point observations."""

    def __init__(self, num_frames: int = NUM_FRAMES):
        self.timestamps = [
            FIRST_TIMESTAMP_NS + i * FRAME_PERIOD_NS for i in range(num_frames)
        ]

        # Ground truth: the rig moves along +x while looking at the wall (+z)
        # and wobbles a little in every axis so that no pose is trivial.
        self.topo_from_imu: list[pycolmap.Rigid3d] = []
        for i in range(num_frames):
            phase = 2.0 * np.pi * i / num_frames
            center = np.array(
                [0.2 * i, 0.3 * np.sin(phase), 0.15 * np.cos(2.0 * phase)]
            )
            rotation = rotation_from_euler(
                2.0 * np.sin(phase),
                5.0 * np.sin(3.0 * phase),
                3.0 * np.cos(phase),
            )
            self.topo_from_imu.append(pycolmap.Rigid3d(rotation, center))

        # The estimate lives in a frame related to the topo frame by this
        # similarity transform (what the evaluation has to recover).
        self.topo_from_est = pycolmap.Sim3d(
            1.7,
            rotation_from_euler(25.0, 0.0, 5.0),
            np.array([12.0, -4.0, 1.5]),
        )
        self.est_from_topo = self.topo_from_est.inverse()

        self.control_points = {cp.tag_id: cp for cp in CONTROL_POINTS}

        # Image naming and single-detection-per-image observations.
        self.image_names: dict[str, list[str]] = {
            key: [f"{key}_{i:03d}.jpg" for i in range(num_frames)]
            for key in CAMERA_IDS
        }
        self.observations: dict[str, tuple[int, np.ndarray]] = {}
        camera = pycolmap.Camera(
            model=CAMERA_MODEL,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            params=CAMERA_PARAMS,
        )
        for i in range(num_frames):
            rig_x = self.topo_from_imu[i].translation[0]
            nearest = min(
                CONTROL_POINTS, key=lambda cp: abs(cp.topo[0] - rig_x)
            )
            for key in CAMERA_IDS:
                cam_from_world = self.cam_from_world_est(i, key)
                point_in_cam = cam_from_world * self.expected_triangulation(
                    nearest.tag_id
                )
                assert point_in_cam[2] > 0, "control point behind the camera"
                detection = camera.img_from_cam(point_in_cam)
                assert 0 <= detection[0] < IMAGE_WIDTH, detection
                assert 0 <= detection[1] < IMAGE_HEIGHT, detection
                self.observations[self.image_names[key][i]] = (
                    nearest.tag_id,
                    detection,
                )

    # ----- geometry ----- #

    def world_from_imu_est(self, index: int) -> pycolmap.Rigid3d:
        """Estimate pose (world_from_imu) of a frame in the estimate frame."""
        gt = self.topo_from_imu[index]
        return _rigid_part(
            self.est_from_topo
            * pycolmap.Sim3d(1.0, gt.rotation, gt.translation)
        )

    def cam_from_world_est(self, index: int, key: str) -> pycolmap.Rigid3d:
        imu_from_world = self.world_from_imu_est(index).inverse()
        return RIG_FROM_CAMERA[key].inverse() * imu_from_world

    def expected_triangulation(self, tag_id: int) -> np.ndarray:
        """Where a control point triangulates in the estimate frame."""
        return self.est_from_topo * self.control_points[tag_id].topo

    def images_per_tag(self, indices=None) -> dict[int, int]:
        """Number of images observing each tag within the given frames."""
        indices = range(len(self.timestamps)) if indices is None else indices
        counts = dict.fromkeys(self.control_points, 0)
        for i in indices:
            for key in CAMERA_IDS:
                tag_id, _ = self.observations[self.image_names[key][i]]
                counts[tag_id] += 1
        return counts

    def triangulable_tags(self, indices=None) -> set[int]:
        return {
            tag for tag, n in self.images_per_tag(indices).items() if n >= 2
        }

    # ----- files ----- #

    def write_calibration(self, path: Path) -> Path:
        calibration = {}
        for key, rig_from_camera in RIG_FROM_CAMERA.items():
            calibration[key] = {
                "model": CAMERA_MODEL,
                "params": CAMERA_PARAMS,
                "resolution": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
                "T_b_s": {
                    "qvec": rig_from_camera.rotation.quat.tolist(),
                    "tvec": rig_from_camera.translation.tolist(),
                },
            }
        path.write_text(json.dumps(calibration, indent=2))
        return path

    def write_control_points(self, path: Path) -> Path:
        origin = np.asarray(CUSTOM_ORIGIN_COORDINATES)
        control_points = {}
        for cp in CONTROL_POINTS:
            measurement = (cp.topo + origin).tolist()
            uncertainty = [CP_UNCERTAINTY_M] * 3
            if not cp.has_height:
                measurement[2] = None
                uncertainty[2] = None
            control_points[cp.geo_id] = {
                "tag_id": [cp.tag_id],
                "measurement": measurement,
                "uncertainty": uncertainty,
                "image_names": [
                    name
                    for name, (tag_id, _) in self.observations.items()
                    if tag_id == cp.tag_id
                ],
            }
        data = {
            "control_points": control_points,
            "images": {
                name: {"detection": detection.tolist()}
                for name, (_, detection) in self.observations.items()
            },
            "timestamps": {
                CAMERA_LABELS[key]: {
                    str(ts): name
                    for ts, name in zip(
                        self.timestamps, self.image_names[key], strict=True
                    )
                }
                for key in CAMERA_IDS
            },
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    def write_estimate(
        self, path: Path, indices=None, sensor: str = "imu"
    ) -> Path:
        """Write the estimate (world_from_sensor, estimate frame)."""
        indices = range(len(self.timestamps)) if indices is None else indices
        lines = []
        for i in indices:
            world_from_imu = self.world_from_imu_est(i)
            if sensor == "imu":
                world_from_sensor = world_from_imu
            else:
                world_from_sensor = world_from_imu * RIG_FROM_CAMERA[sensor]
            lines.append(_format_pose(self.timestamps[i], world_from_sensor))
        path.write_text("".join(lines))
        return path

    def write_ground_truth(self, path: Path, indices=None) -> Path:
        """Write the ground truth (world_from_imu, topo frame)."""
        indices = range(len(self.timestamps)) if indices is None else indices
        path.write_text(
            "".join(
                _format_pose(self.timestamps[i], self.topo_from_imu[i])
                for i in indices
            )
        )
        return path

    def build_reconstruction(
        self, directory: Path, indices=None, sensor: str = "imu"
    ) -> tuple[pycolmap.Reconstruction, ControlPoints]:
        """Run the loading part of ``evaluate_wrt_control_points``."""
        directory.mkdir(parents=True, exist_ok=True)
        calibration = self.write_calibration(directory / "calibration.json")
        cp_json = self.write_control_points(directory / "control_points.json")
        estimate = self.write_estimate(
            directory / "estimate.txt", indices=indices, sensor=sensor
        )

        trajectory = Trajectory.load_from_file(
            estimate, invert_poses=False, corresponding_sensor=sensor
        )
        reconstruction = initialize_reconstruction_from_calibration_file(
            calibration
        )
        control_points, timestamp_to_images = load_cp_json(cp_json)
        reconstruction = trajectory.add_estimate_poses_to_reconstruction(
            reconstruction, timestamp_to_images
        )
        return reconstruction, control_points


# ----- assertion helpers ----- #


def assert_sim3d_close(
    test_case,
    actual: pycolmap.Sim3d,
    expected: pycolmap.Sim3d,
    atol: float = 1e-6,
) -> None:
    test_case.assertIsInstance(actual, pycolmap.Sim3d)
    test_case.assertAlmostEqual(actual.scale, expected.scale, delta=atol)
    np.testing.assert_allclose(
        actual.translation, expected.translation, atol=atol
    )
    test_case.assertLessEqual(actual.rotation.angle_to(expected.rotation), atol)


def errors_by_tag(result: SparseEvalResult) -> dict[int, float]:
    """2D control point errors keyed by tag id (NaN if not triangulated)."""
    return dict(
        zip(result.cp_summary.keys(), calculate_error(result), strict=True)
    )
