import json
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from ..utils.camera import (
    LEFT_CAMERA_STREAM_LABEL,
    RIGHT_CAMERA_STREAM_LABEL,
    add_cameras_to_reconstruction,
)
from ..utils.estimate import (
    round_ns,
)
from ..utils.general import (
    delete_files_in_folder,
    find_closest_timestamp,
)
from ..utils.transformation import (
    get_t_imu_camera_from_json,
)


def add_images_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    estimate_file: Path,
    cp_json_file: Path,
    device_calibration_json: Path,
    slam_input_imu: int = 1,
):
    """Add images to an existing empty reconstruction from a pose estimate file."""
    pose_data = []
    with open(estimate_file) as f:
        lines = f.readlines()

        if len(lines) == 0:
            return reconstruction

        if "#" in lines[0]:
            lines = lines[1:]

        for line in lines:
            parts = line.split()
            if len(parts) < 8:
                continue

            timestamp = round_ns(parts[0])

            tvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q_xyzw = np.array(
                [
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                ]
            )
            T_world_device = pycolmap.Rigid3d(pycolmap.Rotation3d(q_xyzw), tvec)

            pose_data.append((timestamp, T_world_device))

    with open(cp_json_file) as f:
        cp_data = json.load(f)

    image_id = 1
    rig = reconstruction.rig(rig_id=1)

    # if slam_input_imu is 1, then the poses are IMU poses
    # otherwise, the poses are left camera poses (monocular-cam0) (rig poses)
    if slam_input_imu == 1:
        transform = get_t_imu_camera_from_json(
            device_calibration_json=device_calibration_json,
            camera_label="cam0",
        )
    else:
        transform = pycolmap.Rigid3d()

    ts_data_processed = {}
    for label in [LEFT_CAMERA_STREAM_LABEL, RIGHT_CAMERA_STREAM_LABEL]:
        ts_data = cp_data["timestamps"][label]
        ts_data_processed[label] = {int(k): v for k, v in ts_data.items()}
        ts_data_processed[label]["sorted_keys"] = sorted(
            ts_data_processed[label].keys()
        )

    for i, (timestamp, pose) in tqdm(
        enumerate(pose_data),
        total=len(pose_data),
        desc="Adding images to reconstruction",
    ):
        T_world_rig = pose * transform
        frame = pycolmap.Frame()
        frame.rig_id = rig.rig_id
        frame.frame_id = i + 1
        frame.rig_from_world = T_world_rig.inverse()

        images_to_add = []

        for label, camera_id in [
            (LEFT_CAMERA_STREAM_LABEL, 1),
            (RIGHT_CAMERA_STREAM_LABEL, 2),
        ]:
            source_timestamps = ts_data_processed[label]["sorted_keys"]
            closest_timestamp = find_closest_timestamp(
                source_timestamps, timestamp
            )
            if closest_timestamp is None:
                raise ValueError

            image_name = ts_data_processed[label][closest_timestamp]

            im = pycolmap.Image(
                image_name,
                pycolmap.Point2DList(),
                camera_id,
                image_id,
            )
            im.frame_id = frame.frame_id
            frame.add_data_id(im.data_id)

            images_to_add.append(im)
            image_id += 1

        reconstruction.add_frame(frame)
        for im in images_to_add:
            reconstruction.add_image(im)


def create_baseline_reconstruction(
    estimate_file: Path,
    cp_json_file: Path,
    device_calibration_json: Path,
    output_path: Path,
    slam_input_imu: int = 1,
):
    """Create a baseline reconstruction from a pose estimate file.
    The pose estimate file is assumed to be in the format:

    timestamp(ns) tx ty tz q_x q_y q_z q_w

    Args:
        estimate_file (Path): Path to the pose estimate file.
        cp_json_file (Path): Path to the sparse GT json file.
        device_calibration_json (Path): Path to the Aria device calibration json file.
        output_path (Path): Path to the output folder where the reconstruction will be saved.
        slam_input_imu (int, optional): If 1, the poses in the estimate file are IMU poses.
                                        If 0, the poses are left camera poses (monocular-cam0).
    """
    recon_path = output_path / "reconstruction"
    recon_path.mkdir(parents=True, exist_ok=True)
    delete_files_in_folder(recon_path)

    reconstruction = pycolmap.Reconstruction()
    add_cameras_to_reconstruction(
        reconstruction,
        device_calibration_json,
    )

    add_images_to_reconstruction(
        reconstruction,
        estimate_file,
        cp_json_file,
        device_calibration_json,
        slam_input_imu,
    )

    reconstruction.write(recon_path)
