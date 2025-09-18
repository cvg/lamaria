import json
import os
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from ..utils.general import (
    LEFT_CAMERA_STREAM_LABEL,
    RIGHT_CAMERA_STREAM_LABEL,
    add_cameras_to_reconstruction,
    find_closest_timestamp,
    delete_files_in_folder,
    get_t_imu_camera_from_json,
    round_ns,
)


def add_images_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    pred_estimate_file,
    cp_json_file,
    device_calibration_json,
    slam_input_imu,
):
    pose_data = []
    # ESTIMATE FORMAT:
    # timestamp(ns) tx ty tz q_x q_y q_z q_w
    with open(pred_estimate_file, "r") as f:
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

            tvec = np.array(
                [float(parts[1]),
                 float(parts[2]),
                 float(parts[3])])
            q_xyzw = np.array([
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            ])
            T_world_device = pycolmap.Rigid3d(pycolmap.Rotation3d(q_xyzw),
                                              tvec)

            pose_data.append((timestamp, T_world_device))

    with open(cp_json_file, "r") as f:
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
        ts_data_processed[label]["sorted_keys"] = sorted(ts_data_processed[label].keys())

    for i, (timestamp, pose) in tqdm(enumerate(pose_data), 
                                     total=len(pose_data),
                                     desc="Adding images to reconstruction"):

        T_world_rig = pose * transform
        frame = pycolmap.Frame()
        frame.rig_id = rig.rig_id
        frame.frame_id = i + 1
        frame.rig_from_world = T_world_rig.inverse()

        images_to_add = []

        for label, camera_id in [(LEFT_CAMERA_STREAM_LABEL, 1),
                             (RIGHT_CAMERA_STREAM_LABEL, 2)]:

            source_timestamps = ts_data_processed[label]["sorted_keys"]
            closest_timestamp = find_closest_timestamp(
                source_timestamps,
                timestamp
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
    pred_estimate_file: Path,
    eval_folder: Path,
    cp_json_file: Path,
    device_calibration_json: Path,
    slam_input_imu: int,
):
    recon_path = eval_folder / "reconstruction"
    recon_path.mkdir(parents=True, exist_ok=True)
    delete_files_in_folder(recon_path)

    recon = pycolmap.Reconstruction()
    add_cameras_to_reconstruction(
        recon,
        device_calibration_json,
    )

    add_images_to_reconstruction(
        reconstruction=recon,
        pred_estimate_file=pred_estimate_file,
        cp_json_file=cp_json_file,
        device_calibration_json=device_calibration_json,
        slam_input_imu=slam_input_imu,
    )

    recon.write(recon_path)
