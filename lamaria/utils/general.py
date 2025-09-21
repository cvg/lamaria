import os
import json
import numpy as np
import pycolmap
from pathlib import Path
from typing import List, Tuple
import shutil
from bisect import bisect_left
from typing import List
import subprocess
from scipy.spatial.transform import Rotation
from decimal import Decimal, ROUND_HALF_UP

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import get_nearest_pose
from projectaria_tools.core.calibration import CameraCalibration, ImuCalibration

from lamaria import logger

ARIA_CAMERAS = [("cam0", "camera-slam-left"), ("cam1", "camera-slam-right")]
LEFT_CAMERA_STREAM_ID = StreamId("1201-1")
RIGHT_CAMERA_STREAM_ID = StreamId("1201-2")
IMU_STREAM_ID = StreamId("1202-1")
LEFT_CAMERA_STREAM_LABEL = "camera-slam-left"
RIGHT_CAMERA_STREAM_LABEL = "camera-slam-right"
IMU_STREAM_LABEL = "imu-right"
CUSTOM_ORIGIN_COORDINATES = (2683594.4120000005, 1247727.7470000014, 417.307)


def find_closest_timestamp(
    timestamps: list, 
    target_ts: int,
    max_diff: float,
) -> int | None:
    
    """Timestamps must be in nano seconds"""
    index = bisect_left(timestamps, target_ts)
    if index == 0:
        return timestamps[0]
    if index == len(timestamps):
        return timestamps[-1]
    before = timestamps[index - 1]
    after = timestamps[index]
    if abs(target_ts - before) < abs(target_ts - after):
        closest = before
    else:
        closest = after

    if abs(target_ts - closest) > max_diff:
        return None

    return closest


def get_matched_timestamps(
    left_timestamps: list,
    right_timestamps: list,
    max_diff: float,
) -> list[tuple[int, int]]:
    
    matched_timestamps = []

    assert all(isinstance(ts, int) for ts in left_timestamps)
    assert all(isinstance(ts, int) for ts in right_timestamps)

    if len(left_timestamps) < len(right_timestamps):
        for lts in left_timestamps:
            closest_rts = find_closest_timestamp(right_timestamps, lts, max_diff)
            if abs(lts - closest_rts) < max_diff:
                matched_timestamps.append((lts, closest_rts))
    else:
        for rts in right_timestamps:
            closest_lts = find_closest_timestamp(left_timestamps, rts, max_diff)
            if abs(rts - closest_lts) < max_diff:
                matched_timestamps.append((closest_lts, rts))

    return matched_timestamps

def get_t_cam_a_cam_b_from_json(
    calibration_file: Path,
    camera_a_label: str,
    camera_b_label: str,
) -> pycolmap.Rigid3d:

    calib = json.load(calibration_file.open("r"))

    t_body_cam_a = calib[camera_a_label]["T_b_s"]
    t_body_cam_b = calib[camera_b_label]["T_b_s"]

    t_imu_cam_a = pycolmap.Rigid3d(
        pycolmap.Rotation3d(t_body_cam_a["qvec"]),
        t_body_cam_a["tvec"],
    )
    t_imu_cam_b = pycolmap.Rigid3d(
        pycolmap.Rotation3d(t_body_cam_b["qvec"]),
        t_body_cam_b["tvec"],
    )

    t_camera_a_camera_b = t_imu_cam_a.inverse() * t_imu_cam_b

    return t_camera_a_camera_b


def add_cameras_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    calibration_file: Path,
) -> None:

    for i, (key, _) in enumerate(ARIA_CAMERAS):
        cam = camera_colmap_from_json(
            calibration_file=calibration_file,
            camera_label=key,
        )
        cam.camera_id = i + 1
        reconstruction.add_camera(cam)

    rig = pycolmap.Rig(rig_id=1)
    ref_sensor = pycolmap.sensor_t(
        id=1,  # left camera is the rig
        type=pycolmap.SensorType.CAMERA,
    )
    rig.add_ref_sensor(ref_sensor)

    sensor1 = pycolmap.sensor_t(id=2, type=pycolmap.SensorType.CAMERA)
    sensor_from_rig = get_t_cam_a_cam_b_from_json(
        calibration_file=calibration_file,
        camera_a_label="cam1", # right
        camera_b_label="cam0", # left
    )
    rig.add_sensor(sensor1, sensor_from_rig)
    
    reconstruction.add_rig(rig)


def get_qvec_and_tvec_from_transform(transform) -> Tuple[np.ndarray, np.ndarray]:
    """Returns qvec in format x,y,z,w and tvec in format x,y,z"""
    # to_quat() returns in wxyz format
    # https://github.com/facebookresearch/projectaria_tools/blob/867105e65cadbe777db355a407d90533c71d2d06/core/python/sophus/SO3PyBind.h#L161
    qvec = transform.rotation().to_quat()[0]
    tvec = transform.translation()[0]

    qvec = np.roll(
        qvec, -1
    )  # change from w,x,y,z to x,y,z,w for pycolmap format

    return np.array(qvec), np.array(tvec)


def rigid3d_from_transform(transform) -> pycolmap.Rigid3d:
    q, t = get_qvec_and_tvec_from_transform(transform)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)


def get_magnitude_from_transform(transform: pycolmap.Rigid3d) -> Tuple[float, float]:
    translation = transform.translation
    quat_xyzw = transform.rotation.quat
    rotation = Rotation.from_quat(quat_xyzw)
    dr = np.rad2deg(rotation.magnitude())
    dt = np.linalg.norm(translation)

    return dr, dt


def get_camera_params_for_colmap(
    camera_calibration: CameraCalibration,
    camera_model: str,
) -> List[float]:
    # params = [f_u {f_v} c_u c_v [k_0: k_{numK-1}]
    # {p_0 p_1} {s_0 s_1 s_2 s_3}]
    # projection_params is a 15 length vector,
    # starting with focal length, pp, extra coeffs
    camera_params = camera_calibration.get_projection_params()
    f_x, f_y, c_x, c_y = (
        camera_params[0],
        camera_params[0],
        camera_params[1],
        camera_params[2],
    )

    p2, p1 = camera_params[-5], camera_params[-6]

    k1, k2, k3, k4, k5, k6 = camera_params[3:9]

    # FULL_OPENCV model format:
    # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

    # OPENCV_FISHEYE model format:
    # fx, fy, cx, cy, k1, k2, k3, k4

    if camera_model == "OPENCV_FISHEYE":
        params = [f_x, f_y, c_x, c_y, k1, k2, k3, k4]
    elif camera_model == "FULL_OPENCV":
        params = [f_x, f_y, c_x, c_y, k1, k2, p1, p2, k3, k4, k5, k6]
    elif camera_model == "RAD_TAN_THIN_PRISM_FISHEYE":
        aria_fisheye_params = camera_params
        focal_length = aria_fisheye_params[0]
        aria_fisheye_params = np.insert(aria_fisheye_params, 0, focal_length)
        params = aria_fisheye_params

    return params


def camera_colmap_from_calib(calib: CameraCalibration) -> pycolmap.Camera:
    if calib.get_model_name().name != "FISHEYE624":
        raise ValueError(f"Unsupported Aria model {calib.get_model_name().name}")
    model = "RAD_TAN_THIN_PRISM_FISHEYE"
    params = get_camera_params_for_colmap(calib, model)
    width, height = calib.get_image_size()
    return pycolmap.Camera(
        model=model,
        width=width,
        height=height,
        params=params,
    )


def camera_colmap_from_json(
    calibration_file: Path, # open sourced aria calibration files
    camera_label: str
) -> pycolmap.Camera:

    calib = json.load(calibration_file.open("r"))
    camera_data = calib[camera_label]
    if camera_data["model"] != "RAD_TAN_THIN_PRISM_FISHEYE":
        raise ValueError(f"Unsupported Aria model {camera_data['model']}")
    
    model = "RAD_TAN_THIN_PRISM_FISHEYE"
    params = camera_data["params"]
    width = camera_data["resolution"]["width"]
    height = camera_data["resolution"]["height"]
    return pycolmap.Camera(
        model=model,
        width=width,
        height=height,
        params=params,
    )

def round_ns(x: str | int | float) -> int:
    # works for "5120...274.999939", "5.12e11", or ints
    if isinstance(x, int):
        return x
    s = str(x)
    return int(Decimal(s).to_integral_value(rounding=ROUND_HALF_UP))

def delete_files_in_folder(folder, exclude_pattern=None):
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if exclude_pattern is not None and exclude_pattern in filename:
                continue

            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
    else:
        os.makedirs(folder, exist_ok=True)


def extract_images_from_vrs(
    vrs_file: Path,
    image_folder: Path,
    left_subfolder_name="left",
    right_subfolder_name="right",
    rgb_subfolder_name="rgb",
    verbose: bool = False,
    extract_rgb: bool = False,
    extract_left: bool = True,
    extract_right: bool = True,
):
    for camera, stream_id in [
        (left_subfolder_name, "1201-1"),
        (right_subfolder_name, "1201-2"),
        (rgb_subfolder_name, "214-1"),
    ]:
        if camera == rgb_subfolder_name and not extract_rgb:
            continue

        if camera == left_subfolder_name and not extract_left:
            continue

        if camera == right_subfolder_name and not extract_right:
            continue

        output_dir = image_folder / camera
        output_dir.mkdir(parents=True, exist_ok=True)
        delete_files_in_folder(output_dir)
        logger.info(
            "Extracting images for camera %s in VRS %s", camera, vrs_file
        )
        cmd = f"vrs extract-images {vrs_file} --to {output_dir} + {stream_id}"
        stdout = None if verbose else subprocess.PIPE
        out = subprocess.run(
            cmd, shell=True, stderr=subprocess.STDOUT, stdout=stdout
        )
        if out.returncode:
            msg = f"Command '{cmd}' returned {out.returncode}."
            if out.stdout:
                msg += "\n" + out.stdout.decode("utf-8")
            raise subprocess.SubprocessError(msg)
        logger.info("Done!")


def get_closed_loop_data_from_mps(mps_path: Path) -> List[mps.ClosedLoopTrajectoryPose]:
    closed_loop_traj_file = mps_path / "slam" / "closed_loop_trajectory.csv"
    data = mps.read_closed_loop_trajectory(closed_loop_traj_file.as_posix())
    return data

def get_rig_from_worlds_from_estimate(
    estimate_path: Path
) -> List[pycolmap.Rigid3d]:
    """Estimate file format: ts t_x t_y t_z q_x q_y q_z q_w"""
    
    rig_from_worlds = []
    with open(estimate_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            
            tvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            qvec = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            pose = pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), tvec)
            rig_from_world = pose.inverse()
            rig_from_worlds.append(rig_from_world)
    
    return rig_from_worlds

def get_mps_poses_for_timestamps(
    trajectory_data: List[mps.ClosedLoopTrajectoryPose],
    timestamps: List[int],
) -> List:
    
    poses = []
    if trajectory_data:
        for ts in timestamps:
            pose_information = get_nearest_pose(trajectory_data, ts)
            if pose_information:
                t_world_device = pose_information.transform_world_device
                poses.append(t_world_device)
            else:
                poses.append(None)

    return poses


def get_t_rig_world_for_device_time_ns(
    trajectory_data: List[mps.ClosedLoopTrajectoryPose],
    device_time_ns: int,
    imu_calibration: ImuCalibration,
):
    if trajectory_data:
        pose_information = get_nearest_pose(trajectory_data, device_time_ns)
        if pose_information:
            t_world_device = pose_information.transform_world_device
            t_device_imu = imu_calibration.get_transform_device_imu()
            assert (
                t_device_imu is not None
            ), f"No t_device_imu found for device time {device_time_ns}, \
                    check imu calibration"

            t_world_imu = t_world_device @ t_device_imu
            t_imu_world = t_world_imu.inverse()

            return t_imu_world
        else:
            return None
    else:
        raise ValueError("No trajectory data found")


def get_t_imu_camera(
    imu_calib: ImuCalibration,
    camera_calib: CameraCalibration,
    return_qt=False,
    return_matrix=False,
):
    t_device_cam = camera_calib.get_transform_device_camera()
    t_device_imu = imu_calib.get_transform_device_imu()
    t_imu_device = t_device_imu.inverse()

    t_imu_cam = t_imu_device @ t_device_cam

    assert not (
        return_qt and return_matrix
    ), "Only one of return_qt or return_matrix can be True"

    if return_qt:
        qvec, tvec = get_qvec_and_tvec_from_transform(t_imu_cam)
        return qvec, tvec

    if return_matrix:
        t_imu_cam_matrix = t_imu_cam.to_matrix()
        return t_imu_cam_matrix

    return t_imu_cam

def get_t_imu_camera_from_json(
    device_calibration_json: str,
    camera_label: str,
) -> pycolmap.Rigid3d:
    
    with open(device_calibration_json, "r") as f:
        device_calib = json.load(f)
    
    # body is right imu of Aria, therefore T_b_s is T_imu_camera
    t_body_cam = device_calib[camera_label]["T_b_s"]
    t_imu_camera = pycolmap.Rigid3d(
        pycolmap.Rotation3d(t_body_cam["qvec"]),
        t_body_cam["tvec"],
    )

    return t_imu_camera

def get_t_cam_a_cam_b_from_json(
    device_calibration_json: str,
    camera_a_label: str,
    camera_b_label: str,
) -> pycolmap.Rigid3d:
    
    with open(device_calibration_json, "r") as f:
        device_calib = json.load(f)
    
    # body is right imu of Aria, therefore T_b_s is T_imu_camera
    t_body_cam_a = device_calib[camera_a_label]["T_b_s"]
    t_body_cam_b = device_calib[camera_b_label]["T_b_s"]

    t_imu_cam_a = pycolmap.Rigid3d(
        pycolmap.Rotation3d(t_body_cam_a["qvec"]),
        t_body_cam_a["tvec"],
    )
    t_imu_cam_b = pycolmap.Rigid3d(
        pycolmap.Rotation3d(t_body_cam_b["qvec"]),
        t_body_cam_b["tvec"],
    )

    t_camera_a_camera_b = t_imu_cam_a.inverse() * t_imu_cam_b

    return t_camera_a_camera_b


def get_image_names_to_ids(reconstruction_dir: str):
    recon = pycolmap.Reconstruction(reconstruction_dir)
    image_names_to_ids = {}

    for image_id in recon.images.keys():
        image_name = recon.images[image_id].name
        image_names_to_ids[image_name] = image_id

    return image_names_to_ids