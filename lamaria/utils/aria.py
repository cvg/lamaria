import json
import shutil
import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pycolmap
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.calibration import CameraCalibration, ImuCalibration
from projectaria_tools.core.mps.utils import get_nearest_pose
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .. import logger
from .constants import ARIA_CAMERAS, RIGHT_IMU_STREAM_ID
from .timestamps import find_closest_timestamp


# ----- Reconstruction functions ----- #

InitReconstruction: TypeAlias = pycolmap.Reconstruction

def initialize_reconstruction_from_calibration_file(
    calibration_file: Path,
) -> InitReconstruction:
    """Initialize a COLMAP reconstruction from Aria calibration
    json file found on website: https://lamaria.ethz.ch/slam_datasets
    Adds a dummy camera as an IMU along with two cameras.

    Args:
        calibration_file (Path):
        Path to the Aria calibration json file
    Returns:
        InitReconstruction: The initialized COLMAP reconstruction
    """
    reconstruction = pycolmap.Reconstruction()

    imu = pycolmap.Camera(
        camera_id=1,
        model="SIMPLE_PINHOLE",
        params=[0, 0, 0]
    )
    reconstruction.add_camera(imu)

    rig = pycolmap.Rig(rig_id=1)
    ref_sensor = pycolmap.sensor_t(
        id=1,  # imu is the rig
        type=pycolmap.SensorType.CAMERA,
    )
    rig.add_ref_sensor(ref_sensor)

    for i, (key, _) in enumerate(ARIA_CAMERAS):
        cam = camera_colmap_from_json(
            calibration_file=calibration_file,
            camera_label=key,
        )
        cam.camera_id = i + 2  # start from 2 since 1 is imu
        reconstruction.add_camera(cam)

        sensor = pycolmap.sensor_t(id=i + 2, type=pycolmap.SensorType.CAMERA)
        rig_from_sensor = get_t_imu_camera_from_calibration_file(
            calibration_file=calibration_file,
            camera_label=key,
        )
        sensor_from_rig = rig_from_sensor.inverse()
        rig.add_sensor(sensor, sensor_from_rig)

    reconstruction.add_rig(rig)

    return reconstruction


# ----- Camera functions ----- #


def initialize_reconstruction_with_cameras(
    calibration_file: Path,
) -> InitReconstruction:
    """Initialize a COLMAP reconstruction from Aria calibration
    json file found on website: https://lamaria.ethz.ch/slam_datasets
    Adds only the cameras without any dummy IMU.

    Args:
        calibration_file (Path):
        Path to the Aria calibration json file
    Returns:
        InitReconstruction: The initialized COLMAP reconstruction
    """
    reconstruction = pycolmap.Reconstruction()

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
    sensor_from_rig = get_t_cam_a_cam_b_from_calibration_file(
        calibration_file=calibration_file,
        camera_a_label="cam1",  # right
        camera_b_label="cam0",  # left
    )
    rig.add_sensor(sensor1, sensor_from_rig)

    reconstruction.add_rig(rig)

    return reconstruction


def get_camera_params_for_colmap(
    camera_calibration: CameraCalibration,
    camera_model: str,
) -> list[float]:
    """
    Convert Aria CameraCalibration to COLMAP camera parameters.
    Supported models: OPENCV_FISHEYE, FULL_OPENCV, RAD_TAN_THIN_PRISM_FISHEYE
    Args:
        camera_calibration (CameraCalibration):
        The projectaria_tools CameraCalibration object
        camera_model (str): The COLMAP camera model to use
    Returns:
        list[float]: The camera parameters in COLMAP format
    """
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
    """Loads pycolmap camera from Aria CameraCalibration object"""
    if calib.get_model_name().name != "FISHEYE624":
        raise ValueError(
            f"Unsupported Aria model {calib.get_model_name().name}"
        )
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
    calibration_file: Path,  # open sourced aria calibration files
    camera_label: str,
) -> pycolmap.Camera:
    """Loads pycolmap camera from Aria calibration json file found on website:
    https://lamaria.ethz.ch/slam_datasets"""

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


# ----- Transformation functions ----- #


def get_closed_loop_data_from_mps(
    mps_path: Path,
) -> list[mps.ClosedLoopTrajectoryPose]:
    """Get closed loop trajectory data from MPS folder."""
    closed_loop_traj_file = mps_path / "slam" / "closed_loop_trajectory.csv"
    data = mps.read_closed_loop_trajectory(closed_loop_traj_file.as_posix())
    return data


def get_mps_poses_for_timestamps(
    trajectory_data: list[mps.ClosedLoopTrajectoryPose],
    timestamps: list[int],
) -> list:
    """Get T_world_device for a list of device timestamps
    in nanoseconds using MPS trajectory data.
    Returns None for timestamps where
    no trajectory data is found.
    """
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
    trajectory_data: list[mps.ClosedLoopTrajectoryPose],
    device_time_ns: int,
    imu_calibration: ImuCalibration,
):
    """Get T_rig_world (rig is right IMU sensor) for a given device time
    in nanoseconds using MPS trajectory data and IMU calibration.
    Returns None if no trajectory data is found for the given timestamp.
    """
    if trajectory_data:
        pose_information = get_nearest_pose(trajectory_data, device_time_ns)
        if pose_information:
            t_world_device = pose_information.transform_world_device
            t_device_imu = imu_calibration.get_transform_device_imu()
            assert t_device_imu is not None, (
                f"No t_device_imu found for device time {device_time_ns}, \
                    check imu calibration"
            )

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
) -> pycolmap.Rigid3d:
    """Get T_imu_camera from Aria calibrations.
    Returns pycolmap.Rigid3d transform.
    """

    t_device_cam = camera_calib.get_transform_device_camera()
    t_device_imu = imu_calib.get_transform_device_imu()
    t_imu_device = t_device_imu.inverse()

    t_imu_cam = t_imu_device @ t_device_cam

    colmap_t_imu_cam = rigid3d_from_transform(t_imu_cam)

    return colmap_t_imu_cam


def rigid3d_from_transform(transform) -> pycolmap.Rigid3d:
    """Converts projectaria_tools Rigid3d to pycolmap Rigid3d

    Note: to_quat() returns in wxyz format, but pycolmap.Rotation3d
    expects xyzw format."""

    # https://github.com/facebookresearch/projectaria_tools/blob/867105e65cadbe777db355a407d90533c71d2d06/core/python/sophus/SO3PyBind.h#L161

    qvec = transform.rotation().to_quat()[0]
    tvec = transform.translation()[0]
    qvec = np.roll(
        qvec, -1
    )  # change from w,x,y,z to x,y,z,w for pycolmap format
    q = np.array(qvec)
    t = np.array(tvec)

    return pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)


def get_magnitude_from_transform(
    transform: pycolmap.Rigid3d,
) -> tuple[float, float]:
    """Returns rotation (in degrees) and
    translation (in meters) magnitudes
    from a Rigid3d transform
    """
    translation = transform.translation
    quat_xyzw = transform.rotation.quat
    rotation = Rotation.from_quat(quat_xyzw)
    dr = np.rad2deg(rotation.magnitude())
    dt = np.linalg.norm(translation)

    return dr, dt


def get_t_cam_a_cam_b_from_calibration_file(
    calibration_file: Path,
    camera_a_label: str,
    camera_b_label: str,
) -> pycolmap.Rigid3d:
    """Get T_cam_a_cam_b from calibration json file (found on website: https://lamaria.ethz.ch/slam_datasets)
    for given camera labels."""

    calib = json.load(calibration_file.open("r"))

    # body is right imu of Aria, therefore T_b_s is T_imu_camera
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


def get_t_imu_camera_from_calibration_file(
    calibration_file: Path,
    camera_label: str,
) -> pycolmap.Rigid3d:
    """Get T_imu_camera from calibration json file (found on website: https://lamaria.ethz.ch/slam_datasets)
    for a given camera label."""

    calib = json.load(calibration_file.open("r"))

    # body is right imu of Aria, therefore T_b_s is T_imu_camera
    t_body_cam = calib[camera_label]["T_b_s"]
    t_imu_camera = pycolmap.Rigid3d(
        pycolmap.Rotation3d(t_body_cam["qvec"]),
        t_body_cam["tvec"],
    )

    return t_imu_camera


# ----- IMU functions ----- #


def get_online_params_for_imu_from_mps(
    online_calibs_file: Path, stream_label: str, num_error: float = 1e6
):
    online_calibs = mps.read_online_calibration(online_calibs_file.as_posix())
    online_imu_calibs = {}
    num_error = int(num_error)
    for calib in tqdm(
        online_calibs, desc="Reading online IMU calibration data"
    ):
        for imuCalib in calib.imu_calibs:
            if imuCalib.get_label() == stream_label:
                # calib timestamp in microseconds
                # convert to nanoseconds and then quantize to milliseconds
                timestamp = int(calib.tracking_timestamp.total_seconds() * 1e9)
                quantized_timestamp = timestamp // num_error
                online_imu_calibs[quantized_timestamp] = imuCalib

    return online_imu_calibs


def get_imu_data_from_vrs(
    vrs_provider: data_provider.VrsDataProvider,
    mps_folder: Path | None = None,
) -> pycolmap.ImuMeasurements:
    """Get rectified IMU data from VRS file.
    If mps_folder is provided, use online calibration data
    from MPS folder. Otherwise, use device calibration from VRS file."""
    imu_timestamps = sorted(
        vrs_provider.get_timestamps_ns(
            StreamId(RIGHT_IMU_STREAM_ID), TimeDomain.DEVICE_TIME
        )
    )
    imu_stream_label = vrs_provider.get_label_from_stream_id(
        StreamId(RIGHT_IMU_STREAM_ID)
    )

    if mps_folder is not None:
        online_calibs_file = mps_folder / "slam" / "online_calibration.jsonl"
        online_imu_calibs = get_online_params_for_imu_from_mps(
            online_calibs_file, imu_stream_label
        )
        acceptable_diff_ms = 1  # 1 milliseconds
        calib_timestamps = sorted(online_imu_calibs.keys())
    else:
        device_calib = vrs_provider.get_device_calibration()
        calibration = device_calib.get_imu_calib(imu_stream_label)

    ms = pycolmap.ImuMeasurements()
    for timestamp in tqdm(imu_timestamps, desc="Loading rect IMU data"):
        if mps_folder is not None:
            quantized_timestamp = timestamp // int(1e6)
            closest_ts = find_closest_timestamp(
                calib_timestamps, quantized_timestamp, acceptable_diff_ms
            )

            if closest_ts not in online_imu_calibs:
                raise ValueError(
                    f"No calibration found for timestamp {timestamp}"
                )

            calibration = online_imu_calibs[closest_ts]

        imu_data = vrs_provider.get_imu_data_by_time_ns(
            StreamId(RIGHT_IMU_STREAM_ID),
            timestamp,
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )
        if imu_data.accel_valid and imu_data.gyro_valid:
            rectified_acc = calibration.raw_to_rectified_accel(
                imu_data.accel_msec2
            )
            rectified_gyro = calibration.raw_to_rectified_gyro(
                imu_data.gyro_radsec
            )
            ts = float(timestamp) / 1e9  # convert to seconds
            ms.insert(
                pycolmap.ImuMeasurement(ts, rectified_acc, rectified_gyro)
            )

    return ms


# ----- VRS utils ----- #


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
        shutil.rmtree(output_dir)

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
