import json
from pathlib import Path

import numpy as np
import pycolmap
from projectaria_tools.core import mps
from projectaria_tools.core.calibration import CameraCalibration, ImuCalibration
from projectaria_tools.core.mps.utils import get_nearest_pose
from scipy.spatial.transform import Rotation


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
    """Get T_world_device for a list of device timestamps in nanoseconds using MPS trajectory data.
    Returns None for timestamps where no trajectory data is found."""
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
    """Get T_rig_world (rig is right IMU sensor) for a given device time in nanoseconds using MPS trajectory data and IMU calibration.
    Returns None if no trajectory data is found for the given timestamp."""
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
    return_qt=False,
    return_matrix=False,
):
    """Get T_imu_camera from Aria calibrations. Either return as qvec,tvec or 4x4 matrix.
    If neither return_qt or return_matrix is True, returns as projectaria_tools Rigid3d object."""

    t_device_cam = camera_calib.get_transform_device_camera()
    t_device_imu = imu_calib.get_transform_device_imu()
    t_imu_device = t_device_imu.inverse()

    t_imu_cam = t_imu_device @ t_device_cam

    assert not (return_qt and return_matrix), (
        "Only one of return_qt or return_matrix can be True"
    )

    if return_qt:
        qvec, tvec = get_qvec_and_tvec_from_transform(t_imu_cam)
        return qvec, tvec

    if return_matrix:
        t_imu_cam_matrix = t_imu_cam.to_matrix()
        return t_imu_cam_matrix

    return t_imu_cam


def get_qvec_and_tvec_from_transform(
    transform,
) -> tuple[np.ndarray, np.ndarray]:
    """Converts projectaria_tools Rigid3d to qvec and tvec.
    Returns qvec in format x,y,z,w and tvec in format x,y,z"""
    # to_quat() returns in wxyz format
    # https://github.com/facebookresearch/projectaria_tools/blob/867105e65cadbe777db355a407d90533c71d2d06/core/python/sophus/SO3PyBind.h#L161
    qvec = transform.rotation().to_quat()[0]
    tvec = transform.translation()[0]

    qvec = np.roll(
        qvec, -1
    )  # change from w,x,y,z to x,y,z,w for pycolmap format

    return np.array(qvec), np.array(tvec)


def rigid3d_from_transform(transform) -> pycolmap.Rigid3d:
    """Converts projectaria_tools Rigid3d to pycolmap Rigid3d"""
    q, t = get_qvec_and_tvec_from_transform(transform)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)


def get_magnitude_from_transform(
    transform: pycolmap.Rigid3d,
) -> tuple[float, float]:
    """Returns rotation (in degrees) and translation (in meters) magnitudes from a Rigid3d transform"""
    translation = transform.translation
    quat_xyzw = transform.rotation.quat
    rotation = Rotation.from_quat(quat_xyzw)
    dr = np.rad2deg(rotation.magnitude())
    dt = np.linalg.norm(translation)

    return dr, dt


def get_t_cam_a_cam_b_from_json(
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


def get_t_imu_camera_from_json(
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
