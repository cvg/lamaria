from typing import Dict

import numpy as np
import pycolmap
from tqdm import tqdm

from ...config.options import OptIMUOptions
from ..lamaria_reconstruction import LamariaReconstruction


def load_preintegrated_imu_measurements(
    options: OptIMUOptions,
    data: LamariaReconstruction,
) -> Dict[int, pycolmap.PreintegratedImuMeasurement]:
    preintegrated_measurements = {}

    imu_measurements = data.imu_measurements

    colmap_imu_opts = pycolmap.ImuPreintegrationOptions()
    colmap_imu_opts.integration_noise_density = (
        options.integration_noise_density
    )
    imu_calib = load_imu_calibration(options.gyro_infl, options.acc_infl)

    frame_ids = sorted(data.reconstruction.frames.keys())
    timestamps = [data.timestamps[fid] for fid in frame_ids]

    assert len(frame_ids) >= 2, (
        "Need at least two frames to compute preinteg measurements"
    )
    ts_sec = np.asarray(timestamps, dtype=np.float64) / 1e9

    for k in tqdm(
        range(len(frame_ids) - 1),  # skip the last frame
        desc="Loading preintegrated measurements",
    ):
        i = frame_ids[k]
        t1, t2 = ts_sec[k], ts_sec[k + 1]
        ms = imu_measurements.get_measurements_contain_edge(t1, t2)
        if len(ms) == 0:
            continue
        integrated_m = pycolmap.PreintegratedImuMeasurement(
            colmap_imu_opts,
            imu_calib,
            t1,
            t2,
        )
        integrated_m.add_measurements(ms)
        preintegrated_measurements[i] = integrated_m

    return preintegrated_measurements


def load_imu_states(
    data: LamariaReconstruction,
) -> Dict[int, pycolmap.ImuState]:
    imu_states = {}

    frame_ids = sorted(data.reconstruction.frames.keys())
    timestamps = [data.timestamps[fid] for fid in frame_ids]

    assert len(frame_ids) >= 2, "Need at least two frames to compute imu states"

    ts_sec = np.asarray(timestamps, dtype=np.float64) / 1e9

    for k in tqdm(range(len(frame_ids) - 1), desc="Loading IMU states"):
        i = frame_ids[k]
        j = frame_ids[k + 1]

        t1, t2 = ts_sec[k], ts_sec[k + 1]
        dt = t2 - t1
        if dt == 0:
            raise ValueError(f"Zero dt between frames {i} and {j}")

        pi = data.reconstruction.frames[i].rig_from_world.inverse().translation
        pj = data.reconstruction.frames[j].rig_from_world.inverse().translation

        vel = (pj - pi) / dt
        imu_states[i] = pycolmap.ImuState()
        imu_states[i].set_velocity(vel)

    imu_states[frame_ids[-1]] = pycolmap.ImuState()
    vel = imu_states[frame_ids[-2]].velocity
    imu_states[frame_ids[-1]].set_velocity(vel)

    return imu_states


def load_imu_calibration(
    gyro_infl: float = 1.0, acc_infl: float = 1.0
) -> pycolmap.ImuCalibration:
    """Load Aria IMU calibration parameters.
    Here: https://facebookresearch.github.io/projectaria_tools/docs/tech_insights"""

    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = 9.80600
    imu_calib.acc_noise_density = (
        0.8e-4 * imu_calib.gravity_magnitude * acc_infl
    )
    imu_calib.gyro_noise_density = 1e-2 * (np.pi / 180.0) * gyro_infl
    imu_calib.acc_bias_random_walk_sigma = (
        3.5e-5 * imu_calib.gravity_magnitude * np.sqrt(353) * acc_infl
    )  # accel_right BW = 353
    imu_calib.gyro_bias_random_walk_sigma = (
        1.3e-3 * (np.pi / 180.0) * np.sqrt(116) * gyro_infl
    )  # gyro_right BW = 116
    imu_calib.acc_saturation_max = 8.0 * imu_calib.gravity_magnitude
    imu_calib.gyro_saturation_max = 1000.0 * (np.pi / 180.0)
    imu_calib.imu_rate = 1000.0

    return imu_calib
