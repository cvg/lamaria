import numpy as np
import pycolmap
from typing import List
from pathlib import Path

from tqdm import tqdm


def load_preintegrated_imu_measurements(
    rect_imu_data_npy: Path,
    reconstruction: pycolmap.Reconstruction,
    timestamps: List[int],  # must be a sorted list in ns
    gyro_infl: float = 1.0,
    acc_infl: float = 1.0,
    integration_noise_density: float = 1.0
) -> dict[int, pycolmap.PreintegratedImuMeasurement]:

    preintegrated_measurements = {}
    rect_imu_data = np.load(rect_imu_data_npy, allow_pickle=True)
    imu_measurements = pycolmap.ImuMeasurements(rect_imu_data.tolist())
    
    options = pycolmap.ImuPreintegrationOptions()
    options.integration_noise_density = integration_noise_density
    imu_calib = load_imu_calibration(gyro_infl, acc_infl)

    frame_ids = sorted(reconstruction.frames.keys())
    assert len(timestamps) == len(frame_ids), "Unequal timestamps and frames for preinteg calc"
    assert len(frame_ids) >= 2, "Need at least two frames to compute preinteg measurements"
    ts_sec = np.asarray(timestamps, dtype=np.float64) / 1e9

    for k in tqdm(
        range(len(frame_ids) - 1), # skip the last frame
        desc="Loading preintegrated measurements"
    ):
        i = frame_ids[k]
        t1, t2 = ts_sec[k], ts_sec[k + 1]
        ms = imu_measurements.get_measurements_contain_edge(t1, t2)
        if len(ms) == 0:
            continue
        integrated_m = pycolmap.PreintegratedImuMeasurement(
            options,
            imu_calib,
            t1,
            t2,
        )
        integrated_m.add_measurements(ms)
        preintegrated_measurements[i] = integrated_m

    return preintegrated_measurements

def load_imu_states(
    reconstruction: pycolmap.Reconstruction,
    timestamps: List[int] # must be a sorted list in ns
) -> dict[int, pycolmap.ImuState]:
    imu_states = {} 

    frame_ids = sorted(reconstruction.frames.keys())
    assert len(timestamps) == len(frame_ids), "Unequal timestamps and frames for imu state calc"
    assert len(frame_ids) >= 2, "Need at least two frames to compute velocity"

    ts_sec = np.asarray(timestamps, dtype=np.float64) / 1e9

    for k in tqdm(
        range(len(frame_ids) - 1),
        desc="Loading IMU states"
    ):
        i = frame_ids[k]
        j = frame_ids[k + 1]
        
        t1, t2 = ts_sec[k], ts_sec[k + 1]
        dt = t2 - t1
        if dt == 0:
            raise ValueError(f"Zero dt between frames {i} and {j}")
        
        pi = reconstruction.frames[i].rig_from_world.inverse().translation
        pj = reconstruction.frames[j].rig_from_world.inverse().translation

        vel = (pj - pi) / dt
        s = pycolmap.ImuState()
        s.set_velocity(vel)
        imu_states[i] = s
    
    last_id = frame_ids[-1]
    prev_id = frame_ids[-2]
    last_state = pycolmap.ImuState()
    last_state.set_velocity(imu_states[prev_id].velocity)
    imu_states[last_id] = last_state

    return imu_states

def load_imu_calibration(
    gyro_infl: float = 1.0,
    acc_infl: float = 1.0
) -> pycolmap.ImuCalibration:

    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = 9.80600
    imu_calib.acc_noise_density = (
        0.8e-4 * imu_calib.gravity_magnitude * acc_infl
    )
    imu_calib.gyro_noise_density = (
        1e-2 * (np.pi / 180.0) * gyro_infl
    )
    imu_calib.acc_bias_random_walk_sigma = (
        3.5e-5
        * imu_calib.gravity_magnitude
        * np.sqrt(353)
        * acc_infl
    )  # accel_right BW = 353
    imu_calib.gyro_bias_random_walk_sigma = (
        1.3e-3 * (np.pi / 180.0) * np.sqrt(116) * gyro_infl
    )  # gyro_right BW = 116
    imu_calib.acc_saturation_max = 8.0 * imu_calib.gravity_magnitude
    imu_calib.gyro_saturation_max = 1000.0 * (np.pi / 180.0)
    imu_calib.imu_rate = 1000.0
    
    return imu_calib
