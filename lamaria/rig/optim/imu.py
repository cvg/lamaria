import numpy as np
import pycolmap
import pyceres
from typing import List
from pathlib import Path

from tqdm import tqdm

from ... import logger
from .params import IMUParams
from .session import SingleSeqSession


def load_preintegrated_imu_measurements(
    rect_imu_data_npy: Path,
    reconstruction: pycolmap.Reconstruction,
    timestamps: List[int],  # must be a sorted list in ns
    params: IMUParams = IMUParams(),
) -> dict[int, pycolmap.PreintegratedImuMeasurement]:

    preintegrated_measurements = {}
    rect_imu_data = np.load(rect_imu_data_npy, allow_pickle=True)
    imu_measurements = pycolmap.ImuMeasurements(rect_imu_data.tolist())
    
    options = pycolmap.ImuPreintegrationOptions()
    options.integration_noise_density = params.integration_noise_density
    imu_calib = load_imu_calibration(params.gyro_infl, params.acc_infl)

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


class IMUResidualManager:
    """Handles IMU residual setup and constraints"""
    def __init__(self, session: SingleSeqSession):
        self.session = session
    
    def add_residuals(self, problem):
        """Add IMU residuals to the optimization problem"""
        loss = pyceres.TrivialLoss()
        
        frame_ids = sorted(self.session.preintegrated_imu_measurements.keys())
        
        for k in range(len(self.session.preintegrated_imu_measurements)):
            i = frame_ids[k]
            j = frame_ids[k + 1]
            frame_i = self.session.reconstruction.frames[i]
            frame_j = self.session.reconstruction.frames[j]
            i_from_world = frame_i.rig_from_world
            j_from_world = frame_j.rig_from_world

            integrated_m = self.session.preintegrated_imu_measurements[i]

            problem.add_residual_block(
                pycolmap.PreintegratedImuMeasurementCost(integrated_m),
                loss,
                [
                    self.session.imu_from_rig.rotation.quat,
                    self.session.imu_from_rig.translation,
                    self.session.log_scale,
                    self.session.gravity,
                    i_from_world.rotation.quat,
                    i_from_world.translation,
                    self.session.imu_states[i].data,
                    j_from_world.rotation.quat,
                    j_from_world.translation,
                    self.session.imu_states[j].data,
                ],
            )

        self._setup_manifolds_and_constraints(problem)
        logger.info("Added IMU residuals to the problem")
        return problem

    def _setup_manifolds_and_constraints(self, problem):
        """Setup manifolds and parameter constraints"""
        problem.set_manifold(self.session.gravity, pyceres.SphereManifold(3))
        problem.set_manifold(
            self.session.imu_from_rig.rotation.quat,
            pyceres.EigenQuaternionManifold()
        )
        
        # Apply optimization constraints based on configuration
        if not self.session.imu_params.optimize_scale:
            problem.set_parameter_block_constant(self.session.log_scale)
        if not self.session.imu_params.optimize_gravity:
            problem.set_parameter_block_constant(self.session.gravity)
        if not self.session.imu_params.optimize_imu_from_rig:
            problem.set_parameter_block_constant(self.session.imu_from_rig.rotation.quat)
            problem.set_parameter_block_constant(self.session.imu_from_rig.translation)
        if not self.session.imu_params.optimize_bias:
            constant_idxs = np.arange(3, 9)
            for frame_id in self.session.imu_states.keys():
                problem.set_manifold(
                    self.session.imu_states[frame_id].data,
                    pyceres.SubsetManifold(9, constant_idxs),
                )