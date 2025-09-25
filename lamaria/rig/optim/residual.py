import numpy as np
import pyceres
import pycolmap
from tqdm import tqdm

from .session import SingleSeqSession
from ...config.options import OptIMUOptions


class IMUResidualManager:
    """Handles IMU residual setup and constraints"""
    def __init__(
        self,
        imu_options: OptIMUOptions,
        session: SingleSeqSession
    ):
        self.imu_options = imu_options
        self.session = session

    @staticmethod
    def add(
        imu_options: OptIMUOptions,
        session: SingleSeqSession,
        problem,
    ):
        """Entry point for adding IMU residuals to the problem."""
        res_manager = IMUResidualManager(imu_options, session)
        return res_manager.add_residuals(problem)
    
    def add_residuals(self, problem):
        """Add IMU residuals to the optimization problem"""
        loss = pyceres.TrivialLoss()
        
        frame_ids = sorted(self.session.data.reconstruction.frames.keys())
        
        for k in tqdm(
            range(len(self.session.preintegrated_imu_measurements)),
            desc="Adding IMU residuals"
        ):
            i = frame_ids[k]
            j = frame_ids[k + 1]
            frame_i = self.session.data.reconstruction.frames[i]
            frame_j = self.session.data.reconstruction.frames[j]
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

        return problem

    def _setup_manifolds_and_constraints(self, problem):
        """Setup manifolds and parameter constraints"""
        problem.set_manifold(self.session.gravity, pyceres.SphereManifold(3))
        problem.set_manifold(
            self.session.imu_from_rig.rotation.quat,
            pyceres.EigenQuaternionManifold()
        )
        
        # Apply optimization constraints based on configuration
        if not self.imu_options.optimize_scale:
            problem.set_parameter_block_constant(self.session.log_scale)
        if not self.imu_options.optimize_gravity:
            problem.set_parameter_block_constant(self.session.gravity)
        if not self.imu_options.optimize_imu_from_rig:
            problem.set_parameter_block_constant(self.session.imu_from_rig.rotation.quat)
            problem.set_parameter_block_constant(self.session.imu_from_rig.translation)
        if not self.imu_options.optimize_bias:
            constant_idxs = np.arange(3, 9)
            for frame_id in self.session.imu_states.keys():
                problem.set_manifold(
                    self.session.imu_states[frame_id].data,
                    pyceres.SubsetManifold(9, constant_idxs),
                )