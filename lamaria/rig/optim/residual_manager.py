import numpy as np
import pyceres
import pycolmap
from copy import deepcopy
from typing import List, Tuple
import pycolmap.cost_functions
from tqdm import tqdm

from ... import logger
from .session import SingleSeqSession


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


class ImageResidualManager:
    """Handles image residual setup and constraints"""
    def __init__(
        self,
        options: pycolmap.BundleAdjustmentOptions,
        config: pycolmap.BundleAdjustmentConfig,
    ):
        self.options = options
        self.config = config
        self.problem = pyceres.Problem()
        self.summary = pyceres.SolverSummary()
        self.camera_ids = set()
        self.point3D_num_observations = dict()
        logger.info("BundleAdjuster initialized")

    def solve(self, reconstruction: pycolmap.Reconstruction):
        loss = self.options.create_loss_function()
        self.set_up_problem(reconstruction, loss)
        if self.problem.num_residuals() == 0:
            return False
        solver_options = self.set_up_solver_options(
            self.problem, self.options.solver_options
        )
        pyceres.solve(solver_options, self.problem, self.summary)
        return True

    def set_up_problem(
        self,
        session: SingleSeqSession,
        loss: pyceres.LossFunction,
    ):
        logger.info("Setting up optimization problem")
        assert session.reconstruction is not None
        for image_id in tqdm(self.config.image_ids):
            self.add_image_to_problem(session, image_id, loss)
        self.parameterize_cameras(session.reconstruction)
        self.parameterize_points(session.reconstruction)
        logger.info("Optimization problem set up")
        return self.problem

    def set_up_solver_options(
        self, problem: pyceres.Problem, solver_options: pyceres.SolverOptions
    ):
        bundle_adjuster = pycolmap.BundleAdjuster(self.options, self.config)
        return bundle_adjuster.set_up_solver_options(problem, solver_options)

    def add_image_to_problem(
        self,
        session: SingleSeqSession,
        image_id: int,
        loss: pyceres.LossFunction,
    ):
        image = session.reconstruction.images[image_id]
        frame = session.reconstruction.frames[image.frame_id]
        rig = session.reconstruction.rigs[frame.rig_id]
        camera = session.recon.cameras[image.camera_id]
        rig_from_world = frame.rig_from_world
        
        cam_from_rig = None
        for sensor, sensor_from_rig in rig.non_ref_sensors.items():
            if image.camera_id == sensor.id:
                cam_from_rig = sensor_from_rig

        optimize_cam_from_rig = session.cam_params.optimize_cam_from_rig
        optimize_cam_intrinsics = session.cam_params.optimize_cam_intrinsics
        feature_std = session.cam_params.feature_std
        
        num_observations = 0
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            num_observations += 1
            if point2D.point3D_id not in self.point3D_num_observations:
                self.point3D_num_observations[point2D.point3D_id] = 0
            self.point3D_num_observations[point2D.point3D_id] += 1
            
            point3D = session.recon.points3D[point2D.point3D_id]
            assert point3D.track.length() > 1
            if not optimize_cam_from_rig:
                cost = pycolmap.cost_functions.RigReprojErrorCost(
                    camera.model,
                    np.eye(2) * pow(feature_std, 2),
                    point2D.xy,
                    cam_from_rig,
                )
                self.problem.add_residual_block(
                    cost,
                    loss,
                    [
                        rig_from_world.rotation.quat,
                        rig_from_world.translation,
                        point3D.xyz,
                        camera.params,
                    ],
                )
                if not optimize_cam_intrinsics:
                    # do not optimise camera parameters
                    self.problem.set_parameter_block_constant(camera.params)
            else:
                cost = pycolmap.cost_functions.RigReprojErrorCost(
                    camera.model,
                    np.eye(2) * pow(feature_std, 2),
                    point2D.xy
                )
                self.problem.add_residual_block(
                    cost,
                    loss,
                    [
                        cam_from_rig.rotation.quat,
                        cam_from_rig.translation,
                        rig_from_world.rotation.quat,
                        rig_from_world.translation,
                        point3D.xyz,
                        camera.params,
                    ],
                )
                if not optimize_cam_intrinsics:
                    # do not optimise camera parameters
                    self.problem.set_parameter_block_constant(camera.params)

        if num_observations > 0:
            self.camera_ids.add(image.camera_id)
            # Set pose parameterization
            self.problem.set_manifold(
                rig_from_world.rotation.quat,
                pyceres.EigenQuaternionManifold(),
            )

            if optimize_cam_from_rig:
                self.problem.set_manifold(
                    cam_from_rig.rotation.quat,
                    pyceres.EigenQuaternionManifold(),
                )

    def parameterize_cameras(self, reconstruction: pycolmap.Reconstruction):
        constant_camera = (
            (not self.options.refine_focal_length)
            and (not self.options.refine_principal_point)
            and (not self.options.refine_extra_params)
        )
        for camera_id in self.camera_ids:
            camera = reconstruction.cameras[camera_id]
            if constant_camera or self.config.has_constant_cam_intrinsics(
                camera_id
            ):
                self.problem.set_parameter_block_constant(camera.params)
                continue
            const_camera_params = []
            if not self.options.refine_focal_length:
                const_camera_params.extend(camera.focal_length_idxs())
            if not self.options.refine_principal_point:
                const_camera_params.extend(camera.principal_point_idxs())
            if not self.options.refine_extra_params:
                const_camera_params.extend(camera.extra_point_idxs())
            if len(const_camera_params) > 0:
                self.problem.set_manifold(
                    camera.params,
                    pyceres.SubsetManifold(
                        len(camera.params), const_camera_params
                    ),
                )

    def parameterize_points(self, reconstruction: pycolmap.Reconstruction):
        for (
            point3D_id,
            num_observations,
        ) in self.point3D_num_observations.items():
            point3D = reconstruction.points3D[point3D_id]
            if point3D.track.length() > num_observations:
                self.problem.set_parameter_block_constant(point3D.xyz)
        for point3D_id in self.config.constant_point3D_ids:
            point3D = reconstruction.points3D[point3D_id]
            self.problem.set_parameter_block_constant(point3D.xyz)