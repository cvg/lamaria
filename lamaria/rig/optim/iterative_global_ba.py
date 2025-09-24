import pycolmap
from pycolmap import logging
import pyceres
from typing import List, Tuple
from copy import deepcopy
import numpy as np

from ... import logger
from .session import SingleSeqSession
from .residual_manager import BundleAdjuster, IMUResidualManager


class RigConstraintManager:
    """Handles rig-specific constraints and problem setup"""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session
    
    def apply_constraints(self, problem):
        """Apply rig-specific constraints to the problem"""
        # Fix the first rig pose
        frame_ids = sorted(self.session.data.reconstruction.frames.keys())
        first_frame = self.session.data.reconstruction.frames[frame_ids[0]]
        problem.set_parameter_block_constant(first_frame.rotation.quat)
        problem.set_parameter_block_constant(first_frame.translation)

        # Fix 1 DoF translation of the second rig
        second_frame = self.session.data.reconstruction.frames[frame_ids[1]]
        problem.set_manifold(
            second_frame.translation,
            pyceres.SubsetManifold(3, np.array([0])),
        )
        return problem


class VIBundleAdjuster:
    """Visual-Inertial Bundle Adjuster that combines visual and IMU residuals"""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session

    def _init_callback(self):
        frame_ids = sorted(self.session.data.reconstruction.frames.keys())
        poses = list(self.session.data.reconstruction.frames[frame_id].rig_from_world for frame_id in frame_ids)
        callback = RefinementCallback(poses)

        return callback

    def solve(self, ba_options, ba_config):
        """Solve VI bundle adjustment problem"""
        bundle_adjuster = BundleAdjuster(ba_options, ba_config)
        imu_manager = IMUResidualManager(self.session)
        
        bundle_adjuster.set_up_problem(
            self.session,
            ba_options.create_loss_function(),
        )
        problem = bundle_adjuster.problem
        
        solver_options = ba_options.create_solver_options(
            ba_config, problem
        )
        pyceres_solver_options = pyceres.SolverOptions(solver_options)
        
        problem = imu_manager.add_residuals(problem)
        
        # Apply rig constraints
        constraint_manager = RigConstraintManager(self.session)
        problem = constraint_manager.apply_constraints(problem)
        logger.info("Constrained the rig problem")
        
        # Setup solver
        if self.session.opt_options.use_callback:
            callback = self._init_callback()
            pyceres_solver_options.callbacks = [callback]

        pyceres_solver_options.minimizer_progress_to_stdout = True
        pyceres_solver_options.update_state_every_iteration = True
        pyceres_solver_options.max_num_iterations = self.session.opt_options.max_num_iterations
        
        # Solve
        summary = pyceres.SolverSummary()
        pyceres.solve(pyceres_solver_options, problem, summary)
        print(summary.BriefReport())
        
        return summary, problem



class GlobalBundleAdjustment:
    """Strategy for global bundle adjustment with VI integration"""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session

    def adjust(self, mapper, pipeline_options):
        """Perform global bundle adjustment"""
        reconstruction = mapper.reconstruction
        assert reconstruction is not None
        
        reg_image_ids = reconstruction.reg_image_ids()
        if len(reg_image_ids) < 2:
            logger.warning("At least two images must be registered for global bundle-adjustment")

        ba_options = self._configure_ba_options(pipeline_options, len(reg_image_ids))

        # Avoid degeneracies
        mapper.observation_manager.filter_observations_with_negative_depth()
        
        # Configure bundle adjustment
        ba_config = pycolmap.BundleAdjustmentConfig()
        for frame_id in reconstruction.reg_frame_ids():
            frame = reconstruction.frame(frame_id)
            for data_id in frame.data_ids:
                if data_id.sensor_id.type != pycolmap.SensorType.CAMERA:
                    continue
                ba_config.add_image(data_id.id)
        
        # Run bundle adjustment
        vi_bundle_adjuster = VIBundleAdjuster(self.session)
        _, _ = vi_bundle_adjuster.solve(
            ba_options,
            ba_config
        )
    
    def _configure_ba_options(self, pipeline_options, num_reg_images):
        """Configure bundle adjustment options based on number of registered images"""
        ba_options = pipeline_options.get_global_bundle_adjustment()
        ba_options.print_summary = True
        ba_options.refine_focal_length = True
        ba_options.refine_extra_params = True
        
        # Use stricter convergence criteria for first registered images
        if num_reg_images < 10:
            ba_options.solver_options.function_tolerance /= 10
            ba_options.solver_options.gradient_tolerance /= 10
            ba_options.solver_options.parameter_tolerance /= 10
            ba_options.solver_options.max_num_iterations *= 2
            ba_options.solver_options.max_linear_solver_iterations = 200

        return ba_options


class IterativeRefinement:
    """Strategy for iterative global refinement"""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session

    def run(self, mapper, pipeline_options):
        """Run iterative global refinement"""
        reconstruction = mapper.reconstruction
        
        # Initial triangulation
        tri_options = pipeline_options.get_triangulation()
        mapper.complete_and_merge_tracks(tri_options)
        num_retriangulated_observations = mapper.retriangulate(tri_options)
        logging.verbose(
            1, f"=> Retriangulated observations: {num_retriangulated_observations}"
        )

        # Configure mapper options
        mapper_options = self._configure_mapper_options(pipeline_options)
        global_ba_strategy = GlobalBundleAdjustment(self.session)
        
        # Iterative refinement
        for i in range(pipeline_options.ba_global_max_refinements):
            num_observations = reconstruction.compute_num_observations()

            global_ba_strategy.adjust(
                mapper,
                pipeline_options,
            )
            
            if self.session.opt_options.normalize_reconstruction:
                reconstruction.normalize()
            
            # Check convergence
            num_changed_observations = mapper.complete_and_merge_tracks(tri_options)
            num_changed_observations += mapper.filter_points(mapper_options)
            changed = (num_changed_observations / num_observations 
                      if num_observations > 0 else 0)
            
            logging.verbose(1, f"=> Changed observations: {changed:.6f}")
            if changed < pipeline_options.ba_global_max_refinement_change:
                break
    
    def _configure_mapper_options(self, pipeline_options):
        """Configure mapper options"""
        mapper_options = pipeline_options.get_mapper()
        return mapper_options
    

class RefinementCallback(pyceres.IterationCallback):
    def __init__(
        self,
        poses: List[pycolmap.Rigid3d],
        min_pose_change: Tuple[float, float] = (0.001, 0.000001),
        min_iterations: int = 2,
    ):
        pyceres.IterationCallback.__init__(self)
        self.poses = poses
        self.poses_previous = deepcopy(self.poses)
        self.min_pose_change = min_pose_change
        self.min_iterations = min_iterations
        self.pose_changes = []

    def __call__(self, summary: pyceres.IterationSummary):
        if not summary.step_is_successful:
            return pyceres.CallbackReturnType.SOLVER_CONTINUE
        diff = []
        for pose_prev, pose in zip(self.poses_previous, self.poses):
            pose_rel = pose_prev * pose.inverse()
            q_rel, t_rel = pose_rel.rotation.quat, pose_rel.translation
            dr = np.rad2deg(
                np.abs(2 * np.arctan2(np.linalg.norm(q_rel[:-1]), q_rel[-1]))
            )
            dt = np.linalg.norm(t_rel)
            diff.append((dr, dt))
        diff = np.array(diff)
        self.poses_previous = deepcopy(self.poses)
        med, q99, max_ = np.quantile(diff, [0.5, 0.99, 1.0], axis=0)
        logger.info(
            f"{summary.iteration:d} Pose update: "
            f"med/q99/max dR={med[0]:.3f}/{q99[0]:.3f}/{max_[0]:.3f} deg, "
            f"dt={med[1] * 1e2:.3f}/{q99[1] * 1e2:.3f}/{max_[1] * 1e2:.3f} cm"
        )
        self.pose_changes.append((med, q99, max_))
        if summary.iteration >= self.min_iterations and np.all(
            q99 <= self.min_pose_change
        ):
            return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        return pyceres.CallbackReturnType.SOLVER_CONTINUE