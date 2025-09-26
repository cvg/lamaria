import pycolmap
from pycolmap import logging
import pyceres
import numpy as np

from ... import logger
from .session import SingleSeqSession
from .residual import add_imu_residuals_to_problem
from .callback import RefinementCallback
from ...config.options import VIOptimizerOptions


def apply_constraints(problem, session: SingleSeqSession):
    """Apply rig constraints to the problem for fixing gauge freedom."""
    # Fix the first rig pose
    frame_ids = sorted(session.data.reconstruction.frames.keys())
    first_frame_rfw = session.data.reconstruction.frames[frame_ids[0]].rig_from_world
    problem.set_parameter_block_constant(first_frame_rfw.rotation.quat)
    problem.set_parameter_block_constant(first_frame_rfw.translation)

    # Fix 1 DoF translation of the second rig
    second_frame_rfw = session.data.reconstruction.frames[frame_ids[1]].rig_from_world
    problem.set_manifold(
        second_frame_rfw.translation,
        pyceres.SubsetManifold(3, np.array([0])),
    )
    return problem


class VIBundleAdjuster:
    """Visual-Inertial Bundle Adjuster class that 
    adds visual and IMU residuals to the 
    optimization problem."""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session

    @staticmethod
    def run(
        vi_options: VIOptimizerOptions,
        ba_options: pycolmap.BundleAdjustmentOptions,
        ba_config: pycolmap.BundleAdjustmentConfig,
        mapper: pycolmap.IncrementalMapper,
        session: SingleSeqSession,
    ):
        """Entry point for running VI bundle adjustment."""
        vi_ba = VIBundleAdjuster(session)
        vi_ba.solve(
            vi_options,
            ba_options,
            ba_config,
            mapper,
        )

    def solve(
        self,
        vi_options: VIOptimizerOptions,
        ba_options: pycolmap.BundleAdjustmentOptions,
        ba_config: pycolmap.BundleAdjustmentConfig,
        mapper: pycolmap.IncrementalMapper,
    ):
        """Solves the VI bundle adjustment problem."""
        bundle_adjuster = pycolmap.create_default_bundle_adjuster(
            ba_options,
            ba_config,
            mapper.reconstruction,
        )
        
        problem = bundle_adjuster.problem

        solver_options = ba_options.create_solver_options(
            ba_config,
            bundle_adjuster.problem
        )
        pyceres_solver_options = pyceres.SolverOptions(solver_options)
        
        problem = add_imu_residuals_to_problem(
            vi_options.imu,
            self.session,
            problem,
        )
        # problem = apply_constraints(problem, self.session)
        
        # Setup callback if needed
        if vi_options.optim.use_callback:
            callback = self._init_callback(mapper)
            pyceres_solver_options.callbacks = [callback]

        pyceres_solver_options.minimizer_progress_to_stdout = vi_options.optim.minimizer_progress_to_stdout
        pyceres_solver_options.update_state_every_iteration = vi_options.optim.update_state_every_iteration
        pyceres_solver_options.max_num_iterations = vi_options.optim.max_num_iterations
        
        # Solve
        summary = pyceres.SolverSummary()
        pyceres.solve(pyceres_solver_options, bundle_adjuster.problem, summary)
        print(summary.BriefReport())
    
    def _init_callback(self, mapper: pycolmap.IncrementalMapper):
        """Initialize the refinement callback to check pose changes."""
        frame_ids = sorted(mapper.reconstruction.frames.keys())
        poses = list(mapper.reconstruction.frames[frame_id].rig_from_world for frame_id in frame_ids)

        callback = RefinementCallback(poses)
        return callback


class GlobalBundleAdjustment:
    """Global bundle adjustment with visual-inertial constraints."""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session

    @staticmethod
    def run(
        vi_options: VIOptimizerOptions,
        pipeline_options: pycolmap.IncrementalPipelineOptions,
        mapper: pycolmap.IncrementalMapper,
        session: SingleSeqSession,
    ):
        """Entry point for running global bundle adjustment."""
        gba = GlobalBundleAdjustment(session)
        gba.adjust(
            vi_options,
            pipeline_options,
            mapper,
        )
    
    def adjust(
        self,
        vi_options: VIOptimizerOptions,
        pipeline_options: pycolmap.IncrementalPipelineOptions,
        mapper: pycolmap.IncrementalMapper,
    ):
        assert mapper.reconstruction is not None
        
        reg_image_ids = mapper.reconstruction.reg_image_ids()
        if len(reg_image_ids) < 2:
            logger.warning("At least two images must be registered for global bundle-adjustment")

        ba_options = self._configure_ba_options(
            pipeline_options,
            len(reg_image_ids)
        )

        # Avoid negative depths
        mapper.observation_manager.filter_observations_with_negative_depth()
        
        # Setting up bundle adjustment configuration
        ba_config = pycolmap.BundleAdjustmentConfig()
        for frame_id in mapper.reconstruction.reg_frame_ids():
            frame = mapper.reconstruction.frame(frame_id)
            for data_id in frame.data_ids:
                if data_id.sensor_id.type != pycolmap.SensorType.CAMERA:
                    continue
                ba_config.add_image(data_id.id)
        
        # Run bundle adjustment
        VIBundleAdjuster.run(
            vi_options,
            ba_options,
            ba_config,
            mapper,
            self.session,
        )
    
    def _configure_ba_options(self, pipeline_options, num_reg_images):
        """Configure bundle adjustment options based on number of registered images"""
        ba_options = pipeline_options.get_global_bundle_adjustment()
        ba_options.print_summary = True
        ba_options.refine_rig_from_world = True # Refine rig poses
        
        # Use stricter convergence criteria for first registered images
        if num_reg_images < 10:
            ba_options.solver_options.function_tolerance /= 10
            ba_options.solver_options.gradient_tolerance /= 10
            ba_options.solver_options.parameter_tolerance /= 10
            ba_options.solver_options.max_num_iterations *= 2
            ba_options.solver_options.max_linear_solver_iterations = 200

        return ba_options


class IterativeRefinement:
    """Iterative global refinement
    through repeated global bundle adjustments."""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session
    
    @staticmethod
    def run(
        vi_options: VIOptimizerOptions,
        pipeline_options: pycolmap.IncrementalPipelineOptions,
        mapper: pycolmap.IncrementalMapper,
        session: SingleSeqSession,
    ):
        """Entry point for running iterative refinement"""
        iter_refiner = IterativeRefinement(session)
        iter_refiner.refine(
            vi_options,
            pipeline_options,
            mapper,
        )

    def refine(
        self,
        vi_options: VIOptimizerOptions,
        pipeline_options: pycolmap.IncrementalPipelineOptions,
        mapper: pycolmap.IncrementalMapper,
    ):
        # Configure triangulation options
        tri_options = pipeline_options.get_triangulation()
        mapper.complete_and_merge_tracks(tri_options)
        num_retriangulated_observations = mapper.retriangulate(tri_options)
        logging.verbose(
            1, f"=> Retriangulated observations: {num_retriangulated_observations}"
        )

        # Configure mapper options
        mapper_options = pipeline_options.get_mapper()
        
        # Iterative refinement
        for i in range(pipeline_options.ba_global_max_refinements):
            num_observations = mapper.reconstruction.compute_num_observations()

            GlobalBundleAdjustment.run(
                vi_options,
                pipeline_options,
                mapper,
                self.session,
            )

            if vi_options.optim.normalize_reconstruction:
                mapper.reconstruction.normalize()
            
            num_changed_observations = mapper.complete_and_merge_tracks(tri_options)
            num_changed_observations += mapper.filter_points(mapper_options)
            changed = (num_changed_observations / num_observations 
                      if num_observations > 0 else 0)
            
            logging.verbose(1, f"=> Changed observations: {changed:.6f}")
            if changed < pipeline_options.ba_global_max_refinement_change:
                break