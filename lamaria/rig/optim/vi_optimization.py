import pycolmap
from pathlib import Path

from .session import SingleSeqSession
from .iterative_global_ba import IterativeRefinement
from ...config.options import VIOptimizerOptions, TriangulatorOptions


class VIOptimizer:
    """Main Visual-Inertial Optimizer"""
    
    def __init__(
        self,
        vi_options: VIOptimizerOptions,
        triangulator_options: TriangulatorOptions,
        session: SingleSeqSession
    ):
        self.vi_options = vi_options
        self.triangulator_options = triangulator_options
        self.session = session

    @staticmethod
    def optimize(
        vi_options: VIOptimizerOptions,
        triangulator_options: TriangulatorOptions,
        session: SingleSeqSession,
        database_path: Path
    ) -> pycolmap.Reconstruction:
        """Runs the complete VI optimization pipeline"""

        optimizer = VIOptimizer(
            vi_options,
            triangulator_options,
            session
        )

        optimized_recon = optimizer.process(database_path)

        return optimized_recon
    
    def process(self, database_path: Path) -> pycolmap.Reconstruction:
        """Optimize the reconstruction using VI optimization"""
        mapper = self._setup_incremental_mapper(database_path)
        pipeline_options = self._get_incremental_pipeline_options()

        refinement_strategy = IterativeRefinement(self.session)
        refinement_strategy.run(
            pipeline_options,
            mapper
        )

        return mapper.reconstruction

    def _setup_incremental_mapper(self, database_path: Path):
        """Setup the COLMAP incremental mapper"""
        database = pycolmap.Database.open(database_path)
        image_names = [self.session.data.reconstruction.images[image_id].name 
                      for image_id in self.session.data.reconstruction.reg_image_ids()]
        database_cache = pycolmap.DatabaseCache.create(
            database, 15, False, set(image_names)
        )
        mapper = pycolmap.IncrementalMapper(database_cache)
        mapper.begin_reconstruction(self.session.data.reconstruction)
        return mapper

    def _get_incremental_pipeline_options(self):
        """Get incremental pipeline options for COLMAP bundle adjustment"""
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.fix_existing_frames = False
        
        if not self.vi_options.cam.optimize_cam_from_rig:
            pipeline_options.ba_refine_sensor_from_rig = False
        if not self.vi_options.cam.optimize_focal_length:
            pipeline_options.ba_refine_focal_length = False
        if not self.vi_options.cam.optimize_principal_point:
            pipeline_options.ba_refine_principal_point = False
        if not self.vi_options.cam.optimize_extra_params:
            pipeline_options.ba_refine_extra_params = False

        # Setting up relaxed triangulation
        pipeline_options.triangulation.merge_max_reproj_error = self.triangulator_options.merge_max_reproj_error
        pipeline_options.triangulation.complete_max_reproj_error = self.triangulator_options.complete_max_reproj_error
        pipeline_options.triangulation.min_angle = self.triangulator_options.min_angle

        pipeline_options.mapper.filter_max_reproj_error = self.triangulator_options.filter_max_reproj_error
        pipeline_options.mapper.filter_min_tri_angle = self.triangulator_options.filter_min_tri_angle
        
        return pipeline_options