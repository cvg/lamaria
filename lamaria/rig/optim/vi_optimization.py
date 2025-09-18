import pycolmap
from pathlib import Path

from ... import logger
from .session import SingleSeqSession
from .iterative_global_ba import IterativeRefinement


class VIOptimizer:
    """Main Visual-Inertial Optimizer"""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session
    
    def optimize(self, database_path: str, output_folder: Path):
        """Run the complete VI optimization pipeline"""
        
        # Setup mapper
        mapper = self._setup_incremental_mapper(database_path)
        pipeline_options = self._get_incremental_pipeline_options()
        
        # Run iterative refinement
        refinement_strategy = IterativeRefinement(self.session)
        refinement_strategy.run(
            mapper,
            pipeline_options,
        )
        
        # Save results
        optimized_reconstruction = mapper.reconstruction
        output_folder.mkdir(parents=True, exist_ok=True)
        optimized_reconstruction.write(output_folder.as_posix())
        
        return optimized_reconstruction

    def _setup_incremental_mapper(self, database_path: str):
        """Setup the incremental mapper"""
        database = pycolmap.Database.open(database_path)
        image_names = [self.session.reconstruction.images[image_id].name 
                      for image_id in self.session.reconstruction.reg_image_ids()]
        database_cache = pycolmap.DatabaseCache.create(
            database, 15, False, set(image_names)
        )
        mapper = pycolmap.IncrementalMapper(database_cache)
        mapper.begin_reconstruction(self.session.reconstruction)
        return mapper

    def _get_incremental_pipeline_options(self):
        """Get incremental pipeline options"""
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.fix_existing_images = False
        return pipeline_options


def create_vi_optimizer(session: SingleSeqSession) -> VIOptimizer:
    """Factory function to create VI optimizer"""
    return VIOptimizer(session)


def run_vi_optimization(session: SingleSeqSession, database_path: str, 
                       output_folder: Path = None):
    """Run VI optimization with given session and database"""
    optimizer = create_vi_optimizer(session)
    return optimizer.optimize(database_path, output_folder)