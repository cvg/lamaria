import pycolmap
from pathlib import Path

from ..lamaria_reconstruction import LamariaReconstruction
from .session import SingleSeqSession
from .iterative_global_ba import IterativeRefinement


class VIOptimizer:
    """Main Visual-Inertial Optimizer"""
    
    def __init__(self, session: SingleSeqSession):
        self.session = session

    def optimize(self, database_path: Path):
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
        
        return mapper.reconstruction

    def _setup_incremental_mapper(self, database_path: Path):
        """Setup the incremental mapper"""
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
        """Get incremental pipeline options"""
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.fix_existing_images = False
        return pipeline_options


def create_vi_optimizer(session: SingleSeqSession) -> VIOptimizer:
    """Factory function to create VI optimizer"""
    return VIOptimizer(session)

def run(
    session: SingleSeqSession, 
    database_path: Path,
) -> pycolmap.Reconstruction:
    
    """Run VI optimization with given session and database"""
    optimizer = create_vi_optimizer(session)
    optimized_recon = optimizer.optimize(database_path)

    return optimized_recon