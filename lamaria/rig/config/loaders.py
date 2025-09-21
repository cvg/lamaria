from pathlib import Path
from typing import Optional, Sequence, Tuple
from omegaconf import OmegaConf

from .options import (
    MPSOptions,
    PathOptions,
    SensorOptions,
    ToColmapOptions,
    VIOptimizerOptions,
    TriangulatorOptions,
    KeyframeSelectorOptions,
)

class Config:
    def __init__(self, cfg: OmegaConf):
        self.config = cfg
    
    @classmethod
    def load_default(
        cls,
    ) -> 'Config':
        """ Load default config from default file inside lamaria/rig/config/ """
        return cls.load()
    
    @classmethod
    def load(
        cls,
        base_file: str = "lamaria/rig/config/defaults.yaml",
        cli_overrides: Optional[Sequence[str]] = None
    ) -> 'Config':
        
        config = OmegaConf.load(base_file)
        if cli_overrides:
            config = OmegaConf.merge(
                config,
                OmegaConf.from_dotlist(list(cli_overrides))
            )
        
        OmegaConf.resolve(config)
        return cls(config)
    
    def get_paths(self) -> PathOptions:
        """ Get resolved paths from config. """
        return PathOptions.load(self.config)

    def get_mps_options(self) -> MPSOptions:
        """ Get MPS options from config. """
        return MPSOptions.load(self.config)

    def get_sensor_options(self) -> SensorOptions:
        """ Get sensor options from config. """
        return SensorOptions.load(self.config)

    def get_to_colmap_options(self) -> ToColmapOptions:
        """ Get options for to_colmap pipeline from config. """
        return ToColmapOptions.load(self.config)

    def get_keyframing_options(self) -> KeyframeSelectorOptions:
        """ Get keyframing options from config. """
        return KeyframeSelectorOptions.load(self.config)

    def get_triangulator_options(self) -> TriangulatorOptions:
        """ Get triangulation options from config. """
        return TriangulatorOptions.load(self.config)

    def get_vi_optimizer_options(self) -> VIOptimizerOptions:
        """ Get visual-inertial optimization options from config. """
        return VIOptimizerOptions.load(self.config)
        