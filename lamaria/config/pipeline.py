from pathlib import Path
from typing import Optional, Sequence, Any
from omegaconf import OmegaConf, DictConfig
from dataclasses import replace, field

from .options import (
    EstimateToColmapOptions,
    VIOptimizerOptions,
    TriangulatorOptions,
    KeyframeSelectorOptions,
)

class PipelineOptions:
    output_path: Path = Path("/media/lamaria/output/")

    # this should be initialized at load(config)
    estimate_to_colmap_options: EstimateToColmapOptions = field(
        default_factory=EstimateToColmapOptions
    )
    keyframing_options: KeyframeSelectorOptions = field(
        default_factory=KeyframeSelectorOptions
    )
    triangulator_options: TriangulatorOptions = field(
        default_factory=TriangulatorOptions
    )
    vi_optimizer_options: VIOptimizerOptions = field(
        default_factory=VIOptimizerOptions
    )

    @classmethod
    def load(
        cls,
        yaml: Path | str,
        cli_overrides: Optional[Sequence[str]] = None,
    ) -> "PipelineOptions":
        cfg = OmegaConf.load(str(yaml))
        if cli_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(cli_overrides)))
        OmegaConf.resolve(cfg)
        return cls._from_cfg(cfg)
    
    @classmethod
    def _from_cfg(cls, cfg: DictConfig) -> "PipelineOptions":
        obj = cls.__new__(cls)
        obj.output_path = Path(cls._select(cfg, "output_path", obj.output_path))
        obj.estimate_to_colmap_options = EstimateToColmapOptions.load(
            cfg.mps,
            cfg.sensor,
        )
        obj.keyframing_options = KeyframeSelectorOptions.load(cfg.keyframing)
        obj.triangulator_options = TriangulatorOptions.load(cfg.triangulation)
        obj.vi_optimizer_options = VIOptimizerOptions.load(cfg.optimization)
        return obj

    @staticmethod
    def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
        val = OmegaConf.select(cfg, key)
        return default if val is None else val
    
    @property
    def output_path(self) -> Path:
        return self.output_path
    
    @property
    def estimate_to_colmap_options(self) -> EstimateToColmapOptions:
        return self.estimate_to_colmap_options
    
    @property
    def colmap_model(self) -> Path:
        return self.output_path / "initial_recon"
    
    @property
    def keyframing_options(self) -> KeyframeSelectorOptions:
        return self.keyframing_options
    
    @property
    def triangulator_options(self) -> TriangulatorOptions:
        return self.triangulator_options
    
    @property
    def vi_optimizer_options(self) -> VIOptimizerOptions:
        return self.vi_optimizer_options
        