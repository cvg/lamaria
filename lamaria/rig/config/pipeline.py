from pathlib import Path
from typing import Optional, Sequence, Any
from omegaconf import OmegaConf, DictConfig
from dataclasses import replace

from .options import (
    EstimateToColmapOptions,
    VIOptimizerOptions,
    TriangulatorOptions,
    KeyframeSelectorOptions,
)

class PipelineOptions:
    default_cfg: Path = Path("defaults.yaml")
    workspace: Path = Path("/media/lamaria/")
    seq_name: str = "5cp"
    
    def __init__(self) -> None:
        self._cfg: DictConfig = self._load_cfg(self.default_cfg, None)
    
    @classmethod
    def _from_cfg(cls, cfg: DictConfig) -> "PipelineOptions":
        obj = cls.__new__(cls)
        obj._cfg = cfg
        return obj

    @staticmethod
    def _load_cfg(
        cfg_file: Path | str,
        cli_overrides: Optional[Sequence[str]],
    ) -> DictConfig:
        cfg = OmegaConf.load(str(cfg_file))
        if cli_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(cli_overrides)))
        OmegaConf.resolve(cfg)
        return cfg

    @classmethod
    def load(
        cls,
        base_file: Path | str | None = None,
        cli_overrides: Optional[Sequence[str]] = None,
    ) -> "PipelineOptions":
        cfg_path = base_file if base_file is not None else cls.default_cfg
        return cls._from_cfg(cls._load_cfg(cfg_path, cli_overrides))

    @property
    def config(self) -> DictConfig:
        return self._cfg

    @staticmethod
    def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
        val = OmegaConf.select(cfg, key)
        return default if val is None else val

    @property
    def workspace_path(self) -> Path:
        return Path(self._select(self._cfg, "workspace", Path(type(self).workspace)))

    @property
    def seq_name(self) -> str:
        return str(self._select(self._cfg, "seq_name", type(self).seq_name))
    
    @property
    def output_path(self) -> Path:
        return self.workspace / "output" / self.seq_name
    
    def get_estimate_to_colmap_options(self) -> EstimateToColmapOptions:
        """Get EstimateToColmapOptions from config."""
        options = EstimateToColmapOptions.load(
            self._cfg.estimate_to_colmap,
            self._cfg.mps,
            self._cfg.sensor,
        )
        return replace(
            options,
            vrs=self.workspace_path / options.vrs,
            estimate=self.workspace_path / options.estimate,
            images=self.output_path / options.images,
            colmap_model=self.output_path / options.colmap_model,
            output_path=self.output_path,
            mps_folder=(self.workspace_path / options.mps_folder) if options.mps_folder else None,
        )
    
    def get_keyframing_options(self) -> KeyframeSelectorOptions:
        """Get KeyframeSelectorOptions from config."""
        options = KeyframeSelectorOptions.load(self._cfg.keyframing)
        return replace(
            options,
            keyframes=self.output_path / options.keyframes,
            kf_model=self.output_path / options.kf_model,
            output_path=self.output_path,
        )
    
    def get_triangulator_options(self) -> TriangulatorOptions:
        """Get TriangulatorOptions from config."""
        options = TriangulatorOptions.load(self._cfg.triangulation)
        return replace(
            options,
            hloc=self.output_path / options.hloc,
            pairs_file=self.output_path / options.hloc / options.pairs_file,
            tri_model=self.output_path / options.tri_model,
            output_path=self.output_path,
        )
    
    def get_vi_optimizer_options(self) -> VIOptimizerOptions:
        """Get VIOptimizerOptions from config."""
        options = VIOptimizerOptions.load(self._cfg.optimization)
        return replace(
            options,
            optim_model=self.output_path / options.optim.optim_model,
            output_path=self.output_path,
        )
        