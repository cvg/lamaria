from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .options import (
    KeyframeSelectorOptions,
    TriangulatorOptions,
    VIOptimizerOptions,
)


@dataclass
class PipelineOptions:
    _keyframing_options: KeyframeSelectorOptions = field(
        default_factory=KeyframeSelectorOptions
    )
    _triangulator_options: TriangulatorOptions = field(
        default_factory=TriangulatorOptions
    )
    _vi_optimizer_options: VIOptimizerOptions = field(
        default_factory=VIOptimizerOptions
    )

    has_slam_drops: bool = False

    def load(
        self,
        yaml: Path | str,
        cli_overrides: Sequence[str] | None = None,
    ) -> None:
        """Load configuration from a YAML file and apply any overrides."""
        cfg = OmegaConf.load(str(yaml))
        if cli_overrides:
            cfg = OmegaConf.merge(
                cfg, OmegaConf.from_dotlist(list(cli_overrides))
            )
        OmegaConf.resolve(cfg)

        self._update_from_cfg(cfg)

    def _update_from_cfg(self, cfg: DictConfig) -> None:
        """Update object attributes from a config."""
        self._keyframing_options = KeyframeSelectorOptions.load(cfg.keyframing)
        self._triangulator_options = TriangulatorOptions.load(cfg.triangulation)
        self._vi_optimizer_options = VIOptimizerOptions.load(cfg.optimization)

        self.has_slam_drops = cfg.get("has_slam_drops", False)

    # Properties for keyframing
    @property
    def keyframing_options(self) -> KeyframeSelectorOptions:
        return self._keyframing_options

    # Properties for triangulation
    @property
    def triangulator_options(self) -> TriangulatorOptions:
        return self._triangulator_options

    # Properties for visual-inertial optimization
    @property
    def vi_optimizer_options(self) -> VIOptimizerOptions:
        return self._vi_optimizer_options
