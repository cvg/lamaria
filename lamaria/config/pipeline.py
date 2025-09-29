from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .options import (
    EstimateToTimedReconOptions,
    KeyframeSelectorOptions,
    TriangulatorOptions,
    VIOptimizerOptions,
)


class PipelineOptions:
    def __init__(self) -> None:
        self._estimate_to_colmap_options: EstimateToTimedReconOptions = (
            EstimateToTimedReconOptions()
        )
        self._keyframing_options: KeyframeSelectorOptions = (
            KeyframeSelectorOptions()
        )
        self._triangulator_options: TriangulatorOptions = TriangulatorOptions()
        self._vi_optimizer_options: VIOptimizerOptions = VIOptimizerOptions()

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
        self._estimate_to_colmap_options = EstimateToTimedReconOptions.load(
            cfg.sensor,
        )
        self._keyframing_options = KeyframeSelectorOptions.load(cfg.keyframing)
        self._triangulator_options = TriangulatorOptions.load(cfg.triangulation)
        self._vi_optimizer_options = VIOptimizerOptions.load(cfg.optimization)

        out = cfg.get("output_path", None)
        self._output_path = Path(out) if out is not None else Path("/output/")

    # Properties for estimate to COLMAP
    @property
    def estimate_to_colmap_options(self) -> EstimateToTimedReconOptions:
        return self._estimate_to_colmap_options

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
