from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .options import (
    EstimateToLamariaOptions,
    KeyframeSelectorOptions,
    TriangulatorOptions,
    VIOptimizerOptions,
)


class PipelineOptions:
    def __init__(self) -> None:
        self._output_path: Path = Path("output/")
        self._estimate_to_colmap_options: EstimateToLamariaOptions = (
            EstimateToLamariaOptions()
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
        self._estimate_to_colmap_options = EstimateToLamariaOptions.load(
            cfg.mps,
            cfg.sensor,
        )
        self._keyframing_options = KeyframeSelectorOptions.load(cfg.keyframing)
        self._triangulator_options = TriangulatorOptions.load(cfg.triangulation)
        self._vi_optimizer_options = VIOptimizerOptions.load(cfg.optimization)

        out = cfg.get("output_path", None)
        self._output_path = Path(out) if out is not None else Path("/output/")

    @property
    def output_path(self) -> Path:
        """Get the parent output path."""
        return self._output_path

    @output_path.setter
    def output_path(self, path: Path) -> None:
        """Set the parent output path."""
        self._output_path = path

    # Properties for estimate to COLMAP
    @property
    def estimate_to_colmap_options(self) -> EstimateToLamariaOptions:
        return self._estimate_to_colmap_options

    @property
    def images(self) -> Path:
        return self._output_path / "images"

    @property
    def colmap_model(self) -> Path:
        return self._output_path / "initial_recon"

    # Properties for keyframing
    @property
    def keyframing_options(self) -> KeyframeSelectorOptions:
        return self._keyframing_options

    @property
    def keyframes_path(self) -> Path:
        return self._output_path / "keyframes"

    @property
    def kf_model(self) -> Path:
        return self._output_path / "keyframed_recon"

    # Properties for triangulation
    @property
    def triangulator_options(self) -> TriangulatorOptions:
        return self._triangulator_options

    @property
    def hloc(self) -> Path:
        return self._output_path / "hloc"

    @property
    def pairs_file(self) -> Path:
        return self._output_path / "hloc" / "pairs.txt"

    @property
    def tri_model_path(self) -> Path:
        return self._output_path / "triangulated_recon"

    # Properties for visual-inertial optimization
    @property
    def vi_optimizer_options(self) -> VIOptimizerOptions:
        return self._vi_optimizer_options

    @property
    def optim_model_path(self) -> Path:
        return self._output_path / "optim_recon"
