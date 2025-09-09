from pathlib import Path
from typing import Optional, Sequence
from omegaconf import OmegaConf

def _merge_if_exists(cfg, path: Path):
    return OmegaConf.merge(cfg, OmegaConf.load(str(path))) if path.exists() else cfg

def load_cfg(
    base_file: str = "lamaria/rig/config/defaults.yaml",
    local_file: Optional[str] = None,
    cli_overrides: Optional[Sequence[str]] = None,
):
    cfg = OmegaConf.load(base_file)
    if local_file:
        cfg = _merge_if_exists(cfg, Path(local_file))
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(cli_overrides)))
    OmegaConf.resolve(cfg)

    vrs_stem = Path(cfg.run.vrs_file).stem

    base = Path(cfg.paths.base)
    recordings = Path(cfg.paths.recordings)
    mps = Path(cfg.paths.mps)

    parent_rel = cfg.vio.parent_path

    cfg.vrs_file_path = str(recordings / cfg.run.vrs_file)
    cfg.mps_slam_path = str(mps / f"mps_{vrs_stem}_vrs" / "slam")
    cfg.image_stream_path = str(base / parent_rel / vrs_stem / "image_stream")
    
    vio_base = base / parent_rel / vrs_stem / cfg.vio.subfolder

    cfg.vio.parent_path = vio_base
    cfg.vio.keyframes_path = vio_base / cfg.vio.keyframes_subdir
    cfg.vio.output_folder_path = vio_base / cfg.vio.output_folder

    return cfg
