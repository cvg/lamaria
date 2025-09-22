from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional
from pathlib import Path
import pycolmap
from omegaconf import OmegaConf, open_dict

from .helpers import _structured_merge_to_obj


# General options
@dataclass(frozen=True, slots=True)
class MPSOptions:
    use_mps: bool = False
    use_online_calibration: bool = False # when use_mps is true (for online calib file)
    has_slam_drops: bool = False # check vrs json metadata file for each sequence

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> MPSOptions:
        if cfg is None:
            return cls()
        
        return _structured_merge_to_obj(cls, cfg)

@dataclass(frozen=True, slots=True)
class SensorOptions:
    left_cam_stream_id: str = "1201-1"
    right_cam_stream_id: str = "1201-2"
    right_imu_stream_id: str = "1202-1"
    camera_model: str = "RAD_TAN_THIN_PRISM_FISHEYE"

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "SensorOptions":
        if cfg is None:
            return cls()
        
        obj: SensorOptions = _structured_merge_to_obj(cls, cfg)
        return obj

# Estimate to COLMAP options
@dataclass(frozen=True, slots=True)
class EstimateToColmapOptions:
    vrs: str = "recordings/xyz.vrs"
    estimate: str = "estimates/xyz.txt"
    images: str = "image_stream"
    mps_folder: str = "mps/mps_xyz_vrs" # necessary if use_mps is true
    
    mps: MPSOptions = field(default_factory=MPSOptions)
    sensor: SensorOptions = field(default_factory=SensorOptions)

    output_path: Path = Path("/output")

    @classmethod
    def load(
        cls, 
        cfg_estimate_to_colmap: Optional[OmegaConf] = None,
        cfg_mps: Optional[OmegaConf] = None,
        cfg_sensor: Optional[OmegaConf] = None,
    ) -> EstimateToColmapOptions:
        
        if cfg_estimate_to_colmap is None:
            return cls()
        
        base = _structured_merge_to_obj(cls, cfg_estimate_to_colmap)
        return replace(
            base,
            mps=MPSOptions.load(cfg_mps),
            sensor=SensorOptions.load(cfg_sensor)
        )
        

# Keyframing options
@dataclass(frozen=True, slots=True)
class KeyframeSelectorOptions:
    keyframes: str = "keyframes"
    kf_model: str = "keyframe_recon"

    max_rotation: float = 20.0 # degrees
    max_distance: float = 1.0 # meters
    max_elapsed: int = int(1e9) # 1 second in ns

    output_path: Path = Path("/output")

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "KeyframeSelectorOptions":
        if cfg is None:
            return cls()

        cfg = OmegaConf.create(cfg)
        with open_dict(cfg):
            if "max_elapsed" in cfg and isinstance(cfg.max_elapsed, float):
                cfg.max_elapsed = int(cfg.max_elapsed)
        
        obj: KeyframeSelectorOptions = _structured_merge_to_obj(cls, cfg)
        return obj


# Triangulation options
@dataclass(frozen=True, slots=True)
class TriangulatorOptions:
    hloc: str = "hloc"
    pairs_file: str = "pairs.txt"
    tri_model: str = "triangulated_recon"

    feature_conf: str = "aliked-n16"
    matcher_conf: str = "aliked+lightglue"
    retrieval_conf: str = "netvlad"
    num_retrieval_matches: int = 5

    # colmap defaults
    merge_max_reproj_error: float = 4.0
    complete_max_reproj_error: float = 4.0
    min_angle: float = 1.5

    filter_max_reproj_error: float = 4.0
    filter_min_tri_angle: float = 1.5

    output_path: Path = Path("/output")

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "TriangulatorOptions":
        if cfg is None:
            return cls()
        
        return _structured_merge_to_obj(cls, cfg)

# Optimization options
@dataclass(frozen=True, slots=True)
class OptCamOptions:
    feature_std: float = 1.0 # in pixels
    optimize_cam_intrinsics: bool = False
    optimize_cam_from_rig: bool = False

@dataclass(frozen=True, slots=True)
class OptIMUOptions:
    gyro_infl: float = 1.0
    acc_infl: float = 1.0
    integration_noise_density: float = 0.05

    optimize_scale: bool = False
    optimize_gravity: bool = False
    optimize_imu_from_rig: bool = False
    optimize_bias: bool = False

@dataclass(frozen=True, slots=True)
class OptOptions:
    optim_model: Path = Path("optim_recon")
    use_callback: bool = True
    max_num_iterations: int = 10
    normalize_reconstruction: bool = False

@dataclass(frozen=True, slots=True)
class VIOptimizerOptions:
    cam: OptCamOptions = field(default_factory=OptCamOptions)
    imu: OptIMUOptions = field(default_factory=OptIMUOptions)
    optim: OptOptions = field(default_factory=OptOptions)

    colmap_pipeline: pycolmap.IncrementalPipelineOptions = \
        pycolmap.IncrementalPipelineOptions()

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "VIOptimizerOptions":
        if cfg is None:
            return cls()
        
        base = cls()
        cam = _structured_merge_to_obj(OptCamOptions, cfg.optimization.cam)
        imu = _structured_merge_to_obj(OptIMUOptions, cfg.optimization.imu)
        optim = _structured_merge_to_obj(OptOptions, cfg.optimization.general)

        # leave colmap_pipeline as default
        return replace(
            base,
            cam=cam,
            imu=imu,
            optim=optim,
        )
    