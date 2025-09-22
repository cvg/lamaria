from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional
import pycolmap
from omegaconf import OmegaConf

from .helpers import _structured_merge_to_obj


# General options
@dataclass(frozen=True, slots=True)
class MPSOptions:
    use_mps: bool = False
    use_online_calibration: bool = False # when use_mps is true (for online calib file)
    has_slam_drops: bool = False # check vrs json metadata file for each sequence

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> MPSOptions:
        if cfg is None or not hasattr(cfg, 'mps'):
            return cls()
        
        return _structured_merge_to_obj(cls, cfg.mps)

@dataclass(frozen=True, slots=True)
class SensorOptions:
    left_cam_stream_id: str = "1201-1"
    right_cam_stream_id: str = "1201-2"
    right_imu_stream_id: str = "1202-1"
    camera_model: str = "RAD_TAN_THIN_PRISM_FISHEYE"

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "SensorOptions":
        if cfg is None or not hasattr(cfg, 'sensor'):
            return cls()
        
        obj: SensorOptions = _structured_merge_to_obj(cls, cfg.sensor)
        return obj

# To COLMAP options
@dataclass(frozen=True, slots=True)
class EstimateToColmapOptions:
    images: str = "image_stream"
    mps: MPSOptions = field(default_factory=MPSOptions)
    sensor: SensorOptions = field(default_factory=SensorOptions)

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> EstimateToColmapOptions:
        if cfg is None:
            return cls(
                mps=MPSOptions(),
                sensor=SensorOptions(),
            )
        
        return cls(
            images=cfg.estimate_to_colmap.images,
            mps=MPSOptions.load(cfg),
            sensor=SensorOptions.load(cfg),
        )

# Keyframing options
@dataclass(frozen=True, slots=True)
class KeyframeSelectorOptions:
    keyframes: str = "keyframes"
    kf_model: str = "keyframe_recon"
    
    max_rotation: float = 20.0 # degrees
    max_distance: float = 1.0 # meters
    max_elapsed: int = int(1e9) # 1 second in ns

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "KeyframeSelectorOptions":
        if cfg is None or not hasattr(cfg, 'keyframing'):
            return cls()

        obj: KeyframeSelectorOptions = _structured_merge_to_obj(
            cls,
            cfg.keyframing
        )
        obj.replace(max_elapsed=int(obj.max_elapsed))
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

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "TriangulatorOptions":
        if cfg is None or not hasattr(cfg, 'triangulation'):
            return cls()
        
        obj: TriangulatorOptions = _structured_merge_to_obj(cls, cfg.triangulation)
        return obj

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
    optim_model: str = "optim_recon"
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
        if cfg is None or not hasattr(cfg, 'optimization'):
            return cls()
        
        base: VIOptimizerOptions = OmegaConf.to_object(OmegaConf.structured(cls))

        cam = _structured_merge_to_obj(OptCamOptions, cfg.optimization.cam)
        imu = _structured_merge_to_obj(OptIMUOptions, cfg.optimization.imu)
        optim = _structured_merge_to_obj(OptOptions, cfg.optimization.opt)

        # leave colmap_pipeline as default
        return replace(
            base,
            cam=cam,
            imu=imu,
            optim=optim,
        )
    