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
    vrs: Path = Path("recordings/xyz.vrs")
    estimate: Path = Path("estimates/xyz.txt")
    images: Path = Path("image_stream")
    colmap_model: Path = Path("initial_recon")

    mps: MPSOptions = field(default_factory=MPSOptions)
    sensor: SensorOptions = field(default_factory=SensorOptions)

    mps_folder: Optional[Path] = None # necessary if use_mps is true
    output_path: Optional[Path] = None

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

    def set_custom_paths(
        self,
        vrs: Path,
        estimate: Path,
        images: Path,
        colmap_model: Path,
        output_path: Optional[Path] = None,
        mps_folder: Optional[Path] = None,
    ) -> EstimateToColmapOptions:
        return replace(
            self,
            vrs=vrs,
            estimate=estimate,
            images=images,
            colmap_model=colmap_model,
            output_path=output_path if output_path is not None else self.output_path,
            mps_folder=mps_folder if mps_folder is not None else self.mps_folder
        )


# Keyframing options
@dataclass(frozen=True, slots=True)
class KeyframeSelectorOptions:
    keyframes: Path = Path("keyframes")
    kf_model: Path = Path("keyframe_recon")

    max_rotation: float = 20.0 # degrees
    max_distance: float = 1.0 # meters
    max_elapsed: int = int(1e9) # 1 second in ns

    output_path: Optional[Path] = None

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
    
    def set_custom_paths(
        self,
        keyframes: Path,
        kf_model: Path,
        output_path: Optional[Path] = None,
    ) -> KeyframeSelectorOptions:
        return replace(
            self,
            keyframes=keyframes,
            kf_model=kf_model,
            output_path=output_path if output_path is not None else self.output_path,
        )


# Triangulation options
@dataclass(frozen=True, slots=True)
class TriangulatorOptions:
    hloc: Path = Path("hloc")
    pairs_file: Path = Path("pairs.txt")
    tri_model: Path = Path("triangulated_recon")

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

    output_path: Optional[Path] = None

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "TriangulatorOptions":
        if cfg is None:
            return cls()
        
        return _structured_merge_to_obj(cls, cfg)
    
    def set_custom_paths(
        self,
        hloc: Path,
        pairs_file: Path,
        tri_model: Path,
        output_path: Optional[Path] = None,
    ) -> TriangulatorOptions:
        return replace(
            self,
            hloc=hloc,
            pairs_file=pairs_file,
            tri_model=tri_model,
            output_path=output_path if output_path is not None else self.output_path
        )
    

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
    
    output_path: Optional[Path] = None

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "VIOptimizerOptions":
        if cfg is None:
            return cls()
        
        base = cls()
        cam = _structured_merge_to_obj(OptCamOptions, cfg.cam)
        imu = _structured_merge_to_obj(OptIMUOptions, cfg.imu)
        optim = _structured_merge_to_obj(OptOptions, cfg.general)

        # leave colmap_pipeline as default
        return replace(
            base,
            cam=cam,
            imu=imu,
            optim=optim,
        )
    
    def set_custom_paths(
        self,
        optim_model: Path,
        output_path: Optional[Path] = None
    ) -> VIOptimizerOptions:
        return replace(
            self,
            optim=replace(self.optim, optim_model=optim_model),
            output_path=output_path if output_path is not None else self.output_path
        )
