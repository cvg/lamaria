from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
import pycolmap
from omegaconf import OmegaConf

from projectaria_tools.core.stream_id import StreamId
from .helpers import _p, _structured_merge_to_obj


@dataclass(slots=True)
class PathOptions:
    vrs: Optional[Path] = None
    estimate: Optional[Path] = None
    init_model: Optional[Path] = None
    images: Optional[Path] = None
    full_ts: Optional[Path] = None
    mps: Optional[Path] = None

    rect_imu: Optional[Path] = None

    keyframes: Optional[Path] = None
    kf_model: Optional[Path] = None
    kf_ts: Optional[Path] = None
    
    hloc: Optional[Path] = None
    pairs_file: Optional[Path] = None
    tri_model: Optional[Path] = None

    optim_model: Optional[Path] = None

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> PathOptions:
        """
        Build PathOptions from `paths:` block, resolving relative paths
        against `paths.base` and `paths.output.base`.
        Layout from defaults.yaml.
        """
        if cfg is None or not hasattr(cfg, 'paths'):
            return cls()

        cfg_paths = cfg.paths
        base_root = Path(cfg_paths.base).resolve()
        out_root  = _p(cfg_paths.output.base, base_root)

        vrs      = _p(cfg_paths.vrs, base_root)
        estimate = _p(cfg_paths.estimate, base_root)
        mps      = _p(cfg_paths.mps, base_root)

        images     = _p(cfg_paths.output.images, out_root)
        init_model = _p(cfg_paths.output.init_model, out_root)
        
        keyframes  = _p(cfg_paths.output.keyframes, out_root)
        kf_model   = _p(cfg_paths.output.kf_model, out_root)
        
        hloc     = _p(cfg_paths.output.hloc, out_root)
        pairs_file = _p(cfg_paths.output.pairs_file, hloc)
        
        tri_model  = _p(cfg_paths.output.tri_model, out_root)
        optim_model= _p(cfg_paths.output.optim_model, out_root)

        full_ts = _p(cfg_paths.output.full_ts, out_root)
        kf_ts   = _p(cfg_paths.output.kf_ts, out_root)
        rect_imu= _p(cfg_paths.output.rect_imu, out_root)

        return cls(
            vrs=vrs,
            estimate=estimate,
            init_model=init_model,
            images=images,
            full_ts=full_ts,
            mps=mps,
            rect_imu=rect_imu,
            keyframes=keyframes,
            kf_model=kf_model,
            kf_ts=kf_ts,
            hloc=hloc,
            pairs_file=pairs_file,
            tri_model=tri_model,
            optim_model=optim_model,
        )

# General options
@dataclass(frozen=True, slots=True)
class MPSOptions:
    use_mps: bool = False
    use_online_calibration: bool = False # when use_mps is true (for online calib file)

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> MPSOptions:
        if cfg is None or not hasattr(cfg, 'mps'):
            return cls()
        
        return _structured_merge_to_obj(cls, cfg.mps)

@dataclass(frozen=True, slots=True)
class SensorOptions:
    left_cam_stream_id: StreamId = StreamId("1201-1")
    right_cam_stream_id: StreamId = StreamId("1201-2")
    right_imu_stream_id: StreamId = StreamId("1202-1")
    camera_model: str = "RAD_TAN_THIN_PRISM_FISHEYE"

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> SensorOptions:
        if cfg is None or not hasattr(cfg, 'sensor'):
            return cls()
        
        obj: SensorOptions = _structured_merge_to_obj(cls, cfg.sensor)
        if not isinstance(obj.left_cam_stream_id, StreamId):
            object.__setattr__(obj, "left_cam_stream_id", StreamId(str(obj.left_cam_stream_id)))
        if not isinstance(obj.right_cam_stream_id, StreamId):
            object.__setattr__(obj, "right_cam_stream_id", StreamId(str(obj.right_cam_stream_id)))
        if not isinstance(obj.right_imu_stream_id, StreamId):
            object.__setattr__(obj, "right_imu_stream_id", StreamId(str(obj.right_imu_stream_id)))
        return obj

# To COLMAP options
@dataclass(frozen=True, slots=True)
class ToColmapOptions:
    paths: PathOptions = field(default_factory=PathOptions)
    mps: MPSOptions = field(default_factory=MPSOptions)
    sensor: SensorOptions = field(default_factory=SensorOptions)

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> ToColmapOptions:
        if cfg is None:
            return cls(
                paths=PathOptions(),
                mps=MPSOptions(),
                sensor=SensorOptions(),
            )
        
        return cls(
            paths=PathOptions.load(cfg),
            mps=MPSOptions.load(cfg),
            sensor=SensorOptions.load(cfg),
        )

# Keyframing options
@dataclass(frozen=True, slots=True)
class KeyframeSelectorOptions:
    paths: PathOptions = field(default_factory=PathOptions)

    max_rotation: float = 20.0 # degrees
    max_distance: float = 1.0 # meters
    max_elapsed: int = int(1e9) # 1 second in ns

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "KeyframeSelectorOptions":
        if cfg is None or not hasattr(cfg, 'keyframing'):
            return cls(paths=PathOptions())
        
        section = {
            "max_rotation": float(cfg.keyframing.max_rotation),
            "max_distance": float(cfg.keyframing.max_distance),
            "max_elapsed": int(float(cfg.keyframing.max_elapsed)),
        }
        obj: KeyframeSelectorOptions = _structured_merge_to_obj(cls, section)
        return replace(obj, paths=PathOptions.load(cfg))


# Triangulation options
@dataclass(frozen=True, slots=True)
class TriangulatorOptions:
    paths: PathOptions = field(default_factory=PathOptions)

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
            return cls(paths=PathOptions())
        
        obj: TriangulatorOptions = _structured_merge_to_obj(cls, cfg.triangulation)
        return replace(obj, paths=PathOptions.load(cfg))

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
    use_callback: bool = True
    max_num_iterations: int = 10
    normalize_reconstruction: bool = False

@dataclass(frozen=True, slots=True)
class VIOptimizerOptions:
    paths: PathOptions = field(default_factory=PathOptions)

    cam: OptCamOptions = field(default_factory=OptCamOptions)
    imu: OptIMUOptions = field(default_factory=OptIMUOptions)
    optim: OptOptions = field(default_factory=OptOptions)

    colmap_pipeline: pycolmap.IncrementalPipelineOptions = \
        pycolmap.IncrementalPipelineOptions()

    @classmethod
    def load(cls, cfg: Optional[OmegaConf] = None) -> "VIOptimizerOptions":
        if cfg is None or not hasattr(cfg, 'optimization'):
            return cls(paths=PathOptions())
        
        base: VIOptimizerOptions = OmegaConf.to_object(OmegaConf.structured(cls))
        opt = cfg.optimization
        cam = _structured_merge_to_obj(OptCamOptions, {
            "feature_std":            opt.feature_std,
            "optimize_cam_intrinsics": opt.optimize_cam_intrinsics,
            "optimize_cam_from_rig":   opt.optimize_cam_from_rig,
        })
        imu = _structured_merge_to_obj(OptIMUOptions, {
            "gyro_infl":                opt.gyro_infl,
            "acc_infl":                 opt.acc_infl,
            "integration_noise_density": opt.integration_noise_density,
            "optimize_scale":           opt.optimize_scale,
            "optimize_gravity":         opt.optimize_gravity,
            "optimize_imu_from_rig":    opt.optimize_imu_from_rig,
            "optimize_bias":            opt.optimize_bias,
        })
        optim = _structured_merge_to_obj(OptOptions, {
            "use_callback":        opt.use_callback,
            "max_num_iterations":  opt.max_num_iterations,
            "normalize_reconstruction": opt.normalize_reconstruction,
        })
        # leave colmap_pipeline as default
        return replace(
            base,
            paths=PathOptions.load(cfg),
            cam=cam,
            imu=imu,
            optim=optim,
        )
    