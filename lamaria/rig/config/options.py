from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
import pycolmap
from omegaconf import OmegaConf

from projectaria_tools.core.stream_id import StreamId


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
    
    pairs_file: Optional[Path] = None
    tri_model: Optional[Path] = None

    optim_model: Optional[Path] = None

    @classmethod
    def load(cls, cfg: OmegaConf) -> PathOptions:
        pass

# General options
@dataclass(frozen=True, slots=True)
class MPSOptions:
    use_mps: bool = False
    use_online_calibration: bool = False # when use_mps is true (for online calib file)

    @classmethod
    def load(cls, cfg: OmegaConf) -> MPSOptions:
        return MPSOptions(
            use_mps=cfg.get("use_mps", False),
            use_online_calibration=cfg.get("use_online_calibration", False),
        )

@dataclass(frozen=True, slots=True)
class SensorOptions:
    left_cam_stream_id: StreamId = StreamId("1201-1")
    right_cam_stream_id: StreamId = StreamId("1201-2")
    right_imu_stream_id: StreamId = StreamId("1202-1")
    camera_model: str = "RAD_TAN_THIN_PRISM_FISHEYE"

    @classmethod
    def load(cls, cfg: OmegaConf) -> SensorOptions:
        pass

# To COLMAP options
@dataclass(frozen=True, slots=True)
class ToColmapOptions:
    paths: PathOptions = field(default_factory=PathOptions)
    mps_opts: MPSOptions = field(default_factory=MPSOptions)
    sensor_opts: SensorOptions = field(default_factory=SensorOptions)

    @classmethod
    def load(cls, cfg: OmegaConf) -> ToColmapOptions:
        pass

# Keyframing options
@dataclass(frozen=True, slots=True)
class KeyframeSelectorOptions:
    paths: PathOptions = field(default_factory=PathOptions)

    max_rotation: float = 20.0 # degrees
    max_translation: float = 1.0 # meters
    max_elapsed_time: float = 1e9 # 1 second in ns

    @classmethod
    def load(cls, cfg: OmegaConf) -> KeyframeSelectorOptions:
        pass

# Triangulation options
@dataclass(frozen=True, slots=True)
class TriangulatorOptions:
    paths: PathOptions = field(default_factory=PathOptions)

    pairs_path: Optional[Path] = None
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

    cam_opts: OptCamOptions = field(default_factory=OptCamOptions)
    imu_opts: OptIMUOptions = field(default_factory=OptIMUOptions)
    optim_opts: OptOptions = field(default_factory=OptOptions)

    colmap_pipeline_opts: pycolmap.IncrementalPipelineOptions = \
        pycolmap.IncrementalPipelineOptions()
    