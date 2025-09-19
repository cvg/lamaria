from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
import pycolmap

from omegaconf import OmegaConf

from projectaria_tools.core.stream_id import StreamId

# General options
@dataclass(frozen=True, slots=True)
class MPSOptions:
    use_mps: bool = False
    use_device_calibration: bool = True # when use_mps is true (for online calib file)

@dataclass(frozen=True, slots=True)
class SensorOptions:
    left_cam_stream_id: StreamId = StreamId("1201-1")
    right_cam_stream_id: StreamId = StreamId("1201-2")
    right_imu_stream_id: StreamId = StreamId("1202-1")
    camera_model: str = "RAD_TAN_THIN_PRISM_FISHEYE"

# Image extraction options
@dataclass(frozen=True, slots=True)
class ExtractionOptions:
    vrs_file: Optional[Path] = None
    images_dir: Optional[Path] = None

# To COLMAP options
@dataclass(frozen=True, slots=True)
class ToColmapOptions:
    vrs_file: Optional[Path] = None
    estimate_txt: Optional[Path] = None
    init_model: Optional[Path] = None
    images_dir: Optional[Path] = None
    timestamps_npy: Optional[Path] = None

    mps_dir: Optional[Path] = None
    mps_opts: MPSOptions = field(default_factory=MPSOptions)

    sensor_opts: SensorOptions = field(default_factory=SensorOptions)

    rect_imu_data_npy: Optional[Path] = None

# Keyframing options
@dataclass(frozen=True, slots=True)
class KFOptions:
    max_rotation: float = 20.0 # degrees
    max_translation: float = 1.0 # meters
    max_elapsed_time: float = 1e9 # 1 second in ns

@dataclass(frozen=True, slots=True)
class KeyframeSelectorOptions:
    options: KFOptions = field(default_factory=KFOptions)
    init_model: Optional[Path] = None # init model from ToCOLMAP
    keyframes_dir: Optional[Path] = None
    timestamps_npy: Optional[Path] = None
    kf_model: Optional[Path] = None

# Triangulation options
@dataclass(frozen=True, slots=True)
class TriOptions:
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
    

@dataclass(frozen=True, slots=True)
class TriangulatorOptions:
    options: TriOptions = field(default_factory=TriOptions)
    reference_model: Optional[Path] = None
    triangulated_model: Optional[Path] = None

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
    keep_imu_residuals: bool = True

@dataclass(frozen=True, slots=True)
class OptOptions:
    use_callback: bool = True
    max_num_iterations: int = 10
    normalize_reconstruction: bool = False

@dataclass(frozen=True, slots=True)
class VIOptimizerOptions:
    cam_opts: OptCamOptions = field(default_factory=OptCamOptions)
    imu_opts: OptIMUOptions = field(default_factory=OptIMUOptions)
    optim_opts: OptOptions = field(default_factory=OptOptions)

    colmap_pipeline_opts: pycolmap.IncrementalPipelineOptions = \
        pycolmap.IncrementalPipelineOptions()

    init_model: Optional[Path] = None
    timestamps_npy: Optional[Path] = None
    rect_imu_data_npy: Optional[Path] = None

    optimized_model: Optional[Path] = None
    