from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class OptCamParams:
    feature_std: float = 1.0 # in pixels
    optimize_cam_intrinsics: bool = False
    optimize_cam_from_rig: bool = False


@dataclass(frozen=True, slots=True)
class OptIMUParams:
    gyro_infl: float = 1.0
    acc_infl: float = 1.0
    integration_noise_density: float = 0.05

    optimize_scale: bool = False
    optimize_gravity: bool = False
    optimize_imu_from_rig: bool = False
    optimize_bias: bool = False
    keep_imu_residuals: bool = True


@dataclass(frozen=True, slots=True)
class OptParams:
    use_callback: bool = True
    max_num_iterations: int = 10
    normalize_reconstruction: bool = False