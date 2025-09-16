

from dataclasses import dataclass

@dataclass(frozen=True)
class CamParams:
    feature_std: float = 1.0 # in pixels
    optimize_cam_intrinsics: bool = False
    optimize_cam_from_rig: bool = False

    @classmethod
    def from_cfg(cls, cfg) -> "CamParams":
        """
        Accepts the full OmegaConf cfg and returns camera parameters.
        """
        opt = getattr(cfg, "optimization", cfg)

        return cls(
            feature_std=float(opt.feature_std),
            optimize_cam_intrinsics=bool(opt.optimize_cam_intrinsics),
            optimize_cam_from_rig=bool(opt.optimize_cam_from_rig),
        )


@dataclass(frozen=True)
class IMUParams:
    gyro_infl: float = 1.0
    acc_infl: float = 1.0
    integration_noise_density: float = 0.05

    optimize_scale: bool = False
    optimize_gravity: bool = False
    optimize_imu_from_rig: bool = False
    optimize_bias: bool = False
    keep_imu_residuals: bool = True

    @classmethod
    def from_cfg(cls, cfg) -> "IMUParams":
        """
        Accepts the full OmegaConf cfg and returns IMU parameters.
        """
        opt = getattr(cfg, "optimization", cfg)

        return cls(
            gyro_infl=float(opt.imu_inflation_factor_gyro),
            acc_infl=float(opt.imu_inflation_factor_acc),
            integration_noise_density=float(opt.integration_noise_density),
            optimize_scale=bool(opt.optimize_scale),
            optimize_gravity=bool(opt.optimize_gravity),
            optimize_imu_from_rig=bool(opt.optimize_imu_from_rig),
            optimize_bias=bool(opt.optimize_bias),
            keep_imu_residuals=bool(opt.keep_imu_residuals),
        )


@dataclass(frozen=True)
class OptParams:
    use_callback: bool = True
    max_num_iterations: int = 10
    normalize_reconstruction: bool = False
    use_device_calibration: bool = True

    @classmethod
    def from_cfg(cls, cfg) -> "OptParams":
        """
        Accepts the full OmegaConf cfg and returns optimization parameters.
        """
        opt = getattr(cfg, "optimization", cfg)

        return cls(
            use_callback=bool(opt.use_callback),
            max_num_iterations=int(opt.max_num_iterations),
            normalize_reconstruction=bool(opt.normalize_reconstruction),
            use_device_calibration=bool(opt.use_device_calibration),
        )