from __future__ import annotations

from dataclasses import dataclass, field, replace

from omegaconf import OmegaConf, open_dict


def _structured_merge_to_obj(cls, section) -> object:
    """
    Merge a YAML section onto a structured
    config made from the dataclass `cls`,
    then return a dataclass instance.
    """
    base = OmegaConf.structured(cls)
    merged = OmegaConf.merge(base, section or {})
    return OmegaConf.to_object(merged)


# Keyframing options
@dataclass(slots=True)
class KeyframeSelectorOptions:
    max_rotation: float = 20.0  # degrees
    max_distance: float = 1.0  # meters
    max_elapsed: int = int(1e9)  # 1 second in ns

    @classmethod
    def load(cls, cfg: OmegaConf | None = None) -> KeyframeSelectorOptions:
        if cfg is None:
            return cls()

        cfg = OmegaConf.create(cfg)
        with open_dict(cfg):
            if "max_elapsed" in cfg and isinstance(cfg.max_elapsed, float):
                cfg.max_elapsed = int(cfg.max_elapsed)

        obj: KeyframeSelectorOptions = _structured_merge_to_obj(cls, cfg)
        return obj


# Triangulation options
@dataclass(slots=True)
class TriangulatorOptions:
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
    def load(cls, cfg: OmegaConf | None = None) -> TriangulatorOptions:
        if cfg is None:
            return cls()

        return _structured_merge_to_obj(cls, cfg)


# Optimization options
@dataclass(slots=True)
class OptCamOptions:
    feature_std: float = 1.0  # in pixels
    optimize_focal_length: bool = False
    optimize_principal_point: bool = False
    optimize_extra_params: bool = False
    optimize_cam_from_rig: bool = False


@dataclass(slots=True)
class OptIMUOptions:
    gyro_infl: float = 1.0
    acc_infl: float = 1.0
    integration_noise_density: float = 0.05

    optimize_scale: bool = False
    optimize_gravity: bool = False
    optimize_imu_from_rig: bool = False
    optimize_bias: bool = False


@dataclass(slots=True)
class OptOptions:
    use_callback: bool = True
    max_num_iterations: int = 10
    normalize_reconstruction: bool = False
    minimizer_progress_to_stdout: bool = True
    update_state_every_iteration: bool = True


@dataclass(slots=True)
class VIOptimizerOptions:
    cam: OptCamOptions = field(default_factory=OptCamOptions)
    imu: OptIMUOptions = field(default_factory=OptIMUOptions)
    optim: OptOptions = field(default_factory=OptOptions)
    use_mps_online_calibration: bool = False

    @classmethod
    def load(cls, cfg: OmegaConf | None = None) -> VIOptimizerOptions:
        if cfg is None:
            return cls()

        base = cls()
        cam = _structured_merge_to_obj(OptCamOptions, cfg.cam)
        imu = _structured_merge_to_obj(OptIMUOptions, cfg.imu)
        optim = _structured_merge_to_obj(OptOptions, cfg.general)

        return replace(
            base,
            cam=cam,
            imu=imu,
            optim=optim,
            use_mps_online_calibration=cfg.get(
                "use_mps_online_calibration", False
            ),
        )
