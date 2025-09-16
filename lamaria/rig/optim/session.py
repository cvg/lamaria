from __future__ import annotations


from typing import List, Tuple
from pathlib import Path
import numpy as np
import pycolmap


from ..config.loaders import load_cfg
from .imu import (
    load_imu_states,
    load_preintegrated_imu_measurements,
)
from .params import (
    CamParams,
    OptParams,
    IMUParams
)

class SingleSeqSession:
    def __init__(
        self,
        reconstruction: pycolmap.Reconstruction,
        timestamps: List[int],
        rect_imu_data_npy: Path,
        cfg=None
    ):
        cfg = cfg if cfg is not None else load_cfg()
        self.reconstruction: pycolmap.Reconstruction = reconstruction
        self.timestamps: List[int] = timestamps
        self._init_params(cfg)
        self._init_imu_data(rect_imu_data_npy)

    def _init_imu_data(self, rect_imu_data_npy: Path):
        self.preintegrated_imu_measurements = \
            load_preintegrated_imu_measurements(
                rect_imu_data_npy,
                self.reconstruction,
                self.timestamps,
                self.imu_params
            )
        self.imu_states = load_imu_states(
            self.reconstruction,
            self.timestamps
        )

    def _init_params(self, cfg):
        self.cam_params = CamParams.from_cfg(cfg)
        self.imu_params = IMUParams.from_cfg(cfg)
        self.opt_params = OptParams.from_cfg(cfg)


