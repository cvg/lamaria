from __future__ import annotations

from typing import List
from pathlib import Path
import pycolmap
import numpy as np

from .imu import (
    load_imu_states,
    load_preintegrated_imu_measurements,
)
from ..config.options import VIOptimizerOptions
from ..lamaria_reconstruction import LamariaReconstruction

class SingleSeqSession:
    def __init__(
        self,
        options: VIOptimizerOptions,
        data: LamariaReconstruction,
    ):
        self.data = data
        self._init_options(options)
        self._init_imu_data()

    def _init_imu_data(self):
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
        self.imu_from_rig = pycolmap.Rigid3d()
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.log_scale = np.array([0.0])

    def _init_params(self, cfg):
        self.cam_params = CamParams.from_cfg(cfg)
        self.imu_params = IMUParams.from_cfg(cfg)
        self.opt_params = OptParams.from_cfg(cfg)


