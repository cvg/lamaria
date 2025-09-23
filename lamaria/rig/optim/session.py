from __future__ import annotations
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
                self.imu_options,
                self.data
            )
        self.imu_states = load_imu_states(self.data)
        self.imu_from_rig = pycolmap.Rigid3d()
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.log_scale = np.array([0.0])

    def _init_options(self, options: VIOptimizerOptions):
        self.cam_options = options.cam
        self.imu_options = options.imu
        self.opt_options = options.optim


