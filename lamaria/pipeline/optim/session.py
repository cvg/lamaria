from __future__ import annotations

import numpy as np
import pycolmap

from ...config.options import OptIMUOptions
from ..lamaria_reconstruction import LamariaReconstruction
from .imu import (
    load_imu_states,
    preintegrate_imu_measurements,
)


class SingleSeqSession:
    def __init__(
        self,
        imu_options: OptIMUOptions,
        data: LamariaReconstruction,
    ):
        self.data = data
        self._init_imu_data(imu_options)

    def _init_imu_data(self, imu_options):
        self.preintegrated_imu_measurements = (
            preintegrate_imu_measurements(imu_options, self.data)
        )
        self.imu_states = load_imu_states(self.data)
        self.imu_from_rig = pycolmap.Rigid3d()
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.log_scale = np.array([0.0])
