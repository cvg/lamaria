from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from .timed_reconstruction import TimedReconstruction


@dataclass
class VIReconstruction(TimedReconstruction):
    """Visual–Inertial reconstruction: adds IMU measurements."""

    imu_measurements: pycolmap.ImuMeasurements | None = None

    @classmethod
    def read(cls, input_folder: Path) -> "VIReconstruction":
        # Load base data first
        base = super().read(input_folder)

        # Load IMU
        rectified_imu_data_npy = input_folder / "rectified_imu_data.npy"
        assert rectified_imu_data_npy.exists(), (
            f"Rectified IMU data file {rectified_imu_data_npy} does not exist"
        )
        rectified_imu_data = np.load(rectified_imu_data_npy, allow_pickle=True)
        imu_measurements = pycolmap.ImuMeasurements(rectified_imu_data.tolist())

        return cls(
            reconstruction=base.reconstruction,
            timestamps=base.timestamps,
            imu_measurements=imu_measurements,
        )

    def write(self, output_folder: Path) -> None:
        # Write base data first
        super().write_base(output_folder)

        # Write IMU data
        rectified_imu_data_npy = output_folder / "rectified_imu_data.npy"
        np.save(rectified_imu_data_npy, self.imu_measurements.data)
