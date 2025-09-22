import numpy as np
import pycolmap
from typing import List
from pathlib import Path

from .. import logger

class LamariaReconstruction:
    reconstruction: pycolmap.Reconstruction
    timestamps: List[int]
    imu_measurements: pycolmap.ImuMeasurements

    @classmethod
    def read_reconstruction(
        cls,
        input_folder: Path,
    ) -> "LamariaReconstruction":
        assert input_folder.exists(), f"Input folder {input_folder} does not exist"
        instance = cls()
        instance.reconstruction = pycolmap.Reconstruction(input_folder)
        
        ts_path = input_folder / "timestamps.txt"
        assert ts_path.exists(), f"Timestamps file {ts_path} does not exist in {input_folder}"
        with open(ts_path, 'r') as f:
            instance.timestamps = [int(line.strip()) for line in f if line.strip().isdigit()]
        
        rectified_imu_data_npy = input_folder / "rectified_imu_data.npy"
        assert rectified_imu_data_npy.exists(), \
            f"Rectified IMU data file {rectified_imu_data_npy} \
                does not exist in {input_folder}"
        rectified_imu_data = np.load(rectified_imu_data_npy, allow_pickle=True)
        instance.imu_measurements = pycolmap.ImuMeasurements(rectified_imu_data.tolist())

        logger.info(f"Reconstruction read from {input_folder}")
        
        return instance

    def write_reconstruction(
        self,
        output_folder: Path,
    ) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)
        self.reconstruction.write(output_folder.as_posix())
        ts_path = output_folder / "timestamps.txt"
        with open(ts_path, 'w') as f:
            for ts in self.timestamps:
                f.write(f"{ts}\n")

        rectified_imu_data_npy = output_folder / "rectified_imu_data.npy"
        np.save(rectified_imu_data_npy, self.imu_measurements.data)
        logger.info(f"Reconstruction written to {output_folder}")