from pathlib import Path
from typing import Dict

import numpy as np
import pycolmap


class LamariaReconstruction:
    def __init__(self) -> None:
        self.reconstruction = pycolmap.Reconstruction()
        self.timestamps: dict[int, int] = {}
        self.imu_measurements = pycolmap.ImuMeasurements([])

    @staticmethod
    def read(
        input_folder: Path,
    ) -> "LamariaReconstruction":
        assert input_folder.exists(), (
            f"Input folder {input_folder} does not exist"
        )
        instance = LamariaReconstruction()
        instance.reconstruction = pycolmap.Reconstruction(input_folder)

        ts_path = input_folder / "timestamps.txt"
        assert ts_path.exists(), (
            f"Timestamps file {ts_path} does not exist in {input_folder}"
        )
        with open(ts_path) as f:
            lines = f.readlines()
            instance.timestamps = {}
            for line in lines:
                if line.startswith("#"):
                    continue
                frame_id, ts = line.strip().split()
                instance.timestamps[int(frame_id)] = int(ts)

        rectified_imu_data_npy = input_folder / "rectified_imu_data.npy"
        assert rectified_imu_data_npy.exists(), (
            f"Rectified IMU data file {rectified_imu_data_npy} \
                does not exist in {input_folder}"
        )
        rectified_imu_data = np.load(rectified_imu_data_npy, allow_pickle=True)
        instance.imu_measurements = pycolmap.ImuMeasurements(
            rectified_imu_data.tolist()
        )

        return instance

    def write(
        self,
        output_folder: Path,
    ) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)
        self.reconstruction.write(output_folder.as_posix())

        ts_path = output_folder / "timestamps.txt"
        frame_ids = sorted(self.timestamps.keys())

        # sanity check of frame ids in reconstruction and timestamps
        recon_frame_ids = np.array(sorted(self.reconstruction.frames.keys()))
        assert np.array_equal(np.array(frame_ids), recon_frame_ids), (
            "Frame IDs in reconstruction and timestamps do not match"
        )

        with open(ts_path, "w") as f:
            f.write("# FrameID Timestamp(ns)\n")
            for frame_id in frame_ids:
                f.write(f"{frame_id} {self.timestamps[frame_id]}\n")

        rectified_imu_data_npy = output_folder / "rectified_imu_data.npy"
        np.save(rectified_imu_data_npy, self.imu_measurements.data)
