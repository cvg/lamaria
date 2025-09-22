import pycolmap
from typing import List
from pathlib import Path

class LamariaReconstruction:
    reconstruction: pycolmap.Reconstruction
    timestamps: List[int]

    def read_reconstruction_with_timestamps(
        self,
        input_folder: Path,
    ) -> "LamariaReconstruction":
        assert input_folder.exists(), f"Input folder {input_folder} does not exist"
        self.reconstruction = pycolmap.Reconstruction(input_folder)
        ts_path = input_folder / "timestamps.txt"
        assert ts_path.exists(), f"Timestamps file {ts_path} does not exist in {input_folder}"
        with open(ts_path, 'r') as f:
            self.timestamps = [int(line.strip()) for line in f if line.strip().isdigit()]
        
        return self

    def write_reconstruction_with_timestamps(
        self,
        output_folder: Path,
    ) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)
        self.reconstruction.write(output_folder.as_posix())
        ts_path = output_folder / "timestamps.txt"
        with open(ts_path, 'w') as f:
            for ts in self.timestamps:
                f.write(f"{ts}\n")