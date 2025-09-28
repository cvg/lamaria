import json
import shutil
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from .. import logger
from ..utils.constants import (
    LEFT_CAMERA_STREAM_LABEL,
    RIGHT_CAMERA_STREAM_LABEL,
)
from ..utils.synchronization import (
    find_closest_timestamp,
)


def _round_ns(x: str | int | float) -> int:
    # works for "5120...274.999939", "5.12e11", or ints
    if isinstance(x, int):
        return x
    s = str(x)
    return int(Decimal(s).to_integral_value(rounding=ROUND_HALF_UP))


class Estimate:
    """
    Loads and stores an 'estimate' text file with rows:
      ts t_x t_y t_z q_x q_y q_z q_w
    Blank lines and lines starting with '#' are ignored.

    By default, poses are calculated as rig_from_world
    (i.e., inverse of world_from_rig) to satisfy COLMAP format.
    """

    def __init__(self) -> None:
        self.invert_poses = None
        self.corresponding_sensor = None
        self.path: Path | None = None
        self._timestamps: list[int] = []
        self._poses: list[pycolmap.Rigid3d] = []

    def load_from_file(
        self,
        path: str | Path,
        invert_poses: bool = True,
        corresponding_sensor: str = "imu",
    ) -> None:
        """Parse the file, validate format, populate timestamps & poses."""
        self.clear()
        self.path = Path(path)
        self.invert_poses = invert_poses
        self.corresponding_sensor = corresponding_sensor

        if not self.path.exists():
            raise FileNotFoundError(f"Estimate file not found: {self.path}")

        with open(self.path) as f:
            lines = f.readlines()

        state = self._parse(lines)
        if not state:
            raise RuntimeError("Failed to parse estimate file.")

    def add_estimate_poses_to_reconstruction(
        self,
        reconstruction: pycolmap.Reconstruction,
        cp_json_file: Path,
        sensor_from_rig: pycolmap.Rigid3d,
        output_path: Path,
    ) -> Path:
        """
        Adds estimate poses as frames to input reconstruction.
        """

        self._ensure_loaded()

        recon_path = output_path / "reconstruction"
        recon_path.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(recon_path)

        reconstruction = self._add_images_to_reconstruction(
            reconstruction, cp_json_file, sensor_from_rig
        )

        reconstruction.write(recon_path.as_posix())

        return recon_path

    @property
    def timestamps(self) -> list[int]:
        self._ensure_loaded()
        return self._timestamps

    @property
    def poses(self) -> list[pycolmap.Rigid3d]:
        self._ensure_loaded()
        return self._poses

    def as_tuples(self) -> list[tuple[int, pycolmap.Rigid3d]]:
        """Return a list of (timestamp, pose) tuples."""
        self._ensure_loaded()
        return list(zip(self._timestamps, self._poses, strict=True))

    def as_dict(self) -> dict[int, pycolmap.Rigid3d]:
        """Return a dict mapping timestamp to pose."""
        self._ensure_loaded()
        return dict(zip(self._timestamps, self._poses, strict=True))

    def __len__(self) -> int:
        return len(self._timestamps)

    def is_loaded(self) -> bool:
        return len(self._timestamps) > 0

    def clear(self) -> None:
        self._timestamps.clear()
        self._poses.clear()

    def _ensure_loaded(self) -> None:
        if not self._timestamps or not self._poses:
            raise RuntimeError(
                "Estimate not loaded. Call load_from_file() first."
            )

    # Parses the file lines, populating self._timestamps and self._poses
    def _parse(self, lines: list[str]) -> None:
        ts_list: list[int] = []
        pose_list: list[pycolmap.Rigid3d] = []
        exists_lines = False
        for lineno, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            exists_lines = True
            parts = line.split()
            if len(parts) != 8:
                logger.error(
                    f"Line {lineno}: expected 8 values, got {len(parts)}"
                )
                return False

            try:
                ts = _round_ns(parts[0])
                tvec = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
                qvec = np.array(
                    [
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    ]
                )

            except ValueError as e:
                logger.error(f"Line {lineno}: invalid number format: {e}")
                return False

            world_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), tvec)
            pose = (
                world_from_rig.inverse()
                if self.invert_poses
                else world_from_rig
            )

            ts_list.append(ts)
            pose_list.append(pose)

        if not exists_lines:
            logger.error("No valid lines found in the estimate file.")
            return False

        self._timestamps = ts_list
        self._poses = pose_list

    def _add_images_to_reconstruction(
        self,
        reconstruction: pycolmap.Reconstruction,
        cp_json_file: Path,
        sensor_from_rig: pycolmap.Rigid3d,
    ) -> pycolmap.Reconstruction:
        pose_data = self.as_tuples()

        with open(cp_json_file) as f:
            cp_data = json.load(f)

        image_id = 1
        rig = reconstruction.rig(rig_id=1)

        transform = sensor_from_rig.inverse()

        ts_data_processed = {}
        for label in [LEFT_CAMERA_STREAM_LABEL, RIGHT_CAMERA_STREAM_LABEL]:
            ts_data = cp_data["timestamps"][label]
            ts_data_processed[label] = {int(k): v for k, v in ts_data.items()}
            ts_data_processed[label]["sorted_keys"] = sorted(
                ts_data_processed[label].keys()
            )

        for i, (timestamp, pose) in tqdm(
            enumerate(pose_data),
            total=len(pose_data),
            desc="Adding images to reconstruction",
        ):
            if self.invert_poses:
                # poses are in imu/cam_from_world format
                T_world_rig = pose.inverse() * transform
            else:
                # poses are in world_from_cam/imu format
                T_world_rig = pose * transform

            frame = pycolmap.Frame()
            frame.rig_id = rig.rig_id
            frame.frame_id = i + 1
            frame.rig_from_world = T_world_rig.inverse()

            images_to_add = []

            for label, camera_id in [
                (LEFT_CAMERA_STREAM_LABEL, 1),
                (RIGHT_CAMERA_STREAM_LABEL, 2),
            ]:
                source_timestamps = ts_data_processed[label]["sorted_keys"]
                # offsets upto 1 ms (1e6 ns)
                closest_timestamp = find_closest_timestamp(
                    source_timestamps, timestamp, max_diff=1e6
                )
                if closest_timestamp is None:
                    raise ValueError

                image_name = ts_data_processed[label][closest_timestamp]

                im = pycolmap.Image(
                    image_name,
                    pycolmap.Point2DList(),
                    camera_id,
                    image_id,
                )
                im.frame_id = frame.frame_id
                frame.add_data_id(im.data_id)

                images_to_add.append(im)
                image_id += 1

            reconstruction.add_frame(frame)
            for im in images_to_add:
                reconstruction.add_image(im)

        return reconstruction
