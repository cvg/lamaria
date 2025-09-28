import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from ..utils.aria import (
    add_cameras_to_reconstruction as _add_cams,
)
from ..utils.aria import (
    get_t_imu_camera_from_json,
)
from ..utils.constants import (
    LEFT_CAMERA_STREAM_LABEL,
    RIGHT_CAMERA_STREAM_LABEL,
)
from ..utils.general import (
    delete_files_in_folder,
    find_closest_timestamp,
)


def _round_ns(x: str | int | float) -> int:
    # works for "5120...274.999939", "5.12e11", or ints
    if isinstance(x, int):
        return x
    s = str(x)
    return int(Decimal(s).to_integral_value(rounding=ROUND_HALF_UP))


@dataclass(slots=True)
class _BaselineCfg:
    """Configuration for creating a baseline reconstruction."""

    cp_json_file: Path
    device_calibration_json: Path
    output_path: Path
    uses_imu: bool = True  # if False, uses monocular-cam0 poses


class Estimate:
    """
    Loads and stores an 'estimate' text file with rows:
      ts t_x t_y t_z q_x q_y q_z q_w
    Blank lines and lines starting with '#' are ignored.

    By default, poses are calculated as rig_from_world
    (i.e., inverse of world_from_rig) to satisfy COLMAP format.
    """

    def __init__(
        self, invert_poses: bool = True
    ) -> None:
        self.invert_poses = invert_poses
        self.path: Path | None = None
        self._timestamps: list[int] = []
        self._poses: list[pycolmap.Rigid3d] = []
        self._baseline_cfg: _BaselineCfg | None = None

    def load_from_file(self, path: str | Path) -> "Estimate":
        """Parse the file, validate format, populate timestamps & poses."""
        self.clear()
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Estimate file not found: {self.path}")

        with open(self.path) as f:
            lines = f.readlines()

        self._parse(lines)  # raises error if format is invalid
        return self

    def setup_baseline_cfg(
        self,
        cp_json_file: str | Path,
        device_calibration_json: str | Path,
        output_path: str | Path,
        uses_imu: bool,
    ) -> None:
        """
        Store config used by create_baseline_reconstruction().

        uses_imu:
        If True, the poses in the estimate file are IMU poses.
        If False, the poses are left camera poses (monocular-cam0).
        """

        self._baseline_cfg = _BaselineCfg(
            Path(cp_json_file),
            Path(device_calibration_json),
            Path(output_path),
            uses_imu,
        )

    def create_baseline_reconstruction(self) -> pycolmap.Reconstruction:
        """
        Build a COLMAP reconstruction from this Estimate and the
        parameters given via setup_baseline_cfg(). Writes to:
        output_path / "reconstruction"
        Returns pycolmap.Reconstruction.
        """
        self._ensure_loaded()
        if self._baseline_cfg is None:
            raise RuntimeError(
                "Baseline not configured. Call setup_baseline_cfg(...) first."
            )

        cfg = self._baseline_cfg
        recon_path = cfg.output_path / "reconstruction"
        recon_path.mkdir(parents=True, exist_ok=True)
        delete_files_in_folder(recon_path)

        reconstruction = pycolmap.Reconstruction()
        # Adds cameras and rig to the reconstruction
        _add_cams(reconstruction, cfg.device_calibration_json)
        reconstruction = self._add_images_to_reconstruction(
            cfg,
            reconstruction,
        )

        reconstruction.write(str(recon_path))
        return reconstruction

    @property
    def timestamps(self) -> list[int]:
        self._ensure_loaded()
        return self._timestamps

    @property
    def poses(self) -> list[pycolmap.Rigid3d]:
        self._ensure_loaded()
        return self._poses

    @property
    def reconstruction_path(self) -> Path | None:
        if self._baseline_cfg is None:
            return None
        return self._baseline_cfg.output_path / "reconstruction"

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
                raise ValueError(
                    f"{lineno}: expected 8 values, got {len(parts)}"
                )

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
                raise ValueError(
                    f"{lineno}: non-numeric value in {parts!r}"
                ) from e

            world_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), tvec)
            pose = (
                world_from_rig.inverse()
                if self.invert_poses
                else world_from_rig
            )

            ts_list.append(ts)
            pose_list.append(pose)

        if not exists_lines:
            raise ValueError("No valid lines found in estimate file.")

        self._timestamps = ts_list
        self._poses = pose_list

    def _add_images_to_reconstruction(
        self,
        cfg: _BaselineCfg,
        reconstruction: pycolmap.Reconstruction,
    ) -> pycolmap.Reconstruction:
        """Add images to an existing empty
        reconstruction from this pose estimate."""
        pose_data = self.as_tuples()

        with open(cfg.cp_json_file) as f:
            cp_data = json.load(f)

        image_id = 1
        rig = reconstruction.rig(rig_id=1)

        if cfg.uses_imu:
            transform = get_t_imu_camera_from_json(
                cfg.device_calibration_json,
                camera_label="cam0",
            )
        else:
            transform = pycolmap.Rigid3d()

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
