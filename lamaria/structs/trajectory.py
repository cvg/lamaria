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
from ..utils.timestamps import (
    find_closest_timestamp,
    matching_time_indices,
)


def _round_ns(x: str | int | float) -> int:
    # works for "5120...274.999939", "5.12e11", or ints
    if isinstance(x, int):
        return x
    s = str(x)
    return int(Decimal(s).to_integral_value(rounding=ROUND_HALF_UP))


class Trajectory:
    """
    Loads and stores traj data from 'estimate' text file with rows:
      ts t_x t_y t_z q_x q_y q_z q_w
    Blank lines and lines starting with '#' are ignored.

    By default, poses are calculated as rig_from_world
    (i.e., inverse of world_from_rig) to satisfy COLMAP format.
    
    Attributes:
        invert_poses (bool): Whether to invert poses to 
        rig_from_world format.
        corresponding_sensor (str): The reference sensor to in which
        the trajectory is represented ("imu" or "cam0").
    """

    def __init__(self) -> None:
        self.invert_poses = None
        self.corresponding_sensor = None
        self._timestamps: list[int] = []
        self._poses: list[pycolmap.Rigid3d] = []

    @classmethod
    def load_from_file(
        cls,
        path: str | Path,
        invert_poses: bool = True,
        corresponding_sensor: str = "imu",
    ) -> "Trajectory":
        """Parse the file, validate format, populate timestamps & poses."""
        self = cls()
        self.clear()
        path = Path(path)
        self.invert_poses = invert_poses
        self.corresponding_sensor = corresponding_sensor

        if not path.exists():
            raise FileNotFoundError(f"Estimate file not found: {path}")

        with open(path) as f:
            lines = f.readlines()

        state = self._parse(lines)
        if not state:
            raise RuntimeError("Failed to parse estimate file.")

        return self

    def add_estimate_poses_to_reconstruction(
        self,
        reconstruction: pycolmap.Reconstruction,
        timestamp_to_images: dict,
    ) -> pycolmap.Reconstruction:
        """
        Adds estimate poses as frames to input reconstruction.
        """
        self._ensure_loaded()
        reconstruction = self._add_images_to_reconstruction(
            reconstruction, timestamp_to_images
        )

        return reconstruction

    def filter_from_indices(
        self,
        ids: np.ndarray,
    ) -> None:
        """
        From evo package.
        Edits the trajectory based on the provided indices.
        """
        self._ensure_loaded()
        self._timestamps = [self._timestamps[i] for i in ids]
        self._poses = [self._poses[i] for i in ids]

    def transform(self, tgt_from_src: pycolmap.Sim3d) -> None:
        """Apply a similarity transformation to all poses in the trajectory."""
        self._ensure_loaded()
        _tmp_poses = (
            self._poses.copy()
            if self.invert_poses
            else [p.inverse() for p in self._poses]
        )

        # tmp_poses are in rig_from_world format
        for i, p in enumerate(_tmp_poses):
            rig_from_new_world = (
                pycolmap.Sim3d(1, p.rotation, p.translation)
                * tgt_from_src.inverse()
            )
            new_p = pycolmap.Rigid3d(
                rig_from_new_world.rotation,
                rig_from_new_world.translation * tgt_from_src.scale,
            )
            self._poses[i] = new_p if self.invert_poses else new_p.inverse()

    @property
    def timestamps(self) -> list[int]:
        self._ensure_loaded()
        return self._timestamps

    @property
    def poses(self) -> list[pycolmap.Rigid3d]:
        self._ensure_loaded()
        return self._poses

    @property
    def positions(self) -> np.ndarray:
        """Returns Nx3 numpy array of positions."""
        self._ensure_loaded()
        if not self.invert_poses:
            # poses are in world_from_rig format
            return np.array([p.translation for p in self._poses])
        else:
            return np.array([p.inverse().translation for p in self._poses])

    @property
    def orientations(self) -> np.ndarray:
        """Returns Nx4 numpy array of quaternions (x, y, z, w)."""
        self._ensure_loaded()
        if not self.invert_poses:
            # poses are in world_from_rig format
            return np.array([p.rotation.quat for p in self._poses])
        else:
            # poses are in rig_from_world format
            return np.array([p.inverse().rotation.quat for p in self._poses])

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

        return True

    def _add_images_to_reconstruction(
        self,
        reconstruction: pycolmap.Reconstruction,
        timestamp_to_images: dict,
    ) -> pycolmap.Reconstruction:
        pose_data = self.as_tuples()

        image_id = 1
        # imu is the rig in this reconstruction
        rig = reconstruction.rig(rig_id=1)

        if self.corresponding_sensor == "imu":
            transform = pycolmap.Rigid3d()
        else:
            # left camera poses are provided
            # sensor_from_rig == cam0_from_imu
            transform = rig.sensor_from_rig(sensor_id=2)

        for i, (timestamp, pose) in tqdm(
            enumerate(pose_data),
            total=len(pose_data),
            desc="Adding images to reconstruction",
        ):
            frame = pycolmap.Frame()
            frame.rig_id = rig.rig_id
            frame.frame_id = i + 1

            if self.invert_poses:
                # poses are in imu/cam_from_world format
                # if imu: imu_from_world
                # if cam0: cam0_from_world
                T_world_rig = pose.inverse() * transform
            else:
                # poses are in world_from_cam/imu format
                T_world_rig = pose * transform

            frame.rig_from_world = T_world_rig.inverse()

            images_to_add = []

            for label, camera_id in [
                (LEFT_CAMERA_STREAM_LABEL, 2),
                (RIGHT_CAMERA_STREAM_LABEL, 3),
            ]:
                source_timestamps = timestamp_to_images[label]["sorted_keys"]
                # offsets upto 1 ms (1e6 ns)
                closest_timestamp = find_closest_timestamp(
                    source_timestamps, timestamp, max_diff=1e6
                )
                if closest_timestamp is None:
                    raise ValueError

                image_name = timestamp_to_images[label][closest_timestamp]

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


def associate_trajectories(
    traj1: Trajectory,
    traj2: Trajectory,
) -> tuple[Trajectory, Trajectory]:
    """
    From evo package.
    Associate two trajectories based on their timestamps.
    Args:
        traj1 (Trajectory): First trajectory.
        traj2 (Trajectory): Second trajectory.
    Returns:
        tuple[Trajectory, Trajectory]: Associated trajectories.
    """
    if not traj1.is_loaded() or not traj2.is_loaded():
        logger.error("Trajectories must be loaded before association.")
        return None, None

    first_longer = len(traj1) >= len(traj2)
    longer_traj = traj1 if first_longer else traj2
    shorter_traj = traj2 if first_longer else traj1

    short_idx, long_idx = matching_time_indices(
        shorter_traj.timestamps,
        longer_traj.timestamps,
    )
    num_matches = len(long_idx)
    if num_matches == 0:
        logger.error("No matching timestamps found between trajectories.")
        return None, None

    longer_traj.filter_from_indices(long_idx)
    shorter_traj.filter_from_indices(short_idx)

    traj1 = traj1 if first_longer else traj2
    traj2 = traj2 if first_longer else traj1

    return traj1, traj2
