from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pycolmap


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

    By default, poses are returned as rig_from_world
    (i.e., inverse of world_from_rig) to satisfy COLMAP format.
    """

    def __init__(self, invert_poses: bool = True) -> None:
        self.invert_poses = invert_poses
        self.path: Path | None = None
        self._timestamps: list[int] = []
        self._poses: list[pycolmap.Rigid3d] = []

    def load_from_file(self, path: str | Path) -> "Estimate":
        """Parse the file, validate format, populate timestamps & poses."""
        self.clear()
        self.path = Path(path)

        with open(self.path, "r") as f:
            lines = f.readlines()

        self._parse(lines)  # raises error if format is invalid
        return self
    
    @property
    def timestamps(self) -> list[int]:
        self._ensure_loaded()
        return self._timestamps

    @property
    def poses(self) -> list[pycolmap.Rigid3d]:
        self._ensure_loaded()
        return self._poses
    
    def as_dict(self) -> dict[int, pycolmap.Rigid3d]:
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
            raise RuntimeError("Estimate not loaded. Call load_from_file() first.")
        
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
                raise ValueError(f"{lineno}: expected 8 values, got {len(parts)}")
            
            try:
                ts = _round_ns(parts[0])
                tvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                qvec = np.array([float(parts[4]), float(parts[5]),
                                 float(parts[6]), float(parts[7])])
                
            except ValueError as e:
                raise ValueError(f"{lineno}: non-numeric value in {parts!r}") from e
            
            world_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), tvec)
            pose = world_from_rig.inverse() if self.invert_poses else world_from_rig

            ts_list.append(ts)
            pose_list.append(pose)

        if not exists_lines:
            raise ValueError("No valid lines found in estimate file.")
        
        self._timestamps = ts_list
        self._poses = pose_list