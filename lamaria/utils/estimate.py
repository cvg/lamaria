from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pycolmap


def round_ns(x: str | int | float) -> int:
    # works for "5120...274.999939", "5.12e11", or ints
    if isinstance(x, int):
        return x
    s = str(x)
    return int(Decimal(s).to_integral_value(rounding=ROUND_HALF_UP))


def get_rig_from_worlds_from_estimate(
    estimate_path: Path,
) -> list[pycolmap.Rigid3d]:
    """Estimate file format: ts t_x t_y t_z q_x q_y q_z q_w"""

    rig_from_worlds = []
    with open(estimate_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue

            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(
                    f"Estimate file {estimate_path} has invalid format. \
                                 Each line must have 8 values."
                )

            try:
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
            except ValueError:
                raise ValueError(
                    f"Estimate file {estimate_path} has invalid format. \
                                 Each value must be a number."
                )

            pose = pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), tvec)
            rig_from_world = pose.inverse()
            rig_from_worlds.append(rig_from_world)

    return rig_from_worlds


def check_estimate_format(estimate_path: Path) -> bool:
    """Estimate file format: ts t_x t_y t_z q_x q_y q_z q_w"""
    with open(estimate_path) as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith("#")]
        if not lines:
            raise ValueError(
                f"Estimate file {estimate_path} is empty \
                             or has no valid lines."
            )

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(
                    f"Estimate file {estimate_path} has invalid format. \
                                 Each line must have 8 values."
                )

            try:
                [float(part) for part in parts]
            except ValueError:
                raise ValueError(
                    f"Estimate file {estimate_path} has invalid format. \
                                 Each value must be a number."
                )

    return True


def get_estimate_timestamps(estimate_path: Path) -> list[int]:
    """Estimate file format: ts t_x t_y t_z q_x q_y q_z q_w"""
    timestamps = []
    with open(estimate_path) as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith("#")]
        if not lines:
            raise ValueError(
                f"Estimate file {estimate_path} is empty \
                             or has no valid lines."
            )

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(
                    f"Estimate file {estimate_path} has invalid format. \
                                 Each line must have 8 values."
                )

            try:
                ts = round_ns(parts[0])
            except ValueError:
                raise ValueError(
                    f"Estimate file {estimate_path} has invalid format. \
                                 Each value must be a number."
                )

            timestamps.append(ts)

    return timestamps
