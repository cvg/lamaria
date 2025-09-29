from bisect import bisect_left
from copy import deepcopy
from pathlib import Path

import pycolmap

from ...structs.timed_reconstruction import TimedReconstruction
from ...structs.trajectory import (
    Trajectory,
)
from ...utils.types import InitReconstruction


def _image_names_from_folder(
    folder: Path, wrt_to: Path, ext: str = ".jpg"
) -> list[Path]:
    if not folder.is_dir():
        return []
    images = sorted(n for n in folder.iterdir() if n.suffix == ext)
    images = [n.relative_to(wrt_to) for n in images]
    return images


def _match_estimate_ts_to_images(
    timestamps_to_images: dict[int, tuple[Path, Path]],
    est_timestamps: list[int],
    max_diff: int = 1000000,  # 1 ms
) -> tuple[list[tuple[Path, Path]], list[int]]:
    left_ts = list(timestamps_to_images.keys())
    images = list(timestamps_to_images.values())
    assert len(images) == len(left_ts), (
        "Number of images and left timestamps must be equal"
    )

    order = sorted(range(len(left_ts)), key=lambda i: left_ts[i])
    left_ts = [left_ts[i] for i in order]
    images = [images[i] for i in order]

    matched_images: list[tuple[Path, Path]] = []
    matched_timestamps: list[int] = []

    # estimate timestamps will be in nanoseconds like vrs timestamps
    for est in est_timestamps:
        idx = bisect_left(left_ts, est)

        cand_idxs = []
        if idx > 0:
            cand_idxs.append(idx - 1)
        if idx < len(left_ts):
            cand_idxs.append(idx)

        if not cand_idxs:
            continue

        best = min(cand_idxs, key=lambda j: abs(left_ts[j] - est))
        if (max_diff is not None) and (abs(left_ts[best] - est) > max_diff):
            continue

        matched_images.append(images[best])
        matched_timestamps.append(left_ts[best])
    return dict(zip(matched_timestamps, matched_images))


def convert_estimate_into_timed_reconstruction(
    init_reconstruction: InitReconstruction,
    estimate: Trajectory,
    timestamps_to_images: dict[int, tuple[Path, Path]],
) -> TimedReconstruction:
    """
    Populate a TimedReconstruction from a trajectory
    """
    est_timestamps_to_images = _match_estimate_ts_to_images(
        timestamps_to_images, estimate.timestamps
    )

    assert estimate.corresponding_sensor == "imu"
    timestamps = list(est_timestamps_to_images.keys())
    assert len(estimate.poses) == len(timestamps), (
        "The length of traj.poses and timestamps should be equal"
    )

    recon = deepcopy(init_reconstruction)
    image_id = 1
    frame_id_to_timestamp = dict()
    for frame_id, (pose, timestamp) in enumerate(
        zip(estimate.poses, timestamps)
    ):
        frame = pycolmap.Frame()
        frame.rig_id = 1
        frame.frame_id = frame_id
        frame.rig_from_world = pose  # as it corresponds to imu

        image_names = timestamps_to_images(timestamp)
        images_to_add = []
        for cam_id, img_path in [(2, image_names[0]), (3, image_names[1])]:
            im = pycolmap.Image(
                str(img_path),
                pycolmap.Point2DList(),
                cam_id,
                image_id,
            )
            im.frame_id = frame.frame_id
            frame.add_data_id(im.data_id)
            images_to_add.append(im)
            image_id += 1
        recon.add_frame(frame)
        for im in images_to_add:
            recon.add_image(im)
        frame_id_to_timestamp[frame_id] = timestamp

    return TimedReconstruction(
        reconstruction=recon, timestamps=frame_id_to_timestamp
    )
