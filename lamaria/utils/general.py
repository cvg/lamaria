import os
import shutil
import subprocess
from bisect import bisect_left
from pathlib import Path
from typing import Dict

import pycolmap

from lamaria import logger

CUSTOM_ORIGIN_COORDINATES = (2683594.4120000005, 1247727.7470000014, 417.307)


def find_closest_timestamp(
    timestamps: list,
    target_ts: int,
    max_diff: float,
) -> int | None:
    """Timestamps must be in nano seconds"""
    index = bisect_left(timestamps, target_ts)
    if index == 0:
        return timestamps[0]
    if index == len(timestamps):
        return timestamps[-1]
    before = timestamps[index - 1]
    after = timestamps[index]
    if abs(target_ts - before) < abs(target_ts - after):
        closest = before
    else:
        closest = after

    if abs(target_ts - closest) > max_diff:
        return None

    return closest


def get_matched_timestamps(
    left_timestamps: list[int],
    right_timestamps: list[int],
    max_diff: float,
) -> list[tuple[int, int]]:
    matched_timestamps = []

    assert all(isinstance(ts, int) for ts in left_timestamps), (
        "Left timestamps must be integers"
    )
    assert all(isinstance(ts, int) for ts in right_timestamps), (
        "Right timestamps must be integers"
    )

    if len(left_timestamps) < len(right_timestamps):
        for lts in left_timestamps:
            closest_rts = find_closest_timestamp(
                right_timestamps, lts, max_diff
            )
            if closest_rts is not None:
                matched_timestamps.append((lts, closest_rts))
    else:
        for rts in right_timestamps:
            closest_lts = find_closest_timestamp(left_timestamps, rts, max_diff)
            if closest_lts is not None:
                matched_timestamps.append((closest_lts, rts))

    return matched_timestamps


def delete_files_in_folder(folder, exclude_pattern=None):
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if exclude_pattern is not None and exclude_pattern in filename:
                continue

            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
    else:
        os.makedirs(folder, exist_ok=True)


def extract_images_from_vrs(
    vrs_file: Path,
    image_folder: Path,
    left_subfolder_name="left",
    right_subfolder_name="right",
    rgb_subfolder_name="rgb",
    verbose: bool = False,
    extract_rgb: bool = False,
    extract_left: bool = True,
    extract_right: bool = True,
):
    for camera, stream_id in [
        (left_subfolder_name, "1201-1"),
        (right_subfolder_name, "1201-2"),
        (rgb_subfolder_name, "214-1"),
    ]:
        if camera == rgb_subfolder_name and not extract_rgb:
            continue

        if camera == left_subfolder_name and not extract_left:
            continue

        if camera == right_subfolder_name and not extract_right:
            continue

        output_dir = image_folder / camera
        output_dir.mkdir(parents=True, exist_ok=True)
        delete_files_in_folder(output_dir)
        logger.info(
            "Extracting images for camera %s in VRS %s", camera, vrs_file
        )
        cmd = f"vrs extract-images {vrs_file} --to {output_dir} + {stream_id}"
        stdout = None if verbose else subprocess.PIPE
        out = subprocess.run(
            cmd, shell=True, stderr=subprocess.STDOUT, stdout=stdout
        )
        if out.returncode:
            msg = f"Command '{cmd}' returned {out.returncode}."
            if out.stdout:
                msg += "\n" + out.stdout.decode("utf-8")
            raise subprocess.SubprocessError(msg)
        logger.info("Done!")


def get_image_names_to_ids(reconstruction_path: Path) -> Dict[str, int]:
    recon = pycolmap.Reconstruction(reconstruction_path)
    image_names_to_ids = {}

    for image_id in recon.images.keys():
        image_name = recon.images[image_id].name
        image_names_to_ids[image_name] = image_id

    return image_names_to_ids
