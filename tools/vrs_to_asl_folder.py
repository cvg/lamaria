import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from tqdm import tqdm

from lamaria import logger
from lamaria.utils.general import (
    LEFT_CAMERA_STREAM_ID,
    RIGHT_CAMERA_STREAM_ID,
    IMU_STREAM_ID,
    extract_images_from_vrs,
    get_matched_timestamps,
)


def remove_images_when_slam_drops(
    image_folder: Path,
    left_timestamps: List[int],
    right_timestamps: List[int],
    matched_timestamps: List[Tuple[int, int]],
    left_subfolder_name="cam0/data",
    right_subfolder_name="cam1/data",
):
    left_image_folder = os.path.join(image_folder, left_subfolder_name)
    right_image_folder = os.path.join(image_folder, right_subfolder_name)

    assert os.path.exists(left_image_folder) and os.path.exists(
        right_image_folder
    )

    original_left_images = sorted(os.listdir(left_image_folder))
    original_right_images = sorted(os.listdir(right_image_folder))

    assert len(original_left_images) == len(left_timestamps)
    assert len(original_right_images) == len(right_timestamps)

    left_camera_mapping = {
        ts: img for ts, img in zip(left_timestamps, original_left_images)
    }
    right_camera_mapping = {
        ts: img for ts, img in zip(right_timestamps, original_right_images)
    }

    matched_left_ts = [left_ts for left_ts, _ in matched_timestamps]
    matched_right_ts = [right_ts for _, right_ts in matched_timestamps]

    if len(left_timestamps) != len(matched_timestamps):
        for ts, img in left_camera_mapping.items():
            if ts not in matched_left_ts:
                print(f"Removing {img}")
                os.remove(os.path.join(left_image_folder, img))
    elif len(right_timestamps) != len(matched_timestamps):
        for ts, img in right_camera_mapping.items():
            if ts not in matched_right_ts:
                print(f"Removing {img}")
                os.remove(os.path.join(right_image_folder, img))
    else:
        raise ValueError("No images to remove")

    return matched_left_ts


def rename_images_in_folder(
    aria_folder: Path,
    image_timestamps,
    left_subfolder_name="cam0/data",
    right_subfolder_name="cam1/data",
    image_extension=".jpg",
) -> List[int]:
    
    for subfolder in [left_subfolder_name, right_subfolder_name]:
        subfolder_path = aria_folder / subfolder

        if not subfolder_path.exists() and not subfolder_path.is_dir():
            raise ValueError(f"{subfolder_path} does not exist or is not a directory")

        original_images = sorted([f for f in os.listdir(subfolder_path) if f.endswith(image_extension)])
        if len(original_images) == 0:
            raise ValueError(f"No images found in {subfolder_path}")
        
        if len(original_images) != len(image_timestamps):
            raise ValueError(
                f"Number of images {len(original_images)} \
                    does not match number of timestamps \
                    {len(image_timestamps)} in {subfolder_path}"
            )

        for ts, img in zip(image_timestamps, original_images):
            old_image_path = subfolder_path / img
            new_image_path = subfolder_path / f"{ts}{image_extension}"
            os.rename(old_image_path, new_image_path)

    return image_timestamps


def write_image_timestamps_to_txt(image_timestamps: List, txt_file: Path):
    with open(txt_file, "w") as f:
        for timestamp in image_timestamps:
            f.write(f"{timestamp}\n")


def write_image_csv(image_timestamps, cam_folder):
    images = os.listdir(os.path.join(cam_folder, "data"))
    assert len(images) > 0

    images = sorted(images, key=lambda img: int(img.split(".")[0]))

    assert len(images) == len(image_timestamps)
    for ts, img in zip(image_timestamps, images):
        assert int(img.split(".")[0]) == ts, f"{img} != {ts}"

    data_csv = os.path.join(cam_folder, "data.csv")
    if os.path.exists(data_csv):
        os.remove(data_csv)

    with open(data_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for timestamp, image in zip(image_timestamps, images):
            row = [timestamp, image]
            writer.writerow(row)


def write_imu_data_to_csv(vrs_provider, csv_file):
    imu_timestamps = vrs_provider.get_timestamps_ns(
        IMU_STREAM_ID, TimeDomain.DEVICE_TIME
    )

    last_timestamp = None
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            last_row = None
            for last_row in csv.reader(f):
                pass
            if last_row is not None:
                last_timestamp = int(last_row[0])

    if last_timestamp is not None:
        imu_timestamps = [ts for ts in imu_timestamps if ts > last_timestamp]

    if not imu_timestamps:
        logger.info(f"No new IMU data to write to {csv_file}")
        return

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        for timestamp in tqdm(imu_timestamps, desc="Appending IMU data to CSV"):
            imu_data = vrs_provider.get_imu_data_by_time_ns(
                IMU_STREAM_ID,
                timestamp,
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.CLOSEST,
            )
            if imu_data.accel_valid and imu_data.gyro_valid:
                accel = imu_data.accel_msec2
                gyro = imu_data.gyro_radsec

                row = [
                    timestamp,
                    gyro[0],
                    gyro[1],
                    gyro[2],
                    accel[0],
                    accel[1],
                    accel[2],
                ]
                writer.writerow(row)


def form_aria_asl_folder(
    vrs_file: Path,
    output_asl_folder: Path,
    has_slam_drops=False
):
    if output_asl_folder.exists():
        raise ValueError(f"{output_asl_folder=} already exists.")
    
    aria_folder = output_asl_folder / "aria"
    aria_folder.mkdir(parents=True, exist_ok=True)

    dataset_name = vrs_file.stem
    vrs_provider = data_provider.create_vrs_data_provider(vrs_file.as_posix())

    # Get all image timestamps (in ns)
    image_timestamps = vrs_provider.get_timestamps_ns(
        LEFT_CAMERA_STREAM_ID, TimeDomain.DEVICE_TIME
    )
    assert (
        len(image_timestamps) > 0
    ), "No timestamps found"

    right_image_timestamps = None
    matched_timestamps = None
    if has_slam_drops:
        right_image_timestamps = vrs_provider.get_timestamps_ns(
            RIGHT_CAMERA_STREAM_ID, TimeDomain.DEVICE_TIME
        )
        assert (
            len(right_image_timestamps) > 0
        ), "No right camera image timestamps found"
        assert len(right_image_timestamps) != len(
            image_timestamps
        ), "Left and right camera image timestamps are the same"
        matched_timestamps = get_matched_timestamps(
            left_timestamps=image_timestamps,
            right_timestamps=right_image_timestamps,
            max_diff=1e6, # 1 ms in nanoseconds
        )

        assert len(matched_timestamps) > 0, "No matched timestamps found"

    extract_images_from_vrs(
        vrs_file=vrs_file,
        image_folder=aria_folder,
        left_subfolder_name="cam0/data",
        right_subfolder_name="cam1/data",
    )

    if has_slam_drops:
        assert (
            right_image_timestamps is not None
            and matched_timestamps is not None
        )
        image_timestamps = remove_images_when_slam_drops(
            aria_folder,
            image_timestamps,
            right_image_timestamps,
            matched_timestamps,
        )

    image_timestamps = rename_images_in_folder(
        aria_folder,
        image_timestamps,
    )

    imu_folder = output_asl_folder / "aria" / "imu0"
    imu_folder.mkdir(parents=True, exist_ok=True)
    imu_csv = imu_folder / "data.csv"

    write_imu_data_to_csv(
        vrs_provider,
        imu_csv,
    )

    write_image_timestamps_to_txt(
        image_timestamps,
        aria_folder / f"{dataset_name}.txt",
    )
    write_image_csv(image_timestamps, aria_folder / "cam0")
    write_image_csv(image_timestamps, aria_folder / "cam1")


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--vrs_file", type=Path, required=True)
    args.add_argument("--output_asl_folder", type=Path, required=True)
    args.add_argument(
        "--has_slam_drops",
        action="store_true",
        help="Whether the VRS file has dropped SLAM frames",
    )

    args = args.parse_args()

    form_aria_asl_folder(
        args.vrs_file,
        args.output_asl_folder,
        has_slam_drops=args.has_slam_drops,
    )
