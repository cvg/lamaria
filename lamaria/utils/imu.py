import pycolmap
import numpy as np
from pathlib import Path
from tqdm import tqdm
from projectaria_tools.core import mps, data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from lamaria.utils.general import (
    IMU_STREAM_ID,
    find_closest_timestamp,
)

def get_online_params_for_imu_from_mps(
    online_calibs_file: Path,
    stream_label: str,
    num_error: float = 1e6
):
    online_calibs = mps.read_online_calibration(online_calibs_file.as_posix())
    online_imu_calibs = {}
    num_error = int(num_error)
    for calib in tqdm(online_calibs, desc="Reading online IMU calibration data"):
        for imuCalib in calib.imu_calibs:
            if imuCalib.get_label() == stream_label:
                # calib timestamp in microseconds
                # convert to nanoseconds and then quantize to milliseconds
                timestamp = int(calib.tracking_timestamp.total_seconds() * 1e9)
                quantized_timestamp = timestamp // num_error
                online_imu_calibs[quantized_timestamp] = imuCalib

    return online_imu_calibs


def get_online_imu_data_from_vrs(
    vrs_provider: data_provider.VrsDataProvider,
    mps_folder: Path,
):
    imu_timestamps = sorted(
        vrs_provider.get_timestamps_ns(
            IMU_STREAM_ID,
            TimeDomain.DEVICE_TIME
        )
    )
    imu_stream_label = vrs_provider.get_label_from_stream_id(IMU_STREAM_ID)

    online_calibs_file = mps_folder / "slam" / "online_calibration.jsonl"
    online_imu_calibs = get_online_params_for_imu_from_mps(
        online_calibs_file,
        imu_stream_label
    )

    acceptable_diff_ms = 1 # 1 milliseconds

    calib_timestamps = sorted(online_imu_calibs.keys())

    ms = pycolmap.ImuMeasurements()
    for timestamp in tqdm(imu_timestamps, desc="Loading rect IMU data"):
        quantized_timestamp = timestamp // int(1e6)
        closest_ts = find_closest_timestamp(
            calib_timestamps,
            quantized_timestamp,
            acceptable_diff_ms
        )

        if closest_ts not in online_imu_calibs:
            raise ValueError(f"No calibration found for timestamp {timestamp}")

        calibration = online_imu_calibs[closest_ts]
        imu_data = vrs_provider.get_imu_data_by_time_ns(
            IMU_STREAM_ID,
            timestamp,
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )
        if imu_data.accel_valid and imu_data.gyro_valid:
            rectified_acc = calibration.raw_to_rectified_accel(
                imu_data.accel_msec2
            )
            rectified_gyro = calibration.raw_to_rectified_gyro(
                imu_data.gyro_radsec
            )
            ts = float(timestamp) / 1e9  # convert to seconds
            ms.insert(
                pycolmap.ImuMeasurement(ts, rectified_acc, rectified_gyro)
            )

    return ms