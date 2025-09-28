import json
from bisect import bisect_left
from pathlib import Path

from .constants import (
    LEFT_CAMERA_STREAM_LABEL,
    RIGHT_CAMERA_STREAM_LABEL,
)


def get_timestamp_to_images_from_json(
    json_file: str | Path,
):
    with open(json_file) as f:
        data = json.load(f)

    processed_ts_data = {}
    for label in [LEFT_CAMERA_STREAM_LABEL, RIGHT_CAMERA_STREAM_LABEL]:
        ts_data = data["timestamps"][label]
        processed_ts_data[label] = {int(k): v for k, v in ts_data.items()}
        processed_ts_data[label]["sorted_keys"] = sorted(
            processed_ts_data[label].keys()
        )

    return processed_ts_data


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
