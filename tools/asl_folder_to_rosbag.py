from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import rosbag
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu
from tqdm import tqdm


@dataclass(frozen=True)
class CamSample:
    ts_ns: int
    path: Path
    topic: str
    frame_id: str


@dataclass(frozen=True)
class ImuSample:
    ts_ns: int
    gyro: tuple[float, float, float]
    accel: tuple[float, float, float]
    topic: str
    frame_id: str


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def _read_cam_csv(
    csv_path: Path, images_dir: Path, topic: str, frame_id: str
) -> list[CamSample]:
    samples: list[CamSample] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if not _is_int(row[0]):
                continue
            ts_ns = int(row[0])
            img_rel = row[1]
            samples.append(
                CamSample(ts_ns, images_dir / img_rel, topic, frame_id)
            )

    return samples


def _read_imu_csv(csv_path: Path, topic: str, frame_id: str) -> list[ImuSample]:
    samples: list[ImuSample] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if not _is_int(row[0]):
                continue
            ts_ns = int(row[0])
            wx, wy, wz = (float(x) for x in row[1:4])
            ax, ay, az = (float(x) for x in row[4:7])
            samples.append(
                ImuSample(ts_ns, (wx, wy, wz), (ax, ay, az), topic, frame_id)
            )

    return samples


def asl_to_rosbag(
    input_asl_folder: Path,
    output_rosbag: Path,
    cam0_topic: str = "/cam0/image_raw",
    cam1_topic: str = "/cam1/image_raw",
    imu_topic: str = "/imu0",
    cam0_frame: str = "cam0",
    cam1_frame: str = "cam1",
    imu_frame: str = "imu0",
) -> None:
    aria = input_asl_folder / "aria"
    cam0_dir, cam1_dir, imu_dir = aria / "cam0", aria / "cam1", aria / "imu0"

    if not cam0_dir.exists():
        raise FileNotFoundError(f"Missing camera folder: {cam0_dir}")
    if not imu_dir.exists():
        raise FileNotFoundError(f"Missing IMU folder: {imu_dir}")

    cam0 = _read_cam_csv(
        cam0_dir / "data.csv", cam0_dir / "data", cam0_topic, cam0_frame
    )
    cam1 = []
    if cam1_dir.exists() and (cam1_dir / "data.csv").exists():
        cam1 = _read_cam_csv(
            cam1_dir / "data.csv", cam1_dir / "data", cam1_topic, cam1_frame
        )
    imu = _read_imu_csv(imu_dir / "data.csv", imu_topic, imu_frame)

    if not cam0:
        raise RuntimeError("cam0 data is empty.")
    if not imu:
        raise RuntimeError("IMU data is empty.")

    bag_timeline = []
    bag_timeline += [(s.ts_ns, "cam", s) for s in cam0]
    bag_timeline += [(s.ts_ns, "cam", s) for s in cam1]
    bag_timeline += [(s.ts_ns, "imu", s) for s in imu]
    bag_timeline.sort(key=lambda x: x[0])

    output_rosbag.parent.mkdir(parents=True, exist_ok=True)
    bridge = CvBridge()
    bag = rosbag.Bag(str(output_rosbag), "w")

    print(
        f"[asl_to_rosbag] Writing {len(bag_timeline):,} "
        f"messages -> {output_rosbag}"
    )

    for ts_ns, kind, sample in tqdm(
        bag_timeline, desc="Writing bag", unit="msg"
    ):
        stamp = rospy.Time.from_sec(ts_ns * NS_TO_S)

        if kind == "cam":
            cv_image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
            if cv_image is None:
                print(f"Warning: could not load image at path {sample.path}")
                continue
            img_msg = bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = sample.frame_id
            bag.write(sample.topic, img_msg, t=stamp)

        else:
            imu_msg = Imu()
            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = sample.frame_id
            wx, wy, wz = sample.gyro
            ax, ay, az = sample.accel
            (
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ) = wx, wy, wz
            (
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
            ) = ax, ay, az

            imu_msg.orientation_covariance[0] = -1
            imu_msg.angular_velocity_covariance[0] = -1
            imu_msg.linear_acceleration_covariance[0] = -1

            bag.write(sample.topic, imu_msg, t=stamp)

    bag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_asl_folder", type=Path, required=True, help="Input folder"
    )
    parser.add_argument(
        "--output_rosbag", type=Path, required=True, help="Output rosbag file"
    )

    args = parser.parse_args()
    asl_to_rosbag(args.input_asl_folder, args.output_rosbag)
