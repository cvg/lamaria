from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import projectaria_tools.core.mps as mps
import pycolmap
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from ... import logger
from ...config.options import EstimateToLamariaOptions
from ...structs.lamaria_reconstruction import LamariaReconstruction
from ...structs.estimate import (
    Estimate,
)
from ...utils.general import (
    extract_images_from_vrs,
    get_matched_timestamps,
)
from ...utils.aria import (
    get_closed_loop_data_from_mps,
    get_mps_poses_for_timestamps,
    get_t_imu_camera,
    rigid3d_from_transform,
    camera_colmap_from_calib,
    get_imu_data_from_vrs
)


@dataclass
class PerFrameData:
    left_ts: int
    right_ts: int
    left_img: Path
    right_img: Path
    rig_from_world: pycolmap.Rigid3d


class EstimateToLamaria:
    """Converts estimate or MPS data to COLMAP format."""

    def __init__(self, options: EstimateToLamariaOptions):
        self.options = options
        self.data: LamariaReconstruction = LamariaReconstruction()
        self._vrs_provider = None
        self._mps_data_provider = None
        self._left_cam_sid: StreamId | None = None
        self._right_cam_sid: StreamId | None = None
        self._right_imu_sid: StreamId | None = None
        self._per_frame_data: dict[int, PerFrameData] = {}

    @staticmethod
    def convert(
        options: EstimateToLamariaOptions,
        vrs: Path,
        images_path: Path,
        estimate: Path | None = None,
        mps_folder: Path | None = None,
    ) -> LamariaReconstruction:
        """Entry point to run estimate/MPS to colmap conversion."""
        to_colmap = EstimateToLamaria(options)
        return to_colmap.process(vrs, images_path, estimate, mps_folder)

    def process(
        self,
        vrs: Path,
        images_path: Path,
        estimate: Path | None = None,
        mps_folder: Path | None = None,
    ) -> LamariaReconstruction:
        self._init_data(vrs, images_path, estimate, mps_folder)

        if self.options.mps.use_online_calibration:
            self._add_online_sensors()
            self._add_online_frames()
        else:
            self._add_device_sensors()
            self._add_device_frames()

        # IMU + timestamps
        ms = self._get_rectified_imu_data(mps_folder)
        self.data.imu_measurements = ms
        self.data.timestamps = {
            fid: pfd.left_ts for fid, pfd in self._per_frame_data.items()
        }
        return self.data

    # -------- internal methods --------
    def _init_data(
        self,
        vrs: Path,
        image_folder: Path,
        estimate: Path | None = None,
        mps_folder: Path | None = None,
    ) -> None:
        """Initializes data providers and extracts images, timestamps
        and builds per-frame data object.
        Per-frame data is used to create the initial Lamaria reconstruction.
        """

        # Initialize VRS data provider
        self._vrs_provider = data_provider.create_vrs_data_provider(
            vrs.as_posix()
        )

        # Initialize stream IDs
        self._left_cam_sid = StreamId(self.options.sensor.left_cam_stream_id)
        self._right_cam_sid = StreamId(self.options.sensor.right_cam_stream_id)
        self._right_imu_sid = StreamId(self.options.sensor.right_imu_stream_id)

        # Initialize MPS data provider if needed
        if mps_folder is not None:
            data_paths = mps.MpsDataPathsProvider(
                mps_folder.as_posix()
            ).get_data_paths()
            self._mps_data_provider = mps.MpsDataProvider(data_paths)

        # Extract images from VRS file
        extract_images_from_vrs(
            vrs_file=vrs,
            image_folder=image_folder,
        )

        images = self._get_images(image_folder)

        # Get timestamps and build per-frame data
        if estimate is None:
            timestamps = self._get_mps_timestamps()
            closed_loop_data = get_closed_loop_data_from_mps(mps_folder)
            pose_timestamps = [left for left, _ in timestamps]
            mps_poses = get_mps_poses_for_timestamps(
                closed_loop_data, pose_timestamps
            )
            self._per_frame_data = self._build_per_frame_data_from_mps(
                images, timestamps, mps_poses
            )
        else:
            # Raises error if estimate file is invalid
            est = Estimate(invert_poses=True)
            est.load_from_file(estimate)

            timestamps = est.timestamps
            if len(images) != len(timestamps):
                images, timestamps = self._match_estimate_ts_to_images(
                    images, timestamps
                )

            rig_from_worlds = est.poses
            self._per_frame_data = self._build_per_frame_data_from_estimate(
                images, timestamps, rig_from_worlds
            )

    def _build_per_frame_data_from_mps(
        self, images, timestamps, mps_poses
    ) -> dict[int, PerFrameData]:
        per_frame_data: dict[int, PerFrameData] = {}
        imu_stream_label = self._vrs_provider.get_label_from_stream_id(
            self._right_imu_sid
        )

        if not self.options.mps.use_online_calibration:
            device_calibration = self._vrs_provider.get_device_calibration()
            imu_calib = device_calibration.get_imu_calib(imu_stream_label)

        frame_id = 1

        for (left_img, right_img), (left_ts, right_ts), t_world_device in zip(
            images, timestamps, mps_poses
        ):
            if t_world_device is None:
                continue

            if self.options.mps.use_online_calibration:
                ocalib = self._mps_data_provider.get_online_calibration(
                    left_ts, TimeQueryOptions.CLOSEST
                )
                if ocalib is None:
                    continue
                imu_calib = None
                for calib in ocalib.imu_calibs:
                    if calib.get_label() == imu_stream_label:
                        imu_calib = calib
                        break

            t_device_imu = imu_calib.get_transform_device_imu()
            t_world_imu = t_world_device @ t_device_imu
            t_imu_world = t_world_imu.inverse()
            rig_from_world = rigid3d_from_transform(t_imu_world)

            pfd = PerFrameData(
                left_ts=left_ts,
                right_ts=right_ts,
                left_img=left_img,
                right_img=right_img,
                rig_from_world=rig_from_world,
            )
            per_frame_data[frame_id] = pfd
            frame_id += 1

        return per_frame_data

    def _build_per_frame_data_from_estimate(
        self, images, timestamps, rig_from_worlds
    ) -> dict[int, PerFrameData]:
        per_frame_data: dict[int, PerFrameData] = {}
        assert len(images) == len(timestamps) == len(rig_from_worlds), (
            "Number of images, timestamps and poses must be equal"
        )

        frame_id = 1

        for (left_img, right_img), ts, rig_from_world in zip(
            images, timestamps, rig_from_worlds
        ):
            pfd = PerFrameData(
                left_ts=ts,
                right_ts=ts,
                left_img=left_img,
                right_img=right_img,
                rig_from_world=rig_from_world,
            )
            per_frame_data[frame_id] = pfd
            frame_id += 1

        return per_frame_data

    def _images_from_vrs(
        self, folder: Path, wrt_to: Path, ext: str = ".jpg"
    ) -> list[Path]:
        if not folder.is_dir():
            return []
        images = sorted(n for n in folder.iterdir() if n.suffix == ext)
        images = [n.relative_to(wrt_to) for n in images]
        return images

    def _ts_from_vrs(self, sid: StreamId) -> list[int]:
        """Timestamps in nanoseconds"""
        return sorted(
            self._vrs_provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME)
        )

    def _get_images(self, images_path: Path) -> list[tuple[Path, Path]]:
        left_img_dir = images_path / "left"
        right_img_dir = images_path / "right"

        left_images = self._images_from_vrs(left_img_dir, left_img_dir)
        right_images = self._images_from_vrs(right_img_dir, right_img_dir)

        return list(zip(left_images, right_images))

    def _get_mps_timestamps(self, max_diff=1e6) -> list[tuple[int, int]]:
        L = self._ts_from_vrs(self._left_cam_sid)
        R = self._ts_from_vrs(self._right_cam_sid)
        if not self.options.sensor.has_slam_drops:
            assert len(L) == len(R), (
                "Unequal number of left and right timestamps"
            )
            matched = list(zip(L, R))
            if not all(abs(left - right) < max_diff for left, right in matched):
                logger.warning(
                    f"Left and right timestamps differ "
                    f"by more than {max_diff} ns"
                )
        else:
            matched = get_matched_timestamps(L, R, max_diff)

        return matched

    def _match_estimate_ts_to_images(
        self,
        images: list[tuple[Path, Path]],
        est_timestamps: list[int],
        max_diff: int = 1000000,  # 1 ms
    ) -> tuple[list[tuple[Path, Path]], list[int]]:
        left_ts = self._ts_from_vrs(self._left_cam_sid)
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

        return matched_images, matched_timestamps

    def _get_dummy_imu_params(self) -> tuple[int, int, list[float]]:
        """Generates dummy IMU camera parameters for COLMAP."""
        width = 640
        height = 480
        random_params = [
            241.604,
            241.604,
            322.895,
            240.444,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        return width, height, random_params

    def _add_device_sensors(self) -> None:
        """Adds a new rig with device calibrated sensors.
        The rig is consistent across all frames

        Camera ID layout:
        rig_id=1 -> (imu=1, left=2, right=3)
        """
        device_calibration = self._vrs_provider.get_device_calibration()
        imu_stream_label = self._vrs_provider.get_label_from_stream_id(
            self._right_imu_sid
        )
        imu_calib = device_calibration.get_imu_calib(imu_stream_label)

        rig = pycolmap.Rig(rig_id=1)

        w, h, p = self._get_dummy_imu_params()
        # DUMMY CAMERA FOR IMU, IMU ID is 1
        imu = pycolmap.Camera(
            camera_id=1,
            model=self.options.sensor.camera_model,
            width=w,
            height=h,
            params=p,
        )
        self.data.reconstruction.add_camera(imu)
        rig.add_ref_sensor(imu.sensor_id)

        for cam_id, sid in [(2, self._left_cam_sid), (3, self._right_cam_sid)]:
            stream_label = self._vrs_provider.get_label_from_stream_id(sid)
            camera_calib = device_calibration.get_camera_calib(stream_label)
            cam = camera_colmap_from_calib(camera_calib)
            cam.camera_id = cam_id

            t_imu_camera = get_t_imu_camera(
                imu_calib,
                camera_calib,
            )
            sensor_from_rig = t_imu_camera.inverse()

            self.data.reconstruction.add_camera(cam)
            rig.add_sensor(cam.sensor_id, sensor_from_rig)

        self.data.reconstruction.add_rig(rig)

    def _add_online_sensors(self) -> None:
        """Adds a new rig for each frame timestamp.
        Each rig has its own online calibrated sensors.

        Camera ID layout:
        rig_id=1 -> (imu=1, left=2, right=3);
        rig_id=2 -> (imu=4, left=5, right=6);
        rig_id=3 -> (imu=7, left=8, right=9);
        ...
        """
        sensor_id = 1
        for fid, pfd in self._per_frame_data.items():
            t = pfd.left_ts
            calibration = self._mps_data_provider.get_online_calibration(
                t, TimeQueryOptions.CLOSEST
            )
            if calibration is None:
                continue

            rig = pycolmap.Rig(rig_id=fid)
            w, h, p = self._get_dummy_imu_params()
            # DUMMY CAMERA FOR IMU
            imu = pycolmap.Camera(
                camera_id=sensor_id,
                model=self.options.sensor.camera_model,
                width=w,
                height=h,
                params=p,
            )
            self.data.reconstruction.add_camera(imu)
            rig.add_ref_sensor(imu.sensor_id)
            sensor_id += 1

            imu_stream_label = self._vrs_provider.get_label_from_stream_id(
                self._right_imu_sid
            )
            imu_calib = None
            for calib in calibration.imu_calibs:
                if calib.get_label() == imu_stream_label:
                    imu_calib = calib
                    break

            for sid in [self._left_cam_sid, self._right_cam_sid]:
                stream_label = self._vrs_provider.get_label_from_stream_id(sid)
                camera_calib = calibration.get_camera_calib(stream_label)
                cam = camera_colmap_from_calib(camera_calib)
                cam.camera_id = sensor_id
                sensor_id += 1

                t_imu_camera = get_t_imu_camera(
                    imu_calib,
                    camera_calib,
                )
                sensor_from_rig = t_imu_camera.inverse()

                self.data.reconstruction.add_camera(cam)
                rig.add_sensor(cam.sensor_id, sensor_from_rig)

            self.data.reconstruction.add_rig(rig)

    def _add_device_frames(self) -> None:
        """Adds frame data for each rig,
        all rigs share same device calibrated sensors
        """
        image_id = 1

        rig = self.data.reconstruction.rigs[1]
        for fid, pfd in self._per_frame_data.items():
            frame = pycolmap.Frame()
            frame.rig_id = rig.rig_id
            frame.frame_id = fid
            frame.rig_from_world = pfd.rig_from_world

            images_to_add = []
            for cam_id, img_path in [(2, pfd.left_img), (3, pfd.right_img)]:
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

            self.data.reconstruction.add_frame(frame)
            for im in images_to_add:
                self.data.reconstruction.add_image(im)

    def _add_online_frames(self) -> None:
        """Adds frame data for each rig, each rig has
        its own online calibrated sensors

        Frame ID layout:
        fid=1 -> (imu=1, left=2, right=3), (image IDs: 1, 2);
        fid=2 -> (imu=4, left=5, right=6), (image IDs: 3, 4);
        fid=3 -> (imu=7, left=8, right=9), (image IDs: 5, 6);
        ...
        """
        image_id = 1

        for fid, pfd in self._per_frame_data.items():
            frame = pycolmap.Frame()
            frame.rig_id = fid
            frame.frame_id = fid
            frame.rig_from_world = pfd.rig_from_world

            images_to_add = []
            # Camera ID layout from _add_online_sensors():
            # fid=1 -> (imu=1, left=2, right=3);
            # fid=2 -> (imu=4, left=5, right=6); ...
            left_cam_id = 3 * fid - 1
            right_cam_id = 3 * fid

            for cam_id, img_path in [
                (left_cam_id, pfd.left_img),
                (right_cam_id, pfd.right_img),
            ]:
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

            self.data.reconstruction.add_frame(frame)
            for im in images_to_add:
                self.data.reconstruction.add_image(im)

    def _get_rectified_imu_data(
        self,
        mps_folder: Path | None = None,
    ) -> pycolmap.ImuMeasurements:
        """Generates rectified IMU data from VRS file"""
        if self.options.mps.use_online_calibration:
            assert mps_folder is not None, (
                "MPS folder path must be provided if using MPS"
            )
            ms = get_imu_data_from_vrs(
                self._vrs_provider,
                mps_folder,
            )
        else:
            ms = get_imu_data_from_vrs(
                self._vrs_provider,
            )

        return ms
