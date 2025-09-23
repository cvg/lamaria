from __future__ import annotations


from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import pycolmap
from dataclasses import dataclass
from bisect import bisect_left

import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from ... import logger
from ..lamaria_reconstruction import LamariaReconstruction
from ..config.options import EstimateToColmapOptions
from ...utils.general import (
    get_matched_timestamps,
    extract_images_from_vrs,
)
from ...utils.transformation import (
    get_t_imu_camera,
    rigid3d_from_transform,
    get_closed_loop_data_from_mps,
    get_mps_poses_for_timestamps,
)
from ...utils.camera import (
    camera_colmap_from_calib,
)
from ...utils.estimate import (
    check_estimate_format,
    get_estimate_timestamps,
    get_rig_from_worlds_from_estimate,
)
from ...utils.imu import (
    get_imu_data_from_vrs,
)

@dataclass
class PerFrameData:
    left_ts: int
    right_ts: int
    left_img: Path
    right_img: Path
    rig_from_world: pycolmap.Rigid3d


class EstimateToColmap:
    def __init__(
        self,
        options: EstimateToColmapOptions
    ):
        self.options = options
        self._init_io()
        self._init_data()

    def _init_io(self):
        """Initializes output and data providers"""
        self.data = LamariaReconstruction()
        self._vrs_provider = data_provider.create_vrs_data_provider(
            self.options.vrs.as_posix()
        )

        # Initialize stream IDs
        self._left_cam_sid = StreamId(self.options.sensor.left_cam_stream_id)
        self._right_cam_sid = StreamId(self.options.sensor.right_cam_stream_id)
        self._right_imu_sid = StreamId(self.options.sensor.right_imu_stream_id)
        
        # Initialize MPS data provider if needed
        if self.options.mps.use_mps:
            data_paths = mps.MpsDataPathsProvider(self.options.mps_folder.as_posix()).get_data_paths()
            self._mps_data_provider = mps.MpsDataProvider(data_paths)
    
    def _init_data(self):
        """Extracts images, timestamps and builds per-frame data object.
        Per-frame data is used to create the initial Lamaria reconstruction.
        """

        extract_images_from_vrs(
            vrs_file=self.options.vrs,
            image_folder=self.options.images,
        )

        images = self._get_images()

        if self.options.mps.use_mps:
            timestamps = self._get_mps_timestamps()
            closed_loop_data = get_closed_loop_data_from_mps(self.options.mps)
            pose_timestamps = [ l for l, _ in timestamps ]
            mps_poses = get_mps_poses_for_timestamps(closed_loop_data, pose_timestamps)
            self._per_frame_data = self._build_per_frame_data_from_mps(
                images,
                timestamps,
                mps_poses
            )
        else:
            flag = check_estimate_format(self.options.estimate)
            if not flag:
                raise ValueError("Estimate file format is incorrect.")

            timestamps = get_estimate_timestamps(self.options.estimate)
            if len(images) != len(timestamps):
                images, timestamps = self._match_estimate_ts_to_images(images, timestamps)
            
            rig_from_worlds = get_rig_from_worlds_from_estimate(
                self.options.estimate,
            )
            self._per_frame_data = self._build_per_frame_data_from_estimate(
                images,
                timestamps,
                rig_from_worlds
            )

    def _build_per_frame_data_from_mps(
        self,
        images,
        timestamps,
        mps_poses
    ) -> Dict[int, PerFrameData]:
        per_frame_data: Dict[int, PerFrameData] = {}
        imu_stream_label = self._vrs_provider.get_label_from_stream_id(
            self._right_imu_sid
        )
        
        if not self.options.mps.use_online_calibration:
            device_calibration = self._vrs_provider.get_device_calibration()
            imu_calib = device_calibration.get_imu_calib(
                imu_stream_label
            )

        frame_id = 1

        for (left_img, right_img), (left_ts, right_ts), t_world_device \
            in zip(images, timestamps, mps_poses):

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
                rig_from_world=rig_from_world
            )
            per_frame_data[frame_id] = pfd
            frame_id += 1

        return per_frame_data

    def _build_per_frame_data_from_estimate(
        self,
        images,
        timestamps,
        rig_from_worlds
    ) -> Dict[int, PerFrameData]:
        per_frame_data: Dict[int, PerFrameData] = {}
        assert len(images) == len(timestamps) == len(rig_from_worlds), \
            "Number of images, timestamps and poses must be equal"
        
        frame_id = 1
        
        for (left_img, right_img), ts, rig_from_world \
            in zip(images, timestamps, rig_from_worlds):
            pfd = PerFrameData(
                left_ts=ts,
                right_ts=ts,
                left_img=left_img,
                right_img=right_img,
                rig_from_world=rig_from_world
            )
            per_frame_data[frame_id] = pfd
            frame_id += 1
        
        return per_frame_data
    
    def _images_from_vrs(self, folder: Path, wrt_to: Path, ext: str =".jpg") -> List[Path]:
        if not folder.is_dir():
            return []
        images = sorted(n for n in folder.iterdir() if n.suffix == ext)
        images = [n.relative_to(wrt_to) for n in images]
        return images

    def _ts_from_vrs(self, sid: StreamId) -> List[int]:
        """Timestamps in nanoseconds"""
        return sorted(self._vrs_provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME))

    def _get_images(self) -> List[Tuple[Path, Path]]:
        left_img_dir = self.options.images / "left"
        right_img_dir = self.options.images / "right"

        left_images = self._images_from_vrs(left_img_dir, left_img_dir)
        right_images = self._images_from_vrs(right_img_dir, right_img_dir)

        return list(zip(left_images, right_images))

    def _get_mps_timestamps(self, max_diff=1e6) -> List[Tuple[int, int]]:
        if not self.options.mps.has_slam_drops:
            L = self._ts_from_vrs(self._left_cam_sid)
            R = self._ts_from_vrs(self._right_cam_sid)
            assert len(L) == len(R), "Unequal number of left and right timestamps"
            matched = list(zip(L, R))
            if not all(abs(l - r) < max_diff for l, r in matched):
                logger.warning(
                    f"Left and right timestamps differ by more than {max_diff} ns"
                )
        else:
            matched = get_matched_timestamps(L, R, max_diff)

        return matched
    
    def _match_estimate_ts_to_images(
        self,
        images: List[Tuple[Path, Path]],
        est_timestamps: List[int],
        max_diff: int = 1000000, # 1 ms
    ) -> Tuple[List[Tuple[Path, Path]], List[int]]:

        left_ts = self._ts_from_vrs(self._left_cam_sid)
        assert len(images) == len(left_ts), \
            "Number of images and left timestamps must be equal"
        
        order = sorted(range(len(left_ts)), key=lambda i: left_ts[i])
        left_ts = [left_ts[i] for i in order]
        images = [images[i] for i in order]
        
        matched_images: List[Tuple[Path, Path]] = []
        matched_timestamps: List[int] = []

        # estimate timestamps will be in nanoseconds like vrs timestamps
        for est in est_timestamps:
            idx = bisect_left(left_ts, est)

            cand_idxs = []
            if idx > 0: cand_idxs.append(idx - 1)
            if idx < len(left_ts): cand_idxs.append(idx)

            if not cand_idxs:
                continue

            best = min(cand_idxs, key=lambda j: abs(left_ts[j] - est))
            if (max_diff is not None) and (abs(left_ts[best] - est) > max_diff):
                continue

            matched_images.append(images[best])
            matched_timestamps.append(left_ts[best])
        
        return matched_images, matched_timestamps

    def _get_dummy_imu_params(self) -> List:
        # Dummy values for IMU "camera"
        width = 640
        height = 480
        random_params = [241.604, 241.604, 322.895, 240.444, \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                                0.0, 0.0, 0.0, 0.0, 0.0]
        
        return width, height, random_params

    def _add_device_sensors(self) -> None:
        """Adds a new rig with device calibrated sensors.
        The rig is consistent across all frames"""
        device_calibration = self._vrs_provider.get_device_calibration()
        imu_stream_label = self._vrs_provider.get_label_from_stream_id(
            self._right_imu_sid
        )
        imu_calib = device_calibration.get_imu_calib(
            imu_stream_label
        )
        
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

        for cam_id, sid in \
            [(2, self._left_cam_sid),
                (3, self._right_cam_sid)
        ]:
            stream_label = self._vrs_provider.get_label_from_stream_id(
                sid
            )
            camera_calib = device_calibration.get_camera_calib(
                stream_label
            )
            cam = camera_colmap_from_calib(camera_calib)
            cam.camera_id = cam_id

            t_imu_camera = get_t_imu_camera(
                imu_calib,
                camera_calib,
            )
            t_camera_imu = t_imu_camera.inverse()
            sensor_from_rig = rigid3d_from_transform(t_camera_imu)
            
            self.data.reconstruction.add_camera(cam)
            rig.add_sensor(cam.sensor_id, sensor_from_rig)
        
        self.data.reconstruction.add_rig(rig)

    def _add_online_sensors(self) -> None:
        """Adds a new rig for each frame timestamp.
        Each rig has its own online calibrated sensors.
        """
        sensor_id = 1
        for id, pfd in enumerate(self._per_frame_data):
            t = pfd.left_ts
            calibration = self._mps_data_provider.get_online_calibration(
                t, TimeQueryOptions.CLOSEST
            )
            if calibration is None:
                continue

            rig = pycolmap.Rig(rig_id=id + 1)
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

            for sid in \
                [self._left_cam_sid, self._right_cam_sid
            ]:
                stream_label = self._vrs_provider.get_label_from_stream_id(
                    sid
                )
                camera_calib = calibration.get_camera_calib(
                    stream_label
                )
                cam = camera_colmap_from_calib(camera_calib)
                cam.camera_id = sensor_id
                sensor_id += 1

                t_imu_camera = get_t_imu_camera(
                    imu_calib,
                    camera_calib,
                )
                t_camera_imu = t_imu_camera.inverse()
                sensor_from_rig = rigid3d_from_transform(t_camera_imu)

                self.data.reconstruction.add_camera(cam)
                rig.add_sensor(cam.sensor_id, sensor_from_rig)

            self.data.reconstruction.add_rig(rig)

    def _add_device_frames(self) -> None:
        """Adds frame data for each rig, all rigs share same device calibrated sensors"""
        image_id = 1

        rig = self.data.reconstruction.rigs[1]
        for id, pfd in self._per_frame_data.items():
            frame = pycolmap.Frame()
            frame.rig_id = rig.rig_id
            frame.frame_id = id + 1  # unique id
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
        """Adds frame data for each rig, each rig has its own online calibrated sensors"""
        image_id = 1

        for id, pfd in self._per_frame_data.items():
            frame = pycolmap.Frame()
            frame.rig_id = id + 1
            frame.frame_id = id + 1
            frame.rig_from_world = pfd.rig_from_world

            images_to_add = []

            for cam_id, img_path in [(3*id + 2, pfd.left_img), (3*id + 3, pfd.right_img)]:
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

    def _get_rectified_imu_data(self) -> Path:
        """Generates rectified IMU data from VRS file"""
        if self.options.mps.use_online_calibration \
            and self.options.mps.use_mps:
            ms = get_imu_data_from_vrs(
                self._vrs_provider,
                self.options.paths.mps,
            )
        else:
            ms = get_imu_data_from_vrs(
                self._vrs_provider,
            )

        return ms
    
    def create(self) -> LamariaReconstruction:
        """Creates an empty COLMAP reconstruction with cameras and frames"""
        if self.options.mps.use_online_calibration  \
            and self.options.mps.use_mps:
            self._add_online_sensors()
            self._add_online_frames()
        else:
            self._add_device_sensors()
            self._add_device_frames()
        
        ms = self._get_rectified_imu_data()
        self.data.imu_measurements = ms

        return self.data
