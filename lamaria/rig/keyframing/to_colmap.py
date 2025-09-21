from __future__ import annotations


from typing import List, Tuple
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
from ..config.options import ToColmapOptions
from ...utils.general import (
    find_closest_timestamp,
    get_t_imu_camera,
    camera_colmap_from_calib,
    rigid3d_from_transform,
    get_closed_loop_data_from_mps,
    get_mps_poses_for_timestamps,
    get_rig_from_worlds_from_estimate,
    extract_images_from_vrs,
    round_ns,
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


class ToColmap:
    def __init__(
        self,
        options: ToColmapOptions
    ):
        self.options = options
        self._init_io()
        self._init_data()

    def _init_io(self):
        """Initializes output and data providers"""
        self.empty_recons = pycolmap.Reconstruction()
        output_base = self.options.paths.init_model.parent
        output_base.mkdir(parents=True, exist_ok=True)
        
        self.vrs_provider = data_provider.create_vrs_data_provider(
            self.options.paths.vrs.as_posix()
        )
        if self.options.mps.use_mps:
            data_paths = mps.MpsDataPathsProvider(self.cfg.mps_path.as_posix()).get_data_paths()
            self.mps_data_provider = mps.MpsDataProvider(data_paths)
        else:
            assert self.options.paths.estimate is not None, \
                "Estimate path must be provided if MPS is not used"
    
    def _init_data(self):
        """Extracts images, timestamps and builds per-frame data"""

        extract_images_from_vrs(
            vrs_file=self.options.paths.vrs,
            image_folder=self.options.paths.images,
        )

        images = self._get_images()

        if self.options.mps.use_mps:
            timestamps = self._get_mps_timestamps()
            closed_loop_data = get_closed_loop_data_from_mps(self.options.paths.mps)
            pose_timestamps = [ l for l, _ in timestamps ]
            mps_poses = get_mps_poses_for_timestamps(closed_loop_data, pose_timestamps)
            self.per_frame_data = self._build_per_frame_data_from_mps(images, timestamps, mps_poses)
        else:
            timestamps = self._get_estimate_timestamps()
            if len(images) != len(timestamps):
                images, timestamps = self._match_estimate_ts_to_images(images, timestamps)
            
            rig_from_worlds = get_rig_from_worlds_from_estimate(
                self.options.paths.estimate,
            )
            self.per_frame_data = self._build_per_frame_data_from_estimate(images, timestamps, rig_from_worlds)
    
    def _build_per_frame_data_from_mps(self, images, timestamps, mps_poses) -> List[PerFrameData]:
        per_frame_data = []
        imu_stream_label = self.vrs_provider.get_label_from_stream_id(
            self.options.sensor.right_imu_stream_id
        )
        
        if not self.options.mps.use_online_calibration:
            device_calibration = self.vrs_provider.get_device_calibration()
            imu_calib = device_calibration.get_imu_calib(
                imu_stream_label
            )

        for (left_img, right_img), (left_ts, right_ts), t_world_device \
            in zip(images, timestamps, mps_poses):

            if t_world_device is None:
                continue

            if self.options.mps.use_online_calibration:
                ocalib = self.mps_data_provider.get_online_calibration(
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
            per_frame_data.append(pfd)

        return per_frame_data

    def _build_per_frame_data_from_estimate(self, images, timestamps, rig_from_worlds) -> List[PerFrameData]:
        per_frame_data = []
        assert len(images) == len(timestamps) == len(rig_from_worlds), \
            "Number of images, timestamps and poses must be equal"
        for (left_img, right_img), ts, rig_from_world \
            in zip(images, timestamps, rig_from_worlds):
            pfd = PerFrameData(
                left_ts=ts,
                right_ts=ts, # right timestamp is not available in estimate
                left_img=left_img,
                right_img=right_img,
                rig_from_world=rig_from_world
            )
            per_frame_data.append(pfd)
        
        return per_frame_data
    
    def _images_from_vrs(self, folder: Path, wrt_to: Path, ext: str =".jpg") -> List[Path]:
        if not folder.is_dir():
            return []
        images = sorted(n for n in folder.iterdir() if n.suffix == ext)
        images = [n.relative_to(wrt_to) for n in images]
        return images

    def _ts_from_vrs(self, sid: StreamId) -> List[int]:
        """Timestamps in nanoseconds"""
        return sorted(self.vrs_provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME))

    def _get_images(self) -> List[Tuple[Path, Path]]:
        left_img_dir = self.options.paths.images / "left"
        right_img_dir = self.options.paths.images / "right"

        left_images = self._images_from_vrs(left_img_dir, left_img_dir)
        right_images = self._images_from_vrs(right_img_dir, right_img_dir)

        return list(zip(left_images, right_images))

    def _get_mps_timestamps(self, max_diff=1e6) -> List[Tuple[int, int]]:
        if not self.options.mps.has_slam_drops:
            L = self._ts_from_vrs(self.options.sensor.left_cam_stream_id)
            R = self._ts_from_vrs(self.options.sensor.right_cam_stream_id)
            assert len(L) == len(R), "Unequal number of left and right timestamps"
            matched = list(zip(L, R))
            if not all(abs(l - r) < max_diff for l, r in matched):
                logger.warning(
                    f"Left and right timestamps differ by more than {max_diff} ns"
                )
        else:
            matched = self._match_timestamps(max_diff)

        return matched
    
    def _get_estimate_timestamps(self):
        assert self.options.paths.estimate is not None, \
            "Estimate path must be provided if MPS is not used"
        
        with open(self.options.paths.estimate, 'r') as f:
            lines = f.readlines()
        
        timestamps = []
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            ts = round_ns(line.split()[0])
            timestamps.append(ts)
        
        return sorted(timestamps)
    
    def _match_estimate_ts_to_images(
        self,
        images: List[Tuple[Path, Path]],
        est_timestamps: List[int],
        max_diff: int = 1000000, # 1 ms
    ) -> Tuple[List[Tuple[Path, Path]], List[int]]:
        
        left_ts = self._ts_from_vrs(self.options.sensor.left_cam_stream_id)
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
    
    def _match_timestamps(self, max_diff=1e6) -> List[Tuple[int, int]]:
        L = self._ts_from_vrs(self.options.sensor.left_cam_stream_id)
        R = self._ts_from_vrs(self.options.sensor.right_cam_stream_id)
        
        # matching timestamps is only when we have slam drops
        assert len(L) > 0 and len(R) > 0 and len(L) != len(R), \
            "Streams must have data and unequal lengths"
        
        matched = []

        if len(L) < len(R):
            for l in L:
                r = find_closest_timestamp(R, l, max_diff)
                if r is not None:
                    matched.append((l, r))
        else:
            for r in R:
                l = find_closest_timestamp(L, r, max_diff)
                if l is not None:
                    matched.append((l, r))
        return matched

    def _get_dummy_imu_params(self) -> List:
        # Dummy values for IMU "camera"
        width = 640
        height = 480
        random_params = [241.604, 241.604, 322.895, 240.444, \
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                                0.0, 0.0, 0.0, 0.0, 0.0]
        
        return width, height, random_params

    def _add_device_sensors(self) -> None:
        device_calibration = self.vrs_provider.get_device_calibration()
        imu_stream_label = self.vrs_provider.get_label_from_stream_id(
            self.options.sensor.right_imu_stream_id
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
        self.empty_recons.add_camera(imu)
        rig.add_ref_sensor(imu.sensor_id)

        for cam_id, sid in \
            [(2, self.options.sensor.left_cam_stream_id),
                (3, self.options.sensor.right_cam_stream_id)
        ]:
            stream_label = self.vrs_provider.get_label_from_stream_id(
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
            
            self.empty_recons.add_camera(cam)
            rig.add_sensor(cam.sensor_id, sensor_from_rig)
        
        self.empty_recons.add_rig(rig)

    def _add_online_sensors(self) -> None:
        """Adds a new rig for each timestamp, with sensors calibrated online"""
        sensor_id = 1
        for id, pfd in enumerate(self.per_frame_data):
            t = pfd.left_ts
            calibration = self.mps_data_provider.get_online_calibration(
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
            self.empty_recons.add_camera(imu)
            rig.add_ref_sensor(imu.sensor_id)
            sensor_id += 1

            imu_stream_label = self.vrs_provider.get_label_from_stream_id(
                self.options.sensor.right_imu_stream_id
            )
            imu_calib = None
            for calib in calibration.imu_calibs:
                if calib.get_label() == imu_stream_label:
                    imu_calib = calib
                    break

            for sid in \
                [self.options.sensor.left_cam_stream_id, self.options.sensor.right_cam_stream_id
            ]:
                stream_label = self.vrs_provider.get_label_from_stream_id(
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

                self.empty_recons.add_camera(cam)
                rig.add_sensor(cam.sensor_id, sensor_from_rig)

            self.empty_recons.add_rig(rig)

    def _add_device_frames(self) -> None:
        image_id = 1

        rig = self.empty_recons.rigs[1]
        for id, pfd in enumerate(self.per_frame_data):
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

            self.empty_recons.add_frame(frame)
            for im in images_to_add:
                self.empty_recons.add_image(im)

    def _add_online_frames(self) -> None:
        image_id = 1

        for id, pfd in enumerate(self.per_frame_data):
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
            
            self.empty_recons.add_frame(frame)
            for im in images_to_add:
                self.empty_recons.add_image(im)

    def save_imu_measurements(self) -> Path:
        """Saves rectified IMU measurements to a numpy file"""
        if self.options.mps.use_online_calibration \
            and self.options.mps.use_mps:
            ms = get_imu_data_from_vrs(
                self.vrs_provider,
                self.options.paths.mps,
            )
        else:
            ms = get_imu_data_from_vrs(
                self.vrs_provider,
            )
        
        np.save(self.options.paths.rect_imu, ms.data)

        return self.options.paths.rect_imu
    
    def create(self) -> pycolmap.Reconstruction:
        """Creates an empty COLMAP reconstruction with cameras and frames"""
        if self.options.mps.use_online_calibration:
            self._add_online_sensors()
            self._add_online_frames()
        else:
            self._add_device_sensors()
            self._add_device_frames()

        return self.empty_recons

    def write_reconstruction(self) -> Path:
        recon_path = self.options.paths.init_model
        recon_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Init {self.empty_recons.summary()}")
        self.empty_recons.write(recon_path.as_posix())

        return recon_path

    def write_full_timestamps(self) -> Path:
        ts_path = self.options.paths.full_ts
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        left_ts = np.array(sorted([pfd.left_ts for pfd in self.per_frame_data]))
        np.save(ts_path, left_ts)

        return ts_path
