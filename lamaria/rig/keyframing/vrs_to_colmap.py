from __future__ import annotations


from typing import List, Tuple
from pathlib import Path
import numpy as np
import pycolmap
from dataclasses import dataclass

import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from ... import logger
from ..config.loaders import load_cfg
from ...utils.general import (
    find_closest_timestamp,
    get_t_imu_camera,
    camera_colmap_from_calib,
    rigid3d_from_transform,
    get_closed_loop_data_from_mps,
    get_mps_poses_for_timestamps,
)

@dataclass
class PerFrameData:
    left_ts: int
    right_ts: int
    left_img: Path
    right_img: Path
    rig_from_world: pycolmap.Rigid3d


class VrsToColmap:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else load_cfg()
        self._init_io()
        self._init_data()

    def _init_io(self):
        self.empty_recons = pycolmap.Reconstruction()
        self.vrs_provider = data_provider.create_vrs_data_provider(
            self.cfg.vrs_file_path.as_posix()
        )
        data_paths = mps.MpsDataPathsProvider(self.cfg.mps_path.as_posix()).get_data_paths()
        self.mps_data_provider = mps.MpsDataProvider(data_paths)

        self.left_cam_stream_id = StreamId(self.cfg.streams.left_cam_stream_id)
        self.right_cam_stream_id = StreamId(self.cfg.streams.right_cam_stream_id)
        self.imu_stream_id = StreamId(self.cfg.streams.imu_right_stream_id)
    
    def _init_data(self):
        images = self._get_images()
        timestamps = self._get_timestamps()
        closed_loop_data = get_closed_loop_data_from_mps(self.cfg.mps_path)
        pose_timestamps = [ l for l, _ in timestamps ]
        mps_poses = get_mps_poses_for_timestamps(closed_loop_data, pose_timestamps)

        self.per_frame_data = self._build_per_frame_data(images, timestamps, mps_poses)

    def _build_per_frame_data(self, images, timestamps, mps_poses) -> List[PerFrameData]:
        per_frame_data = []
        imu_stream_label = self.vrs_provider.get_label_from_stream_id(
            self.imu_stream_id
        )
        
        if self.cfg.optimization.use_device_calibration:
            device_calibration = self.vrs_provider.get_device_calibration()
            imu_calib = device_calibration.get_imu_calib(
                imu_stream_label
            )

        for (left_img, right_img), (left_ts, right_ts), t_world_device \
            in zip(images, timestamps, mps_poses):

            if t_world_device is None:
                continue

            if not self.cfg.optimization.use_device_calibration:
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
        left_img_dir = self.cfg.image_stream_path / "left"
        right_img_dir = self.cfg.image_stream_path / "right"

        left_images = self._images_from_vrs(left_img_dir, left_img_dir)
        right_images = self._images_from_vrs(right_img_dir, right_img_dir)

        return list(zip(left_images, right_images))

    def _get_timestamps(self, max_diff=1e6) -> List[Tuple[int, int]]:
        if not self.cfg.flags.has_slam_drops:
            L = self._ts_from_vrs(self.left_cam_stream_id)
            R = self._ts_from_vrs(self.right_cam_stream_id)
            assert len(L) == len(R), "Unequal number of left and right timestamps"
            matched = list(zip(L, R))
            if not all(abs(l - r) < max_diff for l, r in matched):
                logger.warning(
                    f"Left and right timestamps differ by more than {max_diff} ns"
                )
        else:
            matched = self._match_timestamps(max_diff)

        return matched
    
    def _match_timestamps(self, max_diff=1e6) -> List[Tuple[int, int]]:
        L = self._ts_from_vrs(self.left_cam_stream_id)
        R = self._ts_from_vrs(self.right_cam_stream_id)
        
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
            self.imu_stream_id
        )
        imu_calib = device_calibration.get_imu_calib(
            imu_stream_label
        )
        
        rig = pycolmap.Rig(rig_id=1)

        w, h, p = self._get_dummy_imu_params()
        # DUMMY CAMERA FOR IMU, IMU ID is 1
        imu = pycolmap.Camera(
            camera_id=1,
            model=self.cfg.camera.model,
            width=w,
            height=h,
            params=p,
        )
        self.empty_recons.add_camera(imu)
        rig.add_ref_sensor(imu.sensor_id)

        for cam_id, sid in [(2, self.left_cam_stream_id), (3, self.right_cam_stream_id)]:
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
                model=self.cfg.camera.model,
                width=w,
                height=h,
                params=p,
            )
            self.empty_recons.add_camera(imu)
            rig.add_ref_sensor(imu.sensor_id)
            sensor_id += 1

            imu_stream_label = self.vrs_provider.get_label_from_stream_id(
                self.imu_stream_id
            )
            imu_calib = None
            for calib in calibration.imu_calibs:
                if calib.get_label() == imu_stream_label:
                    imu_calib = calib
                    break

            for sid in [self.left_cam_stream_id, self.right_cam_stream_id]:
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

    def create(self) -> pycolmap.Reconstruction:
        if self.cfg.optimization.use_device_calibration:
            self._add_device_sensors()
            self._add_device_frames()
        else:
            self._add_online_sensors()
            self._add_online_frames()
        
        return self.empty_recons

    def write_reconstruction(self, recon: pycolmap.Reconstruction, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Init {recon.summary()}")
        recon.write(output_path)

    def write_full_timestamps(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        left_ts = np.array(sorted([pfd.left_ts for pfd in self.per_frame_data]))
        np.save(output_path, left_ts)
