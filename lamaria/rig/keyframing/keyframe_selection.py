from __future__ import annotations

import shutil
import numpy as np
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import pycolmap

from ... import logger
from ..config.loaders import load_cfg
from ...utils.utils import get_magnitude_from_transform

@dataclass
class KFParams:
    max_rotation_deg: float
    max_distance_m: float
    max_elapsed_ns: float


class KeyframeSelector:
    def __init__(self, reconstruction: pycolmap.Reconstruction, timestamps: List[int], cfg=None):
        self.init_recons = reconstruction
        self.timestamps = timestamps
        
        cfg = load_cfg() if cfg is None else cfg
        self.params = KFParams(
            max_rotation_deg=float(cfg.keyframing.max_rotation),
            max_distance_m=float(cfg.keyframing.max_distance),
            max_elapsed_ns=float(cfg.keyframing.max_elapsed) * 1e9,  # convert to ns
        )
        
        self.keyframes_dir = cfg.result.keyframes_path
        self.image_stream_root = cfg.image_stream_path

        self.keyframed_recons = pycolmap.Reconstruction()
        self.keyframe_frame_ids: Optional[List[int]] = None

        self.init_recon_type = "device" if len(self.init_recons.rigs) == 1 else "online"

    def _select_keyframes(self):
        self.keyframe_frame_ids = []  
        dr_dt = np.array([0.0, 0.0])
        dts = 0.0

        init_frame_ids = sorted(self.init_recons.frames.keys())

        for i, (prev, curr) in enumerate(zip(init_frame_ids[:-1], init_frame_ids[1:])):
            if i == 0:
                self.keyframe_frame_ids.append(prev)
                continue

            current_rig_from_world = self.init_recons.frames[curr].rig_from_world
            previous_rig_from_world = self.init_recons.frames[prev].rig_from_world
            current_rig_from_previous_rig = current_rig_from_world * previous_rig_from_world.inverse()

            dr_dt += np.array(get_magnitude_from_transform(current_rig_from_previous_rig))
            dts += self.timestamps[i+1] - self.timestamps[i]

            if dr_dt[0] > self.params.max_rotation_deg or \
                dr_dt[1] > self.params.max_distance_m or \
                    dts > self.params.max_elapsed_ns:
                
                self.keyframe_frame_ids.append(curr)
                dr_dt = np.array([0.0, 0.0])
                dts = 0.0
                
    def _build_device_keyframed_reconstruction(self):
        old_rig = self.init_recons.rigs[1]
        new_rig = pycolmap.Rig(rig_id=1)
        
        ref_sensor_id = old_rig.ref_sensor_id.id
        new_ref_sensor = deepcopy(self.init_recons.cameras[ref_sensor_id])
        self.keyframed_recons.add_camera(new_ref_sensor)
        new_rig.add_ref_sensor(new_ref_sensor.sensor_id)

        for sensor, sensor_from_rig in old_rig.sensors.items():
            new_sensor = deepcopy(self.init_recons.cameras[sensor.id])
            new_sensor_from_rig = deepcopy(sensor_from_rig)
            self.keyframed_recons.add_camera(new_sensor)
            new_rig.add_sensor(new_sensor.sensor_id, new_sensor_from_rig)
        
        self.keyframed_recons.add_rig(new_rig)

        image_id = 1
        for i, frame_id in enumerate(self.keyframe_frame_ids):
            old_frame = self.init_recons.frames[frame_id]
            
            new_frame = pycolmap.Frame()
            new_frame.rig_id = 1
            new_frame.frame_id = i + 1
            new_frame.rig_from_world = deepcopy(old_frame.rig_from_world)

            old_image_ids = sorted([d.id for d in old_frame.data_ids])

            images_to_add = []
            for old_image_id in old_image_ids:
                old_image = self.init_recons.images[old_image_id]

                new_image = pycolmap.Image(
                    old_image.name,
                    pycolmap.Point2DList(),
                    old_image.camera_id,
                    image_id,
                )

                new_image.frame_id = new_frame.frame_id
                new_frame.add_data_id(new_image.data_id)
                images_to_add.append(new_image)
                image_id += 1

            self.keyframed_recons.add_frame(new_frame)
            for img in images_to_add:
                self.keyframed_recons.add_image(img)

    def _clone_online_camera(self, old_camera_id: int, new_camera_id: int) -> pycolmap.Camera:
        old_camera = self.init_recons.cameras[old_camera_id]
        new_camera = deepcopy(old_camera)
        new_camera.camera_id = new_camera_id
        return new_camera

    def _build_online_keyframed_reconstruction(self):
        camera_id = 1
        image_id = 1
        rig_id = 1

        for i, frame_id in enumerate(self.keyframe_frame_ids):
            old_frame = self.init_recons.frames[frame_id]
            old_rig_id = old_frame.rig_id
            old_rig = self.init_recons.rigs[old_rig_id]

            cam_id_map = {}
            
            # IMU cosplaying as a dummy camera
            old_ref_sensor_id = old_rig.ref_sensor_id.id
            new_ref_sensor = self._clone_online_camera(old_ref_sensor_id, camera_id)
            self.keyframed_recons.add_camera(new_ref_sensor)
            cam_id_map[old_ref_sensor_id] = camera_id
            camera_id += 1

            new_rig = pycolmap.Rig(rig_id=rig_id)
            new_rig.add_ref_sensor(new_ref_sensor.sensor_id)

            for old_sensor, sensor_from_rig in old_rig.sensors.items():
                old_sensor_id = old_sensor.id

                if old_sensor_id not in cam_id_map:
                    new_sensor = self._clone_online_camera(old_sensor_id, camera_id)
                    self.keyframed_recons.add_camera(new_sensor)
                    cam_id_map[old_sensor_id] = camera_id
                    camera_id += 1
                
                new_sensor_from_rig = deepcopy(sensor_from_rig)
                new_rig.add_sensor(new_sensor.sensor_id, new_sensor_from_rig)

            self.keyframed_recons.add_rig(new_rig)

            new_frame = pycolmap.Frame()
            new_frame.rig_id = new_rig.rig_id
            new_frame.frame_id = i + 1
            new_frame.rig_from_world = deepcopy(old_frame.rig_from_world)

            old_image_ids = sorted([d.id for d in old_frame.data_ids])
            images_to_add = []
            for old_image_id in old_image_ids:
                old_image = self.init_recons.images[old_image_id]
                mapped_cam_id = cam_id_map[old_image.camera_id]

                new_image = pycolmap.Image(
                    old_image.name,
                    pycolmap.Point2DList(),
                    mapped_cam_id,
                    image_id,
                )
                new_image.frame_id = new_frame.frame_id
                new_frame.add_data_id(new_image.data_id)
                images_to_add.append(new_image)
                image_id += 1
            
            self.keyframed_recons.add_frame(new_frame)
            for img in images_to_add:
                self.keyframed_recons.add_image(img)
            
            rig_id += 1

    def run_keyframing(self) -> pycolmap.Reconstruction:
        self._select_keyframes()
        if self.init_recon_type == "device":
            self._build_device_keyframed_reconstruction()
        else:
            self._build_online_keyframed_reconstruction()
        
        logger.info(f"Created keyframed reconstruction with summary:")
        logger.info(self.keyframed_recons.summary())

        return self.keyframed_recons
    
    def copy_images_to_keyframes_dir(self, output_dir: Path = None) -> None:
        if self.keyframe_frame_ids is None:
            raise ValueError("Keyframes not selected yet. Run `run_keyframing` first.")

        if output_dir is None:
            output_dir = self.keyframes_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        for frame_id in self.keyframe_frame_ids:
            frame = self.init_recons.frames[frame_id]
            for data_id in frame.data_ids:
                image = self.init_recons.images[data_id.id]
                src_path = self.image_stream_root / image.name
                dst_path = output_dir / image.name
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

    def write_reconstruction(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(self.keyframed_recons.summary())
        self.keyframed_recons.write(output_path)


        