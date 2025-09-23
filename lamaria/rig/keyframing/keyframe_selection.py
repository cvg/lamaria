from __future__ import annotations

import shutil
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import List, Optional

import pycolmap

from ..lamaria_reconstruction import LamariaReconstruction
from ..config.options import KeyframeSelectorOptions
from ...utils.transformation import get_magnitude_from_transform


class KeyframeSelector:
    def __init__(
        self,
        options: KeyframeSelectorOptions,
        data: LamariaReconstruction,
    ):
        self.options = options
        self.init_data: LamariaReconstruction = data
        self.init_recons = data.reconstruction # pycolmap.Reconstruction
        self.timestamps = data.timestamps # frame id to timestamp mapping

        self.keyframed_data: LamariaReconstruction = LamariaReconstruction()
        self.keyframe_frame_ids: Optional[List[int]] = None

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
            dts += self.timestamps[curr] - self.timestamps[prev]

            if dr_dt[0] > self.options.max_rotation or \
                dr_dt[1] > self.options.max_distance or \
                    dts > self.options.max_elapsed:
                
                self.keyframe_frame_ids.append(curr)
                dr_dt = np.array([0.0, 0.0])
                dts = 0.0
                
    def _build_device_keyframed_reconstruction(self):
        old_rig = self.init_recons.rigs[1]
        new_rig = pycolmap.Rig(rig_id=1)
        camera_id = 1
        cam_map = {} # new to old
        
        ref_sensor_id = old_rig.ref_sensor_id.id
        cam_map[ref_sensor_id] = camera_id
        new_ref_sensor = self._clone_camera(ref_sensor_id, camera_id)
        camera_id += 1
        self.keyframed_data.reconstruction.add_camera(new_ref_sensor)
        new_rig.add_ref_sensor(new_ref_sensor.sensor_id)

        for sensor, sensor_from_rig in old_rig.non_ref_sensors.items():
            new_sensor = self._clone_camera(sensor.id, camera_id)
            cam_map[sensor.id] = camera_id
            camera_id += 1
            new_sensor_from_rig = deepcopy(sensor_from_rig)
            self.keyframed_data.reconstruction.add_camera(new_sensor)
            new_rig.add_sensor(new_sensor.sensor_id, new_sensor_from_rig)
        
        self.keyframed_data.reconstruction.add_rig(new_rig)

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
                new_cam_id = cam_map[old_image.camera_id]

                new_image = pycolmap.Image(
                    old_image.name,
                    pycolmap.Point2DList(),
                    new_cam_id,
                    image_id,
                )

                new_image.frame_id = new_frame.frame_id
                new_frame.add_data_id(new_image.data_id)
                images_to_add.append(new_image)
                image_id += 1

            self.keyframed_data.reconstruction.add_frame(new_frame)
            for img in images_to_add:
                self.keyframed_data.reconstruction.add_image(img)

    def _clone_camera(self, old_camera_id: int, new_camera_id: int) -> pycolmap.Camera:
        old_camera = self.init_recons.cameras[old_camera_id]
        new_camera = pycolmap.Camera(
            model=old_camera.model,
            width=old_camera.width,
            height=old_camera.height,
            params=old_camera.params,
            camera_id=new_camera_id,
        )
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
            new_ref_sensor = self._clone_camera(old_ref_sensor_id, camera_id)
            self.keyframed_data.reconstruction.add_camera(new_ref_sensor)
            cam_id_map[old_ref_sensor_id] = camera_id
            camera_id += 1

            new_rig = pycolmap.Rig(rig_id=rig_id)
            new_rig.add_ref_sensor(new_ref_sensor.sensor_id)

            for old_sensor, sensor_from_rig in old_rig.non_ref_sensors.items():
                old_sensor_id = old_sensor.id

                if old_sensor_id not in cam_id_map:
                    new_sensor = self._clone_camera(old_sensor_id, camera_id)
                    self.keyframed_data.reconstruction.add_camera(new_sensor)
                    cam_id_map[old_sensor_id] = camera_id
                    camera_id += 1
                
                new_sensor_from_rig = deepcopy(sensor_from_rig)
                new_rig.add_sensor(new_sensor.sensor_id, new_sensor_from_rig)

            self.keyframed_data.reconstruction.add_rig(new_rig)

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
            
            self.keyframed_data.reconstruction.add_frame(new_frame)
            for img in images_to_add:
                self.keyframed_data.reconstruction.add_image(img)
            
            rig_id += 1

    def run_keyframing(self) -> LamariaReconstruction:
        """ Main function to run keyframing on input lamaria reconstruction."""
        self._select_keyframes()
        if len(self.init_recons.rigs.keys()) == 1: # device rig has been added
            self._build_device_keyframed_reconstruction()
        else:
            self._build_online_keyframed_reconstruction()
        
        # set timestamps keys as the new frame ids
        self.keyframed_data.timestamps = {
            frame_id: self.timestamps[frame_id]
            for frame_id in self.keyframe_frame_ids
        }

        self.keyframed_data.imu_measurements = deepcopy(self.init_data.imu_measurements)

        return self.keyframed_data

    def copy_images_to_keyframes_dir(
        self,
        images: Path,
        output: Optional[Path] = None,
    ) -> Path:
        """ Copy images corresponding to keyframes to a separate directory. 
        Images are expected to be in `images/left` and `images/right` subdirectories.
        Check: `extract_images_from_vrs` in `lamaria/utils/general.py` for more details.
        """
        if self.keyframe_frame_ids is None:
            raise ValueError("Keyframes not selected yet. Run `run_keyframing` first.")

        output_dir = output if output is not None else self.options.keyframes

        if output_dir.exists() and any(output_dir.iterdir()):
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)

        for frame_id in self.keyframe_frame_ids:
            frame = self.init_recons.frames[frame_id]
            for data_id in frame.data_ids:
                image = self.init_recons.images[data_id.id]
                
                subdir = "left" if "1201-1" in image.name else "right"
                src_path = images / subdir / image.name
                dst_path = output_dir / image.name
                
                shutil.copy2(src_path, dst_path)
        
        return output_dir