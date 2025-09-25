
import json
import numpy as np
import pycolmap
from pathlib import Path
from typing import List

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.calibration import CameraCalibration

from .transformation import get_t_cam_a_cam_b_from_json


ARIA_CAMERAS = [("cam0", "camera-slam-left"), ("cam1", "camera-slam-right")]
LEFT_CAMERA_STREAM_ID = StreamId("1201-1")
RIGHT_CAMERA_STREAM_ID = StreamId("1201-2")
LEFT_CAMERA_STREAM_LABEL = "camera-slam-left"
RIGHT_CAMERA_STREAM_LABEL = "camera-slam-right"



def add_cameras_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    calibration_file: Path,
) -> None:
    """ Add Aria cameras to COLMAP reconstruction from calibration json file found on website:
    https://lamaria.ethz.ch/slam_datasets
    
    Args:
        reconstruction (pycolmap.Reconstruction): The COLMAP reconstruction to which cameras will be added
        calibration_file (Path): Path to the Aria calibration json file
    """

    for i, (key, _) in enumerate(ARIA_CAMERAS):
        cam = camera_colmap_from_json(
            calibration_file=calibration_file,
            camera_label=key,
        )
        cam.camera_id = i + 1
        reconstruction.add_camera(cam)

    rig = pycolmap.Rig(rig_id=1)
    ref_sensor = pycolmap.sensor_t(
        id=1,  # left camera is the rig
        type=pycolmap.SensorType.CAMERA,
    )
    rig.add_ref_sensor(ref_sensor)

    sensor1 = pycolmap.sensor_t(id=2, type=pycolmap.SensorType.CAMERA)
    sensor_from_rig = get_t_cam_a_cam_b_from_json(
        calibration_file=calibration_file,
        camera_a_label="cam1", # right
        camera_b_label="cam0", # left
    )
    rig.add_sensor(sensor1, sensor_from_rig)
    
    reconstruction.add_rig(rig)


def get_camera_params_for_colmap(
    camera_calibration: CameraCalibration,
    camera_model: str,
) -> List[float]:
    """ Convert Aria CameraCalibration to COLMAP camera parameters.
    Supported models: OPENCV_FISHEYE, FULL_OPENCV, RAD_TAN_THIN_PRISM_FISHEYE
"""
    # params = [f_u {f_v} c_u c_v [k_0: k_{numK-1}]
    # {p_0 p_1} {s_0 s_1 s_2 s_3}]
    # projection_params is a 15 length vector,
    # starting with focal length, pp, extra coeffs
    camera_params = camera_calibration.get_projection_params()
    f_x, f_y, c_x, c_y = (
        camera_params[0],
        camera_params[0],
        camera_params[1],
        camera_params[2],
    )

    p2, p1 = camera_params[-5], camera_params[-6]

    k1, k2, k3, k4, k5, k6 = camera_params[3:9]

    # FULL_OPENCV model format:
    # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

    # OPENCV_FISHEYE model format:
    # fx, fy, cx, cy, k1, k2, k3, k4

    if camera_model == "OPENCV_FISHEYE":
        params = [f_x, f_y, c_x, c_y, k1, k2, k3, k4]
    elif camera_model == "FULL_OPENCV":
        params = [f_x, f_y, c_x, c_y, k1, k2, p1, p2, k3, k4, k5, k6]
    elif camera_model == "RAD_TAN_THIN_PRISM_FISHEYE":
        aria_fisheye_params = camera_params
        focal_length = aria_fisheye_params[0]
        aria_fisheye_params = np.insert(aria_fisheye_params, 0, focal_length)
        params = aria_fisheye_params

    return params


def camera_colmap_from_calib(calib: CameraCalibration) -> pycolmap.Camera:
    """Loads pycolmap camera from Aria CameraCalibration object"""
    if calib.get_model_name().name != "FISHEYE624":
        raise ValueError(f"Unsupported Aria model {calib.get_model_name().name}")
    model = "RAD_TAN_THIN_PRISM_FISHEYE"
    params = get_camera_params_for_colmap(calib, model)
    width, height = calib.get_image_size()
    return pycolmap.Camera(
        model=model,
        width=width,
        height=height,
        params=params,
    )


def camera_colmap_from_json(
    calibration_file: Path, # open sourced aria calibration files
    camera_label: str
) -> pycolmap.Camera:
    """Loads pycolmap camera from Aria calibration json file found on website:
    https://lamaria.ethz.ch/slam_datasets"""

    calib = json.load(calibration_file.open("r"))
    camera_data = calib[camera_label]
    if camera_data["model"] != "RAD_TAN_THIN_PRISM_FISHEYE":
        raise ValueError(f"Unsupported Aria model {camera_data['model']}")
    
    model = "RAD_TAN_THIN_PRISM_FISHEYE"
    params = camera_data["params"]
    width = camera_data["resolution"]["width"]
    height = camera_data["resolution"]["height"]
    return pycolmap.Camera(
        model=model,
        width=width,
        height=height,
        params=params,
    )