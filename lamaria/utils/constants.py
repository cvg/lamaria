from projectaria_tools.core.stream_id import StreamId

# ASL folder name, stream label for Aria cameras
ARIA_CAMERAS: list = [
    ("cam0", "camera-slam-left"),
    ("cam1", "camera-slam-right"),
]

# Aria camera constants
LEFT_CAMERA_STREAM_ID = StreamId("1201-1")
RIGHT_CAMERA_STREAM_ID = StreamId("1201-2")
LEFT_CAMERA_STREAM_LABEL = "camera-slam-left"
RIGHT_CAMERA_STREAM_LABEL = "camera-slam-right"

# Right Aria IMU constants
RIGHT_IMU_STREAM_ID = StreamId("1202-1")
RIGHT_IMU_STREAM_LABEL = "imu-right"

# Custom origin coordinates (LV95 / LN02) for translating large CP coordinates
CUSTOM_ORIGIN_COORDINATES = (2683594.4120000005, 1247727.7470000014, 417.307)
