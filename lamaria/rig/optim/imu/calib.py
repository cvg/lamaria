import numpy as np
import pycolmap

def get_imu_calibration_parameters(
    gyro_infl: float = 1.0,
    acc_infl: float = 1.0
) -> pycolmap.ImuCalibration:

    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = 9.80600
    imu_calib.acc_noise_density = (
        0.8e-4 * imu_calib.gravity_magnitude * acc_infl
    )
    imu_calib.gyro_noise_density = (
        1e-2 * (np.pi / 180.0) * gyro_infl
    )
    imu_calib.acc_bias_random_walk_sigma = (
        3.5e-5
        * imu_calib.gravity_magnitude
        * np.sqrt(353)
        * acc_infl
    )  # accel_right BW = 353
    imu_calib.gyro_bias_random_walk_sigma = (
        1.3e-3 * (np.pi / 180.0) * np.sqrt(116) * gyro_infl
    )  # gyro_right BW = 116
    imu_calib.acc_saturation_max = 8.0 * imu_calib.gravity_magnitude
    imu_calib.gyro_saturation_max = 1000.0 * (np.pi / 180.0)
    imu_calib.imu_rate = 1000.0
    
    return imu_calib
