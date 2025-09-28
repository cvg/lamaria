import pyceres
from tqdm import tqdm
import numpy as np
import pycolmap

from ...config.options import OptIMUOptions
from .session import SingleSeqSession


def add_imu_residuals_to_problem(
    imu_options: OptIMUOptions,
    session: SingleSeqSession,
    problem,
):
    loss = pyceres.TrivialLoss()

    frame_ids = sorted(session.data.reconstruction.frames.keys())

    for k in tqdm(
        range(len(session.preintegrated_imu_measurements)),
        desc="Adding IMU residuals",
    ):
        i = frame_ids[k]
        j = frame_ids[k + 1]
        frame_i = session.data.reconstruction.frames[i]
        frame_j = session.data.reconstruction.frames[j]
        i_from_world = frame_i.rig_from_world
        j_from_world = frame_j.rig_from_world

        integrated_m = session.preintegrated_imu_measurements[i]

        problem.add_residual_block(
            pycolmap.PreintegratedImuMeasurementCost(integrated_m),
            loss,
            [
                session.imu_from_rig.rotation.quat,
                session.imu_from_rig.translation,
                session.log_scale,
                session.gravity,
                i_from_world.rotation.quat,
                i_from_world.translation,
                session.imu_states[i].data,
                j_from_world.rotation.quat,
                j_from_world.translation,
                session.imu_states[j].data,
            ],
        )

    problem = setup_manifolds_and_constraints(
        imu_options,
        session,
        problem,
    )

    return problem


def setup_manifolds_and_constraints(
    imu_options: OptIMUOptions,
    session: SingleSeqSession,
    problem,
):
    """Setup manifolds and parameter constraints"""
    problem.set_manifold(session.gravity, pyceres.SphereManifold(3))
    problem.set_manifold(
        session.imu_from_rig.rotation.quat, pyceres.EigenQuaternionManifold()
    )

    # Apply optimization constraints based on configuration
    if not imu_options.optimize_scale:
        problem.set_parameter_block_constant(session.log_scale)
    if not imu_options.optimize_gravity:
        problem.set_parameter_block_constant(session.gravity)
    if not imu_options.optimize_imu_from_rig:
        problem.set_parameter_block_constant(session.imu_from_rig.rotation.quat)
        problem.set_parameter_block_constant(session.imu_from_rig.translation)
    if not imu_options.optimize_bias:
        constant_idxs = np.arange(3, 9)
        for frame_id in session.imu_states:
            problem.set_manifold(
                session.imu_states[frame_id].data,
                pyceres.SubsetManifold(9, constant_idxs),
            )

    return problem