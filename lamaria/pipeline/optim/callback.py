from copy import deepcopy

import numpy as np
import pyceres
import pycolmap

from ... import logger


class RefinementCallback(pyceres.IterationCallback):
    def __init__(
        self,
        poses: list[pycolmap.Rigid3d],
        min_pose_change: tuple[float, float] = (0.001, 0.000001),
        min_iterations: int = 2,
    ):
        pyceres.IterationCallback.__init__(self)
        self.poses = poses
        self.poses_previous = deepcopy(self.poses)
        self.min_pose_change = min_pose_change
        self.min_iterations = min_iterations
        self.pose_changes = []

    def __call__(self, summary: pyceres.IterationSummary):
        if not summary.step_is_successful:
            return pyceres.CallbackReturnType.SOLVER_CONTINUE
        diff = []
        for pose_prev, pose in zip(self.poses_previous, self.poses):
            pose_rel = pose_prev * pose.inverse()
            q_rel, t_rel = pose_rel.rotation.quat, pose_rel.translation
            dr = np.rad2deg(
                np.abs(2 * np.arctan2(np.linalg.norm(q_rel[:-1]), q_rel[-1]))
            )
            dt = np.linalg.norm(t_rel)
            diff.append((dr, dt))
        diff = np.array(diff)
        self.poses_previous = deepcopy(self.poses)
        med, q99, max_ = np.quantile(diff, [0.5, 0.99, 1.0], axis=0)
        logger.info(
            f"{summary.iteration:d} Pose update: "
            f"med/q99/max dR={med[0]:.3f}/{q99[0]:.3f}/{max_[0]:.3f} deg, "
            f"dt={med[1] * 1e2:.3f}/{q99[1] * 1e2:.3f}/{max_[1] * 1e2:.3f} cm"
        )
        self.pose_changes.append((med, q99, max_))
        if summary.iteration >= self.min_iterations and np.all(
            q99 <= self.min_pose_change
        ):
            return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        return pyceres.CallbackReturnType.SOLVER_CONTINUE
