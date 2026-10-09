import tempfile
import unittest
from pathlib import Path

from evaluate_wrt_mps import run
from lamaria.eval.evo_evaluation import evaluate_wrt_mps
from lamaria.structs.trajectory import Trajectory
from tests.synthetic_scene import NUM_FRAMES, SyntheticScene


def load(path: Path) -> Trajectory:
    return Trajectory.load_from_file(path, invert_poses=False)


class MpsEvaluationTest(unittest.TestCase):
    def test_partial_estimate_must_cover_half_the_duration(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        directory = Path(directory.name)
        scene = SyntheticScene()
        ground_truth = scene.write_ground_truth(directory / "gt.txt")

        # The estimate lives in a different similarity frame; evo's
        # alignment has to absorb it completely.
        estimate = scene.write_estimate(directory / "estimate.txt")
        with self.assertLogs("lamaria", level="INFO") as logs:
            self.assertTrue(run(estimate, ground_truth))
        self.assertIn("ATE RMSE: 0.0000 m", "\n".join(logs.output))

        # Duration is measured between the first and last timestamp, so
        # covering half of it takes one frame more than half the frames.
        long_enough = scene.write_estimate(
            directory / "long_enough.txt", indices=range(NUM_FRAMES // 2 + 1)
        )
        self.assertLess(
            evaluate_wrt_mps(load(long_enough), load(ground_truth)), 1e-6
        )

        too_short = scene.write_estimate(
            directory / "too_short.txt", indices=range(NUM_FRAMES // 2)
        )
        with self.assertLogs("lamaria", level="ERROR") as logs:
            self.assertIsNone(
                evaluate_wrt_mps(load(too_short), load(ground_truth))
            )
            self.assertFalse(run(too_short, ground_truth))
        self.assertIn("too short", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
