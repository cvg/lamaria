import tempfile
import unittest
from pathlib import Path

import numpy as np
import pycolmap

from evaluate_wrt_pgt import run
from lamaria.eval.pgt_evaluation import evaluate_wrt_pgt
from lamaria.structs.sparse_eval import SparseEvalResult
from lamaria.structs.trajectory import Trajectory, associate_trajectories


class PoseRecallTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_trajectory(self, name, indices, offset=0.0):
        path = self.directory / name
        path.write_text(
            "".join(
                f"{1_000_000_000 + i * 50_000_000} {i + offset} 0 0 0 0 0 1\n"
                for i in indices
            )
        )
        return path

    def load_trajectory(self, path):
        return Trajectory.load_from_file(path, invert_poses=False)

    def test_association_preserves_input_identity_and_order(self):
        for estimate_count, gt_count in [(3, 10), (10, 3), (3, 3)]:
            for estimate_first in [True, False]:
                with self.subTest(
                    estimate_count=estimate_count,
                    gt_count=gt_count,
                    estimate_first=estimate_first,
                ):
                    estimate = self.load_trajectory(
                        self.write_trajectory(
                            "estimate.txt", range(estimate_count), offset=100.0
                        )
                    )
                    gt = self.load_trajectory(
                        self.write_trajectory("gt.txt", range(gt_count))
                    )
                    first, second = (
                        (estimate, gt) if estimate_first else (gt, estimate)
                    )
                    associated_first, associated_second = (
                        associate_trajectories(first, second)
                    )
                    self.assertIs(associated_first, first)
                    self.assertIs(associated_second, second)
                    self.assertIsNot(associated_first, associated_second)
                    self.assertEqual(len(first), min(estimate_count, gt_count))
                    self.assertEqual(first.timestamps, second.timestamps)
                    np.testing.assert_allclose(
                        estimate.positions - gt.positions,
                        np.tile([100.0, 0.0, 0.0], (len(gt), 1)),
                    )

    def test_association_without_matches_does_not_filter_inputs(self):
        first = self.load_trajectory(
            self.write_trajectory("first.txt", range(3))
        )
        second = self.load_trajectory(
            self.write_trajectory("second.txt", range(100, 110))
        )
        self.assertEqual(associate_trajectories(first, second), (None, None))
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 10)

    def test_short_wrong_estimate_keeps_its_position_errors(self):
        estimate = self.load_trajectory(
            self.write_trajectory("estimate.txt", range(3), offset=100.0)
        )
        gt = self.load_trajectory(self.write_trajectory("gt.txt", range(10)))
        errors = evaluate_wrt_pgt(estimate, gt, pycolmap.Sim3d())
        np.testing.assert_allclose(errors, [100.0, 100.0, 100.0])

    def test_cli_recall_uses_all_original_gt_keyframes(self):
        gt_path = self.write_trajectory("gt.txt", range(10))
        alignment_path = self.directory / "alignment.npy"
        SparseEvalResult(alignment=pycolmap.Sim3d(), cp_summary={}).save_as_npy(
            alignment_path
        )
        cases = [
            (range(10), 0.0, 100.0),
            (range(10), 100.0, 0.0),
            (range(12), 0.0, 100.0),
            (range(12), 100.0, 0.0),
            (range(3), 0.0, 30.0),
            (range(3), 100.0, 0.0),
            ([0, 1, 2, *range(100, 107)], 0.0, 30.0),
            ([0, 1, 2, *range(100, 107)], 100.0, 0.0),
        ]
        for indices, offset, expected in cases:
            with self.subTest(indices=indices, offset=offset):
                estimate_path = self.write_trajectory(
                    "estimate.txt", indices, offset
                )
                with self.assertLogs("lamaria", level="INFO") as logs:
                    self.assertTrue(run(estimate_path, gt_path, alignment_path))
                for threshold in [1.0, 5.0]:
                    self.assertIn(
                        f"Pose Recall @ {threshold}m: {expected:.4f}",
                        "\n".join(logs.output),
                    )


if __name__ == "__main__":
    unittest.main()
