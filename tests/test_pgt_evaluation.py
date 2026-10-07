import re
import tempfile
import unittest
from pathlib import Path

import evaluate_wrt_control_points
from evaluate_wrt_pgt import run
from lamaria.structs.sparse_eval import SparseEvalResult
from tests.synthetic_scene import NUM_FRAMES, SyntheticScene

RECALL_PATTERN = re.compile(r"Pose Recall @ ([0-9.]+)m: ([0-9.]+)")


class PgtEvaluationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.scene = SyntheticScene()
        self.ground_truth = self.scene.write_ground_truth(
            self.directory / "gt.txt"
        )

    def recall(self, estimate: Path, alignment: Path) -> dict[float, float]:
        with self.assertLogs("lamaria", level="INFO") as logs:
            self.assertTrue(run(estimate, self.ground_truth, alignment))
        found = RECALL_PATTERN.findall("\n".join(logs.output))
        self.assertEqual(len(found), 2)
        return {float(threshold): float(value) for threshold, value in found}

    def test_recall_is_the_fraction_of_ground_truth_covered_by_the_estimate(
        self,
    ):
        # The alignment the control point evaluation would have produced.
        alignment = self.directory / "sparse_eval_result.npy"
        SparseEvalResult(
            alignment=self.scene.topo_from_est, cp_summary={}
        ).save_as_npy(alignment)

        cases = [
            (None, 100.0),
            (range(NUM_FRAMES // 2), 50.0),
            (range(0, NUM_FRAMES, 4), 25.0),
        ]
        for indices, expected in cases:
            with self.subTest(indices=indices):
                estimate = self.scene.write_estimate(
                    self.directory / "estimate.txt", indices=indices
                )
                self.assertEqual(
                    self.recall(estimate, alignment),
                    {1.0: expected, 5.0: expected},
                )

    def test_alignment_saved_by_the_control_point_evaluation_is_consumed(
        self,
    ):
        estimate = self.scene.write_estimate(self.directory / "estimate.txt")
        output = self.directory / "eval_cp"
        self.assertTrue(
            evaluate_wrt_control_points.run(
                estimate,
                self.scene.write_control_points(self.directory / "cp.json"),
                self.scene.write_calibration(self.directory / "calib.json"),
                output,
            )
        )
        self.assertEqual(
            self.recall(estimate, output / "sparse_eval_result.npy"),
            {1.0: 100.0, 5.0: 100.0},
        )


if __name__ == "__main__":
    unittest.main()
