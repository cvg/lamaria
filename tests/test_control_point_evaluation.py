import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_wrt_control_points import run
from lamaria.eval.sparse_evaluation import evaluate_wrt_control_points
from lamaria.structs.control_point import run_control_point_triangulation
from lamaria.structs.sparse_eval import SparseEvalResult
from lamaria.utils.metrics import (
    calculate_control_point_recall,
    calculate_control_point_score,
    piecewise_linear_scoring,
)
from tests.synthetic_scene import (
    CONTROL_POINTS,
    NUM_FRAMES,
    TAGS_WITH_HEIGHT,
    SyntheticScene,
    assert_sim3d_close,
    errors_by_tag,
)


class ControlPointTriangulationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.scene = SyntheticScene()

    def test_full_trajectory_triangulates_every_control_point_exactly(self):
        reconstruction, control_points = self.scene.build_reconstruction(
            self.directory
        )
        run_control_point_triangulation(reconstruction, control_points)

        images_per_tag = self.scene.images_per_tag()
        self.assertEqual(set(control_points), set(self.scene.control_points))
        for tag_id, cp in control_points.items():
            with self.subTest(tag_id=tag_id):
                np.testing.assert_allclose(
                    cp.triangulated,
                    self.scene.expected_triangulation(tag_id),
                    atol=1e-6,
                )
                self.assertEqual(cp.inlier_ratio, 1.0)
                self.assertEqual(
                    len(cp.inlier_detections), images_per_tag[tag_id]
                )

    def test_partial_trajectory_triangulates_only_observed_control_points(
        self,
    ):
        indices = range(NUM_FRAMES // 2)
        reconstruction, control_points = self.scene.build_reconstruction(
            self.directory, indices=indices
        )
        run_control_point_triangulation(reconstruction, control_points)

        triangulable = self.scene.triangulable_tags(indices)
        self.assertTrue(0 < len(triangulable) < len(control_points))
        for tag_id, cp in control_points.items():
            with self.subTest(tag_id=tag_id):
                if tag_id in triangulable:
                    np.testing.assert_allclose(
                        cp.triangulated,
                        self.scene.expected_triangulation(tag_id),
                        atol=1e-6,
                    )
                    for image_id, _ in cp.inlier_detections:
                        self.assertIn(image_id, reconstruction.images)
                else:
                    self.assertIsNone(cp.triangulated)
                    self.assertEqual(cp.inlier_ratio, 0.0)
                    self.assertEqual(cp.inlier_detections, [])


class ControlPointEvaluationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.scene = SyntheticScene()
        self.calibration = self.scene.write_calibration(
            self.directory / "calibration.json"
        )
        self.cp_json = self.scene.write_control_points(
            self.directory / "control_points.json"
        )

    def run_evaluation(self, indices=None) -> tuple[bool, Path]:
        estimate = self.scene.write_estimate(
            self.directory / "estimate.txt", indices=indices
        )
        # The output directory does not exist yet; the script must create it.
        output = self.directory / "outputs" / "eval_cp"
        ok = run(estimate, self.cp_json, self.calibration, output)
        return ok, output / "sparse_eval_result.npy"

    def test_full_trajectory_recovers_the_alignment_and_saves_the_result(
        self,
    ):
        with self.assertLogs("lamaria", level="INFO") as logs:
            ok, result_path = self.run_evaluation()
        self.assertTrue(ok)
        output = "\n".join(logs.output)
        self.assertIn("CP Score: 100.0000", output)
        self.assertIn("CP Recall @ 1m: 100.0000", output)

        result = SparseEvalResult.load_from_npy(result_path)
        assert_sim3d_close(
            self, result.alignment, self.scene.topo_from_est, atol=1e-5
        )
        self.assertEqual(set(result.cp_summary), set(self.scene.control_points))
        for tag_id, error in errors_by_tag(result).items():
            with self.subTest(tag_id=tag_id):
                self.assertLess(error, 1e-4)
        self.assertAlmostEqual(calculate_control_point_score(result), 100.0)
        self.assertAlmostEqual(calculate_control_point_recall(result), 100.0)

    def test_partial_trajectory_scores_missing_control_points_as_misses(self):
        indices = range(NUM_FRAMES // 2)
        ok, result_path = self.run_evaluation(indices=indices)
        self.assertTrue(ok)

        result = SparseEvalResult.load_from_npy(result_path)
        assert_sim3d_close(
            self, result.alignment, self.scene.topo_from_est, atol=1e-5
        )
        triangulable = self.scene.triangulable_tags(indices)
        self.assertEqual(set(result.cp_summary), set(self.scene.control_points))
        for tag_id, error in errors_by_tag(result).items():
            with self.subTest(tag_id=tag_id):
                if tag_id in triangulable:
                    self.assertLess(error, 1e-4)
                else:
                    self.assertIsNone(result.cp_summary[tag_id].triangulated)
                    self.assertTrue(np.isnan(error))

        expected = 100.0 * len(triangulable) / len(CONTROL_POINTS)
        self.assertAlmostEqual(calculate_control_point_score(result), expected)
        self.assertAlmostEqual(calculate_control_point_recall(result), expected)

    def test_partial_trajectory_with_too_few_control_points_fails_cleanly(
        self,
    ):
        indices = range(NUM_FRAMES // 4)
        triangulable_with_height = set(TAGS_WITH_HEIGHT) & (
            self.scene.triangulable_tags(indices)
        )
        self.assertLess(len(triangulable_with_height), 3)

        with self.assertLogs("lamaria", level="ERROR") as logs:
            ok, result_path = self.run_evaluation(indices=indices)
        self.assertFalse(ok)
        self.assertFalse(result_path.exists())
        self.assertIn("Sparse evaluation failed", "\n".join(logs.output))


class SparseAlignmentRegressionTest(unittest.TestCase):
    """The Sim(3) refinement must score the control points as they came out
    of triangulation. Passing the scored arrays to Ceres as free parameter
    blocks let weakly constrained points drift toward their ground truth
    during refinement and inflated the scores (cvg/lamaria#22).
    """

    def test_inconsistent_triangulation_keeps_its_error(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        scene = SyntheticScene()
        reconstruction, control_points = scene.build_reconstruction(
            Path(directory.name)
        )
        run_control_point_triangulation(reconstruction, control_points)

        # Pretend the poses around one control point drifted: its
        # triangulation lands 3 m away from where its detections and
        # ground truth say it is. Refinement must not pull it back.
        bad_tag = CONTROL_POINTS[4].tag_id
        displaced_topo = scene.control_points[bad_tag].topo + [3.0, 0.0, 0.0]
        control_points[bad_tag].triangulated = (
            scene.est_from_topo * displaced_topo
        )
        before = {
            tag_id: cp.triangulated.copy()
            for tag_id, cp in control_points.items()
        }

        result = evaluate_wrt_control_points(reconstruction, control_points)

        for tag_id, triangulated in before.items():
            with self.subTest(tag_id=tag_id):
                np.testing.assert_array_equal(
                    result.cp_summary[tag_id].triangulated, triangulated
                )
        assert_sim3d_close(self, result.alignment, scene.topo_from_est, 1e-3)
        errors = errors_by_tag(result)
        self.assertAlmostEqual(errors.pop(bad_tag), 3.0, delta=0.01)
        self.assertLess(max(errors.values()), 1e-3)

        num_good = len(CONTROL_POINTS) - 1
        expected_score = (
            (20.0 * num_good + float(piecewise_linear_scoring()(3.0)))
            / (20.0 * len(CONTROL_POINTS))
            * 100.0
        )
        self.assertAlmostEqual(
            calculate_control_point_score(result), expected_score, delta=0.1
        )
        self.assertAlmostEqual(
            calculate_control_point_recall(result),
            100.0 * num_good / len(CONTROL_POINTS),
        )


if __name__ == "__main__":
    unittest.main()
