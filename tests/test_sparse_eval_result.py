import tempfile
import unittest
from pathlib import Path

import numpy as np
import pycolmap

from lamaria.structs.control_point import ControlPointSummary
from lamaria.structs.sparse_eval import SparseEvalResult
from tests.synthetic_scene import assert_sim3d_close, rotation_from_euler


class SparseEvalResultTest(unittest.TestCase):
    def test_save_creates_parent_directories_and_round_trips(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        alignment = pycolmap.Sim3d(
            1.7,
            rotation_from_euler(25.0, 0.0, 5.0),
            np.array([12.0, -4.0, 1.5]),
        )
        cp_summary = {
            11: ControlPointSummary(
                name="CP_A",
                topo=[1.0, 2.0, 3.0],
                covariance=np.diag([1e-4, 1e-4, 1e-4]).tolist(),
                triangulated=[0.5, 0.25, 0.125],
            ),
            12: ControlPointSummary(
                name="CP_B",
                topo=[4.0, 5.0, 0.0],
                covariance=np.diag([1e-4, 1e-4, 1e18]).tolist(),
                triangulated=None,
            ),
        }
        result = SparseEvalResult(alignment=alignment, cp_summary=cp_summary)
        path = Path(directory.name) / "nested" / "sparse_eval_result.npy"
        self.assertFalse(path.parent.exists())

        result.save_as_npy(path)
        loaded = SparseEvalResult.load_from_npy(path)

        assert_sim3d_close(self, loaded.alignment, alignment, atol=1e-12)
        self.assertEqual(set(loaded.cp_summary), set(cp_summary))
        for tag_id, expected in cp_summary.items():
            with self.subTest(tag_id=tag_id):
                actual = loaded.cp_summary[tag_id]
                self.assertIsInstance(actual, ControlPointSummary)
                self.assertEqual(actual, expected)
                self.assertEqual(
                    actual.is_triangulated(), expected.is_triangulated()
                )

        with self.assertLogs("lamaria", level="ERROR"):
            self.assertIsNone(
                SparseEvalResult.load_from_npy(path.with_name("missing.npy"))
            )


if __name__ == "__main__":
    unittest.main()
