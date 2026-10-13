import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "map_build"))

from edit_web_map import orthogonalize_axes


class EditorMathTest(unittest.TestCase):
    def test_axis_hints_are_fitted_to_exact_right_angle(self):
        x_axis, y_axis = orthogonalize_axes(
            np.array([100.0, 12.0]), np.array([8.0, -90.0]))
        self.assertAlmostEqual(float(np.linalg.norm(x_axis)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(y_axis)), 1.0)
        self.assertAlmostEqual(float(np.dot(x_axis, y_axis)), 0.0, places=12)
        self.assertGreater(y_axis[0] * x_axis[1] - y_axis[1] * x_axis[0], 0.0)

    def test_conflicting_y_hint_does_not_block_x_direction(self):
        x_axis, y_axis = orthogonalize_axes(
            np.array([50.0, 0.0]), np.array([0.0, 40.0]))
        np.testing.assert_allclose(x_axis, [1.0, 0.0])
        np.testing.assert_allclose(y_axis, [0.0, -1.0])


if __name__ == "__main__":
    unittest.main()
