import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hover_postprocess import extract_instances


class T(unittest.TestCase):
    def test_empty(self):
        result = extract_instances(np.full((8, 8), -9.0), np.zeros((2, 8, 8)))
        self.assertFalse(result.any())

    def test_touching_nuclei_produce_exactly_two_instances(self):
        yy, xx = np.mgrid[:32, :32]
        foreground = (xx - 11) ** 2 + (yy - 16) ** 2 <= 8**2
        foreground |= (xx - 21) ** 2 + (yy - 16) ** 2 <= 8**2
        logits = np.where(foreground, 8.0, -8.0)
        result = extract_instances(logits, np.zeros((2, 32, 32)))
        self.assertEqual(len(np.unique(result)) - 1, 2)

    def test_small_isolated_nucleus_is_not_dropped_beside_large_one(self):
        yy, xx = np.mgrid[:48, :48]
        foreground = (xx - 15) ** 2 + (yy - 24) ** 2 <= 10**2
        foreground |= (xx - 39) ** 2 + (yy - 24) ** 2 <= 2**2
        logits = np.where(foreground, 8.0, -8.0)
        result = extract_instances(logits, np.zeros((2, 48, 48)))
        self.assertEqual(len(np.unique(result)) - 1, 2)

    def test_hv_shape_is_checked(self):
        with self.assertRaisesRegex(ValueError, "HV output"):
            extract_instances(np.ones((8, 8)), np.zeros((8, 8)))


if __name__ == "__main__":
    unittest.main()
