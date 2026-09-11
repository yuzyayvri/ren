import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from phase2_augmentation import augment_geometry


class AugmentationTest(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(12).reshape(2, 2, 3)
        self.semantic = np.array([[1, 2], [3, 4]])
        self.instance = self.semantic.copy()
        self.hv = np.stack((np.ones((2, 2)), np.full((2, 2), 2.0)))

    def test_counterclockwise_rotation_rotates_vectors(self):
        _, semantic, _, hv = augment_geometry(
            self.image, self.semantic, self.instance, self.hv, 1, False, False
        )
        self.assertTrue(np.array_equal(semantic, np.rot90(self.semantic)))
        self.assertTrue(np.all(hv[0] == 2))
        self.assertTrue(np.all(hv[1] == -1))

    def test_horizontal_flip_negates_horizontal_component_only(self):
        _, _, _, hv = augment_geometry(
            self.image, self.semantic, self.instance, self.hv, 0, True, False
        )
        self.assertTrue(np.all(hv[0] == -1))
        self.assertTrue(np.all(hv[1] == 2))

    def test_vertical_flip_negates_vertical_component_only(self):
        _, _, _, hv = augment_geometry(
            self.image, self.semantic, self.instance, self.hv, 0, False, True
        )
        self.assertTrue(np.all(hv[0] == 1))
        self.assertTrue(np.all(hv[1] == -2))


if __name__ == "__main__":
    unittest.main()
