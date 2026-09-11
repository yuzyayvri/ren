import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pannuke_masks import audit_fold, disconnected_instance_keys, instance_key, validate_mask_array


class PanNukeMaskContractTest(unittest.TestCase):
    def test_audit_counts_instance_ids_not_max_id(self):
        masks = np.zeros((1, 256, 256, 6), dtype=np.float64)
        masks[0, 0:2, 0:2, 0] = 17
        masks[0, 3:5, 0:2, 0] = 42
        masks[0, 0:2, 3:5, 3] = 9
        masks[..., 5] = 1
        report = audit_fold(masks)
        self.assertEqual(report["class_instances"], {"neoplastic": 2, "dead": 1})
        self.assertEqual(report["instance_namespace"], "(patch_index, class_index, nonzero_instance_id)")

    def test_numeric_ids_in_different_channels_have_distinct_namespaces(self):
        self.assertNotEqual(instance_key(0, 0, 7), instance_key(0, 1, 7))
        self.assertNotEqual(instance_key(0, 0, 7), instance_key(1, 0, 7))

    def test_rejects_wrong_channel_count(self):
        with self.assertRaisesRegex(ValueError, "256, 256, 6"):
            validate_mask_array(np.zeros((1, 256, 256, 5)))

    def test_rejects_nonbinary_background(self):
        masks = np.zeros((1, 256, 256, 6))
        masks[..., 5] = 2
        with self.assertRaisesRegex(ValueError, "background"):
            validate_mask_array(masks)

    def test_overlap_is_reported_not_silently_resolved(self):
        masks = np.zeros((1, 256, 256, 6))
        masks[0, 0, 0, 0] = 1
        masks[0, 0, 0, 1] = 2
        report = audit_fold(masks)
        self.assertEqual(report["foreground_overlap_pixels"], 1)
        self.assertEqual(report["instances_affected_by_foreground_overlap"], 2)

    def test_background_conflict_and_void_are_reported(self):
        masks = np.zeros((1, 256, 256, 6))
        masks[0, 0, 0, 0] = 9
        masks[0, 0, 0, 5] = 1
        report = audit_fold(masks)
        self.assertEqual(report["foreground_on_background_pixels"], 1)
        self.assertEqual(report["instances_affected_by_background_contradiction"], 1)
        self.assertEqual(report["unlabelled_void_pixels"], 256 * 256 - 1)

    def test_disconnected_remnant_is_reported_without_relabelling(self):
        patch = np.zeros((256, 256, 6))
        patch[0, 0, 2] = 11
        patch[5, 5, 2] = 11
        self.assertEqual(disconnected_instance_keys(patch, 3), {(3, 2, 11)})


if __name__ == "__main__":
    unittest.main()
