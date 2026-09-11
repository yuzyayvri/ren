import unittest

import numpy as np
from evaluate_instance_suppression import (
    false_positive_categories,
    select_validity_examples,
)
from score_adjustment import adjusted_classes, choose


class SuppressionPolicyTests(unittest.TestCase):
    def test_zero_offsets_reproduce_raw_argmax(self):
        scores = np.array([[1.0, 2.0, 0.0, -1.0, 0.5], [0.0, 0.0, 3.0, 1.0, 2.0]])
        np.testing.assert_array_equal(adjusted_classes(scores, np.zeros(5)), [2, 3])

    def test_selection_applies_dead_constraints_and_ties(self):
        entries = [
            {"offsets": [0, 0, 0, 0, 0], "metrics": {"macro_f1": 0.5, "matched_instance_typing_accuracy": 0.7, "classes": {"4": {"recall": 0.1, "f1": 0.2}}}},
            {"offsets": [0, 0, 0, -0.2, 0], "metrics": {"macro_f1": 0.5, "matched_instance_typing_accuracy": 0.7, "classes": {"4": {"recall": 0.2, "f1": 0.2}}}},
        ]
        self.assertEqual(choose(entries)["offsets"][3], -0.2)
    def test_validity_selection_excludes_wholly_ignored_proposal(self):
        rows, labels, excluded = select_validity_examples(
            np.array([10, 11]), np.arange(6, dtype=np.float32).reshape(2, 3),
            {10}, np.array([False, True])
        )
        np.testing.assert_array_equal(rows, [[3, 4, 5]])
        np.testing.assert_array_equal(labels, [0])
        self.assertEqual(excluded, 1)

    def test_unmatched_evaluable_proposal_remains_negative(self):
        rows, labels, _ = select_validity_examples(
            np.array([1]), np.array([[1., 2.]]), set(), np.array([True])
        )
        np.testing.assert_array_equal(rows, [[1., 2.]])
        np.testing.assert_array_equal(labels, [0])

    def test_validity_selection_matched_positive_and_alignment(self):
        rows, labels, _ = select_validity_examples(
            np.array([7, 8, 9]), np.array([[70.], [80.], [90.]]), {7, 9},
            np.array([True, False, True])
        )
        np.testing.assert_array_equal(rows, [[70.], [90.]])
        np.testing.assert_array_equal(labels, [1, 1])

    def test_positive_subthreshold_overlap_is_unresolved(self):
        truth = np.zeros((6, 6), int); truth[1:5, 1:5] = 1
        prediction = np.zeros((6, 6), int); prediction[1, 1] = 1
        prediction[0, 0:5] = 1; prediction[5, 0:5] = 1
        result = false_positive_categories(prediction, truth, np.zeros_like(truth, bool))
        self.assertEqual(result["background_only"], 0)
        self.assertEqual(result["other_unresolved"], 1)

    def test_matched_nucleus_fragment_is_not_background(self):
        truth = np.zeros((5, 5), int)
        truth[1:4, 1:4] = 1
        prediction = np.zeros((5, 5), int)
        prediction[1:3, 1:4] = 1
        prediction[3, 1:3] = 2
        result = false_positive_categories(prediction, truth, np.zeros_like(truth, bool))
        self.assertEqual(result["split_fragment"], 1)
        self.assertEqual(result["background_only"], 0)

    def test_genuine_merge(self):
        truth = np.zeros((4, 6), int)
        truth[1:3, 0:2] = 1
        truth[1:3, 4:6] = 2
        prediction = np.zeros_like(truth)
        prediction[1:3, :] = 1
        self.assertEqual(false_positive_categories(prediction, truth, np.zeros_like(truth, bool))["merge"], 1)

    def test_subthreshold_contact_is_not_merge(self):
        truth = np.zeros((4, 6), int)
        truth[1:3, 0:2] = 1
        truth[1:3, 4:6] = 2
        prediction = np.zeros_like(truth)
        prediction[1:3, 0:2] = 1
        prediction[0, 2] = 1  # one-pixel contact only; matched at IoU > .5
        self.assertEqual(false_positive_categories(prediction, truth, np.zeros_like(truth, bool))["merge"], 0)

    def test_boundary_failure(self):
        truth = np.zeros((5, 5), int)
        truth[1:4, 1:4] = 1
        prediction = np.zeros_like(truth)
        prediction[1, 1:4] = 1
        result = false_positive_categories(prediction, truth, np.zeros_like(truth, bool))
        self.assertEqual(result["boundary_localization"], 1)


if __name__ == "__main__":
    unittest.main()
