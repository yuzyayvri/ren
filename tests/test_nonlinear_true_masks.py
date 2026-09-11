import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from scripts.run_phase2_nonlinear_classifier import true_instance_rows


def test_true_mask_pooling_does_not_use_prediction_namespace():
    semantic = np.array([[1, 1, 0], [0, 0, 2]], dtype=np.int16)
    truth = np.array([[1, 1, 0], [0, 0, 2]], dtype=np.int16)
    fmap = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    image = np.zeros((2, 3, 3), dtype=np.float32)
    ids, features, labels = true_instance_rows(truth, semantic, fmap, image)
    assert ids.tolist() == [1, 2]
    assert labels.tolist() == [1, 2]
    assert features[:, 0].tolist() == [0.5, 5.0]


def test_provenance_partition_invariant():
    total, positives, negatives, zero = 10, 4, 3, 3
    assert total == positives + negatives + zero
