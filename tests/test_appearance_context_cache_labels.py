import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_phase2_appearance_context_cache import (
    PREDICTED_LABEL_IGNORED,
    PREDICTED_LABEL_UNMATCHED,
    matched_proposal_labels,
)


def test_proposal_labels_use_iou_matching_not_numeric_id():
    truth = np.zeros((6, 6), dtype=np.int32)
    truth[1:4, 1:4] = 7
    prediction = np.zeros_like(truth)
    prediction[1:4, 1:4] = 99
    labels = matched_proposal_labels(
        prediction, truth, {7: 3}, np.zeros_like(truth, bool), [99]
    )
    assert labels.tolist() == [3]


def test_proposal_labels_distinguish_unmatched_and_ignored():
    truth = np.zeros((6, 6), dtype=np.int32)
    prediction = np.zeros_like(truth)
    prediction[0:2, 0:2] = 4
    prediction[4:6, 4:6] = 5
    ignored = np.zeros_like(truth, bool)
    ignored[4:6, 4:6] = True
    labels = matched_proposal_labels(prediction, truth, {}, ignored, [4, 5])
    assert labels.tolist() == [PREDICTED_LABEL_UNMATCHED, PREDICTED_LABEL_IGNORED]
