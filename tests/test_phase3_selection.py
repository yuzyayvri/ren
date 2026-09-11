import numpy as np

from scripts.phase3_selection import fit_class_balanced_head, select_detector_threshold


def test_detector_threshold_uses_fixed_grid_and_returns_recall():
    result = select_detector_threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), 2)
    assert result["threshold"] in {round(x, 2) for x in np.arange(.05, .951, .05)}
    assert len(result["per_class_recall"]) == 2


def test_logistic_head_uses_training_only_standardizer_and_fixed_c_grid():
    x = np.array([[-2.], [-1.], [1.], [2.], [-1.5], [1.5]])
    y = np.array([0, 0, 1, 1, 0, 1])
    result = fit_class_balanced_head(x[:4], y[:4], x[4:], y[4:])
    assert result["C"] in {0.01, 0.1, 1.0, 10.0}
    assert len(result["scaler_mean"]) == 1
