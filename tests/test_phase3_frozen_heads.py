import numpy as np
import pytest

from scripts.phase3_frozen_heads import (
    crop_txl,
    normalize_for_path_foundation,
    preprocess_aml,
)


def test_txl_crop_and_aml_preprocess_are_rgb_224():
    image = np.zeros((20, 30, 3), dtype=np.uint8); image[..., 0] = 100
    assert crop_txl(image, (5, 4, 15, 14)).shape == (224, 224, 3)
    assert preprocess_aml(image).shape == (224, 224, 3)


def test_path_input_validation_is_fail_closed():
    with pytest.raises(ValueError): normalize_for_path_foundation(np.zeros((1, 224, 224, 1), dtype=np.float32))
    with pytest.raises(ValueError): preprocess_aml(np.zeros((20, 20), dtype=np.uint8))
