"""Geometry augmentation that preserves HoVer target-vector semantics."""

from __future__ import annotations

import numpy as np


def augment_geometry(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    hv: np.ndarray,
    quarter_turns: int,
    horizontal_flip: bool,
    vertical_flip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a D4 transform and rotate/reflect horizontal and vertical offsets."""
    turns = quarter_turns % 4
    image = np.rot90(image, turns, axes=(0, 1))
    semantic = np.rot90(semantic, turns, axes=(0, 1))
    instance = np.rot90(instance, turns, axes=(0, 1))
    hv = np.rot90(hv, turns, axes=(1, 2))
    horizontal, vertical = hv[0].copy(), hv[1].copy()
    if turns == 1:
        horizontal, vertical = vertical, -horizontal
    elif turns == 2:
        horizontal, vertical = -horizontal, -vertical
    elif turns == 3:
        horizontal, vertical = -vertical, horizontal
    hv = np.stack((horizontal, vertical))

    if horizontal_flip:
        image = np.flip(image, axis=1)
        semantic = np.flip(semantic, axis=1)
        instance = np.flip(instance, axis=1)
        hv = np.flip(hv, axis=2)
        hv[0] *= -1
    if vertical_flip:
        image = np.flip(image, axis=0)
        semantic = np.flip(semantic, axis=0)
        instance = np.flip(instance, axis=0)
        hv = np.flip(hv, axis=1)
        hv[1] *= -1
    return tuple(np.ascontiguousarray(array) for array in (image, semantic, instance, hv))
