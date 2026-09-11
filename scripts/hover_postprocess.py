"""Deterministic local instance extraction from HoVer branch outputs."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _markers(foreground: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Create local markers without a patch-wide size-dependent threshold."""
    coordinates = peak_local_max(
        distance,
        labels=foreground,
        min_distance=8,
        exclude_border=False,
    )
    seeds = np.zeros_like(foreground, dtype=bool)
    if len(coordinates):
        seeds[tuple(coordinates.T)] = True

    # Guarantee at least one seed for every disconnected component. A global
    # peak filter can otherwise suppress a small nucleus near a larger one.
    components, component_count = ndimage.label(foreground)
    seeded_components = set(np.unique(components[seeds]))
    for component_id in range(1, component_count + 1):
        if component_id in seeded_components:
            continue
        component = components == component_id
        location = np.argmax(np.where(component, distance, -1))
        seeds.flat[location] = True
    return ndimage.label(seeds)[0]


def extract_instances(
    np_logits: np.ndarray,
    hv: np.ndarray,
    threshold: float = 0.5,
    marker_percentile: float | None = None,
) -> np.ndarray:
    """Extract instances using local distance markers and oriented HV gradients.

    ``marker_percentile`` remains only for evaluator API compatibility. Local
    markers replace the defective patch-wide percentile.
    """
    del marker_percentile
    probability = 1 / (1 + np.exp(-np_logits))
    foreground = probability > threshold
    if not foreground.any():
        return np.zeros_like(np_logits, dtype=np.int32)
    if hv.shape != (2, *np_logits.shape):
        raise ValueError("HV output must have shape (2, height, width)")

    horizontal_gradient = ndimage.sobel(hv[0], axis=1)
    vertical_gradient = ndimage.sobel(hv[1], axis=0)
    boundary_energy = np.hypot(horizontal_gradient, vertical_gradient)
    distance = ndimage.distance_transform_edt(foreground)
    markers = _markers(foreground, distance)
    return watershed(boundary_energy - distance, markers, mask=foreground).astype(
        np.int32
    )


def assign_types(instances: np.ndarray, type_logits: np.ndarray) -> dict[int, int]:
    labels = type_logits.argmax(0)
    return {
        int(instance_id): int(
            np.bincount(labels[instances == instance_id], minlength=6)[1:].argmax()
            + 1
        )
        for instance_id in np.unique(instances)
        if instance_id
    }
