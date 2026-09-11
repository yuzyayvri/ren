"""Fail-closed PanNuke mask schema validation and instance accounting.

This module deliberately reads the source masks directly.  It does not create
training targets or normalize ambiguous pixels: that policy belongs to the
architecture selected after Phase 2 review.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage

CLASS_NAMES = ("neoplastic", "inflammatory", "connective", "dead", "epithelial")
BACKGROUND_CHANNEL = 5
MASK_CHANNELS = len(CLASS_NAMES) + 1
PATCH_SHAPE = (256, 256)


def fold_paths(data_root: Path, fold: int) -> tuple[Path, Path, Path]:
    """Return the image, mask, and tissue-type paths for an official fold."""
    if fold not in (1, 2, 3):
        raise ValueError(f"fold must be 1, 2, or 3; got {fold}")
    base = data_root / f"fold{fold}" / f"Fold {fold}"
    return (
        base / "images" / f"fold{fold}" / "images.npy",
        base / "masks" / f"fold{fold}" / "masks.npy",
        base / "images" / f"fold{fold}" / "types.npy",
    )


def validate_mask_array(mask: np.ndarray, expected_rows: int | None = None) -> None:
    """Reject schema changes that would silently corrupt instance labels."""
    if mask.ndim != 4 or tuple(mask.shape[1:3]) != PATCH_SHAPE or mask.shape[3] != MASK_CHANNELS:
        raise ValueError(f"expected (N, 256, 256, 6) masks, got {mask.shape}")
    if expected_rows is not None and len(mask) != expected_rows:
        raise ValueError(f"mask row count {len(mask)} differs from images/types {expected_rows}")
    if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
        raise ValueError("masks must be finite numeric values")
    if np.any(mask < 0) or not np.equal(mask, np.floor(mask)).all():
        raise ValueError("masks must contain non-negative integer-valued IDs")
    background = mask[..., BACKGROUND_CHANNEL]
    if not np.isin(background, (0, 1)).all():
        raise ValueError("background channel must be binary (0 or 1)")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash an immutable source array without materialising it in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def instance_key(patch_index: int, class_index: int, instance_id: int) -> tuple[int, int, int]:
    """Canonical source-instance namespace; raw numeric IDs are not global."""
    if patch_index < 0 or class_index not in range(len(CLASS_NAMES)) or instance_id <= 0:
        raise ValueError("invalid (patch, class, ID) instance namespace")
    return patch_index, class_index, instance_id


def disconnected_instance_keys(mask_patch: np.ndarray, patch_index: int) -> set[tuple[int, int, int]]:
    """Return source instances split into multiple 8-connected components.

    This is diagnostic only; no fragment is discarded or relabelled.
    """
    if mask_patch.shape != (*PATCH_SHAPE, MASK_CHANNELS):
        raise ValueError(f"expected (256, 256, 6) patch, got {mask_patch.shape}")
    disconnected: set[tuple[int, int, int]] = set()
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_index in range(len(CLASS_NAMES)):
        channel = mask_patch[..., class_index]
        for instance_id in np.unique(channel):
            if instance_id and ndimage.label(channel == instance_id, structure=structure)[1] > 1:
                disconnected.add(instance_key(patch_index, class_index, int(instance_id)))
    return disconnected


def audit_fold(mask: np.ndarray) -> dict[str, object]:
    """Count namespaced instances and report source contradictions unchanged."""
    validate_mask_array(mask)
    class_instances: Counter[str] = Counter()
    class_pixels: Counter[str] = Counter()
    patches_with_class: Counter[str] = Counter()
    overlap_pixels = 0
    overlap_patches = 0
    foreground_on_background_pixels = 0
    void_pixels = 0
    affected_overlap_instances: set[tuple[int, int, int]] = set()
    affected_background_instances: set[tuple[int, int, int]] = set()
    disconnected_instances: Counter[str] = Counter()

    for patch_index, patch in enumerate(mask):
        for _, class_index, _ in disconnected_instance_keys(patch, patch_index):
            disconnected_instances[CLASS_NAMES[class_index]] += 1
        foreground = patch[..., :BACKGROUND_CHANNEL]
        foreground_count = (foreground > 0).sum(axis=-1)
        overlap = foreground_count > 1
        overlap_pixels += int(overlap.sum())
        overlap_patches += int(overlap.any())
        background = patch[..., BACKGROUND_CHANNEL] > 0
        foreground_on_background_pixels += int((background & (foreground_count > 0)).sum())
        void_pixels += int((~background & (foreground_count == 0)).sum())
        for channel, class_name in enumerate(CLASS_NAMES):
            ids = np.unique(foreground[..., channel])
            ids = ids[ids != 0]
            if len(ids):
                class_instances[class_name] += len(ids)
                patches_with_class[class_name] += 1
                class_pixels[class_name] += int((foreground[..., channel] > 0).sum())
            for instance_id in np.unique(foreground[..., channel][overlap]):
                if instance_id:
                    affected_overlap_instances.add(instance_key(patch_index, channel, int(instance_id)))
            for instance_id in np.unique(foreground[..., channel][background]):
                if instance_id:
                    affected_background_instances.add(instance_key(patch_index, channel, int(instance_id)))

    return {
        "patches": len(mask),
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "class_instances": dict(class_instances),
        "class_pixels": dict(class_pixels),
        "patches_with_class": dict(patches_with_class),
        "background_pixels": int((mask[..., BACKGROUND_CHANNEL] > 0).sum()),
        "foreground_overlap_pixels": overlap_pixels,
        "patches_with_foreground_overlap": overlap_patches,
        "foreground_on_background_pixels": foreground_on_background_pixels,
        "unlabelled_void_pixels": void_pixels,
        "instances_affected_by_foreground_overlap": len(affected_overlap_instances),
        "instances_affected_by_background_contradiction": len(affected_background_instances),
        "disconnected_instances_by_class": dict(disconnected_instances),
        "instance_namespace": "(patch_index, class_index, nonzero_instance_id)",
    }
