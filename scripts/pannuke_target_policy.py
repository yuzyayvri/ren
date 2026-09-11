"""Approved, deterministic source-to-target policy for Phase 2."""
from __future__ import annotations

import numpy as np
from pannuke_masks import (
    BACKGROUND_CHANNEL,
    CLASS_NAMES,
    MASK_CHANNELS,
    PATCH_SHAPE,
    disconnected_instance_keys,
    validate_mask_array,
)

IGNORE = 255

POLICY_ID = "pannuke-target-v1-ignore-ambiguous-instances"

def hover_offsets(instance: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Per-instance centre-relative horizontal/vertical maps, valid nuclei only."""
    hv=np.zeros((2,*PATCH_SHAPE),np.float32)
    for ident in np.unique(instance):
        if not ident: continue
        ys,xs=np.where(instance==ident); cy,cx=ys.mean(),xs.mean()
        scale_y=max(float(np.abs(ys-cy).max()),1.0); scale_x=max(float(np.abs(xs-cx).max()),1.0)
        hv[0,ys,xs]=(xs-cx)/scale_x; hv[1,ys,xs]=(ys-cy)/scale_y
    hv[:,semantic==IGNORE]=0
    return hv


def make_target(mask_patch: np.ndarray, patch_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create type/instance targets; never turn void or ambiguous pixels into background.

    Entire instances touching foreground overlap or disconnected remnants are
    ignored in every branch. Valid background is only source channel 5 == 1.
    A fully void patch consequently has no valid pixels and cannot be a
    negative example. Border-touching valid instances are retained for training;
    primary evaluation retains and scores them normally.
    """
    if mask_patch.shape != (*PATCH_SHAPE, MASK_CHANNELS):
        raise ValueError(f"expected (256, 256, 6), got {mask_patch.shape}")
    validate_mask_array(mask_patch[None])
    fg = mask_patch[..., :BACKGROUND_CHANNEL]
    count = (fg > 0).sum(-1)
    semantic = np.full(PATCH_SHAPE, IGNORE, np.uint8)
    instance = np.zeros(PATCH_SHAPE, np.int32)
    semantic[mask_patch[..., BACKGROUND_CHANNEL] == 1] = 0
    excluded = set(disconnected_instance_keys(mask_patch, patch_index))
    for cls in range(len(CLASS_NAMES)):
        for raw_id in np.unique(fg[..., cls]):
            if not raw_id:
                continue
            pixels = fg[..., cls] == raw_id
            if (count[pixels] > 1).any():
                excluded.add((patch_index, cls, int(raw_id)))
    next_id = 1
    excluded_by_class = {name: 0 for name in CLASS_NAMES}; excluded_pixels_by_class={name:0 for name in CLASS_NAMES}
    for cls in range(len(CLASS_NAMES)):
        for raw_id in np.unique(fg[..., cls]):
            if not raw_id:
                continue
            key = (patch_index, cls, int(raw_id))
            pixels = fg[..., cls] == raw_id
            if key in excluded:
                semantic[pixels] = IGNORE
                excluded_by_class[CLASS_NAMES[cls]] += 1
                excluded_pixels_by_class[CLASS_NAMES[cls]] += int(pixels.sum())
                continue
            semantic[pixels] = cls + 1
            instance[pixels] = next_id
            next_id += 1
    return semantic, instance, {"excluded_instances": len(excluded), **excluded_by_class, **{f"excluded_pixels_{k}":v for k,v in excluded_pixels_by_class.items()}, "valid_pixels": int((semantic != IGNORE).sum())}
