"""Masked, imbalance-aware losses for the Phase 2 HoVer-style model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = 255
LOSS_ID = "ren-hover-loss-v7-fixed-instance-class-original-np"
FOLD1_VALID_INSTANCE_COUNTS = torch.tensor(
    [0, 26062, 10676, 16204, 961, 8806], dtype=torch.float32
)


def fixed_type_weights(device: torch.device) -> torch.Tensor:
    """Inverse-sqrt fold1 instance-frequency weights, normalized over classes."""
    weights = torch.zeros(6, dtype=torch.float32, device=device)
    weights[1:] = FOLD1_VALID_INSTANCE_COUNTS[1:].to(device).rsqrt()
    weights[1:] /= weights[1:].mean()
    return weights


def valid_batch(semantic: torch.Tensor) -> bool:
    return bool((semantic != IGNORE).any())


def masked_hover_loss(
    np_logits: torch.Tensor,
    hv_prediction: torch.Tensor,
    type_logits: torch.Tensor,
    semantic: torch.Tensor,
    instance: torch.Tensor,
    hv_target: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Combine balanced foreground, nucleus-type, and within-nucleus HV losses."""
    valid = semantic != IGNORE
    if not valid.any():
        return None

    foreground = instance > 0
    foreground_float = foreground.float()
    positives = foreground[valid].sum().float()
    negatives = (~foreground & valid).sum().float()
    positive_weight = (negatives / positives.clamp_min(1)).clamp(1, 20)
    np_bce = F.binary_cross_entropy_with_logits(
        np_logits[:, 0][valid],
        foreground_float[valid],
        pos_weight=positive_weight,
    )
    probability = torch.sigmoid(np_logits[:, 0])
    intersection = (probability[valid] * foreground_float[valid]).sum()
    np_dice = 1 - (2 * intersection + 1) / (
        probability[valid].sum() + foreground_float[valid].sum() + 1
    )

    # NP owns foreground/background. Score types per nucleus, so weight every
    # pixel by inverse instance area; otherwise large nuclei dominate an
    # instance-level evaluation metric.
    if foreground.any():
        pixel_loss = F.cross_entropy(
            type_logits.permute(0, 2, 3, 1)[foreground],
            semantic[foreground].long(),
            reduction="none",
        )
        instance_ids = instance[foreground].long()
        areas = torch.bincount(instance_ids)
        instance_weights = areas[instance_ids].float().reciprocal()
        class_ids = semantic[foreground].long()
        weights = instance_weights * fixed_type_weights(type_logits.device)[class_ids]
        type_loss = (pixel_loss * weights).sum() / weights.sum()
    else:
        type_loss = type_logits.sum() * 0

    target = torch.zeros_like(hv_prediction) if hv_target is None else hv_target
    hv_loss = (
        (hv_prediction - target).square().sum(1)[foreground].mean()
        if foreground.any()
        else hv_prediction.sum() * 0
    )
    return np_bce + np_dice + type_loss + hv_loss
