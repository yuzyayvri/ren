#!/usr/bin/env python3
"""Evaluate one HoVer-fast checkpoint on the fixed fold2 development subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hover_fast_model import HoVerFast
from hover_postprocess import assign_types, extract_instances
from nucleus_evaluation import add_scores, empty_score, score_instances, summarize_score
from pannuke_target_policy import IGNORE, make_target

DEFAULT_SUBSET_SIZE = 128
MIN_PATCHES_PER_CLASS = 16


def fold2_arrays() -> tuple[np.ndarray, np.ndarray]:
    base = Path("data/tissue/fold2/Fold 2")
    return (
        np.load(base / "images/fold2/images.npy", mmap_mode="r"),
        np.load(base / "masks/fold2/masks.npy", mmap_mode="r"),
    )


def coverage_indices(
    masks: np.ndarray,
    size: int = DEFAULT_SUBSET_SIZE,
    minimum_patches_per_class: int = MIN_PATCHES_PER_CLASS,
) -> list[int]:
    """Fixed class-stratified subset selected without model outputs or labels from fold3."""
    chosen: list[int] = []
    counts = np.zeros(5, dtype=np.int64)
    for index in range(len(masks)):
        present = (masks[index, ..., :5] > 0).any((0, 1))
        if np.any(present & (counts < minimum_patches_per_class)):
            chosen.append(index)
            counts += present
        if np.all(counts >= minimum_patches_per_class):
            break
    if not np.all(counts >= minimum_patches_per_class):
        raise ValueError(f"could not cover every class in {minimum_patches_per_class} patches")
    chosen += [index for index in range(len(masks)) if index not in chosen][
        : max(0, size - len(chosen))
    ]
    return chosen[:size]


def instance_types(instances: np.ndarray, semantic: np.ndarray) -> dict[int, int]:
    result = {}
    for ident in np.unique(instances):
        if not ident:
            continue
        labels = semantic[instances == ident]
        labels = labels[(labels >= 1) & (labels <= 5)]
        if not len(labels):
            raise ValueError(f"truth instance {ident} has no valid type pixels")
        result[int(ident)] = int(np.bincount(labels, minlength=6)[1:].argmax() + 1)
    return result


def load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return payload.get("state_dict", payload)


def load_model(checkpoint: Path) -> HoVerFast:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = HoVerFast(
        detach_type_features=bool(payload.get("detach_type_features", False)),
        separate_type_decoder=bool(payload.get("separate_type_decoder", False)),
    )
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    return model


def evaluate(
    model: HoVerFast,
    images: np.ndarray,
    masks: np.ndarray,
    indices: list[int],
    np_threshold: float = 0.5,
    marker_percentile: float = 65.0,
) -> dict[str, object]:
    total = empty_score()
    model.eval()
    with torch.inference_mode():
        for index in indices:
            semantic, truth, _ = make_target(masks[index], index)
            image = torch.from_numpy(
                images[index].astype(np.float32).transpose(2, 0, 1) / 255.0
            )[None]
            np_logits, hv, type_logits = model(image)
            prediction = extract_instances(
                np_logits[0, 0].numpy(),
                hv[0].numpy(),
                threshold=np_threshold,
                marker_percentile=marker_percentile,
            )
            patch_score = score_instances(
                prediction,
                truth,
                assign_types(prediction, type_logits[0].numpy()),
                instance_types(truth, semantic),
                ignored=semantic == IGNORE,
            )
            add_scores(total, patch_score)
    return {
        "indices": indices,
        "np_threshold": np_threshold,
        "marker_strategy": "local-distance-peaks-min-distance-8",
        **summarize_score(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--np-threshold", type=float, default=0.5)
    parser.add_argument("--marker-percentile", type=float, default=65.0)
    args = parser.parse_args()
    images, masks = fold2_arrays()
    model = load_model(args.checkpoint)
    report = {
        "checkpoint": str(args.checkpoint),
        "development_fold": 2,
        "fold3_accessed": False,
        **evaluate(
            model,
            images,
            masks,
            coverage_indices(masks),
            args.np_threshold,
            args.marker_percentile,
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
