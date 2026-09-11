#!/usr/bin/env python3
"""Fold2-only foreground-threshold diagnostics for saved HoVer checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import (
    coverage_indices,
    fold2_arrays,
    instance_types,
    load_model,
)
from hover_postprocess import assign_types, extract_instances
from nucleus_evaluation import add_scores, empty_score, score_instances, summarize_score
from pannuke_target_policy import IGNORE, make_target


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    thresholds = args.threshold or [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    images, masks = fold2_arrays()
    indices = coverage_indices(masks)
    reports = []
    seen_hashes: set[str] = set()
    for checkpoint in args.checkpoint:
        checkpoint_hash = file_hash(checkpoint)
        if checkpoint_hash in seen_hashes:
            continue
        seen_hashes.add(checkpoint_hash)
        model = load_model(checkpoint)
        model.eval()
        cached = []
        with torch.inference_mode():
            for index in indices:
                semantic, truth, _ = make_target(masks[index], index)
                image = torch.from_numpy(
                    images[index].astype(np.float32).transpose(2, 0, 1) / 255.0
                )[None]
                np_logits, hv, type_logits = model(image)
                cached.append(
                    (
                        semantic,
                        truth,
                        instance_types(truth, semantic),
                        np_logits[0, 0].numpy(),
                        hv[0].numpy(),
                        type_logits[0].numpy(),
                    )
                )

        curve = []
        for threshold in thresholds:
            total = empty_score()
            pixel_tp = pixel_fp = pixel_fn = 0
            for semantic, truth, truth_types, np_logits, hv, type_logits in cached:
                valid = semantic != IGNORE
                true_foreground = truth > 0
                predicted_foreground = 1 / (1 + np.exp(-np_logits)) > threshold
                pixel_tp += int(np.count_nonzero(predicted_foreground & true_foreground & valid))
                pixel_fp += int(np.count_nonzero(predicted_foreground & ~true_foreground & valid))
                pixel_fn += int(np.count_nonzero(~predicted_foreground & true_foreground & valid))
                prediction = extract_instances(np_logits, hv, threshold=threshold)
                add_scores(
                    total,
                    score_instances(
                        prediction,
                        truth,
                        assign_types(prediction, type_logits),
                        truth_types,
                        ignored=~valid,
                    ),
                )
            summary = summarize_score(total)
            curve.append(
                {
                    "threshold": threshold,
                    "pixel_precision": pixel_tp / (pixel_tp + pixel_fp),
                    "pixel_recall": pixel_tp / (pixel_tp + pixel_fn),
                    "pixel_f1": 2 * pixel_tp / (2 * pixel_tp + pixel_fp + pixel_fn),
                    "detection": summary["detection"],
                    "binary_pq": summary["binary_pq"],
                    "macro_f1": summary["macro_f1"],
                }
            )
        reports.append(
            {
                "checkpoint": str(checkpoint),
                "sha256": checkpoint_hash,
                "curve": curve,
                "best_detection_f1": max(curve, key=lambda item: item["detection"]["f1"]),
            }
        )

    report = {
        "development_fold": 2,
        "fold3_accessed": False,
        "indices": indices,
        "checkpoints": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
