#!/usr/bin/env python3
"""Decompose Phase 2 errors on the fixed fold2 development monitor."""

from __future__ import annotations

import argparse
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
from nucleus_evaluation import (
    add_scores,
    empty_score,
    match_instances,
    score_instances,
    summarize_score,
)
from pannuke_target_policy import IGNORE, hover_offsets, make_target


def perfect_logits(foreground: np.ndarray) -> np.ndarray:
    return np.where(foreground, 20.0, -20.0).astype(np.float32)


def overlap_errors(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """Count predictions spanning truths (merges) and truths spanning predictions (splits)."""
    matrix: dict[tuple[int, int], bool] = {}
    for predicted_id in np.unique(prediction):
        if not predicted_id:
            continue
        truth_ids = np.unique(truth[prediction == predicted_id])
        for truth_id in truth_ids[truth_ids > 0]:
            matrix[int(predicted_id), int(truth_id)] = True
    prediction_ids = {key[0] for key in matrix}
    truth_ids = {key[1] for key in matrix}
    merges = sum(sum(key[0] == ident for key in matrix) > 1 for ident in prediction_ids)
    splits = sum(sum(key[1] == ident for key in matrix) > 1 for ident in truth_ids)
    return {"merge_predictions": merges, "split_truth_instances": splits}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images, masks = fold2_arrays()
    indices = coverage_indices(masks)
    model = load_model(args.checkpoint)
    model.eval()

    scenario_totals = {
        name: empty_score()
        for name in ("predicted", "perfect_foreground_hv", "perfect_foreground_predicted_hv", "predicted_foreground_perfect_hv")
    }
    matched_type_confusion = np.zeros((5, 5), dtype=np.int64)
    truth_mask_type_confusion = np.zeros((5, 5), dtype=np.int64)
    overlap_total = {"merge_predictions": 0, "split_truth_instances": 0}

    with torch.inference_mode():
        for index in indices:
            semantic, truth, _ = make_target(masks[index], index)
            truth_types = instance_types(truth, semantic)
            image = torch.from_numpy(
                images[index].astype(np.float32).transpose(2, 0, 1) / 255.0
            )[None]
            np_logits, predicted_hv, type_logits = model(image)
            predicted_np = np_logits[0, 0].numpy()
            predicted_hv_array = predicted_hv[0].numpy()
            type_array = type_logits[0].numpy()
            foreground = truth > 0
            target_hv = hover_offsets(truth, semantic)
            scenarios = {
                "predicted": extract_instances(predicted_np, predicted_hv_array),
                "perfect_foreground_hv": extract_instances(perfect_logits(foreground), target_hv),
                "perfect_foreground_predicted_hv": extract_instances(perfect_logits(foreground), predicted_hv_array),
                "predicted_foreground_perfect_hv": extract_instances(predicted_np, target_hv),
            }
            for name, prediction in scenarios.items():
                add_scores(
                    scenario_totals[name],
                    score_instances(
                        prediction,
                        truth,
                        assign_types(prediction, type_array),
                        truth_types,
                        ignored=semantic == IGNORE,
                    ),
                )

            prediction = scenarios["predicted"]
            for key, value in overlap_errors(prediction, truth).items():
                overlap_total[key] += value
            prediction_types = assign_types(prediction, type_array)
            matches, _, _ = match_instances(prediction, truth, semantic == IGNORE)
            for predicted_id, truth_id, _ in matches:
                matched_type_confusion[truth_types[truth_id] - 1, prediction_types[predicted_id] - 1] += 1
            true_mask_predictions = assign_types(truth, type_array)
            for truth_id, truth_type in truth_types.items():
                truth_mask_type_confusion[truth_type - 1, true_mask_predictions[truth_id] - 1] += 1

    report = {
        "checkpoint": str(args.checkpoint),
        "development_fold": 2,
        "fold3_accessed": False,
        "indices": indices,
        "scenarios": {name: summarize_score(total) for name, total in scenario_totals.items()},
        "matched_type_confusion_rows_truth_columns_prediction": matched_type_confusion.tolist(),
        "truth_mask_type_confusion_rows_truth_columns_prediction": truth_mask_type_confusion.tolist(),
        "predicted_split_merge_counts": overlap_total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
