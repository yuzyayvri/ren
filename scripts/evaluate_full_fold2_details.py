#!/usr/bin/env python3
"""Detailed full-fold2 evaluation of the frozen two-stage filtered pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import fold2_arrays, instance_types, load_model
from evaluate_instance_suppression import (
    classifier_arrays,
    classify,
    confidence_features,
    false_positive_categories,
    model_features,
    retain,
)
from hover_postprocess import extract_instances
from nucleus_evaluation import (
    add_scores,
    empty_score,
    match_instances,
    score_instances,
    summarize_score,
)
from pannuke_target_policy import IGNORE, make_target
from train_instance_classifier import FeatureCapture, pooled_features, sha256


def validity_probability(features: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (features - arrays["mean"]) / arrays["scale"]
    logits = standardized @ arrays["coefficients"].T + arrays["intercept"]
    return (1 / (1 + np.exp(-logits[:, 0]))).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--type-classifier", type=Path, required=True)
    parser.add_argument("--validity-filter", type=Path, required=True)
    parser.add_argument("--np-threshold", type=float, default=0.60)
    parser.add_argument("--filter-threshold", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    model = load_model(args.segmentation_checkpoint)
    model.eval()
    capture = FeatureCapture(model)
    type_arrays = classifier_arrays(args.type_classifier)
    filter_arrays = classifier_arrays(args.validity_filter)
    images, masks = fold2_arrays()
    true_type_total = empty_score()
    end_total = empty_score()
    confusion = np.zeros((5, 5), dtype=np.int64)
    geometric_support = np.zeros(5, dtype=np.int64)
    geometric_matches = np.zeros(5, dtype=np.int64)
    categories = {"background_only": 0, "merge": 0, "split": 0, "boundary": 0}

    with torch.inference_mode():
        for index in range(len(images)):
            semantic, truth, _ = make_target(masks[index], index)
            truth_types = instance_types(truth, semantic)
            feature_map, np_logits, hv = model_features(model, capture, images[index])
            normalized_image = images[index].astype(np.float32) / 255.0

            truth_ids, truth_features = pooled_features(truth, feature_map, normalized_image)
            truth_predictions = classify(truth_features, type_arrays)
            add_scores(
                true_type_total,
                score_instances(
                    truth,
                    truth,
                    dict(zip(truth_ids.tolist(), truth_predictions.tolist())),
                    truth_types,
                    ignored=semantic == IGNORE,
                ),
            )

            prediction = extract_instances(np_logits, hv, threshold=args.np_threshold)
            predicted_ids, base = pooled_features(prediction, feature_map, normalized_image)
            confidence = confidence_features(prediction, np_logits)
            if len(base):
                types = classify(base, type_arrays)
                scores = validity_probability(
                    np.column_stack((base, confidence)), filter_arrays
                )
            else:
                types = np.empty(0, dtype=np.int64)
                scores = np.empty(0, dtype=np.float32)
            keep = scores >= args.filter_threshold
            filtered = retain(prediction, predicted_ids, keep)
            predicted_types = {
                int(ident): int(class_id)
                for ident, class_id, retained in zip(
                    predicted_ids, types, keep, strict=True
                )
                if retained
            }
            ignored = semantic == IGNORE
            add_scores(
                end_total,
                score_instances(
                    filtered, truth, predicted_types, truth_types, ignored=ignored
                ),
            )
            patch_categories = false_positive_categories(filtered, truth, ignored)
            for name, count in patch_categories.items():
                categories[name] += count
            matches, _, _ = match_instances(filtered, truth, ignored)
            for truth_id, truth_class in truth_types.items():
                geometric_support[truth_class - 1] += 1
            for predicted_id, truth_id, _ in matches:
                truth_class = truth_types[truth_id]
                predicted_class = predicted_types[predicted_id]
                geometric_matches[truth_class - 1] += 1
                confusion[truth_class - 1, predicted_class - 1] += 1
            if (index + 1) % 256 == 0:
                print(f"fold2_details={index + 1}/{len(images)}", flush=True)
    capture.close()
    report = {
        "segmentation_checkpoint": str(args.segmentation_checkpoint),
        "segmentation_sha256": sha256(args.segmentation_checkpoint),
        "type_classifier": str(args.type_classifier),
        "type_classifier_sha256": sha256(args.type_classifier),
        "validity_filter": str(args.validity_filter),
        "validity_filter_sha256": sha256(args.validity_filter),
        "development_fold": 2,
        "fold3_accessed": False,
        "np_threshold": args.np_threshold,
        "filter_threshold": args.filter_threshold,
        "true_instance_classification": summarize_score(true_type_total),
        "end_to_end": {
            **summarize_score(end_total),
            "false_positive_categories": categories,
        },
        "geometric_support_by_class": geometric_support.tolist(),
        "geometric_matches_by_class": geometric_matches.tolist(),
        "geometric_recall_by_class": (
            geometric_matches / geometric_support
        ).tolist(),
        "matched_type_confusion_rows_truth_columns_prediction": confusion.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
