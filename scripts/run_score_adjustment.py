"""Cache frozen fold2 instances and run a bounded additive score experiment."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import (
    coverage_indices,
    fold2_arrays,
    instance_types,
    load_model,
)
from evaluate_instance_suppression import (
    classifier_arrays,
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
from score_adjustment import adjusted_classes, choose, dead_grid
from train_instance_classifier import FeatureCapture, pooled_features, sha256


def raw_scores(features: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    return ((features - arrays["mean"]) / arrays["scale"]) @ arrays["coefficients"].T + arrays["intercept"]


def validity_probability(features: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (features - arrays["mean"]) / arrays["scale"]
    logits = standardized @ arrays["coefficients"].T + arrays["intercept"]
    return (1 / (1 + np.exp(-logits[:, 0]))).astype(np.float32)


def cache_fold2(args: argparse.Namespace, type_arrays, filter_arrays) -> None:
    model = load_model(args.segmentation_checkpoint)
    model.eval()
    capture = FeatureCapture(model)
    images, masks = fold2_arrays()
    records = []
    with torch.inference_mode():
        for index in range(len(images)):
            semantic, truth, _ = make_target(masks[index], index)
            feature_map, np_logits, hv = model_features(model, capture, images[index])
            normalized = images[index].astype(np.float32) / 255.0
            prediction = extract_instances(np_logits, hv, threshold=args.np_threshold)
            ids, features = pooled_features(prediction, feature_map, normalized)
            confidence = confidence_features(prediction, np_logits)
            if len(features):
                validity = validity_probability(np.column_stack((features, confidence)), filter_arrays)
                keep = validity >= args.filter_threshold
                retained = retain(prediction, ids, keep)
                ids, features = ids[keep], features[keep]
                # Scores and retained masks are cached together; geometry is frozen.
                scores = raw_scores(features, type_arrays)
                predicted_types = adjusted_classes(scores, np.zeros(5))
            else:
                validity = np.empty(0, np.float32)
                retained = prediction
                scores = np.empty((0, 5), np.float32)
                predicted_types = np.empty(0, np.int64)
                ids = np.empty(0, np.int64)
            records.append({
                "patch_index": index,
                "prediction": retained,
                "instance_ids": ids,
                "scores": scores.astype(np.float32),
                "baseline_classes": predicted_types,
                "truth": truth,
                "truth_types": instance_types(truth, semantic),
                "ignored": semantic == IGNORE,
            })
            if (index + 1) % 256 == 0:
                print(f"cached_fold2={index + 1}/{len(images)}", flush=True)
    capture.close()
    with (args.output / "score_cache.pkl").open("wb") as handle:
        pickle.dump(records, handle, protocol=5)
    manifest = {
        "schema": "ren-fold2-retained-instance-raw-five-class-scores-v1",
        "records": len(records),
        "segmentation_sha256": sha256(args.segmentation_checkpoint),
        "type_classifier_sha256": sha256(args.type_classifier),
        "validity_filter_sha256": sha256(args.validity_filter),
        "fold3_accessed": False,
    }
    (args.output / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def evaluate(records, offsets: np.ndarray) -> dict[str, object]:
    total = empty_score()
    categories = {key: 0 for key in ("background_only", "split_fragment", "merge", "boundary_localization", "other_unresolved")}
    matched = correct = 0
    predicted_counts = np.zeros(5, dtype=int)
    for record in records:
        classes = adjusted_classes(record["scores"], offsets)
        predicted_counts += np.bincount(classes - 1, minlength=5)
        predicted_types = dict(zip(record["instance_ids"].tolist(), classes.tolist()))
        add_scores(total, score_instances(record["prediction"], record["truth"], predicted_types, record["truth_types"], record["ignored"]))
        matches, _, _ = match_instances(record["prediction"], record["truth"], record["ignored"])
        matched += len(matches)
        correct += sum(predicted_types[p] == record["truth_types"][t] for p, t, _ in matches)
        patch = false_positive_categories(record["prediction"], record["truth"], record["ignored"])
        for key, value in patch.items():
            categories[key] += value
    result = summarize_score(total)
    result["matched_instance_typing_accuracy"] = correct / matched if matched else 0.0
    result["predicted_class_counts"] = predicted_counts.tolist()
    result["false_positive_categories"] = categories
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--type-classifier", type=Path, required=True)
    parser.add_argument("--validity-filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--np-threshold", type=float, default=0.60)
    parser.add_argument("--filter-threshold", type=float, default=0.35)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    type_arrays = classifier_arrays(args.type_classifier)
    filter_arrays = classifier_arrays(args.validity_filter)
    cache_fold2(args, type_arrays, filter_arrays)
    with (args.output / "score_cache.pkl").open("rb") as handle:
        records = pickle.load(handle)
    monitor_indices = set(coverage_indices(fold2_arrays()[1]))
    monitor_records = [record for record in records if record["patch_index"] in monitor_indices]
    curve = []
    for offset in dead_grid():
        offsets = np.zeros(5)
        offsets[3] = offset
        curve.append({"offsets": offsets.tolist(), "metrics": evaluate(monitor_records, offsets)})
    selected = choose(curve)
    selected_offsets = np.asarray(selected["offsets"], dtype=float)
    report = {
        "experiment": "phase2-score-adjustment-v1",
        "score_semantics": "raw five-class decision scores, not calibrated probabilities",
        "dead_only_curve": curve,
        "selected_dead_only": selected,
        "baseline_monitor": curve[0],
        "full_fold2": {
            "baseline": evaluate(records, np.zeros(5)),
            "selected_dead_only": evaluate(records, selected_offsets),
        },
        "monitor_indices": sorted(monitor_indices),
        "fold2_is_development_data": True,
        "fold3_accessed": False,
        "frozen_inputs": {
            "segmentation_checkpoint": str(args.segmentation_checkpoint),
            "segmentation_sha256": sha256(args.segmentation_checkpoint),
            "type_classifier": str(args.type_classifier),
            "type_classifier_sha256": sha256(args.type_classifier),
            "validity_filter": str(args.validity_filter),
            "validity_filter_sha256": sha256(args.validity_filter),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
