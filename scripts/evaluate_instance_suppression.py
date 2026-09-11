#!/usr/bin/env python3
"""Train fold1 proposal suppression and evaluate the fixed pair on fold2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import (
    coverage_indices,
    fold2_arrays,
    instance_types,
    load_model,
)
from hover_postprocess import extract_instances
from nucleus_evaluation import (
    add_scores,
    empty_score,
    match_instances,
    score_instances,
    summarize_score,
)
from pannuke_target_policy import IGNORE, POLICY_ID, make_target
from pilot_hover_fast import fold1_arrays
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train_instance_classifier import FeatureCapture, pooled_features, predict, sha256

FILTER_ID = "ren-instance-validity-v1-fold1-proposals"


def select_validity_examples(
    proposal_ids: np.ndarray,
    proposal_features: np.ndarray,
    matched_proposal_ids: np.ndarray | set[int],
    evaluable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select aligned validity examples, excluding non-evaluable proposals."""
    proposal_ids = np.asarray(proposal_ids)
    proposal_features = np.asarray(proposal_features)
    evaluable = np.asarray(evaluable, dtype=bool)
    if proposal_features.ndim != 2 or len(proposal_features) != len(proposal_ids):
        raise ValueError("proposal ids and feature rows must be aligned")
    if len(evaluable) != len(proposal_ids):
        raise ValueError("evaluable mask must align with proposal ids")
    matched = set(np.asarray(list(matched_proposal_ids), dtype=np.int64).tolist())
    keep = evaluable
    labels = np.asarray([int(ident) in matched for ident in proposal_ids[keep]], dtype=np.int8)
    return proposal_features[keep], labels, int((~keep).sum())


def confidence_features(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0].astype(np.int64)
    if not len(ids):
        return np.empty((0, 4), np.float32)
    probability = 1 / (1 + np.exp(-logits))
    return np.array(
        [
            [
                probability[labels == ident].mean(),
                probability[labels == ident].std(),
                probability[labels == ident].min(),
                probability[labels == ident].max(),
            ]
            for ident in ids
        ],
        dtype=np.float32,
    )


def model_features(model, capture, image):
    tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]
    np_logits, hv, _ = model(tensor)
    feature_map = torch.cat(
        (capture.outputs["encoder1"], capture.outputs["decoder1"]), dim=1
    )[0].numpy()
    return feature_map, np_logits[0, 0].numpy(), hv[0].numpy()


def classifier_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def classify(features: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    return predict(
        features,
        arrays["mean"],
        arrays["scale"],
        arrays["coefficients"],
        arrays["intercept"],
        arrays["classes"],
    )


def retain(labels: np.ndarray, ids: np.ndarray, keep: np.ndarray) -> np.ndarray:
    result = labels.copy()
    rejected = ids[~keep]
    if len(rejected):
        result[np.isin(result, rejected)] = 0
    return result


def false_positive_categories(
    prediction: np.ndarray,
    truth: np.ndarray,
    ignored: np.ndarray,
    minimum_overlap_fraction: float = 0.10,
) -> dict[str, int]:
    """Categorize unmatched proposals using substantial, not one-pixel, overlap.

    An unmatched proposal touching a truth nucleus that is already matched is a
    split fragment; it is not background.  A proposal overlapping multiple
    truth nuclei is a merge.  A substantial overlap with one otherwise
    unmatched truth nucleus is a boundary/localization failure.  Contacts
    below ``minimum_overlap_fraction`` are deliberately ignored.
    """
    matches, false_positives, false_negatives = match_instances(prediction, truth, ignored)
    matched_truth = {truth_id for _, truth_id, _ in matches}
    substantial: dict[int, set[int]] = {}
    any_truth_overlap: dict[int, bool] = {}
    for predicted_id in false_positives:
        mask = (prediction == predicted_id) & ~ignored
        area = int(mask.sum())
        substantial[predicted_id] = set()
        any_truth_overlap[predicted_id] = bool(np.any(truth[mask] > 0))
        for truth_id in np.unique(truth[mask]):
            if not truth_id:
                continue
            overlap = int(np.count_nonzero(mask & (truth == truth_id)))
            truth_area = int(np.count_nonzero((truth == truth_id) & ~ignored))
            if overlap / min(area, truth_area) >= minimum_overlap_fraction:
                substantial[predicted_id].add(int(truth_id))
    result = {
        "background_only": 0,
        "split_fragment": 0,
        "merge": 0,
        "boundary_localization": 0,
        "other_unresolved": 0,
    }
    for predicted_id in false_positives:
        truth_ids = substantial[predicted_id]
        if not any_truth_overlap[predicted_id]:
            result["background_only"] += 1
        elif not truth_ids:
            result["other_unresolved"] += 1
        elif len(truth_ids) > 1:
            result["merge"] += 1
        elif any(truth_id in matched_truth for truth_id in truth_ids):
            result["split_fragment"] += 1
        elif any(truth_id in false_negatives for truth_id in truth_ids):
            result["boundary_localization"] += 1
        else:
            result["other_unresolved"] += 1
    return result


def add_categories(total: dict[str, int], patch: dict[str, int]) -> None:
    for name, count in patch.items():
        total[name] += count


def score_records(records, mode: str, threshold: float):
    total = empty_score()
    categories = {
        "background_only": 0,
        "split_fragment": 0,
        "merge": 0,
        "boundary_localization": 0,
        "other_unresolved": 0,
    }
    for record in records:
        score_record(total, categories, record, mode, threshold)
    return {**summarize_score(total), "false_positive_categories": categories}


def score_record(total, categories, record, mode: str, threshold: float) -> None:
    keep = record[mode] >= threshold
    prediction = retain(record["prediction"], record["ids"], keep)
    predicted_types = {
        int(ident): int(class_id)
        for ident, class_id, retained in zip(
            record["ids"], record["types"], keep, strict=True
        )
        if retained
    }
    add_scores(
        total,
        score_instances(
            prediction,
            record["truth"],
            predicted_types,
            record["truth_types"],
            ignored=record["ignored"],
        ),
    )
    add_categories(
        categories,
        false_positive_categories(prediction, record["truth"], record["ignored"]),
    )


def choose(curve: list[dict[str, object]]) -> dict[str, object]:
    eligible = [entry for entry in curve if entry["metrics"]["classes"]["4"]["recall"] >= 0.20]
    pool = eligible or curve
    return max(pool, key=lambda entry: entry["metrics"]["detection"]["f1"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--type-classifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--np-threshold", type=float, default=0.60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    model = load_model(args.segmentation_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = FeatureCapture(model)
    type_arrays = classifier_arrays(args.type_classifier)

    train_images, train_masks = fold1_arrays()
    filter_features = []
    filter_labels = []
    excluded_ignored_proposals = 0
    with torch.inference_mode():
        for index in range(len(train_images)):
            semantic, truth, _ = make_target(train_masks[index], index)
            feature_map, np_logits, hv = model_features(model, capture, train_images[index])
            prediction = extract_instances(np_logits, hv, threshold=args.np_threshold)
            ids, base = pooled_features(
                prediction,
                feature_map,
                train_images[index].astype(np.float32) / 255.0,
            )
            matched, _, _ = match_instances(prediction, truth, semantic == IGNORE)
            valid_ids = {predicted_id for predicted_id, _, _ in matched}
            confidence = confidence_features(prediction, np_logits)
            evaluable = np.array(
                [np.any((prediction == ident) & ~(semantic == IGNORE)) for ident in ids],
                dtype=bool,
            )
            # Fully ignored proposals are outside the target policy; retain
            # their count for provenance without turning them into negatives.
            selected, labels, excluded = select_validity_examples(
                ids, np.column_stack((base, confidence)), valid_ids, evaluable
            )
            excluded_ignored_proposals += excluded
            filter_features.append(selected)
            filter_labels.append(labels)
            if (index + 1) % 256 == 0:
                print(f"fold1_proposals={index + 1}/{len(train_images)}", flush=True)
    x_train = np.concatenate(filter_features)
    y_train = np.concatenate(filter_labels)
    filter_scaler = StandardScaler().fit(x_train)
    validity = LogisticRegression(
        max_iter=500, class_weight="balanced", solver="lbfgs", random_state=20260909
    ).fit(filter_scaler.transform(x_train), y_train)
    np.savez_compressed(
        args.output / "validity_filter.npz",
        mean=filter_scaler.mean_,
        scale=filter_scaler.scale_,
        coefficients=validity.coef_,
        intercept=validity.intercept_,
        classes=validity.classes_,
    )

    development_images, development_masks = fold2_arrays()
    monitor = set(coverage_indices(development_masks))
    monitor_records = []
    with torch.inference_mode():
        for index in sorted(monitor):
            semantic, truth, _ = make_target(development_masks[index], index)
            feature_map, np_logits, hv = model_features(
                model, capture, development_images[index]
            )
            prediction = extract_instances(np_logits, hv, threshold=args.np_threshold)
            ids, base = pooled_features(
                prediction,
                feature_map,
                development_images[index].astype(np.float32) / 255.0,
            )
            confidence = confidence_features(prediction, np_logits)
            validity_probability = validity.predict_proba(
                filter_scaler.transform(np.column_stack((base, confidence)))
            )[:, 1]
            record = {
                "prediction": prediction,
                "ids": ids,
                "types": classify(base, type_arrays),
                "filter": validity_probability,
                "confidence": confidence[:, 0],
                "truth": truth,
                "truth_types": instance_types(truth, semantic),
                "ignored": semantic == IGNORE,
            }
            monitor_records.append(record)

    filter_curve = [
        {"threshold": threshold, "metrics": score_records(monitor_records, "filter", threshold)}
        for threshold in np.linspace(0.1, 0.9, 17)
    ]
    confidence_curve = [
        {
            "threshold": threshold,
            "metrics": score_records(monitor_records, "confidence", threshold),
        }
        for threshold in np.linspace(0.6, 0.95, 15)
    ]
    selected_filter = choose(filter_curve)
    selected_confidence = choose(confidence_curve)
    configurations = {
        "baseline": ("confidence", 0.0),
        "learned_filter": ("filter", selected_filter["threshold"]),
        "confidence_filter": ("confidence", selected_confidence["threshold"]),
    }
    full_totals = {name: empty_score() for name in configurations}
    full_categories = {
        name: {
            "background_only": 0,
            "split_fragment": 0,
            "merge": 0,
            "boundary_localization": 0,
            "other_unresolved": 0,
        }
        for name in configurations
    }
    with torch.inference_mode():
        for index in range(len(development_images)):
            semantic, truth, _ = make_target(development_masks[index], index)
            feature_map, np_logits, hv = model_features(
                model, capture, development_images[index]
            )
            prediction = extract_instances(np_logits, hv, threshold=args.np_threshold)
            ids, base = pooled_features(
                prediction,
                feature_map,
                development_images[index].astype(np.float32) / 255.0,
            )
            confidence = confidence_features(prediction, np_logits)
            if len(base):
                validity_probability = validity.predict_proba(
                    filter_scaler.transform(np.column_stack((base, confidence)))
                )[:, 1]
                types = classify(base, type_arrays)
            else:
                validity_probability = np.empty(0)
                types = np.empty(0, dtype=np.int64)
            record = {
                "prediction": prediction,
                "ids": ids,
                "types": types,
                "filter": validity_probability,
                "confidence": confidence[:, 0],
                "truth": truth,
                "truth_types": instance_types(truth, semantic),
                "ignored": semantic == IGNORE,
            }
            for name, (mode, threshold) in configurations.items():
                score_record(
                    full_totals[name],
                    full_categories[name],
                    record,
                    mode,
                    threshold,
                )
            if (index + 1) % 256 == 0:
                print(
                    f"fold2_predictions={index + 1}/{len(development_images)}",
                    flush=True,
                )
    capture.close()
    full_results = {
        name: {
            **summarize_score(full_totals[name]),
            "false_positive_categories": full_categories[name],
        }
        for name in configurations
    }
    report = {
        "filter": FILTER_ID,
        "segmentation_checkpoint": str(args.segmentation_checkpoint),
        "segmentation_sha256": sha256(args.segmentation_checkpoint),
        "type_classifier": str(args.type_classifier),
        "type_classifier_sha256": sha256(args.type_classifier),
        "target_policy": POLICY_ID,
        "train_fold": 1,
        "development_fold": 2,
        "fold3_accessed": False,
        "np_threshold": args.np_threshold,
        "train_proposals": len(y_train),
        "train_valid_proposals": int(y_train.sum()),
        "train_excluded_zero_evaluable_proposals": excluded_ignored_proposals,
        "monitor_indices": sorted(monitor),
        "monitor_filter_curve": filter_curve,
        "monitor_confidence_curve": confidence_curve,
        "selected_filter": selected_filter,
        "selected_confidence": selected_confidence,
        "full_fold2": full_results,
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
