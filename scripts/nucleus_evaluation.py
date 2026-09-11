"""One-to-one nucleus detection, typing, and panoptic-quality evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _instance_ids(labels: np.ndarray, ignored: np.ndarray) -> list[int]:
    return [
        int(ident)
        for ident in np.unique(labels)
        if ident and np.any((labels == ident) & ~ignored)
    ]


def match_instances(
    prediction: np.ndarray,
    truth: np.ndarray,
    ignored: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily match instances by descending IoU above a strict threshold."""
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes differ")
    ignored = np.zeros_like(truth, dtype=bool) if ignored is None else ignored.astype(bool)
    if ignored.shape != truth.shape:
        raise ValueError("ignored mask and truth shapes differ")
    predicted_ids = _instance_ids(prediction, ignored)
    truth_ids = _instance_ids(truth, ignored)
    candidates: list[tuple[float, int, int]] = []
    truth_masks = {ident: (truth == ident) & ~ignored for ident in truth_ids}
    for predicted_id in predicted_ids:
        predicted_mask = (prediction == predicted_id) & ~ignored
        for truth_id, truth_mask in truth_masks.items():
            intersection = int(np.count_nonzero(predicted_mask & truth_mask))
            if not intersection:
                continue
            union = int(np.count_nonzero(predicted_mask | truth_mask))
            iou = intersection / union
            if iou > threshold:
                candidates.append((iou, predicted_id, truth_id))

    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, predicted_id, truth_id in sorted(candidates, reverse=True):
        if predicted_id not in used_predictions and truth_id not in used_truth:
            used_predictions.add(predicted_id)
            used_truth.add(truth_id)
            matches.append((predicted_id, truth_id, iou))
    return (
        matches,
        [ident for ident in predicted_ids if ident not in used_predictions],
        [ident for ident in truth_ids if ident not in used_truth],
    )


def empty_counts() -> dict[str, float | int]:
    return {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}


def detection_counts(
    prediction: np.ndarray, truth: np.ndarray, ignored: np.ndarray | None = None
) -> dict[str, float | int]:
    matches, false_positives, false_negatives = match_instances(prediction, truth, ignored)
    return {
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "sum_iou": float(sum(match[2] for match in matches)),
    }


def _validate_types(types: Mapping[int, int], ids: list[int], classes: int, label: str) -> None:
    missing = [ident for ident in ids if ident not in types]
    if missing:
        raise ValueError(f"{label} types missing instance IDs {missing[:8]}")
    invalid = {ident: int(types[ident]) for ident in ids if not 1 <= int(types[ident]) <= classes}
    if invalid:
        raise ValueError(f"{label} types outside 1..{classes}: {invalid}")


def score_instances(
    prediction: np.ndarray,
    truth: np.ndarray,
    predicted_types: Mapping[int, int],
    truth_types: Mapping[int, int],
    ignored: np.ndarray | None = None,
    classes: int = 5,
) -> dict[str, object]:
    """Return raw additive counts; wrong matched types count as one FP and FN."""
    matches, false_positives, false_negatives = match_instances(prediction, truth, ignored)
    predicted_ids = [predicted_id for predicted_id, _, _ in matches] + false_positives
    truth_ids = [truth_id for _, truth_id, _ in matches] + false_negatives
    _validate_types(predicted_types, predicted_ids, classes, "prediction")
    _validate_types(truth_types, truth_ids, classes, "truth")

    detection = {
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "sum_iou": float(sum(match[2] for match in matches)),
    }
    per_class = {str(class_id): empty_counts() for class_id in range(1, classes + 1)}
    for predicted_id in false_positives:
        per_class[str(int(predicted_types[predicted_id]))]["fp"] += 1
    for truth_id in false_negatives:
        per_class[str(int(truth_types[truth_id]))]["fn"] += 1
    for predicted_id, truth_id, iou in matches:
        predicted_class = int(predicted_types[predicted_id])
        truth_class = int(truth_types[truth_id])
        if predicted_class == truth_class:
            counts = per_class[str(truth_class)]
            counts["tp"] += 1
            counts["sum_iou"] += iou
        else:
            per_class[str(predicted_class)]["fp"] += 1
            per_class[str(truth_class)]["fn"] += 1
    return {"detection": detection, "classes": per_class}


def add_scores(total: dict[str, object], score: dict[str, object]) -> None:
    """Add one patch's raw score into an aggregate in place."""
    for key in empty_counts():
        total["detection"][key] += score["detection"][key]
    for class_id, counts in total["classes"].items():
        for key in empty_counts():
            counts[key] += score["classes"][class_id][key]


def empty_score(classes: int = 5) -> dict[str, object]:
    return {
        "detection": empty_counts(),
        "classes": {str(class_id): empty_counts() for class_id in range(1, classes + 1)},
    }


def metrics(counts: Mapping[str, float | int]) -> dict[str, float | int]:
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    denominator = tp + 0.5 * (fp + fn)
    return {
        **counts,
        "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "pq": float(counts["sum_iou"]) / denominator if denominator else 0.0,
    }


def summarize_score(score: dict[str, object]) -> dict[str, object]:
    detection = metrics(score["detection"])
    classes = {class_id: metrics(counts) for class_id, counts in score["classes"].items()}
    return {
        "detection": detection,
        "binary_pq": detection["pq"],
        "classes": classes,
        "macro_f1": float(np.mean([counts["f1"] for counts in classes.values()])),
        "macro_pq": float(np.mean([counts["pq"] for counts in classes.values()])),
    }
