#!/usr/bin/env python3
"""Train a decoupled fold1 instance classifier and evaluate on fold2 only."""

from __future__ import annotations

import argparse
import hashlib
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
from nucleus_evaluation import add_scores, empty_score, score_instances, summarize_score
from pannuke_target_policy import IGNORE, POLICY_ID, make_target
from pilot_hover_fast import fold1_arrays
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSIFIER_ID = "ren-instance-classifier-v1-encoder-decoder-statistics"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureCapture:
    def __init__(self, model: torch.nn.Module):
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = [
            model.encoder1.register_forward_hook(self._hook("encoder1")),
            model.decoder1.register_forward_hook(self._hook("decoder1")),
        ]

    def _hook(self, name: str):
        def save(_module, _inputs, output):
            self.outputs[name] = output.detach()

        return save

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def pooled_features(
    labels: np.ndarray, feature_map: np.ndarray, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool appearance and geometry for every nonzero instance label."""
    ids = np.unique(labels)
    ids = ids[ids > 0].astype(np.int64)
    if not len(ids):
        return ids, np.empty((0, feature_map.shape[0] * 2 + 11), np.float32)
    flat_labels = labels.ravel()
    length = int(flat_labels.max()) + 1
    area = np.bincount(flat_labels, minlength=length).astype(np.float64)
    columns = []
    for channel in feature_map:
        values = channel.ravel().astype(np.float64)
        total = np.bincount(flat_labels, weights=values, minlength=length)
        square = np.bincount(flat_labels, weights=values * values, minlength=length)
        mean = total[ids] / area[ids]
        variance = np.maximum(square[ids] / area[ids] - mean * mean, 0)
        columns.extend((mean, np.sqrt(variance)))
    for channel in image.transpose(2, 0, 1):
        values = channel.ravel().astype(np.float64)
        total = np.bincount(flat_labels, weights=values, minlength=length)
        square = np.bincount(flat_labels, weights=values * values, minlength=length)
        mean = total[ids] / area[ids]
        variance = np.maximum(square[ids] / area[ids] - mean * mean, 0)
        columns.extend((mean, np.sqrt(variance)))
    yy, xx = np.indices(labels.shape)
    for coordinate in (yy, xx):
        total = np.bincount(
            flat_labels, weights=coordinate.ravel(), minlength=length
        )
        columns.append(total[ids] / area[ids] / 255.0)
    columns.extend(
        (
            np.log1p(area[ids]) / 10.0,
            np.sqrt(area[ids]) / 256.0,
            np.ones(len(ids)),
        )
    )
    return ids, np.column_stack(columns).astype(np.float32)


def infer_features(
    model: torch.nn.Module,
    capture: FeatureCapture,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]
    np_logits, hv, _ = model(tensor)
    feature_map = torch.cat(
        (capture.outputs["encoder1"], capture.outputs["decoder1"]), dim=1
    )[0].numpy()
    return feature_map, np_logits[0, 0].numpy(), hv[0].numpy()


def predict(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    logits = ((features - mean) / scale) @ coefficients.T + intercept
    return classes[logits.argmax(1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    model = load_model(args.segmentation_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = FeatureCapture(model)

    train_images, train_masks = fold1_arrays()
    feature_batches = []
    label_batches = []
    with torch.inference_mode():
        for index in range(len(train_images)):
            semantic, instance, _ = make_target(train_masks[index], index)
            feature_map, _, _ = infer_features(model, capture, train_images[index])
            ids, features = pooled_features(
                instance, feature_map, train_images[index].astype(np.float32) / 255.0
            )
            types = instance_types(instance, semantic)
            feature_batches.append(features)
            label_batches.append(np.array([types[int(ident)] for ident in ids]))
            if (index + 1) % 256 == 0:
                print(f"fold1_features={index + 1}/{len(train_images)}", flush=True)
    train_features = np.concatenate(feature_batches)
    train_labels = np.concatenate(label_batches)
    scaler = StandardScaler().fit(train_features)
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=20260908,
    ).fit(scaler.transform(train_features), train_labels)
    np.savez_compressed(
        args.output / "classifier.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficients=classifier.coef_,
        intercept=classifier.intercept_,
        classes=classifier.classes_,
    )

    development_images, development_masks = fold2_arrays()
    indices = coverage_indices(development_masks)
    true_total = empty_score()
    predicted_total = empty_score()
    with torch.inference_mode():
        for index in indices:
            semantic, truth, _ = make_target(development_masks[index], index)
            truth_types = instance_types(truth, semantic)
            image = development_images[index]
            feature_map, np_logits, hv = infer_features(model, capture, image)
            normalized_image = image.astype(np.float32) / 255.0

            truth_ids, truth_features = pooled_features(truth, feature_map, normalized_image)
            truth_predictions = predict(
                truth_features,
                scaler.mean_,
                scaler.scale_,
                classifier.coef_,
                classifier.intercept_,
                classifier.classes_,
            )
            add_scores(
                true_total,
                score_instances(
                    truth,
                    truth,
                    dict(zip(truth_ids.tolist(), truth_predictions.tolist())),
                    truth_types,
                    ignored=semantic == IGNORE,
                ),
            )

            predicted = extract_instances(np_logits, hv, threshold=args.threshold)
            predicted_ids, predicted_features = pooled_features(
                predicted, feature_map, normalized_image
            )
            predicted_classes = predict(
                predicted_features,
                scaler.mean_,
                scaler.scale_,
                classifier.coef_,
                classifier.intercept_,
                classifier.classes_,
            )
            add_scores(
                predicted_total,
                score_instances(
                    predicted,
                    truth,
                    dict(zip(predicted_ids.tolist(), predicted_classes.tolist())),
                    truth_types,
                    ignored=semantic == IGNORE,
                ),
            )
    capture.close()
    report = {
        "classifier": CLASSIFIER_ID,
        "segmentation_checkpoint": str(args.segmentation_checkpoint),
        "segmentation_sha256": sha256(args.segmentation_checkpoint),
        "target_policy": POLICY_ID,
        "train_fold": 1,
        "development_fold": 2,
        "fold3_accessed": False,
        "threshold": args.threshold,
        "train_instances": len(train_labels),
        "train_support": np.bincount(
            train_labels.astype(np.int64), minlength=6
        )[1:].tolist(),
        "development_indices": indices,
        "true_instance_classification": summarize_score(true_total),
        "predicted_instance_end_to_end": summarize_score(predicted_total),
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
