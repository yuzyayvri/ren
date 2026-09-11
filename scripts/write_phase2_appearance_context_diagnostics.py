"""Reconstruct full fold2 diagnostics for the retained appearance/context MLP."""

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from nucleus_evaluation import match_instances
from verify_phase2_appearance_context_cache import load_dataset

ROOT = Path("artifacts")
EXP = ROOT / "phase2_appearance_context_classifier_v1"
OUT = EXP / "diagnostics.json"
CLASS_NAMES = {
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "dead",
    5: "epithelial",
}
IOU_BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0000001))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict(features):
    with np.load(EXP / "standardizer.npz") as scaler:
        x = ((features - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(148, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 5),
    )
    model.load_state_dict(
        torch.load(EXP / "epoch_29.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(x)).argmax(1).numpy().astype(np.int64) + 1


def accuracy_record(correct, total):
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": float(correct / total) if total else None,
    }


def main():
    appearance = json.loads((EXP / "report.json").read_text())
    true_mask = json.loads((EXP / "true_mask_eval_run1.json").read_text())
    details = json.loads(
        (
            ROOT / "phase2_instance_suppression_v1" / "full_fold2_details.json"
        ).read_text()
    )
    truth_rows = load_dataset(EXP / "cache", "fold2_truth")
    proposal_rows = load_dataset(EXP / "cache", "fold2_predicted")
    truth_predictions = predict(
        np.c_[truth_rows["base_features"], truth_rows["added_features"]]
    )
    proposal_predictions = predict(
        np.c_[proposal_rows["base_features"], proposal_rows["added_features"]]
    )
    truth_lookup = {
        (int(p), int(i)): int(y)
        for p, i, y in zip(
            truth_rows["patch_index"], truth_rows["instance_id"], truth_predictions
        )
    }
    proposal_lookup = {
        (int(p), int(i)): int(y)
        for p, i, y in zip(
            proposal_rows["patch_index"],
            proposal_rows["instance_id"],
            proposal_predictions,
        )
    }
    tissue_by_patch = {
        int(p): str(t)
        for p, t in zip(truth_rows["patch_index"], truth_rows["tissue_label"])
    }

    confusion = np.zeros((5, 5), dtype=np.int64)
    paired = {"proposal_correct": 0, "truth_correct": 0, "total": 0}
    tissue = defaultdict(lambda: [0, 0])
    iou_bins = {f"({lo:.1f},{min(hi, 1):.1f}]": [0, 0] for lo, hi in IOU_BINS}
    unmatched = Counter()
    dead_matched_mistyped = 0
    canonical = ROOT / "phase2_nonlinear_classifier_v1" / "fold2_predicted_cache.pkl"
    with canonical.open("rb") as f:
        records = pickle.load(f)
    for patch, record in records.items():
        matches, false_positives, _ = match_instances(
            record["prediction"], record["truth"], record["ignored"]
        )
        for proposal_id, truth_id, iou in matches:
            actual = int(record["truth_types"][truth_id])
            proposal_class = proposal_lookup[(int(patch), int(proposal_id))]
            truth_class = truth_lookup[(int(patch), int(truth_id))]
            confusion[actual - 1, proposal_class - 1] += 1
            paired["proposal_correct"] += proposal_class == actual
            paired["truth_correct"] += truth_class == actual
            paired["total"] += 1
            values = tissue[tissue_by_patch[int(patch)]]
            values[0] += proposal_class == actual
            values[1] += 1
            for lo, hi in IOU_BINS:
                if lo < iou <= hi:
                    values = iou_bins[f"({lo:.1f},{min(hi, 1):.1f}]"]
                    values[0] += proposal_class == actual
                    values[1] += 1
                    break
            dead_matched_mistyped += proposal_class == 4 and actual != 4
        for proposal_id in false_positives:
            unmatched[CLASS_NAMES[proposal_lookup[(int(patch), int(proposal_id))]]] += 1

    diagnostics = {
        "experiment": "phase2-appearance-context-classifier-v1",
        "development_fold": 2,
        "fold3_accessed": False,
        "retention": {
            "retained": True,
            "accepted": False,
            "failing_gate": "end-to-end macro-F1 >= 0.55",
        },
        "full_fold2_end_to_end": appearance["full_fold2_mlp"],
        "augmented_valid_full_fold2_true_mask": true_mask,
        "control_valid_full_fold2_true_mask": json.loads(
            (
                ROOT
                / "phase2_nonlinear_classifier_v1"
                / "full_fold2_true_mask_report.json"
            ).read_text()
        ),
        "matched_type_confusion_rows_truth_columns_prediction": confusion.tolist(),
        "paired_mask_results_same_matches": {
            "proposal_mask": accuracy_record(
                paired["proposal_correct"], paired["total"]
            ),
            "truth_mask": accuracy_record(paired["truth_correct"], paired["total"]),
        },
        "unmatched_retained_prediction_class_counts": dict(sorted(unmatched.items())),
        "tissue_stratified_matched_typing": {
            name: accuracy_record(*values) for name, values in sorted(tissue.items())
        },
        "iou_bin_matched_typing": {
            name: accuracy_record(*values) for name, values in iou_bins.items()
        },
        "dead_false_positives": {
            "mistyped_matched": int(dead_matched_mistyped),
            "unmatched_proposals": int(unmatched["dead"]),
            "total": int(dead_matched_mistyped + unmatched["dead"]),
        },
        "false_positive_categories": appearance["full_fold2_mlp"][
            "false_positive_categories"
        ],
        "geometry_invariant": appearance["geometry_invariant"],
        "hashes": {
            "checkpoint": digest(EXP / "epoch_29.pt"),
            "standardizer": digest(EXP / "standardizer.npz"),
            "segmentation": details["segmentation_sha256"],
            "validity_filter": details["validity_filter_sha256"],
            "proposal_cache": digest(canonical),
            "feature_cache_manifest": digest(EXP / "cache" / "manifest.json"),
        },
    }
    OUT.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(OUT), "matches": paired["total"], "fold3_accessed": False},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
