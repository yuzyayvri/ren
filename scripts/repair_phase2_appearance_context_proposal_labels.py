"""Repair published fold2 proposal labels without changing feature rows."""

from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from build_phase2_appearance_context_cache import matched_proposal_labels
from verify_phase2_appearance_context_cache import SCHEMA, sha

ROOT = Path("artifacts/phase2_appearance_context_classifier_v1/cache")
DATASET = ROOT / "fold2_predicted"
SUPERSEDED = ROOT.parent / "fold2_predicted_labels_superseded"
CANONICAL = Path("artifacts/phase2_nonlinear_classifier_v1/fold2_predicted_cache.pkl")


def main():
    if not SUPERSEDED.exists():
        shutil.copytree(DATASET, SUPERSEDED)
    with CANONICAL.open("rb") as handle:
        records = pickle.load(handle)
    changed = 0
    metas = []
    for path in sorted(DATASET.glob("shard_*.npz")):
        with np.load(path) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        labels = data["labels"].copy()
        for patch in np.unique(data["patch_index"]):
            selected = data["patch_index"] == patch
            record = records[int(patch)]
            labels[selected] = matched_proposal_labels(
                record["prediction"],
                record["truth"],
                record["truth_types"],
                record["ignored"],
                data["instance_id"][selected],
            )
        changed += int(np.count_nonzero(labels != data["labels"]))
        data["labels"] = labels
        temporary = path.with_suffix(".repair.npz")
        np.savez_compressed(temporary, **data)
        os.replace(temporary, path)
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["sha256"] = sha(path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        metas.append(metadata)

    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"] = metas
    labels = np.concatenate(
        [np.load(path)["labels"] for path in sorted(DATASET.glob("shard_*.npz"))]
    )
    manifest["class_support"] = {
        str(int(value)): int(np.count_nonzero(labels == value))
        for value in np.unique(labels)
    }
    manifest["label_semantics"] = {
        "-1": "ignored_no_evaluable_pixels",
        "0": "evaluable_unmatched",
        "1..5": "IoU-matched_truth_class",
    }
    manifest["label_repair"] = {
        "method": "one-to-one IoU > 0.5 matching",
        "changed_rows": changed,
        "fold3_accessed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    root_manifest_path = ROOT / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text())
    root_manifest["datasets"] = [
        manifest if item["dataset"] == "fold2_predicted" else item
        for item in root_manifest["datasets"]
    ]
    root_manifest_path.write_text(json.dumps(root_manifest, indent=2) + "\n")
    assert manifest["schema"] == SCHEMA
    print(
        json.dumps(
            {
                "changed_rows": changed,
                "superseded_copy": str(SUPERSEDED),
                "fold3_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
