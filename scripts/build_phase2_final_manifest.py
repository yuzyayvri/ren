"""Build the transitive immutable production authorization manifest."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/phase2_validity_filter_repair_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    paths = {
        "classifier_epoch_30": "artifacts/phase2_path_foundation_context_classifier_v1/epoch_30.pt",
        "classifier_standardizer": "artifacts/phase2_path_foundation_context_classifier_v1/standardizer.npz",
        "corrected_validity_filter": "artifacts/phase2_validity_filter_repair_v1/fit.npz",
        "segmentation_checkpoint": "artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt",
        "hover_fast_model": "scripts/hover_fast_model.py",
        "segmentation_loader": "scripts/evaluate_hover_checkpoint.py",
        "pooled_proposal_features": "scripts/train_instance_classifier.py",
        "appearance_context_features": "scripts/appearance_context_features.py",
        "validity_feature_extraction": "scripts/evaluate_instance_suppression.py",
        "prediction_entrypoint": "scripts/run_final_protocol.py",
        "shard_verifier": "scripts/final_phase2_evaluation.py",
        "target_policy": "scripts/pannuke_target_policy.py",
        "post_processing": "scripts/hover_postprocess.py",
        "metric_matching": "scripts/nucleus_evaluation.py",
        "evaluator": "scripts/run_final_protocol.py",
        "dependency_lock": "uv.lock",
        "path_foundation_preprocessing": "scripts/extract_embeddings.py",
        "inference_configuration": "artifacts/phase2_validity_filter_repair_v1/final_execution_config.json",
        "classifier": "artifacts/phase2_path_foundation_context_classifier_v1/epoch_30.pt",
        "validity_filter": "artifacts/phase2_validity_filter_repair_v1/fit.npz",
        "configuration": "artifacts/phase2_validity_filter_repair_v1/final_execution_config.json",
    }
    files = {
        key: {"path": value, "sha256": digest(ROOT / value)}
        for key, value in paths.items()
    }
    model = ROOT / "models/vision/path-foundation"
    tree = OUT / "path_foundation_model_tree.json"
    tree_entries = [
        {"path": str(path.relative_to(model)), "sha256": digest(path)}
        for path in sorted(model.rglob("*"))
        if path.is_file()
    ]
    # Publish the closure descriptor atomically before hashing it.  A reader
    # must never observe a partially-written tree and then authorize that
    # truncated content.
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{tree.name}.", suffix=".tmp", dir=tree.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(tree_entries, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, tree)
    finally:
        Path(temporary).unlink(missing_ok=True)
    files["path_foundation_tree"] = {
        "path": str(tree.relative_to(ROOT)),
        "sha256": digest(tree),
    }
    configuration = json.loads((OUT / "final_execution_config.json").read_text())
    manifest = {
        "schema": "phase2-final-readiness-manifest-v3",
        "model_tree_root": {"path": str(model.relative_to(ROOT))},
        "files": files,
        "required_categories": sorted(files),
        "immutable": {
            "execution_configuration": configuration,
            "classifier_epoch": 30,
            "only_refitted_component": "validity_filter",
            "validity_filter_refit_completed": True,
            "classifier_retrained": False,
            "segmentation_retrained": False,
            "threshold_tuning_after_repair": False,
            "candidate_switching_after_repair": False,
        },
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=".final_candidate_manifest.", dir=OUT
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUT / "final_candidate_manifest.json")
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "manifest": str(OUT / "final_candidate_manifest.json"),
                "sha256": digest(OUT / "final_candidate_manifest.json"),
                "categories": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
