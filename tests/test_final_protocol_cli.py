"""Real subprocess lifecycle tests for the canonical production CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hover_fast_model import HoVerFast

CLI = ROOT / "scripts/run_final_protocol.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(directory: Path) -> dict[str, Path]:
    directory.mkdir()
    images = directory / "images.npy"
    np.save(images, np.zeros((2, 256, 256, 3), np.uint8))
    embeddings = directory / "embeddings.npy"
    np.save(embeddings, np.zeros((2, 384), np.float32))
    masks = directory / "masks.npy"
    raw_masks = np.zeros((2, 256, 256, 6), np.int32)
    raw_masks[..., 5] = 1
    np.save(masks, raw_masks)
    types = directory / "types.npy"
    np.save(types, np.asarray(["a", "b"]))
    segmentation = directory / "segmentation.pt"
    torch.save({"state_dict": HoVerFast().state_dict()}, segmentation)
    validity = directory / "validity.npz"
    np.savez(
        validity,
        mean=np.zeros(111),
        scale=np.ones(111),
        coefficients=np.zeros((1, 111)),
        intercept=np.zeros(1),
    )
    standardizer = directory / "standardizer.npz"
    np.savez(standardizer, mean=np.zeros(532), scale=np.ones(532))
    classifier = directory / "classifier.pt"
    head = torch.nn.Sequential(
        torch.nn.Linear(532, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 5),
    )
    torch.save(head.state_dict(), classifier)
    config = directory / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema": "phase2-final-execution-config-v1",
                "np_threshold": 0.6,
                "validity_threshold": 0.35,
                "validity_feature_width": 111,
                "validity_feature_order": "107 pooled proposal features + 4 confidence features",
                "classifier_feature_width": 532,
                "classifier_feature_order": "148 nucleus appearance/context features + 384 Path Foundation patch embedding",
                "target_policy": "pannuke-target-v1-ignore-ambiguous-instances",
                "ignore_label": 255,
                "matching": "strict_iou_greater_than",
                "matching_iou_threshold": 0.5,
                "roles": ["development_rehearsal", "final_evaluation"],
                "gates": {
                    "detection_f1": 0.7,
                    "binary_pq": 0.5,
                    "end_to_end_macro_f1": 0.55,
                    "dead_recall": 0.2,
                    "dead_f1": 0.15,
                },
            },
            sort_keys=True,
        )
    )
    model_directory = directory / "model_tree"
    model_directory.mkdir()
    model = model_directory / "model.bin"
    model.write_bytes(b"model")
    tree = directory / "tree.json"
    tree.write_text(json.dumps([{"path": "model.bin", "sha256": digest(model)}]))
    paths = {
        "segmentation_checkpoint": segmentation,
        "classifier_standardizer": standardizer,
        "classifier": classifier,
        "validity_filter": validity,
        "configuration": config,
        "path_foundation_tree": tree,
    }
    canonical = {
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
        "path_foundation_preprocessing": "scripts/extract_embeddings.py",
        "dependency_lock": "uv.lock",
    }
    files = {
        key: {"path": value.name, "sha256": digest(value), "scope": "manifest"}
        for key, value in paths.items()
    }
    files.update(
        {
            key: {"path": value, "sha256": digest(ROOT / value)}
            for key, value in canonical.items()
        }
    )
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_tree_root": {"path": model_directory.name, "scope": "manifest"},
                "files": files,
                "immutable": {
                    "execution_configuration": json.loads(config.read_text())
                },
            },
            sort_keys=True,
        )
    )
    return {
        "run": directory / "run",
        "manifest": manifest,
        "config": config,
        "images": images,
        "embeddings": embeddings,
        "segmentation": segmentation,
        "validity": validity,
        "standardizer": standardizer,
        "classifier": classifier,
        "masks": masks,
        "types": types,
    }


def command(paths: dict[str, Path], action: str, *extra: str) -> list[str]:
    result = [sys.executable, str(CLI), action]
    for name in (
        "run",
        "manifest",
        "config",
        "images",
        "embeddings",
        "segmentation",
        "validity",
        "standardizer",
        "classifier",
    ):
        result += [f"--{name}", str(paths[name])]
    return [*result, *extra]


def run(
    command_value: list[str], expected: int = 0
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command_value, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def test_real_cli_lifecycle_interrupt_resume_and_rescore(tmp_path):
    paths = fixture(tmp_path / "fixture")
    run(command(paths, "initialize", "--role", "final_evaluation"))
    run(command(paths, "predict", "--role", "development_rehearsal"), 1)
    run(command(paths, "predict", "--stop-after-shards", "1"), 2)
    run(command(paths, "freeze-predictions"), 1)
    run(
        command(
            paths,
            "score",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(paths["types"]),
            "--output",
            str(tmp_path / "early.json"),
        ),
        1,
    )
    partial = paths["run"] / "shards/shard_000001.npz"
    partial.write_bytes(b"interrupted payload")
    (paths["run"] / "shards/shard_000001.meta.json").write_text("{}")
    run(command(paths, "predict"))
    run(command(paths, "freeze-predictions"))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    run(
        command(
            paths,
            "score",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(paths["types"]),
            "--output",
            str(first),
        )
    )
    run(
        command(
            paths,
            "score",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(paths["types"]),
            "--output",
            str(second),
        )
    )
    assert first.read_bytes() == second.read_bytes()
    run(
        command(
            paths,
            "complete",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(paths["types"]),
        )
    )
    run(
        command(
            paths,
            "verify",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(paths["types"]),
        )
    )
    state = json.loads((paths["run"] / "state.json").read_text())
    assert state["state"] == "complete" and state["frozen_coverage"] == 2


def test_cli_rejects_changed_dependency_config_shard_and_truth(tmp_path):
    paths = fixture(tmp_path / "fixture")
    run(command(paths, "initialize", "--role", "final_evaluation"))
    run(command(paths, "predict"))
    run(command(paths, "freeze-predictions"))
    for name in ("segmentation", "validity", "standardizer", "classifier", "config"):
        altered = tmp_path / f"altered_{name}{paths[name].suffix}"
        altered.write_bytes(paths[name].read_bytes() + b"x")
        run(command(dict(paths, **{name: altered}), "verify"), 1)
    changed_images = tmp_path / "changed_images.npy"
    np.save(changed_images, np.ones((2, 256, 256, 3), np.uint8))
    run(command(dict(paths, images=changed_images), "verify"), 1)
    shard = paths["run"] / "shards/shard_000000.npz"
    original = shard.read_bytes()
    shard.write_bytes(original + b" ")
    run(command(paths, "verify"), 1)
    shard.write_bytes(original)
    metadata = paths["run"] / "shards/shard_000000.meta.json"
    original_metadata = metadata.read_bytes()
    metadata.write_bytes(original_metadata + b" ")
    run(command(paths, "verify"), 1)
    metadata.write_bytes(original_metadata)
    shard.unlink()
    run(command(paths, "verify"), 1)
    shard.write_bytes(original)
    state_path = paths["run"] / "state.json"
    state = json.loads(state_path.read_text())
    original_digest = state["prediction_digest"]
    state["completed_shards"].append(0)
    state_path.write_text(json.dumps(state))
    run(command(paths, "verify"), 1)
    state["completed_shards"] = [0, 1]
    state["prediction_digest"] = "0" * 64
    state_path.write_text(json.dumps(state))
    run(command(paths, "verify"), 1)
    state["prediction_digest"] = original_digest
    state_path.write_text(json.dumps(state))
    short_types = tmp_path / "short_types.npy"
    np.save(short_types, np.asarray(["a"]))
    run(
        command(
            paths,
            "score",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(short_types),
            "--output",
            str(tmp_path / "short.json"),
        ),
        1,
    )
    report = tmp_path / "report.json"
    run(
        command(
            paths,
            "score",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(paths["types"]),
            "--output",
            str(report),
        )
    )
    changed_types = tmp_path / "changed_types.npy"
    np.save(changed_types, np.asarray(["x", "b"]))
    run(
        command(
            paths,
            "score",
            "--masks",
            str(paths["masks"]),
            "--types",
            str(changed_types),
            "--output",
            str(tmp_path / "bad.json"),
        ),
        1,
    )


def test_initialization_rejects_missing_dependency_and_semantic_drift(tmp_path):
    paths = fixture(tmp_path / "fixture")
    manifest = json.loads(paths["manifest"].read_text())
    del manifest["files"]["metric_matching"]
    paths["manifest"].write_text(json.dumps(manifest))
    run(command(paths, "initialize", "--role", "final_evaluation"), 1)


def test_manifest_must_authorize_exact_runtime_code_and_complete_model_tree(tmp_path):
    paths = fixture(tmp_path / "fixture")
    original = json.loads(paths["manifest"].read_text())
    categories = (
        "hover_fast_model",
        "segmentation_loader",
        "pooled_proposal_features",
        "appearance_context_features",
        "validity_feature_extraction",
        "prediction_entrypoint",
        "shard_verifier",
        "target_policy",
        "post_processing",
        "metric_matching",
        "evaluator",
        "path_foundation_preprocessing",
        "dependency_lock",
    )
    for offset, category in enumerate(categories):
        changed = json.loads(json.dumps(original))
        replacement = (
            "target_policy" if category != "target_policy" else "hover_fast_model"
        )
        changed["files"][category] = changed["files"][replacement]
        paths["manifest"].write_text(json.dumps(changed))
        run(
            command(
                dict(paths, run=tmp_path / f"run_{offset}"),
                "initialize",
                "--role",
                "final_evaluation",
            ),
            1,
        )
    paths["manifest"].write_text(json.dumps(original))
    extra = tmp_path / "fixture/model_tree/extra.bin"
    extra.write_bytes(b"unauthorized")
    run(
        command(
            dict(paths, run=tmp_path / "tree_run"),
            "initialize",
            "--role",
            "final_evaluation",
        ),
        1,
    )
    extra.unlink()
    changed = json.loads(json.dumps(original))
    changed["immutable"]["execution_configuration"]["np_threshold"] = 0.61
    paths["manifest"].write_text(json.dumps(changed))
    run(
        command(
            dict(paths, run=tmp_path / "immutable_run"),
            "initialize",
            "--role",
            "final_evaluation",
        ),
        1,
    )
    paths = fixture(tmp_path / "second")
    config = json.loads(paths["config"].read_text())
    config["validity_threshold"] = 0.36
    paths["config"].write_text(json.dumps(config))
    manifest = json.loads(paths["manifest"].read_text())
    manifest["files"]["configuration"]["sha256"] = digest(paths["config"])
    paths["manifest"].write_text(json.dumps(manifest))
    run(command(paths, "initialize", "--role", "final_evaluation"), 1)


def test_cli_roles_digest_modes_and_manifest_omissions(tmp_path):
    paths = fixture(tmp_path / "fixture")
    run(command(paths, "initialize", "--role", "development_rehearsal"))
    # A caller cannot override the persisted role, even before prediction.
    run(command(paths, "predict", "--role", "final_evaluation"), 1)
    run(command(paths, "predict", "--stop-after-shards", "1"), 2)
    # Development rehearsal freeze is digest-bound and therefore requires it.
    run(command(paths, "freeze-predictions"), 1)

    final = fixture(tmp_path / "final")
    run(command(final, "initialize", "--role", "final_evaluation"))
    run(command(final, "predict"))
    # Final evaluation has no predeclared digest requirement.
    run(command(final, "freeze-predictions"))

    omitted = json.loads(final["manifest"].read_text())
    del omitted["files"]["appearance_context_features"]
    final["manifest"].write_text(json.dumps(omitted))
    run(
        command(
            dict(final, run=tmp_path / "omitted_run"),
            "initialize",
            "--role",
            "final_evaluation",
        ),
        1,
    )


def test_cli_rejects_coverage_and_state_publication_corruption(tmp_path):
    paths = fixture(tmp_path / "fixture")
    run(command(paths, "initialize", "--role", "final_evaluation"))
    run(command(paths, "predict"))
    shards = paths["run"] / "shards"
    state_path = paths["run"] / "state.json"
    state = json.loads(state_path.read_text())
    # Duplicate coverage in the persisted state is rejected by the verifier.
    state["completed_shards"] = [0, 0, 1]
    state_path.write_text(json.dumps(state))
    run(command(paths, "verify"), 1)
    state["completed_shards"] = [0, 1]
    state_path.write_text(json.dumps(state))

    payload = shards / "shard_000000.npz"
    metadata = shards / "shard_000000.meta.json"
    payload_bytes, metadata_bytes = payload.read_bytes(), metadata.read_bytes()
    payload.write_bytes(payload_bytes + b"tampered")
    run(command(paths, "verify"), 1)
    payload.write_bytes(payload_bytes)
    metadata.write_bytes(metadata_bytes + b"tampered")
    run(command(paths, "verify"), 1)
    metadata.write_bytes(metadata_bytes)
    payload.unlink()
    run(command(paths, "verify"), 1)

    # An uncommitted pair is safely discarded on resume, while a state-listed
    # missing shard remains a hard failure.
    payload.write_bytes(payload_bytes)
    state = json.loads(state_path.read_text())
    state["completed_shards"] = [0]
    # Model an interruption before shard 1's state publication: the payload
    # and metadata exist, but no committed record does.
    state["shard_records"].pop("1")
    state_path.write_text(json.dumps(state))
    orphan = shards / "shard_000001.npz"
    orphan.write_bytes(b"orphan")
    (shards / "shard_000001.meta.json").write_text("{}")
    run(command(paths, "predict"))
    assert orphan.read_bytes() != b"orphan"
    payload.unlink()
    run(command(paths, "verify"), 1)
