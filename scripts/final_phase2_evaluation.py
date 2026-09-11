"""Shared fail-closed validation primitives for the final Phase-2 protocol."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NP_THRESHOLD = 0.60
VALIDITY_THRESHOLD = 0.35
TARGET_POLICY = "pannuke-target-v1-ignore-ambiguous-instances"
IGNORE = 255
ROLES = ["development_rehearsal", "final_evaluation"]
GATES = {
    "detection_f1": 0.70,
    "binary_pq": 0.50,
    "end_to_end_macro_f1": 0.55,
    "dead_recall": 0.20,
    "dead_f1": 0.15,
}
REQUIRED_CATEGORIES = {
    "segmentation_checkpoint",
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
    "path_foundation_tree",
    "path_foundation_preprocessing",
    "classifier",
    "classifier_standardizer",
    "validity_filter",
    "configuration",
    "dependency_lock",
}
ARG_CATEGORIES = {
    "segmentation": "segmentation_checkpoint",
    "validity": "validity_filter",
    "standardizer": "classifier_standardizer",
    "classifier": "classifier",
    "config": "configuration",
}
CANONICAL_CODE = {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def atomic_npz(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(name, **values)
        with Path(name).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _safe_rel(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe manifest path: {value}")
    result = (root / relative).resolve()
    if result != root.resolve() and root.resolve() not in result.parents:
        raise ValueError(f"escaping manifest path: {value}")
    return result


def verify_manifest(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    files = data.get("files")
    if not isinstance(files, dict):
        raise TypeError("manifest files must be an object")
    missing = REQUIRED_CATEGORIES - files.keys()
    if missing:
        raise ValueError(f"missing required manifest categories: {sorted(missing)}")

    def resolve(item: dict) -> Path:
        scope = item.get("scope", "repository")
        if scope not in ("repository", "manifest"):
            raise ValueError("unknown manifest dependency scope")
        base = ROOT if scope == "repository" else path.parent.resolve()
        return _safe_rel(base, item["path"])

    resolved = {}
    for category, item in files.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise TypeError(f"malformed manifest dependency: {category}")
        candidate = resolve(item)
        if not candidate.is_file() or sha256(candidate) != item.get("sha256"):
            raise ValueError(f"manifest dependency mismatch: {category}")
        resolved[category] = {"path": str(candidate), "sha256": item["sha256"]}
    for category, relative in CANONICAL_CODE.items():
        if Path(resolved[category]["path"]) != (ROOT / relative).resolve():
            raise ValueError(
                f"manifest {category} does not authorize canonical runtime code"
            )
    tree = json.loads(Path(resolved["path_foundation_tree"]["path"]).read_text())
    if not isinstance(tree, list) or not tree:
        raise ValueError("empty model tree")
    tree_root_item = data.get("model_tree_root")
    if not isinstance(tree_root_item, dict) or not isinstance(
        tree_root_item.get("path"), str
    ):
        raise TypeError("manifest omits explicit model_tree_root")
    tree_root = resolve(tree_root_item)
    if not tree_root.is_dir():
        raise ValueError("model_tree_root is not a directory")
    seen = set()
    for item in tree:
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("duplicate or malformed model-tree entry")
        seen.add(relative)
        candidate = _safe_rel(tree_root, relative)
        if not candidate.is_file() or sha256(candidate) != item.get("sha256"):
            raise ValueError(f"model-tree mismatch: {relative}")
    actual = {
        str(candidate.relative_to(tree_root))
        for candidate in tree_root.rglob("*")
        if candidate.is_file()
    }
    if actual != seen:
        raise ValueError("model tree has missing or extra files")
    return {
        "manifest_sha256": sha256(path),
        "files": resolved,
        "immutable": data.get("immutable", {}),
        "categories": len(files),
    }


def load_execution_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text())
    expected = {
        "schema": "phase2-final-execution-config-v1",
        "np_threshold": NP_THRESHOLD,
        "validity_threshold": VALIDITY_THRESHOLD,
        "validity_feature_width": 111,
        "validity_feature_order": "107 pooled proposal features + 4 confidence features",
        "classifier_feature_width": 532,
        "classifier_feature_order": "148 nucleus appearance/context features + 384 Path Foundation patch embedding",
        "target_policy": TARGET_POLICY,
        "ignore_label": IGNORE,
        "matching": "strict_iou_greater_than",
        "matching_iou_threshold": 0.5,
        "roles": ROLES,
        "gates": GATES,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"configuration violates frozen {key}")
    return config


def verify_authorized_inputs(manifest: Path, supplied: dict[str, Path]) -> dict:
    verified = verify_manifest(manifest)
    for argument, category in ARG_CATEGORIES.items():
        candidate = supplied.get(argument)
        if candidate is None:
            raise ValueError(f"omitted required dependency: {argument}")
        authorized = verified["files"][category]
        if (
            str(Path(candidate).resolve()) != authorized["path"]
            or sha256(candidate) != authorized["sha256"]
        ):
            raise ValueError(
                f"supplied {argument} is not manifest-authorized as {category}"
            )
    configuration = load_execution_config(supplied["config"])
    if verified["immutable"].get("execution_configuration") != configuration:
        raise ValueError(
            "manifest immutable configuration differs from executed configuration"
        )
    return verified


def array_identity(path: Path) -> dict:
    value = np.load(path, mmap_mode="r")
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "rows": len(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def prediction_digest(shards: list[dict]) -> str:
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(np.asarray(shard["prediction"], np.int32).tobytes())
        pairs = np.asarray(
            sorted(zip(shard["proposal_ids"], shard["predicted_types"])), np.int64
        ).reshape(-1, 2)
        digest.update(pairs.tobytes())
    return digest.hexdigest()


def load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {
            key: value.item() if value.ndim == 0 else value
            for key, value in stored.items()
        }


def validate_shard_payload(payload: dict, patch: int) -> None:
    required = {
        "schema",
        "patch",
        "unfiltered_prediction",
        "prediction",
        "all_proposal_ids",
        "all_validity_scores",
        "keep_decisions",
        "proposal_ids",
        "kept_indices",
        "predicted_types",
        "patch_digest",
        "source_image_sha256",
        "embedding_sha256",
    }
    if (
        set(payload) != required
        or payload.get("schema") != "phase2-production-shard-v2"
        or payload.get("patch") != patch
    ):
        raise ValueError(f"shard schema failure {patch}")
    raw = np.asarray(payload["unfiltered_prediction"], np.int64)
    kept = np.asarray(payload["prediction"], np.int64)
    ids = np.asarray(payload["all_proposal_ids"], np.int64)
    scores = np.asarray(payload["all_validity_scores"], np.float32)
    decisions = np.asarray(payload["keep_decisions"], np.int8)
    indices = np.asarray(payload["kept_indices"], np.int64)
    kept_ids = np.asarray(payload["proposal_ids"], np.int64)
    types = np.asarray(payload["predicted_types"], np.int64)
    raw_ids = np.unique(raw)
    raw_ids = raw_ids[raw_ids > 0]
    if (
        raw.ndim != 2
        or kept.shape != raw.shape
        or len(np.unique(ids)) != len(ids)
        or not np.array_equal(ids, raw_ids)
    ):
        raise ValueError(f"proposal ID coverage failure {patch}")
    exact = (scores >= np.float32(VALIDITY_THRESHOLD)).astype(np.int8)
    exact_indices = np.flatnonzero(exact)
    if (
        len(ids) != len(scores)
        or len(ids) != len(decisions)
        or not np.isfinite(scores).all()
        or not np.array_equal(decisions, exact)
    ):
        raise ValueError(f"validity alignment failure {patch}")
    if not np.array_equal(indices, exact_indices) or not np.array_equal(
        kept_ids, ids[exact_indices]
    ):
        raise ValueError(f"retention alignment failure {patch}")
    if not np.array_equal(kept, np.where(np.isin(raw, kept_ids), raw, 0)):
        raise ValueError(f"retained prediction reconstruction failure {patch}")
    if len(types) != len(kept_ids) or np.any((types < 1) | (types > 5)):
        raise ValueError(f"predicted type failure {patch}")
    local = hashlib.sha256(
        kept.astype(np.int32).tobytes()
        + np.asarray(sorted(zip(kept_ids, types)), np.int64).reshape(-1, 2).tobytes()
    ).hexdigest()
    if payload["patch_digest"] != local:
        raise ValueError(f"patch digest mismatch {patch}")


def validate_prediction_packet(run: Path, state: dict) -> dict:
    expected = int(state["prediction_source"]["rows"])
    completed = state.get("completed_shards")
    records = state.get("shard_records", {})
    if not isinstance(completed, list) or completed != sorted(set(completed)):
        raise ValueError("duplicate or unordered completed coverage")
    if any(
        not isinstance(index, int) or index < 0 or index >= expected
        for index in completed
    ):
        raise ValueError("completed coverage outside source")
    directory = Path(run) / "shards"
    payloads = list(directory.glob("shard_*.npz")) if directory.is_dir() else []
    metadata = list(directory.glob("shard_*.meta.json")) if directory.is_dir() else []
    if {item.name for item in payloads} != {
        f"shard_{index:06d}.npz" for index in completed
    } or {item.name for item in metadata} != {
        f"shard_{index:06d}.meta.json" for index in completed
    }:
        raise ValueError("missing, extra, or incomplete shard file coverage")
    shards = []
    for index in completed:
        record = records.get(str(index))
        if not isinstance(record, dict):
            raise TypeError(f"missing shard record {index}")
        payload_path = directory / f"shard_{index:06d}.npz"
        metadata_path = directory / f"shard_{index:06d}.meta.json"
        if sha256(payload_path) != record.get("payload_sha256") or sha256(
            metadata_path
        ) != record.get("metadata_sha256"):
            raise ValueError(f"state shard hash mismatch {index}")
        payload = load_shard(payload_path)
        validate_shard_payload(payload, index)
        expected_meta = {
            "schema": "phase2-production-shard-metadata-v2",
            "patch": index,
            "payload": payload_path.name,
            "payload_sha256": sha256(payload_path),
            "manifest_sha256": state["manifest_sha256"],
            "configuration_sha256": state["bindings"]["config"],
            "source_image_sha256": record["source_image_sha256"],
            "embedding_sha256": record["embedding_sha256"],
        }
        if json.loads(metadata_path.read_text()) != expected_meta:
            raise ValueError(f"metadata binding mismatch {index}")
        if (
            payload["source_image_sha256"] != record["source_image_sha256"]
            or payload["embedding_sha256"] != record["embedding_sha256"]
        ):
            raise ValueError(f"source binding mismatch {index}")
        shards.append(payload)
    if set(records) != {str(index) for index in completed}:
        raise ValueError("duplicate or extra shard records")
    return {
        "coverage": len(shards),
        "expected_coverage": expected,
        "shards": shards,
        "digest": prediction_digest(shards),
    }
