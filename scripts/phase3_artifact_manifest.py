"""Fail-closed provenance helpers shared by Phase 3 stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_manifest(path: Path, *, stage: str, config: dict, inputs: dict) -> dict:
    if not stage or not isinstance(config, dict) or not isinstance(inputs, dict):
        raise ValueError("invalid manifest fields")
    manifest = {"schema": "phase3-artifact-v1", "stage": stage, "config": config, "inputs": inputs}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return manifest


def require_unexposed(marker: Path) -> None:
    if marker.exists():
        raise RuntimeError(f"test exposure already recorded: {marker}")


def mark_test_exposed(marker: Path, manifest_sha256: str) -> None:
    if marker.exists():
        raise RuntimeError("test exposure marker already exists")
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps({"manifest_sha256": manifest_sha256, "exposed": True}, sort_keys=True) + "\n")
    tmp.replace(marker)
