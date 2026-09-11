"""Shared immutable/provenance helpers for the Phase 4 v3 boundary.

This module is deliberately small and dependency-free.  The v3 stages exchange
only files and hashes; no stage keeps mutable state owned by another stage.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts" / "phase4_protocol_v3"
OBO = ROOT / "data" / "ontology" / "go-basic.obo"
OBO_SHA256 = "c72fc198a86983d55e43aac585d1ffdbeb6e3601475b3f18b6045acdc0a0734c"
QUERY_MODEL = ROOT / "models" / "embed" / "medcpt" / "query"
ARTICLE_MODEL = ROOT / "models" / "embed" / "medcpt" / "article"

# The digest intentionally covers only source code, never generated artifacts.
# Keeping this list explicit makes the transitive code binding auditable.
CODE_FILES = (
    "scripts/phase4_v3_common.py",
    "scripts/phase4_v3_audit.py",
    "scripts/phase4_v3_ingest.py",
    "scripts/phase4_v3_documents.py",
    "scripts/phase4_v3_embed.py",
    "scripts/phase4_v3_index.py",
    "scripts/phase4_v3_query.py",
    "scripts/phase4_v3_benchmark.py",
    "scripts/phase4_v3_evaluate.py",
    "scripts/phase4_v3_verify.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def compact_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    atomic_write_bytes(
        path, compact_json_bytes(value) if compact else canonical_json_bytes(value)
    )


def code_file_hashes() -> dict[str, str]:
    missing = [name for name in CODE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"v3 code files missing: {missing}")
    return {name: sha256_path(ROOT / name) for name in CODE_FILES}


def code_sha256() -> str:
    payload = "".join(
        f"{name}\0{digest}\n" for name, digest in code_file_hashes().items()
    ).encode()
    return sha256_bytes(payload)


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def runtime_record() -> dict[str, Any]:
    lock = ROOT / "uv.lock"
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(
            (
                "numpy",
                "torch",
                "transformers",
                "tokenizers",
                "goatools",
                "pytest",
                "ruff",
            )
        ),
        "uv_lock_sha256": sha256_path(lock) if lock.is_file() else None,
    }


def model_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / ".acquisition-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing acquisition manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = []
    for row in manifest.get("files", []):
        p = path / row["filename"]
        if not p.is_file():
            raise RuntimeError(f"missing model file: {p}")
        actual = {
            "filename": row["filename"],
            "bytes": p.stat().st_size,
            "sha256": sha256_path(p),
        }
        if actual["bytes"] != row.get("bytes") or actual["sha256"] != row.get("sha256"):
            raise RuntimeError(f"model manifest mismatch: {p}")
        checked.append(
            {
                **actual,
                "expected_bytes": row.get("bytes"),
                "expected_sha256": row.get("sha256"),
            }
        )
    return {
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "files": checked,
        "manifest_sha256": sha256_path(manifest_path),
    }


def require_same_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_path(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


@contextmanager
def network_block() -> Any:
    """Fail closed if a stage attempts a network connection.

    Hugging Face offline flags remain enabled separately.  This guard is used
    during evaluation so the report contains executable evidence that network
    access was blocked in the request path.
    """
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    class BlockedSocket(original_socket):
        def connect(self, address):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"network blocked: {address!r}")

        def connect_ex(self, address):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"network blocked: {address!r}")

    def blocked_create_connection(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"network blocked: {args[0] if args else None!r}")

    socket.socket = BlockedSocket  # type: ignore[assignment]
    socket.create_connection = blocked_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
