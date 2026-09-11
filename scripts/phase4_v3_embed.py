"""Fresh, resumable MedCPT article embeddings with strict shard provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    V3,
    atomic_write_json,
    canonical_json_bytes,
    code_sha256,
    model_manifest,
    sha256_bytes,
    sha256_path,
)

SHARD_ROWS = 256
DIM = 768


def _protocol() -> tuple[dict, str]:
    path = V3 / "protocol.json"
    if not path.is_file():
        raise RuntimeError("v3 protocol is not frozen")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_prospective_no_evaluation":
        raise RuntimeError("v3 protocol must be frozen before embedding")
    return protocol, sha256_path(path)


def _load_documents() -> list[dict[str, str]]:
    path = V3 / "documents.json"
    docs = json.loads(path.read_text(encoding="utf-8"))
    if not docs or [row["id"] for row in docs] != sorted(row["id"] for row in docs):
        raise RuntimeError("documents are missing or not GO-ID ordered")
    return docs


def _row_hashes(docs: list[dict[str, str]]) -> list[str]:
    return [sha256_bytes(canonical_json_bytes(doc)) for doc in docs]


def _metadata(
    *,
    start: int,
    end: int,
    docs: list[dict[str, str]],
    row_hashes: list[str],
    protocol_sha: str,
    db_sha: str,
    documents_sha: str,
    code_sha: str,
    model: dict,
) -> dict[str, object]:
    return {
        "schema": 2,
        "start": start,
        "end": end,
        "rows": end - start,
        "indices": list(range(start, end)),
        "go_ids": [doc["id"] for doc in docs[start:end]],
        "row_hashes": row_hashes[start:end],
        "document_sha256": documents_sha,
        "database_sha256": db_sha,
        "source_sha256": json.loads((V3 / "source.json").read_text(encoding="utf-8"))[
            "sha256"
        ],
        "protocol_sha256": protocol_sha,
        "code_sha256": code_sha,
        "model": {
            "repository": model["repository"],
            "revision": model["revision"],
            "manifest_sha256": model["manifest_sha256"],
        },
        "input_contract": {
            "fields": ["title", "abstract"],
            "tokenizer_argument": "list of [title, abstract] pairs",
        },
        "max_length": 512,
        "pooling": "last_hidden_state[:,0,:]",
        "normalization": "row-wise L2",
        "dtype": "float32",
    }


def _validate_shard(path: Path, expected: dict[str, object]) -> None:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            indices = loaded["indices"]
            vectors = loaded["vectors"]
            row_hashes = loaded["row_hashes"]
            metadata_raw = loaded["metadata"].item()
        metadata = json.loads(str(metadata_raw))
    except Exception as exc:  # pragma: no cover - error text is the useful contract
        raise RuntimeError(f"cannot read embedding shard {path}: {exc}") from exc
    if metadata != expected:
        raise RuntimeError(f"embedding shard provenance mismatch: {path}")
    start, end = int(expected["start"]), int(expected["end"])
    if (
        indices.dtype != np.int64
        or indices.shape != (end - start,)
        or not np.array_equal(indices, np.arange(start, end))
    ):
        raise RuntimeError(f"embedding shard index coverage mismatch: {path}")
    expected_hashes = np.asarray(expected["row_hashes"])
    if (
        row_hashes.dtype.kind not in "SU"
        or row_hashes.shape != expected_hashes.shape
        or not np.array_equal(row_hashes, expected_hashes)
    ):
        raise RuntimeError(f"embedding shard row binding mismatch: {path}")
    if (
        vectors.dtype != np.float32
        or vectors.shape != (end - start, DIM)
        or not np.isfinite(vectors).all()
    ):
        raise RuntimeError(f"embedding shard vector shape/finiteness mismatch: {path}")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0) or float(np.max(np.abs(norms - 1))) > 2e-5:
        raise RuntimeError(f"embedding shard normalization mismatch: {path}")


def _save_shard(
    path: Path,
    indices: np.ndarray,
    vectors: np.ndarray,
    row_hashes: list[str],
    metadata: dict[str, object],
) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            indices=indices.astype(np.int64, copy=False),
            vectors=vectors.astype(np.float32, copy=False),
            row_hashes=np.asarray(row_hashes),
            metadata=np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_model():
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(ARTICLE_MODEL, local_files_only=True).eval()
    return tokenizer, model, torch


def embed() -> dict[str, object]:
    _protocol_data, protocol_sha = _protocol()
    docs = _load_documents()
    documents_path = V3 / "documents.json"
    db_path = V3 / "ontology.sqlite"
    documents_sha = sha256_path(documents_path)
    db_sha = sha256_path(db_path)
    source_sha = json.loads((V3 / "source.json").read_text(encoding="utf-8"))["sha256"]
    row_hashes = _row_hashes(docs)
    code_sha = code_sha256()
    model = model_manifest(ARTICLE_MODEL)
    shard_dir = V3 / "embedding_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, encoder, torch = _load_model()
    for start in range(0, len(docs), SHARD_ROWS):
        end = min(start + SHARD_ROWS, len(docs))
        path = shard_dir / f"{start:06d}.npz"
        meta = _metadata(
            start=start,
            end=end,
            docs=docs,
            row_hashes=row_hashes,
            protocol_sha=protocol_sha,
            db_sha=db_sha,
            documents_sha=documents_sha,
            code_sha=code_sha,
            model=model,
        )
        if path.exists():
            _validate_shard(path, meta)
            continue
        inputs = tokenizer(
            [[doc["title"], doc["abstract"]] for doc in docs[start:end]],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            vectors = encoder(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        _save_shard(path, np.arange(start, end), vectors, row_hashes[start:end], meta)
        _validate_shard(path, meta)
    shard_paths = sorted(shard_dir.glob("*.npz"))
    if len(shard_paths) != (len(docs) + SHARD_ROWS - 1) // SHARD_ROWS:
        raise RuntimeError("embedding shard count is incomplete")
    arrays = []
    for path in shard_paths:
        start = int(path.stem)
        end = min(start + SHARD_ROWS, len(docs))
        meta = _metadata(
            start=start,
            end=end,
            docs=docs,
            row_hashes=row_hashes,
            protocol_sha=protocol_sha,
            db_sha=db_sha,
            documents_sha=documents_sha,
            code_sha=code_sha,
            model=model,
        )
        _validate_shard(path, meta)
        with np.load(path, allow_pickle=False) as loaded:
            arrays.append(loaded["vectors"])
    matrix = np.concatenate(arrays, axis=0)
    if matrix.shape != (len(docs), DIM) or not np.isfinite(matrix).all():
        raise RuntimeError("canonical embedding shape/coverage invariant failed")
    canonical = V3 / "article_embeddings.npy"
    tmp = canonical.with_name(canonical.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, matrix.astype(np.float32, copy=False), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, canonical)
    manifest = {
        "schema": 3,
        "backend": "numpy_flat_cosine",
        "rows": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "dtype": "float32",
        "order": "GO ID ascending",
        "documents_sha256": documents_sha,
        "embeddings_sha256": sha256_path(canonical),
        "database_sha256": db_sha,
        "source_sha256": source_sha,
        "protocol_sha256": protocol_sha,
        "code_sha256": code_sha,
        "article_model": {
            "repository": model["repository"],
            "revision": model["revision"],
            "manifest_sha256": model["manifest_sha256"],
        },
        "input_contract": {
            "fields": ["title", "abstract"],
            "tokenizer_argument": "list of [title, abstract] pairs",
        },
        "max_length": 512,
        "pooling": "last_hidden_state[:,0,:]",
        "normalization": "row-wise L2",
        "shard_rows": SHARD_ROWS,
        "shards": len(shard_paths),
    }
    atomic_write_json(V3 / "embedding_manifest.json", manifest)
    return manifest


def main() -> None:
    print(json.dumps(embed(), sort_keys=True))


if __name__ == "__main__":
    main()
