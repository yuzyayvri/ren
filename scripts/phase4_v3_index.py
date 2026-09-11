"""Publish and verify the immutable v3 NumPy flat-cosine index."""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    QUERY_MODEL,
    V3,
    atomic_write_json,
    code_sha256,
    model_manifest,
    runtime_record,
    sha256_path,
)


def publish() -> dict[str, object]:
    protocol_path = V3 / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_prospective_no_evaluation":
        raise RuntimeError("v3 protocol must be frozen before index publication")
    documents = json.loads((V3 / "documents.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(V3 / "ontology.sqlite")
    db_ids = [
        row[0]
        for row in db.execute("SELECT id FROM terms WHERE obsolete=0 ORDER BY id")
    ]
    db.close()
    doc_ids = [row["id"] for row in documents]
    if len(documents) != 38245 or doc_ids != db_ids or doc_ids != sorted(doc_ids):
        raise RuntimeError("index document/database identity order mismatch")
    array_path = V3 / "article_embeddings.npy"
    if not array_path.is_file():
        raise RuntimeError("canonical embeddings are missing")
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if (
        array.shape != (len(documents), 768)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise RuntimeError("canonical embedding invariant failed")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0) or float(np.max(np.abs(norms - 1))) > 2e-5:
        raise RuntimeError("canonical embedding normalization invariant failed")
    source = json.loads((V3 / "source.json").read_text(encoding="utf-8"))
    documents_sha = sha256_path(V3 / "documents.json")
    db_sha = sha256_path(V3 / "ontology.sqlite")
    protocol_sha = sha256_path(protocol_path)
    code_sha = code_sha256()
    article = model_manifest(ARTICLE_MODEL)
    query = model_manifest(QUERY_MODEL)
    manifest = {
        "schema": 3,
        "backend": "numpy_flat_cosine",
        "rows": int(array.shape[0]),
        "dim": int(array.shape[1]),
        "dtype": "float32",
        "order": "GO ID ascending",
        "tie_break": "GO ID ascending",
        "documents_sha256": documents_sha,
        "embeddings_sha256": sha256_path(array_path),
        "database_sha256": db_sha,
        "source_sha256": source["sha256"],
        "protocol_sha256": protocol_sha,
        "code_sha256": code_sha,
        "models": {
            "query": {
                "repository": query["repository"],
                "revision": query["revision"],
                "manifest_sha256": query["manifest_sha256"],
            },
            "article": {
                "repository": article["repository"],
                "revision": article["revision"],
                "manifest_sha256": article["manifest_sha256"],
            },
        },
        "runtime": runtime_record(),
    }
    atomic_write_json(V3 / "index_manifest.json", manifest)
    return manifest


def main() -> None:
    print(json.dumps(publish(), sort_keys=True))


if __name__ == "__main__":
    main()
