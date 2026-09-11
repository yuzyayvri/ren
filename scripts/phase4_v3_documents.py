"""Serialize the frozen GO-term document corpus for MedCPT article encoding."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.phase4_v3_common import (
    ROOT,
    V3,
    atomic_write_bytes,
    atomic_write_json,
    code_sha256,
    sha256_path,
)


def build_documents(db_path: Path = V3 / "ontology.sqlite") -> list[dict[str, str]]:
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT id,name,definition FROM terms WHERE obsolete=0 ORDER BY id"
    ).fetchall()
    syn_rows = db.execute(
        "SELECT term_id,value FROM synonyms ORDER BY term_id,synonym_id"
    ).fetchall()
    db.close()
    synonyms: dict[str, list[str]] = {}
    for term_id, value in syn_rows:
        synonyms.setdefault(term_id, []).append(value)
    return [
        {
            "id": term_id,
            "title": f"{term_id} {name}",
            "abstract": definition
            + (
                " Synonyms: " + "; ".join(synonyms[term_id])
                if term_id in synonyms
                else ""
            ),
        }
        for term_id, name, definition in rows
    ]


def main() -> None:
    db_path = V3 / "ontology.sqlite"
    docs = build_documents(db_path)
    if len(docs) != 38245 or [x["id"] for x in docs] != sorted(x["id"] for x in docs):
        raise RuntimeError("document order/count invariant failed")
    payload = (
        json.dumps(docs, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    documents_path = V3 / "documents.json"
    atomic_write_bytes(documents_path, payload)
    manifest = {
        "schema": 1,
        "serializer": "GO-ID name title; definition + ordered synonyms abstract",
        "article_input": "[[title, abstract]]",
        "rows": len(docs),
        "order": "GO ID ascending",
        "database_sha256": sha256_path(db_path),
        "documents_sha256": sha256_path(documents_path),
        "code_sha256": code_sha256(),
        "source": str(documents_path.relative_to(ROOT)),
    }
    atomic_write_json(V3 / "document_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
