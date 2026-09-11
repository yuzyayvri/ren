"""Freeze and verify the Phase 5 Part 1 synthesis protocol.

freeze: records model hashes, proves every fixture GO identifier resolves
  in the sealed Phase 4 v3 corpus (read-only), and writes an immutable
  manifest binding prompt, schemas, decoding, benchmark, scoring, winner
  rule, lifecycle, and models. Appends the genesis lifecycle entry once.
verify: rechecks every tracked-file hash, benchmark split isolation, model
  file sizes, and lifecycle genesis without running any LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTO = ROOT / "protocols" / "phase5_v1"

TRACKED_FILES = (
    "prompt.md",
    "decoding.json",
    "scoring.json",
    "winner_rule.json",
    "lifecycle.json",
    "benchmark_dev.json",
    "benchmark_final.json",
)

# Phase 4 v3 access is read-only and numpy-free: the sealed SQLite corpus and
# documents are verified by hash against the v3 index manifest, then fixture
# GO identifiers are checked directly against those sealed files. The v3
# python query layer is intentionally not imported: its code-identity gate
# currently fails on this machine (pre-existing, see upstream_notes), while
# every v3 data hash still matches.

OPENBIOLLM = ROOT / "models" / "llm" / "openbiollm-llama3-8b.Q5_K_M.gguf"
MEDGEMMA = ROOT / "models" / "llm" / "medgemma-1.5-4b-it-Q5_K_M.gguf"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_hashes() -> dict[str, str]:
    return {name: sha256_path(PROTO / name) for name in TRACKED_FILES}


def model_records(*, rehash_openbiollm: bool) -> list[dict[str, object]]:
    manifest_path = ROOT / "models" / "llm" / ".acquisition-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    med = next(
        f for f in manifest["files"] if f["filename"] == MEDGEMMA.name
    )
    if not MEDGEMMA.is_file() or MEDGEMMA.stat().st_size != med["bytes"]:
        raise RuntimeError("medgemma file does not match acquisition manifest")
    records = [
        {
            "name": "medgemma-1.5-4b-it",
            "filename": MEDGEMMA.name,
            "repository": manifest["repository"],
            "revision": manifest["revision"],
            "bytes": med["bytes"],
            "sha256": med["sha256"],
        }
    ]
    if not OPENBIOLLM.is_file():
        raise RuntimeError("openbiollm file is missing")
    cache = PROTO / "models.json"
    cached_digest = None
    if cache.is_file() and not rehash_openbiollm:
        cached = {
            r["filename"]: r
            for r in json.loads(cache.read_text(encoding="utf-8"))["models"]
        }.get(OPENBIOLLM.name)
        if cached and cached.get("bytes") == OPENBIOLLM.stat().st_size:
            cached_digest = cached.get("sha256")
    digest = cached_digest or sha256_path(OPENBIOLLM)
    records.append(
        {
            "name": "openbiollm-llama3-8b",
            "filename": OPENBIOLLM.name,
            "repository": "aaditya/OpenBioLLM-Llama3-8B-GGUF",
            "revision": None,
            "bytes": OPENBIOLLM.stat().st_size,
            "sha256": digest,
            "note": "acquisition manifest covers medgemma only; hashed directly",
        }
    )
    return records


def v3_sealed_files() -> dict[str, Path]:
    from scripts.phase4_v3_common import V3 as _V3

    return {
        "index_manifest": _V3 / "index_manifest.json",
        "database": _V3 / "ontology.sqlite",
        "documents": _V3 / "documents.json",
    }


def v3_code_status() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.phase4_v3_common import code_sha256

    manifest = json.loads(v3_sealed_files()["index_manifest"].read_text(encoding="utf-8"))
    current = code_sha256()
    return {
        "frozen_code_sha256": manifest["code_sha256"],
        "working_tree_code_sha256": current,
        "match": current == manifest["code_sha256"],
    }


def check_go_membership() -> dict[str, object]:
    import sqlite3

    paths = v3_sealed_files()
    manifest = json.loads(paths["index_manifest"].read_text(encoding="utf-8"))
    for key in ("database", "documents"):
        if sha256_path(paths[key]) != manifest[f"{key}_sha256"]:
            raise RuntimeError(f"sealed v3 {key} hash mismatch")
    documents = {
        entry["id"]: entry
        for entry in json.loads(paths["documents"].read_text(encoding="utf-8"))
    }
    db = sqlite3.connect(f"file:{paths['database']}?mode=ro", uri=True)
    counts: dict[str, int] = {}
    try:
        for split in ("benchmark_dev.json", "benchmark_final.json"):
            cases = json.loads((PROTO / split).read_text(encoding="utf-8"))["cases"]
            for case in cases:
                for entry in case["context"]:
                    row = db.execute(
                        "SELECT name, definition FROM terms WHERE id=? AND obsolete=0",
                        (entry["go_id"],),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError(
                            f"{case['case_id']}: {entry['go_id']} is not a current v3 term"
                        )
                    sealed_definition = row[1]
                    if row[0] != entry["name"] or entry["definition"] not in sealed_definition:
                        raise RuntimeError(
                            f"{case['case_id']}: {entry['go_id']} name/definition not found in sealed corpus"
                        )
                    doc = documents.get(entry["go_id"])
                    if doc is None or not entry["definition"] in doc["abstract"]:
                        raise RuntimeError(
                            f"{case['case_id']}: {entry['go_id']} text not found in sealed documents"
                        )
            counts[split] = len(cases)
    finally:
        db.close()
    return {"cases": counts, "corpus_rows": manifest["rows"]}


def check_split_isolation() -> None:
    dev = json.loads((PROTO / "benchmark_dev.json").read_text(encoding="utf-8"))["cases"]
    final = json.loads((PROTO / "benchmark_final.json").read_text(encoding="utf-8"))["cases"]
    dev_ids = {c["case_id"] for c in dev}
    final_ids = {c["case_id"] for c in final}
    if dev_ids & final_ids:
        raise RuntimeError(f"benchmark splits share case ids: {dev_ids & final_ids}")
    if any(not c["case_id"].startswith("dev-") for c in dev):
        raise RuntimeError("development case without dev- prefix")
    if any(not c["case_id"].startswith("final-") for c in final):
        raise RuntimeError("final case without final- prefix")


def freeze(*, rehash_openbiollm: bool = False) -> dict[str, object]:
    check_split_isolation()
    go_counts = check_go_membership()
    code_status = v3_code_status()
    models = model_records(rehash_openbiollm=rehash_openbiollm)
    (PROTO / "models.json").write_text(
        json.dumps({"models": models}, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "phase5-freeze-v1",
        "files": file_hashes(),
        "models": {m["filename"]: {"bytes": m["bytes"], "sha256": m["sha256"]} for m in models},  # type: ignore[index]
        "go_membership": go_counts,
        "upstream_notes": {
            "phase4_v3_code_gate": code_status,
            "phase4_v3_code_gate_note": (
                "Pre-existing: the v3 python code-identity gate does not match "
                "its frozen manifest on this machine, so the v3 query layer "
                "refuses to load here. Every v3 data hash (protocol, database, "
                "documents, embeddings, source, models) still matches, and "
                "Phase 5 fixtures are proven directly against the sealed "
                "database and documents. No v3 file was modified by Part 1."
            ),
        },
        "status": "frozen_part1_no_candidate_output",
    }
    (PROTO / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = PROTO / "lifecycle.jsonl"
    if not ledger.is_file():
        entry = {
            "event": "part1_frozen",
            "manifest_sha256": sha256_path(PROTO / "freeze_manifest.json"),
            "note": "Final set sealed; no candidate output exposed.",
        }
        ledger.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify(*, rehash_models: bool = False) -> dict[str, object]:
    manifest_path = PROTO / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("protocol is not frozen; run freeze first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_files = file_hashes()
    if actual_files != manifest["files"]:
        raise RuntimeError("tracked protocol files differ from frozen manifest")
    check_split_isolation()
    go_counts = check_go_membership()
    if go_counts != manifest["go_membership"]:
        raise RuntimeError("GO membership proof differs from frozen manifest")
    if v3_code_status() != manifest["upstream_notes"]["phase4_v3_code_gate"]:
        raise RuntimeError("upstream v3 code-gate state changed since freeze")
    for filename, record in manifest["models"].items():
        path = ROOT / "models" / "llm" / filename
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"model file mismatch: {filename}")
        if rehash_models and sha256_path(path) != record["sha256"]:
            raise RuntimeError(f"model hash mismatch: {filename}")
    ledger = PROTO / "lifecycle.jsonl"
    if not ledger.is_file() or "part1_frozen" not in ledger.read_text(encoding="utf-8"):
        raise RuntimeError("lifecycle genesis entry is missing")
    return {"status": "verified", "go_membership": go_counts}


V2DIR = ROOT / "protocols" / "phase5_v2"

V2_FILES = (
    "prompt.md",
    "decoding.json",
    "response_schema.json",
    "smoke_packet.json",
)

# Carried forward by hash reference from the v1 freeze; never copied.
V2_REFERENCES = (
    "benchmark_dev.json",
    "benchmark_final.json",
    "scoring.json",
    "winner_rule.json",
    "lifecycle.json",
    "models.json",
)


def sync_v2_decoding() -> None:
    """Rebuild decoding.json's embedded schema from response_schema.json.

    The embedded copy is what is actually sent to the server, so it is
    generated mechanically at freeze time. verify_v2 rechecks the sync,
    making divergence machine-detectable instead of silent.
    """
    schema = json.loads((V2DIR / "response_schema.json").read_text(encoding="utf-8"))
    decoding = json.loads((V2DIR / "decoding.json").read_text(encoding="utf-8"))
    decoding["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema}}
    (V2DIR / "decoding.json").write_text(
        json.dumps(decoding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def check_v2_decoding_sync() -> None:
    schema = json.loads((V2DIR / "response_schema.json").read_text(encoding="utf-8"))
    decoding = json.loads((V2DIR / "decoding.json").read_text(encoding="utf-8"))
    if decoding.get("response_format") != {"type": "json_schema", "json_schema": {"schema": schema}}:
        raise RuntimeError("decoding.json embedded schema drifted from response_schema.json")


def freeze_v2() -> dict[str, object]:
    from scripts.phase5_compare import append_lifecycle

    verify()
    sync_v2_decoding()
    v1_manifest_sha = sha256_path(PROTO / "freeze_manifest.json")
    manifest = {
        "schema": "phase5-freeze-v2",
        "files": {name: sha256_path(V2DIR / name) for name in V2_FILES},
        "references": {name: sha256_path(PROTO / name) for name in V2_REFERENCES},
        "supersedes": {"phase5_freeze_v1_manifest_sha256": v1_manifest_sha},
        "status": "frozen_v2_no_candidate_output",
    }
    new_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = V2DIR / "freeze_manifest.json"
    if manifest_path.is_file() and manifest_path.read_bytes() == new_bytes:
        manifest["manifest_sha256"] = sha256_path(manifest_path)
        return manifest
    manifest_path.write_bytes(new_bytes)
    manifest_sha = sha256_path(manifest_path)
    append_lifecycle("part1_superseded", {
        "superseded_manifest_sha256": v1_manifest_sha,
        "superseding_manifest_sha256": manifest_sha,
        "note": "v1 remains valid history; v2 governs forward work.",
    })
    append_lifecycle("phase5_v2_frozen", {
        "manifest_sha256": manifest_sha,
        "note": "Contract revised; no candidate output exposed.",
    })
    manifest["manifest_sha256"] = manifest_sha
    return manifest


def verify_v2() -> dict[str, object]:
    manifest_path = V2DIR / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("v2 protocol is not frozen; run freeze-v2 first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {name: sha256_path(V2DIR / name) for name in V2_FILES}
    if actual != manifest["files"]:
        raise RuntimeError("v2 protocol files differ from frozen manifest")
    check_v2_decoding_sync()
    verify()
    references = {name: sha256_path(PROTO / name) for name in V2_REFERENCES}
    if references != manifest["references"]:
        raise RuntimeError("v1 references changed since v2 freeze")
    if manifest["supersedes"]["phase5_freeze_v1_manifest_sha256"] != sha256_path(
        PROTO / "freeze_manifest.json"
    ):
        raise RuntimeError("superseded v1 manifest changed")
    ledger = (PROTO / "lifecycle.jsonl").read_text(encoding="utf-8")
    if "part1_superseded" not in ledger or "phase5_v2_frozen" not in ledger:
        raise RuntimeError("v2 lifecycle events are missing")
    return {"status": "verified"}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_cmd = sub.add_parser("freeze")
    freeze_cmd.add_argument("--rehash-openbiollm", action="store_true")
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--rehash-models", action="store_true")
    sub.add_parser("freeze-v2")
    sub.add_parser("verify-v2")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(rehash_openbiollm=args.rehash_openbiollm)
    elif args.command == "verify":
        result = verify(rehash_models=args.rehash_models)
    elif args.command == "freeze-v2":
        result = freeze_v2()
    else:
        result = verify_v2()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
