"""Reconciliation boundary for the Phase 4 v3 code-identity gate (issue #2).

Root cause (established from repository evidence, not assumed): the v3
freeze records predate two separately authorized `query.py` changes
(v1 tail membership, v2 postings cache). The working tree exactly matches
the final authorized implementation, so the default load path raises on a
stale reference. Reverting is impossible (no pre-change copy exists) and
re-freezing the stale records is forbidden by the v2 authorization.

This boundary therefore mutates nothing sealed: it machine-checks the
intended state against the sealed records and exposes the estate's own
sanctioned bound-load path (the same binding the recovery code and its
tests use) for consumers such as Phase 5 Part 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V3 = ROOT / "artifacts" / "phase4_protocol_v3"
V1 = V3 / "recovery" / "latency_recovery_v1"
V2 = V3 / "recovery" / "latency_recovery_v2"
REC = V3 / "recovery" / "reconciliation_v1"

EXPECTED_FINAL_CODE = "cce08d213454780fb2967a7857e9435bc7bb915a660a29bb8611bf68eaed5fed"
QUERY_PY = "scripts/phase4_v3_query.py"


class ReconciliationError(RuntimeError):
    """The intended state does not hold. Fail closed; change nothing."""


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> Any:
    if not path.is_file():
        raise ReconciliationError(f"missing sealed record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_sidecar(path: Path) -> str:
    # Sidecars live next to their file: authorization.json -> authorization.sha256.
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ReconciliationError(f"missing sidecar: {sidecar}")
    parts = sidecar.read_text(encoding="utf-8").split()
    # The estate mixes stem-style (v1) and filename-style (v2) labels;
    # accept either. The digest itself is what binds the file.
    if len(parts) != 2 or parts[1] not in (path.stem, path.name):
        raise ReconciliationError(f"malformed sidecar: {sidecar}")
    digest = sha256_path(path)
    if digest != parts[0]:
        raise ReconciliationError(f"sidecar mismatch: {path}")
    return digest


def verify_intended_state() -> dict[str, Any]:
    from scripts.phase4_v3_common import code_sha256

    # 1. Consumed boundary records are intact (sidecars).
    consumed = [
        V1 / "authorization.json",
        V2 / "authorization.json",
        V2 / "implementation_manifest_final.json",
        V2 / "symbolic_postings_manifest.json",
    ]
    sidecars = {p.relative_to(ROOT).as_posix(): check_sidecar(p) for p in consumed}

    # 2. The final authorized implementation is exactly what is on disk.
    impl_final = _json(V2 / "implementation_manifest_final.json")
    if impl_final.get("post_recovery_code_sha256") != EXPECTED_FINAL_CODE:
        raise ReconciliationError("v2 recorded implementation is not the expected code")
    if impl_final.get("changed_code_files") != [QUERY_PY]:
        raise ReconciliationError("v2 recorded change scope is not query.py only")
    current_code = code_sha256()
    if current_code != EXPECTED_FINAL_CODE:
        raise ReconciliationError("working-tree code is not the authorized final code")

    # 3. The other nine files are byte-identical to the v2 pre-change record.
    pre_files = _json(V2 / "authorization.json")["pre_recovery_code_files"]
    nine_ok = True
    for name, digest in pre_files.items():
        if name == QUERY_PY:
            continue
        current = sha256_path(ROOT / name)
        if current != digest:
            nine_ok = False
    if not nine_ok:
        raise ReconciliationError("a non-query v3 code file drifted")

    # 4. Sealed v3 data and models still match the v3 manifest.
    index = _json(V3 / "index_manifest.json")
    from scripts.phase4_v3_common import OBO, model_manifest

    data_files = {
        "protocol.json": "protocol_sha256",
        "ontology.sqlite": "database_sha256",
        "documents.json": "documents_sha256",
        "article_embeddings.npy": "embeddings_sha256",
    }
    for filename, key in data_files.items():
        if sha256_path(V3 / filename) != index[key]:
            raise ReconciliationError(f"sealed v3 data mismatch: {filename}")
    if sha256_path(OBO) != index["source_sha256"]:
        raise ReconciliationError("sealed GO source mismatch")
    from scripts.phase4_v3_common import ARTICLE_MODEL, QUERY_MODEL

    for role, model_root in (("query", QUERY_MODEL), ("article", ARTICLE_MODEL)):
        if (
            model_manifest(model_root)["manifest_sha256"]
            != index["models"][role]["manifest_sha256"]
        ):
            raise ReconciliationError(f"sealed {role} model mismatch")

    return {
        "final_code_sha256": current_code,
        "nine_unchanged": nine_ok,
        "data_match": True,
        "boundary_sidecars": sidecars,
    }


def load_bound_query() -> Any:
    from scripts import phase4_v3_query as query_engine
    from scripts.phase4_latency_recovery_final import (
        _enable_recovery_snapshot_validation,
    )

    _enable_recovery_snapshot_validation()
    query_engine._load_snapshot()
    return query_engine


def reconcile() -> dict[str, Any]:
    authorization = REC / "authorization.json"
    if not authorization.is_file():
        raise ReconciliationError("reconciliation is not authorized; write it first")
    check_sidecar(authorization)
    proof = verify_intended_state()
    manifest = {
        "schema": 1,
        "boundary_id": "phase4_v3_snapshot_reconciliation_v1",
        "proof": proof,
        "mutated_sealed_files": [],
        "mutated_sealed_code": [],
        "status": "reconciled_no_mutation",
    }
    (REC / "reconciliation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    blob = (REC / "reconciliation_manifest.json").read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    (REC / "reconciliation_manifest.sha256").write_text(
        f"{digest}  reconciliation_manifest.json\n", encoding="utf-8"
    )
    ledger = REC / "lifecycle.jsonl"
    if not ledger.is_file():
        ledger.write_text(
            json.dumps(
                {
                    "event": "reconciled",
                    "manifest_sha256": digest,
                    "note": "Intended state proven; no sealed file or code changed.",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    manifest["manifest_sha256"] = digest
    return manifest


def verify() -> dict[str, Any]:
    manifest_path = REC / "reconciliation_manifest.json"
    if not manifest_path.is_file():
        raise ReconciliationError("not reconciled; run reconcile first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_sidecar(manifest_path)
    check_sidecar(REC / "authorization.json")
    proof = verify_intended_state()
    if proof["final_code_sha256"] != manifest["proof"]["final_code_sha256"]:
        raise ReconciliationError("intended state changed since reconciliation")
    if not proof["nine_unchanged"] or not proof["data_match"]:
        raise ReconciliationError("intended state changed since reconciliation")
    ledger = REC / "lifecycle.jsonl"
    if not ledger.is_file() or "reconciled" not in ledger.read_text(encoding="utf-8"):
        raise ReconciliationError("lifecycle genesis entry is missing")
    return {"status": "verified", "final_code_sha256": proof["final_code_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reconcile")
    sub.add_parser("verify")
    args = parser.parse_args()
    result = reconcile() if args.command == "reconcile" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
