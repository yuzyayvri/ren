"""Read-only verifier for the sealed v3 snapshot and final latency recovery."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.phase4_v3_common import (
    CODE_FILES,
    ROOT,
    V3,
    code_file_hashes,
    code_sha256,
    sha256_path,
)

RECOVERY = V3 / "recovery" / "latency_recovery_v2"
FIRST_RECOVERY = V3 / "recovery" / "latency_recovery_v1"


def _json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _digest_json(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _inventory(root: Path, *, exclude_recovery: bool = False) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel_root = path.relative_to(root)
        if exclude_recovery and "recovery" in rel_root.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        result[rel] = {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
    return result


def _check_sidecar(path: Path, target: Path) -> str:
    fields = path.read_text(encoding="ascii").split()
    if len(fields) != 2 or fields[1] != target.name:
        raise RuntimeError(f"invalid sidecar target: {path}")
    digest = sha256_path(target)
    if fields[0] != digest:
        raise RuntimeError(f"sidecar digest mismatch: {path}")
    return digest


def _check_rankings() -> dict[str, Any]:
    report = _json(V3 / "evaluation_report.json")
    evidence = report["ranking_evidence"]
    path = V3 / "rankings.jsonl"
    if evidence["sha256"] != sha256_path(path) or evidence["records"] != 87:
        raise RuntimeError("sealed v3 ranking evidence binding mismatch")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ranking = row["ranking"]
            if len(ranking) != 38245 or len(set(ranking)) != 38245:
                raise RuntimeError(f"stored ranking is not complete: {row['case_id']} {row['mode']}")
            records.append((row["case_id"], row["mode"], _digest_json(ranking)))
    if len(records) != 87 or len({(case, mode) for case, mode, _ in records}) != 87:
        raise RuntimeError("sealed ranking coverage mismatch")
    return {"records": len(records), "sha256": sha256_path(path)}


def verify() -> dict[str, Any]:
    auth = _json(RECOVERY / "authorization.json")
    protocol = _json(RECOVERY / "latency_protocol.json")
    implementation = _json(RECOVERY / "implementation_manifest_final.json")
    postings = _json(RECOVERY / "symbolic_postings_manifest.json")
    equivalence = _json(RECOVERY / "ranking_equivalence_final.json")
    raw = _json(RECOVERY / "latency_samples.json")
    report = _json(RECOVERY / "latency_report.json")
    if auth["status"] != "sealed_before_implementation_change":
        raise RuntimeError("final authorization is not pre-change sealed")
    if protocol["status"] != "frozen_latency_only_no_quality_evaluation":
        raise RuntimeError("final protocol is not frozen")
    if any(value.get("quality_evaluation", False) is not False for value in (protocol, implementation, equivalence, raw, report)):
        raise RuntimeError("quality-evaluation disclosure is not fail-closed")
    _check_sidecar(RECOVERY / "authorization.sha256", RECOVERY / "authorization.json")
    _check_sidecar(RECOVERY / "latency_protocol.sha256", RECOVERY / "latency_protocol.json")
    _check_sidecar(RECOVERY / "implementation_manifest_final.sha256", RECOVERY / "implementation_manifest_final.json")
    _check_sidecar(RECOVERY / "symbolic_postings_manifest.sha256", RECOVERY / "symbolic_postings_manifest.json")
    _check_sidecar(RECOVERY / "ranking_equivalence_final.sha256", RECOVERY / "ranking_equivalence_final.json")
    raw_sha = _check_sidecar(RECOVERY / "latency_samples.sha256", RECOVERY / "latency_samples.json")
    report_sha = _check_sidecar(RECOVERY / "latency_report.sha256", RECOVERY / "latency_report.json")
    if _inventory(V3, exclude_recovery=True) != auth["v3_artifact_inventory"]:
        raise RuntimeError("sealed v3 artifact inventory changed")
    if _inventory(FIRST_RECOVERY) != auth["prior_boundary_inventory"]:
        raise RuntimeError("first latency-recovery boundary changed")
    if code_sha256() != implementation["post_recovery_code_sha256"]:
        raise RuntimeError("final production code binding mismatch")
    before = auth["pre_recovery_code_files"]
    after = code_file_hashes()
    changed = [name for name in CODE_FILES if before.get(name) != after.get(name)]
    if changed != ["scripts/phase4_v3_query.py"]:
        raise RuntimeError(f"unexpected production code changes: {changed}")
    if not implementation.get("v3_artifacts_unchanged") or not implementation.get("first_recovery_unchanged"):
        raise RuntimeError("implementation manifest does not preserve prior evidence")
    if equivalence["record_count"] != 87 or not equivalence["all_87_byte_identical"] or not equivalence["all_87_duplicate_free"]:
        raise RuntimeError("final ranking regression equivalence failed")
    if postings["implementation_code_sha256"] != code_sha256() or postings["implementation_final_sha256"] != sha256_path(RECOVERY / "implementation_manifest_final.json"):
        raise RuntimeError("postings manifest implementation binding mismatch")
    ranking = _check_rankings()
    if raw["execution_id"] != report["execution_id"] or raw["boundary_id"] != report["boundary_id"]:
        raise RuntimeError("raw/report execution binding mismatch")
    if report["status"] != "accepted" or report["all_gates_pass"] is not True:
        raise RuntimeError("final latency report is not accepted")
    for mode in ("symbolic", "vector", "hybrid"):
        values = report["latency"][mode]["samples_ms"]
        if len(values) != 145 or report["latency"][mode]["nearest_rank"] != math.ceil(0.95 * len(values)):
            raise RuntimeError(f"latency sample contract mismatch: {mode}")
        p95 = sorted(values)[report["latency"][mode]["nearest_rank"] - 1]
        if p95 != report["latency"][mode]["p95_ms"] or p95 > 200.0:
            raise RuntimeError(f"latency gate mismatch: {mode}")
    lifecycle = [json.loads(line) for line in (RECOVERY / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if [event.get("event") for event in lifecycle] != ["frozen", "exposure_started", "latency_complete"]:
        raise RuntimeError("final lifecycle is not the sealed one-writer sequence")
    if lifecycle[-1].get("report_sha256") != report_sha or lifecycle[-1].get("raw_samples_sha256") != raw_sha:
        raise RuntimeError("final lifecycle/report binding mismatch")
    return {
        "status": "verified",
        "boundary_id": auth["boundary_id"],
        "report_sha256": report_sha,
        "raw_samples_sha256": raw_sha,
        "ranking_evidence": ranking,
        "code_sha256": code_sha256(),
        "p95_ms": {mode: report["latency"][mode]["p95_ms"] for mode in ("symbolic", "vector", "hybrid")},
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
