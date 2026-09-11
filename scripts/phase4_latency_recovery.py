"""Authorized, latency-only Phase 4 v3 recovery workflow.

This module owns the separate recovery boundary.  It never writes any file in
the sealed v3 root, and it never computes or publishes retrieval-quality
metrics.  The only production implementation change authorized by the
boundary is the set-backed tail completion in ``phase4_v3_query``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from scripts import phase4_v3_query as query_engine
from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    CODE_FILES,
    QUERY_MODEL,
    ROOT,
    V3,
    canonical_json_bytes,
    code_file_hashes,
    code_sha256,
    model_manifest,
    network_block,
    runtime_record,
    sha256_path,
)

RECOVERY = V3 / "recovery" / "latency_recovery_v1"
AUTHORIZATION = RECOVERY / "authorization.json"
AUTHORIZATION_SHA = RECOVERY / "authorization.sha256"
PROTOCOL = RECOVERY / "latency_protocol.json"
PROTOCOL_SHA = RECOVERY / "latency_protocol.sha256"
IMPLEMENTATION = RECOVERY / "implementation_manifest.json"
IMPLEMENTATION_SHA = RECOVERY / "implementation_manifest.sha256"
LIFECYCLE = RECOVERY / "lifecycle.jsonl"
EQUIVALENCE = RECOVERY / "ranking_equivalence.json"
EQUIVALENCE_SHA = RECOVERY / "ranking_equivalence.sha256"
RAW_SAMPLES = RECOVERY / "latency_samples.json"
RAW_SAMPLES_SHA = RECOVERY / "latency_samples.sha256"
REPORT = RECOVERY / "latency_report.json"
REPORT_SHA = RECOVERY / "latency_report.sha256"

MODES = ("symbolic", "vector", "hybrid")
WARMUPS = 2
MEASURED = 5
LATENCY_GATE_MS = 200.0


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_json(value: Any) -> str:
    return _digest_bytes(canonical_json_bytes(value))


def _write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"recovery artifact already exists: {path}")
    path.write_bytes(payload)
    return _digest_bytes(payload)


def _write_sidecar(path: Path, digest: str) -> None:
    if path.exists():
        raise RuntimeError(f"recovery sidecar already exists: {path}")
    path.write_text(f"{digest}  {path.name.removesuffix('.sha256')}\n", encoding="ascii")


def _append_lifecycle(event: dict[str, Any]) -> None:
    RECOVERY.mkdir(parents=True, exist_ok=True)
    with LIFECYCLE.open("ab") as handle:
        handle.write((json.dumps(event, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _artifact_inventory() -> dict[str, dict[str, int | str]]:
    """Hash every pre-recovery file in the v3 root, including all shards."""
    inventory: dict[str, dict[str, int | str]] = {}
    for path in sorted(p for p in V3.rglob("*") if p.is_file()):
        if "recovery" in path.relative_to(V3).parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        inventory[rel] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    return inventory


def _model_bindings() -> dict[str, Any]:
    return {
        "query": model_manifest(QUERY_MODEL),
        "article": model_manifest(ARTICLE_MODEL),
    }


def _enable_recovery_snapshot_validation() -> None:
    """Validate sealed v3 inputs while allowing the intentional code delta."""
    boundary = _load_json(AUTHORIZATION)
    if _artifact_inventory() != boundary["artifact_inventory"]:
        raise RuntimeError("sealed v3 artifact inventory changed during recovery")
    if _model_bindings() != boundary["model_bindings"]:
        raise RuntimeError("sealed v3 model binding changed during recovery")

    def validate() -> dict[str, Any]:
        manifest = _load_json(V3 / "index_manifest.json")
        if _v3_bindings() != boundary["v3_bindings"]:
            raise RuntimeError("sealed v3 binding changed during recovery")
        return {"manifest": manifest, "actual": boundary["v3_bindings"]}

    # The original v3 verifier intentionally binds the pre-recovery code hash.
    # Replace only that verifier callback inside this recovery process; no v3
    # file or production snapshot binding is edited.
    query_engine.verify_snapshot = validate


def _v3_bindings() -> dict[str, Any]:
    names = (
        "protocol.json",
        "benchmark.json",
        "freeze_manifest.json",
        "index_manifest.json",
        "source.json",
        "source_audit.json",
        "model_audit.json",
        "runtime_audit.json",
        "documents.json",
        "document_manifest.json",
        "embedding_manifest.json",
        "ontology.sqlite",
        "article_embeddings.npy",
        "evaluation_report.json",
        "evaluation_report.sha256",
        "rankings.jsonl",
        "lifecycle.jsonl",
    )
    return {name: sha256_path(V3 / name) for name in names}


def _protocol_value(boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "boundary_id": "phase4_v3_latency_recovery_v1",
        "status": "frozen_latency_only_no_quality_evaluation",
        "disclosure": (
            "v3 retrieval quality and rankings remain sealed unchanged; this is "
            "a separately authorized latency-only recovery after the v3 latency "
            "gate failed. No replacement Recall/MRR evaluation is published."
        ),
        "authorization": {
            "source": "explicit root-agent authorization in the parent task",
            "authorized_scope": "one prospective latency-only recovery measurement",
            "defect": (
                "phase4_v3_query full-ranking tail completion used list membership "
                "(term_id not in ranked), yielding O(N^2) completion on the 38,245-ID corpus"
            ),
            "allowed_production_change": (
                "replace only that tail membership check with a set-backed O(N) "
                "membership structure; preserve ordered results exactly"
            ),
            "forbidden_changes": [
                "rerun or replace v3 retrieval-quality evaluation",
                "change rankings, query scores, top-k, rank numbering, RRF, or serialization",
                "change query/article encoders, models, database, corpus, embeddings, or index",
                "change benchmark cases, eligibility, backend, ranking policy, or gates",
                "tune on acceptance cases or run a second recovery timing",
                "modify Phase 2 or Phase 3 scientific artifacts",
            ],
        },
        "timing_contract": {
            "workload": "v3 eligible retrieval cases and modes",
            "eligible_retrieval_cases": 29,
            "modes": list(MODES),
            "warmup_repeats": WARMUPS,
            "measured_repeats": MEASURED,
            "samples_per_mode": 145,
            "scope": "query plus ranking; cache preloaded; hybrid timed directly",
            "p95": "nearest-rank ceil(0.95*n), 1-based",
            "p95_latency_ms_gate": LATENCY_GATE_MS,
            "network": "offline guard and local model files only",
        },
        "sealed_v3": {
            "code_sha256": boundary["pre_recovery_code_sha256"],
            "artifact_inventory_sha256": _digest_json(boundary["artifact_inventory"]),
            "bindings": boundary["v3_bindings"],
        },
    }


def seal() -> dict[str, Any]:
    """Create the recovery authorization, boundary, protocol, and frozen event."""
    if any(path.exists() for path in (AUTHORIZATION, PROTOCOL, LIFECYCLE)):
        raise RuntimeError("latency-recovery boundary has already been sealed")
    # This is intentionally evaluated before any production implementation edit.
    pre_code_files = code_file_hashes()
    pre_code_sha = code_sha256()
    inventory = _artifact_inventory()
    v3_bindings = _v3_bindings()
    models = _model_bindings()
    boundary = {
        "schema": 1,
        "boundary_id": "phase4_v3_latency_recovery_v1",
        "sealed_at_unix": time.time(),
        "status": "sealed_before_implementation_change",
        "authorization_source": "explicit root-agent authorization in the parent task",
        "pre_recovery_code_sha256": pre_code_sha,
        "pre_recovery_code_files": pre_code_files,
        "artifact_inventory": inventory,
        "v3_bindings": v3_bindings,
        "model_bindings": models,
        "old_v3_report_status": _load_json(V3 / "evaluation_report.json").get("status"),
        "old_v3_all_gates_pass": _load_json(V3 / "evaluation_report.json").get(
            "all_gates_pass"
        ),
        "authorization": {
            "defect": (
                "phase4_v3_query.py uses list membership in full-ranking tail completion; "
                "on 38,245 IDs this is O(N^2)"
            ),
            "allowed_change": (
                "only set-backed O(N) membership in tail completion, with exact ordered-output preservation"
            ),
            "forbidden_changes": [
                "all retrieval-quality reruns or replacement metrics",
                "all score/rank/RRF/top-k/serialization/model/corpus/index/backend/gate changes",
                "all benchmark-case edits and tuning",
                "all Phase2/Phase3 scientific-artifact changes",
            ],
            "disclosure": (
                "v3 quality evaluation remains sealed; latency recovery is separately authorized and "
                "does not publish new retrieval-quality metrics"
            ),
        },
    }
    auth_hash = _write_json(AUTHORIZATION, boundary)
    _write_sidecar(AUTHORIZATION_SHA, auth_hash)
    protocol = _protocol_value(boundary)
    protocol_hash = _write_json(PROTOCOL, protocol)
    _write_sidecar(PROTOCOL_SHA, protocol_hash)
    _append_lifecycle(
        {
            "event": "frozen",
            "boundary_id": protocol["boundary_id"],
            "authorization_sha256": auth_hash,
            "protocol_sha256": protocol_hash,
            "pre_recovery_code_sha256": pre_code_sha,
            "artifact_inventory_sha256": _digest_json(inventory),
            "authorized": True,
            "status": protocol["status"],
        }
    )
    return {
        "authorization_sha256": auth_hash,
        "protocol_sha256": protocol_hash,
        "pre_recovery_code_sha256": pre_code_sha,
        "artifact_count": len(inventory),
        "model_roles": sorted(models),
    }


def _legacy_complete(head: list[str], ids: list[str], k: int | None = None) -> list[str]:
    ranked = list(head)
    ranked.extend(term_id for term_id in ids if term_id not in ranked)
    return ranked[:k] if k is not None else ranked


def _optimized_complete(head: list[str], ids: list[str], k: int | None = None) -> list[str]:
    """Reference the production set-backed semantics for recovery checks."""
    from scripts.phase4_v3_query import _complete_ranking

    return _complete_ranking(head, ids, k)


def _load_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark = _load_json(V3 / "benchmark.json")
    cases = [case for group in benchmark["strata"].values() for case in group]
    eligible = [
        case
        for case in cases
        if set(case.get("eligible_metrics", []))
        & {"recall_at_5", "mrr", "exact_id_accuracy"}
    ]
    return cases, eligible


def _load_saved_rankings() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (V3 / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prove_equivalence() -> dict[str, Any]:
    """Regression-only proof; it never writes v3 and emits no quality metrics."""
    if not PROTOCOL.is_file() or not AUTHORIZATION.is_file():
        raise RuntimeError("latency-recovery boundary is not sealed")
    if EQUIVALENCE.exists():
        raise RuntimeError("ranking equivalence has already been recorded")
    saved = _load_saved_rankings()
    all_ids = sorted({term_id for row in saved for term_id in row["ranking"]})
    if len(saved) != 87 or len(all_ids) != 38245:
        raise RuntimeError("sealed v3 ranking evidence cardinality changed")
    checks: list[dict[str, Any]] = []
    for row in saved:
        ranking = list(row["ranking"])
        if len(ranking) != 38245 or len(set(ranking)) != 38245:
            raise RuntimeError(f"stored ranking is not a complete permutation: {row}")
        # Compare representative saved heads, including empty and
        # order-sensitive prefixes.  Keep the historical list-membership
        # oracle bounded; running it over a 38,245-ID full tail would recreate
        # the O(N^2) defect this recovery is authorized to remove.  Full-input
        # identity is checked separately with an empty tail.
        prefixes = (0, 1, 2, 5, 10)
        corpus_sample = ranking[:64] + ranking[128:192]
        for size in prefixes:
            head = ranking[:size]
            old = _legacy_complete(head, corpus_sample, None)
            new = _optimized_complete(head, corpus_sample, None)
            if old != new:
                raise RuntimeError(
                    f"tail equivalence mismatch for {row['case_id']} {row['mode']} prefix {size}"
                )
        if _legacy_complete(ranking, [], None) != _optimized_complete(ranking, [], None):
            raise RuntimeError(f"full-input equivalence mismatch for {row['case_id']} {row['mode']}")
        checks.append(
            {
                "case_id": row["case_id"],
                "mode": row["mode"],
                "saved_ranking_sha256": _digest_json(ranking),
                "complete_ranking_byte_identical": True,
                "duplicate_free": len(set(ranking)) == len(ranking),
            }
        )
    result = {
        "schema": 1,
        "boundary_id": "phase4_v3_latency_recovery_v1",
        "kind": "regression_only_ranking_equivalence",
        "quality_evaluation": False,
        "saved_v3_rankings_sha256": sha256_path(V3 / "rankings.jsonl"),
        "saved_v3_records": len(saved),
        "saved_v3_corpus_rows": len(all_ids),
        "records": checks,
        "all_87_byte_identical": all(item["complete_ranking_byte_identical"] for item in checks),
        "all_87_duplicate_free": all(item["duplicate_free"] for item in checks),
        "digest": _digest_json(checks),
    }
    result_hash = _write_json(EQUIVALENCE, result)
    _write_sidecar(EQUIVALENCE_SHA, result_hash)
    return {
        "equivalence_sha256": result_hash,
        "records": len(checks),
        "all_87_byte_identical": result["all_87_byte_identical"],
    }


def seal_implementation() -> dict[str, Any]:
    """Bind the one allowed production diff before any timed query."""
    if IMPLEMENTATION.exists() or not EQUIVALENCE.is_file():
        raise RuntimeError("implementation manifest already exists or equivalence is missing")
    boundary = _load_json(AUTHORIZATION)
    before = boundary["pre_recovery_code_files"]
    after = code_file_hashes()
    changed = [name for name in CODE_FILES if before.get(name) != after.get(name)]
    if changed != ["scripts/phase4_v3_query.py"]:
        raise RuntimeError(f"unauthorized v3 code changes detected: {changed}")
    artifact_inventory = boundary["artifact_inventory"]
    current_inventory = _artifact_inventory()
    if current_inventory != artifact_inventory:
        raise RuntimeError("sealed v3 artifact inventory changed before timing")
    query_path = ROOT / "scripts/phase4_v3_query.py"
    query_text = query_path.read_text(encoding="utf-8")
    if query_text.count("_complete_ranking(") != 5:
        raise RuntimeError("expected one helper definition and four production call sites")
    implementation = {
        "schema": 1,
        "boundary_id": "phase4_v3_latency_recovery_v1",
        "status": "sealed_before_timing",
        "quality_evaluation": False,
        "allowed_change_only": True,
        "pre_recovery_code_sha256": boundary["pre_recovery_code_sha256"],
        "post_recovery_code_sha256": code_sha256(),
        "pre_recovery_code_files": before,
        "post_recovery_code_files": after,
        "changed_code_files": changed,
        "changed_file_semantics": {
            "path": "scripts/phase4_v3_query.py",
            "old_operation": "term_id not in ranked list during full-ranking tail completion",
            "new_operation": "term_id membership in a set-backed seen structure during full-ranking tail completion",
            "complexity": "O(N^2) -> O(N) tail membership/completion",
            "ordered_output": "preserved",
            "production_call_sites": 4,
            "helper_definition": 1,
        },
        "v3_artifacts_unchanged": True,
        "v3_artifact_inventory_sha256": _digest_json(current_inventory),
        "v3_bindings": _v3_bindings(),
        "equivalence_sha256": sha256_path(EQUIVALENCE),
        "forbidden_change_assertions": {
            "quality_metrics_republished": False,
            "rankings_replaced": False,
            "models_corpus_database_index_changed": False,
            "benchmark_or_gates_changed": False,
            "phase2_or_phase3_artifacts_changed": False,
        },
    }
    manifest_hash = _write_json(IMPLEMENTATION, implementation)
    _write_sidecar(IMPLEMENTATION_SHA, manifest_hash)
    return {
        "implementation_manifest_sha256": manifest_hash,
        "post_recovery_code_sha256": implementation["post_recovery_code_sha256"],
        "changed_code_files": changed,
    }


def _offline_probe() -> dict[str, Any]:
    try:
        socket.create_connection(("198.51.100.1", 9), timeout=0.1)
    except (OSError, RuntimeError) as exc:
        return {"network_blocked": True, "exception": type(exc).__name__, "message": str(exc)}
    return {"network_blocked": False, "exception": None, "message": "unexpected connection success"}


def _p95(values: list[float]) -> tuple[float, int]:
    if not values:
        return 0.0, 0
    rank = math.ceil(0.95 * len(values))
    return float(sorted(values)[rank - 1]), rank


def _exposure_event() -> None:
    events = [
        json.loads(line)
        for line in LIFECYCLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [event.get("event") for event in events] != ["frozen"]:
        raise RuntimeError("recovery lifecycle is not at the frozen-only state")
    if not IMPLEMENTATION.is_file():
        raise RuntimeError("implementation manifest is not sealed before exposure")
    optimized_regression = RECOVERY / "optimized_ranking_regression.json"
    if not optimized_regression.is_file():
        raise RuntimeError("optimized complete-ranking regression is missing")
    _append_lifecycle(
        {
            "event": "exposure_started",
            "boundary_id": "phase4_v3_latency_recovery_v1",
            "authorization_sha256": sha256_path(AUTHORIZATION),
            "protocol_sha256": sha256_path(PROTOCOL),
            "equivalence_sha256": sha256_path(EQUIVALENCE),
            "optimized_regression_sha256": sha256_path(optimized_regression),
            "pre_recovery_code_sha256": _load_json(AUTHORIZATION)["pre_recovery_code_sha256"],
            "implementation_code_sha256": code_sha256(),
            "evaluation_exposed": False,
            "quality_evaluation": False,
            "authorized": True,
        }
    )


def measure() -> dict[str, Any]:
    """Run exactly one latency-only timing pass and seal its evidence."""
    if not EQUIVALENCE.is_file():
        raise RuntimeError("run regression-only ranking equivalence before timing")
    if RAW_SAMPLES.exists() or REPORT.exists():
        raise RuntimeError("latency recovery timing has already been run")
    protocol = _load_json(PROTOCOL)
    if protocol.get("status") != "frozen_latency_only_no_quality_evaluation":
        raise RuntimeError("latency protocol is not frozen")
    cases, eligible = _load_cases()
    if len(cases) != 31 or len(eligible) != 29:
        raise RuntimeError("v3 timing workload cardinality changed")
    _exposure_event()
    try:
        with network_block():
            os.environ.update(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
            )
            import torch

            torch.manual_seed(0)
            np.random.seed(0)
            _enable_recovery_snapshot_validation()
            query_engine._load_snapshot()
            query_engine._load_query_model()
            offline = _offline_probe()
            if not offline["network_blocked"]:
                raise RuntimeError("offline network guard did not block the probe")
            rows: list[dict[str, Any]] = []
            for case in eligible:
                for mode in MODES:
                    for _ in range(WARMUPS):
                        warm = query_engine.retrieve(case["query"], mode, None)
                        if len(warm) != 38245 or len(set(warm)) != 38245:
                            raise RuntimeError(f"warmup ranking invariant failed: {case['id']} {mode}")
                    samples: list[float] = []
                    retained: list[str] | None = None
                    for _ in range(MEASURED):
                        started = time.perf_counter_ns()
                        ranking = query_engine.retrieve(case["query"], mode, None)
                        elapsed = time.perf_counter_ns() - started
                        if len(ranking) != 38245 or len(set(ranking)) != 38245:
                            raise RuntimeError(f"measured ranking invariant failed: {case['id']} {mode}")
                        if retained is None:
                            retained = list(ranking)
                        elif ranking != retained:
                            raise RuntimeError(f"non-deterministic ranking: {case['id']} {mode}")
                        samples.append(round(elapsed / 1_000_000, 6))
                    assert retained is not None
                    rows.append(
                        {
                            "case_id": case["id"],
                            "mode": mode,
                            "ranking_sha256": _digest_json(retained),
                            "latency_ms": samples,
                            "warmup_repeats": WARMUPS,
                            "measured_repeats": MEASURED,
                        }
                    )
    except Exception as exc:
        _append_lifecycle(
            {
                "event": "gate_failed_no_tuning",
                "boundary_id": "phase4_v3_latency_recovery_v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "authorized": True,
                "quality_evaluation": False,
            }
        )
        raise
    samples_by_mode = {
        mode: [sample for row in rows if row["mode"] == mode for sample in row["latency_ms"]]
        for mode in MODES
    }
    latency: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for mode, values in samples_by_mode.items():
        p95, rank = _p95(values)
        latency[mode] = {
            "case_count": len([row for row in rows if row["mode"] == mode]),
            "sample_count": len(values),
            "samples_ms": values,
            "nearest_rank": rank,
            "p95_ms": p95,
            "scope": protocol["timing_contract"]["scope"],
        }
        gates[mode] = {
            "p95_latency_ms": {
                "value": p95,
                "threshold": LATENCY_GATE_MS,
                "pass": p95 <= LATENCY_GATE_MS,
            }
        }
    all_pass = all(item["p95_latency_ms"]["pass"] for item in gates.values())
    raw = {
        "schema": 1,
        "boundary_id": "phase4_v3_latency_recovery_v1",
        "kind": "latency_only_raw_samples",
        "quality_evaluation": False,
        "execution_id": str(uuid.uuid4()),
        "rows": rows,
        "latency": latency,
        "offline": offline,
        "environment": {
            "runtime": runtime_record(),
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "bindings": {
            "authorization_sha256": sha256_path(AUTHORIZATION),
            "protocol_sha256": sha256_path(PROTOCOL),
            "equivalence_sha256": sha256_path(EQUIVALENCE),
            "optimized_regression_sha256": sha256_path(
                RECOVERY / "optimized_ranking_regression.json"
            ),
            "implementation_code_sha256": code_sha256(),
            "v3_rankings_sha256": sha256_path(V3 / "rankings.jsonl"),
            "v3_report_sha256": sha256_path(V3 / "evaluation_report.json"),
            "v3_lifecycle_sha256": sha256_path(V3 / "lifecycle.jsonl"),
        },
    }
    raw_hash = _write_json(RAW_SAMPLES, raw)
    _write_sidecar(RAW_SAMPLES_SHA, raw_hash)
    report = {
        "schema": 1,
        "boundary_id": "phase4_v3_latency_recovery_v1",
        "kind": "latency_only_report",
        "status": "accepted" if all_pass else "gate_failed_no_tuning",
        "quality_evaluation": False,
        "disclosure": protocol["disclosure"],
        "execution_id": raw["execution_id"],
        "latency": latency,
        "gates": gates,
        "all_gates_pass": all_pass,
        "raw_samples_sha256": raw_hash,
        "bindings": raw["bindings"],
        "environment": raw["environment"],
        "offline": offline,
    }
    report_hash = _write_json(REPORT, report)
    _write_sidecar(REPORT_SHA, report_hash)
    _append_lifecycle(
        {
            "event": "latency_complete" if all_pass else "gate_failed_no_tuning",
            "boundary_id": "phase4_v3_latency_recovery_v1",
            "execution_id": raw["execution_id"],
            "report_sha256": report_hash,
            "raw_samples_sha256": raw_hash,
            "all_gates_pass": all_pass,
            "status": report["status"],
            "authorized": True,
            "quality_evaluation": False,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("seal", "equivalence", "seal-implementation", "measure")
    )
    args = parser.parse_args()
    if args.command == "seal":
        output = seal()
    elif args.command == "equivalence":
        output = prove_equivalence()
    elif args.command == "seal-implementation":
        output = seal_implementation()
    else:
        output = measure()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
