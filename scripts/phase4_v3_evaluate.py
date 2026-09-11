"""Execute the one-shot, prospective Phase 4 v3 retrieval evaluation.

The evaluator owns the exposure boundary and writes the retained evidence.  It
does not modify the frozen protocol, benchmark, documents, database, or index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from scripts import phase4_v3_query as query_engine
from scripts.phase4_v3_benchmark import ALLOWED_METRICS
from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    OBO,
    OBO_SHA256,
    QUERY_MODEL,
    V3,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    code_sha256,
    model_manifest,
    network_block,
    runtime_record,
    sha256_bytes,
    sha256_path,
)

MODES = ("symbolic", "vector", "hybrid")
WARMUPS = 2
MEASURED = 5


def _jsonl_append(path: Path, event: dict[str, Any]) -> None:
    payload = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required v3 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _flatten(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for cases in benchmark["strata"].values() for case in cases]


def _preflight() -> dict[str, Any]:
    """Check every immutable binding before the exposure event is written."""
    protocol = _load_json(V3 / "protocol.json")
    benchmark = _load_json(V3 / "benchmark.json")
    freeze = _load_json(V3 / "freeze_manifest.json")
    index = _load_json(V3 / "index_manifest.json")
    source = _load_json(V3 / "source.json")
    source_audit = _load_json(V3 / "source_audit.json")
    model_audit = _load_json(V3 / "model_audit.json")
    runtime_audit = _load_json(V3 / "runtime_audit.json")

    if protocol.get("status") != "frozen_prospective_no_evaluation":
        raise RuntimeError("protocol is not frozen prospectively")
    if benchmark.get("status") != "frozen_prospective_no_evaluation":
        raise RuntimeError("benchmark is not frozen prospectively")
    protocol_sha = sha256_path(V3 / "protocol.json")
    benchmark_sha = sha256_path(V3 / "benchmark.json")
    if (
        freeze.get("protocol_sha256") != protocol_sha
        or freeze.get("benchmark_sha256") != benchmark_sha
    ):
        raise RuntimeError("freeze manifest does not bind current protocol/benchmark")
    if (
        protocol.get("benchmark", {}).get("path")
        != "artifacts/phase4_protocol_v3/benchmark.json"
    ):
        raise RuntimeError("protocol benchmark path is not the v3 boundary")
    if (
        protocol.get("source", {}).get("sha256") != OBO_SHA256
        or benchmark.get("source_sha256") != OBO_SHA256
    ):
        raise RuntimeError("frozen source binding mismatch")
    if sha256_path(OBO) != OBO_SHA256 or source.get("sha256") != OBO_SHA256:
        raise RuntimeError("approved GO source bytes are not bound")

    current_code = code_sha256()
    if protocol.get("provenance", {}).get("code_sha256") != current_code:
        raise RuntimeError("protocol code binding is stale")
    if index.get("code_sha256") != current_code:
        raise RuntimeError("index code binding is stale")
    if (
        source_audit.get("code", {}).get("sha256") != current_code
        or model_audit.get("code", {}).get("sha256") != current_code
    ):
        raise RuntimeError("audit code binding is stale")
    if runtime_audit.get("code_sha256") != current_code:
        raise RuntimeError("runtime audit code binding is stale")

    models = {
        "query": model_manifest(QUERY_MODEL),
        "article": model_manifest(ARTICLE_MODEL),
    }
    for role, manifest in models.items():
        expected = protocol.get("encoders", {}).get(role, {})
        if (
            expected.get("repository") != manifest["repository"]
            or expected.get("revision") != manifest["revision"]
        ):
            raise RuntimeError(f"{role} model revision binding is stale")
        if (
            index.get("models", {}).get(role, {}).get("manifest_sha256")
            != manifest["manifest_sha256"]
        ):
            raise RuntimeError(f"{role} model manifest is not bound by index")

    if index.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("index does not bind the frozen protocol")
    if index.get("source_sha256") != OBO_SHA256:
        raise RuntimeError("index source binding mismatch")
    query_engine.verify_snapshot()

    cases = _flatten(benchmark)
    if len(cases) != 31 or len({case.get("id") for case in cases}) != len(cases):
        raise RuntimeError(
            "frozen benchmark case cardinality/identity invariant failed"
        )
    if any(set(case.get("eligible_metrics", [])) - ALLOWED_METRICS for case in cases):
        raise RuntimeError("frozen benchmark contains an unknown eligible metric")
    if any(
        case.get("provenance", {}).get("ontology_source", {}).get("sha256")
        != OBO_SHA256
        for case in cases
    ):
        raise RuntimeError("case-level ontology provenance is incomplete")

    lifecycle_path = V3 / "lifecycle.jsonl"
    if not lifecycle_path.is_file():
        raise RuntimeError("v3 lifecycle is missing")
    lifecycle_events = [
        json.loads(line)
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lifecycle_events or lifecycle_events[0].get("event") != "freeze":
        raise RuntimeError("v3 lifecycle does not begin with freeze")
    if any(
        event.get("event") in {"exposure", "evaluation_complete", "evaluation_failed"}
        for event in lifecycle_events
    ):
        raise RuntimeError("v3 evaluation has already been exposed or completed")
    for path in (
        V3 / "evaluation_report.json",
        V3 / "evaluation_report.sha256",
        V3 / "rankings.jsonl",
    ):
        if path.exists():
            raise RuntimeError(f"one-shot evaluation artifact already exists: {path}")

    # Verify the canonical array and the document/database order before any
    # query model is loaded.  query_engine.verify_snapshot checks hashes; these
    # checks cover the scientific shape and deterministic row contract.
    import sqlite3

    db = sqlite3.connect(V3 / "ontology.sqlite")
    ids = [
        row[0]
        for row in db.execute("SELECT id FROM terms WHERE obsolete=0 ORDER BY id")
    ]
    db_integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    docs = json.loads((V3 / "documents.json").read_text(encoding="utf-8"))
    if db_integrity != "ok" or [row["id"] for row in docs] != ids:
        raise RuntimeError("database/document order invariant failed")
    array = np.load(V3 / "article_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (
        array.shape != (len(ids), 768)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise RuntimeError("canonical article array invariant failed")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0) or float(np.max(np.abs(norms - 1))) > 2e-5:
        raise RuntimeError("canonical article array normalization invariant failed")

    return {
        "protocol_sha256": protocol_sha,
        "benchmark_sha256": benchmark_sha,
        "freeze_manifest_sha256": sha256_path(V3 / "freeze_manifest.json"),
        "index_manifest_sha256": sha256_path(V3 / "index_manifest.json"),
        "source_sha256": OBO_SHA256,
        "database_sha256": sha256_path(V3 / "ontology.sqlite"),
        "documents_sha256": sha256_path(V3 / "documents.json"),
        "embeddings_sha256": sha256_path(V3 / "article_embeddings.npy"),
        "code_sha256": current_code,
        "models": {
            role: {
                "repository": manifest["repository"],
                "revision": manifest["revision"],
                "manifest_sha256": manifest["manifest_sha256"],
            }
            for role, manifest in models.items()
        },
        "runtime": runtime_record(),
        "case_count": len(cases),
        "corpus_rows": len(ids),
    }


def _ranking_digest(ranking: list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(ranking))


def _metrics(case: dict[str, Any], ranking: list[str]) -> dict[str, Any]:
    if len(ranking) == 0 or len(set(ranking)) != len(ranking):
        raise RuntimeError(f"ranking is empty or contains duplicates: {case['id']}")
    gold = list(case["gold_ids"])
    positions = {term_id: rank for rank, term_id in enumerate(ranking, start=1)}
    target_ranks = {term_id: positions.get(term_id) for term_id in gold}
    first = min(
        (rank for rank in target_ranks.values() if rank is not None), default=None
    )
    top5_hits = [
        term_id for term_id in gold if positions.get(term_id, len(ranking) + 1) <= 5
    ]
    result: dict[str, Any] = {
        "gold_ids": gold,
        "target_ranks": target_ranks,
        "first_relevant_rank": first,
        "top5_hits": top5_hits,
    }
    if "recall_at_5" in case.get("eligible_metrics", []):
        result["recall_at_5"] = len(top5_hits) / len(gold) if gold else 0.0
    if "mrr" in case.get("eligible_metrics", []):
        result["mrr"] = 1.0 / first if first is not None else 0.0
    if "exact_id_accuracy" in case.get("eligible_metrics", []):
        result["exact_id_accuracy"] = bool(first == 1 and len(gold) == 1)
    return result


def _evaluate_retrieval_case(
    case: dict[str, Any], mode: str, ranking_stream: Any
) -> dict[str, Any]:
    retrieve: Callable[[str, str, int | None], list[str]] = query_engine.retrieve
    for _ in range(WARMUPS):
        ranking = retrieve(case["query"], mode, None)
        if len(ranking) == 0:
            raise RuntimeError(f"warmup produced an empty ranking: {case['id']} {mode}")
    samples_ns: list[int] = []
    measured_hashes: list[str] = []
    retained: list[str] | None = None
    for _ in range(MEASURED):
        started = time.perf_counter_ns()
        ranking = retrieve(case["query"], mode, None)
        elapsed = time.perf_counter_ns() - started
        if len(ranking) == 0:
            raise RuntimeError(
                f"measured query produced an empty ranking: {case['id']} {mode}"
            )
        digest = _ranking_digest(ranking)
        samples_ns.append(elapsed)
        measured_hashes.append(digest)
        if retained is None:
            retained = ranking
        elif ranking != retained:
            raise RuntimeError(
                f"non-deterministic measured ranking: {case['id']} {mode}"
            )
    assert retained is not None
    if len(set(measured_hashes)) != 1:
        raise RuntimeError(
            f"non-deterministic measured ranking digest: {case['id']} {mode}"
        )
    ranking_stream.write(
        (
            json.dumps(
                {"case_id": case["id"], "mode": mode, "ranking": retained},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    metrics = _metrics(case, retained)
    return {
        "case_id": case["id"],
        "mode": mode,
        "ranking_rows": len(retained),
        "ranking_sha256": measured_hashes[0],
        "top5": retained[:5],
        "metrics": metrics,
        "latency_ms": [round(value / 1_000_000, 6) for value in samples_ns],
        "warmup_repeats": WARMUPS,
        "measured_repeats": MEASURED,
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)])


def _aggregate(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    eligible = [row for row in rows if metric in row["metrics"]]
    values = [float(row["metrics"][metric]) for row in eligible]
    return {
        "metric": metric,
        "eligible_cases": len(eligible),
        "values": values,
        "mean": sum(values) / len(values) if values else None,
    }


def _traversal_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if case["case_type"] != "traversal":
            continue
        started = time.perf_counter_ns()
        actual = query_engine.traverse(
            case["query"],
            case["direction"],
            int(case["depth"]),
            tuple(case["relations"]),
        )
        elapsed = time.perf_counter_ns() - started
        expected = sorted(case["gold_ids"])
        rows.append(
            {
                "case_id": case["id"],
                "query": case["query"],
                "direction": case["direction"],
                "depth": case["depth"],
                "relations": case["relations"],
                "expected": expected,
                "actual": actual,
                "correct": actual == expected,
                "latency_ms": round(elapsed / 1_000_000, 6),
            }
        )
    return rows


def _offline_evidence() -> dict[str, Any]:
    try:
        socket.create_connection(("198.51.100.1", 9), timeout=0.1)
    except (OSError, RuntimeError) as exc:  # expected: network_block turns this into RuntimeError
        return {
            "network_blocked": True,
            "exception": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "network_blocked": False,
        "exception": None,
        "message": "unexpected connection success",
    }


def evaluate() -> dict[str, Any]:
    preflight = _preflight()
    benchmark = _load_json(V3 / "benchmark.json")
    cases = _flatten(benchmark)
    execution_id = str(uuid.uuid4())
    lifecycle_path = V3 / "lifecycle.jsonl"
    _jsonl_append(
        lifecycle_path,
        {
            "event": "exposure",
            "execution_id": execution_id,
            "protocol_sha256": preflight["protocol_sha256"],
            "benchmark_sha256": preflight["benchmark_sha256"],
            "index_manifest_sha256": preflight["index_manifest_sha256"],
            "code_sha256": preflight["code_sha256"],
            "evaluation_exposed": True,
            "authorized": True,
        },
    )

    ranking_tmp = V3 / "rankings.jsonl.tmp"
    report_path = V3 / "evaluation_report.json"
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
            # Load all state before timing.  The first query therefore cannot
            # hide database/mmap/model initialization in a latency sample.
            query_engine._load_snapshot()
            query_engine._load_query_model()
            offline = _offline_evidence()
            if not offline["network_blocked"]:
                raise RuntimeError("offline network guard did not block the probe")
            ranking_rows: list[dict[str, Any]] = []
            with ranking_tmp.open("wb") as stream:
                for case in cases:
                    if not (
                        set(case.get("eligible_metrics", []))
                        & {"recall_at_5", "mrr", "exact_id_accuracy"}
                    ):
                        continue
                    for mode in MODES:
                        ranking_rows.append(
                            _evaluate_retrieval_case(case, mode, stream)
                        )
                stream.flush()
                os.fsync(stream.fileno())

            # A second un-timed pass demonstrates deterministic reproduction;
            # it is deliberately retained as hashes only to keep the report
            # compact while the first pass retains complete rankings.
            reproduction: list[dict[str, Any]] = []
            for row in ranking_rows:
                case = next(item for item in cases if item["id"] == row["case_id"])
                ranking = query_engine.retrieve(case["query"], row["mode"], None)
                digest = _ranking_digest(ranking)
                reproduction.append(
                    {
                        "case_id": row["case_id"],
                        "mode": row["mode"],
                        "ranking_sha256": digest,
                        "matches": digest == row["ranking_sha256"],
                    }
                )
            if not all(item["matches"] for item in reproduction):
                raise RuntimeError("deterministic reproduction mismatch")
            traversal = _traversal_rows(cases)

        latency: dict[str, Any] = {}
        for mode in MODES:
            mode_rows = [row for row in ranking_rows if row["mode"] == mode]
            samples = [sample for row in mode_rows for sample in row["latency_ms"]]
            latency[mode] = {
                "case_count": len(mode_rows),
                "sample_count": len(samples),
                "p95_ms": _p95(samples),
                "nearest_rank": math.ceil(0.95 * len(samples)) if samples else None,
                "samples_ms": samples,
                "scope": "query plus ranking; cache preloaded; hybrid timed directly",
            }
        aggregates = {
            mode: {
                metric: _aggregate(
                    [row for row in ranking_rows if row["mode"] == mode], metric
                )
                for metric in ("recall_at_5", "mrr", "exact_id_accuracy")
            }
            for mode in MODES
        }
        traversal_correct = (
            sum(bool(row["correct"]) for row in traversal) / len(traversal)
            if traversal
            else 0.0
        )
        gates = _load_json(V3 / "protocol.json")["gates"]
        gate_results: dict[str, Any] = {}
        for mode in MODES:
            mode_aggs = aggregates[mode]
            gate_results[mode] = {
                "recall_at_5": {
                    "value": mode_aggs["recall_at_5"]["mean"],
                    "threshold": gates["recall_at_5"],
                    "pass": (mode_aggs["recall_at_5"]["mean"] or 0)
                    >= gates["recall_at_5"],
                },
                "mrr": {
                    "value": mode_aggs["mrr"]["mean"],
                    "threshold": gates["mrr"],
                    "pass": (mode_aggs["mrr"]["mean"] or 0) >= gates["mrr"],
                },
                "exact_id_accuracy": {
                    "value": mode_aggs["exact_id_accuracy"]["mean"],
                    "threshold": gates["exact_id_accuracy"],
                    "pass": (mode_aggs["exact_id_accuracy"]["mean"] or 0)
                    >= gates["exact_id_accuracy"],
                },
                "traversal_correctness": {
                    "value": traversal_correct,
                    "threshold": gates["traversal_correctness"],
                    "pass": traversal_correct >= gates["traversal_correctness"],
                },
                "p95_latency_ms": {
                    "value": latency[mode]["p95_ms"],
                    "threshold": gates["p95_latency_ms"],
                    "pass": latency[mode]["p95_ms"] <= gates["p95_latency_ms"],
                },
            }
        gate_results["deterministic_reproduction"] = {
            "value": all(item["matches"] for item in reproduction),
            "pass": all(item["matches"] for item in reproduction),
        }
        gate_results["offline"] = {
            "value": offline["network_blocked"],
            "pass": offline["network_blocked"],
        }
        gate_results["provenance"] = {"value": True, "pass": True}
        all_pass = all(
            item["pass"]
            for result in gate_results.values()
            for item in (
                result.values()
                if isinstance(result, dict) and "pass" not in result
                else [result]
            )
        )
        report = {
            "schema": 3,
            "status": "accepted" if all_pass else "gate_failed_no_tuning",
            "execution_id": execution_id,
            "evaluation_exposed": True,
            "disclosure": "v1/v2 were invalid exposed regression evidence; v3 is a separately frozen prospective benchmark/evaluation after reviewed correction.",
            "preflight": preflight,
            "benchmark_case_count": len(cases),
            "modes": list(MODES),
            "rows": ranking_rows,
            "aggregates": aggregates,
            "latency": latency,
            "traversal": traversal,
            "reproduction": reproduction,
            "offline": offline,
            "gates": gate_results,
            "all_gates_pass": all_pass,
            "ranking_evidence": {
                "path": "artifacts/phase4_protocol_v3/rankings.jsonl",
                "sha256": sha256_path(ranking_tmp),
                "records": len(ranking_rows),
                "format": "one complete GO-ID ranking per case/mode",
            },
            "runtime_after": runtime_record(),
        }
        atomic_write_json(report_path, report)
        report_sha = sha256_path(report_path)
        atomic_write_bytes(
            V3 / "evaluation_report.sha256",
            f"{report_sha}  evaluation_report.json\n".encode("ascii"),
        )
        os.replace(ranking_tmp, V3 / "rankings.jsonl")
        _jsonl_append(
            lifecycle_path,
            {
                "event": "evaluation_complete",
                "execution_id": execution_id,
                "report_sha256": report_sha,
                "ranking_evidence_sha256": sha256_path(V3 / "rankings.jsonl"),
                "all_gates_pass": all_pass,
                "status": report["status"],
            },
        )
        return report
    except Exception as exc:
        failure = {
            "schema": 3,
            "status": "evaluation_failed_after_exposure",
            "execution_id": execution_id,
            "evaluation_exposed": True,
            "preflight": preflight,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(V3 / "evaluation_report.json", failure)
        report_sha = sha256_path(V3 / "evaluation_report.json")
        atomic_write_bytes(
            V3 / "evaluation_report.sha256",
            f"{report_sha}  evaluation_report.json\n".encode("ascii"),
        )
        if ranking_tmp.exists():
            os.replace(ranking_tmp, V3 / "rankings.jsonl.failed")
        _jsonl_append(
            lifecycle_path,
            {
                "event": "evaluation_failed",
                "execution_id": execution_id,
                "report_sha256": report_sha,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", action="store_true", help="print the sealed report as JSON"
    )
    args = parser.parse_args()
    report = evaluate()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "all_gates_pass": report["all_gates_pass"],
                    "execution_id": report["execution_id"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
