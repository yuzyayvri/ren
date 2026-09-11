"""Read-only verifier for the sealed Phase 4 v3 snapshot and evaluation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from scripts import phase4_v3_query as query_engine
from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    OBO,
    OBO_SHA256,
    QUERY_MODEL,
    V3,
    code_sha256,
    model_manifest,
    sha256_bytes,
    sha256_path,
)


def _json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing v3 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _report_digest(path: Path) -> str:
    rows = path.read_text(encoding="ascii").split()
    if len(rows) != 2 or rows[1] != "evaluation_report.json":
        raise RuntimeError("evaluation report sidecar format is invalid")
    return rows[0]


def _flatten(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for cases in benchmark["strata"].values() for case in cases]


def _verify_rankings(
    report: dict[str, Any], cases: list[dict[str, Any]], ids: list[str]
) -> dict[str, Any]:
    evidence = report.get("ranking_evidence")
    if not evidence:
        return {"present": False, "records": 0}
    path = V3 / Path(evidence["path"]).name
    if not path.is_file() or sha256_path(path) != evidence["sha256"]:
        raise RuntimeError("ranking evidence hash/path mismatch")
    expected_cases = {
        (case["id"], mode)
        for case in cases
        if set(case.get("eligible_metrics", []))
        & {"recall_at_5", "mrr", "exact_id_accuracy"}
        for mode in ("symbolic", "vector", "hybrid")
    }
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["case_id"], row["mode"])
            if key in seen or key not in expected_cases:
                raise RuntimeError(f"unexpected/duplicate ranking evidence row: {key}")
            ranking = row["ranking"]
            if (
                len(ranking) != len(ids)
                or len(set(ranking)) != len(ids)
                or set(ranking) != set(ids)
            ):
                raise RuntimeError(
                    f"ranking evidence is not a complete corpus permutation: {key}"
                )
            if sha256_bytes(
                (
                    json.dumps(ranking, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
            ) != next(
                item["ranking_sha256"]
                for item in report.get("rows", [])
                if item["case_id"] == row["case_id"] and item["mode"] == row["mode"]
            ):
                raise RuntimeError(f"ranking evidence digest mismatch: {key}")
            seen.add(key)
    if seen != expected_cases:
        raise RuntimeError(
            f"ranking evidence coverage mismatch: {len(seen)} vs {len(expected_cases)}"
        )
    if int(evidence.get("records", -1)) != len(seen):
        raise RuntimeError("ranking evidence record count mismatch")
    return {"present": True, "records": len(seen), "sha256": evidence["sha256"]}


def verify() -> dict[str, Any]:
    protocol = _json(V3 / "protocol.json")
    benchmark = _json(V3 / "benchmark.json")
    freeze = _json(V3 / "freeze_manifest.json")
    index = _json(V3 / "index_manifest.json")
    source = _json(V3 / "source.json")
    report = _json(V3 / "evaluation_report.json")
    report_path = V3 / "evaluation_report.json"
    report_sha = sha256_path(report_path)
    sidecar_sha = _report_digest(V3 / "evaluation_report.sha256")
    if report_sha != sidecar_sha:
        raise RuntimeError("evaluation report sidecar digest mismatch")
    if (
        protocol.get("status") != "frozen_prospective_no_evaluation"
        or benchmark.get("status") != "frozen_prospective_no_evaluation"
    ):
        raise RuntimeError("protocol or benchmark is not frozen")
    protocol_sha = sha256_path(V3 / "protocol.json")
    benchmark_sha = sha256_path(V3 / "benchmark.json")
    if (
        freeze.get("protocol_sha256") != protocol_sha
        or freeze.get("benchmark_sha256") != benchmark_sha
    ):
        raise RuntimeError("freeze manifest mismatch")
    if (
        sha256_path(OBO) != OBO_SHA256
        or source.get("sha256") != OBO_SHA256
        or protocol.get("source", {}).get("sha256") != OBO_SHA256
    ):
        raise RuntimeError("GO source binding mismatch")
    current_code = code_sha256()
    if (
        index.get("code_sha256") != current_code
        or protocol.get("provenance", {}).get("code_sha256") != current_code
    ):
        raise RuntimeError("code digest binding mismatch")
    query_manifest = model_manifest(QUERY_MODEL)
    article_manifest = model_manifest(ARTICLE_MODEL)
    for role, manifest in (("query", query_manifest), ("article", article_manifest)):
        if (
            index.get("models", {}).get(role, {}).get("manifest_sha256")
            != manifest["manifest_sha256"]
        ):
            raise RuntimeError(f"{role} model manifest mismatch")
    query_engine.verify_snapshot()

    db = sqlite3.connect(V3 / "ontology.sqlite")
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    ids = [
        row[0]
        for row in db.execute("SELECT id FROM terms WHERE obsolete=0 ORDER BY id")
    ]
    db.close()
    docs = _json(V3 / "documents.json")
    if integrity != "ok" or len(ids) != 38245 or [row["id"] for row in docs] != ids:
        raise RuntimeError("database/document integrity or order mismatch")
    array = np.load(V3 / "article_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (
        array.shape != (len(ids), 768)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise RuntimeError("article embedding array invariant failed")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0) or float(np.max(np.abs(norms - 1))) > 2e-5:
        raise RuntimeError("article embedding normalization invariant failed")

    lifecycle = [
        json.loads(line)
        for line in (V3 / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lifecycle) < 2 or lifecycle[0].get("event") != "freeze":
        raise RuntimeError("lifecycle freeze event is missing")
    exposure = [event for event in lifecycle if event.get("event") == "exposure"]
    completions = [
        event
        for event in lifecycle
        if event.get("event") in {"evaluation_complete", "evaluation_failed"}
    ]
    if len(exposure) != 1 or len(completions) != 1:
        raise RuntimeError("lifecycle exposure/completion cardinality is invalid")
    execution_id = exposure[0].get("execution_id")
    if (
        not execution_id
        or completions[0].get("execution_id") != execution_id
        or report.get("execution_id") != execution_id
    ):
        raise RuntimeError("lifecycle/report execution binding mismatch")
    if (
        exposure[0].get("protocol_sha256") != protocol_sha
        or exposure[0].get("benchmark_sha256") != benchmark_sha
    ):
        raise RuntimeError("exposure event does not bind frozen artifacts")
    if completions[0].get("report_sha256") != report_sha:
        raise RuntimeError("completion event does not bind report")

    cases = _flatten(benchmark)
    rows = report.get("rows", [])
    row_keys = {(row.get("case_id"), row.get("mode")) for row in rows}
    expected_keys = {
        (case["id"], mode)
        for case in cases
        if set(case.get("eligible_metrics", []))
        & {"recall_at_5", "mrr", "exact_id_accuracy"}
        for mode in ("symbolic", "vector", "hybrid")
    }
    if row_keys != expected_keys or len(rows) != len(expected_keys):
        raise RuntimeError("evaluation row coverage mismatch")
    for row in rows:
        if row.get("ranking_rows") != len(ids) or len(row.get("top5", [])) != 5:
            raise RuntimeError(
                f"evaluation row ranking evidence is incomplete: {row.get('case_id')} {row.get('mode')}"
            )
        if (
            len(row.get("latency_ms", [])) != 5
            or row.get("warmup_repeats") != 2
            or row.get("measured_repeats") != 5
        ):
            raise RuntimeError("latency repeat contract mismatch")
        metrics = row.get("metrics", {})
        if set(metrics.get("target_ranks", {})) != set(metrics.get("gold_ids", [])):
            raise RuntimeError("target rank evidence does not cover every gold ID")
    ranking_evidence = _verify_rankings(report, cases, ids)
    if report.get("status") == "accepted" and not report.get("all_gates_pass"):
        raise RuntimeError("accepted report does not assert all gates pass")
    if report.get("status") == "gate_failed_no_tuning" and report.get("all_gates_pass"):
        raise RuntimeError("gate-failed report asserts all gates pass")
    return {
        "status": "verified",
        "report_status": report.get("status"),
        "execution_id": execution_id,
        "protocol_sha256": protocol_sha,
        "benchmark_sha256": benchmark_sha,
        "report_sha256": report_sha,
        "ranking_evidence": ranking_evidence,
        "case_count": len(cases),
        "evaluation_rows": len(rows),
        "corpus_rows": len(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify()
    print(
        json.dumps(result, sort_keys=True)
        if args.json
        else "v3 immutable snapshot and evaluation verified"
    )


if __name__ == "__main__":
    main()
