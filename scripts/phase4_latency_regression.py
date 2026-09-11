"""Run the clearly labeled, quality-free v3 ranking regression check.

This is deliberately separate from the latency measurement.  It executes the
optimized query path once per saved case/mode, compares the complete ordered
lists to sealed v3 evidence, and emits no Recall/MRR/exact-ID metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts import phase4_v3_query as query_engine
from scripts.phase4_latency_recovery import _enable_recovery_snapshot_validation
from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    QUERY_MODEL,
    V3,
    canonical_json_bytes,
    code_sha256,
    model_manifest,
    network_block,
    runtime_record,
    sha256_path,
)

RECOVERY = V3 / "recovery" / "latency_recovery_v1"
OUTPUT = RECOVERY / "optimized_ranking_regression.json"
OUTPUT_SHA = RECOVERY / "optimized_ranking_regression.sha256"


def _digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _offline_probe() -> dict[str, Any]:
    try:
        socket.create_connection(("198.51.100.1", 9), timeout=0.1)
    except (OSError, RuntimeError) as exc:
        return {"network_blocked": True, "exception": type(exc).__name__, "message": str(exc)}
    return {"network_blocked": False, "exception": None, "message": "unexpected connection success"}


def main() -> None:
    if OUTPUT.exists() or OUTPUT_SHA.exists():
        raise RuntimeError("optimized ranking regression has already been recorded")
    saved = [
        json.loads(line)
        for line in (V3 / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(saved) != 87:
        raise RuntimeError(f"expected 87 sealed rankings, got {len(saved)}")
    benchmark = _load_json(V3 / "benchmark.json")
    cases = {
        case["id"]: case
        for group in benchmark["strata"].values()
        for case in group
    }
    if any(row["case_id"] not in cases for row in saved):
        raise RuntimeError("saved ranking references an unknown benchmark case")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    checks: list[dict[str, Any]] = []
    with network_block():
        _enable_recovery_snapshot_validation()
        query_engine._load_snapshot()
        query_engine._load_query_model()
        offline = _offline_probe()
        if not offline["network_blocked"]:
            raise RuntimeError("offline network guard did not block the probe")
        for row in saved:
            case = cases[row["case_id"]]
            observed = query_engine.retrieve(case["query"], row["mode"], None)
            observed_list = list(observed)
            observed_sha = _digest_json(observed_list)
            checks.append(
                {
                    "case_id": row["case_id"],
                    "mode": row["mode"],
                    "saved_ranking_sha256": _digest_json(row["ranking"]),
                    "observed_ranking_sha256": observed_sha,
                    "byte_identical": observed_list == row["ranking"],
                    "duplicate_free": len(observed_list) == len(set(observed_list)),
                }
            )
            if not checks[-1]["byte_identical"] or not checks[-1]["duplicate_free"]:
                raise RuntimeError(f"optimized ranking regression mismatch: {row['case_id']} {row['mode']}")

    result = {
        "schema": 1,
        "boundary_id": "phase4_v3_latency_recovery_v1",
        "kind": "optimized_complete_ranking_regression",
        "quality_evaluation": False,
        "saved_v3_rankings_sha256": sha256_path(V3 / "rankings.jsonl"),
        "records": checks,
        "record_count": len(checks),
        "all_87_byte_identical": all(item["byte_identical"] for item in checks),
        "all_87_duplicate_free": all(item["duplicate_free"] for item in checks),
        "regression_digest": _digest_json(checks),
        "bindings": {
            "implementation_code_sha256": code_sha256(),
            "query_code_sha256": sha256_path(Path("scripts/phase4_v3_query.py")),
            "v3_report_sha256": sha256_path(V3 / "evaluation_report.json"),
            "v3_lifecycle_sha256": sha256_path(V3 / "lifecycle.jsonl"),
            "query_model_manifest": model_manifest(QUERY_MODEL),
            "article_model_manifest": model_manifest(ARTICLE_MODEL),
        },
        "environment": {
            "runtime": runtime_record(),
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "offline": offline,
    }
    payload = canonical_json_bytes(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    OUTPUT_SHA.write_text(f"{digest}  {OUTPUT.name}\n", encoding="ascii")
    print(json.dumps({"output_sha256": digest, "records": len(checks), "all_87_byte_identical": result["all_87_byte_identical"]}, sort_keys=True))


if __name__ == "__main__":
    main()
