"""Generic official verifier for a canonical stateful production run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from final_phase2_evaluation import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_final_protocol.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "run",
        "manifest",
        "config",
        "images",
        "embeddings",
        "segmentation",
        "validity",
        "standardizer",
        "classifier",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rehearsal-spec", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--types", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    command = [sys.executable, str(CLI), "verify"]
    for name in (
        "run",
        "manifest",
        "config",
        "images",
        "embeddings",
        "segmentation",
        "validity",
        "standardizer",
        "classifier",
    ):
        command += [f"--{name}", str(getattr(args, name))]
    if args.masks is not None:
        command += ["--masks", str(args.masks)]
    if args.types is not None:
        command += ["--types", str(args.types)]
    verified = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if verified.returncode:
        print(verified.stderr, file=sys.stderr)
        return verified.returncode
    packet = json.loads(verified.stdout)
    state = json.loads((args.run / "state.json").read_text())
    result = {
        "schema": "phase2-production-readiness-verification-v2",
        "state": state["state"],
        "coverage": packet["coverage"],
        "prediction_digest": packet["digest"],
        "manifest_sha256": sha256(args.manifest),
        "canonical_verifier": "pass",
    }
    if args.rehearsal_spec:
        specification = json.loads(args.rehearsal_spec.read_text())
        if (
            state["evaluation_role"] != "development_rehearsal"
            or packet["coverage"] != specification["coverage"]
            or packet["digest"] != specification["expected_prediction_digest"]
        ):
            raise RuntimeError(
                "development rehearsal packet differs from specification"
            )
        if args.report:
            report = json.loads(args.report.read_text())
            metrics = report["metrics"]
            measured = {
                "detection_f1": metrics["detection"]["f1"],
                "binary_pq": metrics["binary_pq"],
                "end_to_end_macro_f1": metrics["macro_f1"],
                "dead_recall": metrics["classes"]["4"]["recall"],
                "dead_f1": metrics["classes"]["4"]["f1"],
            }
            if (
                measured != specification["expected_metrics"]
                or report["ignored_only_accounting"]
                != specification["expected_ignored_only_accounting"]
            ):
                raise RuntimeError(
                    "development rehearsal score differs from specification"
                )
            result["report_sha256"] = sha256(args.report)
        result["rehearsal_spec_sha256"] = sha256(args.rehearsal_spec)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
