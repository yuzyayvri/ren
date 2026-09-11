"""Phase 5 Part 3 one-time final acceptance runner (frozen MedGemma winner).

Opens the sealed final set exactly once: aborts unless every verifier is
green and no final-open event exists in the ledger. Runs every final case
through the production pinned path (frozen decoding, winner model, managed
server), scores every repeat with the frozen scorer, and seals the report
plus the accept/stop outcome. Any gate miss records part3_stopped; reruns
need review plus a new ledger entry, never threshold changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V1 = ROOT / "protocols" / "phase5_v1"
V2 = ROOT / "protocols" / "phase5_v2"
OUT = ROOT / "artifacts" / "phase5_final_v1"
FINAL_BENCHMARK = V1 / "benchmark_final.json"
LEDGER = V1 / "lifecycle.jsonl"
REPEATS = 3
TIMEOUT_S = 600.0


class FinalError(RuntimeError):
    """A broken precondition or gate. Recorded, never bypassed."""


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ledger_events() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()]


def check_preconditions() -> dict[str, Any]:
    from scripts.phase4_snapshot_reconciliation import verify_intended_state
    from scripts.phase5_protocol import verify as verify_v1
    from scripts.phase5_protocol import verify_v2

    verify_v1()
    verify_v2()
    bound = verify_intended_state()
    events = [e.get("event") for e in ledger_events()]
    if any(e == "part3_opened_final" for e in ledger_events()):
        raise FinalError("final set was already opened; rerun needs review")
    for required in ("phase5_v2_frozen", "part2_selected"):
        if required not in events:
            raise FinalError(f"lifecycle missing required event: {required}")
    manifest = json.loads((V2 / "freeze_manifest.json").read_text(encoding="utf-8"))
    return {"v1_verified": True, "v2_verified": True,
            "bound_code": bound["final_code_sha256"],
            "v2_manifest_sha256": sha256_path(V2 / "freeze_manifest.json"),
            "v1_manifest_sha256": manifest["supersedes"]["phase5_freeze_v1_manifest_sha256"]}


def append_event(event: str, extra: dict[str, Any]) -> dict[str, Any]:
    if event not in ("part3_opened_final", "part3_accepted", "part3_stopped"):
        raise FinalError(f"illegal lifecycle event: {event}")
    entry = {"event": event, **extra}
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def final_cases() -> list[dict[str, Any]]:
    return json.loads(FINAL_BENCHMARK.read_text(encoding="utf-8"))["cases"]


def run_final(out_root: Path | None = None,
              server_bin: str | None = None,
              cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from scripts.phase5_compare import serving
    from scripts.phase5_evaluate import case_to_packet, score_case
    from scripts.phase5_synthesize import SynthesisProductionError, preflight_winner
    from scripts.phase5_synthesize import synthesize_packet as _synth

    out = out_root or OUT
    preconditions = check_preconditions()
    append_event("part3_opened_final", {
        "v2_manifest_sha256": preconditions["v2_manifest_sha256"],
        "note": "Final set opened exactly once; no tuning follows.",
    })
    model = preflight_winner()
    log_path = out / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "phase5-final-v1", "protocol": "phase5_v2",
        "candidate": model["name"], "preconditions": preconditions,
        "generations": [],
    }
    with serving(model["filename"], server_bin, None, log_path) as (url, _, __):
        for case in cases if cases is not None else final_cases():
            from scripts.phase5_packet import canonical_bytes

            packet = case_to_packet(case)
            packet_bytes = canonical_bytes(packet).decode("utf-8")
            for repeat in range(REPEATS):
                started = time.monotonic()
                entry: dict[str, Any] = {
                    "case_id": case["case_id"], "repeat": repeat,
                    "packet_sha256": hashlib.sha256(packet_bytes.encode()).hexdigest(),
                }
                try:
                    single = _synth(packet, url, timeout_s=TIMEOUT_S)
                except SynthesisProductionError as exc:
                    entry["status"] = "failed"
                    entry["failure"] = f"{type(exc).__name__}: {exc}"
                    entry["latency_s"] = time.monotonic() - started
                    report["generations"].append(entry)
                    continue
                entry.update({k: single[k] for k in
                              ("request", "raw_response", "validated", "note",
                               "raw_content", "failure")
                              if k in single})
                if single.get("status") != "ok":
                    entry["status"] = "failed"
                else:
                    try:
                        entry["score"] = score_case(case, single["validated"])
                        entry["status"] = "ok"
                    except ValueError as exc:
                        entry["status"] = "failed"
                        entry["failure"] = f"{type(exc).__name__}: {exc}"
                entry["latency_s"] = time.monotonic() - started
                report["generations"].append(entry)
    return seal_report(report, out)


def seal_report(report: dict[str, Any], out: Path) -> dict[str, Any]:
    results = {}
    for case_id in {g["case_id"] for g in report["generations"]}:
        items = [g for g in report["generations"] if g["case_id"] == case_id]
        results[case_id] = (
            len(items) == REPEATS
            and all(g["status"] == "ok" and g["score"].get("pass") for g in items)
        )
    report["per_case_pass"] = results
    report["all_pass"] = bool(results) and all(results.values())
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256_path(out / "final_report.json")
    if report["all_pass"]:
        append_event("part3_accepted", {
            "report_sha256": report["report_sha256"],
            "cases": len(results),
        })
    else:
        append_event("part3_stopped", {
            "report_sha256": report["report_sha256"],
            "failing_cases": sorted(c for c, ok in results.items() if not ok),
        })
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preconditions")
    pre.add_argument("--out-root", type=str, default=None)
    run = sub.add_parser("run")
    run.add_argument("--out-root", type=str, default=None)
    run.add_argument("--server-bin", type=str, default=None)
    args = parser.parse_args()
    if args.command == "preconditions":
        print(json.dumps(check_preconditions(), indent=2, sort_keys=True))
        return 0
    out = Path(args.out_root) if args.out_root else OUT
    report = run_final(out_root=out, server_bin=args.server_bin)
    print(json.dumps({"all_pass": report["all_pass"],
                      "per_case_pass": report["per_case_pass"]}, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
