"""Phase 5 Part 2: OpenBioLLM vs MedGemma comparison under the frozen protocol.

Only the frozen development set is ever used. Only one candidate is resident
at a time. Every generation is preserved verbatim; repeats are never
cherry-picked. Selection applies the frozen safety veto before the frozen
winner order. Run `run` first (hours, unattended), then blinded human review,
then `select`. Nothing here touches frozen Part 1 files or sealed Phase 4
state, and the final acceptance set is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTO = ROOT / "protocols" / "phase5_v1"
OUT = ROOT / "artifacts" / "phase5_comparison_v1"
OUT2 = ROOT / "artifacts" / "phase5_comparison_v2"
PROTOCOLS = ('phase5_v1', 'phase5_v2')

REPEATS = 3
TIMEOUT_S = 600.0
PORT = 8080
SERVER_FLAGS = ["-c", "8192", "--host", "127.0.0.1", "--port", str(PORT)]
DEFAULT_SERVER = (
    "/nix/store/rx4ssxdipyj5ls4slf274l46k4idvqc1-llama-cpp-0.3.0/bin/llama-server"
)


class ComparisonError(RuntimeError):
    """Any broken precondition or failed generation. Recorded, never hidden."""


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_models() -> list[dict[str, Any]]:
    return json.loads((PROTO / "models.json").read_text(encoding="utf-8"))["models"]


def protocol_dir(name: str = 'phase5_v1') -> Path:
    if name not in PROTOCOLS:
        raise ComparisonError(f'unknown protocol: {name}')
    return ROOT / 'protocols' / name


def frozen_prompt(proto: str = 'phase5_v1') -> str:
    return (protocol_dir(proto) / 'prompt.md').read_text(encoding='utf-8')


def frozen_decoding(proto: str = 'phase5_v1') -> dict[str, Any]:
    return json.loads((protocol_dir(proto) / 'decoding.json').read_text(encoding='utf-8'))


def dev_cases() -> list[dict[str, Any]]:
    return json.loads((PROTO / "benchmark_dev.json").read_text(encoding="utf-8"))["cases"]


def preflight(model: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / "models" / "llm" / model["filename"]
    if not path.is_file():
        raise ComparisonError(f"model file missing: {path}")
    if path.stat().st_size != model["bytes"]:
        raise ComparisonError(f"model size mismatch: {model['filename']}")
    if sha256_path(path) != model["sha256"]:
        raise ComparisonError(f"model hash mismatch: {model['filename']}")
    return {"filename": model["filename"], "bytes": model["bytes"],
            "sha256": model["sha256"], "name": model["name"]}


def server_argv(server_bin: str, model_path: Path, ngl: int | None = None) -> list[str]:
    argv = [server_bin, "-m", str(model_path), *SERVER_FLAGS]
    if ngl is not None:
        argv += ["-ngl", str(ngl)]
    return argv


def wait_ready(base_url: str, deadline_s: float = 300.0) -> None:
    import httpx

    deadline = time.monotonic() + deadline_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{base_url}/health")
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(1.0)
    raise ComparisonError(f"server never became ready: {last}")


def effective_offload(log_text: str) -> dict[str, Any]:
    import re

    layers = re.findall(r"offloading (\d+) repeating layers to GPU", log_text)
    return {"offloaded_layers": int(layers[-1]) if layers else None}


from contextlib import contextmanager


@contextmanager
def serving(model_filename: str, server_bin: str | None, ngl: int | None,
            log_path: Path):
    from scripts.phase5_client import SynthesisClient  # noqa: F401
    binary = server_bin or os.environ.get('LLAMA_SERVER', DEFAULT_SERVER)
    if not Path(binary).is_file():
        raise ComparisonError(f'llama-server not found at {binary}; set LLAMA_SERVER')
    model_path = ROOT / 'models' / 'llm' / model_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f'http://127.0.0.1:{PORT}'
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(server_argv(binary, model_path, ngl),
                                stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            wait_ready(base_url)
            yield base_url, binary, model_path
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def run_candidate(
    model: dict[str, Any],
    transport: Any = None,
    server_bin: str | None = None,
    timeout_s: float = TIMEOUT_S,
    repeats: int = REPEATS,
    ngl: int | None = None,
    out_root: Path | None = None,
    proto: str = 'phase5_v1',
) -> dict[str, Any]:
    from scripts.phase5_client import SynthesisClient

    identity = preflight(model)
    system_prompt = frozen_prompt(proto)
    decoding = frozen_decoding(proto)
    base_url = f"http://127.0.0.1:{PORT}"
    begun = time.time()
    generations: list[dict[str, Any]] = []

    if transport is None:
        log_path = (out_root or OUT) / model["name"] / "server.log"
        with serving(model["filename"], server_bin, ngl, log_path) as (url, binary, model_path):
            client = SynthesisClient(url, model["filename"], decoding,
                                     system_prompt, timeout_s=timeout_s)
            generations = _generate(client, decoding, repeats)
        offload = effective_offload(log_path.read_text(encoding="utf-8", errors="replace"))
        server_info = {"binary": binary,
                       "argv": server_argv(binary, model_path, ngl),
                       "log": str(log_path.relative_to(ROOT)), **offload}
    else:
        client = SynthesisClient(base_url, model["filename"], decoding,
                                 system_prompt, timeout_s=timeout_s,
                                 transport=transport)
        generations = _generate(client, decoding, repeats)
        server_info = {"binary": "injected-transport", "argv": []}

    record = {
        "schema": "phase5-comparison-run-v1",
        "protocol": proto,
        "candidate": identity,
        "decoding": decoding,
        "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "repeats": repeats,
        "timeout_s": timeout_s,
        "server": server_info,
        "begun_unix": begun,
        "generations": generations,
    }
    out_path = (out_root or OUT) / model["name"] / "run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    record["run_sha256"] = sha256_path(out_path)
    return record


def _generate(client: Any, decoding: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    from scripts.phase5_client import SynthesisError
    from scripts.phase5_evaluate import case_to_packet, score_case
    from scripts.phase5_packet import canonical_bytes
    from scripts.phase5_render import render_note
    from scripts.phase5_validate import ResponseError

    generations: list[dict[str, Any]] = []
    for case in dev_cases():
        packet = case_to_packet(case)
        packet_bytes = canonical_bytes(packet).decode("utf-8")
        for repeat in range(repeats):
            started = time.monotonic()
            entry: dict[str, Any] = {
                "case_id": case["case_id"], "repeat": repeat,
                "packet_sha256": hashlib.sha256(packet_bytes.encode()).hexdigest(),
            }
            try:
                payload, parsed = client.complete(packet)
                entry["request"] = payload
                entry["raw_response"] = parsed
                entry["response_sha256"] = hashlib.sha256(
                    json.dumps(parsed, sort_keys=True).encode()).hexdigest()
                from scripts.phase5_validate import validate_response

                validated = validate_response(packet, parsed)
                entry["validated"] = validated
                entry["score"] = score_case(case, parsed)
                entry["note"] = render_note(packet, validated)
                entry["status"] = "ok"
            except (SynthesisError, ResponseError, ValueError) as exc:
                entry["status"] = "failed"
                entry["failure"] = f"{type(exc).__name__}: {exc}"
                raw = getattr(exc, "raw_content", None)
                if isinstance(raw, str) and raw:
                    entry["raw_content"] = raw
            entry["latency_s"] = time.monotonic() - started
            generations.append(entry)
    return generations


def aggregate(run: dict[str, Any]) -> dict[str, Any]:
    gens = run["generations"]
    total = len(gens)
    ok = [g for g in gens if g["status"] == "ok"]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for g in gens:
        by_case.setdefault(g["case_id"], []).append(g)
    repeatable = 0
    for items in by_case.values():
        notes = {json.dumps(i.get("validated"), sort_keys=True)
                 for i in items if i["status"] == "ok"}
        if len(items) == run["repeats"] and len(notes) == 1:
            repeatable += 1
    finding_recalls = [g["score"].get("finding_recall", 0.0) for g in ok]
    go_recalls = [g["score"].get("go_recall", 0.0) for g in ok]
    forbidden_hits = sum(len(g["score"].get("forbidden_findings_cited", []))
                         + len(g["score"].get("forbidden_claims_found", [])) for g in ok)
    latencies = [g["latency_s"] * 1000.0 for g in ok]
    return {
        "candidate": run["candidate"]["name"],
        "generations": total,
        "valid_rate": len(ok) / total if total else 0.0,
        "mean_finding_recall": sum(finding_recalls) / len(finding_recalls) if finding_recalls else 0.0,
        "mean_go_recall": sum(go_recalls) / len(go_recalls) if go_recalls else 0.0,
        "forbidden_hits": forbidden_hits,
        "repeatability": repeatable / len(by_case) if by_case else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": (sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
                           if latencies else None),
    }


def safety_veto(agg: dict[str, Any], review_verdict: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if agg["valid_rate"] < 1.0:
        reasons.append(f"invalid structures: {1.0 - agg['valid_rate']:.2f} of generations")
    if agg["forbidden_hits"] > 0:
        reasons.append(f"diagnostic overreach or forbidden findings: {agg['forbidden_hits']}")
    if review_verdict == "rejected":
        reasons.append("blinded human review rejected")
    elif review_verdict != "accepted":
        reasons.append(f"human review verdict missing or unclear: {review_verdict!r}")
    return (len(reasons) == 0, reasons)


def select(aggregates: dict[str, dict[str, Any]],
           verdicts: dict[str, str]) -> dict[str, Any]:
    vetoed = {name: safety_veto(agg, verdicts.get(name, "missing"))
              for name, agg in aggregates.items()}
    eligible = [name for name, (ok, _) in vetoed.items() if ok]
    if not eligible:
        return {"winner": None, "eligible": [],
                "vetoes": {n: r for n, (_, r) in vetoed.items()},
                "status": "stopped_neither_eligible"}
    ranked = sorted(
        eligible,
        key=lambda n: (aggregates[n]["mean_finding_recall"]
                       + aggregates[n]["mean_go_recall"],
                       aggregates[n]["repeatability"]),
        reverse=True,
    )
    return {"winner": ranked[0], "eligible": ranked,
            "vetoes": {n: r for n, (_, r) in vetoed.items()},
            "status": "selected"}


def append_lifecycle(event: str, extra: dict[str, Any],
                     ledger: Path | None = None) -> dict[str, Any]:
    if event not in ("part1_superseded", "phase5_v2_frozen", "part2_selected", "part2_stopped"):
        raise ComparisonError(f"illegal lifecycle event: {event}")
    ledger = ledger or PROTO / "lifecycle.jsonl"
    entry = {"event": event, **extra}
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def package_review(runs: dict[str, dict[str, Any]], order: list[str], out_root: Path | None = None) -> dict[str, Path]:
    labels = ["A", "B"]
    mapping = {label: name for label, name in zip(labels, order)}
    review_dir = (out_root or OUT) / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        run = runs[mapping[label]]
        notes = []
        for case in dev_cases():
            packet_notes = [g["note"] for g in run["generations"]
                            if g["case_id"] == case["case_id"] and g["status"] == "ok"]
            notes.append({"case_id": case["case_id"],
                          "notes": packet_notes,
                          "packet": case})
        (review_dir / f"notes_{label}.json").write_text(
            json.dumps(notes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mapping_path = review_dir / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return {"dir": review_dir, "mapping": mapping_path}


def smoke(model: dict[str, Any], proto: str = "phase5_v2",
            server_bin: str | None = None, timeout_s: float = TIMEOUT_S,
            out_root: Path | None = None,
            fixture_path: Path | None = None) -> dict[str, Any]:
    """One synthetic-packet generation per candidate through the live server.

    Mechanism check only: proves the enforcement path works end-to-end.
    The record is marked selection-ineligible and scoring/review commands
    never read it. A failed smoke stops the run before hours are burned.
    """
    from scripts.phase5_client import SynthesisClient, SynthesisError
    from scripts.phase5_evaluate import case_to_packet
    from scripts.phase5_packet import canonical_bytes
    from scripts.phase5_validate import ResponseError, validate_response

    out = out_root or OUT2
    fixture_path = fixture_path or protocol_dir(proto) / "smoke_packet.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("selection_eligible", True):
        raise ComparisonError("smoke fixture must be selection-ineligible")
    identity = preflight(model)
    system_prompt = frozen_prompt(proto)
    decoding = frozen_decoding(proto)
    packet = case_to_packet({k: v for k, v in fixture.items() if k != "selection_eligible"})
    packet_bytes = canonical_bytes(packet).decode("utf-8")
    started = time.monotonic()
    entry: dict[str, Any] = {
        "case_id": packet["case_id"],
        "packet_sha256": hashlib.sha256(packet_bytes.encode()).hexdigest(),
        "fixture_sha256": hashlib.sha256(
            fixture_path.read_bytes()).hexdigest(),
        "selection_eligible": False,
    }
    log_path = out / "smoke" / f"{model['name']}.log"
    with serving(model["filename"], server_bin, None, log_path) as (url, binary, _):
        client = SynthesisClient(url, model["filename"], decoding,
                                 system_prompt, timeout_s=timeout_s)
        try:
            payload, parsed = client.complete(packet)
            entry["request"] = payload
            entry["raw_response"] = parsed
            validated = validate_response(packet, parsed)
            entry["validated"] = validated
            entry["status"] = "ok"
        except (SynthesisError, ResponseError, ValueError) as exc:
            entry["status"] = "failed"
            entry["failure"] = f"{type(exc).__name__}: {exc}"
            raw = getattr(exc, "raw_content", None)
            if isinstance(raw, str) and raw:
                entry["raw_content"] = raw
    entry["latency_s"] = time.monotonic() - started
    entry["server"] = {"binary": binary}
    record = {"schema": "phase5-smoke-v1", "protocol": proto,
              "candidate": identity, "smoke": entry}
    out_path = out / "smoke" / f"{model['name']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--candidate", choices=["openbiollm-llama3-8b", "medgemma-1.5-4b-it", "all"],
                         default="all")
    run_cmd.add_argument("--repeats", type=int, default=REPEATS)
    run_cmd.add_argument("--timeout", type=float, default=TIMEOUT_S)
    run_cmd.add_argument("--ngl", type=int, default=None)
    run_cmd.add_argument("--server-bin", type=str, default=None)
    run_cmd.add_argument("--protocol", choices=list(PROTOCOLS), default="phase5_v1")
    run_cmd.add_argument("--out-root", type=str, default=None)
    score_cmd = sub.add_parser("score")
    score_cmd.add_argument("--out-root", type=str, default=None)
    review_cmd = sub.add_parser("package-review")
    review_cmd.add_argument("--out-root", type=str, default=None)
    smoke_cmd = sub.add_parser("smoke")
    smoke_cmd.add_argument("--candidate", choices=["openbiollm-llama3-8b", "medgemma-1.5-4b-it"],
                           required=True)
    smoke_cmd.add_argument("--protocol", choices=list(PROTOCOLS), default="phase5_v2")
    smoke_cmd.add_argument("--timeout", type=float, default=TIMEOUT_S)
    smoke_cmd.add_argument("--server-bin", type=str, default=None)
    smoke_cmd.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()

    if args.command == "run":
        models = {m["name"]: m for m in frozen_models()}
        names = list(models) if args.candidate == "all" else [args.candidate]
        out = Path(args.out_root) if args.out_root else (OUT2 if args.protocol == "phase5_v2" else OUT)
        for name in names:
            record = run_candidate(models[name], timeout_s=args.timeout,
                                   repeats=args.repeats, proto=args.protocol,
                                   out_root=out,
                                   server_bin=args.server_bin, ngl=args.ngl)
            agg = aggregate(record)
            print(json.dumps({"candidate": name, "aggregate": agg}, indent=2))
        return 0
    if args.command == "smoke":
        models = {m["name"]: m for m in frozen_models()}
        out = Path(args.out_root) if args.out_root else OUT2
        record = smoke(models[args.candidate], proto=args.protocol,
                       timeout_s=args.timeout, server_bin=args.server_bin,
                       out_root=out)
        print(json.dumps({"candidate": args.candidate,
                          "status": record["smoke"]["status"]}, indent=2))
        return 0 if record["smoke"]["status"] == "ok" else 1
    if args.command == "score":
        out = Path(args.out_root) if args.out_root else OUT
        result = {}
        for run_path in sorted(out.glob("*/run.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            result[run["candidate"]["name"]] = aggregate(run)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "package-review":
        out = Path(args.out_root) if args.out_root else OUT
        runs = {}
        for run_path in sorted(out.glob("*/run.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            runs[run["candidate"]["name"]] = run
        if set(runs) != {"openbiollm-llama3-8b", "medgemma-1.5-4b-it"}:
            raise ComparisonError("both candidate runs must exist before packaging review")
        order = sorted(runs)
        paths = package_review(runs, order)
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
