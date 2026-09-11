"""Phase 5 Part 3 production synthesis path (frozen MedGemma winner).

Two input modes share one fail-closed pipeline:

- pinned packet JSON: a fully built packet (used by final acceptance);
- finding labels: observed findings plus per-finding query strings, with
  context retrieved live through the sanctioned bound loader (used for
  real synthesis and integration rehearsals).

Only validated outputs are rendered. Anything else aborts the case with
a recorded failure and no note. Frozen model, prompt, decoding, schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTO = ROOT / "protocols" / "phase5_v2"
V1 = ROOT / "protocols" / "phase5_v1"
OUT = ROOT / "artifacts" / "phase5_synthesis"
WINNER = "medgemma-1.5-4b-it"
RETRIEVAL_K = 5


class SynthesisProductionError(RuntimeError):
    """Any production failure. The case aborts; no note is rendered."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_loopback(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SynthesisProductionError(f"non-loopback server URL refused: {url}")
    return url


def offline_evidence() -> dict[str, Any]:
    try:
        socket.create_connection(("198.51.100.1", 9), timeout=0.5)
    except (OSError, RuntimeError) as exc:
        return {"network_blocked": True, "exception": type(exc).__name__}
    return {"network_blocked": False, "exception": None}


def preflight_winner() -> dict[str, Any]:
    from scripts.phase5_compare import frozen_models, sha256_path

    models = {m["name"]: m for m in frozen_models()}
    model = models[WINNER]
    path = ROOT / "models" / "llm" / model["filename"]
    if not path.is_file() or path.stat().st_size != model["bytes"]:
        raise SynthesisProductionError("winner model file mismatch")
    if sha256_path(path) != model["sha256"]:
        raise SynthesisProductionError("winner model hash mismatch")
    return model


def retrieval_provenance() -> dict[str, Any]:
    from scripts.phase4_snapshot_reconciliation import verify_intended_state

    proof = verify_intended_state()
    return {
        "corpus": "phase4-protocol-v3",
        "corpus_rows": 38245,
        "final_code_sha256": proof["final_code_sha256"],
    }


def retrieve_context(findings: list[dict[str, Any]],
                     queries: dict[str, str]) -> list[dict[str, Any]]:
    from scripts.phase4_snapshot_reconciliation import load_bound_query

    query_engine = load_bound_query()
    import sqlite3

    from scripts.phase4_v3_common import V3 as _V3

    db = sqlite3.connect(f"file:{_V3 / 'ontology.sqlite'}?mode=ro", uri=True)
    try:
        seen: dict[str, dict[str, Any]] = {}
        for finding in findings:
            query_text = queries.get(finding["finding_id"], finding["label"])
            for rank, go_id in enumerate(
                query_engine.retrieve(query_text, mode="hybrid", k=RETRIEVAL_K), start=1
            ):
                if go_id in seen:
                    continue
                row = db.execute(
                    "SELECT name, definition FROM terms WHERE id=? AND obsolete=0",
                    (go_id,),
                ).fetchone()
                if row is None:
                    raise SynthesisProductionError(f"retrieved unknown term: {go_id}")
                seen[go_id] = {
                    "go_id": go_id, "name": row[0], "definition": row[1],
                    "rank": rank, "mode": "hybrid", "query": query_text,
                }
        context = []
        for number, entry in enumerate(seen.values(), start=1):
            entry["evidence_id"] = f"E{number}"
            context.append(entry)
        return context
    finally:
        db.close()


def build_production_packet(case_id: str, source_stage: str,
                            findings: list[dict[str, Any]],
                            queries: dict[str, str],
                            limitations: list[str]) -> dict[str, Any]:
    from scripts.phase5_packet import build_packet, validate_packet

    validate_packet({
        "schema": "phase5-packet-v1", "case_id": case_id,
        "source_stage": source_stage, "findings": findings,
        "context": [], "limitations": limitations,
    })
    return build_packet(case_id, source_stage, findings,
                        retrieve_context(findings, queries), limitations)


def synthesize_packet(packet: dict[str, Any], base_url: str,
                      timeout_s: float = 600.0,
                      transport: Any = None) -> dict[str, Any]:
    from scripts.phase5_client import SynthesisClient, SynthesisError
    from scripts.phase5_render import render_note
    from scripts.phase5_validate import ResponseError, validate_response

    require_loopback(base_url)
    model = preflight_winner()
    system_prompt = (PROTO / "prompt.md").read_text(encoding="utf-8")
    decoding = json.loads((PROTO / "decoding.json").read_text(encoding="utf-8"))
    offline = offline_evidence()
    if not offline["network_blocked"]:
        raise SynthesisProductionError("offline gate failed: network reachable")
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema": "phase5-production-v1",
        "model": {"name": model["name"], "filename": model["filename"],
                  "sha256": model["sha256"]},
        "prompt_sha256": sha256_bytes(system_prompt.encode()),
        "offline": offline,
        "retrieval_provenance": retrieval_provenance(),
    }
    try:
        client = SynthesisClient(base_url, model["filename"], decoding,
                                 system_prompt, timeout_s=timeout_s,
                                 transport=transport)
        payload, parsed = client.complete(packet)
        result["request"] = payload
        result["raw_response"] = parsed
        validated = validate_response(packet, parsed)
        result["validated"] = validated
        result["note"] = render_note(packet, validated)
        result["status"] = "ok"
    except (SynthesisError, ResponseError, ValueError) as exc:
        result["status"] = "failed"
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raw = getattr(exc, "raw_content", None)
        if isinstance(raw, str) and raw:
            result["raw_content"] = raw
    result["latency_s"] = time.monotonic() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    packet_cmd = sub.add_parser("from-packet")
    packet_cmd.add_argument("--packet", type=Path, required=True)
    packet_cmd.add_argument("--server", default="http://127.0.0.1:8080")
    packet_cmd.add_argument("--out", type=Path, required=True)
    packet_cmd.add_argument("--timeout", type=float, default=600.0)
    labels_cmd = sub.add_parser("from-labels")
    labels_cmd.add_argument("--findings", type=Path, required=True)
    labels_cmd.add_argument("--case-id", required=True)
    labels_cmd.add_argument("--source-stage", required=True)
    labels_cmd.add_argument("--server", default="http://127.0.0.1:8080")
    labels_cmd.add_argument("--out", type=Path, required=True)
    labels_cmd.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.command == "from-packet":
        from scripts.phase5_packet import validate_packet

        packet = validate_packet(json.loads(args.packet.read_text(encoding="utf-8")))
    else:
        spec = json.loads(args.findings.read_text(encoding="utf-8"))
        packet = build_production_packet(
            args.case_id, spec.get("source_stage") or args.source_stage,
            spec["findings"], spec.get("queries", {}),
            spec.get("limitations", []),
        )
    result = synthesize_packet(packet, args.server, timeout_s=args.timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "failure": result.get("failure")}, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
