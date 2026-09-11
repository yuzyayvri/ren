"""Phase 5 synthesis input packet: schema, validation, deterministic serialization.

A packet is the only thing the synthesis stage may send to the LLM besides
the frozen prompt. Every fact the model may use must appear here with a
stable evidence ID; anything else the model says is, by definition,
unsupported.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

PACKET_SCHEMA = "phase5-packet-v1"

SOURCE_STAGES = ("phase2", "phase3-txl", "phase3-aml")
QUALIFIERS = ("observed", "uncertain", "absent")

FINDING_LABELS: dict[str, tuple[str, ...]] = {
    "phase2": ("neoplastic", "inflammatory", "connective", "dead", "epithelial"),
    "phase3-txl": ("WBC", "RBC", "Platelets"),
    "phase3-aml": ("blast", "non-blast"),
}

LIMITATIONS = (
    "image-level-only",
    "no-patient-linkage",
    "uncertain-typing",
    "partial-context",
    "low-count",
)

GO_RE = re.compile(r"^GO:\d{7}$")
EVIDENCE_ID_RE = re.compile(r"^E\d+$")
FINDING_ID_RE = re.compile(r"^F\d+$")


class PacketError(ValueError):
    """Raised when a synthesis packet is malformed. Always fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def build_packet(
    case_id: str,
    source_stage: str,
    findings: list[dict[str, Any]],
    context: list[dict[str, Any]],
    limitations: list[str] | tuple[str, ...] = (),
    source_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "case_id": case_id,
        "source_stage": source_stage,
        "findings": findings,
        "context": context,
        "limitations": list(limitations),
    }
    if source_ref is not None:
        packet["source_ref"] = source_ref
    return validate_packet(packet)


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(packet, dict), "packet must be an object")
    _require(packet.get("schema") == PACKET_SCHEMA, "unsupported packet schema")
    case_id = packet.get("case_id")
    _require(isinstance(case_id, str) and case_id.strip(), "case_id is required")
    stage = packet.get("source_stage")
    _require(stage in SOURCE_STAGES, f"unknown source_stage: {stage!r}")
    allowed_labels = FINDING_LABELS[stage]

    findings = packet.get("findings")
    _require(isinstance(findings, list), "findings must be a list")
    seen_findings: set[str] = set()
    for finding in findings:
        _require(isinstance(finding, dict), "finding must be an object")
        fid = finding.get("finding_id")
        _require(
            isinstance(fid, str) and FINDING_ID_RE.fullmatch(fid),
            f"bad finding_id: {fid!r}",
        )
        _require(fid not in seen_findings, f"duplicate finding_id: {fid}")
        seen_findings.add(fid)
        _require(
            finding.get("label") in allowed_labels,
            f"label {finding.get('label')!r} not allowed for {stage}",
        )
        qualifier = finding.get("qualifier", "observed")
        _require(qualifier in QUALIFIERS, f"bad qualifier: {qualifier!r}")
        for key in ("count", "confidence"):
            if key in finding and finding[key] is not None:
                _require(
                    isinstance(finding[key], (int, float))
                    and not isinstance(finding[key], bool),
                    f"{key} must be numeric",
                )
        if "confidence" in finding and finding["confidence"] is not None:
            _require(
                0.0 <= float(finding["confidence"]) <= 1.0,
                "confidence must be in [0, 1]",
            )
        if "count" in finding and finding["count"] is not None:
            _require(int(finding["count"]) >= 0, "count must be non-negative")

    context = packet.get("context")
    _require(isinstance(context, list), "context must be a list")
    seen_evidence: set[str] = set()
    for entry in context:
        _require(isinstance(entry, dict), "context entry must be an object")
        eid = entry.get("evidence_id")
        _require(
            isinstance(eid, str) and EVIDENCE_ID_RE.fullmatch(eid),
            f"bad evidence_id: {eid!r}",
        )
        _require(eid not in seen_evidence, f"duplicate evidence_id: {eid}")
        seen_evidence.add(eid)
        _require(
            isinstance(entry.get("go_id"), str)
            and GO_RE.fullmatch(entry["go_id"]),
            f"bad go_id: {entry.get('go_id')!r}",
        )
        for key in ("name", "definition"):
            _require(
                isinstance(entry.get(key), str) and entry[key].strip(),
                f"context entry {eid} is missing {key}",
            )
        _require(
            entry.get("mode") in ("symbolic", "vector", "hybrid"),
            f"context entry {eid} has bad retrieval mode",
        )
        _require(isinstance(entry.get("rank"), int) and entry["rank"] >= 1,
                  f"context entry {eid} has bad rank")

    limitations = packet.get("limitations", [])
    _require(isinstance(limitations, list), "limitations must be a list")
    for limitation in limitations:
        _require(limitation in LIMITATIONS, f"unknown limitation: {limitation!r}")

    if "source_ref" in packet:
        _require(isinstance(packet["source_ref"], dict), "source_ref must be object")

    return packet


def canonical_bytes(packet: dict[str, Any]) -> bytes:
    validated = validate_packet(packet)
    return (json.dumps(validated, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def packet_digest(packet: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(packet)).hexdigest()
