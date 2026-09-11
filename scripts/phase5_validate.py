"""Phase 5 structured-response validation. Every claim must cite packet evidence."""

from __future__ import annotations

import re
from typing import Any

from scripts.phase5_packet import GO_RE

RESPONSE_SCHEMA = "phase5-response-v1"
GO_MENTION_RE = re.compile(r"GO:\d{7}")


class ResponseError(ValueError):
    """A structured response that must not become a note. Fail closed."""


def _fail(message: str) -> None:
    raise ResponseError(message)


def validate_response(packet: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        _fail("response must be an object")
    if response.get("schema") != RESPONSE_SCHEMA:
        _fail("unsupported response schema")
    if response.get("case_id") != packet.get("case_id"):
        _fail("response case_id does not match packet")

    finding_ids = {f["finding_id"] for f in packet.get("findings", [])}
    evidence_ids = {e["evidence_id"] for e in packet.get("context", [])}
    packet_go_ids = {e["go_id"] for e in packet.get("context", [])}

    abstained = response.get("abstained")
    if not isinstance(abstained, bool):
        _fail("abstained must be boolean")
    claims = response.get("claims")
    if not isinstance(claims, list):
        _fail("claims must be a list")

    if abstained:
        if claims:
            _fail("abstained response must carry zero claims")
        reason = response.get("abstention_reason")
        if not isinstance(reason, str) or not reason.strip():
            _fail("abstained response needs an abstention_reason")
        return response

    if not claims:
        _fail("non-abstained response must carry at least one claim")

    seen_claims: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            _fail("claim must be an object")
        cid = claim.get("claim_id")
        if not isinstance(cid, str) or not cid.strip() or cid in seen_claims:
            _fail(f"bad or duplicate claim_id: {cid!r}")
        seen_claims.add(cid)
        text = claim.get("text")
        if not isinstance(text, str) or not text.strip():
            _fail(f"claim {cid} has empty text")
        cited_evidence = claim.get("evidence_ids")
        cited_findings = claim.get("finding_ids")
        if not isinstance(cited_evidence, list) or not cited_evidence:
            _fail(f"claim {cid} cites no evidence")
        if not isinstance(cited_findings, list) or not cited_findings:
            _fail(f"claim {cid} cites no findings")
        for eid in cited_evidence:
            if eid not in evidence_ids:
                _fail(f"claim {cid} cites unknown evidence {eid!r}")
        for fid in cited_findings:
            if fid not in finding_ids:
                _fail(f"claim {cid} cites unknown finding {fid!r}")
        for go_id in GO_MENTION_RE.findall(text):
            if not GO_RE.fullmatch(go_id) or go_id not in packet_go_ids:
                _fail(f"claim {cid} mentions unsupported GO identifier {go_id}")
    return response
