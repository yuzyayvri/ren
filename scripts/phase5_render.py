"""Deterministic rendering of a validated Phase 5 response into a review note."""

from __future__ import annotations

from typing import Any

ASSIST_WARNING = (
    "Diagnostic-assist only: this note was generated locally from supplied "
    "findings and retrieved context. Every statement must be reviewed and "
    "signed off by a qualified human; nothing here makes an autonomous "
    "clinical call."
)


def render_note(packet: dict[str, Any], response: dict[str, Any]) -> str:
    lines = [f"Case {packet['case_id']} — synthesis review note", ""]
    if response.get("abstained"):
        lines.append("ABSTAINED: no supported finding to report.")
        lines.append(f"Reason: {response['abstention_reason']}")
    else:
        evidence = {e["evidence_id"]: e for e in packet.get("context", [])}
        findings = {f["finding_id"]: f for f in packet.get("findings", [])}
        for claim in response["claims"]:
            cites = ", ".join(
                f"{eid} ({evidence[eid]['go_id']})" for eid in claim["evidence_ids"]
            )
            labels = ", ".join(
                findings[fid]["label"] for fid in claim["finding_ids"]
            )
            lines.append(f"- [{claim['claim_id']}] {claim['text']}")
            lines.append(f"  findings: {labels}; evidence: {cites}")
    limitations = packet.get("limitations", [])
    if limitations:
        lines.append("")
        lines.append("Limitations: " + "; ".join(limitations))
    lines.append("")
    lines.append(ASSIST_WARNING)
    return "\n".join(lines) + "\n"
