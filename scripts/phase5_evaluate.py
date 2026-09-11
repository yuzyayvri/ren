"""Phase 5 machine-checkable scoring of validated structured outputs.

Scoring never touches the LLM. It compares a validated response against the
frozen expectations of one benchmark case: required findings, required GO
terms, forbidden findings, forbidden diagnostic claims, and abstention.
"""

from __future__ import annotations

from typing import Any


def case_to_packet(case: dict[str, Any]) -> dict[str, Any]:
    from scripts.phase5_packet import build_packet

    return build_packet(
        case_id=case["case_id"],
        source_stage=case["source_stage"],
        findings=case["findings"],
        context=case["context"],
        limitations=case.get("limitations", []),
        source_ref=case.get("source_ref"),
    )


def score_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    from scripts.phase5_validate import validate_response

    packet = case_to_packet(case)
    try:
        validated = validate_response(packet, response)
    except ValueError as exc:
        return {"valid_structure": False, "structure_error": str(exc), "pass": False}

    evidence_go = {e["evidence_id"]: e["go_id"] for e in packet["context"]}
    finding_label = {f["finding_id"]: f["label"] for f in packet["findings"]}

    if case.get("expect_abstention"):
        abstained = validated.get("abstained") is True
        return {
            "valid_structure": True,
            "abstained": validated.get("abstained") is True,
            "correct_abstention": abstained,
            "pass": abstained,
        }

    if validated.get("abstained"):
        return {"valid_structure": True, "abstained": True, "pass": False,
                "reason": "abstained on a non-abstention case"}

    cited_findings = {fid for c in validated["claims"] for fid in c["finding_ids"]}
    cited_go = {evidence_go[eid] for c in validated["claims"] for eid in c["evidence_ids"]}
    cited_labels = {finding_label[fid] for fid in cited_findings}
    texts = " ".join(c["text"] for c in validated["claims"]).lower()

    required_labels = set(case.get("required_finding_labels", []))
    required_go = set(case.get("required_go_ids", []))
    forbidden_findings = set(case.get("forbidden_findings", []))
    forbidden_claims = [s.lower() for s in case.get("forbidden_claims", [])]

    missing_labels = sorted(required_labels - cited_labels)
    missing_go = sorted(required_go - cited_go)
    hit_forbidden_findings = sorted(forbidden_findings & cited_labels)
    hit_forbidden_claims = sorted({s for s in forbidden_claims if s in texts})

    finding_recall = (
        len(required_labels & cited_labels) / len(required_labels)
        if required_labels else 1.0
    )
    go_recall = (
        len(required_go & cited_go) / len(required_go) if required_go else 1.0
    )

    passed = not (missing_labels or missing_go or hit_forbidden_findings or hit_forbidden_claims)
    return {
        "valid_structure": True,
        "abstained": False,
        "finding_recall": finding_recall,
        "go_recall": go_recall,
        "missing_finding_labels": missing_labels,
        "missing_go_ids": missing_go,
        "forbidden_findings_cited": hit_forbidden_findings,
        "forbidden_claims_found": hit_forbidden_claims,
        "unsupported_finding_count": 0,
        "hallucinated_go_count": 0,
        "pass": passed,
    }


def score_suite(cases: list[dict[str, Any]], responses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = {case["case_id"]: score_case(case, responses[case["case_id"]]) for case in cases}
    passed = sum(1 for r in results.values() if r.get("pass"))
    return {"total": len(cases), "passed": passed, "results": results}
