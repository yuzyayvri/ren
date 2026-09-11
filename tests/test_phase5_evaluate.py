import json
from pathlib import Path

from scripts.phase5_evaluate import case_to_packet, score_case

DEV = Path("protocols/phase5_v1/benchmark_dev.json")


def _case(case_id):
    cases = json.loads(DEV.read_text())["cases"]
    return next(c for c in cases if c["case_id"] == case_id)


def _good_response(case):
    packet = case_to_packet(case)
    if case.get("expect_abstention"):
        return {"schema": "phase5-response-v1", "case_id": case["case_id"],
                "abstained": True, "abstention_reason": "no supported finding", "claims": []}
    claims = []
    for i, finding in enumerate(packet["findings"]):
        ev = packet["context"][i % len(packet["context"])] if packet["context"] else None
        if ev is None:
            continue
        claims.append({"claim_id": f"C{i+1}",
                       "text": f"{finding['label']} observed with {ev['go_id']} {ev['name']}.",
                       "evidence_ids": [ev["evidence_id"]],
                       "finding_ids": [finding["finding_id"]]})
    return {"schema": "phase5-response-v1", "case_id": case["case_id"],
            "abstained": False, "claims": claims}


def test_all_dev_cases_score_clean_with_supporting_response():
    for case in json.loads(DEV.read_text())["cases"]:
        result = score_case(case, _good_response(case))
        assert result["pass"], (case["case_id"], result)


def test_missing_required_go_fails():
    case = _case("dev-txl-wbc-single")
    resp = {"schema": "phase5-response-v1", "case_id": case["case_id"], "abstained": False,
            "claims": [{"claim_id": "C1", "text": "WBC observed with GO:0050900.",
                        "evidence_ids": ["E2"], "finding_ids": ["F1"]}]}
    result = score_case(case, resp)
    assert not result["pass"] and result["missing_go_ids"] == ["GO:0002443"]


def test_forbidden_claim_fails():
    case = _case("dev-txl-wbc-single")
    resp = {"schema": "phase5-response-v1", "case_id": case["case_id"], "abstained": False,
            "claims": [{"claim_id": "C1", "text": "WBC observed; diagnosis is certain, see GO:0002443.",
                        "evidence_ids": ["E1"], "finding_ids": ["F1"]}]}
    result = score_case(case, resp)
    assert not result["pass"] and "diagnos" in result["forbidden_claims_found"]


def test_wrong_abstention_fails():
    case = _case("dev-txl-wbc-single")
    resp = {"schema": "phase5-response-v1", "case_id": case["case_id"],
            "abstained": True, "abstention_reason": "unsure", "claims": []}
    assert not score_case(case, resp)["pass"]


def test_malformed_response_scores_invalid():
    case = _case("dev-txl-wbc-single")
    result = score_case(case, {"schema": "nope"})
    assert result["valid_structure"] is False and not result["pass"]
