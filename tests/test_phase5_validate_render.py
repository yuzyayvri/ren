import pytest

from scripts.phase5_render import ASSIST_WARNING, render_note
from scripts.phase5_validate import RESPONSE_SCHEMA, ResponseError, validate_response

PACKET = {"schema": "phase5-packet-v1", "case_id": "c1", "source_stage": "phase3-txl",
          "findings": [{"finding_id": "F1", "label": "WBC", "qualifier": "observed"}],
          "context": [{"evidence_id": "E1", "go_id": "GO:0002443", "name": "n",
                       "definition": "d", "rank": 1, "mode": "hybrid", "query": "q"}],
          "limitations": ["image-level-only"]}


def _resp(**over):
    base = {"schema": RESPONSE_SCHEMA, "case_id": "c1", "abstained": False,
            "claims": [{"claim_id": "C1", "text": "WBC observed, see GO:0002443.",
                        "evidence_ids": ["E1"], "finding_ids": ["F1"]}]}
    base.update(over)
    return base


def test_good_response_passes():
    assert validate_response(PACKET, _resp())["claims"][0]["claim_id"] == "C1"


def test_unknown_evidence_rejected():
    bad = _resp()
    bad["claims"][0]["evidence_ids"] = ["E9"]
    with pytest.raises(ResponseError): validate_response(PACKET, bad)


def test_unknown_finding_rejected():
    bad = _resp()
    bad["claims"][0]["finding_ids"] = ["F9"]
    with pytest.raises(ResponseError): validate_response(PACKET, bad)


def test_unsupported_go_mention_rejected():
    bad = _resp()
    bad["claims"][0]["text"] = "See GO:0000001."
    with pytest.raises(ResponseError): validate_response(PACKET, bad)


def test_empty_claims_without_abstention_rejected():
    with pytest.raises(ResponseError): validate_response(PACKET, _resp(claims=[]))


def test_abstention_with_claims_rejected():
    with pytest.raises(ResponseError):
        validate_response(PACKET, _resp(abstained=True, abstention_reason="none"))


def test_case_mismatch_rejected():
    with pytest.raises(ResponseError): validate_response(PACKET, _resp(case_id="c2"))


def test_render_is_deterministic_and_warned():
    first = render_note(PACKET, _resp())
    assert render_note(PACKET, _resp()) == first
    assert ASSIST_WARNING in first and "GO:0002443" in first and "WBC" in first


def test_abstention_renders_reason():
    out = render_note(PACKET, {"schema": RESPONSE_SCHEMA, "case_id": "c1",
                               "abstained": True, "abstention_reason": "nothing supported",
                               "claims": []})
    assert "ABSTAINED" in out and ASSIST_WARNING in out
