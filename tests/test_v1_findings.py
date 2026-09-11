import json

import pytest

from scripts import v1_findings as findings


def _vision():
    return {
        "specimen_sha256": "abc123",
        "models": {"detector": {"sha256": "d1"}, "head": {"sha256": "h1"}},
        "findings": [
            {"finding_id": "F1", "label": "WBC", "confidence": 0.9,
             "region": {"box_xyxy": [1, 2, 3, 4]}, "source": "machine",
             "specimen_id": "s1"},
            {"finding_id": "F2", "label": "RBC", "confidence": 0.8,
             "region": {"box_xyxy": [5, 6, 7, 8]}, "source": "machine",
             "specimen_id": "s1"},
        ],
    }


def _write_vision(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vision.json").write_text(json.dumps(_vision()))
    return directory


def test_enrich_and_packet_adapter():
    enriched = findings.enrich_findings(_vision(), run_id="run1")
    assert all(f["run_id"] == "run1" and f["review_state"] == "unreviewed" for f in enriched)
    assert enriched[0]["model"] == {"detector_sha256": "d1", "head_sha256": "h1"}
    packet = findings.to_packet_findings(enriched)
    assert set(packet[0]) <= {"finding_id", "label", "count", "confidence", "qualifier"}
    from scripts.phase5_packet import validate_packet

    validate_packet({"schema": "phase5-packet-v1", "case_id": "c", "source_stage": "phase3-txl",
                     "findings": packet, "context": [], "limitations": []})


def test_confirm_correct_reject_lifecycle(tmp_path):
    directory = _write_vision(tmp_path / "spec")
    before = findings.sha256_file(directory / "vision.json")
    findings.record_review(directory, "F1", "confirm", reviewer="r1")
    findings.record_review(directory, "F2", "correct",
                           changes={"label": "WBC", "qualifier": "uncertain"},
                           reviewer="r1", reason="looks off")
    assert findings.sha256_file(directory / "vision.json") == before
    effective = {f["finding_id"]: f for f in findings.effective_findings(directory)}
    assert effective["F1"]["review_state"] == "confirmed"
    assert effective["F2"]["review_state"] == "corrected"
    assert effective["F2"]["label"] == "WBC" and effective["F2"]["review_id"]
    assert [f["finding_id"] for f in findings.confirmed_findings(directory)] == ["F1", "F2"]


def test_rejected_excluded_but_auditable(tmp_path):
    directory = _write_vision(tmp_path / "spec")
    findings.record_review(directory, "F1", "reject", reviewer="r1")
    assert findings.confirmed_findings(directory) == []
    assert len(findings.read_reviews(directory)) == 1


def test_invalid_transitions(tmp_path):
    directory = _write_vision(tmp_path / "spec")
    with pytest.raises(findings.FindingError):
        findings.record_review(directory, "F9", "confirm")
    with pytest.raises(findings.FindingError):
        findings.record_review(directory, "F1", "bogus")
    with pytest.raises(findings.FindingError):
        findings.record_review(directory, "F1", "correct")
    with pytest.raises(findings.FindingError):
        findings.record_review(directory, "F1", "correct", changes={"specimen_id": "x"})
