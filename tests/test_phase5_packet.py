import pytest

from scripts.phase5_packet import (
    PacketError,
    canonical_bytes,
    packet_digest,
    validate_packet,
)

FINDINGS = [{"finding_id": "F1", "label": "WBC", "count": 3, "confidence": 0.9, "qualifier": "observed"}]
CONTEXT = [{"evidence_id": "E1", "go_id": "GO:0002443", "name": "leukocyte mediated immunity",
            "definition": "Immune response by a leukocyte.", "rank": 1, "mode": "hybrid", "query": "leukocyte"}]


def _packet(**over):
    base = {"schema": "phase5-packet-v1", "case_id": "c1", "source_stage": "phase3-txl", "findings": FINDINGS,
            "context": CONTEXT, "limitations": []}
    base.update(over)
    return base


def test_valid_packet_passes():
    assert validate_packet(_packet())["schema"] == "phase5-packet-v1"


def test_bad_schema_rejected():
    with pytest.raises(PacketError): validate_packet(_packet() | {"schema": "nope"})


def test_unknown_stage_rejected():
    with pytest.raises(PacketError): validate_packet(_packet(source_stage="phase9"))


def test_wrong_taxonomy_label_rejected():
    bad = [{"finding_id": "F1", "label": "neoplastic", "qualifier": "observed"}]
    with pytest.raises(PacketError): validate_packet(_packet(findings=bad))


def test_confidence_bounds_enforced():
    bad = [{"finding_id": "F1", "label": "WBC", "confidence": 1.5, "qualifier": "observed"}]
    with pytest.raises(PacketError): validate_packet(_packet(findings=bad))


def test_duplicate_ids_rejected():
    dup = FINDINGS + [{"finding_id": "F1", "label": "RBC", "qualifier": "observed"}]
    with pytest.raises(PacketError): validate_packet(_packet(findings=dup))


def test_bad_go_id_rejected():
    ctx = [CONTEXT[0] | {"go_id": "GO:999"}]
    with pytest.raises(PacketError): validate_packet(_packet(context=ctx))


def test_unknown_limitation_rejected():
    with pytest.raises(PacketError): validate_packet(_packet(limitations=["guess"]))


def test_serialization_is_deterministic():
    a = canonical_bytes(_packet())
    b = canonical_bytes(_packet())
    assert a == b and packet_digest(_packet()) == packet_digest(_packet())


def test_empty_findings_allowed_for_abstention_cases():
    assert validate_packet(_packet(findings=[], context=[]))["findings"] == []
