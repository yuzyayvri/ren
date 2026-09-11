import json

import pytest

from scripts import phase5_synthesize as synth


def _packet():
    from scripts.phase5_packet import build_packet

    return build_packet(
        "t1", "phase3-txl",
        [{"finding_id": "F1", "label": "WBC", "qualifier": "observed"}],
        [{"evidence_id": "E1", "go_id": "GO:0002443", "name": "n",
          "definition": "d", "rank": 1, "mode": "hybrid", "query": "q"}],
        [],
    )


def _response(packet):
    return {"choices": [{"message": {"content": json.dumps({
        "schema": "phase5-response-v1", "case_id": packet["case_id"],
        "abstained": False,
        "claims": [{"claim_id": "C1", "text": "WBC seen, GO:0002443.",
                    "evidence_ids": ["E1"], "finding_ids": ["F1"]}]})}}]}


def _patched(monkeypatch):
    monkeypatch.setattr(synth, "preflight_winner",
                        lambda: {"name": "m", "filename": "m.gguf", "sha256": "s"})
    monkeypatch.setattr(synth, "offline_evidence",
                        lambda: {"network_blocked": True, "exception": "x"})
    monkeypatch.setattr(synth, "retrieval_provenance", lambda: {"corpus": "t"})


def test_loopback_refused():
    with pytest.raises(synth.SynthesisProductionError):
        synth.synthesize_packet(_packet(), "http://example.com:8080")


def test_offline_gate(monkeypatch):
    _patched(monkeypatch)
    monkeypatch.setattr(synth, "offline_evidence",
                        lambda: {"network_blocked": False, "exception": None})
    with pytest.raises(synth.SynthesisProductionError):
        synth.synthesize_packet(_packet(), "http://127.0.0.1:8080")


def test_ok_path_renders_only_validated(monkeypatch):
    _patched(monkeypatch)
    packet = _packet()
    result = synth.synthesize_packet(
        packet, "http://127.0.0.1:8080",
        transport=lambda *a: _response(packet))
    assert result["status"] == "ok" and "WBC" in result["note"]
    assert result["model"]["name"] == "m"


def test_invalid_response_fails_without_note(monkeypatch):
    _patched(monkeypatch)
    packet = _packet()
    result = synth.synthesize_packet(
        packet, "http://127.0.0.1:8080",
        transport=lambda *a: {"choices": [{"message": {"content": "prose"}}]})
    assert result["status"] == "failed" and "note" not in result
    assert result["raw_content"] == "prose"


def test_transport_error_fails(monkeypatch):
    _patched(monkeypatch)

    def transport(*a):
        raise RuntimeError("down")

    result = synth.synthesize_packet(_packet(), "http://127.0.0.1:8080",
                                     transport=transport)
    assert result["status"] == "failed" and "note" not in result


def test_retrieve_context_from_sealed_corpus(monkeypatch):
    class Engine:
        def retrieve(self, query, mode="hybrid", k=5):
            assert mode == "hybrid"
            return ["GO:0006915", "GO:0006915"]

    monkeypatch.setattr("scripts.phase4_snapshot_reconciliation.load_bound_query",
                        lambda: Engine())
    findings = [{"finding_id": "F1", "label": "dead", "qualifier": "observed"}]
    context = synth.retrieve_context(findings, {})
    assert [e["evidence_id"] for e in context] == ["E1"]
    assert context[0]["go_id"] == "GO:0006915" and context[0]["rank"] == 1


def test_production_packet_rejects_bad_label():
    from scripts.phase5_packet import PacketError

    with pytest.raises(PacketError):
        synth.build_production_packet(
            "t", "phase3-txl", [{"finding_id": "F1", "label": "nope",
                                 "qualifier": "observed"}], {}, [])
