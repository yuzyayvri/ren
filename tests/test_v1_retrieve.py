
import pytest

from scripts import v1_retrieve as retrieve


def _confirmed(fid="F1", label="WBC", qualifier="observed"):
    return {"finding_id": fid, "label": label, "confidence": 0.9,
            "qualifier": qualifier, "review_state": "confirmed"}


def test_derive_query_deterministic():
    first = retrieve.derive_query(_confirmed())
    assert first == {"query": "white blood cell leukocyte", "rule": "v1-query-derivation-v1",
                     "finding_id": "F1", "qualifier": "observed"}
    assert retrieve.derive_query(_confirmed(qualifier="uncertain"))["qualifier"] == "uncertain"
    with pytest.raises(retrieve.RetrievalError):
        retrieve.derive_query(_confirmed(label="nope"))


def test_unconfirmed_never_retrieves(monkeypatch):
    monkeypatch.setattr("scripts.phase4_snapshot_reconciliation.load_bound_query",
                        lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    with pytest.raises(retrieve.RetrievalError):
        retrieve.retrieve_for_findings([{**_confirmed(), "review_state": "unreviewed"}])


def test_live_retrieval_links_findings():
    sets = retrieve.retrieve_for_findings([_confirmed(), _confirmed("F2", "RBC")])
    assert set(sets) == {"F1", "F2"}
    for fid, group in sets.items():
        assert group["rule"] == "v1-query-derivation-v1"
        assert group["evidence"]
        for entry in group["evidence"]:
            assert entry["finding_id"] == fid and entry["origin"].startswith("auto-")
    assert sets["F1"]["query"] != sets["F2"]["query"]


def test_selection_origins_and_exclusion(tmp_path):
    directory = tmp_path / "spec"
    directory.mkdir()
    sets = {"F1": {"query": "q", "rule": "r", "qualifier": "observed", "retrieved_unix": 1.0,
                   "evidence": [
                       {"evidence_id": "E1", "go_id": "GO:0002443", "name": "n",
                        "definition": "d", "rank": 1, "mode": "hybrid", "query": "q",
                        "origin": "auto-r", "finding_id": "F1"},
                       {"evidence_id": "E2", "go_id": "GO:0050900", "name": "n",
                        "definition": "d", "rank": 2, "mode": "hybrid", "query": "q",
                        "origin": "auto-r", "finding_id": "F1"}],
                   "excluded": [], "manual_adds": []}}
    retrieve.save_evidence(directory, sets)
    retrieve.exclude_evidence(directory, "F1", "E2", reviewer="r1")
    manual = retrieve.add_manual_evidence(directory, "F1", "GO:0006915", reviewer="r1")
    assert manual["origin"] == "human:r1" and manual["name"]
    selected = retrieve.selected_evidence(directory)
    assert [e["evidence_id"] for e in selected] == ["E1", "E2"]
    assert selected[0]["go_id"] == "GO:0002443" and selected[1]["go_id"] == "GO:0006915"
    context = retrieve.to_packet_context(directory)
    assert len(context) == 2
    from scripts.phase5_packet import validate_packet

    validate_packet({"schema": "phase5-packet-v1", "case_id": "c", "source_stage": "phase3-txl",
                     "findings": [{"finding_id": "F1", "label": "WBC", "qualifier": "observed"}],
                     "context": context, "limitations": []})
    with pytest.raises(retrieve.RetrievalError):
        retrieve.add_manual_evidence(directory, "F1", "GO:bad", reviewer="r1")
    with pytest.raises(retrieve.RetrievalError):
        retrieve.exclude_evidence(directory, "F9", "E1")


def test_packet_context_caps_per_finding(tmp_path):
    from scripts import v1_retrieve as retrieve

    directory = tmp_path / "spec"
    directory.mkdir()
    evidence = []
    for rank in range(1, 7):
        evidence.append({"evidence_id": f"E{rank}", "go_id": f"GO:{rank:07d}",
                         "name": "n", "definition": "d", "rank": rank,
                         "mode": "hybrid", "query": "q", "origin": "auto-r",
                         "finding_id": "F1"})
    sets = {"F1": {"query": "q", "rule": "r", "qualifier": "observed",
                   "retrieved_unix": 1.0, "evidence": evidence,
                   "excluded": [], "manual_adds": [
                       {"evidence_id": "M1", "go_id": "GO:0006915", "name": "n",
                        "definition": "d", "rank": 7, "mode": "hybrid", "query": "q",
                        "origin": "human:r1", "finding_id": "F1"}]}}
    retrieve.save_evidence(directory, sets)
    context = retrieve.to_packet_context(directory)
    kept = [e["go_id"] for e in context]
    assert kept[:3] == ["GO:0000001", "GO:0000002", "GO:0000003"]
    assert "GO:0006915" in kept and len(context) == 4
