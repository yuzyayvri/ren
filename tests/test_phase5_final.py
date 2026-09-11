import json

import pytest

from scripts import phase5_final as final


def test_append_event_allowlist(tmp_path, monkeypatch):
    ledger = tmp_path / "lifecycle.jsonl"
    monkeypatch.setattr(final, "LEDGER", ledger)
    final.append_event("part3_opened_final", {"a": 1})
    assert "part3_opened_final" in ledger.read_text()
    with pytest.raises(final.FinalError):
        final.append_event("part3_tampered", {})


def test_preconditions_abort_on_prior_open(tmp_path, monkeypatch):
    ledger = tmp_path / "lifecycle.jsonl"
    ledger.write_text(json.dumps({"event": "part3_opened_final"}) + "\n")
    monkeypatch.setattr(final, "LEDGER", ledger)
    with pytest.raises(final.FinalError):
        final.check_preconditions()


def test_seal_report_accept_and_stop(tmp_path, monkeypatch):
    ledger = tmp_path / "lifecycle.jsonl"
    monkeypatch.setattr(final, "LEDGER", ledger)
    out = tmp_path / "out"
    ok = {"schema": "phase5-final-v1", "generations": [
        {"case_id": "c1", "repeat": 0, "status": "ok", "score": {"pass": True}},
        {"case_id": "c1", "repeat": 1, "status": "ok", "score": {"pass": True}},
        {"case_id": "c1", "repeat": 2, "status": "ok", "score": {"pass": True}}]}
    report = final.seal_report(ok, out)
    assert report["all_pass"] is True
    assert "part3_accepted" in ledger.read_text()
    bad = {"schema": "phase5-final-v1", "generations": [
        {"case_id": "c1", "repeat": 0, "status": "ok", "score": {"pass": True}},
        {"case_id": "c1", "repeat": 1, "status": "failed"},
        {"case_id": "c1", "repeat": 2, "status": "ok", "score": {"pass": True}}]}
    report = final.seal_report(bad, tmp_path / "out2")
    assert report["all_pass"] is False
    assert report["per_case_pass"] == {"c1": False}
    assert "part3_stopped" in ledger.read_text()


def test_run_final_refuses_without_preconditions(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "LEDGER", tmp_path / "lifecycle.jsonl")

    def boom():
        raise final.FinalError("not green")

    monkeypatch.setattr(final, "check_preconditions", boom)
    with pytest.raises(final.FinalError):
        final.run_final(out_root=tmp_path / "out", cases=[])
    assert not (tmp_path / "out").exists()
