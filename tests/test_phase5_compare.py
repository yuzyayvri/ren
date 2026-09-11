import json

import pytest

from scripts import phase5_compare as compare


def _model(**over):
    base = {"name": "m", "filename": "m.gguf", "bytes": 4, "sha256": "x"}
    base.update(over)
    return base


def test_preflight_rejects_size_and_hash(tmp_path):
    target = tmp_path / "m.gguf"
    target.write_bytes(b"1234")
    model = _model(bytes=4, sha256="wrong")
    with pytest.raises(compare.ComparisonError):
        compare.preflight({**model, "filename": str(target)})
    import hashlib

    good = _model(bytes=4, sha256=hashlib.sha256(b"1234").hexdigest())
    assert compare.preflight({**good, "filename": str(target)})["sha256"] == good["sha256"]
    with pytest.raises(compare.ComparisonError):
        compare.preflight({**good, "filename": str(tmp_path / "missing.gguf")})


def test_preflight_uses_frozen_paths():
    import hashlib

    real = {m["name"]: m for m in compare.frozen_models()}
    for model in real.values():
        path = compare.ROOT / "models" / "llm" / model["filename"]
        assert path.stat().st_size == model["bytes"]
        assert hashlib.sha256(path.read_bytes()) != "x"


def test_server_argv_shape():
    argv = compare.server_argv("/bin/srv", compare.ROOT / "m.gguf")
    assert argv[:4] == ["/bin/srv", "-m", str(compare.ROOT / "m.gguf"), "-c"]
    assert "999" not in argv
    assert compare.server_argv("/bin/srv", compare.ROOT / "m.gguf", 33)[-2:] == ["-ngl", "33"]


def test_effective_offload_parsing():
    assert compare.effective_offload("offloading 41 repeating layers to GPU") == {"offloaded_layers": 41}
    assert compare.effective_offload("nothing here") == {"offloaded_layers": None}


def _good_transport(body):
    def transport(url, payload, timeout):
        assert url.endswith("/v1/chat/completions")
        return body
    return transport


def _response_for(case_id, label, go):
    return {"choices": [{"message": {"content": json.dumps({
        "schema": "phase5-response-v1", "case_id": case_id, "abstained": False,
        "claims": [{"claim_id": "C1", "text": f"{label} seen with {go}.",
                    "evidence_ids": ["E1"], "finding_ids": ["F1"]}]})}}]}


def test_run_preserves_all_repeats_and_deterministic_payloads(tmp_path, monkeypatch):
    from scripts.phase5_evaluate import case_to_packet

    def transport(url, payload, timeout):
        import json as _json

        packet = _json.loads(payload["messages"][1]["content"])
        case = next(c for c in compare.dev_cases() if c["case_id"] == packet["case_id"])
        pkt = case_to_packet(case)
        if packet["findings"]:
            f = pkt["findings"][0]
            ev = pkt["context"][0]
            return _response_for(packet["case_id"], f["label"], ev["go_id"])
        return {"choices": [{"message": {"content": json.dumps({
            "schema": "phase5-response-v1", "case_id": packet["case_id"],
            "abstained": True, "abstention_reason": "nothing supported", "claims": []})}}]}

    model = _model()
    monkeypatch.setattr(compare, "preflight", lambda m: {"filename": "m.gguf", "bytes": 1, "sha256": "s", "name": "m"})
    record = compare.run_candidate(model, transport=transport, repeats=2, out_root=tmp_path)
    assert len(record["generations"]) == 8 * 2
    assert all(g["status"] == "ok" for g in record["generations"])
    payloads = {json.dumps(g["request"], sort_keys=True) for g in record["generations"]}
    assert len(payloads) == 8
    agg = compare.aggregate(record)
    assert agg["valid_rate"] == 1.0 and agg["repeatability"] == 1.0 and agg["forbidden_hits"] == 0
    assert (tmp_path / "m" / "run.json").is_file()


def test_failures_recorded_and_vetoed(tmp_path, monkeypatch):
    def transport(url, payload, timeout):
        raise RuntimeError("server down")

    model = _model()
    monkeypatch.setattr(compare, "preflight", lambda m: {"filename": "m.gguf", "bytes": 1, "sha256": "s", "name": "m"})
    record = compare.run_candidate(model, transport=transport, repeats=1, out_root=tmp_path)
    assert all(g["status"] == "failed" for g in record["generations"])
    agg = compare.aggregate(record)
    eligible, reasons = compare.safety_veto(agg, "accepted")
    assert not eligible and reasons


def test_veto_before_ranking():
    bad = {"mean_finding_recall": 1.0, "mean_go_recall": 1.0, "repeatability": 1.0,
           "valid_rate": 1.0, "forbidden_hits": 2}
    good = {"mean_finding_recall": 0.8, "mean_go_recall": 0.8, "repeatability": 0.9,
            "valid_rate": 1.0, "forbidden_hits": 0}
    result = compare.select({"bad": bad, "good": good}, {"bad": "accepted", "good": "accepted"})
    assert result["winner"] == "good" and result["status"] == "selected"
    result = compare.select({"bad": bad, "good": good}, {"bad": "accepted", "good": "missing"})
    assert result["winner"] is None and result["status"] == "stopped_neither_eligible"


def test_lifecycle_allowlists_events(tmp_path):
    ledger = tmp_path / "lifecycle.jsonl"
    compare.append_lifecycle("part2_selected", {"winner": "m"}, ledger=ledger)
    assert "part2_selected" in ledger.read_text()
    with pytest.raises(compare.ComparisonError):
        compare.append_lifecycle("part2_tampered", {}, ledger=ledger)


def test_package_conceals_identities(tmp_path):
    run = {"candidate": {"name": "candidate-x"}, "generations": []}
    paths = compare.package_review({"candidate-x": run, "candidate-y": run}, ["candidate-x", "candidate-y"], out_root=tmp_path)
    a_text = (paths["dir"] / "notes_A.json").read_text()
    assert "candidate-x" not in a_text and "candidate-y" not in a_text
    mapping = json.loads(paths["mapping"].read_text())
    assert set(mapping.values()) == {"candidate-x", "candidate-y"}


def test_response_format_passthrough():
    from scripts.phase5_client import build_request

    packet = {"schema": "phase5-packet-v1", "case_id": "c", "source_stage": "phase3-txl",
              "findings": [], "context": [], "limitations": []}
    v2dec = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 1, "n_predict": 8,
             "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}}
    payload = build_request("sys", packet, v2dec, "m")
    assert payload["response_format"]["type"] == "json_schema"
    payload = build_request("sys", packet, {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                                            "seed": 1, "n_predict": 8}, "m")
    assert payload["response_format"] == {"type": "json_object"}


def test_raw_content_preserved_on_parse_failure(tmp_path, monkeypatch):
    prose = "thought\nThis model writes prose instead of JSON."

    def transport(url, payload, timeout):
        return {"choices": [{"message": {"content": prose}}]}

    monkeypatch.setattr(compare, "preflight",
                        lambda m: {"filename": "m.gguf", "bytes": 1, "sha256": "s", "name": "m"})
    record = compare.run_candidate(_model(), transport=transport, repeats=1, out_root=tmp_path)
    assert len(record["generations"]) == 8
    for entry in record["generations"]:
        assert entry["status"] == "failed"
        assert entry["raw_content"] == prose


def test_smoke_guard_rejects_eligible_fixture(tmp_path, monkeypatch):
    fixture = tmp_path / "smoke_packet.json"
    fixture.write_text(json.dumps({"selection_eligible": True}))
    monkeypatch.setattr(compare, "preflight",
                        lambda m: {"filename": "m.gguf", "bytes": 1, "sha256": "s", "name": "m"})
    with pytest.raises(compare.ComparisonError):
        compare.smoke(_model(), out_root=tmp_path, fixture_path=fixture)


def test_score_ignores_smoke_records(tmp_path, monkeypatch, capsys):
    import sys

    run_dir = tmp_path / "candidate-x"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "candidate": {"name": "candidate-x"}, "repeats": 1, "generations": []}))
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    (smoke_dir / "candidate-x.json").write_text(json.dumps({"selection_eligible": False}))
    monkeypatch.setattr(sys, "argv", ["phase5_compare.py", "score", "--out-root", str(tmp_path)])
    assert compare.main() == 0
    out = capsys.readouterr().out
    assert "candidate-x" in out


def test_lifecycle_v2_events(tmp_path):
    ledger = tmp_path / "lifecycle.jsonl"
    compare.append_lifecycle("part1_superseded", {"a": 1}, ledger=ledger)
    compare.append_lifecycle("phase5_v2_frozen", {"b": 2}, ledger=ledger)
    text = ledger.read_text()
    assert "part1_superseded" in text and "phase5_v2_frozen" in text


def test_response_schema_enforces_shape():
    import json
    from pathlib import Path

    schema = json.loads(Path("protocols/phase5_v2/response_schema.json").read_text())
    assert schema["properties"]["schema"] == {"const": "phase5-response-v1"}
    assert schema["additionalProperties"] is False
    claim = schema["properties"]["claims"]["items"]
    assert claim["additionalProperties"] is False
    assert set(claim["required"]) == {"claim_id", "evidence_ids", "finding_ids", "text"}


def test_relative_out_root_resolves_under_repo(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    target = compare.ROOT / "rel-out-test-residue"
    monkeypatch.setattr(compare, "preflight",
                        lambda m: {"filename": "m.gguf", "bytes": 1, "sha256": "s", "name": "m"})

    def transport(url, payload, timeout):
        raise RuntimeError("down")

    try:
        record = compare.run_candidate(_model(), transport=transport, repeats=1,
                                       out_root=Path("rel-out-test-residue"))
        assert (target / "m" / "run.json").is_file()
        assert record["generations"] and all(g["status"] == "failed" for g in record["generations"])
    finally:
        shutil.rmtree(target, ignore_errors=True)
    assert record["generations"] and all(g["status"] == "failed" for g in record["generations"])
