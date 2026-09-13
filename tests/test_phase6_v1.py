import json
import shutil

import pytest
from fastapi.testclient import TestClient

from scripts import phase6_server as srv

ROOT = srv.ROOT


@pytest.fixture()
def client():
    return TestClient(srv.create_app())


def _txl_bytes():
    return (ROOT / "data/blood/txl-pbc/TXL-PBC/images/val/10c5112dfae4533cb9fa8f6f73a49274.png").read_bytes()


def _import(client, data=None):
    return client.post("/api/specimens/import",
                       files={"file": ("smear.png", data or _txl_bytes(), "image/png")})


def test_import_accepts_and_rejects(client):
    record = _import(client).json()
    assert record["schema"] == "v1-specimen-v1"
    shutil.rmtree(ROOT / "artifacts" / "v1_specimens" / record["specimen_id"])
    bad = client.post("/api/specimens/import",
                      files={"file": ("x.png", b"not an image at all!!!!" * 10, "image/png")})
    assert bad.status_code == 422
    bad = client.post("/api/specimens/import",
                      files={"file": ("x.txt", b"plain text here!!!!!!!!!!" * 10, "text/plain")})
    assert bad.status_code == 422


def test_v1_overlays_need_analysis(client):
    specimen_id = _import(client).json()["specimen_id"]
    try:
        assert client.get(f"/api/specimens/{specimen_id}/overlays").status_code == 409
        assert client.get(f"/api/v1/findings/{specimen_id}").status_code == 409
        assert client.post(f"/api/v1/synthesize/{specimen_id}").status_code == 409
    finally:
        shutil.rmtree(ROOT / "artifacts" / "v1_specimens" / specimen_id)


def test_v1_review_lifecycle(client, monkeypatch):

    specimen_id = _import(client).json()["specimen_id"]
    directory = ROOT / "artifacts" / "v1_specimens" / specimen_id
    try:
        fake_vision = {"specimen_sha256": "x", "models": {},
                       "findings": [{"finding_id": "F1", "label": "WBC", "confidence": 0.9,
                                     "region": {"box_xyxy": [1, 1, 5, 5]}, "source": "machine",
                                     "specimen_id": specimen_id}]}
        (directory / "vision.json").write_text(json.dumps(fake_vision))
        body = {"specimen_id": specimen_id, "finding_id": "F1", "action": "confirm",
                "reviewer": "t"}
        assert client.post("/api/v1/reviews", json=body).json()["action"] == "confirm"
        body = client.get(f"/api/v1/findings/{specimen_id}").json()
        assert body["findings"][0]["review_state"] == "confirmed"
        assert client.post("/api/v1/reviews", json={}).status_code == 422
    finally:
        shutil.rmtree(directory)


def test_v1_retrieve_and_evidence_flow(client, monkeypatch):
    import scripts.phase4_snapshot_reconciliation as rec

    specimen_id = _import(client).json()["specimen_id"]
    directory = ROOT / "artifacts" / "v1_specimens" / specimen_id
    try:
        fake_vision = {"specimen_sha256": "x", "models": {},
                       "findings": [{"finding_id": "F1", "label": "WBC", "confidence": 0.9,
                                     "region": {"box_xyxy": [1, 1, 5, 5]}, "source": "machine",
                                     "specimen_id": specimen_id}]}
        (directory / "vision.json").write_text(json.dumps(fake_vision))
        client.post("/api/v1/reviews", json={
            "specimen_id": specimen_id, "finding_id": "F1", "action": "confirm"})

        class Engine:
            def retrieve(self, query, mode="hybrid", k=5):
                return ["GO:0002443"]

        monkeypatch.setattr(rec, "load_bound_query", lambda: Engine())
        result = client.post(f"/api/v1/retrieve/{specimen_id}").json()
        assert result["evidence"] == 1
        assert client.post("/api/v1/evidence", json={
            "specimen_id": specimen_id, "op": "exclude",
            "finding_id": "F1", "evidence_id": "E1"}).json()["excluded"] == "E1"
        assert client.post("/api/v1/evidence", json={
            "specimen_id": specimen_id, "op": "include",
            "finding_id": "F1", "evidence_id": "E1"}).json()["included"] == "E1"
        assert client.post("/api/v1/evidence", json={"specimen_id": specimen_id,
                                                     "op": "bogus"}).status_code == 422
    finally:
        shutil.rmtree(directory)


def test_v1_synthesize_and_signoff_guards(client, monkeypatch):
    import scripts.phase5_synthesize as synthesis

    specimen_id = _import(client).json()["specimen_id"]
    directory = ROOT / "artifacts" / "v1_specimens" / specimen_id
    try:
        assert client.post(f"/api/v1/signoff/{specimen_id}", json={}).status_code == 409
        fake_vision = {"specimen_sha256": "x", "models": {},
                       "findings": [{"finding_id": "F1", "label": "WBC", "confidence": 0.9,
                                     "region": {"box_xyxy": [1, 1, 5, 5]}, "source": "machine",
                                     "specimen_id": specimen_id}]}
        (directory / "vision.json").write_text(json.dumps(fake_vision))
        client.post("/api/v1/reviews", json={
            "specimen_id": specimen_id, "finding_id": "F1", "action": "confirm"})

        class Engine:
            def retrieve(self, query, mode="hybrid", k=5):
                return ["GO:0002443"]

        import scripts.phase4_snapshot_reconciliation as rec

        monkeypatch.setattr(rec, "load_bound_query", lambda: Engine())
        client.post(f"/api/v1/retrieve/{specimen_id}")

        def fake_synthesize(packet, base_url, timeout_s=600.0, transport=None):
            from scripts.phase5_render import render_note
            from scripts.phase5_validate import validate_response

            validated = validate_response(packet, {
                "schema": "phase5-response-v1", "case_id": packet["case_id"],
                "abstained": False,
                "claims": [{"claim_id": "C1", "text": "WBC seen, GO:0002443.",
                            "evidence_ids": ["E1"], "finding_ids": ["F1"]}]})
            return {"status": "ok", "packet": packet, "validated": validated,
                    "note": render_note(packet, validated)}

        monkeypatch.setattr(synthesis, "synthesize_packet", fake_synthesize)
        job_id = client.post(f"/api/v1/synthesize/{specimen_id}").json()["job_id"]
        import time

        for _ in range(100):
            state = client.get(f"/api/jobs/{job_id}").json()
            if state["state"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert state["state"] == "done"
        signoff = client.post(f"/api/v1/signoff/{specimen_id}",
                              json={"reviewer": "t"}).json()
        assert signoff["reviewer"] == "t" and signoff["packet_sha256"]
        export = client.get(f"/api/v1/export/{specimen_id}").json()
        assert export["schema"] == "v1-export-v1"
        assert export["signoff"]["reviewer"] == "t"
        assert len(export["reviews"]) == 1
    finally:
        shutil.rmtree(directory)


def test_frontend_v1_wiring_present():
    text = (ROOT / "dashboard" / "app.js").read_text()
    for token in ("import-file", "v1RetrieveAll", "v1Synthesize", "export-btn",
                  "/api/v1/signoff", "S.highlight", "boxCodes"):
        assert token in text


def test_retrieve_unanalyzed_is_409_not_500(client):
    specimen_id = _import(client).json()["specimen_id"]
    try:
        r = client.post(f"/api/v1/retrieve/{specimen_id}")
        assert r.status_code == 409, r.text[:200]
    finally:
        shutil.rmtree(ROOT / "artifacts" / "v1_specimens" / specimen_id)


def test_evidence_ops_without_retrieval_are_422(client):
    specimen_id = _import(client).json()["specimen_id"]
    try:
        assert client.post("/api/v1/evidence", json={
            "specimen_id": specimen_id, "op": "exclude",
            "finding_id": "F1", "evidence_id": "E1"}).status_code == 422
        assert client.post("/api/v1/evidence", json={
            "specimen_id": specimen_id, "op": "add",
            "finding_id": "F1", "go_id": "GO:0006915"}).status_code == 422
    finally:
        shutil.rmtree(ROOT / "artifacts" / "v1_specimens" / specimen_id)


def test_bulk_approve_confirms_only_unreviewed(client):
    specimen_id = _import(client).json()["specimen_id"]
    directory = ROOT / "artifacts" / "v1_specimens" / specimen_id
    try:
        fake_vision = {"specimen_sha256": "x", "models": {},
                       "findings": [
                           {"finding_id": "F1", "label": "WBC", "confidence": 0.9,
                            "region": {"box_xyxy": [1, 1, 5, 5]}, "source": "machine",
                            "specimen_id": specimen_id},
                           {"finding_id": "F2", "label": "RBC", "confidence": 0.8,
                            "region": {"box_xyxy": [5, 6, 7, 8]}, "source": "machine",
                            "specimen_id": specimen_id}]}
        (directory / "vision.json").write_text(json.dumps(fake_vision))
        client.post("/api/v1/reviews", json={
            "specimen_id": specimen_id, "finding_id": "F2", "action": "reject"})
        body = client.post("/api/v1/reviews/bulk", json={
            "specimen_id": specimen_id, "action": "confirm"}).json()
        assert body["confirmed"] == ["F1"]
        from scripts import v1_findings as findings

        states = {f["finding_id"]: f["review_state"]
                  for f in findings.effective_findings(directory)}
        assert states == {"F1": "confirmed"}
        decisions = {r["finding_id"]: r["action"] for r in findings.read_reviews(directory)}
        assert decisions["F2"] == "reject"
        reasons = [r["reason"] for r in findings.read_reviews(directory) if r["finding_id"] == "F1"]
        assert reasons == ["bulk auto-approval"]
        assert client.post("/api/v1/reviews/bulk", json={
            "specimen_id": specimen_id, "action": "reject"}).status_code == 422
        assert client.post("/api/v1/reviews/bulk", json={
            "specimen_id": "000000000000", "action": "confirm"}).status_code == 404
    finally:
        shutil.rmtree(directory)


def test_review_mode_toggle_present():
    text = (ROOT / "dashboard" / "app.js").read_text()
    assert "Manual Approve" in text and "Approve For Me" in text
    assert "approve-all" in text and "reviewMode" in text


def _v1_specimen_with_vision(client, label="WBC"):
    specimen_id = _import(client).json()["specimen_id"]
    directory = ROOT / "artifacts" / "v1_specimens" / specimen_id
    vision = {"specimen_sha256": "x", "models": {},
              "findings": [{"finding_id": "F1", "label": label, "confidence": 0.9,
                            "region": {"box_xyxy": [1, 1, 5, 5]}, "source": "machine",
                            "specimen_id": specimen_id}]}
    (directory / "vision.json").write_text(json.dumps(vision))
    client.post("/api/v1/reviews", json={
        "specimen_id": specimen_id, "finding_id": "F1", "action": "confirm"})
    return specimen_id, directory


def _v1_synthesis(directory, label="WBC", go="GO:0002443"):
    from scripts.phase5_packet import build_packet
    from scripts.phase5_render import render_note
    from scripts.phase5_validate import validate_response

    packet = build_packet(
        f"v1-{directory.name}", "phase3-txl",
        [{"finding_id": "F1", "label": label, "confidence": 0.9,
          "qualifier": "observed"}],
        [{"evidence_id": "E1", "go_id": go, "name": "n",
          "definition": "d", "rank": 1, "mode": "hybrid", "query": "q"}],
        ["image-level-only", "no-patient-linkage"])
    response = {"schema": "phase5-response-v1", "case_id": packet["case_id"],
                "abstained": False,
                "claims": [{"claim_id": "C1", "text": f"{label} seen, {go}.",
                            "evidence_ids": ["E1"], "finding_ids": ["F1"]}]}
    validated = validate_response(packet, response)
    (directory / "evidence.json").write_text(json.dumps({
        "F1": {"query": "q", "rule": "r", "qualifier": "observed", "retrieved_unix": 1.0,
               "evidence": [{"evidence_id": "E1", "go_id": go, "name": "n",
                             "definition": "d", "rank": 1, "mode": "hybrid", "query": "q",
                             "origin": "auto-r", "finding_id": "F1"}],
               "excluded": [], "manual_adds": []}}))
    (directory / "synthesis.json").write_text(json.dumps({
        "status": "ok", "packet": packet, "validated": validated,
        "note": render_note(packet, validated)}))
    return packet


def test_signoff_rejects_stale_synthesis(client):
    specimen_id, directory = _v1_specimen_with_vision(client)
    try:
        _v1_synthesis(directory)
        assert client.post(f"/api/v1/signoff/{specimen_id}", json={}).status_code == 200
        client.post("/api/v1/reviews", json={
            "specimen_id": specimen_id, "finding_id": "F1", "action": "correct",
            "changes": {"qualifier": "uncertain"}})
        r = client.post(f"/api/v1/signoff/{specimen_id}", json={})
        assert r.status_code == 409, r.text[:200]
        lines = (directory / "signoffs.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
    finally:
        shutil.rmtree(directory)


def test_cancelled_job_stays_cancelled(client, monkeypatch):
    import time as _time

    started = {}

    def slow(packet, base_url, timeout_s=600.0, transport=None):
        started["ran"] = True
        _time.sleep(2)
        return {"status": "ok", "packet": packet, "validated": {"abstained": True},
                "note": "n"}

    import scripts.phase5_synthesize as synthesis

    monkeypatch.setattr(synthesis, "synthesize_packet", slow)
    packet = {"schema": "phase5-packet-v1", "case_id": "c", "source_stage": "phase3-txl",
              "findings": [], "context": [], "limitations": []}
    job_id = client.post("/api/jobs/synthesize", json={"packet": packet}).json()["job_id"]
    assert client.delete(f"/api/jobs/{job_id}").json()["state"] == "cancelled"
    _time.sleep(2.5)
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "cancelled"


def test_v1_cancel_mid_synthesis_leaves_state(client, monkeypatch):
    import time as _time

    import scripts.phase5_synthesize as synthesis

    specimen_id, directory = _v1_specimen_with_vision(client)
    try:
        client.post("/api/v1/reviews", json={
            "specimen_id": specimen_id, "finding_id": "F1", "action": "confirm"})
        (directory / "evidence.json").write_text(json.dumps({
            "F1": {"query": "q", "rule": "r", "qualifier": "observed", "retrieved_unix": 1.0,
                   "evidence": [{"evidence_id": "E1", "go_id": "GO:0002443", "name": "n",
                                 "definition": "d", "rank": 1, "mode": "hybrid", "query": "q",
                                 "origin": "auto-r", "finding_id": "F1"}],
                   "excluded": [], "manual_adds": []}}))

        def slow(packet, base_url, timeout_s=600.0, transport=None):
            _time.sleep(2)
            return {"status": "ok", "packet": packet,
                    "validated": {"abstained": True}, "note": "n"}

        monkeypatch.setattr(synthesis, "synthesize_packet", slow)
        job_id = client.post(f"/api/v1/synthesize/{specimen_id}").json()["job_id"]
        assert client.delete(f"/api/jobs/{job_id}").json()["state"] == "cancelled"
        _time.sleep(2.5)
        assert client.get(f"/api/jobs/{job_id}").json()["state"] == "cancelled"
        assert not (directory / "synthesis.json").is_file()
    finally:
        shutil.rmtree(directory)


def test_v1_synthesis_reuses_identical_active_job(client, monkeypatch):
    import concurrent.futures
    import threading
    import time as _time

    import scripts.phase5_synthesize as synthesis

    specimen_id, directory = _v1_specimen_with_vision(client)
    try:
        (directory / "evidence.json").write_text(json.dumps({
            "F1": {"query": "q", "rule": "r", "qualifier": "observed",
                   "retrieved_unix": 1.0,
                   "evidence": [{"evidence_id": "E1", "go_id": "GO:0002443",
                                 "name": "n", "definition": "d", "rank": 1,
                                 "mode": "hybrid", "query": "q", "origin": "auto-r",
                                 "finding_id": "F1"}],
                   "excluded": [], "manual_adds": []}}))

        calls = []
        barrier = threading.Barrier(2)

        def slow(packet, base_url, timeout_s=600.0, transport=None):
            calls.append(packet["case_id"])
            _time.sleep(1)
            return {"status": "ok", "packet": packet,
                    "validated": {"abstained": True}, "note": "n"}

        monkeypatch.setattr(synthesis, "synthesize_packet", slow)
        def submit():
            barrier.wait(timeout=5)
            return client.post(f"/api/v1/synthesize/{specimen_id}").json()["job_id"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(lambda _: submit(), range(2)))
        first, second = ids
        assert second == first
        _time.sleep(1.5)
        assert client.get(f"/api/jobs/{first}").json()["state"] == "done"
        assert calls == [f"v1-{specimen_id}"]
    finally:
        shutil.rmtree(directory)

def test_retrieve_returns_finding_mapped_sets(client):
    specimen_id, directory = _v1_specimen_with_vision(client)
    try:
        client.post("/api/v1/reviews", json={
            "specimen_id": specimen_id, "finding_id": "F1", "action": "confirm"})
        (directory / "evidence.json").write_text(json.dumps({
            "F1": {"query": "white blood cell leukocyte", "rule": "r",
                   "qualifier": "observed", "retrieved_unix": 1.0,
                   "evidence": [{"evidence_id": "E1", "go_id": "GO:0002443", "name": "n",
                                 "definition": "d", "rank": 1, "mode": "hybrid",
                                 "query": "white blood cell leukocyte",
                                 "origin": "auto-r", "finding_id": "F1"}],
                   "excluded": [], "manual_adds": []}}))
        import scripts.phase4_snapshot_reconciliation as rec

        class Engine:
            def retrieve(self, query, mode="hybrid", k=5):
                assert query == "white blood cell leukocyte"
                return ["GO:0002443"]

        from unittest import mock

        with mock.patch.object(rec, "load_bound_query", return_value=Engine()):
            body = client.post(f"/api/v1/retrieve/{specimen_id}").json()
        assert set(body["sets"]) == {"F1"}
        group = body["sets"]["F1"]
        assert group["query"] == "white blood cell leukocyte"
        assert group["evidence"][0]["finding_id"] == "F1"
        assert group["evidence"][0]["origin"].startswith("auto-")
    finally:
        shutil.rmtree(directory)


def test_frontend_auto_chain_present():
    text = (ROOT / "dashboard" / "app.js").read_text()
    assert "autoRetrieveQuiet" in text
    assert "manual query remains available" in text


def test_import_preserves_original_filename(client, tmp_path):
    from PIL import Image

    probe = tmp_path / "probe-src.png"
    img = Image.open(ROOT / "data/blood/txl-pbc/TXL-PBC/images/val/02329f7234bd54e68632dd2eb18be58b.png").convert("RGB")
    img.putpixel((0, 0), (7, 77, 177))
    img.save(probe)
    r = client.post("/api/specimens/import",
                    files={"file": ("my-smear.png", probe.read_bytes(), "image/png")}).json()
    try:
        assert r["source_filename"] == "my-smear.png"
        names = [s["name"] for s in client.get("/api/specimens?limit=500").json()]
        assert "my-smear.png" in names
    finally:
        shutil.rmtree(ROOT / "artifacts" / "v1_specimens" / r["specimen_id"])
