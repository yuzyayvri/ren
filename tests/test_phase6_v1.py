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
