import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts import phase6_server as srv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client():
    return TestClient(srv.create_app())


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_specimens_and_image(client):
    items = client.get("/api/specimens?limit=5").json()
    assert len(items) == 5 and {"id", "kind"} <= set(items[0])
    first = items[0]["id"]
    r = client.get(f"/api/specimens/{first}/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/specimens/nope/image").status_code == 404


def test_txl_boxes_use_real_class_names(client):
    body = client.get("/api/specimens/txl-val-10c5112dfae4533cb9fa8f6f73a49274/overlays").json()
    assert body["counts"] and all(" " in label or "(" in label for label in body["counts"])
    assert client.get("/api/specimens/txl-val-10c5112dfae4533cb9fa8f6f73a49274/overlays?source=nope").status_code in (404, 501)


def test_detector_source_reports_unbound(client):
    r = client.get("/api/specimens/txl-val-10c5112dfae4533cb9fa8f6f73a49274/overlays?source=detector")
    assert r.status_code == 501


def test_pannuke_overlay_matches_sealed_counts(client):
    body = client.get("/api/specimens/pannuke-f3-0/overlays").json()
    assert body["counts"] == {"neoplastic": 13, "inflammatory": 1, "connective": 0, "dead": 0, "epithelial": 0}
    assert len(body["instances"]) == 14
    assert body["overlay_url"].endswith("/overlay.png")
    r = client.get("/api/specimens/pannuke-f3-0/overlay.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/specimens/pannuke-f3-999999/overlays").status_code == 404


def test_retrieve_rejects_bad_requests(client):
    assert client.post("/api/retrieve", json={}).status_code == 422
    assert client.post("/api/retrieve", json={"query": "x", "mode": "nope"}).status_code == 422


def test_retrieve_symbolic(client):
    body = client.post("/api/retrieve", json={"query": "apoptosis", "mode": "symbolic"}).json()
    assert body["entries"] and body["entries"][0]["go_id"].startswith("GO:")
    assert all(e["name"] and e["definition"] for e in body["entries"])


def _packet():
    from scripts.phase5_packet import build_packet

    return build_packet(
        "ui-t1", "phase3-txl",
        [{"finding_id": "F1", "label": "WBC", "qualifier": "observed"}],
        [{"evidence_id": "E1", "go_id": "GO:0002443", "name": "n",
          "definition": "d", "rank": 1, "mode": "hybrid", "query": "q"}],
        [],
    )


def test_job_lifecycle_and_golden_packet(client, monkeypatch):
    import scripts.phase5_synthesize as syn

    seen = {}

    def fake(packet, base_url, timeout_s=600.0):
        seen["packet"] = packet
        assert base_url == "http://127.0.0.1:8080"
        return {"status": "ok", "note": "NOTE", "validated": {"abstained": True}}

    monkeypatch.setattr(syn, "synthesize_packet", fake)
    r = client.post("/api/jobs/synthesize", json={"packet": "nope"})
    assert r.status_code == 422
    job_id = client.post("/api/jobs/synthesize", json={"packet": _packet()}).json()["job_id"]
    import time

    for _ in range(100):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["state"] == "done" and state["result"]["note"] == "NOTE"
    from scripts.phase5_packet import canonical_bytes

    assert canonical_bytes(seen["packet"]) == canonical_bytes(_packet())
    assert client.get("/api/jobs/unknown").status_code == 404
    assert client.delete("/api/jobs/unknown").status_code == 404


def test_job_cancel(client, monkeypatch):
    import scripts.phase5_synthesize as syn

    def slow(packet, base_url, timeout_s=600.0):
        import time as _t

        _t.sleep(5)
        return {"status": "ok"}

    monkeypatch.setattr(syn, "synthesize_packet", slow)
    job_id = client.post("/api/jobs/synthesize", json={"packet": _packet()}).json()["job_id"]
    assert client.delete(f"/api/jobs/{job_id}").json()["state"] in ("cancelled", "done", "failed")


def test_reviews_recorded_outside_frozen_tree(client):
    body = {"verdict": "accept", "target": {"specimen": "x"}, "reviewer": "t"}
    record_id = client.post("/api/reviews", json=body).json()["id"]
    path = ROOT / "artifacts" / "phase6_reviews" / f"{record_id}.json"
    assert path.is_file()
    assert json.loads(path.read_text())["verdict"] == "accept"
    assert "phase6_reviews" not in str(ROOT / "artifacts" / "phase2_final_protocol_fold3_final_v1")
    assert client.post("/api/reviews", json={"verdict": "maybe"}).status_code == 422


def test_provenance_lists_sealed_digests(client):
    sealed = client.get("/api/provenance").json()["sealed"]
    assert "protocols/phase5_v2/freeze_manifest.json" in sealed


def test_offline_audit():
    seen = False
    for path in (ROOT / "dashboard").rglob("*"):
        if path.is_file():
            text = path.read_text(errors="replace")
            assert "googleapis" not in text and "cdn.jsdelivr" not in text and "unpkg" not in text
            for token in ("https://", "http://"):
                for line in text.splitlines():
                    if token in line:
                        assert "127.0.0.1" in line, (path.name, line[:100])
        seen = True
    assert seen


def test_script_bootstrap_present():
    text = (ROOT / "scripts" / "phase6_server.py").read_text()
    assert "if str(ROOT) not in _sys.path:" in text


def test_control_bindings_present():
    text = (ROOT / "dashboard" / "app.js").read_text()
    assert '$("overlay-toggle").addEventListener("click", toggleOverlay)' in text
    assert '$("prev").addEventListener("click"' in text
    assert '$("next").addEventListener("click"' in text
    assert "pannuke-f3-${i}" not in text


def test_catalog_serves_pannuke():
    from fastapi.testclient import TestClient

    client = TestClient(srv.create_app())
    ids = [s["id"] for s in client.get("/api/specimens?limit=500").json()]
    assert any(i.startswith("pannuke-f3-") for i in ids)
    assert "pannuke-f3-0" in ids
    assert any(i.startswith("txl-") for i in ids)


def test_pannuke_image_loads():
    from fastapi.testclient import TestClient

    client = TestClient(srv.create_app())
    r = client.get("/api/specimens/pannuke-f3-0/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert len(r.content) > 10000


def test_box_codes_validate_as_packet():
    from fastapi.testclient import TestClient

    from scripts.phase5_packet import build_packet

    client = TestClient(srv.create_app())
    body = client.get("/api/specimens/txl-val-10c5112dfae4533cb9fa8f6f73a49274/overlays").json()
    assert body["boxes"] and all(b["code"] in ("WBC", "RBC", "Platelets") for b in body["boxes"])
    counts: dict[str, int] = {}
    for box in body["boxes"]:
        counts[box["code"]] = counts.get(box["code"], 0) + 1
    packet = build_packet(
        "ui-t1", "phase3-txl",
        [{"finding_id": f"F{i + 1}", "label": label, "count": n, "qualifier": "observed"}
         for i, (label, n) in enumerate(counts.items())],
        [], [])
    assert packet["findings"]


def test_import_errors_report_diagnosed_not_bare(client, monkeypatch):
    import scripts.phase4_snapshot_reconciliation as rec

    def boom():
        raise ModuleNotFoundError("No module named 'scripts'")

    monkeypatch.setattr(rec, "load_bound_query", boom)
    r = client.post("/api/retrieve", json={"query": "x", "mode": "symbolic"})
    assert r.status_code == 502 and "No module named" in r.json()["detail"]
