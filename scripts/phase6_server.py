"""Phase 6 local dashboard backend (FastAPI, loopback-only, offline-first).

Serves specimen imagery with overlays computed from existing sealed
evidence, retrieval through the sanctioned bound loader, synthesis
through the frozen production path, and review records that never
mutate frozen artifacts. No remote dependencies of any kind.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys

if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
CACHE = ROOT / "artifacts" / "phase6_cache"
REVIEWS = ROOT / "artifacts" / "phase6_reviews"
DASHBOARD = ROOT / "dashboard"

PANNUKE_CLASSES = ("background", "neoplastic", "inflammatory", "connective",
                   "dead", "epithelial")
PANNUKE_FOLD = "fold3"
PANNUKE_IMAGES = (ROOT / "data" / "tissue" / "fold3" / "Fold 3" / "images" / "fold3" / "images.npy")
TXL_SPLITS = ("train", "val", "test")
TXL_CODES = ("WBC", "RBC", "Platelets")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def specimen_catalog(limit: int = 60) -> list[dict[str, Any]]:
    import itertools

    txl = ROOT / "data" / "blood" / "txl-pbc" / "TXL-PBC"
    pools: list[list[dict[str, Any]]] = [[], [], []]
    for split in TXL_SPLITS:
        for image in sorted((txl / "images" / split).glob("*.png")):
            pools[0].append({"id": f"txl-{split}-{image.stem}", "kind": "blood-smear",
                             "source": "txl-pbc", "split": split, "name": image.name})
    aml = ROOT / "data" / "blood" / "aml" / "data" / "data"
    if aml.is_dir():
        for image in sorted(aml.glob("*/*.tiff")):
            pools[1].append({"id": f"aml-{image.parent.name}-{image.stem}", "kind": "blood-cell",
                             "source": "aml", "split": image.parent.name, "name": image.name})
    if PANNUKE_IMAGES.is_file():
        import numpy as _np

        count = int(_np.load(str(PANNUKE_IMAGES), mmap_mode="r").shape[0])
        for index in range(count):
            pools[2].append({"id": f"pannuke-f3-{index}", "kind": "tissue-patch",
                             "source": "pannuke", "split": PANNUKE_FOLD,
                             "name": f"fold3 patch {index}"})
    items: list[dict[str, Any]] = []
    for round_robin in itertools.zip_longest(*pools):
        for entry in round_robin:
            if entry is not None:
                items.append(entry)
            if len(items) >= limit:
                return items
    return items


def resolve_specimen(specimen_id: str) -> Path:
    parts = specimen_id.split("-", 2)
    if parts[0] == "txl" and len(parts) == 3:
        path = ROOT / "data" / "blood" / "txl-pbc" / "TXL-PBC" / "images" / parts[1] / f"{parts[2]}.png"
    elif parts[0] == "aml" and len(parts) == 3:
        path = ROOT / "data" / "blood" / "aml" / "data" / "data" / parts[1] / f"{parts[2]}.tiff"
    elif parts[0] == "pannuke":
        raise KeyError("panuke patches are addressed numerically; use pannuke-f3-<index>")
    else:
        raise KeyError(f"unknown specimen: {specimen_id}")
    if not path.is_file():
        raise KeyError(f"unknown specimen: {specimen_id}")
    return path


def render_image(specimen_id: str) -> tuple[bytes, str]:
    from PIL import Image

    if specimen_id.startswith("pannuke-f3-"):
        import numpy as np

        index = int(specimen_id.rsplit("-", 1)[1])
        arr = np.load(PANNUKE_IMAGES, mmap_mode="r")
        if not 0 <= index < arr.shape[0]:
            raise KeyError(specimen_id)
        frame = np.asarray(arr[index]).astype("uint8")
        return _png_bytes(Image.fromarray(frame).convert("RGB")), f"panuke-f3-{index}.png"
    path = resolve_specimen(specimen_id)
    image = Image.open(path).convert("RGB")
    return _png_bytes(image), path.name.replace(".tiff", ".png")


def _png_bytes(image: Any) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def pannuke_overlay(patch: int) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    shard = ROOT / "artifacts" / "phase2_final_protocol_fold3_final_v1" / "shards" / f"shard_{patch:06d}.npz"
    if not shard.is_file():
        raise KeyError(f"no sealed shard for patch {patch}")
    z = np.load(shard)
    mask = np.asarray(z["prediction"], dtype=np.int64)
    by_proposal = dict(zip(np.asarray(z["proposal_ids"], dtype=np.int64).tolist(),
                           np.asarray(z["predicted_types"], dtype=np.int64).tolist()))
    palette = [(0, 0, 0, 0), (230, 57, 70, 140), (241, 196, 15, 140),
               (46, 204, 113, 140), (155, 89, 182, 140), (52, 152, 219, 140)]
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    instances: list[dict[str, Any]] = []
    for proposal_id in sorted(by_proposal):
        hit = mask == proposal_id
        if not bool(hit.any()):
            continue
        nucleus_type = int(by_proposal[proposal_id])
        color = palette[nucleus_type] if 0 <= nucleus_type < len(palette) else (200, 200, 200, 140)
        rgba[hit] = color
        ys, xs = np.nonzero(hit)
        instances.append({
            "id": int(proposal_id),
            "type": PANNUKE_CLASSES[nucleus_type] if 0 <= nucleus_type < len(PANNUKE_CLASSES) else "unknown",
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "pixels": int(hit.sum()),
        })
    overlay = Image.fromarray(rgba, mode="RGBA")
    return {"overlay": _png_bytes(overlay), "instances": instances,
            "counts": {name: sum(1 for i in instances if i["type"] == name)
                       for name in PANNUKE_CLASSES[1:]}}


def txl_boxes(specimen_id: str, source: str = "labels") -> dict[str, Any]:
    from PIL import Image

    path = resolve_specimen(specimen_id)
    if not specimen_id.startswith("txl-"):
        raise KeyError(f"no box overlays for {specimen_id}")
    _, split, stem = specimen_id.split("-", 2)
    width, height = Image.open(path).size
    names = [line.split(None, 1)[1] if " " in line else line
             for line in (ROOT / "data" / "blood" / "txl-pbc" / "TXL-PBC" / "classes.txt").read_text().splitlines()
             if line.strip()]
    boxes: list[dict[str, Any]] = []
    if source == "labels":
        label = ROOT / "data" / "blood" / "txl-pbc" / "TXL-PBC" / "labels" / split / f"{stem}.txt"
        if not label.is_file():
            raise KeyError(f"no labels for {specimen_id}")
        for line in label.read_text().splitlines():
            cls, cx, cy, w, h = line.split()
            cls, cx, cy, w, h = int(cls), float(cx), float(cy), float(w), float(h)
            boxes.append({
                "label": names[cls] if cls < len(names) else str(cls),
                "code": TXL_CODES[cls] if cls < len(TXL_CODES) else str(cls),
                "bbox": [round((cx - w / 2) * width), round((cy - h / 2) * height),
                         round((cx + w / 2) * width), round((cy + h / 2) * height)],
                "confidence": None,
            })
    elif source == "detector":
        boxes = _detector_boxes(path, specimen_id)
    else:
        raise KeyError(f"unknown box source: {source}")
    return {"boxes": boxes, "counts": {b["label"]: sum(1 for x in boxes if x["label"] == b["label"]) for b in boxes}}


def _detector_boxes(path: Path, specimen_id: str) -> list[dict[str, Any]]:
    import json as _json

    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"detector-{specimen_id}.json"
    if cached.is_file():
        return _json.loads(cached.read_text(encoding="utf-8"))["boxes"]
    from fastapi import HTTPException as _HTTPException

    raise _HTTPException(status_code=501,
                         detail="detector-box overlays need the YOLO checkpoint binding")


def describe_terms(go_ids: list[str], query: str, mode: str) -> list[dict[str, Any]]:
    import sqlite3

    from scripts.phase4_v3_common import V3 as _V3

    db = sqlite3.connect(f"file:{_V3 / 'ontology.sqlite'}?mode=ro", uri=True)
    try:
        entries = []
        for rank, go_id in enumerate(go_ids[:25], start=1):
            row = db.execute("SELECT name, definition FROM terms WHERE id=? AND obsolete=0",
                             (go_id,)).fetchone()
            if row is None:
                continue
            entries.append({"go_id": go_id, "name": row[0], "definition": row[1],
                            "rank": rank, "mode": mode, "query": query})
        return entries
    finally:
        db.close()


def create_app() -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="ren workstation", docs_url=None, redoc_url=None, openapi_url=None)
    jobs: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/specimens")
    def specimens(limit: int = 60) -> list[dict[str, Any]]:
        return specimen_catalog(limit=min(max(limit, 1), 500))

    @app.get("/api/specimens/{specimen_id}/image")
    def specimen_image(specimen_id: str) -> Response:
        try:
            data, name = render_image(specimen_id)
        except (KeyError, ValueError, OSError):
            raise HTTPException(status_code=404, detail="unknown specimen")
        return Response(content=data, media_type="image/png",
                        headers={"Content-Disposition": f'inline; filename="{name}"'})

    @app.get("/api/specimens/{specimen_id}/overlays")
    def overlays(specimen_id: str, source: str = "labels") -> Any:
        try:
            if specimen_id.startswith("pannuke-f3-"):
                index = int(specimen_id.rsplit("-", 1)[1])
                result = pannuke_overlay(index)
                return JSONResponse({
                    "instances": result["instances"], "counts": result["counts"],
                    "overlay_url": f"/api/specimens/{specimen_id}/overlay.png"})
            return txl_boxes(specimen_id, source=source)
        except (KeyError, ValueError, OSError):
            raise HTTPException(status_code=404, detail="no overlays")

    @app.get("/api/specimens/{specimen_id}/overlay.png")
    def overlay_png(specimen_id: str) -> Response:
        try:
            index = int(specimen_id.rsplit("-", 1)[1])
            result = pannuke_overlay(index)
        except (KeyError, ValueError, OSError):
            raise HTTPException(status_code=404, detail="no overlays")
        return Response(content=result["overlay"], media_type="image/png")

    @app.post("/api/retrieve")
    def retrieve(body: dict[str, Any]) -> dict[str, Any]:
        from scripts.phase4_snapshot_reconciliation import load_bound_query

        query = body.get("query", "")
        mode = body.get("mode", "hybrid")
        if not isinstance(query, str) or not query.strip() or mode not in ("symbolic", "vector", "hybrid"):
            raise HTTPException(status_code=422, detail="bad retrieval request")
        try:
            engine = load_bound_query()
            return {"query": query, "mode": mode,
                    "entries": describe_terms(engine.retrieve(query, mode=mode), query, mode)}
        except (RuntimeError, ValueError, ImportError) as exc:
            raise HTTPException(status_code=502, detail=f"retrieval failed: {exc}")

    @app.get("/api/server/status")
    def server_status() -> dict[str, Any]:
        proc = app.state.server_proc if hasattr(app.state, "server_proc") else None
        return {"running": proc is not None and proc.poll() is None,
                "model": getattr(app.state, "server_model", None)}

    @app.post("/api/server/start")
    def server_start() -> dict[str, Any]:
        import subprocess

        from scripts.phase5_compare import wait_ready
        from scripts.phase5_synthesize import preflight_winner

        proc = app.state.server_proc if hasattr(app.state, "server_proc") else None
        if proc is not None and proc.poll() is None:
            return {"running": True}
        model = preflight_winner()
        from scripts.phase5_compare import DEFAULT_SERVER

        binary = DEFAULT_SERVER
        if not Path(binary).is_file():
            raise HTTPException(status_code=502, detail="llama-server binary missing")
        CACHE.mkdir(parents=True, exist_ok=True)
        log = (CACHE / "llama-server.log").open("w", encoding="utf-8")
        app.state.server_proc = subprocess.Popen(
            [binary, "-m", str(ROOT / "models" / "llm" / model["filename"]),
             "-c", "8192", "--host", "127.0.0.1", "--port", "8080"],
            stdout=log, stderr=subprocess.STDOUT)
        app.state.server_model = model["filename"]
        try:
            wait_ready("http://127.0.0.1:8080")
        except (RuntimeError, OSError) as exc:
            raise HTTPException(status_code=502, detail=f"server failed: {exc}")
        return {"running": True, "model": model["filename"]}

    @app.post("/api/server/stop")
    def server_stop() -> dict[str, Any]:
        proc = app.state.server_proc if hasattr(app.state, "server_proc") else None
        if proc is not None and proc.poll() is None:
            proc.terminate()
        app.state.server_proc = None
        app.state.server_model = None
        return {"running": False}

    @app.post("/api/jobs/synthesize")
    def synthesize_job(body: dict[str, Any]) -> dict[str, Any]:
        from scripts.phase5_packet import PacketError, validate_packet

        try:
            packet = validate_packet(body["packet"])
        except (KeyError, PacketError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="bad packet")
        job_id = uuid.uuid4().hex[:12]
        with lock:
            jobs[job_id] = {"id": job_id, "state": "queued", "progress": 0.0}
        worker = threading.Thread(target=_run_synthesis_job, args=(app, jobs, lock, job_id, packet),
                                  daemon=True)
        with lock:
            jobs[job_id]["state"] = "running"
        worker.start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: str) -> dict[str, Any]:
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="unknown job")
            return dict(job)

    @app.delete("/api/jobs/{job_id}")
    def job_cancel(job_id: str) -> dict[str, Any]:
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="unknown job")
            if job["state"] in ("done", "failed", "cancelled"):
                return {"id": job_id, "state": job["state"]}
            job["state"] = "cancelled"
            return {"id": job_id, "state": "cancelled"}

    @app.post("/api/reviews")
    def reviews(body: dict[str, Any]) -> dict[str, Any]:
        verdict = body.get("verdict")
        target = body.get("target", {})
        if verdict not in ("accept", "edit", "reject") or not isinstance(target, dict):
            raise HTTPException(status_code=422, detail="bad review")
        REVIEWS.mkdir(parents=True, exist_ok=True)
        record = {"schema": "phase6-review-v1", "verdict": verdict, "target": target,
                  "note": body.get("note"), "reviewer": body.get("reviewer", "local"),
                  "unix_time": time.time(),
                  "target_sha256": sha256_bytes(json.dumps(target, sort_keys=True).encode())}
        path = REVIEWS / f"{uuid.uuid4().hex[:12]}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"id": path.stem, "verdict": verdict}

    @app.get("/api/provenance")
    def provenance() -> dict[str, Any]:
        from scripts.phase5_compare import frozen_models

        out: dict[str, Any] = {"models": frozen_models(), "sealed": {}}
        for name in ("protocols/phase5_v1/freeze_manifest.json",
                     "protocols/phase5_v2/freeze_manifest.json",
                     "artifacts/phase5_comparison_v2/selection_manifest.json",
                     "artifacts/phase5_final_v1/final_report.json"):
            path = ROOT / name
            if path.is_file():
                out["sealed"][name] = sha256_path(path)
        return out

    if DASHBOARD.is_dir():
        app.mount("/", StaticFiles(directory=DASHBOARD, html=True), name="dashboard")
    return app


def _run_synthesis_job(app: Any, jobs: dict[str, Any], lock: threading.Lock,
                       job_id: str, packet: dict[str, Any]) -> None:
    from scripts.phase5_synthesize import SynthesisProductionError, synthesize_packet

    try:
        with lock:
            if jobs[job_id]["state"] == "cancelled":
                return
            jobs[job_id]["progress"] = 0.2
        result = synthesize_packet(packet, "http://127.0.0.1:8080")
        with lock:
            if jobs[job_id]["state"] == "cancelled":
                return
            jobs[job_id].update({"state": "done" if result["status"] == "ok" else "failed",
                                 "progress": 1.0, "result": result,
                                 "error": result.get("failure")})
    except (SynthesisProductionError, OSError, ValueError, RuntimeError, ImportError) as exc:
        with lock:
            jobs[job_id].update({"state": "failed", "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
