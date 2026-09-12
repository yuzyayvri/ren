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

from fastapi import UploadFile

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


V1_SOURCE = "v1-import"


def specimen_catalog(limit: int = 60) -> list[dict[str, Any]]:
    import itertools

    txl = ROOT / "data" / "blood" / "txl-pbc" / "TXL-PBC"
    pools: list[list[dict[str, Any]]] = [[], [], []]
    for split in TXL_SPLITS:
        for image in sorted((txl / "images" / split).glob("*.png")):
            pools[0].append({"id": f"txl-{split}-{image.stem}", "kind": "blood-smear",
                             "source": "txl-pbc", "split": split, "name": image.name})
    v1pool: list[dict[str, Any]] = []
    v1dir = ROOT / "artifacts" / "v1_specimens"
    if v1dir.is_dir():
        for record in sorted(v1dir.glob("*/specimen.json")):
            try:
                meta = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            v1pool.append({"id": meta["specimen_id"], "kind": "blood-smear",
                           "source": V1_SOURCE, "split": "imported",
                           "name": meta.get("source_filename", meta["specimen_id"])})
    pools.insert(0, v1pool)
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


def is_v1_specimen(specimen_id: str) -> bool:
    import re

    return bool(re.fullmatch(r"[0-9a-f]{12}", specimen_id)) and (
        ROOT / "artifacts" / "v1_specimens" / specimen_id / "specimen.json").is_file()


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

    if is_v1_specimen(specimen_id):
        path = ROOT / "artifacts" / "v1_specimens" / specimen_id / "normalized.png"
        return path.read_bytes(), f"{specimen_id}.png"

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

    @app.middleware("http")
    async def _no_store_api(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
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
            if is_v1_specimen(specimen_id):
                from scripts import v1_findings as findings

                directory = ROOT / "artifacts" / "v1_specimens" / specimen_id
                if not (directory / "vision.json").is_file():
                    raise HTTPException(status_code=409, detail="specimen not analyzed yet")
                live = findings.effective_findings(directory)
                boxes = [{
                    "label": f["label"], "code": f["label"],
                    "bbox": [round(v) for v in f["region"]["box_xyxy"]],
                    "confidence": f.get("confidence"),
                    "finding_id": f["finding_id"],
                    "origin": "machine" if f["review_state"] == "unreviewed" else "human-" + f["review_state"],
                } for f in live]
                counts: dict[str, int] = {}
                for box in boxes:
                    counts[box["label"]] = counts.get(box["label"], 0) + 1
                return {"boxes": boxes, "counts": counts}
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

    def _managed_proc(app: Any) -> Any:
        proc = app.state.server_proc if hasattr(app.state, "server_proc") else None
        return proc if proc is not None and proc.poll() is None else None


    def _endpoint_health() -> dict[str, Any]:
        import httpx

        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get("http://127.0.0.1:8080/health")
                if response.status_code != 200:
                    return {"endpoint_healthy": False, "model_match": None}
                try:
                    props = client.get("http://127.0.0.1:8080/props", timeout=2.0).json()
                    model_path = props.get("model_path", "")
                except (ValueError, httpx.HTTPError):
                    model_path = ""
            from scripts.phase5_compare import frozen_models

            winner = next(m["filename"] for m in frozen_models() if m["name"] == "medgemma-1.5-4b-it")
            return {"endpoint_healthy": True,
                    "model_match": model_path.endswith(winner) if model_path else None}
        except httpx.HTTPError:
            return {"endpoint_healthy": False, "model_match": None}


    @app.get("/api/server/status")
    def server_status() -> dict[str, Any]:
        proc = _managed_proc(app)
        health = _endpoint_health()
        return {"running": bool(health["endpoint_healthy"]),
                "managed": proc is not None,
                "model": getattr(app.state, "server_model", None),
                **health}

    @app.post("/api/server/start")
    def server_start() -> dict[str, Any]:
        import subprocess

        from scripts.phase5_compare import wait_ready
        from scripts.phase5_synthesize import preflight_winner

        if _managed_proc(app) is not None:
            return {"running": True, "managed": True}
        if _endpoint_health()["endpoint_healthy"]:
            return {"running": True, "managed": False,
                    "note": "synthesis server already listening; not started here"}
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
        proc = _managed_proc(app)
        if proc is not None:
            proc.terminate()
            app.state.server_proc = None
            app.state.server_model = None
        return server_status()

    @app.post("/api/jobs/synthesize")
    def synthesize_job(body: dict[str, Any]) -> dict[str, Any]:
        from scripts.phase5_packet import PacketError, validate_packet

        try:
            packet = validate_packet(body["packet"])
        except (KeyError, PacketError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="bad packet")
        job_id = _start_job(lambda jid: _run_synthesis_job(app, jobs, lock, jid, packet))
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

    def _evict_jobs() -> None:
        terminal = [key for key, job in jobs.items()
                    if job["state"] in ("done", "failed", "cancelled")]
        for key in terminal[: max(0, len(terminal) + 1 - 128)]:
            del jobs[key]

    def _start_job(fn: Any, *args: Any, metadata: dict[str, Any] | None = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        with lock:
            _evict_jobs()
            jobs[job_id] = {"id": job_id, "state": "queued", "progress": 0.0}
            if metadata:
                jobs[job_id].update(metadata)
        worker = threading.Thread(target=fn, args=(job_id, *args), daemon=True)
        with lock:
            jobs[job_id]["state"] = "running"
        worker.start()
        return job_id

    def _v1_dir(specimen_id: str) -> Path:
        if not is_v1_specimen(specimen_id):
            raise HTTPException(status_code=404, detail="unknown v1 specimen")
        return ROOT / "artifacts" / "v1_specimens" / specimen_id

    @app.post("/api/specimens/import")
    async def specimen_import(file: UploadFile) -> Any:
        import shutil
        import tempfile

        from scripts import v1_ingest as ingest

        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg"):
            raise HTTPException(status_code=422, detail="only PNG/JPEG stills")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmppath = Path(tmp.name)
        try:
            record = ingest.ingest_file(tmppath, source_filename=file.filename or "upload")
        except ingest.IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        finally:
            tmppath.unlink(missing_ok=True)
        return record

    @app.post("/api/v1/analyze/{specimen_id}")
    def v1_analyze(specimen_id: str) -> Any:
        _v1_dir(specimen_id)

        def worker(job_id: str) -> None:
            from scripts import v1_vision as vision

            try:
                with lock:
                    if jobs[job_id]["state"] == "cancelled":
                        return
                    jobs[job_id]["progress"] = 0.3
                result = vision.analyze_specimen(specimen_id)
                with lock:
                    if jobs[job_id]["state"] == "cancelled":
                        return
                    jobs[job_id].update({"state": "done", "progress": 1.0,
                                         "result": {"findings": len(result["findings"]),
                                                    "detections": result["detections"]}})
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                with lock:
                    jobs[job_id].update({"state": "failed",
                                         "error": f"{type(exc).__name__}: {exc}"})

        return {"job_id": _start_job(worker)}

    @app.get("/api/v1/findings/{specimen_id}")
    def v1_findings(specimen_id: str) -> Any:
        from scripts import v1_findings as findings

        directory = _v1_dir(specimen_id)
        if not (directory / "vision.json").is_file():
            raise HTTPException(status_code=409, detail="specimen not analyzed yet")
        return {"findings": findings.effective_findings(directory)}

    @app.post("/api/v1/reviews")
    def v1_reviews(body: dict[str, Any]) -> Any:
        from scripts import v1_findings as findings

        try:
            directory = _v1_dir(body["specimen_id"])
            record = findings.record_review(
                directory, body["finding_id"], body["action"],
                changes=body.get("changes"),
                reviewer=body.get("reviewer", "local"),
                reason=body.get("reason"))
        except (KeyError, TypeError, findings.FindingError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return record

    @app.post("/api/v1/reviews/bulk")
    def v1_reviews_bulk(body: dict[str, Any]) -> Any:
        from scripts import v1_findings as findings

        try:
            directory = _v1_dir(body["specimen_id"])
            action = body.get("action", "confirm")
            if action != "confirm":
                raise findings.FindingError("bulk action supports confirm only")
            reviewer = body.get("reviewer", "local")
            confirmed = []
            for finding in findings.effective_findings(directory):
                if finding["review_state"] != "unreviewed":
                    continue
                findings.record_review(
                    directory, finding["finding_id"], "confirm",
                    reviewer=reviewer, reason="bulk auto-approval")
                confirmed.append(finding["finding_id"])
        except (KeyError, TypeError, findings.FindingError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"confirmed": confirmed}

    @app.post("/api/v1/retrieve/{specimen_id}")
    def v1_retrieve(specimen_id: str) -> Any:
        from scripts import v1_findings as findings
        from scripts import v1_retrieve as retrieval

        directory = _v1_dir(specimen_id)
        if not (directory / "vision.json").is_file():
            raise HTTPException(status_code=409, detail="specimen not analyzed yet")
        try:
            confirmed = findings.confirmed_findings(directory)
            if not confirmed:
                raise HTTPException(status_code=409, detail="no confirmed findings")
            sets = retrieval.retrieve_for_findings(confirmed)
            retrieval.save_evidence(directory, sets)
        except retrieval.RetrievalError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return {"sets": sets,
                "evidence": sum(len(group["evidence"]) for group in sets.values())}

    @app.post("/api/v1/evidence")
    def v1_evidence(body: dict[str, Any]) -> Any:
        from scripts import v1_retrieve as retrieval

        try:
            directory = _v1_dir(body["specimen_id"])
            if body.get("op") == "exclude":
                retrieval.exclude_evidence(directory, body["finding_id"], body["evidence_id"],
                                           reviewer=body.get("reviewer", "local"))
                return {"excluded": body["evidence_id"]}
            if body.get("op") == "add":
                return retrieval.add_manual_evidence(directory, body["finding_id"], body["go_id"],
                                                     reviewer=body.get("reviewer", "local"))
        except (KeyError, TypeError, retrieval.RetrievalError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        raise HTTPException(status_code=422, detail="unknown evidence op")

    @app.post("/api/v1/synthesize/{specimen_id}")
    def v1_synthesize(specimen_id: str) -> Any:
        from scripts import v1_findings as findings
        from scripts import v1_retrieve as retrieval
        from scripts.phase5_packet import build_packet, canonical_bytes

        directory = _v1_dir(specimen_id)
        if not (directory / "vision.json").is_file():
            raise HTTPException(status_code=409, detail="specimen not analyzed yet")
        confirmed = findings.confirmed_findings(directory)
        if not confirmed:
            raise HTTPException(status_code=409, detail="no confirmed findings")
        try:
            packet = build_packet(
                f"v1-{specimen_id}", "phase3-txl",
                findings.to_packet_findings(confirmed),
                retrieval.to_packet_context(directory),
                ["image-level-only", "no-patient-linkage"])
        except (ValueError, OSError, retrieval.RetrievalError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        packet_sha256 = sha256_bytes(canonical_bytes(packet))
        # A browser reload loses its polling timer while the worker keeps
        # running.  Reusing an identical active job makes a subsequent user
        # click attach to that work instead of submitting a concurrent
        # llama-server request (which can otherwise fail with HTTP 500).
        with lock:
            for active_id, active in jobs.items():
                if (active.get("kind") == "v1-synthesis"
                        and active.get("specimen_id") == specimen_id
                        and active.get("packet_sha256") == packet_sha256
                        and active.get("state") in ("queued", "running")):
                    return {"job_id": active_id}

        def worker(job_id: str) -> None:
            from scripts.phase5_synthesize import (
                SynthesisProductionError,
                synthesize_packet,
            )

            try:
                with lock:
                    if jobs[job_id]["state"] == "cancelled":
                        return
                    jobs[job_id]["progress"] = 0.3
                result = synthesize_packet(packet, "http://127.0.0.1:8080")
                with lock:
                    if jobs[job_id]["state"] == "cancelled":
                        return
                    if result["status"] == "ok":
                        (directory / "synthesis.json").write_text(
                            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    jobs[job_id].update({"state": "done" if result["status"] == "ok" else "failed",
                                         "progress": 1.0, "result": result,
                                         "error": result.get("failure")})
            except (SynthesisProductionError, OSError, ValueError, RuntimeError, ImportError) as exc:
                with lock:
                    jobs[job_id].update({"state": "failed",
                                         "error": f"{type(exc).__name__}: {exc}"})

        return {"job_id": _start_job(
            worker, metadata={"kind": "v1-synthesis", "specimen_id": specimen_id,
                              "packet_sha256": packet_sha256})}

    @app.post("/api/v1/signoff/{specimen_id}")
    def v1_signoff(specimen_id: str, body: dict[str, Any]) -> Any:
        from scripts import v1_findings as findings
        from scripts import v1_retrieve as retrieval
        from scripts.phase5_packet import build_packet, canonical_bytes, packet_digest

        directory = _v1_dir(specimen_id)
        synthesis_path = directory / "synthesis.json"
        if not synthesis_path.is_file():
            raise HTTPException(status_code=409, detail="nothing synthesized to sign")
        synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
        if synthesis.get("status") != "ok" or "validated" not in synthesis:
            raise HTTPException(status_code=409, detail="no valid synthesis to sign")
        current = build_packet(
            f"v1-{specimen_id}", "phase3-txl",
            findings.to_packet_findings(findings.confirmed_findings(directory)),
            retrieval.to_packet_context(directory),
            ["image-level-only", "no-patient-linkage"])
        if sha256_bytes(canonical_bytes(current)) != packet_digest(synthesis["packet"]):
            raise HTTPException(status_code=409,
                                detail="findings or evidence changed since synthesis; re-synthesize")
        record = {"schema": "v1-signoff-v1", "specimen_id": specimen_id,
                  "packet_sha256": packet_digest(synthesis["packet"]),
                  "note_sha256": sha256_bytes(synthesis["note"].encode()),
                  "reviewer": body.get("reviewer", "local"),
                  "note": body.get("note"), "unix_time": time.time()}
        (directory / "signoff.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (directory / "signoffs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    @app.get("/api/v1/export/{specimen_id}")
    def v1_export(specimen_id: str) -> Any:
        directory = _v1_dir(specimen_id)
        bundle: dict[str, Any] = {"schema": "v1-export-v1", "specimen_id": specimen_id}
        for name in ("specimen.json", "vision.json", "evidence.json", "synthesis.json", "signoff.json"):
            path = directory / name
            if path.is_file():
                bundle[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
        bundle["reviews"] = [json.loads(line) for line in
                             (directory / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
                             if line.strip()] if (directory / "reviews.jsonl").is_file() else []
        return JSONResponse(bundle)

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
