"""v1 full-system acceptance: unseen specimen end to end, plus failure paths.

Exercises the complete primary workflow on blood-smear PNGs the production
path has never processed: ingest, analyze, review, retrieve, synthesize,
sign, export. Requires the MedGemma server on 127.0.0.1:8080. Asserts
software/provenance/chain correctness, never clinical accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ROOT = Path(__file__).resolve().parents[1]


class AcceptError(AssertionError):
    """Acceptance failure with evidence attached."""


def check(condition: bool, message: str, evidence: Any = None) -> None:
    if not condition:
        raise AcceptError(f"{message} | evidence={json.dumps(evidence, sort_keys=True)[:300]}")


def _chain_bundle(bundle: dict[str, Any]) -> None:
    specimen = bundle["specimen"]
    vision_findings = {f["finding_id"]: f for f in bundle.get("vision", {}).get("findings", [])}
    for record in bundle.get("reviews", []):
        check(record["finding_id"] in vision_findings or record["finding_id"] == "*",
              "review references unknown finding", record["finding_id"])
        check(record.get("review_id"), "review missing identity", record["finding_id"])
    packet_findings = {f["finding_id"] for f in bundle.get("synthesis", {}).get("packet", {}).get("findings", [])} if isinstance(bundle.get("synthesis"), dict) else set()
    for fid in packet_findings:
        check(fid in vision_findings, "packet finding not in vision output", fid)
    check(bundle["signoff"]["packet_sha256"] == bundle["synthesis"]["packet_sha256"],
          "signoff packet hash does not match synthesis input")
    check(bundle["specimen"]["content_sha256"] == specimen["content_sha256"], "specimen hash drift")


def accept_specimen(image: Path, *, workdir: Path, reviewer: str = "acceptance") -> dict[str, Any]:
    from scripts import v1_findings as findings
    from scripts import v1_ingest as ingest
    from scripts import v1_retrieve as retrieval
    from scripts import v1_vision as vision

    record = ingest.ingest_file(image, registry=workdir)
    directory = workdir / record["specimen_id"]
    vision_result = vision.analyze_specimen(record["specimen_id"], registry=workdir)
    check(bool(vision_result["findings"]), "expected findings on acceptance specimen",
          {"detections": vision_result["detections"]})
    for finding in vision_result["findings"]:
        findings.record_review(directory, finding["finding_id"], "confirm", reviewer=reviewer)
    confirmed = findings.confirmed_findings(directory)
    check(len(confirmed) == len(vision_result["findings"]), "confirmation lost findings", None)
    sets = retrieval.retrieve_for_findings(confirmed)
    retrieval.save_evidence(directory, sets)
    check(all(sets[fid]["evidence"] for fid in sets), "empty evidence set", None)

    from scripts.phase5_packet import build_packet
    from scripts.phase5_synthesize import synthesize_packet

    packet = build_packet(
        f"v1-{record['specimen_id']}", "phase3-txl",
        findings.to_packet_findings(confirmed),
        retrieval.to_packet_context(directory),
        ["image-level-only", "no-patient-linkage"])
    result = synthesize_packet(packet, "http://127.0.0.1:8080")
    check(result["status"] == "ok", "synthesis failed", result.get("failure"))
    (directory / "synthesis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    signoff = {"schema": "v1-signoff-v1", "specimen_id": record["specimen_id"],
               "packet_sha256": hashlib.sha256(
                   json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
               "note_sha256": hashlib.sha256(result["note"].encode()).hexdigest(),
               "reviewer": reviewer, "unix_time": time.time()}
    (directory / "signoff.json").write_text(
        json.dumps(signoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle = {"schema": "v1-export-v1", "specimen_id": record["specimen_id"],
              "specimen": record, "vision": vision_result,
              "reviews": findings.read_reviews(directory),
              "evidence": json.loads((directory / "evidence.json").read_text(encoding="utf-8")),
              "synthesis": {"packet": packet, "packet_sha256": signoff["packet_sha256"],
                            "note": result["note"],
                            "model": result["model"], "latency_s": result["latency_s"]},
              "signoff": signoff}
    _chain_bundle(bundle)
    return bundle


def accept_failures(workdir: Path) -> dict[str, int]:
    from scripts import v1_findings as findings
    from scripts import v1_ingest as ingest

    passed = 0
    total = 0

    def expect_reject(name: str, content: bytes, suffix: str) -> None:
        nonlocal passed, total
        total += 1
        path = workdir / f"{name}{suffix}"
        path.write_bytes(content)
        try:
            ingest.ingest_file(path, registry=workdir / "rej")
        except ingest.IngestError:
            passed += 1
        else:
            raise AcceptError(f"corrupt input accepted: {name}")

    expect_reject("text", b"this is not an image at all, just text" * 20, ".png")
    expect_reject("short", b"\x89PNG\r\n\x1a\ntiny", ".png")
    from PIL import Image

    tiny = workdir / "tiny.png"
    Image.new("RGB", (8, 8)).save(tiny)
    total += 1
    try:
        ingest.ingest_file(tiny, registry=workdir / "rej")
    except ingest.IngestError:
        passed += 1
    else:
        raise AcceptError("undersize image accepted")
    total += 1
    try:
        from scripts import v1_retrieve as retrieval

        retrieval.retrieve_for_findings(
            [{"finding_id": "F1", "label": "WBC", "review_state": "unreviewed"}])
    except retrieval.RetrievalError:
        passed += 1
    else:
        raise AcceptError("unconfirmed finding retrieved")
    total += 1
    try:
        findings.record_review(workdir, "F1", "confirm")
    except (findings.FindingError, OSError):
        passed += 1
    else:
        raise AcceptError("review on missing vision accepted")
    return {"passed": passed, "total": total}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "v1_acceptance")
    args = parser.parse_args()
    workdir = args.workdir
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    summary: dict[str, Any] = {"specimens": [], "failures": {}}
    for image in args.images:
        bundle = accept_specimen(image, workdir=workdir)
        summary["specimens"].append({
            "specimen_id": bundle["specimen_id"],
            "findings": len(bundle["vision"]["findings"]),
            "packet_sha256": bundle["signoff"]["packet_sha256"]})
    summary["failures"] = accept_failures(workdir)
    check(summary["failures"]["passed"] == summary["failures"]["total"], "failure paths leaked", summary["failures"])
    (workdir / "acceptance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
