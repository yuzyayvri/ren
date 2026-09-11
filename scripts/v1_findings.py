"""v1 production finding contract and reviewer correction lifecycle.

Machine findings are immutable once written. Human decisions live in a
separate append-only review log linked by finding ID. Only effective
reviewed findings (confirmed as-is or corrected) may enter retrieval;
rejected findings stay auditable but excluded.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ROOT = Path(__file__).resolve().parents[1]

PACKET_KEYS = ("finding_id", "label", "count", "confidence", "qualifier")
ACTIONS = ("confirm", "correct", "reject")


class FindingError(ValueError):
    """Invalid finding or review transition."""


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def enrich_findings(vision_result: dict[str, Any], *, run_id: str | None = None) -> list[dict[str, Any]]:
    run = run_id or uuid.uuid4().hex[:12]
    stamped = time.time()
    models = vision_result.get("models", {})
    enriched = []
    for finding in vision_result.get("findings", []):
        enriched.append({
            **finding,
            "specimen_sha256": vision_result.get("specimen_sha256"),
            "model": {"detector_sha256": models.get("detector", {}).get("sha256"),
                      "head_sha256": models.get("head", {}).get("sha256")},
            "run_id": run,
            "created_unix": stamped,
            "review_state": "unreviewed",
            "evidence_ids": [],
        })
    return enriched


def to_packet_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packet = []
    for finding in findings:
        entry = {key: finding[key] for key in PACKET_KEYS if key in finding and finding[key] is not None}
        entry.setdefault("qualifier", "observed")
        packet.append(entry)
    return packet


def record_review(specimen_dir: Path, finding_id: str, action: str, *,
                  changes: dict[str, Any] | None = None,
                  reviewer: str = "local", reason: str | None = None) -> dict[str, Any]:
    if action not in ACTIONS:
        raise FindingError(f"unknown review action: {action}")
    vision_path = specimen_dir / "vision.json"
    vision = json.loads(vision_path.read_text(encoding="utf-8"))
    originals = {f["finding_id"]: f for f in vision.get("findings", [])}
    if finding_id not in originals:
        raise FindingError(f"unknown finding: {finding_id}")
    if action == "correct" and not changes:
        raise FindingError("correction requires changed fields")
    allowed = {"label", "confidence", "qualifier", "region"}
    if changes and (unknown := set(changes) - allowed):
        raise FindingError(f"uncorrectable fields: {sorted(unknown)}")
    record = {
        "schema": "v1-review-v1",
        "finding_id": finding_id,
        "action": action,
        "original": originals[finding_id],
        "changes": changes or {},
        "reviewer": reviewer,
        "reason": reason,
        "reviewed_unix": time.time(),
        "review_id": uuid.uuid4().hex[:12],
    }
    with (specimen_dir / "reviews.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def read_reviews(specimen_dir: Path) -> list[dict[str, Any]]:
    path = specimen_dir / "reviews.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def effective_findings(specimen_dir: Path) -> list[dict[str, Any]]:
    vision = json.loads((specimen_dir / "vision.json").read_text(encoding="utf-8"))
    decisions: dict[str, dict[str, Any]] = {}
    for record in read_reviews(specimen_dir):
        decisions[record["finding_id"]] = record
    effective = []
    for finding in enrich_findings(vision):
        record = decisions.get(finding["finding_id"])
        if record is None or record["action"] == "reject":
            if record is None:
                finding["review_state"] = "unreviewed"
                effective.append(finding)
            continue
        finding = {**finding, **record["changes"]}
        finding["review_state"] = "confirmed" if record["action"] == "confirm" else "corrected"
        finding["review_id"] = record["review_id"]
        effective.append(finding)
    return effective


def confirmed_findings(specimen_dir: Path) -> list[dict[str, Any]]:
    return [f for f in effective_findings(specimen_dir) if f["review_state"] in ("confirmed", "corrected")]
