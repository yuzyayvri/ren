"""v1 automatic finding-to-retrieval wiring.

Reviewer-confirmed findings drive retrieval through a frozen deterministic
taxonomy-to-query mapping (no generative model). Every query, evidence
item, and selection decision records its origin, so automatic grounding
can never be confused with human-supplied semantics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ROOT = Path(__file__).resolve().parents[1]

DERIVATION_RULE = "v1-query-derivation-v1"
QUERY_MAP = {
    "WBC": "white blood cell leukocyte",
    "RBC": "red blood cell erythrocyte",
    "Platelets": "platelet thrombocyte",
}
TOP_K = 5
PACKET_PER_FINDING = 3


class RetrievalError(ValueError):
    """Retrieval precondition failure."""


def derive_query(finding: dict[str, Any]) -> dict[str, Any]:
    label = finding.get("label")
    if label not in QUERY_MAP:
        raise RetrievalError(f"no retrieval mapping for label: {label!r}")
    return {"query": QUERY_MAP[label], "rule": DERIVATION_RULE,
            "finding_id": finding.get("finding_id"),
            "qualifier": finding.get("qualifier", "observed")}


def retrieve_for_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    for finding in findings:
        if finding.get("review_state") not in ("confirmed", "corrected"):
            raise RetrievalError(
                f"finding {finding.get('finding_id')} is not reviewer-confirmed")
        derive_query(finding)
    from scripts.phase4_snapshot_reconciliation import load_bound_query

    engine = load_bound_query()
    import sqlite3

    from scripts.phase4_v3_common import V3 as _V3

    db = sqlite3.connect(f"file:{_V3 / 'ontology.sqlite'}?mode=ro", uri=True)
    try:
        sets: dict[str, Any] = {}
        for finding in findings:
            derived = derive_query(finding)
            entries = []
            for rank, go_id in enumerate(
                    engine.retrieve(derived["query"], mode="hybrid", k=TOP_K), start=1):
                row = db.execute(
                    "SELECT name, definition FROM terms WHERE id=? AND obsolete=0",
                    (go_id,)).fetchone()
                if row is None:
                    continue
                entries.append({
                    "evidence_id": f"E{len(entries) + 1}", "go_id": go_id,
                    "name": row[0], "definition": row[1], "rank": rank,
                    "mode": "hybrid", "query": derived["query"],
                    "origin": f"auto-{DERIVATION_RULE}",
                    "finding_id": finding["finding_id"],
                })
            sets[finding["finding_id"]] = {
                "query": derived["query"], "rule": DERIVATION_RULE,
                "qualifier": derived["qualifier"], "retrieved_unix": time.time(),
                "evidence": entries, "excluded": [], "manual_adds": [],
            }
        return sets
    finally:
        db.close()


def save_evidence(specimen_dir: Path, sets: dict[str, Any]) -> Path:
    path = specimen_dir / "evidence.json"
    path.write_text(json.dumps(sets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def add_manual_evidence(specimen_dir: Path, finding_id: str, go_id: str, *,
                        reviewer: str = "local") -> dict[str, Any]:
    import re

    if not re.fullmatch(r"GO:\d{7}", go_id or ""):
        raise RetrievalError(f"bad GO identifier: {go_id!r}")
    path = specimen_dir / "evidence.json"
    sets = json.loads(path.read_text(encoding="utf-8"))
    if finding_id not in sets:
        raise RetrievalError(f"unknown finding: {finding_id}")
    import sqlite3

    from scripts.phase4_v3_common import V3 as _V3

    db = sqlite3.connect(f"file:{_V3 / 'ontology.sqlite'}?mode=ro", uri=True)
    try:
        row = db.execute(
            "SELECT name, definition FROM terms WHERE id=? AND obsolete=0",
            (go_id,)).fetchone()
    finally:
        db.close()
    if row is None:
        raise RetrievalError(f"unknown GO term in sealed corpus: {go_id}")
    group = sets[finding_id]
    entry = {"evidence_id": f"M{len(group['manual_adds']) + 1}",
             "go_id": go_id, "name": row[0], "definition": row[1],
             "rank": len(group["evidence"]) + len(group["manual_adds"]) + 1,
             "mode": "hybrid", "query": group["query"],
             "origin": f"human:{reviewer}", "finding_id": finding_id}
    group["manual_adds"].append(entry)
    path.write_text(json.dumps(sets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry


def exclude_evidence(specimen_dir: Path, finding_id: str, evidence_id: str, *,
                     reviewer: str = "local") -> None:
    path = specimen_dir / "evidence.json"
    sets = json.loads(path.read_text(encoding="utf-8"))
    if finding_id not in sets:
        raise RetrievalError(f"unknown finding: {finding_id}")
    sets[finding_id]["excluded"].append({"evidence_id": evidence_id, "reviewer": reviewer})
    path.write_text(json.dumps(sets, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_evidence(specimen_dir: Path) -> list[dict[str, Any]]:
    sets = json.loads((specimen_dir / "evidence.json").read_text(encoding="utf-8"))
    selected = []
    for group in sets.values():
        excluded = {e["evidence_id"] for e in group.get("excluded", [])}
        for entry in group.get("evidence", []):
            if entry["evidence_id"] not in excluded:
                selected.append(entry)
        selected.extend(group.get("manual_adds", []))
    renumbered = []
    for number, entry in enumerate(selected, start=1):
        renumbered.append({**entry, "evidence_id": f"E{number}"})
    return renumbered


def to_packet_context(specimen_dir: Path, per_finding: int = PACKET_PER_FINDING) -> list[dict[str, Any]]:
    """Select packet evidence: top-ranked entries per finding.

    Retrieval keeps the full top-5 sets in evidence.json. The production
    packet carries only the top entries per finding so real specimens fit
    the frozen context window; the cap is deterministic and recorded.
    """
    from scripts.phase5_packet import GO_RE

    if not (specimen_dir / "evidence.json").is_file():
        raise RetrievalError("no retrieved evidence; run retrieval first")
    context = []
    kept: dict[str, int] = {}
    for entry in selected_evidence(specimen_dir):
        # Reviewer-added evidence is never silently dropped by the cap.
        if not str(entry.get("origin", "")).startswith("human:"):
            key = entry.get("finding_id", "")
            kept[key] = kept.get(key, 0) + 1
            if kept[key] > per_finding:
                continue
        if not GO_RE.fullmatch(entry["go_id"]):
            raise RetrievalError(f"bad GO identifier in evidence: {entry['go_id']!r}")
        for key in ("name", "definition"):
            if not entry.get(key):
                raise RetrievalError(f"manual evidence needs {key}: {entry['evidence_id']}")
        context.append({k: entry[k] for k in
                        ("evidence_id", "go_id", "name", "definition", "rank", "mode", "query")})
    return context
