from __future__ import annotations

import json
from pathlib import Path


def test_phase3_source_manifest_is_complete_and_split_disjoint():
    path = Path("artifacts/phase3_protocol_v1/source_manifest.json")
    assert path.exists()
    data = json.loads(path.read_text())
    txl = data["txl_pbc"]["splits"]
    assert {k: len(v) for k, v in txl.items()} == {"train": 882, "val": 252, "test": 126}
    txl_ids = [r["id"] for rows in txl.values() for r in rows]
    assert len(txl_ids) == len(set(txl_ids))
    aml = data["munich_aml"]
    assert aml["seed"] == 20260909
    assert len(aml["rows"]) == 18365
    assert {k: len(v) for k, v in aml["splits"].items()} == {"train": 12854, "val": 2754, "test": 2757}
    split_ids = [r["id"] for rows in aml["splits"].values() for r in rows]
    assert len(split_ids) == len(set(split_ids)) == len(aml["rows"])
    assert aml["label_mapping"]["positive"] == ["MOB", "MYO"]
    assert aml["split_policy"] == "image-level; not patient/specimen-independent"

