import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_phase2_appearance_context_cache import SCHEMA, load_dataset, sha


def make_cache(tmp_path, rows=None):
    rows = rows or [(0, 1, 7, 2), (0, 2, 7, 2)]
    root = tmp_path / "cache" / "fold1_truth"
    root.mkdir(parents=True)
    data = {
        "patch_index": np.array([r[0] for r in rows]), "instance_id": np.array([r[2] for r in rows]),
        "labels": np.array([r[1] for r in rows]), "tissue_label": np.array([r[3] for r in rows]),
        "base_features": np.ones((len(rows), 107), np.float32), "added_features": np.ones((len(rows), 41), np.float32),
        "width": np.array([148]),
    }
    shard = root / "shard_0000.npz"
    np.savez(shard, **data)
    meta = {"shard": 0, "patches": [0], "first_patch": 0, "last_patch": 0, "rows": len(rows),
            "schema": SCHEMA, "base_width": 107, "added_width": 41, "total_width": 148, "sha256": sha(shard)}
    (root / "shard_0000.json").write_text(json.dumps(meta))
    manifest = {"dataset": "fold1_truth", "expected_patch_coverage": [0, 0], "actual_patch_coverage": [0],
                "schema": SCHEMA, "widths": {"base": 107, "added": 41, "total": 148}, "total_rows": len(rows), "shards": [meta]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "cache"


def test_truth_identity_includes_class_scope(tmp_path):
    root = make_cache(tmp_path)
    assert len(load_dataset(root, "fold1_truth")["patch_index"]) == 2


def test_repeated_patch_can_contain_multiple_instances(tmp_path):
    root = make_cache(tmp_path, rows=[(0, 1, 7, 2), (0, 1, 8, 2), (0, 2, 7, 2)])
    data = load_dataset(root, "fold1_truth")
    assert len(data["patch_index"]) == 3
    assert list(zip(data["patch_index"], data["labels"], data["instance_id"])) == [(0, 1, 7), (0, 1, 8), (0, 2, 7)]


@pytest.mark.parametrize("mutation", ["duplicate", "coverage", "hash", "schema", "count"])
def test_fail_closed_integrity_regressions(tmp_path, mutation):
    root = make_cache(tmp_path)
    manifest = json.loads((root / "fold1_truth" / "manifest.json").read_text())
    shard = root / "fold1_truth" / "shard_0000.npz"
    meta = manifest["shards"][0]
    if mutation == "duplicate":
        with np.load(shard) as z: d = dict(z)
        d["labels"][1] = d["labels"][0]
        np.savez(shard, **d); meta["sha256"] = sha(shard); (root / "fold1_truth" / "shard_0000.json").write_text(json.dumps(meta))
    elif mutation == "coverage":
        meta["patches"] = [1]; meta["first_patch"] = meta["last_patch"] = 1
        (root / "fold1_truth" / "shard_0000.json").write_text(json.dumps(meta))
    elif mutation == "hash": meta["sha256"] = "0" * 64
    elif mutation == "schema": manifest["widths"]["total"] = 107
    elif mutation == "count": manifest["total_rows"] = 99
    manifest["shards"][0] = meta
    (root / "fold1_truth" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError): load_dataset(root, "fold1_truth")
