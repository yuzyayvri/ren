import numpy as np
import pytest

from scripts.phase3_recovery_executor import (
    authorize,
    capture_once,
    capture_production,
    require_reproduction,
    score_committed,
)


def pred(): return {'boxes':np.array([[0,0,1,1]],np.float32),'classes':np.array([0],np.int8),'confs':np.array([.8],np.float32),'offsets':np.array([0,1])}
def test_authorize_and_capture_single_use(tmp_path):
 authorize(tmp_path,{'checkpoint':'x'});capture_once(tmp_path,['a'],pred(),{})
 with pytest.raises(RuntimeError):capture_once(tmp_path,['a'],pred(),{})
def test_reproduction_mismatch_fails_closed():
 with pytest.raises(RuntimeError):require_reproduction({'macro_f1':.1},{'macro_f1':.2})


def test_production_capture_uses_manifest_order_and_empty_cases(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"txl_pbc":{"splits":{"test":[{"id":"a","image":"a"},{"id":"b","image":"b"}]}}}')
    seen = []
    def predictor(path):
        seen.append(path)
        return (np.array([[0, 0, 1, 1]], dtype=np.float32), np.array([0], dtype=np.int8), np.array([.8], dtype=np.float32)) if path == "a" else (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float32))
    capture_production(tmp_path / "cache", "checkpoint.pt", manifest, {"checkpoint": "x"}, predictor=predictor)
    assert seen == ["a", "b"]


def test_score_requires_committed_cache_and_publishes_ap(tmp_path):
    capture_once(tmp_path, ["a"], pred(), {})
    records = [("a", [(0, (0, 0, 1, 1))], [(0, (0, 0, 1, 1), .8)])]
    expected = {"macro_f1": 1 / 3, "min_recall": 0.0, "overall_recall": 1.0, "per_class_recall": [1.0, 0.0, 0.0], "tp": 1, "fp": 0, "fn": 0}
    report = score_committed(tmp_path, records, expected, {"cache": "x"})
    assert report["macro_ap50"] > 0


def test_score_mismatch_publishes_no_report(tmp_path):
    capture_once(tmp_path, ["a"], pred(), {})
    records = [("a", [(0, (0, 0, 1, 1))], [(0, (0, 0, 1, 1), .8)])]
    with pytest.raises(RuntimeError): score_committed(tmp_path, records, {"macro_f1": 0}, {})
    assert not (tmp_path / "recovered_report.json").exists()


def test_aggregate_schema_matches_original_reproduction_keys():
    from scripts.phase3_recovery_executor import _aggregate

    records = [("a", [(0, (0, 0, 1, 1))], [(0, (0, 0, 1, 1), .8)])]
    actual = _aggregate(records, .65)
    expected = {"macro_f1": 1 / 3, "min_recall": 0.0, "overall_recall": 1.0, "per_class_recall": [1.0, 0.0, 0.0], "tp": 1, "fp": 0, "fn": 0}
    require_reproduction(actual, expected)
