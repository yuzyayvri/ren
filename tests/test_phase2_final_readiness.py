"""Readiness constants are isolated from the generic production verifier."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = (
    ROOT / "artifacts/phase2_validity_filter_repair_v1/development_rehearsal_spec.json"
)


def test_fold_specific_expectations_live_only_in_rehearsal_spec():
    specification = json.loads(SPEC.read_text())
    assert specification["coverage"] == 2523
    assert specification["expected_ignored_only_accounting"] == {
        "total": 3151,
        "retained": 82,
        "rejected": 3069,
    }
    for path in (
        ROOT / "scripts/run_final_protocol.py",
        ROOT / "scripts/final_phase2_evaluation.py",
    ):
        source = path.read_text()
        assert "2523" not in source
        assert specification["expected_prediction_digest"] not in source


def test_rehearsal_metrics_remain_exactly_frozen():
    metrics = json.loads(SPEC.read_text())["expected_metrics"]
    assert metrics == {
        "detection_f1": 0.7442408090248622,
        "binary_pq": 0.5965614410791367,
        "end_to_end_macro_f1": 0.5693660649671026,
        "dead_recall": 0.43636363636363634,
        "dead_f1": 0.39689922480620154,
    }
