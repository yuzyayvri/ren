"""Evaluate the declared small Dead/Connective follow-up on an existing cache."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from evaluate_hover_checkpoint import coverage_indices, fold2_arrays
from run_score_adjustment import evaluate
from score_adjustment import choose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    with (args.artifact / "score_cache.pkl").open("rb") as handle:
        records = pickle.load(handle)
    monitor_indices = set(coverage_indices(fold2_arrays()[1]))
    monitor = [record for record in records if record["patch_index"] in monitor_indices]
    entries = []
    # Class 5 is the reference (zero); only evidenced connective/dead bias is searched.
    for connective in (0.0, 0.2, 0.4):
        for dead in (-0.6, -0.4, -0.2, 0.0):
            offsets = np.array([0.0, 0.0, connective, dead, 0.0])
            entries.append({"offsets": offsets.tolist(), "metrics": evaluate(monitor, offsets)})
    selected = choose(entries)
    offsets = np.asarray(selected["offsets"], dtype=float)
    report_path = args.artifact / "report.json"
    report = json.loads(report_path.read_text())
    report["stage_b"] = {
        "grid": entries,
        "selection_objective": "Dead recall >= 0.20 and Dead F1 >= 0.15; maximize macro-F1, then Dead F1, matched typing accuracy, smaller absolute Dead offset",
        "selected_multiclass": selected,
        "reference_class_fixed_at_zero": 5,
        "focused_classes": {"connective": 3, "dead": 4},
    }
    report["full_fold2"]["selected_multiclass"] = evaluate(records, offsets)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["stage_b"], indent=2))
    print(json.dumps(report["full_fold2"]["selected_multiclass"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
