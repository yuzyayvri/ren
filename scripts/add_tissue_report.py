"""Add fold2 tissue-stratified metrics to the completed score report."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from evaluate_hover_checkpoint import fold2_arrays
from run_score_adjustment import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    with (args.artifact / "score_cache.pkl").open("rb") as handle:
        records = pickle.load(handle)
    types = np.load("data/tissue/fold2/Fold 2/images/fold2/types.npy", mmap_mode="r")
    report_path = args.artifact / "report.json"
    report = json.loads(report_path.read_text())
    configs = {
        "baseline": np.zeros(5),
        "selected_dead_only": np.asarray(report["selected_dead_only"]["offsets"]),
        "selected_multiclass": np.asarray(report["stage_b"]["selected_multiclass"]["offsets"]),
    }
    tissue = {}
    for label in sorted(set(types.tolist())):
        subset = [r for r in records if types[r["patch_index"]] == label]
        tissue[str(label)] = {
            "patch_support": len(subset),
            "evaluable_nucleus_support": int(sum(len(r["truth_types"]) for r in subset)),
            "configurations": {name: evaluate(subset, offsets) for name, offsets in configs.items()},
        }
    report["tissue_level"] = tissue
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
