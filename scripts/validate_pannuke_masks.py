#!/usr/bin/env python3
"""Validate PanNuke source-mask semantics and emit a small audit JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pannuke_masks import audit_fold, file_sha256, fold_paths, validate_mask_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/tissue"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema": "pannuke-mask-audit-v2",
        "folds": {},
        "semantic_alignment": {
            "status": "unverifiable",
            "reason": "PanNuke arrays contain no patient, slide, or source-image identifier; equal row counts only establish positional pairing.",
        },
    }
    for fold in args.fold or (1, 2, 3):
        images_path, masks_path, types_path = fold_paths(args.data_root, fold)
        if not all(path.is_file() for path in (images_path, masks_path, types_path)):
            raise FileNotFoundError(f"missing PanNuke source array(s) for fold{fold}")
        images = np.load(images_path, mmap_mode="r")
        masks = np.load(masks_path, mmap_mode="r")
        types = np.load(types_path, allow_pickle=False)
        if len(images) != len(types):
            raise ValueError(f"fold{fold}: images/types rows differ: {len(images)} != {len(types)}")
        validate_mask_array(masks, expected_rows=len(images))
        fold_report = audit_fold(masks)
        fold_report["image_shape"] = list(images.shape)
        fold_report["tissue_type_count"] = int(len(np.unique(types)))
        fold_report["source_sha256"] = {
            "images_npy": file_sha256(images_path),
            "masks_npy": file_sha256(masks_path),
            "types_npy": file_sha256(types_path),
        }
        report["folds"][f"fold{fold}"] = fold_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
