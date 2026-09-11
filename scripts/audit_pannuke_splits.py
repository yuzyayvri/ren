#!/usr/bin/env python3
"""Audit duplicate PanNuke image content without changing source arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pannuke_masks import fold_paths


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/tissue"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seen: dict[str, tuple[int, int]] = {}
    duplicate_pairs: list[dict[str, list[int]]] = []
    fold_counts: dict[str, int] = {}
    for fold in (1, 2, 3):
        images_path, _, _ = fold_paths(args.data_root, fold)
        images = np.load(images_path, mmap_mode="r")
        fold_counts[f"fold{fold}"] = len(images)
        for patch_index, image in enumerate(images):
            digest = image_digest(image)
            first = seen.setdefault(digest, (fold, patch_index))
            if first != (fold, patch_index):
                duplicate_pairs.append(
                    {"first": [first[0], first[1]], "duplicate": [fold, patch_index]}
                )

    report = {
        "schema": "pannuke-split-audit-v1",
        "fold_patch_counts": fold_counts,
        "exact_duplicate_image_pair_count": len(duplicate_pairs),
        "exact_duplicate_image_pairs": duplicate_pairs,
        "grouping": {
            "status": "unavailable",
            "reason": "Local PanNuke arrays and README contain no patient or slide identifiers; patient/slide independence cannot be assessed.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
