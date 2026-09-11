"""Read-only metadata audit for the Phase 3 blood datasets.

The audit deliberately avoids decoding image pixels. It records file identities,
annotation schemas, label counts, split counts, and available grouping metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def txl_audit(root: Path) -> dict:
    dataset = root / "TXL-PBC"
    result = {"root": str(dataset), "splits": {}, "classes": [], "metadata_files": []}
    result["classes"] = (dataset / "classes.txt").read_text().splitlines()
    for path in sorted(dataset.iterdir()):
        if path.suffix.lower() in {".xlsx", ".pdf", ".txt", ".yaml"}:
            result["metadata_files"].append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        images = sorted(image_dir.glob("*.png"))
        labels = sorted(label_dir.glob("*.txt"))
        class_counts: Counter[str] = Counter()
        malformed = 0
        for label in labels:
            for line in label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5:
                    malformed += 1
                elif fields[0].isdigit():
                    class_counts[fields[0]] += 1
        result["splits"][split] = {
            "images": len(images), "label_files": len(labels),
            "class_counts": dict(sorted(class_counts.items())),
            "malformed_rows": malformed,
            "unmatched_images": sorted(p.stem for p in images if not (label_dir / f"{p.stem}.txt").exists()),
            "unmatched_labels": sorted(p.stem for p in labels if not (image_dir / f"{p.stem}.png").exists()),
        }
    result["duplicate_image_hashes"] = _duplicate_hashes(dataset / "images")
    return result


def _duplicate_hashes(root: Path) -> dict:
    by_hash: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            by_hash.setdefault(sha256(path), []).append(str(path))
    return {key: value for key, value in by_hash.items() if len(value) > 1}


def aml_audit(root: Path) -> dict:
    result = {"root": str(root), "annotation_files": {}, "class_directories": {}, "grouping_identifiers": []}
    for name in ("annotations.dat", "annotations_augmented.dat", "abbreviations.txt"):
        path = root / name
        result["annotation_files"][name] = {"lines": len(path.read_text().splitlines()), "sha256": sha256(path), "bytes": path.stat().st_size}
    data = root / "data" / "data"
    for directory in sorted(p for p in data.iterdir() if p.is_dir()):
        result["class_directories"][directory.name] = sum(1 for p in directory.rglob("*") if p.is_file())
    result["grouping_identifiers"] = ["none observed in directory names or annotation rows; image path/class is the available grouping"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txl-root", type=Path, required=True)
    parser.add_argument("--aml-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"txl_pbc": txl_audit(args.txl_root), "munich_aml": aml_audit(args.aml_root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
