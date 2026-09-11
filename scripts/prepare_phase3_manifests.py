"""Build fail-closed, immutable Phase 3 source and split manifests.

This is the only pre-training entry point for Phase 3 data.  It hashes source
bytes, preserves TXL's supplied roles, and creates the deterministic AML
image-level split from original eligible images only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 20260909
POSITIVE = {"MYO", "MOB"}
EXCLUDED = {"UNC", "nan"}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def txl_manifest(root: Path) -> dict:
    ds = root / "TXL-PBC"
    raw_classes = [x.strip() for x in (ds / "classes.txt").read_text().splitlines() if x.strip()]
    classes = ["WBC", "RBC", "Platelets"]
    if len(raw_classes) != 3 or not all(any(token in x for token in ("WBC", "RBC", "Platelet")) for x, token in zip(raw_classes, ("WBC", "RBC", "Platelet"))):
        raise ValueError(f"unexpected TXL classes: {classes}")
    out = {"dataset": "TXL-PBC", "classes": classes, "splits": {}}
    for split in ("train", "val", "test"):
        imdir, labdir = ds / "images" / split, ds / "labels" / split
        rows = []
        for image in sorted(imdir.glob("*.png")):
            label = labdir / (image.stem + ".txt")
            if not label.exists():
                raise ValueError(f"missing TXL label: {label}")
            rows.append({"id": image.stem, "image": str(image), "image_sha256": digest(image),
                         "label": str(label), "label_sha256": digest(label)})
        if not rows:
            raise ValueError(f"empty TXL split {split}")
        out["splits"][split] = rows
    hashes = {}
    for split, rows in out["splits"].items():
        for row in rows:
            hashes.setdefault(row["image_sha256"], []).append(split + ":" + row["id"])
    dup = {k: v for k, v in hashes.items() if len(v) > 1}
    if dup:
        raise ValueError(f"cross-split TXL duplicate images: {list(dup)[:3]}")
    return out

def aml_manifest(root: Path) -> dict:
    annotation = root / "annotations.dat"
    data = root / "data" / "data"
    rows = []
    for line in annotation.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"malformed AML annotation: {line!r}")
        rel, label = fields[0], fields[1]
        if label in EXCLUDED:
            continue
        if label not in {"BAS","EBO","EOS","KSC","LYA","LYT","MOB","MON","MMZ","MYB","MYO","NGB","NGS","PMB","PMO"}:
            raise ValueError(f"unknown AML label: {label}")
        image = data / rel
        if not image.exists():
            raise ValueError(f"missing AML image: {image}")
        rows.append({"id": rel, "image": str(image), "label": label,
                     "binary_label": int(label in POSITIVE), "image_sha256": digest(image)})
    by_hash = {}
    for row in rows:
        by_hash.setdefault(row["image_sha256"], []).append(row["id"])
    duplicates = {k: v for k, v in by_hash.items() if len(v) > 1}
    if duplicates:
        raise ValueError(f"duplicate AML image bytes: {list(duplicates)[:3]}")
    rng = random.Random(SEED)
    groups = {0: [], 1: []}
    for row in rows: groups[row["binary_label"]].append(row)
    splits = {"train": [], "val": [], "test": []}
    for label_rows in groups.values():
        rng.shuffle(label_rows)
        n = len(label_rows); ntr = int(n * 0.70); nva = int(n * 0.15)
        splits["train"].extend(label_rows[:ntr]); splits["val"].extend(label_rows[ntr:ntr+nva]); splits["test"].extend(label_rows[ntr+nva:])
    for rows_ in splits.values(): rows_.sort(key=lambda r: r["id"])
    return {"dataset": "Munich-AML", "seed": SEED, "label_mapping": {"positive": sorted(POSITIVE), "negative": "all other named classes", "excluded": sorted(EXCLUDED)}, "rows": rows, "splits": splits, "split_policy": "image-level; not patient/specimen-independent"}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--txl-root", type=Path, required=True); p.add_argument("--aml-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    txl, aml = txl_manifest(a.txl_root), aml_manifest(a.aml_root)
    atomic_json(a.output, {"protocol": "phase3_protocol_v1", "seed": SEED, "txl_pbc": txl, "munich_aml": aml})

if __name__ == "__main__": main()
