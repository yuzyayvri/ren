"""Fail-closed verification of vision embeddings and their provenance."""

import argparse
import sys

import numpy as np
from embedding_artifacts import array_hash, image_hashes
from extract_embeddings import FOLDS, OUT_ROOT

FEATURE_DIMS = {"virchow": 1280, "path-foundation": 384}


def verify_fold(fold: str, backbone: str = "virchow") -> list[str]:
    errors = []
    base = OUT_ROOT / backbone / fold
    try:
        emb = np.load(base / "embeddings.npy")
        labels = np.load(base / "labels.npy", allow_pickle=True)
        source_labels = np.load(FOLDS[fold]["types"], allow_pickle=True)
        source_images = np.load(FOLDS[fold]["images"], mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - verifier reports corrupt/unreadable inputs
        return [f"artifact/source load failure: {exc}"]

    indices = hashes = None
    try:
        with np.load(base / "provenance.npz") as provenance:
            indices = provenance["indices"]
            hashes = provenance["image_sha256"]
            embeddings_hash = provenance["embeddings_sha256"]
            labels_hash = provenance["labels_sha256"]
    except Exception as exc:  # noqa: BLE001 - verifier reports corrupt/unreadable inputs
        errors.append(f"provenance load failure: {exc}")

    n = len(source_images)
    feature_dim = FEATURE_DIMS[backbone]
    if emb.shape != (n, feature_dim): errors.append(f"embedding shape {emb.shape}, expected {(n, feature_dim)}")
    if emb.dtype != np.float32: errors.append(f"embedding dtype {emb.dtype}, expected float32")
    if not np.isfinite(emb).all(): errors.append("embeddings contain non-finite values")
    if emb.ndim == 2 and len(emb) and np.any(np.all(emb == 0, axis=1)): errors.append("embeddings contain all-zero rows")
    if emb.ndim == 2 and len(emb) and np.any(np.ptp(emb, axis=0) == 0): errors.append("embeddings contain constant columns")
    if emb.ndim == 2 and len(np.unique(emb, axis=0)) != len(emb): errors.append("embeddings contain duplicate rows")
    if labels.shape != source_labels.shape or not np.array_equal(labels, source_labels):
        errors.append("labels do not equal source types.npy element-for-element")
    if len(np.unique(source_labels)) != 19: errors.append("source label class count is not 19")
    if indices is not None and not np.array_equal(indices, np.arange(n, dtype=np.int64)):
        errors.append("provenance indices are incomplete, duplicated, or out of order")
    if hashes is not None and (hashes.shape != (n,) or not np.array_equal(hashes, image_hashes(source_images))):
        errors.append("provenance hashes do not match source images in row order")
    if indices is not None and not np.array_equal(embeddings_hash, array_hash(emb)):
        errors.append("canonical embeddings do not match the committed provenance hash")
    if indices is not None and not np.array_equal(labels_hash, array_hash(labels)):
        errors.append("canonical labels do not match the committed provenance hash")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(FEATURE_DIMS), default="virchow")
    args = parser.parse_args()
    ok = True
    for fold in ("fold1", "fold2", "fold3"):
        errors = verify_fold(fold, args.backbone)
        ok = ok and not errors
        print(f"{args.backbone}/{fold}: {'FAIL' if errors else 'OK'}")
        for error in errors:
            print(f"  - {error}")
    print("\nALL FOLDS OK" if ok else "\nPROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
