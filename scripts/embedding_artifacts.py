"""Validated, provenance-preserving storage for batched embedding extraction."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

FEATURE_DIM = 1280


def image_hashes(images: np.ndarray) -> np.ndarray:
    """Return a stable hash for every source image in row order."""
    return np.asarray(
        [hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images],
        dtype="U64",
    )


def array_hash(array: np.ndarray) -> np.ndarray:
    """Hash an array's dtype, shape, and contiguous contents."""
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return np.asarray(digest.hexdigest())


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.save(tmp, array)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_batch(path: Path, start: int, embeddings: np.ndarray, images: np.ndarray) -> None:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    indices = np.arange(start, start + len(embeddings), dtype=np.int64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(indices):
        raise ValueError(f"invalid embedding batch shape {embeddings.shape}")
    if len(images) != len(indices):
        raise ValueError("image and embedding batch lengths differ")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding batch contains non-finite values")
    _atomic_savez(
        path,
        indices=indices,
        embeddings=embeddings,
        image_sha256=image_hashes(images),
    )


def batch_matches_source(
    path: Path, start: int, end: int, images: np.ndarray, feature_dim: int
) -> bool:
    """Return whether a resumable batch is complete and tied to this source slice."""
    try:
        with np.load(path) as batch:
            return bool(
                set(batch.files) == {"indices", "embeddings", "image_sha256"}
                and np.array_equal(batch["indices"], np.arange(start, end, dtype=np.int64))
                and batch["embeddings"].shape == (end - start, feature_dim)
                and batch["embeddings"].dtype == np.float32
                and np.isfinite(batch["embeddings"]).all()
                and np.array_equal(batch["image_sha256"], image_hashes(images))
            )
    except (OSError, ValueError, KeyError):
        return False


def publish_batches(
    out_dir: Path,
    source_images: Path,
    source_labels: Path,
    expected_feature_dim: int | None = None,
) -> tuple[int, int]:
    """Validate exact row coverage and atomically publish canonical artifacts.

    Batch files are retained after publication so provenance can be re-audited.
    Legacy ``embeddings_batch_*.npy`` files are deliberately ignored.
    """
    images = np.load(source_images, mmap_mode="r")
    labels = np.load(source_labels, allow_pickle=True)
    if len(images) != len(labels):
        raise ValueError(f"source image/label lengths differ: {len(images)} != {len(labels)}")

    files = sorted(out_dir.glob("embeddings_batch_*_*.npz"))
    if not files:
        raise ValueError(f"no index-bearing batch files found in {out_dir}")

    embeddings = None
    seen = np.zeros(len(images), dtype=bool)
    hashes = np.empty(len(images), dtype="U64")
    for path in files:
        with np.load(path) as batch:
            required = {"indices", "embeddings", "image_sha256"}
            if set(batch.files) != required:
                raise ValueError(f"{path.name}: fields must be exactly {sorted(required)}")
            indices = batch["indices"]
            feats = batch["embeddings"]
            batch_hashes = batch["image_sha256"]
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{path.name}: invalid indices")
        if len(indices) == 0 or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"{path.name}: indices are not one contiguous interval")
        if indices[0] < 0 or indices[-1] >= len(images):
            raise ValueError(f"{path.name}: indices outside source range")
        if seen[indices].any():
            overlap = indices[seen[indices]].tolist()
            raise ValueError(f"{path.name}: overlapping indices {overlap[:8]}")
        if feats.ndim != 2 or feats.shape[0] != len(indices) or feats.dtype != np.float32:
            raise ValueError(f"{path.name}: invalid embeddings shape/dtype {feats.shape}/{feats.dtype}")
        if expected_feature_dim is not None and feats.shape[1] != expected_feature_dim:
            raise ValueError(f"{path.name}: feature dimension {feats.shape[1]}, expected {expected_feature_dim}")
        if embeddings is None:
            embeddings = np.empty((len(images), feats.shape[1]), dtype=np.float32)
        elif feats.shape[1] != embeddings.shape[1]:
            raise ValueError(f"{path.name}: inconsistent feature dimension {feats.shape[1]}")
        if not np.isfinite(feats).all():
            raise ValueError(f"{path.name}: non-finite embeddings")
        expected_hashes = image_hashes(images[indices])
        if not np.array_equal(batch_hashes, expected_hashes):
            raise ValueError(f"{path.name}: source-image provenance mismatch")
        embeddings[indices] = feats
        hashes[indices] = batch_hashes
        seen[indices] = True

    missing = np.flatnonzero(~seen)
    if len(missing):
        raise ValueError(f"missing {len(missing)} source rows; first indices: {missing[:8].tolist()}")

    assert embeddings is not None
    provenance = {
        "indices": np.arange(len(images), dtype=np.int64),
        "image_sha256": hashes,
        "embeddings_sha256": array_hash(embeddings),
        "labels_sha256": array_hash(labels),
    }
    # provenance.npz is the commit marker and is replaced last. Its content
    # hashes make an interrupted multi-file publication fail verification.
    _atomic_save(out_dir / "embeddings.npy", embeddings)
    _atomic_save(out_dir / "labels.npy", labels)
    _atomic_savez(out_dir / "provenance.npz", **provenance)
    return embeddings.shape
