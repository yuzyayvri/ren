"""Pure, fold1/fold2-only verification helpers for the frozen context run."""
from __future__ import annotations

import numpy as np

EXPECTED_GEOMETRY = {
    "tp": 40894, "fp": 9608, "fn": 18475,
    "detection_f1": 0.7444002512036844,
    "binary_pq": 0.596356927353324,
}
FEATURE_ORDER = "148 existing nucleus features followed by 384 Path Foundation patch context"


def validate_embedding_rows(rows, patch_index, n_embeddings):
    """Validate already-selected rows (duplicates are legitimate here)."""
    rows = np.asarray(rows)
    patch_index = np.asarray(patch_index)
    if rows.ndim != 2 or patch_index.ndim != 1 or rows.shape[0] != len(patch_index):
        raise ValueError("embedding rows and patch index shape mismatch")
    if not np.issubdtype(patch_index.dtype, np.integer):
        raise ValueError("patch indices must be integers")
    if np.any(patch_index < 0) or np.any(patch_index >= n_embeddings):
        raise ValueError("patch index out of range")
    if rows.shape[1] != 384 or not np.isfinite(rows).all():
        raise ValueError("malformed or non-finite embedding rows")
    return rows


def join_patch_embeddings(base, added, embeddings, patch_index):
    base, added = np.asarray(base), np.asarray(added)
    if base.ndim != 2 or added.ndim != 2 or base.shape[0] != added.shape[0]:
        raise ValueError("feature row mismatch")
    embeddings = np.asarray(embeddings)
    patch_index = np.asarray(patch_index)
    if embeddings.ndim != 2 or embeddings.shape[1:] != (384,):
        raise ValueError("embedding matrix must have shape (n, 384)")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding matrix contains non-finite values")
    if patch_index.ndim != 1 or not np.issubdtype(patch_index.dtype, np.integer):
        raise ValueError("patch indices must be a one-dimensional integer array")
    if len(patch_index) != len(base):
        raise ValueError("patch index and feature rows length mismatch")
    if np.any(patch_index < 0) or np.any(patch_index >= len(embeddings)):
        raise ValueError("patch index out of range")
    selected = embeddings[patch_index]
    validate_embedding_rows(selected, patch_index, len(embeddings))
    return np.c_[base, added, selected].astype(np.float32, copy=False)


def verify_frozen_embedding(actual_sha, expected_sha, provenance_sha, expected_provenance_sha):
    if actual_sha != expected_sha or provenance_sha != expected_provenance_sha:
        raise ValueError("frozen embedding provenance mismatch")
    return True


def exact_geometry(metrics, atol=1e-15):
    for key in ("tp", "fp", "fn"):
        if metrics.get(key) != EXPECTED_GEOMETRY[key]:
            return False
    return (np.isfinite(metrics.get("detection_f1", np.nan)) and
            np.isfinite(metrics.get("binary_pq", np.nan)) and
            abs(metrics["detection_f1"] - EXPECTED_GEOMETRY["detection_f1"]) <= atol and
            abs(metrics["binary_pq"] - EXPECTED_GEOMETRY["binary_pq"]) <= atol)


def initialise_augmented_input(control_weight):
    control_weight = np.asarray(control_weight)
    if control_weight.ndim != 2 or control_weight.shape[1] != 148:
        raise ValueError("control first layer must be [hidden, 148]")
    out = np.zeros((control_weight.shape[0], 532), dtype=control_weight.dtype)
    out[:, :148] = control_weight
    return out


def choose_epoch(rows):
    eligible = [r for r in rows if r["dead_recall"] >= .20 and r["dead_f1"] >= .15]
    pool = eligible or list(rows)
    return max(pool, key=lambda r: (r["macro_f1"], r["matched_accuracy"], -r["loss"], -r["epoch"]))
