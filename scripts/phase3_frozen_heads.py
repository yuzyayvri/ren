"""Pure, deterministic Phase 3 crop/provenance and head utilities.

Heavy encoder execution is deliberately kept out of this module; callers pass
validated RGB arrays to the frozen Path Foundation service and persist outputs
with the source identities supplied here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def crop_txl(image: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("TXL crop requires uint8 RGB array")
    h, w, _ = image.shape
    x1, y1, x2, y2 = box_xyxy
    dx, dy = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    x1, y1, x2, y2 = max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))), min(w, int(np.ceil(x2 + dx))), min(h, int(np.ceil(y2 + dy)))
    if x2 <= x1 or y2 <= y1: raise ValueError("invalid crop")
    crop = image[y1:y2, x1:x2]
    side = max(crop.shape[:2]); med = np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8)
    out = np.broadcast_to(med, (side, side, 3)).copy(); oy, ox = (side-crop.shape[0])//2, (side-crop.shape[1])//2
    out[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = crop
    return np.asarray(Image.fromarray(out).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def preprocess_aml(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8: raise ValueError("AML input requires uint8 RGB array")
    h, w, _ = image.shape; side = max(h, w); med = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
    out = np.broadcast_to(med, (side, side, 3)).copy(); oy, ox = (side-h)//2, (side-w)//2; out[oy:oy+h, ox:ox+w] = image
    return np.asarray(Image.fromarray(out).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def normalize_for_path_foundation(batch: np.ndarray) -> np.ndarray:
    x = np.asarray(batch, dtype=np.float32) / 255.0
    if x.ndim != 4 or x.shape[1:] != (224, 224, 3) or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1: raise ValueError("invalid Path Foundation batch")
    return x


def atomic_embedding_cache(path: Path, embeddings: np.ndarray, identities: list[str], config_hash: str) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(identities) or embeddings.shape[1] != 384 or embeddings.dtype != np.float32: raise ValueError("invalid embedding cache")
    payload = {"identities": identities, "config_sha256": config_hash, "embedding_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest()}
    tmp = path.with_suffix(path.suffix + ".tmp.npz"); np.savez_compressed(tmp, embeddings=embeddings, identities=np.asarray(identities), metadata=json.dumps(payload, sort_keys=True)); tmp.replace(path)
