#!/usr/bin/env python3
"""Sequential, resumable Path Foundation extraction for all PanNuke folds."""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ren-matplotlib")

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from embedding_artifacts import batch_matches_source, publish_batches, save_batch
from extract_embeddings import FOLDS, OUT_ROOT

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "vision" / "path-foundation"
LOG_PATH = OUT_ROOT / "path-foundation" / "extract_all.log"
FEATURE_DIM = 384
BATCH_SIZE = 64


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def load_model():
    if not (MODEL_DIR / "saved_model.pb").exists():
        raise FileNotFoundError(f"Path Foundation SavedModel missing from {MODEL_DIR}")
    model = tf.saved_model.load(str(MODEL_DIR))
    inference = model.signatures.get("serving_default")
    if inference is None:
        raise RuntimeError("Path Foundation has no serving_default signature")
    output = inference(tf.zeros((1, 224, 224, 3), dtype=tf.float32))["output_0"]
    if tuple(output.shape) != (1, FEATURE_DIM) or output.dtype != tf.float32:
        raise RuntimeError(f"unexpected Path Foundation output {output.shape}/{output.dtype}")
    return model, inference


def extract_fold(fold: str, inference) -> None:
    meta = FOLDS[fold]
    source = np.load(meta["images"], mmap_mode="r")
    labels = np.load(meta["types"], allow_pickle=True)
    if len(source) != len(labels):
        raise ValueError(f"{fold}: source image/label lengths differ")
    out_dir = OUT_ROOT / "path-foundation" / fold
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"{fold}: {len(source)} images, {(len(source) + BATCH_SIZE - 1) // BATCH_SIZE} batches")

    started = time.time()
    for start in range(0, len(source), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(source))
        path = out_dir / f"embeddings_batch_{start}_{end}.npz"
        if path.exists() and batch_matches_source(path, start, end, source[start:end], FEATURE_DIM):
            continue
        # The model card specifies cropping (0, 0, 224, 224), RGB, and [0,1].
        batch = np.asarray(source[start:end, :224, :224, :], dtype=np.float32) / 255.0
        features = inference(inputs=tf.constant(batch))["output_0"].numpy()
        save_batch(path, start, features, source[start:end])
        elapsed = time.time() - started
        log(f"{fold}: [{start}:{end}] {100 * end / len(source):.1f}% — {end / elapsed:.1f} img/s")

    shape = publish_batches(out_dir, meta["images"], meta["types"], FEATURE_DIM)
    log(f"{fold}: validated and atomically published {shape}; batches retained")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=sorted(FOLDS))
    args = parser.parse_args()
    folds = [args.fold] if args.fold else sorted(FOLDS)
    log(f"Path Foundation extraction started for {folds}")
    _, inference = load_model()
    log("model loaded and one-image signature check passed")
    for fold in folds:
        extract_fold(fold, inference)
    log("ALL PATH FOUNDATION EXTRACTION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
