#!/usr/bin/env python3
"""
Extract Virchow embeddings for all three PanNuke folds.
Resumable: checks for existing output files and skips completed batches.
Designed to run as a long-lived background process via systemd.

Usage:
    python scripts/extract_virchow_all_folds.py [--fold fold1|fold2|fold3]

If run without --fold, extracts all three folds.
Logs progress to embeddings/virchow/extract_all.log.
"""
import os
import sys
import time
from pathlib import Path

# Hardcode env before any imports
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from embedding_artifacts import publish_batches, save_batch
from extract_embeddings import VirchowModel, _virchow_transform_batch

PROJECT_ROOT = Path('/home/yuzy/ren')
STATE_PATH = Path('/tmp/virchow_state.pt')
LOG_PATH = PROJECT_ROOT / 'embeddings' / 'virchow' / 'extract_all.log'

FOLDS = {
    'fold1': {
        'n_images': 2656,
        'img_path': PROJECT_ROOT / 'data' / 'tissue' / 'fold1' / 'Fold 1' / 'images' / 'fold1' / 'images.npy',
        'lbl_path': PROJECT_ROOT / 'data' / 'tissue' / 'fold1' / 'Fold 1' / 'images' / 'fold1' / 'types.npy',
        'outdir': PROJECT_ROOT / 'embeddings' / 'virchow' / 'fold1',
        'batch_size': 64,
    },
    'fold2': {
        'n_images': 2523,
        'img_path': PROJECT_ROOT / 'data' / 'tissue' / 'fold2' / 'Fold 2' / 'images' / 'fold2' / 'images.npy',
        'lbl_path': PROJECT_ROOT / 'data' / 'tissue' / 'fold2' / 'Fold 2' / 'images' / 'fold2' / 'types.npy',
        'outdir': PROJECT_ROOT / 'embeddings' / 'virchow' / 'fold2',
        'batch_size': 64,
    },
    'fold3': {
        'n_images': 2722,
        'img_path': PROJECT_ROOT / 'data' / 'tissue' / 'fold3' / 'Fold 3' / 'images' / 'fold3' / 'images.npy',
        'lbl_path': PROJECT_ROOT / 'data' / 'tissue' / 'fold3' / 'Fold 3' / 'images' / 'fold3' / 'types.npy',
        'outdir': PROJECT_ROOT / 'embeddings' / 'virchow' / 'fold3',
        'batch_size': 64,
    },
}


def log(msg, flush=True):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if flush:
        print(line, flush=True)


def load_model():
    log("Loading model state from /tmp/virchow_state.pt ...")
    state = torch.load(str(STATE_PATH), map_location='cpu', weights_only=True)
    model = VirchowModel()
    model.load_state_dict(state)
    model.eval()
    log(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model


def extract_fold(fold_name, model):
    cfg = FOLDS[fold_name]
    outdir = cfg['outdir']
    outdir.mkdir(parents=True, exist_ok=True)

    # Load images once
    log(f"{fold_name}: loading {cfg['n_images']} images ...")
    source_images = np.load(str(cfg['img_path']), mmap_mode='r')
    images = np.asarray(source_images)
    if images.dtype == np.float64:
        images = images.astype(np.float32) / 255.0
    log(f"{fold_name}: images loaded, dtype={images.dtype}, range=[{images.min():.4f}, {images.max():.4f}]")

    n = cfg['n_images']
    bs = cfg['batch_size']
    n_batches = (n + bs - 1) // bs
    log(f"{fold_name}: {n_batches} batches of {bs} images each")

    start_time = time.time()
    completed_batches = 0

    for batch_idx in range(n_batches):
        start = batch_idx * bs
        end = min(start + bs, n)
        outfile = outdir / f'embeddings_batch_{start}_{end}.npz'

        if outfile.exists():
            completed_batches += 1
            continue

        try:
            batch = images[start:end]
            batch_t = _virchow_transform_batch(batch)
            with torch.inference_mode():
                feats = model(batch_t)[:, 0, :].numpy()

            save_batch(outfile, start, feats, source_images[start:end])
            elapsed = time.time() - start_time
            rate = (end) / elapsed if elapsed > 0 else 0
            log(f"{fold_name}: batch {batch_idx}/{n_batches} [{start}:{end}] "
                f"({end}/{n} images, {100*end/n:.1f}%) — "
                f"{rate:.1f} img/s, {elapsed/60:.1f} min elapsed")
        except Exception as e:  # noqa: BLE001 - preserve any batch failure in the extraction log
            log(f"{fold_name}: ERROR at batch {batch_idx} [{start}:{end}]: {e}")
            import traceback
            traceback.print_exc()
            log(f"{fold_name}: stopping; completed batches remain resumable")
            return False

    # Publish only after validating exact index coverage and source-image provenance.
    log(f"{fold_name}: validating and publishing batch files ...")
    try:
        shape = publish_batches(outdir, cfg['img_path'], cfg['lbl_path'], 1280)
    except ValueError as exc:
        log(f"{fold_name}: NOT PUBLISHED — {exc}")
        return False
    size_mb = os.path.getsize(str(outdir / 'embeddings.npy')) / 1e6
    log(f"{fold_name}: atomically published embeddings.npy ({shape}, {size_mb:.1f} MB)")
    log(f"{fold_name}: retained batch files for provenance/audit")

    total_time = time.time() - start_time
    log(f"{fold_name}: COMPLETE — {total_time/60:.1f} min total, {n/total_time:.1f} img/s avg")
    return True


def main():
    log("=== Virchow all-folds extraction started ===")

    # Check state file
    if not STATE_PATH.exists():
        log(f"ERROR: model state not found at {STATE_PATH}. Run the state-caching step first.")
        sys.exit(1)

    # Determine which folds to run
    target_folds = []
    if len(sys.argv) > 1 and sys.argv[1] == '--fold':
        if len(sys.argv) > 2:
            target_folds = [sys.argv[2]]
        else:
            log("Usage: extract_virchow_all_folds.py [--fold fold1|fold2|fold3]")
            sys.exit(1)
    else:
        target_folds = ['fold1', 'fold2', 'fold3']

    # Always pass requested folds through batch validation/publication. Existing
    # index-bearing batches are skipped, so this remains resumable.
    remaining = []
    for fn in target_folds:
        outdir = FOLDS[fn]['outdir']
        emb_path = outdir / 'embeddings.npy'
        if emb_path.exists():
            emb = np.load(str(emb_path), allow_pickle=True)
            expected = FOLDS[fn]['n_images']
            log(f"{fn}: canonical artifact has {emb.shape[0]}/{expected} rows; validating batches")
        else:
            log(f"{fn}: not yet started — will extract")
        remaining.append(fn)

    log(f"Remaining folds to extract: {remaining}")

    # Load model once
    model = load_model()

    # Extract each fold
    for fold_name in remaining:
        log(f"--- Starting {fold_name} ---")
        success = extract_fold(fold_name, model)
        if not success:
            log(f"WARNING: {fold_name} extraction may have issues — check logs")

    log("=== ALL EXTRACTION COMPLETE ===")


if __name__ == '__main__':
    main()
