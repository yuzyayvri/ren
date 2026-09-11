#!/usr/bin/env python3
"""
Extract a single batch of Virchow embeddings for fold1.
Loads pre-saved model state from /tmp/virchow_state.pt for fast startup.
Usage: python scripts/extract_virchow_batch.py <start> <end>
Saves an atomic, index-bearing batch artifact for later validated assembly.
"""
# Ensure deterministic threading before any torch import
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

sys.path.insert(0, str(Path(__file__).parent))

from embedding_artifacts import save_batch
from extract_embeddings import VirchowModel, _virchow_transform_batch

PROJECT_ROOT = Path('/home/yuzy/ren')
OUT_DIR = PROJECT_ROOT / 'embeddings' / 'virchow' / 'fold1'
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = Path('/tmp/virchow_state.pt')


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <start> <end>', file=sys.stderr)
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2])

    # Load model state (fast — just deserializing from disk)
    state = torch.load(str(STATE_PATH), map_location='cpu', weights_only=True)
    model = VirchowModel()
    model.load_state_dict(state)
    model.eval()

    # Load images slice
    images_path = (PROJECT_ROOT / 'data' / 'tissue' / 'fold1' /
                   'Fold 1' / 'images' / 'fold1' / 'images.npy')
    source_images = np.load(str(images_path), mmap_mode='r')
    images = np.asarray(source_images)
    if images.dtype == np.float64:
        images = images.astype(np.float32) / 255.0

    batch = images[start:end]

    # Transform + forward
    batch_t = _virchow_transform_batch(batch)
    with torch.inference_mode():
        feats = model(batch_t)[:, 0, :].numpy()

    # Save
    out_path = OUT_DIR / f'embeddings_batch_{start}_{end}.npz'
    save_batch(out_path, start, feats, source_images[start:end])
    print(f'DONE_BATCH_{start}_{end}', flush=True)


if __name__ == '__main__':
    main()
