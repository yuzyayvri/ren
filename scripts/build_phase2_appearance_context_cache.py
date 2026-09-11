"""Build validated, resumable 148-column appearance/context cache shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from appearance_context_features import PatchAppearanceContext
from evaluate_hover_checkpoint import fold2_arrays
from nucleus_evaluation import match_instances
from pannuke_target_policy import make_target
from pilot_hover_fast import fold1_arrays

WIDTH = 148
SHARD = 64
PREDICTED_LABEL_UNMATCHED = 0
PREDICTED_LABEL_IGNORED = -1


def matched_proposal_labels(prediction, truth, truth_types, ignored, proposal_ids):
    """Assign proposal labels through IoU matching, never by numeric ID reuse.

    PanNuke truth IDs and model proposal IDs are unrelated namespaces.  Positive
    values are matched truth classes, zero is an evaluable unmatched proposal,
    and -1 is a proposal with no evaluable pixels under the target policy.
    """
    matches, _, _ = match_instances(prediction, truth, ignored)
    matched = {
        int(proposal_id): int(truth_types[truth_id])
        for proposal_id, truth_id, _ in matches
    }
    labels = []
    for proposal_id in np.asarray(proposal_ids, dtype=np.int64):
        evaluable = np.any((prediction == int(proposal_id)) & ~ignored)
        labels.append(
            matched.get(
                int(proposal_id),
                PREDICTED_LABEL_UNMATCHED if evaluable else PREDICTED_LABEL_IGNORED,
            )
        )
    return np.asarray(labels, dtype=np.int64)


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_npz(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, path)


def tissue(fold):
    return np.load(
        Path(f"data/tissue/fold{fold}/Fold {fold}/images/fold{fold}/types.npy"),
        mmap_mode="r",
    )


def build_truth(name, fold, images, masks, base_path, out):
    base = np.load(base_path)
    bfeat = base["features"]
    labels = base.get("labels", base.get("true_class"))
    tissue_arr = tissue(fold)
    rows = []
    pos = 0
    for i, (im, raw) in enumerate(zip(images, masks)):
        _, inst, _ = make_target(raw, i)
        ids = np.unique(inst)
        ids = ids[ids > 0]
        ctx = PatchAppearanceContext(im.astype(np.float32) / 255)
        n = len(ids)
        end = pos + n
        if end > len(bfeat):
            raise ValueError(f"{name} base rows exhausted at patch {i}")
        added = (
            np.stack([ctx.features(inst == int(z)) for z in ids], axis=0)
            if n
            else np.empty((0, 41), np.float32)
        )
        rows.extend(
            [
                (
                    i,
                    int(z),
                    int(labels[pos + j]),
                    str(tissue_arr[i]),
                    bfeat[pos + j],
                    added[j],
                )
                for j, z in enumerate(ids)
            ]
        )
        pos = end
        if (i + 1) % SHARD == 0 or i + 1 == len(images):
            write_shard(name, i // SHARD, rows, out)
            rows = []
    if pos != len(bfeat):
        raise ValueError(f"{name} base rows mismatch {pos} != {len(bfeat)}")


def build_pred(images, out):
    ref = Path("artifacts/phase2_nonlinear_classifier_v1/fold2_predicted_cache.pkl")
    expected = "e764e471a1cc2c801a5d5863909067d5c7d8edd1cf6f186006942bd8ae25e165"
    if digest(ref) != expected:
        raise ValueError("canonical proposal cache hash mismatch")
    with ref.open("rb") as f:
        cache = pickle.load(f)
    rows = []
    tissue_arr = tissue(2)
    for i in range(len(images)):
        r = cache[i]
        ctx = PatchAppearanceContext(images[i].astype(np.float32) / 255)
        ids = r["ids"]
        pred = r["prediction"]
        n = len(ids)
        added = (
            np.stack([ctx.features(pred == int(z)) for z in ids], axis=0)
            if n
            else np.empty((0, 41), np.float32)
        )
        labels = matched_proposal_labels(
            pred, r["truth"], r["truth_types"], r["ignored"], ids
        )
        rows.extend(
            [
                (
                    i,
                    int(z),
                    int(labels[j]),
                    str(tissue_arr[i]),
                    r["features"][j],
                    added[j],
                )
                for j, z in enumerate(ids)
            ]
        )
        if (i + 1) % SHARD == 0 or i + 1 == len(images):
            write_shard("fold2_predicted", i // SHARD, rows, out)
            rows = []


def write_shard(name, shard, rows, out):
    start = shard * SHARD
    path = out / name / f"shard_{shard:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    if rows:
        a = np.asarray(rows, dtype=object)
        patch = a[:, 0].astype(np.int64)
        ident = a[:, 1].astype(np.int64)
        y = a[:, 2].astype(np.int64)
        tissuev = a[:, 3].astype("U32")
        base = np.stack(a[:, 4])
        add = np.stack(a[:, 5])
    else:
        patch = ident = y = tissuev = np.empty(0, np.int64)
        base = np.empty((0, 107), np.float32)
        add = np.empty((0, 41), np.float32)
    if base.shape[1:] != (107,) or add.shape[1:] != (41,):
        raise ValueError("width")
    atomic_npz(
        path,
        patch_index=patch,
        instance_id=ident,
        labels=y,
        tissue_label=tissuev,
        base_features=base.astype(np.float32),
        added_features=add.astype(np.float32),
        width=np.array([WIDTH]),
    )
    z = np.load(path)
    assert (
        np.isfinite(z["base_features"]).all() and np.isfinite(z["added_features"]).all()
    )
    (path.with_suffix(".json")).write_text(
        json.dumps(
            {
                "shard": shard,
                "first_patch": int(patch.min()) if len(patch) else start,
                "last_patch": int(patch.max()) if len(patch) else start,
                "rows": len(patch),
                "sha256": digest(path),
            }
        )
        + "\n"
    )
    with (out / "cache_build.log").open("a") as f:
        f.write(f"{time.time():.3f} complete {name} {shard} {len(patch)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_appearance_context_classifier_v1/cache"),
    )
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    i1, m1 = fold1_arrays()
    i2, m2 = fold2_arrays()
    build_truth(
        "fold1_truth",
        1,
        i1,
        m1,
        Path("artifacts/phase2_nonlinear_classifier_v1/true_fold1_cache.npz"),
        a.output,
    )
    build_truth(
        "fold2_truth",
        2,
        i2,
        m2,
        Path("artifacts/phase2_nonlinear_classifier_v1/true_fold2_cache.npz"),
        a.output,
    )
    build_pred(i2, a.output)


if __name__ == "__main__":
    main()
