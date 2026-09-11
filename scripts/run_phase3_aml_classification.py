"""Frozen Path Foundation image-level Munich AML classification protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from scripts.phase3_artifact_manifest import (
    atomic_manifest,
    mark_test_exposed,
    require_unexposed,
    sha256,
)
from scripts.phase3_frozen_heads import normalize_for_path_foundation, preprocess_aml
from scripts.phase3_selection import C_GRID, THRESHOLDS

ROOT = Path("/home/yuzy/ren")
MODEL = ROOT / "models/vision/path-foundation"
MANIFEST = ROOT / "artifacts/phase3_protocol_v1/source_manifest.json"


def _tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(sha256(p).encode())
    return h.hexdigest()


def _embedder():
    import tensorflow as tf

    model = tf.saved_model.load(str(MODEL))
    sig = model.signatures.get("serving_default") or next(iter(model.signatures.values()))
    name = next(iter(sig.structured_input_signature[1]))

    def run(batch):
        out = sig(**{name: tf.convert_to_tensor(normalize_for_path_foundation(batch))})
        result = next(iter(out.values())).numpy()
        if result.ndim == 3:
            result = result.mean(axis=1)
        if result.shape[1:] != (384,):
            raise RuntimeError(f"unexpected Path Foundation output {result.shape}")
        return np.asarray(result, dtype=np.float32)

    return run


def _split(split):
    return json.loads(MANIFEST.read_text())["munich_aml"]["splits"][split]


def _extract(split, encoder, output, batch_size=32):
    rows = _split(split)
    chunks, labels, ids = [], [], []
    for start in range(0, len(rows), batch_size):
        batch_images = []
        for row in rows[start:start + batch_size]:
            image = np.asarray(Image.open(ROOT / row["image"]).convert("RGB"), dtype=np.uint8)
            batch_images.append(preprocess_aml(image))
            labels.append(int(row["binary_label"]))
            ids.append(row["id"])
        chunks.append(encoder(np.asarray(batch_images, dtype=np.uint8)))
    embeddings = np.concatenate(chunks).astype(np.float32)
    tmp = output.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, embeddings=embeddings, labels=np.asarray(labels, np.int8), identities=np.asarray(ids))
    tmp.replace(output)
    return embeddings, np.asarray(labels, np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "phase3-aml-frozen-path-foundation-v1",
        "seed": 20260909,
        "split_policy": "immutable source manifest; image-level-only, not patient-independent",
        "positive": ["MOB", "MYO"],
        "negative": "all other named classes",
        "excluded": ["UNC", "nan"],
        "preprocessing": "RGB uint8; aspect-preserving 224 square median pad; bilinear antialias; float32 [0,1]",
        "model_tree_sha256": _tree_hash(MODEL),
    }
    (run / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    encoder = _embedder()
    x_train, y_train = _extract("train", encoder, run / "train_embeddings.npz")
    x_val, y_val = _extract("val", encoder, run / "val_embeddings.npz")
    scaler = StandardScaler().fit(x_train)
    train_z, val_z = scaler.transform(x_train), scaler.transform(x_val)
    candidates = []
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=20260909, solver="lbfgs")
        model.fit(train_z, y_train)
        prob = model.predict_proba(val_z)[:, 1]
        choices = []
        for threshold in THRESHOLDS:
            pred = (prob >= threshold).astype(int)
            sensitivity = recall_score(y_val, pred, zero_division=0)
            if sensitivity >= 0.80:
                choices.append((f1_score(y_val, pred, average="macro", zero_division=0), average_precision_score(y_val, prob), -float(threshold), float(threshold), sensitivity))
        if not choices:
            raise RuntimeError(f"no valid threshold for C={c}")
        best = max(choices)
        candidates.append((best[0], best[1], -float(c), c, model, prob, best[3], best[4]))
    _, auprc, _, selected_c, model, val_prob, threshold, val_sensitivity = max(candidates, key=lambda x: x[:3])
    selection = {"C_grid": list(C_GRID), "selected_C": float(selected_c), "threshold": float(threshold), "validation_macro_f1": float(f1_score(y_val, val_prob >= threshold, average="macro")), "validation_sensitivity": float(val_sensitivity), "validation_auprc": float(auprc)}
    atomic_manifest(run / "selection.json", stage="phase3-aml-classification", config=selection, inputs={"source_manifest_sha256": sha256(MANIFEST), "model_tree_sha256": config["model_tree_sha256"]})
    with (run / "head.pkl.tmp").open("wb") as handle:
        pickle.dump(model, handle)
    (run / "head.pkl.tmp").replace(run / "head.pkl")
    np.savez_compressed(run / "standardizer.npz.tmp.npz", mean=scaler.mean_, scale=scaler.scale_)
    (run / "standardizer.npz.tmp.npz").replace(run / "standardizer.npz")
    require_unexposed(run / "test_exposed.marker")
    mark_test_exposed(run / "test_exposed.marker", sha256(run / "selection.json"))
    x_test, y_test = _extract("test", encoder, run / "test_embeddings.npz")
    test_prob = model.predict_proba(scaler.transform(x_test))[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    report = {"selected_C": float(selected_c), "threshold": float(threshold), "validation_macro_f1": selection["validation_macro_f1"], "test_auroc": float(roc_auc_score(y_test, test_prob)), "test_auprc": float(average_precision_score(y_test, test_prob)), "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")), "test_sensitivity": float(recall_score(y_test, test_pred)), "confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(), "supports": np.bincount(y_test, minlength=2).tolist(), "gates": {"auroc": roc_auc_score(y_test, test_prob) >= .90, "auprc": average_precision_score(y_test, test_prob) >= .75, "macro_f1": f1_score(y_test, test_pred, average="macro") >= .75, "sensitivity": recall_score(y_test, test_pred) >= .80}}
    tmp = run / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(run / "report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
