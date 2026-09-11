#!/usr/bin/env python3
"""
Phase 1 — Train a linear classification head on frozen ViT embeddings.

Fits a classifier on top of pre-extracted embeddings to predict PanNuke's
19-way tissue-type label (from types.npy). Trains on folds 1+2, tests on
fold 3.

This is the smoke test for the vision-engine scaffold: prove that
frozen-encoder → embeddings → trainable-head works end to end. The head here
is a plain softmax logistic regression (sklearn LogisticRegression) — not the
final architecture, just a wiring test.

Acceptance criteria (from ROADMAP.md):
    - End-to-end run completes without manual intervention.
    - Test accuracy is meaningfully above chance (>> 1/19 ≈ 5.3%).
      It must beat the held-out majority-class baseline, not merely uniform
      random chance.

Usage:
    python scripts/train_head.py --backbone virchow
    python scripts/train_head.py --backbone path-foundation

Output:
    Prints per-class accuracy, macro-F1, and overall test accuracy to stdout.
    Saves a JSON summary to embeddings/{backbone}/head_results.json for later
    comparison between backbones.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBED_DIR = PROJECT_ROOT / "embeddings"


def load_fold_embeddings(backbone: str, fold: str):
    """Load embeddings.npy + labels.npy for one fold."""
    emb_path = EMBED_DIR / backbone / fold / "embeddings.npy"
    lbl_path = EMBED_DIR / backbone / fold / "labels.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}. "
            f"Run extract_embeddings.py --backbone {backbone} --fold {fold} first."
        )
    embeddings = np.load(emb_path, allow_pickle=True)
    labels = np.load(lbl_path, allow_pickle=True)
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description="Phase 1 head training (smoke test)")
    parser.add_argument(
        "--backbone",
        choices=["virchow", "path-foundation"],
        required=True,
        help="Backbone whose embeddings to train on",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations for LogisticRegression (default 1000)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength (default 1.0)",
    )
    args = parser.parse_args()

    from verify_embeddings import verify_fold

    failures = {
        fold: verify_fold(fold, args.backbone)
        for fold in ("fold1", "fold2", "fold3")
    }
    failures = {fold: errors for fold, errors in failures.items() if errors}
    if failures:
        for fold, errors in failures.items():
            print(f"[error] {fold} failed artifact verification:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        print("[error] Refusing to train on unverified artifacts.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Loading embeddings for backbone={args.backbone}", file=sys.stderr)

    # Load folds 1 and 2 for training
    X1, y1 = load_fold_embeddings(args.backbone, "fold1")
    X2, y2 = load_fold_embeddings(args.backbone, "fold2")
    X3, y3 = load_fold_embeddings(args.backbone, "fold3")

    print(
        f"[info] Fold shapes: 1={X1.shape} 2={X2.shape} 3={X3.shape}",
        file=sys.stderr,
    )

    X_train = np.concatenate([X1, X2], axis=0)
    y_train = np.concatenate([y1, y2], axis=0)

    # Encode string labels → integer classes
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    y_test_int = le.transform(y3)

    class_names = le.classes_.tolist()
    print(f"[info] Classes ({len(class_names)}): {class_names}", file=sys.stderr)

    # Train a linear probe (softmax logistic regression)
    print(
        f"[info] Training LogisticRegression(C={args.C}, max_iter={args.max_iter}, "
        f"n_train={X_train.shape[0]}) ...",
        file=sys.stderr,
    )
    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train_int)

    # Evaluate on fold 3
    y_pred = clf.predict(X3)

    acc = accuracy_score(y_test_int, y_pred)
    train_counts = np.bincount(y_train_int, minlength=len(class_names))
    majority_class = int(np.argmax(train_counts))
    majority_acc = float(np.mean(y_test_int == majority_class))
    macro_f1 = f1_score(y_test_int, y_pred, average="macro")
    per_class_acc = {}
    for i, cname in enumerate(class_names):
        mask = y_test_int == i
        if mask.sum() == 0:
            per_class_acc[cname] = None
        else:
            per_class_acc[cname] = float((y_pred[mask] == i).mean())

    print("\n=== Phase 1 head-training results ===", file=sys.stderr)
    print(f"Backbone: {args.backbone}", file=sys.stderr)
    print(f"Train folds: 1+2 ({X_train.shape[0]} images)", file=sys.stderr)
    print(f"Test fold: 3 ({X3.shape[0]} images)", file=sys.stderr)
    print(f"Overall test accuracy: {acc:.4f} ({acc*100:.2f}%)", file=sys.stderr)
    print(f"Macro F1: {macro_f1:.4f}", file=sys.stderr)
    print(f"Chance level (1/19): {1/19:.4f} ({(1/19)*100:.2f}%)", file=sys.stderr)
    print(
        f"Training-majority baseline ({class_names[majority_class]}): "
        f"{majority_acc:.4f} ({majority_acc*100:.2f}%)",
        file=sys.stderr,
    )
    print("\nPer-class test accuracy:", file=sys.stderr)
    for cname, ca in per_class_acc.items():
        if ca is None:
            print(f"  {cname}: N/A (no test samples)", file=sys.stderr)
        else:
            print(f"  {cname}: {ca:.4f} ({ca*100:.2f}%)", file=sys.stderr)

    # Save summary JSON
    summary = {
        "backbone": args.backbone,
        "train_folds": ["fold1", "fold2"],
        "test_fold": "fold3",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X3.shape[0]),
        "n_classes": len(class_names),
        "class_names": class_names,
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "chance_accuracy": 1.0 / 19.0,
        "majority_class": class_names[majority_class],
        "majority_baseline_accuracy": majority_acc,
        "per_class_accuracy": per_class_acc,
        "model": "LogisticRegression",
        "C": args.C,
        "max_iter": args.max_iter,
    }

    out_path = EMBED_DIR / args.backbone / "head_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[info] Summary saved to {out_path}", file=sys.stderr)

    # Verdict: the learned probe must beat the training-majority predictor.
    if acc <= majority_acc:
        print(
            "\n[fail] Test accuracy does not beat the training-majority baseline; "
            "the Phase 1 smoke test failed.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(
            f"\n[ok] Test accuracy ({acc*100:.2f}%) is above the "
            f"training-majority baseline ({majority_acc*100:.2f}%). "
            "Smoke test passed.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
