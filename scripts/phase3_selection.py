"""Validation-only selection primitives for the frozen Phase 3 stages."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

THRESHOLDS = np.arange(0.05, 0.951, 0.05)
C_GRID = (0.01, 0.1, 1.0, 10.0)


def select_detector_threshold(y_true: np.ndarray, scores: np.ndarray, classes: int) -> dict:
    rows = []
    for threshold in THRESHOLDS:
        pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, pred, average="macro", labels=np.arange(classes), zero_division=0)
        recalls = recall_score(y_true, pred, average=None, labels=np.arange(classes), zero_division=0)
        rows.append((float(f1), float(np.min(recalls)), float(np.mean(recalls)), -float(threshold), threshold, recalls))
    best = max(rows, key=lambda x: x[:4])
    return {"threshold": float(best[4]), "macro_f1": best[0], "min_recall": best[1], "mean_recall": best[2], "per_class_recall": best[5].tolist(), "grid": [float(x) for x in THRESHOLDS]}


def fit_class_balanced_head(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, binary: bool = False) -> dict:
    scaler = StandardScaler().fit(x_train)
    tr, va = scaler.transform(x_train), scaler.transform(x_val)
    candidates = []
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=20260909, solver="lbfgs")
        model.fit(tr, y_train)
        pred = model.predict(va)
        score = f1_score(y_val, pred, average="macro", zero_division=0)
        candidates.append((float(score), -float(c), c, model))
    _, _, c, model = max(candidates, key=lambda row: row[:2])
    result = {"C": float(c), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "model": model, "validation_macro_f1": max(x[0] for x in candidates)}
    if binary:
        probabilities = model.predict_proba(va)[:, 1]
        choices = []
        for threshold in THRESHOLDS:
            pred = (probabilities >= threshold).astype(int)
            sensitivity = recall_score(y_val, pred, zero_division=0)
            if sensitivity < 0.80: continue
            choices.append((f1_score(y_val, pred, average="macro", zero_division=0), average_precision_score(y_val, probabilities), -float(threshold), threshold))
        if not choices: raise ValueError("no AML validation threshold satisfies sensitivity gate")
        best = max(choices)
        result.update({"threshold": float(best[3]), "validation_sensitivity": float(recall_score(y_val, (probabilities >= best[3]).astype(int), zero_division=0)), "validation_auprc": float(best[1])})
    return result

