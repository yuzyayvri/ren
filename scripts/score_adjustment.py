"""Deterministic, bounded score-offset evaluation for cached fold2 instances."""

from __future__ import annotations

import numpy as np


def adjusted_classes(scores: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return argmax classes (1-based); scores are raw decision scores."""
    if scores.ndim != 2 or scores.shape[1] != len(offsets):
        raise ValueError("scores and offsets have incompatible shapes")
    return np.argmax(scores + offsets[None, :], axis=1) + 1


def selection_key(entry: dict[str, object]) -> tuple[float, float, float, float]:
    metrics = entry["metrics"]
    dead = metrics["classes"]["4"]
    return (
        float(metrics["macro_f1"]),
        float(dead["f1"]),
        float(metrics["matched_instance_typing_accuracy"]),
        -abs(float(entry["offsets"][3])),
    )


def choose(entries: list[dict[str, object]]) -> dict[str, object]:
    eligible = [
        entry
        for entry in entries
        if entry["metrics"]["classes"]["4"]["recall"] >= 0.20
        and entry["metrics"]["classes"]["4"]["f1"] >= 0.15
    ]
    if not eligible:
        raise ValueError("no score-offset candidate satisfies Dead constraints")
    return max(eligible, key=selection_key)


def dead_grid() -> list[float]:
    return [float(value) for value in np.arange(0.0, -1.01, -0.10)]

