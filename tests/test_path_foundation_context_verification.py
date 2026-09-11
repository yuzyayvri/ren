import numpy as np
import pytest
from path_foundation_context_verification import (
    choose_epoch,
    exact_geometry,
    initialise_augmented_input,
    join_patch_embeddings,
    verify_frozen_embedding,
)


def test_lookup_preserves_repeats_order_and_exact_values():
    emb = np.arange(5 * 384, dtype=np.float32).reshape(5, 384)
    out = join_patch_embeddings(np.ones((3, 148)), np.empty((3, 0)), emb, np.array([3, 1, 3]))
    assert out.shape == (3, 532)
    np.testing.assert_array_equal(out[:, 148:], emb[[3, 1, 3]])


def test_empty_lookup_has_532_columns():
    out = join_patch_embeddings(np.empty((0, 148)), np.empty((0, 0)), np.zeros((4, 384)), np.array([], dtype=np.int64))
    assert out.shape == (0, 532)


@pytest.mark.parametrize("index", [[-1], [4], [0.5], [[0]]])
def test_embedding_indices_reject_invalid(index):
    with pytest.raises(ValueError):
        join_patch_embeddings(np.ones((len(np.asarray(index).reshape(-1)), 148)), np.empty((len(np.asarray(index).reshape(-1)), 0)), np.zeros((4, 384)), np.asarray(index))


@pytest.mark.parametrize("matrix", [np.zeros((4, 383)), np.zeros((4, 384, 1)), np.full((4, 384), np.nan)])
def test_embedding_matrix_rejects_malformed_or_nonfinite(matrix):
    with pytest.raises(ValueError):
        join_patch_embeddings(np.empty((0, 148)), np.empty((0, 0)), matrix, np.array([], dtype=np.int64))


def test_frozen_hashes_and_zero_initialization_and_bitwise_logits():
    with pytest.raises(ValueError): verify_frozen_embedding("bad", "ok", "p", "p")
    w = np.arange(3 * 148, dtype=np.float32).reshape(3, 148)
    a = initialise_augmented_input(w)
    assert np.array_equal(a[:, :148], w) and np.count_nonzero(a[:, 148:]) == 0
    x = np.zeros((4, 532), dtype=np.float32); x[:, :148] = np.arange(148)
    assert np.array_equal(x @ a.T, x[:, :148] @ w.T)


def test_geometry_fails_closed_for_each_value():
    good = {"tp": 40894, "fp": 9608, "fn": 18475, "detection_f1": .7444002512036844, "binary_pq": .596356927353324}
    assert exact_geometry(good)
    for key in good:
        bad = dict(good); bad[key] = bad[key] + (1 if key in {"tp", "fp", "fn"} else 1e-12)
        assert not exact_geometry(bad)


def test_dead_constrained_selection_and_tie_breaks():
    rows = [{"epoch": 1, "macro_f1": .5, "matched_accuracy": .2, "loss": 1, "dead_recall": .2, "dead_f1": .15},
            {"epoch": 2, "macro_f1": .5, "matched_accuracy": .3, "loss": 1, "dead_recall": .2, "dead_f1": .15},
            {"epoch": 3, "macro_f1": .5, "matched_accuracy": .3, "loss": .9, "dead_recall": .2, "dead_f1": .15},
            {"epoch": 4, "macro_f1": .5, "matched_accuracy": .3, "loss": .9, "dead_recall": .2, "dead_f1": .15}]
    assert choose_epoch(rows)["epoch"] == 3
    rows[0]["dead_recall"] = rows[0]["dead_f1"] = 0
    assert choose_epoch(rows)["epoch"] == 3
