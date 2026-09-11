"""Focused, isolated equivalence checks for the Phase 4 latency recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase4_v3_query import _complete_ranking

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts" / "phase4_protocol_v3"


def _legacy_complete(
    head: list[str], ids: list[str], k: int | None = None
) -> list[str]:
    ranked = list(head)
    ranked.extend(term_id for term_id in ids if term_id not in ranked)
    return ranked[:k] if k is not None else ranked


@pytest.mark.parametrize(
    ("head", "ids", "k"),
    [
        ([], ["GO:0000001", "GO:0000002", "GO:0000003"], None),
        ([], ["GO:0000001", "GO:0000002", "GO:0000003"], 0),
        (["GO:0000002"], ["GO:0000001", "GO:0000002", "GO:0000003"], None),
        (["GO:0000003", "GO:0000001"], ["GO:0000001", "GO:0000002", "GO:0000003"], 2),
        (["GO:0000001", "GO:0000002", "GO:0000003"], ["GO:0000001", "GO:0000002", "GO:0000003"], None),
    ],
)
def test_tail_completion_is_order_and_byte_identical(
    head: list[str], ids: list[str], k: int | None
) -> None:
    old = _legacy_complete(head, ids, k)
    new = _complete_ranking(head, ids, k)
    assert new == old
    assert json.dumps(new, separators=(",", ":")).encode() == json.dumps(
        old, separators=(",", ":")
    ).encode()
    assert len(new) == len(set(new))


def test_tied_head_order_is_stable() -> None:
    # The head is already ordered by the caller's deterministic score/tie rule;
    # tail completion must never reorder it.
    ids = ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"]
    head = ["GO:0000003", "GO:0000001"]
    result = _complete_ranking(head, ids)
    assert result[:2] == head
    assert result == _legacy_complete(head, ids)


def test_all_saved_v3_rankings_are_complete_duplicate_free_and_equivalent() -> None:
    rows = [
        json.loads(line)
        for line in (V3 / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 87
    for row in rows:
        ranking = row["ranking"]
        assert len(ranking) == 38245
        assert len(set(ranking)) == len(ranking)
        # Prefixes cover empty, head-only, tied/order-sensitive, and partial
        # inputs.  A full saved ranking is compared with an empty corpus tail;
        # invoking the historical list-membership implementation over 38,245
        # IDs would intentionally recreate its O(N^2) defect in the test.
        corpus_sample = ranking[:64] + ranking[128:192]
        for size in (0, 1, 2, 5, 10):
            head = ranking[:size]
            old = _legacy_complete(head, corpus_sample)
            new = _complete_ranking(head, corpus_sample)
            assert new == old
            assert json.dumps(new, separators=(",", ":")).encode() == json.dumps(
                old, separators=(",", ":")
            ).encode()
        old_full = _legacy_complete(ranking, [])
        new_full = _complete_ranking(ranking, [])
        assert new_full == old_full == ranking
