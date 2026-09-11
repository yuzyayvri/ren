"""Adversarial, quality-free checks for the final symbolic cache recovery."""

from __future__ import annotations

import json
import re
import sqlite3
from itertools import pairwise
from pathlib import Path

import pytest

from scripts import phase4_v3_query as query_engine
from scripts.phase4_latency_recovery_final import _enable_recovery_snapshot_validation

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts" / "phase4_protocol_v3"


@pytest.fixture(scope="module", autouse=True)
def _load_bound_snapshot() -> None:
    _enable_recovery_snapshot_validation()
    query_engine._load_snapshot()


def _legacy_lexical_candidates(query: str) -> list[tuple[float, str]]:
    db = query_engine._DB
    assert db is not None
    q = query.casefold().strip()
    tokens = set(re.findall(r"[\w]+", q))
    rows = db.execute(
        """
        SELECT a.alias,a.term_id,a.kind,s.scope
        FROM aliases a JOIN terms t ON t.id=a.term_id
        LEFT JOIN synonyms s ON s.synonym_id=a.synonym_id
        WHERE t.obsolete=0
        """
    ).fetchall()
    scores: dict[str, float] = {}
    for alias, term_id, kind, _scope in rows:
        alias = alias.casefold()
        if alias == q:
            scores[term_id] = max(
                scores.get(term_id, 0.0), 10000.0 if kind == "name" else 9000.0
            )
        elif q and (q in alias or alias in q):
            scores[term_id] = max(
                scores.get(term_id, 0.0), 100.0 + len(tokens & set(alias.split()))
            )
    for term_id, name in db.execute("SELECT id,name FROM terms WHERE obsolete=0"):
        name_tokens = set(re.findall(r"[\w]+", name.casefold()))
        overlap = len(tokens & name_tokens)
        if overlap:
            scores[term_id] = max(scores.get(term_id, 0.0), float(overlap))
    return sorted(
        ((score, term_id) for term_id, score in scores.items()),
        key=lambda item: (-item[0], item[1]),
    )


def _benchmark_cases() -> list[dict[str, object]]:
    benchmark = json.loads((V3 / "benchmark.json").read_text(encoding="utf-8"))
    return [case for group in benchmark["strata"].values() for case in group]


def test_cache_is_immutable_and_sqlite_derived() -> None:
    db = query_engine._DB
    assert db is not None
    assert isinstance(query_engine._SYMBOLIC_ALIASES, tuple)
    assert isinstance(query_engine._SYMBOLIC_NAMES, tuple)
    assert all(isinstance(item[3], frozenset) for item in query_engine._SYMBOLIC_ALIASES)
    assert all(isinstance(item[1], frozenset) for item in query_engine._SYMBOLIC_NAMES)
    alias_count = db.execute(
        """
        SELECT count(*)
        FROM aliases a JOIN terms t ON t.id=a.term_id
        WHERE t.obsolete=0
        """
    ).fetchone()[0]
    current_count = db.execute("SELECT count(*) FROM terms WHERE obsolete=0").fetchone()[0]
    assert len(query_engine._SYMBOLIC_ALIASES) == alias_count
    assert len(query_engine._SYMBOLIC_NAMES) == current_count


def test_adversarial_symbolic_scores_order_and_ties_are_exact() -> None:
    benchmark_queries = [
        str(case["query"])
        for case in _benchmark_cases()
        if not re.fullmatch(r"(?i)GO:\d{7}", str(case["query"]).strip())
    ]
    adversarial = [
        "",
        "   ",
        "NeUtRoPhIl",
        "  dead cells  ",
        "ERYTHROCYTE!!!",
        "blood-cell",
        "cell division",
        "a",
        "part_of",
    ]
    for query in benchmark_queries + adversarial:
        assert query_engine._lexical_candidates(query) == _legacy_lexical_candidates(query)
    tied = _legacy_lexical_candidates("cell")
    assert any(tied[index][0] == tied[index + 1][0] for index in range(len(tied) - 1))
    for left, right in pairwise(tied):
        if left[0] == right[0]:
            assert left[1] < right[1]


def test_cache_invalidation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_engine, "_database_stat_sentinel", lambda: (0, 0, 0, 0, 0))
    assert query_engine._symbolic_cache_is_valid() is False
    with pytest.raises(RuntimeError, match="stale cache"):
        query_engine._lexical_candidates("neutrophil")


def test_identifier_redirects_and_traversal_remain_exact() -> None:
    for case in _benchmark_cases():
        case_type = str(case["case_type"])
        if case_type in {"canonical_id", "alternate_id", "obsolete_id"}:
            resolved = query_engine.resolve_identifier(str(case["query"]))
            assert resolved is not None
            assert resolved["targets"] == sorted(case["gold_ids"])
    for case in _benchmark_cases():
        if case["case_type"] != "traversal":
            continue
        actual = query_engine.traverse(
            str(case["query"]),
            str(case["direction"]),
            int(case["depth"]),
            tuple(case["relations"]),
        )
        assert actual == sorted(case["gold_ids"])


def test_tail_completion_remains_duplicate_free() -> None:
    db = sqlite3.connect(V3 / "ontology.sqlite")
    ids = [row[0] for row in db.execute("SELECT id FROM terms WHERE obsolete=0 ORDER BY id")]
    db.close()
    for head in ([], ids[:1], ids[:10], ids[:100], ids):
        completed = query_engine._complete_ranking(head, ids)
        assert completed == ids
        assert len(completed) == len(set(completed))
