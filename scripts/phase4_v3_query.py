"""Offline v3 GO lookup, vector retrieval, RRF fusion, and traversal."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    OBO,
    OBO_SHA256,
    QUERY_MODEL,
    V3,
    code_sha256,
    model_manifest,
    sha256_path,
)

_DB: sqlite3.Connection | None = None
_IDS: list[str] | None = None
_ARRAY: np.ndarray | None = None
_SYMBOLIC_ALIASES: tuple[tuple[str, str, str, frozenset[str]], ...] | None = None
_SYMBOLIC_NAMES: tuple[tuple[str, frozenset[str]], ...] | None = None
_SYMBOLIC_ALIAS_EXACT: Mapping[str, tuple[int, ...]] | None = None
_SYMBOLIC_ALIAS_NGRAMS: tuple[Mapping[str, tuple[int, ...]], ...] | None = None
_SYMBOLIC_NAME_POSTINGS: Mapping[str, tuple[str, ...]] | None = None
_SYMBOLIC_CACHE_BINDING: tuple[int, int, int, int, int] | None = None
_QUERY_MODEL: Any = None
_QUERY_TOKENIZER: Any = None


def _manifest() -> dict[str, Any]:
    path = V3 / "index_manifest.json"
    if not path.is_file():
        raise RuntimeError("v3 index manifest is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_snapshot() -> dict[str, Any]:
    manifest = _manifest()
    protocol = V3 / "protocol.json"
    source = json.loads((V3 / "source.json").read_text(encoding="utf-8"))
    source_bytes_hash = sha256_path(OBO)
    actual = {
        "protocol_sha256": sha256_path(protocol),
        "database_sha256": sha256_path(V3 / "ontology.sqlite"),
        "documents_sha256": sha256_path(V3 / "documents.json"),
        "embeddings_sha256": sha256_path(V3 / "article_embeddings.npy"),
        "source_sha256": source_bytes_hash,
        "code_sha256": code_sha256(),
        "query_model_manifest_sha256": model_manifest(QUERY_MODEL)["manifest_sha256"],
        "article_model_manifest_sha256": model_manifest(ARTICLE_MODEL)[
            "manifest_sha256"
        ],
    }
    expected = {
        "protocol_sha256": manifest["protocol_sha256"],
        "database_sha256": manifest["database_sha256"],
        "documents_sha256": manifest["documents_sha256"],
        "embeddings_sha256": manifest["embeddings_sha256"],
        "source_sha256": manifest["source_sha256"],
        "code_sha256": manifest["code_sha256"],
        "query_model_manifest_sha256": manifest["models"]["query"]["manifest_sha256"],
        "article_model_manifest_sha256": manifest["models"]["article"][
            "manifest_sha256"
        ],
    }
    if actual != expected:
        raise RuntimeError(
            f"v3 immutable snapshot mismatch: expected {expected}, got {actual}"
        )
    if source_bytes_hash != OBO_SHA256 or source.get("sha256") != OBO_SHA256:
        raise RuntimeError("v3 approved GO source hash mismatch")
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    if protocol_data.get("status") != "frozen_prospective_no_evaluation":
        raise RuntimeError("v3 protocol is not frozen")
    return {"manifest": manifest, "actual": actual}


def _load_snapshot() -> None:
    global _DB, _IDS, _ARRAY
    global _SYMBOLIC_ALIASES, _SYMBOLIC_NAMES, _SYMBOLIC_CACHE_BINDING
    global _SYMBOLIC_ALIAS_EXACT, _SYMBOLIC_ALIAS_NGRAMS, _SYMBOLIC_NAME_POSTINGS
    if _DB is not None:
        if not _symbolic_cache_is_valid():
            raise RuntimeError("v3 symbolic cache binding changed; refusing stale cache")
        return
    verify_snapshot()
    _DB = sqlite3.connect(V3 / "ontology.sqlite", check_same_thread=False)
    _DB.execute("PRAGMA query_only=ON")
    _IDS = [
        row[0]
        for row in _DB.execute("SELECT id FROM terms WHERE obsolete=0 ORDER BY id")
    ]
    _ARRAY = np.load(V3 / "article_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if _ARRAY.shape != (len(_IDS), 768) or _ARRAY.dtype != np.float32:
        raise RuntimeError("v3 index shape/dtype mismatch")
    binding = _database_stat_sentinel()
    aliases, names, alias_exact, alias_ngrams, name_postings = _build_symbolic_cache(_DB)
    if _database_stat_sentinel() != binding:
        raise RuntimeError("v3 database changed while building symbolic cache")
    _SYMBOLIC_ALIASES = aliases
    _SYMBOLIC_NAMES = names
    _SYMBOLIC_ALIAS_EXACT = alias_exact
    _SYMBOLIC_ALIAS_NGRAMS = alias_ngrams
    _SYMBOLIC_NAME_POSTINGS = name_postings
    _SYMBOLIC_CACHE_BINDING = binding


def _database_stat_sentinel() -> tuple[int, int, int, int, int]:
    stat = (V3 / "ontology.sqlite").stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _build_symbolic_cache(
    db: sqlite3.Connection,
) -> tuple[
    tuple[tuple[str, str, str, frozenset[str]], ...],
    tuple[tuple[str, frozenset[str]], ...],
    Mapping[str, tuple[int, ...]],
    tuple[Mapping[str, tuple[int, ...]], ...],
    Mapping[str, tuple[str, ...]],
]:
    alias_rows = db.execute(
        """
        SELECT a.alias,a.term_id,a.kind,s.scope
        FROM aliases a JOIN terms t ON t.id=a.term_id
        LEFT JOIN synonyms s ON s.synonym_id=a.synonym_id
        WHERE t.obsolete=0
        """
    ).fetchall()
    aliases = tuple(
        (
            alias.casefold(),
            term_id,
            kind,
            frozenset(alias.casefold().split()),
        )
        for alias, term_id, kind, _scope in alias_rows
    )
    names = tuple(
        (
            term_id,
            frozenset(re.findall(r"[\w]+", name.casefold())),
        )
        for term_id, name in db.execute(
            "SELECT id,name FROM terms WHERE obsolete=0"
        )
    )
    exact: dict[str, list[int]] = {}
    ngrams: list[dict[str, list[int]]] = [{}, {}, {}, {}]
    for index, (alias, _term_id, _kind, _tokens) in enumerate(aliases):
        exact.setdefault(alias, []).append(index)
        for n in (1, 2, 3):
            if len(alias) < n:
                continue
            for start in range(len(alias) - n + 1):
                ngrams[n].setdefault(alias[start : start + n], []).append(index)
    name_postings: dict[str, list[str]] = {}
    for term_id, name_tokens in names:
        for token in name_tokens:
            name_postings.setdefault(token, []).append(term_id)
    frozen_exact = MappingProxyType(
        {key: tuple(value) for key, value in exact.items()}
    )
    frozen_ngrams = tuple(
        MappingProxyType({key: tuple(value) for key, value in postings.items()})
        for postings in ngrams
    )
    frozen_name_postings = MappingProxyType(
        {key: tuple(value) for key, value in name_postings.items()}
    )
    return aliases, names, frozen_exact, frozen_ngrams, frozen_name_postings


def _symbolic_cache_is_valid() -> bool:
    if (
        _SYMBOLIC_ALIASES is None
        or _SYMBOLIC_NAMES is None
        or _SYMBOLIC_ALIAS_EXACT is None
        or _SYMBOLIC_ALIAS_NGRAMS is None
        or _SYMBOLIC_NAME_POSTINGS is None
        or _SYMBOLIC_CACHE_BINDING is None
    ):
        return False
    try:
        current = _database_stat_sentinel()
    except OSError as exc:
        raise RuntimeError("v3 database binding cannot be checked") from exc
    return current == _SYMBOLIC_CACHE_BINDING


def _db() -> sqlite3.Connection:
    _load_snapshot()
    assert _DB is not None
    return _DB


def _canonical_id(value: str) -> str | None:
    match = re.fullmatch(r"(?i)GO:\d{7}", value.strip())
    return match.group(0).upper() if match else None


def resolve_identifier(query: str) -> dict[str, Any] | None:
    identifier = _canonical_id(query)
    if identifier is None:
        return None
    db = _db()
    row = db.execute(
        "SELECT id,obsolete FROM terms WHERE id=?", (identifier,)
    ).fetchone()
    if row and not row[1]:
        return {"input": identifier, "kind": "canonical", "targets": [identifier]}
    alt = db.execute(
        "SELECT target_term_id FROM redirects WHERE old_id=? AND kind='alt_id' AND target_present=1 ORDER BY target_term_id",
        (identifier,),
    ).fetchall()
    if alt:
        targets = sorted({row[0] for row in alt})
        return {"input": identifier, "kind": "alternate", "targets": targets}
    replaced = db.execute(
        "SELECT target_term_id FROM redirects WHERE old_id=? AND kind='replaced_by' AND target_present=1 ORDER BY target_term_id",
        (identifier,),
    ).fetchall()
    if replaced:
        targets = sorted({row[0] for row in replaced})
        return {"input": identifier, "kind": "obsolete_replaced_by", "targets": targets}
    considered = db.execute(
        "SELECT target_term_id FROM redirects WHERE old_id=? AND kind='consider' AND target_present=1 ORDER BY target_term_id",
        (identifier,),
    ).fetchall()
    if considered:
        targets = sorted({row[0] for row in considered})
        return {"input": identifier, "kind": "obsolete_consider", "targets": targets}
    if row and row[1]:
        return {"input": identifier, "kind": "obsolete_unresolved", "targets": []}
    return {"input": identifier, "kind": "unknown_identifier", "targets": []}


def _lexical_candidates(query: str) -> list[tuple[float, str]]:
    _load_snapshot()
    if not _symbolic_cache_is_valid():
        raise RuntimeError("v3 symbolic cache binding changed; refusing stale cache")
    assert (
        _SYMBOLIC_ALIASES is not None
        and _SYMBOLIC_ALIAS_EXACT is not None
        and _SYMBOLIC_ALIAS_NGRAMS is not None
        and _SYMBOLIC_NAME_POSTINGS is not None
    )
    q = query.casefold().strip()
    tokens = set(re.findall(r"[\w]+", q))
    scores: dict[str, float] = {}
    candidates: set[int] = set(_SYMBOLIC_ALIAS_EXACT.get(q, ()))
    if q:
        for start in range(len(q)):
            for end in range(start + 1, len(q) + 1):
                candidates.update(_SYMBOLIC_ALIAS_EXACT.get(q[start:end], ()))
        gram_size = min(3, len(q))
        gram_postings = _SYMBOLIC_ALIAS_NGRAMS[gram_size]
        gram_keys = {
            q[start : start + gram_size]
            for start in range(len(q) - gram_size + 1)
        }
        if gram_keys:
            postings = [gram_postings.get(gram, ()) for gram in gram_keys]
            if all(postings):
                candidate_intersection = set(postings[0])
                for posting in postings[1:]:
                    candidate_intersection.intersection_update(posting)
                candidates.update(
                    index
                    for index in candidate_intersection
                    if q in _SYMBOLIC_ALIASES[index][0]
                )
    for index in candidates:
        alias, term_id, kind, alias_tokens = _SYMBOLIC_ALIASES[index]
        if alias == q:
            scores[term_id] = max(
                scores.get(term_id, 0.0), 10000.0 if kind == "name" else 9000.0
            )
        elif q and (q in alias or alias in q):
            scores[term_id] = max(scores.get(term_id, 0.0), 100.0 + len(tokens & alias_tokens))
    name_overlaps: dict[str, int] = {}
    for token in tokens:
        for term_id in _SYMBOLIC_NAME_POSTINGS.get(token, ()):
            name_overlaps[term_id] = name_overlaps.get(term_id, 0) + 1
    for term_id, overlap in name_overlaps.items():
        scores[term_id] = max(scores.get(term_id, 0.0), float(overlap))
    return sorted(
        ((score, term_id) for term_id, score in scores.items()),
        key=lambda item: (-item[0], item[1]),
    )


def _complete_ranking(
    head: list[str], ids: list[str], k: int | None = None
) -> list[str]:
    """Append unseen corpus IDs in corpus order using O(1) membership checks."""
    ranked = list(head)
    seen = set(ranked)
    for term_id in ids:
        if term_id not in seen:
            ranked.append(term_id)
            seen.add(term_id)
    return ranked[:k] if k is not None else ranked


def symbolic(query: str, k: int | None = None) -> list[str]:
    exact = resolve_identifier(query)
    if exact is not None:
        ranked = list(exact["targets"])
        _load_snapshot()
        assert _IDS is not None
        return _complete_ranking(ranked, _IDS, k)
    ranked = [term_id for _, term_id in _lexical_candidates(query)]
    _load_snapshot()
    assert _IDS is not None
    return _complete_ranking(ranked, _IDS, k)


def _load_query_model() -> tuple[Any, Any]:
    global _QUERY_MODEL, _QUERY_TOKENIZER
    if _QUERY_MODEL is None:
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
        )
        import torch
        from transformers import AutoModel, AutoTokenizer

        _QUERY_TOKENIZER = AutoTokenizer.from_pretrained(
            QUERY_MODEL, local_files_only=True
        )
        _QUERY_MODEL = AutoModel.from_pretrained(
            QUERY_MODEL, local_files_only=True
        ).eval()
        _QUERY_MODEL._v3_torch = torch
    return _QUERY_TOKENIZER, _QUERY_MODEL


def vector(query: str, k: int | None = None) -> list[str]:
    exact = resolve_identifier(query)
    if exact is not None:
        ranked = list(exact["targets"])
        _load_snapshot()
        assert _IDS is not None
        return _complete_ranking(ranked, _IDS, k)
    _load_snapshot()
    tokenizer, model = _load_query_model()
    torch = model._v3_torch
    inputs = tokenizer(
        [query], padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    with torch.no_grad():
        vector_value = (
            model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()[0]
        )
    vector_value /= max(float(np.linalg.norm(vector_value)), 1e-12)
    assert _ARRAY is not None and _IDS is not None
    scores = _ARRAY @ vector_value
    order = np.argsort(-scores, kind="stable")
    ranked = [_IDS[int(index)] for index in order]
    return ranked[:k] if k is not None else ranked


def hybrid(query: str, k: int | None = None) -> list[str]:
    exact = resolve_identifier(query)
    if exact is not None:
        ranked = list(exact["targets"])
        _load_snapshot()
        assert _IDS is not None
        return _complete_ranking(ranked, _IDS, k)
    symbolic_rank = symbolic(query)
    vector_rank = vector(query)
    scores: dict[str, float] = {}
    for rank, term_id in enumerate(symbolic_rank, start=1):
        scores[term_id] = scores.get(term_id, 0.0) + 1.0 / (60 + rank)
    for rank, term_id in enumerate(vector_rank, start=1):
        scores[term_id] = scores.get(term_id, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(scores, key=lambda term_id: (-scores[term_id], term_id))
    return ranked[:k] if k is not None else ranked


def traverse(
    go_id: str,
    direction: str = "ancestors",
    depth: int = 10,
    relations: tuple[str, ...] = ("is_a", "part_of"),
) -> list[str]:
    exact = resolve_identifier(go_id)
    if exact is None or len(exact["targets"]) != 1:
        raise ValueError("traversal requires one canonical GO identifier")
    start = exact["targets"][0]
    if direction not in {"ancestors", "descendants"}:
        raise ValueError("direction must be ancestors or descendants")
    if depth < 0:
        raise ValueError("depth must be non-negative")
    allowed = tuple(sorted(set(relations) & {"is_a", "part_of"}))
    if not allowed:
        return []
    db = _db()
    seen = {start}
    frontier = [start]
    reached: set[str] = set()
    for _ in range(depth):
        if not frontier:
            break
        next_frontier: list[str] = []
        for node in sorted(frontier):
            if direction == "ancestors":
                rows = db.execute(
                    f"SELECT parent FROM edges WHERE child=? AND relation IN ({','.join('?' for _ in allowed)}) ORDER BY parent",
                    (node, *allowed),
                ).fetchall()
            else:
                rows = db.execute(
                    f"SELECT child FROM edges WHERE parent=? AND relation IN ({','.join('?' for _ in allowed)}) ORDER BY child",
                    (node, *allowed),
                ).fetchall()
            for (candidate,) in rows:
                if candidate not in seen:
                    seen.add(candidate)
                    reached.add(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    return sorted(reached)


def retrieve(query: str, mode: str = "hybrid", k: int | None = 5) -> list[str]:
    if mode == "symbolic":
        return symbolic(query, k)
    if mode == "vector":
        return vector(query, k)
    if mode == "hybrid":
        return hybrid(query, k)
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument(
        "--mode", choices=["symbolic", "vector", "hybrid"], default="hybrid"
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "query": args.query,
                "retrieved": retrieve(args.query, args.mode, args.top_k),
            }
        )
    )


if __name__ == "__main__":
    main()
