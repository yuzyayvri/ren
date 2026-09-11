"""Stanza-safe, two-pass GO OBO ingestion for the Phase 4 v3 boundary."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from scripts.phase4_v3_common import (
    OBO,
    OBO_SHA256,
    ROOT,
    V3,
    atomic_write_json,
    sha256_path,
)


def unescape_obo(value: str) -> str:
    out: list[str] = []
    i = 0
    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}
    while i < len(value):
        if value[i] == "\\" and i + 1 < len(value):
            out.append(escapes.get(value[i + 1], value[i + 1]))
            i += 2
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def quoted_value(value: str) -> tuple[str, str]:
    if not value.startswith('"'):
        return value.strip(), ""
    escaped = False
    chars: list[str] = []
    for index in range(1, len(value)):
        char = value[index]
        if escaped:
            chars.append("\\" + char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return unescape_obo("".join(chars)), value[index + 1 :].strip()
        else:
            chars.append(char)
    raise ValueError("unterminated OBO quoted value")


def parse_synonym(value: str) -> dict[str, Any]:
    text, rest = quoted_value(value)
    parts = rest.split(None, 1)
    if not parts:
        raise ValueError(f"synonym scope missing: {value}")
    scope = parts[0].upper()
    if scope not in {"EXACT", "RELATED", "NARROW", "BROAD"}:
        raise ValueError(f"unexpected synonym scope {scope!r}")
    tail = parts[1] if len(parts) > 1 else ""
    xrefs: list[str] = []
    bracket = re.search(r"\[(.*)\]", tail)
    if bracket:
        xrefs = [x.strip() for x in bracket.group(1).split(",") if x.strip()]
        type_part = tail[: bracket.start()].strip()
    else:
        type_part = tail.strip()
    return {
        "value": text,
        "scope": scope,
        "type": type_part,
        "xrefs": xrefs,
        "raw": value,
    }


def parse_definition(value: str) -> tuple[str, list[str]]:
    text, rest = quoted_value(value)
    bracket = re.search(r"\[(.*)\]", rest)
    xrefs = (
        [x.strip() for x in bracket.group(1).split(",") if x.strip()] if bracket else []
    )
    return text, xrefs


def _stanzas(text: str) -> list[tuple[str, str]]:
    markers = list(re.finditer(r"(?m)^\[([^\]]+)\]\s*$", text))
    out = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        out.append((marker.group(1), text[marker.end() : end]))
    return out


def _fields(body: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for line in body.splitlines():
        if not line.strip() or line.startswith("!") or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields.setdefault(key, []).append(value)
    return fields


def parse_obo(path: Path = OBO) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.resolve() == OBO.resolve() and sha256_path(path) != OBO_SHA256:
        raise RuntimeError("approved GO OBO hash mismatch")
    stanzas = _stanzas(raw)
    first_marker = re.search(r"(?m)^\[[^\]]+\]\s*$", raw)
    header_text = raw[: first_marker.start() if first_marker else len(raw)]
    header = {
        "format_version": (
            re.search(r"(?m)^format-version:\s*(\S+)", header_text) or [None, None]
        )[1],
        "data_version": (
            re.search(r"(?m)^data-version:\s*(\S+)", header_text) or [None, None]
        )[1],
        "license_url": (
            re.search(r"(?m)^property_value:\s*terms:license\s+(\S+)", header_text)
            or [None, None]
        )[1],
    }
    terms: list[dict[str, Any]] = []
    typedef_count = 0
    for kind, body in stanzas:
        if kind == "Typedef":
            typedef_count += 1
            continue
        if kind != "Term":
            continue
        fields = _fields(body)
        identifier = fields.get("id", [""])[0]
        if not identifier.startswith("GO:"):
            continue
        definition, def_xrefs = (
            parse_definition(fields.get("def", [""])[0])
            if fields.get("def")
            else ("", [])
        )
        synonyms = [parse_synonym(x) for x in fields.get("synonym", [])]
        relationships = []
        for relation in fields.get("relationship", []):
            parts = relation.split()
            if len(parts) >= 2:
                relationships.append({"relation": parts[0], "target": parts[1]})
        terms.append(
            {
                "id": identifier,
                "name": fields.get("name", [""])[0],
                "namespace": fields.get("namespace", [""])[0],
                "definition": definition,
                "definition_xrefs": def_xrefs,
                "obsolete": fields.get("is_obsolete", ["false"])[0].lower() == "true",
                "alt_ids": fields.get("alt_id", []),
                "replaced_by": [
                    x.split()[0] for x in fields.get("replaced_by", []) if x.split()
                ],
                "consider": [
                    x.split()[0] for x in fields.get("consider", []) if x.split()
                ],
                "is_a": [x.split()[0] for x in fields.get("is_a", []) if x.split()],
                "relationships": relationships,
                "synonyms": synonyms,
                "source_order": len(terms),
            }
        )
    if header["format_version"] != "1.2" or not header["data_version"]:
        raise RuntimeError("GO OBO header lacks format/data version")
    if header["license_url"] != "http://creativecommons.org/licenses/by/4.0/":
        raise RuntimeError(
            "GO OBO license declaration is not the approved CC-BY-4.0 URL"
        )
    ids = {term["id"] for term in terms}
    if len(ids) != len(terms) or len(terms) != 48329:
        raise RuntimeError(f"unexpected GO term count: {len(terms)}")
    if (
        sum(not term["obsolete"] for term in terms) != 38245
        or sum(term["obsolete"] for term in terms) != 10084
    ):
        raise RuntimeError(
            "GO current/obsolete counts do not match the approved source"
        )
    return {
        "header": header,
        "terms": terms,
        "typedef_count": typedef_count,
        "source_sha256": sha256_path(path),
        "source_bytes": path.stat().st_size,
        "term_stanzas": len(terms),
    }


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = DELETE;
PRAGMA synchronous = FULL;
CREATE TABLE terms (
    id TEXT PRIMARY KEY CHECK(id GLOB 'GO:*'),
    name TEXT NOT NULL,
    namespace TEXT NOT NULL,
    definition TEXT NOT NULL,
    definition_xrefs TEXT NOT NULL,
    obsolete INTEGER NOT NULL CHECK(obsolete IN (0,1)),
    source_order INTEGER NOT NULL UNIQUE
);
CREATE TABLE synonyms (
    synonym_id INTEGER PRIMARY KEY,
    term_id TEXT NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
    value TEXT NOT NULL,
    scope TEXT NOT NULL CHECK(scope IN ('EXACT','RELATED','NARROW','BROAD')),
    type TEXT NOT NULL,
    xrefs TEXT NOT NULL,
    raw TEXT NOT NULL,
    UNIQUE(term_id,value,scope,type,raw)
);
CREATE TABLE aliases (
    alias TEXT NOT NULL,
    term_id TEXT NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('name','synonym')),
    synonym_id INTEGER REFERENCES synonyms(synonym_id) ON DELETE CASCADE,
    UNIQUE(alias,term_id,kind,synonym_id)
);
CREATE TABLE redirects (
    old_id TEXT NOT NULL,
    new_id TEXT NOT NULL CHECK(new_id GLOB 'GO:*'),
    target_term_id TEXT REFERENCES terms(id) ON DELETE RESTRICT,
    target_present INTEGER NOT NULL CHECK(target_present IN (0,1)),
    kind TEXT NOT NULL CHECK(kind IN ('alt_id','replaced_by','consider')),
    source_order INTEGER NOT NULL,
    PRIMARY KEY(old_id,new_id,kind)
);
CREATE TABLE edges (
    parent TEXT NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
    child TEXT NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK(relation IN ('is_a','part_of')),
    source_order INTEGER NOT NULL,
    PRIMARY KEY(parent,child,relation)
);
CREATE INDEX aliases_lookup ON aliases(alias);
CREATE INDEX aliases_term ON aliases(term_id);
CREATE INDEX redirects_old ON redirects(old_id,kind,new_id);
CREATE INDEX redirects_new ON redirects(new_id);
CREATE INDEX edges_parent ON edges(parent,relation,child);
CREATE INDEX edges_child ON edges(child,relation,parent);
CREATE INDEX terms_name ON terms(name COLLATE NOCASE);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_database(
    parsed: dict[str, Any], destination: Path = V3 / "ontology.sqlite"
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)
    terms = parsed["terms"]
    ids = {term["id"] for term in terms}
    db.executemany(
        "INSERT INTO terms VALUES(?,?,?,?,?,?,?)",
        (
            (
                term["id"],
                term["name"],
                term["namespace"],
                term["definition"],
                _json(term["definition_xrefs"]),
                int(term["obsolete"]),
                term["source_order"],
            )
            for term in terms
        ),
    )
    synonym_rows = []
    synonym_id = 0
    for term in terms:
        for synonym in term["synonyms"]:
            synonym_rows.append(
                (
                    synonym_id,
                    term["id"],
                    synonym["value"],
                    synonym["scope"],
                    synonym["type"],
                    _json(synonym["xrefs"]),
                    synonym["raw"],
                )
            )
            synonym_id += 1
    db.executemany("INSERT INTO synonyms VALUES(?,?,?,?,?,?,?)", synonym_rows)
    db.executemany(
        "INSERT INTO aliases(alias,term_id,kind,synonym_id) VALUES(?,?,?,?)",
        (
            (term["name"].casefold(), term["id"], "name", None)
            for term in terms
            if term["name"]
        ),
    )
    db.executemany(
        "INSERT INTO aliases(alias,term_id,kind,synonym_id) VALUES(?,?,?,?)",
        ((row[2].casefold(), row[1], "synonym", row[0]) for row in synonym_rows),
    )
    redirects = []
    edges = []
    order = 0
    for term in terms:
        for old in term["alt_ids"]:
            target = term["id"]
            redirects.append(
                (
                    old,
                    target,
                    target if target in ids else None,
                    int(target in ids),
                    "alt_id",
                    order,
                )
            )
            order += 1
        for new in term["replaced_by"]:
            redirects.append(
                (
                    term["id"],
                    new,
                    new if new in ids else None,
                    int(new in ids),
                    "replaced_by",
                    order,
                )
            )
            order += 1
        for new in term["consider"]:
            redirects.append(
                (
                    term["id"],
                    new,
                    new if new in ids else None,
                    int(new in ids),
                    "consider",
                    order,
                )
            )
            order += 1
        for parent in term["is_a"]:
            if parent in ids:
                edges.append((parent, term["id"], "is_a", order))
                order += 1
        for relation in term["relationships"]:
            if relation["relation"] == "part_of" and relation["target"] in ids:
                edges.append((relation["target"], term["id"], "part_of", order))
                order += 1
    db.executemany("INSERT INTO redirects VALUES(?,?,?,?,?,?)", redirects)
    db.executemany("INSERT INTO edges VALUES(?,?,?,?)", edges)
    db.commit()
    counts = {
        "terms": db.execute("SELECT count(*) FROM terms").fetchone()[0],
        "current_terms": db.execute(
            "SELECT count(*) FROM terms WHERE obsolete=0"
        ).fetchone()[0],
        "obsolete_terms": db.execute(
            "SELECT count(*) FROM terms WHERE obsolete=1"
        ).fetchone()[0],
        "synonyms": db.execute("SELECT count(*) FROM synonyms").fetchone()[0],
        "aliases": db.execute("SELECT count(*) FROM aliases").fetchone()[0],
        "redirects": db.execute("SELECT count(*) FROM redirects").fetchone()[0],
        "redirects_by_kind": {
            row[0]: row[1]
            for row in db.execute(
                "SELECT kind,count(*) FROM redirects GROUP BY kind ORDER BY kind"
            )
        },
        "redirects_missing_targets": db.execute(
            "SELECT count(*) FROM redirects WHERE target_present=0"
        ).fetchone()[0],
        "edges": db.execute("SELECT count(*) FROM edges").fetchone()[0],
        "edges_by_relation": {
            row[0]: row[1]
            for row in db.execute(
                "SELECT relation,count(*) FROM edges GROUP BY relation ORDER BY relation"
            )
        },
        "dangling_edges": db.execute(
            "SELECT count(*) FROM edges e LEFT JOIN terms p ON p.id=e.parent LEFT JOIN terms c ON c.id=e.child WHERE p.id IS NULL OR c.id IS NULL"
        ).fetchone()[0],
        "integrity_check": db.execute("PRAGMA integrity_check").fetchone()[0],
    }
    db.close()
    if (
        counts["terms"] != 48329
        or counts["current_terms"] != 38245
        or counts["obsolete_terms"] != 10084
    ):
        raise RuntimeError(f"term count invariant failed: {counts}")
    if counts["dangling_edges"] != 0 or counts["integrity_check"] != "ok":
        raise RuntimeError(f"database integrity invariant failed: {counts}")
    tmp.replace(destination)
    return counts


def main() -> None:
    parsed = parse_obo(OBO)
    counts = build_database(parsed)
    source = {
        "schema": 3,
        "path": str(OBO.relative_to(ROOT)),
        "sha256": parsed["source_sha256"],
        "bytes": parsed["source_bytes"],
        "header": parsed["header"],
        "current_terms": counts["current_terms"],
        "obsolete_terms": counts["obsolete_terms"],
        "total_term_stanzas": counts["terms"],
        "typedef_stanzas_excluded": parsed["typedef_count"],
        "synonym_scopes": ["EXACT", "RELATED", "NARROW", "BROAD"],
        "redirect_policy": "replaced_by authoritative; consider preserved as multi-target advisory; alt_id canonicalized",
        "edge_relations": ["is_a", "part_of"],
        "database_counts": counts,
    }
    atomic_write_json(V3 / "source.json", source)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
