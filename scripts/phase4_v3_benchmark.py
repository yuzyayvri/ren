"""Generate, validate, and freeze the independent Phase 4 v3 benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    OBO,
    OBO_SHA256,
    QUERY_MODEL,
    ROOT,
    V3,
    atomic_write_json,
    code_sha256,
    model_manifest,
    runtime_record,
    sha256_path,
)

ALLOWED_METRICS = {"recall_at_5", "mrr", "exact_id_accuracy", "traversal_correctness"}
OLD_BENCHMARKS = (
    ROOT / "artifacts/phase4_protocol_v1/benchmark.json",
    ROOT / "artifacts/phase4_protocol_v2/benchmark.json",
)


def _source_ref(path: str, locator: str, urls: list[str]) -> dict[str, Any]:
    file_path = ROOT / path
    return {
        "file": path,
        "locator": locator,
        "sha256": sha256_path(file_path),
        "citation_urls": urls,
    }


def _case(
    case_id: str,
    case_type: str,
    query: str,
    gold_ids: list[str],
    eligible_metrics: list[str],
    *,
    input_label: str,
    label_source: dict[str, Any],
    mapping_basis: str,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "id": case_id,
        "case_type": case_type,
        "input_label": input_label,
        "query": query,
        "gold_ids": gold_ids,
        "eligible_metrics": eligible_metrics,
        "mapping_basis": mapping_basis,
        "provenance": {
            "label_source": label_source,
            "ontology_source": {
                "file": "data/ontology/go-basic.obo",
                "sha256": OBO_SHA256,
            },
        },
    }
    result.update(extra)
    return result


def _raw_graph() -> tuple[set[str], list[tuple[str, str, str]]]:
    """Build a traversal graph directly from raw OBO, independent of SQLite."""
    text = OBO.read_text(encoding="utf-8")
    markers = list(re.finditer(r"(?m)^\[([^\]]+)\]\s*$", text))
    terms: list[tuple[str, list[str], list[tuple[str, str]]]] = []
    for index, marker in enumerate(markers):
        if marker.group(1) != "Term":
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        fields: dict[str, list[str]] = defaultdict(list)
        for line in text[marker.end() : end].splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                fields[key].append(value.split(" ! ", 1)[0])
        identifier = fields.get("id", [""])[0]
        if identifier.startswith("GO:"):
            relations = [
                ("is_a", value.split()[0])
                for value in fields.get("is_a", [])
                if value.split()
            ]
            relations.extend(
                (parts[0], parts[1])
                for value in fields.get("relationship", [])
                for parts in [value.split()]
                if len(parts) >= 2 and parts[0] == "part_of"
            )
            terms.append(
                (
                    identifier,
                    fields.get("is_obsolete", ["false"])[0].lower() == "true",
                    relations,
                )
            )
    ids = {identifier for identifier, _, _ in terms}
    edges = [
        (parent, child, relation)
        for child, _, relations in terms
        for relation, parent in relations
        if parent in ids
    ]
    return ids, edges


def _walk_graph(
    edges: list[tuple[str, str, str]], start: str, direction: str, depth: int
) -> list[str]:
    forward: dict[str, list[str]] = defaultdict(list)
    reverse: dict[str, list[str]] = defaultdict(list)
    for parent, child, relation in edges:
        if relation in {"is_a", "part_of"}:
            forward[parent].append(child)
            reverse[child].append(parent)
    adjacency = reverse if direction == "ancestors" else forward
    seen = {start}
    frontier = [start]
    reached: set[str] = set()
    for _ in range(depth):
        next_frontier: list[str] = []
        for node in sorted(frontier):
            for candidate in sorted(adjacency.get(node, [])):
                if candidate not in seen:
                    seen.add(candidate)
                    reached.add(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    return sorted(reached)


def _term_evidence(db: sqlite3.Connection, case: dict[str, Any]) -> None:
    rows = [
        db.execute(
            "SELECT id,name,namespace,definition,obsolete FROM terms WHERE id=?",
            (term_id,),
        ).fetchone()
        for term_id in case["gold_ids"]
    ]
    if any(row is None or row[4] for row in rows):
        raise RuntimeError(f"gold ID is absent/obsolete in {case['id']}")
    case["gold_terms"] = [
        {"id": row[0], "name": row[1], "namespace": row[2], "definition": row[3]}
        for row in rows
    ]


def _build_cases() -> dict[str, list[dict[str, Any]]]:
    pannuke = _source_ref(
        "data/tissue/fold1/Fold 1/README.md",
        "mask class list at lines 9-15",
        ["https://doi.org/10.48550/arXiv.2003.10778"],
    )
    txl = _source_ref(
        "data/blood/txl-pbc/TXL-PBC/classes.txt",
        "class definitions at lines 1-3",
        ["https://doi.org/10.1038/s41597-025-05980-z"],
    )
    aml = _source_ref(
        "artifacts/phase3_protocol_v1/README.md",
        "original AML labels and MYO+MOB blast-equivalent mapping at lines 18-29",
        ["https://doi.org/10.1038/s42256-019-0101-9"],
    )
    go = [
        "https://geneontology.org/docs/download-ontology/",
        "https://geneontology.org/docs/go-citation-policy/",
    ]
    return {
        "tissue_labels": [
            _case(
                "v3-tissue-epithelial",
                "descriptor",
                "glandular epithelial cell differentiation",
                ["GO:0002067"],
                ["recall_at_5", "mrr"],
                input_label="Epithelial",
                label_source=pannuke,
                mapping_basis="PanNuke Epithelial nuclei label; query is the authoritative GO process descriptor GO:0002067.",
                citation_urls=go,
            ),
            _case(
                "v3-tissue-connective",
                "descriptor",
                "connective tissue development",
                ["GO:0061448"],
                ["recall_at_5", "mrr"],
                input_label="Connective/Soft tissue cells",
                label_source=pannuke,
                mapping_basis="PanNuke connective/soft-tissue label; query is the corresponding GO development descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-tissue-inflammatory",
                "descriptor",
                "leukocyte activation involved in inflammatory response",
                ["GO:0002269"],
                ["recall_at_5", "mrr"],
                input_label="Inflammatory",
                label_source=pannuke,
                mapping_basis="PanNuke inflammatory label; query is a GO immune-response descriptor associated with inflammatory leukocytes.",
                citation_urls=go,
            ),
            _case(
                "v3-tissue-dead",
                "descriptor",
                "apoptotic cell death",
                ["GO:0006915"],
                ["recall_at_5", "mrr"],
                input_label="Dead Cells",
                label_source=pannuke,
                mapping_basis="PanNuke Dead Cells label; query is the authoritative GO synonym/descriptor for apoptotic cell death.",
                citation_urls=go,
            ),
            _case(
                "v3-tissue-neoplastic",
                "descriptor",
                "cell population proliferation",
                ["GO:0008283"],
                ["recall_at_5", "mrr"],
                input_label="Neoplastic cells",
                label_source=pannuke,
                mapping_basis="PanNuke Neoplastic label; this is explicitly a related-process retrieval probe, not an identity equivalence claim.",
                citation_urls=go,
            ),
            _case(
                "v3-tissue-cell-division",
                "descriptor",
                "cell division",
                ["GO:0051301"],
                ["recall_at_5", "mrr"],
                input_label="Neoplastic cells",
                label_source=pannuke,
                mapping_basis="PanNuke Neoplastic nuclei label; cell division is an independently adjudicated related-process retrieval probe.",
                citation_urls=go,
            ),
        ],
        "blood_labels": [
            _case(
                "v3-blood-wbc",
                "descriptor",
                "neutrophil differentiation",
                ["GO:0030223"],
                ["recall_at_5", "mrr"],
                input_label="White Blood Cell (WBC), neutrophil morphology",
                label_source=txl,
                mapping_basis="TXL-PBC WBC input with the AML neutrophil morphological label family; GO neutrophil differentiation is the target descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blood-eosinophil",
                "descriptor",
                "eosinophil differentiation",
                ["GO:0030222"],
                ["recall_at_5", "mrr"],
                input_label="EOS",
                label_source=aml,
                mapping_basis="Munich AML EOS morphology label; GO eosinophil differentiation is the target descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blood-monocyte",
                "descriptor",
                "monocyte differentiation",
                ["GO:0030224"],
                ["recall_at_5", "mrr"],
                input_label="MON",
                label_source=aml,
                mapping_basis="Munich AML MON morphology label; GO monocyte differentiation is the target descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blood-lymphocyte",
                "descriptor",
                "lymphocyte differentiation",
                ["GO:0030098"],
                ["recall_at_5", "mrr"],
                input_label="LYT/LYA",
                label_source=aml,
                mapping_basis="Munich AML lymphocyte morphology labels; GO lymphocyte differentiation is the target descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blood-rbc",
                "descriptor",
                "erythrocyte maturation",
                ["GO:0043249"],
                ["recall_at_5", "mrr"],
                input_label="Red Blood Cell (RBC)",
                label_source=txl,
                mapping_basis="TXL-PBC RBC class; GO erythrocyte maturation is the target descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blood-platelet",
                "descriptor",
                "platelet formation",
                ["GO:0030220"],
                ["recall_at_5", "mrr"],
                input_label="Platelet",
                label_source=txl,
                mapping_basis="TXL-PBC Platelet class; GO platelet formation is the target descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blood-myeloid",
                "descriptor",
                "leukocyte differentiation",
                ["GO:0002521"],
                ["recall_at_5", "mrr"],
                input_label="White Blood Cell (WBC), myeloid morphology",
                label_source=txl,
                mapping_basis="TXL-PBC WBC input; leukocyte differentiation is the corresponding GO lineage descriptor.",
                citation_urls=go,
            ),
        ],
        "blast_labels": [
            _case(
                "v3-blast-progenitor",
                "descriptor",
                "myeloid progenitor cell differentiation",
                ["GO:0002318"],
                ["recall_at_5", "mrr"],
                input_label="MYO/MOB blast-equivalent",
                label_source=aml,
                mapping_basis="Matek et al. defines MYO+MOB as blast equivalents; GO myeloid progenitor differentiation is the associated process descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blast-leukocyte",
                "descriptor",
                "myeloid leukocyte differentiation",
                ["GO:0002573"],
                ["recall_at_5", "mrr"],
                input_label="MYO/MOB blast-equivalent",
                label_source=aml,
                mapping_basis="Matek et al. blast-equivalent labels; GO myeloid leukocyte differentiation is a distinct associated descriptor.",
                citation_urls=go,
            ),
            _case(
                "v3-blast-stem",
                "descriptor",
                "hematopoietic stem cell differentiation",
                ["GO:0060218"],
                ["recall_at_5", "mrr"],
                input_label="MYO/MOB blast-equivalent",
                label_source=aml,
                mapping_basis="Matek et al. blast-equivalent labels; hematopoietic stem-cell differentiation is a distinct progenitor-lineage retrieval probe.",
                citation_urls=go,
            ),
        ],
        "identifier_cases": [
            _case(
                "v3-id-canonical-platelet",
                "canonical_id",
                "GO:0030220",
                ["GO:0030220"],
                ["exact_id_accuracy", "mrr"],
                input_label="Platelet",
                label_source=txl,
                mapping_basis="Canonical GO identifier must take exact-ID precedence over lexical/vector processing.",
                citation_urls=go,
            ),
            _case(
                "v3-id-canonical-neutrophil",
                "canonical_id",
                "GO:0030223",
                ["GO:0030223"],
                ["exact_id_accuracy", "mrr"],
                input_label="NGS/NGB neutrophil morphology",
                label_source=aml,
                mapping_basis="Canonical GO identifier must resolve to itself with exact-ID precedence.",
                citation_urls=go,
            ),
            _case(
                "v3-id-alternate-root",
                "alternate_id",
                "GO:0000004",
                ["GO:0008150"],
                ["exact_id_accuracy", "mrr"],
                input_label="WBC process lookup",
                label_source=txl,
                mapping_basis="GO alt_id GO:0000004 is authoritatively redirected to canonical GO:0008150.",
                citation_urls=go,
            ),
            _case(
                "v3-id-alternate-division",
                "alternate_id",
                "GO:0000037",
                ["GO:0006633"],
                ["exact_id_accuracy", "mrr"],
                input_label="Blood-cell process lookup",
                label_source=txl,
                mapping_basis="GO alt_id GO:0000037 is authoritatively redirected to active canonical GO:0006633.",
                citation_urls=go,
            ),
        ],
        "obsolete_cases": [
            _case(
                "v3-obsolete-replaced",
                "obsolete_id",
                "GO:0000003",
                ["GO:0022414"],
                ["recall_at_5", "mrr"],
                input_label="Historical reproduction term",
                label_source=aml,
                mapping_basis="Obsolete GO identifier with replaced_by; replaced_by is authoritative.",
                resolution_kind="replaced_by",
                citation_urls=go,
            ),
            _case(
                "v3-obsolete-consider",
                "obsolete_id",
                "GO:0000005",
                ["GO:0042254", "GO:0044183", "GO:0140309"],
                ["recall_at_5", "mrr"],
                input_label="Historical chaperone term",
                label_source=aml,
                mapping_basis="Obsolete GO identifier with three consider targets; all advisory targets must be preserved.",
                resolution_kind="consider",
                citation_urls=go,
            ),
        ],
        "synonym_cases": [
            _case(
                "v3-syn-exact",
                "synonym",
                "neutrophil granulocytopoiesis",
                ["GO:0030223"],
                ["recall_at_5", "mrr"],
                input_label="NGS/NGB neutrophil morphology",
                label_source=aml,
                mapping_basis="Real EXACT synonym in the frozen OBO source.",
                synonym_scope="EXACT",
                citation_urls=go,
            ),
            _case(
                "v3-syn-related",
                "synonym",
                "eosinophil cell development",
                ["GO:0030222"],
                ["recall_at_5", "mrr"],
                input_label="EOS",
                label_source=aml,
                mapping_basis="Real RELATED synonym in the frozen OBO source.",
                synonym_scope="RELATED",
                citation_urls=go,
            ),
            _case(
                "v3-syn-narrow",
                "synonym",
                "apoptotic program",
                ["GO:0006915"],
                ["recall_at_5", "mrr"],
                input_label="Dead Cells",
                label_source=pannuke,
                mapping_basis="Real NARROW synonym in the frozen OBO source.",
                synonym_scope="NARROW",
                citation_urls=go,
            ),
            _case(
                "v3-syn-broad",
                "synonym",
                "cell suicide",
                ["GO:0006915"],
                ["recall_at_5", "mrr"],
                input_label="Dead Cells",
                label_source=pannuke,
                mapping_basis="Real BROAD synonym in the frozen OBO source.",
                synonym_scope="BROAD",
                citation_urls=go,
            ),
        ],
        "ambiguous_cases": [
            _case(
                "v3-amb-apoptosis",
                "ambiguous",
                "apoptosis",
                ["GO:0006915", "GO:0097194"],
                ["recall_at_5", "mrr"],
                input_label="Dead Cells",
                label_source=pannuke,
                mapping_basis="The exact alias is present as a NARROW synonym for two active GO terms; both candidates are intentionally gold.",
                expected_scope="NARROW",
                citation_urls=go,
            ),
        ],
        "rare_cases": [
            _case(
                "v3-rare-neutrophil-degranulation",
                "rare",
                "neutrophil homeostasis",
                ["GO:0001780"],
                ["recall_at_5", "mrr"],
                input_label="NGS/NGB neutrophil morphology",
                label_source=aml,
                mapping_basis="Active GO term has exactly one active alias (its canonical name) in this release.",
                frequency_criterion={"active_term_count": 1, "alias_count_max": 1},
                citation_urls=go,
            ),
            _case(
                "v3-rare-erythrocyte-enucleation",
                "rare",
                "erythrocyte enucleation",
                ["GO:0043131"],
                ["recall_at_5", "mrr"],
                input_label="Red Blood Cell (RBC)",
                label_source=txl,
                mapping_basis="Active GO term has exactly one active alias (its canonical name) in this release.",
                frequency_criterion={"active_term_count": 1, "alias_count_max": 1},
                citation_urls=go,
            ),
        ],
        "traversal_cases": [
            _case(
                "v3-traverse-neutrophil-anc",
                "traversal",
                "GO:0030223",
                [],
                ["traversal_correctness"],
                input_label="NGS/NGB neutrophil morphology",
                label_source=aml,
                mapping_basis="Independent raw-OBO graph walk of is_a/part_of ancestors.",
                direction="ancestors",
                depth=3,
                relations=["is_a", "part_of"],
                citation_urls=go,
            ),
            _case(
                "v3-traverse-myeloid-desc",
                "traversal",
                "GO:0002573",
                [],
                ["traversal_correctness"],
                input_label="MYO/MOB blast-equivalent",
                label_source=aml,
                mapping_basis="Independent raw-OBO graph walk of is_a/part_of descendants.",
                direction="descendants",
                depth=2,
                relations=["is_a", "part_of"],
                citation_urls=go,
            ),
        ],
    }


def _flatten(strata: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [case for cases in strata.values() for case in cases]


def generate_candidate() -> dict[str, Any]:
    if (V3 / "benchmark_candidate.json").exists():
        raise RuntimeError(
            "v3 benchmark candidate already exists; lifecycle is append-only"
        )
    strata = _build_cases()
    _, edges = _raw_graph()
    for case in strata["traversal_cases"]:
        case["gold_ids"] = _walk_graph(
            edges, case["query"], case["direction"], case["depth"]
        )
        case["gold_generation"] = {
            "method": "independent raw-OBO graph walk",
            "source_sha256": OBO_SHA256,
            "excluded_start": True,
        }
    db = sqlite3.connect(V3 / "ontology.sqlite")
    for case in _flatten(strata):
        _term_evidence(db, case)
    db.close()
    candidate = {
        "schema": 3,
        "status": "candidate_pending_validation",
        "benchmark_id": "phase4_v3_acceptance_new",
        "acceptance_cases_are_new": True,
        "retired_regression_boundaries": [
            "artifacts/phase4_protocol_v1",
            "artifacts/phase4_protocol_v2",
        ],
        "source_sha256": OBO_SHA256,
        "strata": strata,
        "adjudication": {
            "source_policy": "GO basic OBO release is authoritative for names, definitions, synonyms, redirects, and graph edges",
            "official_go_sources": [
                "https://geneontology.org/docs/download-ontology/",
                "https://geneontology.org/docs/go-citation-policy/",
            ],
            "dataset_sources": [
                "https://doi.org/10.48550/arXiv.2003.10778",
                "https://doi.org/10.1038/s41597-025-05980-z",
                "https://doi.org/10.1038/s42256-019-0101-9",
            ],
            "semantic_caution": "Cell-label mappings to GO biological processes are retrieval probes; they do not assert that a morphology class is identical to a GO process or clinical diagnosis.",
        },
    }
    # Ensure acceptance does not silently reuse any v1/v2 acceptance tuple.
    old_tuples = set()
    for path in OLD_BENCHMARKS:
        old = json.loads(path.read_text(encoding="utf-8"))
        for old_case in [
            item for group in old.get("strata", {}).values() for item in group
        ]:
            old_tuples.add(
                (
                    old_case.get("case_type"),
                    old_case.get("query"),
                    tuple(old_case.get("gold_ids", [])),
                )
            )
    for case in _flatten(strata):
        if (case["case_type"], case["query"], tuple(case["gold_ids"])) in old_tuples:
            raise RuntimeError(f"v3 acceptance case reuses an old tuple: {case['id']}")
    atomic_write_json(V3 / "benchmark_candidate.json", candidate)
    return candidate


def validate_candidate(
    candidate_path: Path = V3 / "benchmark_candidate.json",
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        candidate.get("schema") != 3
        or candidate.get("status") != "candidate_pending_validation"
    ):
        raise RuntimeError("v3 candidate status/schema invalid")
    if candidate.get("source_sha256") != OBO_SHA256:
        raise RuntimeError("v3 candidate source binding mismatch")
    required = {
        "tissue_labels",
        "blood_labels",
        "blast_labels",
        "identifier_cases",
        "obsolete_cases",
        "synonym_cases",
        "ambiguous_cases",
        "rare_cases",
        "traversal_cases",
    }
    if set(candidate.get("strata", {})) != required:
        raise RuntimeError("v3 benchmark strata mismatch")
    db = sqlite3.connect(V3 / "ontology.sqlite")
    _, graph_edges = _raw_graph()
    seen_ids: set[str] = set()
    for case in _flatten(candidate["strata"]):
        if (
            case["id"] in seen_ids
            or not case.get("query")
            or not case.get("gold_ids")
            and case["case_type"] != "traversal"
        ):
            raise RuntimeError(f"invalid/duplicate benchmark case: {case.get('id')}")
        seen_ids.add(case["id"])
        if set(case.get("eligible_metrics", [])) - ALLOWED_METRICS:
            raise RuntimeError(f"unknown eligible metric: {case['id']}")
        if not case.get("provenance", {}).get("label_source", {}).get("sha256"):
            raise RuntimeError(f"missing label provenance: {case['id']}")
        if case["case_type"] != "traversal":
            _term_evidence(db, case)
        if case["case_type"] in {"canonical_id", "alternate_id", "obsolete_id"}:
            if not re.fullmatch(r"GO:\d{7}", case["query"]):
                raise RuntimeError(f"identifier case is not a GO ID: {case['id']}")
            if case["case_type"] == "canonical_id":
                row = db.execute(
                    "SELECT obsolete FROM terms WHERE id=?", (case["query"],)
                ).fetchone()
                if not row or row[0] != 0 or case["gold_ids"] != [case["query"]]:
                    raise RuntimeError(
                        f"canonical-ID precedence case invalid: {case['id']}"
                    )
            elif case["case_type"] == "alternate_id":
                targets = [
                    row[0]
                    for row in db.execute(
                        "SELECT new_id FROM redirects WHERE old_id=? AND kind='alt_id' ORDER BY new_id",
                        (case["query"],),
                    )
                ]
                if targets != case["gold_ids"]:
                    raise RuntimeError(f"alternate-ID mapping invalid: {case['id']}")
            else:
                kind = case.get("resolution_kind")
                targets = [
                    row[0]
                    for row in db.execute(
                        "SELECT new_id FROM redirects WHERE old_id=? AND kind=? ORDER BY new_id",
                        (case["query"], kind),
                    )
                ]
                if targets != case["gold_ids"]:
                    raise RuntimeError(f"obsolete mapping invalid: {case['id']}")
        if case["case_type"] == "synonym":
            rows = db.execute(
                "SELECT a.term_id,s.scope FROM aliases a JOIN synonyms s ON s.synonym_id=a.synonym_id JOIN terms t ON t.id=a.term_id WHERE a.alias=? AND t.obsolete=0 AND a.kind='synonym' ORDER BY a.term_id",
                (case["query"].casefold(),),
            ).fetchall()
            if [row[0] for row in rows] != case["gold_ids"] or {
                row[1] for row in rows
            } != {case["synonym_scope"]}:
                raise RuntimeError(f"synonym scope/target invalid: {case['id']}")
        if case["case_type"] == "ambiguous":
            rows = db.execute(
                "SELECT DISTINCT a.term_id,s.scope FROM aliases a JOIN synonyms s ON s.synonym_id=a.synonym_id JOIN terms t ON t.id=a.term_id WHERE a.alias=? AND t.obsolete=0 AND a.kind='synonym' ORDER BY a.term_id",
                (case["query"].casefold(),),
            ).fetchall()
            if (
                [row[0] for row in rows] != case["gold_ids"]
                or len(rows) < 2
                or {row[1] for row in rows} != {case["expected_scope"]}
            ):
                raise RuntimeError(f"ambiguous alias invalid: {case['id']}")
        if case["case_type"] == "rare":
            active_ids = {
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT a.term_id FROM aliases a JOIN terms t ON t.id=a.term_id WHERE a.alias=? AND t.obsolete=0",
                    (case["query"].casefold(),),
                )
            }
            alias_count = db.execute(
                "SELECT count(*) FROM aliases WHERE term_id=?", (case["gold_ids"][0],)
            ).fetchone()[0]
            criterion = case["frequency_criterion"]
            if (
                active_ids != set(case["gold_ids"])
                or criterion["active_term_count"] != len(active_ids)
                or alias_count > criterion["alias_count_max"]
            ):
                raise RuntimeError(f"rare criterion invalid: {case['id']}")
        if case["case_type"] == "traversal":
            if (
                not re.fullmatch(r"GO:\d{7}", case["query"])
                or case["depth"] <= 0
                or not set(case["relations"]) <= {"is_a", "part_of"}
            ):
                raise RuntimeError(f"traversal contract invalid: {case['id']}")
            expected = _walk_graph(
                graph_edges, case["query"], case["direction"], case["depth"]
            )
            if (
                expected != case["gold_ids"]
                or case.get("gold_generation", {}).get("source_sha256") != OBO_SHA256
            ):
                raise RuntimeError(f"traversal gold set invalid: {case['id']}")
    db.close()
    if len(seen_ids) != 31:
        raise RuntimeError(f"unexpected v3 benchmark case count: {len(seen_ids)}")
    return candidate


def protocol_candidate() -> dict[str, Any]:
    source = json.loads((V3 / "source.json").read_text(encoding="utf-8"))
    article = model_manifest(ARTICLE_MODEL)
    query = model_manifest(QUERY_MODEL)
    return {
        "schema": 3,
        "status": "candidate_pending_validation",
        "release": "GO basic archive 2026-06-19",
        "obo_internal_data_version": "releases/2026-06-15",
        "source": {
            "sha256": source["sha256"],
            "bytes": source["bytes"],
            "path": source["path"],
            "license_url": source["header"]["license_url"],
        },
        "corpus": {
            "kind": "go_term_documents",
            "title_field": "GO-ID + canonical name",
            "abstract_field": "definition + ordered synonyms",
            "passage_corpus_present": False,
        },
        "encoders": {
            "query": {
                "repository": query["repository"],
                "revision": query["revision"],
                "max_length": 64,
                "pooling": "last_hidden_state[:,0,:]",
                "normalization": "L2",
            },
            "article": {
                "repository": article["repository"],
                "revision": article["revision"],
                "max_length": 512,
                "pooling": "last_hidden_state[:,0,:]",
                "normalization": "L2",
                "input_fields": ["title", "abstract"],
            },
        },
        "index": {
            "backend": "numpy_flat_cosine",
            "dtype": "float32",
            "order": "GO ID ascending",
            "tie_break": "GO ID ascending",
            "rows": 38245,
            "dim": 768,
        },
        "hybrid": {
            "method": "RRF",
            "cutoff": 60,
            "weights": {"symbolic": 1.0, "vector": 1.0},
            "rank_base": 1,
            "score": "sum(weight/(cutoff+rank)), rank starts at 1",
        },
        "traversal": {
            "relations": ["is_a", "part_of"],
            "directions": ["ancestors", "descendants"],
            "max_depth": 10,
            "depth_is_relation_hops": True,
            "exclude_start": True,
            "depth_zero_returns": [],
            "metric_excluded_from_recall_mrr": True,
        },
        "metrics": {
            "recall_at_5": "top-five set recall over eligible cases",
            "mrr": "full-corpus reciprocal rank of first relevant target",
            "exact_id_accuracy": "top-1 accuracy over canonical/alternate ID cases",
            "traversal_correctness": "exact set equality",
        },
        "latency": {
            "warmup_repeats": 2,
            "measured_repeats": 5,
            "p95": "nearest-rank ceil(0.95*n)",
            "cache_policy": "models, DB, documents, and mmap index are loaded before timing",
            "scope": "query plus ranking; hybrid timed directly",
        },
        "gates": {
            "recall_at_5": 0.8,
            "mrr": 0.7,
            "exact_id_accuracy": 0.95,
            "traversal_correctness": 1.0,
            "p95_latency_ms": 200,
            "deterministic_reproduction": True,
            "provenance": True,
            "offline": True,
        },
        "benchmark": {
            "path": "artifacts/phase4_protocol_v3/benchmark.json",
            "acceptance_cases": "new independent v3 cases only",
            "v1_v2": "exposed regression-only",
        },
        "provenance": {
            "code_sha256": code_sha256(),
            "runtime": runtime_record(),
            "database_sha256": sha256_path(V3 / "ontology.sqlite"),
            "documents_sha256": sha256_path(V3 / "documents.json"),
            "article_model_manifest_sha256": article["manifest_sha256"],
            "query_model_manifest_sha256": query["manifest_sha256"],
        },
        "recovery": "Phase 4 v3 is a separately frozen prospective benchmark/evaluation after reviewed correction; v1/v2 remain invalid exposed regression evidence. A recovery replay is allowed only after an invariant failure is reviewed.",
        "no_llm": True,
    }


def _append_lifecycle(event: dict[str, Any]) -> None:
    path = V3 / "lifecycle.jsonl"
    payload = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        import os

        os.fsync(handle.fileno())


def freeze() -> dict[str, Any]:
    if (
        (V3 / "lifecycle.jsonl").exists()
        or (V3 / "benchmark.json").exists()
        or (V3 / "protocol.json").exists()
    ):
        raise RuntimeError(
            "v3 lifecycle already started; freeze is one-writer append-only"
        )
    candidate = validate_candidate()
    protocol = protocol_candidate()
    protocol["status"] = "frozen_prospective_no_evaluation"
    benchmark = dict(candidate)
    benchmark["status"] = "frozen_prospective_no_evaluation"
    atomic_write_json(V3 / "protocol.json", protocol)
    atomic_write_json(V3 / "benchmark.json", benchmark)
    protocol_sha = sha256_path(V3 / "protocol.json")
    benchmark_sha = sha256_path(V3 / "benchmark.json")
    _append_lifecycle(
        {
            "event": "freeze",
            "protocol_sha256": protocol_sha,
            "benchmark_sha256": benchmark_sha,
            "evaluation_exposed": False,
            "benchmark_case_count": 31,
        }
    )
    atomic_write_json(
        V3 / "freeze_manifest.json",
        {
            "schema": 1,
            "protocol_sha256": protocol_sha,
            "benchmark_sha256": benchmark_sha,
            "lifecycle": "lifecycle.jsonl",
            "status": "frozen_prospective_no_evaluation",
        },
    )
    return {"protocol_sha256": protocol_sha, "benchmark_sha256": benchmark_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["candidate", "validate", "freeze"])
    args = parser.parse_args()
    if args.stage == "candidate":
        result = generate_candidate()
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "case_count": len(_flatten(result["strata"])),
                },
                sort_keys=True,
            )
        )
    elif args.stage == "validate":
        result = validate_candidate()
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "case_count": len(_flatten(result["strata"])),
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(freeze(), sort_keys=True))


if __name__ == "__main__":
    main()
