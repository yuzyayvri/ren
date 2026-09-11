"""Audit the immutable GO source, MedCPT models, runtime, and source labels."""

from __future__ import annotations

import re

from scripts.phase4_v3_common import (
    ARTICLE_MODEL,
    OBO,
    OBO_SHA256,
    QUERY_MODEL,
    ROOT,
    V3,
    atomic_write_json,
    code_file_hashes,
    code_sha256,
    model_manifest,
    runtime_record,
    sha256_path,
)


def _header(text: str) -> dict[str, object]:
    first = re.search(r"(?m)^\[(?:Term|Typedef)\]\s*$", text)
    header = text[: first.start() if first else len(text)]
    version = re.search(r"(?m)^format-version:\s*(\S+)\s*$", header)
    data_version = re.search(r"(?m)^data-version:\s*(\S+)\s*$", header)
    license_match = re.search(r"(?m)^property_value:\s*terms:license\s+(\S+)", header)
    return {
        "format_version": version.group(1) if version else None,
        "data_version": data_version.group(1) if data_version else None,
        "license_url": license_match.group(1) if license_match else None,
        "header_bytes": len(header.encode("utf-8")),
    }


def source_audit() -> dict[str, object]:
    if not OBO.is_file():
        raise RuntimeError(f"missing GO source: {OBO}")
    text = OBO.read_text(encoding="utf-8")
    digest = sha256_path(OBO)
    if digest != OBO_SHA256:
        raise RuntimeError(f"GO source hash mismatch: {digest}")
    term_stanzas = len(re.findall(r"(?m)^\[Term\]\s*$", text))
    typedef_stanzas = len(re.findall(r"(?m)^\[Typedef\]\s*$", text))
    header = _header(text)
    if header["format_version"] != "1.2" or not header["data_version"]:
        raise RuntimeError("GO header version/data-version is incomplete")
    if header["license_url"] != "http://creativecommons.org/licenses/by/4.0/":
        raise RuntimeError(
            "GO source does not declare the expected CC-BY-4.0 license URL"
        )
    return {
        "schema": 2,
        "path": str(OBO.relative_to(ROOT)),
        "sha256": digest,
        "bytes": OBO.stat().st_size,
        "header": header,
        "term_stanzas": term_stanzas,
        "typedef_stanzas_excluded": typedef_stanzas,
        "official_sources": [
            "https://geneontology.org/docs/download-ontology/",
            "https://geneontology.org/docs/go-citation-policy/",
        ],
    }


def label_sources() -> list[dict[str, object]]:
    paths = [
        ROOT / "data/tissue/fold1/Fold 1/README.md",
        ROOT / "data/blood/txl-pbc/README.md",
        ROOT / "data/blood/txl-pbc/TXL-PBC/classes.txt",
        ROOT / "artifacts/phase3_protocol_v1/README.md",
        ROOT / "artifacts/phase3_protocol_v1/source_manifest.json",
    ]
    out = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"missing label provenance file: {path}")
        out.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )
    return out


def main() -> None:
    V3.mkdir(parents=True, exist_ok=True)
    source = source_audit()
    query = model_manifest(QUERY_MODEL)
    article = model_manifest(ARTICLE_MODEL)
    models = {"query": query, "article": article}
    runtime = runtime_record()
    code = {"sha256": code_sha256(), "files": code_file_hashes()}
    atomic_write_json(
        V3 / "source_audit.json",
        {**source, "label_sources": label_sources(), "code": code},
    )
    atomic_write_json(
        V3 / "model_audit.json", {"schema": 2, "models": models, "code": code}
    )
    atomic_write_json(
        V3 / "runtime_audit.json",
        {"schema": 1, **runtime, "code_sha256": code["sha256"]},
    )
    print("source/model/runtime audit complete")


if __name__ == "__main__":
    main()
