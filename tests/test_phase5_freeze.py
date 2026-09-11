import json
from pathlib import Path

PROTO = Path("protocols/phase5_v1")
RUNTIME_MODULES = ["scripts/phase5_packet.py", "scripts/phase5_client.py",
                   "scripts/phase5_validate.py", "scripts/phase5_render.py",
                   "scripts/phase5_evaluate.py"]


def test_frozen_manifest_matches_tracked_files():
    import hashlib

    manifest = json.loads((PROTO / "freeze_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        h = hashlib.sha256()
        with (PROTO / name).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == digest, name


def test_benchmark_splits_are_isolated():
    dev = json.loads((PROTO / "benchmark_dev.json").read_text())["cases"]
    final = json.loads((PROTO / "benchmark_final.json").read_text())["cases"]
    dev_ids = {c["case_id"] for c in dev}
    final_ids = {c["case_id"] for c in final}
    assert dev_ids.isdisjoint(final_ids)
    assert all(i.startswith("dev-") for i in dev_ids)
    assert all(i.startswith("final-") for i in final_ids)


def test_runtime_modules_never_open_final_set():
    for module in RUNTIME_MODULES:
        text = Path(module).read_text()
        assert "benchmark_final" not in text, module


def test_fixture_go_ids_resolve_in_sealed_v3_corpus():
    from scripts.phase5_protocol import check_go_membership

    proof = check_go_membership()
    assert proof["cases"] == {"benchmark_dev.json": 8, "benchmark_final.json": 6}
    assert proof["corpus_rows"] == 38245


def test_lifecycle_genesis_present():
    assert "part1_frozen" in (PROTO / "lifecycle.jsonl").read_text()
