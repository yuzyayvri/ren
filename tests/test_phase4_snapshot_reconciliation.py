import json

from scripts.phase4_snapshot_reconciliation import (
    EXPECTED_FINAL_CODE,
    REC,
    load_bound_query,
    verify,
    verify_intended_state,
)


def test_intended_state_holds():
    proof = verify_intended_state()
    assert proof["final_code_sha256"] == EXPECTED_FINAL_CODE
    assert proof["nine_unchanged"] is True
    assert proof["data_match"] is True
    assert len(proof["boundary_sidecars"]) == 4


def test_reconciliation_manifest_matches_fresh_proof():
    manifest = json.loads((REC / "reconciliation_manifest.json").read_text())
    assert manifest["mutated_sealed_files"] == []
    assert manifest["mutated_sealed_code"] == []
    assert verify()["status"] == "verified"


def test_authorization_forbids_sealed_mutation():
    auth = json.loads((REC / "authorization.json").read_text())
    forbidden = " ".join(auth["forbidden"])
    for term in ("artifact edit", "CODE_FILES", "quality rerun", "tuning"):
        assert term in forbidden


def test_bound_resolve_and_traverse():
    query = load_bound_query()
    resolved = query.resolve_identifier("GO:0006915")
    assert resolved is not None and resolved["targets"] == ["GO:0006915"]
    assert query.traverse("GO:0006915", direction="ancestors", depth=1) == ["GO:0012501"]
    assert load_bound_query() is query


def test_bound_symbolic_matches_sealed_ranking():
    query = load_bound_query()
    sealed = None
    with (REC.parents[1] / "rankings.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if '"v3-amb-apoptosis"' in line and '"symbolic"' in line:
                record = json.loads(line)
                if record["case_id"] == "v3-amb-apoptosis" and record["mode"] == "symbolic":
                    sealed = record["ranking"]
                    break
    assert sealed is not None and len(sealed) == 38245
    assert query.symbolic("apoptosis") == sealed
