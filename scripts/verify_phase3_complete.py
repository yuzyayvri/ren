"""Read-only cross-artifact verifier for completed Phase 3 blood stages."""
import json
from pathlib import Path

ROOT = Path("/home/yuzy/ren")


def main():
    protocol = ROOT / "artifacts/phase3_protocol_v1/source_manifest.json"
    det = ROOT / "artifacts/phase3_detector_v2"
    txl = ROOT / "artifacts/phase3_txl_classification_v1"
    aml = ROOT / "artifacts/phase3_aml_classification_v1"
    assert protocol.exists() and (det / "detector_test_report.json").exists()
    assert (det / "recovery" / "recovered_report.json").exists()
    for stage in (txl, aml):
        for name in ("config.json", "selection.json", "standardizer.npz", "head.pkl", "test_exposed.marker", "report.json"):
            assert (stage / name).exists(), (stage, name)
        report = json.loads((stage / "report.json").read_text())
        assert all(report["gates"].values())
    assert not (det / "recovery" / "recovery_attempt.json").read_text().count('"count": 2')
    print("phase3 cross-artifact verifier: PASS")


if __name__ == "__main__":
    main()
