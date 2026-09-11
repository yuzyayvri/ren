from pathlib import Path

from scripts.audit_phase3_datasets import aml_audit, txl_audit


def test_txl_audit_counts_and_schema(tmp_path: Path):
    root = tmp_path / "txl" / "TXL-PBC"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "classes.txt").write_text("WBC\nRBC\nPlatelets\n")
    (root / "images" / "train" / "a.png").write_bytes(b"image")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    report = txl_audit(tmp_path / "txl")
    assert report["splits"]["train"]["images"] == 1
    assert report["splits"]["train"]["class_counts"] == {"0": 1}
    assert report["splits"]["train"]["malformed_rows"] == 0


def test_aml_audit_reports_annotations_and_classes(tmp_path: Path):
    root = tmp_path / "aml"
    (root / "data" / "data" / "MYO").mkdir(parents=True)
    (root / "data" / "data" / "MYO" / "x.tiff").write_bytes(b"x")
    (root / "annotations.dat").write_text("MYO/MYO_0001.tiff MYO nan nan\n")
    (root / "annotations_augmented.dat").write_text("")
    (root / "abbreviations.txt").write_text("MYO Myeloblast\n")
    report = aml_audit(root)
    assert report["annotation_files"]["annotations.dat"]["lines"] == 1
    assert report["class_directories"]["MYO"] == 1
