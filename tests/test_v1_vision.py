import numpy as np
import pytest

from scripts import v1_vision as vision


def test_checkpoint_binding(monkeypatch):
    monkeypatch.setattr(vision, "sha256_file", lambda p: "wrong")
    with pytest.raises(vision.VisionError):
        vision.load_detector()


def test_input_validation():
    with pytest.raises(vision.VisionError):
        vision.analyze_array(np.zeros((10, 10), np.uint8))
    with pytest.raises(vision.VisionError):
        vision.analyze_array(np.zeros((10, 10, 3), np.float32))


def test_empty_findings_path(monkeypatch):
    class Empty:
        def predict(self, image, **kwargs):
            class Result:
                boxes = None
            return [Result()]

    monkeypatch.setattr(vision, "load_detector", lambda: Empty())
    out = vision.analyze_array(np.zeros((300, 300, 3), np.uint8), specimen_id="s1")
    assert out["findings"] == [] and out["detections"] == 0
    assert out["specimen_id"] == "s1"


def test_live_txl_val_image():
    import json
    from pathlib import Path

    try:
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("vision stack needs GL/X11 libs; run under scripts/vision_python.sh")

    from scripts import v1_ingest as ingest

    record = ingest.ingest_file(
        Path("data/blood/txl-pbc/TXL-PBC/images/val/10c5112dfae4533cb9fa8f6f73a49274.png"))
    out = vision.analyze_specimen(record["specimen_id"])
    assert out["specimen_sha256"] == record["content_sha256"]
    assert len(out["findings"]) >= 10
    labels = {f["label"] for f in out["findings"]}
    assert labels <= {"WBC", "RBC", "Platelets"} and "RBC" in labels
    for finding in out["findings"]:
        box = finding["region"]["box_xyxy"]
        assert 0 <= box[0] < box[2] <= out["image_width"]
        assert 0 <= finding["confidence"] <= 1
        assert finding["source"] == "machine"
    assert out["models"]["detector"]["sha256"] == vision.YOLO_SHA256
    assert (Path("artifacts/v1_specimens") / record["specimen_id"] / "vision.json").is_file()
    print(json.dumps({"findings": len(out["findings"]),
                      "latency_s": round(out["latency_s"], 1)}))
