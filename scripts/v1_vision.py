"""v1 production vision: YOLO detect, Path Foundation embed, frozen head classify.

Sanctioned production wrapper around the proven Phase 3 components. Binds
exact checkpoints by hash, reuses frozen thresholds/preprocessing/taxonomy,
and emits machine findings with full model provenance. Never mixes dataset
labels with live inference output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
YOLO_CHECKPOINT = ROOT / "runs/detect/artifacts/phase3_detector_v2/yolo11n_from_scratch/weights/best.pt"
YOLO_SHA256 = "2a98dc5af8e8623a78cb8e641e737c76df73c7b7c9c8cc7aea4980e6d1171a45"
PF_MODEL = ROOT / "models/vision/path-foundation"
HEAD = ROOT / "artifacts/phase3_txl_classification_v1/head.pkl"
STANDARDIZER = ROOT / "artifacts/phase3_txl_classification_v1/standardizer.npz"

CONF_THRESHOLD = 0.65
NMS_IOU = 0.70
MAX_DET = 300
CLASSES = ("WBC", "RBC", "Platelets")


class VisionError(RuntimeError):
    """Inference precondition or execution failure."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


_DETECTOR = None
_ENCODER = None
_HEAD = None


def load_detector() -> Any:
    global _DETECTOR
    if _DETECTOR is None:
        if not YOLO_CHECKPOINT.is_file() or sha256_file(YOLO_CHECKPOINT) != YOLO_SHA256:
            raise VisionError("YOLO checkpoint missing or hash mismatch")
        from ultralytics import YOLO

        _DETECTOR = YOLO(str(YOLO_CHECKPOINT), task="detect")
    return _DETECTOR


def load_encoder() -> Any:
    global _ENCODER
    if _ENCODER is None:
        import tensorflow as tf

        from scripts.phase3_frozen_heads import normalize_for_path_foundation

        model = tf.saved_model.load(str(PF_MODEL))
        sig = model.signatures.get("serving_default")
        if sig is None:
            sig = next(iter(model.signatures.values()))
        name = next(iter(sig.structured_input_signature[1]))

        def run(batch: Any) -> Any:
            import numpy as np

            out = sig(**{name: tf.convert_to_tensor(normalize_for_path_foundation(batch))})
            feats = next(iter(out.values())).numpy()
            if feats.ndim == 3:
                feats = feats.mean(axis=1)
            if feats.shape[1:] != (384,):
                raise VisionError(f"unexpected Path Foundation output {feats.shape}")
            return np.asarray(feats, np.float32)

        _ENCODER = run
    return _ENCODER


def load_head() -> Any:
    global _HEAD
    if _HEAD is None:
        import numpy as np

        with HEAD.open("rb") as handle:
            model = pickle.load(handle)
        stats = np.load(STANDARDIZER)
        _HEAD = (model, np.asarray(stats["mean"], np.float32), np.asarray(stats["scale"], np.float32))
    return _HEAD


def analyze_array(image: Any, *, specimen_id: str = "adhoc") -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from scripts.phase3_frozen_heads import crop_txl

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise VisionError("vision input must be uint8 RGB array")
    started = time.time()
    detector = load_detector()
    height, width, _ = image.shape
    result = detector.predict(Image.fromarray(image), conf=CONF_THRESHOLD, iou=NMS_IOU,
                              max_det=MAX_DET, device="cpu", verbose=False)[0]
    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes):
        xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=float)
        cls = np.asarray(boxes.cls.cpu().numpy(), dtype=int)
        conf = np.asarray(boxes.conf.cpu().numpy(), dtype=float)
        order = np.argsort(-conf, kind="stable")
        for index in order[:MAX_DET]:
            detections.append({"box": [float(v) for v in xyxy[index]],
                               "class_id": int(cls[index]),
                               "det_confidence": float(conf[index])})
    findings: list[dict[str, Any]] = []
    if detections:
        encoder = load_encoder()
        model, mean, scale = load_head()
        crops = np.asarray([crop_txl(image, tuple(d["box"])) for d in detections], dtype=np.uint8)
        embeddings = encoder(crops.reshape(-1, 224, 224, 3))
        probabilities = model.predict_proba((embeddings - mean) / scale)
        for position, (detection, proba) in enumerate(zip(detections, probabilities)):
            label = int(np.argmax(proba))
            findings.append({
                "finding_id": f"F{position + 1}",
                "label": CLASSES[label],
                "confidence": float(proba[label]),
                "det_confidence": detection["det_confidence"],
                "region": {"box_xyxy": detection["box"], "image_width": width, "image_height": height},
                "source": "machine",
                "specimen_id": specimen_id,
            })
    return {
        "schema": "v1-vision-v1",
        "specimen_id": specimen_id,
        "image_width": width,
        "image_height": height,
        "detections": len(detections),
        "findings": findings,
        "models": {
            "detector": {"checkpoint": str(YOLO_CHECKPOINT.relative_to(ROOT)), "sha256": YOLO_SHA256,
                         "conf_threshold": CONF_THRESHOLD, "nms_iou": NMS_IOU, "max_det": MAX_DET},
            "head": {"checkpoint": str(HEAD.relative_to(ROOT)),
                     "sha256": sha256_file(HEAD), "classes": list(CLASSES)},
        },
        "latency_s": time.time() - started,
    }


def analyze_specimen(specimen_id: str, *, registry: Path | None = None) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from scripts import v1_ingest as ingest

    directory = (registry or ingest.REGISTRY) / specimen_id
    record = json.loads((directory / "specimen.json").read_text(encoding="utf-8"))
    image = np.asarray(Image.open(directory / "normalized.png").convert("RGB"), dtype=np.uint8)
    result = analyze_array(image, specimen_id=specimen_id)
    result["specimen_sha256"] = record["content_sha256"]
    (directory / "vision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_specimen(args.specimen), indent=2, sort_keys=True)[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
