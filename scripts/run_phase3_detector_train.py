from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/yolo11.yaml"
DATA = ROOT / "data/blood/txl-pbc/TXL-PBC/data.yaml"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    if MODEL.read_bytes().__len__() == 0:
        raise RuntimeError("empty canonical architecture")
    settings.update({"sync": False, "api_key": "", "openai_api_key": "", "clearml": False, "comet": False, "dvc": False, "mlflow": False, "raytune": False, "tensorboard": False, "wandb": False})
    model = YOLO(str(MODEL), task="detect")
    model.train(
        data=str(DATA), imgsz=640, epochs=100, patience=20, batch=a.batch,
        workers=4, device="cpu", optimizer="AdamW", pretrained=False,
        seed=20260909, deterministic=True, iou=0.70, lr0=0.001, lrf=0.01,
        momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        warmup_momentum=0.8, warmup_bias_lr=0.1, project=str(a.project),
        name="yolo11n_from_scratch", exist_ok=True, plots=False, verbose=True,
        val=True, save=True, save_period=-1,
    )


if __name__ == "__main__":
    main()
