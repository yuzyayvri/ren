"""v1 specimen ingestion: validate, normalize, register blood-smear stills.

Production intake for previously unseen specimens. Enforces the exact
input contract the production vision path expects (uint8 RGB) and keeps
uploaded specimens strictly separate from frozen development datasets.
Fail closed on anything malformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts" / "v1_specimens"

MAX_DIM = 4096
MAX_PIXELS = 32_000_000
MIN_DIM = 64


class IngestError(ValueError):
    """Rejected specimen. Nothing is registered."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_file(source: Path, *, registry: Path | None = None) -> dict[str, Any]:
    if registry is None:
        registry = REGISTRY
    raw = source.read_bytes() if source.is_file() else b""
    is_png = raw.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = raw.startswith(b"\xff\xd8\xff")
    if (not is_png and not is_jpeg) or len(raw) < 100:
        raise IngestError(f"not a PNG or JPEG file: {source}")
    from PIL import Image, ImageOps

    try:
        image = Image.open(source)
        image.load()
    except (OSError, ValueError) as exc:
        raise IngestError(f"corrupt PNG: {exc}") from exc
    if image.format not in ("PNG", "JPEG", "MPO"):
        raise IngestError(f"not a PNG or JPEG file: {source}")
    w, h = image.size
    if w < MIN_DIM or h < MIN_DIM:
        raise IngestError(f"image too small: {w}x{h}")
    if w > MAX_DIM or h > MAX_DIM or w * h > MAX_PIXELS:
        raise IngestError(f"image too large: {w}x{h}")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if rgb.size != (w, h):
        w, h = rgb.size
    digest = sha256_bytes(raw)
    specimen_id = digest[:12]
    record_dir = registry / specimen_id
    if record_dir.is_dir():
        return json.loads((record_dir / "specimen.json").read_text(encoding="utf-8"))
    record_dir.mkdir(parents=True, exist_ok=False)
    rgb.save(record_dir / "normalized.png")
    thumb = rgb.copy()
    thumb.thumbnail((256, 256))
    thumb.save(record_dir / "thumb.png")
    record = {
        "schema": "v1-specimen-v1",
        "specimen_id": specimen_id,
        "source_filename": source.name,
        "content_sha256": digest,
        "content_bytes": len(raw),
        "width": w,
        "height": h,
        "ingested_unix": time.time(),
    }
    (record_dir / "specimen.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    try:
        record = ingest_file(args.image)
    except IngestError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}))
        return 1
    print(json.dumps({"status": "accepted", **record}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
