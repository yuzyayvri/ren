#!/usr/bin/env python3
"""Pinned, resumable acquisition for selected MedCPT/MedGemma artifacts.

The legacy ``future`` filename and command names are retained for workflow
compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "medcpt-query": ("ncbi/MedCPT-Query-Encoder", ROOT / "models/embed/medcpt/query", [
        "*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md", "LICENSE*"
    ]),
    "medcpt-article": ("ncbi/MedCPT-Article-Encoder", ROOT / "models/embed/medcpt/article", [
        "*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md", "LICENSE*"
    ]),
    "medgemma": ("unsloth/medgemma-1.5-4b-it-GGUF", ROOT / "models/llm", ["medgemma-1.5-4b-it-Q5_K_M.gguf"]),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def acquire(name: str, api: HfApi, update: bool) -> dict:
    repo, dest, patterns = SPECS[name]
    info = api.repo_info(repo, repo_type="model")
    revision = info.sha
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / ".acquisition-manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old and old.get("revision") != revision and not update:
        revision = old["revision"]
    if name == "medgemma":
        files = ["medgemma-1.5-4b-it-Q5_K_M.gguf"]
        target = dest / files[0]
        if target.exists() and target.stat().st_size == 0: target.unlink()
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(dest), allow_patterns=patterns,
                          max_workers=4)
    else:
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(dest), allow_patterns=patterns,
                          ignore_patterns=["*.bin", "pytorch_model.bin", "*.msgpack", "*.h5"],
                          max_workers=4)
        files = sorted(p.name for p in dest.iterdir() if p.is_file() and not p.name.startswith("."))
        if not any(p.endswith(".safetensors") for p in files): raise RuntimeError(f"{name}: no safetensors downloaded")
    records = []
    for rel in files:
        p = dest / rel
        if not p.is_file() or p.stat().st_size == 0: raise RuntimeError(f"missing/truncated file: {p}")
        records.append({"filename": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest = {"schema": 1, "repository": repo, "revision": revision, "files": records}
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, manifest_path)
    return manifest

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--update", action="store_true")
    ap.add_argument("names", nargs="*", choices=[*SPECS, "all"], default=["all"])
    args = ap.parse_args(); names = list(SPECS) if "all" in args.names else args.names
    api = HfApi(); out = {}
    for name in names: out[name] = acquire(name, api, args.update)
    print(json.dumps(out, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
