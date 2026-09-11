#!/usr/bin/env python3
"""
Phase 1 — Vision-engine embedding extraction (smoke test).

Extracts frozen-encoder embeddings for all three PanNuke folds, for both
candidate backbones (Virchow, path-foundation). The embeddings are saved to
disk as .npy arrays; the next stage (train_head.py) loads them and fits a
linear classifier on top.

Design notes:
- Frozen encoder only — never fine-tune the ViT itself. We train a small
  classification head separately.
- Batch inference on CPU (or GPU if ROCm shell is active and torch sees it).
- Per-fold output: {backbone}/{fold}/embeddings.npy + labels.npy, so the
  head-training stage can load one fold at a time.
- Virchow is a ViT-Huge (patch14, 224x224, 630M params). path-foundation is
  a smaller ViT — we load it via timm if a valid HF vision model name is
  available, otherwise fall back to a registered timm name.
- Both backbones need 224x224 RGB input with ImageNet normalization (Virchow's
  config confirms mean/std = ImageNet defaults). PanNuke images are 256x256,
  so we center-crop to 224.

Usage:
    python scripts/extract_embeddings.py --backbone virchow --fold fold1
    python scripts/extract_embeddings.py --backbone virchow --fold fold2
    python scripts/extract_embeddings.py --backbone virchow --fold fold3
    python scripts/extract_embeddings.py --backbone path-foundation --fold fold1
    ... etc.

Or run all folds for one backbone in one call:
    python scripts/extract_embeddings.py --backbone virchow --all-folds

Acceptance (Phase 1 smoke test):
    - Script completes without manual intervention for each fold/backbone.
    - Output embeddings have shape (N_images, D) where D is the encoder's
      feature dimension (Virchow: 1280 for ViT-Huge; path-foundation: check
      on load).
    - labels.npy matches types.npy (19 tissue classes, same order as images).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "tissue"


# ---------------------------------------------------------------------------
# Model loading — Virchow
# ---------------------------------------------------------------------------
#
# Virchow is a ViT-Huge (patch14, 224x224, embed_dim=1280, depth=32,
# num_heads=16) with a gated MLP (SwiGlu-style): fc1 expands 1280→6832 and
# splits into gate+proj (each 3416), SiLU(gate)*proj, then fc2 contracts
# 3416→1280. Layer-scale (init_values=1e-5) is applied to both attn and MLP
# residuals.
#
# Critically, the checkpoint keys use timm-style naming:
#   patch_embed.proj.weight, blocks.N.attn.qkv.weight,
#   blocks.N.attn.proj.weight, blocks.N.mlp.fc1.weight, blocks.N.mlp.fc2.weight,
#   blocks.N.ls1.gamma, blocks.N.ls2.gamma
#
# The gated MLP means we CANNOT use timm's default VisionTransformer (which has
# a standard 2-layer MLP with fc2: hidden→embed_dim). We build the model
# manually with matching key names so load_state_dict works directly.
#
# This is the pragmatic choice for Phase 1: a correct frozen-encoder extraction
# beats spending time on timm integration. If someone wants to upstream a
# proper timm integration later, the block_fn override path is the one to take.

VIRCHOW_EMBED_DIM = 1280
VIRCHOW_DEPTH = 32
VIRCHOW_NUM_HEADS = 16
VIRCHOW_PATCH_SIZE = 14
VIRCHOW_IMG_SIZE = 224
VIRCHOW_MLP_RATIO = 5.3375  # mlp_hidden = 1280 * 5.3375 = 6832
VIRCHOW_INIT_VALUES = 1e-5


class _VirchowGatedMlp(nn.Module):
    """Gated MLP matching Virchow's architecture: fc1(1280→6832) split into
    gate+proj (each 3416), SiLU(gate)*proj, fc2(3416→1280).

    Checkpoint keys: blocks.N.mlp.fc1.{weight,bias}, blocks.N.mlp.fc2.{weight,bias}
    """

    def __init__(self, embed_dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)  # 6832
        proj = hidden // 2  # 3416
        self.fc1 = nn.Linear(embed_dim, hidden)  # 1280 → 6832
        self.fc2 = nn.Linear(proj, embed_dim)  # 3416 → 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)  # [..., 6832]
        gate, proj = h.chunk(2, dim=-1)  # each [..., 3416]
        return self.fc2(F.silu(gate) * proj)


class _VirchowAttention(nn.Module):
    """Self-attention with separate Q/K/V + output projection, matching
    Virchow's checkpoint key names:
      blocks.N.attn.qkv.{weight,bias}  (fused QKV, shape [3*D, D])
      blocks.N.attn.proj.{weight,bias} (shape [D, D])
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 1280/16 = 80
        # Virchow stores QKV as a single fused matrix [3*D, D]
        self.qkv_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.qkv_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.proj_bias = nn.Parameter(torch.empty(embed_dim))
        # Initialize with sensible defaults; will be overwritten by load_state_dict
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.constant_(self.qkv_bias, 0.0)
        nn.init.xavier_uniform_(self.proj_weight)
        nn.init.constant_(self.proj_bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, num_heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return F.linear(x, self.proj_weight, self.proj_bias)


class _VirchowBlock(nn.Module):
    """Transformer block with pre-norm, layer-scale (ls1/ls2.gamma),
    gated MLP — matching Virchow's checkpoint key names exactly.

    Keys: blocks.N.{norm1,attn,ls1.gamma,norm2,mlp,ls2.gamma}
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        init_values: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = _VirchowAttention(embed_dim, num_heads)
        # Virchow names these 'ls1.gamma' and 'ls2.gamma' — use nn.Parameter
        # with that exact name via a small wrapper.
        self.ls1_gamma = nn.Parameter(torch.full((embed_dim,), init_values))
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = _VirchowGatedMlp(embed_dim, mlp_ratio)
        self.ls2_gamma = nn.Parameter(torch.full((embed_dim,), init_values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1_gamma * self.attn(self.norm1(x))
        x = x + self.ls2_gamma * self.mlp(self.norm2(x))
        return x


class VirchowModel(nn.Module):
    """Frozen Virchow ViT-Huge feature extractor.

    Architecture (from config.json):
      - patch14, 224x224, embed_dim=1280, depth=32, num_heads=16
      - gated MLP (SwiGlu), mlp_ratio=5.3375
      - layer-scale with init_values=1e-5
      - num_classes=0 (no classification head; features only)
      - ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

    Checkpoint key naming matches Virchow's HF weights exactly:
      patch_embed.proj.{weight,bias}
      cls_token, pos_embed
      blocks.N.{norm1,attn.qkv,attn.proj,ls1.gamma,norm2,mlp.fc1,mlp.fc2,ls2.gamma}
      norm

    Forward returns [B, N+1, embed_dim] token sequence; caller pools to [B, embed_dim].
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, VIRCHOW_EMBED_DIM, kernel_size=VIRCHOW_PATCH_SIZE, stride=VIRCHOW_PATCH_SIZE
        )
        num_patches = (VIRCHOW_IMG_SIZE // VIRCHOW_PATCH_SIZE) ** 2  # 256
        self.cls_token = nn.Parameter(torch.zeros(1, 1, VIRCHOW_EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, VIRCHOW_EMBED_DIM))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            _VirchowBlock(
                VIRCHOW_EMBED_DIM,
                VIRCHOW_NUM_HEADS,
                VIRCHOW_MLP_RATIO,
                VIRCHOW_INIT_VALUES,
            )
            for _ in range(VIRCHOW_DEPTH)
        )
        self.norm = nn.LayerNorm(VIRCHOW_EMBED_DIM, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, C, H, W] → [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)  # [B, N+1, D]


def load_virchow(model_dir: Path) -> VirchowModel:
    """Load Virchow from local HF download dir, return frozen feature extractor."""
    safetensors_path = model_dir / "model.safetensors"
    pt_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path)
    elif pt_path.exists():
        state_dict = torch.load(pt_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Neither model.safetensors nor pytorch_model.bin in {model_dir}"
        )

    model = VirchowModel()

    # Map Virchow checkpoint keys to our model's keys where naming differs.
    # Virchow uses:
    #   patch_embed.proj.{weight,bias}          → our patch_embed.{weight,bias} (Conv2d)
    #   blocks.N.attn.qkv.{weight,bias}         → our attn.qkv_weight/bias (fused QKV linear)
    #   blocks.N.attn.proj.{weight,bias}        → our attn.proj_weight/bias (output proj)
    #   blocks.N.ls1.gamma / ls2.gamma         → our ls1_gamma / ls2_gamma
    def _remap_key(k: str) -> str:
        k = k.replace("patch_embed.proj.weight", "patch_embed.weight")
        k = k.replace("patch_embed.proj.bias", "patch_embed.bias")
        k = k.replace(".attn.qkv.weight", ".attn.qkv_weight")
        k = k.replace(".attn.qkv.bias", ".attn.qkv_bias")
        k = k.replace(".attn.proj.weight", ".attn.proj_weight")
        k = k.replace(".attn.proj.bias", ".attn.proj_bias")
        k = k.replace(".ls1.gamma", ".ls1_gamma")
        k = k.replace(".ls2.gamma", ".ls2_gamma")
        return k

    remapped = {_remap_key(k): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(remapped)
    except RuntimeError:
        # Some HF refs prefix all keys with "model." — strip and retry once
        if all(k.startswith("model.") for k in state_dict):
            stripped = {k[len("model."):]: v for k, v in state_dict.items()}
            remapped2 = {_remap_key(k): v for k, v in stripped.items()}
            model.load_state_dict(remapped2)
        else:
            raise

    model.eval()
    return model


def _try_load_path_foundation_pt(model_dir: Path):
    """
    Attempt to load path-foundation as a PyTorch model.
    google/path-foundation on HF may expose a pytorch_model.bin or be a TF-only
    SavedModel. We detect which and load accordingly.
    """
    import timm

    has_pt = (model_dir / "pytorch_model.bin").exists() or (
        model_dir / "model.safetensors"
    ).exists()
    has_tf = (model_dir / "saved_model.pb").exists()

    if has_pt:
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            arch = cfg.get("architecture", "")
            if arch and arch in timm.list_models():
                model = timm.create_model(arch, pretrained=False, num_classes=0)
                safetensors_path = model_dir / "model.safetensors"
                pt_path = model_dir / "pytorch_model.bin"
                if safetensors_path.exists():
                    from safetensors.torch import load_file

                    sd = load_file(safetensors_path)
                else:
                    sd = torch.load(pt_path, map_location="cpu")
                try:
                    model.load_state_dict(sd)
                except RuntimeError:
                    stripped = {k.replace("model.", "", 1): v for k, v in sd.items()}
                    model.load_state_dict(stripped)
                model.eval()
                model = model.cuda() if torch.cuda.is_available() else model
                return model, "pytorch"
        raise RuntimeError(
            f"path-foundation has PT weights but no recognized config.architecture. "
            f"Files: {[p.name for p in model_dir.iterdir() if p.is_file()]}"
        )
    elif has_tf:
        import importlib.util

        if importlib.util.find_spec("tensorflow") is None:
            raise RuntimeError(
                "path-foundation is TF-only (SavedModel) and tensorflow is not "
                "installed. Either install tensorflow, or skip path-foundation in "
                "this Phase 1 run."
            )
        return None, "tensorflow"
    else:
        raise FileNotFoundError(
            f"path-foundation directory {model_dir} has neither PT weights nor "
            f"saved_model.pb. Contents: {[p.name for p in model_dir.iterdir()]}"
        )


def load_path_foundation(model_dir: Path):
    """Load path-foundation frozen feature extractor.

    Returns (model_or_None, backend) where backend is 'pytorch' or 'tensorflow'.
    For TF backend, model_or_None is None and extraction happens via
    path_foundation_embed_tf().
    """
    return _try_load_path_foundation_pt(model_dir)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# Virchow config: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], img_size=224
# PanNuke images are 256x256 RGB float in [0,1] (NHWC).
# We resize to 224 (bicubic), center-crop, and normalize to ImageNet stats.

VIRCHOW_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
VIRCHOW_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _virchow_transform_batch(batch: np.ndarray) -> torch.Tensor:
    """Transform a batch of (B, 256, 256, 3) float [0,1] images to
    (B, 3, 224, 224) normalized tensors ready for Virchow.

    Steps: HWC→CHW, resize to 224 (bicubic), center-crop 224, normalize.
    All operations are vectorized over the batch, on CPU.
    """
    # NHWC → NCHW
    x = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float()  # [B, 3, 256, 256]
    # Resize to 224 (bicubic, same as Virchow config interpolation=bicubic)
    x = F.interpolate(
        x, size=224, mode="bicubic", align_corners=False, antialias=True
    )  # [B, 3, 224, 224]
    # Center crop is a no-op since we resized to exactly 224; Virchow's
    # crop_pct=1.0 means no pre-scale crop, and img_size=224 means the model
    # expects 224x224 input. With PanNuke at 256 and resize to 224, we're
    # effectively center-cropping by resizing down.
    # Normalize to ImageNet stats (constants are CPU tensors)
    x = (x - VIRCHOW_MEAN) / VIRCHOW_STD
    return x


# ---------------------------------------------------------------------------
# Virchow forward
# ---------------------------------------------------------------------------

def virchow_embed(batch: torch.Tensor, model: VirchowModel) -> np.ndarray:
    """Forward a batch of (B,3,224,224) tensors through Virchow, return (B,D).

    Uses CLS-token pooling (standard for ViT classification features).
    """
    with torch.no_grad():
        tokens = model(batch)  # [B, N+1, D]
        feats = tokens[:, 0, :]  # CLS token → [B, D]
        return feats.numpy()


# ---------------------------------------------------------------------------
# path-foundation TF forward
# ---------------------------------------------------------------------------

def path_foundation_embed_tf(image_paths_or_arrays, model_dir: Path, batch_size: int = 32):
    """
    Extract features from path-foundation TF SavedModel.
    Yields (B, D) numpy arrays.
    """
    import tensorflow as tf

    # Load the SavedModel
    loaded = tf.saved_model.load(str(model_dir))
    # Find the serving signature
    inference = loaded.signatures.get("serving_default", None)
    if inference is None:
        # Try the first available signature
        sig_keys = list(loaded.signatures.keys())
        if not sig_keys:
            raise RuntimeError(
                "path-foundation SavedModel has no signatures. Cannot extract features."
            )
        inference = loaded.signatures[sig_keys[0]]

    input_name = list(inference.structured_input_signature[1][0].name)
    # inference.structured_input_signature is ((name,), dtype) for each input
    # Simpler: inspect the concrete function
    # Actually: inference.structured_input_signature[1] is a dict-like of TensorSpecs
    specs = inference.structured_input_signature[1]
    if isinstance(specs, dict):
        input_name = next(iter(specs.keys()))
        input_spec = specs[input_name]
    else:
        # It's a tuple/list of TensorSpecs
        input_spec = specs[0]
        input_name = input_spec.name

    # Process in batches
    all_out = []
    for i in range(0, len(image_paths_or_arrays), batch_size):
        batch_arr = image_paths_or_arrays[i : i + batch_size]
        # Convert to TF tensor formatted as the model expects
        # path-foundation likely expects (B, H, W, 3) float [0,1] or [0,255]
        batch_tf = tf.convert_to_tensor(batch_arr, dtype=tf.float32)
        # Ensure shape (B, H, W, C)
        if batch_tf.ndim == 3:
            batch_tf = tf.expand_dims(batch_tf, 0)
        out = inference(**{input_name: batch_tf})
        # out is a dict of tensors; grab the first (feature) output
        feat = next(iter(out.values())).numpy()
        if feat.ndim == 2:
            all_out.append(feat)
        elif feat.ndim == 3:
            feat = feat.mean(axis=1)  # pool patch tokens
            all_out.append(feat)
        else:
            raise RuntimeError(f"Unexpected TF output shape: {feat.shape}")
    return np.concatenate(all_out, axis=0)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

FOLDS = {
    "fold1": {
        "images": DATA_DIR / "fold1" / "Fold 1" / "images" / "fold1" / "images.npy",
        "types": DATA_DIR / "fold1" / "Fold 1" / "images" / "fold1" / "types.npy",
    },
    "fold2": {
        "images": DATA_DIR / "fold2" / "Fold 2" / "images" / "fold2" / "images.npy",
        "types": DATA_DIR / "fold2" / "Fold 2" / "images" / "fold2" / "types.npy",
    },
    "fold3": {
        "images": DATA_DIR / "fold3" / "Fold 3" / "images" / "fold3" / "images.npy",
        "types": DATA_DIR / "fold3" / "Fold 3" / "images" / "fold3" / "types.npy",
    },
}

OUT_ROOT = PROJECT_ROOT / "embeddings"


def extract_fold_virchow(fold: str, model: nn.Module, batch_size: int = 32):
    """Extract Virchow embeddings for one fold; save to embeddings/virchow/{fold}/."""
    meta = FOLDS[fold]
    source_images = np.load(meta["images"], mmap_mode="r")
    images = np.asarray(source_images)
    types = np.load(meta["types"], allow_pickle=True)

    print(
        f"[virchow] {fold}: images={images.shape} dtype={images.dtype} "
        f"types={types.shape} unique={len(np.unique(types))}",
        file=sys.stderr,
    )

    # Normalize images to [0,1] float if needed
    if images.dtype == np.float64 or images.dtype == np.float32:
        if images.max() > 1.0:
            images = images.astype(np.float32) / 255.0
        elif images.min() < 0 or images.max() > 1.0:
            images = images.astype(np.float32)
        else:
            images = images.astype(np.float32)
    elif images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    else:
        images = images.astype(np.float32)

    # Convert NHWC → NCHW and apply transform
    # images is (N, 256, 256, 3) float [0,1]
    n = len(images)
    all_feats = np.empty((n, VIRCHOW_EMBED_DIM), dtype=np.float32)

    # Vectorized transform: HWC→CHW, resize to 224, center-crop, normalize.
    # We process in batches for memory efficiency; the resize/crop/normalize
    # is done on the full batch tensor (not per-image PIL loop).
    for start in tqdm(range(0, n, batch_size), desc=f"virchow {fold}", unit="batch"):
        end = min(start + batch_size, n)
        batch = images[start:end]  # (B, 256, 256, 3) float [0,1]
        batch_t = _virchow_transform_batch(batch)
        feats = virchow_embed(batch_t, model)
        all_feats[start:end] = feats

    out_dir = OUT_ROOT / "virchow" / fold
    out_dir.mkdir(parents=True, exist_ok=True)
    from embedding_artifacts import publish_batches, save_batch

    # Even the single-process path uses the same index/provenance validator as
    # resumable extraction. Canonical files are never published directly.
    batch_path = out_dir / f"embeddings_batch_0_{n}.npz"
    save_batch(batch_path, 0, all_feats, source_images)
    publish_batches(out_dir, meta["images"], meta["types"], VIRCHOW_EMBED_DIM)
    print(
        f"[virchow] {fold}: saved embeddings={all_feats.shape} labels={types.shape} "
        f"to {out_dir}",
        file=sys.stderr,
    )
    return all_feats, types


def extract_fold_path_foundation(fold: str, model_dir: Path, backend: str, batch_size: int = 32):
    """Extract path-foundation embeddings for one fold; save to embeddings/path-foundation/{fold}/."""
    meta = FOLDS[fold]
    images = np.load(meta["images"], allow_pickle=True)
    types = np.load(meta["types"], allow_pickle=True)

    print(
        f"[path-foundation] {fold}: images={images.shape} dtype={images.dtype} "
        f"types={types.shape} unique={len(np.unique(types))}",
        file=sys.stderr,
    )

    if backend == "tensorflow":
        feats = path_foundation_embed_tf(images, model_dir, batch_size=batch_size)
    else:
        raise RuntimeError(f"Unsupported path-foundation backend: {backend}")

    out_dir = OUT_ROOT / "path-foundation" / fold
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", feats)
    np.save(out_dir / "labels.npy", types)
    print(
        f"[path-foundation] {fold}: saved embeddings={feats.shape} labels={types.shape} "
        f"to {out_dir}",
        file=sys.stderr,
    )
    return feats, types


def main():
    parser = argparse.ArgumentParser(description="Phase 1 embedding extraction")
    parser.add_argument(
        "--backbone",
        choices=["virchow", "path-foundation"],
        required=True,
        help="Candidate vision backbone to extract embeddings for",
    )
    parser.add_argument(
        "--fold",
        choices=["fold1", "fold2", "fold3"],
        help="Single fold to process (optional; --all-folds overrides)",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Process all three folds in one call",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default 32)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "vision",
        help="Path to models/vision dir (default: repo/models/vision)",
    )
    args = parser.parse_args()

    if args.fold is None and not args.all_folds:
        parser.error("Specify --fold or --all-folds")

    folds_to_run = ["fold1", "fold2", "fold3"] if args.all_folds else [args.fold]

    if args.backbone == "virchow":
        model_dir = args.models_dir / "virchow"
        if not model_dir.exists():
            print(
                f"[error] Virchow model dir not found: {model_dir}. "
                f"Run 'just models-vision' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[info] Loading Virchow from {model_dir} ...", file=sys.stderr)
        model = load_virchow(model_dir)
        print(f"[info] Virchow loaded. CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
        for fold in folds_to_run:
            extract_fold_virchow(fold, model, batch_size=args.batch_size)

    elif args.backbone == "path-foundation":
        model_dir = args.models_dir / "path-foundation"
        if not model_dir.exists():
            print(
                f"[error] path-foundation model dir not found: {model_dir}. "
                f"Run 'just models-vision' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[info] Loading path-foundation from {model_dir} ...", file=sys.stderr)
        model, backend = load_path_foundation(model_dir)
        if backend == "tensorflow":
            print(
                "[info] path-foundation is TF SavedModel; extracting via tensorflow. "
                "This is slower than a PT backbone — acceptable for a one-time comparison run.",
                file=sys.stderr,
            )
        for fold in folds_to_run:
            extract_fold_path_foundation(fold, model_dir, backend, batch_size=args.batch_size)

    print("[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
