# AGENTS.md — how this repo works

> **This is the "how the repo works" reference.** Roadmap, status, and phased
> task list live in `ROADMAP.md`. Don't merge the two.

## Entering the dev environment

Two `nix develop` shells, picked by what you're doing:

| Shell | Use for | Entry |
|---|---|---|
| `nix develop` (default) | Everything except GPU-accelerated CV feature extraction / head training | `direnv allow` once, then every new shell is already in it (the `.envrc` says `use flake`) |
| `nix develop .#rocm` | CV feature extraction / training with the RX 6600 if you want it GPU-bound instead of CPU-bound | `nix develop .#rocm` explicitly when you need it; leave the default shell for LLM/Vulkan work |

The `direnv` route is the normal one — `use flake` in `.envrc` means `direnv allow` once and subsequent `cd`s into the repo auto-enter the default devShell. The ROCm shell is opt-in on purpose because it sets `HSA_OVERRIDE_GFX_VERSION` (gfx1032 spoofed as gfx1030) and pulls in ROCm libraries that don't belong in the LLM/Vulkan path.

**First time on a fresh checkout:**
```bash
direnv allow            # reads .envrc → use flake
uv sync                 # populates .venv from pyproject.toml / uv.lock
just --list             # see available recipes
```

If `uv sync` fails to start the interpreter, the machine-wide `programs.nix-ld.enable = true;` in `configuration.nix` is missing — `nixos-rebuild switch` after adding it. The flake sets `NIX_LD`/`NIX_LD_LIBRARY_PATH` per shell, but it can't create the `nix-ld` shim itself. (See `SETUP.md`.)

The Python environment is **hedgehog** (pinned): `uv.lock` locks exact versions; `pyproject.toml` is the source of truth for dependencies. Add deps with `uv add <pkg>`, not by editing `pyproject.toml` by hand.

## Running things — the Justfile

`just --list` is the table of contents. The recipes that matter day-to-day:

```
just vulkan-check       # confirm the RX 6600 shows up via RADV (Vulkan path)
just rocm-check         # confirm ROCm sees gfx (ROCm path)
just serve-llm [model]  # llama-server on 127.0.0.1:8080, default model=OpenBioLLM-8B GGUF
```

Model and data recipes (`just models-llm`, `just models-vision`, `just models-embed`, `just data-blood`, `just data-tissue`) are one-time pull steps — the files are already on disk for this project, so you shouldn't need to re-run them unless something got cleared.

**llama-server is the synthesis-stage HTTP endpoint.** It's the only model served as a server; everything else (vision backbone, head training) runs as a script against a model file on disk. Don't design a stage to load llama-server alongside the vision model in the same process — the pipeline is sequential per sample: vision runs and releases VRAM, then llama-server loads separately for synthesis. The hardware doesn't need to support both resident at once.

## Architecture convention — modular / UNIX-style

The project is built as a sequence of small, single-purpose stages, not one
monolithic pipeline. The convention:

- **One concern per script.** A script does one thing: extract embeddings,
  train a head, query the LLM, ingest GO terms, etc. If a script is doing two
  conceptually different things, split it.
- **Plain interfaces between stages.**
  - Vision ↔ retrieval: files and arrays on disk (`.npy`, saved embeddings).
  - LLM stage: HTTP to `llama-server` (`/v1/chat/completions`), nothing else.
  - Retrieval: vector index on disk (LanceDB/Chroma) + SQLite/NetworkX for symbolic GO traversal.
- **No shared mutable state between stages.** Each stage owns its artifacts;
  the next stage reads them. Don't introduce a central state object that every
  stage mutates.
- **Offline-first, and that includes the UI.** Phase 6's dashboard must work
  with no network in the request path — that's an acceptance criterion, not a
  nice-to-have.

## Settled decisions — don't re-litigate

These are in `ROADMAP.md`'s "Settled" section and are carried here for
convenience so a session doesn't re-derive them:

- **Inference backend**: `llama.cpp` built with Vulkan (`vulkanSupport = true`
  in the flake). Never ROCm for the LLM stage. ROCm exists only in the `rocm`
  devShell, only for CV feature extraction with PyTorch.
- **Vision backbone**: frozen `google/path-foundation`; train only a small head
  on top—never fine-tune the encoder. Phase 1 compared it with Virchow and
  selected Path Foundation: 94.78% versus 97.06% accuracy, but roughly 40×
  faster extraction, a roughly 22× smaller checkpoint, and 3.3× smaller
  embeddings. Virchow artifacts are retained only as an audit/reference
  comparison; don't wire both into the final pipeline.
- **Phase 2 exception**: a separate trainable HoVer-Net-fast-style network is
  permitted for PanNuke instance segmentation. This is not permission to
  fine-tune or replace Path Foundation, which remains frozen.
- **Phase 4 retrieval must go through the bound loader**:
  `scripts/phase4_snapshot_reconciliation.py::load_bound_query`. The plain
  default path raises by design — the v3 freeze records predate the two
  authorized query-layer changes (issue #2, reconciled without sealed
  mutation). Never "fix" this by editing sealed v3/v4 artifacts or code.
- **Cognitive synthesis LLMs**: `OpenBioLLM-Llama3-8B` and `MedGemma 1.5 4B
  Q5_K_M`, both GGUF and downloaded. Both are **candidates** to compare; pick a
  winner once you have comparison numbers and drop the other. Don't ship both.
  BioMistral-7B is excluded from Phase 5 and the production/default serving path.
- **Avoid MahmoodLab models** (UNI2-h, CONCH) — they hard-block personal email
  domains at the access-request stage.
- **No monetization** — learning project. Keep attribution to underlying
  foundation models/datasets intact.
- **Diagnostic-assist only** — every output is reviewed and signed off by a
  human; nothing makes an autonomous call.

## Data layout on disk

```
data/
  tissue/                 # PanNuke
    fold_1.zip / fold_2.zip / fold_3.zip   # originals
    fold1/Fold 1/ ...     # unzipped: images/, masks/, types/
    fold2/Fold 2/ ...
    fold3/Fold 3/ ...
  blood/
    txl-pbc/             # TXL-PBC (git clone, YOLO-labeled blood cells)
    aml/                 # Munich AML (TCIA/NBIA, manual download — see SETUP.md)
models/
  vision/
    virchow/            # Phase 1 comparison reference; not the selected pipeline
    path-foundation/    # selected pipeline backbone
  llm/
    openbiollm-llama3-8b.Q5_K_M.gguf
    medgemma-1.5-4b-it-Q5_K_M.gguf
  embed/
    pubmedbert/         # BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
```

PanNuke's per-image tissue-type labels are in `types.npy` (19-way), per the
dataset's layout. Phase 1 uses folds 1+2 to train, fold 3 to test.

## What's already confirmed working

- Flake / devShells / `direnv` / `uv`-managed Python env: working.
- `llama-server` on Vulkan: confirmed end-to-end (verified via a
  `/v1/chat/completions` round-trip with real token/sec numbers).
- Data on disk: PanNuke (fold1/2/3, each with `images.npy`/`masks.npy`/`types.npy`),
  TXL-PBC (cloned), Munich AML (downloaded via Kaggle mirror).
- Models on disk: OpenBioLLM-8B and MedGemma 1.5 4B Q5_K_M (GGUF), Virchow,
  path-foundation, PubMedBERT.

## Verify model download status before starting a phase

Virchow and Path Foundation are confirmed loadable. PubMedBERT is present but
has not yet been load-tested. When starting any phase that depends on a model,
confirm its files are actually there and loadable before writing code against
it—a missing checkpoint is the kind of failure that looks like a code bug.

For PanNuke instance work, also run the source-mask validator before generating
any target artifact. Its report is `artifacts/pannuke_mask_audit.json`; it is
intentionally fail-closed on schema changes and reports source overlaps for an
explicit downstream policy. The full scans are sequential and I/O-heavy—do not
run them alongside another Python vision job.

## Reading the roadmap

`ROADMAP.md` is the plan. Phases are sequential but some data axes (tissue vs.
blood) can progress in parallel once the vision scaffold in Phase 1 is proven.
Each phase has acceptance criteria — if they can't be met with the current
approach, that's a signal to report back and reconsider, not to lower the bar.
