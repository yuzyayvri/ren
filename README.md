# ren

ren is an offline-first learning project for computational cytometry and cell
pathology. It is a diagnostic-assist engine: outputs are reviewed and signed
off by a human, and the system makes no autonomous clinical calls.

## Architecture

The project is organized as small, sequential stages with plain interfaces:

1. **Vision engine** — frozen Google Path Foundation features are used for
   tissue representation and small task-specific heads. Path Foundation is a
   feature extractor: it emits one embedding per crop and performs no
   detection itself. PanNuke instance work also includes a separate
   HoVer-Net-fast-style segmentation network whose frozen outputs supply
   the visible nuclei, classes, and counts; the Path Foundation encoder
   remains frozen.
2. **Knowledge retrieval** — local files/arrays, LanceDB, SQLite, and
   NetworkX support vector and symbolic retrieval workflows.
3. **Cognitive synthesis** — a local `llama-server` endpoint serves GGUF
   candidates over the OpenAI-compatible `/v1/chat/completions` API. Vulkan is
   the intended llama.cpp backend; ROCm is reserved for optional PyTorch CV
   acceleration.
4. **Dashboard/UI** — a local specimen-first workstation served loopback-only
   with no build step or remote dependencies: specimen imagery with sealed
   overlays, bound-loader retrieval, frozen-path synthesis, and explicit
   human review with sign-off. True pyramidal WSI is out of reach of the
   current artifacts and is not claimed.

The code follows a modular, UNIX-style design: scripts own individual stages,
artifacts are passed on disk, and the LLM stage communicates over HTTP.

## Current status

Phases 1–6 are complete: vision scaffold with Path Foundation selected,
one-shot PanNuke fold-3 evaluation accepted, TXL-PBC and image-level Munich
AML blood stages complete, frozen Phase 4 v3 retrieval with latency recovery
accepted, MedGemma selected through blinded comparison and accepted on the
untouched final set, and the local workstation serving specimens, sealed
overlays, retrieval, synthesis, and review. Pending direction is a v1.0.0
refinement (arbitrary-specimen ingestion, automatic finding-to-retrieval
wiring); see ROADMAP.md for the plan when it lands.

Detailed decisions, acceptance criteria, and operational notes are in
[`ROADMAP.md`](ROADMAP.md), [`AGENTS.md`](AGENTS.md), [`MEMORY.md`](MEMORY.md),
and [`SETUP.md`](SETUP.md).

## Setup

This project uses Nix, direnv, and uv. On a fresh checkout, review
[`SETUP.md`](SETUP.md), allow direnv, and run `uv sync`. Use `just --list` to
see the available checks, model/data acquisition recipes, extraction and
verification entry points, and the local LLM serving command. Large datasets,
model weights, embeddings, experiment artifacts, caches, and local environments
are intentionally excluded from version control; the setup recipes document
how to obtain them.

## Repository layout

- `scripts/` — extraction, training, evaluation, audit, retrieval, synthesis,
  and workstation-serving stage entry points
- `dashboard/` — dependency-free static workstation frontend (no build step)
- `protocols/phase5_v1/`, `protocols/phase5_v2/` — frozen synthesis contracts
  (v1 superseded, preserved as history)
- `tests/` — focused regression and protocol tests
- `docs/` — project review documentation
- `flake.nix`, `flake.lock`, `pyproject.toml`, `uv.lock`, `Justfile` —
  reproducible development and execution configuration
- `ROADMAP.md`, `AGENTS.md`, `MEMORY.md`, `SETUP.md` — project documentation and
  operating context

## License

ren is licensed under the GNU General Public License version 3 (GPLv3). See
[`LICENSE`](LICENSE) for the complete license text.
