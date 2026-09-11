# ren

ren is an offline-first learning project for computational cytometry and cell
pathology. It is a diagnostic-assist engine: outputs are reviewed and signed
off by a human, and the system makes no autonomous clinical calls.

## Architecture

The project is organized as small, sequential stages with plain interfaces:

1. **Vision engine** — frozen Google Path Foundation features are used for
   tissue representation and small task-specific heads. PanNuke instance work
   also includes a separate HoVer-Net-fast-style segmentation network; the
   Path Foundation encoder remains frozen.
2. **Knowledge retrieval** — local files/arrays, LanceDB, SQLite, and
   NetworkX support vector and symbolic retrieval workflows.
3. **Cognitive synthesis** — a local `llama-server` endpoint serves GGUF
   candidates over the OpenAI-compatible `/v1/chat/completions` API. Vulkan is
   the intended llama.cpp backend; ROCm is reserved for optional PyTorch CV
   acceleration.
4. **Dashboard/UI** — planned as an offline request-path stage; the current
   repository primarily contains the experiment and protocol tooling.

The code follows a modular, UNIX-style design: scripts own individual stages,
artifacts are passed on disk, and the LLM stage communicates over HTTP.

## Current status

The Nix flake, direnv workflow, uv environment, local models/data, and Vulkan
llama-server path have been exercised on the development machine. Path
Foundation and Virchow embedding extraction and verification are implemented.
PanNuke Phase 2 remains in development: the current fold-2 candidates pass
the geometry gates but the end-to-end macro-F1 acceptance gate is not yet met.
TXL-PBC and Munich AML protocol/audit tooling is present, and Phase 4 local
retrieval protocol tooling is included. Fold 3 remains locked for the pending
Phase 2 decision.

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

- `scripts/` — extraction, training, evaluation, audit, retrieval, and serving
  stage entry points
- `tests/` — focused regression and protocol tests
- `docs/` — project review documentation
- `flake.nix`, `flake.lock`, `pyproject.toml`, `uv.lock`, `Justfile` —
  reproducible development and execution configuration
- `ROADMAP.md`, `AGENTS.md`, `MEMORY.md`, `SETUP.md` — project documentation and
  operating context

## License

ren is licensed under the GNU General Public License version 3 (GPLv3). See
[`LICENSE`](LICENSE) for the complete license text.
