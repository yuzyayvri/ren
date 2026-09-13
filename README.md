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
3. **Cognitive synthesis** — a local `llama-server` endpoint serves the frozen
   MedGemma GGUF over the OpenAI-compatible `/v1/chat/completions` API;
   OpenBioLLM is retained as audit evidence. Vulkan is the intended llama.cpp
   backend; ROCm is reserved for optional PyTorch CV acceleration.
4. **Dashboard/UI** — a local specimen-first workstation served loopback-only
   with no build step or remote dependencies: specimen imagery with sealed
   overlays, bound-loader retrieval, frozen-path synthesis, and explicit
   human review with sign-off. True pyramidal WSI is out of reach of the
   current artifacts and is not claimed.

The code follows a modular, UNIX-style design: scripts own individual stages,
artifacts are passed on disk, and the LLM stage communicates over HTTP.

## v1.0.0 product contract

Ren v1 supports one modality: blood-smear stills (PNG/JPEG). Import validates
and registers the specimen; production vision (bound YOLO detector, frozen
Path Foundation embeddings, frozen logistic head) yields spatial WBC/RBC/
Platelet findings; reviewer confirmation automatically drives deterministic
retrieval (manual retrieval is also available); frozen MedGemma synthesis
drafts the note; review signs it. Everything runs locally with hash-pinned
provenance. No WSI, no tissue inference, no clinical claims. Outputs are
diagnostic-assist drafts requiring human sign-off.

## Current status

Ren v1.0.0 is complete for the supported blood-smear workflow and independently
release-verified. On branch `yuzy/v1-bughunt` at commit
`68dd88a7ef2d7262945cbf51e09df861b2012221`, the fresh unfiltered Phase 6 browser
stability run (`63ccc38b23c2`) scored 100.0/100 across 69 scenarios and 167
checks. It had no crashes, skips, console errors, unexpected failed requests,
or HTTP 5xx responses; the one expected failed request was the intentional S03
dead-backend probe, and teardown was asserted.

The product path accepts arbitrary blood-smear PNG/JPEG stills, runs the bound
YOLO detector plus frozen Path Foundation/logistic head, supports reviewer
confirmation/correction/rejection with automatic deterministic retrieval,
produces frozen MedGemma synthesis, and records sign-off/export provenance.
True pyramidal WSI, tissue production inference, and clinical claims remain
out of scope; every output is a diagnostic-assist draft requiring human
sign-off. PR #16 remains open; no merge has been performed. The authoritative
run and independent verifier are retained in the populated checkout under
`artifacts/benchmark_results/stability/`. The certification binds to the
implementation commit above; subsequent documentation-only commits are not
part of that browser run.

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
- `benchmarks/stability/` — 69-scenario browser stability rubric, runner, and
  scenario definitions for the v1 workstation
- `protocols/phase5_v3/` — frozen production synthesis prompt and decoding
  contract (v1/v2 are retained as protocol history)
- `tests/` — focused regression and protocol tests
- `docs/` — project review documentation
- `flake.nix`, `flake.lock`, `pyproject.toml`, `uv.lock`, `Justfile` —
  reproducible development and execution configuration
- `ROADMAP.md`, `AGENTS.md`, `MEMORY.md`, `SETUP.md` — project documentation and
  operating context

## License

ren is licensed under the GNU General Public License version 3 (GPLv3). See
[`LICENSE`](LICENSE) for the complete license text.
