# ren — first-time setup

## 0. One-time system change

`uv`-managed Python (and the wheels it pulls — torch, opencv, etc.) are
linked against a generic glibc that doesn't exist on NixOS. Add to your
system `configuration.nix` once, machine-wide:

```nix
programs.nix-ld.enable = true;
```

`nixos-rebuild switch` after adding it. The flake sets `NIX_LD`/
`NIX_LD_LIBRARY_PATH` per-shell, but it can't create the `nix-ld` shim
itself — without this, `uv sync` will install fine and then every `uv run`
will fail to start the interpreter.

## 1. Enter the shell

```
direnv allow          # first time only, reads .envrc
just vulkan-check     # confirm the RX 6600 shows up via RADV
uv sync               # populates .venv from pyproject.toml
```

Two shells exist: `nix develop` (default — llama.cpp/Vulkan + the whole
Python stack) and `nix develop .#rocm` (adds ROCm, only needed when you're
running the CV feature-extraction/training step and want it GPU-accelerated
instead of CPU-bound).

## 2. Auth for gated weights

Virchow and Google's `path-foundation` are both gated on Hugging Face —
request access on each model page first, then:

```
hf auth login
```

before `just models-vision` will pull anything. Virchow and path-foundation
both approve fast (Virchow works fine with a personal email; avoid
MahmoodLab's models like UNI2-h/CONCH for this pipeline unless you have an
institutional email on your HF account — they hard-block @gmail/@hotmail/@qq
outright). OpenBioLLM's GGUF mirror is ungated; the MedGemma GGUF is acquired
by the pinned recipe described below.

## 3. Pull everything

```
just models-llm      # OpenBioLLM-8B + MedGemma 1.5 4B Q5_K_M, GGUF, for synthesis
just models-vision    # Virchow + path-foundation, frozen ViT backbones for the vision stage
just models-embed     # PubMedBERT, for the knowledge-retrieval embeddings
just models-future-acquire  # pinned MedCPT + MedGemma acquisition
just models-future-verify    # offline MedCPT + ephemeral llama.cpp checks
just data-blood       # clones TXL-PBC (PBC + Raabin-WBC, YOLO-labeled, no account needed)
just data-tissue      # PanNuke — the one dataset here with no auth wall
```

### Munich AML — the one genuinely manual dataset

TCIA doesn't offer a plain URL for this one, but the actual steps are short:

1. Go to the `AML-Cytomorphology_LMU` collection page on TCIA and add it to
   your cart, then **Download > Download Cart**. This gives you a
   `manifest-XXX.tcia` file — a list of what to fetch, not the images
   themselves.
2. Grab the **Linux** build of the NBIA Data Retriever from TCIA's download
   page (a `.deb`/tarball). It's a generic Linux binary, not built for
   NixOS's non-FHS layout — but you already have `nix-ld` wired up from
   setting up `uv` earlier, so it should just run once extracted, no extra
   patching needed.
3. It has a headless CLI mode, useful here since there's no GUI toolkit in
   this devShell anyway:
   ```
   NBIADataRetriever --cli manifest-XXX.tcia -d data/blood/aml
   ```
   Check `NBIADataRetriever --help` if the flags have shifted — this tool
   isn't updated often, but it does happen.

## 4. Sanity-check the LLM path end to end

```
just serve-llm
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi"}]}'
```

If that round-trips, the Vulkan build is working and every later stage that
talks to `llama-server` over HTTP has something to talk to.

## Why two backends at all

Vulkan handles every GGUF/llama.cpp model in the pipeline — no ROCm, no gfx
override, nothing card-specific to fight with. ROCm only re-enters the
picture because PyTorch doesn't have a real Vulkan backend, and the vision
engine needs PyTorch to run Virchow/UNI and train the classification head
on top. Given that head is small (frozen encoder, you're only training a
thin layer), CPU is a legitimate fallback if the gfx1032 override gives you
grief — worth trying before sinking time into it.

## Pinned retrieval/synthesis model acquisition

The model-acquisition recipe resolves immutable Hugging Face commits, downloads
only MedCPT safetensors plus offline metadata/tokenizers, and the MedGemma
Q5_K_M GGUF. It is idempotent and resumable; rerun after interruption. Use
--update on scripts/acquire_future_models.py only when intentionally changing
revisions. The verification recipe disables network for MedCPT, runs its paired
retrieval smoke test, then starts and stops a local llama.cpp process at context
8192 for one text-only generation. MedCPT remains a future retrieval candidate;
MedGemma is the Phase 5 synthesis candidate paired with OpenBioLLM pending the
comparison, so neither model is a selected production winner yet.
