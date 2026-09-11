# ren dev pipeline — run `just --list` to see all recipes

default:
    just --list

# --- sanity checks -----------------------------------------------------

vulkan-check:
    vulkaninfo --summary | head -30

rocm-check:
    rocminfo | grep -i gfx

# --- models --------------------------------------------------------------

# Cognitive-synthesis LLMs, GGUF, for llama.cpp/Vulkan
models-llm:
    mkdir -p models/llm
    hf download aaditya/OpenBioLLM-Llama3-8B-GGUF --include "*Q5_K_M*" --local-dir models/llm
    hf download unsloth/medgemma-1.5-4b-it-GGUF --include "*Q5_K_M*" --local-dir models/llm

# Vision-engine backbones — Virchow is gated but approves fast; path-foundation
# is gated only by a click-through terms-of-use (no institutional-email wall,
# unlike MahmoodLab's UNI2-h/CONCH, which flat-out deny @gmail/@hotmail/@qq).
models-vision:
    mkdir -p models/vision
    hf download paige-ai/Virchow --local-dir models/vision/virchow
    hf download google/path-foundation --local-dir models/vision/path-foundation

models-embed:
    mkdir -p models/embed
    hf download microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --local-dir models/embed/pubmedbert

# Pinned MedCPT retrieval artifacts plus the MedGemma Phase 5 candidate; verified
# by the dedicated scripts.
models-future-acquire:
    /tmp/run_python.sh scripts/acquire_future_models.py all

models-future-verify:
    /tmp/run_python.sh scripts/verify_future_models.py --output artifacts/future_models_verification.json

# --- data ------------------------------------------------------------------

# No clean single-command pull for Munich AML — see SETUP.md for the manual
# TCIA/NBIA steps. TXL-PBC (integrates PBC + Raabin-WBC, YOLO-labeled) is a
# plain git clone, no account needed at all.
data-blood:
    mkdir -p data/blood
    git clone --depth 1 https://github.com/lugan113/TXL-PBC_Dataset data/blood/txl-pbc
    echo "Munich AML Morphology Dataset: manual TCIA/NBIA steps — see SETUP.md"

data-tissue:
    mkdir -p data/tissue
    echo "PanNuke: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke — direct download, no auth"

# --- serving -----------------------------------------------------------------

serve-llm model="models/llm/medgemma-1.5-4b-it-Q5_K_M.gguf":
    llama-server -m {{model}} --host 127.0.0.1 --port 8080 -ngl 999

# --- phase 5 ---------------------------------------------------------------

# Freeze the Part 1 synthesis protocol (prompt, schemas, decoding, model
# hashes, benchmark, scoring, winner rule, lifecycle). No LLM is run.
freeze-phase5:
    /tmp/run_python.sh scripts/phase5_protocol.py freeze

# Verify the frozen Part 1 protocol without running any LLM.
verify-phase5:
    /tmp/run_python.sh scripts/phase5_protocol.py verify

# Run the Part 2 candidate comparison (hours, unattended). Serves each
# candidate once, preserves every generation. Human review follows.
compare-phase5:
    /tmp/run_python.sh scripts/phase5_compare.py run --all

# Aggregate saved comparison runs without touching any server.
score-phase5:
    /tmp/run_python.sh scripts/phase5_compare.py score

# Freeze/verify the revised v2 synthesis protocol (no LLM is run).
freeze-phase5-v2:
    /tmp/run_python.sh scripts/phase5_protocol.py freeze-v2

verify-phase5-v2:
    /tmp/run_python.sh scripts/phase5_protocol.py verify-v2

# Synthetic smoke gate per candidate (mechanism check only, never scored).
smoke-phase5 candidate:
    /tmp/run_python.sh scripts/phase5_compare.py smoke --candidate {{candidate}} --protocol phase5_v2

# Production synthesis from a pinned packet (frozen winner, loopback only).
synthesize-phase5 packet out:
    /tmp/run_python.sh scripts/phase5_synthesize.py from-packet --packet {{packet}} --out {{out}}

# Guarded final acceptance (preconditions check only shows readiness).
final-preconditions:
    /tmp/run_python.sh scripts/phase5_final.py preconditions

# --- phase 6 ---------------------------------------------------------------

# Local dashboard (loopback-only, offline). Run `just serve-llm` first
# in another shell if note generation is needed.
serve-dashboard port="8081":
    /tmp/run_python.sh scripts/phase6_server.py --port {{port}}

# --- v1 ----------------------------------------------------------------------

# v1 blood-smear product path. Vision steps need scripts/vision_python.sh
# (GL/X11 libs for the detector runtime); synthesis needs serve-llm running.

# Import a blood-smear still into the v1 specimen registry.
v1-ingest image:
    /tmp/run_python.sh scripts/v1_ingest.py {{image}}

# Run production vision on a registered specimen (vision stack).
v1-analyze specimen:
    scripts/vision_python.sh scripts/v1_vision.py --specimen {{specimen}}

# Full v1 acceptance on unseen specimens (needs serve-llm running).
v1-accept *images:
    scripts/vision_python.sh scripts/v1_accept.py --images {{images}}
