# MEMORY.md — persistent context for this project

## Phase 3 blood-axis audit (2026-09-09)

Read-only audit completed for the local TXL-PBC and Munich AML sources. The
reusable audit is `scripts/audit_phase3_datasets.py`; its report is
`artifacts/phase3_protocol_v1/dataset_audit.json`, and focused tests plus the
full suite passed (84 tests). TXL-PBC has matching YOLO image/label files in
882/252/126 train/val/test images, with 18,143 boxes and no malformed rows.
Munich AML has 18,365 original annotation rows, 21,049 augmented rows, and no
observed patient/specimen identifiers. The authoritative source study (Matek
et al., *Nature Machine Intelligence* 2019) explicitly defines myeloblast and
monoblast as blast equivalents and `Pblast = Pmyeloblast + Pmonoblast`; the
protocol therefore confirms MYO+MOB as positive and all other named classes as
negative, excluding UNC/nan. The prospective protocol is
`artifacts/phase3_protocol_v1/README.md`; it uses the supplied TXL split,
deterministic image-level AML splitting, frozen Path Foundation features,
explicit gates, and aborts on leakage or provenance ambiguity. No experiment
or training had started at that audit checkpoint; the later completion and
its disclosed detector recovery are recorded below.

> **This file survives across agent sessions.** When you pick up this repo again,
> read this first, then `ROADMAP.md` for the plan, then `AGENTS.md` for how to
> operate. Don't duplicate information between these three files — each has a
> distinct purpose:
> - `ROADMAP.md` = what's decided, what's left, acceptance criteria per phase
> - `AGENTS.md` = how to enter the dev env, run things, repo conventions
> - `MEMORY.md` = hard-won operational details that would otherwise be
>   re-derived (model quirks, path layouts that bit us, things that failed and
>   why, exact commands that work)

---

## Virchow model — exact architecture and quirks

**What it is:** ViT-Huge, patch14, 224×224, embed_dim=1280, depth=32,
num_heads=16, 631,229,184 parameters. Apache 2.0. From `paige-ai/Virchow`.

**Architecture details (from config.json + checkpoint inspection):**
- Gated MLP (SwiGlu-style): `fc1: 1280→6832`, splits into gate+proj (each 3416),
  `SiLU(gate) * proj`, then `fc2: 3416→1280`. MLP ratio = 5.3375.
- Layer-scale: `ls1.gamma` (attn residual) and `ls2.gamma` (MLP residual),
  both initialized to `1e-5`, shape `(1280,)`.
- Pre-norm: LayerNorm(eps=1e-6) before attention and before MLP.
- Attention: separate Q/K/V as fused `qkv.weight [3*1280, 1280]` +
  `qkv.bias [3*1280]`, plus `proj.weight [1280,1280]` + `proj.bias [1280]`.
  NOT `nn.MultiheadAttention` (which uses fused `in_proj` with different naming).
- `num_classes=0` → no classification head, just feature extraction.
- `pos_embed` shape in checkpoint: `[1, 257, 1280]` = 256 patches + 1 CLS.
  **NOT 2048+1** — I initially thought it might be rotary but it's learned
  positional embeddings for 224×224 input at patch14.
- `cls_token` shape: `[1, 1, 1280]`.
- `patch_embed.proj.weight`: `[1280, 3, 14, 14]`.

**Checkpoint files:** both `model.safetensors` and `pytorch_model.bin` exist and
have been checked tensor-for-tensor. The loader prefers safetensors and falls
back to the `.bin` file.

**Forward output:** `[B, 257, 1280]` token sequence (256 patch tokens + 1 CLS).

**Pooling choice used in this extraction:** CLS-only → `(B, 1280)` features.
This is what's saved in `embeddings.npy` for all three folds.

**⚠ Deviation from Virchow's documented embedding:** Virchow's own documented
feature vector is [CLS ∥ mean(pooled patch tokens)] = 2560-dim (1280+1280),
not CLS-only. CLS-only is a real, usable pooling choice and is not wrong — but
it's different from what Virchow was validated with, so linear-probe accuracy
may come out lower than published numbers that use the 2560-dim embedding.

**Decision for Phase 1:** Keep CLS-only for the smoke test. Re-extraction with
the 2560-dimensional documented pooling would be needed to compare against
published Virchow results. No accuracy range is assumed in advance: the Phase 1
probe must beat the training-majority classifier on held-out fold3.

**Transform pipeline:**
1. PanNuke images: `(N, 256, 256, 3)` float64 in `[0, 255]` (NHWC order,
   RGB channels confirmed — sample pixel `[212, 191, 201]` is a pinkish tissue
   color, looks right).
2. Convert to float32, divide by 255 → `[0, 1]`.
3. NHWC→NCHW (transpose 0,3,1,2).
4. Resize 256→224 bicubic with antialiasing (matches Virchow config: `interpolation=bicubic,
   crop_pct=1.0, img_size=224`). Since crop_pct=1.0 and we resize to exactly
   224, center-crop is a no-op.
5. Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225] (confirmed from config.json pretrained_cfg).
6. Normalize constants are CPU tensors shaped `[1, 3, 1, 1]`.

**Transform implementation:** `_virchow_transform_batch()` in
`scripts/extract_embeddings.py` does all of this vectorized over a batch tensor
(no per-image PIL loop). Uses `torch.nn.functional.interpolate` for resize.

**Key naming remapping (Virchow HF → our model):**
```
patch_embed.proj.weight  → patch_embed.weight   (Conv2d)
patch_embed.proj.bias    → patch_embed.bias
blocks.N.attn.qkv.weight → blocks.N.attn.qkv_weight
blocks.N.attn.qkv.bias   → blocks.N.attn.qkv_bias
blocks.N.attn.proj.weight → blocks.N.attn.proj_weight
blocks.N.attn.proj.bias   → blocks.N.attn.proj_bias
blocks.N.ls1.gamma       → blocks.N.ls1_gamma
blocks.N.ls2.gamma       → blocks.N.ls2_gamma
```
Also handles `"model."` prefix that some HF refs add — strips it and retries.

---

## NixOS environment — what actually works

**The machine:** NixOS, gfx1032 GPU (RX 6600), `nix-ld` NOT enabled system-wide
(`programs.nix-ld.enable = true` missing from configuration.nix). This means
prebuilt Python wheels (torch, numpy) downloaded by uv can't run without manual
`NIX_LD` shim.

**Working Python invocation (CPU mode, no ROCm):**
```bash
export NIX_LD=/nix/store/n51dhmdbik1kfrsm62j5knavmigwrl1a-glibc-2.42-84/lib/ld-linux-x86-64.so.2
export LD_LIBRARY_PATH="/nix/store/0vqb1mcas5j8dv6bhbrshinlgsg6bvgi-gcc-15.3.0-lib/lib:/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib:/nix/store/zks9mfsn4rqr6z9g6pcj2xqzcsplj0nb-zlib-1.3.2/lib"
export LD_PRELOAD=/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib/libgcc_s.so.1
# Then run python via the glibc loader:
$NIX_LD --library-path /nix/store/0vqb1mcas5j8dv6bhbrshinlgsg6bvgi-gcc-15.3.0-lib/lib:/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib:/nix/store/zks9mfsn4rqr6z9g6pcj2xqzcsplj0nb-zlib-1.3.2/lib \
  .venv/bin/python script.py
```

**Critical shared libraries needed (all in /nix/store):**
- `libstdc++.so.6` → `/nix/store/0vqb1mcas5j8dv6bhbrshinlgsg6bvgi-gcc-15.3.0-lib/lib/`
- `libgcc_s.so.1` → `/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib/` (must be LD_PRELOAD'd — libstdc++ depends on it at runtime)
- `libz.so.1` → `/nix/store/zks9mfsn4rqr6z9g6pcj2xqzcsplj0nb-zlib-1.3.2/lib/` (needed by numpy)
- glibc loader → `/nix/store/n51dhmdbik1kfrsm62j5knavmigwrl1a-glibc-2.42-84/lib/ld-linux-x86-64.so.2`

**Wrapper script that works:** `/tmp/run_python.sh` — a bash script with
hardcoded Nix paths that sets the env vars and execs `.venv/bin/python "$@"`
with the correct interpreter path
(`#!/nix/store/90nk33c4fkyg4x4dfk5cykqiryf2nlqq-bash-interactive-5.3p15/bin/bash`).
This is what all extraction commands go through.

**Bash path on this system:** `/nix/store/90nk33c4fkyg4x4dfk5cykqiryf2nlqq-bash-interactive-5.3p15/bin/bash`
(there is no `/bin/bash`). Any shell script that needs to survive a `nohup`
must use this absolute path as its shebang, or it will fail with
`/bin/bash: bad interpreter: No such file or directory`.

**What doesn't work:**
- Running `.venv/bin/python` directly without the `NIX_LD` shim → `libstdc++.so.6: cannot open shared object file`.
- Running python in background via `nohup` → processes die after 1-2 batches
  (signal 9 / OOM-kill suspected, though memory usage looked fine at ~400MB RSS
  per process). The foreground test with `timeout 120` worked fine and produced
  output, so the issue is specific to background/signal handling, not the code.
- Running multiple python processes in parallel → one gets "Killed" (signal 9).
  12 cores, 32GB RAM, but parallel torch inference processes don't coexist
  cleanly. **Stick to sequential batch processing.**
- `nohup /tmp/run_python.sh ... &` then checking `pgrep` → the process is gone
  but no error in log. The process likely got SIGKILL'd by the system, not a
  Python exception. When this happens, check if the output file exists — if it
  does, the batch completed before dying; if not, it died during forward pass.

**What does work:**
- Running python scripts **sequentially in foreground** via
  `/tmp/run_python.sh -u script.py args` — each batch takes ~70-80s and
  completes reliably.
- Running a shell loop that calls `/tmp/run_python.sh` for each batch
  sequentially — the shell survives, each python process starts fresh,
  completes, and exits. This is the approach that successfully extracted
  fold1 (42 batches) and is chewing through fold2.

---

## Batch extraction approach (what works for Virchow)

**Model state caching:** Load Virchow once, save state_dict to
`/tmp/virchow_state.pt` (2.53 GB, torch format). Then each batch script
does `model.load_state_dict(torch.load('/tmp/virchow_state.pt', ...))` —
fast (~5s) compared to loading from the original checkpoint each time.

**Batch script:** `/tmp/run_fold2_batch.sh` and `/tmp/run_fold3_batch.sh`
(written at extraction time, not in the repo). Each:
1. Checks if output file already exists → skips (idempotent, resumable).
2. Writes a temp Python script to `/tmp/fb2_${START}.py` (or `fb3_`) with the
   exact start/end hardcoded.
3. Runs it via `/tmp/run_python.sh -u /tmp/fb2_${START}.py`.
4. Removes the temp script.
5. Logs DONE/FAIL to `embeddings_virchow_foldN.log`.

**Batch size:** 64 images. ~70-80s per batch on CPU (4 threads, ~1 img/s).
Fold1 has 2656 images → 42 batches (41×64 + 1×32). Fold2 has 2523 →
40 batches (39×64 + 1×27). Fold3 has **2722** → 43 batches (42×64 + 1×18).
⚠ Fold3 is 2722, NOT 2444 — the 2444 figure was a wrong hardcoded count that
caused extraction to silently stop at 2444 (verified from images.npy file size:
2722 × 256×256×3×8B + 128B header = 4,281,335,936 bytes; and 2656+2523+2722 =
7901 = the official PanNuke split). Fixed in the scripts; top-up batches
extracted the missing rows 2444:2722 on Sep 7 2026.

**Publishing:** Each new batch is an atomic `.npz` containing explicit source
indices, float32 embeddings, and SHA-256 hashes of its source images. The
publisher rejects malformed batches, gaps, overlaps, non-finite values, and
image-hash mismatches, then atomically writes `embeddings.npy`, `labels.npy`,
and a commit-marker `provenance.npz` containing hashes of both canonical
arrays. Batch files are retained for audit. Legacy unindexed
`embeddings_batch_*.npy` files are not publishable.

**Current artifact status (2026-09-07 repair): verified.** All three folds were
re-extracted after the stitching/truncation audit using index-bearing batches
and antialiased preprocessing. The publisher validated complete, nonoverlapping
coverage and source-image hashes before replacing the canonical files. The
fail-closed verifier reports `ALL FOLDS OK` for fold1=2656, fold2=2523, and
fold3=2722. The retained batch `.npz` files provide row provenance.

**Idempotency:** The batch scripts check for existing output files and skip.
If a batch fails, just re-run that specific start index.

---

## PanNuke data layout (exact paths)

```
data/tissue/
  fold1/Fold 1/images/fold1/images.npy   # (2656, 256, 256, 3) float64 [0,255]
  fold1/Fold 1/images/fold1/types.npy    # (2656,) <U13, 19 unique classes
  fold1/Fold 1/images/fold1/masks.npy    # (exists but not used in Phase 1)
  fold2/Fold 2/images/fold2/images.npy   # (2523, 256, 256, 3)
  fold2/Fold 2/images/fold2/types.npy    # (2523,) 19 classes
  fold3/Fold 3/images/fold3/images.npy   # (2722, 256, 256, 3)
  fold3/Fold 3/images/fold3/types.npy    # (2722,) 19 classes
```

**Note the directory structure:** `foldN/Fold N/images/foldN/` — the "Fold N"
directory has a space in it ("Fold 2", "Fold 3"), and the subfolder is `foldN`
(no space). This bit us during fold2 extraction — the path
`Fold {fold}` with `fold="fold2"` gave `Fold fold2` which doesn't exist.
Correct path: `Fold 2` (hardcoded number) + `fold2` (variable for the
subfolder name).

**labels.npy:** String labels (19 tissue types). Need LabelEncoder before
training the head.

---

## Files on disk (Phase 1 artifacts)

```
embeddings/virchow/fold1/embeddings.npy   # (2656, 1280) float32 — VERIFIED
embeddings/virchow/fold1/labels.npy       # (2656,) string labels — DONE
embeddings/virchow/fold2/embeddings.npy   # (2523, 1280) float32 — VERIFIED
embeddings/virchow/fold2/labels.npy       # (2523,) string labels — DONE
embeddings/virchow/fold3/embeddings.npy   # (2722, 1280) float32 — VERIFIED
embeddings/virchow/fold3/labels.npy       # (2722,) string labels — DONE
scripts/verify_embeddings.py              # fail-closed content/provenance checks
models/vision/virchow/pytorch_model.bin   # 631M param checkpoint
models/vision/virchow/config.json         # architecture + pretrained_cfg
/tmp/virchow_state.pt                     # cached state_dict, 2.53 GB
scripts/extract_embeddings.py             # model definition + virchow loader
scripts/train_head.py                     # LogisticRegression head trainer (not yet run)
```

---

## Scripts in the repo (for reference)

- `scripts/extract_embeddings.py` — model definition (VirchowModel + components),
  `load_virchow()`, `_virchow_transform_batch()`, `virchow_embed()`. Also has
  path-foundation TF loading helpers. This is the canonical Virchow model code.
- `scripts/extract_path_foundation_all_folds.py` — sequential resumable Path
  Foundation extraction using the same index/image-hash provenance publisher.
  The local TensorFlow SavedModel was load-tested with TensorFlow CPU 2.21.0:
  input `(B,224,224,3)` float32 in `[0,1]`, output `(B,384)` float32. Its model
  card specifies a top-left 224×224 crop, which the extractor follows.

**Path Foundation artifact status (2026-09-07): verified.** Sequential CPU
extraction completed at roughly 24–25 images/second. Canonical shapes are
fold1 `(2656,384)`, fold2 `(2523,384)`, and fold3 `(2722,384)`. All batches
were retained, all canonical artifacts have source-image provenance, and
`scripts/verify_embeddings.py --backbone path-foundation` reports
`ALL FOLDS OK`. The linear probe reached 94.78% fold3 accuracy and 0.9255
macro-F1 versus the same 28.47% majority baseline.

**Phase 1 backbone decision:** Path Foundation is the pipeline winner. Virchow
scored 97.06% accuracy and 0.9562 macro-F1, ahead by 2.28 and 3.07 percentage
points, but extracted at only ~0.6 images/second. Path Foundation ran at
~24–25 images/second, has a ~113 MB SavedModel rather than a ~2.53 GB
checkpoint, and produces 384- rather than 1280-dimensional embeddings. That
efficiency is decisive for the offline interactive pipeline while retaining
excellent smoke-test performance. Virchow files/results remain as the Phase 1
audit reference but should not be wired into the final pipeline.
- `scripts/train_head.py` — sklearn LogisticRegression on embeddings. Loads
  from `embeddings/{backbone}/{fold}/`, trains on folds 1+2, tests on fold 3.
  Not yet run — waiting for all 3 folds of embeddings.

---

## Things that failed and why (for whoever picks this up)

1. **nohup + background python dies after 1 batch.** Tested many variants:
   nohup with wrapper script, nohup with inline python -c, bash -c loops,
   parallel submission. All died after 1-2 batches with no error in log.
   Suspect: SIGKILL from system (OOM or process limit), not Python exception.
   Workaround: sequential foreground execution via shell loop that calls each
   batch as a fresh python process.

2. **Parallel batches don't coexist.** When 3 batches were launched simultaneously,
   one got "Killed" (signal 9). Even though total memory looked fine (23GB free),
   torch processes don't play well together on this machine. Sequential only.

3. **Path construction with f-strings for PanNuke dirs.** `fold2/Fold 2/` has a
   space, and the subfolder is `fold2` not `fold2/foldfold2`. Constructing
   `f"Fold {fold}"` with fold="fold2" gives wrong path. Need to hardcode the
   "Fold N" directory name separately from the variable fold name.

4. **nohup can't find /tmp scripts.** `nohup /tmp/run_python.sh ...` failed
   with "No such file or directory" even though the file existed. Root cause:
   the script had `#!/bin/bash` shebang but `/bin/bash` doesn't exist on this
   NixOS system. Fixed by using the absolute bash path in the shebang.

5. **Bash scripts launched via nohup die silently.** The shell script itself
   would start, launch a python batch, then the python process would die and the
   shell loop would continue to the next iteration — but the log showed no new
   entries. Root cause: the python process died, `set -e` in the shell script
   caused the shell to exit. Fixed by NOT using `set -e` and checking return
   codes explicitly, or by running each batch as its own foreground command.

6. **`/tmp` scripts need absolute bash path in shebang.** `/bin/bash` doesn't
   exist on this NixOS system. All scripts in `/tmp` must use
   `#!/nix/store/90nk33c4fkyg4x4dfk5cykqiryf2nlqq-bash-interactive-5.3p15/bin/bash`.

---

## GPU/ROCm status

ROCm not tested for this extraction — everything ran on CPU. The RX 6600 is
gfx1032, needs `HSA_OVERRIDE_GFX_VERSION=10.3.0` to spoof as gfx1030 for ROCm
support. The `rocm` devShell in the flake sets this up. CPU extraction works
and is acceptable per ROADMAP.md ("CPU is an acceptable fallback for that stage
if ROCm fights the gfx1032 override"). Speed: ~1 image/sec on CPU with 4 threads.
If ROCm works, expect significant speedup for fold2/3.

---

## Appearance/context correction (2026-09-09)

The repaired fixed-64-patch cache is in
`artifacts/phase2_appearance_context_classifier_v1/cache/`; the old
row-sliced cache is preserved as `cache_row_sliced_superseded/`. Epoch 29 is
retained as the current development classifier candidate. End-to-end
macro-F1 is **0.484114**, valid true-mask macro-F1 is approximately **0.633960**,
and all five class F1 scores improved. Four of five gates pass; only
end-to-end macro-F1 >= 0.55 fails. Geometry is exactly
TP/FP/FN=40,894/9,608/18,475, detection F1=0.744400, binary PQ=0.596357.
Appearance/context adds useful information, but the remaining gap is
approximately 0.065886. Historical validity-filter contamination remains
indeterminate; fold2 remains development data and fold3 was untouched.

## Phase 1 acceptance criteria (from ROADMAP.md)

- End-to-end run completes without manual intervention.
- Test accuracy exceeds the held-out result of predicting the training-set
  majority class.
- This proves the encoder → embeddings → training loop is correctly wired.

Current status: all three Virchow folds are re-extracted and verified with
source-image provenance. The Phase 1 linear probe completed on 2026-09-07:
97.06% fold3 accuracy and 0.9562 macro-F1, compared with the 28.47% held-out
accuracy of predicting the folds1+2 majority class (Breast). Results are stored
in `embeddings/virchow/head_results.json`; the Phase 1 smoke test passed.

---

## Phase 2 PanNuke masks — verified source contract (2026-09-07)

Run `/tmp/run_python.sh scripts/validate_pannuke_masks.py --output
artifacts/pannuke_mask_audit.json` before a target-generation or segmentation
run. `scripts/pannuke_masks.py` validates the raw shape `(N,256,256,6)`,
finite non-negative integer-valued instance IDs, binary channel-5 background,
and image/mask/type row alignment. The five foreground channels are, in order:
neoplastic, inflammatory, connective, dead, epithelial. IDs are sparse labels,
not counts.

Do not assume foreground channels are mutually exclusive or that background is
their complement: the local audit finds small source overlaps and duplicate IDs
across class channels within patches. Any model-specific conversion must make a
tested, documented policy rather than silently resolving them. Keep folds 1+2
for training and fold3 held out. Full source-mask scans are sequential and
I/O-heavy; never run competing Python vision processes.

**Astra correction:** numeric IDs are scoped by `(patch, class, ID)`; repeats
across class channels are not evidence of ambiguity. Only pixel-level
foreground overlap and unlabelled void pixels need an approved target policy;
the audit found zero foreground/background contradictions. Exact image-content
audit found no cross-fold duplicates, but patient/slide grouping is unavailable.
Phase 2 development is fold1→fold2, then (only after settings
freeze) optional folds1+2 refit and one fold3 evaluation. Patient/slide IDs
are absent locally, so do not claim patient-independent splitting. The separate
trainable HoVer-Net-fast segmentation stage is an explicit Phase 2 exception
to the frozen Path Foundation rule, not permission to fine-tune it.

**Approved target policy:** `pannuke-target-v1-ignore-ambiguous-instances`
(`scripts/pannuke_target_policy.py`). Void pixels are ignore labels; any
foreground-overlap or disconnected instance is wholly ignored and counted by
class; valid channel-5 background remains background. A fully void patch has no
 valid pixels and is never a negative example. Border instances are retained
and scored normally in the primary evaluation; any interior-only analysis is
secondary. Fold2 numerical
gates: detection F1 >=0.70, binary PQ >=0.50, end-to-end macro-F1 >=0.55, Dead
recall >=0.20, Dead F1 >=0.15.

**Current Phase 2 code/artifacts (2026-09-08):** `hover_fast_model.py` is a
compact multiscale encoder–decoder with NP/HV/type branches; it is a local
baseline, not an official HoVer-Net reproduction. Targets, masked losses,
post-processing, and basic instance metrics live in `pannuke_target_policy.py`,
`hover_loss.py`, `hover_postprocess.py`, and `nucleus_evaluation.py`.
`pilot_hover_fast.py` trains fold1 only, validates fold2 only, and never opens
fold3. The coverage-aware 512-step run is in
`artifacts/phase2_development_run_coverage/`: fold2 subset indices
`[0,3,31,155,1,2,4,5,6,7,8,9,10,11,12,13]` covers all five classes with
supports 104/13/55/6/12. It produced TP=15, FP=103, FN=175, sum-IoU=9.59;
this is an early baseline, not acceptance. All tests passed before handoff.

**Phase 2 continuation (2026-09-08):** The evaluator was completed in
`nucleus_evaluation.py` and `evaluate_hover_checkpoint.py`: it now emits
detection precision/recall/F1, binary PQ, five-class end-to-end
precision/recall/F1/PQ/support, macro-F1, and macro-PQ with additive raw counts.
Wrong matched types count as a class FP+FN. Tests cover aggregation, empty
cases, unmatched instances, wrong types, and ignore handling (24 tests pass).

The original 512-step checkpoint evaluates at detection F1 0.0974, binary PQ
0.0623, macro-F1 0.0036, and Dead F1 0; it predicted essentially all detected
nuclei as neoplastic. The v2 loss balances NP foreground with BCE+Dice,
restricts type loss to nucleus pixels with within-batch class weighting, and
uses uniform-class-then-uniform-containing-patch sampling across fold1.

The v2 fixed-budget run in `artifacts/phase2_development_run_v2/` was manually
stopped to conserve session usage after the 1,536-step checkpoint. No process
remains and fold3 was never opened. Monitoring results were:

- step 512: detection F1 0.4000, binary PQ 0.2905, macro-F1 0.0325, Dead F1 0
- step 1024: detection F1 0.4386, binary PQ 0.3231, macro-F1 0.1469, Dead F1 0
- step 1536: detection F1 0.4042, binary PQ 0.2908, macro-F1 0.1763, Dead F1 0

`best_fold2_monitor.pt` is step 1024 by the predeclared normalized-gate score;
`checkpoint.pt` is step 1536. The run did not reach any development gate and
must not proceed to fold3. Future pilot runs persist `progress.json` at every
validation gate so interruption does not lose monitoring history.

**Phase 2 v4/v5 and Astra review (2026-09-08):** The deeper v4 trajectory ran
4,096 steps; v5 deterministically repeated it and extended to 8,192 steps.
The v5 final checkpoint reached detection F1 0.6045, binary PQ 0.4587,
macro-F1 0.2387, Dead recall 0.0772, and Dead F1 0.1176. Step 5,632 had Dead
F1 0.2029 but weaker detection/PQ and zero correctly typed epithelial nuclei;
the old normalized scalar selection therefore favored an unsuitable
trade-off. No v4/v5 checkpoint passed more than one gate, and unchanged
training must not be extended.

Astra independently reproduced a real typing collapse and two instance
extraction defects: both HV Sobel derivatives used the last axis, and a single
patch-wide distance percentile could suppress small nuclei. These are fixed in
`hover_postprocess.py`: derivatives use their explicit horizontal/vertical
axes and component-safe local distance peaks replace the global percentile.
Tests now require exactly two instances for touching nuclei and retention of a
small isolated nucleus. On the fixed 128-patch fold2 monitor, repaired
post-processing with perfect foreground and target HV yields detection F1
0.9688 (TP 2,923, FP 144, FN 44); predicted v5 foreground/HV yields F1 0.6094
(TP 2,144, FP 1,925, FN 823). Perfect foreground with predicted HV is nearly
identical to perfect target HV, while predicted foreground remains poor even
with target HV. Foreground prediction is therefore the dominant geometric
failure on this diagnostic, not HV prediction.

`diagnose_hover_checkpoint.py` records predicted/perfect foreground/HV
comparisons, matched-type and true-mask confusion matrices, and split/merge
counts without opening fold3. V5 true-mask typing predicts only 3/984
epithelial nuclei correctly, confirming typing collapse independently of
extraction. `pilot_hover_fast.py` no longer declares one scalar-score winner:
prospective runs retain separate detection, macro-F1, and rare-class
candidates, and only a checkpoint passing every predeclared gate is accepted.
All 30 tests pass; ruff is clean on the corrected Phase 2 files. Fold3 remains
locked.

**Phase 2 v6/v7 controlled experiments (2026-09-08):** Fold1 contains valid
instance counts `[26062, 10676, 16204, 961, 8806]` for neoplastic,
inflammatory, connective, Dead, and epithelial under the approved target
policy. V6 replaced volatile per-patch class balancing with normalized
inverse-square-root weights derived from those fixed fold1 counts and added
3x foreground-boundary weighting. At its converged step-3,584 candidate it
reached detection F1 0.5424, binary PQ 0.3966, macro-F1 0.2609, Dead recall
0.3211, Dead F1 0.3010, and epithelial F1 0.1359. Diagnostics showed the type
change restored both rare-class and epithelial predictions, but foreground
geometry regressed. With perfect foreground, the same checkpoint reached
detection F1 0.9688 and macro-F1 0.4561; predicted foreground with target HV
reached only detection F1 0.5511. This confirms foreground prediction remains
the dominant geometric limitation and the typing ceiling also remains below
the 0.55 macro-F1 gate.

V7 was the controlled fallback: it retained only the fixed fold1-derived type
weights and restored the original NP BCE+Dice objective. Its detection and
macro candidates both select step 3,584: detection F1 0.5404, binary PQ
0.3949, macro-F1 0.2495, Dead recall 0.2724, Dead F1 0.2696, and epithelial F1
0.1911. The rare-class candidate is step 1,536. Thus fixed weights genuinely
stabilize Dead performance, but both v6 and v7 sacrifice the stronger v5
detection trajectory; boundary weighting is not the sole cause. The evidence
supports a negative-transfer conflict in the shared decoder/features as a
working hypothesis, not yet a proven mechanism. Do not extend v5-v7 unchanged.
The next decision is whether to decouple instance typing into a separate
branch/instance classifier or adopt a more faithful multi-decoder HoVer-Net
design. No process remains, all recorded runs completed cleanly, and fold3 was
never accessed.

**Phase 2 calibration and gradient isolation (2026-09-08):**
`calibrate_hover_thresholds.py` evaluates saved checkpoints on the same fixed
fold2 monitor and reports foreground-pixel and instance metrics. All tested
v5-v7 checkpoints favored the highest predeclared threshold, 0.65. V5 final
improved to pixel F1 0.8015, detection F1 0.6268, and binary PQ 0.4740, so
calibration explains part but not all of the gap.

V8 is a matched 4,096-step v7 intervention with type features detached before
the type head. Initialization, seed, patch sampling, augmentation, optimizer,
losses, and validation steps match v7. At matched step 3,584, detection F1 rose
from v7's 0.5404 to 0.5841 and PQ from 0.3949 to 0.4370. At step 4,096 it was
0.5839/0.4431. Threshold 0.65 raises the final checkpoint to detection F1
0.6161 and PQ 0.4691. Typing collapses (macro-F1 0.0707 calibrated), as a 1x1
head cannot learn useful types from features trained only for NP/HV. The
matched geometry improvement supports harmful type-gradient influence
somewhere in the shared features; it does not localize the conflict to the
decoder.

V9 tested the evidence-backed next architecture: an independent multiscale
type decoder with the encoder still shared. Its uncalibrated detection
candidate (step 3,584) reached detection F1 0.5513/PQ 0.4081. Its macro/rare
candidate (step 4,096) reached macro-F1 0.2694, Dead recall 0.3902, Dead F1
0.2165, and epithelial F1 0.3837. At threshold 0.65 that balanced checkpoint
reaches detection F1 0.5604, PQ 0.4215, and macro-F1 0.2827. Thus task-specific
decoder capacity improves typing but encoder sharing still sacrifices the
isolated segmentation gain, and no acceptance gate set is met. The next
evidence-backed design is a fully decoupled second-stage instance classifier
on top of a frozen segmentation model, evaluated with both true and predicted
instances. A faithful HoVer-Net remains a later benchmark, not the immediate
next intervention. No further training is active; fold3 remains locked.

**Phase 2 decoupled classifier and segmentation-only extension
(2026-09-08):** `train_instance_classifier.py` freezes segmentation, pools
encoder1/decoder1 mean and standard-deviation features plus RGB and geometry
statistics per nucleus, fits a balanced five-class logistic classifier on
fold1 only, and evaluates true and predicted fold2 instances separately. It
saves portable coefficients/scaling and records checkpoint SHA-256, target
policy, folds, threshold, supports, and fold3 access.

The first classifier, paired with v8 step 4,096, trained on 62,709 valid fold1
nuclei and reached true-instance macro-F1 0.6140, proving the frozen features
contain sufficient type signal. End-to-end macro-F1 was 0.3903, limited by
detection F1 0.6161/PQ 0.4691.

V10 deterministically repeated v8 and extended segmentation-only training to
8,192 steps. Its step-7,680 detection candidate calibrates to threshold 0.60:
TP 2,172, FP 1,450, FN 795, detection F1 0.6593, and binary PQ 0.5051. The
paired classifier in `artifacts/phase2_instance_classifier_v2/` reaches
true-instance macro-F1 0.6190 (class F1
0.6626/0.4968/0.5050/0.6329/0.7978). End-to-end macro-F1 is 0.4193; Dead
recall 0.5244 and Dead F1 0.2919 pass comfortably. Binary PQ and both Dead
gates pass; detection F1 (required 0.70) and end-to-end macro-F1 (required
0.55) do not. Task decoupling is validated and remaining work is isolated to
segmentation quality, missed nuclei, and spurious instances. Do not spend more
on type weighting, joint decoders, or unchanged compact-model extension. No
process remains and fold3 stays locked.

**Phase 2 proposal suppression and full-fold2 evaluation (2026-09-09):**
`evaluate_instance_suppression.py` trains a binary validity filter on 85,276
fold1 predictions, 46,702 of which match evaluable fold1 nuclei at IoU > 0.5.
It combines frozen pooled instance features with foreground-confidence
statistics. Selection uses only the fixed 128-patch fold2 monitor, maximizes
detection F1 subject to Dead recall >=0.20, and compares a confidence-only
baseline. The learned threshold is 0.35; the confidence-only threshold is
0.80. Neither segmenter nor type classifier is updated.

On all 2,523 fold2 patches (59,369 evaluable nuclei), the unfiltered pipeline
has detection F1 0.6336/PQ 0.5021. Confidence rejection reaches 0.7085/0.5668.
The learned filter reaches TP 40,894, FP 9,608, FN 18,475, detection F1 0.7444,
and binary PQ 0.5964, passing both geometry gates. Dead recall is 0.4636 and
Dead F1 0.2381, also passing. Meaningful remaining FP categories are 4,880
background-only, 1,641 merge, 672 split, and 2,415 boundary; overlap requires
at least 10% of the smaller instance, so one-pixel contacts do not define a
merge.

`evaluate_full_fold2_details.py` reproduces the fixed filtered pipeline and
adds full-fold true-mask typing, class-wise geometric recall, and matched-type
confusion. True-instance classifier macro-F1 is 0.5453, not the 0.6190 seen on
the enriched monitor. True-instance class F1 is
0.6941/0.6364/0.5656/0.2716/0.5586; Dead has recall 0.8455 but precision
0.1618, demonstrating overprediction. Geometric recall by class is
0.6866/0.7788/0.6289/0.5795/0.7109. Filtered end-to-end class F1 is
0.5420/0.5501/0.3788/0.2381/0.4442, for macro-F1 0.4306. Thus detection F1,
binary PQ, Dead recall, and Dead F1 pass on full fold2; only end-to-end
macro-F1 >=0.55 remains unmet. Classification is not solved, and remaining
errors cannot be assigned solely to segmentation. Artifacts are in
`artifacts/phase2_instance_suppression_v1/`; hashes and fold3=false are
recorded. No process remains and fold3 is locked.

**Phase 2 bounded score adjustment (2026-09-09):** The validity-filter
training code now excludes proposals with zero evaluable pixels under the
ignore policy and records the excluded count; such proposals are neither
positive nor negative examples. FP diagnostics now use substantial overlap
(10% of the smaller instance) and distinguish background-only, split
fragments attached to matched nuclei, merges, boundary/localization failures,
and unresolved cases. Matching, retention, masks, and classifier predictions
are unchanged.

`artifacts/phase2_score_adjustment_v1/` contains the frozen fold2 retained
instance/raw five-class decision-score cache, manifest, complete monitor
curves, focused follow-up grid, and full-fold2 report. Zero-offset full-fold2
reproduction is detection F1 0.744400, binary PQ 0.596357, macro-F1 0.430647.
The monitor-selected Dead-only offset is -0.4 (macro-F1 0.470391); because it
did not reach 0.55, the declared focused grid selected Connective +0.4 and
Dead -0.4 (monitor macro-F1 0.474905). Full fold2 development macro-F1 is
0.446085, with unchanged detection/PQ and Dead recall/F1 0.427273/0.290797.
The 0.55 gate remains unmet. Frozen input hashes are in the report and
fold3_accessed is false; fold3 was not loaded, inspected, enumerated, or
hashed.

## Nonlinear classifier comparison (2026-09-09)

`select_validity_examples` in `scripts/evaluate_instance_suppression.py`
excludes zero-evaluable proposals while preserving aligned rows, and
`false_positive_categories` classifies positive subthreshold overlap as
`other_unresolved`. Direct tests cover both corrections. The frozen validity
artifact is unchanged; its historical report records 85,276 proposals, 46,702
positive evaluable proposals, and 38,574 remaining negative/evaluable-or-
excluded proposals, but cannot establish row-level contamination provenance.

The one prescribed MLP (seed 20260909, 107->64 ReLU, dropout .20, AdamW,
30 epochs) selected epoch 29. Its full fold2 development results were detection
F1 0.744400, binary PQ 0.596357, end-to-end macro-F1 0.471912, matched typing
accuracy 0.694087, Dead recall 0.471591, and Dead F1 0.294013. It was not
retained because the 0.55 end-to-end gate remains unmet. Artifacts are in
`artifacts/phase2_nonlinear_classifier_v1/`; fold3 remained untouched.

Repair conclusion: the invalid predicted-proposal true-mask fields were
superseded. The corrected cache has 59,369 full-fold2 truth instances, 107
finite features, and 2,967 fixed-monitor rows; valid full-fold2 true-mask
macro-F1 is 0.607426 and epoch 29 remains selected. The MLP is retained as the
current development candidate because it improves adjusted-logistic
end-to-end macro-F1 and valid true-mask macro-F1 while preserving both Dead
gates. Proposal-mask feature mismatch is the next investigation. The fold1
provenance scan found 85,276 proposals = 46,702 evaluable positives + 36,407
evaluable negatives + 2,167 zero-evaluable proposals. Historical contamination
is indeterminate because the frozen artifact lacks row-level membership; no
refit was performed. Fold3 remained untouched.

The proposal-mask intervention selected epoch 30 but did not improve the
corrected control: full-fold2 end-to-end macro-F1 0.474432 and true-mask
macro-F1 0.475003. It is not retained; next evidence-backed step is
type-specific appearance/context features. Fold3 remained untouched.

Corrected wording and housekeeping: proposal-trained macro-F1 improved from
0.471912 to 0.474432, with a negligible Dead-F1 trade-off. The duplicate fold2
proposal cache was replaced by a canonical-cache reference manifest; canonical
SHA-256 is `e764e471a1cc2c801a5d5863909067d5c7d8edd1cf6f186006942bd8ae25e165`.
The fixed 41-column extractor is in `scripts/appearance_context_features.py`
and focused tests pass. The augmented cache build was terminated before output;
no model result or retention decision is claimed. Fold3 remained untouched.

The superseding appearance/context run completed with repaired fixed-64-patch
shards. The reusable extractor caches patch-level RGB, grayscale, Sobel,
uniform LBP, and radius-5 disk context once per patch. The single approved
seed-20260909 148-feature MLP selected epoch 29; full-fold2 end-to-end
macro-F1 was 0.484114 versus 0.471912 control, with unchanged geometry
(TP/FP/FN 40,894/9,608/18,475; detection F1 0.744400; binary PQ 0.596357).
The 148-feature epoch-29 classifier is retained as the current best fold2
development candidate because it improves the corrected control. “Accepted”
still means every Phase 2 gate passes; Phase 2 is not accepted because the
end-to-end macro-F1 gate remains unmet. No follow-up experiment was started.
Fold3 remained untouched.

Validity-filter repair (2026-09-09): fold1 fit hashes verified and preserved;
fold2 pre-filter cache regenerated with 2,523 patches and 80,952 unfiltered
proposals. Correct 111-feature validity and separate 532-feature inference pass
all five gates: F1 0.744241, PQ 0.596561, macro-F1 0.569366, Dead recall
0.436364, Dead F1 0.396899. Digest 9ba36ff4739b0358dd61f97bdf91d203e32acc5a94090e5d498eef788d195b80.
No training/refitting/tuning; historical filter preserved; fold3 untouched.

## Future MedCPT/MedGemma candidates and Phase 4 history (2026-09-09)

Historical Phase 4 retrieval protocol v1 (2026-09-10): MedCPT query/article
roles were selected and verified from local pinned manifests.
`scripts/phase4_retrieval.py` implemented separate build/query/evaluate stages,
deterministic symbolic/vector/hybrid ranking, SQLite ontology metadata, and
embedded vectors. Its source audit recorded that the local fixture lacked a
licensed GO OBO snapshot, so that boundary was incomplete. The v1/v2
boundaries are now retained only as invalid exposed regression evidence; do
not use either for scientific gate certification.

Acquisition is implemented by scripts/acquire_future_models.py and records
machine-readable manifests. Query MedCPT revision is
d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc and article revision is
d05a736da4bb84ee4057b7f7999485be6ed85465. Their model.safetensors files are
437951328 bytes with SHA-256 19d78c0d5eaee2f81e6c47c5425bbadcc0c6af016cbb5da4a000d64e59d6e342
and a5d5ffe4d8666c1d0aa15f371b94fc3492ca8f927e5621abd4b3ee9fc845b0f3.
MedGemma revision is 3855f948626b7ae42bccd082757f15078c53e758; its
2829699136-byte Q5_K_M file has SHA-256
1085085f6186e09f075dce216048e5b063ba419a5f8286d0cd66c74a4bb155d8.
Verification passed offline for MedCPT and at llama.cpp context 8192 for
MedGemma. Full timings and scores are in artifacts/future_models_verification.json.
These are future candidates only; selected components, Phase 3/4, RudolfV 2-S,
and fold3 were untouched.

Path Foundation context comparison (2026-09-09): the bounded 532-column
fold1-to-fold2 run selected epoch 30 prospectively. Monitor macro-F1 was
0.5565232993468779 versus 0.5133344600416373 control; full fold2 macro-F1 was
0.5696812332617677. All five frozen fold2 development gates passed and
geometry stayed exactly 40,894/9,608/18,475. The 532-feature candidate is
retained and frozen, awaiting Astra final-readiness review before the one
permitted fold3 evaluation. This is not final Phase 2 completion or a fold3
generalization claim; historical validity-filter provenance remains unresolved.

Final-readiness hardening (2026-09-09) corrected ignored-only accounting and
added descriptive tissue diagnostics. The earlier validity-filter refit is the
sole refit; classifier, segmentation, thresholds, features, and model choice
were unchanged in this pass. The complete transitive manifest is verified and
fold3 remained wholly untouched. Phase 2 is ready for one-shot fold3 evaluation
but is not finally complete.

Final execution-readiness corrections also bound the segmentation checkpoint,
loader/model code, pooled proposal features, generic state machine, recursive
model-tree verification, and prediction-shard verification. Fold2 canonical
tissue alignment reconciles 2,523 patches and 59,369 valid-class instances;
aggregate metrics and the frozen prediction digest are unchanged. No training,
refitting, tuning, or model changes occurred; final evaluation remains unrun.

Final protocol scoring completed from immutable v2 predictions. The scorer
defect was `semantic < 0` instead of the target policy's `IGNORE=255`; after
correction aggregate metrics and digest matched exactly, with byte-identical
complete reproductions and reconciled tissue diagnostics. Fold3 stayed locked.
Final production closeout (2026-09-09): all 2,523 fold2 production shards
passed schema, metadata, score, threshold, and retained-map reconstruction
checks. Corrected reports A/B are byte-identical at
`2ae19f0b3d3519496791d763b264c3721223afd77d093798ef1af6c4e2b3a59e`; ignored-only
accounting is 3,151 = 82 retained + 3,069 rejected. Synthetic final-mode
interruption/resume and self-contained scoring passed. Fold3 remains locked.

Direct fold2 rehearsal v3 completed end-to-end through the canonical state
machine. Freeze digest is
`9ba36ff4739b0358dd61f97bdf91d203e32acc5a94090e5d498eef788d195b80`; reports
A/B are byte-identical with SHA-256
`f3dbddcbcd563aa98ecdc14b7ee6c1c1598ec417c768383097c6d92d8522a928`; metrics
are detection F1 `0.7442408090248622`, binary PQ `0.5965614410791367`, macro-F1
`0.5693660649671026`, Dead recall `0.43636363636363634`, and Dead F1
`0.39689922480620154`; ignored-only is `3151 = 82 retained + 3069 rejected`.
Completion and generic rehearsal verification passed. Readiness marker SHA-256
is `30d04f07d47785f6928afa4ac6f94a62f759662aaf0e17adbe9eedb6f9b3a3e7`.
Fold3 was never accessed.

One-shot fold3 final evaluation (2026-09-09): after Astra GO preflight, one
fresh `final_evaluation` run completed through the canonical CLI under the
persistent external lock. All 2,722 shards committed; freeze digest is
`501b5e04f0a0bc5654215e9a6b9491f34007e7b1601c7089f78c16eae37dd01c`. The
single report `artifacts/phase2_final_protocol_fold3_final_v1/final_report.json`
has SHA-256
`dd2b92cce4eb450ae41039aa1b95c8ed2acd18647c888e39c7b60f7ff368ce1c`.
Metrics: detection F1 `0.7441618925936612`, binary PQ `0.597810029945104`,
end-to-end macro-F1 `0.5555736387146519`, Dead recall `0.33747609942638623`,
Dead F1 `0.3232600732600733`; all gates pass. Ignored-only is
`3803 = 85 retained + 3718 rejected`. Complete and canonical verify passed.
This is the sole fold3 evaluation; preserve artifacts and do not rerun absent
an invariant failure and Astra review.

Phase 3 completed. Detector recovery was disclosed and single-use. TXL frozen
Path Foundation ground-truth crops test macro-F1 0.9905134073389213. AML
original-only frozen Path Foundation selected C=0.1 and threshold=0.8500000000000001;
test AUROC 0.9934856969339728, AUPRC 0.9644221463467961, macro-F1
0.9464670830654076, sensitivity 0.898989898989899; all gates pass. AML is
image-level-only and not patient-independent; no augmented data.

## Phase 4 final retrieval closeout (2026-09-10)

The v1/v2 Phase 4 artifacts remain byte-preserved but are invalid exposed
regression evidence, not gate-certifying results. The forensic record is
`artifacts/phase4_protocol_v3/recovery/v1_v2_forensic_rescore.json`; it is
read-only and documents why those boundaries cannot be used. The v3 quality
boundary is a separate frozen prospective benchmark with 31 cases over 38,245
current GO terms. Its report SHA-256 is
`2d9973a334e69af11fed5a6c286ec679f5a4a16575e1be24275efc35c3c87997`.
Quality and integrity gates pass: symbolic recall@5/MRR/exact-ID
`1.0/1.0/1.0`, vector `0.88/0.8159814323607427/1.0`, hybrid
`1.0/0.9597701149425287/1.0`, plus traversal, provenance, offline, and
deterministic reproduction. Only the v3 latency gate failed initially, at
p95 `8397.242406/7686.852608/8740.750299` ms for symbolic/vector/hybrid.

The first separately frozen latency-only recovery
(`phase4_v3_latency_recovery_v1`) changed only set-backed tail membership and
kept all 87 rankings equivalent and duplicate-free, but its p95 was
`354.853367/119.592767/490.707502` ms (symbolic/vector/hybrid), so it is
retained as a failed recovery with no tuning. Authorization SHA-256 is
`989213789b06d77b15fd06173754eaad96c54b3303080ed3c58205f4e5aa373f`; report
SHA-256 is
`0a70f2367ea7b8475aaba91b810748862ac1ebd9b6e1469f3ad13331fd4ecb0e`.

The final separately authorized latency-only recovery
(`phase4_v3_latency_recovery_v2`) preloads immutable SQLite-derived alias/name
postings and performs exact lexical verification. It changed only
`scripts/phase4_v3_query.py` (final implementation code SHA-256
`cce08d213454780fb2967a7857e9435bc7bb915a660a29bb8611bf68eaed5fed`), did
not rerun retrieval quality, and left v3 plus the first-recovery inventories
unchanged. The sealed postings contain 138,396 alias rows and 38,245 name
rows. Final timing used 29 eligible cases × 3 modes, 2 warmups + 5 measured
repeats (145 samples/mode; nearest rank 138): p95 symbolic `13.026316` ms,
vector `132.087829` ms, hybrid `144.417922` ms, all under 200 ms. Final report
SHA-256 is
`cb08a0344e3bd41cb73d6803b580b3a787d04118717fb5feb2671cfdae48571b`; raw
samples SHA-256 is
`43dbe4bfbae44ea462a95e04f5e24461871566debfd25ad95c62c7b876ff5a9e`.
The final verifier passed twice with identical output; ranking-equivalence
artifact SHA-256 is
`9d64bede63a6f087108ffcbe85bdba91eac477d5f389b68ffcfde22e9fe62623`, and
the v3 ranking file remains SHA-256
`d25b02bdd4451f67517662b9c2e2495b0bfa66b256779b5c6179d677bef938e7`.
Phase 4 is accepted on the sealed v3 quality evidence plus the final
latency-only recovery; no LLM was involved.

Closeout verification: canonical Phase 2 fold3 verification passed at
2,722/2,722 shards with digest
`501b5e04f0a0bc5654215e9a6b9491f34007e7b1601c7089f78c16eae37dd01c`, the
Phase 3 cross-artifact verifier passed, scoped Ruff passed, and the full test
suite passed 126 tests (71 + 31 + 24 sequential chunks). No scientific
evaluation was rerun after the final recovery boundary.

## Phase 4 snapshot reconciliation (2026-09-11)

The v3 freeze records predated two authorized `query.py` changes, so the
default snapshot gate raised while every data hash matched (issue #2).
Proven chain: v3 freeze `2900b86b` -> v1 tail membership `900f445d` ->
v2 postings cache `cce08d21`; the tree matches the v2-recorded final
exactly and the other nine code files match the v2 pre-change record.
Reverting is impossible (no pre-change copy survives) and re-freezing is
forbidden by the v2 authorization. Resolved without mutating any sealed
file or code: new `recovery/reconciliation_v1/` boundary plus
`scripts/phase4_snapshot_reconciliation.py` (intended-state proof and the
sanctioned bound-load path) with regression tests, including bound
symbolic output byte-identical to the sealed v3 ranking. Consumers must
use the bound loader; the plain default path still raises by design.

## Phase 5 Part 2 comparison stop (2026-09-11)

Both candidates ran all 24 dev generations under the frozen protocol and
both are ineligible: valid structured-response rate 0.0 each. MedGemma
emitted prose in all 24; OpenBioLLM emitted near-JSON with wrong field
names, missing case_id, or truncation. Per the frozen winner rule the
phase stops with no winner and no gate weakening; Part 3 is blocked on a
prompt/decoding re-freeze decision. Records in
artifacts/phase5_comparison_v1/; lifecycle event part2_stopped.

## Phase 5 Part 2 v2 selection (2026-09-11)

Rerun under the v2 contract selected MedGemma: valid rate 1.0 with
finding/GO recall 0.875/0.875 and repeatability 0.75, no forbidden hits;
OpenBioLLM valid 0.833 with hallucinated case_ids and a bone-marrow
location contradicting the packet limitations. Blinded review was
unanimous (implementer, user, second reviewer): A accept, B reject;
mapping A=MedGemma, B=OpenBioLLM. Lifecycle event part2_selected;
records in artifacts/phase5_comparison_v2/. serve-llm default now
MedGemma; OpenBioLLM kept as local audit copy only.

## Phase 5 Part 3 final acceptance (2026-09-11)

Final set opened exactly once via the guarded runner: 6 cases x 3
repeats, all pass scoring with byte-identical repeats. Human review
accepted (closest calls: definition-heavy neoplastic synthesis,
duplicate txl-multi claims, distractor citations in epithelial case;
none crossed the rejection bar). Lifecycle event part3_accepted;
report in artifacts/phase5_final_v1/. Phase 5 complete; dashboard
remains Phase 6 work.

## Ren v1.0.0 refinement (2026-09-11)

Shipped on branch yuzy/v1-refinement: M1 PNG/JPEG ingestion + registry
(TXL files are JPEG despite .png names, accepted deliberately); M2
production vision (YOLO best.pt@2a98dc5a, Path Foundation, logistic head;
16 findings match GT composition on val probe; ~10 s cold per smear via
scripts/vision_python.sh, which supplies GL/X11 libs the full-opencv build
needs); M3 finding contract + correction lifecycle; M4 deterministic
taxonomy-to-query wiring with origin tracking; M5 dashboard primary
workflow; M6 acceptance green on 2 unseen smears plus 5 failure paths.
Production packets cap auto evidence at top-3 per finding (manual adds
exempt) against the frozen 8192-token context; decoding n_predict raised
1024->2048 as protocol v3. Open follow-up: proper opencv-headless
packaging instead of the vision_python.sh lib shim.

