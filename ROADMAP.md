# ren — project roadmap

Phase 3 protocol/audit (2026-09-09, at the audit checkpoint): local TXL-PBC and
Munich AML metadata audit is complete; the prospective protocol is frozen in
`artifacts/phase3_protocol_v1/README.md` with its machine-readable audit.
No Phase 3 experiment had started at that audit checkpoint. TXL-PBC has 1,260 images in its supplied
882/252/126 split; AML has no observed patient/specimen identifiers, so its
proposed split is explicitly image-level and not patient-independent. The
protocol predeclares modular detector, frozen Path Foundation classifier, and
AML binary stages plus quantitative gates. The AML blast mapping is confirmed
as MYO+MOB by Matek et al. (2019), whose source study defines these as blast
equivalents; other atypical classes remain non-blast for this task. Revisit
Astra only if split independence, augmented-data lineage, gates, or another
irreversible choice changes.

Final protocol readiness (2026-09-09): fold2 production rehearsal passed
complete 2,523-shard verification and exact reconstruction. Corrected reports
A/B are byte-identical; ignored-only accounting is 3,151 total (82 retained,
3,069 rejected), with unchanged digest. Fold3 remains locked.

Offline-first computational cytometry & cell pathology engine. Pipeline:
`Vision Engine -> Knowledge Retrieval -> Cognitive Synthesis -> Local Dashboard/UI`.
Diagnostic-assist only — every output is reviewed and signed off by a human;
nothing in this system makes an autonomous call.

This file is written to be handed to a coding agent as a project brief. It
states what's already decided and done, and what's left, with concrete
acceptance criteria per phase so a task can be scoped without re-deriving
context. Don't re-litigate the decisions in "Settled" — flag it if one turns
out to be wrong rather than silently working around it.

## Settled (don't re-decide these)

- **Inference backend**: `llama.cpp` built with Vulkan (`vulkanSupport = true`
  in the Nix flake). Never ROCm for the LLM stage — Vulkan is a deliberate
  choice for this GPU, not a placeholder.
- **ROCm** exists only in the `rocm` devShell, only for CV feature extraction
  with PyTorch (no Vulkan backend there). CPU is an acceptable fallback for
  that stage if ROCm fights the gfx1032 override.
- **Vision backbone**: frozen `google/path-foundation` (Google Health AI
  Developer Foundations). Phase 1 selected it over `paige-ai/Virchow`: Virchow
  scored modestly higher, but Path Foundation extraction was about 40× faster
  on this machine with a roughly 22× smaller checkpoint and 3.3× smaller
  embeddings. Train only a small head on top—never fine-tune the encoder.
  Virchow remains only as the recorded Phase 1 comparison reference.
  Avoid MahmoodLab models (UNI2-h, CONCH) for this project: they hard-block
  personal email domains at the access-request stage.
- **Phase 2 exception**: PanNuke instance segmentation may use a separate,
  trainable HoVer-Net-fast-style segmentation network. This does not permit
  fine-tuning or replacing frozen Path Foundation; it is a distinct stage with
  its own artifacts and evaluation gate.
- **Cognitive synthesis LLMs**: `OpenBioLLM-Llama3-8B` and `MedGemma 1.5 4B
  Q5_K_M`, GGUF, served via `llama-server`; BioMistral-7B is excluded from
  Phase 5 and the production/default serving path.
- **Architecture philosophy**: modular, UNIX-style — small pieces
  communicating over plain interfaces (HTTP for the LLM, files/arrays between
  vision and retrieval stages), not one monolithic model or codebase.
- **No monetization** — this is a learning project. Don't design around a
  commercial deployment story, and keep attribution to the underlying
  foundation models/datasets intact.

## Current state

- Flake/devShells, `direnv`, `uv`-managed Python env: working.
- `llama-server` on Vulkan confirmed working end-to-end (verified via a
  `/v1/chat/completions` round-trip with real token/sec numbers).
- Data:
  - **PanNuke** — downloaded and unzipped (`fold1/2/3`, each with
    `images.npy`/`masks.npy`/`types.npy`). Ready to use.
  - **TXL-PBC** (PBC + Raabin-WBC, YOLO-labeled blood cells) — cloned. Ready
    to use.
  - **Munich AML** (blast-cell labels) — downloaded, via a Kaggle mirror
    (sidesteps the Aspera wall entirely). `abbreviations.txt` +
    `annotations.dat`/`annotations_augmented.dat` + `data/` (images). Ready
    to use — Phase 3's blast-vs-non-blast task is no longer blocked.
- Models: OpenBioLLM-8B and MedGemma 1.5 4B Q5_K_M (GGUF) are downloaded.
  Virchow is confirmed loadable and its implementation has been audited against
  the local checkpoint. Path Foundation is load-tested locally under TensorFlow
  2.21 and produces the documented 384-dimensional float32 embeddings.
  PubMedBERT is on disk but not yet load-tested; verify before relying on it.
  Future MedCPT query/article encoders are also pinned and load-tested as
  unselected retrieval candidates. No Phase 3/4 benchmark or Phase 5 winner
  decision was started.
- Phase 1 progress: Virchow embeddings were re-extracted through the validated
  index/provenance pipeline for all folds (fold1 2656, fold2 2523, fold3 2722;
  CLS-only, 1280-dimensional). `scripts/verify_embeddings.py` exits zero with
  all folds passing source-image, label, shape, dtype, finiteness, duplicate,
  and provenance checks. The Phase 1 Virchow linear probe trained on folds 1+2
  and evaluated on fold3 at **97.06% accuracy** and **0.9562 macro-F1**, versus
  a **28.47%** training-majority baseline. Phase 1's smoke-test criterion is met;
  detailed results are in `embeddings/virchow/head_results.json`. Path
  Foundation embeddings are also extracted and verified for all three folds
  (384-dimensional, same row counts and provenance checks). Its linear probe
  reached **94.78% accuracy** and **0.9255 macro-F1**. Virchow led by 2.28 and
  3.07 percentage points respectively, but Path Foundation was selected for the
  final pipeline because it extracted at roughly 24–25 images/second versus
  Virchow's roughly 0.6 images/second and still cleared the smoke-test bar by a
  wide margin. Detailed results are in
  `embeddings/path-foundation/head_results.json`.
- Phase 2 preparation: the local PanNuke mask schema has been directly audited
  and a fail-closed, architecture-neutral validator plus tests are in place.
  Astra approved a separate HoVer-Net-fast-style segmentation architecture as
  a Phase 2 exception while Path Foundation remains frozen. The local compact
  multiscale baseline, masked targets/losses, HV maps, post-processing, and
  seeded fold1→fold2 pilot are implemented. The evaluator now reports additive
  one-to-one counts, detection metrics, binary/per-class PQ, end-to-end
  per-class metrics, macro-F1, and supports. The 8,192-step v5 trajectory
  reached detection F1 0.6045 and binary PQ 0.4587, but macro-F1 was 0.2387,
  Dead recall 0.0772, and Dead F1 0.1176. It did not pass development. Review
  then found and corrected two post-processing defects (wrong HV derivative
  axis and patch-global marker suppression). Diagnostics now separate
  predicted/perfect foreground and HV inputs plus true-mask typing. Future
  runs retain detection, macro-F1, and rare-class candidates independently;
  no scalar checkpoint is called best, and acceptance still requires every
  predeclared gate. Controlled v6/v7 experiments show fixed fold1-derived
  instance-class weights stabilize Dead typing, but neither boundary-weighted
  nor original foreground loss resolves the shared model's detection/typing
  trade-off. V7's strongest balanced checkpoint (step 3,584) reaches detection
  F1 0.5404, binary PQ 0.3949, macro-F1 0.2495, Dead recall 0.2724, Dead F1
  0.2696, and epithelial F1 0.1911. The rare-class gates pass, but overall
  detection and typing remain far below acceptance. Work is paused before the
  next architecture decision; do not extend v5-v7 unchanged. Fold3 remains
  unopened. Continue only on folds1→2. Subsequent calibration found a tuned
  foreground threshold of 0.65 improves v5 detection F1 to 0.6268 and PQ to
  0.4740 but does not close the gates. A matched gradient-isolation run raised
  calibrated detection to 0.6161/PQ 0.4691 while collapsing typing, supporting
  harmful type-gradient influence in shared features. A separate type decoder
  recovered typing (step-4,096 epithelial F1 0.3837; calibrated macro-F1
  0.2827) but calibrated detection remained 0.5604/PQ 0.4215 because the
  encoder was still shared. The next evidence-backed option is a fully
  decoupled second-stage instance classifier paired with the isolated
  segmentation model; this has not been implemented. Do not start another
  joint architecture or open fold3 without reviewing that design boundary.
  The decoupled design has now been tested: a logistic instance classifier
  trained on 62,709 fold1 nuclei from frozen segmentation features reaches
  true-instance fold2 macro-F1 0.6190, Dead F1 0.6329, and epithelial F1
  0.7978. An 8,192-step segmentation-only run selects step 7,680 and threshold
  0.60, reaching binary PQ 0.5051 but detection F1 0.6593. Combined end-to-end
  macro-F1 is 0.4193 with Dead recall 0.5244 and Dead F1 0.2919. The two-stage
  design therefore solves independent typing and passes PQ/rare-class gates,
  but still fails detection F1 and end-to-end macro-F1. Phase 2 remains in
  development; next work must improve frozen segmentation rather than revisit
  typing or extend the same compact model. Full-fold2 evaluation subsequently
  showed that conclusion was too narrow: a fold1-trained proposal-validity
  filter raises detection F1 to 0.7444 and PQ to 0.5964, passing both geometry
  gates while retaining Dead recall 0.4636 and Dead F1 0.2381. However,
  true-instance classification over all 59,369 fold2 nuclei reaches only
  macro-F1 0.5453, and filtered end-to-end macro-F1 is 0.4306. Both residual
  typing error and class-wise geometric misses therefore matter. Phase 2 is
  still blocked only on the end-to-end macro-F1 gate; fold3 stays locked.

  The 2026-09-09 nonlinear classifier comparison is complete in
  `artifacts/phase2_nonlinear_classifier_v1/`. The reusable validity selector
  now excludes zero-evaluable proposals, and FP diagnostics classify positive
  subthreshold overlap as `other_unresolved`. One seed-20260909 107->64-ReLU
  MLP selected epoch 29 prospectively on fold2. Full fold2 reached detection
  F1 0.744400, binary PQ 0.596357, end-to-end macro-F1 0.471912, Dead recall
  0.471591, and Dead F1 0.294013. Geometry counts were unchanged; the MLP was
  retained as the current best fold2 development candidate; Phase 2 remains
  unaccepted because the end-to-end gate remains unmet. Fold3 was untouched.

## Phase 2 experiment note — bounded score adjustment (2026-09-09)

The validity-filter training defect is corrected: proposals with zero
evaluable pixels are excluded, with counts recorded. FP diagnostics now
distinguish background-only, split fragments associated with matched nuclei,
merges, boundary/localization failures, and unresolved cases using substantial
overlap only. These corrections do not change matching, retention, masks, or
classifier predictions.

The frozen raw-score cache and complete experiment are in
`artifacts/phase2_score_adjustment_v1/`. Dead-only selected offset: -0.4;
focused multiclass selection: Connective +0.4, Dead -0.4. Full fold2
development macro-F1 rose from 0.430647 to 0.446085; detection F1 0.744400
and binary PQ 0.596357 stayed unchanged. Dead recall/F1 are 0.427273/0.290797.
The 0.55 gate remains unmet, so Phase 2 is not complete. Fold3 remained
untouched.

## Phase 2 appearance/context feature augmentation (2026-09-09)

Corrected proposal wording: the proposal-trained classifier improved full-fold2
end-to-end macro-F1 from **0.471912** to **0.474432**, while reducing Dead F1;
the trade-off is negligible. The duplicate proposal cache was replaced by a
reference manifest pointing to the canonical cache (SHA-256
`e764e471a1cc2c801a5d5863909067d5c7d8edd1cf6f186006942bd8ae25e165`).
Historical validity-filter contamination remains indeterminate.

The fixed 41-feature extractor is implemented in
`scripts/appearance_context_features.py`; focused tests pass. The bounded
augmented cache build was terminated by the environment before output, so no
training, selection, full-fold2 evaluation, or retention decision is claimed.
Fold3 remained untouched.

## Path Foundation context comparison (2026-09-09)

The single approved 532-column development comparison selected epoch 30 on
monitor macro-F1 0.5565232993468779 versus 0.5133344600416373 for the control.
Full fold2 macro-F1 was 0.5696812332617677; all five gates passed and geometry
was unchanged (TP/FP/FN 40,894/9,608/18,475). The 532-feature candidate passes
all frozen fold2 development gates and is retained and frozen. It is awaiting
Astra final-readiness review before the one permitted fold3 evaluation; this
does not constitute final Phase 2 completion or fold3 generalization.
Historical validity-filter provenance remains unresolved.

## Appearance/context repair completion (2026-09-09)

The corrected fixed-64-patch cache is published in
`artifacts/phase2_appearance_context_classifier_v1/cache/`; the interrupted
row-sliced history is preserved in `cache_row_sliced_superseded/`. Epoch 29 is
retained as the current development classifier candidate. Full-fold2
end-to-end macro-F1 is **0.484114**, valid true-mask macro-F1 is approximately
**0.633960**, and all five per-class F1 scores improved. Four of five gates
remain passed; the sole failure is end-to-end macro-F1 >= 0.55. Geometry is
unchanged (TP/FP/FN 40,894/9,608/18,475; detection F1 0.744400; binary PQ
0.596357). Manifests, hashes, and consolidated comparison are in the artifact
directory. Fold2 remains development data, historical validity-filter
  contamination remains indeterminate, and fold3 remained untouched.

Here, “retained” means the current best fold2 development candidate; “accepted”
means every Phase 2 gate passes. The 148-feature epoch-29 classifier is
retained, but Phase 2 is not accepted: only end-to-end macro-F1 >= 0.55 fails.

## Phase 1 — Vision-engine scaffold (do this first)

Goal: prove frozen-encoder -> embeddings -> trainable-head works at all,
using data already on disk. Not the final head architecture — a smoke test.

- Extract and compare Virchow and Path Foundation embeddings for all three
  PanNuke folds; select one pipeline backbone.
- Fit a classifier on top predicting PanNuke's own tissue-type label
  (`types.npy`, 19-way) using folds 1+2 to train, fold 3 to test.
- **Acceptance criteria**: end-to-end run completes without manual
  intervention; test accuracy exceeds the held-out accuracy of always
  predicting the training-set majority class (uniform chance alone is not a
  sufficient baseline). This proves the encoder, embedding extraction, and
  training loop are correctly wired — not that tissue classification itself
  is the end goal.

## Phase 2 — Real nucleus classification (tissue axis)

Goal: move from the Phase 1 smoke test to PanNuke's actual task — per-nucleus
instance segmentation and classification (5 classes: neoplastic,
inflammatory, connective, dead, epithelial), not just per-image tissue type.

- Use HoVer-Net fast with nuclear foreground, horizontal/vertical offset, and
  five-type-plus-background branches; keep target generation, training,
  inference, and evaluation separate. Path Foundation remains frozen and is
  optional only for later instance features.
- Develop on fold1→fold2, freeze all decisions, optionally refit folds1+2,
  then evaluate fold3 once. Reject checkpoints with unknown PanNuke fold
  exposure.
- **Acceptance criteria**: class-agnostic one-to-one IoU > 0.5 matching;
  detection precision/recall/F1, binary and per-class PQ, plus end-to-end
  five-class precision/recall/F1, macro-F1, and supports. Count unmatched
  predictions/ground truth and wrong types as errors. Predeclare nonzero Dead
  recall >=0.20 and F1 >=0.15 on fold2 development, detection F1 >=0.70,
  binary PQ >=0.50, and end-to-end macro-F1 >=0.55; matched-only typing is
  secondary.

## Phase 3 — Blood-cell axis

Goal: bring up the second data axis (blood smears) in parallel to tissue.

- Detector/localizer for individual cells in a blood smear image, using
  TXL-PBC (already YOLO-labeled).
- Classification head (frozen Virchow/path-foundation features, as in
  Phase 1) for the cell-type taxonomy TXL-PBC provides.
- Blast-vs-non-blast (AML) classification using Munich AML — now unblocked.
- **Acceptance criteria**: detector localizes individual cells with
  reasonable precision/recall on a held-out TXL-PBC split; classifier reaches
  meaningfully-above-chance accuracy on TXL-PBC's cell-type labels.

## Phase 4 — Knowledge retrieval

Goal: offline hybrid vector + symbolic retrieval over Gene Ontology and
whatever clinical text is on hand. Protocol history v1/v2 is retained as
retired regression evidence; the accepted boundary is the separately frozen
v3 protocol/evaluation in `artifacts/phase4_protocol_v3/`. MedCPT is the
selected encoder pair.

- Parse GO (OBO format) offline with `goatools`.
- Embed with the pinned local MedCPT query/article encoders.
- Index in an embedded local vector store (LanceDB or Chroma, no server
  process).
- Add a symbolic layer (SQLite or NetworkX) for exact GO-term parent/child
  traversal — vector similarity alone is weak on exact ontology/code lookups.
- **Acceptance criteria**: given a cell-type label as input (from Phase 1/2/3
  output), retrieval returns relevant GO terms/descriptors. Test this in
  isolation, before wiring it to the LLM — a retrieval bug is much easier to
  spot here than downstream in a synthesized paragraph.

Historical v1 implementation status (2026-09-10, superseded): a staged
offline harness was present in `scripts/phase4_retrieval.py` with embedded
vectors and SQLite exact-ID, alias, namespace, and edge tables. Its fixture
was explicitly incomplete and is not an acceptance result. The v1/v2
boundaries are retained only as invalid exposed regression evidence.

## Phase 5 — Cognitive synthesis

Goal: wire vision output + retrieved context into a coherent note via the
local LLM.

- `llama-server` (already working) takes structured findings + retrieved
  context, produces natural-language synthesis.
- Keep this stage decoupled from vision/retrieval — debug "did the LLM
  explain it right" separately from "did the vision model see it right."
- **Acceptance criteria**: given a known test case (a cell type + its
  correct GO terms), the synthesized note correctly reflects both — no
  hallucinated findings not present in the structured input.

## Phase 6 — Dashboard / UI

Goal: FastAPI backend + minimal frontend, human-in-the-loop review as a
first-class feature, not bolted on.

- Image upload, overlay of detected/classified cells, retrieved knowledge
  panel, synthesized note, explicit accept/edit/reject controls per finding.
- **Acceptance criteria**: a full sample (image in, reviewable output out)
  works without touching the network at any point in the request path.

## Phase 2 final-readiness hardening (2026-09-09)

The validity filter was the sole intentionally refitted component; all other
components, thresholds, and model choices remained frozen. Repaired fold2
passes all five gates. The complete final readiness manifest is hash-verified,
and tissue-stratified results are descriptive development diagnostics only.
Phase 2 is ready for its one-shot fold3 evaluation but is not finally complete;
fold3 remained wholly untouched.

The final execution-readiness corrections are complete: the transitive frozen
manifest and recursive verifiers pass, fold2 tissue diagnostics use canonical
patch-aligned labels, and the frozen aggregate metrics/digest remain unchanged.
No training or tuning occurred; final evaluation remains unrun and fold3 stayed
wholly untouched.

## Notes for whoever/whatever is executing this

- If a phase's acceptance criteria can't be met with the current approach,
  that's a signal to report back and reconsider the approach — not to lower
  the bar or silently ship something that technically runs.
- Data/model choices above were made deliberately for licensing and hardware
  reasons (see "Settled") — if one turns out to be a poor fit in practice,
  flag it rather than quietly substituting something else.

## Nonlinear classifier evidence repair (2026-09-09)

The invalid predicted-proposal true-mask calculation was superseded. The
repaired cache has 59,369 full-fold2 truth instances with 107 finite features;
valid true-mask macro-F1 is 0.607426 and epoch 29 remains selected. The MLP is
retained as the better development candidate because it improves valid
true-mask and adjusted-logistic end-to-end macro-F1 while preserving both Dead
gates; the 0.55 end-to-end gate remains unmet. Fold1 provenance is
85,276 = 46,702 evaluable positives + 36,407 evaluable negatives + 2,167
zero-evaluable proposals. Historical contamination is indeterminate because
row-level membership was not preserved; no refit was performed. Proposal-mask
feature mismatch is the next investigation. Fold3 remained untouched.

The single proposal-mask intervention is in
`artifacts/phase2_proposal_trained_classifier_v1/`: 46,702 matched fold1
proposal rows, epoch 30 selected, full-fold2 end-to-end macro-F1 0.474432,
true-mask macro-F1 0.475003, Dead recall/F1 0.475000/0.274548, and unchanged
geometry (TP/FP/FN 40,894/9,608/18,475; detection F1 0.744400; PQ 0.596357).
It does not improve the corrected control and is not retained. Fold3 remains
untouched.

The appearance/context experiment completed on 2026-09-09. The interrupted
cache attempt was superseded by repaired fixed-64-patch shards; the reusable
extractor computes patch context once while preserving the fixed 41-column
schema. The single approved 148-feature MLP selected epoch 29 prospectively.
Full-fold2 end-to-end macro-F1 was 0.484114 versus 0.471912 for the control;
geometry stayed exactly TP/FP/FN 40,894/9,608/18,475, detection F1 0.744400,
binary PQ 0.596357. The 148-feature epoch-29 classifier is retained as the
current best fold2 development candidate; the 0.55 gate remains unmet, so
Phase 2 is not accepted. Fold3 remained untouched.

Validity-filter repair (2026-09-09): regenerated pre-filter cache covers all
2,523 fold2 patches and 80,952 unfiltered proposals. Correct 111-feature
validity and separate frozen 532-feature inference pass all five gates:
detection F1 0.744241, binary PQ 0.596561, end-to-end macro-F1 0.569366, Dead
recall 0.436364, Dead F1 0.396899. Digest
9ba36ff4739b0358dd61f97bdf91d203e32acc5a94090e5d498eef788d195b80. No
training/refitting/tuning; fold3 untouched; Phase 2 not complete.

The v2 final-protocol scorer was corrected to use `IGNORE=255`. Immutable raw
predictions were scored twice with identical complete reports; all aggregate
metrics, gates, digest, and tissue reconciliation passed. No inference was
rerun, and fold3 remained wholly untouched.

Direct fold2 rehearsal v3 (2026-09-09) completed through the canonical runner:
2,523 shards froze at the expected digest, reports A/B were byte-identical, all
five gates passed, and completion plus generic readiness verification passed.
Fold3 was never accessed.

## Phase 2 one-shot fold3 evaluation (2026-09-09)

Astra-approved final evaluation completed once through the canonical runner in
`artifacts/phase2_final_protocol_fold3_final_v1`, with epoch 30 and all frozen
scientific settings unchanged. All 2,722 shards committed and froze at digest
`501b5e04f0a0bc5654215e9a6b9491f34007e7b1601c7089f78c16eae37dd01c`. The
single final report is `final_report.json` (SHA-256
`dd2b92cce4eb450ae41039aa1b95c8ed2acd18647c888e39c7b60f7ff368ce1c`).
Detection F1 is `0.7441618925936612`, binary PQ `0.597810029945104`,
end-to-end macro-F1 `0.5555736387146519`, Dead recall `0.33747609942638623`,
and Dead F1 `0.3232600732600733`; all five gates pass. Ignored-only accounting
is `3803 = 85 retained + 3718 rejected`. Complete and canonical verification
passed. This is the sole fold3 evaluation; no replacement run or rerun is
permitted absent an invariant failure and review.

Phase 3 blood completion: disclosed detector recovery reproduced frozen test
aggregates after a scorer-schema correction. Frozen Path Foundation TXL
ground-truth crops achieved test macro-F1 0.9905134073389213; AML original-only
classification achieved AUROC 0.9934856969339728, AUPRC 0.9644221463467961,
macro-F1 0.9464670830654076, sensitivity 0.898989898989899; all gates pass.
AML is image-level-only and not patient-independent; no augmented data was used.

## Phase 4 retrieval completion (2026-09-10)

The v1 and v2 Phase 4 boundaries are retained as invalid exposed regression
evidence and are not used for scientific gate certification. The v3 boundary
was separately frozen after reviewed correction, with 31 new benchmark cases,
38,245 current GO terms, pinned MedCPT query/article manifests, and no LLM.
Its quality, traversal, provenance, offline, and deterministic-reproduction
gates pass: symbolic recall@5/MRR/exact-ID are 1.0/1.0/1.0, vector
0.88/0.8159814323607427/1.0, and hybrid 1.0/0.9597701149425287/1.0.
The v3 report is `artifacts/phase4_protocol_v3/evaluation_report.json`
(SHA-256 `2d9973a334e69af11fed5a6c286ec679f5a4a16575e1be24275efc35c3c87997`).

The original v3 timing gate failed at p95 8397.242406 ms (symbolic),
7686.852608 ms (vector), and 8740.750299 ms (hybrid). The first separately
authorized latency-only recovery changed only set-backed tail membership;
it preserved all 87 rankings but failed at 354.853367 ms, 119.592767 ms,
and 490.707502 ms respectively. Its sealed report is
`artifacts/phase4_protocol_v3/recovery/latency_recovery_v1/latency_report.json`
(SHA-256 `0a70f2367ea7b8475aaba91b810748862ac1ebd9b6e1469f3ad13331fd4ecb0e`).

The final separately authorized latency-only recovery added an immutable
SQLite-derived alias/name postings cache with exact lexical verification;
it did not rerun quality evaluation or alter tokenization, scoring, routing,
traversal, encoders, corpus, benchmark, or gates. Across 29 eligible cases,
3 modes, 2 warmups, and 5 measured repeats (145 samples per mode; nearest
rank 138), p95 is 13.026316 ms symbolic, 132.087829 ms vector, and
144.417922 ms hybrid, all below the 200 ms gate. All 87 saved v3 rankings
remain byte-identical and duplicate-free. The final report is
`artifacts/phase4_protocol_v3/recovery/latency_recovery_v2/latency_report.json`
(SHA-256 `cb08a0344e3bd41cb73d6803b580b3a787d04118717fb5feb2671cfdae48571b`),
with raw samples SHA-256
`43dbe4bfbae44ea462a95e04f5e24461871566debfd25ad95c62c7b876ff5a9e`.
The final verifier passed twice with identical output; v3 and first-recovery
inventories are unchanged. Phase 4 is accepted on the combined sealed v3
quality evidence and final latency-only gate.

Snapshot reconciliation (2026-09-11, issue #2): the v3 freeze records
predate the two authorized query-layer changes, so the default snapshot
gate raises; the tree provably matches the v2-recorded final
implementation with all other files and data intact. Ratified with zero
sealed mutation via `recovery/reconciliation_v1/` plus a bound loader and
regression tests; consumers must use the bound loader.

Deferred closeout verification also passed: the canonical Phase 2 fold3
verifier reports 2,722/2,722 shards and digest
`501b5e04f0a0bc5654215e9a6b9491f34007e7b1601c7089f78c16eae37dd01c`, the
Phase 3 cross-artifact verifier passes, the scoped lint check passes, and the
full test suite passes 126 tests. No scientific evaluation was rerun during
closeout.
