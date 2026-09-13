# Phase 2 review packet — PanNuke nuclei

> Historical pre-implementation review packet. The dependency, target,
> training, and fold-3 blockers recorded below were accurate when this packet
> was written and are retained for auditability; they are superseded by the
> completed Phase 2 artifacts and one-shot fold-3 evaluation in `ROADMAP.md`.

## Decision requested

Approve **HoVer-Net fast**, a separate trainable multi-head segmentation
network with nuclear foreground, horizontal/vertical-offset, and
five-type-plus-background branches. This is an explicit Phase 2 exception to
the settled Path Foundation rule: Path Foundation remains frozen, and this
network is not a fine-tune or replacement for it. No model dependency, weight,
target extraction, or training job has been started.

## Local mask facts (direct audit)

`artifacts/pannuke_mask_audit.json` was produced from the three local
`masks.npy` arrays after schema validation.  Every fold has `float64` shape
`(N, 256, 256, 6)`.  Channels 0–4 are non-negative integer-valued instance-ID
masks for `neoplastic`, `inflammatory`, `connective`, `dead`, and `epithelial`;
channel 5 is binary background.  The IDs are sparse—not a count—and must be
counted as distinct nonzero IDs per patch/channel.

| split | patches | neo | inflammatory | connective | dead | epithelial |
|---|---:|---:|---:|---:|---:|---:|
| train fold1 | 2,656 | 26,201 | 10,820 | 16,388 | 967 | 8,842 |
| train fold2 | 2,523 | 22,731 | 10,631 | 16,756 | 884 | 8,870 |
| held-out fold3 | 2,722 | 28,471 | 10,825 | 17,441 | 1,057 | 8,860 |

Raw numeric IDs are only meaningful inside the namespace `(patch, class, ID)`;
the same number in two channels is not by itself ambiguous. The source has
real foreground overlap: 1,216 / 1,572 / 1,530 pixels affecting 471 / 465 /
475 namespaced instances in folds 1–3. It has zero foreground-on-background
pixels, but 2,359,296 / 3,801,371 / 4,325,424 pixels with neither background
nor foreground; ignored/void handling is therefore still required. Background
is not a simple complement of foreground. Target generation remains blocked
under the approved `pannuke-target-v1-ignore-ambiguous-instances` policy:
void pixels are ignored; overlap-affected and disconnected instances are
excluded wholesale from every target/loss and reported by class; valid channel-5
background is retained; fully void patches have zero valid pixels and cannot be
negative examples. Border-touching valid instances are retained for training;
the evaluator must exclude border-touching prediction/ground-truth pairs
symmetrically. Positional image–mask pairing can be
checked for shape and source hashes, but semantic alignment cannot be proven:
the local arrays do not expose source-image, slide, or patient identifiers.

## Split and evaluation contract

- Develop solely with fold1 training and fold2 validation. Freeze every model,
  target, post-processing, loss, and threshold choice before an optional
  folds1+2 refit using that fixed schedule; then evaluate fold3 exactly once.
  No random patch split, pre-training fit, threshold fitting, or augmentation
  derived sample may cross the relevant boundary.
- Use class-agnostic one-to-one matching at IoU > 0.5. Report detection
  precision/recall/F1, binary PQ, per-class PQ, and end-to-end five-class
  precision/recall/F1, macro-F1, and supports. Unmatched predictions are false
  positives; unmatched ground truth are false negatives; a wrong matched type
  is an error against both type classes. Matched-only typing is secondary.
- The Dead class is 1,851 of 121,090 training instances (1.53%). Compute
  sampling weights from fold1 only; do not collapse Dead into background.
  Predeclared fold2 development floors are detection F1 >= 0.70, binary PQ >=
  0.50, end-to-end macro-F1 >= 0.55, Dead recall >= 0.20, and Dead F1 >= 0.15.
  These are gates, not claims about expected final performance.
- Preserve the untouched source arrays and record their SHA-256 hashes, exact
  target-generation settings, model checkpoint hash and provenance, split
  membership, border/empty-image rules, and aggregation policy with each run.
  Reject any pretrained checkpoint whose PanNuke fold exposure is unknown.
  Exact duplicate-image audit found zero pairs across the three folds;
  patient/slide grouping is unavailable locally and must be disclosed rather
  than assumed independent.

## Alternatives considered

**HoVer-Net / TIAToolbox.** HoVer-Net explicitly predicts nuclear pixels,
horizontal/vertical distance maps used for separation, and nuclear type.  It
therefore solves the clustered-nucleus problem directly and has a PanNuke
training precedent.  The official implementation is MIT, but PanNuke-derived
weights inherit PanNuke's CC BY-NC-SA 4.0 restriction.  TIAToolbox is not
installed locally; current releases advertise Python 3.12 compatibility, but
the NixOS wheel/native-dependency and RX 6600 ROCm path have not been proved.
It is feasible as an evaluation dependency only after a pinned, offline cache
and CPU smoke test.  No online weight download may be part of UI inference.

**Frozen Path Foundation plus a light head.** The local SavedModel emits one
384-D embedding for a 224×224 crop.  It is excellent for the Phase 1
image-level task, but that interface does not carry pixel or instance geometry.
A linear head cannot separate touching nuclei.  Adding a spatial decoder,
proposal system, and instance post-processing would recreate the hard parts of
HoVer-Net while giving no established PanNuke separation baseline.  It is not
recommended as the primary Phase 2 route.  It remains suitable later for a
per-instance classifier only if an independent segmentation model supplies
instance crops, with folds respected.

## Compute and operational risk

- CPU-only PyTorch is confirmed; TIAToolbox is absent.  RX 6600 ROCm has not
  been tested for this workload and is optional, never used for the LLM.
- The raw masks are roughly 7.4–8.0 GiB per fold.  The audit is sequential and
  I/O-bound (about two minutes cold-cache for one fold); do not run concurrent
  Python vision processes on this machine.
- A full HoVer-Net training estimate is not credible until a pinned CPU/RX6600
  smoke run establishes batch size and step time.  Gate it behind: package
  install/load test, source-to-target validator, one-batch forward/backward,
  and a held-out evaluator test.  Then report measured epoch time before a
  managed sequential run.
- Pretrained model retrieval is a build-time acquisition concern; inference is
  offline only when weights and all package assets are local and their hashes
  are recorded.

## Acceptance gate

1. Source-mask validator, source hashes, split duplicate audit, and tests pass.
2. Target conversion is deterministic, fail-closed, provenance-bearing, has
   the approved policy above, and reports excluded pixels/instances by class.
3. Target/evaluator tests cover split, merge, missed, spurious, mistyped,
   disconnected-remnant, ignored-region, border, and empty-image cases.
4. Development uses fold1→fold2; fold3 remains untouched until the frozen
   final evaluation, and must clear the predeclared numerical floors above.
5. A pinned implementation and initialization checkpoint pass offline loading,
   one-batch forward/backward, measured memory/epoch time, then final local
   inference latency on a 256×256 patch. Outputs remain diagnostic-assist only.

## Evidence

- Local dataset README: `data/tissue/fold1/Fold 1/README.md` (six-channel
  PanNuke layout and CC BY-NC-SA 4.0 mask license).
- Local audits: `artifacts/pannuke_mask_audit.json` and
  `artifacts/pannuke_split_audit.json`.
- Official HoVer-Net repository: https://github.com/vqdang/hover_net
- TIAToolbox package compatibility/releases: https://pypi.org/project/tiatoolbox/
