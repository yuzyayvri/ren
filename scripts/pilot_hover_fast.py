#!/usr/bin/env python3
"""Fixed-budget fold1 training with periodic fold2 checkpoint selection."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import coverage_indices, evaluate, fold2_arrays
from hover_fast_model import IMPLEMENTATION_ID, INITIALIZATION_PROVENANCE, HoVerFast
from hover_loss import LOSS_ID, masked_hover_loss
from pannuke_target_policy import POLICY_ID, hover_offsets, make_target
from phase2_augmentation import augment_geometry

FLOORS = {
    "detection_f1": 0.70,
    "binary_pq": 0.50,
    "macro_f1": 0.55,
    "dead_recall": 0.20,
    "dead_f1": 0.15,
}


def fold1_arrays() -> tuple[np.ndarray, np.ndarray]:
    base = Path("data/tissue/fold1/Fold 1")
    return (
        np.load(base / "images/fold1/images.npy", mmap_mode="r"),
        np.load(base / "masks/fold1/masks.npy", mmap_mode="r"),
    )


def class_patch_indices(masks: np.ndarray) -> dict[int, np.ndarray]:
    """Index raw fold1 patches by present nucleus class for balanced sampling."""
    present = np.zeros((len(masks), 5), dtype=bool)
    for start in range(0, len(masks), 64):
        batch = masks[start : start + 64, ..., :5]
        present[start : start + len(batch)] = (batch > 0).any(axis=(1, 2))
    result = {class_id: np.flatnonzero(present[:, class_id - 1]) for class_id in range(1, 6)}
    if any(not len(indices) for indices in result.values()):
        raise ValueError("at least one fold1 class has no sampling candidates")
    return result


def training_batch(
    images: np.ndarray, masks: np.ndarray, index: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    semantic, instance, _ = make_target(masks[index], index)
    image_array, semantic, instance, hv_array = augment_geometry(
        images[index].astype(np.float32) / 255.0,
        semantic,
        instance,
        hover_offsets(instance, semantic),
        quarter_turns=int(rng.integers(0, 4)),
        horizontal_flip=bool(rng.integers(0, 2)),
        vertical_flip=bool(rng.integers(0, 2)),
    )
    image = torch.from_numpy(image_array.transpose(2, 0, 1))[None]
    semantic_tensor = torch.from_numpy(semantic)[None].long()
    instance_tensor = torch.from_numpy(instance)[None].long()
    hv = torch.from_numpy(hv_array)[None]
    return image, semantic_tensor, instance_tensor, hv


def gate_status(metrics: dict[str, object]) -> dict[str, bool]:
    dead = metrics["classes"]["4"]
    values = {
        "detection_f1": metrics["detection"]["f1"],
        "binary_pq": metrics["binary_pq"],
        "macro_f1": metrics["macro_f1"],
        "dead_recall": dead["recall"],
        "dead_f1": dead["f1"],
    }
    return {name: values[name] >= floor for name, floor in FLOORS.items()}


def candidate_scores(metrics: dict[str, object]) -> dict[str, float]:
    """Keep distinct development trade-offs; no scalar defines overall best."""
    dead = metrics["classes"]["4"]
    return {
        "detection": float(metrics["detection"]["f1"]),
        "macro_f1": float(metrics["macro_f1"]),
        "rare_class": float(min(dead["recall"], dead["f1"])),
    }


def checkpoint_payload(model: HoVerFast, step: int, seed: int) -> dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "implementation": IMPLEMENTATION_ID,
        "initialization": INITIALIZATION_PROVENANCE,
        "target_policy": POLICY_ID,
        "loss": LOSS_ID,
        "train_fold": 1,
        "development_fold": 2,
        "fold3_accessed": False,
        "step": step,
        "seed": seed,
        "detach_type_features": model.detach_type_features,
        "separate_type_decoder": model.separate_type_decoder,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--validate-every", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--out", type=Path, default=Path("artifacts/phase2_development_run_v2"))
    parser.add_argument("--detach-type-features", action="store_true")
    parser.add_argument("--separate-type-decoder", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.validate_every < 1:
        parser.error("steps and validate-every must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    train_images, train_masks = fold1_arrays()
    development_images, development_masks = fold2_arrays()
    candidates = class_patch_indices(train_masks)
    development_indices = coverage_indices(development_masks)
    rng = np.random.default_rng(args.seed)
    model = HoVerFast(
        detach_type_features=args.detach_type_features,
        separate_type_decoder=args.separate_type_decoder,
    )
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    losses: list[float] = []
    history: list[dict[str, object]] = []
    skipped = 0
    candidate_bests = {name: -1.0 for name in ("detection", "macro_f1", "rare_class")}
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        class_id = int(rng.integers(1, 6))
        index = int(rng.choice(candidates[class_id]))
        image, semantic, instance, hv = training_batch(
            train_images, train_masks, index, rng
        )
        model.train()
        optimizer.zero_grad()
        loss = masked_hover_loss(*model(image), semantic, instance, hv)
        if loss is None:
            skipped += 1
        else:
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        if step % args.validate_every == 0 or step == args.steps:
            development = evaluate(
                model, development_images, development_masks, development_indices
            )
            scores = candidate_scores(development)
            record = {
                "step": step,
                "candidate_scores": scores,
                "gates": gate_status(development),
                "metrics": development,
            }
            history.append(record)
            torch.save(
                checkpoint_payload(model, step, args.seed), args.out / "checkpoint.pt"
            )
            for name, score in scores.items():
                if score > candidate_bests[name]:
                    candidate_bests[name] = score
                    torch.save(
                        checkpoint_payload(model, step, args.seed),
                        args.out / f"candidate_{name}.pt",
                    )
            # Persist monitoring history at every gate so an interrupted run
            # remains auditable even before the final report is written.
            (args.out / "progress.json").write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "train_fold": 1,
                        "development_fold": 2,
                        "fold3_accessed": False,
                        "requested_steps": args.steps,
                        "last_completed_validation_step": step,
                        "history": history,
                        "candidate_steps": {
                            name: max(
                                history,
                                key=lambda item: item["candidate_scores"][name],
                            )["step"]
                            for name in candidate_bests
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"step={step} loss={np.mean(losses[-100:]):.4f} "
                f"det_f1={development['detection']['f1']:.4f} "
                f"pq={development['binary_pq']:.4f} "
                f"macro_f1={development['macro_f1']:.4f} "
                f"dead_f1={development['classes']['4']['f1']:.4f} "
                f"dead_recall={development['classes']['4']['recall']:.4f}",
                flush=True,
            )

    candidates = {
        name: max(history, key=lambda record: record["candidate_scores"][name])
        for name in candidate_bests
    }
    report = {
        "implementation": IMPLEMENTATION_ID,
        "initialization": INITIALIZATION_PROVENANCE,
        "target_policy": POLICY_ID,
        "loss": LOSS_ID,
        "seed": args.seed,
        "train_fold": 1,
        "development_fold": 2,
        "fold3_accessed": False,
        "steps": args.steps,
        "validate_every": args.validate_every,
        "class_sampling": "uniform class then uniform fold1 patch containing class",
        "detach_type_features": args.detach_type_features,
        "separate_type_decoder": args.separate_type_decoder,
        "development_indices": development_indices,
        "floors": FLOORS,
        "skipped_all_ignored_batches": skipped,
        "train_loss": losses,
        "history": history,
        "candidates": candidates,
        "accepted": [record for record in history if all(record["gates"].values())],
        "all_development_gates_pass": any(
            all(record["gates"].values()) for record in history
        ),
        "seconds": time.perf_counter() - started,
        "checkpoint": "checkpoint.pt",
        "candidate_checkpoints": {
            name: f"candidate_{name}.pt" for name in candidate_bests
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
