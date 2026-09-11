"""The sole stateful production CLI for final Phase-2 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from appearance_context_features import PatchAppearanceContext
from evaluate_hover_checkpoint import instance_types, load_model
from evaluate_instance_suppression import (
    confidence_features,
    false_positive_categories,
    model_features,
    retain,
)
from final_phase2_evaluation import (
    GATES,
    NP_THRESHOLD,
    VALIDITY_THRESHOLD,
    array_identity,
    atomic_json,
    atomic_npz,
    load_execution_config,
    sha256,
    validate_prediction_packet,
    validate_shard_payload,
    verify_authorized_inputs,
)
from hover_postprocess import extract_instances
from nucleus_evaluation import (
    add_scores,
    empty_score,
    match_instances,
    score_instances,
    summarize_score,
)
from pannuke_target_policy import IGNORE, make_target
from train_instance_classifier import FeatureCapture, pooled_features


def load_state(run: Path) -> dict:
    return json.loads((run / "state.json").read_text())


def save_state(run: Path, state: dict) -> None:
    atomic_json(run / "state.json", state)


def supplied(args: argparse.Namespace) -> dict[str, Path]:
    return {
        name: getattr(args, name)
        for name in ("config", "segmentation", "validity", "standardizer", "classifier")
    }


def bindings(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "manifest": args.manifest,
        **supplied(args),
        "images": args.images,
        "embeddings": args.embeddings,
    }
    return {name: sha256(path) for name, path in paths.items()}


def require_role(args: argparse.Namespace, state: dict) -> None:
    role = getattr(args, "role", None)
    if role is not None and role != state["evaluation_role"]:
        raise RuntimeError("caller evaluation role differs from initialized state")


def verify_runtime(args: argparse.Namespace, state: dict, packet: bool = False):
    require_role(args, state)
    manifest = verify_authorized_inputs(args.manifest, supplied(args))
    if (
        manifest["manifest_sha256"] != state["manifest_sha256"]
        or bindings(args) != state["bindings"]
    ):
        raise RuntimeError("persisted source/configuration binding changed")
    if (
        array_identity(args.images) != state["prediction_source"]
        or array_identity(args.embeddings) != state["embedding_source"]
    ):
        raise RuntimeError("prediction source identity changed")
    return validate_prediction_packet(args.run, state) if packet else manifest


def clean_uncommitted_publications(run: Path, state: dict) -> None:
    """Remove only shard pairs which never reached the atomic state commit."""
    directory = run / "shards"
    if not directory.is_dir():
        return
    committed = set(state["completed_shards"])
    candidates = set()
    for path in (*directory.glob("shard_*.npz"), *directory.glob("shard_*.meta.json")):
        stem = path.name.removeprefix("shard_").split(".", 1)[0]
        if not stem.isdigit():
            raise RuntimeError(f"unexpected shard publication: {path.name}")
        index = int(stem)
        if index >= state["prediction_source"]["rows"]:
            raise RuntimeError(
                f"uncommitted shard outside authorized coverage: {path.name}"
            )
        candidates.add(index)
    for index in candidates - committed:
        (directory / f"shard_{index:06d}.npz").unlink(missing_ok=True)
        (directory / f"shard_{index:06d}.meta.json").unlink(missing_ok=True)


def initialize(args: argparse.Namespace) -> int:
    if args.run.exists() and any(args.run.iterdir()):
        raise RuntimeError("refuse to overwrite existing run")
    manifest = verify_authorized_inputs(args.manifest, supplied(args))
    images, embeddings = array_identity(args.images), array_identity(args.embeddings)
    if images["rows"] != embeddings["rows"]:
        raise RuntimeError("image/embedding row mismatch")
    state = {
        "schema": "phase2-final-protocol-v4",
        "run_uuid": str(uuid.uuid4()),
        "evaluation_role": args.role,
        "state": "authorized_not_started",
        "manifest_sha256": manifest["manifest_sha256"],
        "bindings": bindings(args),
        "configuration": load_execution_config(args.config),
        "prediction_source": images,
        "embedding_source": embeddings,
        "completed_shards": [],
        "shard_records": {},
        "truth_opened": False,
    }
    args.run.mkdir(parents=True, exist_ok=True)
    save_state(args.run, state)
    print(
        json.dumps(
            {"state": state["state"], "coverage": images["rows"], "role": args.role}
        )
    )
    return 0


def classifier(path: Path) -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Linear(532, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 5),
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(args: argparse.Namespace) -> int:
    state = load_state(args.run)
    verify_runtime(args, state)
    clean_uncommitted_publications(args.run, state)
    validate_prediction_packet(args.run, state)
    if state["state"] not in ("authorized_not_started", "predicting"):
        raise RuntimeError("prediction not permitted in current state")
    state["state"] = "predicting"
    save_state(args.run, state)
    images = np.load(args.images, mmap_mode="r")
    embeddings = np.load(args.embeddings, mmap_mode="r")
    model = load_model(args.segmentation)
    model.eval()
    capture = FeatureCapture(model)
    validity = np.load(args.validity)
    standardizer = np.load(args.standardizer)
    head = classifier(args.classifier)
    if (
        validity["mean"].shape != (111,)
        or validity["scale"].shape != (111,)
        or validity["coefficients"].shape[1] != 111
    ):
        raise RuntimeError("validity artifact feature width/order mismatch")
    if standardizer["mean"].shape != (532,) or standardizer["scale"].shape != (532,):
        raise RuntimeError("classifier standardizer feature width/order mismatch")
    original = len(state["completed_shards"])
    with torch.inference_mode():
        for patch in range(len(images)):
            if patch in state["completed_shards"]:
                continue
            image, embedding = images[patch], embeddings[patch]
            feature_map, np_logits, hv = model_features(model, capture, image)
            unfiltered = extract_instances(np_logits, hv, threshold=NP_THRESHOLD)
            ids, base = pooled_features(
                unfiltered, feature_map, image.astype(np.float32) / 255
            )
            validity_features = np.c_[base, confidence_features(unfiltered, np_logits)]
            if (
                validity_features.shape[1] != 111
                or not np.isfinite(validity_features).all()
            ):
                raise RuntimeError("validity feature integrity failure")
            logits = (
                (validity_features - validity["mean"]) / validity["scale"]
            ) @ validity["coefficients"].T + validity["intercept"]
            scores = np.asarray(1 / (1 + np.exp(-logits[:, 0])), np.float32)
            keep = scores >= np.float32(VALIDITY_THRESHOLD)
            indices = np.flatnonzero(keep)
            retained = retain(unfiltered, ids, keep)
            kept_ids = ids[indices].astype(np.int64)
            context = PatchAppearanceContext(image.astype(np.float32) / 255)
            appearance = (
                np.stack(
                    [context.features(retained == int(ident)) for ident in kept_ids]
                )
                if len(kept_ids)
                else np.empty((0, 41), np.float32)
            )
            features = np.c_[
                base[indices],
                appearance,
                np.repeat(embedding[None], len(indices), axis=0),
            ]
            if features.shape[1] != 532 or not np.isfinite(features).all():
                raise RuntimeError("classifier feature integrity failure")
            normalized = (
                (features - standardizer["mean"]) / standardizer["scale"]
            ).astype(np.float32)
            types = (
                np.asarray(head(torch.from_numpy(normalized)).argmax(1) + 1, np.int64)
                if len(indices)
                else np.empty(0, np.int64)
            )
            image_hash = hashlib.sha256(
                np.ascontiguousarray(image).tobytes()
            ).hexdigest()
            embedding_hash = hashlib.sha256(
                np.ascontiguousarray(embedding).tobytes()
            ).hexdigest()
            local_digest = hashlib.sha256(
                retained.astype(np.int32).tobytes()
                + np.asarray(sorted(zip(kept_ids, types)), np.int64)
                .reshape(-1, 2)
                .tobytes()
            ).hexdigest()
            payload = {
                "schema": "phase2-production-shard-v2",
                "patch": patch,
                "unfiltered_prediction": unfiltered.astype(np.int32),
                "prediction": retained.astype(np.int32),
                "all_proposal_ids": ids.astype(np.int64),
                "all_validity_scores": scores,
                "keep_decisions": keep.astype(np.uint8),
                "proposal_ids": kept_ids,
                "kept_indices": indices.astype(np.int64),
                "predicted_types": types,
                "patch_digest": local_digest,
                "source_image_sha256": image_hash,
                "embedding_sha256": embedding_hash,
            }
            validate_shard_payload(payload, patch)
            directory = args.run / "shards"
            payload_path = directory / f"shard_{patch:06d}.npz"
            meta_path = directory / f"shard_{patch:06d}.meta.json"
            atomic_npz(payload_path, payload)
            metadata = {
                "schema": "phase2-production-shard-metadata-v2",
                "patch": patch,
                "payload": payload_path.name,
                "payload_sha256": sha256(payload_path),
                "manifest_sha256": state["manifest_sha256"],
                "configuration_sha256": state["bindings"]["config"],
                "source_image_sha256": image_hash,
                "embedding_sha256": embedding_hash,
            }
            atomic_json(meta_path, metadata)
            state["completed_shards"].append(patch)
            state["shard_records"][str(patch)] = {
                "payload_sha256": sha256(payload_path),
                "metadata_sha256": sha256(meta_path),
                "source_image_sha256": image_hash,
                "embedding_sha256": embedding_hash,
            }
            save_state(args.run, state)
            if (
                args.stop_after_shards
                and len(state["completed_shards"]) - original >= args.stop_after_shards
            ):
                print(
                    json.dumps(
                        {
                            "state": "predicting",
                            "intentional_stop": True,
                            "completed": len(state["completed_shards"]),
                        }
                    )
                )
                return 2
    capture.close()
    validate_prediction_packet(args.run, state)
    print(
        json.dumps({"state": "predicting", "completed": len(state["completed_shards"])})
    )
    return 0


def freeze(args: argparse.Namespace) -> int:
    state = load_state(args.run)
    packet = verify_runtime(args, state, packet=True)
    if (
        state["state"] != "predicting"
        or packet["coverage"] != packet["expected_coverage"]
    ):
        raise RuntimeError("incomplete prediction coverage")
    if state["evaluation_role"] == "development_rehearsal":
        if not args.expected_digest:
            raise RuntimeError("development rehearsal requires expected digest")
        if packet["digest"] != args.expected_digest:
            raise RuntimeError("development rehearsal digest mismatch")
    state.update(
        state="predictions_frozen",
        prediction_digest=packet["digest"],
        frozen_coverage=packet["coverage"],
    )
    save_state(args.run, state)
    print(json.dumps({"state": state["state"], "digest": packet["digest"]}))
    return 0


def truth_bindings(args: argparse.Namespace) -> dict:
    return {"masks": array_identity(args.masks), "types": array_identity(args.types)}


def build_report(args: argparse.Namespace, state: dict, packet: dict) -> dict:
    masks = np.load(args.masks, mmap_mode="r")
    labels = np.load(args.types, mmap_mode="r")
    total = empty_score()
    groups = {}
    confusion = np.zeros((5, 5), int)
    unmatched = Counter()
    categories = Counter()
    ignored = [0, 0, 0]
    for patch, shard in enumerate(packet["shards"]):
        semantic, truth, _ = make_target(masks[patch], patch)
        prediction = np.asarray(shard["prediction"])
        unfiltered = np.asarray(shard["unfiltered_prediction"])
        ids = np.asarray(shard["all_proposal_ids"])
        keep = np.asarray(shard["keep_decisions"], bool)
        proposal_ids = np.asarray(shard["proposal_ids"])
        predicted = {
            int(i): int(t) for i, t in zip(proposal_ids, shard["predicted_types"])
        }
        actual = instance_types(truth, semantic)
        score_value = score_instances(
            prediction, truth, predicted, actual, ignored=semantic == IGNORE
        )
        add_scores(total, score_value)
        group = groups.setdefault(str(labels[patch]), [0, empty_score(), Counter()])
        group[0] += 1
        add_scores(group[1], score_value)
        group[2].update(actual.values())
        matches, false_positives, _ = match_instances(
            prediction, truth, semantic == IGNORE
        )
        for prediction_id, truth_id, _ in matches:
            confusion[actual[truth_id] - 1, predicted[prediction_id] - 1] += 1
        for prediction_id in false_positives:
            unmatched[str(predicted[prediction_id])] += 1
        categories.update(
            false_positive_categories(prediction, truth, semantic == IGNORE)
        )
        flags = np.asarray(
            [not np.any((unfiltered == ident) & (semantic != IGNORE)) for ident in ids],
            dtype=bool,
        )
        ignored[0] += int(flags.sum())
        ignored[1] += int((flags & keep).sum())
        ignored[2] += int((flags & ~keep).sum())
    metrics = summarize_score(total)
    gates = {
        "detection_f1": metrics["detection"]["f1"] >= GATES["detection_f1"],
        "binary_pq": metrics["binary_pq"] >= GATES["binary_pq"],
        "end_to_end_macro_f1": metrics["macro_f1"] >= GATES["end_to_end_macro_f1"],
        "dead_recall": metrics["classes"]["4"]["recall"] >= GATES["dead_recall"],
        "dead_f1": metrics["classes"]["4"]["f1"] >= GATES["dead_f1"],
    }
    return {
        "schema": "phase2-production-score-v2",
        "evaluation_role": state["evaluation_role"],
        "final_evaluation_data_accessed": state["evaluation_role"]
        == "final_evaluation",
        "coverage": packet["coverage"],
        "prediction_digest": packet["digest"],
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "matched_type_confusion_rows_truth_columns_prediction": confusion.tolist(),
        "unmatched_predictions_by_predicted_class": dict(sorted(unmatched.items())),
        "false_positive_categories": dict(sorted(categories.items())),
        "ignored_only_accounting": {
            "total": ignored[0],
            "retained": ignored[1],
            "rejected": ignored[2],
        },
        "tissue_stratified": {
            name: {
                "patch_count": value[0],
                "ground_truth_class_supports": dict(sorted(value[2].items())),
                "metrics": summarize_score(value[1]),
            }
            for name, value in sorted(groups.items())
        },
        "truth_bindings": state["truth_bindings"],
    }


def score(args: argparse.Namespace) -> int:
    state = load_state(args.run)
    if state["state"] not in ("predictions_frozen", "scoring", "truth_evaluated"):
        raise RuntimeError("truth scoring requires predictions_frozen")
    packet = verify_runtime(args, state, packet=True)
    if packet["coverage"] != state.get("frozen_coverage") or packet[
        "digest"
    ] != state.get("prediction_digest"):
        raise RuntimeError("frozen prediction digest or coverage mismatch")
    current_truth = truth_bindings(args)
    if "truth_bindings" not in state:
        if (
            current_truth["masks"]["rows"] != current_truth["types"]["rows"]
            or current_truth["masks"]["rows"] != state["frozen_coverage"]
        ):
            raise RuntimeError(
                "truth row count differs from frozen prediction coverage"
            )
        state.update(truth_bindings=current_truth, truth_opened=True, state="scoring")
        save_state(args.run, state)
    elif current_truth != state["truth_bindings"]:
        raise RuntimeError("truth source identity changed")
    state["state"] = "scoring"
    save_state(args.run, state)
    report = build_report(args, state, packet)
    if report["prediction_digest"] != state["prediction_digest"]:
        raise RuntimeError("prediction digest mismatch during scoring")
    if truth_bindings(args) != state["truth_bindings"]:
        raise RuntimeError("truth source identity changed during scoring")
    atomic_json(args.output, report)
    state.update(
        state="truth_evaluated",
        report_sha256=sha256(args.output),
        last_report=str(args.output.resolve()),
    )
    save_state(args.run, state)
    print(
        json.dumps(
            {
                "state": state["state"],
                "report": str(args.output),
                "sha256": state["report_sha256"],
            }
        )
    )
    return 0


def complete(args: argparse.Namespace) -> int:
    state = load_state(args.run)
    if state["state"] != "truth_evaluated":
        raise RuntimeError("completion requires truth_evaluated")
    packet = verify_runtime(args, state, packet=True)
    if (
        packet["digest"] != state["prediction_digest"]
        or packet["coverage"] != state["frozen_coverage"]
    ):
        raise RuntimeError("final frozen packet verification failed")
    if truth_bindings(args) != state.get("truth_bindings"):
        raise RuntimeError("truth source identity changed")
    report = Path(state["last_report"])
    if not report.is_file() or sha256(report) != state["report_sha256"]:
        raise RuntimeError("published report changed")
    state["state"] = "complete"
    save_state(args.run, state)
    print(json.dumps({"state": "complete"}))
    return 0


def verify(args: argparse.Namespace) -> int:
    state = load_state(args.run)
    packet = verify_runtime(args, state, packet=True)
    if state["state"] in (
        "predictions_frozen",
        "scoring",
        "truth_evaluated",
        "complete",
    ) and (
        packet["coverage"] != state.get("frozen_coverage")
        or packet["digest"] != state.get("prediction_digest")
    ):
        raise RuntimeError("frozen packet verification failed")
    if state.get("truth_opened"):
        if args.masks is None or args.types is None:
            raise RuntimeError(
                "truth-bound verification requires masks and tissue labels"
            )
        if truth_bindings(args) != state.get("truth_bindings"):
            raise RuntimeError("truth source identity changed")
        report = Path(state["last_report"])
        if not report.is_file() or sha256(report) != state["report_sha256"]:
            raise RuntimeError("published report changed")
    print(
        json.dumps(
            {
                "state": state["state"],
                "coverage": packet["coverage"],
                "expected_coverage": packet["expected_coverage"],
                "digest": packet["digest"],
            }
        )
    )
    return 0


def status(args: argparse.Namespace) -> int:
    print(json.dumps(load_state(args.run), sort_keys=True, indent=2))
    return 0


def add_runtime(parser: argparse.ArgumentParser, role_required: bool = False) -> None:
    for name in (
        "run",
        "manifest",
        "config",
        "images",
        "embeddings",
        "segmentation",
        "validity",
        "standardizer",
        "classifier",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("development_rehearsal", "final_evaluation"),
        required=role_required,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("initialize")
    add_runtime(command, True)
    command.set_defaults(function=initialize)
    command = sub.add_parser("predict")
    add_runtime(command)
    command.add_argument("--stop-after-shards", type=int, default=0)
    command.set_defaults(function=predict)
    command = sub.add_parser("freeze-predictions")
    add_runtime(command)
    command.add_argument("--expected-digest")
    command.set_defaults(function=freeze)
    command = sub.add_parser("score")
    add_runtime(command)
    command.add_argument("--masks", type=Path, required=True)
    command.add_argument("--types", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=score)
    command = sub.add_parser("complete")
    add_runtime(command)
    command.add_argument("--masks", type=Path, required=True)
    command.add_argument("--types", type=Path, required=True)
    command.set_defaults(function=complete)
    command = sub.add_parser("verify")
    add_runtime(command)
    command.add_argument("--masks", type=Path)
    command.add_argument("--types", type=Path)
    command.set_defaults(function=verify)
    command = sub.add_parser("status")
    command.add_argument("--run", type=Path, required=True)
    command.set_defaults(function=status)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
