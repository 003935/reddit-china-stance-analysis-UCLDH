"""Exact runtime for the exploratory conditional-state ModernBERT experiment.

This module is deliberately separate from :mod:`modernbert_training`.  It reuses
the already-proven input, tokenizer and checkpoint primitives, but owns a new
output namespace and never exposes a locked-test entrypoint.  Public receipts
contain aggregate metadata only; row-level development predictions remain in
the private Modal Volume namespace.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

from reddit_china_stance.modernbert_conditional_model import (
    RELEVANCE_LABELS,
    TARGET_LABELS,
    TARGET_STATE_LABELS,
    decode_conditional_logits,
)
from reddit_china_stance.modernbert_conditional_trainer import (
    ConditionalDynamicPaddingCollator,
    ConditionalModelConfig,
    ConditionalOptimisationConfig,
    build_checkpoint_payload,
    build_length_bucket_batches,
    create_adamw,
    create_modernbert_conditional_state_model,
    encode_conditional_label,
    evaluate_epoch,
    load_checkpoint_exact,
    load_pinned_tokenizer,
    restore_training_state,
    save_checkpoint_atomic,
    seed_everything,
    tokenise_record,
    train_epoch,
)
from reddit_china_stance.modernbert_training import (
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    ModernBertContractError,
    _load_runtime_inputs,
    _read_json_object,
    _require_pyarrow,
    _semantic_label_from_teacher_row,
    canonical_sha256,
    file_sha256,
    select_checkpoint,
)
from reddit_china_stance.modernbert_training import (
    validate_private_development_predictions as validate_baseline_predictions,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation import score_semantic_labels

SCHEMA_VERSION = "1.0.0"
OUTPUT_PREFIX = Path("student-modernbert-conditional-v1")
PRIVATE_PREDICTIONS_KIND = "modernbert-conditional-private-development-predictions-v1"
TRIAL_RECEIPT_KIND = "modernbert-conditional-state-trial-receipt-v1"
GPU_TYPE = "L4"
ASHA_RUNGS = (2, 4, 8)
ASHA_COUNTS = {2: 8, 4: 4, 8: 2}
CONFIRMATION_SEEDS = (47, 61, 89)


def _experiment_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_conditional_experiment")


def _baseline_orchestration_module() -> Any:
    return importlib.import_module("reddit_china_stance.modal_modernbert")


def _require_sha256(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _finite_probability(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{where} must be a finite probability")
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def target_state_class_weights(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    """Return capped inverse-square-root weights over material target slots.

    Every material source contributes four states, including explicit ``absent``
    states.  Non-material rows are excluded because target-state loss is masked
    for them by the model contract.
    """

    counts: Counter[int] = Counter()
    slots = 0
    for row in rows:
        label = _semantic_label_from_teacher_row(row)
        encoded = encode_conditional_label(label)
        relevance_index = int(encoded["relevance_labels"])
        if RELEVANCE_LABELS[relevance_index] != "material":
            continue
        states = encoded["target_state_labels"]
        if not isinstance(states, list) or len(states) != len(TARGET_LABELS):
            raise ModernBertContractError("conditional target-state encoding drifted")
        counts.update(int(state) for state in states)
        slots += len(states)
    if slots == 0:
        raise ValueError("conditional class weights require at least one material row")
    if sum(counts.values()) != slots:
        raise ModernBertContractError("conditional target-state slots were not conserved")
    unsupported = [TARGET_STATE_LABELS[index] for index in range(6) if not counts[index]]
    if unsupported:
        raise ModernBertContractError(
            f"conditional target-state weighting lacks supported classes: {unsupported}"
        )
    raw = [math.sqrt(slots / counts[index]) for index in range(6)]
    mean = sum(raw) / len(raw)
    return tuple(min(4.0, max(0.5, value / mean)) for value in raw)


def relevance_class_weights(rows: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    """Return the baseline capped inverse-square-root relevance weights."""

    counts = Counter(str(_semantic_label_from_teacher_row(row)["relevance"]) for row in rows)
    total = len(rows)
    if total == 0:
        raise ValueError("relevance class weights require at least one row")
    raw = [math.sqrt(total / counts[label]) if counts[label] else 0.0 for label in RELEVANCE_LABELS]
    supported = [value for value in raw if value]
    mean = sum(supported) / len(supported)
    return tuple(0.0 if value == 0 else min(4.0, max(0.5, value / mean)) for value in raw)


def _registered_config(
    trial_spec: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    config = trial_spec.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("conditional trial config is missing")
    return trial_spec, config


def select_teacher_rows(
    teacher_rows: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    trial_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select exactly the frozen 5k sweep or 10k confirmation subset."""

    label_budget = trial_spec.get("label_budget")
    ladder_seed = trial_spec.get("ladder_seed")
    phase = trial_spec.get("phase")
    expected = 5_000 if phase == "asha" else 10_000 if phase == "confirmation" else None
    if expected is None:
        raise ValueError("conditional runtime supports only asha or confirmation trials")
    if label_budget != expected or type(ladder_seed) is not int:
        raise ModernBertContractError("conditional label-budget or ladder binding drifted")
    budget = "5k" if expected == 5_000 else "10k"
    fold_limits = set(range(5 if budget == "5k" else 10))
    selected_ids = {
        str(row["sample_id"])
        for row in split_manifest["rows"]
        if row["folds"].get(str(ladder_seed)) in fold_limits
    }
    expected_metadata = split_manifest["ladders"][str(ladder_seed)]["budgets"][budget]
    if len(selected_ids) != expected_metadata["row_count"]:
        raise ModernBertContractError("conditional selected subset differs from split metadata")
    # Bind the selected subset to the exact per-budget manifest recorded by the trial.
    if canonical_sha256(expected_metadata) != trial_spec.get("subset_manifest_sha256"):
        raise ModernBertContractError("conditional selected subset digest drifted")
    by_id = {str(row["sample_id"]): dict(row) for row in teacher_rows}
    if len(by_id) != len(teacher_rows) or not selected_ids <= set(by_id):
        raise ModernBertContractError("conditional teacher rows do not conserve split IDs")
    rows = [by_id[item_id] for item_id in sorted(selected_ids)]
    if len(rows) != expected:
        raise ModernBertContractError(
            f"conditional {phase} trial selected {len(rows)} rows instead of {expected}"
        )
    return rows


def build_optimisation_config(
    trial_spec: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]
) -> ConditionalOptimisationConfig:
    """Translate one frozen experiment recipe into the executable config."""

    outer, registered = _registered_config(trial_spec)
    loss_weights = registered.get("loss_weights", {"relevance": 1.0, "target_state": 1.0})
    if not isinstance(loss_weights, Mapping) or set(loss_weights) != {
        "relevance",
        "target_state",
    }:
        raise ValueError("conditional loss_weights must contain relevance and target_state")
    class_weights = registered.get("target_state_class_weights", "none")
    if class_weights not in {"none", "capped_inverse_sqrt"}:
        raise ValueError("conditional class_weights must be none or capped_inverse_sqrt")
    microbatch = int(outer.get("microbatch_size", 4))
    if microbatch <= 0 or 32 % microbatch:
        raise ValueError("microbatch_size must be a positive divisor of 32")
    weighted = class_weights == "capped_inverse_sqrt"
    return ConditionalOptimisationConfig(
        encoder_learning_rate=float(registered["encoder_learning_rate"]),
        gradient_accumulation_steps=32 // microbatch,
        effective_batch_size=32,
        gradient_checkpointing=bool(outer.get("gradient_checkpointing", True)),
        lambda_relevance=float(loss_weights["relevance"]),
        lambda_target_state=float(loss_weights["target_state"]),
        relevance_class_weights=relevance_class_weights(train_rows),
        target_state_class_weights=(target_state_class_weights(train_rows) if weighted else None),
    )


def _tokenise_rows(
    rows: Sequence[Mapping[str, Any]], *, tokenizer: Any, item_id_field: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    encoded: list[dict[str, Any]] = []
    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get(item_id_field)
        if not isinstance(item_id, str) or not item_id or item_id in reference:
            raise ValueError("conditional rows contain invalid or duplicate item IDs")
        if item_id_field == "sample_id":
            label = _semantic_label_from_teacher_row(row)
        else:
            try:
                label = json.loads(row["label_json"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("conditional development row has invalid label_json") from exc
        if not isinstance(label, dict):
            raise ValueError("conditional semantic label must be an object")
        reference[item_id] = label
        encoded.append(
            {
                "item_id": item_id,
                **tokenise_record(tokenizer, row),
                **encode_conditional_label(label),
            }
        )
    return encoded, reference


def _composite(metrics: Mapping[str, Any]) -> float:
    values = (
        metrics["relevance"]["macro_f1"],
        metrics["targets"]["core"]["micro"]["f1"],
        metrics["end_to_end_core_target_stance"]["micro"]["f1"],
    )
    if any(value is None for value in values):
        return 0.0
    return 0.25 * float(values[0]) + 0.25 * float(values[1]) + 0.50 * float(values[2])


def _binary_metrics(
    *, true_positive: int, false_positive: int, false_negative: int
) -> dict[str, float]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if f1 is None else f1,
    }


def conditional_diagnostics(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the frozen six-state and complete target-by-stance diagnostics.

    Target-state diagnostics use exactly the four slots of reference-material
    rows, matching the candidate loss scope.  All 24 target-by-stance cells are
    retained so the frozen experiment gate can apply its support threshold.
    """

    if set(reference) != set(predictions) or len(reference) != 222:
        raise ModernBertContractError(
            "conditional diagnostics require exact 222-row prediction conservation"
        )
    state_counts = {
        state: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for state in TARGET_STATE_LABELS
    }
    state_support = {state: 0 for state in TARGET_STATE_LABELS}
    state_slots = 0
    target_state_counts = {
        (target, state): {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for target in TARGET_LABELS
        for state in TARGET_STATE_LABELS
    }
    target_state_support = {
        (target, state): 0 for target in TARGET_LABELS for state in TARGET_STATE_LABELS
    }
    for item_id in sorted(reference):
        expected = reference[item_id]
        predicted = predictions[item_id]
        expected_map = {
            str(item["target"]): str(item["stance"])
            for item in expected["target_stances"]
        }
        predicted_map = {
            str(item["target"]): str(item["stance"])
            for item in predicted["target_stances"]
        }
        if expected["relevance"] == "material":
            for target in TARGET_LABELS:
                expected_state = expected_map.get(target, "absent")
                predicted_state = predicted_map.get(target, "absent")
                state_slots += 1
                state_support[expected_state] += 1
                target_state_support[(target, expected_state)] += 1
                for state in TARGET_STATE_LABELS:
                    if expected_state == state and predicted_state == state:
                        state_counts[state]["true_positive"] += 1
                    elif expected_state != state and predicted_state == state:
                        state_counts[state]["false_positive"] += 1
                    elif expected_state == state and predicted_state != state:
                        state_counts[state]["false_negative"] += 1
                    if expected_state == state and predicted_state == state:
                        target_state_counts[(target, state)]["true_positive"] += 1
                    elif expected_state != state and predicted_state == state:
                        target_state_counts[(target, state)]["false_positive"] += 1
                    elif expected_state == state and predicted_state != state:
                        target_state_counts[(target, state)]["false_negative"] += 1
    if not state_slots or sum(state_support.values()) != state_slots:
        raise ModernBertContractError("conditional state diagnostics did not conserve slots")
    per_state = {
        state: {
            "reference_support": state_support[state],
            **_binary_metrics(**state_counts[state]),
        }
        for state in TARGET_STATE_LABELS
    }
    target_stance = {
        target: {
            state: {
                "reference_support": target_state_support[(target, state)],
                **_binary_metrics(**target_state_counts[(target, state)]),
            }
            for state in TARGET_STATE_LABELS
        }
        for target in TARGET_LABELS
    }
    result = {
        "stance_diagnostics": per_state,
        "target_stance_diagnostics": target_stance,
    }
    assert_metadata_only(result, where="conditional confirmation diagnostics")
    return result


def validate_private_development_predictions(value: Any, *, expected_rows: int) -> dict[str, Any]:
    """Validate private logits without returning or publishing row contents."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "kind",
        "row_count",
        "rows",
    }:
        raise ValueError("conditional private prediction schema drifted")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != PRIVATE_PREDICTIONS_KIND
        or type(expected_rows) is not int
        or expected_rows <= 0
        or value["row_count"] != expected_rows
    ):
        raise ValueError("conditional private prediction binding drifted")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError("conditional private prediction row count drifted")
    seen: set[str] = set()
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "source_sample_id",
            "relevance_logits",
            "target_state_logits",
            "decoded_label",
        }:
            raise ValueError("conditional private prediction row schema drifted")
        item_id = row["source_sample_id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("conditional private prediction IDs must be unique")
        seen.add(item_id)

        def vector(raw: Any, size: int, where: str) -> list[float]:
            if not isinstance(raw, list) or len(raw) != size:
                raise ValueError(f"{where} must contain exactly {size} logits")
            result: list[float] = []
            for item in raw:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError(f"{where} must contain finite numbers")
                number = float(item)
                if not math.isfinite(number):
                    raise ValueError(f"{where} must contain finite numbers")
                result.append(number)
            return result

        relevance = vector(row["relevance_logits"], len(RELEVANCE_LABELS), "relevance_logits")
        target_rows = row["target_state_logits"]
        if not isinstance(target_rows, list) or len(target_rows) != len(TARGET_LABELS):
            raise ValueError("target_state_logits must contain four target rows")
        states = [
            vector(item, len(TARGET_STATE_LABELS), "target_state_logits") for item in target_rows
        ]
        # Reuse the encoder as a strict semantic-label validator.
        encode_conditional_label(row["decoded_label"])
        clean_rows.append(
            {
                "source_sample_id": item_id,
                "relevance_logits": relevance,
                "target_state_logits": states,
                "decoded_label": deepcopy(row["decoded_label"]),
            }
        )
    if [row["source_sample_id"] for row in clean_rows] != sorted(seen):
        raise ValueError("conditional private predictions must be sorted by opaque ID")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_PREDICTIONS_KIND,
        "row_count": expected_rows,
        "rows": clean_rows,
    }


def _bind_private_predictions(
    *, experiment: Mapping[str, Any], trial_spec: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind trainer logits to the experiment and deterministic decoder contract."""

    raw = validate_private_development_predictions(value, expected_rows=222)
    contract = _experiment_module()
    rows: list[dict[str, Any]] = []
    for row in raw["rows"]:
        decoded, forced = contract.decode_conditional_logits(
            row["relevance_logits"], row["target_state_logits"]
        )
        if decoded != row["decoded_label"]:
            raise ModernBertContractError("trainer and experiment conditional decoders disagree")
        rows.append(
            {
                **row,
                "forced_target_selection": forced,
            }
        )
    return contract.build_private_development_predictions(
        experiment,
        trial_spec,
        rows=rows,
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    temporary = path.with_suffix(path.suffix + ".new")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_epoch_zero_resume_marker(job: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a claimed trial that has not yet produced its first checkpoint."""

    clean = _validate_job(job)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-conditional-epoch-zero-resume-v1",
        "experiment_run_id": clean["experiment_run_id"],
        "trial_id": clean["trial_spec"]["trial_id"],
        "trial_spec_sha256": clean["trial_spec_sha256"],
        "completed_epochs": 0,
        "state": "claimed_no_checkpoint",
    }


def _validate_epoch_zero_resume_marker(
    value: Mapping[str, Any], *, job: Mapping[str, Any]
) -> dict[str, Any]:
    expected = build_epoch_zero_resume_marker(job)
    if dict(value) != expected:
        raise ModernBertContractError("conditional epoch-zero resume marker binding drifted")
    return expected


def _restore_continuation(
    *,
    experiment_contract: Mapping[str, Any],
    continuation: Mapping[str, Any],
    target_epoch: int,
    run_root: Path,
    optimisation_sha256: str,
    trial_config_sha256: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> tuple[int, int, int]:
    """Restore an exact 2->4 or 4->8 cumulative-ASHA checkpoint."""

    expected_source_epoch = {4: 2, 8: 4}.get(target_epoch)
    if expected_source_epoch is None:
        raise ValueError("continuation target epoch must be 4 or 8")
    experiment = _experiment_module()
    clean_resume = dict(continuation)
    if clean_resume.get("source_config_sha256") != trial_config_sha256:
        raise ModernBertContractError("conditional continuation configuration drifted")
    if clean_resume["completed_epochs"] != expected_source_epoch:
        raise ModernBertContractError("conditional continuation came from the wrong rung")
    source_root = run_root / "phase=asha" / f"trial={clean_resume['source_trial_id']}"
    metadata_descriptor = clean_resume["checkpoint_metadata_artifact"]
    metadata_path = source_root / metadata_descriptor["relative_path"]
    if (
        not metadata_path.is_file()
        or metadata_path.stat().st_size != metadata_descriptor["bytes"]
        or file_sha256(metadata_path) != metadata_descriptor["sha256"]
    ):
        raise ModernBertContractError("conditional continuation metadata is missing or corrupt")
    metadata = _read_json_object(metadata_path, where="conditional checkpoint metadata")
    if metadata.get("checkpoint_id") != clean_resume["checkpoint_id"]:
        raise ModernBertContractError("conditional continuation checkpoint ID drifted")
    source_spec = _read_json_object(
        source_root / "trial-spec.json", where="conditional source trial spec"
    )
    experiment.validate_resume_binding(
        # The experiment is embedded in the checkpoint/trial binding and is
        # identical to the active contract by experiment_run_id.
        experiment_contract,
        source_spec,
        metadata,
        clean_resume,
    )
    checkpoint_descriptor = metadata.get("artifacts", {}).get("model_state")
    if not isinstance(checkpoint_descriptor, Mapping):
        raise ModernBertContractError("conditional checkpoint metadata lacks model state")
    checkpoint = source_root / str(checkpoint_descriptor["relative_path"])
    expected_file = _require_sha256(
        checkpoint_descriptor.get("sha256"), where="checkpoint.model_state.sha256"
    )
    import torch

    preview = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_binding = _require_sha256(
        preview.get("binding_sha256") if isinstance(preview, Mapping) else None,
        where="checkpoint.binding_sha256",
    )
    payload = load_checkpoint_exact(
        checkpoint,
        expected_file_sha256=expected_file,
        expected_binding_sha256=expected_binding,
        map_location="cuda",
    )
    binding = payload["binding"]
    if binding.get("epoch") != expected_source_epoch:
        raise ModernBertContractError("conditional raw checkpoint came from the wrong rung")
    if binding.get("optimisation_config_sha256") != optimisation_sha256:
        raise ModernBertContractError("conditional continuation optimisation config drifted")
    restore_training_state(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=True,
    )
    return int(binding["epoch"]), int(binding["global_step"]), int(binding["optimizer_step"])


def _execute_trial_runtime(
    *,
    job: Mapping[str, Any],
    inputs: Mapping[str, Any],
    staging_root: Path,
    run_root: Path,
    resume: bool,
) -> dict[str, Any]:
    """Execute one pinned L4 trial and return private in-container results."""

    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("conditional ModernBERT training requires a CUDA GPU")
    spec = job["trial_spec"]
    train_rows = select_teacher_rows(
        inputs["teacher_rows"], inputs["split_manifest"], trial_spec=spec
    )
    _, registered = _registered_config(spec)
    seed = int(spec["optimiser_seed"])
    seed_everything(seed)
    optimisation = build_optimisation_config(spec, train_rows)
    model_config = ConditionalModelConfig()
    tokenizer = load_pinned_tokenizer(model_config)
    train_encoded, _ = _tokenise_rows(train_rows, tokenizer=tokenizer, item_id_field="sample_id")
    development_encoded, development_reference = _tokenise_rows(
        inputs["development_rows"],
        tokenizer=tokenizer,
        item_id_field="source_sample_id",
    )
    if len(development_reference) != 222:
        raise ModernBertContractError(
            "conditional evaluation requires exactly 222 development rows"
        )
    collator = ConditionalDynamicPaddingCollator(tokenizer)
    microbatch = 32 // optimisation.gradient_accumulation_steps
    development_batches = build_length_bucket_batches(
        [len(row["input_ids"]) for row in development_encoded],
        batch_size=microbatch,
        seed=0,
        epoch=0,
    )
    development_loader = DataLoader(
        development_encoded,
        batch_sampler=development_batches,
        collate_fn=collator,
        num_workers=0,
    )
    model = create_modernbert_conditional_state_model(
        model_config=model_config,
        optimisation_config=optimisation,
    ).to("cuda")
    optimizer = create_adamw(model, optimisation)
    target_epoch = int(spec["target_epochs"])
    if spec["phase"] == "asha" and target_epoch not in ASHA_RUNGS:
        raise ValueError("conditional sweep target epoch must be an ASHA rung")
    if spec["phase"] == "confirmation" and target_epoch != 8:
        raise ValueError("conditional confirmation must train to at most eight epochs")
    batches_per_epoch = math.ceil(len(train_encoded) / microbatch)
    updates_per_epoch = math.ceil(batches_per_epoch / optimisation.gradient_accumulation_steps)
    # Cumulative ASHA uses one fixed eight-epoch schedule.  A rung-2 checkpoint
    # must retain non-zero learning rate when continued to rung 4 and 8.
    total_updates = 8 * updates_per_epoch
    warmup_updates = int(total_updates * optimisation.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        remaining = max(1, total_updates - warmup_updates)
        return max(0.0, (total_updates - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    checkpoint_dir = staging_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = global_step = optimizer_step = 0
    history: list[dict[str, Any]] = []
    if resume:
        marker = _read_json_object(staging_root / "resume.json", where="conditional resume marker")
        if marker.get("state") == "claimed_no_checkpoint":
            _validate_epoch_zero_resume_marker(marker, job=job)
        else:
            checkpoint = marker.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise ModernBertContractError("conditional resume marker lacks checkpoint binding")
            payload = load_checkpoint_exact(
                staging_root / str(checkpoint["relative_path"]),
                expected_file_sha256=str(checkpoint["sha256"]),
                expected_binding_sha256=str(checkpoint["binding_sha256"]),
                map_location="cuda",
            )
            if payload["binding"].get("trial_manifest_sha256") != job["trial_spec_sha256"]:
                raise ModernBertContractError(
                    "conditional resume checkpoint belongs to another trial"
                )
            restore_training_state(
                payload, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True
            )
            start_epoch = int(payload["binding"]["epoch"])
            global_step = int(payload["binding"]["global_step"])
            optimizer_step = int(payload["binding"]["optimizer_step"])
            history_value = _read_json_object(staging_root / "history.json", where="history")
            history = list(history_value.get("epochs", []))
            if len(history) != start_epoch:
                raise ModernBertContractError(
                    "conditional resume history and checkpoint disagree"
                )
    if start_epoch == 0 and spec.get("resume_binding") is not None:
        continuation = spec["resume_binding"]
        if not isinstance(continuation, Mapping):
            raise ValueError("conditional continuation must be an object")
        start_epoch, global_step, optimizer_step = _restore_continuation(
            experiment_contract=job["experiment_contract"],
            continuation=continuation,
            target_epoch=target_epoch,
            run_root=run_root,
            optimisation_sha256=optimisation.digest(),
            trial_config_sha256=str(registered["config_sha256"]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    train_lengths = [len(row["input_ids"]) for row in train_encoded]
    for epoch in range(start_epoch + 1, target_epoch + 1):
        batches = build_length_bucket_batches(
            train_lengths, batch_size=microbatch, seed=seed, epoch=epoch
        )
        loader = DataLoader(
            train_encoded, batch_sampler=batches, collate_fn=collator, num_workers=0
        )
        train_result = train_epoch(
            model,
            loader,
            optimizer,
            config=optimisation,
            device="cuda",
            scheduler=scheduler,
        )
        evaluation = evaluate_epoch(
            model,
            development_loader,
            device="cuda",
            reference=development_reference,
            config=optimisation,
        )
        evaluation["composite"] = _composite(evaluation["metrics"])
        evaluation.pop("prediction_payload", None)
        global_step += train_result["batches"]
        optimizer_step += train_result["optimizer_steps"]
        history.append({"epoch": epoch, "train": train_result, "development": evaluation})
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            optimisation_config=optimisation,
            binding={
                "dataset_sha256": job["experiment_contract"]["bindings"]["dataset_sha256"],
                "split_manifest_sha256": job["experiment_contract"]["bindings"][
                    "split_manifest_sha256"
                ],
                "trial_manifest_sha256": job["trial_spec_sha256"],
                "code_sha256": job["experiment_contract"]["bindings"]["code_sha256"],
            },
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            seed=seed,
        )
        checkpoint_path = checkpoint_dir / f"epoch-{epoch:02d}.pt"
        descriptor = save_checkpoint_atomic(payload, checkpoint_path)
        _write_json_atomic(staging_root / "history.json", {"epochs": history})
        resume_value = {
            "schema_version": SCHEMA_VERSION,
            "experiment_run_id": job["experiment_run_id"],
            "trial_id": spec["trial_id"],
            "trial_spec_sha256": job["trial_spec_sha256"],
            "checkpoint": {
                "relative_path": str(checkpoint_path.relative_to(staging_root)),
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "binding_sha256": payload["binding_sha256"],
            },
        }
        # Resume markers are mutable pointers, but each update is atomic.
        _write_json_atomic(staging_root / "resume.json", resume_value)
        if spec["phase"] == "confirmation" and epoch >= 2:
            selection = select_checkpoint(
                [
                    {
                        "epoch": item["epoch"],
                        "composite": item["development"]["composite"],
                        "checkpoint_sha256": file_sha256(
                            checkpoint_dir / f"epoch-{item['epoch']:02d}.pt"
                        ),
                    }
                    for item in history
                ]
            )
            if selection["stopped_epoch"] is not None:
                break
    if not history:
        raise ModernBertContractError("conditional trial produced no development history")
    if spec["phase"] == "asha":
        selected_epoch = int(history[-1]["epoch"])
    else:
        selected_epoch = int(
            select_checkpoint(
                [
                    {
                        "epoch": item["epoch"],
                        "composite": item["development"]["composite"],
                        "checkpoint_sha256": file_sha256(
                            checkpoint_dir / f"epoch-{item['epoch']:02d}.pt"
                        ),
                    }
                    for item in history
                ]
            )["selected_epoch"]
        )
    selected_checkpoint = checkpoint_dir / f"epoch-{selected_epoch:02d}.pt"
    preview = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    selected_payload = load_checkpoint_exact(
        selected_checkpoint,
        expected_file_sha256=file_sha256(selected_checkpoint),
        expected_binding_sha256=str(preview["binding_sha256"]),
        map_location="cuda",
    )
    restore_training_state(
        selected_payload, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=False
    )
    selected_evaluation = evaluate_epoch(
        model,
        development_loader,
        device="cuda",
        reference=development_reference,
        config=optimisation,
    )
    selected_evaluation["composite"] = _composite(selected_evaluation["metrics"])
    history_evaluation = next(
        item["development"] for item in history if item["epoch"] == selected_epoch
    )
    if selected_evaluation["composite"] != history_evaluation["composite"]:
        raise ModernBertContractError("conditional selected-checkpoint score is not reproducible")
    predictions = validate_private_development_predictions(
        selected_evaluation.pop("prediction_payload"), expected_rows=222
    )
    elapsed = time.monotonic() - started
    return {
        "history": history,
        "training_rows": len(train_rows),
        "development_rows": 222,
        "truncated_training_rows": sum(row["truncated_tokens"] > 0 for row in train_encoded),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "selected_checkpoint_relative_path": str(selected_checkpoint.relative_to(staging_root)),
        "selected_epoch": selected_epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "selected_checkpoint_global_step": int(selected_payload["binding"]["global_step"]),
        "selected_checkpoint_optimizer_step": int(selected_payload["binding"]["optimizer_step"]),
        "development_predictions": predictions,
        "wall_seconds": elapsed,
        "gpu_seconds": elapsed,
    }


def _public_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    selected_epoch = int(result["selected_epoch"])
    selected = next(
        item for item in result["history"] if int(item.get("epoch", -1)) == selected_epoch
    )
    semantic = selected["development"]["metrics"]
    value = {
        "training_rows": int(result["training_rows"]),
        "development_rows": int(result["development_rows"]),
        "truncated_rows": int(result["truncated_training_rows"]),
        "completed_epochs": int(result["history"][-1]["epoch"]),
        "selected_epoch": selected_epoch,
        "composite": _finite_probability(selected["development"]["composite"], where="composite"),
        "invalid_outputs": int(semantic["invalid_outputs"]),
        "relevance_macro_f1": semantic["relevance"]["macro_f1"],
        "material_recall": semantic["relevance"]["material_recall"],
        "core_target_micro_f1": semantic["targets"]["core"]["micro"]["f1"],
        "core_target_stance_tuple_micro_f1": semantic["end_to_end_core_target_stance"]["micro"][
            "f1"
        ],
        "forced_target_selections": semantic["decoding"]["forced_target_selections"],
    }
    assert_metadata_only(value, where="conditional public trial metrics")
    return value


def _artifact_inventory(root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "checkpoint_metadata": root / "checkpoint-metadata.json",
        "metrics": root / "metrics.json",
        "development_predictions": root / "development-predictions.json",
        "trial_spec": root / "trial-spec.json",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ModernBertContractError(f"conditional artifact is missing: {name}")
        artifacts[name] = {
            "relative_path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return artifacts


def _load_inputs(job: Mapping[str, Any], *, volume_root: Path) -> dict[str, Any]:
    """Use the proven private loader through an exact binding-name adapter."""

    bindings = job["experiment_contract"]["bindings"]
    adapted = deepcopy(dict(job))
    adapted["experiment_contract"] = {
        "bindings": {
            "dataset_parquet_sha256": bindings["dataset_sha256"],
            "split_manifest_sha256": bindings["split_manifest_sha256"],
        }
    }
    inputs = _load_runtime_inputs(adapted, volume_root=volume_root, include_development=True)
    reference_binding = {
        "development_proxy_id": DEVELOPMENT_PROXY_ID,
        "split": "development",
        "labels": [
            {
                "source_sample_id": row["source_sample_id"],
                "label": json.loads(row["label_json"]),
            }
            for row in sorted(inputs["development_rows"], key=lambda item: item["source_sample_id"])
        ],
    }
    if canonical_sha256(reference_binding) != bindings["development_reference_sha256"]:
        raise ModernBertContractError("conditional development-reference binding drifted")
    return inputs


def _load_development_reference(
    *, volume_root: Path, development_reference_sha256: str
) -> dict[str, dict[str, Any]]:
    """Materialise only the exact 222 development labels behind the closeout."""

    _, parquet = _require_pyarrow()
    path = volume_root / "student-modernbert-v1/inputs/development-proxy.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = parquet.read_table(
        path,
        columns=["source_sample_id", "label_json"],
        filters=[("split", "=", "development")],
    ).to_pylist()
    if len(rows) != 222:
        raise ModernBertContractError("confirmation closeout requires exactly 222 reference rows")
    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("source_sample_id")
        if not isinstance(item_id, str) or not item_id or item_id in reference:
            raise ModernBertContractError("development reference IDs are invalid or duplicated")
        try:
            label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("development reference contains invalid label JSON") from exc
        if not isinstance(label, dict):
            raise ValueError("development reference label must be an object")
        encode_conditional_label(label)
        reference[item_id] = label
    binding = {
        "development_proxy_id": DEVELOPMENT_PROXY_ID,
        "split": "development",
        "labels": [
            {"source_sample_id": item_id, "label": reference[item_id]}
            for item_id in sorted(reference)
        ],
    }
    if canonical_sha256(binding) != development_reference_sha256:
        raise ModernBertContractError("development reference content address drifted")
    return reference


def _verify_metric_receipt(
    *, receipt_metrics: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    expected = {
        "relevance_macro_f1": recomputed["relevance"]["macro_f1"],
        "material_recall": recomputed["relevance"]["material_recall"],
        "core_target_micro_f1": recomputed["targets"]["core"]["micro"]["f1"],
        "core_target_stance_tuple_micro_f1": recomputed[
            "end_to_end_core_target_stance"
        ]["micro"]["f1"],
        "invalid_outputs": recomputed["invalid_outputs"],
    }
    for name, value in expected.items():
        observed = receipt_metrics.get(name)
        if isinstance(value, float):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isclose(float(observed), value, abs_tol=1e-12)
            ):
                raise ModernBertContractError(f"recomputed confirmation metric drifted: {name}")
        elif observed != value:
            raise ModernBertContractError(f"recomputed confirmation metric drifted: {name}")


def _bound_gate_result(
    *,
    source_experiment_run_id: str,
    source_phase_run_id: str,
    config_sha256: str,
    trial_id: str,
    receipt_id: str,
    prediction_sha256: str,
    ladder_seed: int,
    optimiser_seed: int,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    metric_payload = {
        "core_target_stance_tuple_micro_f1": metrics[
            "end_to_end_core_target_stance"
        ]["micro"]["f1"],
        "material_recall": metrics["relevance"]["material_recall"],
        "core_target_f1": {
            target: metrics["targets"]["per_class"][target]["f1"]
            for target in TARGET_LABELS[:3]
        },
        "core_target_reference_support": {
            target: metrics["support"]["reference_targets"][target]
            for target in TARGET_LABELS[:3]
        },
        "stance_diagnostics": diagnostics["stance_diagnostics"],
        "target_stance_diagnostics": diagnostics["target_stance_diagnostics"],
        "invalid_outputs": metrics["invalid_outputs"],
    }
    return {
        "source_experiment_run_id": source_experiment_run_id,
        "source_phase_run_id": source_phase_run_id,
        "config_sha256": config_sha256,
        "trial_id": trial_id,
        "receipt_id": receipt_id,
        "prediction_sha256": prediction_sha256,
        "condition": {
            "ladder_seed": ladder_seed,
            "optimiser_seed": optimiser_seed,
        },
        "metric_payload": metric_payload,
        "metric_payload_sha256": canonical_sha256(metric_payload),
    }


def _baseline_prediction_frame(
    *,
    baseline_manifest: Mapping[str, Any],
    volume_root: Path,
    reference: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Validate and privately rescore the three matched 10k baseline trials."""

    baseline = _baseline_orchestration_module()
    clean_manifest = baseline.validate_run_manifest(baseline_manifest)
    if clean_manifest.get("phase") != "confirmatory" or len(clean_manifest["trials"]) != 9:
        raise ModernBertContractError("baseline closeout source must be exact 3x3 confirmatory")
    trials = [
        trial
        for trial in clean_manifest["trials"]
        if trial["config"].get("label_budget") == 10_000
    ]
    expected_pairs = {"101:47", "202:61", "303:89"}
    observed_pairs = {
        f"{trial['config']['ladder_seed']}:{trial['config']['optimiser_seed']}"
        for trial in trials
    }
    if len(trials) != 3 or observed_pairs != expected_pairs:
        raise ModernBertContractError("baseline 10k paired-condition inventory drifted")
    from reddit_china_stance.modernbert_trainer import decode_predictions

    results: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    receipt_ids: list[str] = []
    run_root = (
        volume_root
        / "student-modernbert-v1"
        / f"run={clean_manifest['experiment_run_id']}"
    )
    for trial in sorted(
        trials,
        key=lambda item: (
            item["config"]["ladder_seed"],
            item["config"]["optimiser_seed"],
        ),
    ):
        pair_id = f"{trial['config']['ladder_seed']}:{trial['config']['optimiser_seed']}"
        trial_root = run_root / "phase=confirmatory" / f"trial={trial['trial_id']}"
        receipt = _read_json_object(trial_root / "receipt.json", where="baseline receipt")
        job = {
            "experiment_run_id": clean_manifest["experiment_run_id"],
            "phase_run_id": clean_manifest["phase_run_id"],
            "experiment_contract": clean_manifest["experiment_contract"],
            "trial_spec": trial,
            "trial_spec_sha256": canonical_sha256(trial),
        }
        baseline.validate_trial_receipt(trial_root=trial_root, receipt=receipt, job=job)
        receipt_id = _require_sha256(receipt.get("receipt_id"), where="baseline receipt ID")
        if canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        ) != receipt_id:
            raise ModernBertContractError("baseline receipt content address drifted")
        descriptor = receipt.get("artifacts", {}).get("development_predictions")
        if not isinstance(descriptor, Mapping):
            raise ModernBertContractError("baseline receipt lacks private predictions")
        payload = _read_json_object(
            trial_root / str(descriptor["relative_path"]), where="baseline predictions"
        )
        clean_predictions = validate_baseline_predictions(payload, expected_rows=222)
        if clean_predictions["decoder_target_threshold"] != 0.30:
            raise ModernBertContractError("matched baseline decoder threshold drifted")
        rows = clean_predictions["rows"]
        item_ids = [row["source_sample_id"] for row in rows]
        if set(item_ids) != set(reference) or len(item_ids) != len(set(item_ids)):
            raise ModernBertContractError("baseline predictions do not conserve reference IDs")
        decoded = decode_predictions(
            [row["relevance_logits"] for row in rows],
            [row["target_logits"] for row in rows],
            [row["stance_logits"] for row in rows],
            target_threshold=0.30,
        )
        predictions = dict(zip(item_ids, decoded.labels, strict=True))
        if tuple(row["decoded_label"] for row in rows) != decoded.labels:
            raise ModernBertContractError("stored baseline labels differ from bound logits")
        metrics = score_semantic_labels(reference, predictions)
        _verify_metric_receipt(receipt_metrics=receipt["aggregate_metrics"], recomputed=metrics)
        row_diagnostics = conditional_diagnostics(reference, predictions)
        results.append(
            _bound_gate_result(
                source_experiment_run_id=clean_manifest["experiment_run_id"],
                source_phase_run_id=clean_manifest["phase_run_id"],
                config_sha256=trial["config"]["registered_config"]["config_sha256"],
                trial_id=trial["trial_id"],
                receipt_id=receipt_id,
                prediction_sha256=descriptor["sha256"],
                ladder_seed=trial["config"]["ladder_seed"],
                optimiser_seed=trial["config"]["optimiser_seed"],
                metrics=metrics,
                diagnostics=row_diagnostics,
            )
        )
        diagnostics[pair_id] = row_diagnostics
        receipt_ids.append(receipt_id)
    return results, diagnostics, receipt_ids


def _candidate_prediction_frame(
    *,
    confirmation_manifest: Mapping[str, Any],
    volume_root: Path,
    reference: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[str]]:
    """Validate and privately rescore both candidate recipes across three pairs."""

    contract = _experiment_module()
    clean_manifest = contract.validate_run_manifest(confirmation_manifest)
    if clean_manifest.get("phase") != "confirmation" or len(clean_manifest["trials"]) != 6:
        raise ModernBertContractError("candidate closeout requires exact 2x3 confirmation")
    expected_pairs = {"101:47", "202:61", "303:89"}
    by_config: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, Any] = {}
    receipt_ids: list[str] = []
    for trial in clean_manifest["trials"]:
        config_id = str(trial["config"]["config_sha256"])
        pair_id = f"{trial['ladder_seed']}:{trial['optimiser_seed']}"
        if pair_id not in expected_pairs:
            raise ModernBertContractError("candidate paired condition drifted")
        trial_root = contract.trial_output_root(
            volume_root, clean_manifest["experiment_contract"], trial
        )
        receipt = _read_json_object(trial_root / "receipt.json", where="candidate receipt")
        validate_trial_artifacts(
            trial_root,
            receipt,
            experiment=clean_manifest["experiment_contract"],
            trial_spec=trial,
        )
        descriptor = receipt["artifacts"]["development_predictions"]
        payload = _read_json_object(
            trial_root / descriptor["relative_path"], where="candidate predictions"
        )
        clean_predictions = contract.validate_private_development_predictions(
            payload, clean_manifest["experiment_contract"], trial
        )
        if (
            clean_predictions["development_reference_sha256"]
            != clean_manifest["experiment_contract"]["bindings"][
                "development_reference_sha256"
            ]
        ):
            raise ModernBertContractError("candidate prediction reference binding drifted")
        rows = clean_predictions["rows"]
        item_ids = [row["source_sample_id"] for row in rows]
        if set(item_ids) != set(reference) or len(item_ids) != len(set(item_ids)):
            raise ModernBertContractError("candidate predictions do not conserve reference IDs")
        decoded = decode_conditional_logits(
            [row["relevance_logits"] for row in rows],
            [row["target_state_logits"] for row in rows],
        )
        if tuple(row["decoded_label"] for row in rows) != decoded.predictions:
            raise ModernBertContractError("stored candidate labels differ from bound logits")
        predictions = dict(zip(item_ids, decoded.predictions, strict=True))
        metrics = score_semantic_labels(reference, predictions)
        _verify_metric_receipt(receipt_metrics=receipt["aggregate_metrics"], recomputed=metrics)
        row_diagnostics = conditional_diagnostics(reference, predictions)
        by_config.setdefault(config_id, []).append(
            _bound_gate_result(
                source_experiment_run_id=clean_manifest["experiment_run_id"],
                source_phase_run_id=clean_manifest["phase_run_id"],
                config_sha256=config_id,
                trial_id=trial["trial_id"],
                receipt_id=receipt["receipt_id"],
                prediction_sha256=descriptor["sha256"],
                ladder_seed=trial["ladder_seed"],
                optimiser_seed=trial["optimiser_seed"],
                metrics=metrics,
                diagnostics=row_diagnostics,
            )
        )
        diagnostics.setdefault(config_id, {})[pair_id] = row_diagnostics
        receipt_ids.append(str(receipt["receipt_id"]))
    observed_by_config = {
        config_id: {
            f"{row['condition']['ladder_seed']}:{row['condition']['optimiser_seed']}"
            for row in rows
        }
        for config_id, rows in by_config.items()
    }
    if len(by_config) != 2 or any(
        pairs != expected_pairs for pairs in observed_by_config.values()
    ):
        raise ModernBertContractError("candidate config-by-condition cross-product drifted")
    return by_config, diagnostics, receipt_ids


def closeout_confirmation(
    *,
    confirmation_manifest: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    volume_root: Path,
) -> dict[str, Any]:
    """Privately score and immutably close the exact 2x3 confirmation.

    The function reads row-level labels and logits only inside the private Volume
    boundary.  Its publications contain content-addressed evidence bindings and
    aggregates produced by the frozen experiment gate—never IDs, logits, labels,
    or per-row predictions.
    """

    contract = _experiment_module()
    clean_confirmation = contract.validate_run_manifest(confirmation_manifest)
    if clean_confirmation.get("phase") != "confirmation":
        raise ValueError("conditional closeout requires a confirmation manifest")
    experiment_contract = clean_confirmation["experiment_contract"]
    development_reference_sha256 = experiment_contract["bindings"][
        "development_reference_sha256"
    ]
    reference = _load_development_reference(
        volume_root=volume_root,
        development_reference_sha256=development_reference_sha256,
    )
    baseline_results, _, _ = _baseline_prediction_frame(
        baseline_manifest=baseline_manifest,
        volume_root=volume_root,
        reference=reference,
    )
    candidate_results, _, _ = _candidate_prediction_frame(
        confirmation_manifest=clean_confirmation,
        volume_root=volume_root,
        reference=reference,
    )
    gates = {
        config_id: contract.evaluate_continuation_gate(
            experiment_contract,
            phase_run_id=clean_confirmation["phase_run_id"],
            candidate_config_sha256=config_id,
            baseline_results=baseline_results,
            candidate_results=results,
        )
        for config_id, results in sorted(candidate_results.items())
    }
    selection = contract.select_confirmation_recipe(
        experiment_contract,
        phase_run_id=clean_confirmation["phase_run_id"],
        baseline_results=baseline_results,
        candidate_results_by_config=candidate_results,
    )
    candidate_gate_ids = {
        row["config_sha256"]: row["gate_receipt_id"] for row in selection["candidates"]
    }
    if candidate_gate_ids != {
        config_id: gate["gate_receipt_id"] for config_id, gate in gates.items()
    }:
        raise ModernBertContractError("selection and gate evidence bindings disagree")
    output_root = contract.experiment_output_root(volume_root, experiment_contract) / "closeout"
    gate_descriptors = {
        config_id: contract.publish_immutable_json(
            output_root / f"gate-{config_id}.json", gate
        )
        for config_id, gate in gates.items()
    }
    selection_descriptor = contract.publish_immutable_json(
        output_root / "recipe-selection.json", selection
    )
    result = {
        "status": "complete",
        "experiment_run_id": clean_confirmation["experiment_run_id"],
        "phase_run_id": clean_confirmation["phase_run_id"],
        "development_rows": len(reference),
        "baseline_trials": len(baseline_results),
        "candidate_trials": sum(len(rows) for rows in candidate_results.values()),
        "candidate_recipes": len(candidate_results),
        "gate_receipt_ids": {
            config_id: gate["gate_receipt_id"] for config_id, gate in gates.items()
        },
        "gate_artifacts": gate_descriptors,
        "selection_receipt_id": selection["selection_receipt_id"],
        "selection_artifact": selection_descriptor,
        "selection_verdict": selection["selection_verdict"],
        "selected_config_sha256": selection["selected_config_sha256"],
    }
    assert_metadata_only(result, where="conditional confirmation closeout summary")
    return result


def _validate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "experiment_run_id",
        "phase_run_id",
        "experiment_contract",
        "trial_spec",
    }
    if set(job) not in {frozenset(required), frozenset({*required, "trial_spec_sha256"})}:
        raise ValueError("conditional Modal job has unexpected fields")
    contract = job["experiment_contract"]
    spec = job["trial_spec"]
    if not isinstance(contract, Mapping) or not isinstance(spec, Mapping):
        raise ValueError("conditional job lacks contract or trial spec")
    experiment = _experiment_module()
    clean_contract = experiment.validate_experiment_contract(contract)
    clean_spec = experiment.validate_trial_spec(clean_contract, spec)
    if job["experiment_run_id"] != clean_contract["experiment_run_id"]:
        raise ModernBertContractError("conditional experiment run binding drifted")
    trial_spec_sha256 = canonical_sha256(clean_spec)
    if job.get("trial_spec_sha256", trial_spec_sha256) != trial_spec_sha256:
        raise ModernBertContractError("conditional trial-spec binding drifted")
    if spec.get("phase") not in {"asha", "confirmation"} or spec.get("gpu_type") != GPU_TYPE:
        raise ValueError("conditional runtime accepts only ASHA/confirmation L4 trials")
    bindings = clean_contract.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("conditional contract lacks bindings")
    if bindings.get("dataset_sha256") != DATASET_SHA256:
        raise ModernBertContractError("conditional teacher dataset binding drifted")
    for name in (
        "split_manifest_sha256",
        "source_bundle_sha256",
        "code_sha256",
        "dependency_lock_sha256",
        "development_reference_sha256",
    ):
        _require_sha256(bindings.get(name), where=f"bindings.{name}")
    return {
        **deepcopy(dict(job)),
        "experiment_contract": clean_contract,
        "trial_spec": clean_spec,
        "trial_spec_sha256": trial_spec_sha256,
    }


def _trial_roots(
    *, volume_root: Path, experiment_run_id: str, phase: str, trial_id: str
) -> tuple[Path, Path, Path]:
    run_root = volume_root / OUTPUT_PREFIX / f"run={experiment_run_id}"
    final = run_root / f"phase={phase}" / f"trial={trial_id}"
    staging = run_root / ".incomplete" / f"trial={trial_id}"
    return run_root, final, staging


def validate_trial_artifacts(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash-validate one exact output without returning private contents."""

    assert_metadata_only(receipt, where="conditional public receipt")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "checkpoint_metadata",
        "metrics",
        "development_predictions",
        "trial_spec",
    }:
        raise ModernBertContractError("conditional receipt artifact inventory drifted")
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "relative_path",
            "sha256",
            "bytes",
        }:
            raise ValueError("conditional artifact descriptor schema drifted")
        relative = Path(str(descriptor["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("conditional artifact path is unsafe")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["bytes"]
            or file_sha256(path) != descriptor["sha256"]
        ):
            raise ModernBertContractError(f"conditional artifact is corrupt: {name}")
    contract = _experiment_module()
    stored_spec = _read_json_object(
        root / artifacts["trial_spec"]["relative_path"], where="conditional trial spec"
    )
    if contract.validate_trial_spec(experiment, stored_spec) != dict(trial_spec):
        raise ModernBertContractError("stored conditional trial spec drifted")
    checkpoint = _read_json_object(
        root / artifacts["checkpoint_metadata"]["relative_path"],
        where="conditional checkpoint metadata",
    )
    contract.validate_checkpoint_payload(experiment, trial_spec, checkpoint)
    predictions = _read_json_object(
        root / artifacts["development_predictions"]["relative_path"],
        where="conditional private predictions",
    )
    contract.validate_private_development_predictions(predictions, experiment, trial_spec)
    contract.validate_trial_receipt(experiment, trial_spec, checkpoint, receipt)
    return {"status": "validated", "artifact_count": len(artifacts)}


def run_training_trial(
    *, job: Mapping[str, Any], volume_root: Path, resume: bool = False
) -> dict[str, Any]:
    """Execute and immutably publish one exact conditional-state trial."""

    clean = _validate_job(job)
    spec = clean["trial_spec"]
    run_root, final_root, staging_root = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=clean["experiment_run_id"],
        phase=spec["phase"],
        trial_id=spec["trial_id"],
    )
    if final_root.exists():
        if staging_root.exists():
            raise ModernBertContractError("conditional trial has final and incomplete outputs")
        receipt = _read_json_object(final_root / "receipt.json", where="conditional receipt")
        validate_trial_artifacts(
            final_root,
            receipt,
            experiment=clean["experiment_contract"],
            trial_spec=spec,
        )
        return _receipt_summary(receipt)
    if resume:
        if not (staging_root / "resume.json").is_file():
            raise ModernBertContractError("conditional resume lacks an exact marker")
    else:
        if staging_root.exists():
            raise ModernBertContractError("conditional incomplete trial requires explicit resume")
        staging_root.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(
            staging_root / "resume.json", build_epoch_zero_resume_marker(clean)
        )
    inputs = _load_inputs(clean, volume_root=volume_root)
    result = _execute_trial_runtime(
        job=clean,
        inputs=inputs,
        staging_root=staging_root,
        run_root=run_root,
        resume=resume,
    )
    selected_checkpoint = str(result["selected_checkpoint_relative_path"])
    private_metrics = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "wall_seconds",
            "gpu_seconds",
            "selected_checkpoint_relative_path",
            "development_predictions",
        }
    }
    contract = _experiment_module()
    contract.publish_immutable_json(staging_root / "metrics.json", private_metrics)
    predictions = _bind_private_predictions(
        experiment=clean["experiment_contract"],
        trial_spec=spec,
        value=result["development_predictions"],
    )
    contract.publish_immutable_json(staging_root / "development-predictions.json", predictions)
    contract.publish_immutable_json(staging_root / "trial-spec.json", spec)
    checkpoint_path = staging_root / selected_checkpoint
    raw_checkpoint_descriptor = {
        "relative_path": selected_checkpoint,
        "sha256": file_sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    public_metrics = _public_metrics(private_metrics)
    checkpoint = contract.build_checkpoint_payload(
        clean["experiment_contract"],
        spec,
        completed_epochs=int(result["selected_epoch"]),
        global_step=int(result["selected_checkpoint_global_step"]),
        optimiser_step=int(result["selected_checkpoint_optimizer_step"]),
        artifacts={
            "model_state": raw_checkpoint_descriptor,
            "optimiser_state": raw_checkpoint_descriptor,
            "scheduler_state": raw_checkpoint_descriptor,
            "rng_state": raw_checkpoint_descriptor,
        },
        aggregate_metrics=public_metrics,
    )
    contract.publish_immutable_json(staging_root / "checkpoint-metadata.json", checkpoint)
    artifacts = _artifact_inventory(staging_root)
    gpu_seconds = float(result["gpu_seconds"])
    if not math.isfinite(gpu_seconds) or gpu_seconds < 0:
        raise ValueError("conditional gpu_seconds must be finite and non-negative")
    receipt = contract.build_trial_receipt(
        clean["experiment_contract"],
        spec,
        checkpoint=checkpoint,
        artifacts=artifacts,
        aggregate_metrics=public_metrics,
        gpu_type=GPU_TYPE,
        wall_seconds=float(result["wall_seconds"]),
        gpu_seconds=gpu_seconds,
    )
    contract.publish_immutable_json(staging_root / "receipt.json", receipt)
    (staging_root / "resume.json").unlink(missing_ok=True)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_root, final_root)
    validate_trial_artifacts(
        final_root,
        receipt,
        experiment=clean["experiment_contract"],
        trial_spec=spec,
    )
    return _receipt_summary(receipt)


def _receipt_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "complete",
        "experiment_run_id": receipt["experiment_run_id"],
        "trial_id": receipt["trial_id"],
        "gpu_type": receipt["compute"]["gpu_type"],
        "wall_seconds": receipt["compute"]["wall_seconds"],
        "gpu_seconds": receipt["compute"]["gpu_seconds"],
        "estimated_cost_usd": receipt["compute"]["estimated_cost_usd"],
        "receipt_id": receipt["receipt_id"],
    }


def inspect_trial_output(
    *, manifest: Mapping[str, Any], trial_id: str, volume_root: Path
) -> dict[str, Any]:
    """Return metadata-only status for one exact manifest trial."""

    trials = manifest.get("trials")
    if not isinstance(trials, list):
        raise ValueError("conditional manifest trial inventory is missing")
    matching = [trial for trial in trials if trial.get("trial_id") == trial_id]
    if len(matching) != 1:
        raise ValueError("conditional trial ID is absent or duplicated in manifest")
    trial = matching[0]
    _, final_root, staging_root = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=str(manifest["experiment_run_id"]),
        phase=str(trial["phase"]),
        trial_id=trial_id,
    )
    if final_root.exists() and staging_root.exists():
        raise ModernBertContractError("conditional output has final and incomplete states")
    if final_root.exists():
        receipt = _read_json_object(final_root / "receipt.json", where="conditional receipt")
        if receipt.get("trial_spec_sha256") != canonical_sha256(trial):
            raise ModernBertContractError("conditional receipt trial binding drifted")
        validate_trial_artifacts(
            final_root,
            receipt,
            experiment=manifest["experiment_contract"],
            trial_spec=trial,
        )
        return {"trial_id": trial_id, "state": "complete", "receipt": _receipt_summary(receipt)}
    if staging_root.exists():
        marker = staging_root / "resume.json"
        if not marker.is_file():
            raise ModernBertContractError("conditional incomplete output lacks resume marker")
        value = _read_json_object(marker, where="conditional resume marker")
        if value.get("trial_spec_sha256") != canonical_sha256(trial):
            raise ModernBertContractError("conditional resume marker binding drifted")
        return {"trial_id": trial_id, "state": "resumable", "receipt": None}
    return {"trial_id": trial_id, "state": "missing", "receipt": None}


def aggregate_trial_inspections(
    *, manifest: Mapping[str, Any], inspections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Conserve the exact manifest inventory and return aggregate-only progress."""

    expected = [str(trial["trial_id"]) for trial in manifest["trials"]]
    by_id: dict[str, Mapping[str, Any]] = {}
    for inspection in inspections:
        trial_id = inspection.get("trial_id")
        if trial_id in by_id or trial_id not in expected:
            raise ValueError("conditional inspections contain duplicate or unexpected trial IDs")
        if inspection.get("state") not in {"missing", "resumable", "complete"}:
            raise ValueError("conditional inspection has invalid state")
        by_id[str(trial_id)] = inspection
    if set(by_id) != set(expected):
        raise ValueError("conditional inspections do not conserve manifest trials")
    complete = [trial_id for trial_id in expected if by_id[trial_id]["state"] == "complete"]
    missing = [trial_id for trial_id in expected if by_id[trial_id]["state"] == "missing"]
    resumable = [trial_id for trial_id in expected if by_id[trial_id]["state"] == "resumable"]
    cost = sum(
        (
            Decimal(str(by_id[trial_id]["receipt"]["estimated_cost_usd"]))
            for trial_id in complete
        ),
        Decimal("0"),
    )
    result = {
        "status": "complete" if len(complete) == len(expected) else "incomplete",
        "expected_trials": len(expected),
        "complete_trials": len(complete),
        "complete_trial_ids": complete,
        "missing_trial_ids": missing,
        "resumable_trial_ids": resumable,
        "estimated_cost_usd": format(cost.quantize(Decimal("0.000001")), "f"),
    }
    assert_metadata_only(result, where="conditional aggregate inspection")
    return result


def _completed_receipts(manifest: Mapping[str, Any], *, volume_root: Path) -> list[dict[str, Any]]:
    inspections = [
        inspect_trial_output(
            manifest=manifest,
            trial_id=str(trial["trial_id"]),
            volume_root=volume_root,
        )
        for trial in manifest["trials"]
    ]
    aggregate = aggregate_trial_inspections(manifest=manifest, inspections=inspections)
    if aggregate["status"] != "complete":
        raise ModernBertContractError("conditional transition requires a complete source phase")
    receipts: list[dict[str, Any]] = []
    for inspection in inspections:
        trial_id = inspection["trial_id"]
        _, final_root, _ = _trial_roots(
            volume_root=volume_root,
            experiment_run_id=str(manifest["experiment_run_id"]),
            phase=str(
                next(
                    trial["phase"] for trial in manifest["trials"] if trial["trial_id"] == trial_id
                )
            ),
            trial_id=trial_id,
        )
        receipts.append(_read_json_object(final_root / "receipt.json", where="receipt"))
    return receipts


def _require_preserved_baseline_manifest(
    source_manifest: Mapping[str, Any], derived_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a derived phase to retain the exact baseline descriptor."""

    source_artefacts = source_manifest.get("experiment_artefacts")
    derived_artefacts = derived_manifest.get("experiment_artefacts")
    if not isinstance(source_artefacts, Mapping) or not isinstance(
        derived_artefacts, Mapping
    ):
        raise ModernBertContractError("conditional manifest lacks experiment artefacts")
    baseline = source_artefacts.get("baseline_manifest")
    if not isinstance(baseline, Mapping) or derived_artefacts.get(
        "baseline_manifest"
    ) != dict(baseline):
        raise ModernBertContractError("conditional baseline-manifest descriptor drifted")
    return dict(baseline)


def promote_asha_rung(*, manifest: Mapping[str, Any], volume_root: Path) -> dict[str, Any]:
    """Rank a complete rung and immutably publish the exact next-rung manifest."""

    if manifest.get("phase") != "sweep":
        raise ValueError("only conditional sweep manifests can be promoted")
    asha = manifest.get("asha")
    if not isinstance(asha, Mapping) or asha.get("rung_epochs") not in {2, 4}:
        raise ValueError("conditional promotion requires rung 2 or 4")
    contract = _experiment_module()
    clean_manifest = contract.validate_run_manifest(manifest)
    receipts = _completed_receipts(clean_manifest, volume_root=volume_root)
    by_trial = {trial["trial_id"]: trial for trial in clean_manifest["trials"]}
    results = [
        {
            "config_sha256": by_trial[receipt["trial_id"]]["config"]["config_sha256"],
            "completed_epochs": receipt["aggregate_metrics"]["completed_epochs"],
            "composite": receipt["aggregate_metrics"]["composite"],
            "material_recall": receipt["aggregate_metrics"]["material_recall"],
            "invalid_outputs": receipt["aggregate_metrics"]["invalid_outputs"],
        }
        for receipt in receipts
    ]
    selected_config_ids = contract.select_asha_promotions(
        contract.build_asha_plan(),
        results,
        rung_epochs=int(asha["rung_epochs"]),
        candidate_config_ids=[
            trial["config"]["config_sha256"] for trial in clean_manifest["trials"]
        ],
    )
    next_epoch = {2: 4, 4: 8}[int(asha["rung_epochs"])]
    receipt_by_trial = {receipt["trial_id"]: receipt for receipt in receipts}
    source_by_config = {
        trial["config"]["config_sha256"]: trial for trial in clean_manifest["trials"]
    }
    promoted_trials: list[dict[str, Any]] = []
    for config_id in selected_config_ids:
        source = source_by_config[config_id]
        receipt = receipt_by_trial[source["trial_id"]]
        source_root = contract.trial_output_root(
            volume_root, clean_manifest["experiment_contract"], source
        )
        checkpoint = _read_json_object(
            source_root / receipt["artifacts"]["checkpoint_metadata"]["relative_path"],
            where="conditional promotion checkpoint",
        )
        resume_binding = contract.build_resume_binding(
            clean_manifest["experiment_contract"],
            source,
            checkpoint,
            checkpoint_metadata_artifact=receipt["artifacts"]["checkpoint_metadata"],
        )
        promoted_trials.append(
            contract.freeze_trial_spec(
                clean_manifest["experiment_contract"],
                phase="asha",
                config=source["config"],
                subset_manifest_sha256=source["subset_manifest_sha256"],
                label_budget=5_000,
                training_row_count=source["training_row_count"],
                ladder_seed=source["ladder_seed"],
                optimiser_seed=source["optimiser_seed"],
                target_epochs=next_epoch,
                gpu_type=source["gpu_type"],
                max_gpu_seconds=source["max_gpu_seconds"],
                resume_binding=resume_binding,
            )
        )
    transition_body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-conditional-asha-transition-v1",
        "experiment_run_id": clean_manifest["experiment_run_id"],
        "source_phase_run_id": clean_manifest["phase_run_id"],
        "source_rung_epochs": int(asha["rung_epochs"]),
        "target_rung_epochs": next_epoch,
        "selected_config_ids": selected_config_ids,
        "source_receipt_ids": [
            receipt_by_trial[source_by_config[config_id]["trial_id"]]["receipt_id"]
            for config_id in selected_config_ids
        ],
    }
    transition = {
        **transition_body,
        "transition_receipt_id": canonical_sha256(transition_body),
    }
    run_root = contract.experiment_output_root(volume_root, clean_manifest["experiment_contract"])
    transition_descriptor = contract.publish_immutable_json(
        run_root / "manifests" / f"transition-rung-{next_epoch}.json", transition
    )
    next_manifest = contract.build_run_manifest(
        clean_manifest["experiment_contract"],
        phase="sweep",
        trials=promoted_trials,
        asha_rung_epochs=next_epoch,
        baseline_manifest=clean_manifest["experiment_artefacts"]["baseline_manifest"],
        source_manifest_sha256=canonical_sha256(clean_manifest),
        transition_receipt_sha256=transition["transition_receipt_id"],
    )
    _require_preserved_baseline_manifest(clean_manifest, next_manifest)
    path = (
        volume_root
        / OUTPUT_PREFIX
        / f"run={manifest['experiment_run_id']}"
        / "manifests"
        / f"sweep-rung-{next_epoch}.json"
    )
    descriptor = contract.publish_immutable_json(path, next_manifest)
    return {
        "status": "promoted",
        "source_rung_epochs": int(asha["rung_epochs"]),
        "target_rung_epochs": next_epoch,
        "promoted_trials": len(promoted_trials),
        "phase_run_id": next_manifest["phase_run_id"],
        "manifest": descriptor,
        "transition": transition_descriptor,
    }


def prepare_confirmation_manifest(
    *, manifest: Mapping[str, Any], volume_root: Path
) -> dict[str, Any]:
    """Freeze two winning 10k recipes crossed with three fresh seeds."""

    if (
        manifest.get("phase") != "sweep"
        or not isinstance(manifest.get("asha"), Mapping)
        or manifest["asha"].get("rung_epochs") != 8
    ):
        raise ValueError("confirmation preparation requires the complete rung-8 sweep")
    contract = _experiment_module()
    clean_manifest = contract.validate_run_manifest(manifest)
    receipts = _completed_receipts(clean_manifest, volume_root=volume_root)
    ranked = sorted(
        receipts,
        key=lambda receipt: (
            -float(receipt["aggregate_metrics"]["composite"]),
            str(receipt["trial_id"]),
        ),
    )
    winners = ranked[:2]
    by_trial = {trial["trial_id"]: trial for trial in clean_manifest["trials"]}
    split_path = volume_root / "student-modernbert-v1/inputs/split-manifest.json"
    if (
        file_sha256(split_path)
        != clean_manifest["experiment_contract"]["bindings"]["split_manifest_sha256"]
    ):
        raise ModernBertContractError("conditional confirmation split manifest drifted")
    split_manifest = _read_json_object(split_path, where="conditional split manifest")
    conditions = _experiment_module().CONFIRMATION_CONDITIONS
    trials: list[dict[str, Any]] = []
    for winner in winners:
        config = by_trial[winner["trial_id"]]["config"]
        for condition in conditions:
            ladder_seed = int(condition["ladder_seed"])
            subset_metadata = split_manifest["ladders"][str(ladder_seed)]["budgets"]["10k"]
            trials.append(
                contract.freeze_trial_spec(
                    clean_manifest["experiment_contract"],
                    phase="confirmation",
                    config=config,
                    subset_manifest_sha256=canonical_sha256(subset_metadata),
                    label_budget=10_000,
                    training_row_count=int(subset_metadata["row_count"]),
                    ladder_seed=ladder_seed,
                    optimiser_seed=int(condition["optimiser_seed"]),
                    target_epochs=8,
                    gpu_type=GPU_TYPE,
                    max_gpu_seconds=5_400,
                    resume_binding=None,
                )
            )
    transition_body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-conditional-confirmation-transition-v1",
        "experiment_run_id": clean_manifest["experiment_run_id"],
        "source_phase_run_id": clean_manifest["phase_run_id"],
        "winning_config_ids": [
            by_trial[row["trial_id"]]["config"]["config_sha256"] for row in winners
        ],
        "source_receipt_ids": [row["receipt_id"] for row in winners],
    }
    transition = {
        **transition_body,
        "transition_receipt_id": canonical_sha256(transition_body),
    }
    run_root = contract.experiment_output_root(volume_root, clean_manifest["experiment_contract"])
    transition_descriptor = contract.publish_immutable_json(
        run_root / "manifests" / "transition-confirmation.json", transition
    )
    confirmation = contract.build_run_manifest(
        clean_manifest["experiment_contract"],
        phase="confirmation",
        trials=trials,
        asha_rung_epochs=None,
        baseline_manifest=clean_manifest["experiment_artefacts"]["baseline_manifest"],
        source_manifest_sha256=canonical_sha256(clean_manifest),
        transition_receipt_sha256=transition["transition_receipt_id"],
    )
    _require_preserved_baseline_manifest(clean_manifest, confirmation)
    if len(confirmation.get("trials", [])) != 6:
        raise ModernBertContractError("conditional confirmation must contain two recipes x 3 seeds")
    path = (
        volume_root
        / OUTPUT_PREFIX
        / f"run={manifest['experiment_run_id']}"
        / "manifests"
        / "confirmation.json"
    )
    descriptor = contract.publish_immutable_json(path, confirmation)
    return {
        "status": "prepared",
        "recipes": 2,
        "seeds": list(CONFIRMATION_SEEDS),
        "trials": 6,
        "phase_run_id": confirmation["phase_run_id"],
        "manifest": descriptor,
        "transition": transition_descriptor,
    }


__all__ = [
    "aggregate_trial_inspections",
    "build_epoch_zero_resume_marker",
    "build_optimisation_config",
    "closeout_confirmation",
    "conditional_diagnostics",
    "inspect_trial_output",
    "prepare_confirmation_manifest",
    "promote_asha_rung",
    "relevance_class_weights",
    "run_training_trial",
    "select_teacher_rows",
    "target_state_class_weights",
    "validate_private_development_predictions",
    "validate_trial_artifacts",
]
