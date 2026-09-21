"""Strict contracts for the conditional-state ModernBERT experiment.

This module deliberately contains no Torch, Transformers, or Modal imports. It
freezes the experiment inventory and the metadata boundary used by the runtime.
Private row-level development predictions may be validated here, but only their
immutable file descriptors may enter public receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal
from itertools import product
from pathlib import Path, PurePosixPath
from typing import Any

from reddit_china_stance.modernbert_training import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    TEACHER_GENERATION_RUN_ID,
    TOKENIZER_REVISION,
    canonical_sha256,
)
from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
NAMESPACE = "student-modernbert-conditional-v1"
OUTPUT_PREFIX = Path(NAMESPACE)

EXPERIMENT_KIND = "modernbert-conditional-state-experiment-v1"
ASHA_PLAN_KIND = "modernbert-conditional-state-asha-plan-v1"
TRIAL_SPEC_KIND = "modernbert-conditional-state-trial-spec-v1"
RUN_MANIFEST_KIND = "modernbert-conditional-state-run-manifest-v1"
CHECKPOINT_KIND = "modernbert-conditional-state-checkpoint-v1"
RESUME_BINDING_KIND = "modernbert-conditional-state-resume-binding-v1"
PRIVATE_PREDICTIONS_KIND = "modernbert-conditional-private-development-predictions-v1"
TRIAL_RECEIPT_KIND = "modernbert-conditional-state-trial-receipt-v1"
CONTINUATION_GATE_KIND = "modernbert-conditional-state-continuation-gate-v1"

HARD_COST_CAP_USD = Decimal("200")
DEVELOPMENT_ROWS = 222
TARGETS = ("china_general", "government_ccp", "people_culture", "other")
CORE_TARGETS = TARGETS[:3]
RELEVANCE_CLASSES = ("material", "not_material", "unclear")
TARGET_STATES = (
    "absent",
    "negative",
    "mixed",
    "no_directed_stance",
    "positive",
    "unclear",
)
CONFIRMATION_CONDITIONS = (
    {"ladder_seed": 101, "optimiser_seed": 47},
    {"ladder_seed": 202, "optimiser_seed": 61},
    {"ladder_seed": 303, "optimiser_seed": 89},
)
ASHA_RUNGS = (
    {"epochs": 2, "candidate_count": 8, "promotion_count": 4},
    {"epochs": 4, "candidate_count": 4, "promotion_count": 2},
    {"epochs": 8, "candidate_count": 2, "promotion_count": 0},
)


class ConditionalExperimentContractError(RuntimeError):
    """Raised when a frozen conditional-state experiment contract drifts."""


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc


def _require_sha256(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _require_probability(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number in [0, 1]")
    clean = float(value)
    if not math.isfinite(clean) or not 0 <= clean <= 1:
        raise ValueError(f"{where} must be a finite number in [0, 1]")
    return clean


def _require_nonnegative_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a non-negative finite number")
    clean = float(value)
    if not math.isfinite(clean) or clean < 0:
        raise ValueError(f"{where} must be a non-negative finite number")
    return clean


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001")))


def _fixed_training_settings() -> dict[str, Any]:
    return {
        "head_learning_rate_multiplier": 5.0,
        "optimiser": "adamw",
        "weight_decay": 0.01,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "scheduler": "linear_decay",
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "precision": "bf16",
        "effective_batch_size": 32,
        "max_epochs": 8,
        "max_length": 768,
        "pooling": "masked_mean",
        "relevance_class_weights": "capped_inverse_sqrt",
        "checkpoint_selection": {
            "minimum_epoch": 2,
            "min_delta": 0.001,
            "patience_epochs": 2,
        },
    }


def frozen_trial_configs() -> list[dict[str, Any]]:
    """Return the exact eight-configuration conditional-state sweep."""

    configs: list[dict[str, Any]] = []
    for encoder_lr, relevance_weight, state_weighting in product(
        (3e-5, 5e-5),
        (1.0, 2.0),
        ("none", "capped_inverse_sqrt"),
    ):
        body = {
            "encoder_learning_rate": encoder_lr,
            "loss_weights": {"relevance": relevance_weight, "target_state": 1.0},
            "target_state_class_weights": state_weighting,
            **_fixed_training_settings(),
        }
        configs.append({**body, "config_sha256": canonical_sha256(body)})
    if len(configs) != 8 or len({row["config_sha256"] for row in configs}) != 8:
        raise AssertionError("conditional-state sweep is not eight unique configurations")
    return configs


def _architecture_contract() -> dict[str, Any]:
    return {
        "encoder": "one_shared_encoder",
        "relevance_head": {"classes": list(RELEVANCE_CLASSES)},
        "target_state_heads": {
            "targets": list(TARGETS),
            "states": list(TARGET_STATES),
            "loss_scope": "four_target_slots_on_material_rows",
        },
        "decoder": {
            "relevance_gates_targets": True,
            "material_all_absent_rule": "highest_best_non_absent_minus_absent_margin",
        },
        "target_state_weighting": {
            "source": "active_training_subset_material_target_slots",
            "formula": "sqrt(total_supported_slots/class_count)",
            "normalisation": "supported_class_mean_one",
            "minimum": 0.5,
            "maximum": 4.0,
            "unsupported_class_action": "fail",
        },
    }


def _continuation_criteria() -> dict[str, Any]:
    return {
        "primary_metric": "core_target_stance_tuple_micro_f1",
        "stance_states": list(TARGET_STATES),
        "minimum_mean_gain": 0.03,
        "minimum_improved_pairs": 2,
        "paired_condition_count": 3,
        "maximum_mean_material_recall_decline": 0.02,
        "maximum_supported_core_target_regression": 0.05,
        "maximum_supported_target_stance_regression": 0.10,
        "minimum_reference_positives_for_supported_target": 10,
        "maximum_invalid_outputs": 0,
        "gain_bands": {
            "scrap_below": 0.01,
            "inconclusive_below": 0.03,
        },
    }


def build_asha_plan() -> dict[str, Any]:
    """Build the frozen 5k cumulative 8 -> 4 -> 2 ASHA plan."""

    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": ASHA_PLAN_KIND,
        "label_budget": 5_000,
        "ladder_seed": 101,
        "optimiser_seed": 7,
        "rungs": [dict(rung) for rung in ASHA_RUNGS],
        "configs": frozen_trial_configs(),
        "ranking_metric": "development_composite",
        "minimum_material_recall": 0.80,
        "maximum_invalid_outputs": 0,
    }
    return {**body, "plan_id": canonical_sha256(body)}


def validate_asha_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_asha_plan()
    if dict(value) != expected:
        raise ConditionalExperimentContractError(
            "ASHA plan differs from the frozen 8 -> 4 -> 2 plan"
        )
    return deepcopy(expected)


def select_asha_promotions(
    plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    rung_epochs: int,
    candidate_config_ids: Sequence[str],
) -> list[str]:
    """Select a complete promoting rung with deterministic digest tie-breaking."""

    clean_plan = validate_asha_plan(plan)
    rung = next((row for row in clean_plan["rungs"] if row["epochs"] == rung_epochs), None)
    if rung is None or rung["promotion_count"] == 0:
        raise ValueError("rung_epochs must identify a promoting ASHA rung")
    candidates = list(candidate_config_ids)
    if len(candidates) != rung["candidate_count"] or len(set(candidates)) != len(candidates):
        raise ValueError("candidate inventory differs from the frozen rung")
    registered = {row["config_sha256"] for row in clean_plan["configs"]}
    if not set(candidates) <= registered:
        raise ValueError("candidate inventory contains an unregistered configuration")

    by_id: dict[str, dict[str, Any]] = {}
    expected_fields = {
        "config_sha256",
        "completed_epochs",
        "composite",
        "material_recall",
        "invalid_outputs",
    }
    for raw in results:
        if set(raw) != expected_fields:
            raise ValueError("ASHA result has unexpected fields")
        config_id = _require_sha256(raw["config_sha256"], where="result.config_sha256")
        if config_id in by_id:
            raise ValueError("ASHA result set contains a duplicate configuration")
        if raw["completed_epochs"] != rung_epochs:
            raise ValueError("ASHA result belongs to another rung")
        composite = _require_probability(raw["composite"], where="result.composite")
        material_recall = _require_probability(
            raw["material_recall"], where="result.material_recall"
        )
        invalid_outputs = _require_nonnegative_int(
            raw["invalid_outputs"], where="result.invalid_outputs"
        )
        by_id[config_id] = {
            "config_sha256": config_id,
            "completed_epochs": rung_epochs,
            "composite": composite,
            "material_recall": material_recall,
            "invalid_outputs": invalid_outputs,
        }
    if set(by_id) != set(candidates):
        raise ValueError("ASHA result set does not exactly cover the candidate inventory")
    eligible = [
        row
        for row in by_id.values()
        if row["invalid_outputs"] == 0
        and row["material_recall"] >= clean_plan["minimum_material_recall"]
    ]
    if len(eligible) < rung["promotion_count"]:
        raise ConditionalExperimentContractError("too few ASHA candidates pass the frozen gates")
    ranked = sorted(
        eligible,
        key=lambda row: (-row["composite"], row["config_sha256"]),
    )
    return [row["config_sha256"] for row in ranked[: rung["promotion_count"]]]


def freeze_experiment_contract(
    *,
    dataset_revision: str,
    dataset_sha256: str,
    split_manifest_sha256: str,
    source_bundle_sha256: str,
    code_sha256: str,
    dependency_lock_sha256: str,
    development_reference_sha256: str,
    matched_baseline_receipt_sha256: str,
    teacher_generation_run_id: str,
    rate_card_usd_per_gpu_second: Mapping[str, str | float],
    hard_cost_cap_usd: int | float | str = 200,
) -> dict[str, Any]:
    """Freeze all evidence, implementation, design, and budget bindings."""

    if dataset_revision != DATASET_REVISION or dataset_sha256 != DATASET_SHA256:
        raise ConditionalExperimentContractError("private teacher dataset identity drifted")
    if teacher_generation_run_id != TEACHER_GENERATION_RUN_ID:
        raise ConditionalExperimentContractError("teacher generation identity drifted")
    bindings = {
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "dataset_sha256": _require_sha256(dataset_sha256, where="dataset_sha256"),
        "split_manifest_sha256": _require_sha256(
            split_manifest_sha256, where="split_manifest_sha256"
        ),
        "source_bundle_sha256": _require_sha256(
            source_bundle_sha256, where="source_bundle_sha256"
        ),
        "code_sha256": _require_sha256(code_sha256, where="code_sha256"),
        "dependency_lock_sha256": _require_sha256(
            dependency_lock_sha256, where="dependency_lock_sha256"
        ),
        "development_reference_sha256": _require_sha256(
            development_reference_sha256, where="development_reference_sha256"
        ),
        "matched_baseline_receipt_sha256": _require_sha256(
            matched_baseline_receipt_sha256, where="matched_baseline_receipt_sha256"
        ),
        "teacher_generation_run_id": _require_sha256(
            teacher_generation_run_id, where="teacher_generation_run_id"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    cap = Decimal(str(hard_cost_cap_usd))
    if not cap.is_finite() or cap <= 0 or cap > HARD_COST_CAP_USD:
        raise ValueError("hard cost cap must be positive and no greater than $200")
    if not rate_card_usd_per_gpu_second:
        raise ValueError("at least one GPU rate must be frozen")
    rate_card: dict[str, str] = {}
    for gpu_type, raw_rate in sorted(rate_card_usd_per_gpu_second.items()):
        if not isinstance(gpu_type, str) or not gpu_type:
            raise ValueError("GPU type must be non-empty")
        rate = Decimal(str(raw_rate))
        if not rate.is_finite() or rate <= 0:
            raise ValueError("GPU rates must be positive finite decimals")
        rate_card[gpu_type] = format(rate, "f")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPERIMENT_KIND,
        "namespace": NAMESPACE,
        "bindings": bindings,
        "architecture": _architecture_contract(),
        "registered_design": {
            "asha_plan": build_asha_plan(),
            "confirmation_label_budget": 10_000,
            "confirmation_recipe_count": 2,
            "confirmation_conditions": [dict(row) for row in CONFIRMATION_CONDITIONS],
            "continuation_criteria": _continuation_criteria(),
            "development_rows": DEVELOPMENT_ROWS,
        },
        "budget": {
            "hard_cost_cap_usd": format(cap, "f"),
            "rate_card_usd_per_gpu_second": rate_card,
        },
        "compute": {
            "allowed_gpus": ["L4"],
            "account_gpu_limit": 10,
            "max_concurrent_trials": 8,
            "gpu_fallback_allowed": False,
            "planned_upper_cost_usd": "50",
            "approved_cost_usd": format(cap, "f"),
        },
    }
    assert_metadata_only(body, where="conditional experiment contract")
    return {**body, "experiment_run_id": canonical_sha256(body)}


def validate_experiment_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "namespace",
        "bindings",
        "architecture",
        "registered_design",
        "budget",
        "compute",
        "experiment_run_id",
    }
    if set(value) != required:
        raise ValueError("experiment contract has unexpected fields")
    bindings = value.get("bindings")
    budget = value.get("budget")
    compute = value.get("compute")
    if not all(isinstance(block, Mapping) for block in (bindings, budget, compute)):
        raise ValueError("experiment bindings, budget, and compute must be objects")
    if set(bindings) != {
        "dataset_id",
        "dataset_revision",
        "dataset_sha256",
        "split_manifest_sha256",
        "source_bundle_sha256",
        "code_sha256",
        "dependency_lock_sha256",
        "development_reference_sha256",
        "matched_baseline_receipt_sha256",
        "teacher_generation_run_id",
        "model_id",
        "model_revision",
        "tokenizer_revision",
    }:
        raise ValueError("experiment bindings have unexpected fields")
    if set(budget) != {"hard_cost_cap_usd", "rate_card_usd_per_gpu_second"}:
        raise ValueError("experiment budget has unexpected fields")
    if dict(compute) != {
        "allowed_gpus": ["L4"],
        "account_gpu_limit": 10,
        "max_concurrent_trials": 8,
        "gpu_fallback_allowed": False,
        "planned_upper_cost_usd": "50",
        "approved_cost_usd": budget["hard_cost_cap_usd"],
    }:
        raise ConditionalExperimentContractError("experiment compute contract drifted")
    rebuilt = freeze_experiment_contract(
        dataset_revision=bindings["dataset_revision"],
        dataset_sha256=bindings["dataset_sha256"],
        split_manifest_sha256=bindings["split_manifest_sha256"],
        source_bundle_sha256=bindings["source_bundle_sha256"],
        code_sha256=bindings["code_sha256"],
        dependency_lock_sha256=bindings["dependency_lock_sha256"],
        development_reference_sha256=bindings["development_reference_sha256"],
        matched_baseline_receipt_sha256=bindings["matched_baseline_receipt_sha256"],
        teacher_generation_run_id=bindings["teacher_generation_run_id"],
        rate_card_usd_per_gpu_second=budget["rate_card_usd_per_gpu_second"],
        hard_cost_cap_usd=budget["hard_cost_cap_usd"],
    )
    if dict(value) != rebuilt:
        raise ConditionalExperimentContractError("experiment contract content address drifted")
    return rebuilt


def _validate_artifact_descriptor(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if set(value) != {"relative_path", "sha256", "bytes"}:
        raise ValueError(f"{where} has unexpected fields")
    relative_path = value["relative_path"]
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{where}.relative_path must be non-empty")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{where}.relative_path must be safe and relative")
    return {
        "relative_path": relative_path,
        "sha256": _require_sha256(value["sha256"], where=f"{where}.sha256"),
        "bytes": _require_positive_int(value["bytes"], where=f"{where}.bytes"),
    }


def _validate_repo_artifact_descriptor(
    value: Mapping[str, Any], *, where: str
) -> dict[str, Any]:
    if set(value) != {"repo_relative_path", "sha256", "bytes"}:
        raise ValueError(f"{where} has unexpected fields")
    relative_path = value["repo_relative_path"]
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ValueError(f"{where}.repo_relative_path must be a non-empty POSIX path")
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"{where}.repo_relative_path must be safe and repo-relative")
    return {
        "repo_relative_path": relative_path,
        "sha256": _require_sha256(value["sha256"], where=f"{where}.sha256"),
        "bytes": _require_positive_int(value["bytes"], where=f"{where}.bytes"),
    }


def _validate_resume_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "source_trial_id",
        "source_trial_spec_sha256",
        "source_config_sha256",
        "source_subset_manifest_sha256",
        "source_ladder_seed",
        "source_optimiser_seed",
        "completed_epochs",
        "checkpoint_id",
        "checkpoint_metadata_artifact",
        "resume_binding_id",
    }
    if set(value) != required:
        raise ValueError("resume binding has unexpected fields")
    body = {key: value[key] for key in required - {"resume_binding_id"}}
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != RESUME_BINDING_KIND:
        raise ConditionalExperimentContractError("resume binding schema or kind drifted")
    for field in (
        "experiment_run_id",
        "source_trial_id",
        "source_trial_spec_sha256",
        "source_config_sha256",
        "source_subset_manifest_sha256",
        "checkpoint_id",
    ):
        _require_sha256(value[field], where=f"resume.{field}")
    _require_positive_int(value["source_ladder_seed"], where="resume.source_ladder_seed")
    _require_positive_int(value["source_optimiser_seed"], where="resume.source_optimiser_seed")
    _require_positive_int(value["completed_epochs"], where="resume.completed_epochs")
    _validate_artifact_descriptor(
        value["checkpoint_metadata_artifact"], where="resume.checkpoint_metadata_artifact"
    )
    if value["resume_binding_id"] != canonical_sha256(body):
        raise ConditionalExperimentContractError("resume binding content address drifted")
    return deepcopy(dict(value))


def freeze_trial_spec(
    experiment: Mapping[str, Any],
    *,
    phase: str,
    config: Mapping[str, Any],
    subset_manifest_sha256: str,
    label_budget: int,
    training_row_count: int,
    ladder_seed: int,
    optimiser_seed: int,
    target_epochs: int,
    gpu_type: str,
    max_gpu_seconds: int,
    resume_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one exact ASHA or fresh 10k confirmation trial."""

    clean_experiment = validate_experiment_contract(experiment)
    if phase not in {"asha", "confirmation"}:
        raise ValueError("unsupported conditional-state trial phase")
    clean_config = _json_clone(config)
    config_id = clean_config.get("config_sha256")
    config_body = {key: value for key, value in clean_config.items() if key != "config_sha256"}
    if config_id != canonical_sha256(config_body):
        raise ConditionalExperimentContractError("trial configuration content address drifted")
    registered = {row["config_sha256"] for row in frozen_trial_configs()}
    if config_id not in registered:
        raise ConditionalExperimentContractError("trial uses an unregistered configuration")
    subset_digest = _require_sha256(subset_manifest_sha256, where="subset_manifest_sha256")
    budget = _require_positive_int(label_budget, where="label_budget")
    row_count = _require_positive_int(training_row_count, where="training_row_count")
    if abs(row_count - budget) > 1:
        raise ConditionalExperimentContractError("training row count differs from label budget")
    ladder = _require_positive_int(ladder_seed, where="ladder_seed")
    optimiser = _require_positive_int(optimiser_seed, where="optimiser_seed")
    epochs = _require_positive_int(target_epochs, where="target_epochs")

    clean_resume: dict[str, Any] | None = None
    if phase == "asha":
        if (budget, ladder, optimiser) != (5_000, 101, 7) or epochs not in {2, 4, 8}:
            raise ConditionalExperimentContractError("ASHA trial condition drifted")
        if epochs == 2:
            if resume_binding is not None:
                raise ConditionalExperimentContractError("first ASHA rung cannot have a source")
        else:
            if not isinstance(resume_binding, Mapping):
                raise ConditionalExperimentContractError(
                    "continued ASHA rung lacks a resume binding"
                )
            clean_resume = _validate_resume_binding(resume_binding)
            expected_previous = {4: 2, 8: 4}[epochs]
            if (
                clean_resume["experiment_run_id"] != clean_experiment["experiment_run_id"]
                or clean_resume["source_config_sha256"] != config_id
                or clean_resume["source_subset_manifest_sha256"] != subset_digest
                or clean_resume["source_ladder_seed"] != ladder
                or clean_resume["source_optimiser_seed"] != optimiser
                or clean_resume["completed_epochs"] != expected_previous
            ):
                raise ConditionalExperimentContractError("ASHA resume binding drifted")
    else:
        conditions = {
            (row["ladder_seed"], row["optimiser_seed"])
            for row in CONFIRMATION_CONDITIONS
        }
        if budget != 10_000 or (ladder, optimiser) not in conditions or epochs != 8:
            raise ConditionalExperimentContractError("confirmation trial condition drifted")
        if resume_binding is not None:
            raise ConditionalExperimentContractError("confirmation trials must start fresh")

    seconds = _require_positive_int(max_gpu_seconds, where="max_gpu_seconds")
    rate_card = clean_experiment["budget"]["rate_card_usd_per_gpu_second"]
    if gpu_type not in rate_card:
        raise ValueError("gpu_type is absent from the frozen rate card")
    reservation = Decimal(rate_card[gpu_type]) * Decimal(seconds)
    if reservation > Decimal(clean_experiment["budget"]["hard_cost_cap_usd"]):
        raise ConditionalExperimentContractError("single trial reservation exceeds the cost cap")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_SPEC_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase": phase,
        "config": clean_config,
        "subset_manifest_sha256": subset_digest,
        "label_budget": budget,
        "training_row_count": row_count,
        "ladder_seed": ladder,
        "optimiser_seed": optimiser,
        "target_epochs": epochs,
        "gpu_type": gpu_type,
        "max_gpu_seconds": seconds,
        "reserved_cost_usd": _money(reservation),
        "resume_binding": clean_resume,
    }
    assert_metadata_only(body, where="conditional trial spec")
    return {**body, "trial_id": canonical_sha256(body)}


def validate_trial_spec(
    experiment: Mapping[str, Any], trial_spec: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase",
        "config",
        "subset_manifest_sha256",
        "label_budget",
        "training_row_count",
        "ladder_seed",
        "optimiser_seed",
        "target_epochs",
        "gpu_type",
        "max_gpu_seconds",
        "reserved_cost_usd",
        "resume_binding",
        "trial_id",
    }
    if set(trial_spec) != required:
        raise ValueError("trial spec has unexpected fields")
    rebuilt = freeze_trial_spec(
        experiment,
        phase=trial_spec["phase"],
        config=trial_spec["config"],
        subset_manifest_sha256=trial_spec["subset_manifest_sha256"],
        label_budget=trial_spec["label_budget"],
        training_row_count=trial_spec["training_row_count"],
        ladder_seed=trial_spec["ladder_seed"],
        optimiser_seed=trial_spec["optimiser_seed"],
        target_epochs=trial_spec["target_epochs"],
        gpu_type=trial_spec["gpu_type"],
        max_gpu_seconds=trial_spec["max_gpu_seconds"],
        resume_binding=trial_spec["resume_binding"],
    )
    if dict(trial_spec) != rebuilt:
        raise ConditionalExperimentContractError("trial spec content address or cost drifted")
    return rebuilt


def build_run_manifest(
    experiment: Mapping[str, Any],
    *,
    phase: str,
    trials: Sequence[Mapping[str, Any]],
    asha_rung_epochs: int | None,
    baseline_manifest: Mapping[str, Any],
    source_manifest_sha256: str | None = None,
    transition_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze one exact schedulable sweep rung or six-run two-recipe confirmation."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_baseline_manifest = _validate_repo_artifact_descriptor(
        baseline_manifest,
        where="experiment_artefacts.baseline_manifest",
    )
    clean_trials = [validate_trial_spec(clean_experiment, trial) for trial in trials]
    trial_ids = [trial["trial_id"] for trial in clean_trials]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("run manifest contains duplicate trials")
    if phase == "sweep":
        rung = next((row for row in ASHA_RUNGS if row["epochs"] == asha_rung_epochs), None)
        if rung is None or len(clean_trials) != rung["candidate_count"]:
            raise ConditionalExperimentContractError("sweep manifest inventory drifted")
        if any(
            trial["phase"] != "asha" or trial["target_epochs"] != asha_rung_epochs
            for trial in clean_trials
        ):
            raise ConditionalExperimentContractError("sweep manifest trial condition drifted")
        if len({trial["config"]["config_sha256"] for trial in clean_trials}) != len(
            clean_trials
        ):
            raise ConditionalExperimentContractError("sweep manifest repeats a configuration")
        if asha_rung_epochs == 2:
            if source_manifest_sha256 is not None or transition_receipt_sha256 is not None:
                raise ConditionalExperimentContractError("first sweep rung cannot have a source")
        else:
            _require_sha256(source_manifest_sha256, where="source_manifest_sha256")
            _require_sha256(transition_receipt_sha256, where="transition_receipt_sha256")
        asha: dict[str, Any] | None = {
            "rung_epochs": rung["epochs"],
            "candidate_count": rung["candidate_count"],
            "promotion_count": rung["promotion_count"],
        }
    elif phase == "confirmation":
        if asha_rung_epochs is not None or len(clean_trials) != 6:
            raise ConditionalExperimentContractError("confirmation manifest inventory drifted")
        if any(trial["phase"] != "confirmation" for trial in clean_trials):
            raise ConditionalExperimentContractError("confirmation manifest contains another phase")
        observed_conditions = {
            (trial["ladder_seed"], trial["optimiser_seed"]) for trial in clean_trials
        }
        expected_conditions = {
            (row["ladder_seed"], row["optimiser_seed"])
            for row in CONFIRMATION_CONDITIONS
        }
        config_counts: dict[str, int] = {}
        config_conditions: dict[str, set[tuple[int, int]]] = {}
        for trial in clean_trials:
            config_id = trial["config"]["config_sha256"]
            config_counts[config_id] = config_counts.get(config_id, 0) + 1
            config_conditions.setdefault(config_id, set()).add(
                (trial["ladder_seed"], trial["optimiser_seed"])
            )
        if (
            observed_conditions != expected_conditions
            or len(config_counts) != 2
            or set(config_counts.values()) != {3}
            or any(conditions != expected_conditions for conditions in config_conditions.values())
        ):
            raise ConditionalExperimentContractError("confirmation design drifted")
        _require_sha256(source_manifest_sha256, where="source_manifest_sha256")
        _require_sha256(transition_receipt_sha256, where="transition_receipt_sha256")
        asha = None
    else:
        raise ValueError("run manifest phase must be sweep or confirmation")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_MANIFEST_KIND,
        "namespace": NAMESPACE,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "experiment_contract": clean_experiment,
        "phase": phase,
        "asha": asha,
        "experiment_artefacts": {"baseline_manifest": clean_baseline_manifest},
        "source_binding": {
            "source_manifest_sha256": source_manifest_sha256,
            "transition_receipt_sha256": transition_receipt_sha256,
        },
        "trials": clean_trials,
    }
    assert_metadata_only(body, where="conditional run manifest")
    return {**body, "phase_run_id": canonical_sha256(body)}


def validate_run_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "namespace",
        "experiment_run_id",
        "experiment_contract",
        "phase",
        "asha",
        "experiment_artefacts",
        "source_binding",
        "trials",
        "phase_run_id",
    }
    if set(value) != required:
        raise ValueError("run manifest has unexpected fields")
    source = value.get("source_binding")
    experiment_artefacts = value.get("experiment_artefacts")
    if not isinstance(experiment_artefacts, Mapping) or set(experiment_artefacts) != {
        "baseline_manifest"
    }:
        raise ValueError("run manifest experiment artefacts have unexpected fields")
    if not isinstance(source, Mapping) or set(source) != {
        "source_manifest_sha256",
        "transition_receipt_sha256",
    }:
        raise ValueError("run manifest source binding has unexpected fields")
    asha = value.get("asha")
    if asha is not None and (
        not isinstance(asha, Mapping)
        or set(asha) != {"rung_epochs", "candidate_count", "promotion_count"}
    ):
        raise ValueError("run manifest ASHA block has unexpected fields")
    trials = value.get("trials")
    if not isinstance(trials, list):
        raise ValueError("run manifest trials must be a list")
    rebuilt = build_run_manifest(
        value["experiment_contract"],
        phase=value["phase"],
        trials=trials,
        asha_rung_epochs=None if asha is None else asha["rung_epochs"],
        baseline_manifest=experiment_artefacts["baseline_manifest"],
        source_manifest_sha256=source["source_manifest_sha256"],
        transition_receipt_sha256=source["transition_receipt_sha256"],
    )
    if dict(value) != rebuilt:
        raise ConditionalExperimentContractError("run manifest content address drifted")
    return rebuilt


def _validate_artifacts(
    artifacts: Mapping[str, Any], *, required: frozenset[str] = frozenset()
) -> dict[str, Any]:
    if not isinstance(artifacts, Mapping) or not artifacts or not required <= set(artifacts):
        raise ValueError("artifact inventory is incomplete")
    clean: dict[str, Any] = {}
    for name, descriptor in artifacts.items():
        if not isinstance(name, str) or not name or not isinstance(descriptor, Mapping):
            raise ValueError("artifact inventory contains an invalid entry")
        clean[name] = _validate_artifact_descriptor(descriptor, where=f"artifacts.{name}")
    return clean


def build_checkpoint_payload(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    *,
    completed_epochs: int,
    global_step: int,
    optimiser_step: int,
    artifacts: Mapping[str, Any],
    aggregate_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze every state descriptor needed for exact continuation or crash resume."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_spec = validate_trial_spec(clean_experiment, trial_spec)
    epochs = _require_positive_int(completed_epochs, where="completed_epochs")
    if epochs > clean_spec["target_epochs"]:
        raise ConditionalExperimentContractError("checkpoint exceeds the trial epoch budget")
    clean_artifacts = _validate_artifacts(
        artifacts,
        required=frozenset({"model_state", "optimiser_state", "scheduler_state", "rng_state"}),
    )
    metrics = _json_clone(aggregate_metrics)
    assert_metadata_only(metrics, where="conditional checkpoint aggregate metrics")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "trial_id": clean_spec["trial_id"],
        "trial_spec_sha256": canonical_sha256(clean_spec),
        "config_sha256": clean_spec["config"]["config_sha256"],
        "subset_manifest_sha256": clean_spec["subset_manifest_sha256"],
        "completed_epochs": epochs,
        "global_step": _require_nonnegative_int(global_step, where="global_step"),
        "optimiser_step": _require_nonnegative_int(optimiser_step, where="optimiser_step"),
        "optimiser_seed": clean_spec["optimiser_seed"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "artifacts": clean_artifacts,
        "aggregate_metrics": metrics,
    }
    assert_metadata_only(body, where="conditional checkpoint")
    return {**body, "checkpoint_id": canonical_sha256(body)}


def validate_checkpoint_payload(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "trial_id",
        "trial_spec_sha256",
        "config_sha256",
        "subset_manifest_sha256",
        "completed_epochs",
        "global_step",
        "optimiser_step",
        "optimiser_seed",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "artifacts",
        "aggregate_metrics",
        "checkpoint_id",
    }
    if set(checkpoint) != required:
        raise ValueError("checkpoint payload has unexpected fields")
    rebuilt = build_checkpoint_payload(
        experiment,
        trial_spec,
        completed_epochs=checkpoint["completed_epochs"],
        global_step=checkpoint["global_step"],
        optimiser_step=checkpoint["optimiser_step"],
        artifacts=checkpoint["artifacts"],
        aggregate_metrics=checkpoint["aggregate_metrics"],
    )
    if dict(checkpoint) != rebuilt:
        raise ConditionalExperimentContractError("checkpoint content address or binding drifted")
    return rebuilt


def build_resume_binding(
    experiment: Mapping[str, Any],
    source_trial_spec: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_metadata_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an exact binding from one completed ASHA rung to the next."""

    clean_experiment = validate_experiment_contract(experiment)
    source = validate_trial_spec(clean_experiment, source_trial_spec)
    clean_checkpoint = validate_checkpoint_payload(clean_experiment, source, checkpoint)
    if source["phase"] != "asha" or clean_checkpoint["completed_epochs"] != source["target_epochs"]:
        raise ConditionalExperimentContractError("resume source is not a completed ASHA rung")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": RESUME_BINDING_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "source_trial_id": source["trial_id"],
        "source_trial_spec_sha256": canonical_sha256(source),
        "source_config_sha256": source["config"]["config_sha256"],
        "source_subset_manifest_sha256": source["subset_manifest_sha256"],
        "source_ladder_seed": source["ladder_seed"],
        "source_optimiser_seed": source["optimiser_seed"],
        "completed_epochs": clean_checkpoint["completed_epochs"],
        "checkpoint_id": clean_checkpoint["checkpoint_id"],
        "checkpoint_metadata_artifact": _validate_artifact_descriptor(
            checkpoint_metadata_artifact, where="checkpoint_metadata_artifact"
        ),
    }
    assert_metadata_only(body, where="conditional resume binding")
    return {**body, "resume_binding_id": canonical_sha256(body)}


def validate_resume_binding(
    experiment: Mapping[str, Any],
    source_trial_spec: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_resume_binding(
        experiment,
        source_trial_spec,
        checkpoint,
        checkpoint_metadata_artifact=binding.get("checkpoint_metadata_artifact", {}),
    )
    if dict(binding) != expected:
        raise ConditionalExperimentContractError("resume binding does not match its source")
    return expected


def _finite_vector(value: Any, *, length: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must contain exactly {length} logits")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{where} must contain finite numbers")
        clean = float(item)
        if not math.isfinite(clean):
            raise ValueError(f"{where} must contain finite numbers")
        result.append(clean)
    return result


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def decode_conditional_logits(
    relevance_logits: Sequence[float], target_state_logits: Sequence[Sequence[float]]
) -> tuple[dict[str, Any], bool]:
    """Apply the frozen relevance gate and material all-absent rule."""

    relevance = RELEVANCE_CLASSES[_argmax(relevance_logits)]
    if relevance != "material":
        return {"relevance": relevance, "target_stances": []}, False
    selected: list[dict[str, str]] = []
    for target, logits in zip(TARGETS, target_state_logits, strict=True):
        state_index = _argmax(logits)
        if state_index != 0:
            selected.append({"target": target, "stance": TARGET_STATES[state_index]})
    forced = not selected
    if forced:
        target_index, state_index = max(
            (
                (target_index, _argmax(logits[1:]) + 1)
                for target_index, logits in enumerate(target_state_logits)
            ),
            key=lambda pair: (
                target_state_logits[pair[0]][pair[1]] - target_state_logits[pair[0]][0],
                -pair[0],
            ),
        )
        selected = [
            {"target": TARGETS[target_index], "stance": TARGET_STATES[state_index]}
        ]
    return {"relevance": relevance, "target_stances": selected}, forced


def build_private_development_predictions(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the private, content-addressed 222-row development payload."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_spec = validate_trial_spec(clean_experiment, trial_spec)
    if len(rows) != DEVELOPMENT_ROWS:
        raise ValueError(f"private development payload must contain {DEVELOPMENT_ROWS} rows")
    clean_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "source_sample_id",
            "relevance_logits",
            "target_state_logits",
            "decoded_label",
            "forced_target_selection",
        }:
            raise ValueError("private development row schema drifted")
        item_id = raw["source_sample_id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            raise ValueError("private development IDs must be non-empty and unique")
        seen_ids.add(item_id)
        relevance_logits = _finite_vector(
            raw["relevance_logits"], length=3, where="relevance_logits"
        )
        state_rows = raw["target_state_logits"]
        if not isinstance(state_rows, list) or len(state_rows) != len(TARGETS):
            raise ValueError("target_state_logits must contain four target rows")
        target_state_logits = [
            _finite_vector(item, length=len(TARGET_STATES), where="target_state_logits")
            for item in state_rows
        ]
        decoded, forced = decode_conditional_logits(relevance_logits, target_state_logits)
        if raw["decoded_label"] != decoded or raw["forced_target_selection"] is not forced:
            raise ConditionalExperimentContractError("decoded development prediction drifted")
        clean_rows.append(
            {
                "source_sample_id": item_id,
                "relevance_logits": relevance_logits,
                "target_state_logits": target_state_logits,
                "decoded_label": decoded,
                "forced_target_selection": forced,
            }
        )
    clean_rows.sort(key=lambda row: row["source_sample_id"])
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_PREDICTIONS_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "trial_id": clean_spec["trial_id"],
        "development_reference_sha256": clean_experiment["bindings"][
            "development_reference_sha256"
        ],
        "row_count": DEVELOPMENT_ROWS,
        "rows": clean_rows,
    }
    return {**body, "prediction_set_id": canonical_sha256(body)}


def validate_private_development_predictions(
    value: Mapping[str, Any],
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "trial_id",
        "development_reference_sha256",
        "row_count",
        "rows",
        "prediction_set_id",
    }
    if set(value) != required:
        raise ValueError("private development payload has unexpected fields")
    rebuilt = build_private_development_predictions(
        experiment,
        trial_spec,
        rows=value["rows"],
    )
    if dict(value) != rebuilt:
        raise ConditionalExperimentContractError("private development payload binding drifted")
    return rebuilt


def _validate_public_metrics(value: Mapping[str, Any], *, target_epochs: int) -> dict[str, Any]:
    required = {
        "training_rows",
        "development_rows",
        "completed_epochs",
        "selected_epoch",
        "invalid_outputs",
        "relevance_macro_f1",
        "material_recall",
        "core_target_micro_f1",
        "core_target_stance_tuple_micro_f1",
        "composite",
        "forced_target_selections",
        "truncated_rows",
    }
    if set(value) != required:
        raise ValueError("public trial metrics have unexpected fields")
    clean = {
        "training_rows": _require_positive_int(value["training_rows"], where="training_rows"),
        "development_rows": _require_positive_int(
            value["development_rows"], where="development_rows"
        ),
        "completed_epochs": _require_positive_int(
            value["completed_epochs"], where="completed_epochs"
        ),
        "selected_epoch": _require_positive_int(value["selected_epoch"], where="selected_epoch"),
        "invalid_outputs": _require_nonnegative_int(
            value["invalid_outputs"], where="invalid_outputs"
        ),
        "relevance_macro_f1": _require_probability(
            value["relevance_macro_f1"], where="relevance_macro_f1"
        ),
        "material_recall": _require_probability(
            value["material_recall"], where="material_recall"
        ),
        "core_target_micro_f1": _require_probability(
            value["core_target_micro_f1"], where="core_target_micro_f1"
        ),
        "core_target_stance_tuple_micro_f1": _require_probability(
            value["core_target_stance_tuple_micro_f1"],
            where="core_target_stance_tuple_micro_f1",
        ),
        "composite": _require_probability(value["composite"], where="composite"),
        "forced_target_selections": _require_nonnegative_int(
            value["forced_target_selections"], where="forced_target_selections"
        ),
        "truncated_rows": _require_nonnegative_int(
            value["truncated_rows"], where="truncated_rows"
        ),
    }
    if clean["development_rows"] != DEVELOPMENT_ROWS:
        raise ConditionalExperimentContractError("development row count drifted")
    if not clean["selected_epoch"] <= clean["completed_epochs"] <= target_epochs:
        raise ConditionalExperimentContractError("trial epoch metrics drifted")
    expected_composite = (
        0.25 * clean["relevance_macro_f1"]
        + 0.25 * clean["core_target_micro_f1"]
        + 0.50 * clean["core_target_stance_tuple_micro_f1"]
    )
    if not math.isclose(clean["composite"], expected_composite, abs_tol=1e-12):
        raise ConditionalExperimentContractError("development composite was not recomputed exactly")
    assert_metadata_only(clean, where="conditional trial metrics")
    return clean


def build_trial_receipt(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    aggregate_metrics: Mapping[str, Any],
    gpu_type: str,
    wall_seconds: int | float,
    gpu_seconds: int | float,
) -> dict[str, Any]:
    """Build one complete metadata-only trial receipt."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_spec = validate_trial_spec(clean_experiment, trial_spec)
    clean_checkpoint = validate_checkpoint_payload(clean_experiment, clean_spec, checkpoint)
    if gpu_type != clean_spec["gpu_type"]:
        raise ConditionalExperimentContractError("observed GPU differs from the trial spec")
    clean_gpu_seconds = _require_nonnegative_number(gpu_seconds, where="gpu_seconds")
    clean_wall_seconds = _require_nonnegative_number(wall_seconds, where="wall_seconds")
    if clean_gpu_seconds > clean_spec["max_gpu_seconds"]:
        raise ConditionalExperimentContractError("trial exceeded its GPU-second reservation")
    clean_artifacts = _validate_artifacts(
        artifacts,
        required=frozenset({"checkpoint_metadata", "metrics", "development_predictions"}),
    )
    metrics = _validate_public_metrics(
        aggregate_metrics, target_epochs=clean_spec["target_epochs"]
    )
    rate = Decimal(clean_experiment["budget"]["rate_card_usd_per_gpu_second"][gpu_type])
    cost = rate * Decimal(str(clean_gpu_seconds))
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_RECEIPT_KIND,
        "status": "complete",
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "trial_id": clean_spec["trial_id"],
        "trial_spec_sha256": canonical_sha256(clean_spec),
        "checkpoint_id": clean_checkpoint["checkpoint_id"],
        "bindings": deepcopy(clean_experiment["bindings"]),
        "artifacts": clean_artifacts,
        "aggregate_metrics": metrics,
        "compute": {
            "gpu_type": gpu_type,
            "wall_seconds": clean_wall_seconds,
            "gpu_seconds": clean_gpu_seconds,
            "estimated_cost_usd": _money(cost),
        },
    }
    assert_metadata_only(body, where="conditional trial receipt")
    return {**body, "receipt_id": canonical_sha256(body)}


def validate_trial_receipt(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "status",
        "experiment_run_id",
        "trial_id",
        "trial_spec_sha256",
        "checkpoint_id",
        "bindings",
        "artifacts",
        "aggregate_metrics",
        "compute",
        "receipt_id",
    }
    if set(receipt) != required:
        raise ValueError("trial receipt has unexpected fields")
    compute = receipt.get("compute")
    if not isinstance(compute, Mapping) or set(compute) != {
        "gpu_type",
        "wall_seconds",
        "gpu_seconds",
        "estimated_cost_usd",
    }:
        raise ValueError("trial receipt compute block has unexpected fields")
    rebuilt = build_trial_receipt(
        experiment,
        trial_spec,
        checkpoint=checkpoint,
        artifacts=receipt["artifacts"],
        aggregate_metrics=receipt["aggregate_metrics"],
        gpu_type=compute["gpu_type"],
        wall_seconds=compute["wall_seconds"],
        gpu_seconds=compute["gpu_seconds"],
    )
    if dict(receipt) != rebuilt:
        raise ConditionalExperimentContractError("trial receipt content address drifted")
    return rebuilt


def budget_status(
    experiment: Mapping[str, Any],
    *,
    completed_receipts: Sequence[Mapping[str, Any]] = (),
    active_trial_specs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Account exact completed spend and worst-case active reservations."""

    clean_experiment = validate_experiment_contract(experiment)
    completed_cost = Decimal("0")
    seen_receipts: set[str] = set()
    seen_trials: set[str] = set()
    rate_card = clean_experiment["budget"]["rate_card_usd_per_gpu_second"]
    for receipt in completed_receipts:
        receipt_id = _require_sha256(receipt.get("receipt_id"), where="receipt.receipt_id")
        trial_id = _require_sha256(receipt.get("trial_id"), where="receipt.trial_id")
        if receipt_id in seen_receipts or trial_id in seen_trials:
            raise ValueError("completed receipts contain a duplicate")
        seen_receipts.add(receipt_id)
        seen_trials.add(trial_id)
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        if (
            canonical_sha256(body) != receipt_id
            or receipt.get("status") != "complete"
            or receipt.get("experiment_run_id") != clean_experiment["experiment_run_id"]
        ):
            raise ConditionalExperimentContractError("completed receipt binding drifted")
        compute = receipt.get("compute")
        if not isinstance(compute, Mapping) or set(compute) != {
            "gpu_type",
            "wall_seconds",
            "gpu_seconds",
            "estimated_cost_usd",
        }:
            raise ValueError("completed receipt compute block is invalid")
        gpu_type = compute["gpu_type"]
        if gpu_type not in rate_card:
            raise ConditionalExperimentContractError("completed receipt uses an unknown GPU")
        seconds = _require_nonnegative_number(
            compute["gpu_seconds"], where="receipt.compute.gpu_seconds"
        )
        measured = Decimal(rate_card[gpu_type]) * Decimal(str(seconds))
        if _money(measured) != compute["estimated_cost_usd"]:
            raise ConditionalExperimentContractError("completed receipt cost drifted")
        completed_cost += measured

    reserved_cost = Decimal("0")
    active_ids: set[str] = set()
    for spec in active_trial_specs:
        clean_spec = validate_trial_spec(clean_experiment, spec)
        trial_id = clean_spec["trial_id"]
        if trial_id in active_ids or trial_id in seen_trials:
            raise ValueError("trial is duplicated across active and completed inventory")
        active_ids.add(trial_id)
        reserved_cost += Decimal(str(clean_spec["reserved_cost_usd"]))
    cap = Decimal(clean_experiment["budget"]["hard_cost_cap_usd"])
    projected = completed_cost + reserved_cost
    if projected > cap:
        raise ConditionalExperimentContractError("projected experiment spend exceeds the cap")
    return {
        "hard_cost_cap_usd": _money(cap),
        "completed_cost_usd": _money(completed_cost),
        "active_reserved_cost_usd": _money(reserved_cost),
        "projected_cost_usd": _money(projected),
        "remaining_cost_usd": _money(cap - projected),
        "completed_trials": len(seen_trials),
        "active_trials": len(active_ids),
    }


def _validate_diagnostic_cell(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if set(value) != {"precision", "recall", "f1", "reference_support"}:
        raise ValueError(f"{where} diagnostic schema drifted")
    return {
        "precision": _require_probability(value["precision"], where=f"{where}.precision"),
        "recall": _require_probability(value["recall"], where=f"{where}.recall"),
        "f1": _require_probability(value["f1"], where=f"{where}.f1"),
        "reference_support": _require_nonnegative_int(
            value["reference_support"], where=f"{where}.reference_support"
        ),
    }


def _validate_metric_payload(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    required = {
        "core_target_stance_tuple_micro_f1",
        "material_recall",
        "core_target_f1",
        "core_target_reference_support",
        "stance_diagnostics",
        "target_stance_diagnostics",
        "invalid_outputs",
    }
    if set(value) != required:
        raise ValueError(f"{where} metric payload schema drifted")
    core_target_f1 = value["core_target_f1"]
    core_support = value["core_target_reference_support"]
    stance = value["stance_diagnostics"]
    target_stance = value["target_stance_diagnostics"]
    if not all(isinstance(block, Mapping) for block in (core_target_f1, core_support, stance)):
        raise ValueError(f"{where} metric mappings are invalid")
    if set(core_target_f1) != set(CORE_TARGETS) or set(core_support) != set(CORE_TARGETS):
        raise ValueError(f"{where} core target inventory drifted")
    if set(stance) != set(TARGET_STATES):
        raise ValueError(f"{where} stance diagnostic inventory drifted")
    if not isinstance(target_stance, Mapping) or set(target_stance) != set(TARGETS):
        raise ValueError(f"{where} target-by-stance inventory drifted")
    clean_target_stance: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        target_cells = target_stance[target]
        if not isinstance(target_cells, Mapping) or set(target_cells) != set(TARGET_STATES):
            raise ValueError(f"{where}.{target} stance inventory drifted")
        clean_target_stance[target] = {
            state: _validate_diagnostic_cell(
                target_cells[state], where=f"{where}.target_stance.{target}.{state}"
            )
            for state in TARGET_STATES
        }
    clean_stance = {
        state: _validate_diagnostic_cell(
            stance[state], where=f"{where}.stance_diagnostics.{state}"
        )
        for state in TARGET_STATES
    }
    for state in TARGET_STATES:
        if clean_stance[state]["reference_support"] != sum(
            clean_target_stance[target][state]["reference_support"] for target in TARGETS
        ):
            raise ConditionalExperimentContractError(
                f"{where} aggregate stance support does not conserve target slots"
            )
    clean_core_support = {
        target: _require_nonnegative_int(core_support[target], where=f"{where}.{target}.support")
        for target in CORE_TARGETS
    }
    for target in CORE_TARGETS:
        expected_support = sum(
            clean_target_stance[target][state]["reference_support"]
            for state in TARGET_STATES
            if state != "absent"
        )
        if clean_core_support[target] != expected_support:
            raise ConditionalExperimentContractError(
                f"{where} core target support does not conserve present states"
            )
    return {
        "core_target_stance_tuple_micro_f1": _require_probability(
            value["core_target_stance_tuple_micro_f1"],
            where=f"{where}.core_target_stance_tuple_micro_f1",
        ),
        "material_recall": _require_probability(
            value["material_recall"], where=f"{where}.material_recall"
        ),
        "core_target_f1": {
            target: _require_probability(core_target_f1[target], where=f"{where}.{target}.f1")
            for target in CORE_TARGETS
        },
        "core_target_reference_support": clean_core_support,
        "stance_diagnostics": clean_stance,
        "target_stance_diagnostics": clean_target_stance,
        "invalid_outputs": _require_nonnegative_int(
            value["invalid_outputs"], where=f"{where}.invalid_outputs"
        ),
    }


def _validate_bound_gate_results(
    values: Sequence[Mapping[str, Any]],
    *,
    where: str,
    expected_experiment_run_id: str | None,
    expected_phase_run_id: str | None,
    expected_config_sha256: str | None,
) -> dict[str, dict[str, Any]]:
    expected_conditions = {
        f"{row['ladder_seed']}:{row['optimiser_seed']}": dict(row)
        for row in CONFIRMATION_CONDITIONS
    }
    expected_fields = {
        "source_experiment_run_id",
        "source_phase_run_id",
        "config_sha256",
        "trial_id",
        "receipt_id",
        "prediction_sha256",
        "condition",
        "metric_payload",
        "metric_payload_sha256",
    }
    clean: dict[str, dict[str, Any]] = {}
    seen_by_field = {field: set() for field in ("trial_id", "receipt_id", "prediction_sha256")}
    for raw in values:
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError(f"{where} bound result schema drifted")
        condition = raw["condition"]
        if not isinstance(condition, Mapping) or set(condition) != {
            "ladder_seed",
            "optimiser_seed",
        }:
            raise ValueError(f"{where} condition schema drifted")
        pair_id = f"{condition['ladder_seed']}:{condition['optimiser_seed']}"
        if pair_id not in expected_conditions or dict(condition) != expected_conditions[pair_id]:
            raise ConditionalExperimentContractError(f"{where} condition drifted")
        if pair_id in clean:
            raise ValueError(f"{where} contains a duplicate paired condition")
        digests = {
            field: _require_sha256(raw[field], where=f"{where}.{field}")
            for field in (
                "source_experiment_run_id",
                "source_phase_run_id",
                "config_sha256",
                "trial_id",
                "receipt_id",
                "prediction_sha256",
                "metric_payload_sha256",
            )
        }
        for field in seen_by_field:
            if digests[field] in seen_by_field[field]:
                raise ValueError(f"{where} repeats {field}")
            seen_by_field[field].add(digests[field])
        if (
            (expected_experiment_run_id is not None
             and digests["source_experiment_run_id"] != expected_experiment_run_id)
            or (expected_phase_run_id is not None
                and digests["source_phase_run_id"] != expected_phase_run_id)
            or (expected_config_sha256 is not None
                and digests["config_sha256"] != expected_config_sha256)
        ):
            raise ConditionalExperimentContractError(f"{where} source binding drifted")
        metric_payload = _validate_metric_payload(
            raw["metric_payload"], where=f"{where}.{pair_id}"
        )
        if digests["metric_payload_sha256"] != canonical_sha256(metric_payload):
            raise ConditionalExperimentContractError(f"{where} metric payload digest drifted")
        clean[pair_id] = {
            **digests,
            "condition": dict(condition),
            "metric_payload": metric_payload,
        }
    if set(clean) != set(expected_conditions):
        raise ValueError(f"{where} does not exactly cover the three paired conditions")
    for field in ("source_experiment_run_id", "source_phase_run_id", "config_sha256"):
        if len({row[field] for row in clean.values()}) != 1:
            raise ConditionalExperimentContractError(f"{where} does not share one {field}")
    return clean


def _require_matching_reference_support(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> None:
    pair_ids = sorted(baseline)
    reference = baseline[pair_ids[0]]["metric_payload"]
    for pair_id in pair_ids:
        baseline_metrics = baseline[pair_id]["metric_payload"]
        candidate_metrics = candidate[pair_id]["metric_payload"]
        if baseline_metrics["core_target_reference_support"] != reference[
            "core_target_reference_support"
        ] or candidate_metrics["core_target_reference_support"] != reference[
            "core_target_reference_support"
        ]:
            raise ConditionalExperimentContractError("core target reference support drifted")
        for state in TARGET_STATES:
            expected = reference["stance_diagnostics"][state]["reference_support"]
            if (
                baseline_metrics["stance_diagnostics"][state]["reference_support"] != expected
                or candidate_metrics["stance_diagnostics"][state]["reference_support"]
                != expected
            ):
                raise ConditionalExperimentContractError("aggregate stance support drifted")
        for target in TARGETS:
            for state in TARGET_STATES:
                expected = reference["target_stance_diagnostics"][target][state][
                    "reference_support"
                ]
                if (
                    baseline_metrics["target_stance_diagnostics"][target][state][
                        "reference_support"
                    ]
                    != expected
                    or candidate_metrics["target_stance_diagnostics"][target][state][
                        "reference_support"
                    ]
                    != expected
                ):
                    raise ConditionalExperimentContractError(
                        "target-by-stance reference support drifted"
                    )


def _public_evidence_bindings(
    values: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "condition": deepcopy(values[pair_id]["condition"]),
            "source_experiment_run_id": values[pair_id]["source_experiment_run_id"],
            "source_phase_run_id": values[pair_id]["source_phase_run_id"],
            "config_sha256": values[pair_id]["config_sha256"],
            "trial_id": values[pair_id]["trial_id"],
            "receipt_id": values[pair_id]["receipt_id"],
            "prediction_sha256": values[pair_id]["prediction_sha256"],
            "metric_payload_sha256": values[pair_id]["metric_payload_sha256"],
        }
        for pair_id in sorted(values)
    ]


def evaluate_continuation_gate(
    experiment: Mapping[str, Any],
    *,
    phase_run_id: str,
    candidate_config_sha256: str,
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Gate one confirmed recipe using only exact, content-addressed evidence.

    Every metric payload is bound to its experiment, phase, config, paired
    condition, trial receipt, and private prediction digest. Unbound numeric
    dictionaries are therefore not a valid input to scientific selection.
    """

    clean_experiment = validate_experiment_contract(experiment)
    phase_id = _require_sha256(phase_run_id, where="phase_run_id")
    config_id = _require_sha256(candidate_config_sha256, where="candidate_config_sha256")
    candidate = _validate_bound_gate_results(
        candidate_results,
        where="candidate",
        expected_experiment_run_id=clean_experiment["experiment_run_id"],
        expected_phase_run_id=phase_id,
        expected_config_sha256=config_id,
    )
    baseline = _validate_bound_gate_results(
        baseline_results,
        where="baseline",
        expected_experiment_run_id=None,
        expected_phase_run_id=None,
        expected_config_sha256=None,
    )
    _require_matching_reference_support(baseline, candidate)
    pair_ids = sorted(baseline)
    metric = "core_target_stance_tuple_micro_f1"
    tuple_deltas = [
        candidate[pair_id]["metric_payload"][metric]
        - baseline[pair_id]["metric_payload"][metric]
        for pair_id in pair_ids
    ]
    mean_gain = sum(tuple_deltas) / len(tuple_deltas)
    improved_pairs = sum(delta > 0 for delta in tuple_deltas)
    mean_baseline_recall = sum(
        baseline[pair_id]["metric_payload"]["material_recall"] for pair_id in pair_ids
    ) / 3
    mean_candidate_recall = sum(
        candidate[pair_id]["metric_payload"]["material_recall"] for pair_id in pair_ids
    ) / 3
    recall_decline = mean_baseline_recall - mean_candidate_recall
    target_deltas = {
        target: sum(
            candidate[pair_id]["metric_payload"]["core_target_f1"][target]
            - baseline[pair_id]["metric_payload"]["core_target_f1"][target]
            for pair_id in pair_ids
        )
        / 3
        for target in CORE_TARGETS
    }
    reference = baseline[pair_ids[0]]["metric_payload"]
    support = reference["core_target_reference_support"]
    criteria = _continuation_criteria()
    supported_target_regressions_pass = all(
        delta >= -criteria["maximum_supported_core_target_regression"]
        for target, delta in target_deltas.items()
        if support[target] >= criteria["minimum_reference_positives_for_supported_target"]
    )
    stance_diagnostics = {
        state: {
            "reference_support": reference["stance_diagnostics"][state]["reference_support"],
            "baseline_mean_precision": sum(
                baseline[pair_id]["metric_payload"]["stance_diagnostics"][state]["precision"]
                for pair_id in pair_ids
            )
            / 3,
            "candidate_mean_precision": sum(
                candidate[pair_id]["metric_payload"]["stance_diagnostics"][state]["precision"]
                for pair_id in pair_ids
            )
            / 3,
            "baseline_mean_recall": sum(
                baseline[pair_id]["metric_payload"]["stance_diagnostics"][state]["recall"]
                for pair_id in pair_ids
            )
            / 3,
            "candidate_mean_recall": sum(
                candidate[pair_id]["metric_payload"]["stance_diagnostics"][state]["recall"]
                for pair_id in pair_ids
            )
            / 3,
            "baseline_mean_f1": sum(
                baseline[pair_id]["metric_payload"]["stance_diagnostics"][state]["f1"]
                for pair_id in pair_ids
            )
            / 3,
            "candidate_mean_f1": sum(
                candidate[pair_id]["metric_payload"]["stance_diagnostics"][state]["f1"]
                for pair_id in pair_ids
            )
            / 3,
        }
        for state in TARGET_STATES
    }
    supported_cells: list[dict[str, Any]] = []
    cell_breaches: list[str] = []
    for target in TARGETS:
        for state in TARGET_STATES:
            cell = reference["target_stance_diagnostics"][target][state]
            if cell["reference_support"] < criteria[
                "minimum_reference_positives_for_supported_target"
            ]:
                continue
            baseline_mean = sum(
                baseline[pair_id]["metric_payload"]["target_stance_diagnostics"][target][
                    state
                ]["f1"]
                for pair_id in pair_ids
            ) / 3
            candidate_mean = sum(
                candidate[pair_id]["metric_payload"]["target_stance_diagnostics"][target][
                    state
                ]["f1"]
                for pair_id in pair_ids
            ) / 3
            delta = candidate_mean - baseline_mean
            cell_id = f"{target}:{state}"
            supported_cells.append(
                {
                    "cell": cell_id,
                    "reference_support": cell["reference_support"],
                    "baseline_mean_f1": baseline_mean,
                    "candidate_mean_f1": candidate_mean,
                    "delta": delta,
                }
            )
            if delta < -criteria["maximum_supported_target_stance_regression"]:
                cell_breaches.append(cell_id)
    invalid_outputs = sum(
        baseline[pair_id]["metric_payload"]["invalid_outputs"]
        + candidate[pair_id]["metric_payload"]["invalid_outputs"]
        for pair_id in pair_ids
    )
    guardrails = {
        "mean_gain_pass": mean_gain >= criteria["minimum_mean_gain"],
        "paired_improvement_pass": improved_pairs >= criteria["minimum_improved_pairs"],
        "material_recall_pass": recall_decline
        <= criteria["maximum_mean_material_recall_decline"],
        "supported_target_regression_pass": supported_target_regressions_pass,
        "supported_target_stance_regression_pass": not cell_breaches,
        "invalid_outputs_pass": invalid_outputs <= criteria["maximum_invalid_outputs"],
    }
    if mean_gain < criteria["gain_bands"]["scrap_below"]:
        verdict = "scrap"
    elif mean_gain < criteria["gain_bands"]["inconclusive_below"]:
        verdict = "inconclusive"
    elif all(guardrails.values()):
        verdict = "promote"
    else:
        verdict = "reject_guardrail"
    candidate_evidence = _public_evidence_bindings(candidate)
    baseline_evidence = _public_evidence_bindings(baseline)
    metric_payload_set_sha256 = canonical_sha256(
        {
            "candidate": [row["metric_payload_sha256"] for row in candidate_evidence],
            "baseline": [row["metric_payload_sha256"] for row in baseline_evidence],
        }
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTINUATION_GATE_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase_run_id": phase_id,
        "development_reference_sha256": clean_experiment["bindings"][
            "development_reference_sha256"
        ],
        "candidate_config_sha256": config_id,
        "baseline_config_sha256": baseline_evidence[0]["config_sha256"],
        "candidate_evidence": candidate_evidence,
        "baseline_evidence": baseline_evidence,
        "metric_payload_set_sha256": metric_payload_set_sha256,
        "criteria": criteria,
        "paired_tuple_deltas": tuple_deltas,
        "mean_core_tuple_f1_gain": mean_gain,
        "improved_pair_count": improved_pairs,
        "mean_material_recall_decline": recall_decline,
        "core_target_f1_deltas": target_deltas,
        "core_target_reference_support": support,
        "stance_diagnostics": stance_diagnostics,
        "supported_target_stance_cells": supported_cells,
        "target_stance_regression_breaches": cell_breaches,
        "invalid_outputs": invalid_outputs,
        "guardrails": guardrails,
        "verdict": verdict,
    }
    assert_metadata_only(body, where="conditional continuation gate")
    return {**body, "gate_receipt_id": canonical_sha256(body)}


def select_confirmation_recipe(
    experiment: Mapping[str, Any],
    *,
    phase_run_id: str,
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results_by_config: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Select only from two fully bound, independently gated confirmations."""

    clean_experiment = validate_experiment_contract(experiment)
    phase_id = _require_sha256(phase_run_id, where="phase_run_id")
    if len(candidate_results_by_config) != 2:
        raise ValueError("confirmation selection requires exactly two candidate recipes")
    gates = [
        evaluate_continuation_gate(
            clean_experiment,
            phase_run_id=phase_id,
            candidate_config_sha256=config_id,
            baseline_results=baseline_results,
            candidate_results=results,
        )
        for config_id, results in sorted(candidate_results_by_config.items())
    ]
    baseline_evidence = gates[0]["baseline_evidence"]
    if any(gate["baseline_evidence"] != baseline_evidence for gate in gates[1:]):
        raise ConditionalExperimentContractError("candidate gates use different baseline evidence")
    all_evidence = [
        *baseline_evidence,
        *(row for gate in gates for row in gate["candidate_evidence"]),
    ]
    for field in ("trial_id", "receipt_id", "prediction_sha256"):
        values = [row[field] for row in all_evidence]
        if len(values) != len(set(values)):
            raise ConditionalExperimentContractError(
                f"selection evidence reuses a bound {field}"
            )
    candidates = [
        {
            "config_sha256": gate["candidate_config_sha256"],
            "verdict": gate["verdict"],
            "mean_core_tuple_f1_gain": gate["mean_core_tuple_f1_gain"],
            "gate_receipt_id": gate["gate_receipt_id"],
            "metric_payload_set_sha256": gate["metric_payload_set_sha256"],
            "candidate_evidence": gate["candidate_evidence"],
        }
        for gate in gates
    ]
    eligible = [row for row in candidates if row["verdict"] == "promote"]
    selected = (
        sorted(
            eligible,
            key=lambda row: (-row["mean_core_tuple_f1_gain"], row["config_sha256"]),
        )[0]["config_sha256"]
        if eligible
        else None
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-conditional-state-recipe-selection-v1",
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase_run_id": phase_id,
        "development_reference_sha256": clean_experiment["bindings"][
            "development_reference_sha256"
        ],
        "selection_rule": "highest_passing_mean_gain_then_config_sha256",
        "baseline_evidence": baseline_evidence,
        "candidates": candidates,
        "metric_payload_digest": canonical_sha256(
            [row["metric_payload_set_sha256"] for row in candidates]
        ),
        "selected_config_sha256": selected,
        "selection_verdict": "promote" if selected is not None else "no_promotion",
    }
    assert_metadata_only(body, where="conditional recipe selection")
    return {**body, "selection_receipt_id": canonical_sha256(body)}


def experiment_output_root(volume_root: Path, experiment: Mapping[str, Any]) -> Path:
    clean = validate_experiment_contract(experiment)
    return volume_root / OUTPUT_PREFIX / f"run={clean['experiment_run_id']}"


def trial_output_root(
    volume_root: Path, experiment: Mapping[str, Any], trial_spec: Mapping[str, Any]
) -> Path:
    clean_experiment = validate_experiment_contract(experiment)
    clean_spec = validate_trial_spec(clean_experiment, trial_spec)
    return (
        experiment_output_root(volume_root, clean_experiment)
        / f"phase={clean_spec['phase']}"
        / f"trial={clean_spec['trial_id']}"
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        return (payload + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("publication must be finite JSON") from exc


def _require_namespaced_path(path: Path) -> None:
    if NAMESPACE not in path.parts:
        raise ValueError(f"conditional publication path must be under {NAMESPACE}")


def publication_state(path: Path, expected: Mapping[str, Any]) -> str:
    """Return missing, resume_exact, or complete; reject any immutable drift."""

    _require_namespaced_path(path)
    payload = _canonical_json_bytes(expected)
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ConditionalExperimentContractError(f"immutable publication differs: {path}")
        if incomplete.exists():
            raise ConditionalExperimentContractError(
                f"stale incomplete publication remains: {incomplete}"
            )
        return "complete"
    if incomplete.exists():
        if not incomplete.is_file() or incomplete.read_bytes() != payload:
            raise ConditionalExperimentContractError(
                f"incomplete publication cannot resume exactly: {incomplete}"
            )
        return "resume_exact"
    return "missing"


def publish_immutable_json(
    path: Path, value: Mapping[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    """Atomically publish exact JSON; resume only a byte-identical partial."""

    payload = _canonical_json_bytes(value)
    state = publication_state(path, value)
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    if state == "resume_exact":
        if not resume:
            raise ConditionalExperimentContractError(
                "exact incomplete publication requires explicit resume"
            )
        os.replace(incomplete, path)
    elif state == "missing":
        with incomplete.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(incomplete, path)
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def validate_immutable_publication(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if publication_state(path, expected) != "complete":
        raise ConditionalExperimentContractError("immutable publication is not complete")
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


__all__ = [
    "ASHA_RUNGS",
    "CONFIRMATION_CONDITIONS",
    "CORE_TARGETS",
    "DEVELOPMENT_ROWS",
    "NAMESPACE",
    "OUTPUT_PREFIX",
    "PRIVATE_PREDICTIONS_KIND",
    "RELEVANCE_CLASSES",
    "TARGETS",
    "TARGET_STATES",
    "ConditionalExperimentContractError",
    "budget_status",
    "build_asha_plan",
    "build_checkpoint_payload",
    "build_private_development_predictions",
    "build_resume_binding",
    "build_run_manifest",
    "build_trial_receipt",
    "canonical_sha256",
    "decode_conditional_logits",
    "evaluate_continuation_gate",
    "experiment_output_root",
    "freeze_experiment_contract",
    "freeze_trial_spec",
    "frozen_trial_configs",
    "publication_state",
    "publish_immutable_json",
    "select_asha_promotions",
    "select_confirmation_recipe",
    "trial_output_root",
    "validate_asha_plan",
    "validate_checkpoint_payload",
    "validate_experiment_contract",
    "validate_immutable_publication",
    "validate_private_development_predictions",
    "validate_resume_binding",
    "validate_run_manifest",
    "validate_trial_receipt",
    "validate_trial_spec",
]
