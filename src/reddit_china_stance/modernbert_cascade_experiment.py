"""Frozen contracts for the separate-model ModernBERT cascade experiment.

The experiment deliberately trains two independent encoders: a relevance model and a
target-conditioned six-state model.  This module contains no Torch, Transformers, or Modal
imports.  It owns content-addressed, metadata-only experiment/run/receipt contracts plus the
private prediction envelope used by the runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from reddit_china_stance.modernbert_training import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    MODEL_ID,
    MODEL_REVISION,
    TEACHER_GENERATION_RUN_ID,
    TOKENIZER_REVISION,
    canonical_sha256,
)
from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
NAMESPACE = "student-modernbert-cascade-v1"
OUTPUT_PREFIX = Path(NAMESPACE)

EXPERIMENT_KIND = "modernbert-separate-model-cascade-experiment-v1"
TRIAL_SPEC_KIND = "modernbert-cascade-trial-spec-v1"
RUN_MANIFEST_KIND = "modernbert-cascade-run-manifest-v1"
PRIVATE_PREDICTIONS_KIND = "modernbert-cascade-private-development-predictions-v1"
TRIAL_RECEIPT_KIND = "modernbert-cascade-trial-receipt-v1"
CONTINUATION_GATE_KIND = "modernbert-cascade-continuation-gate-v1"

COMPONENTS = ("relevance", "target_conditioned")
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
PAIRED_CONDITIONS = (
    {"ladder_seed": 101, "optimiser_seed": 47},
    {"ladder_seed": 202, "optimiser_seed": 61},
    {"ladder_seed": 303, "optimiser_seed": 89},
)

DEVELOPMENT_ROWS = 222
RELEVANCE_TRAINING_ROWS = 10_000
MATERIAL_TRAINING_ROWS = 6_436
TARGET_TRAINING_ROWS = 25_744
EXCLUDED_NON_MATERIAL_TARGET_SLOTS = 14_256
TARGET_STATE_COUNTS = {
    "absent": 17_896,
    "negative": 2_235,
    "mixed": 172,
    "no_directed_stance": 4_042,
    "positive": 1_378,
    "unclear": 21,
}
BASELINE_CONFIG_SHA256 = "22607fc61cf6ee0256eb169e5f89319386a4aacd214c0372490eb2b21712f705"
HARD_COST_CAP_USD = Decimal("20")


class CascadeExperimentContractError(RuntimeError):
    """Raised when an immutable cascade experiment contract drifts."""


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc


def _sha(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _probability(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a probability")
    clean = float(value)
    if not math.isfinite(clean) or not 0 <= clean <= 1:
        raise ValueError(f"{where} must be a probability")
    return clean


def _nonnegative_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a non-negative finite number")
    clean = float(value)
    if not math.isfinite(clean) or clean < 0:
        raise ValueError(f"{where} must be a non-negative finite number")
    return clean


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001")))


def _repo_artifact(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if set(value) != {"repo_relative_path", "sha256", "bytes"}:
        raise ValueError(f"{where} has unexpected fields")
    relative = value["repo_relative_path"]
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{where}.repo_relative_path must be a non-empty POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"{where}.repo_relative_path must be safe and relative")
    return {
        "repo_relative_path": relative,
        "sha256": _sha(value["sha256"], where=f"{where}.sha256"),
        "bytes": _positive_int(value["bytes"], where=f"{where}.bytes"),
    }


def _artifact(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if set(value) != {"relative_path", "sha256", "bytes"}:
        raise ValueError(f"{where} has unexpected fields")
    relative = value["relative_path"]
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{where}.relative_path must be non-empty")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{where}.relative_path must be safe and relative")
    return {
        "relative_path": relative,
        "sha256": _sha(value["sha256"], where=f"{where}.sha256"),
        "bytes": _positive_int(value["bytes"], where=f"{where}.bytes"),
    }


def frozen_component_config(component: str) -> dict[str, Any]:
    """Return the one registered configuration for a cascade component."""

    if component not in COMPONENTS:
        raise ValueError("component must be relevance or target_conditioned")
    common = {
        "encoder_learning_rate": 5e-5,
        "head_learning_rate_multiplier": 5.0,
        "dropout": 0.1,
        "effective_batch_size": 32,
        "precision": "bf16",
        "optimiser": "adamw",
        "weight_decay": 0.01,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "scheduler": "linear_decay",
        "gradient_clip": 1.0,
        "max_length": 768,
        "pooling": "masked_mean",
        "attention_implementation": "sdpa",
        "reference_compile": False,
    }
    if component == "relevance":
        body = {
            **common,
            "component": component,
            "classes": list(RELEVANCE_CLASSES),
            "class_weights": "capped_inverse_sqrt",
            "max_epochs": 6,
            "checkpoint_score": {
                "relevance_macro_f1": 0.5,
                "material_recall": 0.5,
            },
        }
    else:
        body = {
            **common,
            "component": component,
            "classes": list(TARGET_STATES),
            "class_weights": "none",
            "max_epochs": 8,
            "checkpoint_score": "core_target_stance_tuple_micro_f1_gold_relevance",
        }
    return {**body, "config_sha256": canonical_sha256(body)}


def _architecture_contract() -> dict[str, Any]:
    return {
        "encoders": "two_separately_instantiated_modernbert_large_models",
        "relevance_model": {
            "input": "target_text [SEP] parent_context [SEP] submission_context",
            "classes": list(RELEVANCE_CLASSES),
        },
        "target_conditioned_model": {
            "input": (
                "[TARGET={registered_target}] [SEP] target_text [SEP] parent_context "
                "[SEP] submission_context"
            ),
            "registered_targets": list(TARGETS),
            "classes": list(TARGET_STATES),
            "training_scope": "four_target_slots_for_teacher_material_rows_only",
        },
        "cascade_decoder": {
            "not_material_or_unclear": "emit_no_targets",
            "material": "run_all_four_target_conditioned_inputs",
            "present_state_rule": "state_is_not_absent",
            "material_all_absent_rule": "highest_best_non_absent_minus_absent_margin",
            "forced_target_count_reported": True,
        },
    }


def continuation_criteria() -> dict[str, Any]:
    """Return the frozen paired end-to-end decision gates."""

    return {
        "primary_metric": "core_target_stance_tuple_micro_f1",
        "minimum_mean_paired_gain": 0.03,
        "minimum_improved_pairs": 2,
        "paired_condition_count": 3,
        "maximum_mean_material_recall_decline": 0.02,
        "maximum_mean_relevance_macro_f1_decline": 0.01,
        "maximum_supported_core_target_regression": 0.05,
        "maximum_supported_target_stance_regression": 0.10,
        "minimum_reference_support": 10,
        "maximum_invalid_outputs": 0,
        "guard_aggregation": "mean_paired_f1_by_reference_cell",
        "gain_bands": {"scrap_below": 0.01, "inconclusive_below": 0.03},
        "passing_action": "independent_human_and_chronological_evaluation",
    }


def freeze_experiment_contract(
    *,
    dataset_revision: str,
    dataset_sha256: str,
    split_manifest_sha256: str,
    source_bundle_sha256: str,
    code_sha256: str,
    dependency_lock_sha256: str,
    development_proxy_sha256: str,
    development_reference_sha256: str,
    baseline_manifest_sha256: str,
    matched_baseline_sha256: str,
    teacher_generation_run_id: str,
    rate_card_usd_per_gpu_second: Mapping[str, str | float],
    hard_cost_cap_usd: str | int | float = "20",
) -> dict[str, Any]:
    """Freeze all data, model, source, baseline, design, compute, and cost bindings."""

    if dataset_revision != DATASET_REVISION or dataset_sha256 != DATASET_SHA256:
        raise CascadeExperimentContractError("private teacher dataset identity drifted")
    if development_proxy_sha256 != DEVELOPMENT_PROXY_ID:
        raise CascadeExperimentContractError("development proxy identity drifted")
    if teacher_generation_run_id != TEACHER_GENERATION_RUN_ID:
        raise CascadeExperimentContractError("teacher generation identity drifted")
    cap = Decimal(str(hard_cost_cap_usd))
    if not cap.is_finite() or cap <= 0 or cap > HARD_COST_CAP_USD:
        raise ValueError("hard cost cap must be positive and no greater than $20")
    if not rate_card_usd_per_gpu_second:
        raise ValueError("at least one GPU rate must be frozen")
    rate_card: dict[str, str] = {}
    for gpu_type, raw in sorted(rate_card_usd_per_gpu_second.items()):
        rate = Decimal(str(raw))
        if not gpu_type or not rate.is_finite() or rate <= 0:
            raise ValueError("GPU rates must bind non-empty types to positive decimals")
        rate_card[gpu_type] = format(rate, "f")
    bindings = {
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "dataset_sha256": _sha(dataset_sha256, where="dataset_sha256"),
        "teacher_generation_run_id": _sha(
            teacher_generation_run_id, where="teacher_generation_run_id"
        ),
        "split_manifest_sha256": _sha(split_manifest_sha256, where="split_manifest_sha256"),
        "development_proxy_sha256": _sha(
            development_proxy_sha256, where="development_proxy_sha256"
        ),
        "development_reference_sha256": _sha(
            development_reference_sha256, where="development_reference_sha256"
        ),
        "baseline_manifest_sha256": _sha(
            baseline_manifest_sha256, where="baseline_manifest_sha256"
        ),
        "matched_baseline_sha256": _sha(matched_baseline_sha256, where="matched_baseline_sha256"),
        "source_bundle_sha256": _sha(source_bundle_sha256, where="source_bundle_sha256"),
        "code_sha256": _sha(code_sha256, where="code_sha256"),
        "dependency_lock_sha256": _sha(dependency_lock_sha256, where="dependency_lock_sha256"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    data_contract = {
        "relevance_training_rows": RELEVANCE_TRAINING_ROWS,
        "material_training_rows": MATERIAL_TRAINING_ROWS,
        "target_conditioned_training_rows": TARGET_TRAINING_ROWS,
        "excluded_non_material_target_slots": EXCLUDED_NON_MATERIAL_TARGET_SLOTS,
        "target_state_counts": dict(TARGET_STATE_COUNTS),
        "development_rows": DEVELOPMENT_ROWS,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPERIMENT_KIND,
        "namespace": NAMESPACE,
        "bindings": bindings,
        "architecture": _architecture_contract(),
        "registered_design": {
            "comparison": "one_no_sweep_separate_model_cascade_vs_frozen_three_head",
            "paired_conditions": [dict(row) for row in PAIRED_CONDITIONS],
            "component_configs": {
                component: frozen_component_config(component) for component in COMPONENTS
            },
            "baseline_config_sha256": BASELINE_CONFIG_SHA256,
            "data": data_contract,
            "continuation_criteria": continuation_criteria(),
        },
        "budget": {
            "hard_cost_cap_usd": format(cap, "f"),
            "rate_card_usd_per_gpu_second": rate_card,
        },
        "compute": {
            "allowed_gpus": ["L4"],
            "account_gpu_limit": 10,
            "max_concurrent_trials": 6,
            "gpu_fallback_allowed": False,
            "planned_upper_cost_usd": "20",
            "approved_cost_usd": format(cap, "f"),
        },
        "locked_test": {
            "authorised": False,
            "predictions_authorised": False,
            "rows_accessed": 0,
        },
    }
    assert_metadata_only(body, where="cascade experiment contract")
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
        "locked_test",
        "experiment_run_id",
    }
    if set(value) != required:
        raise ValueError("cascade experiment contract has unexpected fields")
    bindings = value.get("bindings")
    budget = value.get("budget")
    if not isinstance(bindings, Mapping) or not isinstance(budget, Mapping):
        raise ValueError("cascade bindings and budget must be objects")
    rebuilt = freeze_experiment_contract(
        dataset_revision=bindings["dataset_revision"],
        dataset_sha256=bindings["dataset_sha256"],
        split_manifest_sha256=bindings["split_manifest_sha256"],
        source_bundle_sha256=bindings["source_bundle_sha256"],
        code_sha256=bindings["code_sha256"],
        dependency_lock_sha256=bindings["dependency_lock_sha256"],
        development_proxy_sha256=bindings["development_proxy_sha256"],
        development_reference_sha256=bindings["development_reference_sha256"],
        baseline_manifest_sha256=bindings["baseline_manifest_sha256"],
        matched_baseline_sha256=bindings["matched_baseline_sha256"],
        teacher_generation_run_id=bindings["teacher_generation_run_id"],
        rate_card_usd_per_gpu_second=budget["rate_card_usd_per_gpu_second"],
        hard_cost_cap_usd=budget["hard_cost_cap_usd"],
    )
    if dict(value) != rebuilt:
        raise CascadeExperimentContractError("cascade experiment content address drifted")
    return rebuilt


def freeze_trial_spec(
    experiment: Mapping[str, Any],
    *,
    component: str,
    subset_manifest_sha256: str,
    ladder_seed: int,
    optimiser_seed: int,
    gpu_type: str,
    max_gpu_seconds: int,
) -> dict[str, Any]:
    """Freeze one of the six fresh paired component trials."""

    clean_experiment = validate_experiment_contract(experiment)
    if component not in COMPONENTS:
        raise ValueError("component must be relevance or target_conditioned")
    condition = {(row["ladder_seed"], row["optimiser_seed"]) for row in PAIRED_CONDITIONS}
    ladder = _positive_int(ladder_seed, where="ladder_seed")
    optimiser = _positive_int(optimiser_seed, where="optimiser_seed")
    if (ladder, optimiser) not in condition:
        raise CascadeExperimentContractError("trial condition is not registered")
    seconds = _positive_int(max_gpu_seconds, where="max_gpu_seconds")
    rates = clean_experiment["budget"]["rate_card_usd_per_gpu_second"]
    if gpu_type not in rates or gpu_type != "L4":
        raise CascadeExperimentContractError("cascade trials require the frozen L4 GPU")
    reservation = Decimal(rates[gpu_type]) * Decimal(seconds)
    if reservation > Decimal(clean_experiment["budget"]["hard_cost_cap_usd"]):
        raise CascadeExperimentContractError("single trial reservation exceeds the cost cap")
    config = frozen_component_config(component)
    training_rows = RELEVANCE_TRAINING_ROWS if component == "relevance" else TARGET_TRAINING_ROWS
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_SPEC_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase": "confirmation",
        "component": component,
        "config": config,
        "subset_manifest_sha256": _sha(subset_manifest_sha256, where="subset_manifest_sha256"),
        "source_label_rows": RELEVANCE_TRAINING_ROWS,
        "source_material_rows": MATERIAL_TRAINING_ROWS,
        "training_row_count": training_rows,
        "ladder_seed": ladder,
        "optimiser_seed": optimiser,
        "target_epochs": config["max_epochs"],
        "gpu_type": gpu_type,
        "max_gpu_seconds": seconds,
        "reserved_cost_usd": _money(reservation),
        "fresh_training": True,
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(body, where="cascade trial spec")
    return {**body, "trial_id": canonical_sha256(body)}


def validate_trial_spec(
    trial_spec: Mapping[str, Any], *, experiment: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate one trial; an explicit experiment strengthens identity checking."""

    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase",
        "component",
        "config",
        "subset_manifest_sha256",
        "source_label_rows",
        "source_material_rows",
        "training_row_count",
        "ladder_seed",
        "optimiser_seed",
        "target_epochs",
        "gpu_type",
        "max_gpu_seconds",
        "reserved_cost_usd",
        "fresh_training",
        "locked_test_rows_accessed",
        "trial_id",
    }
    if set(trial_spec) != required:
        raise ValueError("cascade trial spec has unexpected fields")
    if experiment is None:
        if trial_spec["schema_version"] != SCHEMA_VERSION or trial_spec["kind"] != TRIAL_SPEC_KIND:
            raise CascadeExperimentContractError("cascade trial schema or kind drifted")
        if trial_spec["config"] != frozen_component_config(trial_spec["component"]):
            raise CascadeExperimentContractError("cascade trial configuration drifted")
        _sha(trial_spec["experiment_run_id"], where="trial.experiment_run_id")
        _sha(trial_spec["subset_manifest_sha256"], where="trial.subset_manifest_sha256")
        if (
            canonical_sha256({k: v for k, v in trial_spec.items() if k != "trial_id"})
            != trial_spec["trial_id"]
        ):
            raise CascadeExperimentContractError("cascade trial content address drifted")
        return deepcopy(dict(trial_spec))
    rebuilt = freeze_trial_spec(
        experiment,
        component=trial_spec["component"],
        subset_manifest_sha256=trial_spec["subset_manifest_sha256"],
        ladder_seed=trial_spec["ladder_seed"],
        optimiser_seed=trial_spec["optimiser_seed"],
        gpu_type=trial_spec["gpu_type"],
        max_gpu_seconds=trial_spec["max_gpu_seconds"],
    )
    if dict(trial_spec) != rebuilt:
        raise CascadeExperimentContractError("cascade trial content address or cost drifted")
    return rebuilt


def build_run_manifest(
    experiment: Mapping[str, Any],
    *,
    trials: Sequence[Mapping[str, Any]],
    baseline_manifest: Mapping[str, Any],
    matched_baseline: Mapping[str, Any],
    dataset_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the only allowed six-trial confirmation manifest."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_trials = [validate_trial_spec(row, experiment=clean_experiment) for row in trials]
    if len(clean_trials) != 6 or len({row["trial_id"] for row in clean_trials}) != 6:
        raise CascadeExperimentContractError("cascade run requires exactly six unique trials")
    expected = {
        (component, pair["ladder_seed"], pair["optimiser_seed"])
        for component in COMPONENTS
        for pair in PAIRED_CONDITIONS
    }
    observed = {
        (row["component"], row["ladder_seed"], row["optimiser_seed"]) for row in clean_trials
    }
    if observed != expected:
        raise CascadeExperimentContractError("cascade trial inventory drifted")
    clean_profile = _clone(dataset_profile)
    if clean_profile != clean_experiment["registered_design"]["data"]:
        raise CascadeExperimentContractError("cascade dataset profile drifted")
    artefacts = {
        "baseline_manifest": _repo_artifact(baseline_manifest, where="artefacts.baseline_manifest"),
        "matched_baseline": _repo_artifact(matched_baseline, where="artefacts.matched_baseline"),
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_MANIFEST_KIND,
        "namespace": NAMESPACE,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "experiment_contract": clean_experiment,
        "phase": "confirmation",
        "experiment_artefacts": artefacts,
        "dataset_profile": clean_profile,
        "locked_test": deepcopy(clean_experiment["locked_test"]),
        "trials": clean_trials,
    }
    assert_metadata_only(body, where="cascade run manifest")
    return {**body, "phase_run_id": canonical_sha256(body)}


def validate_run_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "namespace",
        "experiment_run_id",
        "experiment_contract",
        "phase",
        "experiment_artefacts",
        "dataset_profile",
        "locked_test",
        "trials",
        "phase_run_id",
    }
    if set(value) != required:
        raise ValueError("cascade run manifest has unexpected fields")
    artefacts = value.get("experiment_artefacts")
    if not isinstance(artefacts, Mapping) or set(artefacts) != {
        "baseline_manifest",
        "matched_baseline",
    }:
        raise ValueError("cascade run artefacts have unexpected fields")
    rebuilt = build_run_manifest(
        value["experiment_contract"],
        trials=value["trials"],
        baseline_manifest=artefacts["baseline_manifest"],
        matched_baseline=artefacts["matched_baseline"],
        dataset_profile=value["dataset_profile"],
    )
    if dict(value) != rebuilt:
        raise CascadeExperimentContractError("cascade run manifest content address drifted")
    return rebuilt


def make_trial_job(manifest: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact metadata-only launch payload for one registered trial."""

    clean_manifest = validate_run_manifest(manifest)
    clean_trial = validate_trial_spec(trial, experiment=clean_manifest["experiment_contract"])
    if clean_trial["trial_id"] not in {row["trial_id"] for row in clean_manifest["trials"]}:
        raise CascadeExperimentContractError("trial is absent from the run manifest")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_run_id": clean_manifest["experiment_run_id"],
        "phase_run_id": clean_manifest["phase_run_id"],
        "run_manifest_sha256": canonical_sha256(clean_manifest),
        "trial_id": clean_trial["trial_id"],
        "trial_spec_sha256": canonical_sha256(clean_trial),
        "component": clean_trial["component"],
        "gpu_type": clean_trial["gpu_type"],
        "experiment_contract": clean_manifest["experiment_contract"],
        "trial_spec": clean_trial,
    }


def _finite_vector(value: Any, *, length: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must contain exactly {length} values")
    clean: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{where} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{where} must contain finite numbers")
        clean.append(number)
    return clean


def build_private_development_predictions(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an exact private logit payload; this artefact must never be public."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_trial = validate_trial_spec(trial_spec, experiment=clean_experiment)
    expected_rows = (
        DEVELOPMENT_ROWS if clean_trial["component"] == "relevance" else DEVELOPMENT_ROWS * 4
    )
    clean_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for raw in rows:
        if clean_trial["component"] == "relevance":
            if set(raw) != {"source_sample_id", "logits"}:
                raise ValueError("private relevance prediction row has unexpected fields")
            target: str | None = None
            logits = _finite_vector(raw["logits"], length=3, where="relevance logits")
        else:
            if set(raw) != {"source_sample_id", "target", "logits"}:
                raise ValueError("private target prediction row has unexpected fields")
            target = raw["target"]
            if target not in TARGETS:
                raise ValueError("private target prediction uses an unregistered target")
            logits = _finite_vector(raw["logits"], length=6, where="target logits")
        sample_id = raw["source_sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("private prediction source ID must be non-empty")
        key = (sample_id, target)
        if key in seen:
            raise ValueError("private prediction rows contain a duplicate")
        seen.add(key)
        row = {"source_sample_id": sample_id, "logits": logits}
        if target is not None:
            row["target"] = target
        clean_rows.append(row)
    if len(clean_rows) != expected_rows:
        raise CascadeExperimentContractError("private development prediction row count drifted")
    if clean_trial["component"] == "target_conditioned":
        by_sample: dict[str, set[str]] = {}
        for row in clean_rows:
            by_sample.setdefault(row["source_sample_id"], set()).add(row["target"])
        if len(by_sample) != DEVELOPMENT_ROWS or any(
            targets != set(TARGETS) for targets in by_sample.values()
        ):
            raise CascadeExperimentContractError(
                "target predictions do not cover four targets per row"
            )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_PREDICTIONS_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "trial_id": clean_trial["trial_id"],
        "component": clean_trial["component"],
        "row_count": len(clean_rows),
        "rows": sorted(
            clean_rows, key=lambda row: (row["source_sample_id"], row.get("target", ""))
        ),
    }
    return {**body, "predictions_id": canonical_sha256(body)}


def validate_private_development_predictions(
    value: Mapping[str, Any],
    *,
    expected_experiment_run_id: str,
    expected_trial_id: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "trial_id",
        "component",
        "row_count",
        "rows",
        "predictions_id",
    }
    if set(value) != required:
        raise ValueError("private cascade predictions have unexpected fields")
    if (
        value["experiment_run_id"] != expected_experiment_run_id
        or value["trial_id"] != expected_trial_id
    ):
        raise CascadeExperimentContractError("private prediction binding drifted")
    component = value["component"]
    if component not in COMPONENTS or not isinstance(value["rows"], list):
        raise ValueError("private prediction component or rows are invalid")
    expected = DEVELOPMENT_ROWS if component == "relevance" else DEVELOPMENT_ROWS * 4
    if value["row_count"] != expected or len(value["rows"]) != expected:
        raise CascadeExperimentContractError("private prediction row count drifted")
    seen: set[tuple[str, str | None]] = set()
    target_coverage: dict[str, set[str]] = {}
    for raw in value["rows"]:
        if not isinstance(raw, Mapping):
            raise ValueError("private prediction row must be an object")
        expected_fields = (
            {"source_sample_id", "logits"}
            if component == "relevance"
            else {"source_sample_id", "target", "logits"}
        )
        if set(raw) != expected_fields:
            raise ValueError("private prediction row has unexpected fields")
        sample_id = raw["source_sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("private prediction source ID must be non-empty")
        target = raw.get("target")
        if target is not None and target not in TARGETS:
            raise ValueError("private prediction uses an unregistered target")
        _finite_vector(
            raw["logits"],
            length=3 if component == "relevance" else 6,
            where="private prediction logits",
        )
        key = (sample_id, target)
        if key in seen:
            raise ValueError("private predictions contain a duplicate row")
        seen.add(key)
        if target is not None:
            target_coverage.setdefault(sample_id, set()).add(target)
    if component == "target_conditioned" and (
        len(target_coverage) != DEVELOPMENT_ROWS
        or any(targets != set(TARGETS) for targets in target_coverage.values())
    ):
        raise CascadeExperimentContractError(
            "target predictions do not cover four targets per development row"
        )
    body = {key: value[key] for key in required - {"predictions_id"}}
    if value["predictions_id"] != canonical_sha256(body):
        raise CascadeExperimentContractError("private prediction content address drifted")
    return deepcopy(dict(value))


def _validate_public_metrics(metrics: Mapping[str, Any], *, component: str) -> dict[str, Any]:
    clean = _clone(metrics)
    required_common = {"selected_epoch", "invalid_outputs"}
    if not required_common <= set(clean):
        raise ValueError("cascade trial metrics lack required fields")
    _positive_int(clean["selected_epoch"], where="metrics.selected_epoch")
    _nonnegative_int(clean["invalid_outputs"], where="metrics.invalid_outputs")
    if component == "relevance":
        for field in ("relevance_macro_f1", "material_recall", "checkpoint_score"):
            _probability(clean.get(field), where=f"metrics.{field}")
    else:
        for field in (
            "target_presence_f1_gold_relevance",
            "stance_accuracy_present_targets",
            "stance_macro_f1_present_targets",
            "core_target_stance_tuple_micro_f1_gold_relevance",
            "checkpoint_score",
        ):
            _probability(clean.get(field), where=f"metrics.{field}")
    assert_metadata_only(clean, where="cascade public trial metrics")
    return clean


def build_trial_receipt(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    *,
    phase_run_id: str,
    run_manifest_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    wall_seconds: int | float,
    gpu_seconds: int | float,
) -> dict[str, Any]:
    """Build one metadata-only receipt with exact source/run/private artefact bindings."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_trial = validate_trial_spec(trial_spec, experiment=clean_experiment)
    required_artifacts = {"checkpoint", "metrics", "private_development_predictions"}
    if set(artifacts) != required_artifacts:
        raise ValueError("cascade receipt artefact inventory drifted")
    clean_artifacts = {
        name: _artifact(value, where=f"artifacts.{name}")
        for name, value in sorted(artifacts.items())
    }
    wall = _nonnegative_number(wall_seconds, where="wall_seconds")
    gpu = _nonnegative_number(gpu_seconds, where="gpu_seconds")
    if gpu > clean_trial["max_gpu_seconds"]:
        raise CascadeExperimentContractError("trial exceeded its GPU reservation")
    rate = Decimal(
        clean_experiment["budget"]["rate_card_usd_per_gpu_second"][clean_trial["gpu_type"]]
    )
    cost = rate * Decimal(str(gpu))
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_RECEIPT_KIND,
        "status": "complete",
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase_run_id": _sha(phase_run_id, where="phase_run_id"),
        "run_manifest_sha256": _sha(run_manifest_sha256, where="run_manifest_sha256"),
        "trial_id": clean_trial["trial_id"],
        "trial_spec_sha256": canonical_sha256(clean_trial),
        "component": clean_trial["component"],
        "ladder_seed": clean_trial["ladder_seed"],
        "optimiser_seed": clean_trial["optimiser_seed"],
        "bindings": deepcopy(clean_experiment["bindings"]),
        "artifacts": clean_artifacts,
        "aggregate_metrics": _validate_public_metrics(metrics, component=clean_trial["component"]),
        "compute": {
            "gpu_type": clean_trial["gpu_type"],
            "wall_seconds": wall,
            "gpu_seconds": gpu,
            "estimated_cost_usd": _money(cost),
        },
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(body, where="cascade trial receipt")
    return {**body, "receipt_id": canonical_sha256(body)}


def validate_trial_receipt(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "status",
        "experiment_run_id",
        "phase_run_id",
        "run_manifest_sha256",
        "trial_id",
        "trial_spec_sha256",
        "component",
        "ladder_seed",
        "optimiser_seed",
        "bindings",
        "artifacts",
        "aggregate_metrics",
        "compute",
        "locked_test_rows_accessed",
        "receipt_id",
    }
    if set(receipt) != required:
        raise ValueError("cascade trial receipt has unexpected fields")
    compute = receipt.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("cascade receipt compute block must be an object")
    rebuilt = build_trial_receipt(
        experiment,
        trial_spec,
        phase_run_id=receipt["phase_run_id"],
        run_manifest_sha256=receipt["run_manifest_sha256"],
        artifacts=receipt["artifacts"],
        metrics=receipt["aggregate_metrics"],
        wall_seconds=compute["wall_seconds"],
        gpu_seconds=compute["gpu_seconds"],
    )
    if dict(receipt) != rebuilt:
        raise CascadeExperimentContractError("cascade trial receipt drifted")
    return rebuilt


def budget_status(
    experiment: Mapping[str, Any],
    *,
    completed_receipts: Sequence[Mapping[str, Any]] = (),
    active_trial_specs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Account measured completed cost plus worst-case active reservations."""

    clean_experiment = validate_experiment_contract(experiment)
    completed = Decimal("0")
    seen: set[str] = set()
    for receipt in completed_receipts:
        trial_id = _sha(receipt.get("trial_id"), where="receipt.trial_id")
        receipt_id = _sha(receipt.get("receipt_id"), where="receipt.receipt_id")
        if trial_id in seen or receipt_id != canonical_sha256(
            {k: v for k, v in receipt.items() if k != "receipt_id"}
        ):
            raise CascadeExperimentContractError("completed receipt inventory drifted")
        seen.add(trial_id)
        completed += Decimal(str(receipt["compute"]["estimated_cost_usd"]))
    active = Decimal("0")
    active_ids: set[str] = set()
    for spec in active_trial_specs:
        clean = validate_trial_spec(spec, experiment=clean_experiment)
        if clean["trial_id"] in seen | active_ids:
            raise ValueError("trial appears more than once in budget inventory")
        active_ids.add(clean["trial_id"])
        active += Decimal(str(clean["reserved_cost_usd"]))
    cap = Decimal(clean_experiment["budget"]["hard_cost_cap_usd"])
    projected = completed + active
    if projected > cap:
        raise CascadeExperimentContractError("projected cascade spend exceeds the cap")
    return {
        "hard_cost_cap_usd": _money(cap),
        "completed_cost_usd": _money(completed),
        "active_reserved_cost_usd": _money(active),
        "projected_cost_usd": _money(projected),
        "remaining_cost_usd": _money(cap - projected),
        "completed_trials": len(seen),
        "active_trials": len(active_ids),
    }


def _gate_metric_payload(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    required = {
        "core_target_stance_tuple_micro_f1",
        "material_recall",
        "relevance_macro_f1",
        "invalid_outputs",
        "core_targets",
        "target_stance_cells",
    }
    if set(value) != required:
        raise ValueError(f"{where} has unexpected fields")
    core_targets = value["core_targets"]
    target_stance = value["target_stance_cells"]
    if not isinstance(core_targets, Mapping) or set(core_targets) != set(CORE_TARGETS):
        raise ValueError(f"{where}.core_targets must cover the three core targets")
    expected_cells = {f"{target}:{state}" for target in CORE_TARGETS for state in TARGET_STATES[1:]}
    if not isinstance(target_stance, Mapping) or set(target_stance) != expected_cells:
        raise ValueError(f"{where}.target_stance_cells must cover every core target-state cell")

    def clean_cells(cells: Mapping[str, Any], *, cell_where: str) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for name, raw in sorted(cells.items()):
            if not isinstance(raw, Mapping) or set(raw) != {"f1", "reference_support"}:
                raise ValueError(f"{cell_where}.{name} has unexpected fields")
            clean[name] = {
                "f1": _probability(raw["f1"], where=f"{cell_where}.{name}.f1"),
                "reference_support": _nonnegative_int(
                    raw["reference_support"],
                    where=f"{cell_where}.{name}.reference_support",
                ),
            }
        return clean

    return {
        "core_target_stance_tuple_micro_f1": _probability(
            value["core_target_stance_tuple_micro_f1"],
            where=f"{where}.core_target_stance_tuple_micro_f1",
        ),
        "material_recall": _probability(value["material_recall"], where=f"{where}.material_recall"),
        "relevance_macro_f1": _probability(
            value["relevance_macro_f1"], where=f"{where}.relevance_macro_f1"
        ),
        "invalid_outputs": _nonnegative_int(
            value["invalid_outputs"], where=f"{where}.invalid_outputs"
        ),
        "core_targets": clean_cells(core_targets, cell_where=f"{where}.core_targets"),
        "target_stance_cells": clean_cells(
            target_stance, cell_where=f"{where}.target_stance_cells"
        ),
    }


def _evaluate_paired_continuation_gate(
    experiment: Mapping[str, Any],
    *,
    paired_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered end-to-end gates to three paired aggregate results."""

    clean_experiment = validate_experiment_contract(experiment)
    criteria = clean_experiment["registered_design"]["continuation_criteria"]
    if len(paired_results) != 3:
        raise ValueError("cascade continuation gate requires exactly three paired results")
    expected_conditions = {(row["ladder_seed"], row["optimiser_seed"]) for row in PAIRED_CONDITIONS}
    clean_results: list[dict[str, Any]] = []
    observed: set[tuple[int, int]] = set()
    for index, raw in enumerate(paired_results):
        if set(raw) != {"ladder_seed", "optimiser_seed", "baseline", "candidate"}:
            raise ValueError("cascade paired result has unexpected fields")
        condition = (raw["ladder_seed"], raw["optimiser_seed"])
        if condition in observed or condition not in expected_conditions:
            raise CascadeExperimentContractError("cascade paired condition inventory drifted")
        observed.add(condition)
        baseline = _gate_metric_payload(raw["baseline"], where=f"pair[{index}].baseline")
        candidate = _gate_metric_payload(raw["candidate"], where=f"pair[{index}].candidate")
        for family in ("core_targets", "target_stance_cells"):
            for name in baseline[family]:
                if (
                    baseline[family][name]["reference_support"]
                    != candidate[family][name]["reference_support"]
                ):
                    raise CascadeExperimentContractError(
                        "baseline and candidate reference support disagree"
                    )
        clean_results.append(
            {
                "ladder_seed": condition[0],
                "optimiser_seed": condition[1],
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    if observed != expected_conditions:
        raise CascadeExperimentContractError("cascade paired result inventory is incomplete")
    clean_results.sort(key=lambda row: (row["ladder_seed"], row["optimiser_seed"]))
    gains = [
        row["candidate"]["core_target_stance_tuple_micro_f1"]
        - row["baseline"]["core_target_stance_tuple_micro_f1"]
        for row in clean_results
    ]
    material_declines = [
        row["baseline"]["material_recall"] - row["candidate"]["material_recall"]
        for row in clean_results
    ]
    relevance_declines = [
        row["baseline"]["relevance_macro_f1"] - row["candidate"]["relevance_macro_f1"]
        for row in clean_results
    ]
    mean_gain = sum(gains) / len(gains)
    improved_pairs = sum(gain > 0 for gain in gains)
    mean_material_decline = sum(material_declines) / len(material_declines)
    mean_relevance_decline = sum(relevance_declines) / len(relevance_declines)

    def supported_regressions(family: str) -> dict[str, float]:
        output: dict[str, float] = {}
        names = clean_results[0]["baseline"][family]
        for name in names:
            support = clean_results[0]["baseline"][family][name]["reference_support"]
            if any(
                row["baseline"][family][name]["reference_support"] != support
                for row in clean_results[1:]
            ):
                raise CascadeExperimentContractError(
                    "reference support differs across paired conditions"
                )
            if support < criteria["minimum_reference_support"]:
                continue
            output[name] = sum(
                row["baseline"][family][name]["f1"] - row["candidate"][family][name]["f1"]
                for row in clean_results
            ) / len(clean_results)
        return output

    target_regressions = supported_regressions("core_targets")
    stance_regressions = supported_regressions("target_stance_cells")
    invalid_outputs = sum(row["candidate"]["invalid_outputs"] for row in clean_results)
    gates = {
        "mean_gain": mean_gain >= criteria["minimum_mean_paired_gain"],
        "improved_pairs": improved_pairs >= criteria["minimum_improved_pairs"],
        "material_recall": (
            mean_material_decline <= criteria["maximum_mean_material_recall_decline"]
        ),
        "relevance_macro_f1": (
            mean_relevance_decline <= criteria["maximum_mean_relevance_macro_f1_decline"]
        ),
        "supported_core_targets": all(
            regression <= criteria["maximum_supported_core_target_regression"]
            for regression in target_regressions.values()
        ),
        "supported_target_stance_cells": all(
            regression <= criteria["maximum_supported_target_stance_regression"]
            for regression in stance_regressions.values()
        ),
        "invalid_outputs": invalid_outputs <= criteria["maximum_invalid_outputs"],
    }
    if all(gates.values()):
        verdict = "continue_to_independent_evaluation"
    elif mean_gain < criteria["gain_bands"]["scrap_below"] or not all(
        value for name, value in gates.items() if name != "mean_gain"
    ):
        verdict = "scrap"
    else:
        verdict = "inconclusive_no_promotion"
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTINUATION_GATE_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "criteria": deepcopy(criteria),
        "paired_conditions": [
            {
                "ladder_seed": row["ladder_seed"],
                "optimiser_seed": row["optimiser_seed"],
                "core_tuple_gain": gains[index],
                "material_recall_decline": material_declines[index],
                "relevance_macro_f1_decline": relevance_declines[index],
            }
            for index, row in enumerate(clean_results)
        ],
        "aggregate": {
            "mean_paired_core_tuple_gain": mean_gain,
            "improved_pair_count": improved_pairs,
            "mean_material_recall_decline": mean_material_decline,
            "mean_relevance_macro_f1_decline": mean_relevance_decline,
            "candidate_invalid_outputs": invalid_outputs,
            "supported_core_target_regressions": target_regressions,
            "supported_target_stance_regressions": stance_regressions,
        },
        "gates": gates,
        "verdict": verdict,
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(body, where="cascade continuation gate")
    return {**body, "gate_receipt_id": canonical_sha256(body)}


def _normalise_closeout_metric_payload(
    value: Mapping[str, Any], *, candidate: bool, where: str
) -> tuple[dict[str, Any], int]:
    payload = value.get("metric_payload", value)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{where} metric payload must be an object")
    core_f1 = payload.get("core_target_f1")
    core_support = payload.get(
        "core_target_support" if candidate else "core_target_reference_support"
    )
    diagnostics = payload.get("target_stance_diagnostics")
    if not all(isinstance(block, Mapping) for block in (core_f1, core_support, diagnostics)):
        raise ValueError(f"{where} lacks target diagnostic blocks")
    core_cells: dict[str, Any] = {}
    stance_cells: dict[str, Any] = {}
    for target in CORE_TARGETS:
        core_cells[target] = {
            "f1": core_f1.get(target),
            "reference_support": core_support.get(target),
        }
        target_diagnostics = diagnostics.get(target)
        if not isinstance(target_diagnostics, Mapping):
            raise ValueError(f"{where} lacks diagnostics for {target}")
        for state in TARGET_STATES[1:]:
            cell = target_diagnostics.get(state)
            if not isinstance(cell, Mapping):
                raise ValueError(f"{where} lacks diagnostics for {target}:{state}")
            stance_cells[f"{target}:{state}"] = {
                "f1": cell.get("f1"),
                "reference_support": cell.get("reference_support"),
            }
    tuple_key = (
        "core_tuple_f1"
        if candidate and "core_tuple_f1" in payload
        else "core_target_stance_tuple_micro_f1"
    )
    if tuple_key not in payload:
        raise ValueError(f"{where} lacks the primary core-tuple metric")
    forced = payload.get("forced_target_selections", 0)
    return (
        {
            "core_target_stance_tuple_micro_f1": payload[tuple_key],
            "material_recall": payload.get("material_recall"),
            "relevance_macro_f1": payload.get("relevance_macro_f1"),
            "invalid_outputs": payload.get("invalid_outputs"),
            "core_targets": core_cells,
            "target_stance_cells": stance_cells,
        },
        _nonnegative_int(forced, where=f"{where}.forced_target_selections"),
    )


def evaluate_continuation_gate(
    experiment: Mapping[str, Any],
    *,
    phase_run_id: str,
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    candidate_trial_receipt_ids: Sequence[str],
) -> dict[str, Any]:
    """Close out three paired cascade results against the frozen three-head baseline."""

    clean_experiment = validate_experiment_contract(experiment)
    phase_id = _sha(phase_run_id, where="phase_run_id")
    receipt_ids = [
        _sha(value, where="candidate_trial_receipt_id") for value in candidate_trial_receipt_ids
    ]
    if len(receipt_ids) != 6 or len(set(receipt_ids)) != 6:
        raise ValueError("cascade closeout requires six unique candidate receipt IDs")

    def by_condition(
        rows: Sequence[Mapping[str, Any]], *, where: str
    ) -> dict[tuple[int, int], Mapping[str, Any]]:
        if len(rows) != 3:
            raise ValueError(f"{where} requires exactly three aggregate results")
        output: dict[tuple[int, int], Mapping[str, Any]] = {}
        for row in rows:
            condition = row.get("condition")
            if not isinstance(condition, Mapping) or set(condition) != {
                "ladder_seed",
                "optimiser_seed",
            }:
                raise ValueError(f"{where} result lacks an exact condition")
            key = (condition["ladder_seed"], condition["optimiser_seed"])
            if key in output:
                raise ValueError(f"{where} contains a duplicate condition")
            output[key] = row
        return output

    baseline_by_condition = by_condition(baseline_results, where="baseline")
    candidate_by_condition = by_condition(candidate_results, where="candidate")
    expected = {(row["ladder_seed"], row["optimiser_seed"]) for row in PAIRED_CONDITIONS}
    if set(baseline_by_condition) != expected or set(candidate_by_condition) != expected:
        raise CascadeExperimentContractError("closeout paired condition inventory drifted")
    paired: list[dict[str, Any]] = []
    forced_target_selections = 0
    for ladder_seed, optimiser_seed in sorted(expected):
        baseline, _ = _normalise_closeout_metric_payload(
            baseline_by_condition[(ladder_seed, optimiser_seed)],
            candidate=False,
            where=f"baseline[{ladder_seed},{optimiser_seed}]",
        )
        candidate, forced = _normalise_closeout_metric_payload(
            candidate_by_condition[(ladder_seed, optimiser_seed)],
            candidate=True,
            where=f"candidate[{ladder_seed},{optimiser_seed}]",
        )
        forced_target_selections += forced
        paired.append(
            {
                "ladder_seed": ladder_seed,
                "optimiser_seed": optimiser_seed,
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    evaluated = _evaluate_paired_continuation_gate(clean_experiment, paired_results=paired)
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONTINUATION_GATE_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase_run_id": phase_id,
        "candidate_trial_receipt_ids": sorted(receipt_ids),
        "criteria": evaluated["criteria"],
        "paired_conditions": evaluated["paired_conditions"],
        "aggregate_evidence": {
            **evaluated["aggregate"],
            "forced_target_selections": forced_target_selections,
        },
        "guards": evaluated["gates"],
        "verdict": evaluated["verdict"],
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(body, where="cascade continuation gate receipt")
    return {**body, "gate_receipt_id": canonical_sha256(body)}


def experiment_output_root(volume_root: Path, experiment: Mapping[str, Any]) -> Path:
    clean = validate_experiment_contract(experiment)
    return volume_root / OUTPUT_PREFIX / f"run={clean['experiment_run_id']}"


def trial_output_root(
    volume_root: Path, experiment: Mapping[str, Any], trial_spec: Mapping[str, Any]
) -> Path:
    clean_experiment = validate_experiment_contract(experiment)
    clean_trial = validate_trial_spec(trial_spec, experiment=clean_experiment)
    return (
        experiment_output_root(volume_root, clean_experiment)
        / "phase=confirmation"
        / f"component={clean_trial['component']}"
        / f"trial={clean_trial['trial_id']}"
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _require_namespaced(path: Path) -> None:
    if NAMESPACE not in path.parts:
        raise ValueError(f"cascade publication path must be under {NAMESPACE}")


def publication_state(path: Path, expected: Mapping[str, Any]) -> str:
    """Return missing, resume_exact, or complete; reject immutable drift."""

    _require_namespaced(path)
    payload = _json_bytes(expected)
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload or incomplete.exists():
            raise CascadeExperimentContractError(f"immutable cascade publication differs: {path}")
        return "complete"
    if incomplete.exists():
        if not incomplete.is_file() or incomplete.read_bytes() != payload:
            raise CascadeExperimentContractError(
                "cascade incomplete publication cannot resume exactly"
            )
        return "resume_exact"
    return "missing"


def publish_immutable_json(
    path: Path, value: Mapping[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    payload = _json_bytes(value)
    state = publication_state(path, value)
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    if state == "resume_exact":
        if not resume:
            raise CascadeExperimentContractError("exact incomplete output requires explicit resume")
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


def validate_immutable_publication(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if publication_state(path, expected) != "complete":
        raise CascadeExperimentContractError("immutable cascade publication is incomplete")
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


__all__ = [
    "BASELINE_CONFIG_SHA256",
    "COMPONENTS",
    "CONTINUATION_GATE_KIND",
    "CORE_TARGETS",
    "DEVELOPMENT_ROWS",
    "MATERIAL_TRAINING_ROWS",
    "NAMESPACE",
    "OUTPUT_PREFIX",
    "PAIRED_CONDITIONS",
    "PRIVATE_PREDICTIONS_KIND",
    "RELEVANCE_CLASSES",
    "RELEVANCE_TRAINING_ROWS",
    "TARGETS",
    "TARGET_STATES",
    "TARGET_STATE_COUNTS",
    "TARGET_TRAINING_ROWS",
    "CascadeExperimentContractError",
    "budget_status",
    "build_private_development_predictions",
    "build_run_manifest",
    "build_trial_receipt",
    "canonical_sha256",
    "continuation_criteria",
    "evaluate_continuation_gate",
    "experiment_output_root",
    "freeze_experiment_contract",
    "freeze_trial_spec",
    "frozen_component_config",
    "make_trial_job",
    "publication_state",
    "publish_immutable_json",
    "trial_output_root",
    "validate_experiment_contract",
    "validate_immutable_publication",
    "validate_private_development_predictions",
    "validate_run_manifest",
    "validate_trial_receipt",
    "validate_trial_spec",
]
