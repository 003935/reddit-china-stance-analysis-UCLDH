"""Immutable contracts for the fixed factorised ModernBERT-v2 comparison.

This module is deliberately free of Torch, Transformers and Modal imports.  It
defines the exact three-component by three-seed experiment, content-addressed
private artefact bindings, public receipts and the aggregate-only B2-versus-B4
decision gate.  Threshold selection, calibration, legacy development frames and
locked tests are outside this experiment's authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    STANCE_CLASSES_B4,
    TARGET_CLASSES,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation_v2 import (
    natural_arm_weighting_config_from_summary,
)

SCHEMA_VERSION = "1.0.0"
NAMESPACE = "student-modernbert-factorised-v2"
OUTPUT_PREFIX = Path(NAMESPACE)

EXPERIMENT_KIND = "modernbert-factorised-experiment-v2"
SOURCE_BUNDLE_KIND = "listed-source-bundle-v1"
TRIAL_SPEC_KIND = "modernbert-factorised-trial-spec-v2"
RUN_MANIFEST_KIND = "modernbert-factorised-run-manifest-v2"
TRIAL_RECEIPT_KIND = "modernbert-factorised-trial-receipt-v2"
REPRESENTATION_GATE_KIND = "modernbert-factorised-representation-gate-v2"

MODEL_ID = "answerdotai/ModernBERT-large"
MODEL_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
TOKENIZER_REVISION = MODEL_REVISION

COMPONENTS = ("relevance", "target_stance_b4", "target_stance_b2")
TARGET_STANCE_COMPONENTS = COMPONENTS[1:]
REGISTERED_SEEDS = (47, 61, 89)
EXPECTED_TRIALS = len(COMPONENTS) * len(REGISTERED_SEEDS)
GPU_TYPE = "L4"
ACCOUNT_GPU_LIMIT = 10
MAX_CONCURRENT_TRIALS = 9
HARD_COST_CAP_USD = Decimal("200")
NATURAL_DEVELOPMENT_ESTIMAND = (
    "unexposed primary-eligible factorised-v2 engineering population"
)


class FactorisedExperimentContractError(RuntimeError):
    """Raised when a frozen factorised-v2 experiment binding drifts."""


def canonical_sha256(value: Any) -> str:
    """Return the repository's canonical finite-JSON SHA-256."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _finite_number(value: Any, *, where: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number >= {minimum}")
    clean = float(value)
    if not math.isfinite(clean) or clean < minimum:
        raise ValueError(f"{where} must be a finite number >= {minimum}")
    return clean


def _probability(value: Any, *, where: str) -> float:
    clean = _finite_number(value, where=where)
    if clean > 1:
        raise ValueError(f"{where} must be in [0, 1]")
    return clean


def _safe_relative(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{where} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"{where} must be a safe relative path")
    return value


def artifact_descriptor(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Validate a private or Volume-relative immutable artefact descriptor."""

    expected = {"relative_path", "sha256", "bytes"}
    optional = {
        "row_count",
        "thread_set_sha256",
        "frame",
        "manifest_id",
        "primary_probability_row_count",
        "selection_component_counts",
    }
    if not expected <= set(value) or set(value) - expected - optional:
        raise ValueError(f"{where} has unexpected fields")
    result: dict[str, Any] = {
        "relative_path": _safe_relative(value["relative_path"], where=f"{where}.relative_path"),
        "sha256": _sha(value["sha256"], where=f"{where}.sha256"),
        "bytes": _positive_int(value["bytes"], where=f"{where}.bytes"),
    }
    if "row_count" in value:
        result["row_count"] = _positive_int(value["row_count"], where=f"{where}.row_count")
    if "thread_set_sha256" in value:
        result["thread_set_sha256"] = _sha(
            value["thread_set_sha256"], where=f"{where}.thread_set_sha256"
        )
    if "frame" in value:
        frame = value["frame"]
        if frame not in {
            "training",
            "development",
            "calibration",
            "acquisition_evaluation",
        }:
            raise ValueError(f"{where}.frame is unsupported")
        result["frame"] = frame
    if "manifest_id" in value:
        result["manifest_id"] = _sha(value["manifest_id"], where=f"{where}.manifest_id")
    if "primary_probability_row_count" in value:
        result["primary_probability_row_count"] = _positive_int(
            value["primary_probability_row_count"],
            where=f"{where}.primary_probability_row_count",
        )
    if "selection_component_counts" in value:
        raw_counts = value["selection_component_counts"]
        if not isinstance(raw_counts, Mapping) or not raw_counts:
            raise ValueError(f"{where}.selection_component_counts must be an object")
        result["selection_component_counts"] = {
            str(key): _positive_int(
                count, where=f"{where}.selection_component_counts.{key}"
            )
            for key, count in sorted(raw_counts.items())
            if isinstance(key, str) and key
        }
        if len(result["selection_component_counts"]) != len(raw_counts):
            raise ValueError(f"{where}.selection_component_counts has an invalid key")
    return result


def repo_artifact_descriptor(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if set(value) != {"repo_relative_path", "sha256", "bytes"}:
        raise ValueError(f"{where} has unexpected fields")
    return {
        "repo_relative_path": _safe_relative(
            value["repo_relative_path"], where=f"{where}.repo_relative_path"
        ),
        "sha256": _sha(value["sha256"], where=f"{where}.sha256"),
        "bytes": _positive_int(value["bytes"], where=f"{where}.bytes"),
    }


def build_source_bundle(repo_root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    """Hash exactly the listed files; unrelated extra repository files are irrelevant."""

    if not relative_paths or len(set(relative_paths)) != len(relative_paths):
        raise ValueError("source bundle requires unique listed files")
    files: dict[str, str] = {}
    for raw in sorted(relative_paths):
        relative = _safe_relative(raw, where="source bundle path")
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = file_sha256(path)
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_BUNDLE_KIND,
        "files": files,
        "code_sha256": canonical_sha256(files),
    }
    return {**body, "source_bundle_id": canonical_sha256(body)}


def validate_source_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "kind",
        "files",
        "code_sha256",
        "source_bundle_id",
    }:
        raise ValueError("source bundle schema drifted")
    raw_files = value.get("files")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != SOURCE_BUNDLE_KIND
        or not isinstance(raw_files, Mapping)
        or not raw_files
    ):
        raise ValueError("source bundle identity or inventory is invalid")
    files: dict[str, str] = {}
    for raw_path, digest in sorted(raw_files.items()):
        relative = _safe_relative(raw_path, where="source bundle path")
        files[relative] = _sha(digest, where=f"source bundle {relative}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_BUNDLE_KIND,
        "files": files,
        "code_sha256": canonical_sha256(files),
    }
    expected = {**body, "source_bundle_id": canonical_sha256(body)}
    if dict(value) != expected:
        raise FactorisedExperimentContractError("source bundle digest drifted")
    return expected


def frozen_component_config(component: str) -> dict[str, Any]:
    """Return the sole registered optimiser/model recipe for one component."""

    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    common = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "encoder_learning_rate": 5e-5,
        "head_learning_rate_multiplier": 5.0,
        "dropout": 0.1,
        "effective_batch_size": 32,
        "per_device_batch_size": 8,
        "gradient_accumulation_steps": 4,
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
        "loss_weighting": "unweighted_proper_losses",
        "checkpoint_selection_uses": "development_only",
        "threshold_or_calibration_selection": False,
    }
    if component == "relevance":
        body = {
            **common,
            "component": component,
            "max_epochs": 6,
            "objective": "binary_relevance_bce_on_codable_rows",
            "checkpoint_score": {
                "relevance_macro_f1": 0.5,
                "material_recall": 0.5,
            },
        }
    else:
        representation = "B4" if component.endswith("b4") else "B2"
        body = {
            **common,
            "component": component,
            "max_epochs": 8,
            "objective": "six_presence_bce_plus_masked_unweighted_stance",
            "stance_representation": representation,
            "checkpoint_score": "conditional_analytic_tuple_micro_f1",
        }
    return {**body, "config_sha256": canonical_sha256(body)}


def representation_criteria() -> dict[str, Any]:
    return {
        "default": "B4",
        "minimum_mean_paired_tuple_f1_gain": 0.015,
        "scrap_below_mean_paired_tuple_f1_gain": 0.005,
        "minimum_improved_pairs": 2,
        "paired_seed_count": 3,
        "maximum_mean_target_presence_f1_decline": 0.01,
        "supported_reference_minimum": 10,
        "maximum_supported_target_or_cell_regression": 0.05,
        "maximum_calibration_or_retained_risk_worsening": 0.01,
        "maximum_invalid_outputs": 0,
    }


def _architecture_contract() -> dict[str, Any]:
    return {
        "encoders": {
            "relevance": "independent_modernbert_large",
            "target_stance_b4": "independent_modernbert_large",
            "target_stance_b2": "independent_modernbert_large",
            "shared_trainable_parameters": False,
        },
        "relevance": {
            "classes": ["not_material", "material"],
            "not_codable": "masked_not_a_class",
        },
        "target_presence": {
            "targets": list(TARGET_CLASSES),
            "objective": "six_independent_binary_logits_on_reference_material_rows",
            "force_at_least_one_target": False,
        },
        "stance": {
            "analytic_targets": list(ANALYTIC_TARGET_CLASSES),
            "states": list(STANCE_CLASSES_B4),
            "mask": "reference_present_analytic_targets_only",
            "representations": {
                "B4": "four_way_categorical_cross_entropy",
                "B2": "two_independent_polarity_bit_binary_cross_entropies",
            },
        },
        "paired_evaluation": {
            "relevance_checkpoint_reuse": "same_seed_and_exact_data_binding_only",
            "target_stance_encoder_reuse": False,
            "full_coverage_thresholds": "fixed_0.5_diagnostic_only",
            "calibration_authorised": False,
        },
    }


def freeze_experiment_contract(
    *,
    teacher_run_id: str,
    teacher_receipt: Mapping[str, Any],
    teacher_ledger: Mapping[str, Any],
    teacher_blinded_input: Mapping[str, Any],
    teacher_private_mapping: Mapping[str, Any],
    source_parquet: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    source_metadata_mapping: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    split_public_manifest: Mapping[str, Any],
    bridge_exposure_register: Mapping[str, Any],
    legacy_proxy: Mapping[str, Any],
    legacy_overlap_audit: Mapping[str, Any],
    training_frame: Mapping[str, Any],
    development_frame: Mapping[str, Any],
    development_probability_design: Mapping[str, Any],
    rubric: Mapping[str, Any],
    schema: Mapping[str, Any],
    bridge_authorisation: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    dependency_lock_sha256: str,
    rate_card_usd_per_gpu_second: Mapping[str, str | float],
    cumulative_measured_spend_usd: str | float,
    active_reservation_usd: str | float,
    planned_phase_upper_usd: str | float,
    hard_cost_cap_usd: str | float = "200",
) -> dict[str, Any]:
    """Freeze fresh-v2 evidence, implementation and total-budget bindings."""

    bundle = validate_source_bundle(source_bundle)
    train = artifact_descriptor(training_frame, where="training_frame")
    development = artifact_descriptor(development_frame, where="development_frame")
    if "row_count" not in train or "thread_set_sha256" not in train:
        raise ValueError("training frame must bind row and thread counts")
    if "row_count" not in development or "thread_set_sha256" not in development:
        raise ValueError("development frame must bind row and thread counts")
    if (
        development.get("row_count") != 600
        or development.get("primary_probability_row_count") != 300
        or development.get("selection_component_counts")
        != {
            "development_context_available": 75,
            "development_multi_target": 75,
            "development_probability": 300,
            "development_rare_target_stance": 150,
        }
    ):
        raise ValueError(
            "development frame must bind the exact 300-primary/300-enrichment design"
        )
    if train["thread_set_sha256"] == development["thread_set_sha256"]:
        raise ValueError("training and development thread sets must be distinct")
    probability_design = dict(development_probability_design)
    weighting = natural_arm_weighting_config_from_summary(
        probability_design,
        estimand=NATURAL_DEVELOPMENT_ESTIMAND,
        include_unweighted_conditional_diagnostics=False,
    )
    probability_weighting = {
        "design_summary": probability_design,
        "design_summary_sha256": canonical_sha256(probability_design),
        "weighting_config_digest": weighting.digest(),
    }

    frozen_teacher_inputs = {
        "teacher_ledger": artifact_descriptor(teacher_ledger, where="teacher_ledger"),
        "teacher_blinded_input": artifact_descriptor(
            teacher_blinded_input, where="teacher_blinded_input"
        ),
        "teacher_private_mapping": artifact_descriptor(
            teacher_private_mapping, where="teacher_private_mapping"
        ),
        "source_parquet": artifact_descriptor(source_parquet, where="source_parquet"),
        "split_manifest": artifact_descriptor(split_manifest, where="split_manifest"),
        "split_public_manifest": artifact_descriptor(
            split_public_manifest, where="split_public_manifest"
        ),
    }
    if any("row_count" not in value for value in frozen_teacher_inputs.values()):
        raise ValueError("every teacher/source/split input must bind its exact row count")
    if len({value["row_count"] for value in frozen_teacher_inputs.values()}) != 1:
        raise FactorisedExperimentContractError(
            "teacher/source/split input row counts do not conserve the source"
        )
    frozen_auxiliary_inputs = {
        "bridge_exposure_register": artifact_descriptor(
            bridge_exposure_register, where="bridge_exposure_register"
        ),
        "legacy_proxy": artifact_descriptor(legacy_proxy, where="legacy_proxy"),
        "legacy_overlap_audit": artifact_descriptor(
            legacy_overlap_audit, where="legacy_overlap_audit"
        ),
    }
    if frozen_auxiliary_inputs["bridge_exposure_register"].get("row_count") != 480:
        raise ValueError("bridge_exposure_register must bind exactly 480 threads")
    if frozen_auxiliary_inputs["legacy_proxy"].get("row_count") != 452:
        raise ValueError("legacy_proxy must bind exactly 452 rows")

    rates: dict[str, str] = {}
    if set(rate_card_usd_per_gpu_second) != {GPU_TYPE}:
        raise ValueError("rate card must bind exactly the L4 rate")
    for gpu, value in rate_card_usd_per_gpu_second.items():
        rate = Decimal(str(value))
        if not rate.is_finite() or rate <= 0:
            raise ValueError("GPU rate must be positive and finite")
        rates[gpu] = format(rate, "f")

    cap = Decimal(str(hard_cost_cap_usd))
    measured = Decimal(str(cumulative_measured_spend_usd))
    active = Decimal(str(active_reservation_usd))
    planned = Decimal(str(planned_phase_upper_usd))
    values = (cap, measured, active, planned)
    if any(not value.is_finite() or value < 0 for value in values) or cap <= 0:
        raise ValueError("cost bindings must be finite and non-negative with a positive cap")
    if cap > HARD_COST_CAP_USD:
        raise ValueError("hard cost cap must be no greater than $200")
    if measured + active + planned > cap:
        raise FactorisedExperimentContractError("planned experiment exceeds the total hard cap")

    bindings = {
        "teacher_run_id": _sha(teacher_run_id, where="teacher_run_id"),
        "teacher_receipt": artifact_descriptor(teacher_receipt, where="teacher_receipt"),
        "source_receipt": artifact_descriptor(source_receipt, where="source_receipt"),
        "source_metadata_mapping": artifact_descriptor(
            source_metadata_mapping, where="source_metadata_mapping"
        ),
        **frozen_teacher_inputs,
        **frozen_auxiliary_inputs,
        "training_frame": train,
        "development_frame": development,
        "development_probability_weighting": probability_weighting,
        "rubric": repo_artifact_descriptor(rubric, where="rubric"),
        "schema": repo_artifact_descriptor(schema, where="schema"),
        "bridge_authorisation": artifact_descriptor(
            bridge_authorisation, where="bridge_authorisation"
        ),
        "source_bundle": bundle,
        "source_bundle_sha256": canonical_sha256(bundle),
        "code_sha256": bundle["code_sha256"],
        "dependency_lock_sha256": _sha(
            dependency_lock_sha256, where="dependency_lock_sha256"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    if bindings["source_metadata_mapping"] != bindings["teacher_private_mapping"]:
        raise FactorisedExperimentContractError(
            "source metadata mapping must be the exact teacher private mapping"
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPERIMENT_KIND,
        "namespace": NAMESPACE,
        "bindings": bindings,
        "architecture": _architecture_contract(),
        "registered_design": {
            "components": list(COMPONENTS),
            "seeds": list(REGISTERED_SEEDS),
            "expected_trials": EXPECTED_TRIALS,
            "configs": {component: frozen_component_config(component) for component in COMPONENTS},
            "representation_criteria": representation_criteria(),
            "development_use": "checkpoint_and_B2_vs_B4_decision_only",
            "development_primary_metric_component": "development_probability",
            "development_probability_weighting_config_digest": weighting.digest(),
            "development_diagnostic_components": [
                "development_context_available",
                "development_multi_target",
                "development_rare_target_stance",
            ],
            "calibration_use": False,
        },
        "compute": {
            "allowed_gpus": [GPU_TYPE],
            "account_gpu_limit": ACCOUNT_GPU_LIMIT,
            "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
            "gpu_fallback_allowed": False,
            "rate_card_usd_per_gpu_second": rates,
            "cumulative_measured_spend_usd": format(measured, "f"),
            "active_reservation_usd": format(active, "f"),
            "planned_phase_upper_usd": format(planned, "f"),
            "hard_cost_cap_usd": format(cap, "f"),
            "remaining_after_plan_usd": format(cap - measured - active - planned, "f"),
        },
        "evidence_boundary": {
            "fresh_v2_only": True,
            "legacy_222_rows_authorised": False,
            "legacy_230_rows_authorised": False,
            "bridge_rows_authorised_for_training": True,
            "bridge_training_label_source": "fresh_full_run_v2_teacher_only",
            "bridge_rows_authorised_for_evaluation": False,
            "locked_test_authorised": False,
            "locked_test_rows_accessed": 0,
            "human_validation_claim_authorised": False,
            "corpus_inference_authorised": False,
        },
    }
    return {**body, "experiment_run_id": canonical_sha256(body)}


def validate_experiment_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "kind",
        "namespace",
        "bindings",
        "architecture",
        "registered_design",
        "compute",
        "evidence_boundary",
        "experiment_run_id",
    }
    if set(value) != expected_top:
        raise ValueError("factorised experiment contract schema drifted")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != EXPERIMENT_KIND
        or value.get("namespace") != NAMESPACE
    ):
        raise FactorisedExperimentContractError("factorised experiment identity drifted")
    bindings = value.get("bindings")
    design = value.get("registered_design")
    compute = value.get("compute")
    boundary = value.get("evidence_boundary")
    if not all(isinstance(item, Mapping) for item in (bindings, design, compute, boundary)):
        raise ValueError("factorised experiment nested contract is invalid")
    if bindings.get("model_id") != MODEL_ID or bindings.get("model_revision") != MODEL_REVISION:
        raise FactorisedExperimentContractError("pinned model identity drifted")
    expected_binding_keys = {
        "teacher_run_id",
        "teacher_receipt",
        "teacher_ledger",
        "teacher_blinded_input",
        "teacher_private_mapping",
        "source_parquet",
        "source_receipt",
        "source_metadata_mapping",
        "split_manifest",
        "split_public_manifest",
        "bridge_exposure_register",
        "legacy_proxy",
        "legacy_overlap_audit",
        "training_frame",
        "development_frame",
        "development_probability_weighting",
        "rubric",
        "schema",
        "bridge_authorisation",
        "source_bundle",
        "source_bundle_sha256",
        "code_sha256",
        "dependency_lock_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
    }
    if set(bindings) != expected_binding_keys:
        raise ValueError("factorised experiment input binding schema drifted")
    _sha(bindings.get("teacher_run_id"), where="teacher_run_id")
    clean_artifacts: dict[str, dict[str, Any]] = {}
    for name in (
        "teacher_receipt",
        "teacher_ledger",
        "teacher_blinded_input",
        "teacher_private_mapping",
        "source_parquet",
        "source_receipt",
        "source_metadata_mapping",
        "split_manifest",
        "split_public_manifest",
        "bridge_exposure_register",
        "legacy_proxy",
        "legacy_overlap_audit",
        "training_frame",
        "development_frame",
        "bridge_authorisation",
    ):
        clean_artifacts[name] = artifact_descriptor(bindings.get(name, {}), where=name)
    teacher_input_names = (
        "teacher_ledger",
        "teacher_blinded_input",
        "teacher_private_mapping",
        "source_parquet",
        "split_manifest",
        "split_public_manifest",
    )
    if any("row_count" not in clean_artifacts[name] for name in teacher_input_names):
        raise ValueError("teacher/source/split bindings omit an exact row count")
    if len({clean_artifacts[name]["row_count"] for name in teacher_input_names}) != 1:
        raise FactorisedExperimentContractError("teacher/source/split row counts drifted")
    if clean_artifacts["bridge_exposure_register"].get("row_count") != 480:
        raise FactorisedExperimentContractError(
            "bridge exposure-register row-count binding drifted"
        )
    if clean_artifacts["legacy_proxy"].get("row_count") != 452:
        raise FactorisedExperimentContractError("legacy proxy row-count binding drifted")
    if clean_artifacts["source_metadata_mapping"] != clean_artifacts[
        "teacher_private_mapping"
    ]:
        raise FactorisedExperimentContractError(
            "source metadata mapping drifted from the teacher private mapping"
        )
    expected_frame_values = {
        "training_frame": "training",
        "development_frame": "development",
    }
    for name, expected_frame in expected_frame_values.items():
        if not {"row_count", "thread_set_sha256"} <= set(clean_artifacts[name]):
            raise ValueError(f"{name} omits row or thread-set conservation")
        if clean_artifacts[name].get("frame") != expected_frame:
            raise FactorisedExperimentContractError(
                f"{name} frame identity binding drifted"
            )
    if (
        clean_artifacts["training_frame"]["thread_set_sha256"]
        == clean_artifacts["development_frame"]["thread_set_sha256"]
    ):
        raise FactorisedExperimentContractError(
            "training and development thread-set bindings are not distinct"
        )
    for name in ("rubric", "schema"):
        repo_artifact_descriptor(bindings.get(name, {}), where=name)
    _sha(bindings.get("dependency_lock_sha256"), where="dependency_lock_sha256")
    if bindings.get("tokenizer_revision") != TOKENIZER_REVISION:
        raise FactorisedExperimentContractError("pinned tokenizer identity drifted")
    bundle = validate_source_bundle(bindings.get("source_bundle", {}))
    if (
        bindings.get("source_bundle_sha256") != canonical_sha256(bundle)
        or bindings.get("code_sha256") != bundle["code_sha256"]
    ):
        raise FactorisedExperimentContractError("source bundle binding drifted")
    if design.get("components") != list(COMPONENTS) or design.get("seeds") != list(
        REGISTERED_SEEDS
    ):
        raise FactorisedExperimentContractError("registered trial design drifted")
    expected_configs = {
        component: frozen_component_config(component) for component in COMPONENTS
    }
    if (
        design.get("configs") != expected_configs
        or design.get("expected_trials") != EXPECTED_TRIALS
    ):
        raise FactorisedExperimentContractError("registered component configuration drifted")
    if design.get("representation_criteria") != representation_criteria():
        raise FactorisedExperimentContractError("representation criteria drifted")
    if (
        design.get("development_primary_metric_component")
        != "development_probability"
        or design.get("development_diagnostic_components")
        != [
            "development_context_available",
            "development_multi_target",
            "development_rare_target_stance",
        ]
    ):
        raise FactorisedExperimentContractError(
            "development primary/diagnostic component binding drifted"
        )
    probability_weighting = bindings.get("development_probability_weighting")
    if not isinstance(probability_weighting, Mapping) or set(
        probability_weighting
    ) != {
        "design_summary",
        "design_summary_sha256",
        "weighting_config_digest",
    }:
        raise ValueError("natural development weighting binding schema drifted")
    probability_summary = probability_weighting["design_summary"]
    if not isinstance(probability_summary, Mapping) or probability_weighting[
        "design_summary_sha256"
    ] != canonical_sha256(probability_summary):
        raise FactorisedExperimentContractError(
            "natural development design-summary binding drifted"
        )
    probability_config = natural_arm_weighting_config_from_summary(
        probability_summary,
        estimand=NATURAL_DEVELOPMENT_ESTIMAND,
        include_unweighted_conditional_diagnostics=False,
    )
    if (
        probability_weighting["weighting_config_digest"]
        != probability_config.digest()
        or design.get("development_probability_weighting_config_digest")
        != probability_config.digest()
    ):
        raise FactorisedExperimentContractError(
            "natural development weighting-config binding drifted"
        )
    expected_compute_keys = {
        "allowed_gpus",
        "account_gpu_limit",
        "max_concurrent_trials",
        "gpu_fallback_allowed",
        "rate_card_usd_per_gpu_second",
        "cumulative_measured_spend_usd",
        "active_reservation_usd",
        "planned_phase_upper_usd",
        "hard_cost_cap_usd",
        "remaining_after_plan_usd",
    }
    if set(compute) != expected_compute_keys:
        raise ValueError("compute binding schema drifted")
    expected_inventory = {
        "allowed_gpus": [GPU_TYPE],
        "account_gpu_limit": ACCOUNT_GPU_LIMIT,
        "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
        "gpu_fallback_allowed": False,
    }
    if any(compute.get(key) != expected for key, expected in expected_inventory.items()):
        raise FactorisedExperimentContractError("compute inventory drifted")
    raw_rates = compute.get("rate_card_usd_per_gpu_second")
    if not isinstance(raw_rates, Mapping) or set(raw_rates) != {GPU_TYPE}:
        raise FactorisedExperimentContractError("compute rate card drifted")
    try:
        rate = Decimal(str(raw_rates[GPU_TYPE]))
        measured = Decimal(str(compute["cumulative_measured_spend_usd"]))
        active = Decimal(str(compute["active_reservation_usd"]))
        planned = Decimal(str(compute["planned_phase_upper_usd"]))
        cap = Decimal(str(compute["hard_cost_cap_usd"]))
        remaining = Decimal(str(compute["remaining_after_plan_usd"]))
    except (InvalidOperation, KeyError, ValueError) as exc:
        raise ValueError("compute cost binding is invalid") from exc
    if not rate.is_finite() or rate <= 0:
        raise FactorisedExperimentContractError("compute rate must be positive and finite")
    if any(
        not amount.is_finite() or amount < 0
        for amount in (measured, active, planned, remaining)
    ) or not cap.is_finite() or cap <= 0:
        raise FactorisedExperimentContractError(
            "compute costs must be finite and non-negative with a positive cap"
        )
    planned_total = measured + active + planned
    if cap > HARD_COST_CAP_USD or planned_total > cap:
        raise FactorisedExperimentContractError("total cost cap is invalid")
    if remaining != cap - planned_total:
        raise FactorisedExperimentContractError("remaining cost binding drifted")
    expected_boundary = {
        "fresh_v2_only": True,
        "legacy_222_rows_authorised": False,
        "legacy_230_rows_authorised": False,
        "bridge_rows_authorised_for_training": True,
        "bridge_training_label_source": "fresh_full_run_v2_teacher_only",
        "bridge_rows_authorised_for_evaluation": False,
        "locked_test_authorised": False,
        "locked_test_rows_accessed": 0,
        "human_validation_claim_authorised": False,
        "corpus_inference_authorised": False,
    }
    if dict(boundary) != expected_boundary:
        raise FactorisedExperimentContractError("evidence boundary drifted")
    body = {key: _clone(value[key]) for key in expected_top - {"experiment_run_id"}}
    if value.get("experiment_run_id") != canonical_sha256(body):
        raise FactorisedExperimentContractError("experiment run ID drifted")
    return _clone(dict(value))


def freeze_trial_spec(
    experiment: Mapping[str, Any],
    *,
    component: str,
    optimiser_seed: int,
    max_gpu_seconds: int,
) -> dict[str, Any]:
    clean = validate_experiment_contract(experiment)
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    if optimiser_seed not in REGISTERED_SEEDS:
        raise ValueError("optimiser seed is not registered")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_SPEC_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "component": component,
        "optimiser_seed": optimiser_seed,
        "pairing_key": f"seed={optimiser_seed}",
        "gpu_type": GPU_TYPE,
        "max_gpu_seconds": _positive_int(max_gpu_seconds, where="max_gpu_seconds"),
        "config": frozen_component_config(component),
        "training_frame_sha256": clean["bindings"]["training_frame"]["sha256"],
        "development_frame_sha256": clean["bindings"]["development_frame"]["sha256"],
        "teacher_ledger_sha256": clean["bindings"]["teacher_ledger"]["sha256"],
        "locked_test_rows_accessed": 0,
    }
    return {**body, "trial_id": canonical_sha256(body)}


def validate_trial_spec(
    value: Mapping[str, Any], *, experiment: Mapping[str, Any]
) -> dict[str, Any]:
    clean_experiment = validate_experiment_contract(experiment)
    expected_keys = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "component",
        "optimiser_seed",
        "pairing_key",
        "gpu_type",
        "max_gpu_seconds",
        "config",
        "training_frame_sha256",
        "development_frame_sha256",
        "teacher_ledger_sha256",
        "locked_test_rows_accessed",
        "trial_id",
    }
    if set(value) != expected_keys:
        raise ValueError("factorised trial schema drifted")
    component = value.get("component")
    seed = value.get("optimiser_seed")
    expected = freeze_trial_spec(
        clean_experiment,
        component=str(component),
        optimiser_seed=seed,
        max_gpu_seconds=value.get("max_gpu_seconds"),
    )
    if dict(value) != expected:
        raise FactorisedExperimentContractError("factorised trial binding drifted")
    return expected


def build_run_manifest(
    experiment: Mapping[str, Any], *, trials: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    clean = validate_experiment_contract(experiment)
    clean_trials = [validate_trial_spec(trial, experiment=clean) for trial in trials]
    expected_pairs = {(component, seed) for component in COMPONENTS for seed in REGISTERED_SEEDS}
    observed_pairs = {(trial["component"], trial["optimiser_seed"]) for trial in clean_trials}
    if len(clean_trials) != EXPECTED_TRIALS or observed_pairs != expected_pairs:
        raise FactorisedExperimentContractError(
            "run manifest must contain the exact three-component by three-seed trial set"
        )
    rate = Decimal(clean["compute"]["rate_card_usd_per_gpu_second"][GPU_TYPE])
    reserved = sum(
        rate * Decimal(trial["max_gpu_seconds"]) for trial in clean_trials
    )
    if reserved > Decimal(clean["compute"]["planned_phase_upper_usd"]):
        raise FactorisedExperimentContractError(
            "registered trial GPU-second reservations exceed the frozen phase budget"
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_MANIFEST_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "experiment_contract": clean,
        "phase": "fixed-representation-comparison",
        "trials": sorted(clean_trials, key=lambda row: (row["component"], row["optimiser_seed"])),
        "expected_trial_count": EXPECTED_TRIALS,
        "cuda_preflight_required": True,
        "locked_test_rows_accessed": 0,
    }
    return {**body, "phase_run_id": canonical_sha256(body)}


def validate_run_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "experiment_contract",
        "phase",
        "trials",
        "expected_trial_count",
        "cuda_preflight_required",
        "locked_test_rows_accessed",
        "phase_run_id",
    }
    if set(value) != expected_keys:
        raise ValueError("factorised run manifest schema drifted")
    if value.get("kind") != RUN_MANIFEST_KIND or value.get("phase") != (
        "fixed-representation-comparison"
    ):
        raise FactorisedExperimentContractError("factorised run manifest identity drifted")
    contract = value.get("experiment_contract")
    trials = value.get("trials")
    if not isinstance(contract, Mapping) or not isinstance(trials, list):
        raise ValueError("factorised run manifest contract or trials are invalid")
    expected = build_run_manifest(contract, trials=trials)
    if dict(value) != expected:
        raise FactorisedExperimentContractError("factorised run manifest binding drifted")
    return expected


def make_trial_job(manifest: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    clean = validate_run_manifest(manifest)
    trial_by_id = {item["trial_id"]: item for item in clean["trials"]}
    raw_id = trial.get("trial_id")
    if raw_id not in trial_by_id or dict(trial) != trial_by_id[raw_id]:
        raise FactorisedExperimentContractError("trial is not registered in this manifest")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "run_manifest_sha256": canonical_sha256(clean),
        "experiment_contract": clean["experiment_contract"],
        "trial_spec": trial_by_id[raw_id],
        "trial_spec_sha256": canonical_sha256(trial_by_id[raw_id]),
        "locked_test_rows_accessed": 0,
    }


def validate_public_trial_metrics(
    value: Mapping[str, Any], *, component: str
) -> dict[str, Any]:
    """Validate the exact metadata-only metrics published for one trial.

    The probability sample is the sole design-weighted development surface.
    The three enriched selection components are unweighted diagnostics.  This
    distinction is part of the immutable publication contract rather than
    presentation metadata: silently dropping or changing it would make a
    diagnostic subset look eligible for the probability-sample estimand.
    """

    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    required = {
        "selected_epoch",
        "development_rows",
        "primary_probability_rows",
        "invalid_outputs",
        "checkpoint_score",
        "development_component_metrics",
    }
    if not required <= set(value):
        raise ValueError("trial metrics omit required aggregate fields")
    metrics = _clone(value)
    _positive_int(metrics["selected_epoch"], where="selected_epoch")
    _positive_int(metrics["development_rows"], where="development_rows")
    if metrics["development_rows"] != 600 or metrics["primary_probability_rows"] != 300:
        raise ValueError("trial metrics must bind the 600/300 development design")
    _nonnegative_int(metrics["invalid_outputs"], where="invalid_outputs")
    _probability(metrics["checkpoint_score"], where="checkpoint_score")
    if component == "relevance":
        for key in ("relevance_macro_f1", "material_recall"):
            if key not in metrics:
                raise ValueError(f"relevance metrics omit {key}")
            _probability(metrics[key], where=key)
    else:
        if "conditional_tuple_micro_f1" not in metrics:
            raise ValueError("target/stance metrics omit conditional_tuple_micro_f1")
        _probability(metrics["conditional_tuple_micro_f1"], where="conditional_tuple_micro_f1")
    component_metrics = metrics["development_component_metrics"]
    expected_counts = {
        "development_context_available": 75,
        "development_multi_target": 75,
        "development_probability": 300,
        "development_rare_target_stance": 150,
    }
    if not isinstance(component_metrics, Mapping) or set(component_metrics) != set(
        expected_counts
    ):
        raise ValueError("trial metrics omit development selection-component diagnostics")
    for selection, row_count in expected_counts.items():
        surface = component_metrics[selection]
        if not isinstance(surface, Mapping) or surface.get("row_count") != row_count:
            raise ValueError("development selection-component metric count drifted")
        expected_scope = (
            "design-weighted-natural-probability-arm"
            if selection == "development_probability"
            else "unweighted-diagnostic-only"
        )
        score_keys = (
            {
                "row_count",
                "evidence_scope",
                "relevance_macro_f1",
                "material_recall",
            }
            if component == "relevance"
            else {"row_count", "evidence_scope", "conditional_tuple_micro_f1"}
        )
        if set(surface) != score_keys:
            raise ValueError("development selection-component metric schema drifted")
        if surface["evidence_scope"] != expected_scope:
            raise ValueError(
                f"{selection}.evidence_scope must be {expected_scope}"
            )
        for key in score_keys - {"row_count", "evidence_scope"}:
            _probability(surface[key], where=f"{selection}.{key}")
    assert_metadata_only(metrics, where="factorised trial metrics")
    return metrics


def build_trial_receipt(
    experiment: Mapping[str, Any],
    trial: Mapping[str, Any],
    *,
    phase_run_id: str,
    run_manifest_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    wall_seconds: float,
    gpu_seconds: float,
) -> dict[str, Any]:
    clean_experiment = validate_experiment_contract(experiment)
    clean_trial = validate_trial_spec(trial, experiment=clean_experiment)
    if set(artifacts) != {"checkpoint", "metrics", "private_development_predictions"}:
        raise ValueError("trial receipt requires exactly three immutable artefacts")
    clean_artifacts = {
        key: artifact_descriptor(value, where=f"artifacts.{key}")
        for key, value in artifacts.items()
    }
    public_metrics = validate_public_trial_metrics(
        metrics, component=clean_trial["component"]
    )
    wall = _finite_number(wall_seconds, where="wall_seconds")
    gpu = _finite_number(gpu_seconds, where="gpu_seconds")
    if gpu > clean_trial["max_gpu_seconds"]:
        raise FactorisedExperimentContractError("trial exceeded its registered GPU-second cap")
    rate = Decimal(
        clean_experiment["compute"]["rate_card_usd_per_gpu_second"][GPU_TYPE]
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_RECEIPT_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase_run_id": _sha(phase_run_id, where="phase_run_id"),
        "run_manifest_sha256": _sha(run_manifest_sha256, where="run_manifest_sha256"),
        "trial_id": clean_trial["trial_id"],
        "component": clean_trial["component"],
        "optimiser_seed": clean_trial["optimiser_seed"],
        "config_sha256": clean_trial["config"]["config_sha256"],
        "artifacts": clean_artifacts,
        "aggregate_metrics": public_metrics,
        "wall_seconds": round(wall, 6),
        "gpu_seconds": round(gpu, 6),
        "estimated_cost_usd": format((rate * Decimal(str(gpu))).quantize(Decimal("0.000001")), "f"),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": canonical_sha256(body)}
    assert_metadata_only(receipt, where="factorised trial receipt")
    return receipt


def validate_trial_receipt(
    experiment: Mapping[str, Any],
    trial: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    phase_run_id: str | None = None,
    run_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    clean_experiment = validate_experiment_contract(experiment)
    clean_trial = validate_trial_spec(trial, experiment=clean_experiment)
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase_run_id",
        "run_manifest_sha256",
        "trial_id",
        "component",
        "optimiser_seed",
        "config_sha256",
        "artifacts",
        "aggregate_metrics",
        "wall_seconds",
        "gpu_seconds",
        "estimated_cost_usd",
        "locked_test_rows_accessed",
        "receipt_id",
    }
    if set(value) != required:
        raise ValueError("factorised trial receipt schema drifted")
    expected = build_trial_receipt(
        clean_experiment,
        clean_trial,
        phase_run_id=phase_run_id or value["phase_run_id"],
        run_manifest_sha256=run_manifest_sha256 or value["run_manifest_sha256"],
        artifacts=value["artifacts"],
        metrics=value["aggregate_metrics"],
        wall_seconds=value["wall_seconds"],
        gpu_seconds=value["gpu_seconds"],
    )
    if dict(value) != expected:
        raise FactorisedExperimentContractError("factorised trial receipt binding drifted")
    return expected


def budget_status(
    experiment: Mapping[str, Any], *, completed_receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    clean = validate_experiment_contract(experiment)
    known_trials: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        for seed in REGISTERED_SEEDS:
            # Maximum seconds are not needed to identify receipts here.  Receipt
            # validation occurs in the run-manifest-aware runtime.
            known_trials[f"{component}:{seed}"] = {"component": component, "seed": seed}
    observed: set[str] = set()
    phase_cost = Decimal("0")
    for receipt in completed_receipts:
        key = f"{receipt.get('component')}:{receipt.get('optimiser_seed')}"
        if key not in known_trials or key in observed:
            raise ValueError("completed receipts contain an unknown or duplicate trial")
        observed.add(key)
        phase_cost += Decimal(str(receipt.get("estimated_cost_usd")))
    compute = clean["compute"]
    total = Decimal(compute["cumulative_measured_spend_usd"]) + phase_cost
    cap = Decimal(compute["hard_cost_cap_usd"])
    result = {
        "completed_trials": len(observed),
        "expected_trials": EXPECTED_TRIALS,
        "phase_cost_usd": format(phase_cost.quantize(Decimal("0.000001")), "f"),
        "cumulative_cost_usd": format(total.quantize(Decimal("0.000001")), "f"),
        "remaining_hard_cap_usd": format((cap - total).quantize(Decimal("0.000001")), "f"),
        "within_hard_cap": total <= cap,
    }
    assert_metadata_only(result, where="factorised budget status")
    return result


def _metric_surface(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    required = {
        "tuple_micro_f1",
        "target_presence_macro_f1",
        "calibration_error",
        "retained_coverage_risk",
        "per_target",
        "per_target_stance_cell",
        "invalid_outputs",
    }
    if set(value) != required:
        raise ValueError(f"{where} metric surface schema drifted")
    result = {
        key: _probability(value[key], where=f"{where}.{key}")
        for key in (
            "tuple_micro_f1",
            "target_presence_macro_f1",
            "calibration_error",
            "retained_coverage_risk",
        )
    }
    result["invalid_outputs"] = _nonnegative_int(
        value["invalid_outputs"], where=f"{where}.invalid_outputs"
    )
    for section in ("per_target", "per_target_stance_cell"):
        raw = value[section]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}.{section} must be an object")
        clean_section: dict[str, Any] = {}
        for key, cell in sorted(raw.items()):
            if not isinstance(key, str) or not key or not isinstance(cell, Mapping):
                raise ValueError(f"{where}.{section} entry is invalid")
            if set(cell) != {"score", "reference_support"}:
                raise ValueError(f"{where}.{section}.{key} schema drifted")
            clean_section[key] = {
                "score": _probability(cell["score"], where=f"{where}.{section}.{key}.score"),
                "reference_support": _nonnegative_int(
                    cell["reference_support"],
                    where=f"{where}.{section}.{key}.reference_support",
                ),
            }
        result[section] = clean_section
    return result


def evaluate_representation_gate(
    experiment: Mapping[str, Any],
    *,
    phase_run_id: str,
    paired_metrics: Sequence[Mapping[str, Any]],
    trial_receipt_ids: Sequence[str],
) -> dict[str, Any]:
    """Apply the preregistered aggregate B2-versus-B4 development gates."""

    clean = validate_experiment_contract(experiment)
    if len(paired_metrics) != len(REGISTERED_SEEDS):
        raise ValueError("representation closeout requires exactly three paired seeds")
    by_seed: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for index, raw in enumerate(paired_metrics):
        if set(raw) != {"optimiser_seed", "B4", "B2"}:
            raise ValueError("paired metric row schema drifted")
        seed = raw["optimiser_seed"]
        if seed not in REGISTERED_SEEDS or seed in by_seed:
            raise ValueError("paired metric seed is unregistered or duplicated")
        if not isinstance(raw["B4"], Mapping) or not isinstance(raw["B2"], Mapping):
            raise ValueError("paired metric variants must be objects")
        by_seed[seed] = (
            _metric_surface(raw["B4"], where=f"paired_metrics[{index}].B4"),
            _metric_surface(raw["B2"], where=f"paired_metrics[{index}].B2"),
        )
    if set(by_seed) != set(REGISTERED_SEEDS):
        raise ValueError("paired metrics do not cover all registered seeds")
    receipt_ids = sorted(_sha(value, where="trial_receipt_id") for value in trial_receipt_ids)
    if len(receipt_ids) != EXPECTED_TRIALS or len(set(receipt_ids)) != EXPECTED_TRIALS:
        raise ValueError("closeout requires nine unique trial receipt IDs")

    criteria = representation_criteria()
    gains: list[float] = []
    presence_declines: list[float] = []
    calibration_worsening: list[float] = []
    risk_worsening: list[float] = []
    supported_regressions: list[dict[str, Any]] = []
    invalid_outputs = 0
    for seed, (b4, b2) in sorted(by_seed.items()):
        gains.append(b2["tuple_micro_f1"] - b4["tuple_micro_f1"])
        presence_declines.append(b4["target_presence_macro_f1"] - b2["target_presence_macro_f1"])
        calibration_worsening.append(b2["calibration_error"] - b4["calibration_error"])
        risk_worsening.append(b2["retained_coverage_risk"] - b4["retained_coverage_risk"])
        invalid_outputs += b4["invalid_outputs"] + b2["invalid_outputs"]
        for section in ("per_target", "per_target_stance_cell"):
            if set(b4[section]) != set(b2[section]):
                raise ValueError(f"paired {section} inventories differ")
            for name, baseline in b4[section].items():
                challenger = b2[section][name]
                if baseline["reference_support"] != challenger["reference_support"]:
                    raise ValueError(f"paired {section} support drifted")
                if baseline["reference_support"] >= criteria["supported_reference_minimum"]:
                    regression = baseline["score"] - challenger["score"]
                    if regression > criteria["maximum_supported_target_or_cell_regression"]:
                        supported_regressions.append(
                            {
                                "seed": seed,
                                "surface": section,
                                "cell": name,
                                "regression": round(regression, 6),
                                "reference_support": baseline["reference_support"],
                            }
                        )

    mean_gain = sum(gains) / len(gains)
    improved = sum(gain > 0 for gain in gains)
    mean_presence_decline = sum(presence_declines) / len(presence_declines)
    maximum_calibration_worsening = max(calibration_worsening)
    maximum_risk_worsening = max(risk_worsening)
    guards = {
        "minimum_gain": mean_gain >= criteria["minimum_mean_paired_tuple_f1_gain"],
        "improved_pairs": improved >= criteria["minimum_improved_pairs"],
        "target_presence": mean_presence_decline
        <= criteria["maximum_mean_target_presence_f1_decline"],
        "supported_targets_and_cells": not supported_regressions,
        "calibration": maximum_calibration_worsening
        <= criteria["maximum_calibration_or_retained_risk_worsening"],
        "retained_coverage_risk": maximum_risk_worsening
        <= criteria["maximum_calibration_or_retained_risk_worsening"],
        "invalid_outputs": invalid_outputs <= criteria["maximum_invalid_outputs"],
    }
    if mean_gain < criteria["scrap_below_mean_paired_tuple_f1_gain"]:
        verdict = "scrap_b2_keep_b4"
    elif all(guards.values()):
        verdict = "promote_b2"
    else:
        verdict = "retain_b4"
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPRESENTATION_GATE_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": _sha(phase_run_id, where="phase_run_id"),
        "trial_receipt_ids": receipt_ids,
        "development_only": True,
        "primary_metric_component": "development_probability",
        "natural_probability_weighting": clean["bindings"][
            "development_probability_weighting"
        ],
        "supported_target_and_cell_gate_scope": (
            "development_probability_design_weighted"
        ),
        "selected_representation": "B2" if verdict == "promote_b2" else "B4",
        "verdict": verdict,
        "aggregate_evidence": {
            "mean_paired_tuple_f1_gain": round(mean_gain, 6),
            "paired_seed_gains": [round(value, 6) for value in gains],
            "improved_pairs": improved,
            "mean_target_presence_f1_decline": round(mean_presence_decline, 6),
            "maximum_calibration_worsening": round(maximum_calibration_worsening, 6),
            "maximum_retained_coverage_risk_worsening": round(maximum_risk_worsening, 6),
            "supported_regression_count": len(supported_regressions),
            "supported_regressions": supported_regressions,
            "invalid_outputs": invalid_outputs,
            "gates": guards,
        },
        "calibration_or_threshold_frozen": False,
        "locked_test_rows_accessed": 0,
        "human_validation_claim_authorised": False,
        "corpus_inference_authorised": False,
    }
    result = {**body, "gate_receipt_id": canonical_sha256(body)}
    assert_metadata_only(result, where="factorised representation gate")
    return result


def publish_immutable_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish one content-addressed metadata artefact without overwrite."""

    assert_metadata_only(value, where=str(path))
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FactorisedExperimentContractError("immutable metadata artefact already differs")
    else:
        temporary = path.with_suffix(path.suffix + ".incomplete")
        if temporary.exists():
            raise FactorisedExperimentContractError("stale incomplete metadata artefact exists")
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
        temporary.replace(path)
    return {
        "relative_path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


__all__ = [
    "ACCOUNT_GPU_LIMIT",
    "COMPONENTS",
    "EXPECTED_TRIALS",
    "GPU_TYPE",
    "HARD_COST_CAP_USD",
    "MAX_CONCURRENT_TRIALS",
    "MODEL_ID",
    "MODEL_REVISION",
    "NAMESPACE",
    "OUTPUT_PREFIX",
    "REGISTERED_SEEDS",
    "TARGET_STANCE_COMPONENTS",
    "TOKENIZER_REVISION",
    "FactorisedExperimentContractError",
    "artifact_descriptor",
    "budget_status",
    "build_run_manifest",
    "build_source_bundle",
    "build_trial_receipt",
    "canonical_sha256",
    "evaluate_representation_gate",
    "file_sha256",
    "freeze_experiment_contract",
    "freeze_trial_spec",
    "frozen_component_config",
    "make_trial_job",
    "publish_immutable_json",
    "repo_artifact_descriptor",
    "representation_criteria",
    "validate_experiment_contract",
    "validate_public_trial_metrics",
    "validate_run_manifest",
    "validate_source_bundle",
    "validate_trial_receipt",
    "validate_trial_spec",
]
