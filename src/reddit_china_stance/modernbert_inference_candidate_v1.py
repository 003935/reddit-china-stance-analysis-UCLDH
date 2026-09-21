"""Registered single-candidate ModernBERT-large/B4 training and inference.

This namespace is intentionally separate from the frozen acquisition-policy
comparison.  It registers one optimiser seed and one one-pass factorised B4
target/stance encoder.  The four-logit stance head is retained for checkpoint
compatibility, but mixed reference instances are removed from stance
cross-entropy at collation and the inference decoder projects explicitly onto
the three non-mixed logits.

Only training, development inference/calibration and a bounded throughput smoke
are represented here.  Locked-test and corpus inference are outside this
module's authority.  There is no GPU fallback, retry or experiment sweep.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    IGNORE_INDEX,
    STANCE_CLASSES_B4,
    TARGET_CLASSES,
    TextRenderConfig,
    encode_v2_label,
    render_factorised_text,
)
from reddit_china_stance.modernbert_factorised_experiment import (
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    artifact_descriptor,
    canonical_sha256,
    file_sha256,
    frozen_component_config,
)
from reddit_china_stance.modernbert_factorised_training import (
    FactorisedDynamicPaddingCollator,
    FactorisedOptimisationConfig,
    collect_development_logits,
    create_adamw,
    create_component_model,
    load_pinned_tokenizer,
    load_private_frame,
    tokenise_factorised_record,
    train_epoch,
)
from reddit_china_stance.modernbert_factorised_training import (
    _thread_set_sha256 as _factorised_thread_set_sha256,
)
from reddit_china_stance.semantic_ontology_v2 import validate_v2_label

SCHEMA_VERSION = "1.0.0"
NAMESPACE = "student-modernbert-inference-candidate-v1"
EXPERIMENT_KIND = "modernbert-inference-candidate-experiment-v1"
RUN_MANIFEST_KIND = "modernbert-inference-candidate-run-manifest-v1"
CALIBRATION_KIND = "modernbert-inference-candidate-development-calibration-v1"
TRAINING_RECEIPT_KIND = "modernbert-inference-candidate-training-receipt-v1"
THROUGHPUT_RECEIPT_KIND = "modernbert-inference-candidate-throughput-smoke-v1"
CHECKPOINT_KIND = "modernbert-inference-candidate-checkpoint-v1"
COMBINED_FRAME_PREPARATION_KIND = "modernbert-inference-candidate-combined-frame-preparation-v1"

COMPONENT = "target_stance_b4"
REGISTERED_SEED = 47
REGISTERED_SEEDS = (REGISTERED_SEED,)
EXPECTED_CANDIDATES = 1
GPU_TYPE = "L4"
MAX_CONCURRENT_CANDIDATES = 1
MAX_TRAINING_ATTEMPTS = 1
CUDA_PREFLIGHT_MAX_GPU_SECONDS = 1_800
TRAIN_MAX_GPU_SECONDS = 7_200
THROUGHPUT_MAX_GPU_SECONDS = 1_800
THROUGHPUT_SMOKE_ROWS = 128
THROUGHPUT_BATCH_SIZE = 32
DEVELOPMENT_ROWS = 600
ACQUISITION_LABEL_ROWS = 2_000
HARD_COST_CAP_USD = Decimal("200")

_ACQUISITION_LABEL_LINEAGE_COLUMNS = {
    "source_sample_id",
    "thread_id",
    "label_json",
    "primary_training_eligible",
}
_COMBINED_TRAINING_LINEAGE_COLUMNS = {
    "item_id",
    "thread_id",
    "label_json",
    "acquisition_source_sample_id",
    "acquisition_label_sha256",
    "primary_training_eligible",
}
_BASE_FRAME_COLUMNS = (
    "item_id",
    "frame",
    "thread_id",
    "target_text",
    "parent_context",
    "submission_context",
    "label_json",
    "selection_component",
    "selection_stratum",
    "inclusion_probability_numerator",
    "inclusion_probability_denominator",
    "inclusion_probability",
    "probability_scope",
)
_ACQUISITION_SOURCE_COLUMNS = {
    "opaque_id",
    "source_sample_id",
    "thread_id",
    "target_text",
    "parent_context",
    "submission_context",
}

MIXED_STANCE = "mixed"
MIXED_LOGIT_INDEX = STANCE_CLASSES_B4.index(MIXED_STANCE)
DECODE_STANCES = tuple(stance for stance in STANCE_CLASSES_B4 if stance != MIXED_STANCE)
DECODE_LOGIT_INDICES = tuple(STANCE_CLASSES_B4.index(stance) for stance in DECODE_STANCES)
TEMPERATURE_GRID = (0.50, 0.67, 0.80, 1.00, 1.25, 1.50, 2.00)
InferenceScope = Literal["development", "throughput_smoke"]

if MIXED_LOGIT_INDEX != 1 or DECODE_LOGIT_INDICES != (0, 2, 3):
    raise RuntimeError("frozen B4 stance order drifted")


class InferenceCandidateContractError(RuntimeError):
    """Raised when the registered candidate or its evidence bindings drift."""


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise InferenceCandidateContractError(
                f"existing immutable candidate artefact differs: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise InferenceCandidateContractError(
            f"stale incomplete candidate artefact exists: {temporary}"
        )
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number")
    clean = float(value)
    if not math.isfinite(clean):
        raise ValueError(f"{where} must be a finite number")
    return clean


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def candidate_optimisation_config() -> FactorisedOptimisationConfig:
    """Return the unchanged optimiser recipe from the retained B4 study."""

    frozen = frozen_component_config(COMPONENT)
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "stance_representation": "B4",
        "max_length": 768,
        "pooling": "masked_mean",
        "attention_implementation": "sdpa",
        "reference_compile": False,
        "precision": "bf16",
        "max_epochs": 8,
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise InferenceCandidateContractError("retained B4 model recipe drifted")
    return FactorisedOptimisationConfig(
        encoder_learning_rate=frozen["encoder_learning_rate"],
        head_learning_rate_multiplier=frozen["head_learning_rate_multiplier"],
        dropout=frozen["dropout"],
        weight_decay=frozen["weight_decay"],
        adam_beta1=frozen["adam_betas"][0],
        adam_beta2=frozen["adam_betas"][1],
        adam_epsilon=frozen["adam_epsilon"],
        warmup_ratio=frozen["warmup_ratio"],
        gradient_clip_norm=frozen["gradient_clip"],
        per_device_batch_size=frozen["per_device_batch_size"],
        gradient_accumulation_steps=frozen["gradient_accumulation_steps"],
        effective_batch_size=frozen["effective_batch_size"],
        use_bf16=frozen["precision"] == "bf16",
    )


def registered_candidate_config() -> dict[str, Any]:
    """Return the sole model/training/decoder candidate registered here."""

    base = frozen_component_config(COMPONENT)
    optimisation_keys = (
        "encoder_learning_rate",
        "head_learning_rate_multiplier",
        "dropout",
        "effective_batch_size",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "precision",
        "optimiser",
        "weight_decay",
        "adam_betas",
        "adam_epsilon",
        "warmup_ratio",
        "scheduler",
        "gradient_clip",
        "max_length",
        "pooling",
        "attention_implementation",
        "reference_compile",
        "loss_weighting",
        "max_epochs",
    )
    body = {
        "candidate_count": EXPECTED_CANDIDATES,
        "component": COMPONENT,
        "optimiser_seed": REGISTERED_SEED,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "encoder_architecture": "single_modernbert_large_one_pass",
        "target_presence_logits": len(TARGET_CLASSES),
        "analytic_targets": list(ANALYTIC_TARGET_CLASSES),
        "stance_head": {
            "representation": "B4",
            "checkpoint_logit_order": list(STANCE_CLASSES_B4),
            "checkpoint_logits_per_target": len(STANCE_CLASSES_B4),
            "mixed_training_treatment": "mask_from_stance_cross_entropy_only",
            "mixed_presence_treatment": "retain_target_presence_supervision",
            "decode_stances": list(DECODE_STANCES),
            "decode_logit_indices": list(DECODE_LOGIT_INDICES),
            "excluded_decode_logit": {
                "stance": MIXED_STANCE,
                "index": MIXED_LOGIT_INDEX,
            },
        },
        "optimisation": {key: base[key] for key in optimisation_keys},
        "retained_b4_source_config_sha256": base["config_sha256"],
        "checkpoint_score": "non_mixed_conditional_stance_accuracy",
        "checkpoint_selection_uses": "development_only",
        "calibration_uses": "development_only",
        "combined_training_frame": {
            "acquisition_label_rows_required": ACQUISITION_LABEL_ROWS,
            "acquisition_row_identity": "item_id_equals_source_sample_id",
            "acquisition_label_binding": "exact_label_json_and_canonical_sha256",
            "acquisition_thread_binding": "exact_thread_id",
            "eligibility_binding": "exact_primary_training_eligible_boolean",
            "training_use": "base_plus_primary_training_eligible_acquisition_rows",
            "development_thread_overlap_allowed": False,
        },
        "locked_test_authorised": False,
        "corpus_inference_authorised": False,
        "gpu_fallback_allowed": False,
        "retry_authorised": False,
        "broad_experiment_authorised": False,
    }
    return {**body, "candidate_config_sha256": canonical_sha256(body)}


def mask_mixed_stance_feature(feature: Mapping[str, Any]) -> dict[str, Any]:
    """Clear only mixed stance supervision while preserving the B4 label and presence.

    The returned feature remains compatible with the frozen B4 collator/model.
    The input is not mutated.  In particular, ``stance_b4_labels`` and both
    target-presence arrays are copied unchanged.
    """

    required = {
        "stance_b4_labels",
        "stance_mask",
        "target_presence_labels",
        "target_presence_mask",
    }
    if not required <= set(feature):
        raise ValueError("candidate feature omits B4 stance or target-presence fields")
    labels = list(feature["stance_b4_labels"])
    masks = list(feature["stance_mask"])
    presence = list(feature["target_presence_labels"])
    presence_mask = list(feature["target_presence_mask"])
    if len(labels) != len(ANALYTIC_TARGET_CLASSES) or len(masks) != len(labels):
        raise ValueError("candidate B4 stance feature width drifted")
    if len(presence) != len(TARGET_CLASSES) or len(presence_mask) != len(presence):
        raise ValueError("candidate target-presence feature width drifted")
    clean_masks: list[int] = []
    for index, (label, mask) in enumerate(zip(labels, masks, strict=True)):
        if mask not in (0, 1, False, True):
            raise ValueError(f"stance_mask[{index}] must be binary")
        if bool(mask):
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 4:
                raise ValueError(f"stance_b4_labels[{index}] is invalid")
            clean_masks.append(int(label != MIXED_LOGIT_INDEX))
        else:
            if label != IGNORE_INDEX:
                raise ValueError("masked source stance labels must use IGNORE_INDEX")
            clean_masks.append(0)
    return {
        **dict(feature),
        "stance_b4_labels": labels,
        "stance_mask": clean_masks,
        "target_presence_labels": presence,
        "target_presence_mask": presence_mask,
    }


class InferenceCandidateB4Collator:
    """Candidate boundary that masks mixed CE while reusing the frozen collator."""

    def __init__(self, tokenizer: Any, *, return_tensors: str | None = "pt") -> None:
        self._base = FactorisedDynamicPaddingCollator(
            tokenizer,
            component=COMPONENT,
            return_tensors=return_tensors,
        )

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return self._base([mask_mixed_stance_feature(feature) for feature in features])


def _softmax(values: Sequence[float], *, temperature: float) -> tuple[float, ...]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _sigmoid(value: float, *, temperature: float) -> float:
    scaled = value / temperature
    if scaled >= 0:
        factor = math.exp(-scaled)
        return 1.0 / (1.0 + factor)
    factor = math.exp(scaled)
    return factor / (1.0 + factor)


def decode_three_class_stance_logits(
    logits: Sequence[float], *, temperature: float = 1.0
) -> dict[str, Any]:
    """Decode only B4 indices 0, 2 and 3; the mixed logit has no effect."""

    values = tuple(
        _finite(value, where=f"stance_logits[{index}]") for index, value in enumerate(logits)
    )
    if len(values) != len(STANCE_CLASSES_B4):
        raise ValueError("stance_logits must contain exactly four B4 logits")
    projected = tuple(values[index] for index in DECODE_LOGIT_INDICES)
    probabilities = _softmax(projected, temperature=temperature)
    selected = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "stance": DECODE_STANCES[selected],
        "probabilities": dict(zip(DECODE_STANCES, probabilities, strict=True)),
        "source_logit_indices": list(DECODE_LOGIT_INDICES),
    }


def validate_inference_scope(scope: str) -> InferenceScope:
    if scope not in {"development", "throughput_smoke"}:
        raise InferenceCandidateContractError(
            "candidate inference is restricted to development and throughput smoke"
        )
    return scope  # type: ignore[return-value]


def _temperature(value: Any, *, where: str) -> float:
    clean = _finite(value, where=where)
    if clean <= 0:
        raise ValueError(f"{where} must be positive")
    return clean


def decode_candidate_logits(
    *,
    target_presence_logits: Sequence[float],
    stance_logits: Sequence[Sequence[float]],
    calibration: Mapping[str, Any],
    scope: str,
) -> dict[str, Any]:
    """Decode one bounded candidate output without forced-target behaviour."""

    validate_inference_scope(scope)
    clean_calibration = validate_development_calibration(calibration)
    presence_values = tuple(
        _finite(value, where=f"target_presence_logits[{index}]")
        for index, value in enumerate(target_presence_logits)
    )
    if len(presence_values) != len(TARGET_CLASSES):
        raise ValueError("target_presence_logits width drifted")
    if len(stance_logits) != len(ANALYTIC_TARGET_CLASSES):
        raise ValueError("stance_logits target width drifted")
    target_probabilities = {
        target: _sigmoid(
            logit,
            temperature=clean_calibration["target_presence_temperature"],
        )
        for target, logit in zip(TARGET_CLASSES, presence_values, strict=True)
    }
    present_targets = [
        target
        for target in TARGET_CLASSES
        if target_probabilities[target] >= clean_calibration["target_presence_threshold"]
    ]
    decoded_stances = {
        target: decode_three_class_stance_logits(
            stance_logits[index],
            temperature=clean_calibration["stance_temperature"],
        )
        for index, target in enumerate(ANALYTIC_TARGET_CLASSES)
        if target in present_targets
    }
    return {
        "target_presence": target_probabilities,
        "present_targets": present_targets,
        "stances": decoded_stances,
        "forced_target_selections": 0,
    }


def _binary_nll(logits: Sequence[float], labels: Sequence[int], temperature: float) -> float:
    losses = []
    for logit, label in zip(logits, labels, strict=True):
        probability = min(max(_sigmoid(logit, temperature=temperature), 1e-12), 1 - 1e-12)
        losses.append(-math.log(probability if label else 1.0 - probability))
    return sum(losses) / len(losses)


def _stance_nll(
    logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float
) -> float:
    losses = []
    decode_index = {b4_index: index for index, b4_index in enumerate(DECODE_LOGIT_INDICES)}
    for row, label in zip(logits, labels, strict=True):
        projected = [row[index] for index in DECODE_LOGIT_INDICES]
        probabilities = _softmax(projected, temperature=temperature)
        losses.append(-math.log(max(probabilities[decode_index[label]], 1e-12)))
    return sum(losses) / len(losses)


def _select_temperature(loss: Any) -> tuple[float, float]:
    candidates = [(float(loss(value)), abs(value - 1.0), value) for value in TEMPERATURE_GRID]
    selected_loss, _, selected_temperature = min(candidates)
    return selected_temperature, selected_loss


def fit_development_calibration(
    development_frame: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit fixed-grid temperatures using the registered development frame only."""

    frame = artifact_descriptor(development_frame, where="development_frame")
    if frame.get("frame") != "development":
        raise InferenceCandidateContractError("calibration source must be development")
    if frame.get("row_count") != len(rows):
        raise InferenceCandidateContractError("calibration rows do not conserve development")
    presence_logits: list[float] = []
    presence_labels: list[int] = []
    stance_logits: list[list[float]] = []
    stance_labels: list[int] = []
    excluded_mixed = 0
    seen: set[str] = set()
    expected_keys = {
        "item_id",
        "target_presence_logits",
        "target_presence_labels",
        "reference_material",
        "stance_logits",
        "stance_labels",
        "stance_mask",
    }
    for row_index, row in enumerate(rows):
        if set(row) != expected_keys:
            raise ValueError(f"development calibration row {row_index} schema drifted")
        item_id = row["item_id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("development calibration IDs are invalid or duplicate")
        seen.add(item_id)
        raw_presence_logits = row["target_presence_logits"]
        raw_presence_labels = row["target_presence_labels"]
        raw_stance_logits = row["stance_logits"]
        raw_stance_labels = row["stance_labels"]
        raw_stance_mask = row["stance_mask"]
        if (
            not isinstance(raw_presence_logits, Sequence)
            or len(raw_presence_logits) != len(TARGET_CLASSES)
            or not isinstance(raw_presence_labels, Sequence)
            or len(raw_presence_labels) != len(TARGET_CLASSES)
            or not isinstance(raw_stance_logits, Sequence)
            or len(raw_stance_logits) != len(ANALYTIC_TARGET_CLASSES)
            or not isinstance(raw_stance_labels, Sequence)
            or len(raw_stance_labels) != len(ANALYTIC_TARGET_CLASSES)
            or not isinstance(raw_stance_mask, Sequence)
            or len(raw_stance_mask) != len(ANALYTIC_TARGET_CLASSES)
        ):
            raise ValueError("development calibration tensor width drifted")
        material = row["reference_material"]
        if not isinstance(material, bool):
            raise ValueError("reference_material must be boolean")
        if material:
            for target_index, (logit, label) in enumerate(
                zip(raw_presence_logits, raw_presence_labels, strict=True)
            ):
                presence_logits.append(
                    _finite(logit, where=f"presence[{row_index}][{target_index}]")
                )
                if label not in (0, 1, False, True):
                    raise ValueError("target-presence calibration labels must be binary")
                presence_labels.append(int(label))
        for target_index, (raw_logits, label, mask) in enumerate(
            zip(raw_stance_logits, raw_stance_labels, raw_stance_mask, strict=True)
        ):
            if mask not in (0, 1, False, True):
                raise ValueError("stance calibration masks must be binary")
            if not bool(mask):
                if label != IGNORE_INDEX:
                    raise ValueError("masked stance calibration labels must use IGNORE_INDEX")
                continue
            if not material:
                raise ValueError("non-material rows cannot expose stance calibration labels")
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 4:
                raise ValueError("stance calibration label is invalid")
            numeric = [
                _finite(value, where=f"stance[{row_index}][{target_index}][{logit_index}]")
                for logit_index, value in enumerate(raw_logits)
            ]
            if len(numeric) != len(STANCE_CLASSES_B4):
                raise ValueError("stance calibration logits must retain four B4 values")
            if label == MIXED_LOGIT_INDEX:
                excluded_mixed += 1
                continue
            stance_logits.append(numeric)
            stance_labels.append(label)
    if not presence_logits or not stance_logits:
        raise InferenceCandidateContractError(
            "development calibration lacks non-mixed stance or presence support"
        )
    presence_temperature, presence_nll = _select_temperature(
        lambda value: _binary_nll(presence_logits, presence_labels, value)
    )
    stance_temperature, stance_nll = _select_temperature(
        lambda value: _stance_nll(stance_logits, stance_labels, value)
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": CALIBRATION_KIND,
        "candidate_config_sha256": registered_candidate_config()["candidate_config_sha256"],
        "source_frame": frame,
        "source_frame_role": "development_only",
        "temperature_grid": list(TEMPERATURE_GRID),
        "target_presence_temperature": presence_temperature,
        "target_presence_nll": presence_nll,
        "target_presence_support": len(presence_logits),
        "target_presence_threshold": 0.5,
        "stance_temperature": stance_temperature,
        "stance_nll": stance_nll,
        "stance_support": len(stance_logits),
        "excluded_mixed_stance_instances": excluded_mixed,
        "decode_stances": list(DECODE_STANCES),
        "decode_logit_indices": list(DECODE_LOGIT_INDICES),
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    return {**body, "calibration_id": canonical_sha256(body)}


def validate_development_calibration(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "candidate_config_sha256",
        "source_frame",
        "source_frame_role",
        "temperature_grid",
        "target_presence_temperature",
        "target_presence_nll",
        "target_presence_support",
        "target_presence_threshold",
        "stance_temperature",
        "stance_nll",
        "stance_support",
        "excluded_mixed_stance_instances",
        "decode_stances",
        "decode_logit_indices",
        "locked_test_rows_accessed",
        "corpus_rows_accessed",
        "calibration_id",
    }
    if set(value) != expected:
        raise ValueError("development calibration schema drifted")
    source = artifact_descriptor(value["source_frame"], where="calibration.source_frame")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != CALIBRATION_KIND
        or value.get("candidate_config_sha256")
        != registered_candidate_config()["candidate_config_sha256"]
        or source.get("frame") != "development"
        or value.get("source_frame_role") != "development_only"
        or value.get("temperature_grid") != list(TEMPERATURE_GRID)
        or value.get("target_presence_threshold") != 0.5
        or value.get("decode_stances") != list(DECODE_STANCES)
        or value.get("decode_logit_indices") != list(DECODE_LOGIT_INDICES)
        or value.get("locked_test_rows_accessed") != 0
        or value.get("corpus_rows_accessed") != 0
    ):
        raise InferenceCandidateContractError("development calibration binding drifted")
    for key in ("target_presence_temperature", "stance_temperature"):
        _temperature(value[key], where=key)
    for key in ("target_presence_nll", "stance_nll"):
        if _finite(value[key], where=key) < 0:
            raise ValueError(f"{key} must be non-negative")
    for key in (
        "target_presence_support",
        "stance_support",
    ):
        _positive_int(value[key], where=key)
    excluded = value["excluded_mixed_stance_instances"]
    if isinstance(excluded, bool) or not isinstance(excluded, int) or excluded < 0:
        raise ValueError("excluded_mixed_stance_instances must be non-negative")
    body = {key: value[key] for key in expected - {"calibration_id"}}
    if value["calibration_id"] != canonical_sha256(body):
        raise InferenceCandidateContractError("development calibration digest drifted")
    if manifest is not None:
        clean_manifest = validate_run_manifest(manifest)
        expected_source = clean_manifest["experiment_contract"]["bindings"]["development_frame"]
        if source != expected_source:
            raise InferenceCandidateContractError(
                "calibration source differs from the registered development frame"
            )
    return _clone(value)


def _decimal(value: Any, *, where: str, positive: bool = False) -> Decimal:
    try:
        clean = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be decimal") from exc
    if not clean.is_finite() or clean < 0 or (positive and clean <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{where} must be finite and {qualifier}")
    return clean


def freeze_experiment_contract(
    *,
    training_frame: Mapping[str, Any],
    development_frame: Mapping[str, Any],
    acquisition_labels: Mapping[str, Any],
    acquisition_receipt: Mapping[str, Any],
    acquisition_run_id: str,
    source_bundle_sha256: str,
    dependency_lock_sha256: str,
    rate_card_usd_per_gpu_second: str | float,
    cumulative_measured_spend_usd: str | float,
    active_reservation_usd: str | float,
    planned_phase_upper_usd: str | float,
    hard_cost_cap_usd: str | float = "200",
) -> dict[str, Any]:
    """Freeze exact labelled inputs and the sole candidate before execution."""

    training = artifact_descriptor(training_frame, where="training_frame")
    development = artifact_descriptor(development_frame, where="development_frame")
    labels = artifact_descriptor(acquisition_labels, where="acquisition_labels")
    receipt = artifact_descriptor(acquisition_receipt, where="acquisition_receipt")
    if training.get("frame") != "training" or "thread_set_sha256" not in training:
        raise ValueError("training frame must bind its training identity and thread set")
    if (
        development.get("frame") != "development"
        or development.get("row_count") != DEVELOPMENT_ROWS
        or "thread_set_sha256" not in development
    ):
        raise ValueError("development frame must bind exactly 600 rows and its thread set")
    if training["thread_set_sha256"] == development["thread_set_sha256"]:
        raise InferenceCandidateContractError("training and development thread sets overlap")
    if labels.get("row_count") != ACQUISITION_LABEL_ROWS:
        raise ValueError("acquisition label artefact must bind exactly 2,000 rows")
    rate = _decimal(
        rate_card_usd_per_gpu_second,
        where="rate_card_usd_per_gpu_second",
        positive=True,
    )
    measured = _decimal(cumulative_measured_spend_usd, where="cumulative_measured_spend_usd")
    active = _decimal(active_reservation_usd, where="active_reservation_usd")
    planned = _decimal(planned_phase_upper_usd, where="planned_phase_upper_usd")
    cap = _decimal(hard_cost_cap_usd, where="hard_cost_cap_usd", positive=True)
    if cap > HARD_COST_CAP_USD:
        raise ValueError("hard cost cap exceeds the shared $200 maximum")
    if measured + active + planned > cap:
        raise InferenceCandidateContractError("candidate plan exceeds the shared cost cap")
    bindings = {
        "training_frame": training,
        "development_frame": development,
        "acquisition_labels": labels,
        "acquisition_receipt": receipt,
        "acquisition_run_id": _sha(acquisition_run_id, where="acquisition_run_id"),
        "source_bundle_sha256": _sha(source_bundle_sha256, where="source_bundle_sha256"),
        "dependency_lock_sha256": _sha(dependency_lock_sha256, where="dependency_lock_sha256"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPERIMENT_KIND,
        "namespace": NAMESPACE,
        "bindings": bindings,
        "registered_design": registered_candidate_config(),
        "compute": {
            "allowed_gpus": [GPU_TYPE],
            "max_concurrent_candidates": MAX_CONCURRENT_CANDIDATES,
            "max_training_attempts": MAX_TRAINING_ATTEMPTS,
            "gpu_fallback_allowed": False,
            "retry_authorised": False,
            "rate_card_usd_per_gpu_second": {GPU_TYPE: format(rate, "f")},
            "cumulative_measured_spend_usd": format(measured, "f"),
            "active_reservation_usd": format(active, "f"),
            "planned_phase_upper_usd": format(planned, "f"),
            "hard_cost_cap_usd": format(cap, "f"),
            "remaining_after_plan_usd": format(cap - measured - active - planned, "f"),
        },
        "evidence_boundary": {
            "development_calibration_only": True,
            "locked_test_authorised": False,
            "locked_test_rows_accessed": 0,
            "corpus_inference_authorised": False,
            "corpus_rows_accessed": 0,
            "human_validation_claim_authorised": False,
        },
    }
    return {**body, "experiment_run_id": canonical_sha256(body)}


def validate_experiment_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "namespace",
        "bindings",
        "registered_design",
        "compute",
        "evidence_boundary",
        "experiment_run_id",
    }
    if set(value) != expected:
        raise ValueError("candidate experiment schema drifted")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != EXPERIMENT_KIND
        or value.get("namespace") != NAMESPACE
        or value.get("registered_design") != registered_candidate_config()
    ):
        raise InferenceCandidateContractError("candidate experiment identity drifted")
    bindings = value.get("bindings")
    compute = value.get("compute")
    boundary = value.get("evidence_boundary")
    if not all(isinstance(item, Mapping) for item in (bindings, compute, boundary)):
        raise ValueError("candidate nested contract is invalid")
    expected_binding_keys = {
        "training_frame",
        "development_frame",
        "acquisition_labels",
        "acquisition_receipt",
        "acquisition_run_id",
        "source_bundle_sha256",
        "dependency_lock_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
    }
    if set(bindings) != expected_binding_keys:
        raise ValueError("candidate input binding schema drifted")
    training = artifact_descriptor(bindings["training_frame"], where="training_frame")
    development = artifact_descriptor(bindings["development_frame"], where="development_frame")
    labels = artifact_descriptor(bindings["acquisition_labels"], where="acquisition_labels")
    artifact_descriptor(bindings["acquisition_receipt"], where="acquisition_receipt")
    if (
        training.get("frame") != "training"
        or development.get("frame") != "development"
        or development.get("row_count") != DEVELOPMENT_ROWS
        or labels.get("row_count") != ACQUISITION_LABEL_ROWS
        or training.get("thread_set_sha256") == development.get("thread_set_sha256")
    ):
        raise InferenceCandidateContractError("candidate frame binding drifted")
    if (
        bindings.get("model_id") != MODEL_ID
        or bindings.get("model_revision") != MODEL_REVISION
        or bindings.get("tokenizer_revision") != TOKENIZER_REVISION
    ):
        raise InferenceCandidateContractError("pinned model binding drifted")
    for key in ("acquisition_run_id", "source_bundle_sha256", "dependency_lock_sha256"):
        _sha(bindings.get(key), where=key)
    expected_compute_keys = {
        "allowed_gpus",
        "max_concurrent_candidates",
        "max_training_attempts",
        "gpu_fallback_allowed",
        "retry_authorised",
        "rate_card_usd_per_gpu_second",
        "cumulative_measured_spend_usd",
        "active_reservation_usd",
        "planned_phase_upper_usd",
        "hard_cost_cap_usd",
        "remaining_after_plan_usd",
    }
    if set(compute) != expected_compute_keys or (
        compute.get("allowed_gpus") != [GPU_TYPE]
        or compute.get("max_concurrent_candidates") != 1
        or compute.get("max_training_attempts") != 1
        or compute.get("gpu_fallback_allowed") is not False
        or compute.get("retry_authorised") is not False
    ):
        raise InferenceCandidateContractError("candidate compute contract drifted")
    rates = compute.get("rate_card_usd_per_gpu_second")
    if not isinstance(rates, Mapping) or set(rates) != {GPU_TYPE}:
        raise ValueError("candidate rate card must bind exactly L4")
    rate = _decimal(rates[GPU_TYPE], where="L4 rate", positive=True)
    measured = _decimal(compute["cumulative_measured_spend_usd"], where="measured")
    active = _decimal(compute["active_reservation_usd"], where="active")
    planned = _decimal(compute["planned_phase_upper_usd"], where="planned")
    cap = _decimal(compute["hard_cost_cap_usd"], where="cap", positive=True)
    remaining = _decimal(compute["remaining_after_plan_usd"], where="remaining")
    if (
        rate <= 0
        or cap > HARD_COST_CAP_USD
        or measured + active + planned > cap
        or remaining != cap - measured - active - planned
    ):
        raise InferenceCandidateContractError("candidate cost binding drifted")
    if dict(boundary) != {
        "development_calibration_only": True,
        "locked_test_authorised": False,
        "locked_test_rows_accessed": 0,
        "corpus_inference_authorised": False,
        "corpus_rows_accessed": 0,
        "human_validation_claim_authorised": False,
    }:
        raise InferenceCandidateContractError("candidate evidence boundary drifted")
    body = {key: value[key] for key in expected - {"experiment_run_id"}}
    if value["experiment_run_id"] != canonical_sha256(body):
        raise InferenceCandidateContractError("candidate experiment digest drifted")
    return _clone(value)


def build_run_manifest(experiment_contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_experiment_contract(experiment_contract)
    preflight_body = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "component": COMPONENT,
        "rows": 1,
        "max_gpu_seconds": CUDA_PREFLIGHT_MAX_GPU_SECONDS,
        "max_attempts": 1,
    }
    preflight = {**preflight_body, "preflight_id": canonical_sha256(preflight_body)}
    training_body = {
        "component": COMPONENT,
        "optimiser_seed": REGISTERED_SEED,
        "candidate_config_sha256": contract["registered_design"]["candidate_config_sha256"],
        "training_frame_sha256": contract["bindings"]["training_frame"]["sha256"],
        "development_frame_sha256": contract["bindings"]["development_frame"]["sha256"],
        "max_gpu_seconds": TRAIN_MAX_GPU_SECONDS,
        "max_attempts": MAX_TRAINING_ATTEMPTS,
    }
    training_job = {**training_body, "job_id": canonical_sha256(training_body)}
    throughput_body = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "component": COMPONENT,
        "rows": THROUGHPUT_SMOKE_ROWS,
        "batch_size": THROUGHPUT_BATCH_SIZE,
        "max_gpu_seconds": THROUGHPUT_MAX_GPU_SECONDS,
        "scope": "throughput_smoke",
    }
    throughput = {**throughput_body, "smoke_id": canonical_sha256(throughput_body)}
    rate = Decimal(contract["compute"]["rate_card_usd_per_gpu_second"][GPU_TYPE])
    reserved = rate * Decimal(
        CUDA_PREFLIGHT_MAX_GPU_SECONDS + TRAIN_MAX_GPU_SECONDS + THROUGHPUT_MAX_GPU_SECONDS
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_MANIFEST_KIND,
        "experiment_run_id": contract["experiment_run_id"],
        "experiment_contract": contract,
        "cuda_preflight": preflight,
        "training_jobs": [training_job],
        "throughput_smoke": throughput,
        "reserved_cost_usd": format(reserved, "f"),
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    return {**body, "phase_run_id": canonical_sha256(body)}


def validate_run_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "experiment_contract",
        "cuda_preflight",
        "training_jobs",
        "throughput_smoke",
        "reserved_cost_usd",
        "locked_test_rows_accessed",
        "corpus_rows_accessed",
        "phase_run_id",
    }
    if set(value) != expected or value.get("kind") != RUN_MANIFEST_KIND:
        raise ValueError("candidate run manifest schema drifted")
    contract = validate_experiment_contract(value["experiment_contract"])
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("experiment_run_id") != contract["experiment_run_id"]
        or value.get("locked_test_rows_accessed") != 0
        or value.get("corpus_rows_accessed") != 0
    ):
        raise InferenceCandidateContractError("candidate run identity drifted")
    expected_manifest = build_run_manifest(contract)
    if dict(value) != expected_manifest:
        raise InferenceCandidateContractError("candidate run manifest drifted")
    return _clone(value)


def _bound_artifact_path(
    *, volume_root: Path, descriptor: Mapping[str, Any], where: str
) -> tuple[Path, dict[str, Any]]:
    clean = artifact_descriptor(descriptor, where=where)
    path = volume_root / clean["relative_path"]
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or file_sha256(path) != clean["sha256"]
    ):
        raise InferenceCandidateContractError(f"{where} is missing or corrupt")
    return path, clean


def _read_parquet_projection(path: Path, *, columns: set[str], where: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - Modal dependency path
        raise RuntimeError("candidate provenance validation requires PyArrow") from exc
    table = pq.read_table(path)
    if not columns <= set(table.column_names):
        missing = sorted(columns - set(table.column_names))
        raise InferenceCandidateContractError(f"{where} omits columns {missing}")
    return table.select(sorted(columns)).to_pylist()


def _validate_acquisition_receipt(
    path: Path,
    *,
    labels_descriptor: Mapping[str, Any],
    acquisition_run_id: str,
) -> None:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("acquisition receipt is not valid UTF-8 JSON") from exc
    if not isinstance(receipt, Mapping):
        raise ValueError("acquisition receipt must contain an object")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("kind") != "sol-teacher-acquisition-v2-receipt-v1"
        or receipt.get("status") != "complete"
        or receipt.get("run_id") != acquisition_run_id
        or receipt.get("row_count") != ACQUISITION_LABEL_ROWS
        or receipt.get("private_labels_parquet_sha256") != labels_descriptor["sha256"]
        or receipt.get("automatic_retry_count") != 0
        or receipt.get("row_replacement_count") != 0
    ):
        raise InferenceCandidateContractError(
            "acquisition receipt does not bind the exact completed 2,000-label artefact"
        )


def thread_set_sha256(values: Sequence[str]) -> str:
    """Use the authoritative factorised-frame thread-set digest."""

    return _factorised_thread_set_sha256(values)


def _validate_local_artifact(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    where: str,
) -> dict[str, Any]:
    clean = artifact_descriptor(descriptor, where=where)
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or file_sha256(path) != clean["sha256"]
    ):
        raise InferenceCandidateContractError(f"{where} is missing or corrupt")
    return clean


def _local_descriptor(
    path: Path,
    *,
    descriptor_root: Path,
    row_count: int | None = None,
    frame: str | None = None,
    thread_digest: str | None = None,
) -> dict[str, Any]:
    try:
        relative_path = path.resolve().relative_to(descriptor_root.resolve())
    except ValueError as exc:
        raise ValueError("candidate artefact is outside descriptor_root") from exc
    descriptor: dict[str, Any] = {
        "relative_path": relative_path.as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        descriptor["row_count"] = row_count
    if frame is not None:
        descriptor["frame"] = frame
    if thread_digest is not None:
        descriptor["thread_set_sha256"] = thread_digest
    return artifact_descriptor(descriptor, where="candidate local artefact")


def materialise_combined_training_frame(
    *,
    base_training_path: Path,
    base_training_descriptor: Mapping[str, Any],
    development_path: Path,
    development_descriptor: Mapping[str, Any],
    acquisition_source_path: Path,
    acquisition_source_descriptor: Mapping[str, Any],
    acquisition_labels_path: Path,
    acquisition_labels_descriptor: Mapping[str, Any],
    acquisition_receipt_path: Path,
    acquisition_receipt_descriptor: Mapping[str, Any],
    acquisition_run_id: str,
    output_path: Path,
    descriptor_root: Path,
) -> dict[str, Any]:
    """Create one immutable base-plus-all-acquisition lineage frame.

    Every acquired label is represented exactly once, including ineligible
    labels. Optimisation later filters only the exact primary-eligible subset.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - Modal dependency path
        raise RuntimeError("combined candidate preparation requires PyArrow") from exc

    base_descriptor = _validate_local_artifact(
        base_training_path,
        base_training_descriptor,
        where="base training frame",
    )
    development_descriptor = _validate_local_artifact(
        development_path,
        development_descriptor,
        where="development frame",
    )
    source_descriptor = _validate_local_artifact(
        acquisition_source_path,
        acquisition_source_descriptor,
        where="acquisition source",
    )
    labels_descriptor = _validate_local_artifact(
        acquisition_labels_path,
        acquisition_labels_descriptor,
        where="acquisition labels",
    )
    receipt_descriptor = _validate_local_artifact(
        acquisition_receipt_path,
        acquisition_receipt_descriptor,
        where="acquisition receipt",
    )
    _validate_acquisition_receipt(
        acquisition_receipt_path,
        labels_descriptor=labels_descriptor,
        acquisition_run_id=_sha(acquisition_run_id, where="acquisition_run_id"),
    )
    base_rows, _ = load_private_frame(
        base_training_path,
        base_descriptor,
        expected_frame="training",
    )
    development_rows, _ = load_private_frame(
        development_path,
        development_descriptor,
        expected_frame="development",
    )
    base_table = pq.read_table(base_training_path)
    if set(base_table.column_names) != set(_BASE_FRAME_COLUMNS):
        raise InferenceCandidateContractError("base training frame column contract drifted")
    source_rows = _read_parquet_projection(
        acquisition_source_path,
        columns=_ACQUISITION_SOURCE_COLUMNS,
        where="acquisition source",
    )
    label_rows = _read_parquet_projection(
        acquisition_labels_path,
        columns=_ACQUISITION_LABEL_LINEAGE_COLUMNS | {"acquisition_arm"},
        where="acquisition labels",
    )
    if (
        len(source_rows) != ACQUISITION_LABEL_ROWS
        or len(label_rows) != ACQUISITION_LABEL_ROWS
        or source_descriptor.get("row_count") != ACQUISITION_LABEL_ROWS
        or labels_descriptor.get("row_count") != ACQUISITION_LABEL_ROWS
    ):
        raise InferenceCandidateContractError(
            "combined candidate inputs must conserve exactly 2,000 acquisition rows"
        )

    base_identity = _read_parquet_projection(
        base_training_path,
        columns={"item_id", "thread_id"},
        where="base training frame",
    )
    development_identity = _read_parquet_projection(
        development_path,
        columns={"item_id", "thread_id"},
        where="development frame",
    )
    base_ids = {row["item_id"] for row in base_identity}
    base_threads = {row["thread_id"] for row in base_identity}
    development_ids = {row["item_id"] for row in development_identity}
    development_threads = {row["thread_id"] for row in development_identity}
    if (
        len(base_ids) != len(base_identity)
        or len(base_threads) != len(base_identity)
        or len(development_ids) != len(development_identity)
        or len(development_threads) != len(development_identity)
        or base_ids & development_ids
        or base_threads & development_threads
    ):
        raise InferenceCandidateContractError(
            "base training and development identity/thread sets are invalid or overlap"
        )
    base_thread_digest = thread_set_sha256(sorted(base_threads))
    development_thread_digest = thread_set_sha256(sorted(development_threads))
    if (
        base_descriptor.get("thread_set_sha256") != base_thread_digest
        or development_descriptor.get("thread_set_sha256") != development_thread_digest
    ):
        raise InferenceCandidateContractError(
            "base training or development thread-set descriptor drifted"
        )
    if len(base_rows) != base_descriptor.get("row_count") or len(
        development_rows
    ) != development_descriptor.get("row_count"):
        raise InferenceCandidateContractError("base or development row count drifted")

    source_by_id: dict[str, dict[str, Any]] = {}
    source_threads: set[str] = set()
    for index, row in enumerate(source_rows):
        sample_id = row.get("source_sample_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or row.get("opaque_id") != sample_id
            or sample_id in source_by_id
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in source_threads
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
            or any(
                row.get(field) is not None and not isinstance(row[field], str)
                for field in ("parent_context", "submission_context")
            )
        ):
            raise InferenceCandidateContractError(
                f"acquisition source row {index} has invalid or duplicate lineage"
            )
        source_by_id[sample_id] = row
        source_threads.add(thread_id)

    labels_by_id: dict[str, dict[str, Any]] = {}
    lineage_rows: list[dict[str, Any]] = []
    for index, row in enumerate(label_rows):
        sample_id = row.get("source_sample_id")
        thread_id = row.get("thread_id")
        label_json = row.get("label_json")
        arm = row.get("acquisition_arm")
        eligible = row.get("primary_training_eligible")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in labels_by_id
            or not isinstance(thread_id, str)
            or not thread_id
            or not isinstance(label_json, str)
            or arm not in {"active", "probability_random"}
            or type(eligible) is not bool
        ):
            raise InferenceCandidateContractError(
                f"acquisition label row {index} has invalid or duplicate lineage"
            )
        try:
            clean_label = validate_v2_label(json.loads(label_json))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("acquisition label_json is invalid") from exc
        label_sha256 = canonical_sha256(clean_label)
        labels_by_id[sample_id] = {**row, "label_sha256": label_sha256}
        lineage_rows.append(
            {
                "source_sample_id": sample_id,
                "thread_id": thread_id,
                "label_sha256": label_sha256,
                "primary_training_eligible": eligible,
            }
        )
    if set(source_by_id) != set(labels_by_id):
        raise InferenceCandidateContractError("acquisition source and label IDs differ")
    if (
        set(source_by_id) & base_ids
        or set(source_by_id) & development_ids
        or source_threads & base_threads
        or source_threads & development_threads
    ):
        raise InferenceCandidateContractError(
            "acquisition identity/thread set overlaps base training or development"
        )

    combined_rows = []
    for row in base_table.select(list(_BASE_FRAME_COLUMNS)).to_pylist():
        combined_rows.append(
            {
                **row,
                "acquisition_source_sample_id": None,
                "acquisition_label_sha256": None,
                "primary_training_eligible": None,
            }
        )
    for sample_id in sorted(source_by_id):
        source = source_by_id[sample_id]
        label = labels_by_id[sample_id]
        if source["thread_id"] != label["thread_id"]:
            raise InferenceCandidateContractError(
                "acquisition source and label thread bindings disagree"
            )
        combined_rows.append(
            {
                "item_id": sample_id,
                "frame": "training",
                "thread_id": source["thread_id"],
                "target_text": source["target_text"],
                "parent_context": source["parent_context"],
                "submission_context": source["submission_context"],
                "label_json": label["label_json"],
                "selection_component": f"acquisition_{label['acquisition_arm']}",
                "selection_stratum": None,
                "inclusion_probability_numerator": None,
                "inclusion_probability_denominator": None,
                "inclusion_probability": None,
                "probability_scope": None,
                "acquisition_source_sample_id": sample_id,
                "acquisition_label_sha256": label["label_sha256"],
                "primary_training_eligible": label["primary_training_eligible"],
            }
        )
    schema = pa.schema(
        [
            *(base_table.schema.field(name) for name in _BASE_FRAME_COLUMNS),
            pa.field("acquisition_source_sample_id", pa.string()),
            pa.field("acquisition_label_sha256", pa.string()),
            pa.field("primary_training_eligible", pa.bool_()),
        ],
        metadata={
            b"kind": COMBINED_FRAME_PREPARATION_KIND.encode(),
            b"acquisition_run_id": acquisition_run_id.encode(),
        },
    )
    table = pa.Table.from_pylist(combined_rows, schema=schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".incomplete")
    if output_path.exists():
        if not pq.read_table(output_path).equals(table):
            raise InferenceCandidateContractError(
                "existing immutable combined training frame differs"
            )
    else:
        if temporary.exists():
            raise InferenceCandidateContractError("stale incomplete combined training frame exists")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output_path)

    combined_threads = base_threads | source_threads
    output_descriptor = _local_descriptor(
        output_path,
        descriptor_root=descriptor_root,
        row_count=len(combined_rows),
        frame="training",
        thread_digest=thread_set_sha256(sorted(combined_threads)),
    )
    local_bindings = {
        "training_frame": output_descriptor,
        "development_frame": _local_descriptor(
            development_path,
            descriptor_root=descriptor_root,
            row_count=len(development_rows),
            frame="development",
            thread_digest=development_thread_digest,
        ),
        "acquisition_labels": _local_descriptor(
            acquisition_labels_path,
            descriptor_root=descriptor_root,
            row_count=len(label_rows),
        ),
        "acquisition_receipt": _local_descriptor(
            acquisition_receipt_path,
            descriptor_root=descriptor_root,
        ),
        "acquisition_run_id": acquisition_run_id,
    }
    _, _, _, validation = load_provenance_bound_training_data(
        volume_root=descriptor_root,
        bindings=local_bindings,
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": COMBINED_FRAME_PREPARATION_KIND,
        "base_training_frame": base_descriptor,
        "development_frame": development_descriptor,
        "acquisition_source": source_descriptor,
        "acquisition_labels": labels_descriptor,
        "acquisition_receipt": receipt_descriptor,
        "acquisition_run_id": acquisition_run_id,
        "combined_training_frame": output_descriptor,
        "validation": validation,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    return {**body, "preparation_id": canonical_sha256(body)}


def load_provenance_bound_training_data(
    *,
    volume_root: Path,
    bindings: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Validate one-to-one acquisition lineage and return authorised training rows.

    The combined frame must represent every one of the 2,000 acquired labels
    exactly once.  Base rows carry null lineage fields.  Acquired rows carry an
    exact source ID, thread, label JSON, canonical label digest and eligibility
    flag.  Only the exact primary-eligible acquired subset is returned for
    optimisation, matching the frozen acquisition-training semantics.
    """

    training_path, training_descriptor = _bound_artifact_path(
        volume_root=volume_root,
        descriptor=bindings["training_frame"],
        where="training_frame",
    )
    development_path, development_descriptor = _bound_artifact_path(
        volume_root=volume_root,
        descriptor=bindings["development_frame"],
        where="development_frame",
    )
    labels_path, labels_descriptor = _bound_artifact_path(
        volume_root=volume_root,
        descriptor=bindings["acquisition_labels"],
        where="acquisition_labels",
    )
    receipt_path, _ = _bound_artifact_path(
        volume_root=volume_root,
        descriptor=bindings["acquisition_receipt"],
        where="acquisition_receipt",
    )
    _validate_acquisition_receipt(
        receipt_path,
        labels_descriptor=labels_descriptor,
        acquisition_run_id=bindings["acquisition_run_id"],
    )
    training_rows, _ = load_private_frame(
        training_path,
        training_descriptor,
        expected_frame="training",
    )
    development_rows, development_reference = load_private_frame(
        development_path,
        development_descriptor,
        expected_frame="development",
    )
    raw_labels = _read_parquet_projection(
        labels_path,
        columns=_ACQUISITION_LABEL_LINEAGE_COLUMNS,
        where="acquisition_labels",
    )
    if len(raw_labels) != ACQUISITION_LABEL_ROWS:
        raise InferenceCandidateContractError("acquisition labels must contain exactly 2,000 rows")
    label_by_id: dict[str, dict[str, Any]] = {}
    label_threads: set[str] = set()
    lineage_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_labels):
        sample_id = row.get("source_sample_id")
        thread_id = row.get("thread_id")
        label_json = row.get("label_json")
        eligible = row.get("primary_training_eligible")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in label_by_id
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in label_threads
            or not isinstance(label_json, str)
            or type(eligible) is not bool
        ):
            raise InferenceCandidateContractError(
                f"acquisition label row {index} has invalid or duplicate lineage"
            )
        try:
            clean_label = validate_v2_label(json.loads(label_json))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("acquisition label_json is invalid") from exc
        label_sha256 = canonical_sha256(clean_label)
        label_by_id[sample_id] = {
            "thread_id": thread_id,
            "label_json": label_json,
            "label_sha256": label_sha256,
            "primary_training_eligible": eligible,
        }
        label_threads.add(thread_id)
        lineage_rows.append(
            {
                "source_sample_id": sample_id,
                "thread_id": thread_id,
                "label_sha256": label_sha256,
                "primary_training_eligible": eligible,
            }
        )
    combined_rows = _read_parquet_projection(
        training_path,
        columns=_COMBINED_TRAINING_LINEAGE_COLUMNS,
        where="combined training frame",
    )
    if len(combined_rows) != len(training_rows):
        raise InferenceCandidateContractError(
            "combined training lineage projection does not conserve rows"
        )
    if len(training_rows) <= ACQUISITION_LABEL_ROWS:
        raise InferenceCandidateContractError(
            "combined training frame must retain a non-empty base training frame"
        )
    training_threads: set[str] = set()
    acquired_counts: dict[str, int] = {}
    marker_by_id: dict[str, str | None] = {}
    for index, row in enumerate(combined_rows):
        item_id = row.get("item_id")
        thread_id = row.get("thread_id")
        marker = row.get("acquisition_source_sample_id")
        label_sha256 = row.get("acquisition_label_sha256")
        eligible = row.get("primary_training_eligible")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in marker_by_id
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in training_threads
        ):
            raise InferenceCandidateContractError(
                f"combined training row {index} has invalid or duplicate identity"
            )
        training_threads.add(thread_id)
        marker_by_id[item_id] = marker
        if marker is None:
            if label_sha256 is not None or eligible is not None or item_id in label_by_id:
                raise InferenceCandidateContractError(
                    "base training row has acquisition identity or shadows an acquired label"
                )
            continue
        if not isinstance(marker, str) or marker != item_id or marker not in label_by_id:
            raise InferenceCandidateContractError(
                "combined acquisition row does not bind item_id to source_sample_id"
            )
        expected = label_by_id[marker]
        if (
            thread_id != expected["thread_id"]
            or row.get("label_json") != expected["label_json"]
            or label_sha256 != expected["label_sha256"]
            or type(eligible) is not bool
            or eligible is not expected["primary_training_eligible"]
        ):
            raise InferenceCandidateContractError(
                "combined acquisition row differs from the exact bound label"
            )
        acquired_counts[marker] = acquired_counts.get(marker, 0) + 1
    if set(acquired_counts) != set(label_by_id) or any(
        count != 1 for count in acquired_counts.values()
    ):
        raise InferenceCandidateContractError(
            "combined training frame must represent all 2,000 acquired labels exactly once"
        )
    development_projection = _read_parquet_projection(
        development_path,
        columns={"item_id", "thread_id"},
        where="development frame",
    )
    development_threads: set[str] = set()
    development_ids: set[str] = set()
    for index, row in enumerate(development_projection):
        item_id = row.get("item_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in development_ids
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in development_threads
        ):
            raise InferenceCandidateContractError(
                f"development row {index} has invalid or duplicate identity"
            )
        development_ids.add(item_id)
        development_threads.add(thread_id)
    training_thread_digest = thread_set_sha256(sorted(training_threads))
    development_thread_digest = thread_set_sha256(sorted(development_threads))
    if (
        training_descriptor.get("thread_set_sha256") != training_thread_digest
        or development_descriptor.get("thread_set_sha256") != development_thread_digest
    ):
        raise InferenceCandidateContractError(
            "training or development thread-set descriptor drifted"
        )
    if training_threads & development_threads:
        raise InferenceCandidateContractError(
            "combined training and development thread sets overlap"
        )
    eligible_acquisition_ids = {
        sample_id for sample_id, row in label_by_id.items() if row["primary_training_eligible"]
    }
    authorised_training_rows = [
        row
        for row in training_rows
        if marker_by_id[row["item_id"]] is None or row["item_id"] in eligible_acquisition_ids
    ]
    if not authorised_training_rows:
        raise InferenceCandidateContractError("authorised combined training frame is empty")
    summary = {
        "combined_training_rows": len(training_rows),
        "base_training_rows": len(training_rows) - ACQUISITION_LABEL_ROWS,
        "acquisition_label_rows": len(raw_labels),
        "acquisition_rows_represented_exactly_once": len(acquired_counts),
        "primary_training_eligible_acquisition_rows": len(eligible_acquisition_ids),
        "authorised_training_rows": len(authorised_training_rows),
        "acquisition_lineage_sha256": canonical_sha256(
            sorted(lineage_rows, key=lambda row: row["source_sample_id"])
        ),
        "training_thread_set_sha256": training_thread_digest,
        "development_thread_set_sha256": development_thread_digest,
        "training_development_thread_overlap": 0,
    }
    return (
        authorised_training_rows,
        development_rows,
        development_reference,
        summary,
    )


def create_candidate_model() -> Any:
    """Instantiate the real pinned one-pass B4 model; no loader override exists."""

    return create_component_model(
        component=COMPONENT,
        config=candidate_optimisation_config(),
    )


def tokenise_inference_record(
    tokenizer: Any,
    *,
    item_id: str,
    row: Mapping[str, Any],
    max_length: int = 768,
) -> dict[str, Any]:
    """Tokenise an unlabelled bounded row under the frozen B4 text contract."""

    if max_length != 768:
        raise ValueError("candidate max_length must remain frozen at 768")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("item_id must be non-empty")
    text = render_factorised_text(
        row,
        separator=tokenizer.sep_token,
        config=TextRenderConfig(),
    )
    raw = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    input_ids = list(raw["input_ids"])
    attention_mask = list(raw.get("attention_mask", [1] * len(input_ids)))
    if not input_ids or len(input_ids) != len(attention_mask):
        raise RuntimeError("tokenizer returned empty or mismatched inputs")
    if len(input_ids) > max_length:
        input_ids = [*input_ids[: max_length - 1], input_ids[-1]]
        attention_mask = [*attention_mask[: max_length - 1], attention_mask[-1]]
    return {
        "item_id": item_id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class CandidateInferenceCollator:
    """Dynamic padding for label-free development and throughput inference."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty inference batch")
        ids = [feature.get("item_id") for feature in features]
        if any(not isinstance(item_id, str) or not item_id for item_id in ids):
            raise ValueError("every inference row requires an item_id")
        batch = dict(
            self.tokenizer.pad(
                [
                    {
                        "input_ids": feature["input_ids"],
                        "attention_mask": feature["attention_mask"],
                    }
                    for feature in features
                ],
                padding=True,
                return_tensors="pt",
            )
        )
        batch["item_ids"] = ids
        return batch


def _require_torch_runtime() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Modal-only dependency path
        raise RuntimeError("candidate execution requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("candidate execution requires the registered L4 CUDA device")
    return torch


def _validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    expected = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase_run_id",
        "candidate_config_sha256",
        "component",
        "optimiser_seed",
        "model_id",
        "model_revision",
        "stance_logits_per_target",
        "selected_epoch",
        "model_state_dict",
    }
    clean = validate_run_manifest(manifest)
    if set(checkpoint) != expected or (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("kind") != CHECKPOINT_KIND
        or checkpoint.get("experiment_run_id") != clean["experiment_run_id"]
        or checkpoint.get("phase_run_id") != clean["phase_run_id"]
        or checkpoint.get("candidate_config_sha256")
        != clean["experiment_contract"]["registered_design"]["candidate_config_sha256"]
        or checkpoint.get("component") != COMPONENT
        or checkpoint.get("optimiser_seed") != REGISTERED_SEED
        or checkpoint.get("model_id") != MODEL_ID
        or checkpoint.get("model_revision") != MODEL_REVISION
        or checkpoint.get("stance_logits_per_target") != 4
        or not isinstance(checkpoint.get("model_state_dict"), Mapping)
    ):
        raise InferenceCandidateContractError("candidate checkpoint metadata drifted")
    _positive_int(checkpoint.get("selected_epoch"), where="selected_epoch")


def _validate_training_provenance(
    value: Any,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("candidate training provenance must be an object")
    expected = {
        "combined_training_rows",
        "base_training_rows",
        "acquisition_label_rows",
        "acquisition_rows_represented_exactly_once",
        "primary_training_eligible_acquisition_rows",
        "authorised_training_rows",
        "acquisition_lineage_sha256",
        "training_thread_set_sha256",
        "development_thread_set_sha256",
        "training_development_thread_overlap",
    }
    clean = validate_run_manifest(manifest)
    bindings = clean["experiment_contract"]["bindings"]
    if set(value) != expected:
        raise ValueError("candidate training provenance schema drifted")
    combined = _positive_int(value.get("combined_training_rows"), where="combined rows")
    base = _positive_int(value.get("base_training_rows"), where="base rows")
    labels = _positive_int(value.get("acquisition_label_rows"), where="label rows")
    represented = _positive_int(
        value.get("acquisition_rows_represented_exactly_once"),
        where="represented acquisition rows",
    )
    eligible = _positive_int(
        value.get("primary_training_eligible_acquisition_rows"),
        where="eligible acquisition rows",
    )
    authorised = _positive_int(
        value.get("authorised_training_rows"), where="authorised training rows"
    )
    if (
        combined != bindings["training_frame"]["row_count"]
        or labels != ACQUISITION_LABEL_ROWS
        or represented != ACQUISITION_LABEL_ROWS
        or base != combined - ACQUISITION_LABEL_ROWS
        or authorised != base + eligible
        or eligible > ACQUISITION_LABEL_ROWS
        or value.get("training_thread_set_sha256")
        != bindings["training_frame"]["thread_set_sha256"]
        or value.get("development_thread_set_sha256")
        != bindings["development_frame"]["thread_set_sha256"]
        or value.get("training_development_thread_overlap") != 0
    ):
        raise InferenceCandidateContractError("candidate training provenance drifted")
    _sha(value.get("acquisition_lineage_sha256"), where="acquisition lineage")
    return _clone(value)


def validate_cuda_preflight_receipt(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase_run_id",
        "model_id",
        "model_revision",
        "gpu_type",
        "seed_count",
        "stance_logits_per_target",
        "mixed_stance_ce_mask_checked",
        "training_provenance",
        "locked_test_rows_accessed",
        "corpus_rows_accessed",
        "receipt_id",
    }
    clean = validate_run_manifest(manifest)
    if set(value) != expected or (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != "modernbert-inference-candidate-cuda-preflight-v1"
        or value.get("experiment_run_id") != clean["experiment_run_id"]
        or value.get("phase_run_id") != clean["phase_run_id"]
        or value.get("model_id") != MODEL_ID
        or value.get("model_revision") != MODEL_REVISION
        or value.get("gpu_type") != GPU_TYPE
        or value.get("seed_count") != 1
        or value.get("stance_logits_per_target") != len(STANCE_CLASSES_B4)
        or value.get("mixed_stance_ce_mask_checked") is not True
        or value.get("locked_test_rows_accessed") != 0
        or value.get("corpus_rows_accessed") != 0
    ):
        raise InferenceCandidateContractError("candidate CUDA preflight receipt drifted")
    _validate_training_provenance(value["training_provenance"], manifest=clean)
    body = {key: value[key] for key in expected - {"receipt_id"}}
    if value.get("receipt_id") != canonical_sha256(body):
        raise InferenceCandidateContractError("candidate CUDA preflight digest drifted")
    return _clone(value)


def validate_training_receipt(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase_run_id",
        "optimiser_seed",
        "selected_epoch",
        "development_three_class_stance_accuracy",
        "history",
        "training_provenance",
        "checkpoint",
        "development_calibration",
        "wall_seconds",
        "gpu_seconds",
        "attempt_count",
        "locked_test_rows_accessed",
        "corpus_rows_accessed",
        "receipt_id",
    }
    clean = validate_run_manifest(manifest)
    if set(value) != expected or (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != TRAINING_RECEIPT_KIND
        or value.get("experiment_run_id") != clean["experiment_run_id"]
        or value.get("phase_run_id") != clean["phase_run_id"]
        or value.get("optimiser_seed") != REGISTERED_SEED
        or value.get("attempt_count") != 1
        or value.get("locked_test_rows_accessed") != 0
        or value.get("corpus_rows_accessed") != 0
    ):
        raise InferenceCandidateContractError("candidate training receipt drifted")
    selected_epoch = _positive_int(value.get("selected_epoch"), where="selected_epoch")
    history = value.get("history")
    expected_epochs = frozen_component_config(COMPONENT)["max_epochs"]
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise InferenceCandidateContractError("candidate training history drifted")
    clean_history = []
    for index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or set(row) != {
            "epoch",
            "train_mean_loss",
            "development_three_class_stance_accuracy",
        }:
            raise ValueError("candidate training history schema drifted")
        loss = _finite(row["train_mean_loss"], where=f"history[{index}].loss")
        score = _finite(
            row["development_three_class_stance_accuracy"],
            where=f"history[{index}].score",
        )
        if row.get("epoch") != index or loss < 0 or not 0 <= score <= 1:
            raise InferenceCandidateContractError("candidate training history values drifted")
        clean_history.append((index, score))
    best_epoch, best_score = max(clean_history, key=lambda item: item[1])
    observed_score = _finite(
        value.get("development_three_class_stance_accuracy"),
        where="development score",
    )
    if selected_epoch != best_epoch or observed_score != best_score:
        raise InferenceCandidateContractError("candidate selected checkpoint score drifted")
    _validate_training_provenance(value["training_provenance"], manifest=clean)
    artifact_descriptor(value["checkpoint"], where="candidate checkpoint")
    artifact_descriptor(value["development_calibration"], where="candidate calibration")
    wall_seconds = _finite(value.get("wall_seconds"), where="training wall seconds")
    gpu_seconds = _finite(value.get("gpu_seconds"), where="training GPU seconds")
    if wall_seconds <= 0 or gpu_seconds != wall_seconds or gpu_seconds > TRAIN_MAX_GPU_SECONDS:
        raise InferenceCandidateContractError("candidate training duration drifted")
    body = {key: value[key] for key in expected - {"receipt_id"}}
    if value.get("receipt_id") != canonical_sha256(body):
        raise InferenceCandidateContractError("candidate training receipt digest drifted")
    return _clone(value)


def _calibration_rows(
    predictions: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for prediction in predictions:
        item_id = prediction["item_id"]
        if item_id not in reference:
            raise InferenceCandidateContractError("development prediction ID is unbound")
        encoding = encode_v2_label(reference[item_id])
        rows.append(
            {
                "item_id": item_id,
                "target_presence_logits": prediction["target_presence_logits"],
                "target_presence_labels": list(encoding.target_presence_labels),
                "reference_material": bool(encoding.target_presence_mask[0]),
                "stance_logits": prediction["stance_logits"],
                "stance_labels": list(encoding.stance_b4_labels),
                "stance_mask": [bool(value) for value in encoding.stance_mask],
            }
        )
    if len(rows) != len(reference):
        raise InferenceCandidateContractError("development predictions do not conserve rows")
    return rows


def _checkpoint_score(rows: Sequence[Mapping[str, Any]]) -> float:
    correct = total = 0
    for row in rows:
        for logits, label, mask in zip(
            row["stance_logits"], row["stance_labels"], row["stance_mask"], strict=True
        ):
            if not mask or label == MIXED_LOGIT_INDEX:
                continue
            prediction = decode_three_class_stance_logits(logits)["stance"]
            correct += int(prediction == STANCE_CLASSES_B4[label])
            total += 1
    if not total:
        raise InferenceCandidateContractError(
            "development lacks non-mixed stance support for checkpoint selection"
        )
    return correct / total


def execute_registered_training(
    manifest: Mapping[str, Any],
    *,
    volume_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Train the sole seed once and calibrate the selected checkpoint on development."""

    clean = validate_run_manifest(manifest)
    if output_root.exists():
        raise InferenceCandidateContractError(
            "candidate output already exists; retries are not authorised"
        )
    bindings = clean["experiment_contract"]["bindings"]
    (
        training_rows,
        development_rows,
        development_reference,
        training_provenance,
    ) = load_provenance_bound_training_data(
        volume_root=volume_root,
        bindings=bindings,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    torch = _require_torch_runtime()
    try:
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - Modal-only dependency path
        raise RuntimeError("candidate runtime dependencies are incomplete") from exc
    from reddit_china_stance.modernbert_trainer import (
        build_length_bucket_batches,
        seed_everything,
    )

    seed_everything(REGISTERED_SEED)
    tokenizer = load_pinned_tokenizer()
    training_features = []
    for row in training_rows:
        feature = tokenise_factorised_record(
            tokenizer,
            item_id=row["item_id"],
            row=row,
            label=json.loads(row["label_json"]),
        )
        if feature["target_presence_mask"][0]:
            training_features.append(feature)
    if not training_features:
        raise InferenceCandidateContractError("candidate training frame has no material rows")
    development_features = [
        tokenise_factorised_record(
            tokenizer,
            item_id=row["item_id"],
            row=row,
            label=json.loads(row["label_json"]),
        )
        for row in development_rows
    ]
    config = candidate_optimisation_config()
    collator = InferenceCandidateB4Collator(tokenizer)
    development_loader = DataLoader(
        development_features,
        batch_sampler=build_length_bucket_batches(
            [len(feature["input_ids"]) for feature in development_features],
            batch_size=config.per_device_batch_size,
            seed=0,
            epoch=0,
        ),
        collate_fn=collator,
        num_workers=0,
    )
    model = create_candidate_model().to("cuda")
    optimizer = create_adamw(model, config)
    max_epochs = frozen_component_config(COMPONENT)["max_epochs"]
    batches_per_epoch = math.ceil(len(training_features) / config.per_device_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    training_lengths = [len(feature["input_ids"]) for feature in training_features]
    checkpoint_path = output_root / "checkpoint.pt"
    best_score = -1.0
    best_epoch = 0
    history = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        training_loader = DataLoader(
            training_features,
            batch_sampler=build_length_bucket_batches(
                training_lengths,
                batch_size=config.per_device_batch_size,
                seed=REGISTERED_SEED,
                epoch=epoch,
            ),
            collate_fn=collator,
            num_workers=0,
        )
        trained = train_epoch(
            model,
            training_loader,
            optimizer,
            config=config,
            device="cuda",
            scheduler=scheduler,
        )
        predictions = collect_development_logits(
            model,
            development_loader,
            component=COMPONENT,
            device="cuda",
            use_bf16=True,
        )
        calibration_rows = _calibration_rows(predictions, development_reference)
        score = _checkpoint_score(calibration_rows)
        history.append(
            {
                "epoch": epoch,
                "train_mean_loss": float(trained["mean_loss"]),
                "development_three_class_stance_accuracy": score,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            payload = {
                "schema_version": SCHEMA_VERSION,
                "kind": CHECKPOINT_KIND,
                "experiment_run_id": clean["experiment_run_id"],
                "phase_run_id": clean["phase_run_id"],
                "candidate_config_sha256": clean["experiment_contract"]["registered_design"][
                    "candidate_config_sha256"
                ],
                "component": COMPONENT,
                "optimiser_seed": REGISTERED_SEED,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "stance_logits_per_target": len(STANCE_CLASSES_B4),
                "selected_epoch": epoch,
                "model_state_dict": model.state_dict(),
            }
            temporary = checkpoint_path.with_suffix(".pt.new")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    _validate_checkpoint_metadata(checkpoint, clean)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_predictions = collect_development_logits(
        model,
        development_loader,
        component=COMPONENT,
        device="cuda",
        use_bf16=True,
    )
    calibration = fit_development_calibration(
        bindings["development_frame"],
        _calibration_rows(final_predictions, development_reference),
    )
    validate_development_calibration(calibration, manifest=clean)
    calibration_path = output_root / "development-calibration.json"
    _write_immutable_json(calibration_path, calibration)
    elapsed = time.monotonic() - started
    if elapsed > TRAIN_MAX_GPU_SECONDS:
        raise InferenceCandidateContractError("candidate exceeded registered GPU seconds")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRAINING_RECEIPT_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "optimiser_seed": REGISTERED_SEED,
        "selected_epoch": best_epoch,
        "development_three_class_stance_accuracy": best_score,
        "history": history,
        "training_provenance": training_provenance,
        "checkpoint": {
            "relative_path": checkpoint_path.relative_to(volume_root).as_posix(),
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "development_calibration": {
            "relative_path": calibration_path.relative_to(volume_root).as_posix(),
            "sha256": file_sha256(calibration_path),
            "bytes": calibration_path.stat().st_size,
        },
        "wall_seconds": elapsed,
        "gpu_seconds": elapsed,
        "attempt_count": 1,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": canonical_sha256(body)}
    _write_immutable_json(output_root / "receipt.json", receipt)
    return receipt


def measure_real_pinned_throughput(
    manifest: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    expected_selected_epoch: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure the real pinned model/tokenizer path on exactly 128 bounded rows."""

    validate_inference_scope("throughput_smoke")
    if len(rows) != THROUGHPUT_SMOKE_ROWS:
        raise ValueError(f"throughput smoke requires exactly {THROUGHPUT_SMOKE_ROWS} rows")
    torch = _require_torch_runtime()
    clean = validate_run_manifest(manifest)
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - Modal-only dependency path
        raise RuntimeError("candidate throughput requires PyTorch DataLoader") from exc
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    _validate_checkpoint_metadata(checkpoint, clean)
    if checkpoint["selected_epoch"] != _positive_int(
        expected_selected_epoch, where="expected_selected_epoch"
    ):
        raise InferenceCandidateContractError(
            "candidate checkpoint and training receipt selected epochs differ"
        )
    tokenizer = load_pinned_tokenizer()
    features = [
        tokenise_inference_record(
            tokenizer,
            item_id=row["item_id"],
            row=row,
        )
        for row in rows
    ]
    loader = DataLoader(
        features,
        batch_size=THROUGHPUT_BATCH_SIZE,
        shuffle=False,
        collate_fn=CandidateInferenceCollator(tokenizer),
        num_workers=0,
    )
    model = create_candidate_model().to("cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    measured_rows = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs = {
                key: value.to("cuda")
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask"}
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**inputs)
            presence = output["target_presence_logits"]
            stances = output["stance_logits"]
            if tuple(presence.shape[1:]) != (len(TARGET_CLASSES),) or tuple(stances.shape[1:]) != (
                len(ANALYTIC_TARGET_CLASSES),
                len(STANCE_CLASSES_B4),
            ):
                raise InferenceCandidateContractError("real model output shape drifted")
            measured_rows += int(presence.shape[0])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if measured_rows != THROUGHPUT_SMOKE_ROWS or elapsed <= 0:
        raise InferenceCandidateContractError("throughput smoke did not conserve rows")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": THROUGHPUT_RECEIPT_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "component": COMPONENT,
        "gpu_type": GPU_TYPE,
        "rows": measured_rows,
        "batch_size": THROUGHPUT_BATCH_SIZE,
        "wall_seconds": elapsed,
        "rows_per_second": measured_rows / elapsed,
        "real_pinned_model_path_measured": True,
        "stance_logits_per_target": len(STANCE_CLASSES_B4),
        "decode_logit_indices": list(DECODE_LOGIT_INDICES),
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    return {**body, "receipt_id": canonical_sha256(body)}


__all__ = [
    "ACQUISITION_LABEL_ROWS",
    "CALIBRATION_KIND",
    "COMPONENT",
    "CUDA_PREFLIGHT_MAX_GPU_SECONDS",
    "DECODE_LOGIT_INDICES",
    "DECODE_STANCES",
    "DEVELOPMENT_ROWS",
    "EXPECTED_CANDIDATES",
    "GPU_TYPE",
    "MAX_CONCURRENT_CANDIDATES",
    "MAX_TRAINING_ATTEMPTS",
    "MIXED_LOGIT_INDEX",
    "MODEL_ID",
    "MODEL_REVISION",
    "NAMESPACE",
    "REGISTERED_SEED",
    "REGISTERED_SEEDS",
    "THROUGHPUT_BATCH_SIZE",
    "THROUGHPUT_SMOKE_ROWS",
    "TOKENIZER_REVISION",
    "CandidateInferenceCollator",
    "InferenceCandidateB4Collator",
    "InferenceCandidateContractError",
    "build_run_manifest",
    "candidate_optimisation_config",
    "create_candidate_model",
    "decode_candidate_logits",
    "decode_three_class_stance_logits",
    "execute_registered_training",
    "fit_development_calibration",
    "freeze_experiment_contract",
    "load_provenance_bound_training_data",
    "mask_mixed_stance_feature",
    "materialise_combined_training_frame",
    "measure_real_pinned_throughput",
    "registered_candidate_config",
    "thread_set_sha256",
    "tokenise_inference_record",
    "validate_cuda_preflight_receipt",
    "validate_development_calibration",
    "validate_experiment_contract",
    "validate_inference_scope",
    "validate_run_manifest",
    "validate_training_receipt",
]
