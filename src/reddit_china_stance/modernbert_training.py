"""Deterministic contracts for the ModernBERT student experiment.

This module intentionally has no Torch, Transformers, or Modal imports.  It
defines the immutable metadata boundary shared by the split builder, trainer,
and cloud launcher; those components remain responsible for doing the work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any

from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_KIND = "modernbert-three-head-experiment-v1"
TRIAL_SPEC_KIND = "modernbert-three-head-trial-spec-v1"
CHECKPOINT_KIND = "modernbert-three-head-checkpoint-metadata-v1"
TRIAL_RECEIPT_KIND = "modernbert-three-head-trial-receipt-v1"
LOCKED_TEST_AUTHORISATION_KIND = "modernbert-locked-test-authorisation-v1"

MODEL_ID = "answerdotai/ModernBERT-large"
MODEL_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
TOKENIZER_REVISION = MODEL_REVISION
DATASET_ID = "aisafteycommons/reddit-china-stance-10k-sol-v1"
DATASET_REVISION = "b0fa1c3e5e3caf4dc3fb43b9024858364bc44834"
DATASET_SHA256 = "f81146988ef504f81fb77b36333084e5ea10ab77908568d5be26bdba296a5bba"
DEVELOPMENT_PROXY_ID = "4ba40fcd0eb33dc2c7ca1c0b9ac75b06f42dc120bed470f9da7f713194a3a66d"
TEACHER_GENERATION_RUN_ID = "bc7854a60140b3584d5cdeb5513b07ff413253fd59cb84877c278de21702afa4"
PROXY_JSON_SHA256 = "6a8f379cd60e1bcf646160b185edad44847d361e59c631581318668a9d3b1a5d"
REFERENCE_ROWS_JSON_SHA256 = "32a5fe0a5a75f5dc48f21a8a60510e9951b8cfe01efb0d4e100b6f1c8c518fcc"
HARD_COST_CAP_USD = Decimal("200")

VOLUME_INPUT_PREFIX = Path("student-modernbert-v1/inputs")
TEACHER_PARQUET_NAME = "teacher-labels.parquet"
SPLIT_MANIFEST_NAME = "split-manifest.json"
DEVELOPMENT_PROXY_PARQUET_NAME = "development-proxy.parquet"
OUTPUT_PREFIX = Path("student-modernbert-v1")

GPU_RATE_USD_PER_SECOND = {
    "L4": Decimal("0.000222"),
}

REGISTERED_LADDERS = (
    {"ladder_seed": 101, "optimiser_seed": 47},
    {"ladder_seed": 202, "optimiser_seed": 61},
    {"ladder_seed": 303, "optimiser_seed": 89},
)
STABILITY_SEEDS = (13, 29)
LABEL_BUDGETS = (2_000, 5_000, 10_000)
ASHA_RUNG_PLAN = (
    {"epochs": 1, "candidate_count": 12, "promotion_count": 4},
    {"epochs": 3, "candidate_count": 4, "promotion_count": 2},
    {"epochs": 6, "candidate_count": 2, "promotion_count": 0},
)
TARGET_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)

_HASH_FIELDS = frozenset(
    {
        "dataset_sha256",
        "split_manifest_sha256",
        "code_sha256",
        "dependency_lock_sha256",
        "config_sha256",
        "model_state_sha256",
        "optimiser_state_sha256",
        "rng_state_sha256",
        "scheduler_state_sha256",
        "trial_manifest_sha256",
    }
)


class ModernBertContractError(RuntimeError):
    """Raised when a frozen student-experiment contract drifts."""


def canonical_sha256(value: Any) -> str:
    """Return a digest over finite canonical JSON."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, where: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return str(value)


def _require_positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _require_nonnegative_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a non-negative finite number")
    clean = float(value)
    if not math.isfinite(clean) or clean < 0:
        raise ValueError(f"{where} must be a non-negative finite number")
    return clean


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001")))


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised in cloud/dev dependency group
        raise RuntimeError("PyArrow is required for ModernBERT private Parquet inputs") from exc
    return pa, pq


def _read_json_object(path: Path, *, where: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain a JSON object")
    return value


def validate_development_proxy_parquet(path: Path) -> dict[str, Any]:
    """Validate the private text-plus-label evaluation input without returning rows."""

    _, pq = _require_pyarrow()
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    expected_columns = [
        "source_sample_id",
        "target_text",
        "parent_context",
        "submission_context",
        "split",
        "resolution",
        "label_json",
    ]
    if table.column_names != expected_columns or table.num_rows != 452:
        raise ModernBertContractError("development proxy Parquet schema or row count drifted")
    metadata = table.schema.metadata or {}
    expected_metadata = {
        b"proxy_id": DEVELOPMENT_PROXY_ID.encode(),
        b"proxy_json_sha256": PROXY_JSON_SHA256.encode(),
        b"reference_rows_json_sha256": REFERENCE_ROWS_JSON_SHA256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ModernBertContractError("development proxy Parquet source binding drifted")
    rows = table.to_pylist()
    ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    from reddit_china_stance.modernbert_trainer import encode_semantic_label

    for index, row in enumerate(rows):
        item_id = row.get("source_sample_id")
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise ValueError(f"development proxy row {index} has an invalid or duplicate ID")
        ids.add(item_id)
        if not isinstance(row.get("target_text"), str) or not row["target_text"].strip():
            raise ValueError(f"development proxy row {index} lacks target text")
        for field in ("parent_context", "submission_context"):
            if row.get(field) is not None and not isinstance(row[field], str):
                raise ValueError(f"development proxy row {index}.{field} is invalid")
        split = row.get("split")
        if split not in {"development", "locked_test_candidate"}:
            raise ValueError(f"development proxy row {index} has an unsupported split")
        split_counts[split] += 1
        if not isinstance(row.get("resolution"), str) or not row["resolution"]:
            raise ValueError(f"development proxy row {index} lacks resolution metadata")
        try:
            label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"development proxy row {index} has invalid label JSON") from exc
        encode_semantic_label(label)
    if dict(split_counts) != {"development": 222, "locked_test_candidate": 230}:
        raise ModernBertContractError("development proxy split counts drifted")
    return {
        "relative_path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "row_count": 452,
        "development_rows": 222,
        "locked_test_candidate_rows": 230,
    }


def assemble_development_proxy_parquet(
    *,
    proxy_json_path: Path,
    reference_rows_json_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Join the frozen proxy labels to private text and publish exact Parquet.

    This is a private preparation helper.  It intentionally emits no public
    receipt because both opaque IDs and text are present in the artefact.
    """

    if file_sha256(proxy_json_path) != PROXY_JSON_SHA256:
        raise ModernBertContractError("frozen development proxy JSON hash drifted")
    if file_sha256(reference_rows_json_path) != REFERENCE_ROWS_JSON_SHA256:
        raise ModernBertContractError("frozen reference-rows JSON hash drifted")
    if output_path.exists():
        return validate_development_proxy_parquet(output_path)
    incomplete = output_path.with_suffix(output_path.suffix + ".incomplete")
    if incomplete.exists():
        raise ModernBertContractError("stale incomplete development proxy Parquet exists")

    proxy = _read_json_object(proxy_json_path, where="development proxy")
    reference = _read_json_object(reference_rows_json_path, where="reference rows")
    if proxy.get("proxy_id") != DEVELOPMENT_PROXY_ID:
        raise ModernBertContractError("development proxy ID drifted")
    proxy_binding = proxy.get("input_binding")
    if (
        not isinstance(proxy_binding, Mapping)
        or proxy_binding.get("reference_packet_sha256") != REFERENCE_ROWS_JSON_SHA256
    ):
        raise ModernBertContractError("development proxy reference-packet binding drifted")
    proxy_rows = proxy.get("rows")
    reference_rows = reference.get("rows")
    if not isinstance(proxy_rows, list) or len(proxy_rows) != 452:
        raise ValueError("development proxy must contain exactly 452 rows")
    if not isinstance(reference_rows, list):
        raise ValueError("reference rows are missing")
    reference_by_id: dict[str, Mapping[str, Any]] = {}
    for row in reference_rows:
        if not isinstance(row, Mapping):
            raise ValueError("reference row must be an object")
        item_id = row.get("source_sample_id")
        if not isinstance(item_id, str) or not item_id or item_id in reference_by_id:
            raise ValueError("reference rows contain an invalid or duplicate source ID")
        reference_by_id[item_id] = row

    joined: list[dict[str, Any]] = []
    for index, row in enumerate(proxy_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"development proxy rows[{index}] must be an object")
        item_id = row.get("source_sample_id")
        source = reference_by_id.get(str(item_id))
        if source is None:
            raise ModernBertContractError("development proxy text join is incomplete")
        label = row.get("label")
        if not isinstance(label, Mapping):
            raise ValueError(f"development proxy rows[{index}].label must be an object")
        joined.append(
            {
                "source_sample_id": item_id,
                "target_text": source.get("target_text"),
                "parent_context": source.get("parent_context"),
                "submission_context": source.get("submission_context"),
                "split": row.get("split"),
                "resolution": row.get("resolution"),
                "label_json": json.dumps(
                    label, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    joined.sort(key=lambda row: str(row["source_sample_id"]))
    pa, pq = _require_pyarrow()
    schema = pa.schema(
        [
            pa.field("source_sample_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("parent_context", pa.string()),
            pa.field("submission_context", pa.string()),
            pa.field("split", pa.string(), nullable=False),
            pa.field("resolution", pa.string(), nullable=False),
            pa.field("label_json", pa.string(), nullable=False),
        ],
        metadata={
            b"proxy_id": DEVELOPMENT_PROXY_ID.encode(),
            b"proxy_json_sha256": PROXY_JSON_SHA256.encode(),
            b"reference_rows_json_sha256": REFERENCE_ROWS_JSON_SHA256.encode(),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(pa.Table.from_pylist(joined, schema=schema), incomplete, compression="zstd")
        validate_development_proxy_parquet(incomplete)
        os.replace(incomplete, output_path)
    except BaseException:
        incomplete.unlink(missing_ok=True)
        raise
    return validate_development_proxy_parquet(output_path)


def _registered_design() -> dict[str, Any]:
    return {
        "asha_plan": build_asha_plan(),
        "stability_seeds": list(STABILITY_SEEDS),
        "label_budgets": list(LABEL_BUDGETS),
        "acquisition_ladders": [dict(item) for item in REGISTERED_LADDERS],
        "target_thresholds": list(TARGET_THRESHOLDS),
        "locked_test_rows": 230,
    }


def frozen_trial_configs() -> list[dict[str, Any]]:
    """Return the registered 3 x 2 x 2 ASHA grid in canonical order."""

    configs: list[dict[str, Any]] = []
    for encoder_lr, loss_weights, class_weights in product(
        (1e-5, 3e-5, 5e-5),
        ((1.0, 1.0, 1.0), (0.5, 1.0, 1.5)),
        ("none", "capped_inverse_sqrt"),
    ):
        body = {
            "encoder_learning_rate": encoder_lr,
            "head_learning_rate_multiplier": 5,
            "loss_weights": {
                "relevance": loss_weights[0],
                "targets": loss_weights[1],
                "stance": loss_weights[2],
            },
            "class_weights": class_weights,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "dropout": 0.1,
            "gradient_clip": 1.0,
            "effective_batch_size": 32,
            "max_epochs": 6,
            "max_length": 768,
            "pooling": "masked_mean",
            "target_threshold": 0.50,
        }
        configs.append({**body, "config_sha256": canonical_sha256(body)})
    if len(configs) != 12 or len({item["config_sha256"] for item in configs}) != 12:
        raise AssertionError("registered sweep grid is not twelve unique configurations")
    return configs


def build_asha_plan() -> dict[str, Any]:
    """Build the frozen 12 -> 4 -> 2 recipe-development plan."""

    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-asha-plan-v1",
        "label_budget": 5_000,
        "ladder_seed": 101,
        "optimiser_seed": 7,
        "fixed_selection_threshold": 0.50,
        "rungs": [dict(rung) for rung in ASHA_RUNG_PLAN],
        "configs": frozen_trial_configs(),
    }
    return {**body, "plan_id": canonical_sha256(body)}


def validate_asha_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_asha_plan()
    if dict(value) != expected:
        raise ModernBertContractError("ASHA plan differs from the registered 12 -> 4 -> 2 grid")
    return deepcopy(expected)


def select_asha_promotions(
    plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    rung_epochs: int,
    candidate_config_ids: Sequence[str],
) -> list[str]:
    """Rank a complete ASHA rung with deterministic tie-breaking.

    Invalid outputs and configurations below the registered 0.80 material
    recall guardrail are ineligible.  Missing candidates fail rather than being
    interpreted as poor scores.
    """

    clean_plan = validate_asha_plan(plan)
    rung = next((item for item in clean_plan["rungs"] if item["epochs"] == rung_epochs), None)
    if rung is None or rung["promotion_count"] == 0:
        raise ValueError("rung_epochs must identify a promoting ASHA rung")
    candidates = list(candidate_config_ids)
    if len(candidates) != rung["candidate_count"] or len(set(candidates)) != len(candidates):
        raise ValueError("candidate config IDs do not match the registered rung cardinality")
    registered = {item["config_sha256"] for item in clean_plan["configs"]}
    if not set(candidates) <= registered:
        raise ValueError("candidate config IDs include an unregistered configuration")

    by_id: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if set(result) != {
            "config_sha256",
            "completed_epochs",
            "composite",
            "material_recall",
            "invalid_outputs",
        }:
            raise ValueError("ASHA result has unexpected fields")
        config_id = _require_sha256(result["config_sha256"], where="result.config_sha256")
        if config_id in by_id:
            raise ValueError("ASHA results contain a duplicate configuration")
        if result["completed_epochs"] != rung_epochs:
            raise ValueError("ASHA result was not measured at the requested rung")
        composite = _require_nonnegative_number(result["composite"], where="result.composite")
        material_recall = _require_nonnegative_number(
            result["material_recall"], where="result.material_recall"
        )
        if composite > 1 or material_recall > 1:
            raise ValueError("ASHA aggregate metrics must be in [0, 1]")
        if not isinstance(result["invalid_outputs"], int) or result["invalid_outputs"] < 0:
            raise ValueError("result.invalid_outputs must be a non-negative integer")
        by_id[config_id] = result
    if set(by_id) != set(candidates):
        raise ValueError("ASHA result set does not exactly match the candidate set")

    eligible = [
        result
        for result in by_id.values()
        if result["invalid_outputs"] == 0 and float(result["material_recall"]) >= 0.80
    ]
    if len(eligible) < rung["promotion_count"]:
        raise ModernBertContractError("too few ASHA candidates pass the frozen guardrails")
    ranked = sorted(
        eligible,
        key=lambda result: (-float(result["composite"]), str(result["config_sha256"])),
    )
    return [str(result["config_sha256"]) for result in ranked[: rung["promotion_count"]]]


def select_checkpoint(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    minimum_epoch: int = 2,
    min_delta: float = 0.001,
    patience_epochs: int = 2,
) -> dict[str, Any]:
    """Apply the registered checkpoint and early-stopping rule."""

    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if minimum_epoch < 1 or min_delta < 0 or patience_epochs < 1:
        raise ValueError("checkpoint-selection parameters are invalid")
    clean: list[dict[str, Any]] = []
    for expected_epoch, checkpoint in enumerate(checkpoints, start=1):
        if set(checkpoint) != {"epoch", "composite", "checkpoint_sha256"}:
            raise ValueError("checkpoint selection input has unexpected fields")
        if checkpoint["epoch"] != expected_epoch:
            raise ValueError("checkpoint epochs must be consecutive and start at one")
        score = _require_nonnegative_number(checkpoint["composite"], where="checkpoint.composite")
        if score > 1:
            raise ValueError("checkpoint.composite must be in [0, 1]")
        digest = _require_sha256(
            checkpoint["checkpoint_sha256"], where="checkpoint.checkpoint_sha256"
        )
        clean.append({"epoch": expected_epoch, "composite": score, "checkpoint_sha256": digest})

    eligible = [item for item in clean if item["epoch"] >= minimum_epoch]
    if not eligible:
        raise ValueError("no checkpoint reaches the registered minimum epoch")
    best = eligible[0]
    significant_reference = best["composite"]
    significant_epoch = best["epoch"]
    stopped_epoch: int | None = None
    for item in eligible[1:]:
        if item["composite"] > best["composite"]:
            best = item
        if item["composite"] >= significant_reference + min_delta:
            significant_reference = item["composite"]
            significant_epoch = item["epoch"]
        if item["epoch"] - significant_epoch >= patience_epochs:
            stopped_epoch = item["epoch"]
            break
    if stopped_epoch is not None and clean[-1]["epoch"] > stopped_epoch:
        raise ModernBertContractError("checkpoints exist after the registered early stop")
    body = {
        "minimum_epoch": minimum_epoch,
        "min_delta": min_delta,
        "patience_epochs": patience_epochs,
        "selected_epoch": best["epoch"],
        "selected_composite": best["composite"],
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "stopped_epoch": stopped_epoch,
        "observed_epochs": clean[-1]["epoch"],
    }
    return {**body, "selection_id": canonical_sha256(body)}


def freeze_experiment_contract(
    *,
    dataset_revision: str,
    dataset_sha256: str,
    split_manifest_sha256: str,
    code_sha256: str,
    dependency_lock_sha256: str,
    development_proxy_id: str,
    teacher_generation_run_id: str,
    rate_card_usd_per_gpu_second: Mapping[str, str | float],
    hard_cost_cap_usd: int | float | str = 200,
) -> dict[str, Any]:
    """Freeze the cross-component identity and budget boundary."""

    if not isinstance(dataset_revision, str) or not dataset_revision:
        raise ValueError("dataset_revision must be non-empty")
    if dataset_revision != DATASET_REVISION or dataset_sha256 != DATASET_SHA256:
        raise ModernBertContractError("private teacher dataset identity drifted")
    if development_proxy_id != DEVELOPMENT_PROXY_ID:
        raise ModernBertContractError("development proxy identity drifted")
    if teacher_generation_run_id != TEACHER_GENERATION_RUN_ID:
        raise ModernBertContractError("teacher generation run identity drifted")
    bindings = {
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "dataset_sha256": _require_sha256(dataset_sha256, where="dataset_sha256"),
        "split_manifest_sha256": _require_sha256(
            split_manifest_sha256, where="split_manifest_sha256"
        ),
        "code_sha256": _require_sha256(code_sha256, where="code_sha256"),
        "dependency_lock_sha256": _require_sha256(
            dependency_lock_sha256, where="dependency_lock_sha256"
        ),
        "development_proxy_id": _require_sha256(development_proxy_id, where="development_proxy_id"),
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
        "bindings": bindings,
        "budget": {
            "hard_cost_cap_usd": format(cap, "f"),
            "rate_card_usd_per_gpu_second": rate_card,
        },
        "registered_design": _registered_design(),
    }
    return {**body, "experiment_run_id": canonical_sha256(body)}


def validate_experiment_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "bindings",
        "budget",
        "registered_design",
        "experiment_run_id",
    }
    if set(value) != required:
        raise ValueError("experiment contract has unexpected fields")
    body = {key: value[key] for key in required - {"experiment_run_id"}}
    if value["experiment_run_id"] != canonical_sha256(body):
        raise ModernBertContractError("experiment contract content address drifted")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != EXPERIMENT_KIND:
        raise ModernBertContractError("experiment contract schema or kind drifted")
    bindings = value["bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError("experiment bindings must be an object")
    if set(bindings) != {
        "dataset_revision",
        "dataset_id",
        "dataset_sha256",
        "split_manifest_sha256",
        "code_sha256",
        "dependency_lock_sha256",
        "development_proxy_id",
        "teacher_generation_run_id",
        "model_id",
        "model_revision",
        "tokenizer_revision",
    }:
        raise ValueError("experiment bindings have unexpected fields")
    for field in _HASH_FIELDS & set(bindings):
        _require_sha256(bindings[field], where=f"bindings.{field}")
    for field in ("development_proxy_id", "teacher_generation_run_id"):
        _require_sha256(bindings[field], where=f"bindings.{field}")
    if (
        bindings.get("dataset_id") != DATASET_ID
        or bindings.get("dataset_revision") != DATASET_REVISION
        or bindings.get("dataset_sha256") != DATASET_SHA256
        or bindings.get("development_proxy_id") != DEVELOPMENT_PROXY_ID
        or bindings.get("teacher_generation_run_id") != TEACHER_GENERATION_RUN_ID
    ):
        raise ModernBertContractError("frozen evidence input identity drifted")
    if bindings.get("model_id") != MODEL_ID or bindings.get("model_revision") != MODEL_REVISION:
        raise ModernBertContractError("pinned ModernBERT identity drifted")
    if bindings.get("tokenizer_revision") != TOKENIZER_REVISION:
        raise ModernBertContractError("pinned tokenizer identity drifted")
    if value["registered_design"] != _registered_design():
        raise ModernBertContractError("registered experiment design drifted")
    budget = value["budget"]
    if not isinstance(budget, Mapping) or set(budget) != {
        "hard_cost_cap_usd",
        "rate_card_usd_per_gpu_second",
    }:
        raise ValueError("experiment budget has unexpected fields")
    rate_card = budget["rate_card_usd_per_gpu_second"]
    if not isinstance(rate_card, Mapping) or not rate_card:
        raise ValueError("experiment rate card must be a non-empty object")
    for gpu_type, raw_rate in rate_card.items():
        if not isinstance(gpu_type, str) or not gpu_type:
            raise ValueError("experiment rate card has an invalid GPU type")
        rate = Decimal(str(raw_rate))
        if not rate.is_finite() or rate <= 0:
            raise ValueError("experiment rate card has an invalid rate")
    cap = Decimal(str(budget["hard_cost_cap_usd"]))
    if cap <= 0 or cap > HARD_COST_CAP_USD:
        raise ModernBertContractError("experiment cost cap exceeds $200")
    return deepcopy(dict(value))


def _validate_trial_condition(
    *,
    phase: str,
    config_sha256: str,
    label_budget: int,
    training_row_count: int,
    ladder_seed: int,
    optimiser_seed: int,
) -> None:
    registered_configs = {item["config_sha256"] for item in frozen_trial_configs()}
    if (
        phase in {"asha", "stability", "confirmatory", "chronological"}
        and config_sha256 not in registered_configs
    ):
        raise ModernBertContractError("full-fine-tune trial uses an unregistered configuration")
    if phase == "asha" and (label_budget, ladder_seed, optimiser_seed) != (5_000, 101, 7):
        raise ModernBertContractError(
            "ASHA trial differs from the registered development condition"
        )
    if phase == "stability" and (
        label_budget != 5_000 or ladder_seed != 101 or optimiser_seed not in STABILITY_SEEDS
    ):
        raise ModernBertContractError("stability trial differs from its registered condition")
    if phase == "confirmatory":
        condition = (ladder_seed, optimiser_seed)
        registered_pairs = {
            (item["ladder_seed"], item["optimiser_seed"]) for item in REGISTERED_LADDERS
        }
        if label_budget not in LABEL_BUDGETS or condition not in registered_pairs:
            raise ModernBertContractError("confirmatory trial differs from the registered design")
    if (
        phase in {"asha", "stability", "confirmatory"}
        and abs(training_row_count - label_budget) > 1
    ):
        raise ModernBertContractError(
            "training row count differs unexpectedly from the nominal acquisition budget"
        )


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
    gpu_type: str,
    max_gpu_seconds: int,
) -> dict[str, Any]:
    """Freeze one schedulable GPU trial and its maximum spend reservation."""

    clean_experiment = validate_experiment_contract(experiment)
    if phase not in {"preflight", "control", "asha", "stability", "confirmatory", "chronological"}:
        raise ValueError("unsupported trial phase")
    clean_config = _json_clone(config)
    config_digest = clean_config.get("config_sha256")
    config_body = {key: value for key, value in clean_config.items() if key != "config_sha256"}
    if config_digest != canonical_sha256(config_body):
        raise ModernBertContractError("trial config content address drifted")
    _require_sha256(subset_manifest_sha256, where="subset_manifest_sha256")
    _require_positive_int(label_budget, where="label_budget")
    _require_positive_int(training_row_count, where="training_row_count")
    _require_positive_int(ladder_seed, where="ladder_seed")
    _require_positive_int(optimiser_seed, where="optimiser_seed")
    _validate_trial_condition(
        phase=phase,
        config_sha256=str(config_digest),
        label_budget=label_budget,
        training_row_count=training_row_count,
        ladder_seed=ladder_seed,
        optimiser_seed=optimiser_seed,
    )
    seconds = _require_positive_int(max_gpu_seconds, where="max_gpu_seconds")
    rate_card = clean_experiment["budget"]["rate_card_usd_per_gpu_second"]
    if gpu_type not in rate_card:
        raise ValueError("gpu_type is not present in the frozen rate card")
    reserved_cost = Decimal(rate_card[gpu_type]) * Decimal(seconds)
    cap = Decimal(clean_experiment["budget"]["hard_cost_cap_usd"])
    if reserved_cost > cap:
        raise ModernBertContractError("single trial reservation exceeds the experiment cap")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_SPEC_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "phase": phase,
        "config": clean_config,
        "subset_manifest_sha256": subset_manifest_sha256,
        "label_budget": label_budget,
        "training_row_count": training_row_count,
        "ladder_seed": ladder_seed,
        "optimiser_seed": optimiser_seed,
        "gpu_type": gpu_type,
        "max_gpu_seconds": seconds,
        "reserved_cost_usd": _money(reserved_cost),
    }
    return {**body, "trial_id": canonical_sha256(body)}


def validate_trial_spec(
    experiment: Mapping[str, Any], trial_spec: Mapping[str, Any]
) -> dict[str, Any]:
    clean_experiment = validate_experiment_contract(experiment)
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
        "gpu_type",
        "max_gpu_seconds",
        "reserved_cost_usd",
        "trial_id",
    }
    if set(trial_spec) != required:
        raise ValueError("trial spec has unexpected fields")
    rebuilt = freeze_trial_spec(
        clean_experiment,
        phase=trial_spec["phase"],
        config=trial_spec["config"],
        subset_manifest_sha256=trial_spec["subset_manifest_sha256"],
        label_budget=trial_spec["label_budget"],
        training_row_count=trial_spec["training_row_count"],
        ladder_seed=trial_spec["ladder_seed"],
        optimiser_seed=trial_spec["optimiser_seed"],
        gpu_type=trial_spec["gpu_type"],
        max_gpu_seconds=trial_spec["max_gpu_seconds"],
    )
    if dict(trial_spec) != rebuilt:
        raise ModernBertContractError("trial spec content address or cost reservation drifted")
    return rebuilt


def build_checkpoint_payload(
    binding: Mapping[str, Any], *, aggregate_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and freeze metadata needed for exact checkpoint resume."""

    required = {
        "dataset_sha256",
        "split_manifest_sha256",
        "trial_manifest_sha256",
        "code_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "epoch",
        "global_step",
        "optimiser_step",
        "seed",
        "config_sha256",
        "model_state_sha256",
    }
    optional = {"optimiser_state_sha256", "scheduler_state_sha256", "rng_state_sha256"}
    if not required <= set(binding) or not set(binding) <= required | optional:
        raise ValueError("checkpoint binding fields differ from the frozen contract")
    clean_binding = _json_clone(binding)
    for field in _HASH_FIELDS & set(clean_binding):
        _require_sha256(clean_binding[field], where=f"checkpoint.{field}")
    if clean_binding["model_id"] != MODEL_ID:
        raise ModernBertContractError("checkpoint model ID drifted")
    if clean_binding["model_revision"] != MODEL_REVISION:
        raise ModernBertContractError("checkpoint model revision drifted")
    if clean_binding["tokenizer_revision"] != TOKENIZER_REVISION:
        raise ModernBertContractError("checkpoint tokenizer revision drifted")
    for field in ("epoch", "global_step", "optimiser_step", "seed"):
        _require_nonnegative_int(clean_binding[field], where=f"checkpoint.{field}")
    metrics = _json_clone(aggregate_metrics)
    assert_metadata_only(metrics, where="checkpoint.aggregate_metrics")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "binding": clean_binding,
        "binding_sha256": canonical_sha256(clean_binding),
        "aggregate_metrics": metrics,
    }
    return {**body, "checkpoint_id": canonical_sha256(body)}


def validate_checkpoint_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "kind",
        "binding",
        "binding_sha256",
        "aggregate_metrics",
        "checkpoint_id",
    }:
        raise ValueError("checkpoint payload has unexpected fields")
    rebuilt = build_checkpoint_payload(
        value["binding"], aggregate_metrics=value["aggregate_metrics"]
    )
    if dict(value) != rebuilt:
        raise ModernBertContractError("checkpoint payload content address drifted")
    return rebuilt


def _validate_artifacts(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    clean = _json_clone(artifacts)
    if not clean:
        raise ValueError("at least one immutable trial artifact is required")
    for name, descriptor in clean.items():
        if not isinstance(name, str) or not name or not isinstance(descriptor, Mapping):
            raise ValueError("artifact descriptors must be named objects")
        if set(descriptor) != {"relative_path", "sha256", "bytes"}:
            raise ValueError("artifact descriptor has unexpected fields")
        if not isinstance(descriptor["relative_path"], str) or not descriptor["relative_path"]:
            raise ValueError("artifact relative_path must be non-empty")
        if (
            Path(descriptor["relative_path"]).is_absolute()
            or ".." in Path(descriptor["relative_path"]).parts
        ):
            raise ValueError("artifact relative_path must be a safe relative path")
        _require_sha256(descriptor["sha256"], where=f"artifacts.{name}.sha256")
        _require_positive_int(descriptor["bytes"], where=f"artifacts.{name}.bytes")
    return clean


def build_trial_receipt(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
    aggregate_metrics: Mapping[str, Any],
    gpu_type: str,
    wall_seconds: int | float,
    gpu_seconds: int | float,
) -> dict[str, Any]:
    """Build a metadata-only complete-trial receipt with measured spend."""

    clean_experiment = validate_experiment_contract(experiment)
    clean_spec = validate_trial_spec(clean_experiment, trial_spec)
    if gpu_type != clean_spec["gpu_type"]:
        raise ModernBertContractError("observed GPU differs from the frozen trial spec")
    clean_gpu_seconds = _require_nonnegative_number(gpu_seconds, where="gpu_seconds")
    clean_wall_seconds = _require_nonnegative_number(wall_seconds, where="wall_seconds")
    if clean_gpu_seconds > clean_spec["max_gpu_seconds"]:
        raise ModernBertContractError("trial exceeded its frozen GPU-second reservation")
    rate = Decimal(clean_experiment["budget"]["rate_card_usd_per_gpu_second"][gpu_type])
    cost = rate * Decimal(str(clean_gpu_seconds))
    metrics = _json_clone(aggregate_metrics)
    assert_metadata_only(metrics, where="trial receipt aggregate_metrics")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_RECEIPT_KIND,
        "status": "complete",
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "trial_id": clean_spec["trial_id"],
        "trial_spec_sha256": canonical_sha256(clean_spec),
        "bindings": deepcopy(clean_experiment["bindings"]),
        "artifacts": _validate_artifacts(artifacts),
        "aggregate_metrics": metrics,
        "compute": {
            "gpu_type": gpu_type,
            "wall_seconds": clean_wall_seconds,
            "gpu_seconds": clean_gpu_seconds,
            "estimated_cost_usd": _money(cost),
        },
    }
    assert_metadata_only(body, where="trial receipt")
    return {**body, "receipt_id": canonical_sha256(body)}


def validate_trial_receipt(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "experiment_run_id",
        "trial_id",
        "trial_spec_sha256",
        "bindings",
        "artifacts",
        "aggregate_metrics",
        "compute",
        "receipt_id",
    }
    if set(receipt) != expected_keys:
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
        artifacts=receipt["artifacts"],
        aggregate_metrics=receipt["aggregate_metrics"],
        gpu_type=compute["gpu_type"],
        wall_seconds=compute["wall_seconds"],
        gpu_seconds=compute["gpu_seconds"],
    )
    if dict(receipt) != rebuilt:
        raise ModernBertContractError("trial receipt content address or accounting drifted")
    return rebuilt


def budget_status(
    experiment: Mapping[str, Any],
    *,
    completed_receipts: Sequence[Mapping[str, Any]] = (),
    active_trial_specs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Account completed spend plus worst-case active reservations."""

    clean_experiment = validate_experiment_contract(experiment)
    receipt_ids: set[str] = set()
    completed = Decimal("0")
    for receipt in completed_receipts:
        receipt_id = _require_sha256(receipt.get("receipt_id"), where="receipt.receipt_id")
        if receipt_id in receipt_ids:
            raise ValueError("completed receipts contain a duplicate")
        receipt_ids.add(receipt_id)
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        if canonical_sha256(body) != receipt_id or receipt.get("status") != "complete":
            raise ModernBertContractError("completed receipt is invalid or incomplete")
        if receipt.get("experiment_run_id") != clean_experiment["experiment_run_id"]:
            raise ModernBertContractError("completed receipt belongs to another experiment")
        compute = receipt.get("compute")
        if not isinstance(compute, Mapping) or set(compute) != {
            "gpu_type",
            "wall_seconds",
            "gpu_seconds",
            "estimated_cost_usd",
        }:
            raise ValueError("completed receipt has an invalid compute block")
        gpu_type = compute["gpu_type"]
        rate_card = clean_experiment["budget"]["rate_card_usd_per_gpu_second"]
        if gpu_type not in rate_card:
            raise ModernBertContractError("completed receipt uses an unregistered GPU")
        gpu_seconds = _require_nonnegative_number(
            compute["gpu_seconds"], where="receipt.compute.gpu_seconds"
        )
        measured_cost = Decimal(rate_card[gpu_type]) * Decimal(str(gpu_seconds))
        if _money(measured_cost) != compute["estimated_cost_usd"]:
            raise ModernBertContractError("completed receipt cost does not match the rate card")
        completed += measured_cost

    trial_ids: set[str] = set()
    reserved = Decimal("0")
    for spec in active_trial_specs:
        clean_spec = validate_trial_spec(clean_experiment, spec)
        if clean_spec["trial_id"] in trial_ids:
            raise ValueError("active trial reservations contain a duplicate")
        trial_ids.add(clean_spec["trial_id"])
        reserved += Decimal(str(clean_spec["reserved_cost_usd"]))
    cap = Decimal(clean_experiment["budget"]["hard_cost_cap_usd"])
    projected = completed + reserved
    if projected > cap:
        raise ModernBertContractError("completed spend plus active reservations exceeds the cap")
    return {
        "hard_cost_cap_usd": _money(cap),
        "completed_cost_usd": _money(completed),
        "active_reserved_cost_usd": _money(reserved),
        "projected_cost_usd": _money(projected),
        "remaining_unreserved_usd": _money(cap - projected),
        "completed_trial_count": len(receipt_ids),
        "active_trial_count": len(trial_ids),
    }


def issue_locked_test_authorisation(
    experiment: Mapping[str, Any],
    *,
    recipe_receipt_id: str,
    threshold_receipt_id: str,
    confirmatory_trial_specs: Sequence[Mapping[str, Any]],
    confirmatory_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Open locked-test access only after all nine confirmatory runs freeze."""

    clean_experiment = validate_experiment_contract(experiment)
    _require_sha256(recipe_receipt_id, where="recipe_receipt_id")
    _require_sha256(threshold_receipt_id, where="threshold_receipt_id")
    if len(confirmatory_trial_specs) != 9 or len(confirmatory_receipts) != 9:
        raise ModernBertContractError("locked test requires exactly nine confirmatory runs")
    specs = [validate_trial_spec(clean_experiment, item) for item in confirmatory_trial_specs]
    if any(item["phase"] != "confirmatory" for item in specs):
        raise ModernBertContractError("locked-test gate received a non-confirmatory trial")
    expected_conditions = {
        (ladder["ladder_seed"], ladder["optimiser_seed"], budget)
        for ladder in REGISTERED_LADDERS
        for budget in LABEL_BUDGETS
    }
    observed_conditions = {
        (item["ladder_seed"], item["optimiser_seed"], item["label_budget"]) for item in specs
    }
    if observed_conditions != expected_conditions:
        raise ModernBertContractError(
            "confirmatory trials do not cover the registered 3 x 3 design"
        )
    if len({item["config"]["config_sha256"] for item in specs}) != 1:
        raise ModernBertContractError("confirmatory trials do not share one frozen recipe")
    specs_by_id = {item["trial_id"]: item for item in specs}
    if len(specs_by_id) != 9:
        raise ValueError("confirmatory trial specs contain duplicates")
    receipts_by_trial: dict[str, dict[str, Any]] = {}
    for receipt in confirmatory_receipts:
        trial_id = receipt.get("trial_id")
        if trial_id not in specs_by_id or trial_id in receipts_by_trial:
            raise ValueError("confirmatory receipts do not map one-to-one to trial specs")
        receipts_by_trial[str(trial_id)] = validate_trial_receipt(
            clean_experiment, specs_by_id[str(trial_id)], receipt
        )
    if set(receipts_by_trial) != set(specs_by_id):
        raise ModernBertContractError("confirmatory receipt set is incomplete")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": LOCKED_TEST_AUTHORISATION_KIND,
        "experiment_run_id": clean_experiment["experiment_run_id"],
        "recipe_receipt_id": recipe_receipt_id,
        "threshold_receipt_id": threshold_receipt_id,
        "confirmatory_receipt_ids": sorted(
            receipt["receipt_id"] for receipt in receipts_by_trial.values()
        ),
        "locked_test_rows": 230,
        "single_access": True,
    }
    assert_metadata_only(body, where="locked-test authorisation")
    return {**body, "authorisation_id": canonical_sha256(body)}


def _validate_modal_job(job: Mapping[str, Any], *, expected_phase: str | None) -> dict[str, Any]:
    required = {
        "experiment_run_id",
        "phase_run_id",
        "experiment_contract",
        "trial_spec",
        "trial_spec_sha256",
    }
    if set(job) != required:
        raise ValueError("Modal training job has unexpected fields")
    contract = job.get("experiment_contract")
    spec = job.get("trial_spec")
    if not isinstance(contract, Mapping) or not isinstance(spec, Mapping):
        raise ValueError("Modal training job contract or trial spec is missing")
    if job["experiment_run_id"] != canonical_sha256(contract):
        raise ModernBertContractError("Modal experiment run binding drifted")
    if job["trial_spec_sha256"] != canonical_sha256(spec):
        raise ModernBertContractError("Modal trial spec binding drifted")
    spec_body = {key: value for key, value in spec.items() if key != "trial_id"}
    if spec.get("trial_id") != canonical_sha256(spec_body):
        raise ModernBertContractError("Modal trial ID content address drifted")
    if expected_phase is not None and spec.get("phase") != expected_phase:
        raise ValueError(f"adapter requires phase={expected_phase}")
    bindings = contract.get("bindings")
    compute = contract.get("compute")
    if not isinstance(bindings, Mapping) or not isinstance(compute, Mapping):
        raise ValueError("Modal experiment contract is incomplete")
    expected_bindings = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_parquet_sha256": DATASET_SHA256,
        "development_proxy_sha256": DEVELOPMENT_PROXY_ID,
    }
    for field, expected in expected_bindings.items():
        if bindings.get(field) != expected:
            raise ModernBertContractError(f"Modal input binding drifted: {field}")
    for field in ("split_manifest_sha256", "trainer_code_sha256", "dependency_lock_sha256"):
        _require_sha256(bindings.get(field), where=f"Modal bindings.{field}")
    approved = Decimal(str(compute.get("approved_cost_usd")))
    hard_max = Decimal(str(compute.get("hard_max_approved_cost_usd")))
    if approved <= 0 or approved > HARD_COST_CAP_USD or hard_max != HARD_COST_CAP_USD:
        raise ModernBertContractError("Modal experiment does not enforce the approved $200 cap")
    gpu_type = spec.get("gpu_type")
    if gpu_type not in GPU_RATE_USD_PER_SECOND:
        raise ValueError("Modal trial requests an unsupported GPU")
    return _json_clone(job)


def _private_input_paths(volume_root: Path) -> dict[str, Path]:
    root = volume_root / VOLUME_INPUT_PREFIX
    return {
        "teacher": root / TEACHER_PARQUET_NAME,
        "split": root / SPLIT_MANIFEST_NAME,
        "proxy": root / DEVELOPMENT_PROXY_PARQUET_NAME,
    }


def _load_runtime_inputs(
    job: Mapping[str, Any], *, volume_root: Path, include_development: bool
) -> dict[str, Any]:
    """Load exact private inputs while exposing only the 222-row development split."""

    _, pq = _require_pyarrow()
    paths = _private_input_paths(volume_root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ModernBERT Volume inputs are missing: {sorted(missing)}")
    bindings = job["experiment_contract"]["bindings"]
    if file_sha256(paths["teacher"]) != bindings["dataset_parquet_sha256"]:
        raise ModernBertContractError("teacher Parquet hash differs from the Modal contract")
    if file_sha256(paths["split"]) != bindings["split_manifest_sha256"]:
        raise ModernBertContractError("split manifest file hash differs from the Modal contract")
    teacher_table = pq.read_table(paths["teacher"])
    if teacher_table.num_rows != 10_000:
        raise ModernBertContractError("teacher Parquet must contain exactly 10,000 rows")
    teacher_rows = teacher_table.to_pylist()
    from reddit_china_stance.modernbert_splits import validate_split_manifest

    split_manifest = validate_split_manifest(
        _read_json_object(paths["split"], where="split manifest"),
        source_rows=teacher_rows,
    )
    proxy_file = pq.ParquetFile(paths["proxy"])
    if proxy_file.metadata.num_rows != 452:
        raise ModernBertContractError("private development proxy must contain 452 rows")
    expected_proxy_columns = {
        "source_sample_id",
        "target_text",
        "parent_context",
        "submission_context",
        "split",
        "resolution",
        "label_json",
    }
    if set(proxy_file.schema_arrow.names) != expected_proxy_columns:
        raise ModernBertContractError("private development proxy schema drifted")
    proxy_metadata = proxy_file.schema_arrow.metadata or {}
    if proxy_metadata.get(b"proxy_id") != DEVELOPMENT_PROXY_ID.encode():
        raise ModernBertContractError("private development proxy ID binding drifted")
    # The locked rows stay behind the explicit gate: training code requests only
    # development rows and never materialises their labels or text.
    development: list[dict[str, Any]] = []
    if include_development:
        development = pq.read_table(
            paths["proxy"],
            filters=[("split", "=", "development")],
        ).to_pylist()
        if len(development) != 222 or any(row.get("split") != "development" for row in development):
            raise ModernBertContractError("development proxy filter did not yield exactly 222 rows")
    return {
        "teacher_rows": teacher_rows,
        "split_manifest": split_manifest,
        "development_rows": development,
    }


def _selected_teacher_rows(
    teacher_rows: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    trial_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = trial_spec.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("trial config is missing")
    phase = str(trial_spec.get("phase"))
    budget_name = "train_5k_101" if phase == "sweep" else config.get("budget")
    if phase == "chronological":
        selected_ids = {
            row["sample_id"]
            for row in split_manifest["rows"]
            if row["chronological_split"] == "train"
        }
    else:
        if not isinstance(budget_name, str) or not budget_name.startswith("train_"):
            raise ValueError("training trial config lacks a registered train budget")
        parts = budget_name.removeprefix("train_").split("_")
        if len(parts) != 2 or parts[0] not in {"2k", "5k", "10k"}:
            raise ValueError("training trial budget must be train_{2k,5k,10k}_{seed}")
        budget, raw_seed = parts
        try:
            ladder_seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError("training trial budget has an invalid ladder seed") from exc
        fold_limits = {"2k": {0, 1}, "5k": set(range(5)), "10k": set(range(10))}
        selected_ids = {
            row["sample_id"]
            for row in split_manifest["rows"]
            if row["folds"].get(str(ladder_seed)) in fold_limits[budget]
        }
        expected = split_manifest["ladders"][str(ladder_seed)]["budgets"][budget]["row_count"]
        if len(selected_ids) != expected:
            raise ModernBertContractError("selected teacher subset does not match split metadata")
    by_id = {str(row["sample_id"]): dict(row) for row in teacher_rows}
    if len(by_id) != len(teacher_rows) or not selected_ids <= set(by_id):
        raise ModernBertContractError("teacher rows do not conserve split-manifest IDs")
    return [by_id[item_id] for item_id in sorted(selected_ids)]


def _semantic_label_from_teacher_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("label_json")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("teacher row has invalid label_json") from exc
    if not isinstance(value, dict):
        raise ValueError("teacher row label_json must contain an object")
    return value


def _class_weight_vectors(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, ...]]:
    from reddit_china_stance.modernbert_trainer import (
        RELEVANCE_LABELS,
        STANCE_LABELS,
        TARGET_LABELS,
    )

    relevance = Counter[str]()
    targets = Counter[str]()
    stances = Counter[str]()
    material_rows = 0
    target_instances = 0
    for row in rows:
        label = _semantic_label_from_teacher_row(row)
        relevance[str(label["relevance"])] += 1
        if label["relevance"] == "material":
            material_rows += 1
        for item in label["target_stances"]:
            targets[str(item["target"])] += 1
            stances[str(item["stance"])] += 1
            target_instances += 1

    def normalised_sqrt(
        labels: Sequence[str], counts: Counter[str], total: int
    ) -> tuple[float, ...]:
        raw = [math.sqrt(total / counts[label]) if counts[label] else 0.0 for label in labels]
        supported = [value for value in raw if value]
        mean = sum(supported) / len(supported)
        return tuple(0.0 if value == 0 else min(4.0, max(0.5, value / mean)) for value in raw)

    relevance_weights = normalised_sqrt(RELEVANCE_LABELS, relevance, len(rows))
    stance_weights = normalised_sqrt(STANCE_LABELS, stances, target_instances)
    target_weights = tuple(
        min(4.0, max(1.0, math.sqrt((material_rows - targets[label]) / targets[label])))
        if targets[label] and material_rows > targets[label]
        else 1.0
        for label in TARGET_LABELS
    )
    return {
        "relevance": relevance_weights,
        "targets": target_weights,
        "stance": stance_weights,
    }


def _optimisation_config(
    trial_spec: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]
) -> Any:
    from reddit_china_stance.modernbert_trainer import OptimisationConfig

    outer_config = trial_spec["config"]
    phase = trial_spec.get("phase")
    has_registered_config = "registered_config" in outer_config
    wrapped_phases = {"sweep", "stability", "confirmatory"}
    if has_registered_config != (phase in wrapped_phases):
        raise ValueError("trial phase and registered configuration wrapper disagree")
    config = outer_config.get("registered_config") if has_registered_config else outer_config
    if not isinstance(config, Mapping):
        raise ValueError("trial lacks its registered configuration")
    if has_registered_config:
        registered = {row["config_sha256"]: row for row in frozen_trial_configs()}
        config_id = config.get("config_sha256")
        if registered.get(config_id) != dict(config):
            raise ModernBertContractError(
                "wrapped trial configuration is not an exact registered recipe"
            )
    loss_weights = config.get("loss_weights", {"relevance": 1, "targets": 1, "stance": 1})
    if not isinstance(loss_weights, Mapping):
        raise ValueError("trial loss weights must be an object")
    weighted = config.get("class_weights") == "capped_inverse_sqrt"
    weights = _class_weight_vectors(train_rows) if weighted else None
    microbatch = int(outer_config.get("microbatch_size", 4))
    if microbatch <= 0 or 32 % microbatch:
        raise ValueError("microbatch_size must be a positive divisor of 32")
    return OptimisationConfig(
        encoder_learning_rate=float(config.get("encoder_learning_rate", "0.00003")),
        gradient_accumulation_steps=32 // microbatch,
        effective_batch_size=32,
        gradient_checkpointing=bool(outer_config.get("gradient_checkpointing", True)),
        lambda_relevance=float(loss_weights["relevance"]),
        lambda_targets=float(loss_weights["targets"]),
        lambda_stance=float(loss_weights["stance"]),
        relevance_class_weights=None if weights is None else weights["relevance"],
        target_positive_weights=None if weights is None else weights["targets"],
        stance_class_weights=None if weights is None else weights["stance"],
    )


def _evaluation_target_threshold(trial_spec: Mapping[str, Any]) -> float:
    """Resolve the registered evaluation threshold, including the frozen Phase 4 override."""

    phase = trial_spec.get("phase")
    outer = trial_spec.get("config")
    if not isinstance(outer, Mapping):
        raise ValueError("trial config is missing")
    registered = outer.get("registered_config", outer)
    if not isinstance(registered, Mapping):
        raise ValueError("trial lacks its registered configuration")
    if phase == "confirmatory":
        if "registered_config" not in outer:
            raise ValueError("confirmatory trial requires a registered configuration wrapper")
        if outer.get("target_threshold") != "0.30":
            raise ModernBertContractError(
                "confirmatory trial must use the separately frozen target threshold 0.30"
            )
        if (
            outer.get("fresh_training") is not True
            or outer.get("continuation") is not None
            or outer.get("locked_test_authorised") is not False
        ):
            raise ModernBertContractError(
                "confirmatory execution broadened fresh-training or locked-test authorisation"
            )
        for field in ("recipe_receipt_id", "threshold_receipt_id"):
            _require_sha256(outer.get(field), where=f"confirmatory.{field}")
        return 0.30
    if "target_threshold" in outer and outer is not registered:
        raise ModernBertContractError("only confirmatory trials may override the recipe threshold")
    threshold = float(registered.get("target_threshold", 0.50))
    if threshold != 0.50:
        raise ModernBertContractError("development recipe threshold drifted from 0.50")
    return threshold


def _tokenise_rows(
    rows: Sequence[Mapping[str, Any]], *, tokenizer: Any, item_id_field: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from reddit_china_stance.modernbert_trainer import encode_semantic_label, tokenise_record

    encoded: list[dict[str, Any]] = []
    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get(item_id_field)
        if not isinstance(item_id, str) or not item_id or item_id in reference:
            raise ValueError("training/evaluation rows contain invalid or duplicate item IDs")
        label = (
            _semantic_label_from_teacher_row(row)
            if item_id_field == "sample_id"
            else json.loads(row["label_json"])
        )
        reference[item_id] = label
        tokenised = tokenise_record(tokenizer, row)
        encoded.append(
            {
                "item_id": item_id,
                **tokenised,
                **encode_semantic_label(label),
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
    return round(0.25 * float(values[0]) + 0.25 * float(values[1]) + 0.50 * float(values[2]), 6)


def _write_resume_marker(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _restore_asha_continuation(
    *,
    continuation: Mapping[str, Any],
    registered_config_id: str,
    target_epoch: int,
    run_root: Path,
    expected_optimisation_sha256: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    torch_load: Any,
    checkpoint_loader: Any,
    restore_training: Any,
) -> tuple[int, int, int]:
    """Validate and restore an exact promoted-rung checkpoint."""

    if continuation.get("config_sha256") != registered_config_id:
        raise ModernBertContractError("ASHA continuation configuration drifted")
    relative = Path(str(continuation.get("checkpoint_relative_path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ASHA continuation checkpoint path is unsafe")
    expected_prefix = Path("phase=sweep") / f"trial={continuation.get('source_trial_id')}"
    if tuple(relative.parts[:2]) != tuple(expected_prefix.parts):
        raise ModernBertContractError("ASHA continuation source-trial path drifted")
    checkpoint_path = run_root / relative
    expected_file_sha256 = _require_sha256(
        continuation.get("checkpoint_sha256"), where="continuation.checkpoint_sha256"
    )
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != expected_file_sha256:
        raise ModernBertContractError("ASHA continuation checkpoint is missing or corrupt")
    preview = torch_load(checkpoint_path, map_location="cpu", weights_only=False)
    binding_sha256 = preview.get("binding_sha256") if isinstance(preview, Mapping) else None
    _require_sha256(binding_sha256, where="continuation.binding_sha256")
    payload = checkpoint_loader(
        checkpoint_path,
        expected_file_sha256=expected_file_sha256,
        expected_binding_sha256=binding_sha256,
        map_location="cuda",
    )
    binding = payload["binding"]
    expected_source_epoch = {3: 1, 6: 3}.get(target_epoch)
    if expected_source_epoch is None or binding.get("epoch") != expected_source_epoch:
        raise ModernBertContractError("ASHA continuation checkpoint is from the wrong rung")
    if binding.get("optimisation_config_sha256") != expected_optimisation_sha256:
        raise ModernBertContractError("ASHA continuation optimisation config drifted")
    restore_training(
        payload, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True
    )
    return (
        int(binding["epoch"]),
        int(binding["global_step"]),
        int(binding["optimizer_step"]),
    )


def _execute_trial_runtime(
    *,
    job: Mapping[str, Any],
    inputs: Mapping[str, Any],
    staging_root: Path,
    resume: bool,
    preflight: bool,
) -> dict[str, Any]:
    """Execute the pinned trainer.  Heavy imports remain inside the GPU path."""

    import torch
    from torch.utils.data import DataLoader

    from reddit_china_stance.modernbert_trainer import (
        DynamicPaddingCollator,
        ModelConfig,
        build_length_bucket_batches,
        create_adamw,
        create_modernbert_three_head_model,
        evaluate_epoch,
        load_checkpoint_exact,
        load_pinned_tokenizer,
        restore_training_state,
        save_checkpoint_atomic,
        seed_everything,
        train_epoch,
    )
    from reddit_china_stance.modernbert_trainer import (
        build_checkpoint_payload as build_trainer_checkpoint,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("ModernBERT production training requires a CUDA GPU")
    spec = job["trial_spec"]
    train_rows = _selected_teacher_rows(
        inputs["teacher_rows"], inputs["split_manifest"], trial_spec=spec
    )
    outer_config = spec["config"]
    registered_config = outer_config.get("registered_config", outer_config)
    if not isinstance(registered_config, Mapping):
        raise ValueError("trial lacks a usable registered configuration")
    seed = int(outer_config.get("seed", 7))
    seed_everything(seed)
    optimisation = _optimisation_config(spec, train_rows)
    evaluation_target_threshold = _evaluation_target_threshold(spec)
    model_config = ModelConfig()
    tokenizer = load_pinned_tokenizer(model_config)
    train_encoded, _ = _tokenise_rows(train_rows, tokenizer=tokenizer, item_id_field="sample_id")
    development_encoded: list[dict[str, Any]] = []
    development_reference: dict[str, dict[str, Any]] = {}
    if not preflight:
        development_encoded, development_reference = _tokenise_rows(
            inputs["development_rows"], tokenizer=tokenizer, item_id_field="source_sample_id"
        )
    collator = DynamicPaddingCollator(tokenizer)
    microbatch = 32 // optimisation.gradient_accumulation_steps
    train_lengths = [len(row["input_ids"]) for row in train_encoded]
    development_loader = None
    if not preflight:
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
    model = create_modernbert_three_head_model(
        model_config=model_config, optimisation_config=optimisation
    ).to("cuda")
    optimizer = create_adamw(model, optimisation)
    maximum_epochs = int(outer_config.get("maximum_epochs", registered_config.get("max_epochs", 6)))
    target_epoch = int(
        outer_config.get("target_epochs", outer_config.get("target_epoch", maximum_epochs))
    )
    if target_epoch < 1 or target_epoch > maximum_epochs:
        raise ValueError("target_epoch must be within the frozen maximum epochs")
    batches_per_epoch = math.ceil(len(train_encoded) / microbatch)
    updates_per_epoch = math.ceil(batches_per_epoch / optimisation.gradient_accumulation_steps)
    total_updates = 200 if preflight else maximum_epochs * updates_per_epoch
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
    development_predictions: dict[str, Any] | None = None
    if resume:
        marker = _read_json_object(staging_root / "resume.json", where="resume marker")
        checkpoint = marker["checkpoint"]
        checkpoint_path = staging_root / checkpoint["relative_path"]
        payload = load_checkpoint_exact(
            checkpoint_path,
            expected_file_sha256=checkpoint["sha256"],
            expected_binding_sha256=checkpoint["binding_sha256"],
            map_location="cuda",
        )
        binding = payload["binding"]
        if binding["trial_manifest_sha256"] != job["trial_spec_sha256"]:
            raise ModernBertContractError("resume checkpoint belongs to another trial")
        restore_training_state(
            payload, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True
        )
        start_epoch = int(binding["epoch"])
        global_step = int(binding["global_step"])
        optimizer_step = int(binding["optimizer_step"])
        history_path = staging_root / "history.json"
        if history_path.exists():
            history_value = _read_json_object(history_path, where="training history")
            history = list(history_value.get("epochs", []))
    elif outer_config.get("continuation") is not None:
        continuation = outer_config["continuation"]
        if not isinstance(continuation, Mapping):
            raise ValueError("ASHA continuation must be an object")
        start_epoch, global_step, optimizer_step = _restore_asha_continuation(
            continuation=continuation,
            registered_config_id=str(registered_config.get("config_sha256")),
            target_epoch=target_epoch,
            run_root=staging_root.parent.parent,
            expected_optimisation_sha256=optimisation.digest(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            torch_load=torch.load,
            checkpoint_loader=load_checkpoint_exact,
            restore_training=restore_training_state,
        )

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    selected_checkpoint_relative_path: str | None = None
    if preflight and start_epoch == 0:
        base_batches = build_length_bucket_batches(
            train_lengths, batch_size=microbatch, seed=seed, epoch=0
        )
        required_batches = 200 * optimisation.gradient_accumulation_steps
        repeated = [base_batches[index % len(base_batches)] for index in range(required_batches)]
        loader = DataLoader(
            train_encoded, batch_sampler=repeated, collate_fn=collator, num_workers=0
        )
        train_result = train_epoch(
            model,
            loader,
            optimizer,
            config=optimisation,
            device="cuda",
            scheduler=scheduler,
        )
        if train_result["optimizer_steps"] != 200:
            raise ModernBertContractError("preflight did not execute exactly 200 optimiser updates")
        history = [{"epoch": 1, "train": train_result}]
        start_epoch = 1
        global_step += train_result["batches"]
        optimizer_step += train_result["optimizer_steps"]
    elif not preflight:
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
                target_threshold=evaluation_target_threshold,
                reference=development_reference,
                config=optimisation,
            )
            evaluation["composite"] = _composite(evaluation["metrics"])
            prediction_payload = evaluation.pop("prediction_payload", None)
            if not isinstance(prediction_payload, Mapping):
                raise ModernBertContractError(
                    "development evaluation did not return private predictions"
                )
            development_predictions = dict(prediction_payload)
            global_step += train_result["batches"]
            optimizer_step += train_result["optimizer_steps"]
            history.append({"epoch": epoch, "train": train_result, "development": evaluation})
            checkpoint_payload = build_trainer_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_config=model_config,
                optimisation_config=optimisation,
                binding={
                    "dataset_sha256": job["experiment_contract"]["bindings"][
                        "dataset_parquet_sha256"
                    ],
                    "split_manifest_sha256": job["experiment_contract"]["bindings"][
                        "split_manifest_sha256"
                    ],
                    "trial_manifest_sha256": job["trial_spec_sha256"],
                    "code_sha256": job["experiment_contract"]["bindings"]["trainer_code_sha256"],
                },
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                seed=seed,
            )
            checkpoint_path = checkpoint_dir / f"epoch-{epoch:02d}.pt"
            descriptor = save_checkpoint_atomic(checkpoint_payload, checkpoint_path)
            relative_path = str(checkpoint_path.relative_to(staging_root))
            _write_resume_marker(
                staging_root / "resume.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_run_id": job["experiment_run_id"],
                    "trial_id": spec["trial_id"],
                    "trial_spec_sha256": job["trial_spec_sha256"],
                    "checkpoint": {
                        "relative_path": relative_path,
                        "sha256": descriptor["sha256"],
                        "bytes": descriptor["bytes"],
                        "binding_sha256": checkpoint_payload["binding_sha256"],
                    },
                },
            )
            _write_resume_marker(staging_root / "history.json", {"epochs": history})
            selected_checkpoint_relative_path = relative_path
            if spec["phase"] != "sweep" and epoch >= 2:
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
                    selected_checkpoint_relative_path = (
                        f"checkpoints/epoch-{selection['selected_epoch']:02d}.pt"
                    )
                    break
        if not history:
            raise ModernBertContractError("training produced no development history")
        if spec["phase"] == "sweep":
            selected_epoch = int(history[-1]["epoch"])
        else:
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
            selected_epoch = int(selection["selected_epoch"])
        selected_checkpoint_relative_path = f"checkpoints/epoch-{selected_epoch:02d}.pt"
        selected_checkpoint_path = staging_root / selected_checkpoint_relative_path
        preview = torch.load(selected_checkpoint_path, map_location="cpu", weights_only=False)
        binding_sha256 = preview.get("binding_sha256") if isinstance(preview, Mapping) else None
        _require_sha256(binding_sha256, where="selected checkpoint binding_sha256")
        selected_payload = load_checkpoint_exact(
            selected_checkpoint_path,
            expected_file_sha256=file_sha256(selected_checkpoint_path),
            expected_binding_sha256=binding_sha256,
            map_location="cuda",
        )
        restore_training_state(
            selected_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=False,
        )
        selected_evaluation = evaluate_epoch(
            model,
            development_loader,
            device="cuda",
            target_threshold=evaluation_target_threshold,
            reference=development_reference,
            config=optimisation,
        )
        selected_evaluation["composite"] = _composite(selected_evaluation["metrics"])
        selected_history = next(item for item in history if int(item["epoch"]) == selected_epoch)
        if selected_evaluation["composite"] != selected_history["development"]["composite"]:
            raise ModernBertContractError(
                "selected checkpoint evaluation differs from its training history"
            )
        prediction_payload = selected_evaluation.pop("prediction_payload", None)
        if not isinstance(prediction_payload, Mapping):
            raise ModernBertContractError(
                "selected checkpoint evaluation did not return private predictions"
            )
        development_predictions = dict(prediction_payload)
    if preflight:
        checkpoint_path = checkpoint_dir / "preflight.pt"
        if not checkpoint_path.exists():
            checkpoint_payload = build_trainer_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_config=model_config,
                optimisation_config=optimisation,
                binding={
                    "dataset_sha256": job["experiment_contract"]["bindings"][
                        "dataset_parquet_sha256"
                    ],
                    "split_manifest_sha256": job["experiment_contract"]["bindings"][
                        "split_manifest_sha256"
                    ],
                    "trial_manifest_sha256": job["trial_spec_sha256"],
                    "code_sha256": job["experiment_contract"]["bindings"]["trainer_code_sha256"],
                },
                epoch=1,
                global_step=global_step,
                optimizer_step=optimizer_step,
                seed=seed,
            )
            descriptor = save_checkpoint_atomic(checkpoint_payload, checkpoint_path)
            _write_resume_marker(staging_root / "history.json", {"epochs": history})
            _write_resume_marker(
                staging_root / "resume.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_run_id": job["experiment_run_id"],
                    "trial_id": spec["trial_id"],
                    "trial_spec_sha256": job["trial_spec_sha256"],
                    "checkpoint": {
                        "relative_path": "checkpoints/preflight.pt",
                        "sha256": descriptor["sha256"],
                        "bytes": descriptor["bytes"],
                        "binding_sha256": checkpoint_payload["binding_sha256"],
                    },
                },
            )
        selected_checkpoint_relative_path = "checkpoints/preflight.pt"
    elapsed = time.monotonic() - started
    return {
        "history": history,
        "training_rows": len(train_rows),
        "development_rows": 0 if preflight else len(development_reference),
        "truncated_training_rows": sum(row["truncated_tokens"] > 0 for row in train_encoded),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "selected_checkpoint_relative_path": selected_checkpoint_relative_path,
        "selected_epoch": None if preflight else selected_epoch,
        "development_predictions": development_predictions,
        "wall_seconds": elapsed,
        "gpu_seconds": elapsed,
    }


def _public_trial_metrics(metrics: Mapping[str, Any], *, preflight: bool) -> dict[str, Any]:
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ModernBertContractError("trial runtime returned no epoch history")
    last = history[-1]
    result: dict[str, Any] = {
        "training_rows": metrics.get("training_rows"),
        "development_rows": metrics.get("development_rows"),
        "truncated_training_rows": metrics.get("truncated_training_rows"),
        "peak_cuda_memory_bytes": metrics.get("peak_cuda_memory_bytes"),
        "completed_epochs": last.get("epoch"),
        "optimizer_steps": sum(
            int(item.get("train", {}).get("optimizer_steps", 0)) for item in history
        ),
    }
    if preflight:
        result["mean_losses"] = last.get("train", {}).get("mean_losses")
    else:
        selected_epoch = metrics.get("selected_epoch")
        if type(selected_epoch) is not int:
            raise ModernBertContractError("trial runtime omitted its selected epoch")
        selected = next((item for item in history if item.get("epoch") == selected_epoch), None)
        if not isinstance(selected, Mapping):
            raise ModernBertContractError("selected epoch is absent from trial history")
        result["selected_epoch"] = selected_epoch
        development = selected.get("development", {})
        semantic = development.get("metrics", {})
        result.update(
            {
                "composite": development.get("composite"),
                "invalid_outputs": semantic.get("invalid_outputs"),
                "relevance_macro_f1": semantic.get("relevance", {}).get("macro_f1"),
                "material_recall": semantic.get("relevance", {}).get("material_recall"),
                "core_target_micro_f1": semantic.get("targets", {})
                .get("core", {})
                .get("micro", {})
                .get("f1"),
                "core_target_stance_tuple_micro_f1": semantic.get(
                    "end_to_end_core_target_stance", {}
                )
                .get("micro", {})
                .get("f1"),
                "forced_target_selections": semantic.get("decoding", {}).get(
                    "forced_target_selections"
                ),
            }
        )
    assert_metadata_only(result, where="ModernBERT public trial metrics")
    return result


def _finite_vector(value: Any, *, length: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must contain exactly {length} logits")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{where} logits must be finite numbers")
        clean = float(item)
        if not math.isfinite(clean):
            raise ValueError(f"{where} logits must be finite numbers")
        result.append(clean)
    return result


def validate_private_development_predictions(value: Any, *, expected_rows: int) -> dict[str, Any]:
    """Validate the private row-level logits needed for threshold calibration.

    This payload intentionally fails the repository's metadata-only contract and
    must stay inside the private run namespace.  Only its file descriptor may be
    copied into the public trial receipt.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "kind",
        "row_count",
        "decoder_target_threshold",
        "rows",
    }:
        raise ValueError("private development prediction payload schema drifted")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != "modernbert-private-development-predictions-v1"
        or value["row_count"] != expected_rows
        or type(expected_rows) is not int
        or expected_rows <= 0
    ):
        raise ValueError("private development prediction payload binding drifted")
    threshold = _require_nonnegative_number(
        value["decoder_target_threshold"], where="decoder_target_threshold"
    )
    if not 0 < threshold < 1:
        raise ValueError("decoder_target_threshold must be in (0, 1)")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError("private development prediction row count drifted")
    clean_rows = []
    seen_ids: set[str] = set()
    relevance_labels = {"material", "not_material", "unclear"}
    target_labels = {"china_general", "government_ccp", "people_culture", "other"}
    stance_labels = {"negative", "mixed", "no_directed_stance", "positive", "unclear"}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "source_sample_id",
            "relevance_logits",
            "target_logits",
            "stance_logits",
            "decoded_label",
        }:
            raise ValueError("private development prediction row schema drifted")
        item_id = row["source_sample_id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            raise ValueError("private development prediction IDs must be non-empty and unique")
        seen_ids.add(item_id)
        relevance_logits = _finite_vector(
            row["relevance_logits"], length=3, where="relevance_logits"
        )
        target_logits = _finite_vector(row["target_logits"], length=4, where="target_logits")
        stance_rows = row["stance_logits"]
        if not isinstance(stance_rows, list) or len(stance_rows) != 4:
            raise ValueError("stance_logits must contain four target rows")
        stance_logits = [
            _finite_vector(item, length=5, where="stance_logits") for item in stance_rows
        ]
        decoded = row["decoded_label"]
        if not isinstance(decoded, Mapping) or set(decoded) != {
            "relevance",
            "target_stances",
        }:
            raise ValueError("decoded development label schema drifted")
        relevance = decoded["relevance"]
        target_stances = decoded["target_stances"]
        if relevance not in relevance_labels or not isinstance(target_stances, list):
            raise ValueError("decoded development label contains an invalid relevance")
        if relevance != "material" and target_stances:
            raise ValueError("non-material development prediction cannot contain targets")
        for target_stance in target_stances:
            if (
                not isinstance(target_stance, Mapping)
                or set(target_stance) != {"target", "stance"}
                or target_stance["target"] not in target_labels
                or target_stance["stance"] not in stance_labels
            ):
                raise ValueError("decoded development target stance is invalid")
        clean_rows.append(
            {
                "source_sample_id": item_id,
                "relevance_logits": relevance_logits,
                "target_logits": target_logits,
                "stance_logits": stance_logits,
                "decoded_label": _json_clone(decoded),
            }
        )
    if [row["source_sample_id"] for row in clean_rows] != sorted(seen_ids):
        raise ValueError("private development predictions must be sorted by opaque ID")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-private-development-predictions-v1",
        "row_count": expected_rows,
        "decoder_target_threshold": threshold,
        "rows": clean_rows,
    }


def score_private_development_thresholds(
    prediction_payload: Any,
    *,
    development_proxy_path: Path,
    expected_composite: float,
) -> dict[str, Any]:
    """Score one private 222-row development payload over the frozen threshold grid.

    The Parquet read is filtered to the development split and requests only opaque
    IDs and labels.  Locked-test text and labels are never materialised.  The
    returned value is metadata-only and contains no row-level predictions.
    """

    from reddit_china_stance.modernbert_trainer import (
        aggregate_semantic_metrics,
        decode_predictions,
    )

    _, pq = _require_pyarrow()
    if not development_proxy_path.is_file():
        raise FileNotFoundError(development_proxy_path)
    parquet = pq.ParquetFile(development_proxy_path)
    expected_columns = {
        "source_sample_id",
        "target_text",
        "parent_context",
        "submission_context",
        "split",
        "resolution",
        "label_json",
    }
    if parquet.metadata.num_rows != 452 or set(parquet.schema_arrow.names) != expected_columns:
        raise ModernBertContractError("development proxy Parquet schema or row count drifted")
    metadata = parquet.schema_arrow.metadata or {}
    expected_metadata = {
        b"proxy_id": DEVELOPMENT_PROXY_ID.encode(),
        b"proxy_json_sha256": PROXY_JSON_SHA256.encode(),
        b"reference_rows_json_sha256": REFERENCE_ROWS_JSON_SHA256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ModernBertContractError("development proxy Parquet source binding drifted")
    development = pq.read_table(
        development_proxy_path,
        columns=["source_sample_id", "label_json"],
        filters=[("split", "=", "development")],
    ).to_pylist()
    if len(development) != 222:
        raise ModernBertContractError("threshold scoring requires exactly 222 development rows")

    reference: dict[str, dict[str, Any]] = {}
    for row in development:
        item_id = row.get("source_sample_id")
        if not isinstance(item_id, str) or not item_id or item_id in reference:
            raise ValueError("development threshold reference has invalid or duplicate IDs")
        try:
            label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("development threshold reference contains invalid label JSON") from exc
        if not isinstance(label, dict):
            raise ValueError("development threshold reference labels must be objects")
        reference[item_id] = label

    clean = validate_private_development_predictions(prediction_payload, expected_rows=222)
    if clean["decoder_target_threshold"] != 0.50:
        raise ModernBertContractError("stored development predictions must bind threshold 0.50")
    prediction_rows = clean["rows"]
    item_ids = [row["source_sample_id"] for row in prediction_rows]
    if set(item_ids) != set(reference):
        raise ModernBertContractError("development prediction and reference IDs differ")
    relevance_logits = [row["relevance_logits"] for row in prediction_rows]
    target_logits = [row["target_logits"] for row in prediction_rows]
    stance_logits = [row["stance_logits"] for row in prediction_rows]
    stored = tuple(row["decoded_label"] for row in prediction_rows)
    recomputed = decode_predictions(
        relevance_logits,
        target_logits,
        stance_logits,
        target_threshold=0.50,
    )
    if recomputed.labels != stored:
        raise ModernBertContractError("stored development labels differ from their bound logits")

    if (
        isinstance(expected_composite, bool)
        or not isinstance(expected_composite, (int, float))
        or not math.isfinite(expected_composite)
        or not 0 <= expected_composite <= 1
    ):
        raise ValueError("expected_composite must be a finite probability")
    threshold_composites: dict[str, float] = {}
    for threshold in TARGET_THRESHOLDS:
        metrics = aggregate_semantic_metrics(
            reference,
            item_ids=item_ids,
            relevance_logits=relevance_logits,
            target_logits=target_logits,
            stance_logits=stance_logits,
            target_threshold=threshold,
        )
        if metrics["invalid_outputs"] != 0:
            raise ModernBertContractError("threshold scoring produced invalid outputs")
        threshold_composites[f"{threshold:.2f}"] = _composite(metrics)
    if threshold_composites["0.50"] != float(expected_composite):
        raise ModernBertContractError(
            "threshold-0.50 composite differs from the selected trial receipt"
        )

    reference_binding = {
        "development_proxy_id": DEVELOPMENT_PROXY_ID,
        "split": "development",
        "labels": [
            {"source_sample_id": item_id, "label": reference[item_id]}
            for item_id in sorted(reference)
        ],
    }
    result = {
        "development_rows": len(reference),
        "development_reference_sha256": canonical_sha256(reference_binding),
        "threshold_composites": threshold_composites,
    }
    assert_metadata_only(result, where="ModernBERT threshold score result")
    return result


def _artifact_inventory(
    root: Path,
    *,
    checkpoint_relative_path: str,
    require_development_predictions: bool,
) -> dict[str, dict[str, Any]]:
    relative = Path(checkpoint_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModernBertContractError("selected checkpoint artifact path is unsafe")
    checkpoint_path = root / relative
    if not checkpoint_path.is_file():
        raise ModernBertContractError("selected checkpoint artifact is missing")
    checkpoint_descriptor = {
        "relative_path": checkpoint_relative_path,
        "sha256": file_sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    prediction_path = root / "development-predictions.json"
    prediction_descriptor: dict[str, Any] | None = None
    if require_development_predictions:
        if not prediction_path.is_file():
            raise ModernBertContractError("development prediction artifact is missing")
        prediction_descriptor = {
            "relative_path": prediction_path.name,
            "sha256": file_sha256(prediction_path),
            "bytes": prediction_path.stat().st_size,
        }
    elif prediction_path.exists():
        raise ModernBertContractError("preflight unexpectedly published development predictions")
    artifacts: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(sorted(item for item in root.rglob("*") if item.is_file())):
        if path.name in {"receipt.json", "resume.json"} or path.name.endswith(".new"):
            continue
        if path == checkpoint_path:
            continue
        if path == prediction_path:
            continue
        artifacts[f"artifact_{index:03d}"] = {
            "relative_path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    if not artifacts:
        raise ModernBertContractError("trial produced no supporting immutable artifacts")
    result = {"checkpoint": checkpoint_descriptor, **artifacts}
    if prediction_descriptor is not None:
        result["development_predictions"] = prediction_descriptor
    return result


def _adapter_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "complete",
        "experiment_run_id": receipt["experiment_run_id"],
        "trial_id": receipt["trial_id"],
        "gpu_type": receipt["compute"]["gpu_type"],
        "wall_seconds": receipt["compute"]["wall_seconds"],
        "gpu_seconds": receipt["compute"]["gpu_seconds"],
        "estimated_cost_usd": receipt["compute"]["estimated_cost_usd"],
        "artifact_count": len(receipt["artifacts"]),
        "receipt_id": receipt["receipt_id"],
    }


def validate_published_trial_artifacts(
    trial_root: Path, receipt: Mapping[str, Any], *, require_development_predictions: bool
) -> dict[str, Any]:
    """Validate every private artifact without returning its contents."""

    assert_metadata_only(receipt, where="ModernBERT public trial receipt")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or "checkpoint" not in artifacts:
        raise ModernBertContractError("trial receipt lacks its checkpoint descriptor")
    if require_development_predictions != ("development_predictions" in artifacts):
        raise ModernBertContractError("development prediction descriptor presence drifted")
    validated = {}
    for name, descriptor in artifacts.items():
        if (
            not isinstance(name, str)
            or not isinstance(descriptor, Mapping)
            or set(descriptor)
            != {
                "relative_path",
                "sha256",
                "bytes",
            }
        ):
            raise ValueError("trial artifact descriptor schema drifted")
        relative = Path(str(descriptor["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("trial artifact descriptor path is unsafe")
        path = trial_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["bytes"]
            or file_sha256(path) != descriptor["sha256"]
        ):
            raise ModernBertContractError(f"trial artifact is missing or corrupt: {name}")
        validated[name] = dict(descriptor)
    if require_development_predictions:
        descriptor = validated["development_predictions"]
        payload = _read_json_object(
            trial_root / descriptor["relative_path"],
            where="private development predictions",
        )
        validate_private_development_predictions(payload, expected_rows=payload.get("row_count"))
    return {"status": "validated", "artifact_count": len(validated)}


def _run_modal_trial(
    *,
    job: Mapping[str, Any],
    volume_root: Path,
    resume: bool,
    preflight: bool,
) -> dict[str, Any]:
    clean_job = _validate_modal_job(job, expected_phase="preflight" if preflight else None)
    spec = clean_job["trial_spec"]
    if not preflight and spec["phase"] == "preflight":
        raise ValueError("training adapter cannot execute a preflight trial")
    run_root = volume_root / OUTPUT_PREFIX / f"run={clean_job['experiment_run_id']}"
    final_root = run_root / f"phase={spec['phase']}" / f"trial={spec['trial_id']}"
    staging_root = run_root / ".incomplete" / f"trial={spec['trial_id']}"
    if final_root.exists():
        if staging_root.exists():
            raise ModernBertContractError("trial has both final and incomplete outputs")
        receipt = _read_json_object(final_root / "receipt.json", where="trial receipt")
        validate_published_trial_artifacts(
            final_root,
            receipt,
            require_development_predictions=not preflight,
        )
        return _adapter_summary(receipt)
    if resume:
        if not (staging_root / "resume.json").is_file():
            raise ModernBertContractError("resume requested without an exact resume marker")
    else:
        if staging_root.exists():
            raise ModernBertContractError("incomplete trial requires explicit resume")
        staging_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_runtime_inputs(
        clean_job, volume_root=volume_root, include_development=not preflight
    )
    result = _execute_trial_runtime(
        job=clean_job,
        inputs=inputs,
        staging_root=staging_root,
        resume=resume,
        preflight=preflight,
    )
    selected_checkpoint = result.get("selected_checkpoint_relative_path")
    if not isinstance(selected_checkpoint, str) or not selected_checkpoint:
        raise ModernBertContractError("trial runtime did not identify its selected checkpoint")
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
    publish_immutable_json(staging_root / "metrics.json", private_metrics)
    development_predictions = result.get("development_predictions")
    if preflight:
        if development_predictions is not None:
            raise ModernBertContractError(
                "preflight runtime unexpectedly returned development predictions"
            )
    else:
        clean_predictions = validate_private_development_predictions(
            development_predictions,
            expected_rows=int(result["development_rows"]),
        )
        publish_immutable_json(staging_root / "development-predictions.json", clean_predictions)
    public_metrics = _public_trial_metrics(private_metrics, preflight=preflight)
    artifacts = _artifact_inventory(
        staging_root,
        checkpoint_relative_path=selected_checkpoint,
        require_development_predictions=not preflight,
    )
    wall_seconds = _require_nonnegative_number(result["wall_seconds"], where="wall_seconds")
    gpu_seconds = _require_nonnegative_number(result["gpu_seconds"], where="gpu_seconds")
    gpu_type = spec["gpu_type"]
    estimated_cost = GPU_RATE_USD_PER_SECOND[gpu_type] * Decimal(str(gpu_seconds))
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment_run_id": clean_job["experiment_run_id"],
        "trial_id": spec["trial_id"],
        "trial_spec_sha256": clean_job["trial_spec_sha256"],
        "bindings": deepcopy(clean_job["experiment_contract"]["bindings"]),
        "artifacts": artifacts,
        "aggregate_metrics": public_metrics,
        "compute": {
            "gpu_type": gpu_type,
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "estimated_cost_usd": format(estimated_cost.quantize(Decimal("0.000001")), "f"),
        },
    }
    assert_metadata_only(body, where="ModernBERT trial receipt")
    receipt = {**body, "receipt_id": canonical_sha256(body)}
    publish_immutable_json(staging_root / "receipt.json", receipt)
    (staging_root / "resume.json").unlink(missing_ok=True)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_root, final_root)
    validate_published_trial_artifacts(
        final_root,
        receipt,
        require_development_predictions=not preflight,
    )
    return _adapter_summary(receipt)


def run_preflight_trial(
    *, job: Mapping[str, Any], volume_root: Path, resume: bool = False
) -> dict[str, Any]:
    """Execute and immutably publish one exact 200-update hardware preflight."""

    return _run_modal_trial(job=job, volume_root=volume_root, resume=resume, preflight=True)


def run_training_trial(
    *, job: Mapping[str, Any], volume_root: Path, resume: bool = False
) -> dict[str, Any]:
    """Execute and immutably publish one exact registered training trial."""

    return _run_modal_trial(job=job, volume_root=volume_root, resume=resume, preflight=False)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def publication_state(path: Path, expected: Mapping[str, Any]) -> str:
    """Return ``missing``, ``resume_exact``, or ``complete``; reject drift."""

    payload = _canonical_json_bytes(expected)
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    if path.exists():
        if path.read_bytes() != payload:
            raise ModernBertContractError(f"immutable publication differs: {path}")
        if incomplete.exists():
            raise ModernBertContractError(f"stale incomplete publication remains: {incomplete}")
        return "complete"
    if incomplete.exists():
        if incomplete.read_bytes() != payload:
            raise ModernBertContractError(
                f"incomplete publication cannot resume exactly: {incomplete}"
            )
        return "resume_exact"
    return "missing"


def publish_immutable_json(
    path: Path, value: Mapping[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    """Atomically publish exact JSON, optionally completing an exact `.incomplete`."""

    payload = _canonical_json_bytes(value)
    state = publication_state(path, value)
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    if state == "resume_exact":
        if not resume:
            raise ModernBertContractError("exact incomplete publication requires explicit resume")
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
        raise ModernBertContractError("immutable publication is not complete")
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


__all__ = [
    "ASHA_RUNG_PLAN",
    "DATASET_ID",
    "DATASET_REVISION",
    "DATASET_SHA256",
    "DEVELOPMENT_PROXY_ID",
    "HARD_COST_CAP_USD",
    "LABEL_BUDGETS",
    "MODEL_ID",
    "MODEL_REVISION",
    "REGISTERED_LADDERS",
    "STABILITY_SEEDS",
    "TEACHER_GENERATION_RUN_ID",
    "TOKENIZER_REVISION",
    "ModernBertContractError",
    "budget_status",
    "build_asha_plan",
    "build_checkpoint_payload",
    "build_trial_receipt",
    "canonical_sha256",
    "freeze_experiment_contract",
    "freeze_trial_spec",
    "frozen_trial_configs",
    "issue_locked_test_authorisation",
    "publication_state",
    "publish_immutable_json",
    "run_preflight_trial",
    "run_training_trial",
    "score_private_development_thresholds",
    "select_asha_promotions",
    "select_checkpoint",
    "validate_asha_plan",
    "validate_checkpoint_payload",
    "validate_experiment_contract",
    "validate_immutable_publication",
    "validate_private_development_predictions",
    "validate_published_trial_artifacts",
    "validate_trial_receipt",
    "validate_trial_spec",
]
