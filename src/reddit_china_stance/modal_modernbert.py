"""Modal orchestration for the frozen ModernBERT three-head experiment.

This module is deliberately orchestration-only.  Importing it constructs Modal
definitions, but does not read private data, contact Modal, or start a GPU.  The
training implementation is imported lazily inside a remote container.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import modal

from reddit_china_stance.modernbert_training import (
    LABEL_BUDGETS,
    REGISTERED_LADDERS,
    STABILITY_SEEDS,
    TARGET_THRESHOLDS,
    build_asha_plan,
    select_asha_promotions,
    select_checkpoint,
    validate_asha_plan,
)
from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-modernbert-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path("student-modernbert-v1")
SCHEMA_VERSION = "1.0.0"

MODEL_ID = "answerdotai/ModernBERT-large"
MODEL_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
DATASET_ID = "aisafteycommons/reddit-china-stance-10k-sol-v1"
DATASET_REVISION = "b0fa1c3e5e3caf4dc3fb43b9024858364bc44834"
DATASET_PARQUET_SHA256 = "f81146988ef504f81fb77b36333084e5ea10ab77908568d5be26bdba296a5bba"
DEVELOPMENT_PROXY_SHA256 = "4ba40fcd0eb33dc2c7ca1c0b9ac75b06f42dc120bed470f9da7f713194a3a66d"

MAX_CONCURRENT_TRIALS = 8
ACCOUNT_GPU_LIMIT = 10
PREFLIGHT_UPDATES = 200
MAX_LENGTH = 768
EFFECTIVE_BATCH_SIZE = 32
HARD_MAX_APPROVAL_USD = Decimal("200")
PLANNED_UPPER_COST_USD = Decimal("155")
PLANNED_CONTINGENCY_USD = HARD_MAX_APPROVAL_USD - PLANNED_UPPER_COST_USD
CONFIRMATORY_PHASE_UPPER_COST_USD = Decimal("65")
ALLOWED_GPUS = ("L4",)
TRAIN_PHASES = ("controls", "sweep", "stability", "confirmatory", "chronological")
TRAIN_PHASE_COUNTS = {
    "controls": 2,
    "sweep": 12,
    "stability": 4,
    "confirmatory": 9,
    "chronological": 1,
}

ENCODER_LEARNING_RATES = ("0.00001", "0.00003", "0.00005")
LOSS_WEIGHT_OPTIONS = (
    {"relevance": "1.0", "targets": "1.0", "stance": "1.0"},
    {"relevance": "0.5", "targets": "1.0", "stance": "1.5"},
)
CLASS_WEIGHT_OPTIONS = ("none", "capped_inverse_sqrt")
ASHA_RUNGS = (1, 3, 6)
ASHA_PROMOTIONS = (12, 4, 2)

RUNTIME_DEPENDENCIES = {
    "accelerate": "1.10.1",
    "huggingface-hub": "0.36.2",
    "iterative-stratification": "0.1.9",
    "pyarrow": "25.0.1",
    "pydantic": "2.13.4",
    "safetensors": "0.8.0",
    "scikit-learn": "1.9.0",
    "torch": "2.8.0",
    "transformers": "4.57.6",
}


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*(f"{name}=={version}" for name, version in RUNTIME_DEPENDENCIES.items()))
    .run_commands(
        'python -c "from huggingface_hub import snapshot_download; '
        "snapshot_download(repo_id='answerdotai/ModernBERT-large', "
        "revision='45bb4654a4d5aaff24dd11d4781fa46d39bf8c13')\""
    )
    .env(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_python_source("reddit_china_stance")
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        parsed = int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    if value != f"{parsed:064x}":
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def enforce_cost_guardrail(*, estimated_cost_usd: Decimal, approved_cost_usd: Decimal) -> None:
    """Fail before cloud submission if approval is invalid or insufficient."""

    if approved_cost_usd <= 0 or approved_cost_usd > HARD_MAX_APPROVAL_USD:
        raise ValueError(f"approved cost must be > 0 and <= {HARD_MAX_APPROVAL_USD}")
    if estimated_cost_usd < 0:
        raise ValueError("estimated cost must be non-negative")
    if estimated_cost_usd > approved_cost_usd:
        raise RuntimeError(
            f"estimated ModernBERT cost ${estimated_cost_usd} exceeds approved ${approved_cost_usd}"
        )


def make_experiment_contract(
    *,
    split_manifest_sha256: str,
    trainer_code_sha256: str,
    dependency_lock_sha256: str,
    approved_cost_usd: Decimal,
) -> dict[str, Any]:
    """Bind the immutable inputs and the approved, non-broadenable experiment family."""

    enforce_cost_guardrail(
        estimated_cost_usd=PLANNED_UPPER_COST_USD,
        approved_cost_usd=approved_cost_usd,
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-three-head-experiment-v1",
        "bindings": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_parquet_sha256": DATASET_PARQUET_SHA256,
            "development_proxy_sha256": DEVELOPMENT_PROXY_SHA256,
            "split_manifest_sha256": _require_sha256(
                "split_manifest_sha256", split_manifest_sha256
            ),
            "trainer_code_sha256": _require_sha256("trainer_code_sha256", trainer_code_sha256),
            "dependency_lock_sha256": _require_sha256(
                "dependency_lock_sha256", dependency_lock_sha256
            ),
        },
        "architecture": {
            "encoder": "ModernBertModel",
            "pooling": "masked_mean",
            "attention_implementation": "sdpa",
            "reference_compile": False,
            "max_length": MAX_LENGTH,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "heads": {
                "relevance_classes": 3,
                "target_labels": 4,
                "stance_shape": [4, 5],
            },
        },
        "sweep": {
            "encoder_learning_rates": list(ENCODER_LEARNING_RATES),
            "loss_weight_options": list(LOSS_WEIGHT_OPTIONS),
            "class_weight_options": list(CLASS_WEIGHT_OPTIONS),
            "asha_rungs": list(ASHA_RUNGS),
            "asha_promotions": list(ASHA_PROMOTIONS),
            "asha_plan_id": build_asha_plan()["plan_id"],
            "grid_size": 12,
            "grid_broadening_allowed": False,
        },
        "compute": {
            "allowed_gpus": list(ALLOWED_GPUS),
            "account_gpu_limit": ACCOUNT_GPU_LIMIT,
            "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
            "preflight_updates": PREFLIGHT_UPDATES,
            "planned_upper_cost_usd": str(PLANNED_UPPER_COST_USD),
            "contingency_usd": str(PLANNED_CONTINGENCY_USD),
            "hard_max_approved_cost_usd": str(HARD_MAX_APPROVAL_USD),
            "approved_cost_usd": str(approved_cost_usd),
            "gpu_fallback_allowed": False,
        },
    }
    return contract


def experiment_run_id(contract: Mapping[str, Any]) -> str:
    validate_experiment_contract(contract)
    return _canonical_sha256(contract)


def validate_experiment_contract(contract: Mapping[str, Any]) -> None:
    """Reject drift from the frozen dataset, model, grid or compute boundary."""

    bindings = contract.get("bindings")
    compute = contract.get("compute")
    sweep = contract.get("sweep")
    architecture = contract.get("architecture")
    if not all(isinstance(value, Mapping) for value in (bindings, compute, sweep, architecture)):
        raise ValueError("experiment contract is incomplete")
    expected_bindings = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_parquet_sha256": DATASET_PARQUET_SHA256,
        "development_proxy_sha256": DEVELOPMENT_PROXY_SHA256,
    }
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise ValueError(f"experiment binding drifted: {key}")
    for key in ("split_manifest_sha256", "trainer_code_sha256", "dependency_lock_sha256"):
        _require_sha256(key, bindings.get(key))
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") != (
        "modernbert-three-head-experiment-v1"
    ):
        raise ValueError("experiment schema or kind drifted")
    if (
        architecture.get("max_length") != MAX_LENGTH
        or architecture.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE
        or architecture.get("attention_implementation") != "sdpa"
        or architecture.get("reference_compile") is not False
    ):
        raise ValueError("frozen architecture drifted")
    expected_sweep = {
        "encoder_learning_rates": list(ENCODER_LEARNING_RATES),
        "loss_weight_options": list(LOSS_WEIGHT_OPTIONS),
        "class_weight_options": list(CLASS_WEIGHT_OPTIONS),
        "asha_rungs": list(ASHA_RUNGS),
        "asha_promotions": list(ASHA_PROMOTIONS),
        "asha_plan_id": build_asha_plan()["plan_id"],
        "grid_size": 12,
        "grid_broadening_allowed": False,
    }
    if dict(sweep) != expected_sweep:
        raise ValueError("frozen 12-configuration sweep drifted")
    if compute.get("max_concurrent_trials") != MAX_CONCURRENT_TRIALS:
        raise ValueError("max concurrent trial count drifted")
    if compute.get("gpu_fallback_allowed") is not False:
        raise ValueError("GPU fallback must remain disabled")
    try:
        approved = Decimal(str(compute.get("approved_cost_usd")))
    except InvalidOperation as error:
        raise ValueError("approved cost is invalid") from error
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(str(compute.get("planned_upper_cost_usd"))),
        approved_cost_usd=approved,
    )


def _trial_spec(
    *,
    phase: str,
    name: str,
    gpu_type: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if gpu_type not in ALLOWED_GPUS:
        raise ValueError(f"gpu_type must be one of {ALLOWED_GPUS}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "name": name,
        "gpu_type": gpu_type,
        "config": dict(config),
    }
    return {**payload, "trial_id": _canonical_sha256(payload)}


def make_preflight_trials(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact 200-update L4 compatibility and throughput trial."""

    validate_experiment_contract(contract)
    base = {
        "budget": "train_5k_101",
        "seed": 7,
        "updates": PREFLIGHT_UPDATES,
        "attention_implementation": "sdpa",
        "encoder_learning_rate": "0.00003",
        "loss_weights": LOSS_WEIGHT_OPTIONS[0],
        "class_weights": "none",
        "semantic_selection_allowed": False,
    }
    return [_trial_spec(phase="preflight", name="preflight-l4", gpu_type="L4", config=base)]


def make_sweep_trials(
    contract: Mapping[str, Any],
    *,
    gpu_type: Literal["L4"],
    rung_epochs: int = 1,
    promotion_receipt: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one ASHA rung, never twelve independent six-epoch trials."""

    validate_experiment_contract(contract)
    plan = validate_asha_plan(build_asha_plan())
    rung = next((row for row in plan["rungs"] if row["epochs"] == rung_epochs), None)
    if rung is None:
        raise ValueError(f"rung_epochs must be one of {ASHA_RUNGS}")
    by_id = {config["config_sha256"]: config for config in plan["configs"]}
    continuation_by_config: dict[str, Mapping[str, Any]] = {}
    if rung_epochs == 1:
        if promotion_receipt is not None:
            raise ValueError("the first ASHA rung cannot have a promotion receipt")
        candidate_ids = [config["config_sha256"] for config in plan["configs"]]
    else:
        clean_promotion = validate_asha_promotion_receipt(
            contract,
            promotion_receipt,
            expected_to_rung_epochs=rung_epochs,
        )
        candidate_ids = [row["config_sha256"] for row in clean_promotion["promoted"]]
        continuation_by_config = {row["config_sha256"]: row for row in clean_promotion["promoted"]}
    if len(candidate_ids) != rung["candidate_count"] or not set(candidate_ids) <= set(by_id):
        raise ValueError("ASHA rung candidate inventory drifted")
    specs = [
        _trial_spec(
            phase="sweep",
            name=f"asha-e{rung_epochs}-{config_id[:12]}",
            gpu_type=gpu_type,
            config={
                "registered_config": by_id[config_id],
                "target_epochs": rung_epochs,
                "attention_implementation": "sdpa",
                "continuation": continuation_by_config.get(config_id),
            },
        )
        for config_id in candidate_ids
    ]
    if len({spec["trial_id"] for spec in specs}) != rung["candidate_count"]:
        raise AssertionError("ASHA rung trial IDs are not unique")
    return specs


def validate_asha_promotion_receipt(
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
    *,
    expected_to_rung_epochs: int,
) -> dict[str, Any]:
    """Validate the immutable ranking bridge between two ASHA rungs."""

    validate_experiment_contract(contract)
    if not isinstance(receipt, Mapping):
        raise ValueError("a prior-rung promotion receipt is required")
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "asha_plan_id",
        "source_phase_run_id",
        "from_rung_epochs",
        "to_rung_epochs",
        "candidate_config_ids",
        "results_sha256",
        "promoted",
        "promotion_receipt_id",
    }
    if set(receipt) != required:
        raise ValueError("ASHA promotion receipt has unexpected fields")
    body = {key: receipt[key] for key in required - {"promotion_receipt_id"}}
    if receipt["promotion_receipt_id"] != _canonical_sha256(body):
        raise ValueError("ASHA promotion receipt content address drifted")
    expected_from = {3: 1, 6: 3}.get(expected_to_rung_epochs)
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["kind"] != "modernbert-asha-promotion-v1"
        or receipt["experiment_run_id"] != experiment_run_id(contract)
        or receipt["asha_plan_id"] != build_asha_plan()["plan_id"]
        or receipt["from_rung_epochs"] != expected_from
        or receipt["to_rung_epochs"] != expected_to_rung_epochs
    ):
        raise ValueError("ASHA promotion receipt binding drifted")
    expected_count = {3: 4, 6: 2}[expected_to_rung_epochs]
    promoted = receipt["promoted"]
    if not isinstance(promoted, list) or len(promoted) != expected_count:
        raise ValueError("ASHA promotion receipt has the wrong promotion count")
    config_ids = [row.get("config_sha256") for row in promoted if isinstance(row, Mapping)]
    if len(config_ids) != expected_count or len(set(config_ids)) != expected_count:
        raise ValueError("ASHA promotion receipt has duplicate or invalid configurations")
    registered = {row["config_sha256"] for row in build_asha_plan()["configs"]}
    if not set(config_ids) <= registered:
        raise ValueError("ASHA promotion receipt contains an unregistered configuration")
    for row in promoted:
        if set(row) != {
            "config_sha256",
            "source_trial_id",
            "checkpoint_relative_path",
            "checkpoint_sha256",
        }:
            raise ValueError("ASHA promoted checkpoint binding drifted")
        _require_sha256("source_trial_id", row["source_trial_id"])
        _require_sha256("checkpoint_sha256", row["checkpoint_sha256"])
        _safe_artifact_path(Path("/"), row["checkpoint_relative_path"])
    return json.loads(json.dumps(receipt, sort_keys=True))


def validate_stability_source_receipt(
    contract: Mapping[str, Any], receipt: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Validate the two final-ASHA configurations authorised for stability."""

    validate_experiment_contract(contract)
    if not isinstance(receipt, Mapping):
        raise ValueError("stability requires a final-ASHA source receipt")
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "asha_plan_id",
        "source_phase_run_id",
        "candidate_configs",
        "source_receipt_ids",
        "stability_source_receipt_id",
    }
    if set(receipt) != required:
        raise ValueError("stability source receipt has unexpected fields")
    body = {key: receipt[key] for key in required - {"stability_source_receipt_id"}}
    if receipt["stability_source_receipt_id"] != _canonical_sha256(body):
        raise ValueError("stability source receipt content address drifted")
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["kind"] != "modernbert-stability-source-v1"
        or receipt["experiment_run_id"] != experiment_run_id(contract)
        or receipt["asha_plan_id"] != build_asha_plan()["plan_id"]
    ):
        raise ValueError("stability source receipt binding drifted")
    configs = receipt["candidate_configs"]
    if not isinstance(configs, list) or len(configs) != 2:
        raise ValueError("stability source must contain exactly two configurations")
    ids = [row.get("config_sha256") for row in configs if isinstance(row, Mapping)]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("stability source configuration IDs are invalid")
    registered = {row["config_sha256"]: row for row in build_asha_plan()["configs"]}
    if any(
        registered.get(config_id) != config for config_id, config in zip(ids, configs, strict=True)
    ):
        raise ValueError("stability source contains an unregistered or changed configuration")
    source_receipts = receipt["source_receipt_ids"]
    if not isinstance(source_receipts, list) or len(source_receipts) != 2:
        raise ValueError("stability source must bind two final-rung trial receipts")
    for value in source_receipts:
        _require_sha256("source_receipt_id", value)
    return json.loads(json.dumps(receipt, sort_keys=True))


def _phase3_semantic_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return every experiment field that Phase 3 is forbidden to change."""

    validate_experiment_contract(contract)
    bindings = dict(contract["bindings"])
    bindings.pop("trainer_code_sha256")
    return {
        "schema_version": contract["schema_version"],
        "kind": contract["kind"],
        "bindings_except_trainer_code": bindings,
        "architecture": contract["architecture"],
        "sweep": contract["sweep"],
        "compute": contract["compute"],
    }


def validate_phase3_transition_receipt(
    destination_contract: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    transition_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the one permitted post-ASHA code-only provenance transition."""

    validate_experiment_contract(destination_contract)
    if not isinstance(transition_receipt, Mapping):
        raise ValueError("cross-contract stability requires a Phase 3 transition receipt")
    required = {
        "schema_version",
        "kind",
        "source_experiment_contract",
        "destination_experiment_contract",
        "source_experiment_run_id",
        "destination_experiment_run_id",
        "source_trainer_code_sha256",
        "destination_trainer_code_sha256",
        "stability_source_receipt_id",
        "final_asha_phase_run_id",
        "final_asha_trial_receipt_ids",
        "code_delta_scope",
        "model_semantics_changed",
        "training_semantics_changed",
        "authorisation",
        "transition_receipt_id",
    }
    if set(transition_receipt) != required:
        raise ValueError("Phase 3 transition receipt has unexpected fields")
    body = {key: transition_receipt[key] for key in required - {"transition_receipt_id"}}
    if transition_receipt["transition_receipt_id"] != _canonical_sha256(body):
        raise ValueError("Phase 3 transition receipt content address drifted")
    source_contract = transition_receipt["source_experiment_contract"]
    target_contract = transition_receipt["destination_experiment_contract"]
    if not isinstance(source_contract, Mapping) or not isinstance(target_contract, Mapping):
        raise ValueError("Phase 3 transition contracts must be objects")
    validate_experiment_contract(source_contract)
    validate_experiment_contract(target_contract)
    source = validate_stability_source_receipt(source_contract, source_receipt)
    if dict(target_contract) != dict(destination_contract):
        raise ValueError("Phase 3 transition destination contract drifted")
    source_run_id = experiment_run_id(source_contract)
    destination_run_id = experiment_run_id(destination_contract)
    if (
        transition_receipt["schema_version"] != SCHEMA_VERSION
        or transition_receipt["kind"] != "modernbert-phase3-code-transition-v1"
        or transition_receipt["source_experiment_run_id"] != source_run_id
        or transition_receipt["destination_experiment_run_id"] != destination_run_id
        or transition_receipt["stability_source_receipt_id"]
        != source["stability_source_receipt_id"]
        or transition_receipt["final_asha_phase_run_id"] != source["source_phase_run_id"]
        or transition_receipt["final_asha_trial_receipt_ids"] != source["source_receipt_ids"]
    ):
        raise ValueError("Phase 3 transition source or destination binding drifted")
    source_code = source_contract["bindings"]["trainer_code_sha256"]
    destination_code = destination_contract["bindings"]["trainer_code_sha256"]
    if (
        source_code == destination_code
        or transition_receipt["source_trainer_code_sha256"] != source_code
        or transition_receipt["destination_trainer_code_sha256"] != destination_code
    ):
        raise ValueError("Phase 3 transition must bind two distinct trainer source digests")
    if _phase3_semantic_contract(source_contract) != _phase3_semantic_contract(
        destination_contract
    ):
        raise ValueError("Phase 3 transition changed a frozen semantic experiment binding")
    allowed_scopes = {
        (
            "private_development_prediction_publication",
            "phase3_stability_orchestration",
        ): False,
        (
            "private_development_prediction_publication",
            "phase3_stability_orchestration",
            "stability_registered_config_adapter_fix",
        ): True,
    }
    scope = tuple(transition_receipt["code_delta_scope"])
    if (
        scope not in allowed_scopes
        or transition_receipt["model_semantics_changed"] is not False
        or transition_receipt["training_semantics_changed"] is not allowed_scopes.get(scope)
    ):
        raise ValueError("Phase 3 transition code-delta claim drifted")
    authorisation = transition_receipt["authorisation"]
    if not isinstance(authorisation, Mapping) or authorisation != {
        "phase": "stability",
        "fresh_training": True,
        "trial_count": 4,
        "candidate_config_sha256": [row["config_sha256"] for row in source["candidate_configs"]],
        "optimiser_seeds": list(STABILITY_SEEDS),
        "label_budget": "train_5k_101",
        "target_epochs": 6,
        "confirmatory_authorised": False,
        "locked_test_authorised": False,
    }:
        raise ValueError("Phase 3 transition authorisation broadened or drifted")
    return json.loads(json.dumps(transition_receipt, sort_keys=True))


def make_stability_trials(
    contract: Mapping[str, Any],
    *,
    gpu_type: Literal["L4"],
    source_receipt: Mapping[str, Any],
    transition_receipt: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create top-two-config x two-seed fresh six-epoch stability trials."""

    if source_receipt.get("experiment_run_id") == experiment_run_id(contract):
        if transition_receipt is not None:
            raise ValueError("same-contract stability must not use a transition receipt")
        source = validate_stability_source_receipt(contract, source_receipt)
        transition_id = None
    else:
        transition = validate_phase3_transition_receipt(
            contract, source_receipt, transition_receipt
        )
        source = validate_stability_source_receipt(
            transition["source_experiment_contract"], source_receipt
        )
        transition_id = transition["transition_receipt_id"]
    trials = [
        _trial_spec(
            phase="stability",
            name=f"stability-{config['config_sha256'][:12]}-seed-{seed}",
            gpu_type=gpu_type,
            config={
                "registered_config": config,
                "budget": "train_5k_101",
                "seed": seed,
                "target_epochs": 6,
                "attention_implementation": "sdpa",
                "continuation": None,
                "fresh_training": True,
                "stability_source_receipt_id": source["stability_source_receipt_id"],
                "phase3_transition_receipt_id": transition_id,
            },
        )
        for config in source["candidate_configs"]
        for seed in STABILITY_SEEDS
    ]
    if len(trials) != 4 or len({trial["trial_id"] for trial in trials}) != 4:
        raise AssertionError("stability must contain four unique fresh trials")
    return trials


def build_confirmatory_split_binding(
    contract: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    split_manifest_file_sha256: str,
) -> dict[str, Any]:
    """Reduce the validated private split manifest to metadata needed for Phase 4."""

    from reddit_china_stance.modernbert_splits import validate_split_manifest

    validate_experiment_contract(contract)
    if split_manifest_file_sha256 != contract["bindings"]["split_manifest_sha256"]:
        raise ValueError("confirmatory split-manifest file binding drifted")
    clean = validate_split_manifest(split_manifest)
    ladders = []
    for registered in REGISTERED_LADDERS:
        ladder_seed = registered["ladder_seed"]
        optimiser_seed = registered["optimiser_seed"]
        source = clean["ladders"][str(ladder_seed)]
        if source["acquisition_seed"] != ladder_seed or source["optimiser_seed"] != optimiser_seed:
            raise ValueError("confirmatory ladder/optimiser pairing drifted")
        budgets = []
        for label_budget in LABEL_BUDGETS:
            short = f"{label_budget // 1000}k"
            metadata = source["budgets"][short]
            budgets.append(
                {
                    "label_budget": label_budget,
                    "budget_name": f"train_{short}_{ladder_seed}",
                    "training_row_count": metadata["row_count"],
                    "sample_ids_sha256": metadata["sample_ids_sha256"],
                }
            )
        ladders.append(
            {
                "ladder_seed": ladder_seed,
                "optimiser_seed": optimiser_seed,
                "budgets": budgets,
            }
        )
    binding = {
        "split_manifest_file_sha256": split_manifest_file_sha256,
        "split_manifest_id": clean["manifest_id"],
        "row_count": clean["contract"]["row_count"],
        "ladders": ladders,
    }
    return validate_confirmatory_split_binding(contract, binding)


def validate_confirmatory_split_binding(
    contract: Mapping[str, Any], binding: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Validate the exact three paired acquisition ladders without exposing row IDs."""

    validate_experiment_contract(contract)
    if not isinstance(binding, Mapping) or set(binding) != {
        "split_manifest_file_sha256",
        "split_manifest_id",
        "row_count",
        "ladders",
    }:
        raise ValueError("confirmatory split binding has unexpected fields")
    if binding["split_manifest_file_sha256"] != contract["bindings"]["split_manifest_sha256"]:
        raise ValueError("confirmatory split binding belongs to another experiment")
    _require_sha256("split_manifest_id", binding["split_manifest_id"])
    if binding["row_count"] != 10_000:
        raise ValueError("confirmatory split binding must conserve 10,000 teacher rows")
    ladders = binding["ladders"]
    if not isinstance(ladders, list) or len(ladders) != len(REGISTERED_LADDERS):
        raise ValueError("confirmatory split binding requires three registered ladders")
    expected_fold_sizes = {
        101: (1001, 1000, 1000, 999, 1000, 999, 1001, 1000, 999, 1001),
        202: (1001, 1000, 1002, 997, 998, 1001, 999, 1000, 1001, 1001),
        303: (1000, 1000, 1001, 1000, 1000, 998, 1001, 999, 1000, 1001),
    }
    expected_names = {2_000: "2k", 5_000: "5k", 10_000: "10k"}
    full_sample_digests = set()
    for ladder, registered in zip(ladders, REGISTERED_LADDERS, strict=True):
        if not isinstance(ladder, Mapping) or set(ladder) != {
            "ladder_seed",
            "optimiser_seed",
            "budgets",
        }:
            raise ValueError("confirmatory ladder binding has unexpected fields")
        ladder_seed = registered["ladder_seed"]
        if (
            ladder["ladder_seed"] != ladder_seed
            or ladder["optimiser_seed"] != registered["optimiser_seed"]
        ):
            raise ValueError("confirmatory ladder/optimiser pairing drifted")
        budgets = ladder["budgets"]
        if not isinstance(budgets, list) or len(budgets) != len(LABEL_BUDGETS):
            raise ValueError("confirmatory ladder requires the registered 2k/5k/10k budgets")
        fold_sizes = expected_fold_sizes[ladder_seed]
        expected_counts = {
            2_000: sum(fold_sizes[:2]),
            5_000: sum(fold_sizes[:5]),
            10_000: sum(fold_sizes),
        }
        for budget, label_budget in zip(budgets, LABEL_BUDGETS, strict=True):
            if not isinstance(budget, Mapping) or set(budget) != {
                "label_budget",
                "budget_name",
                "training_row_count",
                "sample_ids_sha256",
            }:
                raise ValueError("confirmatory budget binding has unexpected fields")
            short = expected_names[label_budget]
            if (
                budget["label_budget"] != label_budget
                or budget["budget_name"] != f"train_{short}_{ladder_seed}"
                or budget["training_row_count"] != expected_counts[label_budget]
            ):
                raise ValueError("confirmatory budget row count or name drifted")
            sample_digest = _require_sha256(
                "confirmatory sample_ids_sha256", budget["sample_ids_sha256"]
            )
            if label_budget == 10_000:
                full_sample_digests.add(sample_digest)
    if len(full_sample_digests) != 1:
        raise ValueError("all confirmatory ladders must share the exact 10k teacher set")
    return json.loads(json.dumps(binding, sort_keys=True))


def make_confirmatory_trials(
    contract: Mapping[str, Any],
    *,
    gpu_type: Literal["L4"],
    recipe_receipt: Mapping[str, Any],
    threshold_receipt: Mapping[str, Any],
    split_binding: Mapping[str, Any],
    transition_receipt: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create the frozen winner x three paired ladders x three label budgets."""

    validate_experiment_contract(contract)
    recipe = validate_stability_recipe_receipt(recipe_receipt)
    threshold = validate_target_threshold_receipt(recipe, threshold_receipt)
    split = validate_confirmatory_split_binding(contract, split_binding)
    run_id = experiment_run_id(contract)
    if recipe["experiment_run_id"] == run_id:
        if transition_receipt is not None:
            raise ValueError("same-contract confirmatory trials must not use a transition receipt")
        transition_id = None
    else:
        transition = validate_confirmatory_transition_receipt(
            contract,
            recipe,
            threshold,
            transition_receipt,
        )
        transition_id = transition["transition_receipt_id"]
    if threshold["selected_config_sha256"] != recipe["selected_config_sha256"]:
        raise ValueError("confirmatory recipe and threshold select different configurations")
    if threshold["selected_threshold"] != "0.30":
        raise ValueError("confirmatory target threshold must be the frozen 0.30 selection")
    configs = {row["config_sha256"]: row for row in build_asha_plan()["configs"]}
    selected_config = configs.get(recipe["selected_config_sha256"])
    if selected_config is None:
        raise ValueError("confirmatory recipe selected an unregistered sweep configuration")
    trials = []
    for ladder in split["ladders"]:
        for budget in ladder["budgets"]:
            short = budget["budget_name"].split("_")[1]
            trials.append(
                _trial_spec(
                    phase="confirmatory",
                    name=(
                        f"confirmatory-ladder-{ladder['ladder_seed']}-{short}"
                        f"-seed-{ladder['optimiser_seed']}"
                    ),
                    gpu_type=gpu_type,
                    config={
                        "registered_config": selected_config,
                        "budget": budget["budget_name"],
                        "label_budget": budget["label_budget"],
                        "training_row_count": budget["training_row_count"],
                        "subset_sample_ids_sha256": budget["sample_ids_sha256"],
                        "ladder_seed": ladder["ladder_seed"],
                        "seed": ladder["optimiser_seed"],
                        "optimiser_seed": ladder["optimiser_seed"],
                        "target_epochs": 6,
                        "attention_implementation": "sdpa",
                        "target_threshold": "0.30",
                        "continuation": None,
                        "fresh_training": True,
                        "recipe_receipt_id": recipe["recipe_receipt_id"],
                        "threshold_receipt_id": threshold["threshold_receipt_id"],
                        "confirmatory_transition_receipt_id": transition_id,
                        "development_evaluation_only": True,
                        "locked_test_authorised": False,
                    },
                )
            )
    expected_conditions = {
        (
            ladder["ladder_seed"],
            ladder["optimiser_seed"],
            label_budget,
        )
        for ladder in REGISTERED_LADDERS
        for label_budget in LABEL_BUDGETS
    }
    observed_conditions = {
        (
            trial["config"]["ladder_seed"],
            trial["config"]["optimiser_seed"],
            trial["config"]["label_budget"],
        )
        for trial in trials
    }
    if (
        observed_conditions != expected_conditions
        or len(trials) != 9
        or len({trial["trial_id"] for trial in trials}) != 9
    ):
        raise AssertionError("confirmatory trials must cover the exact registered 3 x 3 design")
    return trials


def build_confirmatory_transition_receipt(
    source_contract: Mapping[str, Any],
    destination_contract: Mapping[str, Any],
    recipe_receipt: Mapping[str, Any],
    threshold_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorise only Phase 4 after the selection-bound adapter/orchestrator change."""

    validate_experiment_contract(source_contract)
    validate_experiment_contract(destination_contract)
    recipe = validate_stability_recipe_receipt(recipe_receipt)
    threshold = validate_target_threshold_receipt(recipe, threshold_receipt)
    source_run_id = experiment_run_id(source_contract)
    destination_run_id = experiment_run_id(destination_contract)
    if recipe["experiment_run_id"] != source_run_id:
        raise ValueError("confirmatory transition recipe belongs to another source experiment")
    if _phase3_semantic_contract(source_contract) != _phase3_semantic_contract(
        destination_contract
    ):
        raise ValueError("confirmatory transition changed a frozen semantic experiment binding")
    source_code = source_contract["bindings"]["trainer_code_sha256"]
    destination_code = destination_contract["bindings"]["trainer_code_sha256"]
    if source_code == destination_code:
        raise ValueError("confirmatory transition requires a real source-digest change")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-confirmatory-code-transition-v1",
        "source_experiment_contract": dict(source_contract),
        "destination_experiment_contract": dict(destination_contract),
        "source_experiment_run_id": source_run_id,
        "destination_experiment_run_id": destination_run_id,
        "source_trainer_code_sha256": source_code,
        "destination_trainer_code_sha256": destination_code,
        "recipe_receipt_id": recipe["recipe_receipt_id"],
        "threshold_receipt_id": threshold["threshold_receipt_id"],
        "selected_config_sha256": recipe["selected_config_sha256"],
        "selected_target_threshold": threshold["selected_threshold"],
        "code_delta_scope": [
            "confirmatory_manifest_orchestration",
            "confirmatory_registered_config_adapter",
            "confirmatory_frozen_threshold_adapter",
        ],
        "model_architecture_changed": False,
        "registered_training_recipe_changed": False,
        "phase4_execution_enabled": True,
        "authorisation": {
            "phase": "confirmatory",
            "fresh_training": True,
            "trial_count": 9,
            "selected_config_sha256": recipe["selected_config_sha256"],
            "target_threshold": "0.30",
            "paired_ladders": [dict(row) for row in REGISTERED_LADDERS],
            "label_budgets": list(LABEL_BUDGETS),
            "target_epochs": 6,
            "gpu_type": "L4",
            "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
            "approved_cost_usd": str(destination_contract["compute"]["approved_cost_usd"]),
            "confirmatory_phase_upper_cost_usd": str(CONFIRMATORY_PHASE_UPPER_COST_USD),
            "locked_test_authorised": False,
        },
    }
    receipt = {**body, "transition_receipt_id": _canonical_sha256(body)}
    return validate_confirmatory_transition_receipt(
        destination_contract,
        recipe,
        threshold,
        receipt,
    )


def validate_confirmatory_transition_receipt(
    destination_contract: Mapping[str, Any],
    recipe_receipt: Mapping[str, Any],
    threshold_receipt: Mapping[str, Any],
    transition_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed on any broadening of the post-selection Phase 4 bridge."""

    validate_experiment_contract(destination_contract)
    recipe = validate_stability_recipe_receipt(recipe_receipt)
    threshold = validate_target_threshold_receipt(recipe, threshold_receipt)
    if not isinstance(transition_receipt, Mapping):
        raise ValueError("cross-contract confirmatory trials require a transition receipt")
    required = {
        "schema_version",
        "kind",
        "source_experiment_contract",
        "destination_experiment_contract",
        "source_experiment_run_id",
        "destination_experiment_run_id",
        "source_trainer_code_sha256",
        "destination_trainer_code_sha256",
        "recipe_receipt_id",
        "threshold_receipt_id",
        "selected_config_sha256",
        "selected_target_threshold",
        "code_delta_scope",
        "model_architecture_changed",
        "registered_training_recipe_changed",
        "phase4_execution_enabled",
        "authorisation",
        "transition_receipt_id",
    }
    if set(transition_receipt) != required:
        raise ValueError("confirmatory transition receipt has unexpected fields")
    body = {key: transition_receipt[key] for key in required - {"transition_receipt_id"}}
    if transition_receipt["transition_receipt_id"] != _canonical_sha256(body):
        raise ValueError("confirmatory transition receipt content address drifted")
    source_contract = transition_receipt["source_experiment_contract"]
    target_contract = transition_receipt["destination_experiment_contract"]
    if not isinstance(source_contract, Mapping) or not isinstance(target_contract, Mapping):
        raise ValueError("confirmatory transition contracts must be objects")
    validate_experiment_contract(source_contract)
    validate_experiment_contract(target_contract)
    if dict(target_contract) != dict(destination_contract):
        raise ValueError("confirmatory transition destination contract drifted")
    source_run_id = experiment_run_id(source_contract)
    destination_run_id = experiment_run_id(destination_contract)
    source_code = source_contract["bindings"]["trainer_code_sha256"]
    destination_code = destination_contract["bindings"]["trainer_code_sha256"]
    if (
        transition_receipt["schema_version"] != SCHEMA_VERSION
        or transition_receipt["kind"] != "modernbert-confirmatory-code-transition-v1"
        or transition_receipt["source_experiment_run_id"] != source_run_id
        or transition_receipt["destination_experiment_run_id"] != destination_run_id
        or transition_receipt["source_trainer_code_sha256"] != source_code
        or transition_receipt["destination_trainer_code_sha256"] != destination_code
        or source_code == destination_code
    ):
        raise ValueError("confirmatory transition source or destination binding drifted")
    if _phase3_semantic_contract(source_contract) != _phase3_semantic_contract(
        destination_contract
    ):
        raise ValueError("confirmatory transition changed a frozen semantic experiment binding")
    if (
        recipe["experiment_run_id"] != source_run_id
        or transition_receipt["recipe_receipt_id"] != recipe["recipe_receipt_id"]
        or transition_receipt["threshold_receipt_id"] != threshold["threshold_receipt_id"]
        or transition_receipt["selected_config_sha256"] != recipe["selected_config_sha256"]
        or transition_receipt["selected_target_threshold"] != "0.30"
        or threshold["selected_threshold"] != "0.30"
    ):
        raise ValueError("confirmatory transition selection binding drifted")
    if (
        transition_receipt["code_delta_scope"]
        != [
            "confirmatory_manifest_orchestration",
            "confirmatory_registered_config_adapter",
            "confirmatory_frozen_threshold_adapter",
        ]
        or transition_receipt["model_architecture_changed"] is not False
        or transition_receipt["registered_training_recipe_changed"] is not False
        or transition_receipt["phase4_execution_enabled"] is not True
    ):
        raise ValueError("confirmatory transition code-delta claim drifted")
    expected_authorisation = {
        "phase": "confirmatory",
        "fresh_training": True,
        "trial_count": 9,
        "selected_config_sha256": recipe["selected_config_sha256"],
        "target_threshold": "0.30",
        "paired_ladders": [dict(row) for row in REGISTERED_LADDERS],
        "label_budgets": list(LABEL_BUDGETS),
        "target_epochs": 6,
        "gpu_type": "L4",
        "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
        "approved_cost_usd": str(destination_contract["compute"]["approved_cost_usd"]),
        "confirmatory_phase_upper_cost_usd": str(CONFIRMATORY_PHASE_UPPER_COST_USD),
        "locked_test_authorised": False,
    }
    if transition_receipt["authorisation"] != expected_authorisation:
        raise ValueError("confirmatory transition authorisation broadened or drifted")
    return json.loads(json.dumps(transition_receipt, sort_keys=True))


def make_run_manifest(
    *,
    contract: Mapping[str, Any],
    phase: str,
    trials: Sequence[Mapping[str, Any]],
    rung_epochs: int | None = None,
    promotion_receipt: Mapping[str, Any] | None = None,
    stability_source_receipt: Mapping[str, Any] | None = None,
    phase3_transition_receipt: Mapping[str, Any] | None = None,
    recipe_receipt: Mapping[str, Any] | None = None,
    threshold_receipt: Mapping[str, Any] | None = None,
    confirmatory_split_binding: Mapping[str, Any] | None = None,
    confirmatory_transition_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a launchable phase to an immutable contract and exact trial inventory."""

    run_id = experiment_run_id(contract)
    trial_rows = [dict(row) for row in trials]
    if phase == "preflight":
        expected = make_preflight_trials(contract)
        if trial_rows != expected:
            raise ValueError("preflight manifest must contain the exact L4 trial")
    elif phase == "sweep":
        if rung_epochs not in ASHA_RUNGS:
            raise ValueError(f"sweep rung_epochs must be one of {ASHA_RUNGS}")
        gpus = {row.get("gpu_type") for row in trial_rows}
        if len(gpus) != 1 or next(iter(gpus)) not in ALLOWED_GPUS:
            raise ValueError("sweep manifest must freeze exactly one supported GPU")
        expected = make_sweep_trials(
            contract,
            gpu_type=next(iter(gpus)),
            rung_epochs=rung_epochs,
            promotion_receipt=promotion_receipt,
        )
        if trial_rows != expected:
            raise ValueError("sweep manifest broadened or changed the frozen ASHA rung")
    elif phase == "stability":
        gpus = {row.get("gpu_type") for row in trial_rows}
        if len(gpus) != 1 or next(iter(gpus)) not in ALLOWED_GPUS:
            raise ValueError("stability manifest must freeze exactly one supported GPU")
        if trial_rows != make_stability_trials(
            contract,
            gpu_type=next(iter(gpus)),
            source_receipt=stability_source_receipt,
            transition_receipt=phase3_transition_receipt,
        ):
            raise ValueError("stability manifest changed the exact top-two x two-seed design")
    elif phase == "confirmatory":
        gpus = {row.get("gpu_type") for row in trial_rows}
        if len(gpus) != 1 or next(iter(gpus)) not in ALLOWED_GPUS:
            raise ValueError("confirmatory manifest must freeze exactly one supported GPU")
        if trial_rows != make_confirmatory_trials(
            contract,
            gpu_type=next(iter(gpus)),
            recipe_receipt=recipe_receipt,
            threshold_receipt=threshold_receipt,
            split_binding=confirmatory_split_binding,
            transition_receipt=confirmatory_transition_receipt,
        ):
            raise ValueError("confirmatory manifest changed the exact registered 3 x 3 design")
    elif phase not in TRAIN_PHASES:
        raise ValueError(f"phase must be preflight or one of {TRAIN_PHASES}")
    if phase in TRAIN_PHASES:
        expected_count = (
            next(
                row["candidate_count"]
                for row in build_asha_plan()["rungs"]
                if row["epochs"] == rung_epochs
            )
            if phase == "sweep"
            else TRAIN_PHASE_COUNTS[phase]
        )
        if len(trial_rows) != expected_count:
            raise ValueError(f"{phase} manifest must contain exactly {expected_count} trials")
        gpus = {row.get("gpu_type") for row in trial_rows}
        if len(gpus) != 1 or next(iter(gpus)) not in ALLOWED_GPUS:
            raise ValueError("a training phase must freeze exactly one supported GPU")
    if not trial_rows or len({row.get("trial_id") for row in trial_rows}) != len(trial_rows):
        raise ValueError("run manifest requires unique trials")
    for row in trial_rows:
        expected_id = _trial_spec(
            phase=str(row.get("phase")),
            name=str(row.get("name")),
            gpu_type=str(row.get("gpu_type")),
            config=dict(row.get("config", {})),
        )["trial_id"]
        if row.get("phase") != phase or row.get("trial_id") != expected_id:
            raise ValueError("trial does not match its phase or content-addressed ID")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-phase-run-v1",
        "experiment_run_id": run_id,
        "phase": phase,
        "experiment_contract": dict(contract),
        "asha": (
            {
                "plan_id": build_asha_plan()["plan_id"],
                "rung_epochs": rung_epochs,
                "promotion_receipt": (
                    dict(promotion_receipt) if promotion_receipt is not None else None
                ),
            }
            if phase == "sweep"
            else None
        ),
        "trials": trial_rows,
    }
    if phase == "stability":
        payload["stability"] = {
            "source_receipt": dict(stability_source_receipt),
            "phase3_transition_receipt": (
                dict(phase3_transition_receipt) if phase3_transition_receipt is not None else None
            ),
        }
    if phase == "confirmatory":
        recipe = validate_stability_recipe_receipt(recipe_receipt)
        threshold = validate_target_threshold_receipt(recipe, threshold_receipt)
        split = validate_confirmatory_split_binding(contract, confirmatory_split_binding)
        enforce_cost_guardrail(
            estimated_cost_usd=PLANNED_UPPER_COST_USD,
            approved_cost_usd=Decimal(str(contract["compute"]["approved_cost_usd"])),
        )
        payload["confirmatory"] = {
            "transition_receipt": (
                dict(confirmatory_transition_receipt)
                if confirmatory_transition_receipt is not None
                else None
            ),
            "recipe_receipt": recipe,
            "threshold_receipt": threshold,
            "split_binding": split,
            "design": {
                "ladder_count": len(REGISTERED_LADDERS),
                "label_budgets": list(LABEL_BUDGETS),
                "paired_ladders": [dict(row) for row in REGISTERED_LADDERS],
                "trial_count": 9,
                "fresh_training": True,
                "target_epochs": 6,
                "target_threshold": "0.30",
            },
            "cost_guardrail": {
                "confirmatory_phase_upper_cost_usd": str(
                    CONFIRMATORY_PHASE_UPPER_COST_USD
                ),
                "experiment_planned_upper_cost_usd": str(PLANNED_UPPER_COST_USD),
                "approved_cost_usd": str(contract["compute"]["approved_cost_usd"]),
                "hard_max_approved_cost_usd": str(HARD_MAX_APPROVAL_USD),
            },
            "locked_test": {
                "authorised": False,
                "rows_accessed": 0,
                "predictions_authorised": False,
            },
        }
    return {**payload, "phase_run_id": _canonical_sha256(payload)}


def validate_run_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = manifest.get("experiment_contract")
    trials = manifest.get("trials")
    if not isinstance(contract, Mapping) or not isinstance(trials, list):
        raise ValueError("run manifest is incomplete")
    asha = manifest.get("asha")
    stability = manifest.get("stability")
    confirmatory = manifest.get("confirmatory")
    if manifest.get("phase") == "sweep" and not isinstance(asha, Mapping):
        raise ValueError("sweep run manifest lacks its ASHA rung binding")
    rebuilt = make_run_manifest(
        contract=contract,
        phase=str(manifest.get("phase")),
        trials=trials,
        rung_epochs=asha.get("rung_epochs") if isinstance(asha, Mapping) else None,
        promotion_receipt=(asha.get("promotion_receipt") if isinstance(asha, Mapping) else None),
        stability_source_receipt=(
            stability.get("source_receipt") if isinstance(stability, Mapping) else None
        ),
        phase3_transition_receipt=(
            stability.get("phase3_transition_receipt") if isinstance(stability, Mapping) else None
        ),
        recipe_receipt=(
            confirmatory.get("recipe_receipt") if isinstance(confirmatory, Mapping) else None
        ),
        threshold_receipt=(
            confirmatory.get("threshold_receipt") if isinstance(confirmatory, Mapping) else None
        ),
        confirmatory_split_binding=(
            confirmatory.get("split_binding") if isinstance(confirmatory, Mapping) else None
        ),
        confirmatory_transition_receipt=(
            confirmatory.get("transition_receipt")
            if isinstance(confirmatory, Mapping)
            else None
        ),
    )
    if dict(manifest) != rebuilt:
        raise ValueError("run manifest digest or contents drifted")
    return rebuilt


def build_asha_promotion_receipt(
    manifest: Mapping[str, Any], receipts_by_trial_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Rank one complete rung and bind promoted configs to exact checkpoints."""

    clean_manifest = validate_run_manifest(manifest)
    if clean_manifest["phase"] != "sweep":
        raise ValueError("only a sweep rung can produce an ASHA promotion receipt")
    rung_epochs = clean_manifest["asha"]["rung_epochs"]
    next_rung = {1: 3, 3: 6}.get(rung_epochs)
    if next_rung is None:
        raise ValueError("the final ASHA rung does not promote")
    trials = clean_manifest["trials"]
    if set(receipts_by_trial_id) != {trial["trial_id"] for trial in trials}:
        raise ValueError("promotion requires one exact receipt for every rung candidate")
    results: list[dict[str, Any]] = []
    checkpoint_by_config: dict[str, dict[str, Any]] = {}
    candidate_ids: list[str] = []
    for trial in trials:
        receipt = receipts_by_trial_id[trial["trial_id"]]
        if receipt.get("trial_id") != trial["trial_id"] or receipt.get("status") != "complete":
            raise ValueError("promotion input receipt does not bind its exact trial")
        metrics = receipt.get("aggregate_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("promotion input lacks aggregate metrics")
        config_id = trial["config"]["registered_config"]["config_sha256"]
        candidate_ids.append(config_id)
        result = {
            "config_sha256": config_id,
            "completed_epochs": metrics.get("completed_epochs"),
            "composite": metrics.get("composite"),
            "material_recall": metrics.get("material_recall"),
            "invalid_outputs": metrics.get("invalid_outputs"),
        }
        artifacts = receipt.get("artifacts")
        checkpoint = artifacts.get("checkpoint") if isinstance(artifacts, Mapping) else None
        if not isinstance(checkpoint, Mapping):
            raise ValueError("promotion input lacks an immutable checkpoint artifact")
        checkpoint_by_config[config_id] = {
            "config_sha256": config_id,
            "source_trial_id": trial["trial_id"],
            "checkpoint_relative_path": str(
                Path(f"phase=sweep/trial={trial['trial_id']}")
                / str(checkpoint.get("relative_path"))
            ),
            "checkpoint_sha256": checkpoint.get("sha256"),
        }
        results.append(result)
    promoted_ids = select_asha_promotions(
        build_asha_plan(),
        results,
        rung_epochs=rung_epochs,
        candidate_config_ids=candidate_ids,
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-asha-promotion-v1",
        "experiment_run_id": clean_manifest["experiment_run_id"],
        "asha_plan_id": build_asha_plan()["plan_id"],
        "source_phase_run_id": clean_manifest["phase_run_id"],
        "from_rung_epochs": rung_epochs,
        "to_rung_epochs": next_rung,
        "candidate_config_ids": candidate_ids,
        "results_sha256": _canonical_sha256(results),
        "promoted": [checkpoint_by_config[config_id] for config_id in promoted_ids],
    }
    receipt = {**body, "promotion_receipt_id": _canonical_sha256(body)}
    return validate_asha_promotion_receipt(
        clean_manifest["experiment_contract"],
        receipt,
        expected_to_rung_epochs=next_rung,
    )


def _metric_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def build_stability_source_receipt(
    final_asha_manifest: Mapping[str, Any],
    receipts_by_trial_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the two completed final-rung configs as stability inputs."""

    manifest = validate_run_manifest(final_asha_manifest)
    if manifest["phase"] != "sweep" or manifest["asha"]["rung_epochs"] != 6:
        raise ValueError("stability source requires the exact final six-epoch ASHA rung")
    trials = manifest["trials"]
    if len(trials) != 2 or set(receipts_by_trial_id) != {trial["trial_id"] for trial in trials}:
        raise ValueError("stability source requires both exact final-rung receipts")
    configs: list[dict[str, Any]] = []
    receipt_ids: list[str] = []
    for trial in trials:
        receipt = receipts_by_trial_id[trial["trial_id"]]
        metrics = receipt.get("aggregate_metrics")
        if (
            receipt.get("status") != "complete"
            or receipt.get("trial_id") != trial["trial_id"]
            or not isinstance(metrics, Mapping)
            or metrics.get("completed_epochs") != 6
            or metrics.get("invalid_outputs") != 0
        ):
            raise ValueError("final ASHA receipt is incomplete or scientifically invalid")
        _metric_probability("final ASHA composite", metrics.get("composite"))
        _metric_probability("final ASHA material recall", metrics.get("material_recall"))
        receipt_ids.append(_require_sha256("source receipt ID", receipt.get("receipt_id")))
        configs.append(dict(trial["config"]["registered_config"]))
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-stability-source-v1",
        "experiment_run_id": manifest["experiment_run_id"],
        "asha_plan_id": build_asha_plan()["plan_id"],
        "source_phase_run_id": manifest["phase_run_id"],
        "candidate_configs": configs,
        "source_receipt_ids": receipt_ids,
    }
    receipt = {**body, "stability_source_receipt_id": _canonical_sha256(body)}
    return validate_stability_source_receipt(manifest["experiment_contract"], receipt)


def build_phase3_transition_receipt(
    final_asha_manifest: Mapping[str, Any],
    final_asha_receipts_by_trial_id: Mapping[str, Mapping[str, Any]],
    destination_contract: Mapping[str, Any],
    *,
    include_stability_adapter_fix: bool = False,
) -> dict[str, Any]:
    """Authorise only fresh Phase 3 trials after a publication-only code change."""

    manifest = validate_run_manifest(final_asha_manifest)
    source_contract = manifest["experiment_contract"]
    validate_experiment_contract(destination_contract)
    source = build_stability_source_receipt(manifest, final_asha_receipts_by_trial_id)
    if _phase3_semantic_contract(source_contract) != _phase3_semantic_contract(
        destination_contract
    ):
        raise ValueError("cannot transition Phase 3 across semantic experiment drift")
    source_code = source_contract["bindings"]["trainer_code_sha256"]
    destination_code = destination_contract["bindings"]["trainer_code_sha256"]
    if source_code == destination_code:
        raise ValueError("Phase 3 transition requires a real trainer source-digest change")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-phase3-code-transition-v1",
        "source_experiment_contract": dict(source_contract),
        "destination_experiment_contract": dict(destination_contract),
        "source_experiment_run_id": manifest["experiment_run_id"],
        "destination_experiment_run_id": experiment_run_id(destination_contract),
        "source_trainer_code_sha256": source_code,
        "destination_trainer_code_sha256": destination_code,
        "stability_source_receipt_id": source["stability_source_receipt_id"],
        "final_asha_phase_run_id": source["source_phase_run_id"],
        "final_asha_trial_receipt_ids": source["source_receipt_ids"],
        "code_delta_scope": [
            "private_development_prediction_publication",
            "phase3_stability_orchestration",
            *(["stability_registered_config_adapter_fix"] if include_stability_adapter_fix else []),
        ],
        "model_semantics_changed": False,
        "training_semantics_changed": include_stability_adapter_fix,
        "authorisation": {
            "phase": "stability",
            "fresh_training": True,
            "trial_count": 4,
            "candidate_config_sha256": [
                row["config_sha256"] for row in source["candidate_configs"]
            ],
            "optimiser_seeds": list(STABILITY_SEEDS),
            "label_budget": "train_5k_101",
            "target_epochs": 6,
            "confirmatory_authorised": False,
            "locked_test_authorised": False,
        },
    }
    receipt = {**body, "transition_receipt_id": _canonical_sha256(body)}
    return validate_phase3_transition_receipt(destination_contract, source, receipt)


def _development_prediction_descriptor(receipt: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = receipt.get("artifacts")
    prediction = (
        artifacts.get("development_predictions") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(prediction, Mapping):
        raise ValueError("stability receipt lacks development_predictions")
    required = {"relative_path", "sha256", "bytes"}
    if set(prediction) != required:
        raise ValueError("development prediction descriptor drifted")
    _safe_artifact_path(Path("/"), prediction["relative_path"])
    _require_sha256("development prediction SHA-256", prediction["sha256"])
    if type(prediction["bytes"]) is not int or prediction["bytes"] <= 0:
        raise ValueError("development prediction byte count must be positive")
    return dict(prediction)


def _validate_stability_trial_receipt_binding(
    manifest: Mapping[str, Any],
    trial: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Bind one stability result to its exact manifest trial and content address."""

    receipt_id = _require_sha256("stability trial receipt ID", receipt.get("receipt_id"))
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if _canonical_sha256(body) != receipt_id:
        raise ValueError("stability trial receipt content address drifted")
    expected = {
        "experiment_run_id": manifest["experiment_run_id"],
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": _canonical_sha256(trial),
        "bindings": manifest["experiment_contract"]["bindings"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"stability trial receipt binding mismatch: {key}")


def _stability_completion_evidence(
    trial: Mapping[str, Any],
    receipt: Mapping[str, Any],
    history: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove max-epoch completion or the exact registered early-stop rule."""

    config = trial.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("registered_config"), Mapping):
        raise ValueError("stability trial lacks its registered configuration")
    registered = config["registered_config"]
    maximum_epochs = registered.get("max_epochs")
    if maximum_epochs != 6 or config.get("target_epochs") != maximum_epochs:
        raise ValueError("stability completion must remain frozen to maximum epoch 6")
    metrics = receipt.get("aggregate_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("stability receipt lacks aggregate metrics")
    completed = metrics.get("completed_epochs")
    selected = metrics.get("selected_epoch")
    if (
        type(completed) is not int
        or type(selected) is not int
        or not 2 <= selected <= completed <= maximum_epochs
    ):
        raise ValueError("stability completed/selected epochs are invalid")

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("stability receipt lacks immutable artifacts")
    history_descriptors = [
        value
        for value in artifacts.values()
        if isinstance(value, Mapping) and value.get("relative_path") == "history.json"
    ]
    if len(history_descriptors) != 1:
        raise ValueError("stability receipt must bind exactly one history artifact")
    history_descriptor = history_descriptors[0]
    if set(history_descriptor) != {"relative_path", "sha256", "bytes"}:
        raise ValueError("stability history descriptor drifted")
    history_bytes = (
        json.dumps(history, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if hashlib.sha256(history_bytes).hexdigest() != history_descriptor.get("sha256") or len(
        history_bytes
    ) != history_descriptor.get("bytes"):
        raise ValueError("stability history payload differs from its receipt descriptor")
    if not isinstance(history, Mapping) or set(history) != {"epochs"}:
        raise ValueError("stability history schema drifted")
    epochs = history["epochs"]
    if not isinstance(epochs, list) or len(epochs) != completed:
        raise ValueError("stability history does not conserve completed epochs")

    checkpoints_by_epoch: dict[int, Mapping[str, Any]] = {}
    for epoch in range(1, completed + 1):
        expected_path = f"checkpoints/epoch-{epoch:02d}.pt"
        descriptors = [
            value
            for value in artifacts.values()
            if isinstance(value, Mapping) and value.get("relative_path") == expected_path
        ]
        if len(descriptors) != 1:
            raise ValueError(
                "stability receipt does not bind every observed checkpoint exactly once"
            )
        checkpoints_by_epoch[epoch] = descriptors[0]
    selection_inputs = []
    for expected_epoch, row in enumerate(epochs, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != expected_epoch:
            raise ValueError("stability history epochs must be consecutive and start at one")
        development = row.get("development")
        if not isinstance(development, Mapping):
            raise ValueError("stability history epoch lacks development metrics")
        selection_inputs.append(
            {
                "epoch": expected_epoch,
                "composite": _metric_probability(
                    "stability history composite", development.get("composite")
                ),
                "checkpoint_sha256": _require_sha256(
                    "stability checkpoint SHA-256",
                    checkpoints_by_epoch[expected_epoch].get("sha256"),
                ),
            }
        )
    checkpoint_selection = select_checkpoint(selection_inputs)
    if checkpoint_selection["selected_epoch"] != selected or checkpoint_selection[
        "selected_composite"
    ] != metrics.get("composite"):
        raise ValueError("stability receipt differs from registered checkpoint selection")
    if completed < maximum_epochs:
        if checkpoint_selection["stopped_epoch"] != completed:
            raise ValueError("short stability run is not a registered early stop")
        completion = "registered_early_stop"
    else:
        completion = "maximum_epochs"
    selected_checkpoint = artifacts.get("checkpoint")
    if (
        not isinstance(selected_checkpoint, Mapping)
        or selected_checkpoint.get("relative_path") != f"checkpoints/epoch-{selected:02d}.pt"
        or selected_checkpoint.get("sha256") != checkpoint_selection["selected_checkpoint_sha256"]
    ):
        raise ValueError("stability selected checkpoint artifact binding drifted")
    return {
        "completed_epochs": completed,
        "selected_epoch": selected,
        "stopped_epoch": checkpoint_selection["stopped_epoch"],
        "completion": completion,
        "checkpoint_selection_id": checkpoint_selection["selection_id"],
        "history_sha256": history_descriptor["sha256"],
    }


def build_stability_recipe_receipt(
    stability_manifest: Mapping[str, Any],
    receipts_by_trial_id: Mapping[str, Mapping[str, Any]],
    histories_by_trial_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one recipe by mean development composite across both fresh seeds."""

    manifest = validate_run_manifest(stability_manifest)
    if manifest["phase"] != "stability":
        raise ValueError("recipe selection requires the stability phase")
    trials = manifest["trials"]
    if set(receipts_by_trial_id) != {trial["trial_id"] for trial in trials}:
        raise ValueError("recipe selection requires all four exact stability receipts")
    if set(histories_by_trial_id) != {trial["trial_id"] for trial in trials}:
        raise ValueError("recipe selection requires all four exact stability histories")
    by_config: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        receipt = receipts_by_trial_id[trial["trial_id"]]
        _validate_stability_trial_receipt_binding(manifest, trial, receipt)
        metrics = receipt.get("aggregate_metrics")
        if (
            receipt.get("status") != "complete"
            or receipt.get("trial_id") != trial["trial_id"]
            or not isinstance(metrics, Mapping)
            or metrics.get("invalid_outputs") != 0
        ):
            raise ValueError("stability receipt is incomplete or scientifically invalid")
        completion = _stability_completion_evidence(
            trial, receipt, histories_by_trial_id[trial["trial_id"]]
        )
        config = trial["config"]["registered_config"]
        config_id = config["config_sha256"]
        by_config.setdefault(config_id, []).append(
            {
                "seed": trial["config"]["seed"],
                "trial_id": trial["trial_id"],
                "trial_receipt_id": _require_sha256(
                    "stability trial receipt ID", receipt.get("receipt_id")
                ),
                "development_composite": _metric_probability(
                    "development composite", metrics.get("composite")
                ),
                **completion,
                "development_prediction": _development_prediction_descriptor(receipt),
            }
        )
    if set(by_config) != {
        row["config_sha256"] for row in manifest["stability"]["source_receipt"]["candidate_configs"]
    }:
        raise ValueError("stability receipts do not cover both candidate recipes")
    candidates = []
    for config_id in sorted(by_config):
        seeds = sorted(by_config[config_id], key=lambda row: row["seed"])
        if [row["seed"] for row in seeds] != list(STABILITY_SEEDS):
            raise ValueError("each stability recipe requires seeds 13 and 29 exactly once")
        candidates.append(
            {
                "config_sha256": config_id,
                "seed_results": seeds,
                "mean_development_composite": sum(row["development_composite"] for row in seeds)
                / len(seeds),
            }
        )
    selected = sorted(
        candidates,
        key=lambda row: (-row["mean_development_composite"], row["config_sha256"]),
    )[0]
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-stability-recipe-selection-v1",
        "experiment_run_id": manifest["experiment_run_id"],
        "stability_phase_run_id": manifest["phase_run_id"],
        "stability_source_receipt_id": manifest["stability"]["source_receipt"][
            "stability_source_receipt_id"
        ],
        "selection_rule": "highest_mean_development_composite_then_config_sha256",
        "candidates": candidates,
        "selected_config_sha256": selected["config_sha256"],
    }
    return {**body, "recipe_receipt_id": _canonical_sha256(body)}


def validate_stability_recipe_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "experiment_run_id",
        "stability_phase_run_id",
        "stability_source_receipt_id",
        "selection_rule",
        "candidates",
        "selected_config_sha256",
        "recipe_receipt_id",
    }
    if set(receipt) != required:
        raise ValueError("stability recipe receipt has unexpected fields")
    body = {key: receipt[key] for key in required - {"recipe_receipt_id"}}
    if receipt["recipe_receipt_id"] != _canonical_sha256(body):
        raise ValueError("stability recipe receipt content address drifted")
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["kind"] != "modernbert-stability-recipe-selection-v1"
        or receipt["selection_rule"] != "highest_mean_development_composite_then_config_sha256"
    ):
        raise ValueError("stability recipe selection contract drifted")
    candidates = receipt["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("stability recipe receipt must contain two candidates")
    for candidate in candidates:
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"config_sha256", "seed_results", "mean_development_composite"}
            or len(candidate.get("seed_results", [])) != 2
        ):
            raise ValueError("stability recipe candidate seed results drifted")
        _require_sha256("candidate config SHA-256", candidate.get("config_sha256"))
        seed_results = candidate["seed_results"]
        if [row.get("seed") for row in seed_results] != list(STABILITY_SEEDS):
            raise ValueError("stability recipe candidate must bind seeds 13 and 29")
        for row in seed_results:
            if set(row) != {
                "seed",
                "trial_id",
                "trial_receipt_id",
                "development_composite",
                "completed_epochs",
                "selected_epoch",
                "stopped_epoch",
                "completion",
                "checkpoint_selection_id",
                "history_sha256",
                "development_prediction",
            }:
                raise ValueError("stability recipe seed-result schema drifted")
            _require_sha256("stability trial ID", row.get("trial_id"))
            _require_sha256("stability receipt ID", row.get("trial_receipt_id"))
            _metric_probability("stability development composite", row.get("development_composite"))
            completed = row.get("completed_epochs")
            selected_epoch = row.get("selected_epoch")
            stopped_epoch = row.get("stopped_epoch")
            if (
                type(completed) is not int
                or type(selected_epoch) is not int
                or not 2 <= selected_epoch <= completed <= 6
                or row.get("completion") not in {"maximum_epochs", "registered_early_stop"}
            ):
                raise ValueError("stability recipe completion evidence drifted")
            if row["completion"] == "maximum_epochs" and completed != 6:
                raise ValueError("maximum-epoch stability result must complete epoch 6")
            if row["completion"] == "registered_early_stop" and (
                completed >= 6 or stopped_epoch != completed
            ):
                raise ValueError("early-stop stability result is not bound to its stop epoch")
            _require_sha256("stability checkpoint selection ID", row.get("checkpoint_selection_id"))
            _require_sha256("stability history SHA-256", row.get("history_sha256"))
            prediction = row.get("development_prediction")
            if not isinstance(prediction, Mapping):
                raise ValueError("stability recipe candidate lacks prediction binding")
            _development_prediction_descriptor(
                {"artifacts": {"development_predictions": prediction}}
            )
        expected_mean = sum(row["development_composite"] for row in seed_results) / 2
        if candidate.get("mean_development_composite") != expected_mean:
            raise ValueError("stability candidate mean differs from its two seed scores")
    expected = sorted(
        candidates,
        key=lambda row: (-row["mean_development_composite"], row["config_sha256"]),
    )[0]["config_sha256"]
    if receipt["selected_config_sha256"] != expected:
        raise ValueError("stability recipe selection differs from registered rule")
    return json.loads(json.dumps(receipt, sort_keys=True))


def build_target_threshold_receipt(
    recipe_receipt: Mapping[str, Any],
    score_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one global target threshold from metadata-only bound prediction scores."""

    recipe = validate_stability_recipe_receipt(recipe_receipt)
    selected = next(
        row
        for row in recipe["candidates"]
        if row["config_sha256"] == recipe["selected_config_sha256"]
    )
    expected_by_trial = {row["trial_id"]: row for row in selected["seed_results"]}
    inputs = [dict(row) for row in score_inputs]
    if len(inputs) != 2 or {row.get("trial_id") for row in inputs} != set(expected_by_trial):
        raise ValueError("threshold search requires the selected recipe's two prediction sets")
    threshold_keys = [f"{value:.2f}" for value in TARGET_THRESHOLDS]
    clean_inputs = []
    for row in sorted(inputs, key=lambda item: item["trial_id"]):
        if set(row) != {
            "trial_id",
            "prediction_sha256",
            "development_rows",
            "development_reference_sha256",
            "threshold_composites",
        }:
            raise ValueError("threshold score input schema drifted")
        expected = expected_by_trial[row["trial_id"]]
        if row.get("prediction_sha256") != expected["development_prediction"]["sha256"]:
            raise ValueError("threshold score input prediction hash drifted")
        if row.get("development_rows") != 222:
            raise ValueError("threshold score input must bind exactly 222 development rows")
        reference_sha256 = _require_sha256(
            "development reference SHA-256", row.get("development_reference_sha256")
        )
        scores = row.get("threshold_composites")
        if not isinstance(scores, Mapping) or list(scores) != threshold_keys:
            raise ValueError("threshold score input must contain the registered ordered grid")
        clean_inputs.append(
            {
                "trial_id": row["trial_id"],
                "prediction_sha256": row["prediction_sha256"],
                "development_rows": 222,
                "development_reference_sha256": reference_sha256,
                "threshold_composites": {
                    key: _metric_probability(f"threshold composite {key}", scores[key])
                    for key in threshold_keys
                },
            }
        )
    reference_digests = {row["development_reference_sha256"] for row in clean_inputs}
    if len(reference_digests) != 1:
        raise ValueError("threshold score inputs bind different development references")
    means = {
        key: sum(row["threshold_composites"][key] for row in clean_inputs) / len(clean_inputs)
        for key in threshold_keys
    }
    selected_threshold = sorted(
        means,
        key=lambda key: (-means[key], -float(key)),
    )[0]
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-global-target-threshold-v1",
        "experiment_run_id": recipe["experiment_run_id"],
        "recipe_receipt_id": recipe["recipe_receipt_id"],
        "selected_config_sha256": recipe["selected_config_sha256"],
        "development_rows": 222,
        "development_reference_sha256": next(iter(reference_digests)),
        "registered_thresholds": threshold_keys,
        "selection_rule": "highest_mean_composite_then_higher_threshold",
        "score_inputs": clean_inputs,
        "score_inputs_sha256": _canonical_sha256(clean_inputs),
        "mean_composites": means,
        "selected_threshold": selected_threshold,
    }
    return {**body, "threshold_receipt_id": _canonical_sha256(body)}


def validate_target_threshold_receipt(
    recipe_receipt: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    recipe = validate_stability_recipe_receipt(recipe_receipt)
    if receipt.get("recipe_receipt_id") != recipe["recipe_receipt_id"]:
        raise ValueError("threshold receipt binds another recipe")
    rebuilt = build_target_threshold_receipt(recipe, receipt.get("score_inputs", []))
    if dict(receipt) != rebuilt:
        raise ValueError("target threshold receipt content or selection drifted")
    return rebuilt


def _safe_artifact_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("artifact path must be a string")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("artifact path must remain inside the trial output")
    return root.joinpath(*relative.parts)


def validate_trial_receipt(
    *, trial_root: Path, receipt: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a metadata-only receipt and every immutable artifact hash."""

    spec = job["trial_spec"]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment_run_id": job["experiment_run_id"],
        "trial_id": spec["trial_id"],
        "trial_spec_sha256": _canonical_sha256(spec),
        "bindings": job["experiment_contract"]["bindings"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"trial receipt binding mismatch: {key}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError("trial receipt has no immutable artifacts")
    for name, artifact in artifacts.items():
        if not isinstance(name, str) or not isinstance(artifact, Mapping):
            raise RuntimeError("trial receipt artifact contract drifted")
        path = _safe_artifact_path(trial_root, artifact.get("relative_path"))
        if not path.is_file():
            raise RuntimeError(f"trial artifact is missing: {name}")
        if path.stat().st_size != artifact.get("bytes") or _file_sha256(path) != artifact.get(
            "sha256"
        ):
            raise RuntimeError(f"trial artifact hash or size mismatch: {name}")
    compute = receipt.get("compute")
    if not isinstance(compute, Mapping) or compute.get("gpu_type") != spec["gpu_type"]:
        raise RuntimeError("trial compute receipt does not bind the requested GPU")
    for key in ("wall_seconds", "gpu_seconds", "estimated_cost_usd"):
        if key not in compute:
            raise RuntimeError(f"trial compute receipt is missing {key}")
    return {
        "status": "validated",
        "trial_id": spec["trial_id"],
        "phase": spec["phase"],
        "gpu_type": spec["gpu_type"],
        "estimated_cost_usd": str(compute["estimated_cost_usd"]),
        "artifact_count": len(artifacts),
    }


def _trial_job(manifest: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "experiment_contract": manifest["experiment_contract"],
        "trial_spec": dict(trial),
        "trial_spec_sha256": _canonical_sha256(trial),
    }


def _run_root(manifest: Mapping[str, Any]) -> Path:
    return VOLUME_PATH / OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}"


def _freeze_stability_selection(
    manifest: Mapping[str, Any], *, volume_root: Path
) -> dict[str, Any]:
    """Privately score and immutably freeze one stability recipe and threshold."""

    from reddit_china_stance.modernbert_training import (
        DEVELOPMENT_PROXY_PARQUET_NAME,
        VOLUME_INPUT_PREFIX,
        publish_immutable_json,
        score_private_development_thresholds,
    )

    clean = validate_run_manifest(manifest)
    if clean["phase"] != "stability":
        raise ValueError("selection freezing requires the stability phase")
    run_root = volume_root / OUTPUT_PREFIX / f"run={clean['experiment_run_id']}"
    receipts: dict[str, Mapping[str, Any]] = {}
    histories: dict[str, Mapping[str, Any]] = {}
    roots: dict[str, Path] = {}
    for trial in clean["trials"]:
        trial_id = trial["trial_id"]
        trial_root = run_root / "phase=stability" / f"trial={trial_id}"
        receipt_path = trial_root / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError("stability selection requires four complete trial receipts")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_trial_receipt(
            trial_root=trial_root,
            receipt=receipt,
            job=_trial_job(clean, trial),
        )
        history_path = trial_root / "history.json"
        if not history_path.is_file():
            raise RuntimeError("stability selection requires every exact history artifact")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, Mapping):
            raise ValueError("stability history must contain an object")
        receipts[trial_id] = receipt
        histories[trial_id] = history
        roots[trial_id] = trial_root

    recipe = build_stability_recipe_receipt(clean, receipts, histories)
    selected = next(
        candidate
        for candidate in recipe["candidates"]
        if candidate["config_sha256"] == recipe["selected_config_sha256"]
    )
    proxy_path = volume_root / VOLUME_INPUT_PREFIX / DEVELOPMENT_PROXY_PARQUET_NAME
    score_inputs = []
    for seed_result in selected["seed_results"]:
        trial_id = seed_result["trial_id"]
        descriptor = seed_result["development_prediction"]
        prediction_path = _safe_artifact_path(roots[trial_id], descriptor["relative_path"])
        if (
            prediction_path.stat().st_size != descriptor["bytes"]
            or _file_sha256(prediction_path) != descriptor["sha256"]
        ):
            raise RuntimeError("selected development prediction artifact is corrupt")
        prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        scored = score_private_development_thresholds(
            prediction_payload,
            development_proxy_path=proxy_path,
            expected_composite=seed_result["development_composite"],
        )
        score_inputs.append(
            {
                "trial_id": trial_id,
                "prediction_sha256": descriptor["sha256"],
                **scored,
            }
        )
    threshold = build_target_threshold_receipt(recipe, score_inputs)
    validate_target_threshold_receipt(recipe, threshold)
    assert_metadata_only(recipe, where="ModernBERT stability recipe receipt")
    assert_metadata_only(threshold, where="ModernBERT target threshold receipt")

    selection_root = run_root / "selection"
    recipe_descriptor = publish_immutable_json(selection_root / "stability-recipe.json", recipe)
    threshold_descriptor = publish_immutable_json(
        selection_root / "target-threshold.json", threshold
    )
    result = {
        "status": "frozen",
        "experiment_run_id": clean["experiment_run_id"],
        "recipe_receipt_id": recipe["recipe_receipt_id"],
        "threshold_receipt_id": threshold["threshold_receipt_id"],
        "selected_config_sha256": recipe["selected_config_sha256"],
        "selected_threshold": threshold["selected_threshold"],
        "artifacts": {
            "stability_recipe": recipe_descriptor,
            "target_threshold": threshold_descriptor,
        },
    }
    assert_metadata_only(result, where="ModernBERT stability selection summary")
    return result


def _promotion_path(run_root: Path, *, from_rung: int, to_rung: int) -> Path:
    return run_root / "promotions" / f"rung={from_rung}-to={to_rung}.json"


def _validate_asha_predecessor(manifest: Mapping[str, Any], run_root: Path) -> None:
    if manifest["phase"] != "sweep" or manifest["asha"]["rung_epochs"] == 1:
        return
    expected = validate_asha_promotion_receipt(
        manifest["experiment_contract"],
        manifest["asha"]["promotion_receipt"],
        expected_to_rung_epochs=manifest["asha"]["rung_epochs"],
    )
    path = _promotion_path(
        run_root,
        from_rung=expected["from_rung_epochs"],
        to_rung=expected["to_rung_epochs"],
    )
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("prior ASHA promotion receipt is absent or differs on the Volume")
    for promoted in expected["promoted"]:
        checkpoint = _safe_artifact_path(run_root, promoted["checkpoint_relative_path"])
        if not checkpoint.is_file() or _file_sha256(checkpoint) != promoted["checkpoint_sha256"]:
            raise RuntimeError("promoted ASHA checkpoint is absent or corrupt")


def _inspect_exact_trial(manifest: Mapping[str, Any], trial_id: str) -> dict[str, Any]:
    """Validate or classify one exact trial without scanning sibling trials."""

    clean = validate_run_manifest(manifest)
    trials = [trial for trial in clean["trials"] if trial["trial_id"] == trial_id]
    if len(trials) != 1:
        raise ValueError("trial_id must identify exactly one manifest trial")
    root = _run_root(clean)
    _validate_asha_predecessor(clean, root)
    trial = trials[0]
    job = _trial_job(clean, trial)
    final_root = root / f"phase={clean['phase']}" / f"trial={trial_id}"
    incomplete_root = root / ".incomplete" / f"trial={trial_id}"
    if final_root.exists():
        if incomplete_root.exists():
            raise RuntimeError(f"trial has final and incomplete outputs: {trial_id}")
        receipt_path = final_root / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"final trial lacks receipt: {trial_id}")
        result = validate_trial_receipt(
            trial_root=final_root,
            receipt=json.loads(receipt_path.read_text(encoding="utf-8")),
            job=job,
        )
        return {**result, "disposition": "complete"}
    if not incomplete_root.exists():
        return {
            "status": "inspected",
            "trial_id": trial_id,
            "phase": clean["phase"],
            "disposition": "missing",
            "estimated_cost_usd": "0",
        }
    resume_path = incomplete_root / "resume.json"
    if not resume_path.is_file():
        raise RuntimeError(f"incomplete trial has no exact resume marker: {trial_id}")
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    expected_resume = {
        "schema_version": SCHEMA_VERSION,
        "experiment_run_id": clean["experiment_run_id"],
        "trial_id": trial_id,
        "trial_spec_sha256": job["trial_spec_sha256"],
    }
    for key, value in expected_resume.items():
        if resume.get(key) != value:
            raise RuntimeError(f"resume marker binding mismatch for {trial_id}: {key}")
    checkpoint = resume.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"resume marker lacks checkpoint metadata: {trial_id}")
    checkpoint_path = _safe_artifact_path(incomplete_root, checkpoint.get("relative_path"))
    if not checkpoint_path.is_file() or _file_sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise RuntimeError(f"resume checkpoint is missing or corrupt: {trial_id}")
    return {
        "status": "inspected",
        "trial_id": trial_id,
        "phase": clean["phase"],
        "disposition": "resumable",
        "estimated_cost_usd": "0",
    }


def _aggregate_trial_inspections(
    manifest: Mapping[str, Any], inspections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate exact per-trial inspections into the existing public summary."""

    clean = validate_run_manifest(manifest)
    expected = [trial["trial_id"] for trial in clean["trials"]]
    by_id: dict[str, Mapping[str, Any]] = {}
    for result in inspections:
        trial_id = result.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in by_id:
            raise RuntimeError("trial inspections contain an invalid or duplicate trial_id")
        by_id[trial_id] = result
    if set(by_id) != set(expected):
        raise RuntimeError("trial inspections do not exactly cover the manifest")
    complete = [trial_id for trial_id in expected if by_id[trial_id]["disposition"] == "complete"]
    missing = [trial_id for trial_id in expected if by_id[trial_id]["disposition"] == "missing"]
    resumable = [trial_id for trial_id in expected if by_id[trial_id]["disposition"] == "resumable"]
    if len(complete) + len(missing) + len(resumable) != len(expected):
        raise RuntimeError("trial inspection contains an unsupported disposition")
    total_cost = sum(
        (Decimal(str(by_id[trial_id]["estimated_cost_usd"])) for trial_id in complete),
        Decimal("0"),
    )
    return {
        "status": "inspected",
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "phase": clean["phase"],
        "expected_trials": len(expected),
        "complete_trial_ids": complete,
        "missing_trial_ids": missing,
        "resumable_trial_ids": resumable,
        "estimated_completed_cost_usd": str(total_cost),
    }


def _inspect_exact_trials(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Classify exact trials as complete, missing or safely resumable."""

    clean = validate_run_manifest(manifest)
    inspections = [_inspect_exact_trial(clean, trial["trial_id"]) for trial in clean["trials"]]
    return _aggregate_trial_inspections(clean, inspections)


def _invoke_preflight(job: Mapping[str, Any], *, resume: bool) -> Mapping[str, Any]:
    from reddit_china_stance.modernbert_training import run_preflight_trial

    return run_preflight_trial(job=job, volume_root=VOLUME_PATH, resume=resume)


def _invoke_training(job: Mapping[str, Any], *, resume: bool) -> Mapping[str, Any]:
    from reddit_china_stance.modernbert_training import run_training_trial

    return run_training_trial(job=job, volume_root=VOLUME_PATH, resume=resume)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def preflight_l4(job: dict[str, Any], resume: bool = False) -> Mapping[str, Any]:
    return _invoke_preflight(job, resume=resume)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65536,
    timeout=12 * 60 * 60,
    max_containers=MAX_CONCURRENT_TRIALS,
    volumes={str(VOLUME_PATH): volume},
)
def train_l4(job: dict[str, Any], resume: bool = False) -> Mapping[str, Any]:
    return _invoke_training(job, resume=resume)


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def inspect_run(manifest: dict[str, Any]) -> dict[str, Any]:
    return _inspect_exact_trials(manifest)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    max_containers=MAX_CONCURRENT_TRIALS,
    volumes={str(VOLUME_PATH): volume},
)
def inspect_trial(manifest: dict[str, Any], trial_id: str) -> dict[str, Any]:
    return _inspect_exact_trial(manifest, trial_id)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    volumes={str(VOLUME_PATH): volume},
)
def freeze_stability_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    result = _freeze_stability_selection(manifest, volume_root=VOLUME_PATH)
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def promote_asha_rung(manifest: dict[str, Any]) -> dict[str, Any]:
    """Publish one immutable promotion only after the source rung fully validates."""

    clean = validate_run_manifest(manifest)
    inspection = _inspect_exact_trials(clean)
    if inspection["missing_trial_ids"] or inspection["resumable_trial_ids"]:
        raise RuntimeError("ASHA promotion requires a complete validated source rung")
    root = _run_root(clean)
    receipts: dict[str, Mapping[str, Any]] = {}
    for trial in clean["trials"]:
        receipt_path = (
            root / f"phase={clean['phase']}" / f"trial={trial['trial_id']}" / "receipt.json"
        )
        receipts[trial["trial_id"]] = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt = build_asha_promotion_receipt(clean, receipts)
    path = _promotion_path(
        root,
        from_rung=receipt["from_rung_epochs"],
        to_rung=receipt["to_rung_epochs"],
    )
    payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError("immutable ASHA promotion receipt already differs")
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        volume.commit()
    return {
        "status": "promoted",
        "promotion_receipt_id": receipt["promotion_receipt_id"],
        "from_rung_epochs": receipt["from_rung_epochs"],
        "to_rung_epochs": receipt["to_rung_epochs"],
        "promoted_config_ids": [row["config_sha256"] for row in receipt["promoted"]],
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("run manifest must contain a JSON object")
    return validate_run_manifest(value)


def _confirmation(action: str, manifest: Mapping[str, Any]) -> str:
    return (
        f"{action.upper()}_MODERNBERT_{str(manifest['phase']).upper()}_"
        f"{str(manifest['phase_run_id'])[:12]}"
    )


@app.local_entrypoint()
def main(
    action: str = "",
    manifest_path: str = "data/private-modernbert-v1/run-manifest.json",
    approved_cost_usd: str = "200",
    confirm: str = "",
) -> None:
    """Plan, launch preflight/train, or validate one exact immutable phase."""

    if action not in {
        "plan",
        "preflight",
        "train",
        "validate",
        "validate-sharded",
        "promote",
        "freeze",
    }:
        raise ValueError(
            "action must be exactly plan, preflight, train, validate, validate-sharded, promote, "
            "or freeze"
        )
    manifest = _load_manifest(Path(manifest_path))
    approved = Decimal(approved_cost_usd)
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(
            manifest["experiment_contract"]["compute"]["planned_upper_cost_usd"]
        ),
        approved_cost_usd=approved,
    )
    if approved != Decimal(manifest["experiment_contract"]["compute"]["approved_cost_usd"]):
        raise ValueError("CLI approval must exactly match the immutable experiment contract")
    summary: dict[str, Any] = {
        "status": "planned",
        "action": action,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "phase": manifest["phase"],
        "expected_trials": len(manifest["trials"]),
        "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
        "approved_cost_usd": str(approved),
        "required_confirmation": _confirmation(action, manifest),
    }
    if action == "plan":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if action == "preflight" and manifest["phase"] != "preflight":
        raise ValueError("preflight action requires the exact preflight phase manifest")
    if action == "train" and manifest["phase"] not in TRAIN_PHASES:
        raise ValueError("train action requires a registered training phase")
    if action == "promote" and (
        manifest["phase"] != "sweep" or manifest["asha"]["rung_epochs"] not in {1, 3}
    ):
        raise ValueError("promote action requires a non-final ASHA sweep rung")
    if action == "freeze" and manifest["phase"] != "stability":
        raise ValueError("freeze action requires the exact stability phase manifest")
    if action in {"preflight", "train", "freeze"} and confirm != _confirmation(action, manifest):
        raise RuntimeError(f"refusing launch: pass --confirm {_confirmation(action, manifest)}")
    if action == "promote":
        if confirm != _confirmation(action, manifest):
            raise RuntimeError(
                f"refusing promotion: pass --confirm {_confirmation(action, manifest)}"
            )
        summary["promotion"] = promote_asha_rung.remote(manifest)
        summary["status"] = "promoted"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if action == "validate-sharded":
        calls = [inspect_trial.spawn(manifest, trial["trial_id"]) for trial in manifest["trials"]]
        inspection = _aggregate_trial_inspections(manifest, [call.get() for call in calls])
        summary["inspection"] = inspection
        if inspection["missing_trial_ids"] or inspection["resumable_trial_ids"]:
            raise RuntimeError("phase validation requires every exact trial to be complete")
        summary["status"] = "validated"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if action == "freeze":
        # The freezer performs the same exact receipt/artifact validation before
        # scoring. Avoid a duplicate serial hash pass over the large checkpoints.
        summary["selection"] = freeze_stability_selection.remote(manifest)
        summary["status"] = "frozen"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    inspection = inspect_run.remote(manifest)
    summary["inspection"] = inspection
    if action == "validate":
        if inspection["missing_trial_ids"] or inspection["resumable_trial_ids"]:
            raise RuntimeError("phase validation requires every exact trial to be complete")
        summary["status"] = "validated"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    pending = [(trial_id, False) for trial_id in inspection["missing_trial_ids"]] + [
        (trial_id, True) for trial_id in inspection["resumable_trial_ids"]
    ]
    call_ids = []
    for trial_id, resume in pending:
        trial = by_id[trial_id]
        job = _trial_job(manifest, trial)
        if action == "preflight":
            function = preflight_l4
            call_ids.append(function.spawn(job, resume=resume).object_id)
        else:
            function = train_l4
            call_ids.append(function.spawn(job, resume=resume).object_id)
    summary["status"] = "submitted" if call_ids else "already_complete"
    summary["submitted_function_call_ids"] = call_ids
    print(json.dumps(summary, indent=2, sort_keys=True))
