"""Modal boundary for the one-shot ModernBERT acquisition-policy comparison.

This module is deliberately narrow: it constructs the unseen eligible frame, runs
the six already-published B4 checkpoints over two frozen text renderings, and
publishes the selector-authoritative private teacher handoff.  It does *not* train a new
student, decode corpus predictions, read locked labels, or make a promotion
claim.  Row text, opaque identifiers, logits and the acquisition ledger remain
private on the Modal Volume; every returned/public object is metadata only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-modernbert-acquisition-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
DISPATCH_LEASE_DICT_NAME = "reddit-china-stance-modernbert-acquisition-v1-dispatch-leases"
OUTPUT_PREFIX = Path("student-modernbert-acquisition-v1")
POLICY_REPO_PATH = Path("configs/modernbert-acquisition-v1.toml")
POLICY_RUNTIME_PATH = Path("/configs/modernbert-acquisition-v1.toml")
RETAINED_MANIFEST_REPO_PATH = Path(
    "data/private-modernbert-factorised-v2/prepare-publication-contract-v5/run-manifest.json"
)
RETAINED_SOURCE_BUNDLE_REPO_PATH = Path(
    "data/private-modernbert-factorised-v2/prepare-publication-contract-v5/source-bundle.json"
)
RETAINED_MANIFEST_RUNTIME_PATH = Path("/configs/retained-factorised-v5-run-manifest.json")
RETAINED_SOURCE_BUNDLE_RUNTIME_PATH = Path("/configs/retained-factorised-v5-source-bundle.json")
SEMANTIC_V2_SCHEMA_REPO_PATH = Path("schemas/target-stance-v2-pilot.schema.json")
SEMANTIC_V2_SCHEMA_RUNTIME_PATH = Path("/schemas/target-stance-v2-pilot.schema.json")

DATASET_REVISION = "97627c42893bc479c6a4952b1db38bbc33a60ac8"
SOURCE_SCHEMA_VERSION = "1.0.2"
RETRIEVAL_POLICY_DIGEST = "9e8c9050b9512fd83b2a844d4259bd3df8e3f84dc213460fc19407238266769a"
STAGE_A_RUN_ID = "834ff757fb7931e508c782d64db4f9e8761e4273edb8b2635c9c5bff84c406b1"
LANGUAGE_POLICY_DIGEST = "83857eb8b501d6bf262461c4ea2815e6649c9581acc84a031558f5cf83483be2"
LANGUAGE_RUN_ID = "34dbfdea18a7dafe84b8d5be891d6c7771a0ef71e67f5878f0ecc7130229a7b1"
BASE_FACTORISED_RUN_ID = "07527fe72e187bbcfd24bc893c034283ac577e02a54d8871cbdc740f7d35fcc7"

MAX_CONCURRENT_CHECKPOINTS = 6
ACCOUNT_GPU_LIMIT = 10
SCORING_SHARD_COUNT = 10
SCORING_BATCH_SIZE = 32
SCORING_MAX_CONTAINERS = 10
SCORING_RENDERS = ("full", "target_only")
HARD_MAX_APPROVAL_USD = Decimal("200")
ALLOWED_GPUS = ("L4",)
INFERENCE_TIMEOUT_SECONDS = 14_400
# A ceiling, not a scheduling mechanism.  Work is partitioned below so one
# queued 64 GiB container can never consume the entire useful prepare budget.
CPU_PREPARE_TIMEOUT_SECONDS = 21_600
PREPARE_WORKER_CPU = 8
PREPARE_WORKER_MEMORY_MIB = 24_576
PREPARE_MAX_CONTAINERS = 10
PREPARE_REDUCE_CPU = 8
PREPARE_REDUCE_MEMORY_MIB = 40_960
PREPARE_BENCHMARK = {
    "fixed_cells": ["subreddit=China/year=2020", "largest-input-cell"],
    "worker_cpus": [4, 8],
    "worker_memory_mib": [12_288, 24_576],
    "semantic_contract": "identical-to-prepare-shard-v1",
    "execute_cloud_work": True,
}
PREPARED_KIND = "modernbert-acquisition-prepared-v1"
SCORING_KIND = "modernbert-acquisition-checkpoint-logits-v1"
SCORING_SHARD_KIND = "modernbert-acquisition-checkpoint-logit-shard-v1"
SCORING_INDEX_KIND = "modernbert-acquisition-checkpoint-logit-index-v1"
CANDIDATE_SCORE_KIND = "modernbert-acquisition-candidate-score-shard-v1"
PREFLIGHT_KIND = "modernbert-acquisition-cuda-preflight-v1"
CLOSEOUT_KIND = "modernbert-acquisition-packet-receipt-v1"
PREFLIGHT_LOGIT_TOLERANCE = 1e-5
INPUT_SPEC_KIND = "modernbert-acquisition-input-v1"
COMPUTE_LEDGER_KIND = "modernbert-acquisition-compute-ledger-v1"
INPUT_SPEC_KEYS = {
    "schema_version",
    "kind",
    "policy_sha256",
    "base_factorised_run_id",
    "retained_manifest_sha256",
    "retained_source_bundle_sha256",
    "compute",
    "prepared_run_id",
}
DEFAULT_INPUT_SPEC_PATH = Path("data/private-modernbert-acquisition-v1/input.json")
DEFAULT_PREPARED_INPUT_SPEC_PATH = Path(
    "data/private-modernbert-acquisition-v1/prepared-input.json"
)
DEFAULT_PREPARE_RECEIPT_PATH = Path("data/private-modernbert-acquisition-v1/prepare-receipt.json")
MUTATING_ACTIONS = ("prepare", "cuda-preflight", "launch-scoring", "closeout")
PHASE_COST_ACTIONS = MUTATING_ACTIONS
SCORING_JOB_KEYS = {
    "prepared_run_id",
    "checkpoint",
    "checkpoint_bundle_sha256",
    "phase_approval_claim_id",
    "checkpoint_launch_claim_id",
    "approved_cost_usd",
    "estimated_upper_cost_usd",
    "runtime_source_bundle_sha256",
}
SCORING_SHARD_JOB_KEYS = SCORING_JOB_KEYS | {
    "scoring_plan_sha256",
    "scoring_shard",
    "batch_size",
    "attempt_id",
    "dispatch_id",
}
CANDIDATE_REDUCER_JOB_KEYS = {
    "spec",
    "scoring_plan_sha256",
    "scoring_shard",
    "attempt_id",
    "dispatch_id",
}
FINALISER_JOB_KEYS = {"spec", "attempt_id", "dispatch_id"}
SCORING_PHASE_APPROVAL_KEYS = {
    "kind",
    "schema_version",
    "action",
    "prepared_run_id",
    "estimated_upper_cost_usd",
    "approved_cost_usd",
    "cumulative_measured_spend_usd",
    "active_reservation_usd",
    "compute_ledger_sha256",
    "runtime_source_bundle_sha256",
    "checkpoint_bundle_sha256",
    "locked_test_rows_accessed",
    "claim_id",
}
SCORING_CHECKPOINT_CLAIM_KEYS = {
    "kind",
    "schema_version",
    "action",
    "prepared_run_id",
    "component",
    "optimiser_seed",
    "checkpoint_sha256",
    "checkpoint_descriptor_sha256",
    "checkpoint_bundle_sha256",
    "phase_approval_claim_id",
    "estimated_upper_cost_usd",
    "approved_cost_usd",
    "runtime_source_bundle_sha256",
    "status",
    "locked_test_rows_accessed",
    "claim_id",
}

STAGE_ROOT = (
    Path("derived/stage-a")
    / DATASET_REVISION
    / f"source-schema={SOURCE_SCHEMA_VERSION}"
    / f"policy={RETRIEVAL_POLICY_DIGEST}"
    / f"run={STAGE_A_RUN_ID}"
)
LANGUAGE_ROOT = (
    STAGE_ROOT / "language" / f"policy={LANGUAGE_POLICY_DIGEST}" / f"run={LANGUAGE_RUN_ID}"
)

RUNTIME_DEPENDENCIES = {
    "duckdb": "1.4.4",
    "jsonschema": "4.26.0",
    "pydantic": "2.13.4",
    "pyarrow": "25.0.1",
    "torch": "2.8.0",
    "transformers": "4.57.6",
    "safetensors": "0.8.0",
    "huggingface-hub": "0.36.2",
}
REQUIRED_SOURCE_FILES = (
    "configs/modernbert-acquisition-v1.toml",
    "schemas/target-stance-v2-pilot.schema.json",
    "src/reddit_china_stance/modal_modernbert_acquisition.py",
    "src/reddit_china_stance/modernbert_acquisition.py",
    "src/reddit_china_stance/context_assembly.py",
    "src/reddit_china_stance/modernbert_factorised_data.py",
    "src/reddit_china_stance/modernbert_factorised_experiment.py",
    "src/reddit_china_stance/modernbert_factorised_model.py",
    "src/reddit_china_stance/modernbert_factorised_training.py",
    "src/reddit_china_stance/modernbert_trainer.py",
    "src/reddit_china_stance/modal_sol_teacher_packet.py",
    "src/reddit_china_stance/sol_teacher_acquisition_v2.py",
    "src/reddit_china_stance/privacy.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
)
RUNTIME_FILE_MOUNTS = (
    (POLICY_REPO_PATH, POLICY_RUNTIME_PATH),
    (RETAINED_MANIFEST_REPO_PATH, RETAINED_MANIFEST_RUNTIME_PATH),
    (RETAINED_SOURCE_BUNDLE_REPO_PATH, RETAINED_SOURCE_BUNDLE_RUNTIME_PATH),
    (SEMANTIC_V2_SCHEMA_REPO_PATH, SEMANTIC_V2_SCHEMA_RUNTIME_PATH),
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
dispatch_leases = modal.Dict.from_name(
    DISPATCH_LEASE_DICT_NAME,
    environment_name=ENVIRONMENT_NAME,
    create_if_missing=True,
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*(f"{name}=={version}" for name, version in RUNTIME_DEPENDENCIES.items()))
    .run_commands(
        'python -c "from huggingface_hub import snapshot_download; '
        "snapshot_download(repo_id='answerdotai/ModernBERT-large', "
        "revision='45bb4654a4d5aaff24dd11d4781fa46d39bf8c13')\""
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_python_source("reddit_china_stance")
)
for _local_file, _runtime_file in RUNTIME_FILE_MOUNTS:
    image = image.add_local_file(str(_local_file), remote_path=str(_runtime_file))


def _acquisition_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_acquisition")


def _training_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_factorised_training")


def _data_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_factorised_data")


def _experiment_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_factorised_experiment")


def _packet_module() -> Any:
    return importlib.import_module("reddit_china_stance.modal_sol_teacher_packet")


def _runtime_source_bundle() -> dict[str, Any]:
    """Content-address the exact code/configuration executed by Modal.

    ``add_local_python_source`` does not preserve a repository checkout layout in
    the image.  Resolve module files through their imports, while retaining stable
    repository-relative logical names in the bundle.
    """

    module_paths = {
        "src/reddit_china_stance/modal_modernbert_acquisition.py": Path(__file__),
        "src/reddit_china_stance/modernbert_acquisition.py": Path(_acquisition_module().__file__),
        "src/reddit_china_stance/context_assembly.py": Path(
            importlib.import_module("reddit_china_stance.context_assembly").__file__
        ),
        "src/reddit_china_stance/modernbert_factorised_data.py": Path(_data_module().__file__),
        "src/reddit_china_stance/modernbert_factorised_experiment.py": Path(
            _experiment_module().__file__
        ),
        "src/reddit_china_stance/modernbert_factorised_model.py": Path(
            importlib.import_module("reddit_china_stance.modernbert_factorised_model").__file__
        ),
        "src/reddit_china_stance/modernbert_factorised_training.py": Path(
            _training_module().__file__
        ),
        "src/reddit_china_stance/modernbert_trainer.py": Path(
            importlib.import_module("reddit_china_stance.modernbert_trainer").__file__
        ),
        "src/reddit_china_stance/modal_sol_teacher_packet.py": Path(_packet_module().__file__),
        "src/reddit_china_stance/sol_teacher_acquisition_v2.py": Path(
            importlib.import_module("reddit_china_stance.sol_teacher_acquisition_v2").__file__
        ),
        "src/reddit_china_stance/privacy.py": Path(
            importlib.import_module("reddit_china_stance.privacy").__file__
        ),
        "src/reddit_china_stance/semantic_ontology_v2.py": Path(
            importlib.import_module("reddit_china_stance.semantic_ontology_v2").__file__
        ),
        "configs/modernbert-acquisition-v1.toml": _policy_path(),
        "schemas/target-stance-v2-pilot.schema.json": _semantic_v2_schema_path(),
    }
    if set(module_paths) != set(REQUIRED_SOURCE_FILES) or any(
        not path.is_file() for path in module_paths.values()
    ):
        raise RuntimeError("acquisition runtime source inventory is incomplete")
    body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-acquisition-runtime-source-bundle-v1",
        "files": {logical: _file_sha256(path) for logical, path in sorted(module_paths.items())},
        "runtime_dependencies": dict(sorted(RUNTIME_DEPENDENCIES.items())),
    }
    return {**body, "source_bundle_id": _canonical_sha256(body)}


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError("contract must be finite JSON") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _safe_relative(value: Any, *, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty Volume-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{where} must be a safe Volume-relative path")
    return path


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{where} must be JSON") from error
    if not isinstance(result, dict):
        raise ValueError(f"{where} must contain an object")
    return result


def _policy_path() -> Path:
    """Resolve the immutable policy in either local or Modal execution."""

    if POLICY_RUNTIME_PATH.is_file():
        return POLICY_RUNTIME_PATH
    resolved = Path(__file__).resolve()
    if len(resolved.parents) > 2:
        local = resolved.parents[2] / POLICY_REPO_PATH
        if local.is_file():
            return local
    raise FileNotFoundError("frozen ModernBERT acquisition policy is unavailable")


def _semantic_v2_schema_path() -> Path:
    """Resolve the v2 label schema from its exact Modal mount or local checkout."""

    if SEMANTIC_V2_SCHEMA_RUNTIME_PATH.is_file():
        return SEMANTIC_V2_SCHEMA_RUNTIME_PATH
    resolved = Path(__file__).resolve()
    if len(resolved.parents) > 2:
        local = resolved.parents[2] / SEMANTIC_V2_SCHEMA_REPO_PATH
        if local.is_file():
            return local
    raise FileNotFoundError("target-stance v2 label schema is unavailable")


def _retained_evidence_paths() -> tuple[Path, Path]:
    """Resolve the retained v5 manifest and its exact sibling source bundle."""

    if RETAINED_MANIFEST_RUNTIME_PATH.is_file() and RETAINED_SOURCE_BUNDLE_RUNTIME_PATH.is_file():
        return RETAINED_MANIFEST_RUNTIME_PATH, RETAINED_SOURCE_BUNDLE_RUNTIME_PATH
    resolved = Path(__file__).resolve()
    if len(resolved.parents) > 2:
        repo_root = resolved.parents[2]
        local_manifest = repo_root / RETAINED_MANIFEST_REPO_PATH
        local_bundle = repo_root / RETAINED_SOURCE_BUNDLE_REPO_PATH
        if local_manifest.is_file() and local_bundle.is_file():
            return local_manifest, local_bundle
    raise FileNotFoundError("retained factorised-v5 manifest/source bundle is unavailable")


def _validate_retained_evidence(
    *, manifest_path: Path | None = None, source_bundle_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the local retained authority without reading private row artefacts."""

    if (manifest_path is None) != (source_bundle_path is None):
        raise ValueError("retained manifest and source bundle paths must be supplied together")
    if manifest_path is None or source_bundle_path is None:
        manifest_path, source_bundle_path = _retained_evidence_paths()
    manifest = _training_module().validate_run_manifest(
        _read_json(manifest_path, where="retained factorised-v5 run manifest")
    )
    source_bundle = _experiment_module().validate_source_bundle(
        _read_json(source_bundle_path, where="retained factorised-v5 source bundle")
    )
    bindings = manifest.get("experiment_contract", {}).get("bindings", {})
    if (
        manifest.get("experiment_run_id") != BASE_FACTORISED_RUN_ID
        or bindings.get("source_bundle") != source_bundle
        or bindings.get("source_bundle_sha256") != _canonical_sha256(source_bundle)
    ):
        raise RuntimeError("retained factorised-v5 manifest/source bundle binding drifted")
    expected_trials = {
        (component, seed)
        for component in ("relevance", "target_stance_b4", "target_stance_b2")
        for seed in (47, 61, 89)
    }
    actual_trials = {
        (trial.get("component"), trial.get("optimiser_seed"))
        for trial in manifest.get("trials", [])
        if isinstance(trial, Mapping)
    }
    if actual_trials != expected_trials or manifest.get("locked_test_rows_accessed") != 0:
        raise RuntimeError("retained factorised-v5 trial inventory drifted")
    return manifest, source_bundle


def load_frozen_policy() -> tuple[dict[str, Any], str]:
    """Load and validate the sole acquisition-policy source of truth."""

    path = _policy_path()
    # The pure selection module owns schema validation.  This Modal boundary must
    # bind the exact same policy file, rather than carrying a second validator.
    acquisition, _gate = _acquisition_module().load_acquisition_policies(path)
    value = json.loads(
        json.dumps(
            {
                "source": {
                    "dataset_revision": DATASET_REVISION,
                    "source_schema_version": SOURCE_SCHEMA_VERSION,
                    "retrieval_policy_digest": RETRIEVAL_POLICY_DIGEST,
                    "stage_a_run_id": STAGE_A_RUN_ID,
                    "language_policy_digest": LANGUAGE_POLICY_DIGEST,
                    "language_run_id": LANGUAGE_RUN_ID,
                    "language_eligibility": "provisional_english",
                    "language_decision": "explicit-thesis-main-corpus-decision-2026-08-26",
                    "human_language_gate_accepted": False,
                }
            }
        )
    )
    if value["source"] != {
        "dataset_revision": DATASET_REVISION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "retrieval_policy_digest": RETRIEVAL_POLICY_DIGEST,
        "stage_a_run_id": STAGE_A_RUN_ID,
        "language_policy_digest": LANGUAGE_POLICY_DIGEST,
        "language_run_id": LANGUAGE_RUN_ID,
        "language_eligibility": "provisional_english",
        "language_decision": "explicit-thesis-main-corpus-decision-2026-08-26",
        "human_language_gate_accepted": False,
    }:
        raise ValueError("acquisition source policy drifted")
    # Reparse only to retain the complete immutable document in receipts.  The
    # loader above has already failed closed on every field.
    import tomllib

    full = tomllib.loads(path.read_text(encoding="utf-8"))
    if full["source"] != value["source"]:
        raise ValueError("acquisition source policy drifted")
    if acquisition.policy_file_sha256 != _file_sha256(path):
        raise RuntimeError("acquisition policy loader/file digest disagreement")
    return full, acquisition.policy_file_sha256


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"existing immutable output differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _descriptor(path: Path, *, root: Path, row_count: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, Any] = {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        value["row_count"] = row_count
    return value


def enforce_cost_guardrail(*, estimated_cost_usd: Decimal, approved_cost_usd: Decimal) -> None:
    if (
        not approved_cost_usd.is_finite()
        or approved_cost_usd <= 0
        or approved_cost_usd > HARD_MAX_APPROVAL_USD
    ):
        raise ValueError("approved cost must be finite, positive and <= 200")
    if not estimated_cost_usd.is_finite() or estimated_cost_usd < 0:
        raise ValueError("estimated cost must be finite and non-negative")
    if estimated_cost_usd > approved_cost_usd:
        raise RuntimeError("estimated acquisition cost exceeds approved cost")


def _compute_ledger_projection(compute: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "kind": COMPUTE_LEDGER_KIND,
        "cumulative_measured_spend_usd": compute["cumulative_measured_spend_usd"],
        "active_reservation_usd": compute["active_reservation_usd"],
        "phase_upper_cost_usd": dict(compute["phase_upper_cost_usd"]),
        "hard_cost_cap_usd": compute["hard_cost_cap_usd"],
    }


def validate_compute_contract(contract: Mapping[str, Any]) -> None:
    compute = contract.get("compute")
    expected_keys = {
        "allowed_gpus",
        "account_gpu_limit",
        "max_concurrent_checkpoints",
        "gpu_fallback_allowed",
        "rate_card_usd_per_gpu_second",
        "cumulative_measured_spend_usd",
        "active_reservation_usd",
        "phase_upper_cost_usd",
        "hard_cost_cap_usd",
        "remaining_after_plan_usd",
        "ledger_sha256",
    }
    if not isinstance(compute, Mapping) or set(compute) != expected_keys:
        raise ValueError("acquisition compute contract schema drifted")
    if (
        compute["allowed_gpus"] != list(ALLOWED_GPUS)
        or compute["account_gpu_limit"] != ACCOUNT_GPU_LIMIT
        or compute["max_concurrent_checkpoints"] != MAX_CONCURRENT_CHECKPOINTS
        or compute["gpu_fallback_allowed"] is not False
    ):
        raise ValueError("acquisition compute binding drifted")
    if MAX_CONCURRENT_CHECKPOINTS > ACCOUNT_GPU_LIMIT:
        raise ValueError("acquisition checkpoint concurrency exceeds account limit")
    rates = compute["rate_card_usd_per_gpu_second"]
    if not isinstance(rates, Mapping) or set(rates) != {"L4"}:
        raise ValueError("acquisition rate card binding drifted")
    phases = compute["phase_upper_cost_usd"]
    if not isinstance(phases, Mapping) or set(phases) != set(PHASE_COST_ACTIONS):
        raise ValueError("acquisition phase-cost schema drifted")
    try:
        rate, measured, reserved, cap, remaining = (
            Decimal(str(rates["L4"])),
            Decimal(str(compute["cumulative_measured_spend_usd"])),
            Decimal(str(compute["active_reservation_usd"])),
            Decimal(str(compute["hard_cost_cap_usd"])),
            Decimal(str(compute["remaining_after_plan_usd"])),
        )
        phase_costs = [Decimal(str(phases[action])) for action in PHASE_COST_ACTIONS]
    except (InvalidOperation, KeyError, ValueError) as error:
        raise ValueError("acquisition cost binding is invalid") from error
    if (
        not rate.is_finite()
        or rate <= 0
        or any(
            not value.is_finite() or value < 0
            for value in (measured, reserved, remaining, *phase_costs)
        )
        or not cap.is_finite()
        or cap <= 0
    ):
        raise ValueError("acquisition costs must be finite and non-negative")
    planned = sum(phase_costs, Decimal("0"))
    if cap != HARD_MAX_APPROVAL_USD:
        raise ValueError("acquisition compute contract must bind the frozen $200 hard cap")
    if measured + reserved + planned > cap:
        raise RuntimeError("acquisition experiment exceeds the frozen total cost cap")
    if remaining != cap - measured - reserved - planned:
        raise ValueError("acquisition remaining cost binding drifted")
    ledger_sha256 = _sha256(
        compute.get("ledger_sha256"), where="acquisition compute ledger SHA-256"
    )
    if ledger_sha256 != _canonical_sha256(_compute_ledger_projection(compute)):
        raise ValueError("acquisition compute ledger content address drifted")


def _phase_cost(spec: Mapping[str, Any], *, action: str) -> Decimal:
    if action not in PHASE_COST_ACTIONS:
        raise ValueError("action has no mutable phase-cost binding")
    validate_compute_contract(spec)
    try:
        value = Decimal(str(spec["compute"]["phase_upper_cost_usd"][action]))
    except (InvalidOperation, KeyError, TypeError, ValueError) as error:
        raise ValueError("acquisition phase-cost binding is invalid") from error
    return value


def _validate_phase_approval(
    spec: Mapping[str, Any], *, action: str, approved_cost_usd: str
) -> tuple[Decimal, Decimal]:
    try:
        approved = Decimal(approved_cost_usd)
    except InvalidOperation as error:
        raise ValueError("phase approval must be a decimal amount") from error
    estimate = _phase_cost(spec, action=action)
    enforce_cost_guardrail(estimated_cost_usd=estimate, approved_cost_usd=approved)
    if approved != estimate:
        raise ValueError("phase approval must exactly match the immutable phase maximum")
    return estimate, approved


def validate_input_spec(value: Mapping[str, Any], *, action: str | None = None) -> dict[str, Any]:
    """Validate the exact launch-spec surface before any local or remote work."""

    if not isinstance(value, Mapping) or set(value) != INPUT_SPEC_KEYS:
        raise ValueError("acquisition input spec top-level schema drifted")
    clean = dict(value)
    if clean.get("schema_version") != "1.0.0" or clean.get("kind") != INPUT_SPEC_KIND:
        raise ValueError("acquisition input spec identity drifted")
    _sha256(clean.get("policy_sha256"), where="acquisition policy SHA-256")
    if clean.get("base_factorised_run_id") != BASE_FACTORISED_RUN_ID:
        raise ValueError("acquisition input must bind the retained factorised run")
    _sha256(clean.get("retained_manifest_sha256"), where="retained manifest SHA-256")
    _sha256(
        clean.get("retained_source_bundle_sha256"),
        where="retained source bundle SHA-256",
    )
    validate_compute_contract(clean)
    prepared_run_id = clean.get("prepared_run_id")
    if prepared_run_id is not None:
        _sha256(prepared_run_id, where="prepared run ID")
    if action in {"prepare", "validate-volume"} and prepared_run_id is not None:
        raise ValueError(f"{action} requires prepared_run_id to be null")
    if action in {
        "cuda-preflight",
        "launch-scoring",
        "validate-scoring",
        "inspect",
        "closeout",
    } and (prepared_run_id is None):
        raise ValueError(f"{action} requires a prepared_run_id")
    return clean


def _launch_spec_sha256(spec: Mapping[str, Any]) -> str:
    clean = validate_input_spec(spec)
    return _canonical_sha256(
        {key: clean[key] for key in sorted(INPUT_SPEC_KEYS - {"prepared_run_id"})}
    )


def _validate_compute_ledger(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "cumulative_measured_spend_usd",
        "active_reservation_usd",
        "phase_upper_cost_usd",
        "hard_cost_cap_usd",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("authoritative compute ledger schema drifted")
    if value.get("schema_version") != "1.0.0" or value.get("kind") != COMPUTE_LEDGER_KIND:
        raise ValueError("authoritative compute ledger identity drifted")
    phases = value.get("phase_upper_cost_usd")
    if not isinstance(phases, Mapping) or set(phases) != set(PHASE_COST_ACTIONS):
        raise ValueError("authoritative compute ledger phase schema drifted")
    try:
        numbers = [
            Decimal(str(value["cumulative_measured_spend_usd"])),
            Decimal(str(value["active_reservation_usd"])),
            Decimal(str(value["hard_cost_cap_usd"])),
            *(Decimal(str(phases[action])) for action in PHASE_COST_ACTIONS),
        ]
    except (InvalidOperation, KeyError, TypeError, ValueError) as error:
        raise ValueError("authoritative compute ledger values are invalid") from error
    if any(not number.is_finite() or number < 0 for number in numbers):
        raise ValueError("authoritative compute ledger values must be finite and non-negative")
    if numbers[2] != HARD_MAX_APPROVAL_USD:
        raise ValueError("authoritative compute ledger must bind the frozen $200 hard cap")
    if any(number <= 0 for number in numbers[3:]):
        raise ValueError("every mutable phase must have a positive explicit upper cost")
    return dict(value)


def build_local_input_spec(
    *,
    compute_ledger_path: Path,
    manifest_path: Path | None = None,
    source_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Build one deterministic spec from retained v5 evidence and an explicit ledger.

    The compute ledger is mandatory: this builder never guesses cumulative spend,
    active reservations, or CPU/GPU phase maxima.
    """

    manifest, source_bundle = _validate_retained_evidence(
        manifest_path=manifest_path, source_bundle_path=source_bundle_path
    )
    ledger = _validate_compute_ledger(
        _read_json(compute_ledger_path, where="authoritative acquisition compute ledger")
    )
    retained_compute = manifest["experiment_contract"]["compute"]
    rates = retained_compute.get("rate_card_usd_per_gpu_second")
    if not isinstance(rates, Mapping) or set(rates) != {"L4"}:
        raise RuntimeError("retained factorised-v5 rate card is unavailable")
    measured = Decimal(str(ledger["cumulative_measured_spend_usd"]))
    reserved = Decimal(str(ledger["active_reservation_usd"]))
    retained_measured_floor = Decimal(str(retained_compute.get("cumulative_measured_spend_usd")))
    if measured < retained_measured_floor:
        raise ValueError(
            "authoritative cumulative spend is below the retained manifest's measured floor"
        )
    phase_costs = {
        action: str(ledger["phase_upper_cost_usd"][action]) for action in PHASE_COST_ACTIONS
    }
    cap = Decimal(str(ledger["hard_cost_cap_usd"]))
    planned = sum((Decimal(value) for value in phase_costs.values()), Decimal("0"))
    remaining = cap - measured - reserved - planned
    _policy, policy_sha256 = load_frozen_policy()
    result = {
        "schema_version": "1.0.0",
        "kind": INPUT_SPEC_KIND,
        "policy_sha256": policy_sha256,
        "base_factorised_run_id": BASE_FACTORISED_RUN_ID,
        "retained_manifest_sha256": _canonical_sha256(manifest),
        "retained_source_bundle_sha256": _canonical_sha256(source_bundle),
        "compute": {
            "allowed_gpus": list(ALLOWED_GPUS),
            "account_gpu_limit": ACCOUNT_GPU_LIMIT,
            "max_concurrent_checkpoints": MAX_CONCURRENT_CHECKPOINTS,
            "gpu_fallback_allowed": False,
            "rate_card_usd_per_gpu_second": dict(rates),
            "cumulative_measured_spend_usd": str(ledger["cumulative_measured_spend_usd"]),
            "active_reservation_usd": str(ledger["active_reservation_usd"]),
            "phase_upper_cost_usd": phase_costs,
            "hard_cost_cap_usd": str(ledger["hard_cost_cap_usd"]),
            "remaining_after_plan_usd": format(remaining, "f"),
            "ledger_sha256": _canonical_sha256(ledger),
        },
        "prepared_run_id": None,
    }
    return validate_input_spec(result, action="prepare")


def _discover_inputs(*, volume_root: Path = VOLUME_PATH) -> dict[str, list[Path]]:
    candidates = sorted(
        (volume_root / STAGE_ROOT / "candidates").glob(
            "subreddit=*/year=*/unit=*/candidates-*.parquet"
        )
    )
    languages = sorted(
        (volume_root / LANGUAGE_ROOT / "decisions").glob(
            "subreddit=*/year=*/unit=*/language-decisions-*.parquet"
        )
    )
    canonical = sorted(
        (volume_root / "normalised" / DATASET_REVISION / f"schema={SOURCE_SCHEMA_VERSION}").glob(
            "*/year=*/subreddit=*/content_type=*/part-00000.parquet"
        )
    )
    if len(candidates) != 60 or len(languages) != 60 or len(canonical) != 120:
        raise RuntimeError(
            "input inventory mismatch: "
            f"candidates={len(candidates)}, languages={len(languages)}, "
            f"canonical={len(canonical)}"
        )
    if any(
        ".incomplete" in str(path) or ".publishing" in str(path)
        for path in (*candidates, *languages, *canonical)
    ):
        raise RuntimeError("input inventory contains incomplete output")
    return {"candidates": candidates, "languages": languages, "canonical": canonical}


def _inventory(paths: Sequence[Path], *, volume_root: Path = VOLUME_PATH) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(volume_root).as_posix(),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _receipt_backed_inventory(
    paths: Sequence[Path], *, producer: str, volume_root: Path = VOLUME_PATH
) -> list[dict[str, Any]]:
    """Inventory immutable upstream inputs without re-hashing their payloads.

    Canonical normalisation, candidate and language stages already publish
    content-addressed parquet descriptors.  Their path, byte size and Parquet
    footer row count are sufficient to bind that evidence here; a later reader
    validates the descriptor when it actually consumes a shard.  This avoids a
    second full-byte scan merely to make an inventory.
    """
    if producer not in {"candidates", "languages", "canonical"}:
        raise ValueError("unsupported upstream producer")
    result: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file() or any(part.startswith(".") for part in path.parts):
            raise RuntimeError("receipt-backed inventory contains unpublished input")
        relative = path.relative_to(volume_root).as_posix()
        if producer == "canonical":
            # canonical layout: <source>/year=.../subreddit=.../content_type=.../part.parquet
            source_root = path.parents[3]
            receipt_path = source_root / "_receipt.json"
            if not receipt_path.is_file():
                raise RuntimeError(f"canonical producer receipt is absent: {relative}")
            receipt = _read_json(receipt_path, where="canonical producer receipt")
            partition_relative = path.relative_to(source_root).as_posix()
            matches = [
                row
                for row in receipt.get("partitions", ())
                if isinstance(row, Mapping) and row.get("relative_path") == partition_relative
            ]
            if (
                receipt.get("status") != "converted"
                or receipt.get("revision") != DATASET_REVISION
                or receipt.get("source_schema_version") != SOURCE_SCHEMA_VERSION
                or len(matches) != 1
            ):
                raise RuntimeError(f"canonical producer receipt drifted: {relative}")
            partition = matches[0]
            descriptor = {
                "sha256": partition.get("sha256"),
                "bytes": partition.get("bytes"),
                "row_count": partition.get("rows"),
            }
        else:
            receipts = sorted(path.parent.glob("receipt-*.json"))
            if len(receipts) != 1:
                raise RuntimeError(f"expected one content-addressed producer receipt: {relative}")
            receipt_path = receipts[0]
            receipt_sha256 = _file_sha256(receipt_path)
            if receipt_path.name != f"receipt-{receipt_sha256}.json":
                raise RuntimeError(f"producer receipt filename digest drifted: {relative}")
            receipt = _read_json(receipt_path, where="upstream producer receipt")
            if producer == "candidates":
                output = receipt.get("output", {})
                if receipt.get("status") != "complete" or output.get("file") != path.name:
                    raise RuntimeError(f"candidate producer receipt drifted: {relative}")
            else:
                output = receipt.get("outputs", {}).get("decisions", {})
                if (
                    receipt.get("status") != "complete_pending_human_language_audit"
                    or output.get("file") != path.name
                ):
                    raise RuntimeError(f"language producer receipt drifted: {relative}")
            descriptor = {
                "sha256": output.get("sha256"),
                "bytes": output.get("bytes"),
                "row_count": output.get("rows"),
            }
        if (
            not isinstance(descriptor.get("sha256"), str)
            or len(descriptor["sha256"]) != 64
            or descriptor.get("bytes") != path.stat().st_size
            or not isinstance(descriptor.get("row_count"), int)
            or descriptor["row_count"] < 0
        ):
            raise RuntimeError(f"producer receipt descriptor is incomplete: {relative}")
        result.append(
            {
                "relative_path": relative,
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "row_count": descriptor["row_count"],
                "producer_receipt_sha256": _file_sha256(receipt_path),
            }
        )
    return result


def _run_root(run_id: str, *, volume_root: Path = VOLUME_PATH) -> Path:
    return volume_root / OUTPUT_PREFIX / f"run={_sha256(run_id, where='run ID')}"


def _prepared_root(run_id: str, *, volume_root: Path = VOLUME_PATH) -> Path:
    return _run_root(run_id, volume_root=volume_root) / "prepared"


def _prepare_stage_root(
    pre_prepare_id: str, *, volume_root: Path | None = None
) -> Path:
    active_root = VOLUME_PATH if volume_root is None else volume_root
    return (
        active_root
        / OUTPUT_PREFIX
        / "prepare-stages"
        / f"run={_sha256(pre_prepare_id, where='pre-prepare ID')}"
    )


def _pre_prepare_id(spec: Mapping[str, Any], inventories: Mapping[str, Any]) -> str:
    """Stable coordinator namespace, available before any corpus materialisation."""
    return _canonical_sha256(
        {
            "kind": "modernbert-acquisition-prepare-stages-v1",
            "launch_spec_sha256": _launch_spec_sha256(spec),
            "runtime_source_bundle_sha256": _runtime_source_bundle()["source_bundle_id"],
            "inventories": inventories,
        }
    )


def _prepare_shards(paths: Mapping[str, Sequence[Path]]) -> list[dict[str, Any]]:
    """Exact source-cell shards; each candidate/language cell is a work unit."""

    def labels(path: Path) -> dict[str, str]:
        return {
            key: value for part in path.parts if "=" in part for key, value in [part.split("=", 1)]
        }

    candidates: dict[tuple[str, str], list[Path]] = {}
    languages: dict[tuple[str, str], list[Path]] = {}
    for path in paths["candidates"]:
        parts = labels(path)
        candidates.setdefault((parts["subreddit"], parts["year"]), []).append(path)
    for path in paths["languages"]:
        parts = labels(path)
        languages.setdefault((parts["subreddit"], parts["year"]), []).append(path)
    if set(candidates) != set(languages):
        raise RuntimeError("candidate/language shard cells do not match")
    canonical_by_cell: dict[tuple[str, str], list[Path]] = {}
    for path in paths["canonical"]:
        parts = labels(path)
        canonical_by_cell.setdefault((parts["subreddit"], parts["year"]), []).append(path)
    if set(canonical_by_cell) != set(candidates) or any(
        len(value) != 2 for value in canonical_by_cell.values()
    ):
        raise RuntimeError("canonical partitions do not exactly cover source-cell shards")
    return [
        {
            "shard_id": _canonical_sha256(["acquisition-prepare-shard-v1", cell]),
            "cells": [f"subreddit={cell[0]}/year={cell[1]}"],
            "candidates": [str(path) for path in candidates[cell]],
            "languages": [str(path) for path in languages[cell]],
            "canonical": [str(path) for path in canonical_by_cell[cell]],
        }
        for cell in sorted(candidates)
    ]


def _progress_event(
    *, stage: str, expected_shards: int, completed_shards: int, started: float, rows: int = 0
) -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started)
    rate = rows / elapsed if elapsed else None
    remaining = expected_shards - completed_shards
    eta = (elapsed / completed_shards * remaining) if completed_shards else None
    result = {
        "kind": "modernbert-acquisition-prepare-progress-v1",
        "stage": stage,
        "expected_shards": expected_shards,
        "completed_shards": completed_shards,
        "row_count": rows,
        "wall_seconds": round(elapsed, 3),
        "rows_per_second": rate,
        "estimated_remaining_seconds": eta,
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition prepare progress")
    return result


def _shard_paths(shard: Mapping[str, Any]) -> dict[str, list[Path]]:
    candidate_values, language_values = shard.get("candidates"), shard.get("languages")
    if not isinstance(candidate_values, list) or not isinstance(language_values, list):
        raise ValueError("prepare shard paths are malformed")
    return {
        "candidates": [Path(value) for value in candidate_values],
        "languages": [Path(value) for value in language_values],
        "canonical": [Path(value) for value in shard.get("canonical", ())],
    }


def _load_prepare_support(pre_prepare_id: str) -> dict[str, Any]:
    path = _prepare_stage_root(pre_prepare_id) / "support.json"
    support = _read_json(path, where="prepare support receipt")
    required = {
        "pre_prepare_id",
        "launch_spec_sha256",
        "runtime_source_bundle_sha256",
        "inventories",
        "shards",
        "calibration_derivation",
        "calibration_cache",
        "exclusions",
        "support_id",
    }
    if not required <= set(support) or support.get("pre_prepare_id") != pre_prepare_id:
        raise RuntimeError("prepare support receipt binding drifted")
    body = {key: value for key, value in support.items() if key != "support_id"}
    if support.get("support_id") != _canonical_sha256(body):
        raise RuntimeError("prepare support receipt digest drifted")
    exclusions = support["exclusions"]
    if not isinstance(exclusions, Mapping):
        raise RuntimeError("prepare support exclusions are malformed")
    clean_exclusions = dict(exclusions)
    for key in (
        "record_id_sha256",
        "thread_sha256",
        "exact_surface_sha256",
        "normalised_surface_sha256",
    ):
        if not isinstance(clean_exclusions.get(key), list):
            raise RuntimeError("prepare support exclusion set is malformed")
        clean_exclusions[key] = set(clean_exclusions[key])
    shards = support.get("shards")
    inventories = support.get("inventories")
    if (
        not isinstance(shards, list)
        or len(shards) != 60
        or not isinstance(inventories, Mapping)
        or set(inventories) != {"candidates", "languages", "canonical"}
    ):
        raise RuntimeError("prepare support authority is malformed")
    return {**support, "exclusions": clean_exclusions}


def _calibration_cache_root(
    pre_prepare_id: str, *, volume_root: Path | None = None
) -> Path:
    return _prepare_stage_root(pre_prepare_id, volume_root=volume_root) / "calibration"


def _write_prepare_calibration_cache(
    *,
    pre_prepare_id: str,
    rows: Sequence[Mapping[str, Any]],
    derivation: Mapping[str, Any],
    volume_root: Path | None = None,
) -> dict[str, Any]:
    """Persist the small private calibration frame once for restart-safe reuse."""
    active_root = VOLUME_PATH if volume_root is None else volume_root
    root = _calibration_cache_root(pre_prepare_id, volume_root=active_root)
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        cached_rows, receipt = _load_prepare_calibration_cache(
            pre_prepare_id, volume_root=active_root
        )
        if cached_rows != [dict(row) for row in rows] or receipt.get(
            "derivation_id"
        ) != derivation.get("derivation_id"):
            raise RuntimeError("existing calibration cache differs")
        return receipt
    if root.exists():
        raise FileExistsError("calibration cache namespace is incomplete")
    staging = root.parent / ".publishing-calibration"
    if staging.exists():
        raise FileExistsError("stale calibration cache publication requires reconciliation")
    staging.mkdir(parents=True)
    frame = staging / "frame.jsonl"
    with frame.open("xb") as handle:
        for row in rows:
            handle.write(
                (
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )
        handle.flush()
        os.fsync(handle.fileno())
    published = root / frame.name
    descriptor = {
        "relative_path": published.relative_to(active_root).as_posix(),
        "sha256": _file_sha256(frame),
        "bytes": frame.stat().st_size,
        "row_count": len(rows),
    }
    body = {
        "kind": "modernbert-acquisition-calibration-cache-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": pre_prepare_id,
        "derivation_id": derivation.get("derivation_id"),
        "derivation": dict(derivation),
        "row_projection_sha256": _canonical_sha256(rows),
        "frame": descriptor,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(receipt, where="acquisition calibration cache receipt")
    (staging / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(root)
    return receipt


def _load_prepare_calibration_cache(
    pre_prepare_id: str, *, volume_root: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_root = VOLUME_PATH if volume_root is None else volume_root
    root = _calibration_cache_root(pre_prepare_id, volume_root=active_root)
    receipt = _read_json(root / "receipt.json", where="calibration cache receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptor = receipt.get("frame")
    if (
        receipt.get("receipt_id") != _canonical_sha256(body)
        or receipt.get("pre_prepare_id") != pre_prepare_id
        or not isinstance(descriptor, Mapping)
        or not isinstance(receipt.get("derivation"), Mapping)
        or receipt.get("derivation_id")
        != receipt.get("derivation", {}).get("derivation_id")
    ):
        raise RuntimeError("calibration cache receipt binding drifted")
    path = active_root / _safe_relative(
        descriptor.get("relative_path"), where="calibration cache frame"
    )
    if (
        not path.is_file()
        or path.stat().st_size != descriptor.get("bytes")
        or _file_sha256(path) != descriptor.get("sha256")
    ):
        raise RuntimeError("calibration cache frame binding drifted")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        len(rows) != descriptor.get("row_count")
        or _canonical_sha256(rows) != receipt.get("row_projection_sha256")
    ):
        raise RuntimeError("calibration cache row conservation drifted")
    return rows, receipt


def _authorise_prepare_shard(
    *, support: Mapping[str, Any], shard: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    """Require an exact coordinator-published shard; workers accept no ad-hoc paths."""
    if support.get("launch_spec_sha256") != _launch_spec_sha256(spec):
        raise RuntimeError("prepare worker launch spec is not support-bound")
    matches = [
        registered
        for registered in support.get("shards", ())
        if isinstance(registered, Mapping)
        and registered.get("shard_id") == shard.get("shard_id")
    ]
    if len(matches) != 1 or dict(matches[0]) != dict(shard):
        raise RuntimeError("prepare worker shard is not support-authorised")


def _authorise_context_partition(
    *, support: Mapping[str, Any], canonical_descriptor: Mapping[str, Any]
) -> None:
    matches = [
        descriptor
        for descriptor in support.get("inventories", {}).get("canonical", ())
        if isinstance(descriptor, Mapping)
        and descriptor.get("relative_path") == canonical_descriptor.get("relative_path")
    ]
    if len(matches) != 1 or dict(matches[0]) != dict(canonical_descriptor):
        raise RuntimeError("context worker canonical partition is not support-authorised")


def _initialise_prepare_state(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one exact prepare authority before any parallel worker starts."""
    started = time.monotonic()

    def report(stage: str, *, objects: int = 0) -> None:
        event = {
            "kind": "modernbert-acquisition-prepare-initialisation-progress-v1",
            "stage": stage,
            "objects": objects,
            "wall_seconds": round(time.monotonic() - started, 3),
            "locked_test_rows_accessed": 0,
        }
        assert_metadata_only(event, where="prepare initialisation progress")
        print(json.dumps(event, sort_keys=True), flush=True)

    clean = validate_input_spec(spec, action="prepare")
    _policy, policy_sha256 = load_frozen_policy()
    if clean.get("policy_sha256") != policy_sha256:
        raise RuntimeError("acquisition input does not bind the frozen policy digest")
    report("validated")
    paths = _discover_inputs()
    report("discovered-inputs", objects=sum(len(value) for value in paths.values()))
    inventories = {
        key: _receipt_backed_inventory(value, producer=key) for key, value in paths.items()
    }
    report("validated-producer-receipts", objects=sum(len(value) for value in inventories.values()))
    pre_prepare_id = _pre_prepare_id(clean, inventories)
    shards = _prepare_shards(paths)
    support_path = _prepare_stage_root(pre_prepare_id) / "support.json"
    if support_path.is_file():
        support = _load_prepare_support(pre_prepare_id)
        calibration_rows, calibration_cache = _load_prepare_calibration_cache(
            pre_prepare_id
        )
        if (
            support.get("launch_spec_sha256") != _launch_spec_sha256(clean)
            or support.get("runtime_source_bundle_sha256")
            != _runtime_source_bundle()["source_bundle_id"]
            or support.get("inventories") != inventories
            or support.get("shards") != shards
            or not isinstance(support.get("calibration_derivation"), Mapping)
            or support.get("calibration_cache") != calibration_cache
            or calibration_cache.get("derivation")
            != support["calibration_derivation"]
        ):
            raise RuntimeError("existing prepare support authority drifted")
        report("reused-support", objects=len(shards))
        return {
            "spec": clean,
            "policy_sha256": policy_sha256,
            "paths": paths,
            "inventories": inventories,
            "pre_prepare_id": pre_prepare_id,
            "shards": shards,
            "calibration_rows": calibration_rows,
            "calibration_derivation": support["calibration_derivation"],
            "exclusions": support["exclusions"],
        }
    calibration_receipt_path = (
        _calibration_cache_root(pre_prepare_id) / "receipt.json"
    )
    if calibration_receipt_path.is_file():
        calibration_rows, calibration_cache = _load_prepare_calibration_cache(
            pre_prepare_id
        )
        calibration_derivation = dict(calibration_cache["derivation"])
        if (
            calibration_derivation.get("base_factorised_run_id")
            != BASE_FACTORISED_RUN_ID
            or calibration_derivation.get("row_count") != len(calibration_rows)
            or calibration_derivation.get("row_projection_sha256")
            != calibration_cache["row_projection_sha256"]
        ):
            raise RuntimeError("orphaned calibration cache authority drifted")
        report("reused-calibration", objects=len(calibration_rows))
    else:
        calibration_rows, calibration_derivation = _derive_retained_calibration(
            clean, volume_root=VOLUME_PATH
        )
        calibration_cache = _write_prepare_calibration_cache(
            pre_prepare_id=pre_prepare_id,
            rows=calibration_rows,
            derivation=calibration_derivation,
        )
        report("derived-calibration", objects=len(calibration_rows))
    exclusions = _build_exposure_ledger(
        clean, calibration_rows=calibration_rows, volume_root=VOLUME_PATH
    )
    report("derived-exclusions", objects=len(exclusions["record_id_sha256"]))
    support_body = {
        "pre_prepare_id": pre_prepare_id,
        "launch_spec_sha256": _launch_spec_sha256(clean),
        "runtime_source_bundle_sha256": _runtime_source_bundle()["source_bundle_id"],
        "inventories": inventories,
        "shards": shards,
        "calibration_derivation": calibration_derivation,
        "calibration_cache": calibration_cache,
        "exclusions": {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in exclusions.items()
        },
    }
    _write_immutable_json(
        _prepare_stage_root(pre_prepare_id) / "support.json",
        {**support_body, "support_id": _canonical_sha256(support_body)},
    )
    report("published-support", objects=len(shards))
    return {
        "spec": clean,
        "policy_sha256": policy_sha256,
        "paths": paths,
        "inventories": inventories,
        "pre_prepare_id": pre_prepare_id,
        "shards": shards,
        "calibration_rows": calibration_rows,
        "calibration_derivation": calibration_derivation,
        "exclusions": exclusions,
    }


@app.function(
    image=image,
    cpu=8,
    memory=24_576,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=SCORING_MAX_CONTAINERS,
    volumes={str(VOLUME_PATH): volume},
)
def reduce_candidate_scores(job: dict[str, Any]) -> dict[str, Any]:
    """Reduce one prepared-frame shard to compact candidate scores."""

    if not isinstance(job, Mapping) or set(job) != CANDIDATE_REDUCER_JOB_KEYS:
        raise ValueError("candidate score reducer job schema drifted")
    lease = _claim_dispatch_lease(job, stage="reduce-candidate-scores")
    if lease is None:
        return _dispatch_not_acquired(job)
    volume.reload()
    clean = validate_input_spec(job["spec"], action="launch-scoring")
    prepared_run_id = _sha256(clean.get("prepared_run_id"), where="prepared run ID")
    prepared, _frame_path = _load_prepared_contract(
        prepared_run_id, validate_frame_content=False
    )
    _validate_prepared_spec_binding(prepared, clean)
    plan = _scoring_plan_contract(prepared["prepared_frame"]["row_count"])
    shard = job.get("scoring_shard")
    if (
        job.get("scoring_plan_sha256") != plan["plan_id"]
        or not isinstance(shard, Mapping)
        or dict(shard) not in plan["shards"]
        or not isinstance(job.get("attempt_id"), str)
        or not job["attempt_id"]
        or not isinstance(job.get("dispatch_id"), str)
        or not job["dispatch_id"]
    ):
        raise RuntimeError("candidate score reducer plan binding drifted")
    checkpoints = _checkpoint_specs(clean)
    checkpoint_bundle_sha256 = _canonical_sha256(checkpoints)
    rare_cells = prepared.get("rare_cells")
    if not isinstance(rare_cells, list) or len(rare_cells) != 3:
        raise RuntimeError("candidate score reducer rare-cell binding drifted")
    policy_toml, policy_sha256 = load_frozen_policy()
    if clean["policy_sha256"] != policy_sha256:
        raise RuntimeError("candidate score reducer policy binding drifted")
    receipt = _materialise_candidate_score_shard(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        checkpoints=checkpoints,
        rare_cells=rare_cells,
        policy=_acquisition_policy_from_toml(policy_toml),
        shard=shard,
        attempt_id=job["attempt_id"],
        dispatch_id=job["dispatch_id"],
        volume_root=VOLUME_PATH,
    )
    volume.commit()
    return receipt


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def initialise_prepare(spec: dict[str, Any], approved_cost_usd: str) -> dict[str, Any]:
    volume.reload()
    clean = validate_input_spec(spec, action="prepare")
    _validate_phase_approval(clean, action="prepare", approved_cost_usd=approved_cost_usd)
    state = _initialise_prepare_state(clean)
    volume.commit()
    result = {
        "kind": "modernbert-acquisition-prepare-initialised-v1",
        "pre_prepare_id": state["pre_prepare_id"],
        "source_shards": len(state["shards"]),
        "canonical_partitions": len(state["inventories"]["canonical"]),
        "input_bytes": sum(
            row["bytes"]
            for values in state["inventories"].values()
            for row in values
        ),
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition prepare initialisation")
    return result


def _benchmark_prepare_shape(
    pre_prepare_id: str,
    shard: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    threads: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    started = time.monotonic()
    clean = validate_input_spec(spec, action="prepare")
    support = _load_prepare_support(pre_prepare_id)
    _authorise_prepare_shard(support=support, shard=shard, spec=clean)
    rows = _prepare_sql_universe(
        paths=_shard_paths(shard),
        exclusions=support["exclusions"],
        spec=clean,
        reduce=False,
        include_context=False,
        threads=threads,
    )
    scan_elapsed = time.monotonic() - started
    cluster_started = time.monotonic()
    clustered = _cluster_near_duplicates(
        [
            {
                "record_id": row["record_id"],
                "simhash": row["simhash"],
                "normalised_chars": row["normalised_chars"],
            }
            for row in rows
        ]
    )
    cluster_elapsed = time.monotonic() - cluster_started
    parquet_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="acquisition-prepare-benchmark-") as directory:
        parquet_path = Path(directory) / "source.parquet"
        pq.write_table(_prepare_source_table(rows), parquet_path, compression="zstd")
        parquet_file = pq.ParquetFile(parquet_path)
        parquet_rows = parquet_file.metadata.num_rows
        parquet_simhash_type = str(parquet_file.schema_arrow.field("simhash").type)
    parquet_elapsed = time.monotonic() - parquet_started
    result = {
        "kind": "modernbert-acquisition-prepare-benchmark-result-v1",
        "pre_prepare_id": pre_prepare_id,
        "shard_id": shard["shard_id"],
        "cpus": threads,
        "memory_mib": 12_288 if threads == 4 else 24_576,
        "row_count": len(rows),
        "output_projection_sha256": _canonical_sha256(
            [
                {
                    key: row[key]
                    for key in (
                        "record_id",
                        "thread_id",
                        "exact_surface_sha256",
                        "normalised_surface_sha256",
                    )
                }
                for row in rows
            ]
        ),
        "input_bytes": sum(
            path.stat().st_size for values in _shard_paths(shard).values() for path in values
        ),
        "wall_seconds": round(scan_elapsed, 3),
        "rows_per_second": len(rows) / scan_elapsed if scan_elapsed else None,
        "near_cluster_input_count": len(rows),
        "near_cluster_output_count": len(clustered),
        "near_cluster_seconds": round(cluster_elapsed, 3),
        "parquet_roundtrip_row_count": parquet_rows,
        "parquet_simhash_type": parquet_simhash_type,
        "parquet_write_seconds": round(parquet_elapsed, 3),
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition prepare benchmark")
    return result


@app.function(
    image=image,
    cpu=4,
    memory=12_288,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def benchmark_prepare_4cpu(
    pre_prepare_id: str, shard: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    volume.reload()
    return _benchmark_prepare_shape(pre_prepare_id, shard, spec, threads=4)


@app.function(
    image=image,
    cpu=8,
    memory=24_576,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def benchmark_prepare_8cpu(
    pre_prepare_id: str, shard: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    volume.reload()
    return _benchmark_prepare_shape(pre_prepare_id, shard, spec, threads=8)


def _prepare_source_table(source: Sequence[Mapping[str, Any]]) -> Any:
    """Build the durable source-shard table with an explicit wire schema.

    SimHash occupies the full unsigned 64-bit range.  Inferring a schema from
    Python integers therefore fails nondeterministically whenever a shard first
    contains a value above signed int64.  Keep every durable column explicit so
    the bounded benchmark and production publisher exercise the same contract.
    """
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
            pa.field("subreddit", pa.string(), nullable=False),
            pa.field("year", pa.int32(), nullable=False),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("retrieval_mode", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("parent_id", pa.string()),
            pa.field("submission_context", pa.string()),
            pa.field("parent_context", pa.string()),
            pa.field("exact_surface_sha256", pa.string(), nullable=False),
            pa.field("normalised_surface_sha256", pa.string(), nullable=False),
            pa.field("simhash", pa.uint64(), nullable=False),
            pa.field("normalised_chars", pa.int32(), nullable=False),
            pa.field("thread_tiebreak_sha256", pa.string(), nullable=False),
            pa.field("surface_tiebreak_sha256", pa.string(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(list(source), schema=schema)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def run_prepare_benchmark(
    spec: dict[str, Any], approved_cost_usd: str
) -> dict[str, Any]:
    """Run representative and stress cells on two schedulable worker shapes."""
    volume.reload()
    clean = validate_input_spec(spec, action="prepare")
    _validate_phase_approval(clean, action="prepare", approved_cost_usd=approved_cost_usd)
    state = _initialise_prepare_state(clean)
    volume.commit()
    fixed_matches = [
        shard
        for shard in state["shards"]
        if shard["cells"] == ["subreddit=China/year=2020"]
    ]
    if len(fixed_matches) != 1:
        raise RuntimeError("fixed benchmark cell is unavailable")
    fixed = fixed_matches[0]
    largest = max(
        state["shards"],
        key=lambda shard: sum(
            path.stat().st_size
            for values in _shard_paths(shard).values()
            for path in values
        ),
    )
    benchmark_shards = [fixed]
    if largest["shard_id"] != fixed["shard_id"]:
        benchmark_shards.append(largest)
    benchmark_root = _prepare_stage_root(state["pre_prepare_id"]) / "benchmark"
    result_path = benchmark_root / "result.json"
    if result_path.is_file():
        result = _read_json(result_path, where="prepare benchmark result")
        body = {key: value for key, value in result.items() if key != "result_id"}
        if result.get("result_id") != _canonical_sha256(body):
            raise RuntimeError("existing prepare benchmark result drifted")
        return result
    if benchmark_root.exists():
        raise FileExistsError("prepare benchmark namespace is incomplete")
    calls = [
        (shard, benchmark_prepare_4cpu.spawn(state["pre_prepare_id"], shard, clean))
        for shard in benchmark_shards
    ] + [
        (shard, benchmark_prepare_8cpu.spawn(state["pre_prepare_id"], shard, clean))
        for shard in benchmark_shards
    ]
    results = [call.get() for _shard, call in calls]
    for shard in benchmark_shards:
        paired = [row for row in results if row["shard_id"] == shard["shard_id"]]
        if (
            len(paired) != 2
            or paired[0]["row_count"] != paired[1]["row_count"]
            or paired[0]["output_projection_sha256"]
            != paired[1]["output_projection_sha256"]
            or paired[0]["near_cluster_output_count"]
            != paired[1]["near_cluster_output_count"]
            or any(
                row["parquet_roundtrip_row_count"] != row["row_count"]
                or row["parquet_simhash_type"] != "uint64"
                for row in paired
            )
        ):
            raise RuntimeError("prepare benchmark worker shapes are not semantically identical")
    totals = {
        cpus: sum(row["wall_seconds"] for row in results if row["cpus"] == cpus)
        for cpus in (4, 8)
    }
    selected_cpus = min(totals, key=totals.get)
    winner = next(row for row in results if row["cpus"] == selected_cpus)
    body = {
        "kind": "modernbert-acquisition-prepare-benchmark-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": state["pre_prepare_id"],
        "benchmark_cells": [shard["cells"][0] for shard in benchmark_shards],
        "results": results,
        "aggregate_scan_seconds_by_cpu": totals,
        "selected_cpus": selected_cpus,
        "selected_memory_mib": winner["memory_mib"],
        "wall_speedup_over_slower": (
            max(totals.values()) / totals[selected_cpus]
            if totals[selected_cpus]
            else None
        ),
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "result_id": _canonical_sha256(body)}
    _write_immutable_json(result_path, result)
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=PREPARE_WORKER_CPU,
    memory=PREPARE_WORKER_MEMORY_MIB,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=PREPARE_MAX_CONTAINERS,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_shard(
    pre_prepare_id: str, shard: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Materialise one reusable unpublished prepare shard, fail-closed."""
    import pyarrow.parquet as pq

    started = time.monotonic()
    volume.reload()
    clean = validate_input_spec(spec, action="prepare")
    support = _load_prepare_support(pre_prepare_id)
    _authorise_prepare_shard(support=support, shard=shard, spec=clean)
    shard_id = _sha256(shard.get("shard_id"), where="prepare shard ID")
    root = _prepare_stage_root(pre_prepare_id) / "shards" / f"shard={shard_id}"
    staging = root.parent / f".publishing-{shard_id}"
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, where="prepare shard receipt")
        descriptor = receipt.get("frame") if isinstance(receipt, Mapping) else None
        if not isinstance(descriptor, Mapping):
            raise RuntimeError("completed prepare shard receipt is malformed")
        _bound_parquet(
            descriptor,
            volume_root=root,
            where="completed prepare shard",
            expected_rows=descriptor.get("row_count"),
        )
        return receipt
    if root.exists():
        raise FileExistsError("prepare shard namespace is incomplete")
    if staging.exists():
        raise FileExistsError("stale prepare shard publishing namespace requires reconciliation")
    source = _prepare_sql_universe(
        paths=_shard_paths(shard),
        exclusions=support["exclusions"],
        spec=clean,
        reduce=False,
        include_context=False,
    )
    staging.mkdir(parents=True)
    frame = staging / "source.parquet"
    pq.write_table(_prepare_source_table(source), frame, compression="zstd")
    descriptor = _descriptor(frame, root=staging, row_count=len(source))
    body = {
        "kind": "modernbert-acquisition-prepare-shard-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": _sha256(pre_prepare_id, where="pre-prepare ID"),
        "shard_id": shard_id,
        "cells": list(shard["cells"]),
        "frame": descriptor,
        "row_count": len(source),
        "wall_seconds": round(time.monotonic() - started, 3),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    (staging / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(root)
    volume.commit()
    return receipt


def _write_context_requests(
    *,
    pre_prepare_id: str,
    universe: Sequence[Mapping[str, Any]],
    expected_subreddits: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Publish one reusable exact-ID request set per subreddit.

    Submission and parent records cannot cross subreddits.  Partitioning the
    request set here avoids broadcasting millions of IDs through Modal's
    control plane while still allowing every year/content partition to resolve
    cross-year context exactly.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    grouped: dict[str, set[str]] = {}
    all_ids: set[str] = set()
    for row in universe:
        subreddit = row.get("subreddit")
        if not isinstance(subreddit, str) or not subreddit or "/" in subreddit:
            raise RuntimeError("context request row has an invalid subreddit")
        values = {
            value
            for value in (row.get("thread_id"), row.get("parent_id"))
            if isinstance(value, str) and value
        }
        if all_ids.intersection(values):
            # Reddit record IDs are global.  Seeing one request under two
            # subreddits would make the exact partition contract ambiguous.
            other = values.intersection(all_ids)
            if not other.issubset(grouped.setdefault(subreddit, set())):
                raise RuntimeError("context request ID crosses subreddits")
        grouped.setdefault(subreddit, set()).update(values)
        all_ids.update(values)
    if not grouped or not all_ids:
        raise RuntimeError("context request inventory is empty")
    if expected_subreddits is not None:
        if not expected_subreddits or grouped.keys() - expected_subreddits:
            raise RuntimeError("context request subreddit inventory drifted")
        for subreddit in expected_subreddits:
            grouped.setdefault(subreddit, set())

    root = _prepare_stage_root(pre_prepare_id) / "context-requests"
    result: dict[str, dict[str, Any]] = {}
    schema = pa.schema([pa.field("record_id", pa.string(), nullable=False)])
    for subreddit, values in sorted(grouped.items()):
        final = root / f"subreddit={subreddit}"
        receipt_path = final / "receipt.json"
        if receipt_path.is_file():
            receipt = _read_json(receipt_path, where="context request receipt")
            body = {key: value for key, value in receipt.items() if key != "receipt_id"}
            descriptor = receipt.get("requests")
            if (
                receipt.get("receipt_id") != _canonical_sha256(body)
                or receipt.get("pre_prepare_id") != pre_prepare_id
                or receipt.get("subreddit") != subreddit
                or not isinstance(descriptor, Mapping)
            ):
                raise RuntimeError("existing context request receipt drifted")
            _bound_parquet(
                descriptor,
                volume_root=VOLUME_PATH,
                where="existing context requests",
                expected_rows=receipt.get("request_count"),
            )
            result[subreddit] = dict(descriptor)
            continue
        if final.exists():
            raise FileExistsError("context request namespace is incomplete")
        staging = root / f".publishing-subreddit={subreddit}"
        if staging.exists():
            raise FileExistsError("stale context request publication requires reconciliation")
        staging.mkdir(parents=True)
        frame = staging / "requests.parquet"
        rows = [{"record_id": value} for value in sorted(values)]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), frame, compression="zstd")
        published = final / frame.name
        descriptor = {
            "relative_path": published.relative_to(VOLUME_PATH).as_posix(),
            "sha256": _file_sha256(frame),
            "bytes": frame.stat().st_size,
            "row_count": len(rows),
        }
        body = {
            "kind": "modernbert-acquisition-context-requests-v1",
            "schema_version": "1.0.0",
            "pre_prepare_id": pre_prepare_id,
            "subreddit": subreddit,
            "request_count": len(rows),
            "requests": descriptor,
            "locked_test_rows_accessed": 0,
        }
        receipt = {**body, "receipt_id": _canonical_sha256(body)}
        (staging / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(final)
        result[subreddit] = descriptor
    return result


@app.function(
    image=image,
    cpu=PREPARE_WORKER_CPU,
    memory=PREPARE_WORKER_MEMORY_MIB,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=PREPARE_MAX_CONTAINERS,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_context_shard(
    pre_prepare_id: str,
    canonical_descriptor: dict[str, Any],
    request_descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Second registered corpus pass: resolve requested context IDs once/partition."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    volume.reload()
    support = _load_prepare_support(pre_prepare_id)
    _authorise_context_partition(
        support=support, canonical_descriptor=canonical_descriptor
    )
    path = _bound_receipt_parquet(
        canonical_descriptor,
        volume_root=VOLUME_PATH,
        where="context canonical partition",
    )
    request_path = _bound_parquet(
        request_descriptor,
        volume_root=VOLUME_PATH,
        where="context request partition",
        expected_rows=request_descriptor.get("row_count"),
    )
    partition_id = _canonical_sha256(
        [
            "acquisition-context-partition-v1",
            canonical_descriptor,
            request_descriptor,
        ]
    )
    root = _prepare_stage_root(pre_prepare_id) / "contexts" / f"partition={partition_id}"
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, where="context shard receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        descriptor = receipt.get("frame")
        if (
            receipt.get("receipt_id") != _canonical_sha256(body)
            or receipt.get("pre_prepare_id") != pre_prepare_id
            or receipt.get("canonical_sha256") != canonical_descriptor.get("sha256")
            or receipt.get("request_sha256") != request_descriptor.get("sha256")
            or not isinstance(descriptor, Mapping)
        ):
            raise RuntimeError("existing context shard receipt drifted")
        _bound_parquet(
            descriptor,
            volume_root=root,
            where="existing context shard",
            expected_rows=receipt.get("row_count"),
        )
        return receipt
    if root.exists():
        raise FileExistsError("context shard namespace is incomplete")
    database = duckdb.connect(database=":memory:")
    try:
        rows = database.execute(
            "SELECT c.record_id, c.submission_id, c.content_type, c.text "
            "FROM read_parquet(?) c JOIN read_parquet(?) r USING(record_id)",
            [str(path), str(request_path)],
        ).fetchall()
    finally:
        database.close()
    staging = root.parent / f".publishing-{partition_id}"
    if staging.exists():
        raise FileExistsError("stale context shard publishing namespace requires reconciliation")
    staging.mkdir(parents=True)
    frame = staging / "context.parquet"
    context_schema = pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("submission_id", pa.string(), nullable=False),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("bounded_text", pa.string(), nullable=True),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": record_id,
                    "submission_id": submission_id,
                    "content_type": content_type,
                    "bounded_text": _bounded(
                        text,
                        limit=(
                            _packet_module().MAX_CONTEXT_CHARS
                            if content_type == "submission"
                            else _packet_module().MAX_PARENT_CHARS
                        ),
                    ),
                }
                for record_id, submission_id, content_type, text in rows
            ],
            schema=context_schema,
        ),
        frame,
        compression="zstd",
    )
    descriptor = _descriptor(frame, root=staging, row_count=len(rows))
    body = {
        "kind": "modernbert-acquisition-context-shard-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": pre_prepare_id,
        "partition_id": partition_id,
        "canonical_relative_path": path.relative_to(VOLUME_PATH).as_posix(),
        "canonical_sha256": canonical_descriptor["sha256"],
        "request_sha256": request_descriptor["sha256"],
        "requested_ids": request_descriptor["row_count"],
        "frame": descriptor,
        "row_count": len(rows),
        "wall_seconds": round(time.monotonic() - started, 3),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    (staging / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    staging.replace(root)
    volume.commit()
    return receipt


def _load_prepare_stage_frame(
    pre_prepare_id: str, *, stage: str
) -> tuple[dict[str, Any], Path]:
    root = _prepare_stage_root(pre_prepare_id) / stage
    receipt = _read_json(root / "receipt.json", where=f"{stage} receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptor = receipt.get("frame")
    if (
        receipt.get("receipt_id") != _canonical_sha256(body)
        or receipt.get("pre_prepare_id") != pre_prepare_id
        or not isinstance(descriptor, Mapping)
    ):
        raise RuntimeError(f"{stage} receipt binding drifted")
    path = _bound_internal_parquet(
        descriptor,
        volume_root=root,
        where=f"{stage} frame",
    )
    return receipt, path


def _validated_prepare_shard_frames(
    pre_prepare_id: str,
) -> tuple[list[Path], list[dict[str, Any]], int]:
    support = _load_prepare_support(pre_prepare_id)
    frames: list[Path] = []
    receipts: list[dict[str, Any]] = []
    total_rows = 0
    observed: set[str] = set()
    for shard in support["shards"]:
        shard_id = _sha256(shard.get("shard_id"), where="prepare shard ID")
        root = _prepare_stage_root(pre_prepare_id) / "shards" / f"shard={shard_id}"
        receipt = _read_json(root / "receipt.json", where="prepare shard receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        descriptor = receipt.get("frame")
        cells = set(receipt.get("cells", ()))
        if (
            receipt.get("receipt_id") != _canonical_sha256(body)
            or receipt.get("pre_prepare_id") != pre_prepare_id
            or receipt.get("shard_id") != shard_id
            or cells != set(shard["cells"])
            or observed.intersection(cells)
            or not isinstance(descriptor, Mapping)
            or not isinstance(receipt.get("row_count"), int)
        ):
            raise RuntimeError("prepare shard receipt conservation drifted")
        frame = _bound_internal_parquet(
            descriptor,
            volume_root=root,
            where="prepare shard frame",
        )
        observed.update(cells)
        frames.append(frame)
        receipts.append(receipt)
        total_rows += receipt["row_count"]
    expected = {cell for shard in support["shards"] for cell in shard["cells"]}
    if observed != expected or len(frames) != 60:
        raise RuntimeError("prepare source shards are incomplete")
    return frames, receipts, total_rows


@app.function(
    image=image,
    cpu=PREPARE_REDUCE_CPU,
    memory=PREPARE_REDUCE_MEMORY_MIB,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_exact_reduce(pre_prepare_id: str) -> dict[str, Any]:
    """Durably apply global thread and exact-surface reduction in DuckDB."""
    import duckdb
    import pyarrow.parquet as pq

    started = time.monotonic()
    volume.reload()
    root = _prepare_stage_root(pre_prepare_id) / "exact-reduce"
    if (root / "receipt.json").is_file():
        return _load_prepare_stage_frame(pre_prepare_id, stage="exact-reduce")[0]
    if root.exists():
        raise FileExistsError("exact-reduce namespace is incomplete")
    frames, shard_receipts, input_rows = _validated_prepare_shard_frames(pre_prepare_id)
    staging = root.parent / ".publishing-exact-reduce"
    if staging.exists():
        raise FileExistsError("stale exact-reduce publication requires reconciliation")
    staging.mkdir(parents=True)
    output = staging / "exact-reduced.parquet"
    quoted = str(output).replace("'", "''")
    database = duckdb.connect(database=":memory:")
    try:
        database.execute(f"SET threads = {PREPARE_REDUCE_CPU}")
        database.execute("SET preserve_insertion_order = false")
        database.execute(
            f"""
            COPY (
                WITH thread_rows AS (
                    SELECT * EXCLUDE(thread_rank) FROM (
                        SELECT *, row_number() OVER (
                            PARTITION BY thread_id ORDER BY thread_tiebreak_sha256
                        ) AS thread_rank
                        FROM read_parquet(?)
                    ) WHERE thread_rank=1
                ), surface_rows AS (
                    SELECT * EXCLUDE(surface_rank) FROM (
                        SELECT *, row_number() OVER (
                            PARTITION BY normalised_surface_sha256
                            ORDER BY surface_tiebreak_sha256
                        ) AS surface_rank
                        FROM thread_rows
                    ) WHERE surface_rank=1
                )
                SELECT * FROM surface_rows ORDER BY record_id
            ) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """,
            [[str(path) for path in frames]],
        )
    finally:
        database.close()
    output_rows = pq.ParquetFile(output).metadata.num_rows
    if output_rows < 2_600 or output_rows > input_rows:
        raise RuntimeError("exact-reduce row conservation failed")
    descriptor = _descriptor(output, root=staging, row_count=output_rows)
    body = {
        "kind": "modernbert-acquisition-exact-reduce-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": pre_prepare_id,
        "input_shard_receipt_ids": [row["receipt_id"] for row in shard_receipts],
        "input_rows": input_rows,
        "output_rows": output_rows,
        "frame": descriptor,
        "wall_seconds": round(time.monotonic() - started, 3),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    (staging / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    staging.replace(root)
    volume.commit()
    return receipt


@app.function(
    image=image,
    cpu=PREPARE_REDUCE_CPU,
    memory=PREPARE_REDUCE_MEMORY_MIB,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_near_reduce(pre_prepare_id: str) -> dict[str, Any]:
    """Durably apply the exact registered SimHash connected components."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    volume.reload()
    root = _prepare_stage_root(pre_prepare_id) / "near-reduce"
    if (root / "receipt.json").is_file():
        return _load_prepare_stage_frame(pre_prepare_id, stage="near-reduce")[0]
    if root.exists():
        raise FileExistsError("near-reduce namespace is incomplete")
    exact_receipt, exact_path = _load_prepare_stage_frame(
        pre_prepare_id, stage="exact-reduce"
    )
    minimal = pq.read_table(
        exact_path,
        columns=["record_id", "simhash", "normalised_chars"],
    ).to_pylist()
    def report_near_progress(completed: int, expected: int, elapsed: float) -> None:
        remaining = expected - completed
        event = {
            "kind": "modernbert-acquisition-near-reduce-progress-v1",
            "stage": "near-reduce",
            "expected_count": expected,
            "completed_count": completed,
            "wall_seconds": round(elapsed, 3),
            "items_per_second": completed / elapsed if elapsed else None,
            "estimated_remaining_seconds": (
                elapsed / completed * remaining if completed else None
            ),
            "locked_test_rows_accessed": 0,
        }
        assert_metadata_only(event, where="acquisition near-reduce progress")
        print(json.dumps(event, sort_keys=True), flush=True)

    clustered = _cluster_near_duplicates(minimal, progress=report_near_progress)
    if len(clustered) < 2_600 or len(clustered) > len(minimal):
        raise RuntimeError("near-reduce row conservation failed")
    keep = pa.Table.from_pylist(
        [
            {
                "record_id": row["record_id"],
                "near_duplicate_cluster_id": row["near_duplicate_cluster_id"],
            }
            for row in clustered
        ]
    )
    staging = root.parent / ".publishing-near-reduce"
    if staging.exists():
        raise FileExistsError("stale near-reduce publication requires reconciliation")
    staging.mkdir(parents=True)
    output = staging / "eligible-base.parquet"
    quoted = str(output).replace("'", "''")
    database = duckdb.connect(database=":memory:")
    try:
        database.execute(f"SET threads = {PREPARE_REDUCE_CPU}")
        database.register("near_keep", keep)
        database.execute(
            f"""
            COPY (
                SELECT e.* EXCLUDE(
                    thread_tiebreak_sha256, surface_tiebreak_sha256
                ), k.near_duplicate_cluster_id
                FROM read_parquet(?) e JOIN near_keep k USING(record_id)
                ORDER BY e.record_id
            ) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """,
            [str(exact_path)],
        )
    finally:
        database.close()
    output_rows = pq.ParquetFile(output).metadata.num_rows
    if output_rows != len(clustered):
        raise RuntimeError("near-reduce publication did not conserve representatives")
    descriptor = _descriptor(output, root=staging, row_count=output_rows)
    body = {
        "kind": "modernbert-acquisition-near-reduce-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": pre_prepare_id,
        "exact_reduce_receipt_id": exact_receipt["receipt_id"],
        "input_rows": len(minimal),
        "output_rows": output_rows,
        "frame": descriptor,
        "wall_seconds": round(time.monotonic() - started, 3),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    (staging / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    staging.replace(root)
    volume.commit()
    return receipt


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_context_requests_stage(pre_prepare_id: str) -> dict[str, Any]:
    """Publish compact, resumable context-request frames after deduplication."""
    import pyarrow.parquet as pq

    started = time.monotonic()
    volume.reload()
    support = _load_prepare_support(pre_prepare_id)
    near_receipt, near_path = _load_prepare_stage_frame(pre_prepare_id, stage="near-reduce")
    rows = pq.read_table(
        near_path,
        columns=["subreddit", "thread_id", "parent_id"],
    ).to_pylist()
    expected_subreddits = {
        part.split("=", 1)[1]
        for descriptor in support["inventories"]["canonical"]
        for part in Path(descriptor["relative_path"]).parts
        if part.startswith("subreddit=")
    }
    descriptors = _write_context_requests(
        pre_prepare_id=pre_prepare_id,
        universe=rows,
        expected_subreddits=expected_subreddits,
    )
    volume.commit()
    body = {
        "kind": "modernbert-acquisition-context-request-index-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": pre_prepare_id,
        "near_reduce_receipt_id": near_receipt["receipt_id"],
        "subreddit_count": len(descriptors),
        "request_rows": sum(row["row_count"] for row in descriptors.values()),
        "request_descriptors_sha256": _canonical_sha256(descriptors),
        "wall_seconds": round(time.monotonic() - started, 3),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    _write_immutable_json(
        _prepare_stage_root(pre_prepare_id) / "context-requests" / "index.json",
        receipt,
    )
    volume.commit()
    return receipt


def _hydrate_context_frames(
    *, near_path: Path, context_paths: Sequence[Path | str], output: Path, threads: int
) -> dict[str, int]:
    """Exact cross-year context join shared by production and regression tests."""
    import duckdb
    import pyarrow.parquet as pq

    if threads not in {4, 8} or not near_path.is_file() or not context_paths:
        raise ValueError("hydration inputs are invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    quoted = str(output).replace("'", "''")
    database = duckdb.connect(database=":memory:")
    try:
        database.execute(f"SET threads = {threads}")
        database.execute(
            "CREATE TEMP TABLE context AS SELECT * FROM read_parquet(?)",
            [[str(path) for path in context_paths]],
        )
        duplicate_context = database.execute(
            "SELECT count(*)-count(DISTINCT record_id) FROM context"
        ).fetchone()[0]
        universe_rows, comment_rows = database.execute(
            """
            SELECT count(*), count(*) FILTER (WHERE content_type='comment')
            FROM read_parquet(?)
            """,
            [str(near_path)],
        ).fetchone()
        missing_submission, invalid_submission = database.execute(
            """
            SELECT
              count(*) FILTER (WHERE s.record_id IS NULL),
              count(*) FILTER (
                WHERE s.record_id IS NOT NULL AND s.content_type<>'submission'
              )
            FROM read_parquet(?) u
            LEFT JOIN context s ON u.thread_id=s.record_id
            WHERE u.content_type='comment'
            """,
            [str(near_path)],
        ).fetchone()
        missing_parent, invalid_parent = database.execute(
            """
            SELECT
              count(*) FILTER (WHERE p.record_id IS NULL),
              count(*) FILTER (
                WHERE p.record_id IS NOT NULL AND (
                  p.content_type<>'comment' OR p.submission_id<>u.thread_id
                )
              )
            FROM read_parquet(?) u
            LEFT JOIN context p ON u.parent_id=p.record_id
            WHERE u.content_type='comment'
              AND starts_with(CAST(u.parent_id AS VARCHAR), 't1_')
            """,
            [str(near_path)],
        ).fetchone()
        invalid_top_level_parent = database.execute(
            """
            SELECT count(*)
            FROM read_parquet(?)
            WHERE content_type='comment'
              AND starts_with(CAST(parent_id AS VARCHAR), 't3_')
              AND parent_id<>thread_id
            """,
            [str(near_path)],
        ).fetchone()[0]
        # Missing context is a source-coverage fact, not a relational error:
        # comments can reference submissions or parents outside the retained
        # 2020-2025 corpus. A present record with the wrong type/thread remains
        # a fail-closed integrity violation.
        if (
            duplicate_context
            or invalid_submission
            or invalid_parent
            or invalid_top_level_parent
        ):
            raise RuntimeError("cross-year context hydration failed exact thread validation")
        database.execute(
            f"""
            COPY (
                SELECT u.* EXCLUDE(submission_context, parent_context),
                       CASE WHEN u.content_type='comment' THEN s.bounded_text END
                           AS submission_context,
                       CASE WHEN u.content_type='comment'
                                  AND starts_with(CAST(u.parent_id AS VARCHAR), 't1_')
                            THEN p.bounded_text END
                           AS parent_context
                FROM read_parquet(?) u
                LEFT JOIN context s ON u.thread_id=s.record_id
                LEFT JOIN context p
                  ON starts_with(CAST(u.parent_id AS VARCHAR), 't1_')
                 AND u.parent_id=p.record_id
                 AND p.content_type='comment'
                ORDER BY u.record_id
            ) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """,
            [str(near_path)],
        )
        packet = _packet_module()
        packet_bound_violations, top_level_parent_contexts = database.execute(
            f"""
            SELECT
              count(*) FILTER (
                WHERE target_text IS NULL OR target_text=''
                   OR length(target_text)>{packet.MAX_TARGET_CHARS}
                   OR (submission_context IS NOT NULL AND (
                     submission_context='' OR
                     length(submission_context)>{packet.MAX_CONTEXT_CHARS}
                   ))
                   OR (parent_context IS NOT NULL AND (
                     parent_context='' OR length(parent_context)>{packet.MAX_PARENT_CHARS}
                   ))
              ),
              count(*) FILTER (
                WHERE content_type='comment'
                  AND starts_with(CAST(parent_id AS VARCHAR), 't3_')
                  AND parent_context IS NOT NULL
              )
            FROM read_parquet(?)
            """,
            [str(output)],
        ).fetchone()
        if packet_bound_violations or top_level_parent_contexts:
            raise RuntimeError("hydrated acquisition frame differs from exact packet bounds")
    finally:
        database.close()
    output_rows = pq.ParquetFile(output).metadata.num_rows
    if output_rows != universe_rows:
        raise RuntimeError("cross-year context hydration did not conserve universe rows")
    return {
        "universe_rows": universe_rows,
        "comment_rows": comment_rows,
        "missing_submission_contexts": missing_submission,
        "missing_parent_contexts": missing_parent,
        "packet_bound_violations": packet_bound_violations,
        "top_level_parent_contexts": top_level_parent_contexts,
        "output_rows": output_rows,
    }


@app.function(
    image=image,
    cpu=PREPARE_REDUCE_CPU,
    memory=PREPARE_REDUCE_MEMORY_MIB,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_hydrate(pre_prepare_id: str) -> dict[str, Any]:
    """Join exact cross-year context frames and persist the final private universe."""
    started = time.monotonic()
    volume.reload()
    root = _prepare_stage_root(pre_prepare_id) / "hydrated"
    if (root / "receipt.json").is_file():
        return _load_prepare_stage_frame(pre_prepare_id, stage="hydrated")[0]
    if root.exists():
        raise FileExistsError("hydrated namespace is incomplete")
    near_receipt, near_path = _load_prepare_stage_frame(pre_prepare_id, stage="near-reduce")
    support = _load_prepare_support(pre_prepare_id)
    context_paths: list[str] = []
    observed: set[str] = set()
    for canonical in support["inventories"]["canonical"]:
        subreddit = next(
            part.split("=", 1)[1]
            for part in Path(canonical["relative_path"]).parts
            if part.startswith("subreddit=")
        )
        request_receipt = _read_json(
            _prepare_stage_root(pre_prepare_id)
            / "context-requests"
            / f"subreddit={subreddit}"
            / "receipt.json",
            where="context request receipt",
        )
        request = request_receipt["requests"]
        partition_id = _canonical_sha256(
            ["acquisition-context-partition-v1", canonical, request]
        )
        context_root = (
            _prepare_stage_root(pre_prepare_id)
            / "contexts"
            / f"partition={partition_id}"
        )
        receipt = _read_json(context_root / "receipt.json", where="context shard receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        descriptor = receipt.get("frame")
        if (
            receipt.get("receipt_id") != _canonical_sha256(body)
            or receipt.get("canonical_relative_path") != canonical["relative_path"]
            or receipt.get("canonical_sha256") != canonical["sha256"]
            or canonical["relative_path"] in observed
            or not isinstance(descriptor, Mapping)
        ):
            raise RuntimeError("context shard conservation drifted before hydration")
        context_paths.append(
            str(
                _bound_internal_parquet(
                    descriptor,
                    volume_root=context_root,
                    where="context shard frame",
                )
            )
        )
        observed.add(canonical["relative_path"])
    if observed != {
        row["relative_path"] for row in support["inventories"]["canonical"]
    }:
        raise RuntimeError("context hydration lacks canonical partitions")
    staging = root.parent / ".publishing-hydrated"
    if staging.exists():
        raise FileExistsError("stale hydrated publication requires reconciliation")
    staging.mkdir(parents=True)
    output = staging / "eligible-frame.parquet"
    hydration = _hydrate_context_frames(
        near_path=near_path,
        context_paths=context_paths,
        output=output,
        threads=PREPARE_REDUCE_CPU,
    )
    output_rows = hydration["output_rows"]
    if output_rows != near_receipt["output_rows"]:
        raise RuntimeError("hydrated universe did not conserve near-reduced rows")
    descriptor = _descriptor(output, root=staging, row_count=output_rows)
    body = {
        "kind": "modernbert-acquisition-hydrated-v1",
        "schema_version": "1.0.0",
        "pre_prepare_id": pre_prepare_id,
        "near_reduce_receipt_id": near_receipt["receipt_id"],
        "context_partition_count": len(context_paths),
        "context_coverage": hydration,
        "output_rows": output_rows,
        "frame": descriptor,
        "wall_seconds": round(time.monotonic() - started, 3),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    (staging / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    staging.replace(root)
    volume.commit()
    return receipt


def _checkpoint_root(run_id: str, *, volume_root: Path = VOLUME_PATH) -> Path:
    return _run_root(run_id, volume_root=volume_root) / "checkpoint-scoring"


def _normalise_surface(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _simhash(text: str) -> int:
    tokens = _normalise_surface(text).split()
    features = (
        [" ".join(tokens[index : index + 3]) for index in range(max(1, len(tokens) - 2))]
        if len(tokens) >= 3
        else tokens or [""]
    )
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, weight in enumerate(weights) if weight >= 0)


def _bounded(text: Any, *, limit: int) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str) or not text:
        raise ValueError("source context must be null or non-empty text")
    return (
        importlib.import_module("reddit_china_stance.context_assembly")
        .deterministic_head_tail(text, max_chars=limit)
        .text
    )


def _cluster_near_duplicates(
    rows: Sequence[dict[str, Any]],
    *,
    progress: Callable[[int, int, float], None] | None = None,
    progress_interval: int = 100_000,
) -> list[dict[str, Any]]:
    """Greedily cluster the full eligible universe under the registered relation.

    Four 16-bit SimHash bands give an exact pigeonhole candidate index for a
    Hamming distance at most three.  Every qualifying pair is still checked
    against the registered length and Hamming predicate, so buckets are only an
    acceleration, never an approximation.  The canonical cluster representative
    is deterministic and is the only member returned for probability sampling.
    """
    if progress_interval < 1:
        raise ValueError("near-duplicate progress interval must be positive")
    packet = _packet_module()
    started = time.monotonic()
    ordered = sorted(
        rows, key=lambda row: _canonical_sha256(["acquisition-near", row["record_id"]])
    )
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(ordered):
        value = row["simhash"]
        candidates: set[int] = set()
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            candidates.update(buckets.get(key, ()))
        for other_index in candidates:
            other = ordered[other_index]
            if packet._is_near_duplicate(
                simhash=value,
                chars=row["normalised_chars"],
                comparisons=[(other["simhash"], other["normalised_chars"])],
            ):
                union(index, other_index)
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            buckets.setdefault(key, []).append(index)
        completed = index + 1
        if progress is not None and (
            completed % progress_interval == 0 or completed == len(ordered)
        ):
            progress(completed, len(ordered), time.monotonic() - started)
    representatives: dict[int, int] = {}
    for index in range(len(ordered)):
        root = find(index)
        representatives.setdefault(root, index)
    for index, row in enumerate(ordered):
        representative = representatives[find(index)]
        row["near_duplicate_cluster_id"] = _canonical_sha256(
            ["acquisition-near-cluster-v1", ordered[representative]["record_id"]]
        )
    return [row for index, row in enumerate(ordered) if representatives[find(index)] == index]


def _bound_parquet(
    descriptor: Mapping[str, Any], *, volume_root: Path, where: str, expected_rows: int
) -> Path:
    import pyarrow.parquet as pq

    required = {"relative_path", "sha256", "bytes", "row_count"}
    if not isinstance(descriptor, Mapping) or not required <= set(descriptor):
        raise ValueError(f"{where} descriptor is incomplete")
    path = volume_root / _safe_relative(descriptor["relative_path"], where=where)
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != _sha256(descriptor["sha256"], where=f"{where} SHA-256")
        or descriptor["row_count"] != expected_rows
        or pq.ParquetFile(path).metadata.num_rows != expected_rows
    ):
        raise RuntimeError(f"{where} descriptor or row count drifted")
    return path


def _bound_internal_parquet(
    descriptor: Mapping[str, Any], *, volume_root: Path, where: str
) -> Path:
    """Validate a Parquet produced inside this staged preparation run."""
    expected_rows = descriptor.get("row_count") if isinstance(descriptor, Mapping) else None
    if not isinstance(expected_rows, int) or expected_rows < 0:
        raise ValueError(f"{where} descriptor lacks a valid row count")
    return _bound_parquet(
        descriptor,
        volume_root=volume_root,
        where=where,
        expected_rows=expected_rows,
    )


def _bound_receipt_parquet(
    descriptor: Mapping[str, Any], *, volume_root: Path, where: str
) -> Path:
    """Validate an immutable upstream Parquet against its receipt descriptor."""
    import pyarrow.parquet as pq

    required = {
        "relative_path",
        "sha256",
        "bytes",
        "row_count",
        "producer_receipt_sha256",
    }
    if not isinstance(descriptor, Mapping) or set(descriptor) != required:
        raise ValueError(f"{where} receipt-backed descriptor is incomplete")
    path = volume_root / _safe_relative(descriptor["relative_path"], where=where)
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path)
        != _sha256(descriptor["sha256"], where=f"{where} SHA-256")
        or pq.ParquetFile(path).metadata.num_rows != descriptor["row_count"]
        or not isinstance(descriptor["producer_receipt_sha256"], str)
        or len(descriptor["producer_receipt_sha256"]) != 64
    ):
        raise RuntimeError(f"{where} producer-receipt binding drifted")
    return path


def _bound_file(descriptor: Mapping[str, Any], *, volume_root: Path, where: str) -> Path:
    required = {"relative_path", "sha256", "bytes"}
    if not isinstance(descriptor, Mapping) or not required <= set(descriptor):
        raise ValueError(f"{where} descriptor is incomplete")
    path = volume_root / _safe_relative(descriptor["relative_path"], where=where)
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != _sha256(descriptor["sha256"], where=f"{where} SHA-256")
    ):
        raise RuntimeError(f"{where} descriptor drifted")
    return path


def _retained_manifest(spec: Mapping[str, Any], *, volume_root: Path) -> dict[str, Any]:
    del volume_root  # The manifest is image-bound; its descriptors bind retained Volume files.
    manifest, source_bundle = _validate_retained_evidence()
    if spec.get("retained_manifest_sha256") != _canonical_sha256(manifest) or spec.get(
        "retained_source_bundle_sha256"
    ) != _canonical_sha256(source_bundle):
        raise RuntimeError("acquisition spec retained evidence binding drifted")
    return manifest


def _derive_retained_calibration(
    spec: Mapping[str, Any], *, volume_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialise the exact formerly-unused 600-row calibration membership.

    The caller cannot provide a frame.  This reconstructs it from the retained
    split plus the exact source, mapping, blinded-input, teacher-label and bridge
    artefacts bound by the successful factorised run.
    """

    training = _training_module()
    manifest = _retained_manifest(spec, volume_root=volume_root)
    bindings = manifest["experiment_contract"]["bindings"]
    expected = {
        "teacher_ledger",
        "teacher_blinded_input",
        "teacher_private_mapping",
        "source_parquet",
        "split_manifest",
        "bridge_exposure_register",
    }
    if not expected <= set(bindings):
        raise RuntimeError("retained run omits calibration derivation artefacts")
    paths = {
        "teacher_ledger": _bound_parquet(
            bindings["teacher_ledger"],
            volume_root=volume_root,
            where="teacher ledger",
            expected_rows=10_000,
        ),
        "teacher_private_mapping": _bound_parquet(
            bindings["teacher_private_mapping"],
            volume_root=volume_root,
            where="teacher private mapping",
            expected_rows=10_000,
        ),
        "source_parquet": _bound_parquet(
            bindings["source_parquet"],
            volume_root=volume_root,
            where="teacher source",
            expected_rows=10_000,
        ),
        "teacher_blinded_input": _bound_file(
            bindings["teacher_blinded_input"],
            volume_root=volume_root,
            where="teacher blinded input",
        ),
        "split_manifest": _bound_file(
            bindings["split_manifest"],
            volume_root=volume_root,
            where="retained split membership",
        ),
        "bridge_exposure_register": _bound_file(
            bindings["bridge_exposure_register"],
            volume_root=volume_root,
            where="bridge exposure register",
        ),
    }
    joined = training._joined_rows_from_teacher_artifacts(
        teacher_labels_parquet_path=paths["teacher_ledger"],
        blinded_input_json_path=paths["teacher_blinded_input"],
        private_mapping_parquet_path=paths["teacher_private_mapping"],
        source_parquet_path=paths["source_parquet"],
        expected_teacher_run_id=bindings["teacher_run_id"],
    )
    if len(joined) != 10_000:
        raise RuntimeError("retained calibration join does not conserve 10,000 rows")
    joined_by_id = {row["item_id"]: row for row in joined}
    if len(joined_by_id) != len(joined):
        raise RuntimeError("retained calibration join has duplicate sample IDs")

    membership = _read_json(paths["split_manifest"], where="retained split membership")
    required_membership = {
        "schema_version",
        "kind",
        "contract_digest",
        "source_count",
        "probability_designs",
        "rare_cell_support",
        "memberships",
        "membership_id",
    }
    if (
        set(membership) != required_membership
        or membership.get("kind") != "modernbert-factorised-private-membership-v1"
        or membership.get("source_count") != 10_000
    ):
        raise RuntimeError("retained split membership schema drifted")
    membership_body = {key: membership[key] for key in required_membership - {"membership_id"}}
    if membership.get("membership_id") != training.canonical_sha256(membership_body):
        raise RuntimeError("retained split membership digest drifted")
    members = membership.get("memberships")
    if not isinstance(members, list) or len(members) != 10_000:
        raise RuntimeError("retained split membership does not conserve source rows")

    bridge = training.validate_private_exposure_register(
        _read_json(paths["bridge_exposure_register"], where="bridge exposure register"),
        scope="bridge",
    )
    bridge_threads = set(bridge["thread_ids"])
    if len(bridge_threads) != 480:
        raise RuntimeError("retained bridge register does not contain exactly 480 threads")

    rows: list[dict[str, Any]] = []
    for member in members:
        if not isinstance(member, Mapping) or member.get("frame") != "calibration":
            continue
        item_id = member.get("item_id")
        source = joined_by_id.get(item_id)
        if source is None:
            raise RuntimeError("calibration membership references an unknown source row")
        if (
            member.get("thread_id") != source["thread_id"]
            or member.get("bridge_exposed") is not False
            or source["thread_id"] in bridge_threads
            or member.get("quality_tier") != source["quality_tier"]
            or member.get("primary_training_eligible") is not source["primary_training_eligible"]
            or member.get("label_sha256") != source["label_sha256"]
            or member.get("source_row_sha256") != source["source_row_sha256"]
            or source["primary_training_eligible"] is not True
        ):
            raise RuntimeError("calibration membership/source/teacher binding drifted")
        rows.append(
            {
                "item_id": item_id,
                "frame": "acquisition_evaluation",
                "thread_id": source["thread_id"],
                "target_text": source["target_text"],
                "parent_context": source["parent_context"],
                "submission_context": source["submission_context"],
                "label_json": source["label_json"],
                "quality_tier": source["quality_tier"],
                "selection_component": member["selection_component"],
                "selection_stratum": member["selection_stratum"],
                "inclusion_probability_numerator": member["inclusion_probability_numerator"],
                "inclusion_probability_denominator": member["inclusion_probability_denominator"],
                "inclusion_probability": member["inclusion_probability"],
                "probability_scope": member["probability_scope"],
            }
        )
    rows.sort(key=lambda row: row["item_id"])
    if len(rows) != 600 or len({row["thread_id"] for row in rows}) != 600:
        raise RuntimeError("retained calibration frame is not exactly 600 unique threads")
    derivation = {
        "base_factorised_run_id": BASE_FACTORISED_RUN_ID,
        "membership_id": membership["membership_id"],
        "row_count": 600,
        "thread_set_sha256": _canonical_sha256(sorted(row["thread_id"] for row in rows)),
        "row_projection_sha256": _canonical_sha256(rows),
        "source_artifacts": {
            key: {
                "sha256": bindings[key]["sha256"],
                "bytes": bindings[key]["bytes"],
                **(
                    {"row_count": bindings[key]["row_count"]}
                    if "row_count" in bindings[key]
                    else {}
                ),
            }
            for key in sorted(paths)
        },
    }
    return rows, {**derivation, "derivation_id": _canonical_sha256(derivation)}


def _packet_bounded_fields(row: Mapping[str, Any]) -> dict[str, str | None]:
    packet = _packet_module()
    return {
        "target_text": _bounded(row.get("target_text"), limit=packet.MAX_TARGET_CHARS),
        "submission_context": _bounded(
            row.get("submission_context"), limit=packet.MAX_CONTEXT_CHARS
        ),
        "parent_context": _bounded(row.get("parent_context"), limit=packet.MAX_PARENT_CHARS),
    }


def _build_exposure_ledger(
    spec: Mapping[str, Any],
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    volume_root: Path,
) -> dict[str, Any]:
    """Derive all acquisition exclusions from immutable upstream artefacts.

    The register is deliberately derived here rather than accepted as an opaque
    caller file.  It retains only hashes plus SimHash/length pairs and binds the
    four exposure universes required by the protocol.
    """
    import pyarrow.parquet as pq

    training = _training_module()
    manifest = _retained_manifest(spec, volume_root=volume_root)
    bindings = manifest["experiment_contract"]["bindings"]
    descriptors = {
        "teacher_10k": bindings["source_parquet"],
        "legacy_452": bindings["legacy_proxy"],
        "bridge_480": bindings["bridge_exposure_register"],
        "development_600": bindings["development_frame"],
        "acquisition_evaluation_600": {
            "row_count": len(calibration_rows),
            "sha256": _canonical_sha256(calibration_rows),
        },
    }
    sources = {
        "teacher_10k": _bound_parquet(
            bindings["source_parquet"],
            volume_root=volume_root,
            where="teacher source",
            expected_rows=10_000,
        ),
        "development_600": _bound_parquet(
            bindings["development_frame"],
            volume_root=volume_root,
            where="factorised development",
            expected_rows=600,
        ),
        "legacy_452": _bound_parquet(
            bindings["legacy_proxy"],
            volume_root=volume_root,
            where="legacy development proxy",
            expected_rows=452,
        ),
    }
    bridge_path = _bound_file(
        bindings["bridge_exposure_register"],
        volume_root=volume_root,
        where="bridge exposure register",
    )
    bridge = training.validate_private_exposure_register(
        _read_json(bridge_path, where="bridge exposure register"), scope="bridge"
    )
    if len(bridge["thread_ids"]) != 480:
        raise RuntimeError("bridge exposure register does not contain 480 threads")
    packet = _packet_module()
    record_hashes: set[str] = set()
    thread_hashes: set[str] = set()
    exact_hashes: set[str] = set()
    normalised_hashes: set[str] = set()
    references: set[tuple[int, int]] = set()
    exposure_rows: list[tuple[str | None, str | None, str]] = []
    teacher_table = pq.read_table(
        sources["teacher_10k"], columns=["sample_id", "thread_id", "target_text"]
    )
    if tuple(teacher_table.column_names) != ("sample_id", "thread_id", "target_text"):
        raise RuntimeError("teacher source exposure schema drifted")
    exposure_rows.extend(
        (row["sample_id"], row["thread_id"], row["target_text"])
        for row in teacher_table.to_pylist()
    )
    development_table = pq.read_table(
        sources["development_600"], columns=["item_id", "thread_id", "target_text"]
    )
    if tuple(development_table.column_names) != ("item_id", "thread_id", "target_text"):
        raise RuntimeError("development-frame exposure schema drifted")
    exposure_rows.extend(
        (row["item_id"], row["thread_id"], row["target_text"])
        for row in development_table.to_pylist()
    )
    legacy_table = pq.read_table(sources["legacy_452"])
    legacy_columns = (
        "source_sample_id",
        "target_text",
        "parent_context",
        "submission_context",
        "split",
        "resolution",
        "label_json",
    )
    if tuple(legacy_table.column_names) != legacy_columns:
        raise RuntimeError("legacy 452 exposure schema drifted")
    exposure_rows.extend(
        (row["source_sample_id"], None, row["target_text"])
        for row in legacy_table.select(["source_sample_id", "target_text"]).to_pylist()
    )
    exposure_rows.extend(
        (row["item_id"], row["thread_id"], row["target_text"]) for row in calibration_rows
    )
    if len(exposure_rows) != 10_000 + 600 + 452 + 600:
        raise RuntimeError("exposure source rows do not conserve registered counts")
    for record_id, thread, raw_text in exposure_rows:
        bounded = _packet_bounded_fields(
            {"target_text": raw_text, "submission_context": None, "parent_context": None}
        )["target_text"]
        if not isinstance(bounded, str) or not bounded:
            raise RuntimeError("exposure surface is invalid after exact packet bounding")
        normalised = _normalise_surface(bounded)
        if record_id is not None:
            if not isinstance(record_id, str) or not record_id:
                raise RuntimeError("exposure record ID is invalid")
            record_hashes.add(hashlib.sha256(record_id.encode()).hexdigest())
        if thread is not None:
            if not isinstance(thread, str) or not thread:
                raise RuntimeError("exposure thread ID is invalid")
            thread_hashes.add(hashlib.sha256(thread.encode()).hexdigest())
        exact_hashes.add(hashlib.sha256(bounded.encode()).hexdigest())
        normalised_hashes.add(hashlib.sha256(normalised.encode()).hexdigest())
        if len(normalised) >= packet.NEAR_DUPLICATE_MIN_CHARS:
            references.add((packet._simhash(bounded), len(normalised)))
    for thread in bridge["thread_ids"]:
        thread_hashes.add(hashlib.sha256(thread.encode()).hexdigest())
    return {
        "record_id_sha256": record_hashes,
        "thread_sha256": thread_hashes,
        "exact_surface_sha256": exact_hashes,
        "normalised_surface_sha256": normalised_hashes,
        "near_duplicate_references": sorted(references),
        "source_descriptors_sha256": _canonical_sha256(descriptors),
    }


def _prepare_sql_universe(
    *,
    paths: Mapping[str, Sequence[Path]],
    exclusions: Mapping[str, Any],
    spec: Mapping[str, Any],
    reduce: bool = True,
    include_context: bool = True,
    threads: int = PREPARE_WORKER_CPU,
) -> list[dict[str, Any]]:
    """Filter row exposures, then retain one deterministic row per shard/thread.

    Global reduction resolves threads crossing shards, exact surfaces and corpus
    near-duplicate components after all source-shard receipts are conserved.
    """
    import duckdb

    database = duckdb.connect(database=":memory:")
    if threads not in {4, 8}:
        raise ValueError("prepare SQL threads must be exactly 4 or 8")
    database.execute(f"SET threads = {threads}")
    candidate_paths = [str(path) for path in paths["candidates"]]
    language_paths = [str(path) for path in paths["languages"]]
    canonical_paths = [str(path) for path in paths["canonical"]]
    try:
        database.execute(
            """
            CREATE TEMP TABLE english AS
            SELECT c.record_id, c.content_type, c.retrieval_channels
            FROM read_parquet(?) c JOIN read_parquet(?) l USING(record_id)
            WHERE l.provisional_status = 'provisional_english'
            """,
            [candidate_paths, language_paths],
        )
        total, distinct_total = database.execute(
            "SELECT count(*), count(DISTINCT record_id) FROM english"
        ).fetchone()
        if total != distinct_total:
            raise RuntimeError("provisional-English source inventory is non-unique")
        database.execute(
            """
            CREATE TEMP TABLE canonical_records AS
            SELECT record_id, submission_id, subreddit, year, content_type, text, parent_id
            FROM read_parquet(?)
            """,
            [canonical_paths],
        )
        database.execute(
            """
            CREATE TEMP TABLE base AS
            SELECT s.record_id, s.submission_id AS thread_id, s.subreddit, s.year,
                   s.content_type,
                   CASE WHEN list_contains(e.retrieval_channels, 'direct_lexical')
                        THEN 'direct' ELSE 'expanded_only' END AS retrieval_mode,
                   s.text AS target_text, s.parent_id
            FROM canonical_records s
            JOIN english e ON s.record_id=e.record_id AND s.content_type=e.content_type
            """,
        )
        observed, distinct_observed = database.execute(
            "SELECT count(*), count(DISTINCT record_id) FROM base"
        ).fetchone()
        if observed != total or observed != distinct_observed:
            raise RuntimeError("candidate/canonical join did not conserve provisional-English rows")
        if include_context:
            # Compatibility path for bounded local tests.  Production shards
            # deliberately defer this to the registered second corpus pass so
            # cross-year parents/submissions are resolved exactly once.
            missing_cross_thread = database.execute(
                """
                SELECT count(*) FROM base b
                JOIN canonical_records p ON b.parent_id=p.record_id
                WHERE b.content_type='comment' AND p.submission_id<>b.thread_id
                """
            ).fetchone()[0]
            if missing_cross_thread:
                raise RuntimeError("canonical parent context crosses submission thread")
            row_cursor = database.execute(
                """
                SELECT b.record_id, b.thread_id, b.subreddit, b.year, b.content_type,
                       b.retrieval_mode, b.target_text, b.parent_id,
                       submission.text AS submission_context, parent.text AS parent_context
                FROM base b
                LEFT JOIN canonical_records submission
                    ON b.content_type='comment' AND submission.record_id=b.thread_id
                       AND submission.content_type='submission'
                LEFT JOIN canonical_records parent
                    ON b.content_type='comment' AND parent.record_id=b.parent_id
                       AND parent.submission_id=b.thread_id
                """
            )
        else:
            row_cursor = database.execute(
                """
                SELECT record_id, thread_id, subreddit, year, content_type,
                       retrieval_mode, target_text, parent_id,
                       NULL::VARCHAR AS submission_context,
                       NULL::VARCHAR AS parent_context
                FROM base
                """
            )
        packet = _packet_module()
        if packet.NEAR_DUPLICATE_MAX_HAMMING != 3:
            raise RuntimeError("acquisition exposure index requires the frozen Hamming radius 3")
        reference_buckets: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for reference in exclusions["near_duplicate_references"]:
            if (
                not isinstance(reference, (list, tuple))
                or len(reference) != 2
                or not isinstance(reference[0], int)
                or not isinstance(reference[1], int)
            ):
                raise RuntimeError("near-duplicate exposure reference drifted")
            pair = (reference[0], reference[1])
            for band in range(4):
                key = (band, (pair[0] >> (band * 16)) & 0xFFFF)
                reference_buckets.setdefault(key, set()).add(pair)
        thread_representatives: dict[str, dict[str, Any]] = {}
        while batch := row_cursor.fetchmany(10_000):
            for (
                record_id,
                thread_id,
                subreddit,
                year,
                content_type,
                retrieval_mode,
                target_text,
                _parent_id,
                submission,
                parent,
            ) in batch:
                if not all(
                    isinstance(item, str) and item
                    for item in (
                        record_id,
                        thread_id,
                        subreddit,
                        content_type,
                        retrieval_mode,
                        target_text,
                    )
                ):
                    raise RuntimeError("canonical source has an invalid required field")
                bounded = _packet_bounded_fields(
                    {
                        "target_text": target_text,
                        "submission_context": submission,
                        "parent_context": parent,
                    }
                )
                bounded_target = bounded["target_text"]
                if not isinstance(bounded_target, str) or not bounded_target:
                    raise RuntimeError("canonical target is empty after packet bounding")
                normalised = _normalise_surface(bounded_target)
                exact = hashlib.sha256(bounded_target.encode()).hexdigest()
                normalised_sha = hashlib.sha256(normalised.encode()).hexdigest()
                if (
                    hashlib.sha256(record_id.encode()).hexdigest()
                    in exclusions["record_id_sha256"]
                    or hashlib.sha256(thread_id.encode()).hexdigest()
                    in exclusions["thread_sha256"]
                    or exact in exclusions["exact_surface_sha256"]
                    or normalised_sha in exclusions["normalised_surface_sha256"]
                ):
                    continue
                simhash = packet._simhash(bounded_target)
                normalised_chars = len(normalised)
                candidate_references: set[tuple[int, int]] = set()
                for band in range(4):
                    key = (band, (simhash >> (band * 16)) & 0xFFFF)
                    candidate_references.update(reference_buckets.get(key, ()))
                if packet._is_near_duplicate(
                    simhash=simhash,
                    chars=normalised_chars,
                    comparisons=sorted(candidate_references),
                ):
                    continue
                candidate = {
                    "record_id": record_id,
                    "thread_id": thread_id,
                    "subreddit": subreddit,
                    "year": int(year),
                    "content_type": content_type,
                    "retrieval_mode": retrieval_mode,
                    "target_text": bounded_target,
                    "parent_id": _parent_id,
                    "submission_context": bounded["submission_context"],
                    "parent_context": bounded["parent_context"],
                    "exact_surface_sha256": exact,
                    "normalised_surface_sha256": normalised_sha,
                    "simhash": simhash,
                    "normalised_chars": normalised_chars,
                    "thread_tiebreak_sha256": _canonical_sha256(
                        ["acquisition-thread-v1", record_id]
                    ),
                    "surface_tiebreak_sha256": _canonical_sha256(
                        ["acquisition-surface-v1", record_id]
                    ),
                }
                previous = thread_representatives.get(thread_id)
                if (
                    previous is None
                    or candidate["thread_tiebreak_sha256"]
                    < previous["thread_tiebreak_sha256"]
                ):
                    thread_representatives[thread_id] = candidate
        source = list(thread_representatives.values())
    finally:
        database.close()
    if not reduce:
        return sorted(source, key=lambda row: row["record_id"])
    return _reduce_prepare_source(source)


def _reduce_prepare_source(source: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply global thread, surface and near-duplicate representative ranking."""
    by_thread: dict[str, dict[str, Any]] = {}
    for row in source:
        previous = by_thread.get(row["thread_id"])
        if previous is None or _canonical_sha256(
            ["acquisition-thread-v1", row["record_id"]]
        ) < _canonical_sha256(["acquisition-thread-v1", previous["record_id"]]):
            by_thread[row["thread_id"]] = row
    by_surface: dict[str, dict[str, Any]] = {}
    for row in by_thread.values():
        key = row["normalised_surface_sha256"]
        previous = by_surface.get(key)
        if previous is None or _canonical_sha256(
            ["acquisition-surface-v1", row["record_id"]]
        ) < _canonical_sha256(["acquisition-surface-v1", previous["record_id"]]):
            by_surface[key] = row
    result = _cluster_near_duplicates(list(by_surface.values()))
    if len(result) < 2_600:
        raise RuntimeError(
            "eligible acquisition universe cannot support both arms and evaluation reserve"
        )
    return sorted(result, key=lambda row: row["record_id"])


def _prepared_contract(
    spec: Mapping[str, Any],
    *,
    inventories: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    universe: Sequence[Mapping[str, Any]] | None,
    rare_cells: Sequence[Mapping[str, Any]],
    rare_cell_support_summary: Mapping[str, Any],
    calibration_derivation: Mapping[str, Any],
    runtime_source_bundle: Mapping[str, Any],
    policy_sha256: str,
    eligible_frame_path: Path | None = None,
) -> dict[str, Any]:
    if spec.get("base_factorised_run_id") != BASE_FACTORISED_RUN_ID:
        raise ValueError("acquisition must bind the exact retained B4 source run")
    if (
        calibration_derivation.get("base_factorised_run_id") != BASE_FACTORISED_RUN_ID
        or calibration_derivation.get("row_count") != 600
        or not isinstance(calibration_derivation.get("derivation_id"), str)
    ):
        raise ValueError("acquisition must derive the retained 600-row evaluation frame")
    clean_rare_cells = list(_acquisition_module().validate_rare_cells(rare_cells))
    if (
        len(clean_rare_cells) != 3
        or rare_cell_support_summary.get("supported_cell_count") != 3
        or rare_cell_support_summary.get("retained_cell_count") != 8
    ):
        raise ValueError("acquisition requires exactly three evaluation-supported rare cells")
    source_bundle_id = _sha256(
        runtime_source_bundle.get("source_bundle_id"),
        where="acquisition runtime source bundle ID",
    )
    projection_fields = (
        "record_id",
        "thread_id",
        "near_duplicate_cluster_id",
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
        "exact_surface_sha256",
        "normalised_surface_sha256",
    )
    if (universe is None) == (eligible_frame_path is None):
        raise ValueError("prepared contract needs exactly one eligible-frame source")
    if universe is not None:
        eligible_frame_sha256 = _canonical_sha256(
            [{key: row[key] for key in projection_fields} for row in universe]
        )
        eligible_population_rows = len(universe)
    else:
        eligible_frame_sha256, eligible_population_rows = (
            _canonical_parquet_projection_sha256(
                eligible_frame_path,
                columns=projection_fields,
            )
        )
    return {
        "kind": PREPARED_KIND,
        "schema_version": "1.0.0",
        "policy_sha256": _sha256(policy_sha256, where="acquisition policy SHA-256"),
        "base_factorised_run_id": BASE_FACTORISED_RUN_ID,
        "launch_spec_sha256": _launch_spec_sha256(spec),
        "runtime_source_bundle": dict(runtime_source_bundle),
        "runtime_source_bundle_sha256": source_bundle_id,
        "analysis_language": {
            "main_corpus_status": "provisional_english",
            "human_language_gate_accepted": False,
        },
        "source_inventory": inventories,
        "source_inventory_sha256": _canonical_sha256(inventories),
        "exclusion_ledger_sha256": _canonical_sha256(
            {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in exclusions.items()
            }
        ),
        "eligible_frame_sha256": eligible_frame_sha256,
        "eligible_population_rows": eligible_population_rows,
        "rare_cells": clean_rare_cells,
        "rare_cell_list_sha256": _canonical_sha256(clean_rare_cells),
        "rare_cell_support_summary": dict(rare_cell_support_summary),
        "acquisition_evaluation": {
            "row_count": 600,
            "role": "unused_calibration_membership_repurposed_once",
            "derivation_id": calibration_derivation["derivation_id"],
            "thread_set_sha256": calibration_derivation["thread_set_sha256"],
            "row_projection_sha256": calibration_derivation["row_projection_sha256"],
        },
        "locked_test_rows_accessed": 0,
    }


def _canonical_parquet_projection_sha256(
    path: Path, *, columns: Sequence[str]
) -> tuple[str, int]:
    """Stream the exact canonical JSON-list digest without Python row materialisation."""
    import pyarrow.parquet as pq

    if not path.is_file() or not columns:
        raise ValueError("canonical Parquet projection source is invalid")
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=list(columns), batch_size=65_536):
        for row in batch.to_pylist():
            if count:
                digest.update(b",")
            digest.update(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            count += 1
    digest.update(b"]")
    return digest.hexdigest(), count


def _freeze_rare_cells(
    spec: Mapping[str, Any],
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    volume_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain supported cells from the exact prior eight-cell membership list."""

    training = _training_module()
    manifest = _retained_manifest(spec, volume_root=volume_root)
    bindings = manifest["experiment_contract"]["bindings"]
    descriptor = bindings["training_frame"]
    path = volume_root / descriptor["relative_path"]
    rows, _reference = training.load_private_frame(path, descriptor, expected_frame="training")
    if len(rows) != 8_147:
        raise RuntimeError("rare-cell support frame does not contain the frozen 8,147 rows")
    split_path = _bound_file(
        bindings["split_manifest"],
        volume_root=volume_root,
        where="retained split membership",
    )
    membership = _read_json(split_path, where="retained split membership")
    retained = membership.get("rare_cell_support")
    if not isinstance(retained, list) or len(retained) != 8:
        raise RuntimeError("retained split does not bind the exact eight rare cells")
    retained_cells: set[tuple[str, str]] = set()
    for row in retained:
        if not isinstance(row, Mapping) or not {"target", "stance"} <= set(row):
            raise RuntimeError("retained rare-cell membership schema drifted")
        cell = (row["target"], row["stance"])
        if cell in retained_cells:
            raise RuntimeError("retained rare-cell membership contains duplicates")
        retained_cells.add(cell)

    data = _data_module()
    training_support = {cell: 0 for cell in retained_cells}
    evaluation_support = {cell: 0 for cell in retained_cells}

    def add_support(
        source_rows: Sequence[Mapping[str, Any]], counts: dict[tuple[str, str], int]
    ) -> None:
        for source_row in source_rows:
            encoding = data.encode_v2_label(json.loads(source_row["label_json"]))
            for index, target in enumerate(data.ANALYTIC_TARGET_CLASSES):
                if not encoding.stance_mask[index]:
                    continue
                stance = data.STANCE_CLASSES_B4[encoding.stance_b4_labels[index]]
                if (target, stance) in counts:
                    counts[(target, stance)] += 1

    add_support(rows, training_support)
    add_support(calibration_rows, evaluation_support)
    policy, _policy_sha256 = load_frozen_policy()
    minimum = policy["gate"]["rare_cell_minimum_evaluation_support"]
    if type(minimum) is not int or minimum < 10:
        raise RuntimeError("rare-cell evaluation support gate drifted")
    supported = sorted(cell for cell in retained_cells if evaluation_support[cell] >= minimum)
    if len(supported) != 3:
        raise RuntimeError("retained evaluation support must yield exactly three frozen rare cells")
    frozen = [
        {
            "target": target,
            "stance": stance,
            "training_support": training_support[(target, stance)],
        }
        for target, stance in supported
    ]
    clean = list(_acquisition_module().validate_rare_cells(frozen))
    if len(clean) != 3:
        raise RuntimeError("canonical rare-cell validation did not preserve exactly three cells")
    summary_body = {
        "retained_cell_count": 8,
        "evaluation_minimum_support": minimum,
        "supported_cell_count": len(clean),
        "excluded_unsupported_cell_count": 8 - len(clean),
        "evaluation_support_by_cell": [
            {
                "target": target,
                "stance": stance,
                "evaluation_support": evaluation_support[(target, stance)],
                "training_support": training_support[(target, stance)],
                "retained_for_gate": (target, stance) in supported,
            }
            for target, stance in sorted(retained_cells)
        ],
    }
    return clean, {**summary_body, "summary_id": _canonical_sha256(summary_body)}


def _write_private_frame(
    *, root: Path, contract: Mapping[str, Any], universe: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    final = root / "eligible-frame.parquet"
    if final.exists():
        if pq.ParquetFile(final).metadata.num_rows != len(universe):
            raise RuntimeError("existing acquisition frame has wrong row count")
        contract_path = root / "contract.json"
        if not contract_path.is_file() or _read_json(
            contract_path, where="existing acquisition frame contract"
        ) != dict(contract):
            raise RuntimeError("existing acquisition frame contract drifted")
    else:
        staging = root / ".eligible-frame.incomplete"
        if staging.exists():
            raise FileExistsError("stale acquisition frame staging requires manual reconciliation")
        staging.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(list(universe)), staging / final.name, compression="zstd"
        )
        (staging / "contract.json").write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(root / "published")
        (root / "published" / final.name).replace(final)
        (root / "published" / "contract.json").replace(root / "contract.json")
        (root / "published").rmdir()
    return _descriptor(final, root=root, row_count=len(universe))


def _write_private_frame_from_stage(
    *,
    root: Path,
    contract: Mapping[str, Any],
    source_path: Path,
    source_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote a validated hydrated stage without materialising rows in Python."""
    import shutil

    import pyarrow.parquet as pq

    expected_rows = source_descriptor.get("row_count")
    if (
        not source_path.is_file()
        or not isinstance(expected_rows, int)
        or expected_rows < 2_600
        or source_path.stat().st_size != source_descriptor.get("bytes")
        or _file_sha256(source_path) != source_descriptor.get("sha256")
        or pq.ParquetFile(source_path).metadata.num_rows != expected_rows
    ):
        raise RuntimeError("hydrated source descriptor drifted before promotion")
    final = root / "eligible-frame.parquet"
    contract_path = root / "contract.json"
    if final.exists():
        if (
            _file_sha256(final) != source_descriptor["sha256"]
            or pq.ParquetFile(final).metadata.num_rows != expected_rows
            or not contract_path.is_file()
            or _read_json(contract_path, where="existing acquisition frame contract")
            != dict(contract)
        ):
            raise RuntimeError("existing promoted acquisition frame drifted")
    else:
        staging = root / ".eligible-frame.incomplete"
        if staging.exists():
            raise FileExistsError("stale acquisition frame staging requires reconciliation")
        staging.mkdir(parents=True)
        shutil.copyfile(source_path, staging / final.name)
        (staging / "contract.json").write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(root / "published")
        (root / "published" / final.name).replace(final)
        (root / "published" / "contract.json").replace(contract_path)
        (root / "published").rmdir()
    return _descriptor(final, root=root, row_count=expected_rows)


def _prepared_frame_shards_root(run_id: str, *, volume_root: Path = VOLUME_PATH) -> Path:
    return _prepared_root(run_id, volume_root=volume_root) / "frame-shards"


def _materialise_prepared_frame_shards(
    *,
    run_id: str,
    prepared_frame_path: Path,
    prepared_frame: Mapping[str, Any],
    volume_root: Path = VOLUME_PATH,
) -> dict[str, Any]:
    """Atomically publish ten source-bound frame shards in canonical row order."""

    import pyarrow.parquet as pq

    row_count = prepared_frame.get("row_count")
    if (
        not isinstance(row_count, int)
        or not prepared_frame_path.is_file()
        or pq.ParquetFile(prepared_frame_path).metadata.num_rows != row_count
        or _file_sha256(prepared_frame_path) != prepared_frame.get("sha256")
    ):
        raise RuntimeError("prepared frame drifted before scoring-shard publication")
    plan = _scoring_plan_contract(row_count)
    final = _prepared_frame_shards_root(run_id, volume_root=volume_root)
    if (final / "receipt.json").is_file():
        return _validate_prepared_frame_shards(
            run_id=run_id,
            prepared_frame=prepared_frame,
            volume_root=volume_root,
        )
    if final.exists():
        raise FileExistsError("stale prepared-frame shard publication requires reconciliation")
    if any(final.parent.glob(".frame-shards.*.incomplete")):
        raise FileExistsError("stale prepared-frame shard attempt requires reconciliation")
    staging = final.parent / f".frame-shards.{uuid.uuid4().hex}.incomplete"
    staging.mkdir(parents=True)
    frames: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    for shard in plan["shards"]:
        table = _read_parquet_slice(
            prepared_frame_path,
            start=shard["start"],
            row_count=shard["row_count"],
        )
        ids = table["record_id"].to_pylist()
        if len(ids) != len(set(ids)) or any(not isinstance(value, str) for value in ids):
            raise RuntimeError("prepared-frame shard contains duplicate or invalid record IDs")
        ordered_ids.extend(ids)
        path = staging / f"shard={shard['shard_id']}" / "frame.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(table, path, compression="zstd")
        frames.append(
            {
                **shard,
                "frame": _descriptor(path, root=staging, row_count=shard["row_count"]),
            }
        )
    if len(ordered_ids) != row_count or len(set(ordered_ids)) != row_count:
        raise RuntimeError("prepared-frame shards do not conserve the eligible frame")
    body = {
        "kind": "modernbert-acquisition-prepared-frame-shards-v1",
        "schema_version": "1.0.0",
        "prepared_run_id": run_id,
        "prepared_frame_sha256": prepared_frame["sha256"],
        "prepared_frame_rows": row_count,
        "scoring_plan_sha256": plan["plan_id"],
        "frame_shards": frames,
        "record_id_sequence_sha256": _acquisition_module().canonical_sequence_sha256(
            ordered_ids
        ),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(receipt, where="prepared-frame shard receipt")
    _write_immutable_json(staging / "receipt.json", receipt)
    os.replace(staging, final)
    return receipt


def _validate_prepared_frame_shards(
    *,
    run_id: str,
    prepared_frame: Mapping[str, Any],
    volume_root: Path = VOLUME_PATH,
) -> dict[str, Any]:
    final = _prepared_frame_shards_root(run_id, volume_root=volume_root)
    receipt = _read_json(final / "receipt.json", where="prepared-frame shard receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    plan = _scoring_plan_contract(prepared_frame["row_count"])
    frames = receipt.get("frame_shards")
    if (
        receipt.get("receipt_id") != _canonical_sha256(body)
        or receipt.get("kind") != "modernbert-acquisition-prepared-frame-shards-v1"
        or receipt.get("schema_version") != "1.0.0"
        or receipt.get("prepared_run_id") != run_id
        or receipt.get("prepared_frame_sha256") != prepared_frame["sha256"]
        or receipt.get("prepared_frame_rows") != prepared_frame["row_count"]
        or receipt.get("scoring_plan_sha256") != plan["plan_id"]
        or not isinstance(frames, list)
        or len(frames) != SCORING_SHARD_COUNT
        or receipt.get("locked_test_rows_accessed") != 0
    ):
        raise RuntimeError("prepared-frame shard receipt binding drifted")
    if [
        {key: frame[key] for key in ("shard_id", "start", "stop", "row_count")}
        for frame in frames
        if isinstance(frame, Mapping)
    ] != plan["shards"]:
        raise RuntimeError("prepared-frame shard boundaries drifted")
    return receipt


def _load_prepared_frame_shard(
    *,
    run_id: str,
    prepared: Mapping[str, Any],
    shard: Mapping[str, Any],
    volume_root: Path = VOLUME_PATH,
) -> Path:
    receipt = _validate_prepared_frame_shards(
        run_id=run_id,
        prepared_frame=prepared["prepared_frame"],
        volume_root=volume_root,
    )
    matches = [
        frame
        for frame in receipt["frame_shards"]
        if isinstance(frame, Mapping) and frame.get("shard_id") == shard.get("shard_id")
    ]
    if len(matches) != 1 or any(matches[0].get(key) != shard.get(key) for key in shard):
        raise RuntimeError("prepared-frame shard lookup drifted")
    descriptor = matches[0].get("frame")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "relative_path",
        "sha256",
        "bytes",
        "row_count",
    }:
        raise RuntimeError("prepared-frame shard descriptor drifted")
    root = _prepared_frame_shards_root(run_id, volume_root=volume_root)
    path = root / _safe_relative(descriptor["relative_path"], where="prepared-frame shard path")
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != descriptor["sha256"]
        or descriptor["row_count"] != shard["row_count"]
    ):
        raise RuntimeError("prepared-frame shard content drifted")
    return path


def _materialise_acquisition_evaluation(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the exact reconstructed 600-row calibration frame once."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if (
        len(rows) != 600
        or derivation.get("row_count") != 600
        or derivation.get("row_projection_sha256") != _canonical_sha256(list(rows))
    ):
        raise RuntimeError("acquisition evaluation derivation drifted before publication")
    schema = pa.schema(
        [
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("frame", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("parent_context", pa.string()),
            pa.field("submission_context", pa.string()),
            pa.field("label_json", pa.string(), nullable=False),
            pa.field("quality_tier", pa.string(), nullable=False),
            pa.field("selection_component", pa.string(), nullable=False),
            pa.field("selection_stratum", pa.string()),
            pa.field("inclusion_probability_numerator", pa.int64()),
            pa.field("inclusion_probability_denominator", pa.int64()),
            pa.field("inclusion_probability", pa.float64()),
            pa.field("probability_scope", pa.string()),
        ]
    )
    expected = pa.Table.from_pylist(list(rows), schema=schema)
    final = root / "acquisition-evaluation.parquet"
    if final.exists():
        if not pq.read_table(final).equals(expected):
            raise RuntimeError("existing acquisition evaluation differs from derivation")
    else:
        staging = root / ".acquisition-evaluation.incomplete"
        if staging.exists():
            raise FileExistsError("stale acquisition evaluation staging requires reconciliation")
        staging.mkdir(parents=True)
        pq.write_table(expected, staging / final.name, compression="zstd")
        (staging / final.name).replace(final)
        staging.rmdir()
    if pq.ParquetFile(final).metadata.num_rows != 600:
        raise RuntimeError("materialised acquisition evaluation does not conserve 600 rows")
    return _descriptor(final, root=root, row_count=600)


def _public_preparation_receipt(
    *,
    contract: Mapping[str, Any],
    prepared_frame: Mapping[str, Any],
    acquisition_evaluation_frame: Mapping[str, Any],
) -> dict[str, Any]:
    prepared_run_id = _canonical_sha256(contract)
    body = {
        **{key: value for key, value in contract.items() if key != "source_inventory"},
        "prepared_run_id": prepared_run_id,
        "prepared_frame": dict(prepared_frame),
        "acquisition_evaluation_frame": dict(acquisition_evaluation_frame),
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(receipt, where="acquisition preparation receipt")
    return receipt


def promote_prepared_input_spec(
    *,
    input_spec: Mapping[str, Any],
    prepare_receipt: Mapping[str, Any],
    prepared_run_id: str,
) -> dict[str, Any]:
    """Bind a returned preparation ID into a new exact local launch spec.

    The public receipt is deliberately sufficient for this local transition: its
    canonical receipt ID binds the returned run ID and the exact pre-prepare input
    spec digest, while the private source inventory remains only on the Volume.
    """

    clean = validate_input_spec(input_spec, action="prepare")
    run_id = _sha256(prepared_run_id, where="prepared run ID")
    if not isinstance(prepare_receipt, Mapping):
        raise ValueError("preparation receipt must be a JSON object")
    receipt = dict(prepare_receipt)
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if (
        receipt.get("receipt_id") != _canonical_sha256(body)
        or receipt.get("kind") != PREPARED_KIND
        or receipt.get("schema_version") != "1.0.0"
        or receipt.get("prepared_run_id") != run_id
        or receipt.get("launch_spec_sha256") != _launch_spec_sha256(clean)
        or receipt.get("policy_sha256") != clean["policy_sha256"]
        or receipt.get("base_factorised_run_id") != clean["base_factorised_run_id"]
        or receipt.get("locked_test_rows_accessed") != 0
        or "source_inventory" in receipt
    ):
        raise RuntimeError("public preparation receipt/input-spec binding drifted")
    assert_metadata_only(receipt, where="acquisition preparation receipt promotion")
    promoted = {**clean, "prepared_run_id": run_id}
    return validate_input_spec(promoted, action="cuda-preflight")


def promote_prepared_input_file(
    *,
    input_spec_path: Path,
    prepare_receipt_path: Path,
    output_spec_path: Path,
    prepared_run_id: str,
) -> dict[str, Any]:
    """Write the post-prepare spec immutably, never replacing its input spec."""

    if input_spec_path.resolve() == output_spec_path.resolve():
        raise ValueError("promoted input spec must use a new output path")
    promoted = promote_prepared_input_spec(
        input_spec=_read_json(input_spec_path, where="pre-prepare acquisition input spec"),
        prepare_receipt=_read_json(
            prepare_receipt_path, where="public acquisition preparation receipt"
        ),
        prepared_run_id=prepared_run_id,
    )
    _write_immutable_json(output_spec_path, promoted)
    return promoted


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def prepare(spec: dict[str, Any], approved_cost_usd: str) -> dict[str, Any]:
    """Coordinate resumable shards then publish exactly one validated frame."""
    volume.reload()
    spec = validate_input_spec(spec, action="prepare")
    _validate_phase_approval(spec, action="prepare", approved_cost_usd=approved_cost_usd)
    state = _initialise_prepare_state(spec)
    policy_sha256 = state["policy_sha256"]
    paths = state["paths"]
    inventories = state["inventories"]
    pre_prepare_id = state["pre_prepare_id"]
    shards = state["shards"]
    calibration_rows = state["calibration_rows"]
    calibration_derivation = state["calibration_derivation"]
    exclusions = state["exclusions"]
    volume.commit()
    # Modal maps each small worker independently.  Valid receipt-bearing shards
    # are reused by the worker after interruption; no coordinator creates a
    # final namespace until exact conservation has passed.
    source_started = time.monotonic()
    shard_receipts: list[dict[str, Any]] = []
    shard_rows = 0
    for receipt in prepare_shard.map(
        [pre_prepare_id] * len(shards), shards, [spec] * len(shards)
    ):
        shard_receipts.append(receipt)
        shard_rows += int(receipt.get("row_count", 0))
        print(
            json.dumps(
                _progress_event(
                    stage="source-shards",
                    expected_shards=len(shards),
                    completed_shards=len(shard_receipts),
                    started=source_started,
                    rows=shard_rows,
                ),
                sort_keys=True,
            )
        )
    if len(shard_receipts) != len(shards):
        raise RuntimeError("prepare shard scheduler did not return every receipt")
    exact_started = time.monotonic()
    exact_receipt = prepare_exact_reduce.remote(pre_prepare_id)
    print(
        json.dumps(
            _progress_event(
                stage="exact-reduce",
                expected_shards=1,
                completed_shards=1,
                started=exact_started,
                rows=exact_receipt["output_rows"],
            ),
            sort_keys=True,
        )
    )
    near_started = time.monotonic()
    near_receipt = prepare_near_reduce.remote(pre_prepare_id)
    print(
        json.dumps(
            _progress_event(
                stage="near-reduce",
                expected_shards=1,
                completed_shards=1,
                started=near_started,
                rows=near_receipt["output_rows"],
            ),
            sort_keys=True,
        )
    )
    prepare_context_requests_stage.remote(pre_prepare_id)
    volume.reload()
    canonical_descriptors = {
        descriptor["relative_path"]: descriptor for descriptor in inventories["canonical"]
    }
    request_descriptors: dict[str, dict[str, Any]] = {}
    for descriptor in inventories["canonical"]:
        subreddit = next(
            part.split("=", 1)[1]
            for part in Path(descriptor["relative_path"]).parts
            if part.startswith("subreddit=")
        )
        if subreddit in request_descriptors:
            continue
        request_receipt = _read_json(
            _prepare_stage_root(pre_prepare_id)
            / "context-requests"
            / f"subreddit={subreddit}"
            / "receipt.json",
            where="context request receipt",
        )
        request_descriptor = request_receipt.get("requests")
        if not isinstance(request_descriptor, Mapping):
            raise RuntimeError("context request receipt lacks its frame")
        request_descriptors[subreddit] = dict(request_descriptor)
    context_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in paths["canonical"]:
        labels = {
            key: value
            for part in path.parts
            if "=" in part
            for key, value in [part.split("=", 1)]
        }
        subreddit = labels.get("subreddit")
        relative = path.relative_to(VOLUME_PATH).as_posix()
        if subreddit not in request_descriptors or relative not in canonical_descriptors:
            raise RuntimeError("context job lacks exact canonical/request descriptors")
        context_jobs.append(
            (dict(canonical_descriptors[relative]), dict(request_descriptors[subreddit]))
        )
    context_started = time.monotonic()
    context_receipts: list[dict[str, Any]] = []
    context_rows = 0
    for receipt in prepare_context_shard.map(
        [pre_prepare_id] * len(context_jobs),
        [job[0] for job in context_jobs],
        [job[1] for job in context_jobs],
    ):
        context_receipts.append(receipt)
        context_rows += int(receipt.get("row_count", 0))
        print(
            json.dumps(
                _progress_event(
                    stage="context-shards",
                    expected_shards=len(context_jobs),
                    completed_shards=len(context_receipts),
                    started=context_started,
                    rows=context_rows,
                ),
                sort_keys=True,
            )
        )
    if len(context_receipts) != len(context_jobs):
        raise RuntimeError("context scheduler did not return every canonical partition receipt")
    observed_context_partitions: set[str] = set()
    for receipt in context_receipts:
        partition_id = receipt.get("partition_id")
        if not isinstance(partition_id, str) or receipt.get("pre_prepare_id") != pre_prepare_id:
            raise RuntimeError("context shard receipt binding drifted")
        canonical_relative = receipt.get("canonical_relative_path")
        if (
            not isinstance(canonical_relative, str)
            or canonical_relative in observed_context_partitions
            or canonical_relative not in canonical_descriptors
            or receipt.get("canonical_sha256")
            != canonical_descriptors[canonical_relative]["sha256"]
        ):
            raise RuntimeError("context shard partition conservation drifted")
        observed_context_partitions.add(canonical_relative)
    if observed_context_partitions != set(canonical_descriptors):
        raise RuntimeError("context stage did not conserve all canonical partitions")
    hydration_started = time.monotonic()
    hydrated_receipt = prepare_hydrate.remote(pre_prepare_id)
    volume.reload()
    hydrated_receipt, hydrated_path = _load_prepare_stage_frame(
        pre_prepare_id, stage="hydrated"
    )
    progress = _progress_event(
        stage="hydrated",
        expected_shards=1,
        completed_shards=1,
        started=hydration_started,
        rows=hydrated_receipt["output_rows"],
    )
    print(json.dumps(progress, sort_keys=True))
    if calibration_rows is None:
        calibration_rows, observed_derivation = _derive_retained_calibration(
            spec, volume_root=VOLUME_PATH
        )
        if observed_derivation != calibration_derivation:
            raise RuntimeError("cached calibration derivation drifted before publication")
    rare_cells, rare_cell_support_summary = _freeze_rare_cells(
        spec, calibration_rows=calibration_rows, volume_root=VOLUME_PATH
    )
    runtime_source_bundle = _runtime_source_bundle()
    contract = _prepared_contract(
        spec,
        inventories=inventories,
        exclusions=exclusions,
        universe=None,
        rare_cells=rare_cells,
        rare_cell_support_summary=rare_cell_support_summary,
        calibration_derivation=calibration_derivation,
        runtime_source_bundle=runtime_source_bundle,
        policy_sha256=policy_sha256,
        eligible_frame_path=hydrated_path,
    )
    run_id = _canonical_sha256(contract)
    root = _prepared_root(run_id)
    descriptor = _write_private_frame_from_stage(
        root=root,
        contract=contract,
        source_path=hydrated_path,
        source_descriptor=hydrated_receipt["frame"],
    )
    _materialise_prepared_frame_shards(
        run_id=run_id,
        prepared_frame_path=root / descriptor["relative_path"],
        prepared_frame=descriptor,
    )
    evaluation_descriptor = _materialise_acquisition_evaluation(
        root=root, rows=calibration_rows, derivation=calibration_derivation
    )
    public = _public_preparation_receipt(
        contract=contract,
        prepared_frame=descriptor,
        acquisition_evaluation_frame=evaluation_descriptor,
    )
    if public["prepared_run_id"] != run_id:
        raise RuntimeError("prepared run ID/public receipt binding drifted")
    _write_immutable_json(root / "receipt.json", public)
    volume.commit()
    return public


def _checkpoint_specs(
    spec: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> list[dict[str, Any]]:
    """Resolve, rather than accept, the six retained B4 checkpoint descriptors.

    The image-bound retained-v5 manifest supplies the trial inventory.  Each
    descriptor is reconstructed from that manifest and its final receipt; a
    hand-written checkpoint path/hash inventory is deliberately not accepted.
    """
    training = _training_module()
    manifest = _retained_manifest(spec, volume_root=volume_root)
    if manifest.get("experiment_run_id") != BASE_FACTORISED_RUN_ID:
        raise RuntimeError("retained factorised run ID drifted")
    expected = {
        (component, seed)
        for component in ("relevance", "target_stance_b4")
        for seed in (47, 61, 89)
    }
    seen: set[tuple[str, int]] = set()
    clean: list[dict[str, Any]] = []
    for trial in manifest["trials"]:
        component, seed = trial["component"], trial["optimiser_seed"]
        if (component, seed) not in expected:
            continue
        final, _attempts = training._trial_roots(  # Receipt layout authority.
            volume_root=volume_root,
            experiment_run_id=manifest["experiment_run_id"],
            component=component,
            trial_id=trial["trial_id"],
        )
        receipt = _read_json(final / "receipt.json", where="retained checkpoint receipt")
        clean_receipt = training.validate_trial_artifacts(
            final,
            receipt,
            experiment=manifest["experiment_contract"],
            trial_spec=trial,
            phase_run_id=manifest["phase_run_id"],
            run_manifest_sha256=training.canonical_sha256(manifest),
        )
        descriptor = clean_receipt["artifacts"]["checkpoint"]
        checkpoint = final / descriptor["relative_path"]
        if not checkpoint.is_file() or _file_sha256(checkpoint) != descriptor["sha256"]:
            raise RuntimeError("retained checkpoint artefact drifted")
        item = {
            "component": component,
            "optimiser_seed": seed,
            "trial_spec": dict(trial),
            "relative_path": checkpoint.relative_to(volume_root).as_posix(),
            "sha256": _sha256(descriptor["sha256"], where="checkpoint SHA-256"),
            "trial_id": trial["trial_id"],
            "development_predictions_relative_path": (
                final
                / clean_receipt["artifacts"]["private_development_predictions"]["relative_path"]
            )
            .relative_to(volume_root)
            .as_posix(),
            "development_predictions_sha256": _sha256(
                clean_receipt["artifacts"]["private_development_predictions"]["sha256"],
                where="development predictions SHA-256",
            ),
            "development_frame_relative_path": _safe_relative(
                manifest["experiment_contract"]["bindings"]["development_frame"]["relative_path"],
                where="development frame path",
            ).as_posix(),
            "development_frame_sha256": _sha256(
                manifest["experiment_contract"]["bindings"]["development_frame"]["sha256"],
                where="development frame SHA-256",
            ),
        }
        if (component, seed) in seen:
            raise ValueError("checkpoint inventory has duplicates")
        seen.add((component, seed))
        clean.append(item)
    if seen != expected:
        raise ValueError("checkpoint inventory is not the exact B4 three-seed cross product")
    return sorted(clean, key=lambda row: (row["component"], row["optimiser_seed"]))


def preflight_local(spec: Mapping[str, Any], *, volume_root: Path | None = None) -> dict[str, Any]:
    """Read-only launch validation that never invokes a Modal function."""

    clean = validate_input_spec(spec)
    policy, policy_sha256 = load_frozen_policy()
    if clean["policy_sha256"] != policy_sha256:
        raise RuntimeError("acquisition local preflight policy digest drifted")
    manifest, source_bundle = _validate_retained_evidence()
    if clean["retained_manifest_sha256"] != _canonical_sha256(manifest) or clean[
        "retained_source_bundle_sha256"
    ] != _canonical_sha256(source_bundle):
        raise RuntimeError("acquisition local preflight retained evidence drifted")
    bindings = manifest["experiment_contract"]["bindings"]
    volume_paths = sorted(
        {
            descriptor["relative_path"]
            for descriptor in bindings.values()
            if isinstance(descriptor, Mapping) and "relative_path" in descriptor
        }
    )
    for index, relative_path in enumerate(volume_paths):
        _safe_relative(relative_path, where=f"retained Volume binding {index}")
    retained_trials = [
        trial
        for trial in manifest["trials"]
        if trial["component"] in {"relevance", "target_stance_b4"}
    ]
    if len(retained_trials) != 6:
        raise RuntimeError("retained local manifest does not resolve six B4 trial bundles")
    resolved = 0
    checkpoint_bundle_sha256: str | None = None
    resolution = "trial-specifications-only"
    if volume_root is not None:
        checkpoints = _checkpoint_specs(clean, volume_root=volume_root)
        resolved = len(checkpoints)
        checkpoint_bundle_sha256 = _canonical_sha256(checkpoints)
        resolution = "checkpoint-artifacts-and-receipts"
    result = {
        "status": "validated",
        "policy_sha256": policy_sha256,
        "retained_manifest_sha256": clean["retained_manifest_sha256"],
        "retained_source_bundle_sha256": clean["retained_source_bundle_sha256"],
        "retained_volume_binding_count": len(volume_paths),
        "checkpoint_trial_bundles": len(retained_trials),
        "checkpoint_artifacts_resolved": resolved,
        "checkpoint_resolution": resolution,
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "prepared_run_id": clean["prepared_run_id"],
        "compute_ledger_sha256": clean["compute"]["ledger_sha256"],
        "phase_upper_cost_usd": dict(clean["compute"]["phase_upper_cost_usd"]),
        "locked_test_rows_accessed": 0,
        "policy_kind": policy["kind"],
    }
    assert_metadata_only(result, where="acquisition local preflight")
    return result


def validate_volume_bindings(
    spec: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    """Read-only validation of the retained v5 checkpoint bundles on the Volume."""

    clean = validate_input_spec(spec, action="validate-volume")
    _policy, policy_sha256 = load_frozen_policy()
    if clean["policy_sha256"] != policy_sha256:
        raise RuntimeError("acquisition Volume validation policy digest drifted")
    manifest = _retained_manifest(clean, volume_root=volume_root)
    checkpoints = _checkpoint_specs(clean, volume_root=volume_root)
    artefact_bindings: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_path = volume_root / _safe_relative(
            checkpoint["relative_path"], where="retained checkpoint path"
        )
        prediction_path = volume_root / _safe_relative(
            checkpoint["development_predictions_relative_path"],
            where="retained development prediction path",
        )
        receipt_path = checkpoint_path.parent / "receipt.json"
        if (
            not checkpoint_path.is_file()
            or _file_sha256(checkpoint_path) != checkpoint["sha256"]
            or not prediction_path.is_file()
            or _file_sha256(prediction_path) != checkpoint["development_predictions_sha256"]
            or not receipt_path.is_file()
        ):
            raise RuntimeError("retained checkpoint bundle artefact drifted")
        artefact_bindings.append(
            {
                "component": checkpoint["component"],
                "optimiser_seed": checkpoint["optimiser_seed"],
                "checkpoint_sha256": checkpoint["sha256"],
                "development_predictions_sha256": checkpoint["development_predictions_sha256"],
                "receipt_sha256": _file_sha256(receipt_path),
            }
        )
    if len(artefact_bindings) != 6:
        raise RuntimeError("retained Volume validation did not resolve six checkpoint bundles")
    body = {
        "status": "validated",
        "kind": "modernbert-acquisition-volume-validation-v1",
        "schema_version": "1.0.0",
        "policy_sha256": policy_sha256,
        "retained_manifest_sha256": _canonical_sha256(manifest),
        "retained_source_bundle_sha256": clean["retained_source_bundle_sha256"],
        "checkpoint_bundles": 6,
        "checkpoint_bundle_sha256": _canonical_sha256(checkpoints),
        "checkpoint_artefact_bindings_sha256": _canonical_sha256(artefact_bindings),
        "compute_ledger_sha256": clean["compute"]["ledger_sha256"],
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "validation_id": _canonical_sha256(body)}
    assert_metadata_only(result, where="acquisition Volume validation")
    return result


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def validate_volume(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate retained Volume inputs without starting preparation or GPU work."""

    volume.reload()
    return validate_volume_bindings(spec, volume_root=VOLUME_PATH)


def _acquisition_policy_from_toml(policy: Mapping[str, Any]) -> Any:
    """Adapt the sole TOML policy into the pure deterministic selector contract."""

    random_arm = policy["random"]
    active = policy["active"]
    bucket_rows = active["bucket_rows"]
    return _acquisition_module().AcquisitionPolicy(
        probability_rows=random_arm["rows"],
        rare_cell_rows=bucket_rows["rare_cell"],
        boundary_rows=bucket_rows["boundary"],
        multi_context_rows=bucket_rows["multi_context"],
        uncertainty_disagreement_rows=bucket_rows["uncertainty_disagreement"],
        probability_seed=random_arm["seed"],
        active_seed=active["seed"],
        expected_seed_count=3,
        policy_file_sha256=_file_sha256(_policy_path()),
    )


def _load_prepared_contract(
    run_id: str,
    *,
    volume_root: Path = VOLUME_PATH,
    validate_frame_content: bool = True,
) -> tuple[dict[str, Any], Path]:
    root = _prepared_root(run_id, volume_root=volume_root)
    _policy, policy_sha256 = load_frozen_policy()
    runtime_source_bundle = _runtime_source_bundle()
    receipt = _read_json(root / "receipt.json", where="acquisition preparation receipt")
    private_contract = _read_json(root / "contract.json", where="private preparation contract")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if (
        _canonical_sha256(private_contract) != run_id
        or receipt.get("prepared_run_id") != run_id
        or receipt.get("receipt_id") != _canonical_sha256(receipt_body)
        or "source_inventory" in receipt
        or any(
            receipt.get(key) != value
            for key, value in private_contract.items()
            if key != "source_inventory"
        )
        or receipt.get("kind") != PREPARED_KIND
        or receipt.get("policy_sha256") != policy_sha256
        or receipt.get("base_factorised_run_id") != BASE_FACTORISED_RUN_ID
        or receipt.get("runtime_source_bundle") != runtime_source_bundle
        or receipt.get("runtime_source_bundle_sha256") != runtime_source_bundle["source_bundle_id"]
        or receipt.get("analysis_language")
        != {"main_corpus_status": "provisional_english", "human_language_gate_accepted": False}
        or receipt.get("locked_test_rows_accessed") != 0
    ):
        raise RuntimeError("prepared acquisition contract drifted")
    descriptor = receipt.get("prepared_frame")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "relative_path",
        "sha256",
        "bytes",
        "row_count",
    }:
        raise RuntimeError("prepared acquisition frame descriptor drifted")
    path = root / str(descriptor["relative_path"])
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or (validate_frame_content and _file_sha256(path) != descriptor["sha256"])
    ):
        raise RuntimeError("prepared acquisition frame content drifted")
    return receipt, path


def _validate_prepared_spec_binding(prepared: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    if prepared.get("launch_spec_sha256") != _launch_spec_sha256(spec):
        raise RuntimeError("prepared acquisition launch-spec binding drifted")


def _render(row: Mapping[str, Any], *, mode: str, separator: str) -> str:
    if mode not in {"full", "target_only"}:
        raise ValueError("unknown acquisition render")
    rendered_row = dict(row)
    if mode == "target_only":
        rendered_row["parent_context"] = None
        rendered_row["submission_context"] = None
    return _data_module().render_factorised_text(rendered_row, separator=separator)


def _tokenise_render(tokenizer: Any, text: str) -> dict[str, list[int]]:
    """Apply the frozen factorised token truncation rule (retain final token)."""
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
    maximum = _training_module().MAX_LENGTH
    if len(input_ids) > maximum:
        input_ids = [*input_ids[: maximum - 1], input_ids[-1]]
        attention_mask = [*attention_mask[: maximum - 1], attention_mask[-1]]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _load_checkpoint_model(checkpoint: Mapping[str, Any]) -> Any:
    import torch

    training = _training_module()
    trial_spec = checkpoint.get("trial_spec")
    if not isinstance(trial_spec, Mapping):
        raise RuntimeError("checkpoint descriptor omits its exact trial specification")
    config = training.build_optimisation_config(trial_spec)
    model = training.create_component_model(component=checkpoint["component"], config=config)
    state = torch.load(
        VOLUME_PATH / checkpoint["relative_path"], map_location="cpu", weights_only=True
    )
    if not isinstance(state, Mapping) or set(state) != {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase_run_id",
        "trial_id",
        "component",
        "optimiser_seed",
        "config_sha256",
        "training_frame_sha256",
        "development_frame_sha256",
        "teacher_ledger_sha256",
        "source_bundle_sha256",
        "dependency_lock_sha256",
        "model_id",
        "model_revision",
        "selected_epoch",
        "model_state_dict",
    }:
        raise RuntimeError("checkpoint payload is not a state mapping")
    if (
        state["kind"] != "modernbert-factorised-checkpoint-v2"
        or state["experiment_run_id"] != BASE_FACTORISED_RUN_ID
        or state["component"] != checkpoint["component"]
        or state["optimiser_seed"] != checkpoint["optimiser_seed"]
        or state["trial_id"] != checkpoint["trial_id"]
        or state["config_sha256"] != trial_spec["config"]["config_sha256"]
        or state["development_frame_sha256"] != checkpoint["development_frame_sha256"]
        or not isinstance(state["model_state_dict"], Mapping)
    ):
        raise RuntimeError("checkpoint provenance or component binding drifted")
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to("cuda").eval()


def _max_abs_difference(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right):
        return max((_max_abs_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return max(
            (_max_abs_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0
        )
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise ValueError("parity logits are not numeric")
    return abs(float(left) - float(right))


def _collect_unlabelled_logits(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    component: str,
    render: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score an unlabelled frame with the exact training-time input mechanics."""

    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("acquisition scoring requires CUDA BF16 support")
    encoded: list[dict[str, Any]] = []
    for row in rows:
        item_id = row.get("record_id", row.get("item_id"))
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("unlabelled acquisition row lacks a private item ID")
        feature = _tokenise_render(
            tokenizer,
            _render(row, mode=render, separator=tokenizer.sep_token),
        )
        encoded.append({"item_id": item_id, **feature})
    build_batches = importlib.import_module(
        "reddit_china_stance.modernbert_trainer"
    ).build_length_bucket_batches
    batches = build_batches(
        [len(row["input_ids"]) for row in encoded],
        batch_size=batch_size,
        seed=0,
        epoch=0,
    )
    output_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for indices in batches:
            features = [encoded[index] for index in indices]
            batch = tokenizer.pad(
                [
                    {
                        "input_ids": row["input_ids"],
                        "attention_mask": row["attention_mask"],
                    }
                    for row in features
                ],
                padding=True,
                return_tensors="pt",
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )
            if component == "relevance":
                logits = output["relevance_logits"].detach().float().cpu().tolist()
                output_rows.extend(
                    {"item_id": row["item_id"], "relevance_logit": float(logit)}
                    for row, logit in zip(features, logits, strict=True)
                )
            else:
                targets = output["target_presence_logits"].detach().float().cpu().tolist()
                stances = output["stance_logits"].detach().float().cpu().tolist()
                output_rows.extend(
                    {
                        "item_id": row["item_id"],
                        "target_presence_logits": target,
                        "stance_logits": stance,
                    }
                    for row, target, stance in zip(features, targets, stances, strict=True)
                )
    if len(output_rows) != len(rows) or len({row["item_id"] for row in output_rows}) != len(rows):
        raise RuntimeError("unlabelled acquisition scoring did not conserve rows")
    return sorted(output_rows, key=lambda row: row["item_id"])


def _replay_checkpoint_parity(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the published 600-row full render and probe target-only determinism."""
    from torch.utils.data import DataLoader

    training = _training_module()
    frame_descriptor = {
        "relative_path": checkpoint["development_frame_relative_path"],
        "sha256": checkpoint["development_frame_sha256"],
        "bytes": (VOLUME_PATH / checkpoint["development_frame_relative_path"]).stat().st_size,
        "frame": "development",
    }
    # The manifest descriptor is authoritative for row count/thread-set fields;
    # load_private_frame only needs its integrity and frame identity fields.
    frame_path = VOLUME_PATH / checkpoint["development_frame_relative_path"]
    rows, labels = training.load_private_frame(
        frame_path, frame_descriptor, expected_frame="development"
    )
    predictions_path = VOLUME_PATH / checkpoint["development_predictions_relative_path"]
    if _file_sha256(predictions_path) != checkpoint["development_predictions_sha256"]:
        raise RuntimeError("stored development predictions drifted")
    expected = json.loads(predictions_path.read_text(encoding="utf-8"))
    if (
        not isinstance(expected, Mapping)
        or expected.get("experiment_run_id") != BASE_FACTORISED_RUN_ID
        or expected.get("trial_id") != checkpoint["trial_id"]
        or expected.get("component") != checkpoint["component"]
        or expected.get("optimiser_seed") != checkpoint["optimiser_seed"]
        or expected.get("development_frame_sha256") != checkpoint["development_frame_sha256"]
        or not isinstance(expected.get("rows"), list)
    ):
        raise RuntimeError("stored development prediction binding drifted")
    if len(rows) != 600 or len(expected["rows"]) != 600:
        raise RuntimeError("development parity frame does not conserve 600 rows")
    tokenizer = training.load_pinned_tokenizer()
    model = _load_checkpoint_model(checkpoint)
    optimisation = training.build_optimisation_config(checkpoint["trial_spec"])
    encoded = [
        training.tokenise_factorised_record(
            tokenizer, item_id=row["item_id"], row=row, label=labels[row["item_id"]]
        )
        for row in rows
    ]
    batches = importlib.import_module(
        "reddit_china_stance.modernbert_trainer"
    ).build_length_bucket_batches(
        [len(row["input_ids"]) for row in encoded],
        batch_size=optimisation.per_device_batch_size,
        seed=0,
        epoch=0,
    )
    loader = DataLoader(
        encoded,
        batch_sampler=batches,
        collate_fn=training.FactorisedDynamicPaddingCollator(
            tokenizer, component=checkpoint["component"]
        ),
        num_workers=0,
    )
    observed_rows = training.collect_development_logits(
        model,
        loader,
        component=checkpoint["component"],
        device="cuda",
        use_bf16=True,
    )
    observed = {row["item_id"]: row for row in observed_rows}
    probe = rows[:32]
    first_target_only = _collect_unlabelled_logits(
        model=model,
        tokenizer=tokenizer,
        rows=probe,
        component=checkpoint["component"],
        render="target_only",
        batch_size=optimisation.per_device_batch_size,
    )
    second_target_only = _collect_unlabelled_logits(
        model=model,
        tokenizer=tokenizer,
        rows=probe,
        component=checkpoint["component"],
        render="target_only",
        batch_size=optimisation.per_device_batch_size,
    )
    target_only_max_difference = _max_abs_difference(
        [
            {key: value for key, value in row.items() if key != "item_id"}
            for row in first_target_only
        ],
        [
            {key: value for key, value in row.items() if key != "item_id"}
            for row in second_target_only
        ],
    )
    expected_index = {row["item_id"]: row for row in expected["rows"]}
    max_difference = 0.0
    for item_id, actual_row in observed.items():
        target = expected_index[item_id]
        actual = (
            actual_row["relevance_logit"]
            if checkpoint["component"] == "relevance"
            else {
                "target_presence_logits": actual_row["target_presence_logits"],
                "stance_logits": actual_row["stance_logits"],
            }
        )
        expected_logits = (
            target["relevance_logit"]
            if checkpoint["component"] == "relevance"
            else {
                "target_presence_logits": target["target_presence_logits"],
                "stance_logits": target["stance_logits"],
            }
        )
        max_difference = max(max_difference, _max_abs_difference(actual, expected_logits))
    if (
        max_difference > PREFLIGHT_LOGIT_TOLERANCE
        or target_only_max_difference > PREFLIGHT_LOGIT_TOLERANCE
    ):
        raise RuntimeError("checkpoint CUDA parity exceeds frozen tolerance")
    return {
        "development_rows": 600,
        "full_max_abs_logit_difference": max_difference,
        "target_only_probe_rows": len(probe),
        "target_only_repeat_max_abs_logit_difference": target_only_max_difference,
    }


def _write_parquet_immutable(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if path.exists():
        existing = pq.ParquetFile(path).metadata.num_rows
        if existing != len(rows):
            raise RuntimeError("existing immutable checkpoint logits row count drifted")
        return _descriptor(path, root=path.parent, row_count=existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.{uuid.uuid4().hex}.incomplete"
    if any(path.parent.glob(f".{path.name}.*.incomplete")):
        raise FileExistsError("stale checkpoint-logit staging requires manual reconciliation")
    staging.mkdir(parents=True)
    temporary = staging / path.name
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
    os.replace(temporary, path)
    staging.rmdir()
    return _descriptor(path, root=path.parent, row_count=len(rows))


def _read_parquet_slice(
    path: Path,
    *,
    start: int,
    row_count: int,
    columns: Sequence[str] | None = None,
) -> Any:
    """Read one contiguous slice without materialising the full prepared frame."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if start < 0 or row_count <= 0:
        raise ValueError("Parquet slice bounds must be non-negative and non-empty")
    parquet = pq.ParquetFile(path)
    if start + row_count > parquet.metadata.num_rows:
        raise ValueError("Parquet slice exceeds the prepared frame")
    batches: list[Any] = []
    observed = 0
    stop = start + row_count
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        batch_stop = observed + batch.num_rows
        if batch_stop <= start:
            observed = batch_stop
            continue
        if observed >= stop:
            break
        local_start = max(0, start - observed)
        local_stop = min(batch.num_rows, stop - observed)
        batches.append(batch.slice(local_start, local_stop - local_start))
        observed = batch_stop
        if observed >= stop:
            break
    table = pa.Table.from_batches(batches)
    if table.num_rows != row_count:
        raise RuntimeError("Parquet slice did not conserve requested rows")
    return table


def _checkpoint_scoring_root(
    *, run_id: str, checkpoint: Mapping[str, Any], volume_root: Path
) -> Path:
    component = checkpoint.get("component")
    seed = checkpoint.get("optimiser_seed")
    if (
        not isinstance(component, str)
        or component not in {"relevance", "target_stance_b4"}
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed not in {47, 61, 89}
    ):
        raise ValueError("checkpoint scoring identity is outside the frozen six trials")
    return (
        _checkpoint_root(run_id, volume_root=volume_root)
        / f"component={component}"
        / f"seed={seed}"
    )


def _scoring_shard_plan(row_count: int) -> list[dict[str, int | str]]:
    """Partition the prepared frame into ten deterministic contiguous slices."""

    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < SCORING_SHARD_COUNT
    ):
        raise ValueError("scoring requires at least ten prepared rows")
    quotient, remainder = divmod(row_count, SCORING_SHARD_COUNT)
    plan: list[dict[str, int | str]] = []
    start = 0
    for index in range(SCORING_SHARD_COUNT):
        size = quotient + (1 if index < remainder else 0)
        stop = start + size
        plan.append(
            {
                "shard_id": f"{index:02d}",
                "start": start,
                "stop": stop,
                "row_count": size,
            }
        )
        start = stop
    if start != row_count or sum(int(shard["row_count"]) for shard in plan) != row_count:
        raise RuntimeError("scoring shard plan does not conserve prepared rows")
    return plan


def _scoring_plan_contract(row_count: int) -> dict[str, Any]:
    body = {
        "kind": "modernbert-acquisition-scoring-shard-plan-v1",
        "schema_version": "1.0.0",
        "row_count": row_count,
        "shard_count": SCORING_SHARD_COUNT,
        "renders": list(SCORING_RENDERS),
        "batch_size": SCORING_BATCH_SIZE,
        "shards": _scoring_shard_plan(row_count),
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "plan_id": _canonical_sha256(body)}
    assert_metadata_only(result, where="acquisition scoring shard plan")
    return result


def _scoring_shard_root(
    *,
    run_id: str,
    checkpoint: Mapping[str, Any],
    shard: Mapping[str, Any],
    volume_root: Path,
) -> Path:
    shard_id = shard.get("shard_id")
    if not isinstance(shard_id, str) or len(shard_id) != 2 or not shard_id.isdigit():
        raise ValueError("scoring shard ID is invalid")
    return (
        _checkpoint_scoring_root(
            run_id=run_id,
            checkpoint=checkpoint,
            volume_root=volume_root,
        )
        / "shards"
        / f"shard={shard_id}"
    )


def _validate_scoring_job_approval(
    job: Mapping[str, Any],
    *,
    prepared: Mapping[str, Any],
    volume_root: Path = VOLUME_PATH,
) -> dict[str, Any]:
    """Validate persisted coordinator claims before any checkpoint GPU work."""

    if not isinstance(job, Mapping) or set(job) != SCORING_JOB_KEYS:
        raise ValueError("checkpoint scoring job approval schema drifted")
    run_id = _sha256(job.get("prepared_run_id"), where="prepared run ID")
    checkpoint = job.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint scoring job lacks checkpoint descriptor")
    checkpoint_sha256 = _sha256(checkpoint.get("sha256"), where="checkpoint SHA-256")
    checkpoint_descriptor_sha256 = _canonical_sha256(checkpoint)
    checkpoint_bundle_sha256 = _sha256(
        job.get("checkpoint_bundle_sha256"), where="checkpoint bundle SHA-256"
    )
    phase_claim_id = _sha256(job.get("phase_approval_claim_id"), where="phase approval claim ID")
    checkpoint_claim_id = _sha256(
        job.get("checkpoint_launch_claim_id"), where="checkpoint launch claim ID"
    )
    runtime_sha256 = _sha256(
        job.get("runtime_source_bundle_sha256"),
        where="scoring job runtime source bundle SHA-256",
    )
    if runtime_sha256 != prepared.get("runtime_source_bundle_sha256"):
        raise RuntimeError("checkpoint scoring runtime binding drifted")
    try:
        estimated = Decimal(str(job.get("estimated_upper_cost_usd")))
        approved = Decimal(str(job.get("approved_cost_usd")))
    except InvalidOperation as error:
        raise ValueError("checkpoint scoring approval amounts are invalid") from error
    enforce_cost_guardrail(estimated_cost_usd=estimated, approved_cost_usd=approved)
    if estimated != approved:
        raise ValueError("checkpoint scoring approval must exactly match its estimate")

    phase_path = _checkpoint_root(run_id, volume_root=volume_root) / "phase-approval.json"
    phase = _read_json(phase_path, where="acquisition scoring phase approval")
    phase_body = {key: value for key, value in phase.items() if key != "claim_id"}
    try:
        cumulative = Decimal(str(phase.get("cumulative_measured_spend_usd")))
        reservation = Decimal(str(phase.get("active_reservation_usd")))
    except InvalidOperation as error:
        raise ValueError("scoring phase spend binding is invalid") from error
    if (
        set(phase) != SCORING_PHASE_APPROVAL_KEYS
        or phase.get("claim_id") != _canonical_sha256(phase_body)
        or phase.get("claim_id") != phase_claim_id
        or phase.get("kind") != "modernbert-acquisition-phase-approval-v1"
        or phase.get("schema_version") != "1.0.0"
        or phase.get("action") != "launch-scoring"
        or phase.get("prepared_run_id") != run_id
        or phase.get("checkpoint_bundle_sha256") != checkpoint_bundle_sha256
        or phase.get("runtime_source_bundle_sha256") != runtime_sha256
        or phase.get("estimated_upper_cost_usd") != format(estimated, "f")
        or phase.get("approved_cost_usd") != format(approved, "f")
        or not cumulative.is_finite()
        or cumulative < 0
        or not reservation.is_finite()
        or reservation < 0
        or phase.get("locked_test_rows_accessed") != 0
    ):
        raise RuntimeError("scoring phase approval claim binding drifted")
    _sha256(phase.get("compute_ledger_sha256"), where="compute ledger SHA-256")

    root = _checkpoint_scoring_root(run_id=run_id, checkpoint=checkpoint, volume_root=volume_root)
    launch = _read_json(root / "launch-claim.json", where="checkpoint launch claim")
    launch_body = {key: value for key, value in launch.items() if key != "claim_id"}
    if (
        set(launch) != SCORING_CHECKPOINT_CLAIM_KEYS
        or launch.get("claim_id") != _canonical_sha256(launch_body)
        or launch.get("claim_id") != checkpoint_claim_id
        or launch.get("kind") != "modernbert-acquisition-checkpoint-launch-claim-v1"
        or launch.get("schema_version") != "1.0.0"
        or launch.get("action") != "score-checkpoint"
        or launch.get("prepared_run_id") != run_id
        or launch.get("component") != checkpoint.get("component")
        or launch.get("optimiser_seed") != checkpoint.get("optimiser_seed")
        or launch.get("checkpoint_sha256") != checkpoint_sha256
        or launch.get("checkpoint_descriptor_sha256") != checkpoint_descriptor_sha256
        or launch.get("checkpoint_bundle_sha256") != checkpoint_bundle_sha256
        or launch.get("phase_approval_claim_id") != phase_claim_id
        or launch.get("estimated_upper_cost_usd") != format(estimated, "f")
        or launch.get("approved_cost_usd") != format(approved, "f")
        or launch.get("runtime_source_bundle_sha256") != runtime_sha256
        or launch.get("status") != "claimed_before_spawn"
        or launch.get("locked_test_rows_accessed") != 0
    ):
        raise RuntimeError("checkpoint launch claim binding drifted")
    return {
        "phase_approval_claim_id": phase_claim_id,
        "checkpoint_launch_claim_id": checkpoint_claim_id,
        "approved_cost_usd": format(approved, "f"),
        "estimated_upper_cost_usd": format(estimated, "f"),
        "runtime_source_bundle_sha256": runtime_sha256,
    }


def _scoring_shard_job(
    *,
    prepared_run_id: str,
    checkpoint: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    phase_claim: Mapping[str, Any],
    checkpoint_claim: Mapping[str, Any],
    scoring_plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    attempt_id: str,
    dispatch_id: str,
) -> dict[str, Any]:
    return {
        "prepared_run_id": prepared_run_id,
        "checkpoint": dict(checkpoint),
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "phase_approval_claim_id": phase_claim["claim_id"],
        "checkpoint_launch_claim_id": checkpoint_claim["claim_id"],
        "estimated_upper_cost_usd": phase_claim["estimated_upper_cost_usd"],
        "approved_cost_usd": phase_claim["approved_cost_usd"],
        "runtime_source_bundle_sha256": phase_claim["runtime_source_bundle_sha256"],
        "scoring_plan_sha256": scoring_plan["plan_id"],
        "scoring_shard": dict(shard),
        "batch_size": SCORING_BATCH_SIZE,
        "attempt_id": attempt_id,
        "dispatch_id": dispatch_id,
    }


def _validate_scoring_shard_job(
    job: Mapping[str, Any],
    *,
    prepared: Mapping[str, Any],
    volume_root: Path = VOLUME_PATH,
) -> dict[str, Any]:
    """Bind a GPU unit to one exact prepared-frame slice and one render."""

    if not isinstance(job, Mapping) or set(job) != SCORING_SHARD_JOB_KEYS:
        raise ValueError("checkpoint scoring shard job schema drifted")
    base_job = {key: job[key] for key in SCORING_JOB_KEYS}
    approval = _validate_scoring_job_approval(
        base_job,
        prepared=prepared,
        volume_root=volume_root,
    )
    descriptor = prepared.get("prepared_frame")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError("prepared scoring frame descriptor is missing")
    row_count = descriptor.get("row_count")
    scoring_plan = _scoring_plan_contract(row_count)
    shard = job.get("scoring_shard")
    if (
        job.get("batch_size") != SCORING_BATCH_SIZE
        or job.get("scoring_plan_sha256") != scoring_plan["plan_id"]
        or not isinstance(shard, Mapping)
        or dict(shard) not in scoring_plan["shards"]
        or not isinstance(job.get("attempt_id"), str)
        or not job["attempt_id"]
        or not isinstance(job.get("dispatch_id"), str)
        or not job["dispatch_id"]
    ):
        raise RuntimeError("checkpoint scoring shard plan binding drifted")
    return {
        **approval,
        "scoring_plan_sha256": scoring_plan["plan_id"],
        "scoring_shard": dict(shard),
        "batch_size": SCORING_BATCH_SIZE,
        "attempt_id": job["attempt_id"],
        "dispatch_id": job["dispatch_id"],
    }


def _validate_scoring_shard_receipt(
    job: Mapping[str, Any],
    *,
    prepared: Mapping[str, Any],
    volume_root: Path = VOLUME_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one immutable private shard and its metadata-only receipt."""

    import pyarrow.parquet as pq

    approval = _validate_scoring_shard_job(
        job,
        prepared=prepared,
        volume_root=volume_root,
    )
    run_id = _sha256(job.get("prepared_run_id"), where="prepared run ID")
    checkpoint = job["checkpoint"]
    shard = job["scoring_shard"]
    root = _scoring_shard_root(
        run_id=run_id,
        checkpoint=checkpoint,
        shard=shard,
        volume_root=volume_root,
    )
    receipt = _read_json(root / "receipt.json", where="checkpoint scoring shard receipt")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    expected_keys = {
        "kind",
        "schema_version",
        "prepared_run_id",
        "prepared_frame_sha256",
        "checkpoint",
        "checkpoint_bundle_sha256",
        "checkpoint_sha256",
        "phase_approval_claim_id",
        "checkpoint_launch_claim_id",
        "approved_cost_usd",
        "estimated_upper_cost_usd",
        "runtime_source_bundle_sha256",
        "scoring_plan_sha256",
        "scoring_shard",
        "batch_size",
        "attempt_id",
        "dispatch_id",
        "text_renderer",
        "token_truncation",
        "padding",
        "precision",
        "row_count",
        "render_count",
        "logit_rows",
        "logits",
        "wall_seconds",
        "rows_per_worker_second",
        "locked_test_rows_accessed",
        "receipt_id",
    }
    wall_seconds = receipt.get("wall_seconds")
    rows_per_second = receipt.get("rows_per_worker_second")
    if (
        set(receipt) != expected_keys
        or receipt.get("receipt_id") != _canonical_sha256(receipt_body)
        or receipt.get("kind") != SCORING_SHARD_KIND
        or receipt.get("schema_version") != "1.0.0"
        or receipt.get("prepared_run_id") != run_id
        or receipt.get("prepared_frame_sha256") != prepared["prepared_frame"]["sha256"]
        or receipt.get("checkpoint") != checkpoint
        or receipt.get("checkpoint_bundle_sha256") != job["checkpoint_bundle_sha256"]
        or receipt.get("checkpoint_sha256") != checkpoint["sha256"]
        or any(receipt.get(key) != approval[key] for key in approval)
        or receipt.get("row_count") != shard["row_count"]
        or receipt.get("render_count") != len(SCORING_RENDERS)
        or receipt.get("logit_rows") != shard["row_count"] * len(SCORING_RENDERS)
        or not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) < 0
        or not isinstance(rows_per_second, (int, float))
        or isinstance(rows_per_second, bool)
        or not math.isfinite(float(rows_per_second))
        or float(rows_per_second) < 0
        or receipt.get("locked_test_rows_accessed") != 0
    ):
        raise RuntimeError("checkpoint scoring shard receipt binding drifted")
    descriptor = receipt.get("logits")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "relative_path",
        "sha256",
        "bytes",
        "row_count",
    }:
        raise RuntimeError("checkpoint scoring shard lacks private logits descriptor")
    path = root / _safe_relative(descriptor["relative_path"], where="scoring shard logits path")
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != descriptor["sha256"]
        or descriptor["row_count"] != shard["row_count"] * len(SCORING_RENDERS)
    ):
        raise RuntimeError("checkpoint scoring shard logits drifted")
    rows = pq.read_table(path).to_pylist()
    observed = {(row.get("opaque_id"), row.get("render")) for row in rows}
    if (
        len(rows) != shard["row_count"] * len(SCORING_RENDERS)
        or len(observed) != shard["row_count"] * len(SCORING_RENDERS)
        or {render for _opaque, render in observed} != set(SCORING_RENDERS)
        or any(not isinstance(opaque, str) or not opaque for opaque, _render in observed)
    ):
        raise RuntimeError("checkpoint scoring shard rows do not conserve identities")
    return receipt, rows


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    timeout=INFERENCE_TIMEOUT_SECONDS,
    max_containers=SCORING_MAX_CONTAINERS,
    volumes={str(VOLUME_PATH): volume},
)
def score_checkpoint(job: dict[str, Any]) -> dict[str, Any]:
    """Score one checkpoint/data shard on both renders with batch size 32."""

    if not isinstance(job, Mapping) or set(job) != SCORING_SHARD_JOB_KEYS:
        raise ValueError("checkpoint scoring shard job schema drifted")
    lease = _claim_dispatch_lease(job, stage="score-checkpoint")
    if lease is None:
        return _dispatch_not_acquired(job)
    import pyarrow.parquet as pq
    import torch

    volume.reload()
    run_id = _sha256(job.get("prepared_run_id"), where="prepared run ID")
    prepared, _frame_path = _load_prepared_contract(run_id, validate_frame_content=False)
    checkpoint = job.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint scoring shard job lacks checkpoint descriptor")
    approval = _validate_scoring_shard_job(job, prepared=prepared)
    shard = approval["scoring_shard"]
    root = _scoring_shard_root(
        run_id=run_id,
        checkpoint=checkpoint,
        shard=shard,
        volume_root=VOLUME_PATH,
    )
    if (root / "receipt.json").is_file():
        receipt, _rows = _validate_scoring_shard_receipt(job, prepared=prepared)
        return receipt
    attempt = root / "attempts" / f"attempt={job['attempt_id']}.incomplete"
    if attempt.exists():
        raise FileExistsError("scoring shard attempt already exists")
    attempt.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("acquisition scoring requires L4 CUDA")
    relative = _safe_relative(checkpoint.get("relative_path"), where="checkpoint path")
    source = VOLUME_PATH / relative
    if not source.is_file() or _file_sha256(source) != _sha256(
        checkpoint.get("sha256"), where="checkpoint SHA-256"
    ):
        raise RuntimeError("frozen B4 checkpoint drifted")
    prepared_shard_path = _load_prepared_frame_shard(
        run_id=run_id,
        prepared=prepared,
        shard=shard,
        volume_root=VOLUME_PATH,
    )
    rows = pq.read_table(prepared_shard_path).to_pylist()
    if len(rows) != shard["row_count"]:
        raise RuntimeError("prepared acquisition scoring slice drifted")
    training = _training_module()
    tokenizer = training.load_pinned_tokenizer()
    model = _load_checkpoint_model(checkpoint)
    started = time.monotonic()
    output_rows: list[dict[str, Any]] = []
    for render in SCORING_RENDERS:
        scored = _collect_unlabelled_logits(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            component=checkpoint["component"],
            render=render,
            batch_size=SCORING_BATCH_SIZE,
        )
        output_rows.extend(
            {
                "opaque_id": row["item_id"],
                "render": render,
                **{key: value for key, value in row.items() if key != "item_id"},
            }
            for row in scored
        )
    expected_logits = shard["row_count"] * len(SCORING_RENDERS)
    if len(output_rows) != expected_logits or len(
        {(row["opaque_id"], row["render"]) for row in output_rows}
    ) != expected_logits:
        raise RuntimeError("checkpoint scoring shard did not conserve render pairs")
    attempt_descriptor = _write_parquet_immutable(
        attempt / "logits.parquet",
        sorted(output_rows, key=lambda row: row["opaque_id"]),
    )
    descriptor = {
        **attempt_descriptor,
        "relative_path": (attempt / "logits.parquet").relative_to(root).as_posix(),
    }
    wall_seconds = max(0.001, round(time.monotonic() - started, 3))
    body = {
        "kind": SCORING_SHARD_KIND,
        "schema_version": "1.0.0",
        "prepared_run_id": run_id,
        "prepared_frame_sha256": prepared["prepared_frame"]["sha256"],
        "checkpoint": dict(checkpoint),
        "checkpoint_bundle_sha256": job["checkpoint_bundle_sha256"],
        "checkpoint_sha256": _file_sha256(source),
        **approval,
        "text_renderer": "factorised-target-parent-submission-v2",
        "token_truncation": "manual-retain-final-token-max-768",
        "padding": "dynamic-per-length-bucket-batch",
        "precision": "bf16-autocast",
        "row_count": len(rows),
        "render_count": len(SCORING_RENDERS),
        "logit_rows": len(output_rows),
        "logits": descriptor,
        "wall_seconds": wall_seconds,
        "rows_per_worker_second": len(rows) / wall_seconds,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(receipt, where="acquisition checkpoint shard receipt")
    _write_immutable_json(attempt / "receipt.json", receipt)
    _write_immutable_json(root / "receipt.json", receipt)
    volume.commit()
    return receipt


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32_768,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def cuda_preflight(spec: dict[str, Any], approved_cost_usd: str) -> dict[str, Any]:
    """Validate actual tokenizer/model/checkpoint loading before six scored calls."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("acquisition preflight requires L4 CUDA")
    spec = validate_input_spec(spec, action="cuda-preflight")
    _validate_phase_approval(spec, action="cuda-preflight", approved_cost_usd=approved_cost_usd)
    _policy, policy_sha256 = load_frozen_policy()
    if spec.get("policy_sha256") != policy_sha256:
        raise RuntimeError("acquisition preflight policy digest drifted")
    prepared_run_id = _sha256(spec.get("prepared_run_id"), where="prepared run ID")
    prepared, _frame_path = _load_prepared_contract(prepared_run_id)
    _validate_prepared_spec_binding(prepared, spec)
    checkpoints = _checkpoint_specs(spec)
    parity: dict[str, Any] = {}
    for checkpoint in checkpoints:
        key = f"{checkpoint['component']}:{checkpoint['optimiser_seed']}"
        parity[key] = _replay_checkpoint_parity(checkpoint)
        torch.cuda.empty_cache()
    body = {
        "kind": PREFLIGHT_KIND,
        "schema_version": "1.0.0",
        "prepared_run_id": prepared_run_id,
        "policy_sha256": policy_sha256,
        "checkpoint_bundle_sha256": _canonical_sha256(checkpoints),
        "runtime_source_bundle_sha256": prepared["runtime_source_bundle_sha256"],
        "gpu_type": "L4",
        "parity": parity,
        "logit_tolerance": PREFLIGHT_LOGIT_TOLERANCE,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(receipt, where="acquisition preflight receipt")
    _write_immutable_json(_checkpoint_root(prepared_run_id) / "cuda-preflight.json", receipt)
    volume.commit()
    return receipt


def validate_cuda_preflight(
    spec: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    spec = validate_input_spec(spec, action="launch-scoring")
    run_id = _sha256(spec.get("prepared_run_id"), where="prepared run ID")
    _policy, policy_sha256 = load_frozen_policy()
    if spec.get("policy_sha256") != policy_sha256:
        raise RuntimeError("acquisition validation policy digest drifted")
    receipt = _read_json(
        _checkpoint_root(run_id, volume_root=volume_root) / "cuda-preflight.json",
        where="acquisition CUDA preflight",
    )
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    expected_keys = {
        "kind",
        "schema_version",
        "prepared_run_id",
        "policy_sha256",
        "checkpoint_bundle_sha256",
        "runtime_source_bundle_sha256",
        "gpu_type",
        "parity",
        "logit_tolerance",
        "locked_test_rows_accessed",
        "receipt_id",
    }
    if receipt.get("policy_sha256") != policy_sha256:
        raise RuntimeError("acquisition CUDA preflight binding drifted")
    expected_bundle = _canonical_sha256(_checkpoint_specs(spec, volume_root=volume_root))
    prepared, _frame_path = _load_prepared_contract(run_id, volume_root=volume_root)
    _validate_prepared_spec_binding(prepared, spec)
    if (
        set(receipt) != expected_keys
        or receipt.get("receipt_id") != _canonical_sha256(receipt_body)
        or receipt.get("kind") != PREFLIGHT_KIND
        or receipt.get("schema_version") != "1.0.0"
        or receipt.get("prepared_run_id") != run_id
        or receipt.get("checkpoint_bundle_sha256") != expected_bundle
        or receipt.get("runtime_source_bundle_sha256") != prepared["runtime_source_bundle_sha256"]
        or receipt.get("gpu_type") != "L4"
        or not isinstance(receipt.get("parity"), Mapping)
        or len(receipt["parity"]) != 6
        or receipt.get("logit_tolerance") != PREFLIGHT_LOGIT_TOLERANCE
        or receipt.get("locked_test_rows_accessed") != 0
    ):
        raise RuntimeError("acquisition CUDA preflight binding drifted")
    return receipt


def _scoring_phase_approval_claim(
    *,
    spec: Mapping[str, Any],
    prepared: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    estimated_cost_usd: Decimal,
    approved_cost_usd: Decimal,
) -> dict[str, Any]:
    body = {
        "kind": "modernbert-acquisition-phase-approval-v1",
        "schema_version": "1.0.0",
        "action": "launch-scoring",
        "prepared_run_id": _sha256(spec.get("prepared_run_id"), where="prepared run ID"),
        "estimated_upper_cost_usd": format(estimated_cost_usd, "f"),
        "approved_cost_usd": format(approved_cost_usd, "f"),
        "cumulative_measured_spend_usd": spec["compute"]["cumulative_measured_spend_usd"],
        "active_reservation_usd": spec["compute"]["active_reservation_usd"],
        "compute_ledger_sha256": spec["compute"]["ledger_sha256"],
        "runtime_source_bundle_sha256": _sha256(
            prepared.get("runtime_source_bundle_sha256"),
            where="prepared runtime source bundle SHA-256",
        ),
        "checkpoint_bundle_sha256": _sha256(
            checkpoint_bundle_sha256, where="checkpoint bundle SHA-256"
        ),
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "claim_id": _canonical_sha256(body)}
    assert_metadata_only(result, where="acquisition scoring phase approval")
    return result


def _checkpoint_launch_claim(
    *,
    prepared_run_id: str,
    checkpoint: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    phase_claim: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "kind": "modernbert-acquisition-checkpoint-launch-claim-v1",
        "schema_version": "1.0.0",
        "action": "score-checkpoint",
        "prepared_run_id": prepared_run_id,
        "component": checkpoint["component"],
        "optimiser_seed": checkpoint["optimiser_seed"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_descriptor_sha256": _canonical_sha256(checkpoint),
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "phase_approval_claim_id": phase_claim["claim_id"],
        "estimated_upper_cost_usd": phase_claim["estimated_upper_cost_usd"],
        "approved_cost_usd": phase_claim["approved_cost_usd"],
        "runtime_source_bundle_sha256": phase_claim["runtime_source_bundle_sha256"],
        "status": "claimed_before_spawn",
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "claim_id": _canonical_sha256(body)}
    assert_metadata_only(result, where="acquisition checkpoint launch claim")
    return result


def _function_call_state(function_call_id: Any) -> str:
    """Poll one persisted Modal call without treating active work as failed."""

    if not isinstance(function_call_id, str) or not function_call_id:
        raise ValueError("persisted function call ID is invalid")
    try:
        modal.FunctionCall.from_id(function_call_id).get(timeout=0)
    except modal.exception.TimeoutError as error:
        if type(error) is modal.exception.TimeoutError:
            return "active"
        if isinstance(
            error,
            (modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError),
        ):
            return "failed"
        raise
    except (
        modal.exception.ExecutionError,
        modal.exception.RemoteError,
        modal.exception.UserCodeException,
    ):
        return "failed"
    return "completed"


_DISPATCH_INTENT_KEYS = {
    "attempt_id",
    "dispatch_id",
    "generation",
    "job_sha256",
    "stage",
    "status",
}
_DISPATCH_CALL_KEYS = {"dispatch_id", "function_call_id", "job_sha256", "status"}
_DISPATCH_LEASE_KEYS = {
    "attempt_id",
    "dispatch_id",
    "function_call_id",
    "job_sha256",
    "stage",
}


def _dispatch_lease_key(*, stage: str, dispatch_id: str) -> str:
    return f"{stage}:{dispatch_id}"


def _claim_dispatch_lease(
    job: Mapping[str, Any], *, stage: str
) -> dict[str, Any] | None:
    """Atomically fence duplicate calls before any expensive worker work."""

    attempt_id = job.get("attempt_id")
    dispatch_id = job.get("dispatch_id")
    function_call_id = modal.current_function_call_id()
    if not all(
        isinstance(value, str) and value
        for value in (stage, attempt_id, dispatch_id, function_call_id)
    ):
        raise RuntimeError("dispatch lease binding is incomplete")
    claim = {
        "attempt_id": attempt_id,
        "dispatch_id": dispatch_id,
        "function_call_id": function_call_id,
        "job_sha256": _canonical_sha256(job),
        "stage": stage,
    }
    key = _dispatch_lease_key(stage=stage, dispatch_id=dispatch_id)
    if dispatch_leases.put(key, claim, skip_if_exists=True):
        return claim
    existing = dispatch_leases.get(key)
    if existing == claim:
        return claim
    if not isinstance(existing, Mapping) or set(existing) != _DISPATCH_LEASE_KEYS:
        raise RuntimeError("dispatch lease state drifted")
    return None


def _dispatch_not_acquired(claim: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "status": "dispatch_lease_not_acquired",
        "dispatch_id": claim.get("dispatch_id"),
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="dispatch lease rejection")
    return result


def _dispatch_intent(
    *, stage: str, generation: int, attempt_id: str, dispatch_id: str, job: Mapping[str, Any]
) -> dict[str, Any]:
    if generation < 0 or not all(
        isinstance(value, str) and value for value in (stage, attempt_id, dispatch_id)
    ):
        raise ValueError("dispatch intent binding is invalid")
    intent = {
        "attempt_id": attempt_id,
        "dispatch_id": dispatch_id,
        "generation": generation,
        "job_sha256": _canonical_sha256(job),
        "stage": stage,
        "status": "intent_committed_before_spawn",
    }
    assert_metadata_only(intent, where="dispatch intent")
    return intent


def _dispatch_directory(dispatch_root: Path, dispatch_id: str) -> Path:
    return dispatch_root / f"dispatch={_sha256(dispatch_id, where='dispatch ID')}"


def _record_dispatch_call(
    *, dispatch_root: Path, intent: Mapping[str, Any], function_call_id: str
) -> None:
    if not isinstance(function_call_id, str) or not function_call_id:
        raise RuntimeError("spawned function call ID is invalid")
    receipt = {
        "dispatch_id": intent["dispatch_id"],
        "function_call_id": function_call_id,
        "job_sha256": intent["job_sha256"],
        "status": "spawned",
    }
    path = (
        _dispatch_directory(dispatch_root, intent["dispatch_id"])
        / "calls"
        / f"call={_canonical_sha256(receipt)}.json"
    )
    _write_immutable_json(path, receipt)


def _spawn_committed_dispatch(
    *,
    dispatch_root: Path,
    intent: Mapping[str, Any],
    job: Mapping[str, Any],
    function: Any,
) -> str:
    """Commit an immutable intent before spawning, then record the returned call ID."""

    if (
        set(intent) != _DISPATCH_INTENT_KEYS
        or intent.get("job_sha256") != _canonical_sha256(job)
        or intent.get("dispatch_id") != job.get("dispatch_id")
        or intent.get("attempt_id") != job.get("attempt_id")
    ):
        raise RuntimeError("dispatch intent/job binding drifted")
    directory = _dispatch_directory(dispatch_root, intent["dispatch_id"])
    _write_immutable_json(directory / "intent.json", intent)
    volume.commit()
    call = function.spawn(dict(job))
    _record_dispatch_call(
        dispatch_root=dispatch_root,
        intent=intent,
        function_call_id=call.object_id,
    )
    volume.commit()
    return call.object_id


def _load_dispatch_intents(*, dispatch_root: Path, stage: str) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []
    for path in sorted(dispatch_root.glob("dispatch=*/intent.json")):
        intent = _read_json(path, where=f"{stage} dispatch intent")
        if (
            set(intent) != _DISPATCH_INTENT_KEYS
            or intent.get("stage") != stage
            or intent.get("status") != "intent_committed_before_spawn"
            or not isinstance(intent.get("generation"), int)
            or intent["generation"] < 0
            or not all(
                isinstance(intent.get(key), str) and intent[key]
                for key in ("attempt_id", "dispatch_id", "job_sha256")
            )
            or path
            != _dispatch_directory(dispatch_root, intent["dispatch_id"])
            / "intent.json"
        ):
            raise RuntimeError(f"{stage} dispatch intent drifted")
        intents.append(intent)
    generations = sorted(intent["generation"] for intent in intents)
    if generations != list(range(len(intents))):
        raise RuntimeError(f"{stage} dispatch generations are not contiguous")
    return sorted(intents, key=lambda intent: intent["generation"])


def _load_dispatch_calls(
    *, dispatch_root: Path, intent: Mapping[str, Any]
) -> list[dict[str, Any]]:
    calls = [
        _read_json(path, where="dispatch call receipt")
        for path in sorted(
            (_dispatch_directory(dispatch_root, intent["dispatch_id"]) / "calls").glob(
                "call=*.json"
            )
        )
    ]
    if any(
        set(call) != _DISPATCH_CALL_KEYS
        or call.get("dispatch_id") != intent.get("dispatch_id")
        or call.get("job_sha256") != intent.get("job_sha256")
        or call.get("status") != "spawned"
        or not isinstance(call.get("function_call_id"), str)
        or not call["function_call_id"]
        for call in calls
    ):
        raise RuntimeError("dispatch call receipt drifted")
    return calls


def _dispatch_attempt_state(
    *, dispatch_root: Path, intent: Mapping[str, Any]
) -> str:
    """Resolve the lease owner, or conservatively reconcile the same intent."""

    key = _dispatch_lease_key(stage=intent["stage"], dispatch_id=intent["dispatch_id"])
    lease = dispatch_leases.get(key)
    if lease is not None:
        if (
            not isinstance(lease, Mapping)
            or set(lease) != _DISPATCH_LEASE_KEYS
            or lease.get("attempt_id") != intent.get("attempt_id")
            or lease.get("dispatch_id") != intent.get("dispatch_id")
            or lease.get("job_sha256") != intent.get("job_sha256")
            or lease.get("stage") != intent.get("stage")
            or not isinstance(lease.get("function_call_id"), str)
            or not lease["function_call_id"]
        ):
            raise RuntimeError("dispatch lease binding drifted")
        return _function_call_state(lease["function_call_id"])
    calls = _load_dispatch_calls(dispatch_root=dispatch_root, intent=intent)
    call_states = _persisted_dispatch_call_states(calls)
    return "active" if "active" in call_states else "reconcile"


def _persisted_dispatch_call_states(
    dispatches: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Return exact terminal/active states for every immutable dispatch attempt."""

    states: set[str] = set()
    for dispatch in dispatches:
        function_call_id = dispatch.get("function_call_id")
        if not isinstance(function_call_id, str) or not function_call_id:
            raise RuntimeError("persisted dispatch function call ID is invalid")
        states.add(_function_call_state(function_call_id))
    return states


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def coordinate_scoring(spec: dict[str, Any], approved_cost_usd: str) -> dict[str, Any]:
    volume.reload()
    spec = validate_input_spec(spec, action="launch-scoring")
    estimate, approved = _validate_phase_approval(
        spec, action="launch-scoring", approved_cost_usd=approved_cost_usd
    )
    validate_cuda_preflight(spec)
    checkpoints = _checkpoint_specs(spec)
    checkpoint_bundle_sha256 = _canonical_sha256(checkpoints)
    prepared_run_id = _sha256(spec.get("prepared_run_id"), where="prepared run ID")
    prepared, _frame_path = _load_prepared_contract(prepared_run_id)
    _validate_prepared_spec_binding(prepared, spec)
    phase_claim = _scoring_phase_approval_claim(
        spec=spec,
        prepared=prepared,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        estimated_cost_usd=estimate,
        approved_cost_usd=approved,
    )
    scoring_root = _checkpoint_root(prepared_run_id)
    _write_immutable_json(scoring_root / "phase-approval.json", phase_claim)
    scoring_plan = _scoring_plan_contract(prepared["prepared_frame"]["row_count"])
    _write_immutable_json(scoring_root / "scoring-plan.json", scoring_plan)
    volume.commit()
    calls: list[str] = []
    completed_shards = 0
    reused_shards = 0
    dispatched_shards = 0
    completed_checkpoints = 0
    for checkpoint in checkpoints:
        root = _checkpoint_scoring_root(
            run_id=prepared_run_id,
            checkpoint=checkpoint,
            volume_root=VOLUME_PATH,
        )
        if (root / "receipt.json").exists():
            completed_checkpoints += 1
        claim = root / "launch-claim.json"
        payload = _checkpoint_launch_claim(
            prepared_run_id=prepared_run_id,
            checkpoint=checkpoint,
            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
            phase_claim=phase_claim,
        )
        _write_immutable_json(claim, payload)
        volume.commit()
        for shard in scoring_plan["shards"]:
            shard_root = _scoring_shard_root(
                run_id=prepared_run_id,
                checkpoint=checkpoint,
                shard=shard,
                volume_root=VOLUME_PATH,
            )
            if (shard_root / "receipt.json").is_file():
                persisted = _read_json(
                    shard_root / "receipt.json", where="checkpoint scoring shard receipt"
                )
                job = _scoring_shard_job(
                    prepared_run_id=prepared_run_id,
                    checkpoint=checkpoint,
                    checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                    phase_claim=phase_claim,
                    checkpoint_claim=payload,
                    scoring_plan=scoring_plan,
                    shard=shard,
                    attempt_id=str(persisted.get("attempt_id")),
                    dispatch_id=str(persisted.get("dispatch_id")),
                )
                _validate_scoring_shard_receipt(job, prepared=prepared)
                completed_shards += 1
                reused_shards += 1
                continue
            dispatch_root = root / "dispatches" / f"shard={shard['shard_id']}"
            intents = _load_dispatch_intents(
                dispatch_root=dispatch_root, stage="score-checkpoint"
            )
            intent = intents[-1] if intents else None
            job: dict[str, Any] | None = None
            state = "new"
            if intent is not None:
                job = _scoring_shard_job(
                    prepared_run_id=prepared_run_id,
                    checkpoint=checkpoint,
                    checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                    phase_claim=phase_claim,
                    checkpoint_claim=payload,
                    scoring_plan=scoring_plan,
                    shard=shard,
                    attempt_id=intent["attempt_id"],
                    dispatch_id=intent["dispatch_id"],
                )
                if intent["job_sha256"] != _canonical_sha256(job):
                    raise RuntimeError("checkpoint scoring dispatch job binding drifted")
                state = _dispatch_attempt_state(
                    dispatch_root=dispatch_root, intent=intent
                )
                if state in {"completed", "failed"}:
                    volume.reload()
                    if (shard_root / "receipt.json").is_file():
                        persisted = _read_json(
                            shard_root / "receipt.json",
                            where="checkpoint scoring shard receipt",
                        )
                        job = _scoring_shard_job(
                            prepared_run_id=prepared_run_id,
                            checkpoint=checkpoint,
                            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                            phase_claim=phase_claim,
                            checkpoint_claim=payload,
                            scoring_plan=scoring_plan,
                            shard=shard,
                            attempt_id=str(persisted.get("attempt_id")),
                            dispatch_id=str(persisted.get("dispatch_id")),
                        )
                        _validate_scoring_shard_receipt(job, prepared=prepared)
                        completed_shards += 1
                        reused_shards += 1
                        continue
                    if state == "completed":
                        raise RuntimeError(
                            "completed checkpoint scoring call lacks its final shard receipt"
                        )
                if state == "active":
                    dispatched_shards += 1
                    continue
            if state in {"new", "failed"}:
                generation = len(intents)
                attempt_id = uuid.uuid4().hex
                dispatch_id = _canonical_sha256(
                    {
                        "prepared_run_id": prepared_run_id,
                        "checkpoint_sha256": checkpoint["sha256"],
                        "scoring_shard": shard,
                        "generation": generation,
                        "attempt_id": attempt_id,
                    }
                )
                job = _scoring_shard_job(
                    prepared_run_id=prepared_run_id,
                    checkpoint=checkpoint,
                    checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                    phase_claim=phase_claim,
                    checkpoint_claim=payload,
                    scoring_plan=scoring_plan,
                    shard=shard,
                    attempt_id=attempt_id,
                    dispatch_id=dispatch_id,
                )
                intent = _dispatch_intent(
                    stage="score-checkpoint",
                    generation=generation,
                    attempt_id=attempt_id,
                    dispatch_id=dispatch_id,
                    job=job,
                )
            if intent is None or job is None:
                raise AssertionError("checkpoint scoring dispatch was not constructed")
            call_id = _spawn_committed_dispatch(
                dispatch_root=dispatch_root,
                intent=intent,
                job=job,
                function=score_checkpoint,
            )
            calls.append(call_id)
            dispatched_shards += 1
    expected_shards = len(checkpoints) * SCORING_SHARD_COUNT
    candidate_score_receipts = 0
    reducer_calls: list[str] = []
    rare_cells = prepared.get("rare_cells")
    if not isinstance(rare_cells, list) or len(rare_cells) != 3:
        raise RuntimeError("candidate reducer rare-cell binding drifted")
    if completed_shards == expected_shards:
        reducer_dispatch_root = _candidate_score_root(prepared_run_id) / "dispatches"
        for shard in scoring_plan["shards"]:
            score_root = _candidate_score_root(prepared_run_id) / f"shard={shard['shard_id']}"
            if (score_root / "receipt.json").is_file():
                _validate_candidate_score_shard(
                    prepared_run_id=prepared_run_id,
                    prepared=prepared,
                    checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                    rare_cells=rare_cells,
                    shard=shard,
                    volume_root=VOLUME_PATH,
                )
                candidate_score_receipts += 1
                continue
            dispatch_root = reducer_dispatch_root / f"shard={shard['shard_id']}"
            intents = _load_dispatch_intents(
                dispatch_root=dispatch_root, stage="reduce-candidate-scores"
            )
            intent = intents[-1] if intents else None
            reducer_job: dict[str, Any] | None = None
            state = "new"
            if intent is not None:
                reducer_job = {
                    "spec": spec,
                    "scoring_plan_sha256": scoring_plan["plan_id"],
                    "scoring_shard": shard,
                    "attempt_id": intent["attempt_id"],
                    "dispatch_id": intent["dispatch_id"],
                }
                if intent["job_sha256"] != _canonical_sha256(reducer_job):
                    raise RuntimeError("candidate reducer dispatch job binding drifted")
                state = _dispatch_attempt_state(
                    dispatch_root=dispatch_root, intent=intent
                )
                if state in {"completed", "failed"}:
                    volume.reload()
                    if (score_root / "receipt.json").is_file():
                        _validate_candidate_score_shard(
                            prepared_run_id=prepared_run_id,
                            prepared=prepared,
                            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                            rare_cells=rare_cells,
                            shard=shard,
                            volume_root=VOLUME_PATH,
                        )
                        candidate_score_receipts += 1
                        continue
                    if state == "completed":
                        raise RuntimeError(
                            "completed candidate reducer call lacks its final shard receipt"
                        )
                if state == "active":
                    continue
            if state in {"new", "failed"}:
                generation = len(intents)
                attempt_id = uuid.uuid4().hex
                dispatch_id = _canonical_sha256(
                    {
                        "prepared_run_id": prepared_run_id,
                        "scoring_plan_sha256": scoring_plan["plan_id"],
                        "scoring_shard": shard,
                        "generation": generation,
                        "attempt_id": attempt_id,
                    }
                )
                reducer_job = {
                    "spec": spec,
                    "scoring_plan_sha256": scoring_plan["plan_id"],
                    "scoring_shard": shard,
                    "attempt_id": attempt_id,
                    "dispatch_id": dispatch_id,
                }
                intent = _dispatch_intent(
                    stage="reduce-candidate-scores",
                    generation=generation,
                    attempt_id=attempt_id,
                    dispatch_id=dispatch_id,
                    job=reducer_job,
                )
            if intent is None or reducer_job is None:
                raise AssertionError("candidate reducer dispatch was not constructed")
            reducer_calls.append(
                _spawn_committed_dispatch(
                    dispatch_root=dispatch_root,
                    intent=intent,
                    job=reducer_job,
                    function=reduce_candidate_scores,
                )
            )
    finalisation_complete = (scoring_root / "finalisation.json").is_file()
    finalisation_submitted = False
    if candidate_score_receipts == SCORING_SHARD_COUNT and not finalisation_complete:
        dispatch_root = scoring_root / "finalisation-dispatches"
        intents = _load_dispatch_intents(
            dispatch_root=dispatch_root, stage="finalise-scoring"
        )
        intent = intents[-1] if intents else None
        finaliser_job: dict[str, Any] | None = None
        state = "new"
        if intent is not None:
            finaliser_job = {
                "spec": spec,
                "attempt_id": intent["attempt_id"],
                "dispatch_id": intent["dispatch_id"],
            }
            if intent["job_sha256"] != _canonical_sha256(finaliser_job):
                raise RuntimeError("scoring finalisation dispatch job binding drifted")
            state = _dispatch_attempt_state(
                dispatch_root=dispatch_root, intent=intent
            )
            if state in {"completed", "failed"}:
                volume.reload()
                if (scoring_root / "finalisation.json").is_file():
                    finalisation_complete = True
                elif state == "completed":
                    raise RuntimeError("completed finalisation call lacks its final receipt")
            elif state == "active":
                pass
        if not finalisation_complete and state not in {"active", "completed"}:
            if state in {"new", "failed"}:
                generation = len(intents)
                attempt_id = uuid.uuid4().hex
                dispatch_id = _canonical_sha256(
                    {
                        "prepared_run_id": prepared_run_id,
                        "scoring_plan_sha256": scoring_plan["plan_id"],
                        "generation": generation,
                        "attempt_id": attempt_id,
                    }
                )
                finaliser_job = {
                    "spec": spec,
                    "attempt_id": attempt_id,
                    "dispatch_id": dispatch_id,
                }
                intent = _dispatch_intent(
                    stage="finalise-scoring",
                    generation=generation,
                    attempt_id=attempt_id,
                    dispatch_id=dispatch_id,
                    job=finaliser_job,
                )
            if intent is None or finaliser_job is None:
                raise AssertionError("scoring finalisation dispatch was not constructed")
            _spawn_committed_dispatch(
                dispatch_root=dispatch_root,
                intent=intent,
                job=finaliser_job,
                function=finalise_scoring,
            )
            finalisation_submitted = True
    if finalisation_complete:
        status = "already_complete"
    elif finalisation_submitted:
        status = "finalisation_submitted"
    elif reducer_calls:
        status = "reduction_submitted"
    elif calls:
        status = "submitted"
    else:
        status = "in_progress"
    return {
        "status": status,
        "expected_checkpoints": 6,
        "completed_checkpoints": completed_checkpoints,
        "expected_scoring_shards": expected_shards,
        "completed_scoring_shards": completed_shards,
        "dispatched_scoring_shards": dispatched_shards,
        "reused_scoring_shards": reused_shards,
        "submitted_calls": len(calls),
        "completed_candidate_score_shards": candidate_score_receipts,
        "submitted_reducer_calls": len(reducer_calls),
        "batch_size": SCORING_BATCH_SIZE,
        "max_concurrent_scoring_jobs": SCORING_MAX_CONTAINERS,
        "finalisation_submitted": finalisation_submitted,
        "finalisation_complete": finalisation_complete,
        "phase_approval_claim_id": phase_claim["claim_id"],
        "locked_test_rows_accessed": 0,
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def _assemble_candidates(
    *, prepared_run_id: str, spec: Mapping[str, Any], volume_root: Path
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    prepared, frame_path = _load_prepared_contract(prepared_run_id, volume_root=volume_root)
    base = pq.read_table(frame_path).to_pylist()
    checkpoints = _checkpoint_specs(spec, volume_root=volume_root)
    checkpoint_bundle_sha256 = _canonical_sha256(checkpoints)
    by_key: dict[tuple[str, int], dict[tuple[str, str], Mapping[str, Any]]] = {}
    for checkpoint in checkpoints:
        root = (
            _checkpoint_root(prepared_run_id, volume_root=volume_root)
            / f"component={checkpoint['component']}"
            / f"seed={checkpoint['optimiser_seed']}"
        )
        receipt = _read_json(root / "receipt.json", where="checkpoint scoring receipt")
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        expected_receipt_keys = {
            "kind",
            "schema_version",
            "prepared_run_id",
            "prepared_frame_sha256",
            "checkpoint",
            "checkpoint_bundle_sha256",
            "checkpoint_sha256",
            "phase_approval_claim_id",
            "checkpoint_launch_claim_id",
            "approved_cost_usd",
            "estimated_upper_cost_usd",
            "runtime_source_bundle_sha256",
            "text_renderer",
            "token_truncation",
            "padding",
            "precision",
            "row_count",
            "render_count",
            "logit_rows",
            "logits",
            "locked_test_rows_accessed",
            "receipt_id",
        }
        if (
            set(receipt) != expected_receipt_keys
            or receipt.get("receipt_id") != _canonical_sha256(receipt_body)
            or receipt.get("kind") != SCORING_KIND
            or receipt.get("schema_version") != "1.0.0"
            or receipt.get("prepared_run_id") != prepared_run_id
            or receipt.get("checkpoint") != checkpoint
            or receipt.get("prepared_frame_sha256") != prepared["prepared_frame"]["sha256"]
            or receipt.get("checkpoint_bundle_sha256") != checkpoint_bundle_sha256
            or receipt.get("checkpoint_sha256") != checkpoint["sha256"]
            or receipt.get("approved_cost_usd")
            != str(spec["compute"]["phase_upper_cost_usd"]["launch-scoring"])
            or receipt.get("estimated_upper_cost_usd") != receipt.get("approved_cost_usd")
            or not isinstance(receipt.get("phase_approval_claim_id"), str)
            or len(receipt["phase_approval_claim_id"]) != 64
            or not isinstance(receipt.get("checkpoint_launch_claim_id"), str)
            or len(receipt["checkpoint_launch_claim_id"]) != 64
            or receipt.get("row_count") != len(base)
            or receipt.get("render_count") != 2
            or receipt.get("logit_rows") != len(base) * 2
            or receipt.get("runtime_source_bundle_sha256")
            != prepared["runtime_source_bundle_sha256"]
            or receipt.get("locked_test_rows_accessed") != 0
        ):
            raise RuntimeError("checkpoint scoring receipt binding drifted")
        descriptor = receipt.get("logits")
        if not isinstance(descriptor, Mapping):
            raise RuntimeError("checkpoint scoring lacks private logits descriptor")
        path = root / str(descriptor["relative_path"])
        if (
            set(descriptor) != {"relative_path", "sha256", "bytes", "row_count"}
            or not path.is_file()
            or path.stat().st_size != descriptor["bytes"]
            or _file_sha256(path) != descriptor["sha256"]
            or descriptor["row_count"] != len(base) * 2
        ):
            raise RuntimeError("checkpoint logits drifted")
        rows = pq.read_table(path).to_pylist()
        index = {(str(row["opaque_id"]), str(row["render"])): row for row in rows}
        if len(index) != len(base) * 2:
            raise RuntimeError("checkpoint logits do not conserve render pairs")
        by_key[(checkpoint["component"], checkpoint["optimiser_seed"])] = index
    result: list[dict[str, Any]] = []
    data = _data_module()
    presence_targets = tuple(data.TARGET_CLASSES)
    analytic_targets = tuple(data.ANALYTIC_TARGET_CLASSES)
    stance_labels = tuple(data.STANCE_CLASSES_B4)
    if len(presence_targets) != 6 or len(analytic_targets) != 5 or len(stance_labels) != 4:
        raise RuntimeError("canonical factorised ontology widths drifted")
    for row in base:
        opaque = row["record_id"]
        outputs: dict[str, Any] = {}
        for seed in (47, 61, 89):
            relevance_index = by_key[("relevance", seed)]
            target_index = by_key[("target_stance_b4", seed)]
            renders: dict[str, Any] = {}
            for render in ("full", "target_only"):
                relevance = relevance_index[(opaque, render)]
                target = target_index[(opaque, render)]
                renders[render] = {
                    "relevance": _sigmoid(float(relevance["relevance_logit"])),
                    "target_presence": {
                        key: _sigmoid(float(value))
                        for key, value in zip(
                            presence_targets,
                            target["target_presence_logits"],
                            strict=True,
                        )
                        if key in analytic_targets
                    },
                    "stance": {
                        key: {
                            label: value
                            for label, value in zip(
                                stance_labels,
                                _softmax(logits),
                                strict=True,
                            )
                        }
                        for key, logits in zip(
                            analytic_targets, target["stance_logits"], strict=True
                        )
                    },
                }
            outputs[str(seed)] = renders
        result.append(
            {
                "opaque_id": opaque,
                "thread_id": row["thread_id"],
                "near_duplicate_cluster_id": row["near_duplicate_cluster_id"],
                "subreddit": row["subreddit"],
                "year": int(row["year"]),
                "content_type": row["content_type"],
                "retrieval_mode": row["retrieval_mode"],
                "seed_outputs": outputs,
            }
        )
    return result


def _load_scoring_authority(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    volume_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scoring_root = _checkpoint_root(prepared_run_id, volume_root=volume_root)
    phase_claim = _read_json(scoring_root / "phase-approval.json", where="scoring phase claim")
    checkpoint_root = _checkpoint_scoring_root(
        run_id=prepared_run_id,
        checkpoint=checkpoint,
        volume_root=volume_root,
    )
    checkpoint_claim = _read_json(
        checkpoint_root / "launch-claim.json",
        where="checkpoint launch claim",
    )
    scoring_plan = _read_json(scoring_root / "scoring-plan.json", where="scoring shard plan")
    expected_plan = _scoring_plan_contract(prepared["prepared_frame"]["row_count"])
    if scoring_plan != expected_plan:
        raise RuntimeError("persisted scoring shard plan drifted")
    return phase_claim, checkpoint_claim, scoring_plan


def _checkpoint_shard_jobs(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    volume_root: Path,
) -> list[dict[str, Any]]:
    phase_claim, checkpoint_claim, scoring_plan = _load_scoring_authority(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint=checkpoint,
        volume_root=volume_root,
    )
    jobs: list[dict[str, Any]] = []
    for shard in scoring_plan["shards"]:
        root = _scoring_shard_root(
            run_id=prepared_run_id,
            checkpoint=checkpoint,
            shard=shard,
            volume_root=volume_root,
        )
        receipt = _read_json(root / "receipt.json", where="checkpoint scoring shard receipt")
        jobs.append(
            _scoring_shard_job(
            prepared_run_id=prepared_run_id,
            checkpoint=checkpoint,
            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
            phase_claim=phase_claim,
            checkpoint_claim=checkpoint_claim,
            scoring_plan=scoring_plan,
            shard=shard,
            attempt_id=str(receipt.get("attempt_id")),
            dispatch_id=str(receipt.get("dispatch_id")),
        )
        )
    return jobs


def _validated_checkpoint_shard_union(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    volume_root: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Reject missing, duplicate, or mispartitioned private scoring shards."""

    receipts: list[dict[str, Any]] = []
    paths: list[Path] = []
    jobs = _checkpoint_shard_jobs(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint=checkpoint,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        volume_root=volume_root,
    )
    for job in jobs:
        shard = job["scoring_shard"]
        prepared_shard_path = _load_prepared_frame_shard(
            run_id=prepared_run_id,
            prepared=prepared,
            shard=shard,
            volume_root=volume_root,
        )
        expected_ids = set(
            _read_parquet_slice(
                prepared_shard_path,
                start=0,
                row_count=shard["row_count"],
                columns=["record_id"],
            )["record_id"].to_pylist()
        )
        receipt, rows = _validate_scoring_shard_receipt(
            job,
            prepared=prepared,
            volume_root=volume_root,
        )
        observed_ids = {row["opaque_id"] for row in rows}
        if len(expected_ids) != shard["row_count"] or observed_ids != expected_ids:
            raise RuntimeError("checkpoint scoring shard union drifted from prepared rows")
        root = _scoring_shard_root(
            run_id=prepared_run_id,
            checkpoint=checkpoint,
            shard=shard,
            volume_root=volume_root,
        )
        receipts.append(receipt)
        paths.append(root / receipt["logits"]["relative_path"])
    if (
        len(receipts) != SCORING_SHARD_COUNT
        or len({receipt["receipt_id"] for receipt in receipts}) != len(receipts)
    ):
        raise RuntimeError("checkpoint scoring shard receipts are missing or duplicated")
    return receipts, paths


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _finalise_checkpoint_scoring_index(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    volume_root: Path,
) -> dict[str, Any]:
    """Publish a metadata-only index over ten complete private logit shards."""

    receipts, _shard_paths = _validated_checkpoint_shard_union(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint=checkpoint,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        volume_root=volume_root,
    )
    jobs = _checkpoint_shard_jobs(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint=checkpoint,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        volume_root=volume_root,
    )
    approval = _validate_scoring_job_approval(
        {key: jobs[0][key] for key in SCORING_JOB_KEYS},
        prepared=prepared,
        volume_root=volume_root,
    )
    root = _checkpoint_scoring_root(
        run_id=prepared_run_id,
        checkpoint=checkpoint,
        volume_root=volume_root,
    )
    expected_rows = prepared["prepared_frame"]["row_count"] * len(SCORING_RENDERS)
    checkpoint_path = volume_root / _safe_relative(
        checkpoint.get("relative_path"), where="checkpoint path"
    )
    if not checkpoint_path.is_file() or _file_sha256(checkpoint_path) != checkpoint["sha256"]:
        raise RuntimeError("frozen B4 checkpoint drifted during finalisation")
    shard_bindings = sorted(
        (
            {
                "scoring_shard": receipt["scoring_shard"],
                "receipt_id": receipt["receipt_id"],
                "logits_sha256": receipt["logits"]["sha256"],
                "logit_rows": receipt["logit_rows"],
            }
            for receipt in receipts
        ),
        key=lambda value: value["scoring_shard"]["shard_id"],
    )
    if sum(binding["logit_rows"] for binding in shard_bindings) != expected_rows:
        raise RuntimeError("checkpoint scoring shard index does not conserve render pairs")
    body = {
        "kind": SCORING_INDEX_KIND,
        "schema_version": "1.0.0",
        "prepared_run_id": prepared_run_id,
        "prepared_frame_sha256": prepared["prepared_frame"]["sha256"],
        "checkpoint": dict(checkpoint),
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "checkpoint_sha256": checkpoint["sha256"],
        **approval,
        "scoring_plan_sha256": jobs[0]["scoring_plan_sha256"],
        "batch_size": SCORING_BATCH_SIZE,
        "row_count": prepared["prepared_frame"]["row_count"],
        "render_count": len(SCORING_RENDERS),
        "logit_rows": expected_rows,
        "shard_count": len(shard_bindings),
        "shards": shard_bindings,
        "locked_test_rows_accessed": 0,
    }
    index_receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(index_receipt, where="acquisition checkpoint shard index")
    _write_immutable_json(root / "receipt.json", index_receipt)
    return index_receipt


def _candidate_score_root(run_id: str, *, volume_root: Path = VOLUME_PATH) -> Path:
    return _checkpoint_root(run_id, volume_root=volume_root) / "candidate-scores"


def _compact_candidate_score_rows(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    checkpoints: Sequence[Mapping[str, Any]],
    rare_cells: Sequence[Mapping[str, Any]],
    policy: Any,
    shard: Mapping[str, Any],
    volume_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reduce one data shard's six-checkpoint logits to compact selector records."""

    import pyarrow.parquet as pq

    frame_path = _load_prepared_frame_shard(
        run_id=prepared_run_id,
        prepared=prepared,
        shard=shard,
        volume_root=volume_root,
    )
    base = pq.read_table(frame_path).to_pylist()
    base_ids = {row["record_id"] for row in base}
    if len(base) != shard["row_count"] or len(base_ids) != len(base):
        raise RuntimeError("prepared-frame shard does not conserve unique rows")
    by_key: dict[tuple[str, int], dict[tuple[str, str], Mapping[str, Any]]] = {}
    source_receipt_ids: list[str] = []
    for checkpoint in checkpoints:
        jobs = _checkpoint_shard_jobs(
            prepared_run_id=prepared_run_id,
            prepared=prepared,
            checkpoint=checkpoint,
            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
            volume_root=volume_root,
        )
        matches = [job for job in jobs if job["scoring_shard"] == dict(shard)]
        if len(matches) != 1:
            raise RuntimeError("candidate score reducer could not resolve one scoring shard")
        receipt, logits = _validate_scoring_shard_receipt(
            matches[0],
            prepared=prepared,
            volume_root=volume_root,
        )
        index = {(row["opaque_id"], row["render"]): row for row in logits}
        expected_pairs = {
            (opaque_id, render) for opaque_id in base_ids for render in SCORING_RENDERS
        }
        if set(index) != expected_pairs:
            raise RuntimeError("candidate score reducer logit union drifted")
        by_key[(checkpoint["component"], checkpoint["optimiser_seed"])] = index
        source_receipt_ids.append(receipt["receipt_id"])
    if len(by_key) != 6 or len(set(source_receipt_ids)) != 6:
        raise RuntimeError("candidate score reducer requires six distinct checkpoint shards")
    data = _data_module()
    presence_targets = tuple(data.TARGET_CLASSES)
    analytic_targets = tuple(data.ANALYTIC_TARGET_CLASSES)
    stance_labels = tuple(data.STANCE_CLASSES_B4)
    result: list[dict[str, Any]] = []
    selector = _acquisition_module()
    for row in base:
        opaque_id = row["record_id"]
        seed_outputs: dict[str, Any] = {}
        for seed in (47, 61, 89):
            renders: dict[str, Any] = {}
            for render in SCORING_RENDERS:
                relevance = by_key[("relevance", seed)][(opaque_id, render)]
                target = by_key[("target_stance_b4", seed)][(opaque_id, render)]
                renders[render] = {
                    "relevance": _sigmoid(float(relevance["relevance_logit"])),
                    "target_presence": {
                        key: _sigmoid(float(value))
                        for key, value in zip(
                            presence_targets,
                            target["target_presence_logits"],
                            strict=True,
                        )
                        if key in analytic_targets
                    },
                    "stance": {
                        key: dict(
                            zip(stance_labels, _softmax(logits), strict=True)
                        )
                        for key, logits in zip(
                            analytic_targets,
                            target["stance_logits"],
                            strict=True,
                        )
                    },
                }
            seed_outputs[str(seed)] = renders
        compact = selector.compute_compact_score_record(
                {
                    "opaque_id": opaque_id,
                    "thread_id": row["thread_id"],
                    "near_duplicate_cluster_id": row["near_duplicate_cluster_id"],
                    "subreddit": row["subreddit"],
                    "year": int(row["year"]),
                    "content_type": row["content_type"],
                    "retrieval_mode": row["retrieval_mode"],
                    "seed_outputs": seed_outputs,
                },
                rare_cells=rare_cells,
                expected_seed_count=3,
            )
        ranking_source = dict(compact)
        compact["probability_tiebreak_sha256"] = selector.build_probability_selection_row(
            ranking_source,
            quota=1,
            population_rows=1,
            policy=policy,
        )["selection_tiebreak_sha256"]
        for bucket in policy.bucket_quotas:
            compact[f"{bucket}_tiebreak_sha256"] = selector.active_rank_key(
                ranking_source,
                bucket=bucket,
                policy=policy,
            )[1]
        result.append(compact)
    if len(result) != len(base) or {row["opaque_id"] for row in result} != base_ids:
        raise RuntimeError("compact candidate scores do not conserve prepared rows")
    return result, sorted(source_receipt_ids)


def _validate_candidate_score_shard(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    rare_cells: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
    volume_root: Path,
) -> tuple[dict[str, Any], Path]:
    import pyarrow.parquet as pq

    root = _candidate_score_root(prepared_run_id, volume_root=volume_root) / (
        f"shard={shard['shard_id']}"
    )
    receipt = _read_json(root / "receipt.json", where="candidate score shard receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptor = receipt.get("candidate_scores")
    if (
        receipt.get("receipt_id") != _canonical_sha256(body)
        or receipt.get("kind") != CANDIDATE_SCORE_KIND
        or receipt.get("schema_version") != "1.0.0"
        or receipt.get("prepared_run_id") != prepared_run_id
        or receipt.get("prepared_frame_sha256") != prepared["prepared_frame"]["sha256"]
        or receipt.get("checkpoint_bundle_sha256") != checkpoint_bundle_sha256
        or receipt.get("rare_cell_list_sha256") != _canonical_sha256(list(rare_cells))
        or receipt.get("scoring_shard") != dict(shard)
        or receipt.get("row_count") != shard["row_count"]
        or not isinstance(receipt.get("source_scoring_receipt_ids"), list)
        or len(receipt["source_scoring_receipt_ids"]) != 6
        or len(set(receipt["source_scoring_receipt_ids"])) != 6
        or not isinstance(receipt.get("attempt_id"), str)
        or not receipt["attempt_id"]
        or not isinstance(receipt.get("dispatch_id"), str)
        or not receipt["dispatch_id"]
        or not isinstance(descriptor, Mapping)
        or set(descriptor) != {"relative_path", "sha256", "bytes", "row_count"}
        or receipt.get("locked_test_rows_accessed") != 0
    ):
        raise RuntimeError("candidate score shard receipt binding drifted")
    path = root / _safe_relative(
        descriptor["relative_path"], where="candidate score shard path"
    )
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != descriptor["sha256"]
        or descriptor["row_count"] != shard["row_count"]
        or pq.ParquetFile(path).metadata.num_rows != shard["row_count"]
    ):
        raise RuntimeError("candidate score shard content drifted")
    return receipt, path


def _materialise_candidate_score_shard(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    checkpoints: Sequence[Mapping[str, Any]],
    rare_cells: Sequence[Mapping[str, Any]],
    policy: Any,
    shard: Mapping[str, Any],
    attempt_id: str,
    dispatch_id: str,
    volume_root: Path,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = _candidate_score_root(prepared_run_id, volume_root=volume_root) / (
        f"shard={shard['shard_id']}"
    )
    if (root / "receipt.json").is_file():
        receipt, _path = _validate_candidate_score_shard(
            prepared_run_id=prepared_run_id,
            prepared=prepared,
            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
            rare_cells=rare_cells,
            shard=shard,
            volume_root=volume_root,
        )
        return receipt
    attempt = root / "attempts" / f"attempt={attempt_id}.incomplete"
    if attempt.exists():
        raise FileExistsError("candidate score shard attempt already exists")
    attempt.mkdir(parents=True)
    rows, source_receipt_ids = _compact_candidate_score_rows(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        checkpoints=checkpoints,
        rare_cells=rare_cells,
        policy=policy,
        shard=shard,
        volume_root=volume_root,
    )
    path = attempt / "candidate-scores.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    descriptor = _descriptor(path, root=root, row_count=len(rows))
    score_projection = [
        {
            "opaque_id": row["opaque_id"],
            **{
                bucket: row[bucket]
                for bucket in (
                    "uncertainty_disagreement",
                    "boundary",
                    "rare_cell",
                    "multi_context",
                )
            },
        }
        for row in sorted(rows, key=lambda value: value["opaque_id"])
    ]
    body = {
        "kind": CANDIDATE_SCORE_KIND,
        "schema_version": "1.0.0",
        "prepared_run_id": prepared_run_id,
        "prepared_frame_sha256": prepared["prepared_frame"]["sha256"],
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "rare_cell_list_sha256": _canonical_sha256(list(rare_cells)),
        "scoring_shard": dict(shard),
        "source_scoring_receipt_ids": source_receipt_ids,
        "attempt_id": attempt_id,
        "dispatch_id": dispatch_id,
        "row_count": len(rows),
        "candidate_score_projection_sha256": _acquisition_module().canonical_sequence_sha256(
            score_projection
        ),
        "candidate_scores": descriptor,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(receipt, where="candidate score shard receipt")
    _write_immutable_json(attempt / "receipt.json", receipt)
    _write_immutable_json(root / "receipt.json", receipt)
    return receipt


def _validated_candidate_score_union(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoint_bundle_sha256: str,
    checkpoints: Sequence[Mapping[str, Any]],
    rare_cells: Sequence[Mapping[str, Any]],
    volume_root: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    import pyarrow.parquet as pq

    plan = _scoring_plan_contract(prepared["prepared_frame"]["row_count"])
    receipts: list[dict[str, Any]] = []
    paths: list[Path] = []
    all_ids: set[str] = set()
    for shard in plan["shards"]:
        receipt, path = _validate_candidate_score_shard(
            prepared_run_id=prepared_run_id,
            prepared=prepared,
            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
            rare_cells=rare_cells,
            shard=shard,
            volume_root=volume_root,
        )
        expected_source_ids: list[str] = []
        for checkpoint in checkpoints:
            jobs = _checkpoint_shard_jobs(
                prepared_run_id=prepared_run_id,
                prepared=prepared,
                checkpoint=checkpoint,
                checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                volume_root=volume_root,
            )
            matches = [job for job in jobs if job["scoring_shard"] == dict(shard)]
            if len(matches) != 1:
                raise RuntimeError("candidate score source shard binding is incomplete")
            source_root = _scoring_shard_root(
                run_id=prepared_run_id,
                checkpoint=checkpoint,
                shard=shard,
                volume_root=volume_root,
            )
            source_receipt = _read_json(
                source_root / "receipt.json", where="candidate score source receipt"
            )
            expected_source_ids.append(source_receipt["receipt_id"])
        if receipt["source_scoring_receipt_ids"] != sorted(expected_source_ids):
            raise RuntimeError("candidate score source receipt binding drifted")
        score_ids = set(pq.read_table(path, columns=["opaque_id"])["opaque_id"].to_pylist())
        frame_path = _load_prepared_frame_shard(
            run_id=prepared_run_id,
            prepared=prepared,
            shard=shard,
            volume_root=volume_root,
        )
        frame_ids = set(
            pq.read_table(frame_path, columns=["record_id"])["record_id"].to_pylist()
        )
        if (
            len(score_ids) != shard["row_count"]
            or len(frame_ids) != shard["row_count"]
            or score_ids != frame_ids
            or all_ids & score_ids
        ):
            raise RuntimeError("candidate score shard union drifted from prepared rows")
        all_ids.update(score_ids)
        receipts.append(receipt)
        paths.append(path)
    if len(all_ids) != prepared["prepared_frame"]["row_count"]:
        raise RuntimeError("candidate score shards do not conserve the prepared universe")
    return receipts, paths


@app.function(
    image=image,
    cpu=16,
    memory=65_536,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def finalise_scoring(job: dict[str, Any]) -> dict[str, Any]:
    """Publish six shard indices after ten parallel compact reducers complete."""

    if not isinstance(job, Mapping) or set(job) != FINALISER_JOB_KEYS:
        raise ValueError("scoring finaliser job schema drifted")
    lease = _claim_dispatch_lease(job, stage="finalise-scoring")
    if lease is None:
        return _dispatch_not_acquired(job)
    volume.reload()
    clean = validate_input_spec(job["spec"], action="launch-scoring")
    validate_cuda_preflight(clean)
    prepared_run_id = _sha256(clean.get("prepared_run_id"), where="prepared run ID")
    prepared, _frame_path = _load_prepared_contract(prepared_run_id)
    _validate_prepared_spec_binding(prepared, clean)
    _validate_prepared_frame_shards(
        run_id=prepared_run_id,
        prepared_frame=prepared["prepared_frame"],
    )
    checkpoints = _checkpoint_specs(clean)
    checkpoint_bundle_sha256 = _canonical_sha256(checkpoints)
    checkpoint_receipts: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_receipts.append(
            _finalise_checkpoint_scoring_index(
                prepared_run_id=prepared_run_id,
                prepared=prepared,
                checkpoint=checkpoint,
                checkpoint_bundle_sha256=checkpoint_bundle_sha256,
                volume_root=VOLUME_PATH,
            )
        )
        volume.commit()
    rare_cells = prepared.get("rare_cells")
    if (
        not isinstance(rare_cells, list)
        or len(rare_cells) != 3
        or prepared.get("rare_cell_list_sha256") != _canonical_sha256(rare_cells)
    ):
        raise RuntimeError("prepared acquisition rare-cell freeze drifted")
    scoring_plan = _scoring_plan_contract(prepared["prepared_frame"]["row_count"])
    score_receipts, _score_paths = _validated_candidate_score_union(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        checkpoints=checkpoints,
        rare_cells=rare_cells,
        volume_root=VOLUME_PATH,
    )
    body = {
        "kind": "modernbert-acquisition-scoring-finalisation-v1",
        "schema_version": "1.0.0",
        "prepared_run_id": prepared_run_id,
        "prepared_frame_sha256": prepared["prepared_frame"]["sha256"],
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "scoring_plan_sha256": scoring_plan["plan_id"],
        "attempt_id": job["attempt_id"],
        "dispatch_id": job["dispatch_id"],
        "checkpoint_receipt_ids": sorted(
            receipt["receipt_id"] for receipt in checkpoint_receipts
        ),
        "candidate_score_receipt_ids": sorted(
            receipt["receipt_id"] for receipt in score_receipts
        ),
        "checkpoint_count": len(checkpoint_receipts),
        "candidate_score_shard_count": len(score_receipts),
        "scoring_shard_count": len(checkpoints) * SCORING_SHARD_COUNT,
        "batch_size": SCORING_BATCH_SIZE,
        "max_concurrent_scoring_jobs": SCORING_MAX_CONTAINERS,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": _canonical_sha256(body)}
    assert_metadata_only(receipt, where="scoring finalisation receipt")
    _write_immutable_json(_checkpoint_root(prepared_run_id) / "finalisation.json", receipt)
    volume.commit()
    return receipt


def _compact_record_from_sql(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "opaque_id",
            "thread_id",
            "near_duplicate_cluster_id",
            "subreddit",
            "year",
            "content_type",
            "retrieval_mode",
            "uncertainty_disagreement",
            "boundary",
            "rare_cell",
            "multi_context",
        )
    }


def _bounded_acquisition_ledger(
    *,
    score_paths: Sequence[Path],
    rare_cells: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, str],
    policy: Any,
    temporary_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select exactly with DuckDB external sorts and only arm-sized Python state."""

    import duckdb

    selector = _acquisition_module()
    source_list = "[" + ",".join(_sql_literal(path) for path in score_paths) + "]"
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = '4GB'")
        connection.execute(f"SET temp_directory = {_sql_literal(temporary_root)}")
        connection.execute(
            "CREATE TEMP VIEW candidates AS SELECT * FROM read_parquet(" + source_list + ")"
        )
        population, opaque_count, thread_count, cluster_count = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT opaque_id), COUNT(DISTINCT thread_id), "
            "COUNT(DISTINCT near_duplicate_cluster_id) FROM candidates"
        ).fetchone()
        if len({population, opaque_count, thread_count, cluster_count}) != 1:
            raise RuntimeError("compact candidate universe contains duplicate identities")
        raw_strata = connection.execute(
            "SELECT subreddit, year, content_type, retrieval_mode, COUNT(*) AS population_rows "
            "FROM candidates GROUP BY ALL"
        ).fetchall()
        strata = selector.allocate_probability_stratum_counts(
            [
                {
                    "stratum": {
                        "subreddit": row[0],
                        "year": int(row[1]),
                        "content_type": row[2],
                        "retrieval_mode": row[3],
                    },
                    "population_rows": int(row[4]),
                }
                for row in raw_strata
            ],
            policy,
        )
        probability_rows: list[dict[str, Any]] = []
        for stratum in strata:
            quota = stratum["sample_rows"]
            if quota == 0:
                continue
            value = stratum["stratum"]
            cursor = connection.execute(
                "SELECT * FROM candidates WHERE subreddit = ? AND year = ? "
                "AND content_type = ? AND retrieval_mode = ? "
                "ORDER BY probability_tiebreak_sha256 LIMIT ?",
                [
                    value["subreddit"],
                    value["year"],
                    value["content_type"],
                    value["retrieval_mode"],
                    quota,
                ],
            )
            names = [description[0] for description in cursor.description]
            selected = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
            if len(selected) != quota:
                raise RuntimeError("probability stratum selection did not conserve its quota")
            probability_rows.extend(
                selector.build_probability_selection_row(
                    _compact_record_from_sql(row),
                    quota=quota,
                    population_rows=stratum["population_rows"],
                    policy=policy,
                )
                for row in selected
            )
        probability_rows.sort(
            key=lambda row: (
                selector.canonical_sha256(row["stratum"]),
                row["selection_tiebreak_sha256"],
            )
        )
        if len(probability_rows) != policy.probability_rows:
            raise RuntimeError("probability selection does not conserve its arm")
        connection.execute(
            "CREATE TEMP TABLE probability_exclusions "
            "(opaque_id VARCHAR, near_duplicate_cluster_id VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO probability_exclusions VALUES (?, ?)",
            [
                (row["opaque_id"], row["near_duplicate_cluster_id"])
                for row in probability_rows
            ],
        )
        pool_count = connection.execute(
            "SELECT COUNT(*) FROM candidates c WHERE NOT EXISTS ("
            "SELECT 1 FROM probability_exclusions p WHERE p.opaque_id = c.opaque_id "
            "OR p.near_duplicate_cluster_id = c.near_duplicate_cluster_id)"
        ).fetchone()[0]
        if pool_count < sum(policy.bucket_quotas.values()):
            raise ValueError("active pool cannot satisfy the frozen bucket quotas")
        active_rows: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        selected_clusters = {
            row["near_duplicate_cluster_id"] for row in probability_rows
        }
        for bucket, quota in policy.bucket_quotas.items():
            cursor = connection.execute(
                "SELECT c.* FROM candidates c WHERE NOT EXISTS ("
                "SELECT 1 FROM probability_exclusions p WHERE p.opaque_id = c.opaque_id "
                "OR p.near_duplicate_cluster_id = c.near_duplicate_cluster_id) "
                f"ORDER BY {bucket} DESC, {bucket}_tiebreak_sha256"
            )
            names = [description[0] for description in cursor.description]
            accepted = 0
            rank = 0
            while accepted < quota:
                batch = cursor.fetchmany(4096)
                if not batch:
                    break
                for values in batch:
                    rank += 1
                    row = dict(zip(names, values, strict=True))
                    if (
                        row["opaque_id"] in selected_ids
                        or row["near_duplicate_cluster_id"] in selected_clusters
                    ):
                        continue
                    compact = _compact_record_from_sql(row)
                    expected_key = selector.active_rank_key(
                        compact,
                        bucket=bucket,
                        policy=policy,
                    )
                    if expected_key != (-row[bucket], row[f"{bucket}_tiebreak_sha256"]):
                        raise RuntimeError("persisted active ranking key drifted")
                    active_rows.append(
                        selector.build_active_selection_row(
                            compact,
                            bucket=bucket,
                            bucket_rank=rank,
                            policy=policy,
                        )
                    )
                    selected_ids.add(row["opaque_id"])
                    selected_clusters.add(row["near_duplicate_cluster_id"])
                    accepted += 1
                    if accepted == quota:
                        break
            if accepted != quota:
                raise ValueError(f"active bucket {bucket} cannot satisfy its quota")
        cursor = connection.execute(
            "SELECT opaque_id, uncertainty_disagreement, boundary, rare_cell, multi_context "
            "FROM candidates ORDER BY opaque_id"
        )

        def score_records() -> Any:
            while batch := cursor.fetchmany(8192):
                for row in batch:
                    yield {
                        "opaque_id": row[0],
                        "uncertainty_disagreement": row[1],
                        "boundary": row[2],
                        "rare_cell": row[3],
                        "multi_context": row[4],
                    }

        score_digest = selector.canonical_sequence_sha256(score_records())
        return selector.finalise_acquisition_ledger(
            eligible_population_rows=population,
            candidate_score_digest=score_digest,
            probability_rows=probability_rows,
            probability_strata=strata,
            active_rows=active_rows,
            rare_cells=rare_cells,
            input_bindings=input_bindings,
            policy=policy,
        )
    finally:
        connection.close()


def _scoring_artifact_digest(
    *,
    prepared_run_id: str,
    prepared: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    volume_root: Path,
) -> str:
    """Content-address six metadata indices and their sixty private logit shards."""

    bindings: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        root = (
            _checkpoint_root(prepared_run_id, volume_root=volume_root)
            / f"component={checkpoint['component']}"
            / f"seed={checkpoint['optimiser_seed']}"
        )
        receipt = _read_json(root / "receipt.json", where="checkpoint scoring receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        shards = receipt.get("shards")
        if (
            receipt.get("receipt_id") != _canonical_sha256(body)
            or receipt.get("kind") != SCORING_INDEX_KIND
            or receipt.get("prepared_run_id") != prepared_run_id
            or receipt.get("checkpoint") != checkpoint
            or not isinstance(shards, list)
            or len(shards) != SCORING_SHARD_COUNT
            or receipt.get("locked_test_rows_accessed") != 0
        ):
            raise RuntimeError("checkpoint scoring shard index drifted")
        source_receipts, _paths = _validated_checkpoint_shard_union(
            prepared_run_id=prepared_run_id,
            prepared=prepared,
            checkpoint=checkpoint,
            checkpoint_bundle_sha256=_canonical_sha256(list(checkpoints)),
            volume_root=volume_root,
        )
        expected_shards = sorted(
            (
                {
                    "scoring_shard": source["scoring_shard"],
                    "receipt_id": source["receipt_id"],
                    "logits_sha256": source["logits"]["sha256"],
                    "logit_rows": source["logit_rows"],
                }
                for source in source_receipts
            ),
            key=lambda value: value["scoring_shard"]["shard_id"],
        )
        if shards != expected_shards:
            raise RuntimeError("checkpoint scoring index/source shard binding drifted")
        bindings.append(
            {
                "component": checkpoint["component"],
                "optimiser_seed": checkpoint["optimiser_seed"],
                "receipt_id": _sha256(receipt.get("receipt_id"), where="checkpoint receipt ID"),
                "shard_bindings_sha256": _canonical_sha256(shards),
                "logit_rows": receipt.get("logit_rows"),
            }
        )
    return _canonical_sha256(
        sorted(bindings, key=lambda row: (row["component"], row["optimiser_seed"]))
    )


def _validate_teacher_handoff_read_only(
    *, ledger: Mapping[str, Any], acquisition_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Exercise the real teacher validator using only ephemeral memory-backed files."""

    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    teacher = importlib.import_module("reddit_china_stance.sol_teacher_acquisition_v2")
    memory_root = Path("/dev/shm")
    temporary_parent = memory_root if memory_root.is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix="acquisition-read-only-validation-", dir=temporary_parent
    ) as raw:
        root = Path(raw)
        ledger_path = root / "acquisition-ledger.json"
        ledger_path.write_text(
            json.dumps(ledger, ensure_ascii=False, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        source_path = root / "acquisition-source.parquet"
        pq.write_table(pa.Table.from_pylist(list(acquisition_rows)), source_path)
        checked_source, checked_ledger, interleaved = teacher._validate_source_and_ledger(
            source_path, ledger_path
        )
    expected_projection = [
        {key: row[key] for key in teacher.SOURCE_COLUMNS} for row in acquisition_rows
    ]
    if (
        checked_source != expected_projection
        or checked_ledger != dict(ledger)
        or len(interleaved) != teacher.EXPECTED_ROWS
    ):
        raise RuntimeError("read-only teacher handoff validation changed reconstructed material")
    return {
        "teacher_packet_rows": len(checked_source),
        "teacher_source_projection_sha256": _canonical_sha256(expected_projection),
        "teacher_interleave_sha256": _canonical_sha256(interleaved),
    }


def _reconstruct_scoring_material(
    spec: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    """Reconstruct the exact closeout material without writing to the Volume."""

    import pyarrow.parquet as pq

    clean = validate_input_spec(spec, action="validate-scoring")
    policy_toml, policy_sha256 = load_frozen_policy()
    if clean["policy_sha256"] != policy_sha256:
        raise RuntimeError("acquisition scoring validation policy digest drifted")
    preflight = validate_cuda_preflight(clean, volume_root=volume_root)
    prepared_run_id = _sha256(clean.get("prepared_run_id"), where="prepared run ID")
    prepared, frame_path = _load_prepared_contract(prepared_run_id, volume_root=volume_root)
    _validate_prepared_spec_binding(prepared, clean)
    checkpoints = _checkpoint_specs(clean, volume_root=volume_root)
    checkpoint_bundle_sha256 = _canonical_sha256(checkpoints)
    scoring_artifact_sha256 = _scoring_artifact_digest(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoints=checkpoints,
        volume_root=volume_root,
    )
    rare_cells = prepared.get("rare_cells")
    if (
        not isinstance(rare_cells, list)
        or len(rare_cells) != 3
        or prepared.get("rare_cell_list_sha256") != _canonical_sha256(rare_cells)
    ):
        raise RuntimeError("prepared acquisition rare-cell freeze drifted")
    bindings = {
        "eligible_frame_sha256": prepared["eligible_frame_sha256"],
        "source_inventory_sha256": prepared["source_inventory_sha256"],
        "exclusion_ledger_sha256": prepared["exclusion_ledger_sha256"],
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "scoring_artifact_sha256": scoring_artifact_sha256,
        "policy_file_sha256": policy_sha256,
    }
    _score_receipts, score_paths = _validated_candidate_score_union(
        prepared_run_id=prepared_run_id,
        prepared=prepared,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        checkpoints=checkpoints,
        rare_cells=rare_cells,
        volume_root=volume_root,
    )
    with tempfile.TemporaryDirectory(prefix="bounded-acquisition-selection-") as raw_temp:
        ledger, public = _bounded_acquisition_ledger(
            score_paths=score_paths,
            rare_cells=rare_cells,
            input_bindings=bindings,
            policy=_acquisition_policy_from_toml(policy_toml),
            temporary_root=Path(raw_temp),
        )
    ledger_rows = [*ledger["probability_arm"]["rows"], *ledger["active_arm"]["rows"]]
    ledger_ids = [row["opaque_id"] for row in ledger_rows]
    if len(ledger_ids) != 2_000 or len(set(ledger_ids)) != 2_000:
        raise RuntimeError("acquisition selections do not conserve two arms")
    selected_ids = set(ledger_ids)
    frame: dict[str, dict[str, Any]] = {}
    for shard in _scoring_shard_plan(prepared["prepared_frame"]["row_count"]):
        prepared_shard_path = _load_prepared_frame_shard(
            run_id=prepared_run_id,
            prepared=prepared,
            shard=shard,
            volume_root=volume_root,
        )
        for row in pq.read_table(prepared_shard_path).to_pylist():
            if row["record_id"] in selected_ids:
                if row["record_id"] in frame:
                    raise RuntimeError("selected eligible-frame identity is duplicated")
                frame[row["record_id"]] = row
    if set(frame) != selected_ids:
        raise RuntimeError("selected eligible-frame rows are missing from prepared shards")
    acquisition_rows: list[dict[str, Any]] = []
    for opaque_id in ledger_ids:
        row = frame[opaque_id]
        bounded = _packet_bounded_fields(row)
        if bounded != {
            "target_text": row["target_text"],
            "submission_context": row["submission_context"],
            "parent_context": row["parent_context"],
        }:
            raise RuntimeError("eligible frame text differs from exact packet bounds")
        acquisition_rows.append(
            {
                "opaque_id": opaque_id,
                "source_sample_id": opaque_id,
                "thread_id": row["thread_id"],
                **bounded,
            }
        )
    teacher_validation = _validate_teacher_handoff_read_only(
        ledger=ledger, acquisition_rows=acquisition_rows
    )
    return {
        "prepared_run_id": prepared_run_id,
        "prepared": prepared,
        "frame_path": frame_path,
        "checkpoints": checkpoints,
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "scoring_artifact_sha256": scoring_artifact_sha256,
        "preflight": preflight,
        "ledger": ledger,
        "public": public,
        "acquisition_rows": acquisition_rows,
        "teacher_validation": teacher_validation,
    }


def validate_scoring_artifacts(
    spec: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    """Read-only scoring and closeout reconstruction with metadata-only output."""

    material = _reconstruct_scoring_material(spec, volume_root=volume_root)
    body = {
        "status": "validated",
        "kind": "modernbert-acquisition-scoring-validation-v1",
        "schema_version": "1.0.0",
        "prepared_run_id": material["prepared_run_id"],
        "cuda_preflight_receipt_id": _sha256(
            material["preflight"].get("receipt_id"), where="CUDA preflight receipt ID"
        ),
        "checkpoint_bundles": len(material["checkpoints"]),
        "checkpoint_bundle_sha256": material["checkpoint_bundle_sha256"],
        "scoring_artifact_sha256": material["scoring_artifact_sha256"],
        "ledger_id": _sha256(material["ledger"].get("ledger_id"), where="ledger ID"),
        "ledger_sha256": _canonical_sha256(material["ledger"]),
        "public_receipt_id": _sha256(
            material["public"].get("receipt_id"), where="acquisition public receipt ID"
        ),
        **material["teacher_validation"],
        "runtime_source_bundle_sha256": material["prepared"]["runtime_source_bundle_sha256"],
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "validation_id": _canonical_sha256(body)}
    assert_metadata_only(result, where="acquisition scoring validation")
    return result


@app.function(
    image=image,
    cpu=16,
    memory=65_536,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def validate_scoring(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate all scoring artefacts and closeout reconstruction without publication."""

    volume.reload()
    return validate_scoring_artifacts(spec, volume_root=VOLUME_PATH)


@app.function(
    image=image,
    cpu=16,
    memory=65_536,
    timeout=CPU_PREPARE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def closeout(spec: dict[str, Any], approved_cost_usd: str) -> dict[str, Any]:
    """Build the exact private acquisition ledger and ledger-ordered teacher source."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    volume.reload()
    spec = validate_input_spec(spec, action="closeout")
    _validate_phase_approval(spec, action="closeout", approved_cost_usd=approved_cost_usd)
    material = _reconstruct_scoring_material(spec, volume_root=VOLUME_PATH)
    prepared_run_id = material["prepared_run_id"]
    prepared = material["prepared"]
    ledger = material["ledger"]
    public = material["public"]
    acquisition_rows = material["acquisition_rows"]
    policy_sha256 = spec["policy_sha256"]
    root = _run_root(prepared_run_id)
    final = root / "acquisition"
    staging = final.parent / ".acquisition.incomplete"
    if final.exists():
        public_existing = _read_json(final / "receipt.json", where="acquisition receipt")
        expected_keys = set(public) | {
            "kind",
            "policy_sha256",
            "prepared_run_id",
            "teacher_packet_rows",
            "runtime_source_bundle_sha256",
            "private_artifacts",
            "locked_test_rows_accessed",
            "receipt_id",
        }
        if (
            set(public_existing) != expected_keys
            or any(public_existing.get(key) != value for key, value in public.items())
            or public_existing.get("kind") != CLOSEOUT_KIND
            or public_existing.get("policy_sha256") != policy_sha256
            or public_existing.get("prepared_run_id") != prepared_run_id
        ):
            raise RuntimeError("existing acquisition closeout differs")
        return public_existing
    if staging.exists():
        raise FileExistsError("stale acquisition closeout staging requires manual reconciliation")
    staging.mkdir(parents=True)
    ledger_path = staging / "acquisition-ledger.json"
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_schema = pa.schema(
        [
            pa.field("opaque_id", pa.string(), nullable=False),
            pa.field("source_sample_id", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("submission_context", pa.string()),
            pa.field("parent_context", pa.string()),
        ]
    )
    source_path = staging / "acquisition-source.parquet"
    pq.write_table(
        pa.Table.from_pylist(acquisition_rows, schema=source_schema),
        source_path,
        compression="zstd",
    )
    persisted_source = pq.read_table(source_path)
    if tuple(persisted_source.column_names) != tuple(field.name for field in source_schema):
        raise RuntimeError("acquisition teacher source schema drifted after publication")
    persisted_rows = persisted_source.select(["opaque_id", "source_sample_id"]).to_pylist()
    if any(row["source_sample_id"] != row["opaque_id"] for row in persisted_rows):
        raise RuntimeError("acquisition teacher source ID aliases drifted after publication")
    teacher = importlib.import_module("reddit_china_stance.sol_teacher_acquisition_v2")
    checked_source, checked_ledger, _interleaved = teacher._validate_source_and_ledger(
        source_path, ledger_path
    )
    expected_projection = [
        {key: row[key] for key in teacher.SOURCE_COLUMNS} for row in acquisition_rows
    ]
    if checked_source != expected_projection or checked_ledger != ledger:
        raise RuntimeError("teacher handoff validation changed acquisition artefacts")
    private_descriptors = {
        "acquisition-ledger.json": _descriptor(ledger_path, root=staging, row_count=2_000),
        "acquisition-source.parquet": _descriptor(source_path, root=staging, row_count=2_000),
    }
    receipt_body = {
        **public,
        "kind": CLOSEOUT_KIND,
        "policy_sha256": policy_sha256,
        "prepared_run_id": prepared_run_id,
        "teacher_packet_rows": 2_000,
        "runtime_source_bundle_sha256": prepared["runtime_source_bundle_sha256"],
        "private_artifacts": {
            name: {key: value for key, value in descriptor.items() if key != "relative_path"}
            for name, descriptor in private_descriptors.items()
        },
        "locked_test_rows_accessed": 0,
    }
    receipt = {**receipt_body, "receipt_id": _canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="acquisition closeout receipt")
    (staging / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    staging.replace(final)
    volume.commit()
    return receipt


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def inspect(spec: dict[str, Any]) -> dict[str, Any]:
    volume.reload()
    spec = validate_input_spec(spec, action="inspect")
    run_id = _sha256(spec.get("prepared_run_id"), where="prepared run ID")
    prepared, _frame_path = _load_prepared_contract(run_id)
    _validate_prepared_spec_binding(prepared, spec)
    root = _checkpoint_root(run_id)
    scoring_plan = _scoring_plan_contract(prepared["prepared_frame"]["row_count"])
    completed_checkpoints = len(list(root.glob("component=*/seed=*/receipt.json")))
    shard_receipts = list(root.glob("component=*/seed=*/shards/shard=*/receipt.json"))
    dispatches = list(root.glob("component=*/seed=*/dispatches/shard=*/dispatch=*/intent.json"))
    worker_seconds = 0.0
    scored_rows = 0
    for path in shard_receipts:
        receipt = _read_json(path, where="scoring inspection shard receipt")
        if (
            receipt.get("kind") != SCORING_SHARD_KIND
            or receipt.get("scoring_plan_sha256") != scoring_plan["plan_id"]
            or receipt.get("batch_size") != SCORING_BATCH_SIZE
            or receipt.get("render_count") != len(SCORING_RENDERS)
            or receipt.get("locked_test_rows_accessed") != 0
        ):
            raise RuntimeError("scoring inspection shard binding drifted")
        worker_seconds += float(receipt.get("wall_seconds", 0.0))
        scored_rows += int(receipt.get("row_count", 0))
    expected_shards = 6 * SCORING_SHARD_COUNT
    completed_shards = len(shard_receipts)
    completed_candidate_score_shards = len(
        list(root.glob("candidate-scores/shard=*/receipt.json"))
    )
    finalisation_complete = (root / "finalisation.json").is_file()
    remaining = expected_shards - completed_shards
    estimated_remaining = (
        worker_seconds / completed_shards * remaining / SCORING_MAX_CONTAINERS
        if completed_shards
        else None
    )
    if finalisation_complete:
        stage = "complete"
    elif completed_shards == expected_shards:
        stage = "finalisation"
    else:
        stage = "scoring-shards"
    incomplete = len(list(root.glob("**/*.incomplete")))
    result = {
        "prepared_run_id": run_id,
        "stage": stage,
        "completed_checkpoint_jobs": completed_checkpoints,
        "expected_checkpoint_jobs": 6,
        "completed_scoring_shards": completed_shards,
        "expected_scoring_shards": expected_shards,
        "completed_candidate_score_shards": completed_candidate_score_shards,
        "expected_candidate_score_shards": SCORING_SHARD_COUNT,
        "finalisation_complete": finalisation_complete,
        "dispatched_scoring_shards": len(dispatches),
        "scored_prepared_rows": scored_rows,
        "aggregate_worker_seconds": round(worker_seconds, 3),
        "rows_per_worker_second": scored_rows / worker_seconds if worker_seconds else None,
        "estimated_remaining_wall_seconds": estimated_remaining,
        "batch_size": SCORING_BATCH_SIZE,
        "max_concurrent_scoring_jobs": SCORING_MAX_CONTAINERS,
        "incomplete_entries": incomplete,
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition inspection")
    return result


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=15 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def inspect_prepare(spec: dict[str, Any]) -> dict[str, Any]:
    """Metadata-only view of a pre-publication prepare namespace."""
    volume.reload()
    clean = validate_input_spec(spec, action="prepare")
    inventories = {
        key: _receipt_backed_inventory(value, producer=key)
        for key, value in _discover_inputs().items()
    }
    pre_prepare_id = _pre_prepare_id(clean, inventories)
    shards = _prepare_shards(_discover_inputs())
    completed = 0
    rows = 0
    worker_seconds = 0.0
    root = _prepare_stage_root(pre_prepare_id)
    for shard in shards:
        receipt_path = root / "shards" / f"shard={shard['shard_id']}" / "receipt.json"
        if receipt_path.is_file():
            receipt = _read_json(receipt_path, where="prepare inspection shard receipt")
            if receipt.get("shard_id") == shard["shard_id"] and isinstance(
                receipt.get("row_count"), int
            ):
                completed += 1
                rows += receipt["row_count"]
                worker_seconds += float(receipt.get("wall_seconds", 0.0))
    context_completed = 0
    context_rows = 0
    context_worker_seconds = 0.0
    contexts_root = root / "contexts"
    if contexts_root.exists():
        for receipt_path in contexts_root.glob("partition=*/receipt.json"):
            receipt = _read_json(receipt_path, where="context inspection receipt")
            if isinstance(receipt.get("row_count"), int):
                context_completed += 1
                context_rows += receipt["row_count"]
                context_worker_seconds += float(receipt.get("wall_seconds", 0.0))
    durable = {
        stage: (root / stage / "receipt.json").is_file()
        for stage in ("exact-reduce", "near-reduce", "hydrated")
    }
    durable["context-requests"] = (root / "context-requests" / "index.json").is_file()
    if completed < len(shards):
        stage = "source-shards"
    elif not durable["exact-reduce"]:
        stage = "exact-reduce"
    elif not durable["near-reduce"]:
        stage = "near-reduce"
    elif not durable["context-requests"]:
        stage = "context-requests"
    elif context_completed < 120:
        stage = "context-shards"
    elif not durable["hydrated"]:
        stage = "hydrated"
    else:
        stage = "final-publication"
    base_rate = rows / worker_seconds if worker_seconds else None
    context_rate = context_rows / context_worker_seconds if context_worker_seconds else None
    base_eta = (
        worker_seconds / completed * (len(shards) - completed) / PREPARE_MAX_CONTAINERS
        if completed
        else None
    )
    context_eta = (
        context_worker_seconds
        / context_completed
        * (120 - context_completed)
        / PREPARE_MAX_CONTAINERS
        if context_completed
        else None
    )
    result = {
        "pre_prepare_id": pre_prepare_id,
        "stage": stage,
        "source_shards": {
            "expected": len(shards),
            "completed": completed,
            "row_count": rows,
            "aggregate_worker_seconds": round(worker_seconds, 3),
            "rows_per_worker_second": base_rate,
            "estimated_remaining_wall_seconds": base_eta,
        },
        "durable_stages": durable,
        "context_shards": {
            "expected": 120,
            "completed": context_completed,
            "row_count": context_rows,
            "aggregate_worker_seconds": round(context_worker_seconds, 3),
            "rows_per_worker_second": context_rate,
            "estimated_remaining_wall_seconds": context_eta,
        },
        "incomplete_entries": (
            len(list(root.glob("**/.publishing-*")))
            + len(list(root.glob("**/*.incomplete")))
            if root.exists()
            else 0
        ),
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition prepare inspection")
    return result


def required_confirmation(action: str, spec: Mapping[str, Any]) -> str:
    if action == "prepare":
        return "PREPARE_MODERNBERT_ACQUISITION_V1"
    prefixes = {
        "cuda-preflight": "CUDA_PREFLIGHT_MODERNBERT_ACQUISITION",
        "launch-scoring": "LAUNCH_SCORING_MODERNBERT_ACQUISITION",
        "closeout": "CLOSEOUT_MODERNBERT_ACQUISITION",
    }
    if action not in prefixes:
        raise ValueError("action has no confirmation token")
    run_id = _sha256(spec.get("prepared_run_id"), where="prepared run ID")
    return f"{prefixes[action]}_{run_id[:12]}"


@app.local_entrypoint()
def main(
    action: str = "",
    input_spec_path: str = "",
    prepare_receipt_path: str = str(DEFAULT_PREPARE_RECEIPT_PATH),
    prepared_run_id: str = "",
    promoted_input_spec_path: str = str(DEFAULT_PREPARED_INPUT_SPEC_PATH),
    compute_ledger_path: str = "data/private-modernbert-acquisition-v1/compute-ledger.json",
    retained_manifest_path: str = str(RETAINED_MANIFEST_REPO_PATH),
    retained_source_bundle_path: str = str(RETAINED_SOURCE_BUNDLE_REPO_PATH),
    local_volume_root: str = "",
    approved_cost_usd: str = "0",
    confirm: str = "",
) -> None:
    actions = {
        "build-input-local",
        "promote-prepared-input-local",
        "validate",
        "preflight-local",
        "validate-volume",
        "prepare",
        "cuda-preflight",
        "launch-scoring",
        "validate-scoring",
        "inspect",
        "inspect-prepare",
        "benchmark-plan-local",
        "benchmark-prepare",
        "closeout",
    }
    if action not in actions:
        raise ValueError("action must be exactly " + ", ".join(sorted(actions)))
    later_actions = {
        "cuda-preflight",
        "launch-scoring",
        "validate-scoring",
        "inspect",
        "closeout",
    }
    resolved_input_spec_path = (
        Path(input_spec_path)
        if input_spec_path
        else (
            DEFAULT_PREPARED_INPUT_SPEC_PATH if action in later_actions else DEFAULT_INPUT_SPEC_PATH
        )
    )
    if action == "build-input-local":
        result = build_local_input_spec(
            compute_ledger_path=Path(compute_ledger_path),
            manifest_path=Path(retained_manifest_path),
            source_bundle_path=Path(retained_source_bundle_path),
        )
        _write_immutable_json(resolved_input_spec_path, result)
        print(json.dumps(result, sort_keys=True, indent=2))
        return
    if action == "benchmark-plan-local":
        result = {
            "kind": "modernbert-acquisition-prepare-benchmark-plan-v1",
            **PREPARE_BENCHMARK,
            "locked_test_rows_accessed": 0,
        }
        assert_metadata_only(result, where="acquisition prepare benchmark plan")
        print(json.dumps(result, sort_keys=True, indent=2))
        return
    if action == "promote-prepared-input-local":
        result = promote_prepared_input_file(
            input_spec_path=resolved_input_spec_path,
            prepare_receipt_path=Path(prepare_receipt_path),
            output_spec_path=Path(promoted_input_spec_path),
            prepared_run_id=prepared_run_id,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return
    spec = _read_json(resolved_input_spec_path, where="acquisition input spec")
    validation_action = (
        None
        if action in {"validate", "preflight-local"}
        else ("prepare" if action == "benchmark-prepare" else action)
    )
    spec = validate_input_spec(spec, action=validation_action)
    if action in {"validate", "preflight-local"}:
        root = (
            Path(local_volume_root) if action == "preflight-local" and local_volume_root else None
        )
        result = preflight_local(spec, volume_root=root)
    elif action == "validate-volume":
        result = validate_volume.remote(spec)
    elif action == "validate-scoring":
        result = validate_scoring.remote(spec)
    elif action == "inspect":
        result = inspect.remote(spec)
    elif action == "inspect-prepare":
        result = inspect_prepare.remote(spec)
    elif action == "benchmark-prepare":
        result = run_prepare_benchmark.remote(spec, approved_cost_usd)
    else:
        _validate_phase_approval(spec, action=action, approved_cost_usd=approved_cost_usd)
        required = required_confirmation(action, spec)
        if confirm != required:
            raise RuntimeError(f"refusing mutation: pass --confirm {required}")
        if action == "prepare":
            result = prepare.remote(spec, approved_cost_usd)
            _write_immutable_json(Path(prepare_receipt_path), result)
        elif action == "cuda-preflight":
            result = cuda_preflight.remote(spec, approved_cost_usd)
        elif action == "launch-scoring":
            result = coordinate_scoring.remote(spec, approved_cost_usd)
        else:
            result = closeout.remote(spec, approved_cost_usd)
    print(json.dumps(result, sort_keys=True, indent=2))


__all__ = [
    "ACCOUNT_GPU_LIMIT",
    "ALLOWED_GPUS",
    "APP_NAME",
    "BASE_FACTORISED_RUN_ID",
    "CLOSEOUT_KIND",
    "COMPUTE_LEDGER_KIND",
    "DATASET_REVISION",
    "ENVIRONMENT_NAME",
    "HARD_MAX_APPROVAL_USD",
    "INPUT_SPEC_KIND",
    "MAX_CONCURRENT_CHECKPOINTS",
    "OUTPUT_PREFIX",
    "PREFLIGHT_KIND",
    "REQUIRED_SOURCE_FILES",
    "SCORING_BATCH_SIZE",
    "SCORING_INDEX_KIND",
    "SCORING_KIND",
    "SCORING_SHARD_COUNT",
    "SCORING_SHARD_KIND",
    "VOLUME_NAME",
    "VOLUME_PATH",
    "_assemble_candidates",
    "_canonical_sha256",
    "_checkpoint_specs",
    "_discover_inputs",
    "_prepared_contract",
    "_public_preparation_receipt",
    "_run_root",
    "_validate_scoring_job_approval",
    "build_local_input_spec",
    "closeout",
    "coordinate_scoring",
    "cuda_preflight",
    "enforce_cost_guardrail",
    "finalise_scoring",
    "inspect",
    "main",
    "preflight_local",
    "prepare",
    "promote_prepared_input_file",
    "promote_prepared_input_spec",
    "reduce_candidate_scores",
    "required_confirmation",
    "score_checkpoint",
    "validate_compute_contract",
    "validate_cuda_preflight",
    "validate_input_spec",
    "validate_scoring",
    "validate_scoring_artifacts",
    "validate_volume",
    "validate_volume_bindings",
]
