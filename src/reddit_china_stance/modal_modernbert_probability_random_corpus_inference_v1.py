"""Execute exact sharded inference for the calibrated probability-random ensemble.

This is a separate, content-addressed corpus authority.  It scores the 908,141-row
post-acquisition eligible candidate corpus (not all Reddit) with six frozen
checkpoints, applies the frozen calibration, and writes only private row-level
artefacts plus metadata-only receipts.  It cannot train, call a provider, read the
consumed locked test, choose another model/GPU, or silently retry failed work.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance import modernbert_probability_random_corpus_inference_v1 as corpus
from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-modernbert-probability-random-corpus-inference-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path("student-modernbert-probability-random-corpus-inference-v1")
POLICY_REPO_PATH = "configs/modernbert-probability-random-corpus-inference-v1.toml"
POLICY_RUNTIME_PATH = "/configs/modernbert-probability-random-corpus-inference-v1.toml"

ELIGIBLE_RELATIVE_PATH = Path(
    "student-modernbert-acquisition-v1/"
    f"run={candidate.ELIGIBLE_PREPARED_RUN_ID}/prepared/eligible-frame.parquet"
)
PREPARED_RECEIPT_RELATIVE_PATH = ELIGIBLE_RELATIVE_PATH.parent / "receipt.json"
ACQUISITION_LEDGER_RELATIVE_PATH = Path(
    "student-modernbert-acquisition-v1/"
    f"run={candidate.ELIGIBLE_PREPARED_RUN_ID}/acquisition/acquisition-ledger.json"
)
ELIGIBLE_FRAME_BYTES = 295_349_714

SAMPLE_RUN_ID = "247c077975b5527203e2de709e8c884dd90ca8e74976130d4b30da5a4a5c957c"
SAMPLE_RECEIPT_ID = "62c3bd62d9e6d547a441f3c05499f6e80a9f244d0830d9a271c50f804122cc32"
CHECKPOINT_COHORT_ID = "ad01414410538d5001ac5fb0070751ccfd251d31c5b75c5460bcd5a0dbe06f9b"
TEACHER_RUN_ID = "75c48c76db22be4f3eb009ec379cdf75e8ac88761b7475e6d9ce31f4f3ec0949"
TEACHER_RECEIPT_ID = "809fe8528f6928c0a1fa9b7ba55a18fa7a9beb0ced570f73eb60df239710ae31"
TEACHER_RECEIPT_FILE_SHA256 = "d3a7bf0ca0fe0e6162b90dc91a458251676a6ec79229e2fa915e7d5daf02f69e"
CALIBRATION_RUN_ID = "3a40abaa140214f4f4d7ac2cffac44bcdbff0da879b157f83f0c97b67aa5beec"
CALIBRATION_RECEIPT_ID = "f59dd582cb24b0f38b0e69aa56dd889e8031b6234f409acb71dbfd021d50e9af"
CALIBRATION_ID = "07b429e62653fb8513364327f67e7f620cf44496471c8443023767ad5f66526c"
CALIBRATION_FILE_SHA256 = "f593ce9baa9cd565184de0d673dd56a7c27843ce573fd399958d7559e6ab3c14"
CALIBRATION_FILE_BYTES = 5_347
GPU_TIMEOUT_SECONDS = corpus.MAXIMUM_WORKER_SECONDS
CPU_TIMEOUT_SECONDS = 60 * 60
LAUNCH_PARENT_TIMEOUT_SECONDS = 24 * 60 * 60


def _repository_root(module_path: Path) -> Path:
    resolved = module_path.resolve()
    if (
        len(resolved.parents) >= 3
        and resolved.parent.name == "reddit_china_stance"
        and resolved.parent.parent.name == "src"
    ):
        return resolved.parents[2]
    return resolved.parent


REPO_ROOT = _repository_root(Path(__file__))
REQUIRED_SOURCE_FILES = (
    POLICY_REPO_PATH,
    "src/reddit_china_stance/modal_modernbert_probability_random_corpus_inference_v1.py",
    "src/reddit_china_stance/modernbert_probability_random_corpus_inference_v1.py",
    "src/reddit_china_stance/modal_modernbert_probability_random_calibration_v1.py",
    "src/reddit_china_stance/modernbert_probability_random_calibration_v1.py",
    "src/reddit_china_stance/modernbert_probability_random_candidate_v1.py",
    "src/reddit_china_stance/modernbert_factorised_data.py",
    "src/reddit_china_stance/modernbert_factorised_experiment.py",
    "src/reddit_china_stance/modernbert_factorised_model.py",
    "src/reddit_china_stance/modernbert_factorised_training.py",
    "src/reddit_china_stance/modernbert_model.py",
    "src/reddit_china_stance/modernbert_trainer.py",
    "src/reddit_china_stance/human_seeded_consensus_v1.py",
    "src/reddit_china_stance/privacy.py",
    "src/reddit_china_stance/semantic_evaluation_v2.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
)
RUNTIME_DEPENDENCIES = {
    "accelerate": "1.10.1",
    "duckdb": "1.4.4",
    "huggingface-hub": "0.36.2",
    "jsonschema": "4.26.0",
    "numpy": "2.5.2",
    "pyarrow": "25.0.1",
    "pydantic": "2.13.4",
    "safetensors": "0.8.0",
    "torch": "2.8.0",
    "transformers": "4.57.6",
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME,
    environment_name=ENVIRONMENT_NAME,
    create_if_missing=False,
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
    .add_local_file(POLICY_REPO_PATH, remote_path=POLICY_RUNTIME_PATH)
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"existing immutable state differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise RuntimeError(f"stale immutable staging file exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _descriptor(path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(VOLUME_PATH).as_posix(),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "row_count": row_count,
    }


def _validate_descriptor(
    descriptor: Mapping[str, Any], *, where: str, expected_rows: int | None = None
) -> Path:
    import pyarrow.parquet as pq

    if set(descriptor) != {"relative_path", "sha256", "bytes", "row_count"}:
        raise RuntimeError(f"{where} descriptor schema drifted")
    relative = Path(str(descriptor["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{where} path is unsafe")
    path = VOLUME_PATH / relative
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != descriptor["sha256"]
        or pq.ParquetFile(path).metadata.num_rows != descriptor["row_count"]
        or (expected_rows is not None and descriptor["row_count"] != expected_rows)
    ):
        raise RuntimeError(f"{where} content drifted")
    return path


def _private_identity_sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("private identity sequence contains an invalid value")
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_source_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    files = value.get("files")
    if (
        set(value) != {"schema_version", "kind", "files", "code_sha256", "source_bundle_id"}
        or value.get("schema_version") != corpus.SCHEMA_VERSION
        or value.get("kind") != "listed-source-bundle-v1"
        or not isinstance(files, Mapping)
        or set(files) != set(REQUIRED_SOURCE_FILES)
        or any(
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not candidate.is_sha256(digest)
            for path, digest in files.items()
        )
    ):
        raise ValueError("corpus-inference source bundle schema drifted")
    clean_files = dict(sorted(files.items()))
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "listed-source-bundle-v1",
        "files": clean_files,
        "code_sha256": candidate.canonical_sha256(clean_files),
    }
    expected = {**body, "source_bundle_id": candidate.canonical_sha256(body)}
    if dict(value) != expected:
        raise RuntimeError("corpus-inference source bundle digest drifted")
    return expected


def build_source_bundle(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    files = {
        relative: _file_sha256(repo_root / relative) for relative in sorted(REQUIRED_SOURCE_FILES)
    }
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "listed-source-bundle-v1",
        "files": files,
        "code_sha256": candidate.canonical_sha256(files),
    }
    return validate_source_bundle({**body, "source_bundle_id": candidate.canonical_sha256(body)})


def _runtime_policy(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    bundle = validate_source_bundle(source_bundle)
    path = Path(POLICY_RUNTIME_PATH)
    if not path.is_file() or _file_sha256(path) != bundle["files"][POLICY_REPO_PATH]:
        raise RuntimeError("runtime corpus-inference policy differs from the source bundle")
    policy = tomllib.loads(path.read_text(encoding="utf-8"))
    eligible = policy.get("eligible_frame", {})
    selected = policy.get("candidate", {})
    execution = policy.get("execution", {})
    phases = policy.get("phase_availability", {})
    analysis = policy.get("analysis", {})
    projection = corpus.compute_projection()
    compute = policy.get("compute", {})
    compute_contract = {
        **projection,
        "gpu_type": compute.get("gpu_type"),
        "gpu_fallback_allowed": compute.get("gpu_fallback_allowed"),
        "retry_authorised": execution.get("retry_authorised"),
        "maximum_locked_test_rows": compute.get("maximum_locked_test_rows"),
        "maximum_corpus_rows": compute.get("maximum_corpus_rows"),
        "full_phase_operational_monitoring_target_usd": compute.get(
            "full_phase_operational_monitoring_target_usd"
        ),
        "cumulative_operational_monitoring_target_usd": compute.get(
            "cumulative_operational_monitoring_target_usd"
        ),
        "provider_billing_hard_cap_enforced": compute.get(
            "provider_billing_hard_cap_enforced"
        ),
        "proceed_without_provider_hard_cap_authorised": compute.get(
            "proceed_without_provider_hard_cap_authorised"
        ),
    }
    corpus.validate_compute_authority(compute_contract)
    if (
        policy.get("schema_version") != corpus.SCHEMA_VERSION
        or policy.get("kind") != "modernbert-probability-random-corpus-inference-policy-v1"
        or eligible.get("prepared_run_id") != candidate.ELIGIBLE_PREPARED_RUN_ID
        or eligible.get("source_row_count") != corpus.SOURCE_ROWS
        or eligible.get("post_acquisition_row_count") != corpus.CORPUS_ROWS
        or eligible.get("source_sha256") != candidate.ELIGIBLE_FRAME_SHA256
        or eligible.get("source_projection_sha256") != candidate.ELIGIBLE_PROJECTION_SHA256
        or eligible.get("acquisition_exclusion_count") != candidate.ACQUISITION_ROWS
        or eligible.get("acquisition_ledger_id") != candidate.ACQUISITION_LEDGER_ID
        or eligible.get("acquisition_ledger_file_sha256")
        != candidate.ACQUISITION_LEDGER_FILE_SHA256
        or selected.get("sample_run_id") != SAMPLE_RUN_ID
        or selected.get("sample_receipt_id") != SAMPLE_RECEIPT_ID
        or selected.get("teacher_run_id") != TEACHER_RUN_ID
        or selected.get("teacher_receipt_id") != TEACHER_RECEIPT_ID
        or selected.get("teacher_receipt_file_sha256") != TEACHER_RECEIPT_FILE_SHA256
        or selected.get("checkpoint_cohort_id") != CHECKPOINT_COHORT_ID
        or selected.get("calibration_run_id") != CALIBRATION_RUN_ID
        or selected.get("calibration_receipt_id") != CALIBRATION_RECEIPT_ID
        or selected.get("calibration_id") != CALIBRATION_ID
        or selected.get("checkpoint_count") != len(candidate.CHECKPOINT_SPECS)
        or selected.get("mixed_in_estimand") is not False
        or execution.get("input_shards") != corpus.SHARD_COUNT
        or execution.get("maximum_concurrent_l4s") != corpus.MAXIMUM_CONCURRENT_L4S
        or execution.get("batch_size") != 8
        or execution.get("smoke_rows") != corpus.SMOKE_ROWS
        or execution.get("retry_authorised") is not False
        or execution.get("gpu_job_topology")
        != (
            "120-shards-at-most-10-concurrent-each-tokenised-once-and-scored-by-all-six-"
            "checkpoints-sequentially"
        )
        or execution.get("launch_completion") != "parent-remains-live-and-gathers-all-child-calls"
        or execution.get("launch_parent_timeout_seconds") != LAUNCH_PARENT_TIMEOUT_SECONDS
        or execution.get("platform_preemption_recovery_authorised") is not True
        or execution.get("maximum_platform_preemption_recoveries_per_call") != 1
        or execution.get("maximum_platform_preemption_recoveries_global")
        != corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES
        or execution.get("maximum_platform_execution_attempts_per_call")
        != corpus.MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL
        or execution.get("execution_attempt_permit_gate")
        != "entry-time-atomic-per-job-queue-partition"
        or execution.get("execution_attempt_permit_partition_ttl_seconds") != 604_800
        or execution.get("provider_budget_gate_required") is not False
        or execution.get("provider_budget_gate_verified") is not False
        or execution.get("proceed_without_provider_hard_cap_authorised") is not True
        or compute.get("smoke_phase_operational_monitoring_target_usd")
        != format(corpus.NEW_SMOKE_COST_MONITORING_TARGET_USD, ".2f")
        or compute.get("rate_card_usd_per_gpu_second")
        != format(corpus.GPU_RATE_USD_PER_SECOND, "f")
        or compute.get("measured_steady_state_rows_per_second")
        != projection["measured_rows_per_second"]
        or compute.get("measured_one_time_model_load_seconds_per_worker")
        != format(corpus.MODEL_LOAD_SECONDS_PER_WORKER, "f")
        or compute.get("projected_steady_state_gpu_seconds")
        != projection["projected_steady_state_gpu_seconds"]
        or compute.get("projected_model_load_gpu_seconds")
        != projection["projected_model_load_gpu_seconds"]
        or compute.get("projected_gpu_seconds") != projection["projected_gpu_seconds"]
        or compute.get("projected_cost_usd") != projection["projected_cost_usd"]
        or compute.get("conservative_multiplier") != projection["conservative_multiplier"]
        or compute.get("conservative_gpu_seconds") != projection["conservative_gpu_seconds"]
        or compute.get("conservative_cost_usd") != projection["conservative_cost_usd"]
        or compute.get("maximum_worker_seconds") != projection["maximum_worker_seconds"]
        or compute.get("maximum_platform_execution_attempts_per_call")
        != projection["maximum_platform_execution_attempts_per_call"]
        or compute.get("budgeted_base_attempt_gpu_seconds")
        != projection["budgeted_base_attempt_gpu_seconds"]
        or compute.get("budgeted_base_attempt_cost_usd")
        != projection["budgeted_base_attempt_cost_usd"]
        or compute.get("maximum_platform_preemption_recoveries")
        != projection["maximum_platform_preemption_recoveries"]
        or compute.get("budgeted_recovery_attempt_gpu_seconds")
        != projection["budgeted_recovery_attempt_gpu_seconds"]
        or compute.get("budgeted_recovery_attempt_cost_usd")
        != projection["budgeted_recovery_attempt_cost_usd"]
        or compute.get("queue_bounded_authorised_execution_gpu_seconds")
        != projection["queue_bounded_authorised_execution_gpu_seconds"]
        or compute.get("queue_bounded_authorised_execution_cost_usd")
        != projection["queue_bounded_authorised_execution_cost_usd"]
        or compute.get("prior_failed_gpu_seconds_estimate")
        != projection["prior_failed_gpu_seconds_estimate"]
        or compute.get("prior_failed_cost_estimate_usd")
        != projection["prior_failed_cost_estimate_usd"]
        or compute.get("prior_smoke_cost_usd") != projection["prior_smoke_cost_usd"]
        or compute.get("new_smoke_cost_monitoring_target_usd")
        != projection["new_smoke_cost_monitoring_target_usd"]
        or compute.get("queue_bounded_cumulative_cost_projection_usd")
        != projection["queue_bounded_cumulative_cost_projection_usd"]
        or analysis
        != {
            "scored_scope_rows": corpus.CORPUS_ROWS,
            "calibration_member_rows": candidate.CALIBRATION_ROWS,
            "default_label_unseen_rows": corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS,
            "default_thesis_scope": "post-acquisition-corpus-excluding-calibration-members",
        }
        or phases
        != {
            "prepare": True,
            "smoke": True,
            "full_corpus_inference": True,
            "aggregate": True,
            "validate": True,
            "training": False,
            "provider_execution": False,
            "locked_test": False,
            "hf_upload": False,
            "commit": False,
            "push": False,
        }
    ):
        raise RuntimeError("runtime corpus-inference authority drifted")
    return policy


def _plan() -> dict[str, Any]:
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-shard-plan-v1",
        "row_count": corpus.CORPUS_ROWS,
        "shard_count": corpus.SHARD_COUNT,
        "batch_size": 8,
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "gpu_job_count": corpus.SHARD_COUNT,
        "maximum_concurrent_l4s": corpus.MAXIMUM_CONCURRENT_L4S,
        "forward_pass_shard_count": corpus.SHARD_COUNT * len(candidate.CHECKPOINT_SPECS),
        "gpu_job_topology": (
            "120-shards-at-most-10-concurrent-each-tokenised-once-and-scored-by-all-six-"
            "checkpoints-sequentially"
        ),
        "launch_completion": "parent-remains-live-and-gathers-all-child-calls",
        "launch_parent_timeout_seconds": LAUNCH_PARENT_TIMEOUT_SECONDS,
        "platform_preemption_recovery_authorised": True,
        "maximum_platform_preemption_recoveries_per_call": 1,
        "maximum_platform_preemption_recoveries_global": (
            corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES
        ),
        "maximum_platform_execution_attempts_per_call": (
            corpus.MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL
        ),
        "execution_attempt_permit_gate": "entry-time-atomic-per-job-queue-partition",
        "execution_attempt_permit_partition_ttl_seconds": 604_800,
        "shards": corpus.shard_plan(),
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "plan_id": candidate.canonical_sha256(body)}
    assert_metadata_only(result, where="corpus-inference shard plan")
    return result


def _require_operational_cost_authority(policy: Mapping[str, Any]) -> None:
    execution = policy.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("provider_budget_gate_required") is not False
        or execution.get("provider_budget_gate_verified") is not False
        or execution.get("proceed_without_provider_hard_cap_authorised") is not True
    ):
        raise RuntimeError(
            "full corpus operational cost authority is absent or drifted"
        )


def _authority(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    bundle = validate_source_bundle(source_bundle)
    policy = _runtime_policy(bundle)
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-inference-authority-v1",
        "source_bundle_id": bundle["source_bundle_id"],
        "policy_sha256": bundle["files"][POLICY_REPO_PATH],
        "eligible_prepared_run_id": candidate.ELIGIBLE_PREPARED_RUN_ID,
        "eligible_source_rows": corpus.SOURCE_ROWS,
        "post_acquisition_corpus_rows": corpus.CORPUS_ROWS,
        "eligible_frame_sha256": candidate.ELIGIBLE_FRAME_SHA256,
        "eligible_projection_sha256": candidate.ELIGIBLE_PROJECTION_SHA256,
        "acquisition_exclusion_rows": candidate.ACQUISITION_ROWS,
        "acquisition_ledger_id": candidate.ACQUISITION_LEDGER_ID,
        "acquisition_ledger_file_sha256": candidate.ACQUISITION_LEDGER_FILE_SHA256,
        "sample_run_id": SAMPLE_RUN_ID,
        "sample_receipt_id": SAMPLE_RECEIPT_ID,
        "teacher_run_id": TEACHER_RUN_ID,
        "teacher_receipt_id": TEACHER_RECEIPT_ID,
        "teacher_receipt_file_sha256": TEACHER_RECEIPT_FILE_SHA256,
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "calibration_run_id": CALIBRATION_RUN_ID,
        "calibration_receipt_id": CALIBRATION_RECEIPT_ID,
        "calibration_id": CALIBRATION_ID,
        "plan_id": _plan()["plan_id"],
        "compute": {
            **corpus.compute_projection(),
            "gpu_type": corpus.GPU_TYPE,
            "full_phase_operational_monitoring_target_usd": policy["compute"][
                "full_phase_operational_monitoring_target_usd"
            ],
            "cumulative_operational_monitoring_target_usd": policy["compute"][
                "cumulative_operational_monitoring_target_usd"
            ],
            "provider_billing_hard_cap_enforced": False,
            "proceed_without_provider_hard_cap_authorised": True,
            "gpu_fallback_allowed": False,
            "retry_authorised": False,
            "maximum_locked_test_rows": 0,
            "maximum_corpus_rows": corpus.CORPUS_ROWS,
        },
        "calibration_members_in_scored_corpus": candidate.CALIBRATION_ROWS,
        "default_label_unseen_analysis_rows": corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS,
        "training_authorised": False,
        "provider_execution_authorised": False,
        "locked_test_authorised": False,
        "locked_test_rows_accessed": 0,
        "corpus_inference_authorised": True,
        "provider_budget_gate_required": False,
        "provider_budget_gate_verified": False,
        "proceed_without_provider_hard_cap_authorised": True,
        "evidence_boundary": (
            "provisional model-assisted predictions for the post-acquisition eligible candidate "
            "corpus; not independent human validation or all Reddit"
        ),
    }
    authority = {**body, "authority_id": candidate.canonical_sha256(body)}
    assert_metadata_only(authority, where="corpus-inference authority")
    return authority


def _run_root(authority_id: str) -> Path:
    if not candidate.is_sha256(authority_id):
        raise ValueError("corpus-inference authority ID is invalid")
    return VOLUME_PATH / OUTPUT_PREFIX / f"run={authority_id}"


def _recovery_queue_name(authority: Mapping[str, Any]) -> str:
    authority_id = str(authority.get("authority_id", ""))
    if not candidate.is_sha256(authority_id):
        raise ValueError("corpus recovery queue authority is invalid")
    return f"reddit-corpus-recovery-{authority_id[:32]}"


def _recovery_queue(authority: Mapping[str, Any], *, create_if_missing: bool) -> modal.Queue:
    return modal.Queue.from_name(
        _recovery_queue_name(authority),
        environment_name=ENVIRONMENT_NAME,
        create_if_missing=create_if_missing,
    )


def _attempt_queue_name(authority_or_job: Mapping[str, Any]) -> str:
    authority_id = str(authority_or_job.get("authority_id", ""))
    if not candidate.is_sha256(authority_id):
        raise ValueError("corpus attempt queue authority is invalid")
    return f"reddit-corpus-attempts-{authority_id[:32]}"


def _attempt_queue(
    authority_or_job: Mapping[str, Any], *, create_if_missing: bool
) -> modal.Queue:
    return modal.Queue.from_name(
        _attempt_queue_name(authority_or_job),
        environment_name=ENVIRONMENT_NAME,
        create_if_missing=create_if_missing,
    )


def _attempt_tokens(job: Mapping[str, Any]) -> list[str]:
    authority_id = str(job.get("authority_id", ""))
    job_id = str(job.get("job_id", ""))
    if not candidate.is_sha256(authority_id) or not candidate.is_sha256(job_id):
        raise ValueError("corpus execution-attempt token binding is invalid")
    return [
        candidate.canonical_sha256(
            {
                "kind": "corpus-platform-execution-attempt-permit-v1",
                "authority_id": authority_id,
                "job_id": job_id,
                "platform_attempt_index": index,
            }
        )
        for index in range(corpus.MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL)
    ]


def _recovery_tokens(authority: Mapping[str, Any]) -> list[str]:
    return [
        candidate.canonical_sha256(
            {
                "kind": "corpus-platform-preemption-recovery-token-v1",
                "authority_id": authority["authority_id"],
                "index": index,
            }
        )
        for index in range(corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES)
    ]


def _validate_upstream() -> dict[str, Any]:
    import pyarrow.parquet as pq

    eligible_path = VOLUME_PATH / ELIGIBLE_RELATIVE_PATH
    prepared_receipt = _read_json(
        VOLUME_PATH / PREPARED_RECEIPT_RELATIVE_PATH, where="prepared eligible-frame receipt"
    )
    prepared_descriptor = prepared_receipt.get("prepared_frame")
    if (
        prepared_receipt.get("prepared_run_id") != candidate.ELIGIBLE_PREPARED_RUN_ID
        or prepared_receipt.get("locked_test_rows_accessed") != 0
        or not isinstance(prepared_descriptor, Mapping)
        or prepared_descriptor.get("row_count") != corpus.SOURCE_ROWS
        or prepared_descriptor.get("sha256") != candidate.ELIGIBLE_FRAME_SHA256
        or not eligible_path.is_file()
        or eligible_path.stat().st_size != ELIGIBLE_FRAME_BYTES
        or _file_sha256(eligible_path) != candidate.ELIGIBLE_FRAME_SHA256
        or pq.ParquetFile(eligible_path).metadata.num_rows != corpus.SOURCE_ROWS
    ):
        raise RuntimeError("corrected eligible-frame binding drifted")
    ledger_path = VOLUME_PATH / ACQUISITION_LEDGER_RELATIVE_PATH
    ledger = _read_json(ledger_path, where="acquisition exclusion ledger")
    if (
        not ledger_path.is_file()
        or _file_sha256(ledger_path) != candidate.ACQUISITION_LEDGER_FILE_SHA256
        or ledger.get("ledger_id") != candidate.ACQUISITION_LEDGER_ID
    ):
        raise RuntimeError("acquisition exclusion ledger drifted")

    sample_root = VOLUME_PATH / candidate.NAMESPACE / f"run={SAMPLE_RUN_ID}"
    sample_receipt = _read_json(sample_root / "receipt.json", where="calibration sample receipt")
    sample_body = {key: value for key, value in sample_receipt.items() if key != "receipt_id"}
    membership_descriptor = sample_receipt.get("membership_artifact")
    if (
        sample_receipt.get("receipt_id") != SAMPLE_RECEIPT_ID
        or candidate.canonical_sha256(sample_body) != SAMPLE_RECEIPT_ID
        or sample_receipt.get("post_acquisition_population_count") != corpus.CORPUS_ROWS
        or sample_receipt.get("sample_count") != candidate.CALIBRATION_ROWS
        or sample_receipt.get("locked_test_access_count") != 0
        or not isinstance(membership_descriptor, Mapping)
    ):
        raise RuntimeError("calibration sample receipt drifted")
    membership_path = _validate_descriptor(
        membership_descriptor,
        where="calibration membership",
        expected_rows=candidate.CALIBRATION_ROWS,
    )

    cohort = _read_json(sample_root / "checkpoint-cohort.json", where="checkpoint cohort")
    if (
        cohort.get("cohort_id") != CHECKPOINT_COHORT_ID
        or candidate.canonical_sha256(
            {key: value for key, value in cohort.items() if key != "cohort_id"}
        )
        != CHECKPOINT_COHORT_ID
        or cohort.get("locked_test_access_count") != 0
        or [member.get("trial_id") for member in cohort.get("members", [])]
        != [spec.trial_id for spec in candidate.CHECKPOINT_SPECS]
    ):
        raise RuntimeError("checkpoint cohort drifted")
    for member, spec in zip(cohort["members"], candidate.CHECKPOINT_SPECS, strict=True):
        descriptor = member.get("checkpoint")
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("relative_path") != candidate.checkpoint_relative_path(spec)
            or descriptor.get("sha256") != spec.checkpoint_sha256
        ):
            raise RuntimeError("checkpoint cohort member drifted")
        path = VOLUME_PATH / str(descriptor["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
            or _file_sha256(path) != spec.checkpoint_sha256
        ):
            raise RuntimeError("checkpoint artefact drifted")

    teacher_path = sample_root / "teacher" / "receipt.json"
    teacher = _read_json(teacher_path, where="calibration teacher receipt")
    if (
        _file_sha256(teacher_path) != TEACHER_RECEIPT_FILE_SHA256
        or candidate.canonical_sha256(teacher) != TEACHER_RECEIPT_ID
        or teacher.get("run_id") != TEACHER_RUN_ID
        or teacher.get("row_count") != candidate.CALIBRATION_ROWS
        or teacher.get("status") != "complete"
        or teacher.get("failed_attempt_count") != 0
        or teacher.get("automatic_retry_count") != 0
        or teacher.get("receipt_contains_raw_text") is not False
        or teacher.get("receipt_contains_row_ids") is not False
        or teacher.get("receipt_contains_row_level_labels") is not False
        or teacher.get("receipt_contains_thread_ids") is not False
    ):
        raise RuntimeError("calibration teacher receipt drifted")

    calibration_root = sample_root / f"calibration={CALIBRATION_RUN_ID}"
    calibration_receipt = _read_json(
        calibration_root / "receipt.json", where="completed calibration receipt"
    )
    calibration_body = {
        key: value for key, value in calibration_receipt.items() if key != "receipt_id"
    }
    calibration_descriptor = calibration_receipt.get("calibration_artifact")
    if (
        calibration_receipt.get("receipt_id") != CALIBRATION_RECEIPT_ID
        or candidate.canonical_sha256(calibration_body) != CALIBRATION_RECEIPT_ID
        or calibration_receipt.get("calibration_run_id") != CALIBRATION_RUN_ID
        or calibration_receipt.get("sample_receipt_id") != SAMPLE_RECEIPT_ID
        or calibration_receipt.get("checkpoint_cohort_id") != CHECKPOINT_COHORT_ID
        or calibration_receipt.get("teacher_run_id") != TEACHER_RUN_ID
        or calibration_receipt.get("calibration_id") != CALIBRATION_ID
        or calibration_receipt.get("locked_test_rows_accessed") != 0
        or calibration_receipt.get("corpus_rows_accessed") != 0
        or not isinstance(calibration_descriptor, Mapping)
    ):
        raise RuntimeError("completed calibration receipt drifted")
    calibration_path = VOLUME_PATH / str(calibration_descriptor["relative_path"])
    if (
        not calibration_path.is_file()
        or calibration_path.stat().st_size != CALIBRATION_FILE_BYTES
        or _file_sha256(calibration_path) != CALIBRATION_FILE_SHA256
    ):
        raise RuntimeError("completed calibration artefact drifted")
    fitted = _read_json(calibration_path, where="completed calibration artefact")
    if (
        fitted.get("calibration_id") != CALIBRATION_ID
        or candidate.canonical_sha256(
            {key: value for key, value in fitted.items() if key != "calibration_id"}
        )
        != CALIBRATION_ID
        or fitted.get("locked_test_rows_accessed") != 0
        or fitted.get("corpus_rows_accessed") != 0
        or fitted.get("stance_estimand_classes") != ["negative", "no_directed_stance", "positive"]
    ):
        raise RuntimeError("completed calibration settings drifted")
    return {
        "eligible_path": eligible_path,
        "ledger": ledger,
        "membership_path": membership_path,
        "cohort": cohort,
        "fitted_calibration": fitted,
    }


def _exclusion_rows(ledger: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for arm in ("probability_arm", "active_arm"):
        value = ledger.get(arm)
        members = value.get("rows") if isinstance(value, Mapping) else None
        if not isinstance(members, list) or len(members) != 1_000:
            raise RuntimeError("acquisition exclusion arm does not contain 1,000 rows")
        for row in members:
            selected = {
                field: row.get(field)
                for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id")
            }
            if any(not isinstance(item, str) or not item for item in selected.values()):
                raise RuntimeError("acquisition exclusion row is invalid")
            rows.append(selected)
    for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id"):
        values = [row[field] for row in rows]
        if len(values) != candidate.ACQUISITION_ROWS or len(set(values)) != len(values):
            raise RuntimeError(f"acquisition exclusions contain duplicate {field}")
    return rows


def _validate_preparation(authority: Mapping[str, Any], *, validate_files: bool) -> dict[str, Any]:
    root = _run_root(str(authority["authority_id"])) / "preparation"
    receipt = _read_json(root / "receipt.json", where="corpus preparation receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    shards = receipt.get("input_shards")
    if (
        receipt.get("receipt_id") != candidate.canonical_sha256(body)
        or receipt.get("kind") != "modernbert-probability-random-corpus-preparation-receipt-v1"
        or receipt.get("authority_id") != authority["authority_id"]
        or receipt.get("plan_id") != authority["plan_id"]
        or receipt.get("eligible_source_rows") != corpus.SOURCE_ROWS
        or receipt.get("acquisition_exclusion_rows") != candidate.ACQUISITION_ROWS
        or receipt.get("corpus_rows") != corpus.CORPUS_ROWS
        or receipt.get("calibration_member_rows") != candidate.CALIBRATION_ROWS
        or receipt.get("label_unseen_rows") != corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS
        or receipt.get("locked_test_rows_accessed") != 0
        or not isinstance(shards, list)
        or len(shards) != corpus.SHARD_COUNT
    ):
        raise RuntimeError("corpus preparation receipt drifted")
    expected = _plan()["shards"]
    observed = [
        {key: row.get(key) for key in ("shard_id", "start", "stop", "row_count")}
        for row in shards
        if isinstance(row, Mapping)
    ]
    if observed != expected:
        raise RuntimeError("corpus preparation shard bounds drifted")
    if validate_files:
        for row in shards:
            _validate_descriptor(
                row["input_artifact"],
                where=f"corpus input shard {row['shard_id']}",
                expected_rows=row["row_count"],
            )
    return receipt


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    timeout=CPU_TIMEOUT_SECONDS,
    volumes={str(VOLUME_PATH): volume},
)
def prepare(source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Freeze and materialise ten exact private shards after both acquisition arms."""

    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    bundle = validate_source_bundle(source_bundle)
    _runtime_policy(bundle)
    authority = _authority(bundle)
    volume.reload()
    root = _run_root(authority["authority_id"])
    if (root / "preparation" / "receipt.json").is_file():
        return _validate_preparation(authority, validate_files=True)
    if root.exists():
        raise FileExistsError("corpus-inference run namespace exists without preparation")
    upstream = _validate_upstream()
    exclusions = _exclusion_rows(upstream["ledger"])
    membership_ids = pq.read_table(upstream["membership_path"], columns=["opaque_id"]).to_pylist()
    membership = [row["opaque_id"] for row in membership_ids]
    if len(membership) != candidate.CALIBRATION_ROWS or len(set(membership)) != len(membership):
        raise RuntimeError("calibration membership identities drifted")
    staging = root.parent / f".run={authority['authority_id']}.preparing"
    if staging.exists():
        raise FileExistsError("stale corpus preparation staging namespace exists")
    preparation = staging / "preparation"
    preparation.mkdir(parents=True)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 8")
        connection.execute("SET memory_limit = '24GB'")
        connection.execute("SET temp_directory = '/tmp/corpus-inference-duckdb'")
        connection.register("excluded", pa.Table.from_pylist(exclusions))
        connection.register(
            "calibration_members",
            pa.Table.from_pylist([{"opaque_id": value} for value in membership]),
        )
        connection.from_parquet(str(upstream["eligible_path"])).create_view("eligible")
        counts = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT record_id), COUNT(DISTINCT thread_id), "
            "COUNT(DISTINCT near_duplicate_cluster_id) FROM eligible"
        ).fetchone()
        if counts != (corpus.SOURCE_ROWS,) * 4:
            raise RuntimeError("eligible corpus identity conservation failed")
        connection.execute(
            "CREATE TEMP TABLE corpus AS "
            "WITH remaining AS ("
            " SELECT e.* FROM eligible e WHERE NOT EXISTS ("
            "  SELECT 1 FROM excluded x WHERE x.opaque_id=e.record_id "
            "  OR x.thread_id=e.thread_id "
            "  OR x.near_duplicate_cluster_id=e.near_duplicate_cluster_id"
            " )"
            "), ordered AS ("
            " SELECT row_number() OVER (ORDER BY record_id) - 1 AS corpus_position, r.*, "
            "        c.opaque_id IS NOT NULL AS calibration_member"
            " FROM remaining r LEFT JOIN calibration_members c ON c.opaque_id=r.record_id"
            ") SELECT * FROM ordered ORDER BY corpus_position"
        )
        remaining = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT record_id), COUNT(DISTINCT thread_id), "
            "COUNT(DISTINCT near_duplicate_cluster_id), "
            "SUM(CASE WHEN calibration_member THEN 1 ELSE 0 END), "
            "MIN(corpus_position), MAX(corpus_position) FROM corpus"
        ).fetchone()
        if remaining != (
            corpus.CORPUS_ROWS,
            corpus.CORPUS_ROWS,
            corpus.CORPUS_ROWS,
            corpus.CORPUS_ROWS,
            candidate.CALIBRATION_ROWS,
            0,
            corpus.CORPUS_ROWS - 1,
        ):
            raise RuntimeError("post-acquisition corpus conservation failed")
        input_shards = []
        all_ids: list[str] = []
        for shard in _plan()["shards"]:
            path = preparation / "input-shards" / f"shard={shard['shard_id']}" / "frame.parquet"
            path.parent.mkdir(parents=True)
            table = connection.execute(
                "SELECT CAST(corpus_position AS BIGINT) AS corpus_position, "
                "record_id AS opaque_id, subreddit, CAST(year AS INTEGER) AS year, "
                "content_type, retrieval_mode, target_text, parent_context, submission_context, "
                "calibration_member FROM corpus WHERE corpus_position >= ? "
                "AND corpus_position < ? ORDER BY corpus_position",
                [shard["start"], shard["stop"]],
            ).fetch_arrow_table()
            pq.write_table(table, path, compression="zstd")
            positions = table["corpus_position"].to_pylist()
            identities = table["opaque_id"].to_pylist()
            if positions != list(range(shard["start"], shard["stop"])):
                raise RuntimeError("prepared corpus shard positions drifted")
            if len(identities) != shard["row_count"] or len(set(identities)) != len(identities):
                raise RuntimeError("prepared corpus shard identities drifted")
            all_ids.extend(identities)
            final_path = (
                root
                / "preparation"
                / "input-shards"
                / f"shard={shard['shard_id']}"
                / "frame.parquet"
            )
            input_shards.append(
                {
                    **shard,
                    "input_artifact": {
                        **_descriptor(path, row_count=shard["row_count"]),
                        "relative_path": final_path.relative_to(VOLUME_PATH).as_posix(),
                    },
                    "private_identity_sequence_sha256": _private_identity_sequence_sha256(
                        identities
                    ),
                }
            )
    finally:
        connection.close()
    if len(all_ids) != corpus.CORPUS_ROWS or len(set(all_ids)) != corpus.CORPUS_ROWS:
        raise RuntimeError("prepared corpus shards do not form an exact identity union")
    receipt_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-preparation-receipt-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": bundle["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "eligible_prepared_run_id": candidate.ELIGIBLE_PREPARED_RUN_ID,
        "eligible_frame_sha256": candidate.ELIGIBLE_FRAME_SHA256,
        "eligible_source_rows": corpus.SOURCE_ROWS,
        "acquisition_ledger_id": candidate.ACQUISITION_LEDGER_ID,
        "acquisition_exclusion_rows": candidate.ACQUISITION_ROWS,
        "corpus_rows": corpus.CORPUS_ROWS,
        "calibration_member_rows": candidate.CALIBRATION_ROWS,
        "label_unseen_rows": corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS,
        "input_shards": input_shards,
        "private_identity_sequence_sha256": _private_identity_sequence_sha256(all_ids),
        "locked_test_rows_accessed": 0,
        "status": "complete",
    }
    receipt = {**receipt_body, "receipt_id": candidate.canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="corpus preparation receipt")
    _write_immutable_json(staging / "authority.json", authority)
    _write_immutable_json(staging / "source-bundle.json", bundle)
    _write_immutable_json(staging / "plan.json", _plan())
    _write_immutable_json(preparation / "receipt.json", receipt)
    os.replace(staging, root)
    volume.commit()
    return receipt


def _load_authority(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    expected = _authority(source_bundle)
    path = _run_root(expected["authority_id"]) / "authority.json"
    observed = _read_json(path, where="corpus-inference authority")
    if observed != expected:
        raise RuntimeError("persisted corpus-inference authority drifted")
    return observed


def _input_shard(
    authority: Mapping[str, Any], shard: Mapping[str, Any], *, validate_content: bool
) -> tuple[dict[str, Any], Path]:
    preparation = _validate_preparation(authority, validate_files=False)
    matches = [
        row
        for row in preparation["input_shards"]
        if isinstance(row, Mapping) and row.get("shard_id") == shard.get("shard_id")
    ]
    if len(matches) != 1 or any(matches[0].get(key) != value for key, value in shard.items()):
        raise RuntimeError("corpus input shard is not authorised by the exact plan")
    descriptor = matches[0].get("input_artifact")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError("corpus input shard descriptor is missing")
    if validate_content:
        path = _validate_descriptor(
            descriptor,
            where=f"corpus input shard {shard['shard_id']}",
            expected_rows=int(shard["row_count"]),
        )
    else:
        relative = Path(str(descriptor.get("relative_path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("corpus input shard path is unsafe")
        path = VOLUME_PATH / relative
    return dict(matches[0]), path


def _spec_by_trial(trial_id: str) -> candidate.CheckpointSpec:
    matches = [spec for spec in candidate.CHECKPOINT_SPECS if spec.trial_id == trial_id]
    if len(matches) != 1:
        raise ValueError("checkpoint trial is outside the frozen cohort")
    return matches[0]


def _checkpoint_payload(path: Path, spec: candidate.CheckpointSpec) -> Any:
    import torch

    if not path.is_file() or _file_sha256(path) != spec.checkpoint_sha256:
        raise RuntimeError("frozen probability-random checkpoint drifted")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "kind": "modernbert-acquisition-checkpoint-v1",
        "experiment_run_id": candidate.ACQUISITION_TRAINING_RUN_ID,
        "phase_run_id": candidate.ACQUISITION_TRAINING_PHASE_ID,
        "trial_id": spec.trial_id,
        "arm": "random",
        "component": spec.component,
        "optimiser_seed": spec.seed,
        "selected_epoch": spec.selected_epoch,
        "training_frame_sha256": candidate.RANDOM_TRAINING_FRAME_SHA256,
        "model_id": "answerdotai/ModernBERT-large",
        "model_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    }
    if (
        not isinstance(payload, Mapping)
        or any(payload.get(key) != value for key, value in expected.items())
        or not isinstance(payload.get("model_state_dict"), Mapping)
    ):
        raise RuntimeError(f"checkpoint payload drifted for {spec.component} seed {spec.seed}")
    return payload


def _tokenise_rows(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from reddit_china_stance.modernbert_factorised_data import render_factorised_text
    from reddit_china_stance.modernbert_factorised_training import MAX_LENGTH

    features: list[dict[str, Any]] = []
    for row in rows:
        opaque_id = row.get("opaque_id")
        if not isinstance(opaque_id, str) or not opaque_id:
            raise ValueError("private corpus input lacks an opaque identity")
        rendered = render_factorised_text(row, separator=tokenizer.sep_token)
        raw = tokenizer(
            rendered,
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
        if len(input_ids) > MAX_LENGTH:
            input_ids = [*input_ids[: MAX_LENGTH - 1], input_ids[-1]]
            attention_mask = [*attention_mask[: MAX_LENGTH - 1], attention_mask[-1]]
        features.append(
            {
                "corpus_position": row["corpus_position"],
                "opaque_id": opaque_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
    return features


def _score_features(
    *,
    model: Any,
    tokenizer: Any,
    features: Sequence[Mapping[str, Any]],
    component: str,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    import torch

    if component not in candidate.COMPONENTS or batch_size != 8:
        raise ValueError("corpus scoring component or batch size drifted")
    result: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            members = features[start : start + batch_size]
            batch = tokenizer.pad(
                [
                    {
                        "input_ids": row["input_ids"],
                        "attention_mask": row["attention_mask"],
                    }
                    for row in members
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
                result.extend(
                    {
                        "corpus_position": row["corpus_position"],
                        "opaque_id": row["opaque_id"],
                        "relevance_logit": float(logit),
                    }
                    for row, logit in zip(members, logits, strict=True)
                )
            else:
                targets = output["target_presence_logits"].detach().float().cpu().tolist()
                stances = output["stance_logits"].detach().float().cpu().tolist()
                result.extend(
                    {
                        "corpus_position": row["corpus_position"],
                        "opaque_id": row["opaque_id"],
                        "target_presence_logits": target,
                        "stance_logits": stance,
                    }
                    for row, target, stance in zip(members, targets, stances, strict=True)
                )
    if (
        len(result) != len(features)
        or [row["corpus_position"] for row in result]
        != [row["corpus_position"] for row in features]
        or [row["opaque_id"] for row in result] != [row["opaque_id"] for row in features]
    ):
        raise RuntimeError("checkpoint scoring did not conserve private input order")
    return result


def _load_model(spec: candidate.CheckpointSpec) -> Any:

    from reddit_china_stance.modernbert_factorised_training import (
        FactorisedOptimisationConfig,
        create_component_model,
    )

    path = VOLUME_PATH / candidate.checkpoint_relative_path(spec)
    payload = _checkpoint_payload(path, spec)
    model = create_component_model(component=spec.component, config=FactorisedOptimisationConfig())
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    return model.to("cuda").eval()


def _dispatch_id(authority: Mapping[str, Any], *, stage: str) -> str:
    if stage not in {"full-scoring", "reducers"}:
        raise ValueError("unknown corpus dispatch stage")
    return candidate.canonical_sha256(
        {
            "kind": f"modernbert-probability-random-corpus-{stage}-dispatch-v1",
            "authority_id": authority["authority_id"],
            "plan_id": authority["plan_id"],
            "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
            "calibration_id": CALIBRATION_ID,
        }
    )


def _score_job(
    authority: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    dispatch_id = _dispatch_id(authority, stage="full-scoring")
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-ensemble-shard-job-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": authority["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": dispatch_id,
        "scoring_shard": dict(shard),
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "ensemble": "unweighted-arithmetic-mean-aligned-raw-logits-all-three-seeds",
        "batch_size": 8,
        "gpu_type": corpus.GPU_TYPE,
        "retry_authorised": False,
        "platform_preemption_recovery_authorised": True,
        "maximum_platform_preemption_recoveries_per_call": 1,
        "maximum_platform_preemption_recoveries_global": (
            corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES
        ),
        "maximum_platform_execution_attempts_per_call": (
            corpus.MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL
        ),
        "execution_attempt_permit_gate": "entry-time-atomic-per-job-queue-partition",
        "locked_test_rows_accessed": 0,
    }
    return {**body, "job_id": candidate.canonical_sha256(body)}


def _validate_score_job(job: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    shard = job.get("scoring_shard")
    if not isinstance(shard, Mapping) or dict(shard) not in _plan()["shards"]:
        raise ValueError("checkpoint scoring job shard is outside the plan")
    expected = _score_job(authority, shard)
    if dict(job) != expected:
        raise RuntimeError("checkpoint scoring job authority drifted")
    return dict(shard)


def _score_root(authority: Mapping[str, Any], shard: Mapping[str, Any]) -> Path:
    return (
        _run_root(str(authority["authority_id"]))
        / "full"
        / "ensemble-shards"
        / f"shard={shard['shard_id']}"
    )


def _validate_score_receipt(
    job: Mapping[str, Any], authority: Mapping[str, Any], *, read_rows: bool
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    import pyarrow.parquet as pq

    shard = _validate_score_job(job, authority)
    root = _score_root(authority, shard)
    receipt = _read_json(root / "receipt.json", where="ensemble corpus shard receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptor = receipt.get("private_logits_artifact")
    expected_keys = {
        "schema_version",
        "kind",
        "authority_id",
        "source_bundle_id",
        "plan_id",
        "dispatch_id",
        "job_id",
        "scoring_shard",
        "checkpoint_cohort_id",
        "checkpoint_count",
        "ensemble",
        "batch_size",
        "gpu_type",
        "platform_preemption_recovery_authorised",
        "maximum_platform_preemption_recoveries_per_call",
        "maximum_platform_preemption_recoveries_global",
        "maximum_platform_execution_attempts_per_call",
        "execution_attempt_permit_gate",
        "precision",
        "row_count",
        "forward_pass_rows",
        "private_identity_sequence_sha256",
        "private_logits_artifact",
        "started_at_unix_seconds",
        "completed_at_unix_seconds",
        "wall_seconds",
        "rows_per_second",
        "estimated_cost_usd",
        "corpus_rows_accessed",
        "locked_test_rows_accessed",
        "status",
        "platform_attempt_index",
        "platform_preemption_recovery_count",
        "receipt_id",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("receipt_id") != candidate.canonical_sha256(body)
        or receipt.get("kind") != "modernbert-probability-random-corpus-ensemble-shard-receipt-v1"
        or any(
            receipt.get(key) != job[key]
            for key in (
                "authority_id",
                "source_bundle_id",
                "plan_id",
                "dispatch_id",
                "job_id",
                "scoring_shard",
                "checkpoint_cohort_id",
                "checkpoint_count",
                "ensemble",
                "batch_size",
                "gpu_type",
                "platform_preemption_recovery_authorised",
                "maximum_platform_preemption_recoveries_per_call",
                "maximum_platform_preemption_recoveries_global",
                "maximum_platform_execution_attempts_per_call",
                "execution_attempt_permit_gate",
            )
        )
        or receipt.get("row_count") != shard["row_count"]
        or receipt.get("forward_pass_rows") != shard["row_count"] * len(candidate.CHECKPOINT_SPECS)
        or receipt.get("corpus_rows_accessed") != shard["row_count"]
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("status") != "complete"
        or receipt.get("platform_attempt_index") not in {0, 1}
        or receipt.get("platform_preemption_recovery_count")
        != receipt.get("platform_attempt_index")
        or not isinstance(descriptor, Mapping)
    ):
        raise RuntimeError("ensemble corpus shard receipt drifted")
    path = _validate_descriptor(
        descriptor,
        where=f"ensemble corpus shard {shard['shard_id']}",
        expected_rows=int(shard["row_count"]),
    )
    rows = pq.read_table(path).to_pylist() if read_rows else None
    if rows is not None:
        positions = [row.get("corpus_position") for row in rows]
        identities = [row.get("opaque_id") for row in rows]
        if (
            positions != list(range(shard["start"], shard["stop"]))
            or _private_identity_sequence_sha256(identities)
            != receipt["private_identity_sequence_sha256"]
        ):
            raise RuntimeError("ensemble corpus shard private rows drifted")
    return receipt, rows


@app.function(
    image=image,
    gpu=corpus.GPU_TYPE,
    cpu=4,
    memory=65_536,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=corpus.MAXIMUM_CONCURRENT_L4S,
    volumes={str(VOLUME_PATH): volume},
)
def score_ensemble_shard(job: dict[str, Any], source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Tokenise once, then score all six frozen checkpoints for one exact shard."""

    attempt_tokens = _attempt_tokens(job)
    attempt_token = _attempt_queue(job, create_if_missing=False).get(
        block=False, partition=job["job_id"]
    )
    if attempt_token not in attempt_tokens:
        raise RuntimeError("per-call platform execution-attempt budget is exhausted")
    platform_attempt_index = attempt_tokens.index(attempt_token)
    recovery_token: str | None = None
    if platform_attempt_index == 1:
        recovery_token = _recovery_queue(job, create_if_missing=False).get(block=False)
        if recovery_token not in _recovery_tokens(job):
            raise RuntimeError("global platform pre-emption recovery budget is exhausted")
    started_at = time.time()

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from reddit_china_stance import modernbert_probability_random_calibration_v1 as calibration

    bundle = validate_source_bundle(source_bundle)
    _runtime_policy(bundle)
    volume.reload()
    authority = _load_authority(bundle)
    shard = _validate_score_job(job, authority)
    root = _score_root(authority, shard)
    if (root / "receipt.json").is_file():
        receipt, _rows = _validate_score_receipt(job, authority, read_rows=False)
        return receipt
    if root.exists():
        raise FileExistsError("ensemble corpus shard namespace is incomplete")
    function_call_id = modal.current_function_call_id()
    if not isinstance(function_call_id, str) or not function_call_id:
        raise RuntimeError("ensemble worker lacks a Modal FunctionCall identity")
    attempts_root = (
        _run_root(authority["authority_id"])
        / "full"
        / "worker-attempts"
        / f"shard={shard['shard_id']}"
    )
    existing_attempts = sorted(attempts_root.glob("attempt=*.json"))
    if any(path.name == f"attempt={platform_attempt_index}.json" for path in existing_attempts):
        raise RuntimeError("ensemble worker execution-attempt marker already exists")
    if platform_attempt_index == 0 and existing_attempts:
        raise RuntimeError("initial ensemble worker execution has unexpected prior markers")
    for path in existing_attempts:
        prior = _read_json(path, where="ensemble worker prior-attempt marker")
        prior_body = {key: value for key, value in prior.items() if key != "attempt_id"}
        if (
            prior.get("attempt_id") != candidate.canonical_sha256(prior_body)
            or prior.get("authority_id") != authority["authority_id"]
            or prior.get("job_id") != job["job_id"]
            or prior.get("shard_id") != shard["shard_id"]
            or prior.get("function_call_id") != function_call_id
            or prior.get("platform_attempt_index") != 0
        ):
            raise RuntimeError("ensemble worker prior execution-attempt marker drifted")
    attempt_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-worker-attempt-v1",
        "authority_id": authority["authority_id"],
        "job_id": job["job_id"],
        "shard_id": shard["shard_id"],
        "function_call_id": function_call_id,
        "platform_attempt_index": platform_attempt_index,
        "platform_preemption_recovery": platform_attempt_index == 1,
        "execution_attempt_permit": attempt_token,
        "recovery_token": recovery_token,
        "started_at_unix_seconds": started_at,
    }
    _write_immutable_json(
        attempts_root / f"attempt={platform_attempt_index}.json",
        {**attempt_body, "attempt_id": candidate.canonical_sha256(attempt_body)},
    )
    volume.commit()
    staging = root.parent / (
        f".shard={shard['shard_id']}.job={job['job_id']}.attempt={platform_attempt_index}"
    )
    if staging.exists():
        raise FileExistsError("ensemble corpus shard attempt already exists")
    staging.mkdir(parents=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("corpus scoring requires the requested L4 with BF16 support")
    input_receipt, input_path = _input_shard(authority, shard, validate_content=True)
    rows = pq.read_table(input_path).to_pylist()
    positions = [row["corpus_position"] for row in rows]
    identities = [row["opaque_id"] for row in rows]
    if (
        positions != list(range(shard["start"], shard["stop"]))
        or _private_identity_sequence_sha256(identities)
        != input_receipt["private_identity_sequence_sha256"]
    ):
        raise RuntimeError("ensemble corpus shard input sequence drifted")
    from reddit_china_stance.modernbert_factorised_training import load_pinned_tokenizer

    tokenizer = load_pinned_tokenizer()
    features = _tokenise_rows(tokenizer, rows)
    per_component: dict[str, list[list[dict[str, Any]]]] = {
        component: [] for component in candidate.COMPONENTS
    }
    for spec in candidate.CHECKPOINT_SPECS:
        model = _load_model(spec)
        scored = _score_features(
            model=model,
            tokenizer=tokenizer,
            features=features,
            component=spec.component,
        )
        per_component[spec.component].append(
            [
                {
                    "item_id": row["opaque_id"],
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"opaque_id", "corpus_position"}
                    },
                }
                for row in scored
            ]
        )
        del model, scored
        torch.cuda.empty_cache()
    relevance = calibration.mean_aligned_logits(per_component["relevance"], component="relevance")
    targets = calibration.mean_aligned_logits(
        per_component["target_stance_b4"], component="target_stance_b4"
    )
    relevance_by_id = {row["item_id"]: row for row in relevance}
    targets_by_id = {row["item_id"]: row for row in targets}
    output = [
        {
            "corpus_position": int(row["corpus_position"]),
            "opaque_id": row["opaque_id"],
            **{
                key: value
                for key, value in relevance_by_id[row["opaque_id"]].items()
                if key != "item_id"
            },
            **{
                key: value
                for key, value in targets_by_id[row["opaque_id"]].items()
                if key != "item_id"
            },
        }
        for row in rows
    ]
    del features, per_component, relevance, targets
    logits_path = staging / "ensemble-logits.parquet"
    pq.write_table(pa.Table.from_pylist(output), logits_path, compression="zstd")
    identities = [row["opaque_id"] for row in output]
    completed_at = time.time()
    wall_seconds = completed_at - started_at
    descriptor = {
        **_descriptor(logits_path, row_count=len(output)),
        "relative_path": (root / "ensemble-logits.parquet").relative_to(VOLUME_PATH).as_posix(),
    }
    receipt_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-ensemble-shard-receipt-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": bundle["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": job["dispatch_id"],
        "job_id": job["job_id"],
        "scoring_shard": dict(shard),
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "ensemble": "unweighted-arithmetic-mean-aligned-raw-logits-all-three-seeds",
        "batch_size": 8,
        "gpu_type": corpus.GPU_TYPE,
        "platform_preemption_recovery_authorised": job[
            "platform_preemption_recovery_authorised"
        ],
        "maximum_platform_preemption_recoveries_per_call": job[
            "maximum_platform_preemption_recoveries_per_call"
        ],
        "maximum_platform_preemption_recoveries_global": job[
            "maximum_platform_preemption_recoveries_global"
        ],
        "maximum_platform_execution_attempts_per_call": job[
            "maximum_platform_execution_attempts_per_call"
        ],
        "execution_attempt_permit_gate": job["execution_attempt_permit_gate"],
        "precision": "bf16-autocast",
        "row_count": len(output),
        "forward_pass_rows": len(output) * len(candidate.CHECKPOINT_SPECS),
        "private_identity_sequence_sha256": _private_identity_sequence_sha256(identities),
        "private_logits_artifact": descriptor,
        "started_at_unix_seconds": started_at,
        "completed_at_unix_seconds": completed_at,
        "wall_seconds": wall_seconds,
        "rows_per_second": len(output) / wall_seconds,
        "estimated_cost_usd": format(
            Decimal(str(wall_seconds)) * corpus.GPU_RATE_USD_PER_SECOND, ".6f"
        ),
        "corpus_rows_accessed": len(output),
        "locked_test_rows_accessed": 0,
        "status": "complete",
        "platform_attempt_index": platform_attempt_index,
        "platform_preemption_recovery_count": platform_attempt_index,
    }
    receipt = {**receipt_body, "receipt_id": candidate.canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="ensemble corpus shard receipt")
    _write_immutable_json(staging / "receipt.json", receipt)
    os.replace(staging, root)
    volume.commit()
    return receipt


def _smoke_root(authority: Mapping[str, Any]) -> Path:
    return _run_root(str(authority["authority_id"])) / "smoke"


def _validate_smoke(authority: Mapping[str, Any]) -> dict[str, Any]:
    root = _smoke_root(authority)
    receipt = _read_json(root / "receipt.json", where="corpus-inference smoke receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptor = receipt.get("private_predictions_artifact")
    expected_keys = {
        "schema_version",
        "kind",
        "authority_id",
        "source_bundle_id",
        "plan_id",
        "checkpoint_cohort_id",
        "calibration_id",
        "row_count",
        "checkpoint_count",
        "forward_pass_rows",
        "batch_size",
        "gpu_type",
        "precision",
        "peak_gpu_bytes",
        "private_identity_sequence_sha256",
        "private_predictions_artifact",
        "started_at_unix_seconds",
        "completed_at_unix_seconds",
        "wall_seconds",
        "rows_per_second",
        "estimated_cost_usd",
        "smoke_phase_operational_monitoring_target_usd",
        "corpus_rows_accessed",
        "locked_test_rows_accessed",
        "status",
        "evidence_boundary",
        "receipt_id",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("receipt_id") != candidate.canonical_sha256(body)
        or receipt.get("kind") != "modernbert-probability-random-corpus-smoke-receipt-v1"
        or receipt.get("authority_id") != authority["authority_id"]
        or receipt.get("plan_id") != authority["plan_id"]
        or receipt.get("checkpoint_cohort_id") != CHECKPOINT_COHORT_ID
        or receipt.get("calibration_id") != CALIBRATION_ID
        or receipt.get("row_count") != corpus.SMOKE_ROWS
        or receipt.get("checkpoint_count") != len(candidate.CHECKPOINT_SPECS)
        or receipt.get("forward_pass_rows") != corpus.SMOKE_ROWS * len(candidate.CHECKPOINT_SPECS)
        or receipt.get("corpus_rows_accessed") != corpus.SMOKE_ROWS
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("status") != "complete"
        or Decimal(str(receipt.get("estimated_cost_usd")))
        > corpus.NEW_SMOKE_COST_MONITORING_TARGET_USD
        or not isinstance(descriptor, Mapping)
    ):
        raise RuntimeError("corpus-inference smoke receipt drifted")
    _validate_descriptor(
        descriptor,
        where="corpus-inference smoke predictions",
        expected_rows=corpus.SMOKE_ROWS,
    )
    return receipt


@app.function(
    image=image,
    gpu=corpus.GPU_TYPE,
    cpu=4,
    memory=36_864,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def smoke(source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Run all six checkpoints and calibrated decoding on the first 128 corpus rows."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from reddit_china_stance import modernbert_probability_random_calibration_v1 as calibration
    from reddit_china_stance.modernbert_factorised_training import load_pinned_tokenizer

    started_at = time.time()
    bundle = validate_source_bundle(source_bundle)
    _runtime_policy(bundle)
    volume.reload()
    authority = _load_authority(bundle)
    _validate_preparation(authority, validate_files=False)
    root = _smoke_root(authority)
    if (root / "receipt.json").is_file():
        return _validate_smoke(authority)
    if root.exists():
        raise FileExistsError("corpus-inference smoke namespace is incomplete")
    staging = root.parent / ".smoke.incomplete"
    if staging.exists():
        raise FileExistsError("stale corpus-inference smoke attempt exists")
    staging.mkdir(parents=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("corpus-inference smoke requires the requested L4 with BF16 support")
    first_shard = _plan()["shards"][0]
    input_receipt, input_path = _input_shard(authority, first_shard, validate_content=True)
    rows = pq.read_table(input_path).slice(0, corpus.SMOKE_ROWS).to_pylist()
    if (
        [row["corpus_position"] for row in rows] != list(range(corpus.SMOKE_ROWS))
        or _private_identity_sequence_sha256([row["opaque_id"] for row in rows])
        != _private_identity_sequence_sha256(
            pq.read_table(input_path, columns=["opaque_id"])
            .slice(0, corpus.SMOKE_ROWS)["opaque_id"]
            .to_pylist()
        )
        or input_receipt["start"] != 0
    ):
        raise RuntimeError("corpus-inference smoke input drifted")
    tokenizer = load_pinned_tokenizer()
    features = _tokenise_rows(tokenizer, rows)
    per_component: dict[str, list[list[dict[str, Any]]]] = {
        component: [] for component in candidate.COMPONENTS
    }
    peak_gpu_bytes = 0
    for spec in candidate.CHECKPOINT_SPECS:
        model = _load_model(spec)
        scored = _score_features(
            model=model,
            tokenizer=tokenizer,
            features=features,
            component=spec.component,
        )
        per_component[spec.component].append(
            [
                {
                    "item_id": row["opaque_id"],
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"opaque_id", "corpus_position"}
                    },
                }
                for row in scored
            ]
        )
        peak_gpu_bytes = max(peak_gpu_bytes, int(torch.cuda.max_memory_allocated()))
        del model, scored
        torch.cuda.empty_cache()
    relevance = calibration.mean_aligned_logits(per_component["relevance"], component="relevance")
    target = calibration.mean_aligned_logits(
        per_component["target_stance_b4"], component="target_stance_b4"
    )
    calibration_path = (
        VOLUME_PATH
        / candidate.NAMESPACE
        / f"run={SAMPLE_RUN_ID}"
        / f"calibration={CALIBRATION_RUN_ID}"
        / "calibration.json"
    )
    if _file_sha256(calibration_path) != CALIBRATION_FILE_SHA256:
        raise RuntimeError("smoke calibration artefact drifted")
    fitted = _read_json(calibration_path, where="smoke calibration artefact")
    relevance_by_id = {row["item_id"]: row for row in relevance}
    target_by_id = {row["item_id"]: row for row in target}
    predictions = [
        corpus.build_private_prediction_row(
            corpus_position=int(row["corpus_position"]),
            opaque_id=row["opaque_id"],
            subreddit=row["subreddit"],
            year=int(row["year"]),
            content_type=row["content_type"],
            retrieval_mode=row["retrieval_mode"],
            calibration_member=bool(row["calibration_member"]),
            logits={
                **relevance_by_id[row["opaque_id"]],
                **target_by_id[row["opaque_id"]],
            },
            fitted_calibration=fitted,
        )
        for row in rows
    ]
    corpus.validate_private_prediction_rows(predictions, start=0, stop=corpus.SMOKE_ROWS)
    predictions_path = staging / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(predictions), predictions_path, compression="zstd")
    completed_at = time.time()
    wall_seconds = completed_at - started_at
    descriptor = {
        **_descriptor(predictions_path, row_count=len(predictions)),
        "relative_path": (root / "predictions.parquet").relative_to(VOLUME_PATH).as_posix(),
    }
    receipt_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-smoke-receipt-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": bundle["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "calibration_id": CALIBRATION_ID,
        "row_count": len(predictions),
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "forward_pass_rows": len(predictions) * len(candidate.CHECKPOINT_SPECS),
        "batch_size": 8,
        "gpu_type": corpus.GPU_TYPE,
        "precision": "bf16-autocast",
        "peak_gpu_bytes": peak_gpu_bytes,
        "private_identity_sequence_sha256": _private_identity_sequence_sha256(
            [row["opaque_id"] for row in predictions]
        ),
        "private_predictions_artifact": descriptor,
        "started_at_unix_seconds": started_at,
        "completed_at_unix_seconds": completed_at,
        "wall_seconds": wall_seconds,
        "rows_per_second": len(predictions) / wall_seconds,
        "estimated_cost_usd": format(
            Decimal(str(wall_seconds)) * corpus.GPU_RATE_USD_PER_SECOND, ".6f"
        ),
        "smoke_phase_operational_monitoring_target_usd": format(
            corpus.NEW_SMOKE_COST_MONITORING_TARGET_USD, ".2f"
        ),
        "corpus_rows_accessed": len(predictions),
        "locked_test_rows_accessed": 0,
        "status": "complete",
        "evidence_boundary": "bounded engineering smoke only; not corpus analysis evidence",
    }
    if (
        Decimal(receipt_body["estimated_cost_usd"])
        > corpus.NEW_SMOKE_COST_MONITORING_TARGET_USD
    ):
        raise RuntimeError("corpus-inference smoke exceeded its operational monitoring target")
    receipt = {**receipt_body, "receipt_id": candidate.canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="corpus-inference smoke receipt")
    _write_immutable_json(staging / "receipt.json", receipt)
    os.replace(staging, root)
    volume.commit()
    return receipt


def _launch_claim(
    authority: Mapping[str, Any], *, stage: str, jobs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    dispatch_id = _dispatch_id(authority, stage=stage)
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": f"modernbert-probability-random-corpus-{stage}-launch-claim-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": authority["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": dispatch_id,
        "job_count": len(jobs),
        "job_ids_sha256": candidate.canonical_sha256([job["job_id"] for job in jobs]),
        "retry_authorised": False,
        "platform_preemption_recovery_authorised": stage == "full-scoring",
        "maximum_platform_preemption_recoveries_global": (
            corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES if stage == "full-scoring" else 0
        ),
        "locked_test_rows_accessed": 0,
    }
    result = {**body, "claim_id": candidate.canonical_sha256(body)}
    assert_metadata_only(result, where=f"corpus {stage} launch claim")
    return result


def _validate_full_completion(
    authority: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    full_root = _run_root(str(authority["authority_id"])) / "full"
    dispatch = _read_json(full_root / "dispatch.json", where="full corpus dispatch receipt")
    completion = _read_json(
        full_root / "completion.json", where="full corpus completion receipt"
    )
    expected_keys = {
        "schema_version",
        "kind",
        "authority_id",
        "plan_id",
        "dispatch_id",
        "dispatch_receipt_id",
        "completed_jobs",
        "platform_preemption_recovery_count",
        "scoring_recovery_receipt_count",
        "remaining_recovery_tokens",
        "remaining_execution_attempt_permits",
        "ensemble_receipt_ids_sha256",
        "locked_test_rows_accessed",
        "status",
        "receipt_id",
    }
    body = {key: value for key, value in completion.items() if key != "receipt_id"}
    recovery_count = completion.get("platform_preemption_recovery_count")
    scoring_recovery_count = completion.get("scoring_recovery_receipt_count")
    if (
        set(completion) != expected_keys
        or completion.get("receipt_id") != candidate.canonical_sha256(body)
        or completion.get("kind")
        != "modernbert-probability-random-corpus-full-scoring-completion-v1"
        or completion.get("authority_id") != authority["authority_id"]
        or completion.get("plan_id") != authority["plan_id"]
        or completion.get("dispatch_id") != _dispatch_id(authority, stage="full-scoring")
        or completion.get("dispatch_receipt_id") != dispatch.get("receipt_id")
        or completion.get("completed_jobs") != len(jobs)
        or type(recovery_count) is not int
        or not 0 <= recovery_count <= corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES
        or type(scoring_recovery_count) is not int
        or not 0 <= scoring_recovery_count <= recovery_count
        or completion.get("remaining_recovery_tokens")
        != corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES - recovery_count
        or completion.get("remaining_execution_attempt_permits")
        != len(jobs) - recovery_count
        or completion.get("locked_test_rows_accessed") != 0
        or completion.get("status") != "complete"
    ):
        raise RuntimeError("full corpus completion receipt drifted")
    return completion


@app.function(
    image=image,
    cpu=2,
    memory=4_096,
    timeout=LAUNCH_PARENT_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def launch_full(source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Persist one immutable 120-job intent, then gather it exactly once."""

    bundle = validate_source_bundle(source_bundle)
    policy = _runtime_policy(bundle)
    volume.reload()
    authority = _load_authority(bundle)
    _validate_preparation(authority, validate_files=True)
    smoke_receipt = _validate_smoke(authority)
    _require_operational_cost_authority(policy)
    jobs = [_score_job(authority, shard) for shard in _plan()["shards"]]
    claim = _launch_claim(authority, stage="full-scoring", jobs=jobs)
    full_root = _run_root(authority["authority_id"]) / "full"
    claim_path = full_root / "launch-claim.json"
    dispatch_path = full_root / "dispatch.json"
    completion_path = full_root / "completion.json"
    if completion_path.is_file():
        return _validate_full_completion(authority, jobs)
    if dispatch_path.is_file():
        recovery_queue = _recovery_queue(authority, create_if_missing=False)
        attempt_queue = _attempt_queue(authority, create_if_missing=False)
        dispatch = _read_json(dispatch_path, where="full corpus dispatch receipt")
        dispatch_body = {key: value for key, value in dispatch.items() if key != "receipt_id"}
        if (
            dispatch.get("receipt_id") != candidate.canonical_sha256(dispatch_body)
            or dispatch.get("dispatch_id") != claim["dispatch_id"]
            or dispatch.get("claim_id") != claim["claim_id"]
            or dispatch.get("submitted_jobs") != len(jobs)
            or dispatch.get("maximum_concurrent_l4s") != corpus.MAXIMUM_CONCURRENT_L4S
            or dispatch.get("recovery_queue_name") != _recovery_queue_name(authority)
            or dispatch.get("execution_attempt_queue_name") != _attempt_queue_name(authority)
            or not isinstance(dispatch.get("function_call_ids"), list)
            or len(dispatch["function_call_ids"]) != len(jobs)
            or len(set(dispatch["function_call_ids"])) != len(jobs)
        ):
            raise RuntimeError("existing full corpus dispatch receipt drifted")
        function_calls = [
            modal.FunctionCall.from_id(function_call_id)
            for function_call_id in dispatch["function_call_ids"]
        ]
    else:
        recovery_queue = _recovery_queue(authority, create_if_missing=True)
        attempt_queue = _attempt_queue(authority, create_if_missing=True)
        if claim_path.exists():
            raise RuntimeError(
                "full corpus launch claim exists without dispatch receipt; "
                "manual reconciliation required"
            )
        if recovery_queue.len() != 0:
            raise RuntimeError("corpus recovery queue is not empty before first dispatch")
        if attempt_queue.len(total=True) != 0:
            raise RuntimeError("corpus execution-attempt queue is not empty before first dispatch")
        recovery_queue.put_many(_recovery_tokens(authority))
        if recovery_queue.len() != corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES:
            raise RuntimeError("corpus recovery queue did not initialise exactly")
        for job in jobs:
            attempt_queue.put_many(
                _attempt_tokens(job), partition=job["job_id"], partition_ttl=604_800
            )
        expected_attempt_permits = (
            len(jobs) * corpus.MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL
        )
        if attempt_queue.len(total=True) != expected_attempt_permits:
            raise RuntimeError("corpus execution-attempt queue did not initialise exactly")
        _write_immutable_json(claim_path, claim)
        volume.commit()
        function_calls = [score_ensemble_shard.spawn(dict(job), bundle) for job in jobs]
        function_call_ids = [call.object_id for call in function_calls]
        dispatch_body = {
            "schema_version": corpus.SCHEMA_VERSION,
            "kind": "modernbert-probability-random-corpus-full-scoring-dispatch-v1",
            "authority_id": authority["authority_id"],
            "plan_id": authority["plan_id"],
            "dispatch_id": claim["dispatch_id"],
            "claim_id": claim["claim_id"],
            "smoke_receipt_id": smoke_receipt["receipt_id"],
            "submitted_jobs": len(jobs),
            "maximum_concurrent_l4s": corpus.MAXIMUM_CONCURRENT_L4S,
            "function_call_ids": function_call_ids,
            "platform_preemption_recovery_authorised": True,
            "maximum_platform_preemption_recoveries_per_call": 1,
            "maximum_platform_preemption_recoveries_global": (
                corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES
            ),
            "recovery_queue_name": _recovery_queue_name(authority),
            "maximum_platform_execution_attempts_per_call": (
                corpus.MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL
            ),
            "execution_attempt_queue_name": _attempt_queue_name(authority),
            "execution_attempt_permit_gate": "entry-time-atomic-per-job-queue-partition",
            "full_phase_operational_monitoring_target_usd": "25.00",
            "retry_authorised": False,
            "locked_test_rows_accessed": 0,
            "status": "submitted",
        }
        dispatch = {**dispatch_body, "receipt_id": candidate.canonical_sha256(dispatch_body)}
        assert_metadata_only(dispatch, where="full corpus scoring dispatch receipt")
        _write_immutable_json(dispatch_path, dispatch)
        volume.commit()
    results: list[dict[str, Any]] = []
    for function_call, job in zip(function_calls, jobs, strict=True):
        try:
            result = function_call.get()
        except Exception:
            volume.reload()
            result, _rows = _validate_score_receipt(job, authority, read_rows=False)
        results.append(result)
    volume.reload()
    validated = [_validate_score_receipt(job, authority, read_rows=False)[0] for job in jobs]
    if [row.get("receipt_id") for row in results] != [row["receipt_id"] for row in validated]:
        raise RuntimeError("gathered ensemble shard results differ from persisted receipts")
    remaining_recovery_tokens = recovery_queue.len()
    remaining_attempt_permits = attempt_queue.len(total=True)
    per_job_remaining_attempt_permits = [
        attempt_queue.len(partition=job["job_id"]) for job in jobs
    ]
    if any(value not in {0, 1} for value in per_job_remaining_attempt_permits):
        raise RuntimeError("per-call execution-attempt accounting drifted")
    recovery_count = sum(1 - value for value in per_job_remaining_attempt_permits)
    scoring_recovery_receipts = sum(
        row["platform_preemption_recovery_count"] for row in validated
    )
    if (
        recovery_count > corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES
        or scoring_recovery_receipts > recovery_count
        or remaining_recovery_tokens
        != corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES - recovery_count
        or remaining_attempt_permits != len(jobs) - recovery_count
        or any(
            receipt["platform_preemption_recovery_count"] > 1 - remaining
            for receipt, remaining in zip(
                validated, per_job_remaining_attempt_permits, strict=True
            )
        )
    ):
        raise RuntimeError("platform execution-attempt or recovery accounting drifted")
    completion_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-full-scoring-completion-v1",
        "authority_id": authority["authority_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": claim["dispatch_id"],
        "dispatch_receipt_id": dispatch["receipt_id"],
        "completed_jobs": len(validated),
        "platform_preemption_recovery_count": recovery_count,
        "scoring_recovery_receipt_count": scoring_recovery_receipts,
        "remaining_recovery_tokens": remaining_recovery_tokens,
        "remaining_execution_attempt_permits": remaining_attempt_permits,
        "ensemble_receipt_ids_sha256": candidate.canonical_sha256(
            [row["receipt_id"] for row in validated]
        ),
        "locked_test_rows_accessed": 0,
        "status": "complete",
    }
    completion = {
        **completion_body,
        "receipt_id": candidate.canonical_sha256(completion_body),
    }
    assert_metadata_only(completion, where="full corpus scoring completion receipt")
    _write_immutable_json(completion_path, completion)
    volume.commit()
    return completion


def _reducer_job(authority: Mapping[str, Any], shard: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-reducer-job-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": authority["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": _dispatch_id(authority, stage="reducers"),
        "scoring_dispatch_id": _dispatch_id(authority, stage="full-scoring"),
        "scoring_shard": dict(shard),
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "calibration_id": CALIBRATION_ID,
        "retry_authorised": False,
        "locked_test_rows_accessed": 0,
    }
    return {**body, "job_id": candidate.canonical_sha256(body)}


def _validate_reducer_job(job: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    shard = job.get("scoring_shard")
    if not isinstance(shard, Mapping) or dict(shard) not in _plan()["shards"]:
        raise ValueError("corpus reducer shard is outside the exact plan")
    expected = _reducer_job(authority, shard)
    if dict(job) != expected:
        raise RuntimeError("corpus reducer job authority drifted")
    return dict(shard)


def _prediction_root(authority: Mapping[str, Any], shard: Mapping[str, Any]) -> Path:
    return (
        _run_root(str(authority["authority_id"]))
        / "full"
        / "prediction-shards"
        / f"shard={shard['shard_id']}"
    )


def _source_score_receipts(
    authority: Mapping[str, Any], shard: Mapping[str, Any], *, read_rows: bool
) -> list[tuple[dict[str, Any], list[dict[str, Any]] | None]]:
    return [_validate_score_receipt(_score_job(authority, shard), authority, read_rows=read_rows)]


def _validate_prediction_receipt(
    job: Mapping[str, Any], authority: Mapping[str, Any], *, read_rows: bool
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    import pyarrow.parquet as pq

    shard = _validate_reducer_job(job, authority)
    root = _prediction_root(authority, shard)
    receipt = _read_json(root / "receipt.json", where="corpus prediction shard receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptor = receipt.get("private_predictions_artifact")
    expected_keys = {
        "schema_version",
        "kind",
        "authority_id",
        "source_bundle_id",
        "plan_id",
        "dispatch_id",
        "scoring_dispatch_id",
        "job_id",
        "scoring_shard",
        "checkpoint_cohort_id",
        "calibration_id",
        "source_score_receipt_count",
        "source_score_receipt_ids_sha256",
        "row_count",
        "calibration_member_rows",
        "label_unseen_rows",
        "private_identity_sequence_sha256",
        "private_predictions_artifact",
        "wall_seconds",
        "corpus_rows_accessed",
        "locked_test_rows_accessed",
        "status",
        "receipt_id",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("receipt_id") != candidate.canonical_sha256(body)
        or receipt.get("kind") != "modernbert-probability-random-corpus-prediction-shard-receipt-v1"
        or any(
            receipt.get(key) != job[key]
            for key in (
                "authority_id",
                "source_bundle_id",
                "plan_id",
                "dispatch_id",
                "scoring_dispatch_id",
                "job_id",
                "scoring_shard",
                "checkpoint_cohort_id",
                "calibration_id",
            )
        )
        or receipt.get("source_score_receipt_count") != 1
        or receipt.get("row_count") != shard["row_count"]
        or not isinstance(receipt.get("calibration_member_rows"), int)
        or not isinstance(receipt.get("label_unseen_rows"), int)
        or int(receipt["calibration_member_rows"]) + int(receipt["label_unseen_rows"])
        != shard["row_count"]
        or receipt.get("corpus_rows_accessed") != shard["row_count"]
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("status") != "complete"
        or not isinstance(descriptor, Mapping)
    ):
        raise RuntimeError("corpus prediction shard receipt drifted")
    source_receipts = _source_score_receipts(authority, shard, read_rows=False)
    source_ids = [value[0]["receipt_id"] for value in source_receipts]
    if candidate.canonical_sha256(source_ids) != receipt["source_score_receipt_ids_sha256"]:
        raise RuntimeError("corpus prediction shard source receipt binding drifted")
    path = _validate_descriptor(
        descriptor,
        where=f"corpus prediction shard {shard['shard_id']}",
        expected_rows=int(shard["row_count"]),
    )
    rows = pq.read_table(path).to_pylist() if read_rows else None
    if rows is not None:
        corpus.validate_private_prediction_rows(
            rows, start=int(shard["start"]), stop=int(shard["stop"])
        )
        if (
            _private_identity_sequence_sha256([row["opaque_id"] for row in rows])
            != receipt["private_identity_sequence_sha256"]
        ):
            raise RuntimeError("corpus prediction shard private identity digest drifted")
    return receipt, rows


@app.function(
    image=image,
    cpu=8,
    memory=49_152,
    timeout=CPU_TIMEOUT_SECONDS,
    max_containers=corpus.MAXIMUM_CONCURRENT_L4S,
    volumes={str(VOLUME_PATH): volume},
)
def reduce_prediction_shard(job: dict[str, Any], source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Calibrate and decode one private pre-ensembled shard."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    bundle = validate_source_bundle(source_bundle)
    _runtime_policy(bundle)
    volume.reload()
    authority = _load_authority(bundle)
    shard = _validate_reducer_job(job, authority)
    root = _prediction_root(authority, shard)
    if (root / "receipt.json").is_file():
        receipt, _rows = _validate_prediction_receipt(job, authority, read_rows=False)
        return receipt
    if root.exists():
        raise FileExistsError("corpus prediction shard namespace is incomplete")
    staging = root.parent / f".shard={shard['shard_id']}.job={job['job_id']}"
    if staging.exists():
        raise FileExistsError("corpus prediction reducer attempt already exists")
    staging.mkdir(parents=True)
    input_receipt, input_path = _input_shard(authority, shard, validate_content=True)
    input_columns = [
        "corpus_position",
        "opaque_id",
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
        "calibration_member",
    ]
    inputs = pq.read_table(input_path, columns=input_columns).to_pylist()
    input_ids = [row["opaque_id"] for row in inputs]
    if [row["corpus_position"] for row in inputs] != list(
        range(shard["start"], shard["stop"])
    ) or _private_identity_sequence_sha256(input_ids) != input_receipt[
        "private_identity_sequence_sha256"
    ]:
        raise RuntimeError("corpus reducer input sequence drifted")
    sources = _source_score_receipts(authority, shard, read_rows=True)
    source_receipt_ids = [receipt["receipt_id"] for receipt, _rows in sources]
    score_rows = sources[0][1]
    if score_rows is None:
        raise RuntimeError("corpus reducer source rows are missing")
    if [row["opaque_id"] for row in score_rows] != input_ids:
        raise RuntimeError("corpus reducer source identities differ from input")
    logits_by_id = {
        row["opaque_id"]: {
            key: value for key, value in row.items() if key not in {"opaque_id", "corpus_position"}
        }
        for row in score_rows
    }
    calibration_path = (
        VOLUME_PATH
        / candidate.NAMESPACE
        / f"run={SAMPLE_RUN_ID}"
        / f"calibration={CALIBRATION_RUN_ID}"
        / "calibration.json"
    )
    if (
        not calibration_path.is_file()
        or calibration_path.stat().st_size != CALIBRATION_FILE_BYTES
        or _file_sha256(calibration_path) != CALIBRATION_FILE_SHA256
    ):
        raise RuntimeError("corpus reducer calibration artefact drifted")
    fitted = _read_json(calibration_path, where="corpus reducer calibration artefact")
    predictions = [
        corpus.build_private_prediction_row(
            corpus_position=int(row["corpus_position"]),
            opaque_id=row["opaque_id"],
            subreddit=row["subreddit"],
            year=int(row["year"]),
            content_type=row["content_type"],
            retrieval_mode=row["retrieval_mode"],
            calibration_member=bool(row["calibration_member"]),
            logits=logits_by_id[row["opaque_id"]],
            fitted_calibration=fitted,
        )
        for row in inputs
    ]
    corpus.validate_private_prediction_rows(
        predictions, start=int(shard["start"]), stop=int(shard["stop"])
    )
    predictions_path = staging / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(predictions), predictions_path, compression="zstd")
    descriptor = {
        **_descriptor(predictions_path, row_count=len(predictions)),
        "relative_path": (root / "predictions.parquet").relative_to(VOLUME_PATH).as_posix(),
    }
    calibration_members = sum(bool(row["calibration_member"]) for row in predictions)
    receipt_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-prediction-shard-receipt-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": bundle["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": job["dispatch_id"],
        "scoring_dispatch_id": job["scoring_dispatch_id"],
        "job_id": job["job_id"],
        "scoring_shard": dict(shard),
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "calibration_id": CALIBRATION_ID,
        "source_score_receipt_count": len(source_receipt_ids),
        "source_score_receipt_ids_sha256": candidate.canonical_sha256(source_receipt_ids),
        "row_count": len(predictions),
        "calibration_member_rows": calibration_members,
        "label_unseen_rows": len(predictions) - calibration_members,
        "private_identity_sequence_sha256": _private_identity_sequence_sha256(
            [row["opaque_id"] for row in predictions]
        ),
        "private_predictions_artifact": descriptor,
        "wall_seconds": time.monotonic() - started,
        "corpus_rows_accessed": len(predictions),
        "locked_test_rows_accessed": 0,
        "status": "complete",
    }
    receipt = {**receipt_body, "receipt_id": candidate.canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="corpus prediction shard receipt")
    _write_immutable_json(staging / "receipt.json", receipt)
    os.replace(staging, root)
    volume.commit()
    return receipt


@app.function(
    image=image,
    cpu=2,
    memory=4_096,
    timeout=LAUNCH_PARENT_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def launch_reducers(source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Dispatch 120 reducers only after all 120 ensemble shards exact-validate."""

    bundle = validate_source_bundle(source_bundle)
    _runtime_policy(bundle)
    volume.reload()
    authority = _load_authority(bundle)
    _validate_smoke(authority)
    for shard in _plan()["shards"]:
        _validate_score_receipt(_score_job(authority, shard), authority, read_rows=False)
    jobs = [_reducer_job(authority, shard) for shard in _plan()["shards"]]
    claim = _launch_claim(authority, stage="reducers", jobs=jobs)
    full_root = _run_root(authority["authority_id"]) / "full"
    claim_path = full_root / "reducer-launch-claim.json"
    dispatch_path = full_root / "reducer-dispatch.json"
    if dispatch_path.is_file():
        dispatch = _read_json(dispatch_path, where="corpus reducer dispatch receipt")
        if (
            dispatch.get("dispatch_id") != claim["dispatch_id"]
            or dispatch.get("claim_id") != claim["claim_id"]
            or dispatch.get("submitted_jobs") != len(jobs)
        ):
            raise RuntimeError("existing corpus reducer dispatch receipt drifted")
        return dispatch
    if claim_path.exists():
        raise RuntimeError(
            "corpus reducer launch claim exists without dispatch receipt; "
            "manual reconciliation required"
        )
    _write_immutable_json(claim_path, claim)
    volume.commit()
    function_calls = [reduce_prediction_shard.spawn(dict(job), bundle) for job in jobs]
    function_call_ids = [call.object_id for call in function_calls]
    dispatch_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-reducer-dispatch-v1",
        "authority_id": authority["authority_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": claim["dispatch_id"],
        "claim_id": claim["claim_id"],
        "submitted_jobs": len(jobs),
        "maximum_concurrent_reducers": corpus.MAXIMUM_CONCURRENT_L4S,
        "function_call_ids": function_call_ids,
        "retry_authorised": False,
        "locked_test_rows_accessed": 0,
        "status": "submitted",
    }
    dispatch = {**dispatch_body, "receipt_id": candidate.canonical_sha256(dispatch_body)}
    assert_metadata_only(dispatch, where="corpus reducer dispatch receipt")
    _write_immutable_json(dispatch_path, dispatch)
    volume.commit()
    results = modal.FunctionCall.gather(*function_calls)
    volume.reload()
    validated = [_validate_prediction_receipt(job, authority, read_rows=False)[0] for job in jobs]
    if [row.get("receipt_id") for row in results] != [row["receipt_id"] for row in validated]:
        raise RuntimeError("gathered prediction shard results differ from persisted receipts")
    completion_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-reducer-completion-v1",
        "authority_id": authority["authority_id"],
        "plan_id": authority["plan_id"],
        "dispatch_id": claim["dispatch_id"],
        "dispatch_receipt_id": dispatch["receipt_id"],
        "completed_jobs": len(validated),
        "prediction_receipt_ids_sha256": candidate.canonical_sha256(
            [row["receipt_id"] for row in validated]
        ),
        "locked_test_rows_accessed": 0,
        "status": "complete",
    }
    completion = {
        **completion_body,
        "receipt_id": candidate.canonical_sha256(completion_body),
    }
    assert_metadata_only(completion, where="corpus reducer completion receipt")
    _write_immutable_json(full_root / "reducer-completion.json", completion)
    volume.commit()
    return completion


def _aggregate_scope(connection: Any, *, where: str) -> dict[str, Any]:
    from reddit_china_stance.modernbert_factorised_data import (
        ANALYTIC_TARGET_CLASSES,
        TARGET_CLASSES,
    )
    from reddit_china_stance.modernbert_probability_random_calibration_v1 import (
        STANCE_ESTIMAND_CLASSES,
    )

    dimensions: dict[str, Any] = {}
    dimension_fields = {
        "overall": (),
        "year": ("year",),
        "subreddit": ("subreddit",),
        "content_type": ("content_type",),
        "year_subreddit": ("year", "subreddit"),
    }
    measures = [
        "COUNT(*) AS row_count",
        "SUM(CASE WHEN relevance='material' THEN 1 ELSE 0 END) AS material_count",
        "SUM(CASE WHEN relevance='not_material' THEN 1 ELSE 0 END) AS not_material_count",
    ]
    for target in TARGET_CLASSES:
        measures.append(
            f"SUM(CASE WHEN target_{target}_present THEN 1 ELSE 0 END) AS target_{target}_count"
        )
        if target in ANALYTIC_TARGET_CLASSES:
            for state in STANCE_ESTIMAND_CLASSES:
                measures.append(
                    f"SUM(CASE WHEN stance_{target}='{state}' THEN 1 ELSE 0 END) "
                    f"AS stance_{target}_{state}_count"
                )
    for name, fields in dimension_fields.items():
        select_fields = [*fields, *measures]
        group = "" if not fields else " GROUP BY " + ", ".join(fields)
        order = "" if not fields else " ORDER BY " + ", ".join(fields)
        cursor = connection.execute(
            "SELECT " + ", ".join(select_fields) + f" FROM corpus WHERE {where}" + group + order
        )
        names = [item[0] for item in cursor.description]
        records = []
        for raw in cursor.fetchall():
            flat = dict(zip(names, raw, strict=True))
            record = {
                "group": {field: flat[field] for field in fields},
                "row_count": int(flat["row_count"]),
                "relevance_counts": {
                    "material": int(flat["material_count"]),
                    "not_material": int(flat["not_material_count"]),
                },
                "target_presence_counts": {
                    target: int(flat[f"target_{target}_count"]) for target in TARGET_CLASSES
                },
                "target_stance_counts": {
                    target: {
                        state: int(flat[f"stance_{target}_{state}_count"])
                        for state in STANCE_ESTIMAND_CLASSES
                    }
                    for target in ANALYTIC_TARGET_CLASSES
                },
            }
            records.append(record)
        dimensions[name] = records
    return {"dimensions": dimensions}


def _validate_final(authority: Mapping[str, Any]) -> dict[str, Any]:
    root = _run_root(str(authority["authority_id"])) / "final"
    receipt = _read_json(root / "receipt.json", where="corpus-inference final receipt")
    full_completion = _validate_full_completion(
        authority, [_score_job(authority, shard) for shard in _plan()["shards"]]
    )
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptor = receipt.get("aggregate_artifact")
    if (
        receipt.get("receipt_id") != candidate.canonical_sha256(body)
        or receipt.get("kind") != "modernbert-probability-random-corpus-inference-receipt-v1"
        or receipt.get("authority_id") != authority["authority_id"]
        or receipt.get("plan_id") != authority["plan_id"]
        or receipt.get("checkpoint_cohort_id") != CHECKPOINT_COHORT_ID
        or receipt.get("calibration_id") != CALIBRATION_ID
        or receipt.get("full_scoring_completion_receipt_id")
        != full_completion["receipt_id"]
        or receipt.get("corpus_rows") != corpus.CORPUS_ROWS
        or receipt.get("calibration_member_rows") != candidate.CALIBRATION_ROWS
        or receipt.get("label_unseen_rows") != corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS
        or receipt.get("ensemble_shard_receipts") != corpus.SHARD_COUNT
        or receipt.get("prediction_shard_receipts") != corpus.SHARD_COUNT
        or receipt.get("locked_test_rows_accessed") != 0
        or not isinstance(receipt.get("platform_preemption_recovery_count"), int)
        or receipt.get("platform_preemption_recovery_count", -1)
        > corpus.MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES
        or receipt.get("provider_billing_hard_cap_enforced") is not False
        or Decimal(str(receipt.get("queue_bounded_cumulative_cost_projection_usd")))
        > corpus.CUMULATIVE_MONITORING_TARGET_USD
        or receipt.get("status") != "complete"
        or not isinstance(descriptor, Mapping)
    ):
        raise RuntimeError("corpus-inference final receipt drifted")
    if set(descriptor) != {"relative_path", "sha256", "bytes"}:
        raise RuntimeError("corpus aggregate descriptor schema drifted")
    aggregate_path = VOLUME_PATH / str(descriptor["relative_path"])
    if (
        not aggregate_path.is_file()
        or aggregate_path.stat().st_size != descriptor["bytes"]
        or _file_sha256(aggregate_path) != descriptor["sha256"]
    ):
        raise RuntimeError("corpus aggregate artefact drifted")
    aggregate = _read_json(aggregate_path, where="corpus aggregate artefact")
    if aggregate.get("aggregate_id") != receipt.get("aggregate_id") or candidate.canonical_sha256(
        {key: value for key, value in aggregate.items() if key != "aggregate_id"}
    ) != aggregate.get("aggregate_id"):
        raise RuntimeError("corpus aggregate identity drifted")
    assert_metadata_only(aggregate, where="corpus aggregate artefact")
    return receipt


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    timeout=CPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def finalise(source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Exact-validate all private shards and publish aggregate-only corpus evidence."""

    import duckdb

    bundle = validate_source_bundle(source_bundle)
    _runtime_policy(bundle)
    volume.reload()
    authority = _load_authority(bundle)
    _validate_preparation(authority, validate_files=True)
    smoke_receipt = _validate_smoke(authority)
    score_jobs = [_score_job(authority, shard) for shard in _plan()["shards"]]
    full_completion = _validate_full_completion(authority, score_jobs)
    final_root = _run_root(authority["authority_id"]) / "final"
    if (final_root / "receipt.json").is_file():
        return _validate_final(authority)
    if final_root.exists():
        raise FileExistsError("corpus finalisation namespace is incomplete")
    score_receipts: list[dict[str, Any]] = []
    prediction_receipts: list[dict[str, Any]] = []
    prediction_paths: list[Path] = []
    input_paths: list[Path] = []
    for shard in _plan()["shards"]:
        input_receipt, input_path = _input_shard(authority, shard, validate_content=True)
        prediction_job = _reducer_job(authority, shard)
        prediction_receipt, rows = _validate_prediction_receipt(
            prediction_job, authority, read_rows=True
        )
        if rows is None:
            raise RuntimeError("corpus finalisation lacks private prediction rows")
        if (
            prediction_receipt["private_identity_sequence_sha256"]
            != input_receipt["private_identity_sequence_sha256"]
        ):
            raise RuntimeError("corpus prediction identity sequence differs from input")
        prediction_receipts.append(prediction_receipt)
        prediction_paths.append(
            VOLUME_PATH / prediction_receipt["private_predictions_artifact"]["relative_path"]
        )
        input_paths.append(input_path)
        score_receipts.extend(
            receipt
            for receipt, _score_rows in _source_score_receipts(authority, shard, read_rows=False)
        )
    if (
        len(score_receipts) != corpus.SHARD_COUNT
        or len({row["receipt_id"] for row in score_receipts}) != len(score_receipts)
        or len(prediction_receipts) != corpus.SHARD_COUNT
        or len({row["receipt_id"] for row in prediction_receipts}) != len(prediction_receipts)
    ):
        raise RuntimeError("corpus finalisation receipt union is incomplete or duplicated")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 8")
        for shard, input_path, prediction_path in zip(
            _plan()["shards"], input_paths, prediction_paths, strict=True
        ):
            mismatch = connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?) i FULL OUTER JOIN read_parquet(?) p "
                "USING(corpus_position, opaque_id) "
                "WHERE i.opaque_id IS NULL OR p.opaque_id IS NULL",
                [str(input_path), str(prediction_path)],
            ).fetchone()[0]
            if mismatch != 0:
                raise RuntimeError(
                    f"corpus prediction shard {shard['shard_id']} differs from its input identities"
                )
        connection.from_parquet([str(path) for path in prediction_paths]).create_view("corpus")
        columns = tuple(row[0] for row in connection.execute("DESCRIBE corpus").fetchall())
        if columns != corpus.private_prediction_columns():
            raise RuntimeError("corpus prediction union column contract drifted")
        counts = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT corpus_position), COUNT(DISTINCT opaque_id), "
            "MIN(corpus_position), MAX(corpus_position), "
            "SUM(CASE WHEN calibration_member THEN 1 ELSE 0 END) FROM corpus"
        ).fetchone()
        if counts != (
            corpus.CORPUS_ROWS,
            corpus.CORPUS_ROWS,
            corpus.CORPUS_ROWS,
            0,
            corpus.CORPUS_ROWS - 1,
            candidate.CALIBRATION_ROWS,
        ):
            raise RuntimeError("corpus prediction union does not exactly conserve the corpus")
        all_scope = _aggregate_scope(connection, where="TRUE")
        unseen_scope = _aggregate_scope(connection, where="NOT calibration_member")
    finally:
        connection.close()
    if (
        all_scope["dimensions"]["overall"][0]["row_count"] != corpus.CORPUS_ROWS
        or unseen_scope["dimensions"]["overall"][0]["row_count"]
        != corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS
    ):
        raise RuntimeError("corpus aggregate scopes do not conserve registered populations")
    aggregate_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-aggregate-v1",
        "authority_id": authority["authority_id"],
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "calibration_id": CALIBRATION_ID,
        "full_scoring_completion_receipt_id": full_completion["receipt_id"],
        "scopes": {
            "all_scored_post_acquisition_corpus": {
                "row_count": corpus.CORPUS_ROWS,
                **all_scope,
            },
            "default_label_unseen_analysis": {
                "row_count": corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS,
                "excludes_calibration_members": True,
                **unseen_scope,
            },
        },
        "stance_estimand_classes": ["negative", "no_directed_stance", "positive"],
        "mixed_in_estimand": False,
        "locked_test_rows_accessed": 0,
        "evidence_boundary": (
            "provisional model-assisted aggregate predictions; not independent human validation, "
            "causal evidence, prevalence truth or all Reddit"
        ),
    }
    aggregate = {
        **aggregate_body,
        "aggregate_id": candidate.canonical_sha256(aggregate_body),
    }
    assert_metadata_only(aggregate, where="corpus aggregate artefact")
    total_gpu_seconds = sum(Decimal(str(row["wall_seconds"])) for row in score_receipts)
    actual_cost = total_gpu_seconds * corpus.GPU_RATE_USD_PER_SECOND
    scoring_recovery_receipt_count = sum(
        int(row["platform_preemption_recovery_count"]) for row in score_receipts
    )
    platform_recovery_count = int(full_completion["platform_preemption_recovery_count"])
    if scoring_recovery_receipt_count > platform_recovery_count:
        raise RuntimeError("completed corpus scoring exceeded its platform recovery budget")
    recovery_gpu_seconds_allowance = Decimal(
        platform_recovery_count * corpus.MAXIMUM_WORKER_SECONDS
    )
    recovery_cost_allowance = (
        recovery_gpu_seconds_allowance * corpus.GPU_RATE_USD_PER_SECOND
    )
    receipted_plus_recovery_allowance_cost = actual_cost + recovery_cost_allowance
    if receipted_plus_recovery_allowance_cost > corpus.FULL_PHASE_MONITORING_TARGET_USD:
        raise RuntimeError(
            "completed corpus scoring exceeded the full-phase operational monitoring target"
        )
    queue_bounded_cumulative_cost_projection = (
        corpus.PRIOR_FAILED_COST_ESTIMATE_USD
        + corpus.PRIOR_SMOKE_COST_USD
        + Decimal(str(smoke_receipt["estimated_cost_usd"]))
        + receipted_plus_recovery_allowance_cost
    )
    if queue_bounded_cumulative_cost_projection > corpus.CUMULATIVE_MONITORING_TARGET_USD:
        raise RuntimeError(
            "Queue-bounded cumulative corpus-inference projection exceeded its monitoring target"
        )
    earliest_start = min(float(row["started_at_unix_seconds"]) for row in score_receipts)
    latest_finish = max(float(row["completed_at_unix_seconds"]) for row in score_receipts)
    inference_wall_seconds = latest_finish - earliest_start
    staging = final_root.parent / ".final.incomplete"
    if staging.exists():
        raise FileExistsError("stale corpus finalisation attempt exists")
    staging.mkdir(parents=True)
    aggregate_path = staging / "aggregate.json"
    _write_immutable_json(aggregate_path, aggregate)
    aggregate_descriptor = {
        "relative_path": (final_root / "aggregate.json").relative_to(VOLUME_PATH).as_posix(),
        "sha256": _file_sha256(aggregate_path),
        "bytes": aggregate_path.stat().st_size,
    }
    prediction_index = [
        {
            "shard_id": row["scoring_shard"]["shard_id"],
            "receipt_id": row["receipt_id"],
            "private_predictions_artifact": row["private_predictions_artifact"],
        }
        for row in prediction_receipts
    ]
    receipt_body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-inference-receipt-v1",
        "authority_id": authority["authority_id"],
        "source_bundle_id": bundle["source_bundle_id"],
        "plan_id": authority["plan_id"],
        "sample_receipt_id": SAMPLE_RECEIPT_ID,
        "teacher_receipt_id": TEACHER_RECEIPT_ID,
        "teacher_receipt_file_sha256": TEACHER_RECEIPT_FILE_SHA256,
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "calibration_run_id": CALIBRATION_RUN_ID,
        "calibration_receipt_id": CALIBRATION_RECEIPT_ID,
        "calibration_id": CALIBRATION_ID,
        "full_scoring_completion_receipt_id": full_completion["receipt_id"],
        "corpus_rows": corpus.CORPUS_ROWS,
        "calibration_member_rows": candidate.CALIBRATION_ROWS,
        "label_unseen_rows": corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS,
        "ensemble_shard_receipts": len(score_receipts),
        "prediction_shard_receipts": len(prediction_receipts),
        "ensemble_receipt_ids_sha256": candidate.canonical_sha256(
            [row["receipt_id"] for row in score_receipts]
        ),
        "prediction_index": prediction_index,
        "aggregate_id": aggregate["aggregate_id"],
        "aggregate_artifact": aggregate_descriptor,
        "successful_receipted_l4_gpu_seconds": format(total_gpu_seconds, ".6f"),
        "successful_receipted_estimated_cost_usd": format(actual_cost, ".6f"),
        "platform_preemption_recovery_count": platform_recovery_count,
        "scoring_recovery_receipt_count": scoring_recovery_receipt_count,
        "recovery_gpu_seconds_allowance": format(recovery_gpu_seconds_allowance, ".6f"),
        "recovery_cost_allowance_usd": format(recovery_cost_allowance, ".6f"),
        "receipted_plus_recovery_allowance_cost_usd": format(
            receipted_plus_recovery_allowance_cost, ".6f"
        ),
        "prior_failed_gpu_seconds_estimate": format(
            corpus.PRIOR_FAILED_GPU_SECONDS_ESTIMATE, ".6f"
        ),
        "prior_failed_cost_estimate_usd": format(
            corpus.PRIOR_FAILED_COST_ESTIMATE_USD, ".6f"
        ),
        "prior_smoke_cost_usd": format(corpus.PRIOR_SMOKE_COST_USD, ".6f"),
        "current_smoke_cost_usd": smoke_receipt["estimated_cost_usd"],
        "queue_bounded_cumulative_cost_projection_usd": format(
            queue_bounded_cumulative_cost_projection, ".6f"
        ),
        "full_phase_operational_monitoring_target_usd": "25.00",
        "cumulative_operational_monitoring_target_usd": "30.00",
        "provider_billing_hard_cap_enforced": False,
        "proceed_without_provider_hard_cap_authorised": True,
        "inference_wall_seconds": inference_wall_seconds,
        "effective_parallel_rows_per_second": corpus.CORPUS_ROWS / inference_wall_seconds,
        "reducer_wall_seconds_sum": sum(float(row["wall_seconds"]) for row in prediction_receipts),
        "locked_test_rows_accessed": 0,
        "status": "complete",
        "evidence_boundary": (
            "provisional model-assisted predictions for the post-acquisition eligible candidate "
            "corpus; default thesis aggregates exclude the 600 calibration members"
        ),
    }
    receipt = {**receipt_body, "receipt_id": candidate.canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="corpus-inference final receipt")
    _write_immutable_json(staging / "receipt.json", receipt)
    os.replace(staging, final_root)
    volume.commit()
    return receipt


@app.function(image=image, cpu=2, memory=4_096, timeout=30 * 60, volumes={str(VOLUME_PATH): volume})
def inspect(source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Return metadata-only durable progress without launching or retrying work."""

    bundle = validate_source_bundle(source_bundle)
    _runtime_policy(bundle)
    volume.reload()
    authority = _authority(bundle)
    root = _run_root(authority["authority_id"])
    if not root.exists():
        return {
            "status": "not_prepared",
            "authority_id": authority["authority_id"],
            "completed_ensemble_shards": 0,
            "expected_ensemble_shards": corpus.SHARD_COUNT,
            "completed_prediction_shards": 0,
            "expected_prediction_shards": corpus.SHARD_COUNT,
            "locked_test_rows_accessed": 0,
        }
    _load_authority(bundle)
    prepared = (root / "preparation" / "receipt.json").is_file()
    smoke_complete = (root / "smoke" / "receipt.json").is_file()
    ensemble_complete = 0
    prediction_complete = 0
    for shard in _plan()["shards"]:
        if (_score_root(authority, shard) / "receipt.json").is_file():
            ensemble_complete += 1
        if (_prediction_root(authority, shard) / "receipt.json").is_file():
            prediction_complete += 1
    final_complete = (root / "final" / "receipt.json").is_file()
    status = (
        "complete"
        if final_complete
        else "reducers_complete"
        if prediction_complete == corpus.SHARD_COUNT
        else "scoring_complete"
        if ensemble_complete == corpus.SHARD_COUNT
        else "in_progress"
        if (root / "full" / "dispatch.json").is_file()
        else "smoke_complete"
        if smoke_complete
        else "prepared"
        if prepared
        else "incomplete"
    )
    result = {
        "status": status,
        "authority_id": authority["authority_id"],
        "prepared": prepared,
        "smoke_complete": smoke_complete,
        "completed_ensemble_shards": ensemble_complete,
        "expected_ensemble_shards": corpus.SHARD_COUNT,
        "completed_prediction_shards": prediction_complete,
        "expected_prediction_shards": corpus.SHARD_COUNT,
        "final_complete": final_complete,
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="corpus-inference progress")
    return result


@app.local_entrypoint()
def main(action: str = "inspect") -> None:
    bundle = build_source_bundle()
    if action == "prepare":
        result = prepare.remote(bundle)
    elif action == "smoke":
        result = smoke.remote(bundle)
    elif action == "launch-full":
        result = launch_full.remote(bundle)
    elif action == "launch-reducers":
        result = launch_reducers.remote(bundle)
    elif action == "finalise":
        result = finalise.remote(bundle)
    elif action == "inspect":
        result = inspect.remote(bundle)
    else:
        raise ValueError(
            "action must be prepare, smoke, launch-full, launch-reducers, finalise or inspect"
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


__all__ = [
    "APP_NAME",
    "REQUIRED_SOURCE_FILES",
    "build_source_bundle",
    "finalise",
    "inspect",
    "launch_full",
    "launch_reducers",
    "prepare",
    "reduce_prediction_shard",
    "score_ensemble_shard",
    "smoke",
    "validate_source_bundle",
]
