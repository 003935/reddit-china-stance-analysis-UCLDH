"""Modal orchestration for the frozen paired ModernBERT acquisition comparison.

The launcher exposes only preparation, CUDA preflight, the exact twelve L4
trials, metadata-only inspection/validation and one-time closeout.  It has no
B2, calibration, locked-test, corpus-inference, fallback-GPU or retry action.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_acquisition_training as training
from reddit_china_stance import modernbert_factorised_experiment as factorised_experiment
from reddit_china_stance import modernbert_factorised_training as factorised_training
from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-modernbert-acquisition-training-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path(training.NAMESPACE)
POLICY_REMOTE_PATH = Path("/configs/modernbert-acquisition-v1.toml")
MAX_CONCURRENT_TRIALS = 10
ACCOUNT_GPU_LIMIT = 10
ALLOWED_GPUS = ("L4",)
HARD_MAX_APPROVAL_USD = Decimal("200")
PHASE_MAX_APPROVAL_USD = Decimal("25")
PREPARE_TIMEOUT_SECONDS = 1_800
TRAIN_TIMEOUT_SECONDS = training.TRIAL_MAX_GPU_SECONDS
CUDA_PREFLIGHT_KIND = "modernbert-acquisition-training-cuda-preflight-v1"
DISPATCH_KIND = "modernbert-acquisition-training-dispatch-v1"


def _resolve_local_repo_root(source_file: Path) -> Path | None:
    resolved = source_file.resolve()
    return resolved.parents[2] if len(resolved.parents) > 2 else None


REPO_ROOT = _resolve_local_repo_root(Path(__file__))
REQUIRED_SOURCE_FILES = (
    "configs/modernbert-acquisition-v1.toml",
    "schemas/target-stance-v2-pilot.schema.json",
    "src/reddit_china_stance/modal_modernbert_acquisition_training.py",
    "src/reddit_china_stance/modernbert_acquisition_training.py",
    "src/reddit_china_stance/modernbert_acquisition.py",
    "src/reddit_china_stance/modernbert_factorised_data.py",
    "src/reddit_china_stance/modernbert_factorised_experiment.py",
    "src/reddit_china_stance/modernbert_factorised_model.py",
    "src/reddit_china_stance/modernbert_factorised_training.py",
    "src/reddit_china_stance/modernbert_model.py",
    "src/reddit_china_stance/modernbert_trainer.py",
    "src/reddit_china_stance/privacy.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
    "src/reddit_china_stance/sol_teacher_acquisition_v2.py",
)

RUNTIME_DEPENDENCIES = {
    "accelerate": "1.10.1",
    "huggingface-hub": "0.36.2",
    "jsonschema": "4.26.0",
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
    .uv_pip_install(
        *(f"{name}=={version}" for name, version in RUNTIME_DEPENDENCIES.items())
    )
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
    .add_local_file(
        "configs/modernbert-acquisition-v1.toml",
        remote_path=str(POLICY_REMOTE_PATH),
    )
    .add_local_file(
        "schemas/target-stance-v2-pilot.schema.json",
        remote_path="/schemas/target-stance-v2-pilot.schema.json",
    )
)


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"existing immutable state differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _volume_relative(value: Any, *, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty Volume-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{where} must be a safe Volume-relative path")
    return path


def _file_sha256(path: Path) -> str:
    return training.file_sha256(path)


def _validate_bound_file(
    descriptor: Mapping[str, Any], *, volume_root: Path, where: str
) -> Path:
    required = {"relative_path", "sha256", "bytes"}
    if not required <= set(descriptor):
        raise ValueError(f"{where} descriptor is incomplete")
    path = volume_root / _volume_relative(descriptor["relative_path"], where=where)
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != descriptor["sha256"]
    ):
        raise RuntimeError(f"{where} is missing or corrupt")
    return path


def enforce_cost_guardrail(
    *, estimated_cost_usd: Decimal, approved_cost_usd: Decimal
) -> None:
    if (
        not approved_cost_usd.is_finite()
        or approved_cost_usd <= 0
        or approved_cost_usd > PHASE_MAX_APPROVAL_USD
    ):
        raise ValueError("approved phase cost must be finite, positive and <= 25")
    if not estimated_cost_usd.is_finite() or estimated_cost_usd < 0:
        raise ValueError("estimated cost must be finite and non-negative")
    if estimated_cost_usd > approved_cost_usd:
        raise RuntimeError(
            f"estimated acquisition-training cost ${estimated_cost_usd} exceeds "
            f"approved ${approved_cost_usd}"
        )


def validate_compute_contract(contract: Mapping[str, Any]) -> None:
    compute = contract.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("acquisition experiment lacks compute contract")
    if (
        compute.get("allowed_gpus") != ["L4"]
        or compute.get("account_gpu_limit") != ACCOUNT_GPU_LIMIT
        or compute.get("max_concurrent_trials") != MAX_CONCURRENT_TRIALS
        or compute.get("gpu_fallback_allowed") is not False
    ):
        raise ValueError("acquisition compute inventory drifted")
    try:
        cap = Decimal(str(compute["hard_cost_cap_usd"]))
        measured = Decimal(str(compute["cumulative_measured_spend_usd"]))
        active = Decimal(str(compute["active_reservation_usd"]))
        planned = Decimal(str(compute["planned_phase_upper_usd"]))
        remaining = Decimal(str(compute["remaining_after_plan_usd"]))
    except (InvalidOperation, KeyError, ValueError) as exc:
        raise ValueError("acquisition cost binding is invalid") from exc
    if (
        cap > HARD_MAX_APPROVAL_USD
        or measured + active + planned > cap
        or remaining != cap - measured - active - planned
    ):
        raise RuntimeError("acquisition experiment exceeds the shared $200 cap")


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = training.validate_run_manifest(
        _read_json(path, where="acquisition training manifest")
    )
    validate_compute_contract(manifest["experiment_contract"])
    return manifest


def _run_root(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return volume_root / OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}"


def _preflight_path(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> Path:
    return (
        _run_root(manifest, volume_root=volume_root)
        / "cuda-preflight"
        / f"phase={manifest['phase_run_id']}.json"
    )


def _launch_claim_root(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> Path:
    return _run_root(manifest, volume_root=volume_root) / "launch-claims"


def _dispatch_path(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> Path:
    return _launch_claim_root(manifest, volume_root=volume_root) / "dispatch.json"


def _trial_claim(
    manifest: Mapping[str, Any], trial: Mapping[str, Any], *, approved_cost_usd: str
) -> dict[str, Any]:
    body = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": "modernbert-acquisition-training-trial-launch-claim-v1",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "run_manifest_sha256": training.canonical_sha256(manifest),
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": training.canonical_sha256(trial),
        "approved_cost_usd": approved_cost_usd,
        "reserved_cost_usd": manifest["reserved_cost_usd"],
        "max_gpu_seconds": training.TRIAL_MAX_GPU_SECONDS,
        "retry_authorised": False,
        "locked_test_rows_accessed": 0,
    }
    return {**body, "claim_id": training.canonical_sha256(body)}


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"durable launch claim already exists: {path}") from exc


def create_launch_claims(
    manifest: Mapping[str, Any],
    *,
    approved_cost_usd: Decimal,
    volume_root: Path,
) -> dict[str, dict[str, Any]]:
    """Atomically claim one phase, then durably claim every registered trial."""

    clean = training.validate_run_manifest(manifest)
    validate_compute_contract(clean["experiment_contract"])
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(clean["reserved_cost_usd"]),
        approved_cost_usd=approved_cost_usd,
    )
    approved = format(approved_cost_usd, "f")
    root = _launch_claim_root(clean, volume_root=volume_root)
    phase_body = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": "modernbert-acquisition-training-phase-launch-claim-v1",
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "run_manifest_sha256": training.canonical_sha256(clean),
        "approved_cost_usd": approved,
        "reserved_cost_usd": clean["reserved_cost_usd"],
        "expected_trials": training.EXPECTED_TRIALS,
        "retry_authorised": False,
        "locked_test_rows_accessed": 0,
    }
    phase_claim = {
        **phase_body,
        "claim_id": training.canonical_sha256(phase_body),
    }
    _write_exclusive_json(root / "phase.json", phase_claim)
    claims: dict[str, dict[str, Any]] = {}
    for trial in clean["trials"]:
        if trial["max_gpu_seconds"] != TRAIN_TIMEOUT_SECONDS:
            raise RuntimeError("trial reservation and Modal timeout differ")
        claim = _trial_claim(clean, trial, approved_cost_usd=approved)
        _write_exclusive_json(root / f"trial={trial['trial_id']}.json", claim)
        claims[trial["trial_id"]] = claim
    return claims


def _validate_trial_claim(
    manifest: Mapping[str, Any],
    trial: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    volume_root: Path,
) -> dict[str, Any]:
    expected = _trial_claim(
        manifest, trial, approved_cost_usd=str(claim.get("approved_cost_usd"))
    )
    path = _launch_claim_root(manifest, volume_root=volume_root) / (
        f"trial={trial['trial_id']}.json"
    )
    if dict(claim) != expected or _read_json(path, where="trial launch claim") != expected:
        raise RuntimeError("trial launch claim binding drifted")
    return expected


def _dispatch_receipt(
    manifest: Mapping[str, Any], submissions: list[Mapping[str, Any]]
) -> dict[str, Any]:
    clean = training.validate_run_manifest(manifest)
    expected_trial_ids = [trial["trial_id"] for trial in clean["trials"]]
    observed_trial_ids: list[str] = []
    observed_call_ids: list[str] = []
    clean_submissions: list[dict[str, str]] = []
    for submission in submissions:
        if set(submission) != {"trial_id", "function_call_id"}:
            raise ValueError("dispatch submission schema drifted")
        trial_id = submission["trial_id"]
        call_id = submission["function_call_id"]
        if (
            not isinstance(trial_id, str)
            or not isinstance(call_id, str)
            or not call_id.startswith("fc-")
            or len(call_id) <= 3
        ):
            raise ValueError("dispatch submission identity is invalid")
        observed_trial_ids.append(trial_id)
        observed_call_ids.append(call_id)
        clean_submissions.append({"trial_id": trial_id, "function_call_id": call_id})
    if (
        observed_trial_ids != expected_trial_ids
        or len(set(observed_call_ids)) != training.EXPECTED_TRIALS
    ):
        raise RuntimeError("dispatch does not exactly cover the registered trials")
    body = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": DISPATCH_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "run_manifest_sha256": training.canonical_sha256(clean),
        "dispatch_mode": "detached_ephemeral_app",
        "expected_trials": training.EXPECTED_TRIALS,
        "submissions": clean_submissions,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "dispatch_id": training.canonical_sha256(body)}
    assert_metadata_only(receipt, where="acquisition dispatch receipt")
    return receipt


def _validate_dispatch_receipt(
    manifest: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "kind",
        "experiment_run_id",
        "phase_run_id",
        "run_manifest_sha256",
        "dispatch_mode",
        "expected_trials",
        "submissions",
        "locked_test_rows_accessed",
        "dispatch_id",
    }:
        raise ValueError("dispatch receipt schema drifted")
    expected = _dispatch_receipt(manifest, list(value["submissions"]))
    if dict(value) != expected:
        raise RuntimeError("dispatch receipt binding drifted")
    return expected


def validate_preparation_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "base_training_frame",
        "acquisition_source",
        "acquisition_labels",
        "acquisition_ledger",
        "acquisition_teacher_receipt",
        "checkpoint_selection_frame",
        "factorised_parent_manifest",
        "factorised_representation_gate",
        "acquisition_config_sha256",
        "source_files",
        "source_bundle_sha256",
        "dependency_lock_sha256",
        "rate_card_usd_per_gpu_second",
        "cumulative_measured_spend_usd",
        "active_reservation_usd",
        "planned_phase_upper_usd",
        "hard_cost_cap_usd",
    }
    if (
        set(spec) != expected
        or spec.get("schema_version") != training.SCHEMA_VERSION
        or spec.get("kind")
        != "modernbert-acquisition-training-preparation-spec-v1"
    ):
        raise ValueError("acquisition preparation spec schema drifted")
    for key in (
        "base_training_frame",
        "acquisition_source",
        "acquisition_labels",
        "acquisition_ledger",
        "acquisition_teacher_receipt",
        "checkpoint_selection_frame",
        "factorised_parent_manifest",
        "factorised_representation_gate",
    ):
        factorised_experiment.artifact_descriptor(spec[key], where=key)
    if spec.get("source_files") != list(REQUIRED_SOURCE_FILES):
        raise ValueError("acquisition preparation source-file inventory drifted")
    for key in (
        "acquisition_config_sha256",
        "source_bundle_sha256",
        "dependency_lock_sha256",
    ):
        value = spec.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{key} must be a lowercase SHA-256")
    try:
        rate = Decimal(str(spec["rate_card_usd_per_gpu_second"]))
        measured = Decimal(str(spec["cumulative_measured_spend_usd"]))
        active = Decimal(str(spec["active_reservation_usd"]))
        planned = Decimal(str(spec["planned_phase_upper_usd"]))
        cap = Decimal(str(spec["hard_cost_cap_usd"]))
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        raise ValueError("acquisition preparation cost fields must be decimal") from exc
    if (
        not rate.is_finite()
        or rate <= 0
        or any(not value.is_finite() or value < 0 for value in (measured, active))
        or planned != PHASE_MAX_APPROVAL_USD
        or cap != HARD_MAX_APPROVAL_USD
        or measured + active + planned > cap
    ):
        raise ValueError("acquisition preparation cost contract drifted")
    return json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))


def _preparation_id(spec: Mapping[str, Any]) -> str:
    return training.canonical_sha256(validate_preparation_spec(spec))


def _validate_parent_representation_gate(
    parent: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        gate.get("schema_version") != training.SCHEMA_VERSION
        or gate.get("kind") != "modernbert-factorised-representation-gate-v2"
        or gate.get("experiment_run_id") != parent["experiment_run_id"]
        or gate.get("phase_run_id") != parent["phase_run_id"]
        or gate.get("selected_representation") != "B4"
        or gate.get("verdict") not in {"retain_b4", "scrap_b2_keep_b4"}
        or gate.get("development_only") is not True
        or gate.get("calibration_or_threshold_frozen") is not False
        or gate.get("locked_test_rows_accessed") != 0
        or gate.get("human_validation_claim_authorised") is not False
        or gate.get("corpus_inference_authorised") is not False
    ):
        raise RuntimeError("factorised parent gate does not exact-bind retained B4")
    body = {key: value for key, value in gate.items() if key != "gate_receipt_id"}
    if gate.get("gate_receipt_id") != training.canonical_sha256(body):
        raise RuntimeError("factorised parent gate receipt digest drifted")
    return json.loads(json.dumps(gate, sort_keys=True, allow_nan=False))


def _materialise_parent_calibration(
    *, spec: Mapping[str, Any], output_root: Path
) -> tuple[Path, dict[str, Any]]:
    parent_path = _validate_bound_file(
        spec["factorised_parent_manifest"],
        volume_root=VOLUME_PATH,
        where="factorised parent manifest",
    )
    gate_path = _validate_bound_file(
        spec["factorised_representation_gate"],
        volume_root=VOLUME_PATH,
        where="factorised representation gate",
    )
    parent = factorised_experiment.validate_run_manifest(
        _read_json(parent_path, where="factorised parent manifest")
    )
    gate = _validate_parent_representation_gate(
        parent, _read_json(gate_path, where="factorised representation gate")
    )
    bindings = parent["experiment_contract"]["bindings"]
    if (
        factorised_experiment.artifact_descriptor(
            bindings["training_frame"], where="parent training frame"
        )
        != factorised_experiment.artifact_descriptor(
            spec["base_training_frame"], where="base training frame"
        )
        or factorised_experiment.artifact_descriptor(
            bindings["development_frame"], where="parent development frame"
        )
        != factorised_experiment.artifact_descriptor(
            spec["checkpoint_selection_frame"], where="checkpoint selection frame"
        )
    ):
        raise RuntimeError("factorised parent frame bindings drifted")

    def bound(name: str, descriptor: Mapping[str, Any]) -> Path:
        return _validate_bound_file(
            descriptor,
            volume_root=VOLUME_PATH,
            where=f"factorised parent {name}",
        )

    materialised = factorised_training.materialise_private_frames(
        teacher_labels_parquet_path=bound("teacher ledger", bindings["teacher_ledger"]),
        blinded_input_json_path=bound(
            "teacher blinded input", bindings["teacher_blinded_input"]
        ),
        private_mapping_parquet_path=bound(
            "teacher private mapping", bindings["teacher_private_mapping"]
        ),
        source_parquet_path=bound("source parquet", bindings["source_parquet"]),
        membership_json_path=bound("split manifest", bindings["split_manifest"]),
        split_public_manifest_json_path=bound(
            "split public manifest", bindings["split_public_manifest"]
        ),
        bridge_exposure_register_json_path=bound(
            "bridge exposure register", bindings["bridge_exposure_register"]
        ),
        legacy_proxy_parquet_path=bound("legacy proxy", bindings["legacy_proxy"]),
        legacy_proxy_descriptor=bindings["legacy_proxy"],
        source_parquet_descriptor=bindings["source_parquet"],
        output_root=output_root / "factorised-parent-frames",
        descriptor_root=VOLUME_PATH,
        expected_teacher_run_id=bindings["teacher_run_id"],
    )
    calibration = factorised_experiment.artifact_descriptor(
        materialised["calibration"], where="materialised calibration frame"
    )
    if calibration.get("frame") != "calibration" or calibration.get("row_count") != 600:
        raise RuntimeError("factorised parent calibration frame drifted")
    provenance = training.validate_factorised_parent_provenance(
        {
            "experiment_run_id": parent["experiment_run_id"],
            "phase_run_id": parent["phase_run_id"],
            "run_manifest": spec["factorised_parent_manifest"],
            "representation_gate": spec["factorised_representation_gate"],
            "gate_receipt_id": gate["gate_receipt_id"],
            "split_manifest": bindings["split_manifest"],
            "selected_representation": gate["selected_representation"],
            "verdict": gate["verdict"],
        }
    )
    return VOLUME_PATH / calibration["relative_path"], provenance


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    volumes={str(VOLUME_PATH): volume},
    timeout=PREPARE_TIMEOUT_SECONDS,
)
def prepare_frames(spec: dict[str, Any]) -> dict[str, Any]:
    preparation_id = _preparation_id(spec)
    base = _validate_bound_file(
        spec["base_training_frame"], volume_root=VOLUME_PATH, where="base training frame"
    )
    source = _validate_bound_file(
        spec["acquisition_source"], volume_root=VOLUME_PATH, where="acquisition source"
    )
    labels = _validate_bound_file(
        spec["acquisition_labels"], volume_root=VOLUME_PATH, where="acquisition labels"
    )
    ledger = _validate_bound_file(
        spec["acquisition_ledger"], volume_root=VOLUME_PATH, where="acquisition ledger"
    )
    teacher_receipt = _validate_bound_file(
        spec["acquisition_teacher_receipt"],
        volume_root=VOLUME_PATH,
        where="acquisition teacher receipt",
    )
    checkpoint_selection = _validate_bound_file(
        spec["checkpoint_selection_frame"],
        volume_root=VOLUME_PATH,
        where="checkpoint-selection frame",
    )
    output = (
        VOLUME_PATH / OUTPUT_PREFIX / "prepared" / f"input={preparation_id}"
    )
    evaluation, parent_provenance = _materialise_parent_calibration(
        spec=spec, output_root=output
    )
    summary = training.prepare_acquisition_training_frames(
        base_training_path=base,
        acquisition_source_path=source,
        acquisition_labels_path=labels,
        acquisition_ledger_path=ledger,
        acquisition_teacher_receipt_path=teacher_receipt,
        checkpoint_selection_path=checkpoint_selection,
        acquisition_evaluation_path=evaluation,
        acquisition_config_path=POLICY_REMOTE_PATH,
        expected_config_sha256=spec["acquisition_config_sha256"],
        factorised_parent_provenance=parent_provenance,
        output_root=output,
        descriptor_root=VOLUME_PATH,
    )
    receipt_body = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": "modernbert-acquisition-training-preparation-receipt-v1",
        "preparation_spec_sha256": training.canonical_sha256(spec),
        "preparation": summary,
    }
    receipt = {**receipt_body, "receipt_id": training.canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="acquisition preparation receipt")
    _write_immutable_json(output / "receipt.json", receipt)
    volume.commit()
    return receipt


def freeze_prepared_manifest(
    *,
    spec: Mapping[str, Any],
    preparation_receipt: Mapping[str, Any],
    repo_root: Path | None = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze one exact experiment/run manifest from validated preparation evidence."""

    if repo_root is None:
        raise RuntimeError("manifest freezing requires the local repository checkout")
    clean_spec = validate_preparation_spec(spec)
    receipt = training.validate_preparation_receipt(preparation_receipt)
    if receipt["preparation_spec_sha256"] != training.canonical_sha256(clean_spec):
        raise RuntimeError("preparation receipt does not bind the exact input spec")
    source_bundle = factorised_experiment.build_source_bundle(
        repo_root, clean_spec["source_files"]
    )
    factorised_experiment.validate_source_bundle(source_bundle)
    source_bundle_sha256 = training.canonical_sha256(source_bundle)
    dependency_lock_sha256 = _file_sha256(repo_root / "uv.lock")
    policy_path = repo_root / "configs/modernbert-acquisition-v1.toml"
    if (
        source_bundle_sha256 != clean_spec["source_bundle_sha256"]
        or dependency_lock_sha256 != clean_spec["dependency_lock_sha256"]
        or _file_sha256(policy_path) != clean_spec["acquisition_config_sha256"]
    ):
        raise RuntimeError("preparation source bundle, lockfile, or policy digest drifted")
    _, gate_policy = training.load_acquisition_policies(policy_path)
    preparation = receipt["preparation"]
    rare_cells = preparation["validated_teacher_inputs"]["rare_cells"]
    if (
        len(rare_cells) != training.EXPECTED_RARE_CELLS
        or preparation["validated_teacher_inputs"]["rare_cell_list_sha256"]
        != training.canonical_sha256(rare_cells)
    ):
        raise RuntimeError("preparation does not bind the canonical three rare cells")
    contract = training.freeze_experiment_contract(
        preparation=preparation,
        acquisition_config={
            "repo_relative_path": "configs/modernbert-acquisition-v1.toml",
            "sha256": clean_spec["acquisition_config_sha256"],
            "bytes": policy_path.stat().st_size,
        },
        gate_policy=gate_policy,
        rare_cells=rare_cells,
        loss_contribution_counts_by_arm=preparation[
            "loss_contribution_counts_by_arm"
        ],
        source_bundle_sha256=source_bundle_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        rate_card_usd_per_gpu_second=clean_spec[
            "rate_card_usd_per_gpu_second"
        ],
        cumulative_measured_spend_usd=clean_spec[
            "cumulative_measured_spend_usd"
        ],
        active_reservation_usd=clean_spec["active_reservation_usd"],
        planned_phase_upper_usd=clean_spec["planned_phase_upper_usd"],
        hard_cost_cap_usd=clean_spec["hard_cost_cap_usd"],
    )
    manifest = training.build_run_manifest(contract)
    validate_compute_contract(manifest["experiment_contract"])
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(manifest["reserved_cost_usd"]),
        approved_cost_usd=PHASE_MAX_APPROVAL_USD,
    )
    return manifest, source_bundle


@app.function(image=image, cpu=2, memory=4_096, volumes={str(VOLUME_PATH): volume})
def publish_prepared_manifest(
    manifest: dict[str, Any],
    source_bundle: dict[str, Any],
    preparation_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Publish the exact validated preparation and run authority to the Volume."""

    clean = training.validate_run_manifest(manifest)
    validate_compute_contract(clean["experiment_contract"])
    bundle = factorised_experiment.validate_source_bundle(source_bundle)
    receipt = training.validate_preparation_receipt(preparation_receipt)
    bindings = clean["experiment_contract"]["bindings"]
    if (
        bindings["source_bundle_sha256"] != training.canonical_sha256(bundle)
        or bindings["preparation_id"] != receipt["preparation"]["preparation_id"]
        or len(bindings["rare_cells"]) != training.EXPECTED_RARE_CELLS
        or bindings["rare_cells_sha256"]
        != training.canonical_sha256(bindings["rare_cells"])
    ):
        raise RuntimeError("prepared manifest publication binding drifted")
    root = _run_root(clean) / "manifests"
    _write_immutable_json(root / "run-manifest.json", clean)
    _write_immutable_json(root / "source-bundle.json", bundle)
    _write_immutable_json(root / "preparation-receipt.json", receipt)
    volume.commit()
    result = {
        "status": "published",
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "run_manifest_sha256": training.canonical_sha256(clean),
        "source_bundle_sha256": training.canonical_sha256(bundle),
        "preparation_receipt_id": receipt["receipt_id"],
        "expected_trials": training.EXPECTED_TRIALS,
        "rare_cell_count": len(bindings["rare_cells"]),
        "reserved_cost_usd": clean["reserved_cost_usd"],
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition prepared-manifest publication")
    return result


def prepare_manifest(
    *,
    preparation_spec_path: Path,
    manifest_path: Path,
    source_bundle_path: Path,
    preparation_receipt_path: Path,
    approved_cost_usd: Decimal,
) -> dict[str, Any]:
    """Materialise frames remotely, freeze locally, then publish immutable authority."""

    if any(
        path.exists()
        for path in (manifest_path, source_bundle_path, preparation_receipt_path)
    ):
        raise RuntimeError("refusing to overwrite existing acquisition preparation authority")
    spec = validate_preparation_spec(
        _read_json(preparation_spec_path, where="acquisition preparation spec")
    )
    receipt = training.validate_preparation_receipt(prepare_frames.remote(spec))
    manifest, source_bundle = freeze_prepared_manifest(
        spec=spec,
        preparation_receipt=receipt,
    )
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(manifest["reserved_cost_usd"]),
        approved_cost_usd=approved_cost_usd,
    )
    _write_immutable_json(preparation_receipt_path, receipt)
    _write_immutable_json(source_bundle_path, source_bundle)
    _write_immutable_json(manifest_path, manifest)
    publication = publish_prepared_manifest.remote(manifest, source_bundle, receipt)
    return {
        **publication,
        "status": "prepared_and_published",
        "local_manifest_path": str(manifest_path),
        "local_source_bundle_path": str(source_bundle_path),
        "local_preparation_receipt_path": str(preparation_receipt_path),
        "cuda_preflight_required": True,
        "cuda_preflight_executed": False,
    }


def _validate_preflight_receipt(
    manifest: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "schema_version", "kind", "experiment_run_id", "phase_run_id",
        "run_manifest_sha256", "policy_file_sha256", "gpu_type",
        "cuda_available", "components_checked", "checkpoint_selection_rows_checked",
        "acquisition_evaluation_rows_checked", "evaluation_frames_distinct",
        "private_prediction_contract_checked", "publication_contract_checked",
        "invalid_outputs", "locked_test_rows_accessed", "receipt_id",
    }
    if set(value) != expected or value.get("kind") != CUDA_PREFLIGHT_KIND:
        raise ValueError("CUDA preflight receipt schema drifted")
    if (
        value["experiment_run_id"] != manifest["experiment_run_id"]
        or value["phase_run_id"] != manifest["phase_run_id"]
        or value["run_manifest_sha256"] != training.canonical_sha256(manifest)
        or value["policy_file_sha256"]
        != manifest["experiment_contract"]["bindings"]["acquisition_config"]["sha256"]
        or value["gpu_type"] != "L4"
        or value["cuda_available"] is not True
        or value["components_checked"] != list(training.COMPONENTS)
        or value["checkpoint_selection_rows_checked"] != 600
        or value["acquisition_evaluation_rows_checked"] != 600
        or value["evaluation_frames_distinct"] is not True
        or value["private_prediction_contract_checked"] is not True
        or value["publication_contract_checked"] is not True
        or value["invalid_outputs"] != 0
        or value["locked_test_rows_accessed"] != 0
    ):
        raise RuntimeError("CUDA preflight binding or evidence drifted")
    body = {key: value[key] for key in expected - {"receipt_id"}}
    if value["receipt_id"] != training.canonical_sha256(body):
        raise RuntimeError("CUDA preflight receipt digest drifted")
    return dict(value)


def _validate_cuda_model_output(
    output: Any,
    *,
    torch_module: Any,
) -> Any:
    """Return the exact finite scalar loss emitted by a component model."""

    if not isinstance(output, Mapping) or "loss" not in output:
        raise RuntimeError("CUDA preflight model output omitted the loss mapping")
    loss = output["loss"]
    try:
        scalar = loss.numel() == 1
        finite = bool(torch_module.isfinite(loss).all().item())
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise RuntimeError("CUDA preflight produced an invalid loss tensor") from exc
    if not scalar or not finite:
        raise RuntimeError("CUDA preflight produced a non-finite scalar loss")
    return loss


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=3_600,
    volumes={str(VOLUME_PATH): volume},
)
def cuda_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    """Run the exact CUDA/model/frame/publication boundary before paid trials."""

    import torch

    clean = training.validate_run_manifest(manifest)
    policy_digest = _file_sha256(POLICY_REMOTE_PATH)
    if policy_digest != clean["experiment_contract"]["bindings"]["acquisition_config"][
        "sha256"
    ]:
        raise RuntimeError("preflight acquisition config digest drifted")
    checkpoint_descriptor = clean["experiment_contract"]["bindings"][
        "checkpoint_selection_frame"
    ]
    evaluation_descriptor = clean["experiment_contract"]["bindings"][
        "acquisition_evaluation_frame"
    ]
    checkpoint_rows, checkpoint_reference = training.load_private_frame(
        VOLUME_PATH / checkpoint_descriptor["relative_path"],
        checkpoint_descriptor,
        expected_frame="development",
    )
    evaluation_rows, _ = training.load_private_frame(
        VOLUME_PATH / evaluation_descriptor["relative_path"],
        evaluation_descriptor,
        expected_frame="acquisition_evaluation",
    )
    # Exercise real pinned model construction on CUDA for both components.  Full
    # training is deliberately not repeated inside the preflight.
    tokenizer = training.load_pinned_tokenizer()
    first = checkpoint_rows[0]
    feature = training.tokenise_factorised_record(
        tokenizer,
        item_id=first["item_id"],
        row=first,
        label=json.loads(first["label_json"]),
    )
    checked = []
    for component in training.COMPONENTS:
        trial = next(row for row in clean["trials"] if row["component"] == component)
        config = training.build_optimisation_config(trial)
        model = training.create_component_model(component=component, config=config).to("cuda")
        collator = training.FactorisedDynamicPaddingCollator(tokenizer, component=component)
        batch = collator([feature])
        inputs = {
            key: value.to("cuda") if hasattr(value, "to") else value
            for key, value in batch.items()
            if key != "item_ids"
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        _validate_cuda_model_output(output, torch_module=torch)
        checked.append(component)
        del model
        torch.cuda.empty_cache()
    # Round-trip the full private row contracts and scorer shapes without
    # exposing their contents.
    synthetic: dict[str, list[dict[str, Any]]] = {
        "relevance": [
            {"item_id": row["item_id"], "relevance_logit": 0.0}
            for row in evaluation_rows
        ],
        "target_stance_b4": [
            {
                "item_id": row["item_id"],
                "target_presence_logits": [0.0] * 6,
                "stance_logits": [[0.0] * 4 for _ in range(5)],
            }
            for row in evaluation_rows
        ],
    }
    checkpoint_synthetic: dict[str, list[dict[str, Any]]] = {
        "relevance": [
            {"item_id": row["item_id"], "relevance_logit": 0.0}
            for row in checkpoint_rows
        ],
        "target_stance_b4": [
            {
                "item_id": row["item_id"],
                "target_presence_logits": [0.0] * 6,
                "stance_logits": [[0.0] * 4 for _ in range(5)],
            }
            for row in checkpoint_rows
        ],
    }
    for component in training.COMPONENTS:
        trial = next(row for row in clean["trials"] if row["component"] == component)
        training._build_private_predictions(clean, trial, synthetic[component])
        training._score_checkpoint(
            component=component,
            reference=checkpoint_reference,
            rows=checkpoint_synthetic[component],
            selected_epoch=1,
            loss_counts=trial["loss_contribution_counts"],
        )
    body = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": CUDA_PREFLIGHT_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "run_manifest_sha256": training.canonical_sha256(clean),
        "policy_file_sha256": policy_digest,
        "gpu_type": "L4",
        "cuda_available": True,
        "components_checked": checked,
        "checkpoint_selection_rows_checked": len(checkpoint_rows),
        "acquisition_evaluation_rows_checked": len(evaluation_rows),
        "evaluation_frames_distinct": (
            checkpoint_descriptor["sha256"] != evaluation_descriptor["sha256"]
        ),
        "private_prediction_contract_checked": True,
        "publication_contract_checked": True,
        "invalid_outputs": 0,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": training.canonical_sha256(body)}
    _validate_preflight_receipt(clean, receipt)
    _write_immutable_json(_preflight_path(clean), receipt)
    volume.commit()
    return receipt


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=36_864,
    timeout=TRAIN_TIMEOUT_SECONDS,
    volumes={str(VOLUME_PATH): volume},
    max_containers=MAX_CONCURRENT_TRIALS,
)
def train_l4(job: dict[str, Any]) -> Mapping[str, Any]:
    if set(job) != {"training_job", "launch_claim"}:
        raise ValueError("claimed acquisition training job schema drifted")
    volume.reload()
    manifest, trial = training._validate_job(job["training_job"])
    _validate_trial_claim(
        manifest,
        trial,
        job["launch_claim"],
        volume_root=VOLUME_PATH,
    )
    if trial["max_gpu_seconds"] != TRAIN_TIMEOUT_SECONDS:
        raise RuntimeError("registered GPU seconds differ from Modal timeout")
    result = training.run_training_trial(
        job=job["training_job"],
        volume_root=VOLUME_PATH,
        trial_executor=training.execute_registered_gpu_trial,
    )
    volume.commit()
    return result


@app.function(image=image, cpu=2, memory=4_096, volumes={str(VOLUME_PATH): volume})
def coordinate_training(
    manifest: dict[str, Any], approved_cost_usd: str
) -> dict[str, Any]:
    clean = training.validate_run_manifest(manifest)
    try:
        approved = Decimal(approved_cost_usd)
    except InvalidOperation as exc:
        raise ValueError("approved_cost_usd must be decimal") from exc
    validate_compute_contract(clean["experiment_contract"])
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(clean["reserved_cost_usd"]),
        approved_cost_usd=approved,
    )
    volume.reload()
    _validate_preflight_receipt(
        clean, _read_json(_preflight_path(clean), where="CUDA preflight receipt")
    )
    for trial in clean["trials"]:
        inspection = training.inspect_trial_output(
            manifest=clean, trial_id=trial["trial_id"], volume_root=VOLUME_PATH
        )
        if inspection["status"] != "missing":
            raise RuntimeError("registered trial already has output or attempt evidence")
    claims = create_launch_claims(
        clean, approved_cost_usd=approved, volume_root=VOLUME_PATH
    )
    volume.commit()
    submissions: list[dict[str, str]] = []
    for trial in clean["trials"]:
        call = train_l4.spawn(
            {
                "training_job": training.build_trial_job(clean, trial),
                "launch_claim": claims[trial["trial_id"]],
            }
        )
        submissions.append(
            {"trial_id": trial["trial_id"], "function_call_id": call.object_id}
        )
    dispatch = _dispatch_receipt(clean, submissions)
    _write_exclusive_json(_dispatch_path(clean), dispatch)
    volume.commit()
    return {
        "submitted_trials": len(submissions),
        "expected_trials": training.EXPECTED_TRIALS,
        "dispatch_id": dispatch["dispatch_id"],
        "dispatch_mode": dispatch["dispatch_mode"],
        "submitted_function_call_ids": [
            submission["function_call_id"] for submission in submissions
        ],
        "locked_test_rows_accessed": 0,
    }


@app.function(image=image, cpu=2, memory=4_096, volumes={str(VOLUME_PATH): volume})
def inspect_run(manifest: dict[str, Any]) -> dict[str, Any]:
    clean = training.validate_run_manifest(manifest)
    volume.reload()
    inspections = [
        training.inspect_trial_output(
            manifest=clean, trial_id=trial["trial_id"], volume_root=VOLUME_PATH
        )
        for trial in clean["trials"]
    ]
    result = training.aggregate_trial_inspections(manifest=clean, inspections=inspections)
    dispatch_path = _dispatch_path(clean)
    if dispatch_path.exists():
        dispatch = _validate_dispatch_receipt(
            clean, _read_json(dispatch_path, where="dispatch receipt")
        )
        result.update(
            {
                "dispatch_id": dispatch["dispatch_id"],
                "dispatch_mode": dispatch["dispatch_mode"],
                "submitted_trials": len(dispatch["submissions"]),
                "submitted_function_call_ids": [
                    submission["function_call_id"]
                    for submission in dispatch["submissions"]
                ],
            }
        )
    else:
        result["submitted_trials"] = 0
        result["submitted_function_call_ids"] = []
    assert_metadata_only(result, where="acquisition run dispatch inspection")
    return result


@app.function(image=image, cpu=4, memory=16_384, volumes={str(VOLUME_PATH): volume})
def closeout(manifest: dict[str, Any]) -> dict[str, Any]:
    clean = training.validate_run_manifest(manifest)
    volume.reload()
    inspections = [
        training.inspect_trial_output(
            manifest=clean, trial_id=trial["trial_id"], volume_root=VOLUME_PATH
        )
        for trial in clean["trials"]
    ]
    aggregate = training.aggregate_trial_inspections(
        manifest=clean, inspections=inspections
    )
    if aggregate["status"] != "complete":
        raise RuntimeError("cannot close out before all twelve trials validate")
    result = training.closeout_comparison(
        manifest=clean,
        volume_root=VOLUME_PATH,
        policy_path=POLICY_REMOTE_PATH,
    )
    volume.commit()
    return result


def _confirmation(action: str, manifest: Mapping[str, Any]) -> str:
    return f"{action.upper()}_MODERNBERT_ACQUISITION_{manifest['phase_run_id'][:12]}"


@app.local_entrypoint()
def main(
    action: str,
    manifest_path: str,
    approved_cost_usd: str = "25",
    confirm: str = "",
    preparation_spec_path: str = "",
) -> None:
    manifest_file = Path(manifest_path)
    try:
        approved = Decimal(approved_cost_usd)
    except InvalidOperation as exc:
        raise ValueError("approved_cost_usd must be decimal") from exc
    if action == "prepare":
        if not preparation_spec_path:
            raise ValueError("prepare requires preparation_spec_path")
        result = prepare_manifest(
            preparation_spec_path=Path(preparation_spec_path),
            manifest_path=manifest_file,
            source_bundle_path=manifest_file.parent / "source-bundle.json",
            preparation_receipt_path=(
                manifest_file.parent / "preparation-receipt.json"
            ),
            approved_cost_usd=approved,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    manifest = _load_manifest(manifest_file)
    reserved = Decimal(manifest["reserved_cost_usd"])
    enforce_cost_guardrail(estimated_cost_usd=reserved, approved_cost_usd=approved)
    if action == "cuda-preflight":
        result = cuda_preflight.remote(manifest)
    elif action == "launch":
        required = _confirmation(action, manifest)
        if confirm != required:
            raise ValueError(f"launch requires --confirm {required}")
        result = coordinate_training.remote(manifest, format(approved, "f"))
    elif action == "inspect":
        result = inspect_run.remote(manifest)
    elif action == "validate-sharded":
        result = inspect_run.remote(manifest)
        if result["status"] != "complete":
            raise RuntimeError("sharded validation is incomplete")
    elif action == "closeout":
        required = _confirmation(action, manifest)
        if confirm != required:
            raise ValueError(f"closeout requires --confirm {required}")
        result = closeout.remote(manifest)
    else:
        raise ValueError(
            "action must be prepare, cuda-preflight, launch, inspect, "
            "validate-sharded or closeout"
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
