"""Modal launcher for the registered single ModernBERT inference candidate.

The command surface is deliberately narrow: prepare one immutable manifest,
preflight its real pinned L4 path, train the sole seed once, inspect metadata or
measure one bounded throughput smoke.  There is no retry, alternate GPU,
locked-test, corpus or sweep action.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_factorised_experiment as factorised_experiment
from reddit_china_stance import modernbert_inference_candidate_v1 as candidate
from reddit_china_stance import semantic_ontology_v2

APP_NAME = "reddit-china-stance-modernbert-inference-candidate-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path(candidate.NAMESPACE)
ALLOWED_GPUS = ("L4",)
MAX_CONCURRENT_CANDIDATES = 1
PAID_ACTIONS = frozenset({"cuda-preflight", "train", "throughput-smoke"})
HARD_MAX_APPROVAL_USD = Decimal("200")
PHASE_MAX_APPROVAL_USD = Decimal("5")


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
SCHEMA_REPO_PATH = "schemas/target-stance-v2-pilot.schema.json"
SCHEMA_RUNTIME_PATH = "/schemas/target-stance-v2-pilot.schema.json"

REQUIRED_SOURCE_FILES = (
    SCHEMA_REPO_PATH,
    "src/reddit_china_stance/modal_modernbert_inference_candidate_v1.py",
    "src/reddit_china_stance/modernbert_inference_candidate_v1.py",
    "src/reddit_china_stance/modernbert_factorised_data.py",
    "src/reddit_china_stance/modernbert_factorised_experiment.py",
    "src/reddit_china_stance/modernbert_factorised_model.py",
    "src/reddit_china_stance/modernbert_factorised_training.py",
    "src/reddit_china_stance/modernbert_model.py",
    "src/reddit_china_stance/modernbert_trainer.py",
    "src/reddit_china_stance/privacy.py",
    "src/reddit_china_stance/semantic_evaluation_v2.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
)

RUNTIME_DEPENDENCIES = {
    "accelerate": "1.10.1",
    "huggingface-hub": "0.36.2",
    "jsonschema": "4.26.0",
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
        f"snapshot_download(repo_id='{candidate.MODEL_ID}', "
        f"revision='{candidate.MODEL_REVISION}')\""
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
        SCHEMA_REPO_PATH,
        remote_path=SCHEMA_RUNTIME_PATH,
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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"existing immutable state differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise RuntimeError(f"stale incomplete immutable state exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _volume_relative(value: Any, *, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty Volume-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{where} must be a safe Volume-relative path")
    return path


def _validate_bound_file(descriptor: Mapping[str, Any], *, volume_root: Path, where: str) -> Path:
    clean = factorised_experiment.artifact_descriptor(descriptor, where=where)
    path = volume_root / _volume_relative(clean["relative_path"], where=where)
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or factorised_experiment.file_sha256(path) != clean["sha256"]
    ):
        raise RuntimeError(f"{where} is missing or corrupt")
    return path


def _validate_runtime_schema(source_bundle: Mapping[str, Any]) -> dict[str, str]:
    bundle = factorised_experiment.validate_source_bundle(source_bundle)
    expected_sha256 = bundle["files"].get(SCHEMA_REPO_PATH)
    if not isinstance(expected_sha256, str):
        raise RuntimeError("candidate source bundle does not bind the runtime schema")
    schema_path = Path(SCHEMA_RUNTIME_PATH)
    if (
        not schema_path.is_file()
        or factorised_experiment.file_sha256(schema_path) != expected_sha256
    ):
        raise RuntimeError("mounted runtime schema is missing or corrupt")
    semantic_ontology_v2.load_v2_label_schema(schema_path)
    return {
        "status": "exact-valid",
        "runtime_path": SCHEMA_RUNTIME_PATH,
        "sha256": expected_sha256,
    }


def enforce_cost_guardrail(*, estimated_cost_usd: Decimal, approved_cost_usd: Decimal) -> None:
    if (
        not approved_cost_usd.is_finite()
        or approved_cost_usd <= 0
        or approved_cost_usd > PHASE_MAX_APPROVAL_USD
    ):
        raise ValueError("approved candidate cost must be finite, positive and <= 5")
    if not estimated_cost_usd.is_finite() or estimated_cost_usd < 0:
        raise ValueError("estimated candidate cost must be finite and non-negative")
    if estimated_cost_usd > approved_cost_usd:
        raise RuntimeError(
            f"estimated candidate cost ${estimated_cost_usd} exceeds approved ${approved_cost_usd}"
        )


def _enforce_manifest_cost(manifest: Mapping[str, Any], approved_cost_usd: str) -> None:
    try:
        approved = Decimal(approved_cost_usd)
    except InvalidOperation as exc:
        raise ValueError("approved_cost_usd must be decimal") from exc
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(str(manifest["reserved_cost_usd"])),
        approved_cost_usd=approved,
    )


def validate_compute_contract(contract: Mapping[str, Any]) -> None:
    clean = candidate.validate_experiment_contract(contract)
    compute = clean["compute"]
    if (
        compute["allowed_gpus"] != ["L4"]
        or compute["max_concurrent_candidates"] != MAX_CONCURRENT_CANDIDATES
        or compute["max_training_attempts"] != 1
        or compute["gpu_fallback_allowed"] is not False
        or compute["retry_authorised"] is not False
    ):
        raise RuntimeError("candidate compute inventory drifted")
    cap = Decimal(compute["hard_cost_cap_usd"])
    measured = Decimal(compute["cumulative_measured_spend_usd"])
    active = Decimal(compute["active_reservation_usd"])
    planned = Decimal(compute["planned_phase_upper_usd"])
    remaining = Decimal(compute["remaining_after_plan_usd"])
    if (
        cap > HARD_MAX_APPROVAL_USD
        or measured + active + planned > cap
        or remaining != cap - measured - active - planned
    ):
        raise RuntimeError("candidate exceeds the shared cost cap")


def validate_preparation_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "training_frame",
        "development_frame",
        "acquisition_labels",
        "acquisition_receipt",
        "acquisition_run_id",
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
        or spec.get("schema_version") != candidate.SCHEMA_VERSION
        or spec.get("kind") != "modernbert-inference-candidate-preparation-spec-v1"
    ):
        raise ValueError("candidate preparation spec schema drifted")
    for key in (
        "training_frame",
        "development_frame",
        "acquisition_labels",
        "acquisition_receipt",
    ):
        factorised_experiment.artifact_descriptor(spec[key], where=key)
    if spec.get("source_files") != list(REQUIRED_SOURCE_FILES):
        raise ValueError("candidate preparation source-file inventory drifted")
    for key in (
        "acquisition_run_id",
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
        raise ValueError("candidate preparation cost fields must be decimal") from exc
    if (
        not rate.is_finite()
        or rate <= 0
        or any(not value.is_finite() or value < 0 for value in (measured, active))
        or planned > PHASE_MAX_APPROVAL_USD
        or cap > HARD_MAX_APPROVAL_USD
        or cap <= 0
        or measured + active + planned > cap
    ):
        raise ValueError("candidate preparation cost contract drifted")
    return json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))


def freeze_prepared_manifest(
    spec: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the exact sources, lockfile, labelled inputs and sole candidate."""

    clean = validate_preparation_spec(spec)
    source_bundle = factorised_experiment.build_source_bundle(
        repo_root,
        clean["source_files"],
    )
    factorised_experiment.validate_source_bundle(source_bundle)
    if (
        candidate.canonical_sha256(source_bundle) != clean["source_bundle_sha256"]
        or candidate.file_sha256(repo_root / "uv.lock") != clean["dependency_lock_sha256"]
    ):
        raise RuntimeError("candidate source bundle or dependency lock drifted")
    contract = candidate.freeze_experiment_contract(
        training_frame=clean["training_frame"],
        development_frame=clean["development_frame"],
        acquisition_labels=clean["acquisition_labels"],
        acquisition_receipt=clean["acquisition_receipt"],
        acquisition_run_id=clean["acquisition_run_id"],
        source_bundle_sha256=clean["source_bundle_sha256"],
        dependency_lock_sha256=clean["dependency_lock_sha256"],
        rate_card_usd_per_gpu_second=clean["rate_card_usd_per_gpu_second"],
        cumulative_measured_spend_usd=clean["cumulative_measured_spend_usd"],
        active_reservation_usd=clean["active_reservation_usd"],
        planned_phase_upper_usd=clean["planned_phase_upper_usd"],
        hard_cost_cap_usd=clean["hard_cost_cap_usd"],
    )
    manifest = candidate.build_run_manifest(contract)
    validate_compute_contract(contract)
    if Decimal(manifest["reserved_cost_usd"]) > Decimal(clean["planned_phase_upper_usd"]):
        raise RuntimeError("candidate reservation exceeds its prepared phase bound")
    return manifest, source_bundle


def _run_root(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return volume_root / OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}"


def _authority_root(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return _run_root(manifest, volume_root=volume_root) / "authority"


def _training_root(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return _run_root(manifest, volume_root=volume_root) / "training"


def _preflight_path(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return _run_root(manifest, volume_root=volume_root) / "cuda-preflight.json"


def _throughput_path(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return _run_root(manifest, volume_root=volume_root) / "throughput-smoke.json"


@app.function(image=image, cpu=2, memory=4_096, volumes={str(VOLUME_PATH): volume})
def publish_manifest(manifest: dict[str, Any], source_bundle: dict[str, Any]) -> dict[str, Any]:
    clean = candidate.validate_run_manifest(manifest)
    factorised_experiment.validate_source_bundle(source_bundle)
    if (
        candidate.canonical_sha256(source_bundle)
        != clean["experiment_contract"]["bindings"]["source_bundle_sha256"]
    ):
        raise RuntimeError("published source bundle differs from candidate binding")
    root = _authority_root(clean)
    _write_immutable_json(root / "source-bundle.json", source_bundle)
    _write_immutable_json(root / "run-manifest.json", clean)
    volume.commit()
    return {
        "status": "published",
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "training_jobs": 1,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }


def prepare_manifest(
    *,
    preparation_spec_path: Path,
    manifest_path: Path,
    source_bundle_path: Path,
    approved_cost_usd: Decimal,
) -> dict[str, Any]:
    if manifest_path.exists() or source_bundle_path.exists():
        raise RuntimeError("refusing to overwrite existing local candidate authority")
    spec = validate_preparation_spec(
        _read_json(preparation_spec_path, where="candidate preparation spec")
    )
    manifest, source_bundle = freeze_prepared_manifest(spec)
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(manifest["reserved_cost_usd"]),
        approved_cost_usd=approved_cost_usd,
    )
    _write_immutable_json(source_bundle_path, source_bundle)
    _write_immutable_json(manifest_path, manifest)
    publication = publish_manifest.remote(manifest, source_bundle)
    return {
        **publication,
        "status": "prepared_and_published",
        "local_manifest_path": str(manifest_path),
        "local_source_bundle_path": str(source_bundle_path),
        "cuda_preflight_required": True,
        "cuda_preflight_executed": False,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=candidate.CUDA_PREFLIGHT_MAX_GPU_SECONDS,
    volumes={str(VOLUME_PATH): volume},
)
def cuda_preflight(
    manifest: dict[str, Any], approved_cost_usd: str, confirmation: str
) -> dict[str, Any]:
    """Exercise one exact labelled row through the real pinned B4 CUDA path."""

    import torch

    clean = candidate.validate_run_manifest(manifest)
    _enforce_manifest_cost(clean, approved_cost_usd)
    _require_phase_confirmation("cuda-preflight", clean, confirmation)
    volume.reload()
    preflight_path = _preflight_path(clean)
    if preflight_path.exists():
        raise RuntimeError("candidate CUDA preflight already has immutable evidence")
    bindings = clean["experiment_contract"]["bindings"]
    _, development_rows, _, training_provenance = candidate.load_provenance_bound_training_data(
        volume_root=VOLUME_PATH,
        bindings=bindings,
    )
    tokenizer = candidate.load_pinned_tokenizer()
    row = development_rows[0]
    feature = candidate.tokenise_factorised_record(
        tokenizer,
        item_id=row["item_id"],
        row=row,
        label=json.loads(row["label_json"]),
    )
    feature = {
        **feature,
        "target_presence_labels": [1, *feature["target_presence_labels"][1:]],
        "target_presence_mask": [1] * len(candidate.TARGET_CLASSES),
        "stance_b4_labels": [
            candidate.MIXED_LOGIT_INDEX,
            *([candidate.IGNORE_INDEX] * (len(candidate.ANALYTIC_TARGET_CLASSES) - 1)),
        ],
        "stance_mask": [1, *([0] * (len(candidate.ANALYTIC_TARGET_CLASSES) - 1))],
    }
    batch = candidate.InferenceCandidateB4Collator(tokenizer)([feature])
    if (
        not bool(batch["target_presence_labels"][0, 0].item())
        or int(batch["stance_labels"][0, 0].item()) != candidate.MIXED_LOGIT_INDEX
        or bool(batch["stance_known_mask"][0, 0].item())
    ):
        raise RuntimeError("candidate mixed-mask/presence boundary drifted")
    model = candidate.create_candidate_model().to("cuda")
    inputs = {key: value.to("cuda") for key, value in batch.items() if key != "item_ids"}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**inputs)
    if (
        not torch.isfinite(output["loss"])
        or tuple(output["target_presence_logits"].shape) != (1, 6)
        or tuple(output["stance_logits"].shape) != (1, 5, 4)
    ):
        raise RuntimeError("candidate CUDA preflight output drifted")
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "modernbert-inference-candidate-cuda-preflight-v1",
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "model_id": candidate.MODEL_ID,
        "model_revision": candidate.MODEL_REVISION,
        "gpu_type": "L4",
        "seed_count": 1,
        "stance_logits_per_target": 4,
        "mixed_stance_ce_mask_checked": True,
        "training_provenance": training_provenance,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": candidate.canonical_sha256(body)}
    _write_immutable_json(preflight_path, receipt)
    volume.commit()
    return receipt


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=36_864,
    timeout=candidate.TRAIN_MAX_GPU_SECONDS,
    volumes={str(VOLUME_PATH): volume},
    max_containers=MAX_CONCURRENT_CANDIDATES,
)
def train_candidate(
    manifest: dict[str, Any], approved_cost_usd: str, confirmation: str
) -> dict[str, Any]:
    clean = candidate.validate_run_manifest(manifest)
    _enforce_manifest_cost(clean, approved_cost_usd)
    _require_phase_confirmation("train", clean, confirmation)
    volume.reload()
    preflight_path = _preflight_path(clean)
    if not preflight_path.is_file():
        raise RuntimeError("candidate CUDA preflight is required before training")
    candidate.validate_cuda_preflight_receipt(
        _read_json(preflight_path, where="candidate CUDA preflight receipt"),
        manifest=clean,
    )
    result = candidate.execute_registered_training(
        clean,
        volume_root=VOLUME_PATH,
        output_root=_training_root(clean),
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=candidate.THROUGHPUT_MAX_GPU_SECONDS,
    volumes={str(VOLUME_PATH): volume},
    max_containers=1,
)
def throughput_smoke(
    manifest: dict[str, Any], approved_cost_usd: str, confirmation: str
) -> dict[str, Any]:
    """Measure the trained real pinned model on 128 development rows."""

    clean = candidate.validate_run_manifest(manifest)
    _enforce_manifest_cost(clean, approved_cost_usd)
    _require_phase_confirmation("throughput-smoke", clean, confirmation)
    volume.reload()
    throughput_path = _throughput_path(clean)
    if throughput_path.exists():
        raise RuntimeError("candidate throughput smoke already has immutable evidence")
    receipt_path = _training_root(clean) / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("candidate training must complete before throughput smoke")
    training_receipt = candidate.validate_training_receipt(
        _read_json(receipt_path, where="candidate training receipt"),
        manifest=clean,
    )
    checkpoint_path = _validate_bound_file(
        training_receipt["checkpoint"],
        volume_root=VOLUME_PATH,
        where="candidate checkpoint",
    )
    calibration_path = _validate_bound_file(
        training_receipt["development_calibration"],
        volume_root=VOLUME_PATH,
        where="candidate development calibration",
    )
    candidate.validate_development_calibration(
        _read_json(calibration_path, where="candidate development calibration"),
        manifest=clean,
    )
    bindings = clean["experiment_contract"]["bindings"]
    development_path = _validate_bound_file(
        bindings["development_frame"],
        volume_root=VOLUME_PATH,
        where="development frame",
    )
    development_rows, _ = candidate.load_private_frame(
        development_path,
        bindings["development_frame"],
        expected_frame="development",
    )
    result = candidate.measure_real_pinned_throughput(
        clean,
        checkpoint_path=checkpoint_path,
        expected_selected_epoch=training_receipt["selected_epoch"],
        rows=development_rows[: candidate.THROUGHPUT_SMOKE_ROWS],
    )
    _write_immutable_json(throughput_path, result)
    volume.commit()
    return result


@app.function(image=image, cpu=2, memory=4_096, volumes={str(VOLUME_PATH): volume})
def inspect_run(manifest: dict[str, Any], source_bundle: dict[str, Any]) -> dict[str, Any]:
    clean = candidate.validate_run_manifest(manifest)
    bundle = factorised_experiment.validate_source_bundle(source_bundle)
    if (
        candidate.canonical_sha256(bundle)
        != clean["experiment_contract"]["bindings"]["source_bundle_sha256"]
    ):
        raise RuntimeError("inspect source bundle does not match candidate authority")
    volume.reload()
    paths = {
        "cuda_preflight": _preflight_path(clean),
        "training_receipt": _training_root(clean) / "receipt.json",
        "throughput_smoke": _throughput_path(clean),
    }
    return {
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "artefacts": {
            name: "present" if path.is_file() else "missing" for name, path in paths.items()
        },
        "runtime_schema": _validate_runtime_schema(bundle),
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }


def _confirmation(action: str, manifest: Mapping[str, Any]) -> str:
    if action not in PAID_ACTIONS:
        raise ValueError(f"{action} is not a registered paid candidate action")
    return f"RUN_INFERENCE_CANDIDATE_{manifest['phase_run_id'][:12]}"


def _require_phase_confirmation(
    action: str, manifest: Mapping[str, Any], confirmation: str
) -> None:
    if action not in PAID_ACTIONS:
        raise ValueError(f"{action} is not a registered paid candidate action")
    required = _confirmation(action, manifest)
    if confirmation != required:
        raise ValueError(f"{action} requires --confirm {required}")


@app.local_entrypoint()
def main(
    action: str,
    manifest_path: str,
    approved_cost_usd: str = "5",
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
            approved_cost_usd=approved,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    manifest = candidate.validate_run_manifest(
        _read_json(manifest_file, where="candidate manifest")
    )
    enforce_cost_guardrail(
        estimated_cost_usd=Decimal(manifest["reserved_cost_usd"]),
        approved_cost_usd=approved,
    )
    if action in PAID_ACTIONS:
        _require_phase_confirmation(action, manifest, confirm)
        if action == "cuda-preflight":
            result = cuda_preflight.remote(
                manifest,
                format(approved, "f"),
                confirm,
            )
        elif action == "train":
            result = train_candidate.remote(
                manifest,
                format(approved, "f"),
                confirm,
            )
        else:
            result = throughput_smoke.remote(
                manifest,
                format(approved, "f"),
                confirm,
            )
    elif action == "inspect":
        source_bundle = _read_json(
            manifest_file.parent / "source-bundle.json",
            where="candidate source bundle",
        )
        result = inspect_run.remote(manifest, source_bundle)
    else:
        raise ValueError(
            "action must be prepare, cuda-preflight, train, throughput-smoke or inspect"
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
