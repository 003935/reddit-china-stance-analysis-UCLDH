"""Modal orchestration for the fixed factorised ModernBERT-v2 comparison.

The launcher can prepare fresh-v2 frames, validate contracts, execute one CUDA and
real-development scoring gate, launch exactly nine registered L4 trials, inspect/validate immutable
outputs, and close out the development-only B2/B4 comparison.  It has no legacy
frame, locked-test, calibration, corpus-inference, replacement-GPU or implicit
resubmission authority.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import modal

APP_NAME = "reddit-china-stance-modernbert-factorised-v2"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path("student-modernbert-factorised-v2")
CUDA_PREFLIGHT_RECEIPT_KIND = "modernbert-factorised-cuda-preflight-v4"

PREPARATION_IDENTITY_FIELDS = (
    "teacher_run_id",
    "teacher_receipt",
    "teacher_ledger",
    "teacher_blinded_input",
    "teacher_private_mapping",
    "source_parquet",
    "source_receipt",
    "source_metadata_mapping",
    "bridge_exposure_register",
    "legacy_proxy",
    "bridge_authorisation",
    "rubric",
    "schema",
    "teacher_receipt_relative_path",
    "teacher_labels_parquet_relative_path",
    "blinded_input_json_relative_path",
    "private_mapping_parquet_relative_path",
    "source_parquet_relative_path",
    "source_receipt_relative_path",
    "source_metadata_mapping_relative_path",
    "bridge_exposure_register_json_relative_path",
    "bridge_receipt_relative_path",
    "legacy_proxy_parquet_relative_path",
)

MAX_CONCURRENT_TRIALS = 9
ACCOUNT_GPU_LIMIT = 10
HARD_MAX_APPROVAL_USD = Decimal("200")
ALLOWED_GPUS = ("L4",)
TRAIN_TIMEOUT_SECONDS = 18_000

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

REQUIRED_SOURCE_FILES = (
    "src/reddit_china_stance/modal_modernbert_factorised.py",
    "src/reddit_china_stance/modernbert_factorised_data.py",
    "src/reddit_china_stance/modernbert_factorised_experiment.py",
    "src/reddit_china_stance/modernbert_factorised_model.py",
    "src/reddit_china_stance/modernbert_factorised_splits.py",
    "src/reddit_china_stance/modernbert_factorised_training.py",
    "src/reddit_china_stance/prepare_modernbert_factorised_inputs.py",
    "src/reddit_china_stance/semantic_evaluation_v2.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
    "src/reddit_china_stance/privacy.py",
)
RUNTIME_FILE_MOUNTS = {
    "docs/rubrics/target-stance-v2-pilot.md": "/docs/rubrics/target-stance-v2-pilot.md",
    "schemas/target-stance-v2-pilot.schema.json": "/schemas/target-stance-v2-pilot.schema.json",
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
    .add_local_file(
        "schemas/target-stance-v2-pilot.schema.json",
        remote_path=RUNTIME_FILE_MOUNTS["schemas/target-stance-v2-pilot.schema.json"],
    )
    .add_local_file(
        "docs/rubrics/target-stance-v2-pilot.md",
        remote_path=RUNTIME_FILE_MOUNTS["docs/rubrics/target-stance-v2-pilot.md"],
    )
)


def _experiment_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_factorised_experiment")


def _training_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_factorised_training")


def _require_callable(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        raise RuntimeError(f"factorised implementation lacks required callable {name}")
    return value


def _require_sha256(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
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
        raise RuntimeError(
            f"estimated factorised cost ${estimated_cost_usd} exceeds approved "
            f"${approved_cost_usd}"
        )


def validate_compute_contract(contract: Mapping[str, Any]) -> None:
    compute = contract.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("factorised experiment contract lacks compute binding")
    expected_keys = {
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
    if set(compute) != expected_keys:
        raise ValueError("factorised compute binding schema drifted")
    expected = {
        "allowed_gpus": list(ALLOWED_GPUS),
        "account_gpu_limit": ACCOUNT_GPU_LIMIT,
        "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
        "gpu_fallback_allowed": False,
    }
    if any(compute.get(key) != value for key, value in expected.items()):
        raise ValueError("factorised compute or concurrency binding drifted")
    if MAX_CONCURRENT_TRIALS > ACCOUNT_GPU_LIMIT:
        raise ValueError("factorised concurrency exceeds the account GPU limit")
    rates = compute.get("rate_card_usd_per_gpu_second")
    if not isinstance(rates, Mapping) or set(rates) != {"L4"}:
        raise ValueError("factorised rate card binding drifted")
    try:
        rate = Decimal(str(rates["L4"]))
        estimate = Decimal(str(compute["planned_phase_upper_usd"]))
        cap = Decimal(str(compute["hard_cost_cap_usd"]))
        measured = Decimal(str(compute["cumulative_measured_spend_usd"]))
        active = Decimal(str(compute["active_reservation_usd"]))
        remaining = Decimal(str(compute["remaining_after_plan_usd"]))
    except (InvalidOperation, KeyError, ValueError) as exc:
        raise ValueError("factorised cost binding is invalid") from exc
    if not rate.is_finite() or rate <= 0:
        raise ValueError("factorised GPU rate must be positive and finite")
    if any(
        not amount.is_finite() or amount < 0
        for amount in (measured, active, estimate, remaining)
    ) or not cap.is_finite() or cap <= 0:
        raise ValueError(
            "factorised costs must be finite and non-negative with a positive cap"
        )
    committed = measured + active
    if cap > HARD_MAX_APPROVAL_USD or committed + estimate > cap:
        raise RuntimeError("factorised experiment exceeds the frozen total cost cap")
    if remaining != cap - committed - estimate:
        raise ValueError("factorised remaining cost binding drifted")


def verify_frozen_source_bundle(*, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    """Verify listed files only; unrelated new repository files are tolerated."""

    contract = manifest.get("experiment_contract")
    bindings = contract.get("bindings") if isinstance(contract, Mapping) else None
    if not isinstance(bindings, Mapping):
        raise ValueError("factorised manifest lacks source bindings")
    bundle = bindings.get("source_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("factorised manifest lacks listed source bundle")
    clean = _require_callable(_experiment_module(), "validate_source_bundle")(bundle)
    if bindings.get("source_bundle_sha256") != _canonical_sha256(clean):
        raise RuntimeError("factorised source bundle content address drifted")
    files = clean["files"]
    if not set(REQUIRED_SOURCE_FILES) <= set(files):
        raise RuntimeError("factorised source bundle omits required implementation files")
    root = _repo_root()
    for relative, expected_hash in files.items():
        candidate = root / relative
        if not candidate.is_file() or _file_sha256(candidate) != _require_sha256(
            expected_hash, where=f"source hash for {relative}"
        ):
            raise RuntimeError(f"frozen source changed: {relative}")
    lock = root / "uv.lock"
    if not lock.is_file() or _file_sha256(lock) != _require_sha256(
        bindings.get("dependency_lock_sha256"), where="dependency lock binding"
    ):
        raise RuntimeError("frozen uv.lock changed")
    # Sibling source-bundle.json is a convenient human-readable copy, not a
    # second authority.  If present, it must exactly match the embedded bundle.
    sibling = manifest_path.parent / "source-bundle.json"
    if sibling.exists() and _read_json(sibling, where="source bundle") != clean:
        raise RuntimeError("sibling source bundle differs from the manifest binding")


def _validate_manifest_inventory(manifest: Mapping[str, Any]) -> None:
    trials = manifest.get("trials")
    if (
        manifest.get("phase") != "fixed-representation-comparison"
        or not isinstance(trials, list)
        or len(trials) != 9
    ):
        raise ValueError("factorised manifest must contain exactly nine fixed trials")
    pairs = {
        (trial.get("component"), trial.get("optimiser_seed"))
        for trial in trials
        if isinstance(trial, Mapping)
    }
    expected = {
        (component, seed)
        for component in ("relevance", "target_stance_b4", "target_stance_b2")
        for seed in (47, 61, 89)
    }
    if pairs != expected or any(trial.get("gpu_type") != "L4" for trial in trials):
        raise ValueError("factorised trial set is not the exact L4 component/seed cross product")
    if manifest.get("locked_test_rows_accessed") != 0:
        raise ValueError("factorised manifest must bind zero locked-test access")


def _load_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path, where="factorised run manifest")
    clean = _require_callable(_experiment_module(), "validate_run_manifest")(value)
    if not isinstance(clean, dict):
        raise RuntimeError("factorised manifest validator must return an object")
    validate_compute_contract(clean["experiment_contract"])
    _validate_manifest_inventory(clean)
    return clean


def _run_root(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return volume_root / OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}"


def _preflight_path(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return (
        _run_root(manifest, volume_root=volume_root)
        / "cuda-preflight"
        / f"phase={manifest['phase_run_id']}.json"
    )


def _preparation_id_from_spec(spec: Mapping[str, Any]) -> str:
    missing = [field for field in PREPARATION_IDENTITY_FIELDS if field not in spec]
    if missing:
        raise ValueError(
            "factorised preparation identity omits bindings: "
            + ", ".join(sorted(missing))
        )
    source_bundle_sha256 = _require_sha256(
        spec.get("source_bundle_sha256"), where="preparation source bundle"
    )
    dependency_lock_sha256 = _require_sha256(
        spec.get("dependency_lock_sha256"), where="preparation dependency lock"
    )
    return _canonical_sha256(
        {
            "input_bindings": {
                field: spec[field] for field in PREPARATION_IDENTITY_FIELDS
            },
            "source_bundle_sha256": source_bundle_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
        }
    )


def _prepared_output_root(
    preparation_id: str, *, volume_root: Path = VOLUME_PATH
) -> Path:
    return (
        volume_root
        / OUTPUT_PREFIX
        / "prepared"
        / f"input={_require_sha256(preparation_id, where='preparation_id')}"
    )


def _zero_development_predictions(
    component: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if component == "relevance":
        return [
            {"item_id": row["item_id"], "relevance_logit": 0.0} for row in rows
        ]
    stance_width = 4 if component == "target_stance_b4" else 2
    if component not in {"target_stance_b4", "target_stance_b2"}:
        raise ValueError("unknown factorised preflight component")
    return [
        {
            "item_id": row["item_id"],
            "target_presence_logits": [0.0] * 6,
            "stance_logits": [[0.0] * stance_width for _ in range(5)],
        }
        for row in rows
    ]


def _validate_preflight_publication_contract(
    metrics: Mapping[str, Any],
    *,
    component: str,
    manifest: Mapping[str, Any],
    trial: Mapping[str, Any],
    private_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Round-trip exact scorer output through the real publication contract.

    This deliberately exercises every contract boundary that a paid worker reaches
    after training: private-prediction construction, metrics artefact publication,
    receipt construction, receipt validation and exact final-directory validation.
    Only stable digests leave the ephemeral preflight directory.
    """

    experiment = _experiment_module()
    training = _training_module()
    public_metrics = _require_callable(
        experiment, "validate_public_trial_metrics"
    )(metrics, component=component)
    if not isinstance(public_metrics, dict):
        raise RuntimeError("factorised public metrics validator must return an object")
    if public_metrics != dict(metrics):
        raise RuntimeError(
            f"factorised public metrics validator changed scorer output for {component}"
        )
    if trial.get("component") != component:
        raise RuntimeError("factorised preflight trial/component binding drifted")
    experiment_contract = manifest.get("experiment_contract")
    phase_run_id = manifest.get("phase_run_id")
    if not isinstance(experiment_contract, Mapping) or not isinstance(
        phase_run_id, str
    ):
        raise RuntimeError("factorised preflight manifest binding is incomplete")
    run_manifest_sha256 = _canonical_sha256(manifest)

    with tempfile.TemporaryDirectory(prefix=f"factorised-preflight-{component}-") as raw:
        root = Path(raw)
        checkpoint_path = root / "checkpoint.pt"
        checkpoint_path.write_bytes(
            f"factorised preflight checkpoint: {component}\n".encode()
        )
        predictions = _require_callable(
            training, "build_private_development_predictions"
        )(experiment_contract, trial, rows=private_rows)
        prediction_path = root / "development-predictions.json"
        _require_callable(training, "_write_json_atomic")(
            prediction_path, predictions
        )
        metrics_path = root / "metrics.json"
        _require_callable(training, "_write_json_atomic")(
            metrics_path,
            {
                "schema_version": "1.0.0",
                "kind": training.TRIAL_METRICS_KIND,
                "component": component,
                "aggregate_metrics": public_metrics,
                "epoch_history": [],
                "peak_gpu_bytes": 0,
            },
        )

        def descriptor(path: Path) -> dict[str, Any]:
            return {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }

        artifacts = {
            "checkpoint": descriptor(checkpoint_path),
            "metrics": descriptor(metrics_path),
            "private_development_predictions": descriptor(prediction_path),
        }
        receipt = _require_callable(experiment, "build_trial_receipt")(
            experiment_contract,
            trial,
            phase_run_id=phase_run_id,
            run_manifest_sha256=run_manifest_sha256,
            artifacts=artifacts,
            metrics=public_metrics,
            wall_seconds=1.0,
            gpu_seconds=1.0,
        )
        receipt_path = root / "receipt.json"
        _require_callable(training, "_write_json_atomic")(receipt_path, receipt)
        validated = _require_callable(training, "validate_trial_artifacts")(
            root,
            receipt,
            experiment=experiment_contract,
            trial_spec=trial,
            phase_run_id=phase_run_id,
            run_manifest_sha256=run_manifest_sha256,
        )
        if validated != receipt:
            raise RuntimeError("factorised preflight receipt round trip changed output")

    return {
        "status": "passed",
        "validator": "build_trial_receipt+validate_trial_artifacts",
        "public_metrics_sha256": _canonical_sha256(public_metrics),
        "trial_spec_sha256": _canonical_sha256(trial),
        "receipt_id": receipt["receipt_id"],
        "artifact_bindings_sha256": _canonical_sha256(receipt["artifacts"]),
    }


def _real_development_preflight_evidence(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    """Exercise exact frame loading, natural-design extraction and IPW scoring."""

    training = _training_module()
    bindings = manifest["experiment_contract"]["bindings"]
    descriptor = bindings["development_frame"]
    rows, reference = _require_callable(training, "load_private_frame")(
        volume_root / descriptor["relative_path"],
        descriptor,
        expected_frame="development",
    )
    weighting, design_summary = _require_callable(
        training, "_load_natural_development_weighting"
    )(bindings=bindings, volume_root=volume_root)
    natural_design = _require_callable(
        training, "_natural_design_from_development_rows"
    )(rows)
    components: dict[str, Any] = {}
    publication_contracts: dict[str, Any] = {}
    selection_components = {
        row["item_id"]: row["selection_component"] for row in rows
    }
    for component in ("relevance", "target_stance_b4", "target_stance_b2"):
        private_rows = _zero_development_predictions(component, rows)
        metrics = _require_callable(training, "_score_checkpoint")(
            component=component,
            reference=reference,
            private_rows=private_rows,
            selection_component_by_item=selection_components,
            natural_design=natural_design,
            natural_weighting=weighting,
            selected_epoch=1,
        )
        component_trials = [
            trial for trial in manifest.get("trials", [])
            if isinstance(trial, Mapping) and trial.get("component") == component
        ]
        if not component_trials:
            raise RuntimeError(f"factorised preflight lacks a registered {component} trial")
        publication_contracts[component] = _validate_preflight_publication_contract(
            metrics,
            component=component,
            manifest=manifest,
            trial=component_trials[0],
            private_rows=private_rows,
        )
        probability_design = metrics.get("probability_design")
        if (
            metrics.get("development_rows") != len(rows)
            or metrics.get("primary_probability_rows") != 300
            or metrics.get("natural_arm_weighting_config_digest")
            != weighting.digest()
            or not isinstance(probability_design, Mapping)
            or probability_design.get("design_digest")
            != design_summary.get("design_digest")
        ):
            raise RuntimeError(
                f"real development/IPW preflight drifted for {component}"
            )
        components[component] = {
            "checkpoint_scoring_executed": True,
            "development_rows": len(rows),
            "primary_probability_rows": 300,
            "invalid_outputs": metrics["invalid_outputs"],
            "natural_arm_weighting_config_digest": weighting.digest(),
            "probability_design_digest": probability_design["design_digest"],
        }
    return {
        "development_frame_sha256": descriptor["sha256"],
        "development_frame_value": "development",
        "development_rows": len(rows),
        "natural_design_digest": design_summary["design_digest"],
        "natural_arm_weighting_config_digest": weighting.digest(),
        "component_checkpoint_scoring": components,
        "publication_contract_validation": {
            "status": "passed",
            "validator": "build_trial_receipt+validate_trial_artifacts",
            "components": publication_contracts,
        },
    }


def _trial_job(manifest: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    return _require_callable(_experiment_module(), "make_trial_job")(manifest, trial)


def _inspect_exact_trial(manifest: Mapping[str, Any], trial_id: str) -> dict[str, Any]:
    result = _require_callable(_training_module(), "inspect_trial_output")(
        manifest=manifest, trial_id=trial_id, volume_root=VOLUME_PATH
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("factorised trial inspection must return an object")
    return dict(result)


def _aggregate_inspections(
    manifest: Mapping[str, Any], inspections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = _require_callable(_training_module(), "aggregate_trial_inspections")(
        manifest=manifest, inspections=inspections
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("factorised aggregate inspection must return an object")
    return dict(result)


def _validate_volume_artifact(path: Path, descriptor: Mapping[str, Any], *, where: str) -> None:
    clean = _require_callable(_experiment_module(), "artifact_descriptor")(
        descriptor, where=where
    )
    try:
        relative = path.relative_to(VOLUME_PATH).as_posix()
    except ValueError as exc:
        raise ValueError(f"{where} must remain inside the registered Volume") from exc
    if clean["relative_path"] != relative:
        raise RuntimeError(f"{where} descriptor path binding drifted")
    if not path.is_file():
        raise FileNotFoundError(path)
    if (
        path.stat().st_size != clean["bytes"]
        or _file_sha256(path) != clean["sha256"]
    ):
        raise RuntimeError(f"{where} Volume artefact is missing or corrupt")


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_frames(spec: dict[str, Any]) -> dict[str, Any]:
    """Materialise exact fresh-v2 train/development frames on the Volume."""

    volume.reload()
    labels_relative = _volume_relative(
        spec.get("teacher_labels_parquet_relative_path"), where="teacher labels"
    )
    teacher_receipt_relative = _volume_relative(
        spec.get("teacher_receipt_relative_path"), where="teacher receipt"
    )
    blinded_relative = _volume_relative(
        spec.get("blinded_input_json_relative_path"), where="blinded input"
    )
    mapping_relative = _volume_relative(
        spec.get("private_mapping_parquet_relative_path"), where="private mapping"
    )
    source_relative = _volume_relative(
        spec.get("source_parquet_relative_path"), where="source Parquet"
    )
    source_receipt_relative = _volume_relative(
        spec.get("source_receipt_relative_path"), where="source receipt"
    )
    source_mapping_relative = _volume_relative(
        spec.get("source_metadata_mapping_relative_path"),
        where="source metadata mapping",
    )
    bridge_register_relative = _volume_relative(
        spec.get("bridge_exposure_register_json_relative_path"),
        where="bridge exposure register",
    )
    legacy_proxy_relative = _volume_relative(
        spec.get("legacy_proxy_parquet_relative_path"), where="legacy proxy"
    )
    bridge_receipt_relative = _volume_relative(
        spec.get("bridge_receipt_relative_path"), where="bridge receipt"
    )
    teacher_receipt = VOLUME_PATH / teacher_receipt_relative
    labels = VOLUME_PATH / labels_relative
    blinded = VOLUME_PATH / blinded_relative
    mapping = VOLUME_PATH / mapping_relative
    source = VOLUME_PATH / source_relative
    source_receipt = VOLUME_PATH / source_receipt_relative
    source_mapping = VOLUME_PATH / source_mapping_relative
    bridge_register = VOLUME_PATH / bridge_register_relative
    legacy_proxy = VOLUME_PATH / legacy_proxy_relative
    bridge_receipt = VOLUME_PATH / bridge_receipt_relative
    teacher_receipt_descriptor = spec.get("teacher_receipt")
    labels_descriptor = spec.get("teacher_ledger")
    blinded_descriptor = spec.get("teacher_blinded_input")
    mapping_descriptor = spec.get("teacher_private_mapping")
    source_descriptor = spec.get("source_parquet")
    source_receipt_descriptor = spec.get("source_receipt")
    source_mapping_descriptor = spec.get("source_metadata_mapping")
    bridge_register_descriptor = spec.get("bridge_exposure_register")
    legacy_proxy_descriptor = spec.get("legacy_proxy")
    bridge_receipt_descriptor = spec.get("bridge_authorisation")
    rubric_descriptor = spec.get("rubric")
    schema_descriptor = spec.get("schema")
    descriptors = {
        "teacher receipt": (teacher_receipt, teacher_receipt_descriptor),
        "teacher labels": (labels, labels_descriptor),
        "teacher blinded input": (blinded, blinded_descriptor),
        "teacher private mapping": (mapping, mapping_descriptor),
        "source Parquet": (source, source_descriptor),
        "source receipt": (source_receipt, source_receipt_descriptor),
        "source metadata mapping": (source_mapping, source_mapping_descriptor),
        "bridge exposure register": (bridge_register, bridge_register_descriptor),
        "bridge receipt": (bridge_receipt, bridge_receipt_descriptor),
        "legacy proxy": (legacy_proxy, legacy_proxy_descriptor),
    }
    if any(not isinstance(descriptor, Mapping) for _, descriptor in descriptors.values()):
        raise ValueError("preparation spec lacks a frozen input descriptor")
    if not isinstance(rubric_descriptor, Mapping) or not isinstance(
        schema_descriptor, Mapping
    ):
        raise ValueError("preparation spec lacks frozen rubric/schema descriptors")
    for where, (path, descriptor) in descriptors.items():
        _validate_volume_artifact(path, descriptor, where=where)
    preparation_id = _require_sha256(spec.get("preparation_id"), where="preparation_id")
    if preparation_id != _preparation_id_from_spec(spec):
        raise RuntimeError("factorised preparation identity drifted")
    output = _prepared_output_root(preparation_id)
    result = _require_callable(
        _training_module(), "prepare_evidence_frames_from_artifacts"
    )(
        teacher_receipt_path=teacher_receipt,
        teacher_receipt_descriptor=teacher_receipt_descriptor,
        teacher_labels_parquet_path=labels,
        teacher_labels_descriptor=labels_descriptor,
        blinded_input_json_path=blinded,
        blinded_input_descriptor=blinded_descriptor,
        private_mapping_parquet_path=mapping,
        private_mapping_descriptor=mapping_descriptor,
        source_parquet_path=source,
        source_parquet_descriptor=source_descriptor,
        source_receipt_path=source_receipt,
        source_receipt_descriptor=source_receipt_descriptor,
        source_metadata_mapping_path=source_mapping,
        source_metadata_mapping_descriptor=source_mapping_descriptor,
        bridge_receipt_path=bridge_receipt,
        bridge_receipt_descriptor=bridge_receipt_descriptor,
        bridge_exposure_register_json_path=bridge_register,
        bridge_exposure_register_descriptor=bridge_register_descriptor,
        legacy_proxy_parquet_path=legacy_proxy,
        legacy_proxy_descriptor=legacy_proxy_descriptor,
        private_split_root=output / "private-split",
        public_split_root=output / "public-split",
        frame_output_root=output / "frames",
        descriptor_root=VOLUME_PATH,
        expected_teacher_run_id=_require_sha256(
            spec.get("teacher_run_id"), where="teacher_run_id"
        ),
        rubric_sha256=_require_sha256(
            rubric_descriptor.get("sha256"), where="rubric SHA-256"
        ),
        schema_sha256=_require_sha256(
            schema_descriptor.get("sha256"), where="schema SHA-256"
        ),
    )
    volume.commit()
    return dict(result)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65536,
    timeout=TRAIN_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def cuda_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    """Backpropagate all models and score the exact real development/IPW path."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("factorised CUDA preflight requires the requested L4")
    training = _training_module()
    optimisation = training.FactorisedOptimisationConfig()
    shapes: dict[str, Any] = {}
    gradients: dict[str, bool] = {}
    for component in ("relevance", "target_stance_b4", "target_stance_b2"):
        model = training.create_component_model(component=component, config=optimisation).to(
            "cuda"
        )
        input_ids = torch.randint(0, 128, (2, 16), device="cuda")
        attention_mask = torch.ones_like(input_ids)
        if component == "relevance":
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                relevance_labels=torch.tensor([0, 1], device="cuda"),
                codable_mask=torch.tensor([True, True], device="cuda"),
            )
            shapes[component] = list(output["relevance_logits"].shape)
        else:
            stance = (
                torch.zeros((2, 5), dtype=torch.long, device="cuda")
                if component.endswith("b4")
                else torch.zeros((2, 5, 2), device="cuda")
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_presence_labels=torch.ones((2, 6), device="cuda"),
                stance_labels=stance,
                reference_material_mask=torch.tensor([True, True], device="cuda"),
                stance_known_mask=torch.ones((2, 5), dtype=torch.bool, device="cuda"),
            )
            shapes[component] = {
                "target": list(output["target_presence_logits"].shape),
                "stance": list(output["stance_logits"].shape),
            }
        output["loss"].backward()
        gradients[component] = any(
            parameter.grad is not None for parameter in model.parameters()
        )
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    if not all(gradients.values()):
        raise RuntimeError("factorised CUDA preflight did not reach every component")
    real_development = _real_development_preflight_evidence(manifest)
    bindings = manifest["experiment_contract"]["bindings"]
    receipt = {
        "status": "passed",
        "kind": CUDA_PREFLIGHT_RECEIPT_KIND,
        "gpu_type": "L4",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "source_bundle_sha256": bindings["source_bundle_sha256"],
        "dependency_lock_sha256": bindings["dependency_lock_sha256"],
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
        "synthetic_rows": 2,
        "synthetic_sequence_length": 16,
        "output_shapes": shapes,
        "component_gradients": gradients,
        "real_development_scoring": real_development,
        "locked_test_rows_accessed": 0,
    }
    _write_immutable_json(_preflight_path(manifest), receipt)
    volume.commit()
    return receipt


def validate_cuda_preflight_receipt(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    path = _preflight_path(manifest, volume_root=volume_root)
    if not path.is_file():
        raise RuntimeError("factorised launch requires an exact CUDA preflight receipt")
    receipt = _read_json(path, where="factorised CUDA preflight receipt")
    bindings = manifest["experiment_contract"]["bindings"]
    expected = {
        "status": "passed",
        "kind": CUDA_PREFLIGHT_RECEIPT_KIND,
        "gpu_type": "L4",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "source_bundle_sha256": bindings["source_bundle_sha256"],
        "dependency_lock_sha256": bindings["dependency_lock_sha256"],
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
        "synthetic_rows": 2,
        "synthetic_sequence_length": 16,
        "output_shapes": {
            "relevance": [2],
            "target_stance_b4": {"target": [2, 6], "stance": [2, 5, 4]},
            "target_stance_b2": {"target": [2, 6], "stance": [2, 5, 2]},
        },
        "component_gradients": {
            "relevance": True,
            "target_stance_b4": True,
            "target_stance_b2": True,
        },
        "real_development_scoring": _real_development_preflight_evidence(
            manifest, volume_root=volume_root
        ),
        "locked_test_rows_accessed": 0,
    }
    if receipt != expected:
        raise RuntimeError("factorised CUDA preflight binding drifted")
    return receipt


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65536,
    timeout=TRAIN_TIMEOUT_SECONDS,
    max_containers=MAX_CONCURRENT_TRIALS,
    volumes={str(VOLUME_PATH): volume},
)
def train_l4(job: dict[str, Any]) -> Mapping[str, Any]:
    training = _training_module()
    result = _require_callable(training, "run_training_trial")(
        job=job,
        volume_root=VOLUME_PATH,
        trial_executor=_require_callable(training, "execute_registered_gpu_trial"),
    )
    # The caller must not observe success before the immutable final directory is
    # durably visible to inspection containers.
    volume.commit()
    return result


def _claim_path(manifest: Mapping[str, Any], trial_id: str) -> Path:
    return _run_root(manifest) / "launch-claims" / f"trial={trial_id}.json"


def _dispatch_path(manifest: Mapping[str, Any], trial_id: str) -> Path:
    return _run_root(manifest) / "launch-dispatches" / f"trial={trial_id}.json"


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def coordinate_training(manifest: dict[str, Any]) -> dict[str, Any]:
    """Claim every missing registered trial before spawning it exactly once."""

    volume.reload()
    validate_cuda_preflight_receipt(manifest)
    inspections = [
        _inspect_exact_trial(manifest, trial["trial_id"]) for trial in manifest["trials"]
    ]
    aggregate = _aggregate_inspections(manifest, inspections)
    if aggregate["incomplete_trial_ids"]:
        raise RuntimeError("incomplete factorised trials require manual reconciliation")
    by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    calls: list[str] = []
    for trial_id in aggregate["missing_trial_ids"]:
        claim = _claim_path(manifest, trial_id)
        dispatch = _dispatch_path(manifest, trial_id)
        if claim.exists() or dispatch.exists():
            raise RuntimeError("existing launch claim requires manual reconciliation")
        payload = {
            "experiment_run_id": manifest["experiment_run_id"],
            "phase_run_id": manifest["phase_run_id"],
            "trial_id": trial_id,
            "trial_spec_sha256": _canonical_sha256(by_id[trial_id]),
            "status": "claimed_before_spawn",
        }
        _write_immutable_json(claim, payload)
        volume.commit()
        call = train_l4.spawn(_trial_job(manifest, by_id[trial_id]))
        _write_immutable_json(
            dispatch,
            {**payload, "status": "spawned", "function_call_id": call.object_id},
        )
        volume.commit()
        calls.append(call.object_id)
    return {
        "status": "submitted" if calls else "already_complete",
        "expected_trials": 9,
        "submitted_function_call_ids": calls,
        "complete_trial_ids": aggregate["complete_trial_ids"],
        "locked_test_rows_accessed": 0,
    }


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def inspect_run(manifest: dict[str, Any]) -> dict[str, Any]:
    volume.reload()
    return _aggregate_inspections(
        manifest,
        [_inspect_exact_trial(manifest, trial["trial_id"]) for trial in manifest["trials"]],
    )


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    max_containers=MAX_CONCURRENT_TRIALS,
    volumes={str(VOLUME_PATH): volume},
)
def inspect_trial(manifest: dict[str, Any], trial_id: str) -> dict[str, Any]:
    volume.reload()
    return _inspect_exact_trial(manifest, trial_id)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def closeout_comparison(manifest: dict[str, Any]) -> dict[str, Any]:
    volume.reload()
    result = _require_callable(_training_module(), "closeout_fixed_comparison")(
        manifest=manifest, volume_root=VOLUME_PATH
    )
    volume.commit()
    return dict(result)


def _confirmation(action: str, manifest: Mapping[str, Any]) -> str:
    return f"{action.upper()}_MODERNBERT_FACTORISED_{str(manifest['phase_run_id'])[:12]}"


def _prepare_manifest(
    *, input_spec_path: Path, manifest_path: Path, source_bundle_path: Path
) -> dict[str, Any]:
    """Materialise fresh frames remotely, then freeze the exact local manifest."""

    if manifest_path.exists() or source_bundle_path.exists():
        raise RuntimeError("refusing to overwrite an existing factorised preparation")
    spec = _read_json(input_spec_path, where="factorised prepare spec")
    required = {
        "teacher_run_id",
        "teacher_receipt",
        "teacher_receipt_relative_path",
        "teacher_ledger",
        "teacher_labels_parquet_relative_path",
        "teacher_blinded_input",
        "blinded_input_json_relative_path",
        "teacher_private_mapping",
        "private_mapping_parquet_relative_path",
        "source_parquet",
        "source_parquet_relative_path",
        "source_receipt",
        "source_receipt_relative_path",
        "source_metadata_mapping",
        "source_metadata_mapping_relative_path",
        "bridge_exposure_register",
        "bridge_exposure_register_json_relative_path",
        "bridge_receipt_relative_path",
        "legacy_proxy",
        "legacy_proxy_parquet_relative_path",
        "rubric",
        "schema",
        "bridge_authorisation",
        "source_files",
        "rate_card_usd_per_gpu_second",
        "cumulative_measured_spend_usd",
        "active_reservation_usd",
        "planned_phase_upper_usd",
        "hard_cost_cap_usd",
        "max_gpu_seconds_by_component",
        "source_bundle_sha256",
        "dependency_lock_sha256",
    }
    if set(spec) != required:
        raise ValueError("factorised prepare spec schema drifted")
    source_files = spec["source_files"]
    if not isinstance(source_files, list) or not set(REQUIRED_SOURCE_FILES) <= set(source_files):
        raise ValueError("prepare spec omits required source files")
    experiment = _experiment_module()
    source_bundle = experiment.build_source_bundle(_repo_root(), source_files)
    source_bundle_sha256 = _canonical_sha256(source_bundle)
    dependency_lock_sha256 = _file_sha256(_repo_root() / "uv.lock")
    if (
        spec["source_bundle_sha256"] != source_bundle_sha256
        or spec["dependency_lock_sha256"] != dependency_lock_sha256
    ):
        raise RuntimeError(
            "factorised prepare spec source bundle or dependency lock drifted"
        )
    preparation_id = _preparation_id_from_spec(spec)
    frames = prepare_frames.remote({**spec, "preparation_id": preparation_id})
    if frames.get("teacher_labels_sha256") != spec["teacher_ledger"]["sha256"]:
        raise RuntimeError("prepared frame teacher binding drifted")
    if frames.get("blinded_input_sha256") != spec["teacher_blinded_input"]["sha256"]:
        raise RuntimeError("prepared frame blinded-input binding drifted")
    if frames.get("private_mapping_sha256") != spec["teacher_private_mapping"]["sha256"]:
        raise RuntimeError("prepared frame private-mapping binding drifted")
    if frames.get("source_parquet_sha256") != spec["source_parquet"]["sha256"]:
        raise RuntimeError("prepared frame source binding drifted")
    if frames.get("bridge_exposure_register_sha256") != spec[
        "bridge_exposure_register"
    ]["sha256"]:
        raise RuntimeError("prepared frame bridge-register binding drifted")
    if frames.get("legacy_proxy_sha256") != spec["legacy_proxy"]["sha256"]:
        raise RuntimeError("prepared frame legacy-proxy binding drifted")
    contract = experiment.freeze_experiment_contract(
        teacher_run_id=spec["teacher_run_id"],
        teacher_receipt=spec["teacher_receipt"],
        teacher_ledger=spec["teacher_ledger"],
        teacher_blinded_input=spec["teacher_blinded_input"],
        teacher_private_mapping=spec["teacher_private_mapping"],
        source_parquet=spec["source_parquet"],
        source_receipt=spec["source_receipt"],
        source_metadata_mapping=spec["source_metadata_mapping"],
        split_manifest=frames["split_manifest"],
        split_public_manifest=frames["split_public_manifest"],
        bridge_exposure_register=spec["bridge_exposure_register"],
        legacy_proxy=spec["legacy_proxy"],
        legacy_overlap_audit=frames["legacy_overlap_audit"],
        training_frame=frames["training"],
        development_frame=frames["development"],
        development_probability_design=frames["probability_designs"][
            "development_probability"
        ],
        rubric=spec["rubric"],
        schema=spec["schema"],
        bridge_authorisation=spec["bridge_authorisation"],
        source_bundle=source_bundle,
        dependency_lock_sha256=dependency_lock_sha256,
        rate_card_usd_per_gpu_second=spec["rate_card_usd_per_gpu_second"],
        cumulative_measured_spend_usd=spec["cumulative_measured_spend_usd"],
        active_reservation_usd=spec["active_reservation_usd"],
        planned_phase_upper_usd=spec["planned_phase_upper_usd"],
        hard_cost_cap_usd=spec["hard_cost_cap_usd"],
    )
    caps = spec["max_gpu_seconds_by_component"]
    if not isinstance(caps, Mapping) or set(caps) != {
        "relevance",
        "target_stance_b4",
        "target_stance_b2",
    }:
        raise ValueError("prepare spec GPU-second caps are invalid")
    trials = [
        experiment.freeze_trial_spec(
            contract,
            component=component,
            optimiser_seed=seed,
            max_gpu_seconds=caps[component],
        )
        for component in ("relevance", "target_stance_b4", "target_stance_b2")
        for seed in (47, 61, 89)
    ]
    manifest = experiment.build_run_manifest(contract, trials=trials)
    _write_immutable_json(source_bundle_path, source_bundle)
    _write_immutable_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "expected_trials": 9,
        "training_rows": frames["training"]["row_count"],
        "development_rows": frames["development"]["row_count"],
        "thread_overlap": frames["thread_overlap"],
        "cuda_preflight_required": True,
        "cuda_preflight_executed": False,
        "locked_test_rows_accessed": 0,
    }


@app.local_entrypoint()
def main(
    action: str = "",
    manifest_path: str = "data/private-modernbert-factorised-v2/run-manifest.json",
    input_spec_path: str = "data/private-modernbert-factorised-v2/prepare-input.json",
    approved_cost_usd: str = "0",
    confirm: str = "",
) -> None:
    actions = {
        "prepare",
        "validate",
        "cuda-preflight",
        "launch",
        "inspect",
        "validate-sharded",
        "closeout",
    }
    if action not in actions:
        raise ValueError("action must be exactly " + ", ".join(sorted(actions)))
    manifest_file = Path(manifest_path)
    if action == "prepare":
        if confirm != "PREPARE_MODERNBERT_FACTORISED_V2":
            raise RuntimeError(
                "refusing preparation: pass --confirm PREPARE_MODERNBERT_FACTORISED_V2"
            )
        result = _prepare_manifest(
            input_spec_path=Path(input_spec_path),
            manifest_path=manifest_file,
            source_bundle_path=manifest_file.parent / "source-bundle.json",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    manifest = _load_manifest(manifest_file)
    try:
        approved = Decimal(approved_cost_usd)
        estimate = Decimal(
            manifest["experiment_contract"]["compute"]["planned_phase_upper_usd"]
        )
    except (InvalidOperation, KeyError) as exc:
        raise ValueError("CLI or immutable cost binding is invalid") from exc
    if action in {"cuda-preflight", "launch", "closeout"}:
        enforce_cost_guardrail(estimated_cost_usd=estimate, approved_cost_usd=approved)
        if approved != estimate:
            raise ValueError("CLI approval must exactly match the immutable phase maximum")
    summary: dict[str, Any] = {
        "status": "validated" if action == "validate" else "planned",
        "action": action,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "expected_trials": 9,
        "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
        "required_confirmation": _confirmation(action, manifest),
        "locked_test_rows_accessed": 0,
    }
    if action == "validate":
        verify_frozen_source_bundle(manifest_path=manifest_file, manifest=manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if action in {"cuda-preflight", "launch", "closeout"}:
        required = _confirmation(action, manifest)
        if confirm != required:
            raise RuntimeError(f"refusing mutation: pass --confirm {required}")
        verify_frozen_source_bundle(manifest_path=manifest_file, manifest=manifest)
    if action == "cuda-preflight":
        summary["preflight"] = cuda_preflight.remote(manifest)
        summary["status"] = "preflight-complete"
    elif action == "launch":
        summary["submission"] = coordinate_training.remote(manifest)
        summary["status"] = "submitted"
    elif action == "inspect":
        summary["inspection"] = inspect_run.remote(manifest)
        summary["status"] = "inspected"
    elif action == "validate-sharded":
        calls = [inspect_trial.spawn(manifest, trial["trial_id"]) for trial in manifest["trials"]]
        inspection = _aggregate_inspections(manifest, [call.get() for call in calls])
        if inspection["status"] != "complete" or inspection["incomplete_trial_ids"]:
            raise RuntimeError("factorised validation requires nine exact final trials")
        summary["inspection"] = inspection
        summary["status"] = "validated"
    elif action == "closeout":
        inspection = inspect_run.remote(manifest)
        if inspection["status"] != "complete" or inspection["incomplete_trial_ids"]:
            raise RuntimeError("factorised closeout requires nine exact final trials")
        summary["closeout"] = closeout_comparison.remote(manifest)
        summary["status"] = "closed-out"
    print(json.dumps(summary, indent=2, sort_keys=True))


__all__ = [
    "ACCOUNT_GPU_LIMIT",
    "ALLOWED_GPUS",
    "APP_NAME",
    "CUDA_PREFLIGHT_RECEIPT_KIND",
    "ENVIRONMENT_NAME",
    "HARD_MAX_APPROVAL_USD",
    "MAX_CONCURRENT_TRIALS",
    "OUTPUT_PREFIX",
    "REQUIRED_SOURCE_FILES",
    "VOLUME_NAME",
    "VOLUME_PATH",
    "coordinate_training",
    "cuda_preflight",
    "enforce_cost_guardrail",
    "inspect_run",
    "inspect_trial",
    "main",
    "prepare_frames",
    "validate_compute_contract",
    "validate_cuda_preflight_receipt",
    "verify_frozen_source_bundle",
]
