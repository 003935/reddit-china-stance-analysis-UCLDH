"""Modal orchestration for the separate-model ModernBERT cascade experiment.

The launcher exposes only synthetic CUDA preflight, the exact six registered
component trials, metadata-only inspection/validation, and development-only
closeout.  It has no locked-test or corpus-inference authority.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import modal

APP_NAME = "reddit-china-stance-modernbert-cascade-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path("student-modernbert-cascade-v1")

MAX_CONCURRENT_TRIALS = 6
ACCOUNT_GPU_LIMIT = 10
HARD_MAX_APPROVAL_USD = Decimal("20")
ALLOWED_GPUS = ("L4",)
TRAIN_TIMEOUT_SECONDS = 18_000

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


def _experiment_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_cascade_experiment")


def _runtime_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_cascade_runtime")


def _require_callable(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        raise RuntimeError(f"cascade implementation lacks required callable {name}")
    return value


def _require_sha256(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def enforce_cost_guardrail(*, estimated_cost_usd: Decimal, approved_cost_usd: Decimal) -> None:
    if approved_cost_usd <= 0 or approved_cost_usd > HARD_MAX_APPROVAL_USD:
        raise ValueError(f"approved cost must be > 0 and <= {HARD_MAX_APPROVAL_USD}")
    if estimated_cost_usd < 0:
        raise ValueError("estimated cost must be non-negative")
    if estimated_cost_usd > approved_cost_usd:
        raise RuntimeError(
            f"estimated cascade cost ${estimated_cost_usd} exceeds approved ${approved_cost_usd}"
        )


def validate_compute_contract(contract: Mapping[str, Any]) -> None:
    compute = contract.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("cascade experiment contract lacks compute binding")
    expected = {
        "allowed_gpus": list(ALLOWED_GPUS),
        "account_gpu_limit": ACCOUNT_GPU_LIMIT,
        "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
        "gpu_fallback_allowed": False,
    }
    if any(compute.get(key) != value for key, value in expected.items()):
        raise ValueError("cascade compute or concurrency binding drifted")
    if MAX_CONCURRENT_TRIALS > ACCOUNT_GPU_LIMIT:
        raise ValueError("cascade concurrency exceeds the account GPU limit")
    try:
        estimate = Decimal(str(compute["planned_upper_cost_usd"]))
        approved = Decimal(str(compute["approved_cost_usd"]))
    except (InvalidOperation, KeyError) as error:
        raise ValueError("cascade cost binding is invalid") from error
    enforce_cost_guardrail(estimated_cost_usd=estimate, approved_cost_usd=approved)


def verify_frozen_source_bundle(*, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    contract = manifest.get("experiment_contract")
    bindings = contract.get("bindings") if isinstance(contract, Mapping) else None
    if not isinstance(bindings, Mapping):
        raise ValueError("cascade manifest lacks source bindings")
    bundle_path = manifest_path.parent / "source-bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError("sibling source-bundle.json is required")
    if _file_sha256(bundle_path) != _require_sha256(
        bindings.get("source_bundle_sha256"), where="source bundle binding"
    ):
        raise RuntimeError("source bundle content address drifted")
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("source bundle must contain JSON") from error
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("source_glob") != "src/reddit_china_stance/*.py"
        or not isinstance(bundle.get("files"), Mapping)
    ):
        raise ValueError("source bundle inventory is invalid")
    files = bundle["files"]
    if bundle.get("code_sha256") != _canonical_sha256(files):
        raise RuntimeError("source bundle code content address drifted")
    if bundle["code_sha256"] != _require_sha256(bindings.get("code_sha256"), where="code binding"):
        raise RuntimeError("manifest code binding drifted")
    root = _repo_root()
    current = {
        path.relative_to(root).as_posix()
        for path in (root / "src/reddit_china_stance").glob("*.py")
        if path.is_file()
    }
    if current != set(files):
        raise RuntimeError("frozen source inventory changed")
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not relative.startswith("src/"):
            raise ValueError("source bundle contains an unsafe path")
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


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cascade run manifest must contain an object")
    clean = _require_callable(_experiment_module(), "validate_run_manifest")(value)
    if not isinstance(clean, dict):
        raise RuntimeError("cascade manifest validator must return an object")
    validate_compute_contract(clean["experiment_contract"])
    _validate_manifest_inventory(clean)
    return clean


def _trial_component(trial: Mapping[str, Any]) -> str:
    component = trial.get("component", trial.get("task"))
    if component not in {"relevance", "target_conditioned"}:
        config = trial.get("config")
        component = config.get("component") if isinstance(config, Mapping) else None
    if component not in {"relevance", "target_conditioned"}:
        raise ValueError("cascade trial component is invalid")
    return str(component)


def _trial_seed(trial: Mapping[str, Any]) -> int:
    value = trial.get("optimiser_seed", trial.get("seed"))
    if type(value) is not int:
        config = trial.get("config")
        if isinstance(config, Mapping):
            value = config.get("optimiser_seed", config.get("seed"))
    if type(value) is not int:
        raise ValueError("cascade trial seed is invalid")
    return value


def _validate_manifest_inventory(manifest: Mapping[str, Any]) -> None:
    trials = manifest.get("trials")
    phase = manifest.get("phase")
    if phase not in {"confirmation", "cascade"} or not isinstance(trials, list):
        raise ValueError("cascade manifest phase or trials are invalid")
    if len(trials) != 6:
        raise ValueError("cascade manifest must contain exactly six trials")
    ids = [trial.get("trial_id") for trial in trials if isinstance(trial, Mapping)]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("cascade trial IDs must be unique")
    if any(trial.get("gpu_type") != "L4" for trial in trials):
        raise ValueError("cascade trials must use exactly L4")
    pairs = [(_trial_component(trial), _trial_seed(trial)) for trial in trials]
    seeds = {seed for _, seed in pairs}
    if len(seeds) != 3 or set(pairs) != {
        (component, seed)
        for component in ("relevance", "target_conditioned")
        for seed in seeds
    }:
        raise ValueError("cascade trials must be an exact two-component by three-seed design")
    locked = manifest.get("locked_test", manifest.get("locked_test_access"))
    if locked is not None and (
        not isinstance(locked, Mapping) or locked.get("rows_accessed") != 0
    ):
        raise ValueError("cascade manifest must bind zero locked-test access")


def _trial_job(manifest: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    builder = getattr(_experiment_module(), "make_trial_job", None)
    compact = builder(manifest, trial) if callable(builder) else {}
    if not isinstance(compact, Mapping):
        raise RuntimeError("cascade trial-job builder must return an object")
    job = {
        **dict(compact),
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "experiment_contract": dict(manifest["experiment_contract"]),
        "trial_spec": dict(trial),
        "trial_spec_sha256": _canonical_sha256(trial),
        "run_manifest_sha256": _canonical_sha256(manifest),
    }
    return job


def _run_root(manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH) -> Path:
    return volume_root / OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}"


def _inspect_exact_trial(manifest: Mapping[str, Any], trial_id: str) -> dict[str, Any]:
    result = _require_callable(_runtime_module(), "inspect_trial_output")(
        manifest=manifest, trial_id=trial_id, volume_root=VOLUME_PATH
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("cascade trial inspection must return an object")
    return dict(result)


def _aggregate_inspections(
    manifest: Mapping[str, Any], inspections: list[Mapping[str, Any]]
) -> dict[str, Any]:
    result = _require_callable(_runtime_module(), "aggregate_trial_inspections")(
        manifest=manifest, inspections=inspections
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("cascade aggregate inspection must return an object")
    return dict(result)


def _invoke_training(job: Mapping[str, Any], *, resume: bool) -> Mapping[str, Any]:
    result = _require_callable(_runtime_module(), "run_training_trial")(
        job=job, volume_root=VOLUME_PATH, resume=resume
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("cascade training runtime must return an object")
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65536,
    timeout=TRAIN_TIMEOUT_SECONDS,
    max_containers=MAX_CONCURRENT_TRIALS,
    volumes={str(VOLUME_PATH): volume},
)
def train_l4(job: dict[str, Any], resume: bool = False) -> Mapping[str, Any]:
    return _invoke_training(job, resume=resume)


def _cuda_preflight_receipt_path(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> Path:
    return (
        _run_root(manifest, volume_root=volume_root)
        / "cuda-preflight-receipts"
        / f"phase-run={manifest['phase_run_id']}.json"
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            f"existing immutable state requires manual reconciliation: {path}"
        ) from error


def _publish_idempotent_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError("immutable preflight receipt already differs")
        return
    _write_immutable_json(path, payload)


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
    """Exercise both independently loaded component models on synthetic data."""

    import torch

    trainer = importlib.import_module("reddit_china_stance.modernbert_cascade_trainer")
    config_type = trainer.CascadeOptimisationConfig
    create_models = _require_callable(trainer, "create_modernbert_cascade_models")
    if not torch.cuda.is_available():
        raise RuntimeError("cascade CUDA preflight requires the requested L4")
    models = create_models(
        optimisation_config=config_type(encoder_learning_rate=5e-5)
    )
    batch_size, sequence_length = 2, 16
    input_ids = torch.randint(0, 128, (batch_size, sequence_length), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    relevance = models.relevance.to("cuda")
    relevance.train()
    relevance_output = relevance(
        input_ids=input_ids,
        attention_mask=attention_mask,
        relevance_labels=torch.tensor([0, 1], device="cuda"),
    )
    relevance_output["loss"].backward()
    relevance.to("cpu")
    torch.cuda.empty_cache()
    target = models.target_conditioned.to("cuda")
    target.train()
    target_output = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_state_labels=torch.tensor([0, 1], device="cuda"),
    )
    target_output["loss"].backward()
    receipt = {
        "status": "passed",
        "kind": "modernbert-cascade-cuda-preflight-v1",
        "gpu_type": "L4",
        "synthetic_rows": 2,
        "synthetic_sequence_length": 16,
        "relevance_shape": list(relevance_output["relevance_logits"].shape),
        "target_conditioned_shape": list(target_output["target_state_logits"].shape),
        "component_gradients": {
            "relevance": any(
                parameter.grad is not None for parameter in models.relevance.parameters()
            ),
            "target_conditioned": any(
                parameter.grad is not None for parameter in models.target_conditioned.parameters()
            ),
        },
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        **{
            key: manifest["experiment_contract"]["bindings"][key]
            for key in (
                "source_bundle_sha256",
                "code_sha256",
                "dependency_lock_sha256",
                "model_id",
                "model_revision",
            )
        },
    }
    if not all(receipt["component_gradients"].values()):
        raise RuntimeError("CUDA preflight did not backpropagate through both components")
    _publish_idempotent_json(_cuda_preflight_receipt_path(manifest), receipt)
    volume.commit()
    return receipt


def validate_cuda_preflight_receipt(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    path = _cuda_preflight_receipt_path(manifest, volume_root=volume_root)
    if not path.is_file():
        raise RuntimeError("cascade launch requires an exact CUDA preflight receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("cascade CUDA preflight receipt is not JSON") from error
    bindings = manifest["experiment_contract"]["bindings"]
    expected = {
        "status": "passed",
        "kind": "modernbert-cascade-cuda-preflight-v1",
        "gpu_type": "L4",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "source_bundle_sha256": bindings["source_bundle_sha256"],
        "code_sha256": bindings["code_sha256"],
        "dependency_lock_sha256": bindings["dependency_lock_sha256"],
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
        "relevance_shape": [2, 3],
        "target_conditioned_shape": [2, 6],
        "component_gradients": {"relevance": True, "target_conditioned": True},
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("cascade CUDA preflight receipt binding drifted")
    return receipt


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
    """Claim each missing trial before spawning it; ambiguous claims fail closed."""

    volume.reload()
    validate_cuda_preflight_receipt(manifest)
    inspections = [
        _inspect_exact_trial(manifest, trial["trial_id"]) for trial in manifest["trials"]
    ]
    aggregate = _aggregate_inspections(manifest, inspections)
    if aggregate.get("resumable_trial_ids"):
        raise RuntimeError("resumable cascade trials require explicit manual resume")
    by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    calls: list[str] = []
    for trial_id in aggregate.get("missing_trial_ids", []):
        claim_path = _claim_path(manifest, trial_id)
        dispatch_path = _dispatch_path(manifest, trial_id)
        if claim_path.exists() or dispatch_path.exists():
            raise RuntimeError("existing launch claim requires manual reconciliation")
        trial = by_id[trial_id]
        claim = {
            "experiment_run_id": manifest["experiment_run_id"],
            "phase_run_id": manifest["phase_run_id"],
            "trial_id": trial_id,
            "trial_spec_sha256": _canonical_sha256(trial),
            "status": "claimed_before_spawn",
        }
        _write_immutable_json(claim_path, claim)
        volume.commit()
        call = train_l4.spawn(_trial_job(manifest, trial), resume=False)
        _write_immutable_json(
            dispatch_path,
            {**claim, "status": "spawned", "function_call_id": call.object_id},
        )
        volume.commit()
        calls.append(call.object_id)
    return {
        "status": "submitted" if calls else "already_complete",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "submitted_function_call_ids": calls,
        "complete_trial_ids": aggregate.get("complete_trial_ids", []),
    }


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def inspect_run(manifest: dict[str, Any]) -> dict[str, Any]:
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
    return _inspect_exact_trial(manifest, trial_id)


def _load_bound_baseline_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artefacts = manifest.get("experiment_artefacts")
    descriptor = artefacts.get("baseline_manifest") if isinstance(artefacts, Mapping) else None
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "repo_relative_path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError("cascade closeout requires an exact baseline manifest binding")
    raw_relative = descriptor["repo_relative_path"]
    relative = PurePosixPath(raw_relative) if isinstance(raw_relative, str) else None
    if relative is None or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("cascade baseline manifest path is unsafe")
    path = _repo_root().joinpath(*relative.parts)
    if (
        type(descriptor["bytes"]) is not int
        or not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or _file_sha256(path) != _require_sha256(descriptor["sha256"], where="baseline binding")
    ):
        raise RuntimeError("cascade baseline manifest binding drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("cascade baseline manifest must contain an object")
    from reddit_china_stance.privacy import assert_metadata_only

    assert_metadata_only(value, where="cascade closeout baseline manifest")
    return value


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=20 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def closeout_confirmation(
    manifest: dict[str, Any], baseline_manifest: dict[str, Any]
) -> dict[str, Any]:
    result = _require_callable(_runtime_module(), "closeout_confirmation")(
        confirmation_manifest=manifest,
        baseline_manifest=baseline_manifest,
        volume_root=VOLUME_PATH,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("cascade closeout must return metadata")
    volume.commit()
    return dict(result)


def _confirmation(action: str, manifest: Mapping[str, Any]) -> str:
    return f"{action.upper()}_MODERNBERT_CASCADE_{str(manifest['phase_run_id'])[:12]}"


def _submit_training(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch to the Volume-mounted coordinator without reading ``/data`` locally.

    The coordinator reloads the Volume and validates the exact persisted CUDA
    preflight receipt before it creates any trial claim or function call.
    """

    result = coordinate_training.remote(dict(manifest))
    if not isinstance(result, Mapping):
        raise RuntimeError("cascade training coordinator must return an object")
    return dict(result)


@app.local_entrypoint()
def main(
    action: str = "",
    manifest_path: str = (
        "data/private-modernbert-cascade-v1/confirmation-v3/run-manifest.json"
    ),
    approved_cost_usd: str = "20",
    confirm: str = "",
) -> None:
    actions = {"plan", "cuda-preflight", "train", "inspect", "validate-sharded", "closeout"}
    if action not in actions:
        raise ValueError("action must be exactly " + ", ".join(sorted(actions)))
    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    try:
        approved = Decimal(approved_cost_usd)
        compute = manifest["experiment_contract"]["compute"]
        immutable_approval = Decimal(str(compute["approved_cost_usd"]))
        estimate = Decimal(str(compute["planned_upper_cost_usd"]))
    except (InvalidOperation, KeyError) as error:
        raise ValueError("CLI or immutable cost binding is invalid") from error
    enforce_cost_guardrail(estimated_cost_usd=estimate, approved_cost_usd=approved)
    if approved != immutable_approval:
        raise ValueError("CLI approval must exactly match the immutable contract")
    summary: dict[str, Any] = {
        "status": "planned",
        "action": action,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "expected_trials": 6,
        "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
        "approved_cost_usd": str(approved),
        "required_confirmation": _confirmation(action, manifest),
    }
    if action == "plan":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if action in {"cuda-preflight", "train", "closeout"}:
        required = _confirmation(action, manifest)
        if confirm != required:
            raise RuntimeError(f"refusing mutation: pass --confirm {required}")
    if action in {"cuda-preflight", "train", "closeout"}:
        verify_frozen_source_bundle(manifest_path=manifest_file, manifest=manifest)
    if action == "cuda-preflight":
        summary["preflight"] = cuda_preflight.remote(manifest)
        summary["status"] = "preflight-complete"
    elif action == "train":
        summary["submission"] = _submit_training(manifest)
        summary["status"] = "submitted"
    elif action == "inspect":
        summary["inspection"] = inspect_run.remote(manifest)
        summary["status"] = "inspected"
    elif action == "validate-sharded":
        calls = [inspect_trial.spawn(manifest, trial["trial_id"]) for trial in manifest["trials"]]
        inspection = _aggregate_inspections(manifest, [call.get() for call in calls])
        if inspection.get("status") != "complete":
            raise RuntimeError("cascade validation requires all six exact trials")
        summary["inspection"] = inspection
        summary["status"] = "validated"
    elif action == "closeout":
        inspection = inspect_run.remote(manifest)
        if inspection.get("status") != "complete":
            raise RuntimeError("cascade closeout requires all six validated trials")
        baseline = _load_bound_baseline_manifest(manifest)
        summary["closeout"] = closeout_confirmation.remote(manifest, baseline)
        summary["status"] = "closed-out"
    print(json.dumps(summary, indent=2, sort_keys=True))
