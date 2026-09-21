"""Modal orchestration for the isolated conditional-state ModernBERT study.

The module owns cloud boundaries only.  It deliberately has no evaluation-data
entrypoint and never creates an experiment manifest from local private inputs.
The conditional experiment module owns the immutable contract, trial receipts,
and training implementation; imports of it are lazy so importing this module is
safe on a developer machine without the optional ML dependencies.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import modal

APP_NAME = "reddit-china-stance-modernbert-conditional-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path("student-modernbert-conditional-v1")

MAX_CONCURRENT_TRIALS = 8
ACCOUNT_GPU_LIMIT = 10
HARD_MAX_APPROVAL_USD = Decimal("200")
ALLOWED_GPUS = ("L4",)
ASHA_RUNGS = (2, 4, 8)
ASHA_COUNTS = {2: 8, 4: 4, 8: 2}
TRAIN_TIMEOUT_SECONDS = 5_400

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
    """Load the isolated contract/runtime implementation only when required."""

    return importlib.import_module("reddit_china_stance.modernbert_conditional_experiment")


def _runtime_module() -> Any:
    """Load the private-data runtime only inside a Modal container."""

    return importlib.import_module("reddit_china_stance.modernbert_conditional_runtime")


def _require_callable(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        raise RuntimeError(f"conditional experiment implementation lacks {name}")
    return value


def enforce_cost_guardrail(*, estimated_cost_usd: Decimal, approved_cost_usd: Decimal) -> None:
    """Fail closed before any submission outside the approved bounded study."""

    if approved_cost_usd <= 0 or approved_cost_usd > HARD_MAX_APPROVAL_USD:
        raise ValueError(f"approved cost must be > 0 and <= {HARD_MAX_APPROVAL_USD}")
    if estimated_cost_usd < 0:
        raise ValueError("estimated cost must be non-negative")
    if estimated_cost_usd > approved_cost_usd:
        raise RuntimeError(
            "estimated conditional ModernBERT cost "
            f"${estimated_cost_usd} exceeds approved ${approved_cost_usd}"
        )


def validate_compute_contract(contract: Mapping[str, Any]) -> None:
    """Enforce the account, GPU and cost boundary independently of the runtime."""

    compute = contract.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("conditional experiment contract lacks compute binding")
    if compute.get("allowed_gpus") != list(ALLOWED_GPUS):
        raise ValueError("conditional experiment GPU binding drifted")
    if compute.get("account_gpu_limit") != ACCOUNT_GPU_LIMIT:
        raise ValueError("conditional experiment account GPU limit drifted")
    if compute.get("max_concurrent_trials") != MAX_CONCURRENT_TRIALS:
        raise ValueError("conditional experiment concurrency limit drifted")
    if MAX_CONCURRENT_TRIALS > ACCOUNT_GPU_LIMIT:
        raise ValueError("conditional experiment concurrency exceeds account GPU limit")
    if compute.get("gpu_fallback_allowed") is not False:
        raise ValueError("conditional experiment GPU fallback must remain disabled")
    try:
        estimate = Decimal(str(compute["planned_upper_cost_usd"]))
        approved = Decimal(str(compute["approved_cost_usd"]))
    except (InvalidOperation, KeyError) as error:
        raise ValueError("conditional experiment cost binding is invalid") from error
    enforce_cost_guardrail(estimated_cost_usd=estimate, approved_cost_usd=approved)


def _require_sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_frozen_source_bundle(*, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    """Reject submission if the local source or dependency bundle drifted."""

    contract = manifest.get("experiment_contract")
    bindings = contract.get("bindings") if isinstance(contract, Mapping) else None
    if not isinstance(bindings, Mapping):
        raise ValueError("conditional manifest lacks source bindings")
    bundle_path = manifest_path.parent / "source-bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError("sibling source-bundle.json is required before submission")
    bundle_bytes = bundle_path.read_bytes()
    if _file_sha256(bundle_path) != _require_sha256(
        bindings.get("source_bundle_sha256"), where="source bundle binding"
    ):
        raise RuntimeError("source bundle content address drifted")
    try:
        bundle = json.loads(bundle_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("source bundle must contain JSON") from error
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("files"), Mapping):
        raise ValueError("source bundle file inventory is invalid")
    if bundle.get("source_glob") != "src/reddit_china_stance/*.py":
        raise RuntimeError("source bundle glob drifted")
    files = bundle["files"]
    if bundle.get("code_sha256") != _canonical_sha256(files):
        raise RuntimeError("source bundle code content address drifted")
    if bundle["code_sha256"] != _require_sha256(bindings.get("code_sha256"), where="code binding"):
        raise RuntimeError("manifest code binding drifted")
    root = _repo_root()
    current_inventory = {
        path.relative_to(root).as_posix()
        for path in (root / "src/reddit_china_stance").glob("*.py")
        if path.is_file()
    }
    if current_inventory != set(files):
        raise RuntimeError("frozen source inventory changed")
    for relative_path, expected_hash in files.items():
        if not isinstance(relative_path, str) or not relative_path.startswith("src/"):
            raise ValueError("source bundle contains an unsafe source path")
        candidate = root / relative_path
        if not candidate.is_file() or _file_sha256(candidate) != _require_sha256(
            expected_hash, where=f"source bundle hash for {relative_path}"
        ):
            raise RuntimeError(f"frozen source changed: {relative_path}")
    lock_path = root / "uv.lock"
    if not lock_path.is_file() or _file_sha256(lock_path) != _require_sha256(
        bindings.get("dependency_lock_sha256"), where="dependency lock binding"
    ):
        raise RuntimeError("frozen uv.lock changed")


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("conditional run manifest must contain a JSON object")
    module = _experiment_module()
    manifest = _require_callable(module, "validate_run_manifest")(value)
    if not isinstance(manifest, dict):
        raise RuntimeError("conditional manifest validator must return an object")
    contract = manifest.get("experiment_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("conditional run manifest lacks an experiment contract")
    validate_compute_contract(contract)
    bindings = contract.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("conditional experiment contract lacks source bindings")
    _require_sha256(bindings.get("source_bundle_sha256"), where="source bundle binding")
    _validate_manifest_inventory(manifest)
    return manifest


def _validate_manifest_inventory(manifest: Mapping[str, Any]) -> None:
    """Guard the launcher even if a lower layer accidentally broadens a phase."""

    phase = manifest.get("phase")
    trials = manifest.get("trials")
    if phase not in {"sweep", "confirmation"} or not isinstance(trials, list):
        raise ValueError("conditional manifest phase or trials are invalid")
    if not trials or len(trials) > MAX_CONCURRENT_TRIALS:
        raise ValueError("conditional manifest trial inventory exceeds the launch boundary")
    if phase == "sweep":
        asha = manifest.get("asha")
        if not isinstance(asha, Mapping) or asha.get("rung_epochs") not in ASHA_RUNGS:
            raise ValueError("conditional sweep manifest lacks a registered ASHA rung")
        expected = ASHA_COUNTS[asha["rung_epochs"]]
        if len(trials) != expected:
            raise ValueError("conditional ASHA rung trial count drifted")
    elif len(trials) != 6:
        raise ValueError("conditional confirmation must contain exactly six fresh 10k trials")
    trial_ids = {trial.get("trial_id") for trial in trials if isinstance(trial, Mapping)}
    if len(trial_ids) != len(trials):
        raise ValueError("conditional manifest trial IDs must be unique")
    if any(
        not isinstance(trial, Mapping) or trial.get("gpu_type") != "L4" for trial in trials
    ):
        raise ValueError("conditional trials must use exactly L4")
    if phase == "confirmation":
        pairs = []
        for trial in trials:
            config = trial.get("config")
            registered = config.get("registered_config") if isinstance(config, Mapping) else None
            if not isinstance(registered, Mapping) or not isinstance(config.get("seed"), int):
                # The experiment validator owns the full trial schema.  This
                # guard adds an exact cross-product check whenever that schema
                # is present to the launcher.
                break
            pairs.append((registered.get("config_sha256"), config["seed"]))
        else:
            config_ids = {config_id for config_id, _ in pairs}
            seeds = {seed for _, seed in pairs}
            if (
                len(config_ids) != 2
                or len(seeds) != 3
                or len(pairs) != len(set(pairs))
                or set(pairs) != {(config_id, seed) for config_id in config_ids for seed in seeds}
            ):
                raise ValueError(
                    "conditional confirmation must be an exact two-recipe by three-seed design"
                )


def _trial_job(manifest: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    module = _experiment_module()
    builder = getattr(module, "make_trial_job", None)
    if callable(builder):
        job = builder(manifest, trial)
    else:
        job = {
            "experiment_run_id": manifest["experiment_run_id"],
            "phase_run_id": manifest["phase_run_id"],
            "experiment_contract": dict(manifest["experiment_contract"]),
            "trial_spec": dict(trial),
            "trial_spec_sha256": _canonical_sha256(trial),
        }
    if not isinstance(job, Mapping):
        raise RuntimeError("conditional trial-job builder must return an object")
    return dict(job)


def _run_root(manifest: Mapping[str, Any]) -> Path:
    return VOLUME_PATH / OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}"


def _inspect_exact_trial(manifest: Mapping[str, Any], trial_id: str) -> dict[str, Any]:
    module = _runtime_module()
    inspector = _require_callable(module, "inspect_trial_output")
    result = inspector(manifest=manifest, trial_id=trial_id, volume_root=VOLUME_PATH)
    if not isinstance(result, Mapping):
        raise RuntimeError("conditional trial inspector must return an object")
    return dict(result)


def _aggregate_inspections(
    manifest: Mapping[str, Any], inspections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    module = _runtime_module()
    aggregate = _require_callable(module, "aggregate_trial_inspections")
    result = aggregate(manifest=manifest, inspections=inspections)
    if not isinstance(result, Mapping):
        raise RuntimeError("conditional inspection aggregator must return an object")
    return dict(result)


def _invoke_training(job: Mapping[str, Any], *, resume: bool) -> Mapping[str, Any]:
    module = _runtime_module()
    runner = _require_callable(module, "run_training_trial")
    result = runner(job=job, volume_root=VOLUME_PATH, resume=resume)
    if not isinstance(result, Mapping):
        raise RuntimeError("conditional training runtime must return an object")
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
    """Run one exact conditional trial; its runtime owns resume and publication."""

    return _invoke_training(job, resume=resume)


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
    """Prove the pinned GPU model path with synthetic tensors only."""

    import torch

    from reddit_china_stance.modernbert_conditional_trainer import (
        ConditionalOptimisationConfig,
        create_modernbert_conditional_state_model,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requires the requested L4 device")
    model = create_modernbert_conditional_state_model(
        optimisation_config=ConditionalOptimisationConfig(encoder_learning_rate=3e-5)
    ).to("cuda")
    model.train()
    batch_size, sequence_length = 2, 16
    input_ids = torch.randint(0, 128, (batch_size, sequence_length), device="cuda")
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device="cuda")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        relevance_labels=torch.tensor([0, 1], device="cuda"),
        target_state_labels=torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], device="cuda"),
    )
    outputs["loss"].backward()
    head_gradients = {
        "relevance": model.relevance_head.weight.grad is not None,
        "target_state": model.target_state_head.weight.grad is not None,
    }
    if not all(head_gradients.values()):
        raise RuntimeError("CUDA preflight did not backpropagate through every conditional head")
    bindings = manifest["experiment_contract"]["bindings"]
    receipt = {
        "status": "passed",
        "kind": "modernbert-conditional-cuda-preflight-v1",
        "gpu_type": "L4",
        "synthetic_rows": batch_size,
        "synthetic_sequence_length": sequence_length,
        "relevance_shape": list(outputs["relevance_logits"].shape),
        "target_state_shape": list(outputs["target_state_logits"].shape),
        "head_gradients": head_gradients,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "source_bundle_sha256": bindings["source_bundle_sha256"],
        "code_sha256": bindings["code_sha256"],
        "dependency_lock_sha256": bindings["dependency_lock_sha256"],
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
    }
    _publish_idempotent_immutable_json(_cuda_preflight_receipt_path(manifest), receipt)
    volume.commit()
    return receipt


def _cuda_preflight_receipt_path(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> Path:
    return (
        volume_root
        / OUTPUT_PREFIX
        / f"run={manifest['experiment_run_id']}"
        / "cuda-preflight-receipts"
        / f"phase={manifest['phase']}"
        / f"phase-run={manifest['phase_run_id']}.json"
    )


def validate_cuda_preflight_receipt(
    manifest: Mapping[str, Any], *, volume_root: Path = VOLUME_PATH
) -> dict[str, Any]:
    """Require the exact persisted synthetic GPU proof before training claims."""

    path = _cuda_preflight_receipt_path(manifest, volume_root=volume_root)
    if not path.is_file():
        raise RuntimeError("training launch requires an exact CUDA preflight receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("CUDA preflight receipt is not JSON") from error
    if not isinstance(receipt, dict):
        raise RuntimeError("CUDA preflight receipt must be an object")
    bindings = manifest["experiment_contract"]["bindings"]
    expected = {
        "status": "passed",
        "kind": "modernbert-conditional-cuda-preflight-v1",
        "gpu_type": "L4",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "source_bundle_sha256": bindings["source_bundle_sha256"],
        "code_sha256": bindings["code_sha256"],
        "dependency_lock_sha256": bindings["dependency_lock_sha256"],
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
        "relevance_shape": [2, 3],
        "target_state_shape": [2, 4, 6],
        "head_gradients": {"relevance": True, "target_state": True},
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("CUDA preflight receipt binding drifted")
    return receipt


def _claim_path(manifest: Mapping[str, Any], trial_id: str) -> Path:
    return (
        _run_root(manifest)
        / "launch-claims"
        / f"phase={manifest['phase']}"
        / f"trial={trial_id}.json"
    )


def _dispatch_path(manifest: Mapping[str, Any], trial_id: str) -> Path:
    return (
        _run_root(manifest)
        / "launch-dispatches"
        / f"phase={manifest['phase']}"
        / f"trial={trial_id}.json"
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
            f"existing launch claim requires manual reconciliation: {path}"
        ) from error


def _publish_idempotent_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError("immutable CUDA preflight receipt already differs")
        return
    _write_immutable_json(path, payload)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
)
def coordinate_training(manifest: dict[str, Any]) -> dict[str, Any]:
    """Claim each missing trial before spawn; never infer a safe retry after a crash."""

    volume.reload()
    validate_cuda_preflight_receipt(manifest)
    inspections = [
        _inspect_exact_trial(manifest, trial["trial_id"]) for trial in manifest["trials"]
    ]
    inspection = _aggregate_inspections(manifest, inspections)
    if inspection.get("resumable_trial_ids"):
        raise RuntimeError("resumable conditional trials require manual reconciliation")
    by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    call_ids: list[str] = []
    for trial_id in inspection.get("missing_trial_ids", []):
        claim_path = _claim_path(manifest, trial_id)
        if claim_path.exists() or _dispatch_path(manifest, trial_id).exists():
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
        dispatch = {**claim, "status": "spawned", "function_call_id": call.object_id}
        _write_immutable_json(_dispatch_path(manifest, trial_id), dispatch)
        volume.commit()
        call_ids.append(call.object_id)
    return {
        "status": "submitted" if call_ids else "already_complete",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "submitted_function_call_ids": call_ids,
        "complete_trial_ids": inspection.get("complete_trial_ids", []),
    }


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def inspect_run(manifest: dict[str, Any]) -> dict[str, Any]:
    inspections = [
        _inspect_exact_trial(manifest, trial["trial_id"]) for trial in manifest["trials"]
    ]
    return _aggregate_inspections(manifest, inspections)


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


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def promote_asha_rung(manifest: dict[str, Any]) -> dict[str, Any]:
    """Publish one exact promotion only after full source-rung validation."""

    inspections = [
        _inspect_exact_trial(manifest, trial["trial_id"]) for trial in manifest["trials"]
    ]
    inspection = _aggregate_inspections(manifest, inspections)
    if inspection.get("missing_trial_ids") or inspection.get("resumable_trial_ids"):
        raise RuntimeError("ASHA promotion requires a complete validated source rung")
    module = _runtime_module()
    promoter = _require_callable(module, "promote_asha_rung")
    result = promoter(manifest=manifest, volume_root=VOLUME_PATH)
    if not isinstance(result, Mapping):
        raise RuntimeError("conditional ASHA promoter must return an object")
    volume.commit()
    return dict(result)


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def prepare_confirmation(manifest: dict[str, Any]) -> dict[str, Any]:
    """Publish the immutable six-trial confirmation manifest on the Volume."""

    module = _runtime_module()
    preparer = _require_callable(module, "prepare_confirmation_manifest")
    result = preparer(manifest=manifest, volume_root=VOLUME_PATH)
    if not isinstance(result, Mapping):
        raise RuntimeError("conditional confirmation preparer must return an object")
    volume.commit()
    return dict(result)


def _load_bound_baseline_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Load only the exact local baseline manifest bound by the experiment."""

    artefacts = manifest.get("experiment_artefacts")
    if not isinstance(artefacts, Mapping):
        raise RuntimeError(
            "confirmation closeout is blocked: the frozen experiment does not bind "
            "an exact baseline manifest path and hash"
        )
    descriptor = artefacts.get("baseline_manifest")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "repo_relative_path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError("confirmation closeout baseline manifest path is not exactly bound")
    relative_value = descriptor["repo_relative_path"]
    relative = PurePosixPath(relative_value) if isinstance(relative_value, str) else None
    if (
        relative is None
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
    ):
        raise RuntimeError("confirmation closeout baseline manifest path is unsafe")
    root = _repo_root().resolve()
    path = root.joinpath(*relative.parts)
    expected_bytes = descriptor["bytes"]
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
        or not path.is_file()
        or path.stat().st_size != expected_bytes
        or _file_sha256(path)
        != _require_sha256(descriptor["sha256"], where="baseline manifest binding")
    ):
        raise RuntimeError("confirmation closeout baseline manifest binding drifted")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("confirmation closeout baseline manifest is not JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("confirmation closeout baseline manifest must be an object")
    from reddit_china_stance.privacy import assert_metadata_only

    assert_metadata_only(value, where="conditional closeout baseline manifest")
    return value


@app.function(image=image, cpu=4, memory=16384, volumes={str(VOLUME_PATH): volume})
def closeout_confirmation(
    manifest: dict[str, Any], baseline_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Close a completed confirmation only with its frozen baseline artefact."""

    module = _runtime_module()
    closeout = _require_callable(module, "closeout_confirmation")
    result = closeout(
        confirmation_manifest=manifest,
        baseline_manifest=baseline_manifest,
        volume_root=VOLUME_PATH,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("conditional confirmation closeout must return metadata")
    volume.commit()
    return dict(result)


def _confirmation(action: str, manifest: Mapping[str, Any]) -> str:
    return (
        f"{action.upper()}_MODERNBERT_CONDITIONAL_"
        f"{str(manifest['phase']).upper()}_{str(manifest['phase_run_id'])[:12]}"
    )


def _validate_action_phase(action: str, manifest: Mapping[str, Any]) -> None:
    """Keep phase-specific mutation authority exact and independently testable."""

    if action == "prepare-confirmation" and (
        manifest["phase"] != "sweep" or manifest["asha"]["rung_epochs"] != 8
    ):
        raise ValueError("prepare-confirmation requires the final ASHA sweep manifest")
    if action == "promote" and (
        manifest["phase"] != "sweep" or manifest["asha"]["rung_epochs"] not in {2, 4}
    ):
        raise ValueError("promote requires a non-final ASHA sweep manifest")
    if action == "closeout-confirmation" and manifest["phase"] != "confirmation":
        raise ValueError("closeout-confirmation requires the exact confirmation phase manifest")


@app.local_entrypoint()
def main(
    action: str = "",
    manifest_path: str = "data/private-modernbert-conditional-v1/sweep-rung2-v5/run-manifest.json",
    approved_cost_usd: str = "200",
    confirm: str = "",
) -> None:
    """Plan/prepare, submit, inspect, validate, promote, or prepare confirmation."""

    actions = {
        "plan",
        "prepare",
        "train",
        "inspect",
        "validate",
        "validate-sharded",
        "promote",
        "prepare-confirmation",
        "cuda-preflight",
        "closeout-confirmation",
    }
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
        raise ValueError("CLI or immutable approval is invalid") from error
    enforce_cost_guardrail(estimated_cost_usd=estimate, approved_cost_usd=approved)
    if approved != immutable_approval:
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
    if action in {"plan", "prepare"}:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    _validate_action_phase(action, manifest)
    if action in {
        "train",
        "promote",
        "prepare-confirmation",
        "cuda-preflight",
        "closeout-confirmation",
    }:
        required = _confirmation(action, manifest)
        if confirm != required:
            raise RuntimeError(f"refusing mutation: pass --confirm {required}")
    if action in {"train", "cuda-preflight", "closeout-confirmation"}:
        verify_frozen_source_bundle(manifest_path=manifest_file, manifest=manifest)
    if action == "inspect":
        summary["inspection"] = inspect_run.remote(manifest)
        summary["status"] = "inspected"
    elif action == "validate-sharded":
        calls = [inspect_trial.spawn(manifest, trial["trial_id"]) for trial in manifest["trials"]]
        inspection = _aggregate_inspections(manifest, [call.get() for call in calls])
        if inspection.get("missing_trial_ids") or inspection.get("resumable_trial_ids"):
            raise RuntimeError("phase validation requires every exact trial to be complete")
        summary["inspection"] = inspection
        summary["status"] = "validated"
    elif action == "validate":
        inspection = inspect_run.remote(manifest)
        if inspection.get("missing_trial_ids") or inspection.get("resumable_trial_ids"):
            raise RuntimeError("phase validation requires every exact trial to be complete")
        summary["inspection"] = inspection
        summary["status"] = "validated"
    elif action == "promote":
        summary["promotion"] = promote_asha_rung.remote(manifest)
        summary["status"] = "promoted"
    elif action == "prepare-confirmation":
        summary["confirmation"] = prepare_confirmation.remote(manifest)
        summary["status"] = "confirmation_prepared"
    elif action == "cuda-preflight":
        summary["cuda_preflight"] = cuda_preflight.remote(manifest)
        summary["status"] = "cuda_preflight_passed"
    elif action == "closeout-confirmation":
        baseline_manifest = _load_bound_baseline_manifest(manifest)
        summary["closeout"] = closeout_confirmation.remote(manifest, baseline_manifest)
        summary["status"] = "closed_out"
    else:  # train
        coordination = coordinate_training.remote(manifest)
        summary["coordination"] = coordination
        summary["submitted_function_call_ids"] = coordination["submitted_function_call_ids"]
        summary["status"] = coordination["status"]
    print(json.dumps(summary, indent=2, sort_keys=True))
