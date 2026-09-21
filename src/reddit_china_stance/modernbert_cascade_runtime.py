"""Private runtime for the separate-model ModernBERT cascade experiment.

Row text, labels, IDs and logits remain inside the private Modal Volume.  The
only public artefacts are content-addressed trial receipts and aggregate-only
development gates.  There is deliberately no locked-test entrypoint.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

from reddit_china_stance.modernbert_cascade_model import (
    TARGET_STATE_LABELS,
    decode_cascade_logits,
)
from reddit_china_stance.modernbert_training import (
    DEVELOPMENT_PROXY_ID,
    ModernBertContractError,
    _load_runtime_inputs,
    _semantic_label_from_teacher_row,
    canonical_sha256,
    file_sha256,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation import (
    RELEVANCE_LABELS,
    STANCE_LABELS,
    TARGET_LABELS,
    score_semantic_labels,
)

SCHEMA_VERSION = "1.0.0"
OUTPUT_PREFIX = Path("student-modernbert-cascade-v1")
GPU_TYPE = "L4"
DEVELOPMENT_ROWS = 222
COMPONENTS = ("relevance", "target_conditioned")


def _experiment_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_cascade_experiment")


def _trainer_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_cascade_trainer")


def _require_sha256(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{where} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite incomplete JSON: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _descriptor(path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _validate_descriptor(root: Path, value: Mapping[str, Any], *, where: str) -> Path:
    if set(value) != {"relative_path", "sha256", "bytes"}:
        raise ValueError(f"{where} descriptor schema drifted")
    relative = Path(str(value["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{where} descriptor path is unsafe")
    path = root / relative
    if (
        type(value["bytes"]) is not int
        or not path.is_file()
        or path.stat().st_size != value["bytes"]
        or file_sha256(path) != _require_sha256(value["sha256"], where=f"{where}.sha256")
    ):
        raise ModernBertContractError(f"{where} artifact is missing or corrupt")
    return path


def _validate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "experiment_run_id",
        "phase_run_id",
        "experiment_contract",
        "trial_spec",
        "trial_spec_sha256",
        "run_manifest_sha256",
    }
    allowed_extras = {"schema_version", "trial_id", "component", "gpu_type"}
    if not required <= set(job) or set(job) - required - allowed_extras:
        raise ValueError("cascade Modal job has unexpected fields")
    contract = job.get("experiment_contract")
    trial = job.get("trial_spec")
    if not isinstance(contract, Mapping) or not isinstance(trial, Mapping):
        raise ValueError("cascade job lacks experiment contract or trial spec")
    experiment = _experiment_module()
    clean_contract = experiment.validate_experiment_contract(contract)
    clean_trial = experiment.validate_trial_spec(trial, experiment=clean_contract)
    if job["experiment_run_id"] != clean_contract["experiment_run_id"]:
        raise ModernBertContractError("cascade job experiment binding drifted")
    _require_sha256(job["phase_run_id"], where="phase_run_id")
    if job["trial_spec_sha256"] != canonical_sha256(clean_trial):
        raise ModernBertContractError("cascade job trial binding drifted")
    _require_sha256(job["run_manifest_sha256"], where="run_manifest_sha256")
    if clean_trial["locked_test_rows_accessed"] != 0 or clean_contract["locked_test"] != {
        "authorised": False,
        "predictions_authorised": False,
        "rows_accessed": 0,
    }:
        raise ModernBertContractError("cascade runtime has no locked-test authority")
    return {
        **{key: deepcopy(job[key]) for key in required},
        "experiment_contract": clean_contract,
        "trial_spec": clean_trial,
    }


def build_epoch_zero_resume_marker(job: Mapping[str, Any]) -> dict[str, Any]:
    clean = _validate_job(job)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-cascade-epoch-zero-resume-v1",
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "trial_id": clean["trial_spec"]["trial_id"],
        "trial_spec_sha256": clean["trial_spec_sha256"],
        "completed_epochs": 0,
        "cumulative_wall_seconds": 0.0,
        "cumulative_gpu_seconds": 0.0,
        "state": "claimed_no_checkpoint",
    }


def _trial_roots(
    *, volume_root: Path, experiment_run_id: str, component: str, trial_id: str
) -> tuple[Path, Path, Path]:
    if component not in COMPONENTS:
        raise ValueError("cascade trial component is invalid")
    root = volume_root / OUTPUT_PREFIX / f"run={experiment_run_id}"
    final = root / "phase=confirmation" / f"component={component}" / f"trial={trial_id}"
    incomplete = (
        root
        / "phase=confirmation.incomplete"
        / f"component={component}"
        / f"trial={trial_id}"
    )
    return root, final, incomplete


def _reconcile_resume_staging(staging_root: Path, *, completed_epochs: int) -> None:
    """Discard only uncommitted next-epoch files after an explicit resume.

    ``resume.json`` is the transaction commit marker.  Files for epochs above
    that marker may exist if the container stopped between atomic writes; they
    are never considered evidence and are deterministically regenerated from
    the committed checkpoint.
    """

    if type(completed_epochs) is not int or completed_epochs < 0:
        raise ValueError("completed_epochs must be a non-negative integer")
    for directory, pattern in (
        (staging_root, "epoch-*-predictions.json"),
        (staging_root, "epoch-*-predictions.json.new"),
        (staging_root / "checkpoints", "epoch-*.pt"),
        (staging_root / "checkpoints", "epoch-*.pt.incomplete"),
    ):
        if not directory.is_dir():
            continue
        for path in directory.glob(pattern):
            match = re.fullmatch(
                r"epoch-(\d+)(?:-predictions\.json(?:\.new)?|\.pt(?:\.incomplete)?)",
                path.name,
            )
            if match is None:
                raise ModernBertContractError("cascade staging contains an invalid epoch file")
            if int(match.group(1)) > completed_epochs:
                path.unlink()
    history_path = staging_root / "history.json"
    for transient in (staging_root / "history.json.new", staging_root / "resume.json.new"):
        if transient.exists():
            if not transient.is_file():
                raise ModernBertContractError("cascade staging transaction file is invalid")
            transient.unlink()
    if history_path.is_file():
        history = _read_json(history_path, where="cascade history").get("epochs")
        if not isinstance(history, list) or len(history) < completed_epochs:
            raise ModernBertContractError("cascade history is behind its resume marker")
        if len(history) > completed_epochs:
            _write_json_atomic(history_path, {"epochs": history[:completed_epochs]})


def _load_inputs(job: Mapping[str, Any], *, volume_root: Path) -> dict[str, Any]:
    bindings = job["experiment_contract"]["bindings"]
    adapted = deepcopy(dict(job))
    adapted["experiment_contract"] = {
        "bindings": {
            "dataset_parquet_sha256": bindings["dataset_sha256"],
            "split_manifest_sha256": bindings["split_manifest_sha256"],
        }
    }
    inputs = _load_runtime_inputs(adapted, volume_root=volume_root, include_development=True)
    reference_binding = {
        "development_proxy_id": DEVELOPMENT_PROXY_ID,
        "split": "development",
        "labels": [
            {
                "source_sample_id": row["source_sample_id"],
                "label": json.loads(row["label_json"]),
            }
            for row in sorted(inputs["development_rows"], key=lambda row: row["source_sample_id"])
        ],
    }
    if canonical_sha256(reference_binding) != bindings["development_reference_sha256"]:
        raise ModernBertContractError("cascade development-reference binding drifted")
    return inputs


def select_teacher_rows(
    teacher_rows: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    trial_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select the exact registered 10k ladder and, for Model B, material rows."""

    seed = trial_spec.get("ladder_seed")
    if type(seed) is not int:
        raise ValueError("cascade trial lacks a ladder seed")
    metadata = split_manifest["ladders"][str(seed)]["budgets"]["10k"]
    if canonical_sha256(metadata) != trial_spec["subset_manifest_sha256"]:
        raise ModernBertContractError("cascade 10k subset binding drifted")
    selected_ids = {
        str(row["sample_id"])
        for row in split_manifest["rows"]
        if row["folds"].get(str(seed)) in set(range(10))
    }
    by_id = {str(row["sample_id"]): dict(row) for row in teacher_rows}
    if len(selected_ids) != 10_000 or len(by_id) != 10_000 or selected_ids != set(by_id):
        raise ModernBertContractError("cascade training input is not the exact 10k teacher set")
    rows = [by_id[item_id] for item_id in sorted(selected_ids)]
    if trial_spec["component"] == "target_conditioned":
        rows = [
            row
            for row in rows
            if _semantic_label_from_teacher_row(row)["relevance"] == "material"
        ]
        if len(rows) != 6_436:
            raise ModernBertContractError(
                "target-conditioned source must contain 6,436 material rows"
            )
    return rows


def relevance_class_weights(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    counts = Counter(_semantic_label_from_teacher_row(row)["relevance"] for row in rows)
    if sum(counts.values()) != 10_000 or any(not counts[label] for label in RELEVANCE_LABELS):
        raise ModernBertContractError("relevance weighting requires all exact teacher rows/classes")
    raw = [math.sqrt(10_000 / counts[label]) for label in RELEVANCE_LABELS]
    mean = sum(raw) / len(raw)
    return tuple(min(4.0, max(0.5, value / mean)) for value in raw)  # type: ignore[return-value]


def build_optimisation_config(
    trial_spec: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]
) -> Any:
    trainer = _trainer_module()
    config = trial_spec["config"]
    if config.get("encoder_learning_rate") != 5e-5 or config.get("effective_batch_size") != 32:
        raise ModernBertContractError("cascade optimiser configuration drifted")
    return trainer.CascadeOptimisationConfig(
        encoder_learning_rate=5e-5,
        head_learning_rate_multiplier=5.0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        warmup_ratio=0.06,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=4,
        effective_batch_size=32,
        use_bf16=True,
        gradient_checkpointing=False,
        relevance_class_weights=(
            relevance_class_weights(train_rows)
            if trial_spec["component"] == "relevance"
            else None
        ),
        target_state_class_weights=None,
    )


def _semantic_label(row: Mapping[str, Any], *, teacher: bool) -> dict[str, Any]:
    if teacher:
        return _semantic_label_from_teacher_row(row)
    try:
        label = json.loads(row["label_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("cascade development row contains invalid label JSON") from error
    if not isinstance(label, dict):
        raise ValueError("cascade semantic label must be an object")
    return label


def _tokenise_component_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    component: str,
    teacher: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    trainer = _trainer_module()
    item_field = "sample_id" if teacher else "source_sample_id"
    encoded: list[dict[str, Any]] = []
    reference: dict[str, dict[str, Any]] = {}
    for raw in rows:
        item_id = raw.get(item_field)
        if not isinstance(item_id, str) or not item_id or item_id in reference:
            raise ValueError("cascade rows contain invalid or duplicate IDs")
        label = _semantic_label(raw, teacher=teacher)
        reference[item_id] = label
        if component == "relevance":
            encoded.append(
                {
                    "item_id": item_id,
                    **trainer.tokenise_relevance_record(tokenizer, raw),
                    **trainer.encode_relevance_label(label),
                }
            )
        else:
            for target in TARGET_LABELS:
                encoded.append(
                    {
                        "item_id": item_id,
                        **trainer.tokenise_target_conditioned_record(
                            tokenizer, raw, target=target
                        ),
                        **trainer.encode_target_conditioned_label(label, target=target),
                    }
                )
    expected = len(rows) if component == "relevance" else len(rows) * 4
    if len(encoded) != expected:
        raise ModernBertContractError("cascade tokenisation did not conserve component rows")
    return encoded, reference


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _relevance_metrics(
    reference: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_id = {row["source_sample_id"]: row["logits"] for row in rows}
    if set(by_id) != set(reference) or len(by_id) != DEVELOPMENT_ROWS:
        raise ModernBertContractError("relevance predictions do not conserve development rows")
    counts = {label: [0, 0, 0] for label in RELEVANCE_LABELS}
    for item_id, expected in reference.items():
        predicted = RELEVANCE_LABELS[max(range(3), key=by_id[item_id].__getitem__)]
        actual = expected["relevance"]
        for label in RELEVANCE_LABELS:
            if actual == label and predicted == label:
                counts[label][0] += 1
            elif actual != label and predicted == label:
                counts[label][1] += 1
            elif actual == label and predicted != label:
                counts[label][2] += 1
    f1 = {label: _binary_f1(*counts[label]) for label in RELEVANCE_LABELS}
    tp, _, fn = counts["material"]
    recall = tp / (tp + fn) if tp + fn else 0.0
    macro = sum(f1.values()) / len(f1)
    return {
        "selected_epoch": 1,
        "invalid_outputs": 0,
        "relevance_macro_f1": macro,
        "material_recall": recall,
        "checkpoint_score": 0.5 * macro + 0.5 * recall,
    }


def _target_metrics(
    reference: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_key = {(row["source_sample_id"], row["target"]): row["logits"] for row in rows}
    if len(by_key) != DEVELOPMENT_ROWS * 4:
        raise ModernBertContractError("target predictions do not conserve development target slots")
    item_ids = sorted(reference)
    target_logits = [
        [by_key[(item_id, target)] for target in TARGET_LABELS] for item_id in item_ids
    ]
    gold_relevance_logits = []
    for item_id in item_ids:
        vector = [-1.0, -1.0, -1.0]
        vector[RELEVANCE_LABELS.index(reference[item_id]["relevance"])] = 1.0
        gold_relevance_logits.append(vector)
    decoded = decode_cascade_logits(gold_relevance_logits, target_logits)
    semantic = score_semantic_labels(
        reference, dict(zip(item_ids, decoded.predictions, strict=True))
    )
    presence = [0, 0, 0]
    stance_total = stance_correct = 0
    stance_counts = {stance: [0, 0, 0] for stance in STANCE_LABELS}
    for item_id in item_ids:
        expected = reference[item_id]
        if expected["relevance"] != "material":
            continue
        expected_map = {row["target"]: row["stance"] for row in expected["target_stances"]}
        for target in TARGET_LABELS:
            state = max(range(6), key=by_key[(item_id, target)].__getitem__)
            present = state != 0
            actual_present = target in expected_map
            if present and actual_present:
                presence[0] += 1
            elif present:
                presence[1] += 1
            elif actual_present:
                presence[2] += 1
            if actual_present:
                stance_total += 1
                actual = expected_map[target]
                predicted = TARGET_STATE_LABELS[state] if present else "absent"
                if predicted == actual:
                    stance_correct += 1
                for stance in STANCE_LABELS:
                    if actual == stance and predicted == stance:
                        stance_counts[stance][0] += 1
                    elif actual != stance and predicted == stance:
                        stance_counts[stance][1] += 1
                    elif actual == stance and predicted != stance:
                        stance_counts[stance][2] += 1
    macro = sum(_binary_f1(*stance_counts[state]) for state in STANCE_LABELS) / len(
        STANCE_LABELS
    )
    tuple_f1 = semantic["end_to_end_core_target_stance"]["micro"]["f1"] or 0.0
    return {
        "selected_epoch": 1,
        "invalid_outputs": semantic["invalid_outputs"],
        "target_presence_f1_gold_relevance": _binary_f1(*presence),
        "stance_accuracy_present_targets": stance_correct / stance_total if stance_total else 0.0,
        "stance_macro_f1_present_targets": macro,
        "core_target_stance_tuple_micro_f1_gold_relevance": tuple_f1,
        "checkpoint_score": tuple_f1,
    }


def _component_metrics(
    component: str,
    reference: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
) -> dict[str, Any]:
    metrics = (
        _relevance_metrics(reference, rows)
        if component == "relevance"
        else _target_metrics(reference, rows)
    )
    metrics["selected_epoch"] = epoch
    return metrics


def _normalise_prediction_rows(component: str, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = result.get("prediction_payload")
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise RuntimeError("cascade evaluation did not return private predictions")
    clean = []
    for row in rows:
        source_id = row["source_sample_id"]
        if component == "relevance":
            clean.append({"source_sample_id": source_id, "logits": row["relevance_logits"]})
        else:
            clean.append(
                {
                    "source_sample_id": source_id,
                    "target": row["target"],
                    "logits": row["target_state_logits"],
                }
            )
    return clean


def _execute_trial_runtime(
    *,
    job: Mapping[str, Any],
    inputs: Mapping[str, Any],
    staging_root: Path,
    resume: bool,
    invocation_started: float,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("cascade ModernBERT training requires CUDA")
    trainer = _trainer_module()
    spec = job["trial_spec"]
    component = spec["component"]
    source_rows = select_teacher_rows(
        inputs["teacher_rows"], inputs["split_manifest"], trial_spec=spec
    )
    optimisation = build_optimisation_config(spec, source_rows)
    seed = int(spec["optimiser_seed"])
    trainer.seed_everything(seed)
    tokenizer = trainer.load_pinned_tokenizer(trainer.CascadeModelConfig())
    train_encoded, _ = _tokenise_component_rows(
        source_rows, tokenizer=tokenizer, component=component, teacher=True
    )
    dev_encoded, reference = _tokenise_component_rows(
        inputs["development_rows"],
        tokenizer=tokenizer,
        component=component,
        teacher=False,
    )
    if len(reference) != DEVELOPMENT_ROWS:
        raise ModernBertContractError("cascade evaluation requires exactly 222 rows")
    create_component = getattr(trainer, "create_modernbert_cascade_component", None)
    if not callable(create_component):
        raise RuntimeError("cascade trainer lacks single-component model construction")
    model_config = trainer.CascadeModelConfig()
    model = create_component(
        component=component,
        optimisation_config=optimisation,
        model_config=model_config,
    ).to("cuda")
    optimizer = trainer.create_adamw(model, optimisation)
    microbatch = optimisation.effective_batch_size // optimisation.gradient_accumulation_steps
    if microbatch != 8:
        raise ModernBertContractError("cascade microbatch binding drifted")
    collator = (
        trainer.RelevanceDynamicPaddingCollator(tokenizer)
        if component == "relevance"
        else trainer.TargetConditionedDynamicPaddingCollator(tokenizer)
    )
    dev_batches = trainer.build_length_bucket_batches(
        [len(row["input_ids"]) for row in dev_encoded], batch_size=microbatch, seed=0, epoch=0
    )
    dev_loader = DataLoader(
        dev_encoded, batch_sampler=dev_batches, collate_fn=collator, num_workers=0
    )
    target_epochs = int(spec["target_epochs"])
    updates_per_epoch = math.ceil(math.ceil(len(train_encoded) / microbatch) / 4)
    total_updates = updates_per_epoch * target_epochs
    warmup = int(total_updates * optimisation.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        remaining = max(1, total_updates - warmup)
        return max(0.0, (total_updates - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    history_path = staging_root / "history.json"
    history: list[dict[str, Any]] = []
    start_epoch = global_step = optimizer_step = 0
    marker = _read_json(staging_root / "resume.json", where="cascade resume marker")
    prior_wall_seconds = float(marker.get("cumulative_wall_seconds", 0.0))
    prior_gpu_seconds = float(marker.get("cumulative_gpu_seconds", 0.0))
    if min(prior_wall_seconds, prior_gpu_seconds) < 0:
        raise ModernBertContractError("cascade resume cost counters are invalid")
    if resume and marker["completed_epochs"]:
        checkpoint = marker.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ModernBertContractError("cascade resume marker lacks checkpoint")
        checkpoint_path = staging_root / checkpoint["relative_path"]
        payload = trainer.load_checkpoint_exact(
            checkpoint_path,
            expected_file_sha256=checkpoint["sha256"],
            expected_binding_sha256=checkpoint["binding_sha256"],
            map_location="cuda",
        )
        if payload["binding"]["trial_manifest_sha256"] != job["trial_spec_sha256"]:
            raise ModernBertContractError("cascade resume checkpoint belongs to another trial")
        trainer.restore_training_state(
            payload, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=True
        )
        start_epoch = int(payload["binding"]["epoch"])
        global_step = int(payload["binding"]["global_step"])
        optimizer_step = int(payload["binding"]["optimizer_step"])
        history = list(_read_json(history_path, where="cascade history")["epochs"])
        if len(history) != start_epoch:
            raise ModernBertContractError("cascade resume history and checkpoint disagree")
    elif resume and marker != build_epoch_zero_resume_marker(job):
        raise ModernBertContractError("cascade epoch-zero resume marker drifted")
    torch.cuda.reset_peak_memory_stats()
    train_lengths = [len(row["input_ids"]) for row in train_encoded]
    for epoch in range(start_epoch + 1, target_epochs + 1):
        batches = trainer.build_length_bucket_batches(
            train_lengths, batch_size=microbatch, seed=seed, epoch=epoch
        )
        loader = DataLoader(
            train_encoded,
            batch_sampler=batches,
            collate_fn=collator,
            num_workers=0,
        )
        trained = trainer.train_component_epoch(
            model,
            loader,
            optimizer,
            component=component,
            config=optimisation,
            device="cuda",
            scheduler=scheduler,
        )
        global_step += int(trained["batches"])
        optimizer_step += int(trained["optimizer_steps"])
        evaluated = trainer.evaluate_component_epoch(
            model,
            dev_loader,
            component=component,
            device="cuda",
            config=optimisation,
            collect_predictions=True,
        )
        private_rows = _normalise_prediction_rows(component, evaluated)
        metrics = _component_metrics(component, reference, private_rows, epoch=epoch)
        predictions = _experiment_module().build_private_development_predictions(
            job["experiment_contract"], spec, rows=private_rows
        )
        prediction_path = staging_root / f"epoch-{epoch:02d}-predictions.json"
        _write_json_atomic(prediction_path, predictions)
        payload = trainer.build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            optimisation_config=optimisation,
            binding={
                "dataset_sha256": job["experiment_contract"]["bindings"]["dataset_sha256"],
                "split_manifest_sha256": job["experiment_contract"]["bindings"][
                    "split_manifest_sha256"
                ],
                "trial_manifest_sha256": job["trial_spec_sha256"],
                "code_sha256": job["experiment_contract"]["bindings"]["code_sha256"],
            },
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            seed=seed,
        )
        checkpoint_path = staging_root / "checkpoints" / f"epoch-{epoch:02d}.pt"
        descriptor = trainer.save_checkpoint_atomic(payload, checkpoint_path)
        history.append(
            {
                "epoch": epoch,
                "train_mean_loss": trained["mean_loss"],
                "development_mean_loss": evaluated["mean_loss"],
                "aggregate_metrics": metrics,
                "checkpoint": {
                    "relative_path": str(checkpoint_path.relative_to(staging_root)),
                    "sha256": descriptor["sha256"],
                    "bytes": descriptor["bytes"],
                    "binding_sha256": payload["binding_sha256"],
                },
                "predictions": _descriptor(prediction_path, root=staging_root),
            }
        )
        _write_json_atomic(history_path, {"epochs": history})
        _write_json_atomic(
            staging_root / "resume.json",
            {
                **build_epoch_zero_resume_marker(job),
                "completed_epochs": epoch,
                "cumulative_wall_seconds": prior_wall_seconds
                + (time.monotonic() - invocation_started),
                "cumulative_gpu_seconds": prior_gpu_seconds
                + (time.monotonic() - invocation_started),
                "state": "checkpointed",
                "checkpoint": history[-1]["checkpoint"],
            },
        )
    key = "checkpoint_score"
    selected = max(history, key=lambda row: (row["aggregate_metrics"][key], -row["epoch"]))
    invocation_seconds = time.monotonic() - invocation_started
    return {
        "history": history,
        "selected": selected,
        "wall_seconds": prior_wall_seconds + invocation_seconds,
        "gpu_seconds": prior_gpu_seconds + invocation_seconds,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _public_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    selected = deepcopy(result["selected"]["aggregate_metrics"])
    selected["selected_epoch"] = result["selected"]["epoch"]
    return selected


def run_training_trial(
    *, job: Mapping[str, Any], volume_root: Path, resume: bool = False
) -> dict[str, Any]:
    """Run or explicitly resume one trial and atomically publish exact artefacts."""

    clean = _validate_job(job)
    spec = clean["trial_spec"]
    _, final_root, staging_root = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=clean["experiment_run_id"],
        component=spec["component"],
        trial_id=spec["trial_id"],
    )
    publication_root = final_root.with_name(f"{final_root.name}.publishing")
    if final_root.exists():
        receipt = _read_json(final_root / "receipt.json", where="cascade receipt")
        validate_trial_artifacts(
            final_root,
            receipt,
            experiment=clean["experiment_contract"],
            trial_spec=spec,
            phase_run_id=clean["phase_run_id"],
            run_manifest_sha256=clean["run_manifest_sha256"],
        )
        if staging_root.exists() or publication_root.exists():
            if not resume:
                raise RuntimeError("valid final trial has stale staging; explicit resume required")
            for path in (staging_root, publication_root):
                if path.exists():
                    shutil.rmtree(path)
        return {
            "status": "complete",
            "experiment_run_id": clean["experiment_run_id"],
            "trial_id": spec["trial_id"],
            "component": spec["component"],
            "receipt_id": receipt["receipt_id"],
            "estimated_cost_usd": receipt["compute"]["estimated_cost_usd"],
        }
    if resume:
        if not staging_root.is_dir() or not (staging_root / "resume.json").is_file():
            raise RuntimeError("cascade resume requested without recoverable staging evidence")
        marker = _read_json(staging_root / "resume.json", where="cascade resume marker")
        _reconcile_resume_staging(
            staging_root, completed_epochs=int(marker.get("completed_epochs", -1))
        )
        if publication_root.exists():
            shutil.rmtree(publication_root)
    else:
        if staging_root.exists() or publication_root.exists():
            raise RuntimeError("incomplete cascade trial requires explicit resume")
        staging_root.mkdir(parents=True)
        _write_json_atomic(staging_root / "resume.json", build_epoch_zero_resume_marker(clean))
    function_started = time.monotonic()
    inputs = _load_inputs(clean, volume_root=volume_root)
    result = _execute_trial_runtime(
        job=clean,
        inputs=inputs,
        staging_root=staging_root,
        resume=resume,
        invocation_started=function_started,
    )
    selected = result["selected"]
    selected_epoch = selected["epoch"]
    selected_checkpoint = staging_root / selected["checkpoint"]["relative_path"]
    selected_predictions = staging_root / selected["predictions"]["relative_path"]
    publication_root.mkdir(parents=True)
    checkpoint_final = publication_root / "checkpoint.pt"
    predictions_final = publication_root / "development-predictions.json"
    shutil.copy2(selected_checkpoint, checkpoint_final)
    shutil.copy2(selected_predictions, predictions_final)
    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-cascade-trial-metrics-v1",
        "component": spec["component"],
        "selected_epoch": selected_epoch,
        "aggregate_metrics": _public_metrics(result),
        "epoch_history": [
            {
                "epoch": row["epoch"],
                "train_mean_loss": row["train_mean_loss"],
                "development_mean_loss": row["development_mean_loss"],
                "checkpoint_score": row["aggregate_metrics"]["checkpoint_score"],
            }
            for row in result["history"]
        ],
        "peak_gpu_bytes": result["peak_gpu_bytes"],
    }
    assert_metadata_only(metrics_payload, where="cascade trial metrics")
    metrics_path = publication_root / "metrics.json"
    _write_json_atomic(metrics_path, metrics_payload)
    artifacts = {
        "checkpoint": _descriptor(checkpoint_final, root=publication_root),
        "metrics": _descriptor(metrics_path, root=publication_root),
        "private_development_predictions": _descriptor(
            predictions_final, root=publication_root
        ),
    }
    receipt = _experiment_module().build_trial_receipt(
        clean["experiment_contract"],
        spec,
        phase_run_id=clean["phase_run_id"],
        run_manifest_sha256=clean["run_manifest_sha256"],
        artifacts=artifacts,
        metrics=_public_metrics(result),
        wall_seconds=result["wall_seconds"],
        gpu_seconds=result["gpu_seconds"],
    )
    _write_json_atomic(publication_root / "receipt.json", receipt)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(publication_root, final_root)
    shutil.rmtree(staging_root)
    for parent in (staging_root.parent, staging_root.parent.parent):
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    return {
        "status": "complete",
        "experiment_run_id": clean["experiment_run_id"],
        "trial_id": spec["trial_id"],
        "component": spec["component"],
        "receipt_id": receipt["receipt_id"],
        "estimated_cost_usd": receipt["compute"]["estimated_cost_usd"],
    }


def validate_trial_artifacts(
    trial_root: Path,
    receipt: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase_run_id: str,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    clean = _experiment_module().validate_trial_receipt(experiment, trial_spec, receipt)
    if clean["phase_run_id"] != phase_run_id or clean["run_manifest_sha256"] != run_manifest_sha256:
        raise ModernBertContractError("cascade receipt run-manifest binding drifted")
    paths = {
        name: _validate_descriptor(trial_root, descriptor, where=name)
        for name, descriptor in clean["artifacts"].items()
    }
    predictions = _read_json(
        paths["private_development_predictions"], where="cascade private predictions"
    )
    _experiment_module().validate_private_development_predictions(
        predictions,
        expected_experiment_run_id=experiment["experiment_run_id"],
        expected_trial_id=trial_spec["trial_id"],
    )
    metrics = _read_json(paths["metrics"], where="cascade metrics")
    if metrics.get("aggregate_metrics") != clean["aggregate_metrics"]:
        raise ModernBertContractError("cascade receipt and metrics artifact disagree")
    expected_files = {
        "receipt.json",
        *[descriptor["relative_path"] for descriptor in clean["artifacts"].values()],
    }
    observed = {
        str(path.relative_to(trial_root))
        for path in trial_root.rglob("*")
        if path.is_file()
    }
    if observed != expected_files:
        raise ModernBertContractError("cascade final trial artifact inventory drifted")
    return clean


def inspect_trial_output(
    *, manifest: Mapping[str, Any], trial_id: str, volume_root: Path
) -> dict[str, Any]:
    trials = manifest.get("trials")
    if not isinstance(trials, list):
        raise ValueError("cascade inspection manifest lacks trials")
    matches = [trial for trial in trials if trial.get("trial_id") == trial_id]
    if len(matches) != 1:
        raise ValueError("cascade inspection trial is absent or duplicated")
    trial = matches[0]
    _, final_root, incomplete_root = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=str(manifest["experiment_run_id"]),
        component=trial["component"],
        trial_id=trial_id,
    )
    if final_root.exists() and incomplete_root.exists():
        raise ModernBertContractError("cascade trial has final and incomplete outputs")
    if final_root.is_dir():
        receipt = _read_json(final_root / "receipt.json", where="cascade receipt")
        validate_trial_artifacts(
            final_root,
            receipt,
            experiment=manifest["experiment_contract"],
            trial_spec=trial,
            phase_run_id=manifest["phase_run_id"],
            run_manifest_sha256=canonical_sha256(manifest),
        )
        return {
            "trial_id": trial_id,
            "component": trial["component"],
            "state": "complete",
            "receipt": {
                "receipt_id": receipt["receipt_id"],
                "estimated_cost_usd": receipt["compute"]["estimated_cost_usd"],
            },
        }
    if incomplete_root.is_dir():
        marker = _read_json(incomplete_root / "resume.json", where="cascade resume marker")
        return {
            "trial_id": trial_id,
            "component": trial["component"],
            "state": "resumable",
            "completed_epochs": marker.get("completed_epochs"),
            "receipt": None,
        }
    return {
        "trial_id": trial_id,
        "component": trial["component"],
        "state": "missing",
        "receipt": None,
    }


def aggregate_trial_inspections(
    *, manifest: Mapping[str, Any], inspections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {trial["trial_id"] for trial in manifest["trials"]}
    observed = {row.get("trial_id") for row in inspections}
    if len(inspections) != len(expected) or observed != expected:
        raise ValueError("cascade inspections do not conserve the trial inventory")
    complete = sorted(row["trial_id"] for row in inspections if row["state"] == "complete")
    resumable = sorted(row["trial_id"] for row in inspections if row["state"] == "resumable")
    missing = sorted(row["trial_id"] for row in inspections if row["state"] == "missing")
    unknown = [
        row
        for row in inspections
        if row["state"] not in {"complete", "resumable", "missing"}
    ]
    if unknown:
        raise ValueError("cascade inspection contains an unknown state")
    cost = sum(
        (
            Decimal(str(row["receipt"]["estimated_cost_usd"]))
            for row in inspections
            if row["receipt"] is not None
        ),
        start=Decimal("0"),
    )
    result = {
        "status": (
            "complete" if len(complete) == 6 and not resumable and not missing else "incomplete"
        ),
        "expected_trials": 6,
        "complete_trial_ids": complete,
        "resumable_trial_ids": resumable,
        "missing_trial_ids": missing,
        "estimated_cost_usd": format(cost.quantize(Decimal("0.000001")), "f"),
    }
    assert_metadata_only(result, where="cascade trial inspection")
    return result


def _load_component_predictions(
    *, manifest: Mapping[str, Any], trial: Mapping[str, Any], volume_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, root, _ = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=manifest["experiment_run_id"],
        component=trial["component"],
        trial_id=trial["trial_id"],
    )
    receipt = _read_json(root / "receipt.json", where="cascade receipt")
    validate_trial_artifacts(
        root,
        receipt,
        experiment=manifest["experiment_contract"],
        trial_spec=trial,
        phase_run_id=manifest["phase_run_id"],
        run_manifest_sha256=canonical_sha256(manifest),
    )
    descriptor = receipt["artifacts"]["private_development_predictions"]
    predictions = _read_json(
        root / descriptor["relative_path"], where="cascade private predictions"
    )
    clean = _experiment_module().validate_private_development_predictions(
        predictions,
        expected_experiment_run_id=manifest["experiment_run_id"],
        expected_trial_id=trial["trial_id"],
    )
    return receipt, clean


def _candidate_frames(
    *, manifest: Mapping[str, Any], volume_root: Path, reference: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    receipt_ids: list[str] = []
    for trial in manifest["trials"]:
        pair = (trial["ladder_seed"], trial["optimiser_seed"])
        receipt, payload = _load_component_predictions(
            manifest=manifest, trial=trial, volume_root=volume_root
        )
        by_pair.setdefault(pair, {})[trial["component"]] = payload
        receipt_ids.append(receipt["receipt_id"])
    if any(set(parts) != set(COMPONENTS) for parts in by_pair.values()) or len(by_pair) != 3:
        raise ModernBertContractError("cascade closeout lacks exact paired components")
    from reddit_china_stance.modernbert_conditional_runtime import conditional_diagnostics

    results = []
    for (ladder_seed, optimiser_seed), parts in sorted(by_pair.items()):
        relevance_rows = parts["relevance"]["rows"]
        target_rows = parts["target_conditioned"]["rows"]
        item_ids = [row["source_sample_id"] for row in relevance_rows]
        by_target = {
            (row["source_sample_id"], row["target"]): row["logits"] for row in target_rows
        }
        nested = [
            [by_target[(item_id, target)] for target in TARGET_LABELS] for item_id in item_ids
        ]
        decoded = decode_cascade_logits([row["logits"] for row in relevance_rows], nested)
        predictions = dict(zip(item_ids, decoded.predictions, strict=True))
        if set(predictions) != set(reference):
            raise ModernBertContractError("cascade closeout does not conserve development IDs")
        metrics = score_semantic_labels(reference, predictions)
        diagnostics = conditional_diagnostics(reference, predictions)
        results.append(
            {
                "condition": {
                    "ladder_seed": ladder_seed,
                    "optimiser_seed": optimiser_seed,
                },
                "core_tuple_f1": metrics["end_to_end_core_target_stance"]["micro"]["f1"],
                "material_recall": metrics["relevance"]["material_recall"],
                "relevance_macro_f1": metrics["relevance"]["macro_f1"],
                "core_target_f1": {
                    target: metrics["targets"]["per_class"][target]["f1"]
                    for target in TARGET_LABELS[:3]
                },
                "core_target_support": {
                    target: metrics["support"]["reference_targets"][target]
                    for target in TARGET_LABELS[:3]
                },
                "target_stance_diagnostics": diagnostics["target_stance_diagnostics"],
                "invalid_outputs": metrics["invalid_outputs"],
                "forced_target_selections": decoded.forced_target_selections,
            }
        )
    return results, sorted(receipt_ids)


def _baseline_frames(
    *,
    baseline_manifest: Mapping[str, Any],
    volume_root: Path,
    reference: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from reddit_china_stance.modernbert_conditional_runtime import _baseline_prediction_frame

    frames, diagnostics, _ = _baseline_prediction_frame(
        baseline_manifest=baseline_manifest, volume_root=volume_root, reference=reference
    )
    baseline_by_pair = {
        (row["condition"]["ladder_seed"], row["condition"]["optimiser_seed"]): row
        for row in frames
    }
    # The validated baseline receipt stores relevance macro-F1, which the old
    # conditional gate frame did not need to copy.
    run_root = (
        volume_root
        / "student-modernbert-v1"
        / f"run={baseline_manifest['experiment_run_id']}"
        / "phase=confirmatory"
    )
    for trial in baseline_manifest["trials"]:
        config = trial.get("config", {})
        if config.get("label_budget") != 10_000:
            continue
        pair = (config["ladder_seed"], config["optimiser_seed"])
        receipt = _read_json(
            run_root / f"trial={trial['trial_id']}" / "receipt.json", where="baseline receipt"
        )
        baseline_by_pair[pair]["metric_payload"]["relevance_macro_f1"] = receipt[
            "aggregate_metrics"
        ]["relevance_macro_f1"]
        pair_id = f"{pair[0]}:{pair[1]}"
        baseline_by_pair[pair]["metric_payload"]["target_stance_diagnostics"] = diagnostics[
            pair_id
        ]["target_stance_diagnostics"]
    return list(baseline_by_pair.values())


def _evaluate_gate(
    *,
    experiment: Mapping[str, Any],
    phase_run_id: str,
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    candidate_receipt_ids: Sequence[str],
) -> dict[str, Any]:
    return _experiment_module().evaluate_continuation_gate(
        experiment,
        phase_run_id=phase_run_id,
        baseline_results=baseline,
        candidate_results=candidate,
        candidate_trial_receipt_ids=candidate_receipt_ids,
    )


def closeout_confirmation(
    *,
    confirmation_manifest: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    volume_root: Path,
) -> dict[str, Any]:
    """Privately pair A+B per seed, recompute the cascade, and freeze one verdict."""

    experiment_module = _experiment_module()
    manifest = experiment_module.validate_run_manifest(confirmation_manifest)
    if manifest["locked_test"]["rows_accessed"] != 0:
        raise ModernBertContractError("cascade closeout cannot access the locked test")
    from reddit_china_stance.modernbert_conditional_runtime import _load_development_reference

    reference = _load_development_reference(
        volume_root=volume_root,
        development_reference_sha256=manifest["experiment_contract"]["bindings"][
            "development_reference_sha256"
        ],
    )
    candidate, receipt_ids = _candidate_frames(
        manifest=manifest, volume_root=volume_root, reference=reference
    )
    baseline = _baseline_frames(
        baseline_manifest=baseline_manifest, volume_root=volume_root, reference=reference
    )
    gate = _evaluate_gate(
        experiment=manifest["experiment_contract"],
        phase_run_id=manifest["phase_run_id"],
        baseline=baseline,
        candidate=candidate,
        candidate_receipt_ids=receipt_ids,
    )
    output_root = (
        volume_root
        / OUTPUT_PREFIX
        / f"run={manifest['experiment_run_id']}"
        / "closeout"
    )
    descriptor = experiment_module.publish_immutable_json(
        output_root / "continuation-gate.json", gate
    )
    result = {
        "status": "complete",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "development_rows": DEVELOPMENT_ROWS,
        "component_trials": 6,
        "paired_cascades": 3,
        "gate_receipt_id": gate["gate_receipt_id"],
        "gate_artifact": descriptor,
        "verdict": gate["verdict"],
        "aggregate_evidence": gate["aggregate_evidence"],
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="cascade closeout summary")
    return result
