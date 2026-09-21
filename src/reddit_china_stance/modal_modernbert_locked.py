"""One-time Modal evaluation of the frozen ModernBERT student locked test."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from reddit_china_stance.modal_modernbert import (
    HARD_MAX_APPROVAL_USD,
    OUTPUT_PREFIX,
    VOLUME_PATH,
    app,
    image,
    inspect_trial,
    validate_run_manifest,
    volume,
)
from reddit_china_stance.modernbert_locked_test import (
    LOCKED_TEST_ROWS,
    PREDICTION_KIND,
    TARGET_THRESHOLD,
    build_final_receipt,
    build_locked_test_aggregate,
    build_locked_test_authorisation,
    build_locked_test_manifest,
    build_locked_trial_receipt,
    validate_locked_test_manifest,
    validate_locked_trial_receipt,
    validate_private_locked_predictions,
)
from reddit_china_stance.modernbert_training import (
    DEVELOPMENT_PROXY_PARQUET_NAME,
    VOLUME_INPUT_PREFIX,
    _load_runtime_inputs,
    _optimisation_config,
    _selected_teacher_rows,
    _tokenise_rows,
    canonical_sha256,
    file_sha256,
    publish_immutable_json,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation import score_semantic_labels

LOCKED_TEST_PHASE_COST_ESTIMATE_USD = Decimal("5")
LOCKED_OUTPUT_NAME = "locked-test"
MANIFEST_FILE = "manifest.json"
AUTHORISATION_FILE = "authorisation.json"
PREDICTIONS_FILE = "predictions.json"
TRIAL_RECEIPT_FILE = "receipt.json"
AGGREGATE_FILE = "aggregate.json"
FINAL_RECEIPT_FILE = "final-receipt.json"


def _json_object(path: Path, *, where: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _file_descriptor(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    return {
        "relative_path": relative_path or path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _safe_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("artifact path must be a string")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("artifact path is unsafe")
    return root.joinpath(*relative.parts)


def _source_run_root(source_manifest: Mapping[str, Any]) -> Path:
    return VOLUME_PATH / OUTPUT_PREFIX / f"run={source_manifest['experiment_run_id']}"


def _locked_root(source_experiment_run_id: str, locked_test_run_id: str) -> Path:
    return (
        VOLUME_PATH
        / OUTPUT_PREFIX
        / f"run={source_experiment_run_id}"
        / LOCKED_OUTPUT_NAME
        / f"run={locked_test_run_id}"
    )


def _load_locked_manifest(source_experiment_run_id: str, locked_test_run_id: str) -> dict[str, Any]:
    path = _locked_root(source_experiment_run_id, locked_test_run_id) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError("exact locked-test manifest is absent")
    clean = validate_locked_test_manifest(_json_object(path, where="locked-test manifest"))
    if clean["locked_test_run_id"] != locked_test_run_id:
        raise ValueError("locked-test run path and manifest differ")
    return clean


def _source_receipts(source_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    run_root = _source_run_root(source_manifest)
    receipts = {}
    for trial in source_manifest["trials"]:
        trial_root = run_root / "phase=confirmatory" / f"trial={trial['trial_id']}"
        path = trial_root / TRIAL_RECEIPT_FILE
        if not path.is_file():
            raise FileNotFoundError("confirmatory trial receipt is missing")
        receipt = _json_object(path, where="confirmatory trial receipt")
        _validate_source_receipt(source_manifest, trial, receipt)
        receipts[trial["trial_id"]] = receipt
    return receipts


def _validate_source_receipt(
    source_manifest: Mapping[str, Any],
    trial: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
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
        raise ValueError("confirmatory source receipt schema drifted")
    expected = {
        "status": "complete",
        "experiment_run_id": source_manifest["experiment_run_id"],
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": canonical_sha256(trial),
        "bindings": source_manifest["experiment_contract"]["bindings"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("confirmatory source receipt binding drifted")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if receipt.get("receipt_id") != canonical_sha256(body):
        raise ValueError("confirmatory source receipt content address drifted")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or "checkpoint" not in artifacts:
        raise ValueError("confirmatory source receipt lacks a checkpoint")
    return dict(receipt)


def _prepare_locked_test(
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    evaluator_code_sha256: str,
    dependency_lock_sha256: str,
) -> dict[str, Any]:
    source = validate_run_manifest(source_manifest)
    if source["phase"] != "confirmatory":
        raise ValueError("locked test requires the exact confirmatory source phase")
    receipts = _source_receipts(source)
    authorisation = build_locked_test_authorisation(source, list(receipts.values()))
    locked = build_locked_test_manifest(
        source,
        confirmatory_receipts=list(receipts.values()),
        authorisation=authorisation,
        source_manifest_sha256=source_manifest_sha256,
        evaluator_code_sha256=evaluator_code_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
    )
    root = _locked_root(source["experiment_run_id"], locked["locked_test_run_id"])
    publish_immutable_json(root / AUTHORISATION_FILE, authorisation)
    publish_immutable_json(root / MANIFEST_FILE, locked)
    return {
        "status": "prepared",
        "locked_test_run_id": locked["locked_test_run_id"],
        "authorisation_id": authorisation["authorisation_id"],
        "trial_count": len(locked["trials"]),
        "locked_test_row_count": LOCKED_TEST_ROWS,
    }


def _locked_reference_rows() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    proxy_path = VOLUME_PATH / VOLUME_INPUT_PREFIX / DEVELOPMENT_PROXY_PARQUET_NAME
    rows = pq.read_table(
        proxy_path,
        filters=[("split", "=", "locked_test_candidate")],
    ).to_pylist()
    if len(rows) != LOCKED_TEST_ROWS or any(
        row.get("split") != "locked_test_candidate" for row in rows
    ):
        raise RuntimeError("locked-test filter did not yield exactly 230 rows")
    return rows


def _reference_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reference = {}
    for row in rows:
        item_id = row.get("source_sample_id")
        if not isinstance(item_id, str) or not item_id or item_id in reference:
            raise ValueError("locked-test reference has invalid or duplicate IDs")
        label = json.loads(row["label_json"])
        if not isinstance(label, dict):
            raise ValueError("locked-test reference label is invalid")
        reference[item_id] = label
    return reference


def _evaluate_locked_trial(
    source_manifest: Mapping[str, Any],
    *,
    locked_test_run_id: str,
    source_trial_id: str,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from reddit_china_stance.modernbert_trainer import (
        DynamicPaddingCollator,
        ModelConfig,
        build_length_bucket_batches,
        create_modernbert_three_head_model,
        decode_predictions,
        evaluate_epoch,
        load_checkpoint_exact,
        load_pinned_tokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("locked-test inference requires a CUDA GPU")
    source = validate_run_manifest(source_manifest)
    locked = _load_locked_manifest(source["experiment_run_id"], locked_test_run_id)
    trials = [row for row in locked["trials"] if row["source_trial_id"] == source_trial_id]
    source_trials = [row for row in source["trials"] if row["trial_id"] == source_trial_id]
    if len(trials) != 1 or len(source_trials) != 1:
        raise ValueError("locked-test source trial must resolve exactly once")
    trial = trials[0]
    source_trial = source_trials[0]
    root = _locked_root(source["experiment_run_id"], locked_test_run_id)
    trial_root = root / f"trial={source_trial_id}"
    prediction_path = trial_root / PREDICTIONS_FILE
    receipt_path = trial_root / TRIAL_RECEIPT_FILE
    if receipt_path.is_file():
        receipt = validate_locked_trial_receipt(
            locked, _json_object(receipt_path, where="locked-test trial receipt")
        )
        descriptor = receipt["prediction_artifact"]
        if (
            not prediction_path.is_file()
            or prediction_path.stat().st_size != descriptor["bytes"]
            or file_sha256(prediction_path) != descriptor["sha256"]
        ):
            raise RuntimeError("complete locked-test prediction artifact is corrupt")
        return {
            "status": "already_complete",
            "locked_test_run_id": locked_test_run_id,
            "source_trial_id": source_trial_id,
            "receipt_id": receipt["receipt_id"],
            "estimated_cost_usd": receipt["compute"]["estimated_cost_usd"],
        }

    rows = _locked_reference_rows()
    reference = _reference_map(rows)
    started = time.monotonic()
    if prediction_path.is_file():
        payload = validate_private_locked_predictions(
            _json_object(prediction_path, where="private locked-test predictions"),
            expected_trial=trial,
            run_id=locked_test_run_id,
        )
        predictions = {row["source_sample_id"]: row["decoded_label"] for row in payload["rows"]}
        decoded = decode_predictions(
            [row["relevance_logits"] for row in payload["rows"]],
            [row["target_logits"] for row in payload["rows"]],
            [row["stance_logits"] for row in payload["rows"]],
            target_threshold=float(TARGET_THRESHOLD),
        )
        if list(decoded.labels) != [row["decoded_label"] for row in payload["rows"]]:
            raise RuntimeError("stored locked-test labels differ from their logits")
        metrics = score_semantic_labels(reference, predictions)
        metrics["decoding"] = {
            "target_threshold": float(TARGET_THRESHOLD),
            "forced_target_selections": decoded.forced_target_selections,
        }
    else:
        job = {
            "experiment_run_id": source["experiment_run_id"],
            "phase_run_id": source["phase_run_id"],
            "experiment_contract": source["experiment_contract"],
            "trial_spec": source_trial,
            "trial_spec_sha256": canonical_sha256(source_trial),
        }
        inputs = _load_runtime_inputs(job, volume_root=VOLUME_PATH, include_development=False)
        train_rows = _selected_teacher_rows(
            inputs["teacher_rows"], inputs["split_manifest"], trial_spec=source_trial
        )
        optimisation = _optimisation_config(source_trial, train_rows)
        model_config = ModelConfig()
        tokenizer = load_pinned_tokenizer(model_config)
        encoded, encoded_reference = _tokenise_rows(
            rows, tokenizer=tokenizer, item_id_field="source_sample_id"
        )
        if encoded_reference != reference:
            raise RuntimeError("locked-test tokenisation changed the reference binding")
        collator = DynamicPaddingCollator(tokenizer)
        microbatch = 32 // optimisation.gradient_accumulation_steps
        batches = build_length_bucket_batches(
            [len(row["input_ids"]) for row in encoded],
            batch_size=microbatch,
            seed=0,
            epoch=0,
        )
        loader = DataLoader(
            encoded,
            batch_sampler=batches,
            collate_fn=collator,
            num_workers=0,
        )
        model = create_modernbert_three_head_model(
            model_config=model_config,
            optimisation_config=optimisation,
        ).to("cuda")
        source_trial_root = (
            _source_run_root(source) / "phase=confirmatory" / f"trial={source_trial_id}"
        )
        source_receipt = _json_object(
            source_trial_root / TRIAL_RECEIPT_FILE,
            where="confirmatory source receipt",
        )
        _validate_source_receipt(source, source_trial, source_receipt)
        if source_receipt["receipt_id"] != trial["source_receipt_id"]:
            raise RuntimeError("locked trial and source receipt differ")
        checkpoint_path = _safe_path(source_trial_root, trial["checkpoint"]["relative_path"])
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != trial["checkpoint"]["bytes"]
            or file_sha256(checkpoint_path) != trial["checkpoint"]["sha256"]
        ):
            raise RuntimeError("selected checkpoint is missing or corrupt")
        preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        binding_sha256 = preview.get("binding_sha256") if isinstance(preview, Mapping) else None
        if not isinstance(binding_sha256, str):
            raise RuntimeError("selected checkpoint binding is absent")
        checkpoint = load_checkpoint_exact(
            checkpoint_path,
            expected_file_sha256=trial["checkpoint"]["sha256"],
            expected_binding_sha256=binding_sha256,
            map_location="cuda",
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        evaluation = evaluate_epoch(
            model,
            loader,
            device="cuda",
            target_threshold=float(TARGET_THRESHOLD),
            reference=reference,
            config=optimisation,
        )
        raw_payload = evaluation.pop("prediction_payload")
        metrics = evaluation["metrics"]
        payload = {
            "schema_version": "1.0.0",
            "kind": PREDICTION_KIND,
            "locked_test_run_id": locked_test_run_id,
            "authorisation_id": locked["authorisation"]["authorisation_id"],
            "source_trial_id": source_trial_id,
            "source_receipt_id": trial["source_receipt_id"],
            "row_count": raw_payload["row_count"],
            "decoder_target_threshold": raw_payload["decoder_target_threshold"],
            "rows": raw_payload["rows"],
        }
        payload = validate_private_locked_predictions(
            payload, expected_trial=trial, run_id=locked_test_run_id
        )
        publish_immutable_json(prediction_path, payload)
    wall_seconds = time.monotonic() - started
    prediction_descriptor = _file_descriptor(prediction_path)
    receipt = build_locked_trial_receipt(
        locked,
        trial=trial,
        prediction_descriptor=prediction_descriptor,
        metrics=metrics,
        wall_seconds=wall_seconds,
    )
    publish_immutable_json(receipt_path, receipt)
    return {
        "status": "complete",
        "locked_test_run_id": locked_test_run_id,
        "source_trial_id": source_trial_id,
        "receipt_id": receipt["receipt_id"],
        "estimated_cost_usd": receipt["compute"]["estimated_cost_usd"],
    }


def _locked_status(source_experiment_run_id: str, locked_test_run_id: str) -> dict[str, Any]:
    locked = _load_locked_manifest(source_experiment_run_id, locked_test_run_id)
    root = _locked_root(source_experiment_run_id, locked_test_run_id)
    complete = []
    partial = []
    for trial in locked["trials"]:
        trial_root = root / f"trial={trial['source_trial_id']}"
        has_prediction = (trial_root / PREDICTIONS_FILE).is_file()
        has_receipt = (trial_root / TRIAL_RECEIPT_FILE).is_file()
        if has_prediction and has_receipt:
            complete.append(trial["source_trial_id"])
        elif has_prediction or has_receipt:
            partial.append(trial["source_trial_id"])
    return {
        "status": "inspected",
        "locked_test_run_id": locked_test_run_id,
        "expected_trials": 9,
        "complete_trials": len(complete),
        "partial_trials": len(partial),
        "aggregate_complete": (root / AGGREGATE_FILE).is_file()
        and (root / FINAL_RECEIPT_FILE).is_file(),
    }


def _aggregate_locked_test(
    source_experiment_run_id: str, locked_test_run_id: str
) -> dict[str, Any]:
    locked = _load_locked_manifest(source_experiment_run_id, locked_test_run_id)
    root = _locked_root(source_experiment_run_id, locked_test_run_id)
    reference = _reference_map(_locked_reference_rows())
    payloads = {}
    receipts = []
    for trial in locked["trials"]:
        trial_root = root / f"trial={trial['source_trial_id']}"
        prediction_path = trial_root / PREDICTIONS_FILE
        receipt_path = trial_root / TRIAL_RECEIPT_FILE
        if not prediction_path.is_file() or not receipt_path.is_file():
            raise RuntimeError("locked-test aggregate requires all nine trial outputs")
        receipt = validate_locked_trial_receipt(
            locked, _json_object(receipt_path, where="locked-test trial receipt")
        )
        descriptor = receipt["prediction_artifact"]
        if (
            prediction_path.stat().st_size != descriptor["bytes"]
            or file_sha256(prediction_path) != descriptor["sha256"]
        ):
            raise RuntimeError("locked-test prediction hash differs from its receipt")
        payloads[trial["source_trial_id"]] = _json_object(
            prediction_path, where="private locked-test prediction"
        )
        receipts.append(receipt)
    aggregate = build_locked_test_aggregate(
        locked,
        reference=reference,
        prediction_payloads=payloads,
        trial_receipts=receipts,
    )
    aggregate_path = root / AGGREGATE_FILE
    publish_immutable_json(aggregate_path, aggregate)
    descriptor = _file_descriptor(aggregate_path)
    final_receipt = build_final_receipt(
        locked,
        aggregate=aggregate,
        aggregate_descriptor=descriptor,
    )
    publish_immutable_json(root / FINAL_RECEIPT_FILE, final_receipt)
    result = {
        "status": "complete",
        "locked_test_run_id": locked_test_run_id,
        "aggregate_id": aggregate["aggregate_id"],
        "final_receipt_id": final_receipt["receipt_id"],
        "row_count": aggregate["row_count"],
        "budget_summary": aggregate["budget_summary"],
        "paired_differences": aggregate["paired_differences"],
        "paired_hierarchical_bootstrap": aggregate["paired_hierarchical_bootstrap"],
        "core_target_regressions": aggregate["core_target_regressions"],
        "scale_gate": aggregate["scale_gate"],
        "invalid_outputs_total": aggregate["invalid_outputs_total"],
        "estimated_evaluation_cost_usd": aggregate["estimated_evaluation_cost_usd"],
    }
    assert_metadata_only(result, where="locked-test aggregate result")
    return result


def _validate_locked_final(
    source_experiment_run_id: str, locked_test_run_id: str
) -> dict[str, Any]:
    locked = _load_locked_manifest(source_experiment_run_id, locked_test_run_id)
    root = _locked_root(source_experiment_run_id, locked_test_run_id)
    aggregate_path = root / AGGREGATE_FILE
    receipt_path = root / FINAL_RECEIPT_FILE
    if not aggregate_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("locked-test final output is incomplete")
    aggregate = _json_object(aggregate_path, where="locked-test aggregate")
    receipt = _json_object(receipt_path, where="locked-test final receipt")
    rebuilt = build_final_receipt(
        locked,
        aggregate=aggregate,
        aggregate_descriptor=_file_descriptor(aggregate_path),
    )
    if receipt != rebuilt:
        raise RuntimeError("locked-test final receipt differs from immutable output")
    return {
        "status": "validated",
        "locked_test_run_id": locked_test_run_id,
        "aggregate_id": aggregate["aggregate_id"],
        "final_receipt_id": receipt["receipt_id"],
        "row_count": receipt["rows_accessed"],
        "scale_verdict": receipt["scale_verdict"],
        "scale_gate_passed": receipt["scale_gate_passed"],
        "estimated_evaluation_cost_usd": receipt["estimated_evaluation_cost_usd"],
    }


@app.function(image=image, cpu=4, memory=16384, timeout=30 * 60, volumes={str(VOLUME_PATH): volume})
def prepare_locked_test(
    source_manifest: dict[str, Any],
    source_manifest_sha256: str,
    evaluator_code_sha256: str,
    dependency_lock_sha256: str,
) -> dict[str, Any]:
    result = _prepare_locked_test(
        source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        evaluator_code_sha256=evaluator_code_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65536,
    timeout=60 * 60,
    max_containers=8,
    volumes={str(VOLUME_PATH): volume},
)
def evaluate_locked_trial(
    source_manifest: dict[str, Any], locked_test_run_id: str, source_trial_id: str
) -> dict[str, Any]:
    result = _evaluate_locked_trial(
        source_manifest,
        locked_test_run_id=locked_test_run_id,
        source_trial_id=source_trial_id,
    )
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=30 * 60, volumes={str(VOLUME_PATH): volume})
def inspect_locked_test(source_experiment_run_id: str, locked_test_run_id: str) -> dict[str, Any]:
    return _locked_status(source_experiment_run_id, locked_test_run_id)


@app.function(image=image, cpu=8, memory=32768, timeout=60 * 60, volumes={str(VOLUME_PATH): volume})
def aggregate_locked_test(source_experiment_run_id: str, locked_test_run_id: str) -> dict[str, Any]:
    result = _aggregate_locked_test(source_experiment_run_id, locked_test_run_id)
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=30 * 60, volumes={str(VOLUME_PATH): volume})
def validate_locked_final(source_experiment_run_id: str, locked_test_run_id: str) -> dict[str, Any]:
    return _validate_locked_final(source_experiment_run_id, locked_test_run_id)


def _code_digest(repo_root: Path) -> str:
    relative_paths = (
        "src/reddit_china_stance/modal_modernbert_locked.py",
        "src/reddit_china_stance/modernbert_locked_test.py",
        "src/reddit_china_stance/modal_modernbert.py",
        "src/reddit_china_stance/modernbert_training.py",
        "src/reddit_china_stance/modernbert_trainer.py",
        "src/reddit_china_stance/modernbert_model.py",
        "src/reddit_china_stance/semantic_evaluation.py",
    )
    bindings = {path: file_sha256(repo_root / path) for path in relative_paths}
    return canonical_sha256(bindings)


def _confirmation(action: str, phase_run_id: str) -> str:
    return f"{action.upper()}_MODERNBERT_LOCKED_TEST_{phase_run_id[:12]}"


@app.local_entrypoint()
def locked_main(
    action: str = "",
    source_manifest_path: str = "data/private-modernbert-v1/confirmatory/run-manifest.json",
    locked_test_run_id: str = "",
    approved_cost_usd: str = "200",
    confirm: str = "",
) -> None:
    if action not in {"plan", "run", "status", "aggregate", "validate"}:
        raise ValueError("action must be exactly plan, run, status, aggregate, or validate")
    source_path = Path(source_manifest_path)
    source = validate_run_manifest(_json_object(source_path, where="source manifest"))
    approved = Decimal(approved_cost_usd)
    if approved != HARD_MAX_APPROVAL_USD:
        raise ValueError("locked-test CLI approval must exactly match the frozen $200 cap")
    if approved < LOCKED_TEST_PHASE_COST_ESTIMATE_USD:
        raise RuntimeError("locked-test cost estimate exceeds approved spend")
    required = _confirmation(action, source["phase_run_id"])
    summary = {
        "status": "planned",
        "action": action,
        "source_experiment_run_id": source["experiment_run_id"],
        "source_phase_run_id": source["phase_run_id"],
        "expected_trials": 9,
        "locked_test_row_count": LOCKED_TEST_ROWS,
        "estimated_cost_usd": str(LOCKED_TEST_PHASE_COST_ESTIMATE_USD),
        "approved_cost_usd": str(approved),
        "required_confirmation": required,
    }
    if action == "plan":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if action == "run":
        if confirm != required:
            raise RuntimeError(f"refusing locked-test launch: pass --confirm {required}")
        validations = [inspect_trial.spawn(source, trial["trial_id"]) for trial in source["trials"]]
        if any(call.get().get("disposition") != "complete" for call in validations):
            raise RuntimeError("locked-test gate requires nine exact complete source trials")
        repo_root = Path.cwd()
        prepared = prepare_locked_test.remote(
            source,
            file_sha256(source_path),
            _code_digest(repo_root),
            file_sha256(repo_root / "uv.lock"),
        )
        run_id = prepared["locked_test_run_id"]
        calls = [
            evaluate_locked_trial.spawn(source, run_id, trial["trial_id"])
            for trial in source["trials"]
        ]
        summary.update(
            {
                "status": "submitted",
                "locked_test_run_id": run_id,
                "authorisation_id": prepared["authorisation_id"],
                "submitted_function_call_ids": [call.object_id for call in calls],
            }
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if not locked_test_run_id:
        raise ValueError("status, aggregate and validate require --locked-test-run-id")
    if action == "status":
        summary["inspection"] = inspect_locked_test.remote(
            source["experiment_run_id"], locked_test_run_id
        )
        summary["status"] = "inspected"
    elif action == "aggregate":
        if confirm != required:
            raise RuntimeError(f"refusing locked-test aggregation: pass --confirm {required}")
        summary["result"] = aggregate_locked_test.remote(
            source["experiment_run_id"], locked_test_run_id
        )
        summary["status"] = "complete"
    else:
        summary["validation"] = validate_locked_final.remote(
            source["experiment_run_id"], locked_test_run_id
        )
        summary["status"] = "validated"
    print(json.dumps(summary, indent=2, sort_keys=True))
