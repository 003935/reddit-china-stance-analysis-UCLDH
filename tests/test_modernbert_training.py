from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import reddit_china_stance.modernbert_training as training
from reddit_china_stance.modernbert_training import (
    LABEL_BUDGETS,
    REGISTERED_LADDERS,
    ModernBertContractError,
    assemble_development_proxy_parquet,
    budget_status,
    build_asha_plan,
    build_checkpoint_payload,
    build_trial_receipt,
    canonical_sha256,
    freeze_experiment_contract,
    freeze_trial_spec,
    frozen_trial_configs,
    issue_locked_test_authorisation,
    publication_state,
    publish_immutable_json,
    run_preflight_trial,
    run_training_trial,
    select_asha_promotions,
    select_checkpoint,
    validate_checkpoint_payload,
    validate_experiment_contract,
    validate_immutable_publication,
    validate_trial_receipt,
    validate_trial_spec,
)

SHA = "a" * 64


def _experiment(*, cap: int = 200, rate: str = "0.001") -> dict[str, object]:
    return freeze_experiment_contract(
        dataset_revision="b0fa1c3e5e3caf4dc3fb43b9024858364bc44834",
        dataset_sha256="f81146988ef504f81fb77b36333084e5ea10ab77908568d5be26bdba296a5bba",
        split_manifest_sha256="2" * 64,
        code_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        development_proxy_id="4ba40fcd0eb33dc2c7ca1c0b9ac75b06f42dc120bed470f9da7f713194a3a66d",
        teacher_generation_run_id=(
            "bc7854a60140b3584d5cdeb5513b07ff413253fd59cb84877c278de21702afa4"
        ),
        rate_card_usd_per_gpu_second={"L4": rate},
        hard_cost_cap_usd=cap,
    )


def _spec(
    experiment: dict[str, object],
    *,
    phase: str = "asha",
    budget: int = 5_000,
    ladder_seed: int = 101,
    optimiser_seed: int = 7,
    max_gpu_seconds: int = 1_000,
) -> dict[str, object]:
    return freeze_trial_spec(
        experiment,
        phase=phase,
        config=frozen_trial_configs()[0],
        subset_manifest_sha256="7" * 64,
        label_budget=budget,
        training_row_count=budget,
        ladder_seed=ladder_seed,
        optimiser_seed=optimiser_seed,
        gpu_type="L4",
        max_gpu_seconds=max_gpu_seconds,
    )


def _receipt(
    experiment: dict[str, object],
    spec: dict[str, object],
    *,
    gpu_seconds: int = 100,
) -> dict[str, object]:
    return build_trial_receipt(
        experiment,
        spec,
        artifacts={
            "checkpoint_artifact": {
                "relative_path": "checkpoints/final.safetensors",
                "sha256": "8" * 64,
                "bytes": 100,
            }
        },
        aggregate_metrics={"composite": 0.75, "invalid_outputs": 0},
        gpu_type="L4",
        wall_seconds=120,
        gpu_seconds=gpu_seconds,
    )


def test_registered_grid_and_asha_plan_are_exact_and_content_addressed() -> None:
    configs = frozen_trial_configs()
    plan = build_asha_plan()

    assert len(configs) == len({item["config_sha256"] for item in configs}) == 12
    assert [(rung["candidate_count"], rung["promotion_count"]) for rung in plan["rungs"]] == [
        (12, 4),
        (4, 2),
        (2, 0),
    ]
    assert plan["plan_id"] == canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_id"}
    )


def test_asha_promotion_is_complete_guarded_and_tie_broken_by_digest() -> None:
    plan = build_asha_plan()
    ids = [item["config_sha256"] for item in plan["configs"]]
    results = [
        {
            "config_sha256": config_id,
            "completed_epochs": 1,
            "composite": 0.5,
            "material_recall": 0.9,
            "invalid_outputs": 0,
        }
        for config_id in reversed(ids)
    ]

    promoted = select_asha_promotions(plan, results, rung_epochs=1, candidate_config_ids=ids)
    assert promoted == sorted(ids)[:4]

    results[0]["invalid_outputs"] = 1
    results[1]["material_recall"] = 0.79
    assert len(select_asha_promotions(plan, results, rung_epochs=1, candidate_config_ids=ids)) == 4
    with pytest.raises(ValueError, match="exactly match"):
        select_asha_promotions(plan, results[:-1], rung_epochs=1, candidate_config_ids=ids)


def test_checkpoint_selection_freezes_min_delta_patience_and_rejects_late_epochs() -> None:
    checkpoints = [
        {"epoch": 1, "composite": 0.40, "checkpoint_sha256": "1" * 64},
        {"epoch": 2, "composite": 0.50, "checkpoint_sha256": "2" * 64},
        {"epoch": 3, "composite": 0.5005, "checkpoint_sha256": "3" * 64},
        {"epoch": 4, "composite": 0.5007, "checkpoint_sha256": "4" * 64},
    ]
    selected = select_checkpoint(checkpoints)
    assert selected["selected_epoch"] == 4
    assert selected["stopped_epoch"] == 4
    with pytest.raises(ModernBertContractError, match="after the registered early stop"):
        select_checkpoint(
            [
                *checkpoints,
                {"epoch": 5, "composite": 0.8, "checkpoint_sha256": "5" * 64},
            ]
        )


def test_experiment_and_trial_bind_every_identity_and_cap_at_200() -> None:
    experiment = _experiment()
    assert validate_experiment_contract(experiment) == experiment
    assert experiment["budget"]["hard_cost_cap_usd"] == "200"
    assert experiment["registered_design"]["acquisition_ladders"] == list(REGISTERED_LADDERS)

    spec = _spec(experiment)
    assert validate_trial_spec(experiment, spec) == spec
    assert spec["reserved_cost_usd"] == 1.0

    drifted = deepcopy(experiment)
    drifted["bindings"]["model_revision"] = "f" * 40
    with pytest.raises(ModernBertContractError, match="content address drifted"):
        validate_experiment_contract(drifted)
    with pytest.raises(ValueError, match=r"no greater than \$200"):
        _experiment(cap=201)


def test_trial_reservation_and_observed_gpu_cannot_exceed_frozen_bounds() -> None:
    experiment = _experiment(rate="0.1")
    with pytest.raises(ModernBertContractError, match="single trial reservation"):
        _spec(experiment, max_gpu_seconds=2_001)

    experiment = _experiment()
    spec = _spec(experiment, max_gpu_seconds=100)
    with pytest.raises(ModernBertContractError, match="GPU-second reservation"):
        _receipt(experiment, spec, gpu_seconds=101)


def test_checkpoint_payload_requires_exact_resume_bindings_and_metadata_only_metrics() -> None:
    binding = {
        "dataset_sha256": "1" * 64,
        "split_manifest_sha256": "2" * 64,
        "trial_manifest_sha256": "3" * 64,
        "code_sha256": "4" * 64,
        "model_id": "answerdotai/ModernBERT-large",
        "model_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        "tokenizer_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        "epoch": 2,
        "global_step": 100,
        "optimiser_step": 100,
        "seed": 7,
        "config_sha256": "5" * 64,
        "model_state_sha256": "6" * 64,
        "optimiser_state_sha256": "7" * 64,
        "scheduler_state_sha256": "8" * 64,
        "rng_state_sha256": "9" * 64,
    }
    payload = build_checkpoint_payload(binding, aggregate_metrics={"composite": 0.71})
    assert validate_checkpoint_payload(payload) == payload

    initial = deepcopy(binding)
    initial.update(epoch=0, global_step=0, optimiser_step=0, seed=0)
    assert validate_checkpoint_payload(
        build_checkpoint_payload(initial, aggregate_metrics={"composite": 0.0})
    )
    with pytest.raises(ValueError, match="private field"):
        build_checkpoint_payload(binding, aggregate_metrics={"predictions": ["private"]})


def test_receipt_is_metadata_only_content_addressed_and_recomputes_cost() -> None:
    experiment = _experiment()
    spec = _spec(experiment)
    receipt = _receipt(experiment, spec, gpu_seconds=250)

    assert receipt["compute"]["estimated_cost_usd"] == 0.25
    assert receipt["bindings"] == experiment["bindings"]
    assert validate_trial_receipt(experiment, spec, receipt) == receipt
    assert "private" not in json.dumps(receipt)

    drifted = deepcopy(receipt)
    drifted["compute"]["estimated_cost_usd"] = 0.01
    with pytest.raises(ModernBertContractError, match="accounting drifted"):
        validate_trial_receipt(experiment, spec, drifted)
    with pytest.raises(ValueError, match="private field"):
        build_trial_receipt(
            experiment,
            spec,
            artifacts={
                "checkpoint": {
                    "relative_path": "checkpoint.bin",
                    "sha256": "9" * 64,
                    "bytes": 1,
                }
            },
            aggregate_metrics={"rows": ["not public"]},
            gpu_type="L4",
            wall_seconds=1,
            gpu_seconds=1,
        )


def test_budget_accounts_completed_plus_active_reservations_under_one_cap() -> None:
    experiment = _experiment(cap=2)
    complete_spec = _spec(experiment, max_gpu_seconds=1_000)
    completed = _receipt(experiment, complete_spec, gpu_seconds=600)
    active = _spec(
        experiment,
        phase="stability",
        optimiser_seed=13,
        max_gpu_seconds=1_000,
    )
    status = budget_status(experiment, completed_receipts=[completed], active_trial_specs=[active])
    assert status["completed_cost_usd"] == 0.6
    assert status["active_reserved_cost_usd"] == 1.0
    assert status["remaining_unreserved_usd"] == 0.4

    too_large = _spec(
        experiment,
        phase="stability",
        optimiser_seed=29,
        max_gpu_seconds=500,
    )
    with pytest.raises(ModernBertContractError, match="exceeds the cap"):
        budget_status(
            experiment,
            completed_receipts=[completed],
            active_trial_specs=[active, too_large],
        )


def test_locked_test_gate_requires_exact_complete_nine_run_design() -> None:
    experiment = _experiment()
    specs = [
        _spec(
            experiment,
            phase="confirmatory",
            budget=budget,
            ladder_seed=ladder["ladder_seed"],
            optimiser_seed=ladder["optimiser_seed"],
        )
        for ladder in REGISTERED_LADDERS
        for budget in LABEL_BUDGETS
    ]
    receipts = [_receipt(experiment, spec) for spec in specs]
    authorisation = issue_locked_test_authorisation(
        experiment,
        recipe_receipt_id="a" * 64,
        threshold_receipt_id="b" * 64,
        confirmatory_trial_specs=specs,
        confirmatory_receipts=receipts,
    )
    assert authorisation["locked_test_rows"] == 230
    assert len(authorisation["confirmatory_receipt_ids"]) == 9
    assert authorisation["single_access"] is True
    with pytest.raises(ModernBertContractError, match="exactly nine"):
        issue_locked_test_authorisation(
            experiment,
            recipe_receipt_id="a" * 64,
            threshold_receipt_id="b" * 64,
            confirmatory_trial_specs=specs[:-1],
            confirmatory_receipts=receipts[:-1],
        )


def test_immutable_publication_requires_explicit_exact_resume(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"kind": "synthetic", "count": 3}
    incomplete = path.with_suffix(".json.incomplete")
    incomplete.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    assert publication_state(path, value) == "resume_exact"
    with pytest.raises(ModernBertContractError, match="explicit resume"):
        publish_immutable_json(path, value)
    descriptor = publish_immutable_json(path, value, resume=True)
    assert descriptor == validate_immutable_publication(path, value)
    assert publication_state(path, value) == "complete"
    assert publish_immutable_json(path, value) == descriptor


def test_immutable_publication_never_repairs_mismatched_partial_or_final(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    expected = {"count": 1}
    incomplete = path.with_suffix(".json.incomplete")
    incomplete.write_text('{"count":2}\n', encoding="utf-8")
    with pytest.raises(ModernBertContractError, match="cannot resume exactly"):
        publish_immutable_json(path, expected, resume=True)
    incomplete.unlink()
    path.write_text('{"count":2}\n', encoding="utf-8")
    with pytest.raises(ModernBertContractError, match="immutable publication differs"):
        publish_immutable_json(path, expected)


def test_private_proxy_assembly_joins_exact_452_and_validates_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy_path = tmp_path / "proxy.json"
    reference_path = tmp_path / "reference.json"
    proxy_rows = []
    reference_rows = []
    for index in range(452):
        item_id = f"item-{index:03d}"
        split = "development" if index < 222 else "locked_test_candidate"
        label = {"relevance": "not_material", "target_stances": []}
        proxy_rows.append(
            {
                "source_sample_id": item_id,
                "split": split,
                "resolution": "synthetic",
                "label": label,
            }
        )
        reference_rows.append(
            {
                "source_sample_id": item_id,
                "target_text": f"synthetic text {index}",
                "parent_context": None,
                "submission_context": "synthetic context",
            }
        )
    proxy_path.write_text(
        json.dumps(
            {
                "proxy_id": "d" * 64,
                "input_binding": {"reference_packet_sha256": "e" * 64},
                "rows": proxy_rows,
            }
        ),
        encoding="utf-8",
    )
    reference_path.write_text(json.dumps({"rows": reference_rows}), encoding="utf-8")
    monkeypatch.setattr(training, "DEVELOPMENT_PROXY_ID", "d" * 64)
    monkeypatch.setattr(training, "PROXY_JSON_SHA256", training.file_sha256(proxy_path))
    monkeypatch.setattr(
        training, "REFERENCE_ROWS_JSON_SHA256", training.file_sha256(reference_path)
    )
    proxy_value = json.loads(proxy_path.read_text(encoding="utf-8"))
    proxy_value["input_binding"]["reference_packet_sha256"] = training.file_sha256(reference_path)
    proxy_path.write_text(json.dumps(proxy_value), encoding="utf-8")
    monkeypatch.setattr(training, "PROXY_JSON_SHA256", training.file_sha256(proxy_path))
    output = tmp_path / "development-proxy.parquet"

    descriptor = assemble_development_proxy_parquet(
        proxy_json_path=proxy_path,
        reference_rows_json_path=reference_path,
        output_path=output,
    )
    assert descriptor["row_count"] == 452
    assert descriptor["development_rows"] == 222
    assert descriptor["locked_test_candidate_rows"] == 230
    assert (
        assemble_development_proxy_parquet(
            proxy_json_path=proxy_path,
            reference_rows_json_path=reference_path,
            output_path=output,
        )
        == descriptor
    )


def _modal_job(*, preflight: bool) -> dict[str, object]:
    import reddit_china_stance.modal_modernbert as orchestrator

    contract = orchestrator.make_experiment_contract(
        split_manifest_sha256="1" * 64,
        trainer_code_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
        approved_cost_usd=training.Decimal("200"),
    )
    if preflight:
        trial = orchestrator.make_preflight_trials(contract)[0]
        manifest = orchestrator.make_run_manifest(
            contract=contract,
            phase="preflight",
            trials=orchestrator.make_preflight_trials(contract),
        )
    else:
        trials = orchestrator.make_sweep_trials(contract, gpu_type="L4")
        trial = trials[0]
        manifest = orchestrator.make_run_manifest(
            contract=contract, phase="sweep", trials=trials, rung_epochs=1
        )
    return orchestrator._trial_job(manifest, trial)


def test_preflight_adapter_executes_once_and_publishes_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _modal_job(preflight=True)
    monkeypatch.setattr(training, "_load_runtime_inputs", lambda *args, **kwargs: {"ok": True})

    def fake_runtime(**kwargs):
        assert kwargs["preflight"] is True
        checkpoint = kwargs["staging_root"] / "checkpoints/fake.bin"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return {
            "history": [
                {
                    "epoch": 1,
                    "train": {
                        "optimizer_steps": 200,
                        "mean_losses": {"loss": 0.5},
                    },
                }
            ],
            "training_rows": 5_000,
            "development_rows": 0,
            "truncated_training_rows": 5,
            "peak_cuda_memory_bytes": 100,
            "selected_checkpoint_relative_path": "checkpoints/fake.bin",
            "wall_seconds": 10.0,
            "gpu_seconds": 10.0,
        }

    monkeypatch.setattr(training, "_execute_trial_runtime", fake_runtime)
    summary = run_preflight_trial(job=job, volume_root=tmp_path)
    assert summary["status"] == "complete"
    assert summary["estimated_cost_usd"] == "0.002220"
    assert run_preflight_trial(job=job, volume_root=tmp_path) == summary


def test_training_adapter_resumes_only_explicit_bound_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _modal_job(preflight=False)
    monkeypatch.setattr(training, "_load_runtime_inputs", lambda *args, **kwargs: {"ok": True})

    def interrupted_runtime(**kwargs):
        staging = kwargs["staging_root"]
        checkpoint = staging / "checkpoints/epoch-01.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        training._write_resume_marker(
            staging / "resume.json",
            {
                "schema_version": training.SCHEMA_VERSION,
                "experiment_run_id": job["experiment_run_id"],
                "trial_id": job["trial_spec"]["trial_id"],
                "trial_spec_sha256": job["trial_spec_sha256"],
                "checkpoint": {
                    "relative_path": "checkpoints/epoch-01.pt",
                    "sha256": training.file_sha256(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "binding_sha256": "4" * 64,
                },
            },
        )
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(training, "_execute_trial_runtime", interrupted_runtime)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_training_trial(job=job, volume_root=tmp_path)
    with pytest.raises(ModernBertContractError, match="explicit resume"):
        run_training_trial(job=job, volume_root=tmp_path)

    def resumed_runtime(**kwargs):
        assert kwargs["resume"] is True
        return {
            "history": [
                {
                    "epoch": 1,
                    "train": {"optimizer_steps": 100},
                    "development": {
                        "composite": 0.5,
                        "metrics": {
                            "invalid_outputs": 0,
                            "relevance": {"macro_f1": 0.8, "material_recall": 0.9},
                            "targets": {"core": {"micro": {"f1": 0.7}}},
                            "end_to_end_core_target_stance": {"micro": {"f1": 0.5}},
                            "decoding": {"forced_target_selections": 1},
                        },
                    },
                }
            ],
            "training_rows": 5_000,
            "development_rows": 1,
            "truncated_training_rows": 5,
            "peak_cuda_memory_bytes": 100,
            "selected_checkpoint_relative_path": "checkpoints/epoch-01.pt",
            "selected_epoch": 1,
            "development_predictions": {
                "schema_version": training.SCHEMA_VERSION,
                "kind": "modernbert-private-development-predictions-v1",
                "row_count": 1,
                "decoder_target_threshold": 0.5,
                "rows": [
                    {
                        "source_sample_id": "private-one",
                        "relevance_logits": [1.0, 0.0, -1.0],
                        "target_logits": [1.0, -1.0, -1.0, -1.0],
                        "stance_logits": [[1.0, 0.0, -1.0, -2.0, -3.0]] * 4,
                        "decoded_label": {
                            "relevance": "material",
                            "target_stances": [{"target": "china_general", "stance": "negative"}],
                        },
                    }
                ],
            },
            "wall_seconds": 20.0,
            "gpu_seconds": 20.0,
        }

    monkeypatch.setattr(training, "_execute_trial_runtime", resumed_runtime)
    summary = run_training_trial(job=job, volume_root=tmp_path, resume=True)
    assert summary["status"] == "complete"
    assert summary["artifact_count"] >= 3

    import reddit_china_stance.modal_modernbert as orchestrator

    trials = orchestrator.make_sweep_trials(
        job["experiment_contract"], gpu_type="L4", rung_epochs=1
    )
    manifest = orchestrator.make_run_manifest(
        contract=job["experiment_contract"],
        phase="sweep",
        trials=trials,
        rung_epochs=1,
    )
    receipts = {}
    for trial in trials:
        receipts[trial["trial_id"]] = {
            "status": "complete",
            "trial_id": trial["trial_id"],
            "aggregate_metrics": {
                "completed_epochs": 1,
                "composite": 0.5,
                "material_recall": 0.9,
                "invalid_outputs": 0,
            },
            "artifacts": {
                "checkpoint": {
                    "relative_path": "checkpoints/epoch-01.pt",
                    "sha256": "5" * 64,
                    "bytes": 10,
                }
            },
        }
    final_root = (
        tmp_path
        / training.OUTPUT_PREFIX
        / f"run={job['experiment_run_id']}"
        / "phase=sweep"
        / f"trial={job['trial_spec']['trial_id']}"
    )
    receipts[job["trial_spec"]["trial_id"]] = json.loads(
        (final_root / "receipt.json").read_text(encoding="utf-8")
    )
    published_receipt = receipts[job["trial_spec"]["trial_id"]]
    assert "development_predictions" in published_receipt["artifacts"]
    assert "rows" not in published_receipt
    private_path = (
        final_root / published_receipt["artifacts"]["development_predictions"]["relative_path"]
    )
    private_payload = json.loads(private_path.read_text(encoding="utf-8"))
    assert private_payload["rows"][0]["source_sample_id"] == "private-one"
    promotion = orchestrator.build_asha_promotion_receipt(manifest, receipts)
    assert len(promotion["promoted"]) == 4
    private_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModernBertContractError, match="missing or corrupt"):
        run_training_trial(job=job, volume_root=tmp_path)


def test_sweep_adapter_reads_nested_registered_config_and_fixed_seed_budget() -> None:
    import reddit_china_stance.modal_modernbert as orchestrator

    contract = orchestrator.make_experiment_contract(
        split_manifest_sha256="1" * 64,
        trainer_code_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
        approved_cost_usd=training.Decimal("200"),
    )
    trial = orchestrator.make_sweep_trials(contract, gpu_type="L4", rung_epochs=1)[0]
    rows = [
        {
            "sample_id": f"sample-{index}",
            "label_json": json.dumps(
                {
                    "relevance": "material" if index == 0 else "not_material",
                    "target_stances": (
                        [{"target": "china_general", "stance": "negative"}] if index == 0 else []
                    ),
                }
            ),
        }
        for index in range(4)
    ]
    config = training._optimisation_config(trial, rows)
    registered = trial["config"]["registered_config"]
    assert config.encoder_learning_rate == registered["encoder_learning_rate"]
    assert config.lambda_relevance == registered["loss_weights"]["relevance"]
    assert trial["config"]["target_epochs"] == 1


def test_stability_adapter_reads_nested_registered_config() -> None:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "label_json": json.dumps(
                {
                    "relevance": "material" if index == 0 else "not_material",
                    "target_stances": (
                        [{"target": "china_general", "stance": "negative"}] if index == 0 else []
                    ),
                }
            ),
        }
        for index in range(4)
    ]
    registered = next(
        row
        for row in training.frozen_trial_configs()
        if row["encoder_learning_rate"] == 0.00005
        and row["loss_weights"]["relevance"] == 0.5
        and row["class_weights"] == "capped_inverse_sqrt"
    )
    trial = {
        "phase": "stability",
        "config": {
            "registered_config": registered,
            "microbatch_size": 4,
        },
    }
    config = training._optimisation_config(trial, rows)
    assert config.encoder_learning_rate == 0.00005
    assert config.lambda_relevance == 0.5
    assert config.lambda_targets == 1.0
    assert config.lambda_stance == 1.5
    assert config.relevance_class_weights is not None


def test_stability_adapter_fails_without_registered_config() -> None:
    with pytest.raises(ValueError, match="wrapper disagree"):
        training._optimisation_config(
            {"phase": "stability", "config": {"encoder_learning_rate": 0.00005}}, []
        )


def test_confirmatory_adapter_uses_registered_recipe_and_frozen_outer_threshold() -> None:
    registered = training.frozen_trial_configs()[0]
    trial = {
        "phase": "confirmatory",
        "config": {
            "registered_config": registered,
            "target_threshold": "0.30",
            "fresh_training": True,
            "continuation": None,
            "locked_test_authorised": False,
            "recipe_receipt_id": "a" * 64,
            "threshold_receipt_id": "b" * 64,
        },
    }
    optimisation = training._optimisation_config(trial, [])
    assert optimisation.encoder_learning_rate == registered["encoder_learning_rate"]
    assert training._evaluation_target_threshold(trial) == 0.30


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"target_threshold": "0.40"}, "frozen target threshold 0.30"),
        ({"locked_test_authorised": True}, "locked-test authorisation"),
        ({"fresh_training": False}, "fresh-training"),
    ],
)
def test_confirmatory_adapter_rejects_semantic_or_authorisation_drift(
    mutation: dict[str, object], match: str
) -> None:
    trial = {
        "phase": "confirmatory",
        "config": {
            "registered_config": training.frozen_trial_configs()[0],
            "target_threshold": "0.30",
            "fresh_training": True,
            "continuation": None,
            "locked_test_authorised": False,
            "recipe_receipt_id": "a" * 64,
            "threshold_receipt_id": "b" * 64,
        },
    }
    trial["config"].update(mutation)
    with pytest.raises(training.ModernBertContractError, match=match):
        training._evaluation_target_threshold(trial)


def test_confirmatory_adapter_rejects_changed_registered_recipe() -> None:
    registered = json.loads(json.dumps(training.frozen_trial_configs()[0]))
    registered["encoder_learning_rate"] = 0.123
    trial = {"phase": "confirmatory", "config": {"registered_config": registered}}
    with pytest.raises(training.ModernBertContractError, match="exact registered recipe"):
        training._optimisation_config(trial, [])


def test_public_metrics_use_selected_epoch_not_last_epoch() -> None:
    metrics = {
        "history": [
            {
                "epoch": epoch,
                "train": {"optimizer_steps": 10},
                "development": {
                    "composite": composite,
                    "metrics": {
                        "invalid_outputs": 0,
                        "relevance": {"macro_f1": composite, "material_recall": 0.9},
                        "targets": {"core": {"micro": {"f1": composite}}},
                        "end_to_end_core_target_stance": {"micro": {"f1": composite}},
                        "decoding": {"forced_target_selections": 0},
                    },
                },
            }
            for epoch, composite in ((5, 0.7), (6, 0.6))
        ],
        "selected_epoch": 5,
        "training_rows": 5_000,
        "development_rows": 222,
        "truncated_training_rows": 0,
        "peak_cuda_memory_bytes": 1,
    }
    public = training._public_trial_metrics(metrics, preflight=False)
    assert public["completed_epochs"] == 6
    assert public["selected_epoch"] == 5
    assert public["composite"] == 0.7


def test_private_threshold_scorer_reads_only_development_and_returns_metadata(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    proxy_path = tmp_path / "development-proxy.parquet"
    rows = [
        {
            "source_sample_id": f"development-{index:03d}",
            "target_text": "synthetic development text",
            "parent_context": None,
            "submission_context": None,
            "split": "development",
            "resolution": "synthetic",
            "label_json": json.dumps(
                {"relevance": "not_material", "target_stances": []}, sort_keys=True
            ),
        }
        for index in range(222)
    ] + [
        {
            "source_sample_id": f"locked-{index:03d}",
            "target_text": "synthetic locked text",
            "parent_context": None,
            "submission_context": None,
            "split": "locked_test_candidate",
            "resolution": "synthetic",
            "label_json": json.dumps(
                {"relevance": "material", "target_stances": []}, sort_keys=True
            ),
        }
        for index in range(230)
    ]
    schema = pa.schema(
        [
            ("source_sample_id", pa.string()),
            ("target_text", pa.string()),
            ("parent_context", pa.string()),
            ("submission_context", pa.string()),
            ("split", pa.string()),
            ("resolution", pa.string()),
            ("label_json", pa.string()),
        ],
        metadata={
            b"proxy_id": training.DEVELOPMENT_PROXY_ID.encode(),
            b"proxy_json_sha256": training.PROXY_JSON_SHA256.encode(),
            b"reference_rows_json_sha256": training.REFERENCE_ROWS_JSON_SHA256.encode(),
        },
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), proxy_path)
    payload = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": "modernbert-private-development-predictions-v1",
        "row_count": 222,
        "decoder_target_threshold": 0.5,
        "rows": [
            {
                "source_sample_id": f"development-{index:03d}",
                "relevance_logits": [0.0, 5.0, 0.0],
                "target_logits": [-5.0] * 4,
                "stance_logits": [[5.0, 0.0, 0.0, 0.0, 0.0]] * 4,
                "decoded_label": {"relevance": "not_material", "target_stances": []},
            }
            for index in range(222)
        ],
    }

    result = training.score_private_development_thresholds(
        payload,
        development_proxy_path=proxy_path,
        expected_composite=0.0,
    )

    assert result["development_rows"] == 222
    assert len(result["development_reference_sha256"]) == 64
    assert result["threshold_composites"] == {
        "0.30": 0.0,
        "0.40": 0.0,
        "0.50": 0.0,
        "0.60": 0.0,
        "0.70": 0.0,
    }
    assert "rows" not in result
    assert "source_sample_id" not in json.dumps(result)

    corrupted = deepcopy(payload)
    corrupted["rows"][0]["decoded_label"] = {
        "relevance": "unclear",
        "target_stances": [],
    }
    with pytest.raises(ModernBertContractError, match="differ from their bound logits"):
        training.score_private_development_thresholds(
            corrupted,
            development_proxy_path=proxy_path,
            expected_composite=0.0,
        )


def test_promoted_rung_restores_exact_source_checkpoint_without_restart(tmp_path: Path) -> None:
    source_trial = "source-trial"
    relative = Path("phase=sweep") / f"trial={source_trial}" / "checkpoints/epoch-01.pt"
    checkpoint = tmp_path / relative
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"bound checkpoint")
    digest = training.file_sha256(checkpoint)
    restored = []

    def checkpoint_loader(path, **kwargs):
        assert path == checkpoint
        assert kwargs["expected_file_sha256"] == digest
        return {
            "binding": {
                "epoch": 1,
                "global_step": 100,
                "optimizer_step": 50,
                "optimisation_config_sha256": "9" * 64,
            }
        }

    counters = training._restore_asha_continuation(
        continuation={
            "config_sha256": "8" * 64,
            "source_trial_id": source_trial,
            "checkpoint_relative_path": str(relative),
            "checkpoint_sha256": digest,
        },
        registered_config_id="8" * 64,
        target_epoch=3,
        run_root=tmp_path,
        expected_optimisation_sha256="9" * 64,
        model=object(),
        optimizer=object(),
        scheduler=object(),
        torch_load=lambda *args, **kwargs: {"binding_sha256": "7" * 64},
        checkpoint_loader=checkpoint_loader,
        restore_training=lambda *args, **kwargs: restored.append((args, kwargs)),
    )
    assert counters == (1, 100, 50)
    assert len(restored) == 1
