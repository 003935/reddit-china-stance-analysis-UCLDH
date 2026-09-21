from __future__ import annotations

import json
from pathlib import Path

import pytest

from reddit_china_stance import modernbert_conditional_experiment as experiment
from reddit_china_stance.modernbert_conditional_runtime import (
    _bound_gate_result,
    _public_metrics,
    _require_preserved_baseline_manifest,
    _trial_roots,
    _validate_epoch_zero_resume_marker,
    _validate_job,
    _write_json_atomic,
    aggregate_trial_inspections,
    build_epoch_zero_resume_marker,
    build_optimisation_config,
    closeout_confirmation,
    conditional_diagnostics,
    prepare_confirmation_manifest,
    promote_asha_rung,
    run_training_trial,
    select_teacher_rows,
    target_state_class_weights,
    validate_private_development_predictions,
)
from reddit_china_stance.modernbert_training import (
    DATASET_REVISION,
    DATASET_SHA256,
    TEACHER_GENERATION_RUN_ID,
    ModernBertContractError,
    canonical_sha256,
)
from reddit_china_stance.semantic_evaluation import score_semantic_labels

DIGEST = "a" * 64


def _baseline_manifest_descriptor() -> dict[str, object]:
    return {
        "repo_relative_path": (
            "data/private-modernbert-conditional-v1/initial/matched-baseline.json"
        ),
        "sha256": "f" * 64,
        "bytes": 123,
    }


def _row(item_id: str, relevance: str, target_stances: list[dict[str, str]]) -> dict:
    return {
        "sample_id": item_id,
        "label_json": json.dumps(
            {"relevance": relevance, "target_stances": target_stances},
            sort_keys=True,
        ),
    }


def _weighted_rows() -> list[dict]:
    stances = ["negative", "mixed", "no_directed_stance", "positive", "unclear"]
    return [
        _row(
            f"row-{index}",
            "material",
            [{"target": "government_ccp", "stance": stance}],
        )
        for index, stance in enumerate(stances)
    ]


def _contract(*, split_manifest_sha256: str = "1" * 64) -> dict:
    return experiment.freeze_experiment_contract(
        dataset_revision=DATASET_REVISION,
        dataset_sha256=DATASET_SHA256,
        split_manifest_sha256=split_manifest_sha256,
        source_bundle_sha256="2" * 64,
        code_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        development_reference_sha256="5" * 64,
        matched_baseline_receipt_sha256="6" * 64,
        teacher_generation_run_id=TEACHER_GENERATION_RUN_ID,
        rate_card_usd_per_gpu_second={"L4": "0.000222"},
    )


def _trial(contract: dict, config: dict | None = None, *, epochs: int = 2) -> dict:
    return experiment.freeze_trial_spec(
        contract,
        phase="asha",
        config=config or experiment.frozen_trial_configs()[0],
        subset_manifest_sha256="7" * 64,
        label_budget=5_000,
        training_row_count=5_000,
        ladder_seed=101,
        optimiser_seed=7,
        target_epochs=epochs,
        gpu_type="L4",
        max_gpu_seconds=10_000,
        resume_binding=None,
    )


def _public_metric(composite: float, *, epochs: int = 2) -> dict:
    # Choose components whose exact registered weighted mean is composite.
    return {
        "training_rows": 5_000,
        "development_rows": 222,
        "completed_epochs": epochs,
        "selected_epoch": epochs,
        "invalid_outputs": 0,
        "relevance_macro_f1": composite,
        "material_recall": 0.9,
        "core_target_micro_f1": composite,
        "core_target_stance_tuple_micro_f1": composite,
        "composite": composite,
        "forced_target_selections": 0,
        "truncated_rows": 0,
    }


def test_target_state_weights_use_all_material_slots_and_cap() -> None:
    weights = target_state_class_weights(_weighted_rows())

    assert len(weights) == 6
    assert all(0.5 <= weight <= 4.0 for weight in weights)
    assert weights[0] < weights[1]

    with pytest.raises(ModernBertContractError, match="lacks supported classes"):
        target_state_class_weights(_weighted_rows()[:1])


def test_build_optimisation_config_binds_relevance_and_state_weights() -> None:
    config = experiment.frozen_trial_configs()[1]
    spec = {"config": config}
    optimisation = build_optimisation_config(spec, _weighted_rows())

    assert optimisation.encoder_learning_rate == config["encoder_learning_rate"]
    assert optimisation.lambda_relevance == config["loss_weights"]["relevance"]
    assert optimisation.lambda_target_state == 1.0
    assert optimisation.relevance_class_weights is not None
    assert optimisation.target_state_class_weights is not None


def test_exact_5k_teacher_selection_is_bound_to_split_metadata() -> None:
    teacher = [{"sample_id": f"id-{index:05d}"} for index in range(10_000)]
    split_rows = [
        {
            "sample_id": row["sample_id"],
            "folds": {"101": index % 10},
        }
        for index, row in enumerate(teacher)
    ]
    budget_metadata = {"row_count": 5_000, "opaque_id_set_sha256": DIGEST}
    split = {
        "rows": split_rows,
        "ladders": {"101": {"budgets": {"5k": budget_metadata}}},
    }
    spec = {
        "phase": "asha",
        "label_budget": 5_000,
        "ladder_seed": 101,
        "subset_manifest_sha256": canonical_sha256(budget_metadata),
    }

    selected = select_teacher_rows(teacher, split, trial_spec=spec)

    assert len(selected) == 5_000
    assert selected[0]["sample_id"] == "id-00000"
    with pytest.raises(ModernBertContractError, match="subset digest drifted"):
        select_teacher_rows(
            teacher,
            split,
            trial_spec={**spec, "subset_manifest_sha256": "0" * 64},
        )


def test_private_prediction_validation_is_sorted_finite_and_schema_exact() -> None:
    def prediction(item_id: str) -> dict:
        return {
            "source_sample_id": item_id,
            "relevance_logits": [2.0, 0.0, -1.0],
            "target_state_logits": [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "decoded_label": {
                "relevance": "material",
                "target_stances": [{"target": "government_ccp", "stance": "negative"}],
            },
        }

    value = {
        "schema_version": "1.0.0",
        "kind": "modernbert-conditional-private-development-predictions-v1",
        "row_count": 2,
        "rows": [prediction("a"), prediction("b")],
    }
    assert validate_private_development_predictions(value, expected_rows=2) == value
    with pytest.raises(ValueError, match="sorted"):
        validate_private_development_predictions(
            {**value, "rows": list(reversed(value["rows"]))}, expected_rows=2
        )


def test_job_validation_binds_contract_source_bundle_and_trial() -> None:
    contract = _contract()
    trial = _trial(contract)
    job = {
        "experiment_run_id": contract["experiment_run_id"],
        "phase_run_id": "8" * 64,
        "experiment_contract": contract,
        "trial_spec": trial,
        "trial_spec_sha256": canonical_sha256(trial),
    }
    assert _validate_job(job) == job
    without_digest = {key: value for key, value in job.items() if key != "trial_spec_sha256"}
    assert _validate_job(without_digest)["trial_spec_sha256"] == canonical_sha256(trial)
    with pytest.raises(Exception, match=r"source_bundle_sha256|content address drifted"):
        drifted = json.loads(json.dumps(job))
        drifted["experiment_contract"]["bindings"]["source_bundle_sha256"] = "0" * 64
        _validate_job(drifted)


def test_epoch_zero_claim_is_bound_atomic_and_resumable_without_deleting_evidence(
    tmp_path: Path,
) -> None:
    contract = _contract()
    trial = _trial(contract)
    job = {
        "experiment_run_id": contract["experiment_run_id"],
        "phase_run_id": "8" * 64,
        "experiment_contract": contract,
        "trial_spec": trial,
    }
    marker = build_epoch_zero_resume_marker(job)
    path = tmp_path / "resume.json"
    _write_json_atomic(path, marker)

    assert json.loads(path.read_text()) == marker
    assert _validate_epoch_zero_resume_marker(marker, job=job) == marker
    assert not path.with_suffix(".json.new").exists()
    with pytest.raises(ModernBertContractError, match=r"epoch-zero.*drifted"):
        _validate_epoch_zero_resume_marker({**marker, "trial_id": "0" * 64}, job=job)


def test_trial_publishes_epoch_zero_marker_before_private_input_initialisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    trial = _trial(contract)
    job = {
        "experiment_run_id": contract["experiment_run_id"],
        "phase_run_id": "8" * 64,
        "experiment_contract": contract,
        "trial_spec": trial,
    }

    def fail_input_load(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic input initialisation failure")

    monkeypatch.setattr(
        "reddit_china_stance.modernbert_conditional_runtime._load_inputs",
        fail_input_load,
    )
    with pytest.raises(RuntimeError, match="synthetic input initialisation failure"):
        run_training_trial(job=job, volume_root=tmp_path)

    _, _, incomplete = _trial_roots(
        volume_root=tmp_path,
        experiment_run_id=contract["experiment_run_id"],
        phase="asha",
        trial_id=trial["trial_id"],
    )
    marker_path = incomplete / "resume.json"
    marker = json.loads(marker_path.read_text())
    assert _validate_epoch_zero_resume_marker(marker, job=job) == marker

    with pytest.raises(RuntimeError, match="synthetic input initialisation failure"):
        run_training_trial(job=job, volume_root=tmp_path, resume=True)
    assert json.loads(marker_path.read_text()) == marker


def test_aggregate_inspections_conserves_inventory_and_cost() -> None:
    manifest = {"trials": [{"trial_id": "a"}, {"trial_id": "b"}]}
    result = aggregate_trial_inspections(
        manifest=manifest,
        inspections=[
            {
                "trial_id": "a",
                "state": "complete",
                "receipt": {"estimated_cost_usd": 1.25},
            },
            {"trial_id": "b", "state": "missing", "receipt": None},
        ],
    )
    assert result["status"] == "incomplete"
    assert result["complete_trial_ids"] == ["a"]
    assert result["missing_trial_ids"] == ["b"]
    assert result["estimated_cost_usd"] == "1.250000"
    all_missing = aggregate_trial_inspections(
        manifest=manifest,
        inspections=[
            {"trial_id": "a", "state": "missing", "receipt": None},
            {"trial_id": "b", "state": "missing", "receipt": None},
        ],
    )
    assert all_missing["estimated_cost_usd"] == "0.000000"
    with pytest.raises(ValueError, match="conserve"):
        aggregate_trial_inspections(manifest=manifest, inspections=[])


def test_runtime_paths_and_experiment_publication_are_isolated_and_immutable(
    tmp_path: Path,
) -> None:
    _, final, incomplete = _trial_roots(
        volume_root=tmp_path,
        experiment_run_id=DIGEST,
        phase="asha",
        trial_id="b" * 64,
    )
    assert "student-modernbert-conditional-v1" in final.parts
    assert "student-modernbert-v1" not in final.parts
    assert incomplete.parent.name == ".incomplete"

    path = tmp_path / "student-modernbert-conditional-v1" / "manifest.json"
    expected = {"status": "frozen", "digest": DIGEST}
    experiment.publish_immutable_json(path, expected)
    experiment.publish_immutable_json(path, expected)
    with pytest.raises(Exception, match="immutable publication differs"):
        experiment.publish_immutable_json(path, {**expected, "status": "drifted"})


def test_public_metrics_match_exact_contract_shape() -> None:
    metric = _public_metric(0.7)
    private = {
        "training_rows": 5_000,
        "development_rows": 222,
        "truncated_training_rows": 3,
        "history": [
            {
                "epoch": 2,
                "train": {"optimizer_steps": 10},
                "development": {
                    "composite": 0.7,
                    "metrics": {
                        "invalid_outputs": 0,
                        "relevance": {"macro_f1": 0.7, "material_recall": 0.9},
                        "targets": {"core": {"micro": {"f1": 0.7}}},
                        "end_to_end_core_target_stance": {"micro": {"f1": 0.7}},
                        "decoding": {"forced_target_selections": 0},
                    },
                },
            }
        ],
        "selected_epoch": 2,
    }
    assert _public_metrics(private) == {**metric, "truncated_rows": 3}


def test_promotion_uses_registered_ranking_and_publishes_exact_next_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    configs = experiment.frozen_trial_configs()
    trials = [_trial(contract, config) for config in configs]
    manifest = experiment.build_run_manifest(
        contract,
        phase="sweep",
        trials=trials,
        asha_rung_epochs=2,
        baseline_manifest=_baseline_manifest_descriptor(),
    )
    receipts = []
    descriptor = {
        "relative_path": "checkpoint.pt",
        "sha256": "9" * 64,
        "bytes": 1,
    }
    for index, trial in enumerate(trials):
        metrics = _public_metric(0.50 + index * 0.01)
        checkpoint = experiment.build_checkpoint_payload(
            contract,
            trial,
            completed_epochs=2,
            global_step=100,
            optimiser_step=50,
            artifacts={
                name: descriptor
                for name in ("model_state", "optimiser_state", "scheduler_state", "rng_state")
            },
            aggregate_metrics=metrics,
        )
        root = experiment.trial_output_root(tmp_path, contract, trial)
        root.mkdir(parents=True)
        checkpoint_path = root / "checkpoint-metadata.json"
        experiment.publish_immutable_json(checkpoint_path, checkpoint)
        receipts.append(
            {
                "trial_id": trial["trial_id"],
                "receipt_id": f"{index:064x}",
                "aggregate_metrics": metrics,
                "artifacts": {
                    "checkpoint_metadata": {
                        "relative_path": checkpoint_path.name,
                        "sha256": __import__("hashlib")
                        .sha256(checkpoint_path.read_bytes())
                        .hexdigest(),
                        "bytes": checkpoint_path.stat().st_size,
                    }
                },
            }
        )
    monkeypatch.setattr(
        "reddit_china_stance.modernbert_conditional_runtime._completed_receipts",
        lambda *_args, **_kwargs: receipts,
    )

    result = promote_asha_rung(manifest=manifest, volume_root=tmp_path)

    assert result["target_rung_epochs"] == 4
    assert result["promoted_trials"] == 4
    published = json.loads(
        (
            tmp_path
            / "student-modernbert-conditional-v1"
            / f"run={contract['experiment_run_id']}"
            / "manifests"
            / "sweep-rung-4.json"
        ).read_text()
    )
    assert experiment.validate_run_manifest(published) == published
    assert all(trial["resume_binding"] is not None for trial in published["trials"])
    assert published["experiment_artefacts"]["baseline_manifest"] == (
        _baseline_manifest_descriptor()
    )


def test_confirmation_preparation_freezes_two_recipes_by_three_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path = tmp_path / "student-modernbert-v1/inputs/split-manifest.json"
    split_path.parent.mkdir(parents=True)
    split_manifest = {
        "ladders": {
            str(seed): {
                "budgets": {
                    "10k": {
                        "folds": list(range(10)),
                        "row_count": 10_000,
                        "sample_ids_sha256": f"{index + 1:064x}",
                        "feature_support": {},
                    }
                }
            }
            for index, seed in enumerate((101, 202, 303))
        }
    }
    split_path.write_text(json.dumps(split_manifest), encoding="utf-8")
    import hashlib

    contract = _contract(split_manifest_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest())
    configs = experiment.frozen_trial_configs()[:2]
    source_trials = [
        {
            "trial_id": f"{index + 10:064x}",
            "config": config,
        }
        for index, config in enumerate(configs)
    ]
    source_manifest = {
        "phase": "sweep",
        "asha": {"rung_epochs": 8},
        "experiment_run_id": contract["experiment_run_id"],
        "phase_run_id": "8" * 64,
        "experiment_contract": contract,
        "experiment_artefacts": {
            "baseline_manifest": _baseline_manifest_descriptor()
        },
        "trials": source_trials,
    }
    receipts = [
        {
            "trial_id": trial["trial_id"],
            "receipt_id": f"{index + 20:064x}",
            "aggregate_metrics": {"composite": 0.8 - index * 0.01},
        }
        for index, trial in enumerate(source_trials)
    ]
    monkeypatch.setattr(experiment, "validate_run_manifest", lambda value: dict(value))
    monkeypatch.setattr(
        "reddit_china_stance.modernbert_conditional_runtime._completed_receipts",
        lambda *_args, **_kwargs: receipts,
    )

    result = prepare_confirmation_manifest(manifest=source_manifest, volume_root=tmp_path)

    assert result["recipes"] == 2
    assert result["trials"] == 6
    published = json.loads(
        (
            tmp_path
            / "student-modernbert-conditional-v1"
            / f"run={contract['experiment_run_id']}"
            / "manifests"
            / "confirmation.json"
        ).read_text()
    )
    assert len(published["trials"]) == 6
    assert len({trial["config"]["config_sha256"] for trial in published["trials"]}) == 2
    assert {trial["max_gpu_seconds"] for trial in published["trials"]} == {5_400}
    assert published["experiment_artefacts"]["baseline_manifest"] == (
        _baseline_manifest_descriptor()
    )


def test_derived_manifest_lineage_rejects_baseline_descriptor_drift() -> None:
    source = {
        "experiment_artefacts": {
            "baseline_manifest": _baseline_manifest_descriptor()
        }
    }
    assert _require_preserved_baseline_manifest(source, source) == (
        _baseline_manifest_descriptor()
    )
    drifted = {
        "experiment_artefacts": {
            "baseline_manifest": {
                **_baseline_manifest_descriptor(),
                "sha256": "e" * 64,
            }
        }
    }
    with pytest.raises(ModernBertContractError, match=r"baseline-manifest.*drifted"):
        _require_preserved_baseline_manifest(source, drifted)


def test_private_confirmation_closeout_uses_bound_builder_and_publishes_only_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    config_ids = [row["config_sha256"] for row in experiment.frozen_trial_configs()[:2]]
    phase_run_id = "8" * 64
    confirmation = {
        "phase": "confirmation",
        "experiment_run_id": contract["experiment_run_id"],
        "phase_run_id": phase_run_id,
        "experiment_contract": contract,
    }
    reference = {
        f"row-{index:03d}": (
            {
                "relevance": "material",
                "target_stances": [
                    {
                        "target": ("china_general", "government_ccp", "people_culture")[
                            index % 3
                        ],
                        "stance": (
                            "negative",
                            "mixed",
                            "no_directed_stance",
                            "positive",
                            "unclear",
                        )[index % 5],
                    }
                ],
            }
            if index < 180
            else {"relevance": "not_material", "target_stances": []}
        )
        for index in range(222)
    }
    metrics = score_semantic_labels(reference, reference)
    diagnostics = conditional_diagnostics(reference, reference)
    conditions = ((101, 47), (202, 61), (303, 89))

    def result(
        *, experiment_run_id: str, config_id: str, prefix: int
    ) -> list[dict[str, object]]:
        return [
            _bound_gate_result(
                source_experiment_run_id=experiment_run_id,
                source_phase_run_id=f"{prefix + 1:064x}",
                config_sha256=config_id,
                trial_id=f"{prefix + 10 + index:064x}",
                receipt_id=f"{prefix + 20 + index:064x}",
                prediction_sha256=f"{prefix + 30 + index:064x}",
                ladder_seed=ladder_seed,
                optimiser_seed=optimiser_seed,
                metrics=metrics,
                diagnostics=diagnostics,
            )
            for index, (ladder_seed, optimiser_seed) in enumerate(conditions)
        ]

    baseline_results = result(
        experiment_run_id="d" * 64, config_id="e" * 64, prefix=100
    )
    candidate_results = {
        config_id: [
            {
                **row,
                "source_phase_run_id": phase_run_id,
            }
            for row in result(
                experiment_run_id=contract["experiment_run_id"],
                config_id=config_id,
                prefix=200 + 100 * config_index,
            )
        ]
        for config_index, config_id in enumerate(config_ids)
    }
    monkeypatch.setattr(experiment, "validate_run_manifest", lambda value: dict(value))
    monkeypatch.setattr(
        "reddit_china_stance.modernbert_conditional_runtime._load_development_reference",
        lambda **_kwargs: reference,
    )
    monkeypatch.setattr(
        "reddit_china_stance.modernbert_conditional_runtime._baseline_prediction_frame",
        lambda **_kwargs: (baseline_results, {}, []),
    )
    monkeypatch.setattr(
        "reddit_china_stance.modernbert_conditional_runtime._candidate_prediction_frame",
        lambda **_kwargs: (candidate_results, {}, []),
    )

    closeout = closeout_confirmation(
        confirmation_manifest=confirmation,
        baseline_manifest={"phase": "confirmatory"},
        volume_root=tmp_path,
    )

    assert closeout["development_rows"] == 222
    assert closeout["baseline_trials"] == 3
    assert closeout["candidate_trials"] == 6
    assert closeout["selection_verdict"] == "no_promotion"
    output = (
        tmp_path
        / "student-modernbert-conditional-v1"
        / f"run={contract['experiment_run_id']}"
        / "closeout"
    )
    assert len(list(output.glob("gate-*.json"))) == 2
    assert (output / "recipe-selection.json").is_file()
    assert all("row-" not in path.read_text() for path in output.glob("*.json"))
