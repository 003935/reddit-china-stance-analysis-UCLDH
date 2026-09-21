from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from reddit_china_stance import modernbert_conditional_experiment as experiment
from reddit_china_stance.modernbert_conditional_model import TARGET_STATE_LABELS
from reddit_china_stance.modernbert_training import (
    DATASET_REVISION,
    DATASET_SHA256,
    TEACHER_GENERATION_RUN_ID,
)
from reddit_china_stance.privacy import assert_metadata_only

SHA = "a" * 64


def _experiment(*, cap: int = 200) -> dict:
    return experiment.freeze_experiment_contract(
        dataset_revision=DATASET_REVISION,
        dataset_sha256=DATASET_SHA256,
        split_manifest_sha256="1" * 64,
        source_bundle_sha256="2" * 64,
        code_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        development_reference_sha256="5" * 64,
        matched_baseline_receipt_sha256="6" * 64,
        teacher_generation_run_id=TEACHER_GENERATION_RUN_ID,
        rate_card_usd_per_gpu_second={"L4": "0.000222"},
        hard_cost_cap_usd=cap,
    )


def _trial(
    frozen: dict,
    *,
    config: dict | None = None,
    target_epochs: int = 2,
    resume_binding: dict | None = None,
    phase: str = "asha",
    ladder_seed: int = 101,
    optimiser_seed: int = 7,
) -> dict:
    return experiment.freeze_trial_spec(
        frozen,
        phase=phase,
        config=config or experiment.frozen_trial_configs()[0],
        subset_manifest_sha256="7" * 64,
        label_budget=5_000 if phase == "asha" else 10_000,
        training_row_count=5_000 if phase == "asha" else 10_000,
        ladder_seed=ladder_seed,
        optimiser_seed=optimiser_seed,
        target_epochs=target_epochs,
        gpu_type="L4",
        max_gpu_seconds=3_600,
        resume_binding=resume_binding,
    )


def _descriptors(*names: str) -> dict:
    return {
        name: {"relative_path": f"{name}.json", "sha256": str(index) * 64, "bytes": 10}
        for index, name in enumerate(names, start=1)
    }


def _baseline_manifest_descriptor() -> dict:
    return {
        "repo_relative_path": "data/private-modernbert-v1/baseline/run-manifest.json",
        "sha256": "f" * 64,
        "bytes": 1_024,
    }


def _checkpoint(frozen: dict, trial: dict, *, completed_epochs: int | None = None) -> dict:
    return experiment.build_checkpoint_payload(
        frozen,
        trial,
        completed_epochs=completed_epochs or trial["target_epochs"],
        global_step=100,
        optimiser_step=100,
        artifacts=_descriptors(
            "model_state",
            "optimiser_state",
            "scheduler_state",
            "rng_state",
        ),
        aggregate_metrics={"composite": 0.5},
    )


def _metrics(trial: dict) -> dict:
    relevance = 0.6
    target = 0.5
    target_state = 0.4
    return {
        "training_rows": trial["training_row_count"],
        "development_rows": experiment.DEVELOPMENT_ROWS,
        "completed_epochs": trial["target_epochs"],
        "selected_epoch": trial["target_epochs"],
        "invalid_outputs": 0,
        "relevance_macro_f1": relevance,
        "material_recall": 0.9,
        "core_target_micro_f1": target,
        "core_target_stance_tuple_micro_f1": target_state,
        "composite": 0.25 * relevance + 0.25 * target + 0.5 * target_state,
        "forced_target_selections": 0,
        "truncated_rows": 0,
    }


def _receipt(frozen: dict, trial: dict) -> dict:
    return experiment.build_trial_receipt(
        frozen,
        trial,
        checkpoint=_checkpoint(frozen, trial),
        artifacts=_descriptors(
            "checkpoint_metadata",
            "metrics",
            "development_predictions",
        ),
        aggregate_metrics=_metrics(trial),
        gpu_type="L4",
        wall_seconds=100,
        gpu_seconds=100,
    )


def test_frozen_inventory_matches_model_and_final_design() -> None:
    assert experiment.NAMESPACE == "student-modernbert-conditional-v1"
    assert experiment.TARGET_STATES == TARGET_STATE_LABELS
    assert experiment.ASHA_RUNGS == (
        {"epochs": 2, "candidate_count": 8, "promotion_count": 4},
        {"epochs": 4, "candidate_count": 4, "promotion_count": 2},
        {"epochs": 8, "candidate_count": 2, "promotion_count": 0},
    )
    configs = experiment.frozen_trial_configs()
    assert len(configs) == len({row["config_sha256"] for row in configs}) == 8
    assert {row["encoder_learning_rate"] for row in configs} == {3e-5, 5e-5}
    assert {row["loss_weights"]["relevance"] for row in configs} == {1.0, 2.0}
    assert {row["loss_weights"]["target_state"] for row in configs} == {1.0}
    assert {row["target_state_class_weights"] for row in configs} == {
        "none",
        "capped_inverse_sqrt",
    }
    assert {row["relevance_class_weights"] for row in configs} == {
        "capped_inverse_sqrt"
    }
    assert {row["max_epochs"] for row in configs} == {8}
    assert {row["max_length"] for row in configs} == {768}
    criteria = _experiment()["registered_design"]["continuation_criteria"]
    assert criteria["primary_metric"] == "core_target_stance_tuple_micro_f1"
    assert criteria["stance_states"] == list(TARGET_STATE_LABELS)
    assert criteria["maximum_supported_target_stance_regression"] == 0.10


def test_experiment_is_content_addressed_metadata_only_and_drift_fails() -> None:
    frozen = _experiment()
    assert frozen["experiment_run_id"] == experiment.canonical_sha256(
        {key: value for key, value in frozen.items() if key != "experiment_run_id"}
    )
    assert frozen["compute"] == {
        "allowed_gpus": ["L4"],
        "account_gpu_limit": 10,
        "max_concurrent_trials": 8,
        "gpu_fallback_allowed": False,
        "planned_upper_cost_usd": "50",
        "approved_cost_usd": "200",
    }
    assert_metadata_only(frozen)
    assert experiment.validate_experiment_contract(frozen) == frozen
    drifted = deepcopy(frozen)
    drifted["architecture"]["decoder"]["relevance_gates_targets"] = False
    with pytest.raises(experiment.ConditionalExperimentContractError):
        experiment.validate_experiment_contract(drifted)
    with pytest.raises(ValueError, match="no greater than"):
        _experiment(cap=201)


def test_asha_promotion_requires_complete_exact_inventory_and_gates() -> None:
    plan = experiment.build_asha_plan()
    config_ids = [row["config_sha256"] for row in plan["configs"]]
    results = [
        {
            "config_sha256": config_id,
            "completed_epochs": 2,
            "composite": 0.5 + index / 100,
            "material_recall": 0.9,
            "invalid_outputs": 0,
        }
        for index, config_id in enumerate(config_ids)
    ]
    promoted = experiment.select_asha_promotions(
        plan,
        results,
        rung_epochs=2,
        candidate_config_ids=config_ids,
    )
    assert promoted == [row["config_sha256"] for row in results[-1:-5:-1]]
    with pytest.raises(ValueError, match="exactly cover"):
        experiment.select_asha_promotions(
            plan,
            results[:-1],
            rung_epochs=2,
            candidate_config_ids=config_ids,
        )
    blocked = deepcopy(results)
    for row in blocked:
        row["invalid_outputs"] = 1
    with pytest.raises(experiment.ConditionalExperimentContractError, match="too few"):
        experiment.select_asha_promotions(
            plan,
            blocked,
            rung_epochs=2,
            candidate_config_ids=config_ids,
        )


def test_trial_checkpoint_and_resume_are_exactly_bound() -> None:
    frozen = _experiment()
    first = _trial(frozen)
    checkpoint = _checkpoint(frozen, first)
    descriptor = {"relative_path": "checkpoint.json", "sha256": "8" * 64, "bytes": 12}
    resume = experiment.build_resume_binding(
        frozen,
        first,
        checkpoint,
        checkpoint_metadata_artifact=descriptor,
    )
    assert experiment.validate_resume_binding(frozen, first, checkpoint, resume) == resume
    second = _trial(frozen, target_epochs=4, resume_binding=resume)
    assert second["resume_binding"]["completed_epochs"] == 2
    assert experiment.validate_trial_spec(frozen, second) == second
    drifted = deepcopy(resume)
    drifted["completed_epochs"] = 1
    with pytest.raises(experiment.ConditionalExperimentContractError):
        _trial(frozen, target_epochs=4, resume_binding=drifted)
    changed_checkpoint = deepcopy(checkpoint)
    changed_checkpoint["global_step"] += 1
    with pytest.raises(experiment.ConditionalExperimentContractError):
        experiment.validate_checkpoint_payload(frozen, first, changed_checkpoint)


def test_run_manifest_inventory_and_content_address_are_exact() -> None:
    frozen = _experiment()
    trials = [
        _trial(frozen, config=config)
        for config in experiment.frozen_trial_configs()
    ]
    manifest = experiment.build_run_manifest(
        frozen,
        phase="sweep",
        trials=trials,
        asha_rung_epochs=2,
        baseline_manifest=_baseline_manifest_descriptor(),
    )
    assert manifest["phase_run_id"] == experiment.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "phase_run_id"}
    )
    assert manifest["experiment_artefacts"]["baseline_manifest"] == (
        _baseline_manifest_descriptor()
    )
    assert experiment.validate_run_manifest(manifest) == manifest
    changed_descriptor = {**_baseline_manifest_descriptor(), "sha256": "e" * 64}
    changed_manifest = experiment.build_run_manifest(
        frozen,
        phase="sweep",
        trials=trials,
        asha_rung_epochs=2,
        baseline_manifest=changed_descriptor,
    )
    assert changed_manifest["phase_run_id"] != manifest["phase_run_id"]
    with pytest.raises(experiment.ConditionalExperimentContractError, match="inventory"):
        experiment.build_run_manifest(
            frozen,
            phase="sweep",
            trials=trials[:-1],
            asha_rung_epochs=2,
            baseline_manifest=_baseline_manifest_descriptor(),
        )
    duplicate = [*trials[:-1], trials[0]]
    with pytest.raises(ValueError, match="duplicate"):
        experiment.build_run_manifest(
            frozen,
            phase="sweep",
            trials=duplicate,
            asha_rung_epochs=2,
            baseline_manifest=_baseline_manifest_descriptor(),
        )


@pytest.mark.parametrize(
    "descriptor",
    [
        {
            "repo_relative_path": "/tmp/baseline.json",
            "sha256": "f" * 64,
            "bytes": 1,
        },
        {
            "repo_relative_path": "data/../baseline.json",
            "sha256": "f" * 64,
            "bytes": 1,
        },
        {
            "repo_relative_path": "data/baseline.json",
            "sha256": "invalid",
            "bytes": 1,
        },
        {
            "repo_relative_path": "data/baseline.json",
            "sha256": "f" * 64,
            "bytes": 0,
        },
    ],
)
def test_run_manifest_rejects_unsafe_baseline_descriptor(descriptor: dict) -> None:
    frozen = _experiment()
    trials = [
        _trial(frozen, config=config)
        for config in experiment.frozen_trial_configs()
    ]
    with pytest.raises(ValueError):
        experiment.build_run_manifest(
            frozen,
            phase="sweep",
            trials=trials,
            asha_rung_epochs=2,
            baseline_manifest=descriptor,
        )


def test_confirmation_manifest_is_two_recipes_by_three_paired_conditions() -> None:
    frozen = _experiment()
    configs = experiment.frozen_trial_configs()[:2]
    trials = [
        _trial(
            frozen,
            config=config,
            phase="confirmation",
            target_epochs=8,
            ladder_seed=condition["ladder_seed"],
            optimiser_seed=condition["optimiser_seed"],
        )
        for config in configs
        for condition in experiment.CONFIRMATION_CONDITIONS
    ]
    manifest = experiment.build_run_manifest(
        frozen,
        phase="confirmation",
        trials=trials,
        asha_rung_epochs=None,
        baseline_manifest=_baseline_manifest_descriptor(),
        source_manifest_sha256="8" * 64,
        transition_receipt_sha256="9" * 64,
    )
    assert len(manifest["trials"]) == 6
    assert manifest["experiment_artefacts"]["baseline_manifest"] == (
        _baseline_manifest_descriptor()
    )
    assert experiment.validate_run_manifest(manifest) == manifest
    with pytest.raises(experiment.ConditionalExperimentContractError, match="inventory"):
        experiment.build_run_manifest(
            frozen,
            phase="confirmation",
            trials=trials[:3],
            asha_rung_epochs=None,
            baseline_manifest=_baseline_manifest_descriptor(),
            source_manifest_sha256="8" * 64,
            transition_receipt_sha256="9" * 64,
        )


def test_private_predictions_validate_decoder_and_content_address() -> None:
    frozen = _experiment()
    trial = _trial(frozen)
    relevance_logits = [3.0, 0.0, 0.0]
    state_logits = [[4.0, 1.0, 0.0, 0.0, 0.0, 0.0] for _ in experiment.TARGETS]
    decoded, forced = experiment.decode_conditional_logits(relevance_logits, state_logits)
    assert decoded == {
        "relevance": "material",
        "target_stances": [{"target": "china_general", "stance": "negative"}],
    }
    assert forced is True
    rows = [
        {
            "source_sample_id": f"synthetic-{index:03d}",
            "relevance_logits": relevance_logits,
            "target_state_logits": state_logits,
            "decoded_label": decoded,
            "forced_target_selection": forced,
        }
        for index in reversed(range(experiment.DEVELOPMENT_ROWS))
    ]
    payload = experiment.build_private_development_predictions(frozen, trial, rows=rows)
    assert payload["row_count"] == experiment.DEVELOPMENT_ROWS
    assert payload["rows"][0]["source_sample_id"] == "synthetic-000"
    assert experiment.validate_private_development_predictions(payload, frozen, trial) == payload
    drifted = deepcopy(payload)
    drifted["rows"][0]["decoded_label"]["target_stances"][0]["stance"] = "positive"
    with pytest.raises(experiment.ConditionalExperimentContractError):
        experiment.validate_private_development_predictions(drifted, frozen, trial)


def test_receipt_is_metadata_only_content_addressed_and_costed() -> None:
    frozen = _experiment()
    trial = _trial(frozen)
    checkpoint = _checkpoint(frozen, trial)
    receipt = _receipt(frozen, trial)
    assert receipt["compute"]["estimated_cost_usd"] == 0.0222
    assert_metadata_only(receipt)
    assert experiment.validate_trial_receipt(frozen, trial, checkpoint, receipt) == receipt
    drifted = deepcopy(receipt)
    drifted["aggregate_metrics"]["composite"] = 0.1
    with pytest.raises(experiment.ConditionalExperimentContractError):
        experiment.validate_trial_receipt(frozen, trial, checkpoint, drifted)
    private_metrics = _metrics(trial)
    private_metrics["rows"] = []
    with pytest.raises(ValueError):
        experiment.build_trial_receipt(
            frozen,
            trial,
            checkpoint=checkpoint,
            artifacts=_descriptors(
                "checkpoint_metadata",
                "metrics",
                "development_predictions",
            ),
            aggregate_metrics=private_metrics,
            gpu_type="L4",
            wall_seconds=1,
            gpu_seconds=1,
        )


def test_budget_rejects_duplicates_and_projected_overspend() -> None:
    frozen = _experiment(cap=0.8)
    trial = _trial(frozen)
    receipt = _receipt(frozen, trial)
    status = experiment.budget_status(frozen, completed_receipts=[receipt])
    assert status["completed_cost_usd"] == 0.0222
    with pytest.raises(ValueError, match="duplicate"):
        experiment.budget_status(frozen, completed_receipts=[receipt, receipt])
    expensive = experiment.freeze_trial_spec(
        frozen,
        phase="asha",
        config=experiment.frozen_trial_configs()[1],
        subset_manifest_sha256="7" * 64,
        label_budget=5_000,
        training_row_count=5_000,
        ladder_seed=101,
        optimiser_seed=7,
        target_epochs=2,
        gpu_type="L4",
        max_gpu_seconds=3_600,
    )
    with pytest.raises(experiment.ConditionalExperimentContractError, match="projected"):
        experiment.budget_status(
            frozen,
            completed_receipts=[receipt],
            active_trial_specs=[expensive],
        )


def test_immutable_publication_requires_exact_explicit_resume(tmp_path: Path) -> None:
    path = tmp_path / experiment.NAMESPACE / "receipt.json"
    value = {"kind": "synthetic", "digest": SHA}
    incomplete = path.with_suffix(".json.incomplete")
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text(
        '{\n  "digest": "' + SHA + '",\n  "kind": "synthetic"\n}\n',
        encoding="utf-8",
    )
    with pytest.raises(experiment.ConditionalExperimentContractError, match="explicit resume"):
        experiment.publish_immutable_json(path, value)
    descriptor = experiment.publish_immutable_json(path, value, resume=True)
    assert descriptor == experiment.validate_immutable_publication(path, value)
    assert experiment.publish_immutable_json(path, value) == descriptor
    with pytest.raises(experiment.ConditionalExperimentContractError, match="differs"):
        experiment.publish_immutable_json(path, {**value, "digest": "b" * 64})
    with pytest.raises(ValueError, match="must be under"):
        experiment.publish_immutable_json(tmp_path / "outside.json", value)


def _metric_payload(
    tuple_f1: float,
    recall: float,
    target_f1: float,
    cell_f1: float,
    *,
    regressed_cell: tuple[str, str] | None = None,
) -> dict:
    target_stance = {}
    for target in experiment.TARGETS:
        target_stance[target] = {}
        for state in experiment.TARGET_STATES:
            support = 20 if state == "absent" else 10 if state == "negative" else 0
            f1 = 0.5 if regressed_cell == (target, state) else cell_f1
            target_stance[target][state] = {
                "precision": f1,
                "recall": f1,
                "f1": f1,
                "reference_support": support,
            }
    stance = {
        state: {
            "precision": cell_f1,
            "recall": cell_f1,
            "f1": cell_f1,
            "reference_support": sum(
                target_stance[target][state]["reference_support"]
                for target in experiment.TARGETS
            ),
        }
        for state in experiment.TARGET_STATES
    }
    return {
        "core_target_stance_tuple_micro_f1": tuple_f1,
        "material_recall": recall,
        "core_target_f1": {target: target_f1 for target in experiment.CORE_TARGETS},
        "core_target_reference_support": {
            target: 10 for target in experiment.CORE_TARGETS
        },
        "stance_diagnostics": stance,
        "target_stance_diagnostics": target_stance,
        "invalid_outputs": 0,
    }


def _bound_gate_rows(
    frozen: dict,
    *,
    phase_run_id: str,
    config_sha256: str,
    tuple_f1: float,
    recall: float,
    target_f1: float,
    cell_f1: float,
    baseline: bool = False,
    regressed_cell: tuple[str, str] | None = None,
) -> list[dict]:
    rows = []
    for index, condition in enumerate(experiment.CONFIRMATION_CONDITIONS):
        metrics = _metric_payload(
            tuple_f1,
            recall,
            target_f1,
            cell_f1,
            regressed_cell=regressed_cell,
        )
        prefix = "baseline" if baseline else "candidate"
        rows.append(
            {
                "source_experiment_run_id": (
                    experiment.canonical_sha256({"source": "baseline-experiment"})
                    if baseline
                    else frozen["experiment_run_id"]
                ),
                "source_phase_run_id": (
                    experiment.canonical_sha256({"source": "baseline-phase"})
                    if baseline
                    else phase_run_id
                ),
                "config_sha256": config_sha256,
                "trial_id": experiment.canonical_sha256(
                    {
                        "source": prefix,
                        "config": config_sha256,
                        "type": "trial",
                        "index": index,
                    }
                ),
                "receipt_id": experiment.canonical_sha256(
                    {
                        "source": prefix,
                        "config": config_sha256,
                        "type": "receipt",
                        "index": index,
                    }
                ),
                "prediction_sha256": experiment.canonical_sha256(
                    {
                        "source": prefix,
                        "config": config_sha256,
                        "type": "prediction",
                        "index": index,
                    }
                ),
                "condition": dict(condition),
                "metric_payload": metrics,
                "metric_payload_sha256": experiment.canonical_sha256(metrics),
            }
        )
    return rows


def test_continuation_gate_encodes_promote_inconclusive_and_scrap_bands() -> None:
    frozen = _experiment()
    phase_run_id = "e" * 64
    baseline_config = "d" * 64
    candidate_config = "a" * 64
    baseline = _bound_gate_rows(
        frozen,
        phase_run_id=phase_run_id,
        config_sha256=baseline_config,
        tuple_f1=0.60,
        recall=0.95,
        target_f1=0.70,
        cell_f1=0.70,
        baseline=True,
    )
    candidate = _bound_gate_rows(
        frozen,
        phase_run_id=phase_run_id,
        config_sha256=candidate_config,
        tuple_f1=0.64,
        recall=0.94,
        target_f1=0.72,
        cell_f1=0.72,
    )
    promoted = experiment.evaluate_continuation_gate(
        frozen,
        phase_run_id=phase_run_id,
        candidate_config_sha256=candidate_config,
        baseline_results=baseline,
        candidate_results=candidate,
    )
    assert promoted["verdict"] == "promote"
    assert promoted["guardrails"] == {
        "mean_gain_pass": True,
        "paired_improvement_pass": True,
        "material_recall_pass": True,
        "supported_target_regression_pass": True,
        "supported_target_stance_regression_pass": True,
        "invalid_outputs_pass": True,
    }
    assert set(promoted["stance_diagnostics"]) == set(experiment.TARGET_STATES)
    assert len(promoted["candidate_evidence"]) == len(promoted["baseline_evidence"]) == 3
    assert promoted["experiment_run_id"] == frozen["experiment_run_id"]
    assert promoted["phase_run_id"] == phase_run_id
    assert promoted["development_reference_sha256"] == "5" * 64
    inconclusive = experiment.evaluate_continuation_gate(
        frozen,
        phase_run_id=phase_run_id,
        candidate_config_sha256=candidate_config,
        baseline_results=baseline,
        candidate_results=_bound_gate_rows(
            frozen,
            phase_run_id=phase_run_id,
            config_sha256=candidate_config,
            tuple_f1=0.62,
            recall=0.95,
            target_f1=0.70,
            cell_f1=0.70,
        ),
    )
    assert inconclusive["verdict"] == "inconclusive"
    scrapped = experiment.evaluate_continuation_gate(
        frozen,
        phase_run_id=phase_run_id,
        candidate_config_sha256=candidate_config,
        baseline_results=baseline,
        candidate_results=_bound_gate_rows(
            frozen,
            phase_run_id=phase_run_id,
            config_sha256=candidate_config,
            tuple_f1=0.605,
            recall=0.95,
            target_f1=0.70,
            cell_f1=0.70,
        ),
    )
    assert scrapped["verdict"] == "scrap"

    second_candidate = _bound_gate_rows(
        frozen,
        phase_run_id=phase_run_id,
        config_sha256="b" * 64,
        tuple_f1=0.63,
        recall=0.94,
        target_f1=0.71,
        cell_f1=0.71,
    )
    selection = experiment.select_confirmation_recipe(
        frozen,
        phase_run_id=phase_run_id,
        baseline_results=baseline,
        candidate_results_by_config={
            "a" * 64: candidate,
            "b" * 64: second_candidate,
        },
    )
    assert selection["selected_config_sha256"] == "a" * 64
    assert selection["selection_verdict"] == "promote"
    assert len(selection["candidates"][0]["candidate_evidence"]) == 3


def test_gate_rejects_supported_target_stance_regression_and_unbound_metrics() -> None:
    frozen = _experiment()
    phase_run_id = "e" * 64
    baseline = _bound_gate_rows(
        frozen,
        phase_run_id=phase_run_id,
        config_sha256="d" * 64,
        tuple_f1=0.60,
        recall=0.95,
        target_f1=0.70,
        cell_f1=0.70,
        baseline=True,
    )
    regressed = _bound_gate_rows(
        frozen,
        phase_run_id=phase_run_id,
        config_sha256="a" * 64,
        tuple_f1=0.64,
        recall=0.94,
        target_f1=0.72,
        cell_f1=0.72,
        regressed_cell=("government_ccp", "negative"),
    )
    result = experiment.evaluate_continuation_gate(
        frozen,
        phase_run_id=phase_run_id,
        candidate_config_sha256="a" * 64,
        baseline_results=baseline,
        candidate_results=regressed,
    )
    assert result["verdict"] == "reject_guardrail"
    assert result["guardrails"]["supported_target_stance_regression_pass"] is False
    assert result["target_stance_regression_breaches"] == ["government_ccp:negative"]

    unbound = [
        {
            "core_target_stance_tuple_micro_f1": 0.64,
            "material_recall": 0.94,
        }
    ] * 3
    with pytest.raises(ValueError, match="bound result schema"):
        experiment.select_confirmation_recipe(
            frozen,
            phase_run_id=phase_run_id,
            baseline_results=baseline,
            candidate_results_by_config={"a" * 64: unbound, "b" * 64: unbound},
        )

    digest_drift = deepcopy(regressed)
    digest_drift[0]["metric_payload_sha256"] = "f" * 64
    with pytest.raises(experiment.ConditionalExperimentContractError, match="digest drifted"):
        experiment.evaluate_continuation_gate(
            frozen,
            phase_run_id=phase_run_id,
            candidate_config_sha256="a" * 64,
            baseline_results=baseline,
            candidate_results=digest_drift,
        )
