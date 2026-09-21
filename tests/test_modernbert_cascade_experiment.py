from __future__ import annotations

import json
from pathlib import Path

import pytest

from reddit_china_stance import modernbert_cascade_experiment as experiment
from reddit_china_stance.modernbert_training import (
    DATASET_REVISION,
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    TEACHER_GENERATION_RUN_ID,
)

SHA = "a" * 64


def frozen_experiment() -> dict:
    return experiment.freeze_experiment_contract(
        dataset_revision=DATASET_REVISION,
        dataset_sha256=DATASET_SHA256,
        split_manifest_sha256="1" * 64,
        source_bundle_sha256="2" * 64,
        code_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        development_proxy_sha256=DEVELOPMENT_PROXY_ID,
        development_reference_sha256="5" * 64,
        baseline_manifest_sha256="6" * 64,
        matched_baseline_sha256="7" * 64,
        teacher_generation_run_id=TEACHER_GENERATION_RUN_ID,
        rate_card_usd_per_gpu_second={"L4": "0.000222"},
        hard_cost_cap_usd="20",
    )


def frozen_trials(contract: dict) -> list[dict]:
    return [
        experiment.freeze_trial_spec(
            contract,
            component=component,
            subset_manifest_sha256="8" * 64,
            ladder_seed=condition["ladder_seed"],
            optimiser_seed=condition["optimiser_seed"],
            gpu_type="L4",
            max_gpu_seconds=6_000 if component == "relevance" else 18_000,
        )
        for component in experiment.COMPONENTS
        for condition in experiment.PAIRED_CONDITIONS
    ]


def descriptor(name: str, digest: str = SHA) -> dict:
    return {"repo_relative_path": name, "sha256": digest, "bytes": 123}


def test_contract_freezes_actual_separate_model_design_and_gates() -> None:
    contract = frozen_experiment()

    assert experiment.validate_experiment_contract(contract) == contract
    assert contract["namespace"] == "student-modernbert-cascade-v1"
    assert contract["architecture"]["encoders"] == (
        "two_separately_instantiated_modernbert_large_models"
    )
    assert contract["registered_design"]["data"] == {
        "relevance_training_rows": 10_000,
        "material_training_rows": 6_436,
        "target_conditioned_training_rows": 25_744,
        "excluded_non_material_target_slots": 14_256,
        "target_state_counts": {
            "absent": 17_896,
            "negative": 2_235,
            "mixed": 172,
            "no_directed_stance": 4_042,
            "positive": 1_378,
            "unclear": 21,
        },
        "development_rows": 222,
    }
    criteria = contract["registered_design"]["continuation_criteria"]
    assert criteria["minimum_mean_paired_gain"] == 0.03
    assert criteria["minimum_improved_pairs"] == 2
    assert criteria["maximum_mean_material_recall_decline"] == 0.02
    assert criteria["maximum_mean_relevance_macro_f1_decline"] == 0.01
    assert criteria["maximum_supported_core_target_regression"] == 0.05
    assert criteria["maximum_supported_target_stance_regression"] == 0.10
    assert contract["compute"] == {
        "allowed_gpus": ["L4"],
        "account_gpu_limit": 10,
        "max_concurrent_trials": 6,
        "gpu_fallback_allowed": False,
        "planned_upper_cost_usd": "20",
        "approved_cost_usd": "20",
    }
    assert contract["locked_test"]["rows_accessed"] == 0


def test_component_configs_are_fixed_and_target_ce_is_unweighted() -> None:
    relevance = experiment.frozen_component_config("relevance")
    target = experiment.frozen_component_config("target_conditioned")

    assert relevance["encoder_learning_rate"] == 5e-5
    assert relevance["class_weights"] == "capped_inverse_sqrt"
    assert relevance["max_epochs"] == 6
    assert relevance["checkpoint_score"] == {
        "relevance_macro_f1": 0.5,
        "material_recall": 0.5,
    }
    assert target["class_weights"] == "none"
    assert target["max_epochs"] == 8
    assert target["classes"] == list(experiment.TARGET_STATES)


def test_contract_rejects_identity_or_cost_drift() -> None:
    kwargs = {
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "split_manifest_sha256": "1" * 64,
        "source_bundle_sha256": "2" * 64,
        "code_sha256": "3" * 64,
        "dependency_lock_sha256": "4" * 64,
        "development_proxy_sha256": DEVELOPMENT_PROXY_ID,
        "development_reference_sha256": "5" * 64,
        "baseline_manifest_sha256": "6" * 64,
        "matched_baseline_sha256": "7" * 64,
        "teacher_generation_run_id": TEACHER_GENERATION_RUN_ID,
        "rate_card_usd_per_gpu_second": {"L4": "0.000222"},
    }
    with pytest.raises(experiment.CascadeExperimentContractError):
        experiment.freeze_experiment_contract(**{**kwargs, "dataset_revision": "drift"})
    with pytest.raises(ValueError, match="no greater than \\$20"):
        experiment.freeze_experiment_contract(**kwargs, hard_cost_cap_usd="21")


def test_manifest_requires_exact_two_components_by_three_pairs() -> None:
    contract = frozen_experiment()
    trials = frozen_trials(contract)
    manifest = experiment.build_run_manifest(
        contract,
        trials=trials,
        baseline_manifest=descriptor("baseline.json", "6" * 64),
        matched_baseline=descriptor("matched.json", "7" * 64),
        dataset_profile=contract["registered_design"]["data"],
    )

    assert experiment.validate_run_manifest(manifest) == manifest
    assert len(manifest["trials"]) == 6
    assert {row["component"] for row in manifest["trials"]} == set(experiment.COMPONENTS)
    assert {row["training_row_count"] for row in manifest["trials"]} == {
        10_000,
        25_744,
    }
    with pytest.raises(experiment.CascadeExperimentContractError):
        experiment.build_run_manifest(
            contract,
            trials=trials[:-1],
            baseline_manifest=descriptor("baseline.json", "6" * 64),
            matched_baseline=descriptor("matched.json", "7" * 64),
            dataset_profile=contract["registered_design"]["data"],
        )


def test_trial_job_is_self_contained_and_bound_to_manifest() -> None:
    contract = frozen_experiment()
    trials = frozen_trials(contract)
    manifest = experiment.build_run_manifest(
        contract,
        trials=trials,
        baseline_manifest=descriptor("baseline.json", "6" * 64),
        matched_baseline=descriptor("matched.json", "7" * 64),
        dataset_profile=contract["registered_design"]["data"],
    )
    job = experiment.make_trial_job(manifest, trials[0])

    assert job["experiment_contract"] == contract
    assert job["trial_spec"] == trials[0]
    assert job["phase_run_id"] == manifest["phase_run_id"]
    assert job["run_manifest_sha256"] == experiment.canonical_sha256(manifest)


def test_private_prediction_contract_has_component_specific_exact_rows() -> None:
    contract = frozen_experiment()
    relevance_trial, target_trial = frozen_trials(contract)[0], frozen_trials(contract)[3]
    relevance_rows = [
        {"source_sample_id": f"row-{index}", "logits": [0.1, 0.2, 0.3]} for index in range(222)
    ]
    relevance = experiment.build_private_development_predictions(
        contract, relevance_trial, rows=relevance_rows
    )
    assert (
        experiment.validate_private_development_predictions(
            relevance,
            expected_experiment_run_id=contract["experiment_run_id"],
            expected_trial_id=relevance_trial["trial_id"],
        )
        == relevance
    )

    target_rows = [
        {
            "source_sample_id": f"row-{index}",
            "target": target,
            "logits": [0, 1, 2, 3, 4, 5],
        }
        for index in range(222)
        for target in experiment.TARGETS
    ]
    target = experiment.build_private_development_predictions(
        contract, target_trial, rows=target_rows
    )
    assert target["row_count"] == 888
    broken = dict(target)
    broken["rows"] = target["rows"][:-1]
    with pytest.raises(experiment.CascadeExperimentContractError):
        experiment.validate_private_development_predictions(
            broken,
            expected_experiment_run_id=contract["experiment_run_id"],
            expected_trial_id=target_trial["trial_id"],
        )


def test_receipt_and_budget_are_content_addressed() -> None:
    contract = frozen_experiment()
    trial = frozen_trials(contract)[0]
    artifacts = {
        name: {"relative_path": f"{name}.json", "sha256": SHA, "bytes": 12}
        for name in ("checkpoint", "metrics", "private_development_predictions")
    }
    metrics = {
        "selected_epoch": 4,
        "invalid_outputs": 0,
        "relevance_macro_f1": 0.8,
        "material_recall": 0.9,
        "checkpoint_score": 0.85,
    }
    receipt = experiment.build_trial_receipt(
        contract,
        trial,
        phase_run_id="9" * 64,
        run_manifest_sha256="b" * 64,
        artifacts=artifacts,
        metrics=metrics,
        wall_seconds=10,
        gpu_seconds=100,
    )

    assert experiment.validate_trial_receipt(contract, trial, receipt) == receipt
    status = experiment.budget_status(contract, completed_receipts=[receipt])
    assert status["completed_trials"] == 1
    assert status["completed_cost_usd"] == 0.0222


def closeout_row(condition: dict, *, gain: float, candidate: bool) -> dict:
    core_support = {target: 20 for target in experiment.CORE_TARGETS}
    core_f1 = {target: 0.8 + (gain if candidate else 0) for target in experiment.CORE_TARGETS}
    diagnostics = {
        target: {
            state: {
                "f1": 0.7 + (gain if candidate else 0),
                "reference_support": 12,
            }
            for state in experiment.TARGET_STATES
        }
        for target in experiment.CORE_TARGETS
    }
    payload = {
        "material_recall": 0.9,
        "relevance_macro_f1": 0.8,
        "core_target_f1": core_f1,
        "target_stance_diagnostics": diagnostics,
        "invalid_outputs": 0,
    }
    if candidate:
        payload.update(
            {
                "core_tuple_f1": 0.63 + gain,
                "core_target_support": core_support,
                "forced_target_selections": 1,
            }
        )
        return {"condition": condition, **payload}
    payload["core_target_reference_support"] = core_support
    payload["core_target_stance_tuple_micro_f1"] = 0.63
    return {"condition": condition, "metric_payload": payload}


def test_continuation_gate_uses_exact_paired_guards_and_receipt_bindings() -> None:
    contract = frozen_experiment()
    conditions = [dict(row) for row in experiment.PAIRED_CONDITIONS]
    baseline = [closeout_row(row, gain=0, candidate=False) for row in conditions]
    candidate = [closeout_row(row, gain=0.04, candidate=True) for row in conditions]

    gate = experiment.evaluate_continuation_gate(
        contract,
        phase_run_id="9" * 64,
        baseline_results=baseline,
        candidate_results=candidate,
        candidate_trial_receipt_ids=[str(index) * 64 for index in range(1, 7)],
    )

    assert gate["verdict"] == "continue_to_independent_evaluation"
    assert all(gate["guards"].values())
    assert gate["aggregate_evidence"]["improved_pair_count"] == 3
    assert gate["aggregate_evidence"]["forced_target_selections"] == 3
    assert gate["locked_test_rows_accessed"] == 0


def test_immutable_publication_requires_namespace_and_exact_resume(tmp_path: Path) -> None:
    path = tmp_path / experiment.NAMESPACE / "receipt.json"
    value = {"kind": "metadata-only", "value": 1}

    descriptor_value = experiment.publish_immutable_json(path, value)
    assert descriptor_value["sha256"]
    assert experiment.publication_state(path, value) == "complete"
    with pytest.raises(experiment.CascadeExperimentContractError):
        experiment.publish_immutable_json(path, {"kind": "metadata-only", "value": 2})
    with pytest.raises(ValueError):
        experiment.publication_state(tmp_path / "outside.json", value)


def test_contract_is_finite_json() -> None:
    json.dumps(frozen_experiment(), allow_nan=False)
