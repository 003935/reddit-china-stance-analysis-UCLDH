from __future__ import annotations

from copy import deepcopy

import pytest

from reddit_china_stance import modernbert_factorised_experiment as experiment


def _artifact(
    name: str,
    digest: str,
    *,
    row_count: int | None = None,
    thread_set_sha256: str | None = None,
    frame: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "relative_path": f"inputs/{name}",
        "sha256": digest * 64,
        "bytes": 123,
    }
    if row_count is not None:
        value["row_count"] = row_count
    if thread_set_sha256 is not None:
        value["thread_set_sha256"] = thread_set_sha256
    if frame is not None:
        value["frame"] = frame
    return value


def _source_bundle() -> dict[str, object]:
    files = {"src/reddit_china_stance/frozen.py": "1" * 64}
    body = {
        "schema_version": experiment.SCHEMA_VERSION,
        "kind": experiment.SOURCE_BUNDLE_KIND,
        "files": files,
        "code_sha256": experiment.canonical_sha256(files),
    }
    return {**body, "source_bundle_id": experiment.canonical_sha256(body)}


def _probability_design() -> dict[str, object]:
    return {
        "component": "development_probability",
        "design_digest": "6" * 64,
        "expected_sample_rows": 300,
        "represented_population_rows": 8_000,
        "expected_strata": 20,
        "expected_probability_min": 0.02,
        "expected_probability_max": 0.08,
        "design_effective_sample_size": 240.0,
        "inverse_weight_ratio": 4.0,
    }


def frozen_contract(*, planned: str = "30", measured: str = "10") -> dict:
    return experiment.freeze_experiment_contract(
        teacher_run_id="a" * 64,
        teacher_receipt=_artifact("teacher-receipt.json", "2"),
        teacher_ledger=_artifact("labels.parquet", "3", row_count=10_000),
        teacher_blinded_input=_artifact("blinded-input.json", "4", row_count=10_000),
        teacher_private_mapping=_artifact("mapping.parquet", "5", row_count=10_000),
        source_parquet=_artifact("source.parquet", "6", row_count=10_000),
        source_receipt=_artifact("source-receipt.json", "e"),
        source_metadata_mapping=_artifact("mapping.parquet", "5", row_count=10_000),
        split_manifest=_artifact("membership.json", "7", row_count=10_000),
        split_public_manifest=_artifact(
            "split-public.json", "a", row_count=10_000
        ),
        bridge_exposure_register=_artifact(
            "bridge-exposure.json", "f", row_count=480
        ),
        legacy_proxy=_artifact("legacy-proxy.parquet", "1", row_count=452),
        legacy_overlap_audit=_artifact("legacy-overlap.json", "0"),
        training_frame=_artifact(
            "training.parquet",
            "8",
            row_count=8_100,
            thread_set_sha256="8" * 64,
            frame="training",
        ),
        development_frame=_artifact(
            "development.parquet",
            "9",
            row_count=600,
            thread_set_sha256="9" * 64,
            frame="development",
        )
        | {
            "primary_probability_row_count": 300,
            "selection_component_counts": {
                "development_context_available": 75,
                "development_multi_target": 75,
                "development_probability": 300,
                "development_rare_target_stance": 150,
            },
        },
        development_probability_design=_probability_design(),
        rubric={
            "repo_relative_path": "docs/rubrics/target-stance-v2-pilot.md",
            "sha256": "b" * 64,
            "bytes": 10,
        },
        schema={
            "repo_relative_path": "schemas/target-stance-v2-pilot.schema.json",
            "sha256": "c" * 64,
            "bytes": 10,
        },
        bridge_authorisation=_artifact("bridge-receipt.json", "d"),
        source_bundle=_source_bundle(),
        dependency_lock_sha256="e" * 64,
        rate_card_usd_per_gpu_second={"L4": "0.000222"},
        cumulative_measured_spend_usd=measured,
        active_reservation_usd="5",
        planned_phase_upper_usd=planned,
        hard_cost_cap_usd="200",
    )


def fixed_manifest() -> dict:
    contract = frozen_contract()
    trials = [
        experiment.freeze_trial_spec(
            contract,
            component=component,
            optimiser_seed=seed,
            max_gpu_seconds=3_600,
        )
        for component in experiment.COMPONENTS
        for seed in experiment.REGISTERED_SEEDS
    ]
    return experiment.build_run_manifest(contract, trials=trials)


def _metric_surface(*, tuple_f1: float, target_f1: float = 0.8) -> dict:
    return {
        "tuple_micro_f1": tuple_f1,
        "target_presence_macro_f1": target_f1,
        "calibration_error": 0.1,
        "retained_coverage_risk": 1.0 - tuple_f1,
        "per_target": {
            "government_ccp": {"score": tuple_f1, "reference_support": 20}
        },
        "per_target_stance_cell": {
            "government_ccp:negative": {
                "score": tuple_f1,
                "reference_support": 12,
            }
        },
        "invalid_outputs": 0,
    }


def _trial_metrics(component: str) -> dict:
    component_metrics = {}
    for selection, row_count in {
        "development_context_available": 75,
        "development_multi_target": 75,
        "development_probability": 300,
        "development_rare_target_stance": 150,
    }.items():
        surface = {
            "row_count": row_count,
            "evidence_scope": (
                "design-weighted-natural-probability-arm"
                if selection == "development_probability"
                else "unweighted-diagnostic-only"
            ),
        }
        if component == "relevance":
            surface |= {"relevance_macro_f1": 0.8, "material_recall": 0.75}
        else:
            surface["conditional_tuple_micro_f1"] = 0.7
        component_metrics[selection] = surface
    metrics = {
        "selected_epoch": 2,
        "development_rows": 600,
        "primary_probability_rows": 300,
        "invalid_outputs": 0,
        "checkpoint_score": 0.75,
        "development_component_metrics": component_metrics,
    }
    if component == "relevance":
        metrics |= {"relevance_macro_f1": 0.8, "material_recall": 0.75}
    else:
        metrics["conditional_tuple_micro_f1"] = 0.7
    return metrics


def test_contract_freezes_exact_model_recipe_inputs_and_total_cost() -> None:
    contract = frozen_contract()
    assert experiment.validate_experiment_contract(contract) == contract
    assert contract["experiment_run_id"] == experiment.canonical_sha256(
        {key: value for key, value in contract.items() if key != "experiment_run_id"}
    )
    assert contract["architecture"]["encoders"]["shared_trainable_parameters"] is False
    assert contract["compute"]["max_concurrent_trials"] == 9
    assert contract["evidence_boundary"]["bridge_rows_authorised_for_training"] is True
    assert (
        contract["evidence_boundary"]["bridge_training_label_source"]
        == "fresh_full_run_v2_teacher_only"
    )
    assert contract["evidence_boundary"]["bridge_rows_authorised_for_evaluation"] is False
    assert contract["evidence_boundary"]["locked_test_rows_accessed"] == 0
    for component in experiment.COMPONENTS:
        config = contract["registered_design"]["configs"][component]
        assert config["encoder_learning_rate"] == 5e-5
        assert config["head_learning_rate_multiplier"] == 5.0
        assert config["effective_batch_size"] == 32
        assert config["precision"] == "bf16"
        assert config["loss_weighting"] == "unweighted_proper_losses"
    assert contract["registered_design"]["configs"]["relevance"]["max_epochs"] == 6
    assert contract["registered_design"]["configs"]["target_stance_b4"]["max_epochs"] == 8


@pytest.mark.parametrize("component", experiment.COMPONENTS)
def test_public_trial_metrics_preserve_exact_evidence_scope_for_all_components(
    component: str,
) -> None:
    metrics = _trial_metrics(component)
    assert experiment.validate_public_trial_metrics(
        metrics, component=component
    ) == metrics

    contract = frozen_contract()
    trial = experiment.freeze_trial_spec(
        contract,
        component=component,
        optimiser_seed=47,
        max_gpu_seconds=100,
    )
    receipt = experiment.build_trial_receipt(
        contract,
        trial,
        phase_run_id="f" * 64,
        run_manifest_sha256="1" * 64,
        artifacts={
            "checkpoint": _artifact("checkpoint.pt", "2"),
            "metrics": _artifact("metrics.json", "3"),
            "private_development_predictions": _artifact(
                "development-predictions.parquet", "4", row_count=600
            ),
        },
        metrics=metrics,
        wall_seconds=10.0,
        gpu_seconds=10.0,
    )
    assert experiment.validate_trial_receipt(contract, trial, receipt) == receipt


def test_public_trial_metrics_reject_missing_evidence_scope() -> None:
    metrics = _trial_metrics("relevance")
    del metrics["development_component_metrics"]["development_probability"][
        "evidence_scope"
    ]
    with pytest.raises(ValueError, match="metric schema drifted"):
        experiment.validate_public_trial_metrics(metrics, component="relevance")


@pytest.mark.parametrize(
    ("selection", "wrong_scope"),
    [
        ("development_probability", "unweighted-diagnostic-only"),
        (
            "development_context_available",
            "design-weighted-natural-probability-arm",
        ),
    ],
)
def test_public_trial_metrics_reject_wrong_evidence_scope(
    selection: str, wrong_scope: str
) -> None:
    metrics = _trial_metrics("target_stance_b4")
    metrics["development_component_metrics"][selection]["evidence_scope"] = wrong_scope
    with pytest.raises(ValueError, match=r"evidence_scope must be"):
        experiment.validate_public_trial_metrics(
            metrics, component="target_stance_b4"
        )


def test_public_trial_metrics_reject_extra_scope_field() -> None:
    metrics = _trial_metrics("target_stance_b2")
    metrics["development_component_metrics"]["development_multi_target"][
        "scientific_aggregate_eligible"
    ] = False
    with pytest.raises(ValueError, match="metric schema drifted"):
        experiment.validate_public_trial_metrics(
            metrics, component="target_stance_b2"
        )


def test_contract_fails_on_row_conservation_or_total_cost_drift() -> None:
    mismatched = _artifact("mapping.parquet", "5", row_count=9_999)
    with pytest.raises(experiment.FactorisedExperimentContractError, match="row counts"):
        experiment.freeze_experiment_contract(
            **{
                **{
                    "teacher_run_id": "a" * 64,
                    "teacher_receipt": _artifact("teacher-receipt.json", "2"),
                    "teacher_ledger": _artifact("labels.parquet", "3", row_count=10_000),
                    "teacher_blinded_input": _artifact(
                        "blinded-input.json", "4", row_count=10_000
                    ),
                    "teacher_private_mapping": mismatched,
                        "source_parquet": _artifact("source.parquet", "6", row_count=10_000),
                        "source_receipt": _artifact("source-receipt.json", "e"),
                        "source_metadata_mapping": mismatched,
                        "split_manifest": _artifact("membership.json", "7", row_count=10_000),
                        "split_public_manifest": _artifact(
                            "split-public.json", "a", row_count=10_000
                        ),
                        "bridge_exposure_register": _artifact(
                            "bridge-exposure.json", "f", row_count=480
                        ),
                        "legacy_proxy": _artifact(
                            "legacy-proxy.parquet", "1", row_count=452
                        ),
                        "legacy_overlap_audit": _artifact(
                            "legacy-overlap.json", "0"
                        ),
                    "training_frame": _artifact(
                        "training.parquet",
                        "8",
                        row_count=8_100,
                        thread_set_sha256="8" * 64,
                        frame="training",
                    ),
                        "development_frame": _artifact(
                        "development.parquet",
                        "9",
                        row_count=600,
                        thread_set_sha256="9" * 64,
                        frame="development",
                        )
                        | {
                            "primary_probability_row_count": 300,
                            "selection_component_counts": {
                                "development_context_available": 75,
                                "development_multi_target": 75,
                                "development_probability": 300,
                                "development_rare_target_stance": 150,
                            },
                        },
                    "development_probability_design": _probability_design(),
                    "rubric": {
                        "repo_relative_path": "docs/rubric.md",
                        "sha256": "b" * 64,
                        "bytes": 10,
                    },
                    "schema": {
                        "repo_relative_path": "schemas/schema.json",
                        "sha256": "c" * 64,
                        "bytes": 10,
                    },
                    "bridge_authorisation": _artifact("bridge.json", "d"),
                    "source_bundle": _source_bundle(),
                    "dependency_lock_sha256": "e" * 64,
                    "rate_card_usd_per_gpu_second": {"L4": "0.000222"},
                    "cumulative_measured_spend_usd": "10",
                    "active_reservation_usd": "5",
                    "planned_phase_upper_usd": "30",
                }
            }
        )
    with pytest.raises(experiment.FactorisedExperimentContractError, match="hard cap"):
        frozen_contract(planned="190", measured="10")


def test_run_manifest_is_exact_three_components_by_three_seeds() -> None:
    manifest = fixed_manifest()
    assert experiment.validate_run_manifest(manifest) == manifest
    assert len(manifest["trials"]) == 9
    assert {
        (trial["component"], trial["optimiser_seed"]) for trial in manifest["trials"]
    } == {
        (component, seed)
        for component in experiment.COMPONENTS
        for seed in experiment.REGISTERED_SEEDS
    }
    assert all(trial["gpu_type"] == "L4" for trial in manifest["trials"])
    with pytest.raises(experiment.FactorisedExperimentContractError, match="exact"):
        experiment.build_run_manifest(
            manifest["experiment_contract"], trials=manifest["trials"][:-1]
        )

    oversized = [
        experiment.freeze_trial_spec(
            manifest["experiment_contract"],
            component=component,
            optimiser_seed=seed,
            max_gpu_seconds=100_000,
        )
        for component in experiment.COMPONENTS
        for seed in experiment.REGISTERED_SEEDS
    ]
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="GPU-second reservations",
    ):
        experiment.build_run_manifest(
            manifest["experiment_contract"], trials=oversized
        )


def test_content_addressing_rejects_binding_or_configuration_drift() -> None:
    contract = frozen_contract()
    drifted = deepcopy(contract)
    drifted["bindings"]["teacher_private_mapping"]["sha256"] = "f" * 64
    with pytest.raises(experiment.FactorisedExperimentContractError):
        experiment.validate_experiment_contract(drifted)
    trial = experiment.freeze_trial_spec(
        contract, component="relevance", optimiser_seed=47, max_gpu_seconds=100
    )
    trial["config"]["encoder_learning_rate"] = 1e-4
    with pytest.raises(experiment.FactorisedExperimentContractError):
        experiment.validate_trial_spec(trial, experiment=contract)


def test_rehashed_contract_cannot_bypass_cost_cap_with_negative_or_nonfinite_values() -> None:
    def rehash(value: dict) -> dict:
        value["experiment_run_id"] = experiment.canonical_sha256(
            {key: item for key, item in value.items() if key != "experiment_run_id"}
        )
        return value

    negative = deepcopy(frozen_contract())
    negative["compute"]["cumulative_measured_spend_usd"] = "-500"
    negative["compute"]["remaining_after_plan_usd"] = "665"
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="finite and non-negative",
    ):
        experiment.validate_experiment_contract(rehash(negative))

    nonfinite = deepcopy(frozen_contract())
    nonfinite["compute"]["rate_card_usd_per_gpu_second"]["L4"] = "NaN"
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="positive and finite",
    ):
        experiment.validate_experiment_contract(rehash(nonfinite))

    mismatched_remaining = deepcopy(frozen_contract())
    mismatched_remaining["compute"]["remaining_after_plan_usd"] = "999"
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="remaining cost binding drifted",
    ):
        experiment.validate_experiment_contract(rehash(mismatched_remaining))


def test_contract_requires_source_metadata_mapping_to_be_teacher_mapping() -> None:
    drifted = deepcopy(frozen_contract())
    drifted["bindings"]["source_metadata_mapping"] = _artifact(
        "other-mapping.parquet", "f", row_count=10_000
    )
    drifted["experiment_run_id"] = experiment.canonical_sha256(
        {key: value for key, value in drifted.items() if key != "experiment_run_id"}
    )
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="source metadata mapping drifted",
    ):
        experiment.validate_experiment_contract(drifted)


def test_three_seed_gate_promotes_only_a_material_b2_gain() -> None:
    contract = frozen_contract()
    paired = [
        {
            "optimiser_seed": seed,
            "B4": _metric_surface(tuple_f1=0.70),
            "B2": _metric_surface(tuple_f1=0.72),
        }
        for seed in experiment.REGISTERED_SEEDS
    ]
    gate = experiment.evaluate_representation_gate(
        contract,
        phase_run_id="f" * 64,
        paired_metrics=paired,
        trial_receipt_ids=[f"{index:064x}" for index in range(1, 10)],
    )
    assert gate["verdict"] == "promote_b2"
    assert gate["selected_representation"] == "B2"
    assert gate["aggregate_evidence"]["mean_paired_tuple_f1_gain"] == 0.02
    assert gate["calibration_or_threshold_frozen"] is False
    assert gate["locked_test_rows_accessed"] == 0

    too_small = deepcopy(paired)
    for row in too_small:
        row["B2"]["tuple_micro_f1"] = 0.703
        row["B2"]["retained_coverage_risk"] = 0.297
    gate = experiment.evaluate_representation_gate(
        contract,
        phase_run_id="f" * 64,
        paired_metrics=too_small,
        trial_receipt_ids=[f"{index:064x}" for index in range(1, 10)],
    )
    assert gate["verdict"] == "scrap_b2_keep_b4"
    assert gate["selected_representation"] == "B4"
