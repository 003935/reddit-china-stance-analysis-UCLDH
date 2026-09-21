from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

import reddit_china_stance.modal_modernbert as orchestrator


def _contract(
    *,
    approved: Decimal = Decimal("200"),
    trainer_code_sha256: str = "2" * 64,
    dependency_lock_sha256: str = "3" * 64,
) -> dict[str, object]:
    return orchestrator.make_experiment_contract(
        split_manifest_sha256="1" * 64,
        trainer_code_sha256=trainer_code_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        approved_cost_usd=approved,
    )


def _write_artifact(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _refresh_trial_receipt_id(receipt: dict[str, object]) -> None:
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = orchestrator._canonical_sha256(body)


def test_contract_pins_inputs_grid_concurrency_and_cost() -> None:
    contract = _contract()
    assert contract["bindings"]["model_revision"] == orchestrator.MODEL_REVISION
    assert contract["bindings"]["dataset_parquet_sha256"] == (orchestrator.DATASET_PARQUET_SHA256)
    assert contract["sweep"]["grid_size"] == 12
    assert contract["architecture"]["attention_implementation"] == "sdpa"
    assert contract["architecture"]["reference_compile"] is False
    assert contract["compute"]["max_concurrent_trials"] == 8
    assert contract["compute"]["gpu_fallback_allowed"] is False
    assert contract["compute"]["hard_max_approved_cost_usd"] == "200"
    assert len(orchestrator.experiment_run_id(contract)) == 64
    assert (
        orchestrator.RUNTIME_DEPENDENCIES
        | {
            "iterative-stratification": "0.1.9",
            "scikit-learn": "1.9.0",
            "torch": "2.8.0",
            "transformers": "4.57.6",
            "safetensors": "0.8.0",
        }
        == orchestrator.RUNTIME_DEPENDENCIES
    )


def test_cost_guardrail_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="exceeds approved"):
        orchestrator.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("155"), approved_cost_usd=Decimal("100")
        )
    with pytest.raises(ValueError, match="<= 200"):
        orchestrator.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("155"), approved_cost_usd=Decimal("201")
        )


def test_preflight_is_exact_200_update_l4_trial() -> None:
    trials = orchestrator.make_preflight_trials(_contract())
    assert len(trials) == 1
    assert {trial["gpu_type"] for trial in trials} == {"L4"}
    assert {trial["config"]["updates"] for trial in trials} == {200}
    assert all(not trial["config"]["semantic_selection_allowed"] for trial in trials)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(), phase="preflight", trials=trials
    )
    assert orchestrator.validate_run_manifest(manifest) == manifest


def test_sweep_is_exact_unique_three_by_two_by_two_grid() -> None:
    trials = orchestrator.make_sweep_trials(_contract(), gpu_type="L4")
    assert len(trials) == 12
    assert len({trial["trial_id"] for trial in trials}) == 12
    assert {trial["gpu_type"] for trial in trials} == {"L4"}
    observed = {
        (
            trial["config"]["registered_config"]["encoder_learning_rate"],
            tuple(trial["config"]["registered_config"]["loss_weights"].values()),
            trial["config"]["registered_config"]["class_weights"],
        )
        for trial in trials
    }
    assert len(observed) == 12
    assert {trial["config"]["target_epochs"] for trial in trials} == {1}
    manifest = orchestrator.make_run_manifest(
        contract=_contract(), phase="sweep", trials=trials, rung_epochs=1
    )
    assert orchestrator.validate_run_manifest(manifest) == manifest


def test_sweep_manifest_rejects_grid_broadening() -> None:
    trials = orchestrator.make_sweep_trials(_contract(), gpu_type="L4")
    trials[0]["config"]["registered_config"]["encoder_learning_rate"] = 0.0001
    with pytest.raises(ValueError, match=r"content-addressed ID|broadened"):
        orchestrator.make_run_manifest(
            contract=_contract(), phase="sweep", trials=trials, rung_epochs=1
        )


def _asha_receipts(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    rung = manifest["asha"]["rung_epochs"]
    return {
        trial["trial_id"]: {
            "status": "complete",
            "trial_id": trial["trial_id"],
            "receipt_id": hashlib.sha256(f"receipt:{trial['trial_id']}".encode()).hexdigest(),
            "aggregate_metrics": {
                "completed_epochs": rung,
                "composite": 0.5,
                "material_recall": 0.9,
                "invalid_outputs": 0,
            },
            "artifacts": {
                "checkpoint": {
                    "relative_path": "checkpoint.bin",
                    "sha256": hashlib.sha256(trial["trial_id"].encode()).hexdigest(),
                    "bytes": len(trial["trial_id"]),
                }
            },
        }
        for trial in manifest["trials"]
    }


def test_asha_rungs_are_gated_12_to_4_to_2() -> None:
    contract = _contract()
    rung1_trials = orchestrator.make_sweep_trials(contract, gpu_type="L4", rung_epochs=1)
    rung1 = orchestrator.make_run_manifest(
        contract=contract, phase="sweep", trials=rung1_trials, rung_epochs=1
    )
    promotion_1_3 = orchestrator.build_asha_promotion_receipt(rung1, _asha_receipts(rung1))
    assert len(promotion_1_3["promoted"]) == 4

    rung3_trials = orchestrator.make_sweep_trials(
        contract,
        gpu_type="L4",
        rung_epochs=3,
        promotion_receipt=promotion_1_3,
    )
    rung3 = orchestrator.make_run_manifest(
        contract=contract,
        phase="sweep",
        trials=rung3_trials,
        rung_epochs=3,
        promotion_receipt=promotion_1_3,
    )
    assert len(rung3_trials) == 4
    assert {trial["config"]["target_epochs"] for trial in rung3_trials} == {3}
    promotion_3_6 = orchestrator.build_asha_promotion_receipt(rung3, _asha_receipts(rung3))
    rung6_trials = orchestrator.make_sweep_trials(
        contract,
        gpu_type="L4",
        rung_epochs=6,
        promotion_receipt=promotion_3_6,
    )
    assert len(rung6_trials) == 2
    assert {trial["config"]["target_epochs"] for trial in rung6_trials} == {6}


def test_asha_promotion_rejects_incomplete_rung_results() -> None:
    contract = _contract()
    trials = orchestrator.make_sweep_trials(contract, gpu_type="L4", rung_epochs=1)
    manifest = orchestrator.make_run_manifest(
        contract=contract, phase="sweep", trials=trials, rung_epochs=1
    )
    receipts = _asha_receipts(manifest)
    receipts.pop(next(iter(receipts)))
    with pytest.raises(ValueError, match="every rung candidate"):
        orchestrator.build_asha_promotion_receipt(manifest, receipts)


def _final_asha_manifest() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    contract = _contract()
    rung1_trials = orchestrator.make_sweep_trials(contract, gpu_type="L4", rung_epochs=1)
    rung1 = orchestrator.make_run_manifest(
        contract=contract, phase="sweep", trials=rung1_trials, rung_epochs=1
    )
    promotion_1_3 = orchestrator.build_asha_promotion_receipt(rung1, _asha_receipts(rung1))
    rung3_trials = orchestrator.make_sweep_trials(
        contract, gpu_type="L4", rung_epochs=3, promotion_receipt=promotion_1_3
    )
    rung3 = orchestrator.make_run_manifest(
        contract=contract,
        phase="sweep",
        trials=rung3_trials,
        rung_epochs=3,
        promotion_receipt=promotion_1_3,
    )
    promotion_3_6 = orchestrator.build_asha_promotion_receipt(rung3, _asha_receipts(rung3))
    rung6_trials = orchestrator.make_sweep_trials(
        contract, gpu_type="L4", rung_epochs=6, promotion_receipt=promotion_3_6
    )
    rung6 = orchestrator.make_run_manifest(
        contract=contract,
        phase="sweep",
        trials=rung6_trials,
        rung_epochs=6,
        promotion_receipt=promotion_3_6,
    )
    return rung6, _asha_receipts(rung6)


def _stability_evidence(
    manifest: dict[str, object], *, tied_recipe_means: bool = False
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    config_ids = sorted(
        {trial["config"]["registered_config"]["config_sha256"] for trial in manifest["trials"]}
    )
    receipts = {}
    histories = {}
    for trial in manifest["trials"]:
        config_id = trial["config"]["registered_config"]["config_sha256"]
        config_index = config_ids.index(config_id)
        seed = trial["config"]["seed"]
        composite = 0.70 if tied_recipe_means else (0.72 if config_index == 1 else 0.68)
        selected_composite = composite + (0.01 if seed == 13 else -0.01)
        prediction_payload = f"predictions:{trial['trial_id']}".encode()
        history = {
            "epochs": [
                {
                    "epoch": epoch,
                    "train": {},
                    "development": {"composite": selected_composite - 0.01 * (6 - epoch)},
                }
                for epoch in range(1, 7)
            ]
        }
        history_payload = (
            json.dumps(history, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode()
        checkpoint_artifacts = {
            f"artifact_{epoch:03d}": {
                "relative_path": f"checkpoints/epoch-{epoch:02d}.pt",
                "sha256": hashlib.sha256(
                    f"checkpoint:{trial['trial_id']}:{epoch}".encode()
                ).hexdigest(),
                "bytes": 64,
            }
            for epoch in range(1, 6)
        }
        selected_checkpoint = {
            "relative_path": "checkpoints/epoch-06.pt",
            "sha256": hashlib.sha256(f"checkpoint:{trial['trial_id']}:6".encode()).hexdigest(),
            "bytes": 64,
        }
        receipts[trial["trial_id"]] = {
            "schema_version": orchestrator.SCHEMA_VERSION,
            "status": "complete",
            "experiment_run_id": manifest["experiment_run_id"],
            "trial_id": trial["trial_id"],
            "trial_spec_sha256": orchestrator._canonical_sha256(trial),
            "bindings": manifest["experiment_contract"]["bindings"],
            "aggregate_metrics": {
                "completed_epochs": 6,
                "selected_epoch": 6,
                "composite": selected_composite,
                "material_recall": 0.9,
                "invalid_outputs": 0,
            },
            "artifacts": {
                **checkpoint_artifacts,
                "history": {
                    "relative_path": "history.json",
                    "sha256": hashlib.sha256(history_payload).hexdigest(),
                    "bytes": len(history_payload),
                },
                "development_predictions": {
                    "relative_path": "development-predictions.parquet",
                    "sha256": hashlib.sha256(prediction_payload).hexdigest(),
                    "bytes": len(prediction_payload),
                },
                "checkpoint": selected_checkpoint,
            },
        }
        _refresh_trial_receipt_id(receipts[trial["trial_id"]])
        histories[trial["trial_id"]] = history
    return receipts, histories


def test_stability_is_exact_top_two_by_seeds_and_fresh_training() -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    trials = orchestrator.make_stability_trials(_contract(), gpu_type="L4", source_receipt=source)
    assert len(trials) == 4
    assert {trial["config"]["seed"] for trial in trials} == {13, 29}
    assert {trial["config"]["target_epochs"] for trial in trials} == {6}
    assert all(trial["config"]["fresh_training"] for trial in trials)
    assert all(trial["config"]["continuation"] is None for trial in trials)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(),
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
    )
    assert orchestrator.validate_run_manifest(manifest) == manifest


def test_phase3_transition_authorises_only_four_fresh_stability_trials() -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    destination = _contract(trainer_code_sha256="4" * 64)
    with pytest.raises(ValueError, match="transition receipt"):
        orchestrator.make_stability_trials(destination, gpu_type="L4", source_receipt=source)
    transition = orchestrator.build_phase3_transition_receipt(
        final_manifest, final_receipts, destination
    )
    assert transition["source_trainer_code_sha256"] == "2" * 64
    assert transition["destination_trainer_code_sha256"] == "4" * 64
    assert transition["model_semantics_changed"] is False
    assert transition["training_semantics_changed"] is False
    assert transition["authorisation"]["confirmatory_authorised"] is False
    assert transition["authorisation"]["locked_test_authorised"] is False
    trials = orchestrator.make_stability_trials(
        destination,
        gpu_type="L4",
        source_receipt=source,
        transition_receipt=transition,
    )
    assert len(trials) == 4
    assert all(
        trial["config"]["phase3_transition_receipt_id"] == transition["transition_receipt_id"]
        for trial in trials
    )
    manifest = orchestrator.make_run_manifest(
        contract=destination,
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
        phase3_transition_receipt=transition,
    )
    assert orchestrator.validate_run_manifest(manifest) == manifest


def test_phase3_transition_can_bind_stability_adapter_fix() -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    destination = _contract(trainer_code_sha256="4" * 64)
    transition = orchestrator.build_phase3_transition_receipt(
        final_manifest,
        final_receipts,
        destination,
        include_stability_adapter_fix=True,
    )
    assert transition["code_delta_scope"][-1] == "stability_registered_config_adapter_fix"
    assert transition["training_semantics_changed"] is True
    assert (
        orchestrator.validate_phase3_transition_receipt(destination, source, transition)
        == transition
    )


def test_phase3_transition_rejects_semantic_or_authorisation_drift() -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    drifted_dependency = _contract(
        trainer_code_sha256="4" * 64,
        dependency_lock_sha256="5" * 64,
    )
    with pytest.raises(ValueError, match="semantic experiment drift"):
        orchestrator.build_phase3_transition_receipt(
            final_manifest, final_receipts, drifted_dependency
        )

    destination = _contract(trainer_code_sha256="4" * 64)
    transition = orchestrator.build_phase3_transition_receipt(
        final_manifest, final_receipts, destination
    )
    transition["authorisation"]["confirmatory_authorised"] = True
    body = {key: value for key, value in transition.items() if key != "transition_receipt_id"}
    transition["transition_receipt_id"] = orchestrator._canonical_sha256(body)
    with pytest.raises(ValueError, match="authorisation broadened"):
        orchestrator.validate_phase3_transition_receipt(destination, source, transition)


def test_recipe_uses_mean_across_both_seeds_and_digest_tie_break() -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    trials = orchestrator.make_stability_trials(_contract(), gpu_type="L4", source_receipt=source)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(),
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
    )
    receipts, histories = _stability_evidence(manifest)
    receipt = orchestrator.build_stability_recipe_receipt(manifest, receipts, histories)
    assert orchestrator.validate_stability_recipe_receipt(receipt) == receipt
    assert (
        receipt["selected_config_sha256"]
        == sorted(
            receipt["candidates"],
            key=lambda row: (-row["mean_development_composite"], row["config_sha256"]),
        )[0]["config_sha256"]
    )

    tied_receipts, tied_histories = _stability_evidence(manifest, tied_recipe_means=True)
    tied = orchestrator.build_stability_recipe_receipt(manifest, tied_receipts, tied_histories)
    assert tied["selected_config_sha256"] == min(row["config_sha256"] for row in tied["candidates"])


def test_recipe_accepts_only_proven_registered_early_stop() -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    trials = orchestrator.make_stability_trials(_contract(), gpu_type="L4", source_receipt=source)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(),
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
    )
    receipts, histories = _stability_evidence(manifest)
    trial_id = manifest["trials"][0]["trial_id"]
    receipt = receipts[trial_id]
    history = {
        "epochs": [
            {"epoch": 1, "train": {}, "development": {"composite": 0.40}},
            {"epoch": 2, "train": {}, "development": {"composite": 0.70}},
            {"epoch": 3, "train": {}, "development": {"composite": 0.69}},
            {"epoch": 4, "train": {}, "development": {"composite": 0.68}},
        ]
    }
    history_payload = (
        json.dumps(history, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    prediction = receipt["artifacts"]["development_predictions"]
    receipt["aggregate_metrics"].update(
        {"completed_epochs": 4, "selected_epoch": 2, "composite": 0.70}
    )
    receipt["artifacts"] = {
        "checkpoint": {
            "relative_path": "checkpoints/epoch-02.pt",
            "sha256": hashlib.sha256(f"checkpoint:{trial_id}:2".encode()).hexdigest(),
            "bytes": 64,
        },
        "checkpoint_01": {
            "relative_path": "checkpoints/epoch-01.pt",
            "sha256": hashlib.sha256(f"checkpoint:{trial_id}:1".encode()).hexdigest(),
            "bytes": 64,
        },
        "checkpoint_03": {
            "relative_path": "checkpoints/epoch-03.pt",
            "sha256": hashlib.sha256(f"checkpoint:{trial_id}:3".encode()).hexdigest(),
            "bytes": 64,
        },
        "checkpoint_04": {
            "relative_path": "checkpoints/epoch-04.pt",
            "sha256": hashlib.sha256(f"checkpoint:{trial_id}:4".encode()).hexdigest(),
            "bytes": 64,
        },
        "history": {
            "relative_path": "history.json",
            "sha256": hashlib.sha256(history_payload).hexdigest(),
            "bytes": len(history_payload),
        },
        "development_predictions": prediction,
    }
    _refresh_trial_receipt_id(receipts[trial_id])
    histories[trial_id] = history

    recipe = orchestrator.build_stability_recipe_receipt(manifest, receipts, histories)
    result = next(
        row
        for candidate in recipe["candidates"]
        for row in candidate["seed_results"]
        if row["trial_id"] == trial_id
    )
    assert result["completion"] == "registered_early_stop"
    assert result["completed_epochs"] == 4
    assert result["selected_epoch"] == 2
    assert result["stopped_epoch"] == 4

    invalid_history = json.loads(json.dumps(history))
    invalid_history["epochs"][3]["development"]["composite"] = 0.702
    invalid_payload = (
        json.dumps(invalid_history, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    receipts[trial_id]["aggregate_metrics"].update({"selected_epoch": 4, "composite": 0.702})
    receipts[trial_id]["artifacts"]["checkpoint"] = receipts[trial_id]["artifacts"].pop(
        "checkpoint_04"
    )
    receipts[trial_id]["artifacts"]["checkpoint"]["relative_path"] = "checkpoints/epoch-04.pt"
    receipts[trial_id]["artifacts"]["checkpoint_02"] = {
        "relative_path": "checkpoints/epoch-02.pt",
        "sha256": hashlib.sha256(f"checkpoint:{trial_id}:2".encode()).hexdigest(),
        "bytes": 64,
    }
    receipts[trial_id]["artifacts"]["history"].update(
        {"sha256": hashlib.sha256(invalid_payload).hexdigest(), "bytes": len(invalid_payload)}
    )
    _refresh_trial_receipt_id(receipts[trial_id])
    histories[trial_id] = invalid_history
    with pytest.raises(ValueError, match="not a registered early stop"):
        orchestrator.build_stability_recipe_receipt(manifest, receipts, histories)


@pytest.mark.parametrize("mutation", ["aggregate", "history_descriptor"])
def test_recipe_rejects_mutated_trial_receipt_without_new_content_address(
    mutation: str,
) -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    trials = orchestrator.make_stability_trials(_contract(), gpu_type="L4", source_receipt=source)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(),
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
    )
    receipts, histories = _stability_evidence(manifest)
    receipt = receipts[manifest["trials"][0]["trial_id"]]
    if mutation == "aggregate":
        receipt["aggregate_metrics"]["material_recall"] = 0.91
    else:
        receipt["artifacts"]["history"]["bytes"] += 1

    with pytest.raises(ValueError, match="content address drifted"):
        orchestrator.build_stability_recipe_receipt(manifest, receipts, histories)


def test_threshold_search_uses_two_bound_predictions_and_higher_tie_break() -> None:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    trials = orchestrator.make_stability_trials(_contract(), gpu_type="L4", source_receipt=source)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(),
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
    )
    receipts, histories = _stability_evidence(manifest)
    recipe = orchestrator.build_stability_recipe_receipt(manifest, receipts, histories)
    selected = next(
        row
        for row in recipe["candidates"]
        if row["config_sha256"] == recipe["selected_config_sha256"]
    )
    scores = []
    for seed_result in selected["seed_results"]:
        scores.append(
            {
                "trial_id": seed_result["trial_id"],
                "prediction_sha256": seed_result["development_prediction"]["sha256"],
                "development_rows": 222,
                "development_reference_sha256": "8" * 64,
                "threshold_composites": {
                    "0.30": 0.60,
                    "0.40": 0.70,
                    "0.50": 0.70,
                    "0.60": 0.65,
                    "0.70": 0.55,
                },
            }
        )
    threshold = orchestrator.build_target_threshold_receipt(recipe, scores)
    assert threshold["selected_threshold"] == "0.50"
    assert orchestrator.validate_target_threshold_receipt(recipe, threshold) == threshold
    scores[0]["prediction_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prediction hash drifted"):
        orchestrator.build_target_threshold_receipt(recipe, scores)


def _confirmatory_split_binding() -> dict[str, object]:
    counts = {
        101: (2_001, 5_000, 10_000),
        202: (2_001, 4_998, 10_000),
        303: (2_000, 5_001, 10_000),
    }
    return {
        "split_manifest_file_sha256": "1" * 64,
        "split_manifest_id": "9" * 64,
        "row_count": 10_000,
        "ladders": [
            {
                "ladder_seed": ladder["ladder_seed"],
                "optimiser_seed": ladder["optimiser_seed"],
                "budgets": [
                    {
                        "label_budget": label_budget,
                        "budget_name": (
                            f"train_{label_budget // 1000}k_{ladder['ladder_seed']}"
                        ),
                        "training_row_count": row_count,
                        "sample_ids_sha256": (
                            "f" * 64
                            if label_budget == 10_000
                            else hashlib.sha256(
                                f"{ladder['ladder_seed']}:{label_budget}".encode()
                            ).hexdigest()
                        ),
                    }
                    for label_budget, row_count in zip(
                        orchestrator.LABEL_BUDGETS,
                        counts[ladder["ladder_seed"]],
                        strict=True,
                    )
                ],
            }
            for ladder in orchestrator.REGISTERED_LADDERS
        ],
    }


def _confirmatory_selection() -> tuple[dict[str, object], dict[str, object]]:
    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    trials = orchestrator.make_stability_trials(_contract(), gpu_type="L4", source_receipt=source)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(),
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
    )
    receipts, histories = _stability_evidence(manifest)
    recipe = orchestrator.build_stability_recipe_receipt(manifest, receipts, histories)
    selected = next(
        row
        for row in recipe["candidates"]
        if row["config_sha256"] == recipe["selected_config_sha256"]
    )
    scores = [
        {
            "trial_id": row["trial_id"],
            "prediction_sha256": row["development_prediction"]["sha256"],
            "development_rows": 222,
            "development_reference_sha256": "8" * 64,
            "threshold_composites": {
                "0.30": 0.70,
                "0.40": 0.69,
                "0.50": 0.68,
                "0.60": 0.67,
                "0.70": 0.66,
            },
        }
        for row in selected["seed_results"]
    ]
    threshold = orchestrator.build_target_threshold_receipt(recipe, scores)
    assert threshold["selected_threshold"] == "0.30"
    return recipe, threshold


def test_confirmatory_manifest_is_exact_registered_nine_run_design() -> None:
    contract = _contract()
    recipe, threshold = _confirmatory_selection()
    split = _confirmatory_split_binding()
    trials = orchestrator.make_confirmatory_trials(
        contract,
        gpu_type="L4",
        recipe_receipt=recipe,
        threshold_receipt=threshold,
        split_binding=split,
    )
    assert len(trials) == 9
    assert len({row["trial_id"] for row in trials}) == 9
    assert {
        (
            row["config"]["ladder_seed"],
            row["config"]["optimiser_seed"],
            row["config"]["label_budget"],
        )
        for row in trials
    } == {
        (ladder["ladder_seed"], ladder["optimiser_seed"], budget)
        for ladder in orchestrator.REGISTERED_LADDERS
        for budget in orchestrator.LABEL_BUDGETS
    }
    assert {row["config"]["target_threshold"] for row in trials} == {"0.30"}
    assert {row["config"]["registered_config"]["config_sha256"] for row in trials} == {
        recipe["selected_config_sha256"]
    }
    assert all(row["config"]["fresh_training"] is True for row in trials)
    assert all(row["config"]["locked_test_authorised"] is False for row in trials)

    manifest = orchestrator.make_run_manifest(
        contract=contract,
        phase="confirmatory",
        trials=trials,
        recipe_receipt=recipe,
        threshold_receipt=threshold,
        confirmatory_split_binding=split,
    )
    assert orchestrator.validate_run_manifest(manifest) == manifest
    assert manifest["confirmatory"]["locked_test"] == {
        "authorised": False,
        "rows_accessed": 0,
        "predictions_authorised": False,
    }
    assert manifest["confirmatory"]["cost_guardrail"] == {
        "confirmatory_phase_upper_cost_usd": "65",
        "experiment_planned_upper_cost_usd": "155",
        "approved_cost_usd": "200",
        "hard_max_approved_cost_usd": "200",
    }


def test_confirmatory_manifest_rejects_ladder_or_selection_drift() -> None:
    contract = _contract()
    recipe, threshold = _confirmatory_selection()
    split = _confirmatory_split_binding()
    split["ladders"][1]["budgets"][1]["training_row_count"] = 5_000
    with pytest.raises(ValueError, match="row count or name drifted"):
        orchestrator.make_confirmatory_trials(
            contract,
            gpu_type="L4",
            recipe_receipt=recipe,
            threshold_receipt=threshold,
            split_binding=split,
        )

    valid_split = _confirmatory_split_binding()
    threshold["selected_threshold"] = "0.40"
    with pytest.raises(ValueError, match=r"content or selection drifted|frozen 0.30"):
        orchestrator.make_confirmatory_trials(
            contract,
            gpu_type="L4",
            recipe_receipt=recipe,
            threshold_receipt=threshold,
            split_binding=valid_split,
        )


def test_confirmatory_transition_authorises_only_exact_nine_run_phase() -> None:
    source = _contract()
    destination = _contract(trainer_code_sha256="4" * 64)
    recipe, threshold = _confirmatory_selection()
    transition = orchestrator.build_confirmatory_transition_receipt(
        source,
        destination,
        recipe,
        threshold,
    )
    assert (
        orchestrator.validate_confirmatory_transition_receipt(
            destination,
            recipe,
            threshold,
            transition,
        )
        == transition
    )
    assert transition["source_experiment_run_id"] == orchestrator.experiment_run_id(source)
    assert transition["destination_experiment_run_id"] == orchestrator.experiment_run_id(
        destination
    )
    assert transition["authorisation"]["locked_test_authorised"] is False
    split = _confirmatory_split_binding()
    trials = orchestrator.make_confirmatory_trials(
        destination,
        gpu_type="L4",
        recipe_receipt=recipe,
        threshold_receipt=threshold,
        split_binding=split,
        transition_receipt=transition,
    )
    assert len(trials) == 9
    assert {row["config"]["confirmatory_transition_receipt_id"] for row in trials} == {
        transition["transition_receipt_id"]
    }
    manifest = orchestrator.make_run_manifest(
        contract=destination,
        phase="confirmatory",
        trials=trials,
        recipe_receipt=recipe,
        threshold_receipt=threshold,
        confirmatory_split_binding=split,
        confirmatory_transition_receipt=transition,
    )
    assert orchestrator.validate_run_manifest(manifest) == manifest


def test_confirmatory_transition_rejects_authorisation_broadening() -> None:
    source = _contract()
    destination = _contract(trainer_code_sha256="4" * 64)
    recipe, threshold = _confirmatory_selection()
    transition = orchestrator.build_confirmatory_transition_receipt(
        source,
        destination,
        recipe,
        threshold,
    )
    transition["authorisation"]["locked_test_authorised"] = True
    body = {key: value for key, value in transition.items() if key != "transition_receipt_id"}
    transition["transition_receipt_id"] = orchestrator._canonical_sha256(body)
    with pytest.raises(ValueError, match="authorisation broadened"):
        orchestrator.validate_confirmatory_transition_receipt(
            destination,
            recipe,
            threshold,
            transition,
        )


def test_private_stability_freezer_publishes_only_two_metadata_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reddit_china_stance.modernbert_training as training

    final_manifest, final_receipts = _final_asha_manifest()
    source = orchestrator.build_stability_source_receipt(final_manifest, final_receipts)
    trials = orchestrator.make_stability_trials(_contract(), gpu_type="L4", source_receipt=source)
    manifest = orchestrator.make_run_manifest(
        contract=_contract(),
        phase="stability",
        trials=trials,
        stability_source_receipt=source,
    )
    receipts, histories = _stability_evidence(manifest)
    run_root = (
        tmp_path
        / orchestrator.OUTPUT_PREFIX
        / f"run={manifest['experiment_run_id']}"
        / "phase=stability"
    )
    for trial in trials:
        trial_id = trial["trial_id"]
        trial_root = run_root / f"trial={trial_id}"
        trial_root.mkdir(parents=True)
        history_payload = (
            json.dumps(
                histories[trial_id],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        (trial_root / "history.json").write_bytes(history_payload)
        private_payload = json.dumps({"private_trial": trial_id}).encode()
        prediction = receipts[trial_id]["artifacts"]["development_predictions"]
        prediction.update(
            {
                "relative_path": "development-predictions.json",
                "sha256": hashlib.sha256(private_payload).hexdigest(),
                "bytes": len(private_payload),
            }
        )
        _refresh_trial_receipt_id(receipts[trial_id])
        (trial_root / "development-predictions.json").write_bytes(private_payload)
        (trial_root / "receipt.json").write_text(json.dumps(receipts[trial_id]), encoding="utf-8")

    monkeypatch.setattr(orchestrator, "validate_trial_receipt", lambda **_kwargs: {})
    monkeypatch.setattr(
        training,
        "score_private_development_thresholds",
        lambda _payload, *, development_proxy_path, expected_composite: {
            "development_rows": 222,
            "development_reference_sha256": "8" * 64,
            "threshold_composites": {
                "0.30": expected_composite,
                "0.40": expected_composite,
                "0.50": expected_composite,
                "0.60": expected_composite,
                "0.70": expected_composite,
            },
        },
    )

    result = orchestrator._freeze_stability_selection(manifest, volume_root=tmp_path)
    repeated = orchestrator._freeze_stability_selection(manifest, volume_root=tmp_path)

    assert repeated == result
    assert result["status"] == "frozen"
    selection_root = (
        tmp_path / orchestrator.OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}" / "selection"
    )
    assert sorted(path.name for path in selection_root.iterdir()) == [
        "stability-recipe.json",
        "target-threshold.json",
    ]
    public_text = "\n".join(path.read_text() for path in selection_root.iterdir())
    assert "private_trial" not in public_text
    assert "source_sample_id" not in public_text


def test_promoted_rung_cannot_advance_without_published_receipt_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    rung1_trials = orchestrator.make_sweep_trials(contract, gpu_type="L4", rung_epochs=1)
    rung1 = orchestrator.make_run_manifest(
        contract=contract, phase="sweep", trials=rung1_trials, rung_epochs=1
    )
    promotion = orchestrator.build_asha_promotion_receipt(rung1, _asha_receipts(rung1))
    rung3_trials = orchestrator.make_sweep_trials(
        contract, gpu_type="L4", rung_epochs=3, promotion_receipt=promotion
    )
    rung3 = orchestrator.make_run_manifest(
        contract=contract,
        phase="sweep",
        trials=rung3_trials,
        rung_epochs=3,
        promotion_receipt=promotion,
    )
    monkeypatch.setattr(orchestrator, "VOLUME_PATH", tmp_path)
    with pytest.raises(RuntimeError, match="promotion receipt"):
        orchestrator._inspect_exact_trials(rung3)

    run_root = tmp_path / orchestrator.OUTPUT_PREFIX / f"run={rung3['experiment_run_id']}"
    promotion_path = orchestrator._promotion_path(run_root, from_rung=1, to_rung=3)
    promotion_path.parent.mkdir(parents=True)
    promotion_path.write_text(json.dumps(promotion), encoding="utf-8")
    for promoted in promotion["promoted"]:
        checkpoint_path = run_root / promoted["checkpoint_relative_path"]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(promoted["source_trial_id"], encoding="utf-8")
    inspection = orchestrator._inspect_exact_trials(rung3)
    assert len(inspection["missing_trial_ids"]) == 4


def test_later_phase_rejects_extra_trials_or_unsupported_gpu() -> None:
    contract = _contract()
    trial = orchestrator._trial_spec(
        phase="chronological",
        name="chronological-one",
        gpu_type="L4",
        config={"frozen_recipe_sha256": "4" * 64, "seed": 13},
    )
    with pytest.raises(ValueError, match="exactly 1 trials"):
        orchestrator.make_run_manifest(
            contract=contract, phase="chronological", trials=[trial, dict(trial)]
        )
    with pytest.raises(ValueError, match="gpu_type must be one of"):
        orchestrator._trial_spec(
            phase="stability",
            name="unsupported",
            gpu_type="L40S",
            config={"frozen_recipe_sha256": "4" * 64, "seed": 1},
        )


def test_validate_trial_receipt_checks_bindings_and_artifact_hashes(tmp_path: Path) -> None:
    contract = _contract()
    trial = orchestrator.make_sweep_trials(contract, gpu_type="L4")[0]
    # A one-row manifest is not a valid launch manifest; construct the job shape directly for
    # receipt validation, which is intentionally trial-local.
    job = {
        "experiment_run_id": orchestrator.experiment_run_id(contract),
        "experiment_contract": contract,
        "trial_spec": trial,
    }
    artifact = _write_artifact(tmp_path / "metrics.json", b"{}\n")
    receipt = {
        "schema_version": orchestrator.SCHEMA_VERSION,
        "status": "complete",
        "experiment_run_id": job["experiment_run_id"],
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": orchestrator._canonical_sha256(trial),
        "bindings": contract["bindings"],
        "artifacts": {"metrics": artifact},
        "compute": {
            "gpu_type": "L4",
            "wall_seconds": 10,
            "gpu_seconds": 10,
            "estimated_cost_usd": "0.01",
        },
    }
    result = orchestrator.validate_trial_receipt(trial_root=tmp_path, receipt=receipt, job=job)
    assert result["status"] == "validated"
    (tmp_path / "metrics.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash or size mismatch"):
        orchestrator.validate_trial_receipt(trial_root=tmp_path, receipt=receipt, job=job)


def test_sharded_trial_inspection_validates_and_aggregates_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    trial = orchestrator._trial_spec(
        phase="chronological",
        name="chronological-one",
        gpu_type="L4",
        config={"frozen_recipe_sha256": "4" * 64, "seed": 13},
    )
    manifest = orchestrator.make_run_manifest(
        contract=contract, phase="chronological", trials=[trial]
    )
    monkeypatch.setattr(orchestrator, "VOLUME_PATH", tmp_path)
    final_root = (
        tmp_path
        / orchestrator.OUTPUT_PREFIX
        / f"run={manifest['experiment_run_id']}"
        / "phase=chronological"
        / f"trial={trial['trial_id']}"
    )
    artifact = _write_artifact(final_root / "metrics.json", b"{}\n")
    job = orchestrator._trial_job(manifest, trial)
    receipt = {
        "schema_version": orchestrator.SCHEMA_VERSION,
        "status": "complete",
        "experiment_run_id": manifest["experiment_run_id"],
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": job["trial_spec_sha256"],
        "bindings": contract["bindings"],
        "artifacts": {"metrics": artifact},
        "compute": {
            "gpu_type": "L4",
            "wall_seconds": 10,
            "gpu_seconds": 10,
            "estimated_cost_usd": "0.01",
        },
    }
    (final_root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    result = orchestrator._inspect_exact_trial(manifest, trial["trial_id"])
    assert result["disposition"] == "complete"
    aggregate = orchestrator._aggregate_trial_inspections(manifest, [result])
    assert aggregate["complete_trial_ids"] == [trial["trial_id"]]
    assert aggregate["estimated_completed_cost_usd"] == "0.01"

    with pytest.raises(RuntimeError, match="exactly cover"):
        orchestrator._aggregate_trial_inspections(manifest, [])


def test_sharded_trial_inspection_rejects_final_and_incomplete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    trial = orchestrator._trial_spec(
        phase="chronological",
        name="chronological-one",
        gpu_type="L4",
        config={"frozen_recipe_sha256": "4" * 64, "seed": 13},
    )
    manifest = orchestrator.make_run_manifest(
        contract=contract, phase="chronological", trials=[trial]
    )
    monkeypatch.setattr(orchestrator, "VOLUME_PATH", tmp_path)
    run_root = tmp_path / orchestrator.OUTPUT_PREFIX / f"run={manifest['experiment_run_id']}"
    (run_root / "phase=chronological" / f"trial={trial['trial_id']}").mkdir(parents=True)
    (run_root / ".incomplete" / f"trial={trial['trial_id']}").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="final and incomplete"):
        orchestrator._inspect_exact_trial(manifest, trial["trial_id"])


def test_inspection_resumes_only_exact_bound_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    # The one-run chronological phase keeps this resume test metadata-only.
    trial = orchestrator._trial_spec(
        phase="chronological",
        name="chronological-one",
        gpu_type="L4",
        config={"frozen_recipe_sha256": "4" * 64, "seed": 13},
    )
    manifest = orchestrator.make_run_manifest(
        contract=contract, phase="chronological", trials=[trial]
    )
    monkeypatch.setattr(orchestrator, "VOLUME_PATH", tmp_path)
    incomplete = (
        tmp_path
        / orchestrator.OUTPUT_PREFIX
        / f"run={manifest['experiment_run_id']}"
        / ".incomplete"
        / f"trial={trial['trial_id']}"
    )
    checkpoint = _write_artifact(incomplete / "checkpoint.bin", b"checkpoint")
    resume = {
        "schema_version": orchestrator.SCHEMA_VERSION,
        "experiment_run_id": manifest["experiment_run_id"],
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": orchestrator._canonical_sha256(trial),
        "checkpoint": checkpoint,
    }
    (incomplete / "resume.json").write_text(json.dumps(resume), encoding="utf-8")
    result = orchestrator._inspect_exact_trials(manifest)
    assert result["resumable_trial_ids"] == [trial["trial_id"]]
    resume["trial_spec_sha256"] = "0" * 64
    (incomplete / "resume.json").write_text(json.dumps(resume), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding mismatch"):
        orchestrator._inspect_exact_trials(manifest)
