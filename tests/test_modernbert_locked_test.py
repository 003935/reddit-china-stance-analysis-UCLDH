from __future__ import annotations

from typing import Any

import reddit_china_stance.modernbert_locked_test as locked
from reddit_china_stance.modernbert_training import (
    LABEL_BUDGETS,
    REGISTERED_LADDERS,
    canonical_sha256,
)
from reddit_china_stance.semantic_evaluation import score_semantic_labels

SHA = "a" * 64


def _source_evidence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials = []
    receipts = []
    for ladder in REGISTERED_LADDERS:
        for budget in LABEL_BUDGETS:
            config = {
                "label_budget": budget,
                "training_row_count": budget,
                "ladder_seed": ladder["ladder_seed"],
                "optimiser_seed": ladder["optimiser_seed"],
            }
            trial_body = {
                "schema_version": "1.0.0",
                "phase": "confirmatory",
                "gpu_type": "L4",
                "name": f"synthetic-{ladder['ladder_seed']}-{budget}",
                "config": config,
            }
            trial = {**trial_body, "trial_id": canonical_sha256(trial_body)}
            trials.append(trial)
            receipt_body = {
                "status": "complete",
                "experiment_run_id": "b" * 64,
                "trial_id": trial["trial_id"],
                "trial_spec_sha256": canonical_sha256(trial),
                "artifacts": {
                    "checkpoint": {
                        "relative_path": "checkpoints/selected.pt",
                        "sha256": canonical_sha256({"checkpoint": trial["trial_id"]}),
                        "bytes": 100,
                    }
                },
            }
            receipts.append({**receipt_body, "receipt_id": canonical_sha256(receipt_body)})
    source = {
        "phase": "confirmatory",
        "experiment_run_id": "b" * 64,
        "phase_run_id": "c" * 64,
        "experiment_contract": {
            "bindings": {
                "model_id": "answerdotai/ModernBERT-large",
                "model_revision": "d" * 40,
                "development_proxy_sha256": "e" * 64,
            }
        },
        "trials": trials,
        "confirmatory": {
            "recipe_receipt": {"recipe_receipt_id": "f" * 64},
            "threshold_receipt": {"threshold_receipt_id": "1" * 64},
            "locked_test": {
                "authorised": False,
                "rows_accessed": 0,
                "predictions_authorised": False,
            },
        },
    }
    return source, receipts


def test_locked_manifest_predictions_and_plateau_aggregate(monkeypatch: Any) -> None:
    monkeypatch.setattr(locked, "BOOTSTRAP_REPLICATES", 25)
    source, source_receipts = _source_evidence()
    authorisation = locked.build_locked_test_authorisation(source, source_receipts)
    manifest = locked.build_locked_test_manifest(
        source,
        confirmatory_receipts=source_receipts,
        authorisation=authorisation,
        source_manifest_sha256=SHA,
        evaluator_code_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
    )
    locked.validate_locked_test_manifest(manifest)

    targets = ("china_general", "government_ccp", "people_culture")
    reference = {
        f"opaque-{index:03d}": {
            "relevance": "material",
            "target_stances": [{"target": targets[index % len(targets)], "stance": "negative"}],
        }
        for index in range(locked.LOCKED_TEST_ROWS)
    }
    payloads = {}
    trial_receipts = []
    for trial in manifest["trials"]:
        rows = [
            {
                "source_sample_id": item_id,
                "relevance_logits": [10.0, 0.0, 0.0],
                "target_logits": [
                    10.0 if target == label["target_stances"][0]["target"] else -10.0
                    for target in (*targets, "other")
                ],
                "stance_logits": [[10.0, 0.0, 0.0, 0.0, 0.0] for _ in range(4)],
                "decoded_label": label,
            }
            for item_id, label in sorted(reference.items())
        ]
        payload = {
            "schema_version": "1.0.0",
            "kind": locked.PREDICTION_KIND,
            "locked_test_run_id": manifest["locked_test_run_id"],
            "authorisation_id": manifest["authorisation"]["authorisation_id"],
            "source_trial_id": trial["source_trial_id"],
            "source_receipt_id": trial["source_receipt_id"],
            "row_count": locked.LOCKED_TEST_ROWS,
            "decoder_target_threshold": 0.3,
            "rows": rows,
        }
        predictions = {row["source_sample_id"]: row["decoded_label"] for row in rows}
        metrics = score_semantic_labels(reference, predictions)
        metrics["decoding"] = {"target_threshold": 0.3, "forced_target_selections": 0}
        trial_receipt = locked.build_locked_trial_receipt(
            manifest,
            trial=trial,
            prediction_descriptor={
                "relative_path": "predictions.json",
                "sha256": canonical_sha256(payload),
                "bytes": 1,
            },
            metrics=metrics,
            wall_seconds=1.0,
        )
        payloads[trial["source_trial_id"]] = payload
        trial_receipts.append(trial_receipt)

    aggregate = locked.build_locked_test_aggregate(
        manifest,
        reference=reference,
        prediction_payloads=payloads,
        trial_receipts=trial_receipts,
    )
    assert aggregate["scale_gate"] == {
        "verdict": "plateau_stop",
        "passed": False,
        "criteria": {
            "mean_gain_at_least_0_02": False,
            "bootstrap_interval_excludes_zero": False,
            "no_supported_core_target_regression_over_0_05": True,
        },
        "regression_breaches": [],
    }
    assert aggregate["invalid_outputs_total"] == 0
