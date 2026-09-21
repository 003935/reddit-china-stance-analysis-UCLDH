"""Frozen one-time locked-test contracts and aggregate scoring for ModernBERT.

This module contains no Modal or Torch imports. It freezes the exact evaluation
design before the 230 locked rows are materialised and keeps all public outputs
metadata-only.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal
from typing import Any

from reddit_china_stance.modernbert_training import (
    GPU_RATE_USD_PER_SECOND,
    LABEL_BUDGETS,
    REGISTERED_LADDERS,
    canonical_sha256,
    validate_private_development_predictions,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation import CORE_TARGETS, score_semantic_labels

SCHEMA_VERSION = "1.0.0"
MANIFEST_KIND = "modernbert-locked-test-manifest-v1"
PREDICTION_KIND = "modernbert-private-locked-test-predictions-v1"
TRIAL_RECEIPT_KIND = "modernbert-locked-test-trial-receipt-v1"
AGGREGATE_KIND = "modernbert-locked-test-aggregate-v1"
FINAL_RECEIPT_KIND = "modernbert-locked-test-final-receipt-v1"
AUTHORISATION_KIND = "modernbert-modal-locked-test-authorisation-v1"

LOCKED_TEST_ROWS = 230
TARGET_THRESHOLD = "0.30"
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = "0.95"

SCALE_GATES = {
    "mean_core_tuple_f1_gain_min": "0.02",
    "bootstrap_interval_excludes_zero": True,
    "core_target_support_min": 10,
    "per_core_target_f1_regression_max": "0.05",
    "plateau_below": "0.01",
    "inconclusive_below": "0.02",
}


def build_locked_test_authorisation(
    confirmatory_manifest: Mapping[str, Any],
    confirmatory_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authorise one access after the exact current nine-run Modal contract validates."""

    trials = confirmatory_manifest.get("trials")
    confirmatory = confirmatory_manifest.get("confirmatory")
    if (
        confirmatory_manifest.get("phase") != "confirmatory"
        or not isinstance(trials, list)
        or len(trials) != 9
        or not isinstance(confirmatory, Mapping)
    ):
        raise ValueError("locked-test authorisation requires the exact confirmatory manifest")
    specs = {str(row.get("trial_id")): row for row in trials}
    receipts = {str(row.get("trial_id")): row for row in confirmatory_receipts}
    if len(specs) != 9 or len(receipts) != 9 or set(specs) != set(receipts):
        raise ValueError("locked-test authorisation requires nine exact receipts")
    expected_conditions = {
        (int(row["ladder_seed"]), int(row["optimiser_seed"]), budget)
        for row in REGISTERED_LADDERS
        for budget in LABEL_BUDGETS
    }
    observed_conditions = set()
    receipt_ids = []
    for trial_id, trial in specs.items():
        receipt = receipts[trial_id]
        config = trial.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("confirmatory trial config is missing")
        if (
            receipt.get("status") != "complete"
            or receipt.get("experiment_run_id") != confirmatory_manifest.get("experiment_run_id")
            or receipt.get("trial_spec_sha256") != canonical_sha256(trial)
        ):
            raise ValueError("confirmatory receipt binding drifted")
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        receipt_id = _require_sha256(receipt.get("receipt_id"), where="confirmatory receipt ID")
        if canonical_sha256(receipt_body) != receipt_id:
            raise ValueError("confirmatory receipt content address drifted")
        observed_conditions.add(
            (config["ladder_seed"], config["optimiser_seed"], config["label_budget"])
        )
        receipt_ids.append(receipt_id)
    if observed_conditions != expected_conditions:
        raise ValueError("confirmatory receipts do not cover the registered 3 x 3 design")
    locked_contract = confirmatory.get("locked_test")
    if locked_contract != {
        "authorised": False,
        "rows_accessed": 0,
        "predictions_authorised": False,
    }:
        raise ValueError("source confirmatory manifest has already broadened locked-test access")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": AUTHORISATION_KIND,
        "source_experiment_run_id": confirmatory_manifest["experiment_run_id"],
        "source_phase_run_id": confirmatory_manifest["phase_run_id"],
        "recipe_receipt_id": confirmatory["recipe_receipt"]["recipe_receipt_id"],
        "threshold_receipt_id": confirmatory["threshold_receipt"]["threshold_receipt_id"],
        "confirmatory_receipt_ids": sorted(receipt_ids),
        "locked_test_rows": LOCKED_TEST_ROWS,
        "single_access": True,
    }
    assert_metadata_only(body, where="locked-test authorisation")
    return {**body, "authorisation_id": canonical_sha256(body)}


def _require_sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    try:
        parsed = int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest") from exc
    if value != f"{parsed:064x}":
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _descriptor(value: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if set(value) != {"relative_path", "sha256", "bytes"}:
        raise ValueError(f"{where} descriptor has unexpected fields")
    path = value.get("relative_path")
    size = value.get("bytes")
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"{where} descriptor path is unsafe")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"{where} descriptor size must be positive")
    return {
        "relative_path": path,
        "sha256": _require_sha256(value.get("sha256"), where=f"{where}.sha256"),
        "bytes": size,
    }


def build_locked_test_manifest(
    confirmatory_manifest: Mapping[str, Any],
    *,
    confirmatory_receipts: Sequence[Mapping[str, Any]],
    authorisation: Mapping[str, Any],
    source_manifest_sha256: str,
    evaluator_code_sha256: str,
    dependency_lock_sha256: str,
) -> dict[str, Any]:
    """Freeze the exact nine-checkpoint evaluation before locked-row access."""

    trials = confirmatory_manifest.get("trials")
    confirmatory = confirmatory_manifest.get("confirmatory")
    contract = confirmatory_manifest.get("experiment_contract")
    if (
        confirmatory_manifest.get("phase") != "confirmatory"
        or not isinstance(trials, list)
        or len(trials) != 9
        or not isinstance(confirmatory, Mapping)
        or not isinstance(contract, Mapping)
    ):
        raise ValueError("locked-test preparation requires the exact confirmatory manifest")
    receipts = {str(row.get("trial_id")): dict(row) for row in confirmatory_receipts}
    if len(receipts) != 9 or set(receipts) != {str(row.get("trial_id")) for row in trials}:
        raise ValueError("locked-test preparation requires nine exact confirmatory receipts")
    authorisation_id = _require_sha256(
        authorisation.get("authorisation_id"), where="authorisation.authorisation_id"
    )
    authorisation_body = {
        key: value for key, value in authorisation.items() if key != "authorisation_id"
    }
    if canonical_sha256(authorisation_body) != authorisation_id:
        raise ValueError("locked-test authorisation content address drifted")
    if (
        authorisation.get("kind") != AUTHORISATION_KIND
        or authorisation.get("source_experiment_run_id")
        != confirmatory_manifest.get("experiment_run_id")
        or authorisation.get("source_phase_run_id") != confirmatory_manifest.get("phase_run_id")
        or authorisation.get("recipe_receipt_id")
        != confirmatory["recipe_receipt"]["recipe_receipt_id"]
        or authorisation.get("threshold_receipt_id")
        != confirmatory["threshold_receipt"]["threshold_receipt_id"]
        or authorisation.get("locked_test_rows") != LOCKED_TEST_ROWS
        or authorisation.get("single_access") is not True
    ):
        raise ValueError("locked-test authorisation binding drifted")
    if sorted(authorisation.get("confirmatory_receipt_ids", [])) != sorted(
        receipt["receipt_id"] for receipt in receipts.values()
    ):
        raise ValueError("locked-test authorisation receipt set drifted")

    frozen_trials = []
    for trial in trials:
        trial_id = str(trial["trial_id"])
        receipt = receipts[trial_id]
        config = trial.get("config")
        artifacts = receipt.get("artifacts")
        if not isinstance(config, Mapping) or not isinstance(artifacts, Mapping):
            raise ValueError("confirmatory trial or receipt is incomplete")
        checkpoint = artifacts.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("confirmatory receipt lacks its selected checkpoint")
        frozen_trials.append(
            {
                "source_trial_id": trial_id,
                "source_trial_spec_sha256": canonical_sha256(trial),
                "source_receipt_id": _require_sha256(
                    receipt.get("receipt_id"), where="confirmatory receipt ID"
                ),
                "label_budget": int(config["label_budget"]),
                "training_row_count": int(config["training_row_count"]),
                "ladder_seed": int(config["ladder_seed"]),
                "optimiser_seed": int(config["optimiser_seed"]),
                "checkpoint": _descriptor(checkpoint, where="selected checkpoint"),
            }
        )
    frozen_trials.sort(key=lambda row: (row["ladder_seed"], row["label_budget"]))
    expected = {
        (int(row["ladder_seed"]), int(row["optimiser_seed"]), budget)
        for row in REGISTERED_LADDERS
        for budget in LABEL_BUDGETS
    }
    observed = {
        (row["ladder_seed"], row["optimiser_seed"], row["label_budget"]) for row in frozen_trials
    }
    if observed != expected:
        raise ValueError("locked-test trials do not cover the registered 3 x 3 design")

    bindings = contract.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("confirmatory experiment bindings are absent")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "source_experiment_run_id": confirmatory_manifest["experiment_run_id"],
        "source_phase_run_id": confirmatory_manifest["phase_run_id"],
        "source_manifest_sha256": _require_sha256(
            source_manifest_sha256, where="source_manifest_sha256"
        ),
        "evaluator_code_sha256": _require_sha256(
            evaluator_code_sha256, where="evaluator_code_sha256"
        ),
        "dependency_lock_sha256": _require_sha256(
            dependency_lock_sha256, where="dependency_lock_sha256"
        ),
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
        "development_proxy_sha256": bindings["development_proxy_sha256"],
        "recipe_receipt_id": authorisation["recipe_receipt_id"],
        "threshold_receipt_id": authorisation["threshold_receipt_id"],
        "authorisation": deepcopy(dict(authorisation)),
        "locked_test": {
            "split": "locked_test_candidate",
            "row_count": LOCKED_TEST_ROWS,
            "single_access": True,
            "target_threshold": TARGET_THRESHOLD,
        },
        "bootstrap": {
            "method": "paired_hierarchical_pairs_then_threads_percentile",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "estimand": "mean_10k_minus_5k_core_target_stance_tuple_micro_f1",
        },
        "scale_gates": dict(SCALE_GATES),
        "trials": frozen_trials,
    }
    assert_metadata_only(body, where="locked-test manifest")
    return {**body, "locked_test_run_id": canonical_sha256(body)}


def validate_locked_test_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "source_experiment_run_id",
        "source_phase_run_id",
        "source_manifest_sha256",
        "evaluator_code_sha256",
        "dependency_lock_sha256",
        "model_id",
        "model_revision",
        "development_proxy_sha256",
        "recipe_receipt_id",
        "threshold_receipt_id",
        "authorisation",
        "locked_test",
        "bootstrap",
        "scale_gates",
        "trials",
        "locked_test_run_id",
    }
    if set(value) != expected_keys or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("locked-test manifest schema drifted")
    body = {key: deepcopy(item) for key, item in value.items() if key != "locked_test_run_id"}
    if value.get("kind") != MANIFEST_KIND or value.get("locked_test_run_id") != canonical_sha256(
        body
    ):
        raise ValueError("locked-test manifest content address drifted")
    for key in (
        "source_experiment_run_id",
        "source_phase_run_id",
        "source_manifest_sha256",
        "evaluator_code_sha256",
        "dependency_lock_sha256",
        "development_proxy_sha256",
        "recipe_receipt_id",
        "threshold_receipt_id",
    ):
        _require_sha256(value.get(key), where=f"locked manifest.{key}")
    if value.get("locked_test") != {
        "split": "locked_test_candidate",
        "row_count": LOCKED_TEST_ROWS,
        "single_access": True,
        "target_threshold": TARGET_THRESHOLD,
    }:
        raise ValueError("locked-test access contract drifted")
    if (
        value.get("bootstrap")
        != {
            "method": "paired_hierarchical_pairs_then_threads_percentile",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "estimand": "mean_10k_minus_5k_core_target_stance_tuple_micro_f1",
        }
        or value.get("scale_gates") != SCALE_GATES
    ):
        raise ValueError("locked-test statistical contract drifted")
    trials = value.get("trials")
    if not isinstance(trials, list) or len(trials) != 9:
        raise ValueError("locked-test manifest must contain nine trials")
    expected = {
        (int(row["ladder_seed"]), int(row["optimiser_seed"]), budget)
        for row in REGISTERED_LADDERS
        for budget in LABEL_BUDGETS
    }
    observed = set()
    source_ids = set()
    receipt_ids = set()
    for row in trials:
        if set(row) != {
            "source_trial_id",
            "source_trial_spec_sha256",
            "source_receipt_id",
            "label_budget",
            "training_row_count",
            "ladder_seed",
            "optimiser_seed",
            "checkpoint",
        }:
            raise ValueError("locked-test trial binding schema drifted")
        source_ids.add(_require_sha256(row["source_trial_id"], where="source_trial_id"))
        receipt_ids.add(_require_sha256(row["source_receipt_id"], where="source_receipt_id"))
        _require_sha256(row["source_trial_spec_sha256"], where="source_trial_spec_sha256")
        _descriptor(row["checkpoint"], where="selected checkpoint")
        observed.add((row["ladder_seed"], row["optimiser_seed"], row["label_budget"]))
    if len(source_ids) != 9 or len(receipt_ids) != 9 or observed != expected:
        raise ValueError("locked-test trial set drifted")
    assert_metadata_only(value, where="locked-test manifest")
    return deepcopy(dict(value))


def validate_private_locked_predictions(
    value: Mapping[str, Any], *, expected_trial: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "locked_test_run_id",
        "authorisation_id",
        "source_trial_id",
        "source_receipt_id",
        "row_count",
        "decoder_target_threshold",
        "rows",
    }
    if set(value) != expected_keys or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("private locked-test prediction schema drifted")
    if (
        value.get("kind") != PREDICTION_KIND
        or value.get("locked_test_run_id") != run_id
        or value.get("source_trial_id") != expected_trial["source_trial_id"]
        or value.get("source_receipt_id") != expected_trial["source_receipt_id"]
        or value.get("row_count") != LOCKED_TEST_ROWS
        or float(value.get("decoder_target_threshold")) != float(TARGET_THRESHOLD)
    ):
        raise ValueError("private locked-test prediction binding drifted")
    projected = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-private-development-predictions-v1",
        "row_count": value["row_count"],
        "decoder_target_threshold": value["decoder_target_threshold"],
        "rows": value["rows"],
    }
    validate_private_development_predictions(projected, expected_rows=LOCKED_TEST_ROWS)
    return deepcopy(dict(value))


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "items",
        "invalid_outputs",
        "relevance_macro_f1",
        "material_recall",
        "core_target_micro_f1",
        "core_target_stance_tuple_micro_f1",
        "fixed_reference_core_target_stance_accuracy",
        "forced_target_selections",
        "core_target_f1",
        "core_target_support",
    }
    if set(metrics) == expected_keys:
        return deepcopy(dict(metrics))
    return {
        "items": metrics["items"],
        "invalid_outputs": metrics["invalid_outputs"],
        "relevance_macro_f1": metrics["relevance"]["macro_f1"],
        "material_recall": metrics["relevance"]["material_recall"],
        "core_target_micro_f1": metrics["targets"]["core"]["micro"]["f1"],
        "core_target_stance_tuple_micro_f1": metrics["end_to_end_core_target_stance"]["micro"][
            "f1"
        ],
        "fixed_reference_core_target_stance_accuracy": metrics["stance"][
            "fixed_reference_target_core"
        ]["accuracy"],
        "forced_target_selections": metrics.get("decoding", {}).get("forced_target_selections", 0),
        "core_target_f1": {
            target: metrics["targets"]["per_class"][target]["f1"] for target in CORE_TARGETS
        },
        "core_target_support": {
            target: metrics["targets"]["per_class"][target]["support"] for target in CORE_TARGETS
        },
    }


def build_locked_trial_receipt(
    manifest: Mapping[str, Any],
    *,
    trial: Mapping[str, Any],
    prediction_descriptor: Mapping[str, Any],
    metrics: Mapping[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    clean_manifest = validate_locked_test_manifest(manifest)
    matching = [
        row
        for row in clean_manifest["trials"]
        if row["source_trial_id"] == trial.get("source_trial_id")
    ]
    if len(matching) != 1 or dict(trial) != matching[0]:
        raise ValueError("locked-test trial is not in the frozen manifest")
    if not isinstance(wall_seconds, (int, float)) or not math.isfinite(wall_seconds):
        raise ValueError("locked-test runtime must be finite")
    if wall_seconds < 0:
        raise ValueError("locked-test runtime must be non-negative")
    cost = Decimal(str(wall_seconds)) * GPU_RATE_USD_PER_SECOND["L4"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_RECEIPT_KIND,
        "status": "complete",
        "locked_test_run_id": clean_manifest["locked_test_run_id"],
        "authorisation_id": clean_manifest["authorisation"]["authorisation_id"],
        "source_trial_id": trial["source_trial_id"],
        "source_receipt_id": trial["source_receipt_id"],
        "label_budget": trial["label_budget"],
        "training_row_count": trial["training_row_count"],
        "ladder_seed": trial["ladder_seed"],
        "optimiser_seed": trial["optimiser_seed"],
        "prediction_artifact": _descriptor(prediction_descriptor, where="locked-test prediction"),
        "aggregate_metrics": _metric_summary(metrics),
        "compute": {
            "gpu_type": "L4",
            "wall_seconds": wall_seconds,
            "gpu_seconds": wall_seconds,
            "estimated_cost_usd": f"{cost.quantize(Decimal('0.000001')):.6f}",
        },
    }
    assert_metadata_only(body, where="locked-test trial receipt")
    return {**body, "receipt_id": canonical_sha256(body)}


def validate_locked_trial_receipt(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    clean_manifest = validate_locked_test_manifest(manifest)
    trial = next(
        (
            row
            for row in clean_manifest["trials"]
            if row["source_trial_id"] == receipt.get("source_trial_id")
        ),
        None,
    )
    if trial is None:
        raise ValueError("locked-test receipt belongs to an unknown trial")
    rebuilt = build_locked_trial_receipt(
        clean_manifest,
        trial=trial,
        prediction_descriptor=receipt.get("prediction_artifact", {}),
        metrics=receipt.get("aggregate_metrics", {}),
        wall_seconds=receipt.get("compute", {}).get("wall_seconds"),
    )
    if dict(receipt) != rebuilt:
        raise ValueError("locked-test trial receipt content address drifted")
    return rebuilt


def _decoded_predictions(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["source_sample_id"]): dict(row["decoded_label"]) for row in payload["rows"]}


def _core_tuple_f1(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    sampled_ids: Sequence[str],
) -> float:
    true_positive = false_positive = false_negative = 0
    core = set(CORE_TARGETS)
    for item_id in sampled_ids:
        expected = {
            (row["target"], row["stance"])
            for row in reference[item_id]["target_stances"]
            if row["target"] in core
        }
        predicted = {
            (row["target"], row["stance"])
            for row in predictions[item_id]["target_stances"]
            if row["target"] in core
        }
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_locked_test_aggregate(
    manifest: Mapping[str, Any],
    *,
    reference: Mapping[str, Mapping[str, Any]],
    prediction_payloads: Mapping[str, Mapping[str, Any]],
    trial_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score nine locked predictions and apply the frozen scale gates."""

    clean_manifest = validate_locked_test_manifest(manifest)
    if len(reference) != LOCKED_TEST_ROWS:
        raise ValueError("locked-test aggregate requires exactly 230 reference rows")
    expected_ids = {row["source_trial_id"] for row in clean_manifest["trials"]}
    if set(prediction_payloads) != expected_ids:
        raise ValueError("locked-test aggregate prediction set is incomplete")
    receipts = {
        str(receipt.get("source_trial_id")): validate_locked_trial_receipt(clean_manifest, receipt)
        for receipt in trial_receipts
    }
    if len(receipts) != 9 or set(receipts) != expected_ids:
        raise ValueError("locked-test aggregate receipt set is incomplete")

    results = []
    predictions_by_condition: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for trial in clean_manifest["trials"]:
        trial_id = trial["source_trial_id"]
        payload = validate_private_locked_predictions(
            prediction_payloads[trial_id],
            expected_trial=trial,
            run_id=clean_manifest["locked_test_run_id"],
        )
        if set(row["source_sample_id"] for row in payload["rows"]) != set(reference):
            raise ValueError("locked-test prediction and reference IDs differ")
        predictions = _decoded_predictions(payload)
        metrics = score_semantic_labels(reference, predictions)
        metrics["decoding"] = {
            "target_threshold": float(TARGET_THRESHOLD),
            "forced_target_selections": receipts[trial_id]["aggregate_metrics"][
                "forced_target_selections"
            ],
        }
        summary = _metric_summary(metrics)
        if summary != receipts[trial_id]["aggregate_metrics"]:
            raise ValueError("locked-test stored metrics do not recompute exactly")
        condition = (trial["ladder_seed"], trial["label_budget"])
        predictions_by_condition[condition] = predictions
        results.append({"trial": trial, "metrics": summary})

    metric_names = (
        "relevance_macro_f1",
        "material_recall",
        "core_target_micro_f1",
        "core_target_stance_tuple_micro_f1",
        "fixed_reference_core_target_stance_accuracy",
    )
    budget_summary = {}
    for budget in LABEL_BUDGETS:
        group = [row["metrics"] for row in results if row["trial"]["label_budget"] == budget]
        budget_summary[str(budget)] = {
            "runs": len(group),
            "mean": {
                metric: round(statistics.mean(float(row[metric]) for row in group), 6)
                for metric in metric_names
            },
            "sample_sd": {
                metric: round(statistics.stdev(float(row[metric]) for row in group), 6)
                for metric in metric_names
            },
        }

    paired_differences = {}
    for low, high in ((2_000, 5_000), (5_000, 10_000), (2_000, 10_000)):
        values = []
        for ladder in REGISTERED_LADDERS:
            ladder_seed = int(ladder["ladder_seed"])
            by_budget = {
                row["trial"]["label_budget"]: row["metrics"]
                for row in results
                if row["trial"]["ladder_seed"] == ladder_seed
            }
            values.append(
                round(
                    float(by_budget[high]["core_target_stance_tuple_micro_f1"])
                    - float(by_budget[low]["core_target_stance_tuple_micro_f1"]),
                    6,
                )
            )
        paired_differences[f"{high}_minus_{low}"] = {
            "core_target_stance_tuple_micro_f1": values,
            "mean": round(statistics.mean(values), 6),
            "positive_pairs": sum(value > 0 for value in values),
            "negative_pairs": sum(value < 0 for value in values),
        }

    row_ids = sorted(reference)
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap_values = []
    pair_seeds = [int(row["ladder_seed"]) for row in REGISTERED_LADDERS]
    for _ in range(BOOTSTRAP_REPLICATES):
        differences = []
        for ladder_seed in rng.choices(pair_seeds, k=len(pair_seeds)):
            sampled_ids = rng.choices(row_ids, k=len(row_ids))
            score_5k = _core_tuple_f1(
                reference, predictions_by_condition[(ladder_seed, 5_000)], sampled_ids
            )
            score_10k = _core_tuple_f1(
                reference, predictions_by_condition[(ladder_seed, 10_000)], sampled_ids
            )
            differences.append(score_10k - score_5k)
        bootstrap_values.append(statistics.mean(differences))
    interval = {
        "lower": round(_percentile(bootstrap_values, 0.025), 6),
        "upper": round(_percentile(bootstrap_values, 0.975), 6),
    }

    per_target = {}
    regression_breaches = []
    for target in CORE_TARGETS:
        support_values = {row["metrics"]["core_target_support"][target] for row in results}
        if len(support_values) != 1:
            raise ValueError("locked-test target support differs across trials")
        support = support_values.pop()
        means = {}
        for budget in (5_000, 10_000):
            values = [
                row["metrics"]["core_target_f1"][target]
                for row in results
                if row["trial"]["label_budget"] == budget
            ]
            defined = [float(value) for value in values if value is not None]
            means[str(budget)] = None if not defined else round(statistics.mean(defined), 6)
        delta = (
            None
            if means["10000"] is None or means["5000"] is None
            else round(means["10000"] - means["5000"], 6)
        )
        gated = support >= int(SCALE_GATES["core_target_support_min"])
        if gated and delta is None:
            raise ValueError("supported locked-test core target lacks a defined F1")
        breached = (
            gated
            and delta is not None
            and delta < -float(SCALE_GATES["per_core_target_f1_regression_max"])
        )
        if breached:
            regression_breaches.append(target)
        per_target[target] = {
            "support": support,
            "mean_f1_5k": means["5000"],
            "mean_f1_10k": means["10000"],
            "delta": delta,
            "gated": gated,
            "regression_breach": breached,
        }

    mean_gain = paired_differences["10000_minus_5000"]["mean"]
    criteria = {
        "mean_gain_at_least_0_02": mean_gain >= float(SCALE_GATES["mean_core_tuple_f1_gain_min"]),
        "bootstrap_interval_excludes_zero": interval["lower"] > 0 or interval["upper"] < 0,
        "no_supported_core_target_regression_over_0_05": not regression_breaches,
    }
    supported = all(criteria.values())
    if supported:
        verdict = "supported"
    elif mean_gain < float(SCALE_GATES["plateau_below"]):
        verdict = "plateau_stop"
    elif mean_gain < float(SCALE_GATES["inconclusive_below"]):
        verdict = "inconclusive"
    else:
        verdict = "rejected_by_registered_gate"

    total_cost = sum(
        Decimal(receipt["compute"]["estimated_cost_usd"]) for receipt in receipts.values()
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": AGGREGATE_KIND,
        "locked_test_run_id": clean_manifest["locked_test_run_id"],
        "authorisation_id": clean_manifest["authorisation"]["authorisation_id"],
        "row_count": LOCKED_TEST_ROWS,
        "runs": 9,
        "target_threshold": TARGET_THRESHOLD,
        "budget_summary": budget_summary,
        "paired_differences": paired_differences,
        "paired_hierarchical_bootstrap": {
            **clean_manifest["bootstrap"],
            "interval": interval,
        },
        "core_target_regressions": per_target,
        "scale_gate": {
            "verdict": verdict,
            "passed": supported,
            "criteria": criteria,
            "regression_breaches": regression_breaches,
        },
        "invalid_outputs_total": sum(row["metrics"]["invalid_outputs"] for row in results),
        "trial_receipt_ids_sha256": canonical_sha256(
            sorted(receipt["receipt_id"] for receipt in receipts.values())
        ),
        "prediction_artifacts_sha256": canonical_sha256(
            sorted(receipt["prediction_artifact"]["sha256"] for receipt in receipts.values())
        ),
        "estimated_evaluation_cost_usd": f"{total_cost.quantize(Decimal('0.000001')):.6f}",
    }
    assert_metadata_only(body, where="locked-test aggregate")
    return {**body, "aggregate_id": canonical_sha256(body)}


def build_final_receipt(
    manifest: Mapping[str, Any],
    *,
    aggregate: Mapping[str, Any],
    aggregate_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    clean_manifest = validate_locked_test_manifest(manifest)
    aggregate_body = {key: value for key, value in aggregate.items() if key != "aggregate_id"}
    if (
        aggregate.get("kind") != AGGREGATE_KIND
        or aggregate.get("locked_test_run_id") != clean_manifest["locked_test_run_id"]
        or aggregate.get("aggregate_id") != canonical_sha256(aggregate_body)
    ):
        raise ValueError("locked-test aggregate content address drifted")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": FINAL_RECEIPT_KIND,
        "status": "complete",
        "locked_test_run_id": clean_manifest["locked_test_run_id"],
        "authorisation_id": clean_manifest["authorisation"]["authorisation_id"],
        "aggregate_id": aggregate["aggregate_id"],
        "aggregate_artifact": _descriptor(aggregate_descriptor, where="locked-test aggregate"),
        "scale_verdict": aggregate["scale_gate"]["verdict"],
        "scale_gate_passed": aggregate["scale_gate"]["passed"],
        "rows_accessed": LOCKED_TEST_ROWS,
        "single_access_complete": True,
        "estimated_evaluation_cost_usd": aggregate["estimated_evaluation_cost_usd"],
    }
    assert_metadata_only(body, where="locked-test final receipt")
    return {**body, "receipt_id": canonical_sha256(body)}


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "LOCKED_TEST_ROWS",
    "PREDICTION_KIND",
    "TARGET_THRESHOLD",
    "build_final_receipt",
    "build_locked_test_aggregate",
    "build_locked_test_authorisation",
    "build_locked_test_manifest",
    "build_locked_trial_receipt",
    "validate_locked_test_manifest",
    "validate_locked_trial_receipt",
    "validate_private_locked_predictions",
]
