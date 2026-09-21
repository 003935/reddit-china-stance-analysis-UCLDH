from __future__ import annotations

import json

import pytest

from reddit_china_stance.semantic_evaluation import (
    MISSING_TARGET,
    evaluate_capability_gate,
    score_semantic_labels,
)


def _material(target: str, stance: str) -> dict[str, object]:
    return {
        "relevance": "material",
        "target_stances": [{"target": target, "stance": stance}],
    }


def _reference() -> dict[str, dict[str, object]]:
    return {
        "one": _material("china_general", "positive"),
        "two": _material("government_ccp", "negative"),
        "three": _material("people_culture", "no_directed_stance"),
        "four": _material("other", "mixed"),
        "five": {"relevance": "not_material", "target_stances": []},
        "six": {"relevance": "unclear", "target_stances": []},
    }


def test_perfect_predictions_report_fieldwise_metrics_and_pass_gate() -> None:
    reference = _reference()
    metrics = score_semantic_labels(reference, reference)

    assert metrics["items"] == 6
    assert metrics["invalid_outputs"] == 0
    assert metrics["relevance"]["macro_f1"] == 1.0
    assert metrics["relevance"]["material_recall"] == 1.0
    assert metrics["targets"]["core"]["micro"]["f1"] == 1.0
    assert metrics["targets"]["core"]["macro"]["f1"] == 1.0
    assert metrics["stance"]["fixed_reference_target"]["accuracy"] == 1.0
    assert metrics["stance"]["fixed_reference_target"]["cohen_kappa"] == 1.0
    assert metrics["end_to_end_core_target_stance"]["micro"]["f1"] == 1.0
    assert metrics["initial_label_projection"] == {
        "relevance": {"accepted": 5, "correct": 5, "precision": 1.0, "row_coverage": 0.833333},
        "core_targets": {
            "accepted_instances": 3,
            "correct_instances": 3,
            "precision": 1.0,
            "row_coverage": 0.5,
        },
        "core_stance_tuples": {
            "accepted_instances": 3,
            "correct_instances": 3,
            "precision": 1.0,
            "row_coverage": 0.5,
        },
    }
    assert metrics["diagnostics"]["exact_target_set"]["accuracy"] == 1.0
    assert metrics["diagnostics"]["exact_whole_row"]["accuracy"] == 1.0
    assert evaluate_capability_gate(metrics)["passed"] is True
    json.dumps(metrics)


def test_missed_target_is_a_stance_failure_instead_of_being_dropped() -> None:
    reference = _reference()
    predictions = dict(reference)
    predictions["two"] = _material("china_general", "negative")

    metrics = score_semantic_labels(reference, predictions)
    fixed = metrics["stance"]["fixed_reference_target"]
    government = metrics["targets"]["per_class"]["government_ccp"]

    assert fixed["support"] == 4
    assert fixed["missing_predicted_targets"] == 1
    assert fixed["accuracy"] == 0.75
    assert fixed["confusion_matrix"]["negative"][MISSING_TARGET] == 1
    assert government["false_negative"] == 1
    assert metrics["end_to_end_core_target_stance"]["micro"] == {
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 1,
        "support": 3,
        "predicted": 3,
        "precision": 0.666667,
        "recall": 0.666667,
        "f1": 0.666667,
    }
    gate = evaluate_capability_gate(metrics)
    assert gate["passed"] is False
    assert "core_target_micro_f1" in gate["failed_criteria"]


def test_missing_and_malformed_outputs_remain_in_all_relevant_denominators() -> None:
    reference = _reference()
    predictions = dict(reference)
    predictions.pop("one")
    predictions["two"] = {"relevance": "material", "target_stances": []}

    metrics = score_semantic_labels(reference, predictions)

    assert metrics["missing_outputs"] == 1
    assert metrics["malformed_outputs"] == 1
    assert metrics["invalid_outputs"] == 2
    assert metrics["valid_outputs"] == 4
    assert metrics["relevance"]["confusion_matrix"]["material"]["__invalid__"] == 2
    assert metrics["relevance"]["material_recall"] == 0.5
    assert metrics["stance"]["fixed_reference_target"]["missing_predicted_targets"] == 2
    assert metrics["diagnostics"]["exact_target_set"]["accuracy"] == 0.666667
    assert evaluate_capability_gate(metrics)["criteria"]["invalid_outputs"]["passed"] is False


def test_multi_target_scoring_is_set_based_and_whole_row_is_stance_sensitive() -> None:
    reference = {
        "one": {
            "relevance": "material",
            "target_stances": [
                {"target": "government_ccp", "stance": "negative"},
                {"target": "people_culture", "stance": "positive"},
            ],
        }
    }
    predictions = {
        "one": {
            "relevance": "material",
            "target_stances": [
                {"target": "people_culture", "stance": "negative"},
                {"target": "government_ccp", "stance": "negative"},
            ],
        }
    }

    metrics = score_semantic_labels(reference, predictions)

    assert metrics["diagnostics"]["exact_target_set"]["accuracy"] == 1.0
    assert metrics["diagnostics"]["exact_whole_row"]["accuracy"] == 0.0
    assert metrics["stance"]["fixed_reference_target"]["accuracy"] == 0.5
    assert metrics["end_to_end_core_target_stance"]["micro"]["f1"] == 0.5
    assert metrics["support"]["reference_multi_target_rows"] == 1


def test_rejects_packet_binding_errors_and_invalid_reference() -> None:
    reference = _reference()
    with pytest.raises(ValueError, match="unexpected item IDs"):
        score_semantic_labels(reference, {**reference, "extra": reference["one"]})
    bad_reference = dict(reference)
    bad_reference["one"] = {
        "relevance": "not_material",
        "target_stances": [{"target": "china_general", "stance": "positive"}],
    }
    with pytest.raises(ValueError, match="reference contains an invalid label"):
        score_semantic_labels(bad_reference, reference)


def test_gate_fails_closed_when_required_metric_is_undefined() -> None:
    reference = {
        "one": {"relevance": "not_material", "target_stances": []},
        "two": {"relevance": "unclear", "target_stances": []},
    }
    metrics = score_semantic_labels(reference, reference)
    gate = evaluate_capability_gate(metrics)

    assert gate["passed"] is False
    assert gate["criteria"]["material_recall"]["value"] is None
    assert gate["criteria"]["material_recall"]["passed"] is False
