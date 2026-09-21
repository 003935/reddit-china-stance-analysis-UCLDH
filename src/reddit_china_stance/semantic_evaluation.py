"""Fieldwise evaluation for a frozen development-proxy semantic-label contract.

The scorer deliberately keeps missing and invalid predictions in the denominators.
In particular, stance is evaluated for every reference target instance; a
missing predicted target is represented by a sentinel rather than silently
dropping the instance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

RELEVANCE_LABELS = ("material", "not_material", "unclear")
TARGET_LABELS = ("china_general", "government_ccp", "people_culture", "other")
CORE_TARGETS = ("china_general", "government_ccp", "people_culture")
STANCE_LABELS = ("negative", "positive", "mixed", "no_directed_stance", "unclear")

INVALID_PREDICTION = "__invalid__"
MISSING_TARGET = "__missing_target__"

CAPABILITY_GATE_THRESHOLDS = {
    "invalid_outputs_max": 0,
    "relevance_macro_f1_min": 0.85,
    "material_recall_min": 0.90,
    "core_target_micro_f1_min": 0.80,
    "fixed_reference_core_target_stance_accuracy_min": 0.75,
    "core_target_stance_tuple_micro_f1_min": 0.75,
    "projected_relevance_precision_min": 0.90,
    "projected_core_target_precision_min": 0.85,
    "projected_core_stance_tuple_precision_min": 0.80,
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _mean(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return None if not defined else round(sum(defined) / len(defined), 6)


def _binary_metrics(
    *, true_positive: int, false_positive: int, false_negative: int
) -> dict[str, Any]:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "support": true_positive + false_negative,
        "predicted": true_positive + false_positive,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * true_positive, f1_denominator),
    }


def _validate_label(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise the frozen two-field semantic label."""

    if set(value) != {"relevance", "target_stances"}:
        raise ValueError("semantic label must contain exactly relevance and target_stances")
    relevance = value.get("relevance")
    if relevance not in RELEVANCE_LABELS:
        raise ValueError("semantic label has an unsupported relevance")
    raw_target_stances = value.get("target_stances")
    if not isinstance(raw_target_stances, list) or len(raw_target_stances) > len(TARGET_LABELS):
        raise ValueError("semantic label target_stances must be a bounded list")

    target_stances: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for raw in raw_target_stances:
        if not isinstance(raw, Mapping) or set(raw) != {"target", "stance"}:
            raise ValueError("each target stance must contain exactly target and stance")
        target = raw.get("target")
        stance = raw.get("stance")
        if target not in TARGET_LABELS or stance not in STANCE_LABELS:
            raise ValueError("semantic label has an unsupported target or stance")
        if target in seen_targets:
            raise ValueError("semantic label has duplicate target entries")
        seen_targets.add(target)
        target_stances.append({"target": str(target), "stance": str(stance)})

    if relevance == "material" and not target_stances:
        raise ValueError("material semantic label must contain at least one target")
    if relevance != "material" and target_stances:
        raise ValueError("non-material or unclear semantic label cannot contain targets")

    order = {target: index for index, target in enumerate(TARGET_LABELS)}
    target_stances.sort(key=lambda item: order[item["target"]])
    return {"relevance": str(relevance), "target_stances": target_stances}


def _target_stance_map(label: Mapping[str, Any]) -> dict[str, str]:
    return {str(item["target"]): str(item["stance"]) for item in label["target_stances"]}


def _multiclass_metrics(
    pairs: Sequence[tuple[str, str]],
    *,
    reference_labels: Sequence[str],
    prediction_labels: Sequence[str],
) -> dict[str, Any]:
    matrix = {
        expected: {predicted: 0 for predicted in prediction_labels} for expected in reference_labels
    }
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for label in reference_labels:
        true_positive = matrix[label].get(label, 0)
        false_negative = sum(matrix[label].values()) - true_positive
        false_positive = sum(
            matrix[other].get(label, 0) for other in reference_labels if other != label
        )
        per_class[label] = _binary_metrics(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

    correct = sum(expected == predicted for expected, predicted in pairs)
    return {
        "labels": list(reference_labels),
        "prediction_labels": list(prediction_labels),
        "support": len(pairs),
        "correct": correct,
        "accuracy": _ratio(correct, len(pairs)),
        "macro_f1": _mean([per_class[label]["f1"] for label in reference_labels]),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _target_metrics(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    counts = {
        target: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for target in TARGET_LABELS
    }
    for item_id, expected in reference.items():
        expected_targets = set(_target_stance_map(expected))
        predicted = predictions[item_id]
        predicted_targets = set(_target_stance_map(predicted)) if predicted is not None else set()
        for target in TARGET_LABELS:
            if target in expected_targets and target in predicted_targets:
                counts[target]["true_positive"] += 1
            elif target in predicted_targets:
                counts[target]["false_positive"] += 1
            elif target in expected_targets:
                counts[target]["false_negative"] += 1

    per_class = {target: _binary_metrics(**counts[target]) for target in TARGET_LABELS}
    core_counts = {
        key: sum(counts[target][key] for target in CORE_TARGETS)
        for key in ("true_positive", "false_positive", "false_negative")
    }
    all_counts = {
        key: sum(counts[target][key] for target in TARGET_LABELS)
        for key in ("true_positive", "false_positive", "false_negative")
    }
    return {
        "per_class": per_class,
        "all": {
            "classes": list(TARGET_LABELS),
            "micro": _binary_metrics(**all_counts),
            "macro": {
                "precision": _mean([per_class[target]["precision"] for target in TARGET_LABELS]),
                "recall": _mean([per_class[target]["recall"] for target in TARGET_LABELS]),
                "f1": _mean([per_class[target]["f1"] for target in TARGET_LABELS]),
            },
        },
        "core": {
            "classes": list(CORE_TARGETS),
            "micro": _binary_metrics(**core_counts),
            "macro": {
                "precision": _mean([per_class[target]["precision"] for target in CORE_TARGETS]),
                "recall": _mean([per_class[target]["recall"] for target in CORE_TARGETS]),
                "f1": _mean([per_class[target]["f1"] for target in CORE_TARGETS]),
            },
        },
    }


def _cohen_kappa(pairs: Sequence[tuple[str, str]], *, labels: Sequence[str]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(expected == predicted for expected, predicted in pairs) / total
    expected_marginal = {label: 0 for label in labels}
    predicted_marginal = {label: 0 for label in labels}
    for expected, predicted in pairs:
        expected_marginal[expected] += 1
        predicted_marginal[predicted] += 1
    chance = sum(
        (expected_marginal[label] / total) * (predicted_marginal[label] / total) for label in labels
    )
    if chance == 1.0:
        return None
    return round((observed - chance) / (1.0 - chance), 6)


def _fixed_reference_target_stance_metrics(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    targets: Sequence[str],
) -> dict[str, Any]:
    included_targets = set(targets)
    pairs: list[tuple[str, str]] = []
    missing_targets = 0
    for item_id, expected in reference.items():
        expected_stances = _target_stance_map(expected)
        predicted = predictions[item_id]
        predicted_stances = _target_stance_map(predicted) if predicted is not None else {}
        for target, expected_stance in expected_stances.items():
            if target not in included_targets:
                continue
            predicted_stance = predicted_stances.get(target, MISSING_TARGET)
            missing_targets += predicted_stance == MISSING_TARGET
            pairs.append((expected_stance, predicted_stance))

    prediction_labels = (*STANCE_LABELS, MISSING_TARGET)
    metrics = _multiclass_metrics(
        pairs,
        reference_labels=STANCE_LABELS,
        prediction_labels=prediction_labels,
    )
    metrics["targets"] = list(targets)
    metrics["missing_predicted_targets"] = missing_targets
    metrics["cohen_kappa"] = _cohen_kappa(pairs, labels=prediction_labels)
    return metrics


def _tuple_metrics(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    targets: Sequence[str],
) -> dict[str, Any]:
    included_targets = set(targets)
    expected_tuples: set[tuple[str, str, str]] = set()
    predicted_tuples: set[tuple[str, str, str]] = set()
    for item_id, expected in reference.items():
        for target, stance in _target_stance_map(expected).items():
            if target in included_targets:
                expected_tuples.add((item_id, target, stance))
        predicted = predictions[item_id]
        if predicted is not None:
            for target, stance in _target_stance_map(predicted).items():
                if target in included_targets:
                    predicted_tuples.add((item_id, target, stance))
    true_positive = len(expected_tuples & predicted_tuples)
    return _binary_metrics(
        true_positive=true_positive,
        false_positive=len(predicted_tuples - expected_tuples),
        false_negative=len(expected_tuples - predicted_tuples),
    )


def _projection_metrics(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    accepted_relevance = correct_relevance = 0
    accepted_target_instances = correct_target_instances = 0
    accepted_stance_tuples = correct_stance_tuples = 0
    accepted_target_rows: set[str] = set()
    accepted_stance_rows: set[str] = set()
    accepted_stances = {"negative", "positive", "no_directed_stance"}

    for item_id, expected in reference.items():
        predicted = predictions[item_id]
        if predicted is None:
            continue
        if predicted["relevance"] != "unclear":
            accepted_relevance += 1
            correct_relevance += predicted["relevance"] == expected["relevance"]

        expected_map = _target_stance_map(expected)
        predicted_map = _target_stance_map(predicted)
        for target, stance in predicted_map.items():
            if target not in CORE_TARGETS:
                continue
            accepted_target_rows.add(item_id)
            accepted_target_instances += 1
            correct_target_instances += target in expected_map
            if stance in accepted_stances:
                accepted_stance_rows.add(item_id)
                accepted_stance_tuples += 1
                correct_stance_tuples += expected_map.get(target) == stance

    items = len(reference)
    return {
        "relevance": {
            "accepted": accepted_relevance,
            "correct": correct_relevance,
            "precision": _ratio(correct_relevance, accepted_relevance),
            "row_coverage": _ratio(accepted_relevance, items),
        },
        "core_targets": {
            "accepted_instances": accepted_target_instances,
            "correct_instances": correct_target_instances,
            "precision": _ratio(correct_target_instances, accepted_target_instances),
            "row_coverage": _ratio(len(accepted_target_rows), items),
        },
        "core_stance_tuples": {
            "accepted_instances": accepted_stance_tuples,
            "correct_instances": correct_stance_tuples,
            "precision": _ratio(correct_stance_tuples, accepted_stance_tuples),
            "row_coverage": _ratio(len(accepted_stance_rows), items),
        },
    }


def score_semantic_labels(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Return metadata-only aggregates for semantic labels keyed by item ID.

    Missing, ``None``, and structurally invalid predictions are counted as invalid
    outputs. Unexpected prediction IDs are rejected because they signal a packet
    binding error. Reference labels must all be valid.
    """

    if not reference:
        raise ValueError("semantic evaluation reference is empty")
    unexpected = set(predictions) - set(reference)
    if unexpected:
        raise ValueError("semantic predictions contain unexpected item IDs")

    clean_reference: dict[str, dict[str, Any]] = {}
    for item_id, label in reference.items():
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("semantic reference item IDs must be non-empty strings")
        try:
            clean_reference[item_id] = _validate_label(label)
        except (TypeError, ValueError) as error:
            raise ValueError("semantic reference contains an invalid label") from error

    clean_predictions: dict[str, dict[str, Any] | None] = {}
    missing_outputs = malformed_outputs = 0
    for item_id in clean_reference:
        if item_id not in predictions or predictions[item_id] is None:
            clean_predictions[item_id] = None
            missing_outputs += 1
            continue
        raw_prediction = predictions[item_id]
        try:
            clean_predictions[item_id] = _validate_label(raw_prediction)
        except (TypeError, ValueError, AttributeError):
            clean_predictions[item_id] = None
            malformed_outputs += 1

    relevance_pairs = [
        (
            expected["relevance"],
            clean_predictions[item_id]["relevance"]
            if clean_predictions[item_id] is not None
            else INVALID_PREDICTION,
        )
        for item_id, expected in clean_reference.items()
    ]
    relevance = _multiclass_metrics(
        relevance_pairs,
        reference_labels=RELEVANCE_LABELS,
        prediction_labels=(*RELEVANCE_LABELS, INVALID_PREDICTION),
    )
    relevance["material_precision"] = relevance["per_class"]["material"]["precision"]
    relevance["material_recall"] = relevance["per_class"]["material"]["recall"]
    relevance["predicted_unclear_rate"] = _ratio(
        sum(
            prediction is not None and prediction["relevance"] == "unclear"
            for prediction in clean_predictions.values()
        ),
        len(clean_reference),
    )

    exact_target_sets = exact_whole_rows = exact_target_sets_material = 0
    reference_material_rows = 0
    for item_id, expected in clean_reference.items():
        predicted = clean_predictions[item_id]
        expected_targets = set(_target_stance_map(expected))
        predicted_targets = set(_target_stance_map(predicted)) if predicted is not None else set()
        target_set_exact = predicted is not None and expected_targets == predicted_targets
        exact_target_sets += target_set_exact
        if expected["relevance"] == "material":
            reference_material_rows += 1
            exact_target_sets_material += target_set_exact
        exact_whole_rows += predicted is not None and predicted == expected

    reference_target_support = {target: 0 for target in TARGET_LABELS}
    reference_stance_support = {stance: 0 for stance in STANCE_LABELS}
    for expected in clean_reference.values():
        for target, stance in _target_stance_map(expected).items():
            reference_target_support[target] += 1
            reference_stance_support[stance] += 1

    invalid_outputs = missing_outputs + malformed_outputs
    result = {
        "items": len(clean_reference),
        "valid_outputs": len(clean_reference) - invalid_outputs,
        "invalid_outputs": invalid_outputs,
        "invalid_output_rate": _ratio(invalid_outputs, len(clean_reference)),
        "missing_outputs": missing_outputs,
        "malformed_outputs": malformed_outputs,
        "support": {
            "reference_relevance": {
                label: sum(expected["relevance"] == label for expected in clean_reference.values())
                for label in RELEVANCE_LABELS
            },
            "reference_targets": reference_target_support,
            "reference_stances": reference_stance_support,
            "reference_material_rows": reference_material_rows,
            "reference_multi_target_rows": sum(
                len(expected["target_stances"]) > 1 for expected in clean_reference.values()
            ),
        },
        "relevance": relevance,
        "targets": _target_metrics(clean_reference, clean_predictions),
        "stance": {
            "fixed_reference_target": _fixed_reference_target_stance_metrics(
                clean_reference,
                clean_predictions,
                targets=TARGET_LABELS,
            ),
            "fixed_reference_target_core": _fixed_reference_target_stance_metrics(
                clean_reference,
                clean_predictions,
                targets=CORE_TARGETS,
            ),
        },
        "end_to_end_core_target_stance": {
            "micro": _tuple_metrics(
                clean_reference,
                clean_predictions,
                targets=CORE_TARGETS,
            )
        },
        "end_to_end_target_stance": {
            "micro": _tuple_metrics(
                clean_reference,
                clean_predictions,
                targets=TARGET_LABELS,
            )
        },
        "initial_label_projection": _projection_metrics(clean_reference, clean_predictions),
        "diagnostics": {
            "exact_target_set": {
                "correct": exact_target_sets,
                "accuracy": _ratio(exact_target_sets, len(clean_reference)),
            },
            "exact_target_set_reference_material": {
                "correct": exact_target_sets_material,
                "support": reference_material_rows,
                "accuracy": _ratio(exact_target_sets_material, reference_material_rows),
            },
            "exact_whole_row": {
                "correct": exact_whole_rows,
                "accuracy": _ratio(exact_whole_rows, len(clean_reference)),
            },
        },
    }
    return result


def evaluate_capability_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the conjunctive locked-test gate frozen in the experiment brief."""

    def criterion(
        value: int | float | None, operator: str, threshold: int | float
    ) -> dict[str, Any]:
        passed = value is not None and (
            value <= threshold if operator == "<=" else value >= threshold
        )
        return {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }

    relevance = metrics["relevance"]
    targets = metrics["targets"]
    stance = metrics["stance"]["fixed_reference_target_core"]
    tuple_micro = metrics["end_to_end_core_target_stance"]["micro"]
    projection = metrics["initial_label_projection"]
    criteria = {
        "invalid_outputs": criterion(
            metrics["invalid_outputs"], "<=", CAPABILITY_GATE_THRESHOLDS["invalid_outputs_max"]
        ),
        "relevance_macro_f1": criterion(
            relevance["macro_f1"], ">=", CAPABILITY_GATE_THRESHOLDS["relevance_macro_f1_min"]
        ),
        "material_recall": criterion(
            relevance["material_recall"], ">=", CAPABILITY_GATE_THRESHOLDS["material_recall_min"]
        ),
        "core_target_micro_f1": criterion(
            targets["core"]["micro"]["f1"],
            ">=",
            CAPABILITY_GATE_THRESHOLDS["core_target_micro_f1_min"],
        ),
        "fixed_reference_core_target_stance_accuracy": criterion(
            stance["accuracy"],
            ">=",
            CAPABILITY_GATE_THRESHOLDS["fixed_reference_core_target_stance_accuracy_min"],
        ),
        "core_target_stance_tuple_micro_f1": criterion(
            tuple_micro["f1"],
            ">=",
            CAPABILITY_GATE_THRESHOLDS["core_target_stance_tuple_micro_f1_min"],
        ),
        "projected_relevance_precision": criterion(
            projection["relevance"]["precision"],
            ">=",
            CAPABILITY_GATE_THRESHOLDS["projected_relevance_precision_min"],
        ),
        "projected_core_target_precision": criterion(
            projection["core_targets"]["precision"],
            ">=",
            CAPABILITY_GATE_THRESHOLDS["projected_core_target_precision_min"],
        ),
        "projected_core_stance_tuple_precision": criterion(
            projection["core_stance_tuples"]["precision"],
            ">=",
            CAPABILITY_GATE_THRESHOLDS["projected_core_stance_tuple_precision_min"],
        ),
    }
    failed = [name for name, result in criteria.items() if not result["passed"]]
    return {
        "gate": "development-proxy-capability-v1",
        "conjunctive": True,
        "passed": not failed,
        "criteria": criteria,
        "failed_criteria": failed,
    }
