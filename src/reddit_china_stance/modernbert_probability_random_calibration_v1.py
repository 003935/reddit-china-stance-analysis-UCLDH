"""Weighted calibration and aggregate evaluation for the retained random ensemble.

The module is storage agnostic.  It consumes private, row-aligned mean logits and
teacher labels, fits only the frozen grids, and returns metadata-only calibration
and evaluation artefacts.  Mixed stance labels remain valid target-presence evidence
but are excluded from the registered three-class stance estimand.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    STANCE_CLASSES_B4,
    TARGET_CLASSES,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation_v2 import (
    PROBABILITY_SCOPE,
    DecodeThresholds,
    NaturalArmWeightingConfig,
    decode_factorised_probabilities,
    natural_arm_design_digest,
    score_natural_probability_arm,
)
from reddit_china_stance.semantic_ontology_v2 import validate_v2_label

CALIBRATION_KIND = "modernbert-probability-random-weighted-calibration-v1"
VALIDATION_KIND = "modernbert-probability-random-aggregate-validation-v1"
THRESHOLD_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
ECE_BINS = 10
STANCE_ESTIMAND_CLASSES = ("negative", "no_directed_stance", "positive")
STANCE_ESTIMAND_INDICES = tuple(STANCE_CLASSES_B4.index(state) for state in STANCE_ESTIMAND_CLASSES)
MIXED_STATE = "mixed"


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be finite numeric")
    clean = float(value)
    if not math.isfinite(clean):
        raise ValueError(f"{where} must be finite numeric")
    return clean


def _sigmoid(logit: float, temperature: float) -> float:
    scaled = logit / temperature
    if scaled >= 0:
        factor = math.exp(-scaled)
        return 1.0 / (1.0 + factor)
    factor = math.exp(scaled)
    return factor / (1.0 + factor)


def _softmax(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def mean_aligned_logits(
    per_checkpoint: Sequence[Sequence[Mapping[str, Any]]], *, component: str
) -> list[dict[str, Any]]:
    """Take the unweighted mean of all three aligned seed logits."""

    if component not in candidate.COMPONENTS:
        raise ValueError("component is outside the retained cohort")
    if isinstance(per_checkpoint, (str, bytes)) or len(per_checkpoint) != len(candidate.SEEDS):
        raise ValueError("mean-logit aggregation requires exactly three checkpoints")
    if any(len(rows) == 0 for rows in per_checkpoint):
        raise ValueError("checkpoint logits cannot be empty")
    ids = [[row.get("item_id") for row in rows] for rows in per_checkpoint]
    if any(values != ids[0] for values in ids[1:]) or len(ids[0]) != len(set(ids[0])):
        raise ValueError("checkpoint logits are not uniquely aligned")
    result: list[dict[str, Any]] = []
    for row_index, item_id in enumerate(ids[0]):
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("checkpoint logits contain an invalid item ID")
        if component == "relevance":
            values = [
                _finite(rows[row_index].get("relevance_logit"), where="relevance_logit")
                for rows in per_checkpoint
            ]
            result.append({"item_id": item_id, "relevance_logit": sum(values) / len(values)})
            continue
        target_values = [rows[row_index].get("target_presence_logits") for rows in per_checkpoint]
        stance_values = [rows[row_index].get("stance_logits") for rows in per_checkpoint]
        if any(
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != len(TARGET_CLASSES)
            for values in target_values
        ) or any(
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != len(ANALYTIC_TARGET_CLASSES)
            for values in stance_values
        ):
            raise ValueError("target/stance checkpoint tensor width drifted")
        targets = [
            sum(_finite(values[index], where="target_presence_logit") for values in target_values)
            / len(target_values)
            for index in range(len(TARGET_CLASSES))
        ]
        stances: list[list[float]] = []
        for target_index in range(len(ANALYTIC_TARGET_CLASSES)):
            if any(
                not isinstance(values[target_index], Sequence)
                or isinstance(values[target_index], (str, bytes))
                or len(values[target_index]) != len(STANCE_CLASSES_B4)
                for values in stance_values
            ):
                raise ValueError("stance checkpoint class width drifted")
            stances.append(
                [
                    sum(
                        _finite(
                            values[target_index][class_index],
                            where="stance_logit",
                        )
                        for values in stance_values
                    )
                    / len(stance_values)
                    for class_index in range(len(STANCE_CLASSES_B4))
                ]
            )
        result.append(
            {
                "item_id": item_id,
                "target_presence_logits": targets,
                "stance_logits": stances,
            }
        )
    return result


def _clean_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {
        "item_id",
        "label",
        "inclusion_probability",
        "relevance_logit",
        "target_presence_logits",
        "stance_logits",
    }
    if isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("calibration rows must be a non-empty sequence")
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, raw in enumerate(rows):
        if set(raw) != expected:
            raise ValueError(f"calibration row {row_index} schema drifted")
        item_id = raw["item_id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("calibration row IDs are invalid or duplicate")
        seen.add(item_id)
        probability = _finite(raw["inclusion_probability"], where="inclusion_probability")
        if not 0.0 < probability <= 1.0:
            raise ValueError("inclusion_probability must be inside (0, 1]")
        targets = raw["target_presence_logits"]
        stances = raw["stance_logits"]
        if (
            not isinstance(targets, Sequence)
            or isinstance(targets, (str, bytes))
            or len(targets) != len(TARGET_CLASSES)
            or not isinstance(stances, Sequence)
            or isinstance(stances, (str, bytes))
            or len(stances) != len(ANALYTIC_TARGET_CLASSES)
        ):
            raise ValueError("calibration tensor width drifted")
        clean_stances = []
        for target_index, values in enumerate(stances):
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or len(values) != len(STANCE_CLASSES_B4)
            ):
                raise ValueError(f"stance target {target_index} width drifted")
            clean_stances.append([_finite(value, where="stance_logit") for value in values])
        clean.append(
            {
                "item_id": item_id,
                "label": validate_v2_label(raw["label"]),
                "inclusion_probability": probability,
                "relevance_logit": _finite(raw["relevance_logit"], where="relevance_logit"),
                "target_presence_logits": [
                    _finite(value, where="target_presence_logit") for value in targets
                ],
                "stance_logits": clean_stances,
            }
        )
    return clean


def _weighted_binary_nll(
    logits: Sequence[float], labels: Sequence[bool], weights: Sequence[float], temperature: float
) -> float:
    total_weight = sum(weights)
    return (
        sum(
            weight
            * -math.log(
                max(
                    _sigmoid(logit, temperature) if label else 1.0 - _sigmoid(logit, temperature),
                    1e-12,
                )
            )
            for logit, label, weight in zip(logits, labels, weights, strict=True)
        )
        / total_weight
    )


def _weighted_categorical_nll(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float],
    temperature: float,
) -> float:
    total_weight = sum(weights)
    return (
        sum(
            weight * -math.log(max(_softmax(values, temperature)[label], 1e-12))
            for values, label, weight in zip(logits, labels, weights, strict=True)
        )
        / total_weight
    )


def _select_temperature(loss: Any) -> tuple[float, float]:
    ranked = [
        (float(loss(temperature)), abs(temperature - 1.0), temperature)
        for temperature in candidate.TEMPERATURE_GRID
    ]
    selected_loss, _, selected_temperature = min(ranked)
    return selected_temperature, selected_loss


def _binary_counts(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    weights: Sequence[float],
    threshold: float,
) -> dict[str, float | None]:
    tp = fp = fn = 0.0
    for probability, label, weight in zip(probabilities, labels, weights, strict=True):
        predicted = probability >= threshold
        tp += weight * predicted * label
        fp += weight * predicted * (not label)
        fn += weight * (not predicted) * label
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def _select_threshold(
    probabilities: Sequence[float], labels: Sequence[bool], weights: Sequence[float]
) -> tuple[float, dict[str, float | None]]:
    ranked = []
    metrics_by_threshold = {}
    for threshold in THRESHOLD_GRID:
        metrics = _binary_counts(probabilities, labels, weights, threshold)
        metrics_by_threshold[threshold] = metrics
        f1 = -1.0 if metrics["f1"] is None else float(metrics["f1"])
        recall = -1.0 if metrics["recall"] is None else float(metrics["recall"])
        ranked.append((-f1, -recall, abs(threshold - 0.5), threshold))
    selected = min(ranked)[-1]
    return selected, metrics_by_threshold[selected]


def _weighted_binary_ece(
    probabilities: Sequence[float], labels: Sequence[bool], weights: Sequence[float]
) -> float:
    total_weight = sum(weights)
    total = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == ECE_BINS - 1 and probability == 1.0)
        ]
        bin_weight = sum(weights[index] for index in members)
        if not bin_weight:
            continue
        confidence = sum(weights[index] * probabilities[index] for index in members) / bin_weight
        accuracy = sum(weights[index] * labels[index] for index in members) / bin_weight
        total += bin_weight * abs(confidence - accuracy)
    return total / total_weight


def _weighted_categorical_ece(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int], weights: Sequence[float]
) -> float:
    confidence = [max(values) for values in probabilities]
    correct = [
        values.index(max(values)) == label
        for values, label in zip(probabilities, labels, strict=True)
    ]
    return _weighted_binary_ece(confidence, correct, weights)


def _target_lookup(label: Mapping[str, Any]) -> dict[str, str | None]:
    return {item["target"]: item.get("stance") for item in label["targets"]}


def fit_weighted_calibration(
    rows: Sequence[Mapping[str, Any]], *, bindings: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit the frozen weighted temperature/threshold grids without seed selection."""

    clean = _clean_rows(rows)
    weights = [1.0 / row["inclusion_probability"] for row in clean]
    codable = [row for row in clean if row["label"]["codability"] == "codable"]
    codable_weights = [
        weight
        for row, weight in zip(clean, weights, strict=True)
        if row["label"]["codability"] == "codable"
    ]
    if not codable:
        raise ValueError("calibration frame lacks codable relevance support")
    relevance_logits = [row["relevance_logit"] for row in codable]
    relevance_labels = [row["label"]["relevance"] == "material" for row in codable]
    relevance_temperature, relevance_nll = _select_temperature(
        lambda temperature: _weighted_binary_nll(
            relevance_logits, relevance_labels, codable_weights, temperature
        )
    )
    relevance_probabilities = [_sigmoid(value, relevance_temperature) for value in relevance_logits]
    relevance_threshold, relevance_threshold_metrics = _select_threshold(
        relevance_probabilities, relevance_labels, codable_weights
    )

    material_indices = [
        index
        for index, row in enumerate(clean)
        if row["label"]["codability"] == "codable" and row["label"]["relevance"] == "material"
    ]
    if not material_indices:
        raise ValueError("calibration frame lacks material target support")
    target_calibration: dict[str, Any] = {}
    stance_calibration: dict[str, Any] = {}
    mixed_excluded = 0
    for target_index, target in enumerate(TARGET_CLASSES):
        target_logits = [
            clean[index]["target_presence_logits"][target_index] for index in material_indices
        ]
        target_labels = [
            target in _target_lookup(clean[index]["label"]) for index in material_indices
        ]
        target_weights = [weights[index] for index in material_indices]
        temperature, nll = _select_temperature(
            lambda value, logits=target_logits, labels=target_labels, row_weights=target_weights: (
                _weighted_binary_nll(logits, labels, row_weights, value)
            )
        )
        probabilities = [_sigmoid(value, temperature) for value in target_logits]
        threshold, threshold_metrics = _select_threshold(
            probabilities, target_labels, target_weights
        )
        target_calibration[target] = {
            "temperature": temperature,
            "weighted_nll": nll,
            "weighted_ece": _weighted_binary_ece(probabilities, target_labels, target_weights),
            "threshold": threshold,
            "threshold_metrics": threshold_metrics,
            "sample_support": len(target_labels),
            "sample_positive_support": sum(target_labels),
        }
        if target not in ANALYTIC_TARGET_CLASSES:
            continue
        stance_logits: list[list[float]] = []
        stance_labels: list[int] = []
        stance_weights: list[float] = []
        for index in material_indices:
            stance = _target_lookup(clean[index]["label"]).get(target)
            if stance is None:
                continue
            if stance == MIXED_STATE:
                mixed_excluded += 1
                continue
            stance_logits.append(
                [
                    clean[index]["stance_logits"][target_index][source_index]
                    for source_index in STANCE_ESTIMAND_INDICES
                ]
            )
            stance_labels.append(STANCE_ESTIMAND_CLASSES.index(stance))
            stance_weights.append(weights[index])
        if not stance_logits:
            raise ValueError(f"calibration frame lacks three-class stance support for {target}")
        stance_temperature, stance_nll = _select_temperature(
            lambda value, logits=stance_logits, labels=stance_labels, row_weights=stance_weights: (
                _weighted_categorical_nll(logits, labels, row_weights, value)
            )
        )
        stance_probabilities = [_softmax(values, stance_temperature) for values in stance_logits]
        stance_calibration[target] = {
            "temperature": stance_temperature,
            "weighted_nll": stance_nll,
            "weighted_ece": _weighted_categorical_ece(
                stance_probabilities, stance_labels, stance_weights
            ),
            "sample_support": len(stance_labels),
        }
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": CALIBRATION_KIND,
        "bindings": dict(bindings),
        "sample_rows": len(clean),
        "codable_relevance_rows": len(codable),
        "material_target_rows": len(material_indices),
        "temperature_grid": list(candidate.TEMPERATURE_GRID),
        "threshold_grid": list(THRESHOLD_GRID),
        "ece_bins": ECE_BINS,
        "weighting": "inverse-inclusion-probability-hajek-ratio",
        "ensemble": "unweighted-arithmetic-mean-aligned-raw-logits-all-three-seeds",
        "relevance": {
            "temperature": relevance_temperature,
            "weighted_nll": relevance_nll,
            "weighted_ece": _weighted_binary_ece(
                relevance_probabilities, relevance_labels, codable_weights
            ),
            "threshold": relevance_threshold,
            "threshold_metrics": relevance_threshold_metrics,
            "sample_support": len(relevance_labels),
            "sample_positive_support": sum(relevance_labels),
        },
        "target_presence": target_calibration,
        "stance": stance_calibration,
        "stance_checkpoint_classes": list(STANCE_CLASSES_B4),
        "stance_estimand_classes": list(STANCE_ESTIMAND_CLASSES),
        "excluded_mixed_stance_instances": mixed_excluded,
        "abstention_margin": 0.0,
        "stance_min_confidence": 0.0,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
        "evidence_boundary": (
            "in-frame weighted calibration diagnostics on model-assisted silver labels; "
            "not human-gold or independent validation"
        ),
    }
    artefact = {**body, "calibration_id": candidate.canonical_sha256(body)}
    assert_metadata_only(artefact, where="probability-random weighted calibration")
    return artefact


def thresholds_from_calibration(calibration: Mapping[str, Any]) -> DecodeThresholds:
    return DecodeThresholds(
        stance_representation="B4",
        relevance_threshold=float(calibration["relevance"]["threshold"]),
        relevance_abstention_margin=0.0,
        target_thresholds=tuple(
            float(calibration["target_presence"][target]["threshold"]) for target in TARGET_CLASSES
        ),
        target_abstention_margins=(0.0,) * len(TARGET_CLASSES),
        stance_min_confidence=0.0,
    )


def calibrated_probabilities(
    row: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    relevance = _sigmoid(
        _finite(row["relevance_logit"], where="relevance_logit"),
        float(calibration["relevance"]["temperature"]),
    )
    targets = tuple(
        _sigmoid(
            _finite(row["target_presence_logits"][index], where="target_presence_logit"),
            float(calibration["target_presence"][target]["temperature"]),
        )
        for index, target in enumerate(TARGET_CLASSES)
    )
    stances = []
    for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES):
        projected = [
            _finite(row["stance_logits"][target_index][source_index], where="stance_logit")
            for source_index in STANCE_ESTIMAND_INDICES
        ]
        three_class = _softmax(projected, float(calibration["stance"][target]["temperature"]))
        by_class = dict(zip(STANCE_ESTIMAND_CLASSES, three_class, strict=True))
        stances.append(
            tuple(0.0 if state == MIXED_STATE else by_class[state] for state in STANCE_CLASSES_B4)
        )
    return {
        "relevance_probability": relevance,
        "target_probabilities": targets,
        "stance_probabilities": tuple(stances),
    }


def build_aggregate_validation(
    rows: Sequence[Mapping[str, Any]],
    membership: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the exact probability sample and return aggregate-only evidence."""

    clean = _clean_rows(rows)
    if len(clean) != len(membership):
        raise ValueError("membership and calibration row counts differ")
    by_id = {row["item_id"]: row for row in clean}
    if set(by_id) != {row.get("opaque_id") for row in membership}:
        raise ValueError("membership and calibration identities differ")
    thresholds = thresholds_from_calibration(calibration)
    reference = {item_id: row["label"] for item_id, row in by_id.items()}
    predictions = {}
    design = {}
    for membership_row in membership:
        item_id = membership_row["opaque_id"]
        probabilities = calibrated_probabilities(by_id[item_id], calibration)
        predictions[item_id] = decode_factorised_probabilities(
            relevance_probability=probabilities["relevance_probability"],
            target_probabilities=probabilities["target_probabilities"],
            stance_probabilities=probabilities["stance_probabilities"],
            thresholds=thresholds,
        )
        design[item_id] = {
            "frame": "calibration",
            "selection_component": membership_row["selection_component"],
            "selection_stratum": membership_row["selection_stratum"],
            "inclusion_probability_numerator": membership_row["inclusion_probability_numerator"],
            "inclusion_probability_denominator": membership_row[
                "inclusion_probability_denominator"
            ],
            "inclusion_probability": membership_row["inclusion_probability"],
            "probability_scope": PROBABILITY_SCOPE,
        }
    probabilities = [float(row["inclusion_probability"]) for row in membership]
    weights = [1.0 / value for value in probabilities]
    weighting = NaturalArmWeightingConfig(
        selection_component="calibration_probability",
        estimand="post-acquisition eligible candidate population conditional on codability",
        design_digest=natural_arm_design_digest(design),
        expected_sample_rows=len(membership),
        expected_population_rows=candidate.REMAINING_POPULATION_ROWS,
        expected_strata=len({row["selection_stratum"] for row in membership}),
        expected_probability_min=min(probabilities),
        expected_probability_max=max(probabilities),
        minimum_effective_sample_size=sum(weights) ** 2 / sum(value * value for value in weights),
        maximum_inverse_weight_ratio=max(weights) / min(weights),
        include_unweighted_conditional_diagnostics=True,
    )
    score = score_natural_probability_arm(
        reference,
        predictions,
        design,
        thresholds=thresholds,
        weighting=weighting,
        excluded_stance_states=(MIXED_STATE,),
    )
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": VALIDATION_KIND,
        "calibration_id": calibration["calibration_id"],
        "sample_rows": len(clean),
        "score": score,
        "source_membership_probability_scope": ("post-acquisition-eligible-candidate-frame"),
        "evaluated_probability_scope": PROBABILITY_SCOPE,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
        "evidence_boundary": (
            "design-weighted in-frame model-assisted silver diagnostics; mixed is outside "
            "the three-class stance estimand; not independent or human-gold validation"
        ),
    }
    artefact = {**body, "validation_id": candidate.canonical_sha256(body)}
    assert_metadata_only(artefact, where="probability-random aggregate validation")
    return artefact


__all__ = [
    "ECE_BINS",
    "STANCE_ESTIMAND_CLASSES",
    "THRESHOLD_GRID",
    "build_aggregate_validation",
    "calibrated_probabilities",
    "fit_weighted_calibration",
    "mean_aligned_logits",
    "thresholds_from_calibration",
]
