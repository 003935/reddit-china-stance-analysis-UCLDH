"""Aggregate-only evaluation for factorised ontology-v2 predictions.

The scorer keeps conditional diagnostics (reference upstream state) separate from
end-to-end metrics (predicted upstream state).  Missing outputs and selective
abstentions remain in population denominators.  Row-level labels, probabilities and
identifiers are consumed privately and never returned.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    B2_TO_STANCE,
    STANCE_CLASSES_B4,
    TARGET_CLASSES,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_ontology_v2 import validate_v2_label

EVALUATION_SCHEMA_VERSION = "semantic-evaluation-v2"
STANCE_REPRESENTATIONS = ("B4", "B2")
NATURAL_ARM_COMPONENTS = ("calibration_probability", "development_probability")
PROBABILITY_SCOPE = "conditional-within-frozen-metadata-stratum"
WEIGHTING_ESTIMATOR = "inverse-probability-hajek-ratio-with-ht-support-v1"
NATURAL_DESIGN_DIGEST_FIELDS = (
    "frame",
    "selection_component",
    "selection_stratum",
    "inclusion_probability_numerator",
    "inclusion_probability_denominator",
    "inclusion_probability",
    "probability_scope",
)
NATURAL_COMPONENT_FRAMES = {
    "calibration_probability": "calibration",
    "development_probability": "development",
}


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation contract must be finite JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _probability(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite probability")
    clean = float(value)
    if not math.isfinite(clean) or not 0.0 <= clean <= 1.0:
        raise ValueError(f"{where} must be a finite probability")
    return clean


def _probability_vector(value: Sequence[Any], *, size: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{where} must contain exactly {size} probabilities")
    return tuple(_probability(item, where=f"{where}[{index}]") for index, item in enumerate(value))


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else round(sum(present) / len(present), 6)


@dataclass(frozen=True, slots=True)
class DecodeThresholds:
    """Exact-bound full-coverage and selective decoding configuration."""

    stance_representation: str
    relevance_threshold: float = 0.5
    relevance_abstention_margin: float = 0.0
    target_thresholds: tuple[float, ...] = (0.5,) * len(TARGET_CLASSES)
    target_abstention_margins: tuple[float, ...] = (0.0,) * len(TARGET_CLASSES)
    stance_min_confidence: float = 0.0
    b2_bit_thresholds: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if self.stance_representation not in STANCE_REPRESENTATIONS:
            raise ValueError("stance_representation must be exactly B4 or B2")
        relevance_threshold = _probability(self.relevance_threshold, where="relevance_threshold")
        relevance_margin = _probability(
            self.relevance_abstention_margin,
            where="relevance_abstention_margin",
        )
        if not 0.0 < relevance_threshold < 1.0:
            raise ValueError("relevance_threshold must be strictly inside (0, 1)")
        if relevance_margin >= min(relevance_threshold, 1.0 - relevance_threshold):
            raise ValueError("relevance_abstention_margin covers an entire decision side")

        if type(self.target_thresholds) is not tuple:
            raise ValueError("target_thresholds must be an immutable tuple")
        if type(self.target_abstention_margins) is not tuple:
            raise ValueError("target_abstention_margins must be an immutable tuple")
        if type(self.b2_bit_thresholds) is not tuple:
            raise ValueError("b2_bit_thresholds must be an immutable tuple")
        targets = _probability_vector(
            self.target_thresholds,
            size=len(TARGET_CLASSES),
            where="target_thresholds",
        )
        margins = _probability_vector(
            self.target_abstention_margins,
            size=len(TARGET_CLASSES),
            where="target_abstention_margins",
        )
        for index, (threshold, margin) in enumerate(zip(targets, margins, strict=True)):
            if not 0.0 < threshold < 1.0:
                raise ValueError(f"target_thresholds[{index}] must be inside (0, 1)")
            if margin >= min(threshold, 1.0 - threshold):
                raise ValueError(
                    f"target_abstention_margins[{index}] covers an entire decision side"
                )
        _probability(self.stance_min_confidence, where="stance_min_confidence")
        bits = _probability_vector(
            self.b2_bit_thresholds,
            size=2,
            where="b2_bit_thresholds",
        )
        if any(not 0.0 < threshold < 1.0 for threshold in bits):
            raise ValueError("B2 bit thresholds must be strictly inside (0, 1)")

    def digest(self) -> str:
        return canonical_sha256(
            {"schema_version": EVALUATION_SCHEMA_VERSION, "thresholds": asdict(self)}
        )


@dataclass(frozen=True, slots=True)
class NaturalArmWeightingConfig:
    """Exact frozen sampling-design binding for one natural probability arm.

    Every bound is supplied by the prepared evidence-frame manifest.  There are no
    permissive defaults because changing the sample or weighting diagnostics after
    seeing outcomes would change the estimand.
    """

    selection_component: str
    estimand: str
    design_digest: str
    expected_sample_rows: int
    expected_population_rows: int
    expected_strata: int
    expected_probability_min: float
    expected_probability_max: float
    minimum_effective_sample_size: float
    maximum_inverse_weight_ratio: float
    include_unweighted_conditional_diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.selection_component not in NATURAL_ARM_COMPONENTS:
            raise ValueError("selection_component is not a registered natural probability arm")
        if not isinstance(self.estimand, str) or not self.estimand:
            raise ValueError("estimand must be a non-empty registered description")
        if not _is_sha256(self.design_digest):
            raise ValueError("design_digest must be a lowercase SHA-256")
        for name in ("expected_sample_rows", "expected_population_rows", "expected_strata"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.expected_population_rows < self.expected_sample_rows:
            raise ValueError("expected population cannot be smaller than the sample")
        probability_min = _probability(
            self.expected_probability_min,
            where="expected_probability_min",
        )
        probability_max = _probability(
            self.expected_probability_max,
            where="expected_probability_max",
        )
        if probability_min <= 0 or probability_min > probability_max:
            raise ValueError("registered inclusion-probability bounds are invalid")
        if (
            isinstance(self.minimum_effective_sample_size, bool)
            or not isinstance(self.minimum_effective_sample_size, (int, float))
            or not math.isfinite(self.minimum_effective_sample_size)
            or not 0 < self.minimum_effective_sample_size <= self.expected_sample_rows
        ):
            raise ValueError("minimum_effective_sample_size is outside the sample bounds")
        if (
            isinstance(self.maximum_inverse_weight_ratio, bool)
            or not isinstance(self.maximum_inverse_weight_ratio, (int, float))
            or not math.isfinite(self.maximum_inverse_weight_ratio)
            or self.maximum_inverse_weight_ratio < 1.0
        ):
            raise ValueError("maximum_inverse_weight_ratio must be finite and at least one")
        if type(self.include_unweighted_conditional_diagnostics) is not bool:
            raise ValueError("include_unweighted_conditional_diagnostics must be boolean")

    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "weighting_estimator": WEIGHTING_ESTIMATOR,
                "natural_arm": asdict(self),
            }
        )


def natural_arm_weighting_config_from_summary(
    summary: Mapping[str, Any],
    *,
    estimand: str,
    include_unweighted_conditional_diagnostics: bool = False,
) -> NaturalArmWeightingConfig:
    """Build the exact evaluator binding from a validated split-design summary."""

    required = {
        "component",
        "design_digest",
        "expected_sample_rows",
        "represented_population_rows",
        "expected_strata",
        "expected_probability_min",
        "expected_probability_max",
        "design_effective_sample_size",
        "inverse_weight_ratio",
    }
    if not isinstance(summary, Mapping) or not set(summary) >= required:
        raise ValueError("probability-design summary is missing required weighting fields")
    return NaturalArmWeightingConfig(
        selection_component=summary["component"],
        estimand=estimand,
        design_digest=summary["design_digest"],
        expected_sample_rows=summary["expected_sample_rows"],
        expected_population_rows=summary["represented_population_rows"],
        expected_strata=summary["expected_strata"],
        expected_probability_min=summary["expected_probability_min"],
        expected_probability_max=summary["expected_probability_max"],
        minimum_effective_sample_size=summary["design_effective_sample_size"],
        maximum_inverse_weight_ratio=summary["inverse_weight_ratio"],
        include_unweighted_conditional_diagnostics=(include_unweighted_conditional_diagnostics),
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def natural_arm_design_digest(design: Mapping[str, Mapping[str, Any]]) -> str:
    """Bind the private membership identity and exact probability-design fields."""

    if not design:
        raise ValueError("natural-arm design is empty")
    records: list[dict[str, Any]] = []
    for private_identity, raw in design.items():
        if not isinstance(private_identity, str) or not private_identity:
            raise ValueError("natural-arm design identities must be non-empty strings")
        if not isinstance(raw, Mapping) or any(
            field not in raw for field in NATURAL_DESIGN_DIGEST_FIELDS
        ):
            raise ValueError("natural-arm design row is missing a required field")
        records.append(
            {
                "private_identity": private_identity,
                **{field: raw[field] for field in NATURAL_DESIGN_DIGEST_FIELDS},
            }
        )
    records.sort(key=lambda record: record["private_identity"])
    return canonical_sha256(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "weighting_estimator": WEIGHTING_ESTIMATOR,
            "records": records,
        }
    )


def _validate_natural_design(
    reference_keys: Sequence[str],
    design: Mapping[str, Mapping[str, Any]],
    *,
    config: NaturalArmWeightingConfig,
) -> tuple[list[float], dict[str, Any]]:
    if set(design) != set(reference_keys):
        raise ValueError("natural-arm design identities must exactly match the evaluation frame")
    if len(reference_keys) != config.expected_sample_rows:
        raise ValueError("natural-arm sample row count drifted")
    if natural_arm_design_digest(design) != config.design_digest:
        raise ValueError("natural-arm design digest drifted")

    expected_frame = NATURAL_COMPONENT_FRAMES[config.selection_component]
    strata: dict[str, dict[str, Any]] = {}
    probabilities: list[float] = []
    weights: list[float] = []
    for private_identity in reference_keys:
        row = design[private_identity]
        if row["frame"] != expected_frame:
            raise ValueError("natural-arm membership frame disagrees with its component")
        if row["selection_component"] != config.selection_component:
            raise ValueError("natural-arm membership mixes selection components")
        if row["probability_scope"] != PROBABILITY_SCOPE:
            raise ValueError("natural-arm probability scope drifted")
        stratum = row["selection_stratum"]
        numerator = row["inclusion_probability_numerator"]
        denominator = row["inclusion_probability_denominator"]
        probability = row["inclusion_probability"]
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("natural-arm selection stratum must be non-empty")
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or numerator <= 0
            or denominator < numerator
        ):
            raise ValueError("natural-arm inclusion-probability fraction is invalid")
        clean_probability = _probability(
            probability,
            where="natural-arm inclusion_probability",
        )
        if clean_probability <= 0 or not math.isclose(
            clean_probability,
            numerator / denominator,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("natural-arm inclusion probability disagrees with its fraction")
        existing = strata.setdefault(
            stratum,
            {"numerator": numerator, "denominator": denominator, "selected": 0},
        )
        if existing["numerator"] != numerator or existing["denominator"] != denominator:
            raise ValueError("natural-arm stratum has inconsistent inclusion fractions")
        existing["selected"] += 1
        probabilities.append(clean_probability)
        weights.append(1.0 / clean_probability)

    if len(strata) != config.expected_strata:
        raise ValueError("natural-arm observed stratum count drifted")
    for stratum in strata.values():
        if stratum["selected"] != stratum["numerator"]:
            raise ValueError("natural-arm stratum does not contain its exact selected quota")
    population_rows = sum(stratum["denominator"] for stratum in strata.values())
    if population_rows != config.expected_population_rows:
        raise ValueError("natural-arm represented population row count drifted")
    if not math.isclose(
        sum(weights),
        float(config.expected_population_rows),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("natural-arm Horvitz-Thompson support does not recover the population")
    probability_min = min(probabilities)
    probability_max = max(probabilities)
    if not math.isclose(
        probability_min,
        config.expected_probability_min,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        probability_max,
        config.expected_probability_max,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("natural-arm inclusion-probability bounds drifted")
    weight_min = min(weights)
    weight_max = max(weights)
    weight_ratio = weight_max / weight_min
    if weight_ratio > config.maximum_inverse_weight_ratio + 1e-12:
        raise ValueError("natural-arm inverse-weight ratio exceeds its registered maximum")
    effective_sample_size = sum(weights) ** 2 / sum(weight * weight for weight in weights)
    if effective_sample_size + 1e-9 < config.minimum_effective_sample_size:
        raise ValueError("natural-arm effective sample size is below its registered minimum")

    summary = {
        "weighting_estimator": WEIGHTING_ESTIMATOR,
        "weighting_config_digest": config.digest(),
        "design_digest": config.design_digest,
        "selection_component": config.selection_component,
        "estimand": config.estimand,
        "sample_rows": len(reference_keys),
        "represented_population_rows": population_rows,
        "observed_strata": len(strata),
        "inclusion_probability_min": round(probability_min, 12),
        "inclusion_probability_max": round(probability_max, 12),
        "inverse_weight_min": round(weight_min, 6),
        "inverse_weight_max": round(weight_max, 6),
        "inverse_weight_ratio": round(weight_ratio, 6),
        "effective_sample_size": round(effective_sample_size, 6),
        "effective_sample_size_fraction": _ratio(effective_sample_size, len(reference_keys)),
        "registered_gates_passed": True,
    }
    assert_metadata_only(summary, where="natural_arm_weighting")
    return weights, summary


@dataclass(frozen=True, slots=True)
class FactorisedPrediction:
    """Private decoded output, optionally retaining probabilities for proper scores."""

    config_digest: str
    stance_representation: str
    full_relevance: str
    selective_relevance: str | None
    full_target_presence: tuple[bool, ...]
    selective_target_presence: tuple[bool | None, ...]
    full_stances: tuple[str, ...]
    selective_stances: tuple[str | None, ...]
    relevance_probability: float | None = None
    target_probabilities: tuple[float, ...] | None = None
    stance_class_probabilities: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not _is_sha256(self.config_digest):
            raise ValueError("config_digest must be a lowercase SHA-256")
        if self.stance_representation not in STANCE_REPRESENTATIONS:
            raise ValueError("unsupported stance representation")
        if self.full_relevance not in {"material", "not_material"}:
            raise ValueError("full_relevance must be binary")
        if self.selective_relevance not in {"material", "not_material", None}:
            raise ValueError("selective_relevance must be binary or abstained")
        if (
            type(self.full_target_presence) is not tuple
            or len(self.full_target_presence) != len(TARGET_CLASSES)
            or any(type(value) is not bool for value in self.full_target_presence)
        ):
            raise ValueError("full_target_presence has an invalid shape or dtype")
        if (
            type(self.selective_target_presence) is not tuple
            or len(self.selective_target_presence) != len(TARGET_CLASSES)
            or any(
                value is not None and type(value) is not bool
                for value in self.selective_target_presence
            )
        ):
            raise ValueError("selective_target_presence has an invalid shape or dtype")
        if (
            type(self.full_stances) is not tuple
            or len(self.full_stances) != len(ANALYTIC_TARGET_CLASSES)
            or any(value not in STANCE_CLASSES_B4 for value in self.full_stances)
        ):
            raise ValueError("full_stances has an invalid shape or state")
        if (
            type(self.selective_stances) is not tuple
            or len(self.selective_stances) != len(ANALYTIC_TARGET_CLASSES)
            or any(
                value is not None and value not in STANCE_CLASSES_B4
                for value in self.selective_stances
            )
        ):
            raise ValueError("selective_stances has an invalid shape or state")

        probability_fields = (
            self.relevance_probability,
            self.target_probabilities,
            self.stance_class_probabilities,
        )
        if all(value is None for value in probability_fields):
            return
        if any(value is None for value in probability_fields):
            raise ValueError("probability fields must be supplied as one complete bundle")
        _probability(self.relevance_probability, where="relevance_probability")
        _probability_vector(
            self.target_probabilities,
            size=len(TARGET_CLASSES),
            where="target_probabilities",
        )
        if type(self.target_probabilities) is not tuple:
            raise ValueError("target_probabilities must be an immutable tuple")
        if type(self.stance_class_probabilities) is not tuple or any(
            type(values) is not tuple for values in self.stance_class_probabilities
        ):
            raise ValueError("stance_class_probabilities must use immutable tuples")
        if len(self.stance_class_probabilities) != len(ANALYTIC_TARGET_CLASSES):
            raise ValueError("stance_class_probabilities has the wrong target width")
        for target_index, values in enumerate(self.stance_class_probabilities):
            clean = _probability_vector(
                values,
                size=len(STANCE_CLASSES_B4),
                where=f"stance_class_probabilities[{target_index}]",
            )
            if not math.isclose(sum(clean), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("stance class probabilities must sum to one")


def _binary_decode(probability: float, threshold: float, margin: float) -> tuple[bool, bool | None]:
    full = probability >= threshold
    selective = None if abs(probability - threshold) < margin else full
    return full, selective


def _b2_class_probabilities(negative: float, positive: float) -> tuple[float, ...]:
    by_state = {
        "negative": negative * (1.0 - positive),
        "positive": (1.0 - negative) * positive,
        "mixed": negative * positive,
        "no_directed_stance": (1.0 - negative) * (1.0 - positive),
    }
    return tuple(by_state[state] for state in STANCE_CLASSES_B4)


def decode_factorised_probabilities(
    *,
    relevance_probability: float,
    target_probabilities: Sequence[float],
    stance_probabilities: Sequence[Sequence[float]],
    thresholds: DecodeThresholds,
) -> FactorisedPrediction:
    """Decode B4 or B2 probabilities without ever forcing a target selection."""

    relevance_probability = _probability(relevance_probability, where="relevance_probability")
    target_values = _probability_vector(
        target_probabilities,
        size=len(TARGET_CLASSES),
        where="target_probabilities",
    )
    if len(stance_probabilities) != len(ANALYTIC_TARGET_CLASSES):
        raise ValueError("stance_probabilities has the wrong target width")

    relevance_full, relevance_selective = _binary_decode(
        relevance_probability,
        thresholds.relevance_threshold,
        thresholds.relevance_abstention_margin,
    )
    full_targets: list[bool] = []
    selective_targets: list[bool | None] = []
    for probability, threshold, margin in zip(
        target_values,
        thresholds.target_thresholds,
        thresholds.target_abstention_margins,
        strict=True,
    ):
        full, selective = _binary_decode(probability, threshold, margin)
        full_targets.append(full)
        selective_targets.append(selective)

    full_stances: list[str] = []
    selective_stances: list[str | None] = []
    class_probabilities: list[tuple[float, ...]] = []
    for target_index, raw in enumerate(stance_probabilities):
        if thresholds.stance_representation == "B4":
            classes = _probability_vector(
                raw,
                size=len(STANCE_CLASSES_B4),
                where=f"stance_probabilities[{target_index}]",
            )
            if not math.isclose(sum(classes), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("B4 stance probabilities must sum to one")
            state_index = max(range(len(classes)), key=classes.__getitem__)
            state = STANCE_CLASSES_B4[state_index]
            confidence = classes[state_index]
        else:
            bits = _probability_vector(
                raw,
                size=2,
                where=f"stance_probabilities[{target_index}]",
            )
            decoded_bits = (
                int(bits[0] >= thresholds.b2_bit_thresholds[0]),
                int(bits[1] >= thresholds.b2_bit_thresholds[1]),
            )
            state = B2_TO_STANCE[decoded_bits]
            classes = _b2_class_probabilities(*bits)
            confidence = classes[STANCE_CLASSES_B4.index(state)]
        full_stances.append(state)
        selective_stances.append(state if confidence >= thresholds.stance_min_confidence else None)
        class_probabilities.append(classes)

    return FactorisedPrediction(
        config_digest=thresholds.digest(),
        stance_representation=thresholds.stance_representation,
        full_relevance="material" if relevance_full else "not_material",
        selective_relevance=(
            None
            if relevance_selective is None
            else "material"
            if relevance_selective
            else "not_material"
        ),
        full_target_presence=tuple(full_targets),
        selective_target_presence=tuple(selective_targets),
        full_stances=tuple(full_stances),
        selective_stances=tuple(selective_stances),
        relevance_probability=relevance_probability,
        target_probabilities=target_values,
        stance_class_probabilities=tuple(class_probabilities),
    )


def _binary_metrics(
    expected: Sequence[bool],
    observed: Sequence[bool | None],
) -> dict[str, Any]:
    if len(expected) != len(observed):
        raise ValueError("binary metric vectors disagree in length")
    true_positive = false_positive = false_negative = true_negative = abstained = 0
    for reference, value in zip(expected, observed, strict=True):
        abstained += value is None
        true_positive += reference and value is True
        false_positive += not reference and value is True
        false_negative += reference and value is not True
        true_negative += not reference and value is False
    support = len(expected)
    correct = true_positive + true_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "support": support,
        "positive_support": sum(expected),
        "predicted_positive": true_positive + false_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "abstained": abstained,
        "accuracy": _ratio(correct, support),
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": _ratio(2 * true_positive, f1_denominator),
    }


def _selective_summary(expected: Sequence[Any], observed: Sequence[Any | None]) -> dict[str, Any]:
    if len(expected) != len(observed):
        raise ValueError("selective metric vectors disagree in length")
    accepted = sum(value is not None for value in observed)
    correct = sum(
        value is not None and value == reference
        for reference, value in zip(expected, observed, strict=True)
    )
    errors = accepted - correct
    abstained = len(expected) - accepted
    return {
        "support": len(expected),
        "accepted": accepted,
        "correct": correct,
        "errors": errors,
        "abstained": abstained,
        "coverage": _ratio(accepted, len(expected)),
        "accepted_risk": _ratio(errors, accepted),
        "population_errors_including_abstentions": errors + abstained,
        "population_risk_including_abstentions": _ratio(errors + abstained, len(expected)),
    }


def _binary_probability_scores(
    expected: Sequence[bool], probabilities: Sequence[float | None]
) -> dict[str, Any]:
    if len(expected) != len(probabilities):
        raise ValueError("probability metric vectors disagree in length")
    pairs = [
        (int(reference), probability)
        for reference, probability in zip(expected, probabilities, strict=True)
        if probability is not None
    ]
    support = len(expected)
    available = len(pairs)
    if not pairs:
        return {
            "support": support,
            "available": 0,
            "missing": support,
            "coverage": 0.0,
            "brier": None,
            "nll": None,
        }
    epsilon = 1e-15
    brier = sum((probability - reference) ** 2 for reference, probability in pairs) / available
    nll = (
        -sum(
            reference * math.log(max(probability, epsilon))
            + (1 - reference) * math.log(max(1.0 - probability, epsilon))
            for reference, probability in pairs
        )
        / available
    )
    return {
        "support": support,
        "available": available,
        "missing": support - available,
        "coverage": _ratio(available, support),
        "brier": round(brier, 6),
        "nll": round(nll, 6),
    }


def _categorical_metrics(
    expected: Sequence[str],
    observed: Sequence[str | None],
) -> dict[str, Any]:
    if len(expected) != len(observed):
        raise ValueError("categorical metric vectors disagree in length")
    per_state: dict[str, dict[str, Any]] = {}
    for state in STANCE_CLASSES_B4:
        per_state[state] = _binary_metrics(
            [value == state for value in expected],
            [None if value is None else value == state for value in observed],
        )
    correct = sum(value == reference for reference, value in zip(expected, observed, strict=True))
    return {
        "support": len(expected),
        "correct": correct,
        "abstained": sum(value is None for value in observed),
        "accuracy": _ratio(correct, len(expected)),
        "macro_f1": _mean([per_state[state]["f1"] for state in STANCE_CLASSES_B4]),
        "per_state": per_state,
    }


def _categorical_probability_scores(
    expected: Sequence[str],
    probabilities: Sequence[tuple[float, ...] | None],
) -> dict[str, Any]:
    if len(expected) != len(probabilities):
        raise ValueError("categorical probability vectors disagree in length")
    pairs = [
        (reference, values)
        for reference, values in zip(expected, probabilities, strict=True)
        if values is not None
    ]
    support = len(expected)
    available = len(pairs)
    if not pairs:
        return {
            "support": support,
            "available": 0,
            "missing": support,
            "coverage": 0.0,
            "brier": None,
            "nll": None,
        }
    epsilon = 1e-15
    brier_sum = nll_sum = 0.0
    for reference, values in pairs:
        index = STANCE_CLASSES_B4.index(reference)
        brier_sum += sum(
            (probability - int(class_index == index)) ** 2
            for class_index, probability in enumerate(values)
        )
        nll_sum -= math.log(max(values[index], epsilon))
    return {
        "support": support,
        "available": available,
        "missing": support - available,
        "coverage": _ratio(available, support),
        "brier": round(brier_sum / available, 6),
        "nll": round(nll_sum / available, 6),
    }


def _weighted_count(value: float) -> float:
    return round(value, 6)


def _validate_metric_weights(weights: Sequence[float], *, expected_size: int) -> None:
    if len(weights) != expected_size or any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
        for weight in weights
    ):
        raise ValueError("metric weights must be positive finite values with exact row alignment")


def _weighted_binary_metrics(
    expected: Sequence[bool],
    observed: Sequence[bool | None],
    weights: Sequence[float],
) -> dict[str, Any]:
    if len(expected) != len(observed):
        raise ValueError("weighted binary metric vectors disagree in length")
    _validate_metric_weights(weights, expected_size=len(expected))
    true_positive = false_positive = false_negative = true_negative = abstained = 0.0
    for reference, value, weight in zip(expected, observed, weights, strict=True):
        abstained += weight * (value is None)
        true_positive += weight * (reference and value is True)
        false_positive += weight * (not reference and value is True)
        false_negative += weight * (reference and value is not True)
        true_negative += weight * (not reference and value is False)
    support = sum(weights)
    positive_support = sum(
        weight for reference, weight in zip(expected, weights, strict=True) if reference
    )
    predicted_positive = true_positive + false_positive
    correct = true_positive + true_negative
    return {
        "sample_support": len(expected),
        "estimated_population_support": _weighted_count(support),
        "estimated_positive_support": _weighted_count(positive_support),
        "estimated_predicted_positive": _weighted_count(predicted_positive),
        "estimated_true_positive": _weighted_count(true_positive),
        "estimated_false_positive": _weighted_count(false_positive),
        "estimated_false_negative": _weighted_count(false_negative),
        "estimated_true_negative": _weighted_count(true_negative),
        "estimated_abstained": _weighted_count(abstained),
        "accuracy": _ratio(correct, support),
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": _ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
    }


def _weighted_selective_summary(
    expected: Sequence[Any],
    observed: Sequence[Any | None],
    weights: Sequence[float],
) -> dict[str, Any]:
    if len(expected) != len(observed):
        raise ValueError("weighted selective metric vectors disagree in length")
    _validate_metric_weights(weights, expected_size=len(expected))
    support = sum(weights)
    accepted = sum(
        weight for value, weight in zip(observed, weights, strict=True) if value is not None
    )
    correct = sum(
        weight
        for reference, value, weight in zip(expected, observed, weights, strict=True)
        if value is not None and value == reference
    )
    errors = accepted - correct
    abstained = support - accepted
    return {
        "sample_support": len(expected),
        "estimated_population_support": _weighted_count(support),
        "estimated_accepted": _weighted_count(accepted),
        "estimated_correct": _weighted_count(correct),
        "estimated_errors": _weighted_count(errors),
        "estimated_abstained": _weighted_count(abstained),
        "coverage": _ratio(accepted, support),
        "accepted_risk": _ratio(errors, accepted),
        "estimated_population_errors_including_abstentions": _weighted_count(errors + abstained),
        "population_risk_including_abstentions": _ratio(errors + abstained, support),
    }


def _weighted_binary_probability_scores(
    expected: Sequence[bool],
    probabilities: Sequence[float | None],
    weights: Sequence[float],
) -> dict[str, Any]:
    if len(expected) != len(probabilities):
        raise ValueError("weighted probability metric vectors disagree in length")
    _validate_metric_weights(weights, expected_size=len(expected))
    triples = [
        (int(reference), probability, weight)
        for reference, probability, weight in zip(expected, probabilities, weights, strict=True)
        if probability is not None
    ]
    support = sum(weights)
    available = sum(weight for _, _, weight in triples)
    if not triples:
        return {
            "sample_support": len(expected),
            "estimated_population_support": _weighted_count(support),
            "estimated_available": 0.0,
            "estimated_missing": _weighted_count(support),
            "coverage": 0.0,
            "brier": None,
            "nll": None,
        }
    epsilon = 1e-15
    brier = (
        sum(weight * (probability - reference) ** 2 for reference, probability, weight in triples)
        / available
    )
    nll = (
        -sum(
            weight
            * (
                reference * math.log(max(probability, epsilon))
                + (1 - reference) * math.log(max(1.0 - probability, epsilon))
            )
            for reference, probability, weight in triples
        )
        / available
    )
    return {
        "sample_support": len(expected),
        "estimated_population_support": _weighted_count(support),
        "estimated_available": _weighted_count(available),
        "estimated_missing": _weighted_count(support - available),
        "coverage": _ratio(available, support),
        "brier": round(brier, 6),
        "nll": round(nll, 6),
    }


def _weighted_categorical_metrics(
    expected: Sequence[str],
    observed: Sequence[str | None],
    weights: Sequence[float],
) -> dict[str, Any]:
    if len(expected) != len(observed):
        raise ValueError("weighted categorical metric vectors disagree in length")
    _validate_metric_weights(weights, expected_size=len(expected))
    per_state = {
        state: _weighted_binary_metrics(
            [value == state for value in expected],
            [None if value is None else value == state for value in observed],
            weights,
        )
        for state in STANCE_CLASSES_B4
    }
    support = sum(weights)
    correct = sum(
        weight
        for reference, value, weight in zip(expected, observed, weights, strict=True)
        if value == reference
    )
    abstained = sum(
        weight for value, weight in zip(observed, weights, strict=True) if value is None
    )
    return {
        "sample_support": len(expected),
        "estimated_population_support": _weighted_count(support),
        "estimated_correct": _weighted_count(correct),
        "estimated_abstained": _weighted_count(abstained),
        "accuracy": _ratio(correct, support),
        "macro_f1": _mean([per_state[state]["f1"] for state in STANCE_CLASSES_B4]),
        "per_state": per_state,
    }


def _weighted_categorical_probability_scores(
    expected: Sequence[str],
    probabilities: Sequence[tuple[float, ...] | None],
    weights: Sequence[float],
) -> dict[str, Any]:
    if len(expected) != len(probabilities):
        raise ValueError("weighted categorical probability vectors disagree in length")
    _validate_metric_weights(weights, expected_size=len(expected))
    triples = [
        (reference, values, weight)
        for reference, values, weight in zip(expected, probabilities, weights, strict=True)
        if values is not None
    ]
    support = sum(weights)
    available = sum(weight for _, _, weight in triples)
    if not triples:
        return {
            "sample_support": len(expected),
            "estimated_population_support": _weighted_count(support),
            "estimated_available": 0.0,
            "estimated_missing": _weighted_count(support),
            "coverage": 0.0,
            "brier": None,
            "nll": None,
        }
    epsilon = 1e-15
    brier_sum = nll_sum = 0.0
    for reference, values, weight in triples:
        reference_index = STANCE_CLASSES_B4.index(reference)
        brier_sum += weight * sum(
            (probability - int(class_index == reference_index)) ** 2
            for class_index, probability in enumerate(values)
        )
        nll_sum -= weight * math.log(max(values[reference_index], epsilon))
    return {
        "sample_support": len(expected),
        "estimated_population_support": _weighted_count(support),
        "estimated_available": _weighted_count(available),
        "estimated_missing": _weighted_count(support - available),
        "coverage": _ratio(available, support),
        "brier": round(brier_sum / available, 6),
        "nll": round(nll_sum / available, 6),
    }


@dataclass(frozen=True, slots=True)
class _ReferenceState:
    codable: bool
    material: bool
    targets: tuple[bool, ...]
    stances: tuple[str | None, ...]


def _reference_state(value: Mapping[str, Any]) -> _ReferenceState:
    clean = validate_v2_label(value)
    codable = clean["codability"] == "codable"
    material = codable and clean["relevance"] == "material"
    targets = [False] * len(TARGET_CLASSES)
    stances: list[str | None] = [None] * len(ANALYTIC_TARGET_CLASSES)
    for item in clean["targets"]:
        index = TARGET_CLASSES.index(item["target"])
        targets[index] = True
        if index < len(ANALYTIC_TARGET_CLASSES):
            stances[index] = item["stance"]
    return _ReferenceState(codable, material, tuple(targets), tuple(stances))


def _target_metric_surface(
    expected: Mapping[str, list[bool]],
    observed: Mapping[str, list[bool | None]],
    probabilities: Mapping[str, list[float | None]],
) -> dict[str, Any]:
    per_target = {
        target: {
            **_binary_metrics(expected[target], observed[target]),
            "coverage_risk": _selective_summary(expected[target], observed[target]),
            "proper_scores": _binary_probability_scores(expected[target], probabilities[target]),
        }
        for target in TARGET_CLASSES
    }
    expected_all = [value for target in TARGET_CLASSES for value in expected[target]]
    observed_all = [value for target in TARGET_CLASSES for value in observed[target]]
    probabilities_all = [value for target in TARGET_CLASSES for value in probabilities[target]]
    micro = _binary_metrics(expected_all, observed_all)
    return {
        "micro": {
            **micro,
            "coverage_risk": _selective_summary(expected_all, observed_all),
            "proper_scores": _binary_probability_scores(expected_all, probabilities_all),
        },
        "macro_f1": _mean([per_target[target]["f1"] for target in TARGET_CLASSES]),
        "per_target": per_target,
    }


def _weighted_target_metric_surface(
    expected: Mapping[str, list[bool]],
    observed: Mapping[str, list[bool | None]],
    probabilities: Mapping[str, list[float | None]],
    row_weights: Sequence[float],
) -> dict[str, Any]:
    per_target = {
        target: {
            **_weighted_binary_metrics(expected[target], observed[target], row_weights),
            "coverage_risk": _weighted_selective_summary(
                expected[target], observed[target], row_weights
            ),
            "proper_scores": _weighted_binary_probability_scores(
                expected[target], probabilities[target], row_weights
            ),
        }
        for target in TARGET_CLASSES
    }
    expected_all = [value for target in TARGET_CLASSES for value in expected[target]]
    observed_all = [value for target in TARGET_CLASSES for value in observed[target]]
    probabilities_all = [value for target in TARGET_CLASSES for value in probabilities[target]]
    weights_all = list(row_weights) * len(TARGET_CLASSES)
    micro = _weighted_binary_metrics(expected_all, observed_all, weights_all)
    return {
        "micro": {
            **micro,
            "coverage_risk": _weighted_selective_summary(expected_all, observed_all, weights_all),
            "proper_scores": _weighted_binary_probability_scores(
                expected_all, probabilities_all, weights_all
            ),
        },
        "macro_f1": _mean([per_target[target]["f1"] for target in TARGET_CLASSES]),
        "per_target": per_target,
    }


def _tuple_surface(
    expected_sets: Sequence[set[tuple[str, str]]],
    observed_sets: Sequence[set[tuple[str, str]]],
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    per_target_counts = {target: {"tp": 0, "fp": 0, "fn": 0} for target in ANALYTIC_TARGET_CLASSES}
    per_cell = {
        target: {state: {"support": 0, "correct": 0} for state in STANCE_CLASSES_B4}
        for target in ANALYTIC_TARGET_CLASSES
    }
    for expected, observed in zip(expected_sets, observed_sets, strict=True):
        true_positive += len(expected & observed)
        false_positive += len(observed - expected)
        false_negative += len(expected - observed)
        for target in ANALYTIC_TARGET_CLASSES:
            expected_target = {item for item in expected if item[0] == target}
            observed_target = {item for item in observed if item[0] == target}
            per_target_counts[target]["tp"] += len(expected_target & observed_target)
            per_target_counts[target]["fp"] += len(observed_target - expected_target)
            per_target_counts[target]["fn"] += len(expected_target - observed_target)
        for target, state in expected:
            per_cell[target][state]["support"] += 1
            per_cell[target][state]["correct"] += (target, state) in observed

    def tuple_metric(counts: Mapping[str, int]) -> dict[str, Any]:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "support": tp + fn,
            "predicted_positive": tp + fp,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        }

    for target in ANALYTIC_TARGET_CLASSES:
        for state in STANCE_CLASSES_B4:
            cell = per_cell[target][state]
            cell["accuracy"] = _ratio(cell["correct"], cell["support"])
    return {
        "micro": tuple_metric({"tp": true_positive, "fp": false_positive, "fn": false_negative}),
        "per_target": {
            target: tuple_metric(per_target_counts[target]) for target in ANALYTIC_TARGET_CLASSES
        },
        "reference_cells": per_cell,
    }


def _weighted_tuple_surface(
    expected_sets: Sequence[set[tuple[str, str]]],
    observed_sets: Sequence[set[tuple[str, str]]],
    row_weights: Sequence[float],
) -> dict[str, Any]:
    if len(expected_sets) != len(observed_sets):
        raise ValueError("weighted tuple vectors disagree in length")
    _validate_metric_weights(row_weights, expected_size=len(expected_sets))
    true_positive = false_positive = false_negative = 0.0
    sample_true_positive = sample_false_positive = sample_false_negative = 0
    per_target_counts = {
        target: {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "sample_tp": 0,
            "sample_fp": 0,
            "sample_fn": 0,
        }
        for target in ANALYTIC_TARGET_CLASSES
    }
    per_cell = {
        target: {
            state: {
                "sample_support": 0,
                "sample_correct": 0,
                "estimated_support": 0.0,
                "estimated_correct": 0.0,
            }
            for state in STANCE_CLASSES_B4
        }
        for target in ANALYTIC_TARGET_CLASSES
    }
    for expected, observed, weight in zip(expected_sets, observed_sets, row_weights, strict=True):
        sample_true_positive += len(expected & observed)
        sample_false_positive += len(observed - expected)
        sample_false_negative += len(expected - observed)
        true_positive += weight * len(expected & observed)
        false_positive += weight * len(observed - expected)
        false_negative += weight * len(expected - observed)
        for target in ANALYTIC_TARGET_CLASSES:
            expected_target = {item for item in expected if item[0] == target}
            observed_target = {item for item in observed if item[0] == target}
            per_target_counts[target]["sample_tp"] += len(expected_target & observed_target)
            per_target_counts[target]["sample_fp"] += len(observed_target - expected_target)
            per_target_counts[target]["sample_fn"] += len(expected_target - observed_target)
            per_target_counts[target]["tp"] += weight * len(expected_target & observed_target)
            per_target_counts[target]["fp"] += weight * len(observed_target - expected_target)
            per_target_counts[target]["fn"] += weight * len(expected_target - observed_target)
        for target, state in expected:
            per_cell[target][state]["sample_support"] += 1
            per_cell[target][state]["sample_correct"] += (target, state) in observed
            per_cell[target][state]["estimated_support"] += weight
            per_cell[target][state]["estimated_correct"] += weight * ((target, state) in observed)

    def tuple_metric(counts: Mapping[str, float | int]) -> dict[str, Any]:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        sample_tp = int(counts["sample_tp"])
        sample_fp = int(counts["sample_fp"])
        sample_fn = int(counts["sample_fn"])
        return {
            "sample_true_positive": sample_tp,
            "sample_false_positive": sample_fp,
            "sample_false_negative": sample_fn,
            "sample_support": sample_tp + sample_fn,
            "sample_predicted_positive": sample_tp + sample_fp,
            "estimated_true_positive": _weighted_count(tp),
            "estimated_false_positive": _weighted_count(fp),
            "estimated_false_negative": _weighted_count(fn),
            "estimated_support": _weighted_count(tp + fn),
            "estimated_predicted_positive": _weighted_count(tp + fp),
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        }

    for target in ANALYTIC_TARGET_CLASSES:
        for state in STANCE_CLASSES_B4:
            cell = per_cell[target][state]
            support = cell["estimated_support"]
            correct = cell["estimated_correct"]
            cell["estimated_support"] = _weighted_count(support)
            cell["estimated_correct"] = _weighted_count(correct)
            cell["accuracy"] = _ratio(correct, support)
    return {
        "micro": tuple_metric(
            {
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "sample_tp": sample_true_positive,
                "sample_fp": sample_false_positive,
                "sample_fn": sample_false_negative,
            }
        ),
        "per_target": {
            target: tuple_metric(per_target_counts[target]) for target in ANALYTIC_TARGET_CLASSES
        },
        "reference_cells": per_cell,
    }


def score_factorised_predictions(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, FactorisedPrediction | None],
    *,
    thresholds: DecodeThresholds,
    excluded_stance_states: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Score private v2 rows and return a privacy-safe aggregate surface only."""

    if not reference:
        raise ValueError("factorised evaluation reference is empty")
    if (
        type(excluded_stance_states) is not tuple
        or len(set(excluded_stance_states)) != len(excluded_stance_states)
        or any(state not in STANCE_CLASSES_B4 for state in excluded_stance_states)
    ):
        raise ValueError("excluded_stance_states must be unique frozen B4 states")
    if unexpected := set(predictions) - set(reference):
        raise ValueError(f"factorised outputs contain {len(unexpected)} unexpected identifiers")

    states: list[_ReferenceState] = []
    outputs: list[FactorisedPrediction | None] = []
    for item_key, raw_reference in reference.items():
        if not isinstance(item_key, str) or not item_key:
            raise ValueError("reference identifiers must be non-empty strings")
        try:
            states.append(_reference_state(raw_reference))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("reference contains an invalid ontology-v2 label") from exc
        output = predictions.get(item_key)
        if output is not None and not isinstance(output, FactorisedPrediction):
            raise ValueError("outputs must be FactorisedPrediction instances or null")
        if output is not None:
            if output.config_digest != thresholds.digest():
                raise ValueError("output threshold/config digest drifted")
            if output.stance_representation != thresholds.stance_representation:
                raise ValueError("output stance representation drifted")
        outputs.append(output)

    codable_indices = [index for index, state in enumerate(states) if state.codable]
    material_indices = [index for index, state in enumerate(states) if state.material]
    excluded_stance_instances = sum(
        stance in excluded_stance_states
        for state in states
        for stance in state.stances
        if stance is not None
    )

    relevance_expected = [states[index].material for index in codable_indices]
    relevance_full: list[bool | None] = []
    relevance_selective: list[bool | None] = []
    relevance_probabilities: list[float | None] = []
    for index in codable_indices:
        output = outputs[index]
        relevance_full.append(None if output is None else output.full_relevance == "material")
        relevance_selective.append(
            None
            if output is None or output.selective_relevance is None
            else output.selective_relevance == "material"
        )
        relevance_probabilities.append(None if output is None else output.relevance_probability)
    relevance_material = _binary_metrics(relevance_expected, relevance_full)
    relevance_not_material = _binary_metrics(
        [not value for value in relevance_expected],
        [None if value is None else not value for value in relevance_full],
    )
    relevance_surface = {
        "support": len(relevance_expected),
        "accuracy": relevance_material["accuracy"],
        "macro_f1": _mean([relevance_material["f1"], relevance_not_material["f1"]]),
        "material": relevance_material,
        "not_material": relevance_not_material,
        "selective": _selective_summary(relevance_expected, relevance_selective),
        "proper_scores": _binary_probability_scores(relevance_expected, relevance_probabilities),
    }

    conditional_expected = {target: [] for target in TARGET_CLASSES}
    conditional_full = {target: [] for target in TARGET_CLASSES}
    conditional_selective = {target: [] for target in TARGET_CLASSES}
    conditional_probabilities = {target: [] for target in TARGET_CLASSES}
    for index in material_indices:
        output = outputs[index]
        for target_index, target in enumerate(TARGET_CLASSES):
            conditional_expected[target].append(states[index].targets[target_index])
            conditional_full[target].append(
                None if output is None else output.full_target_presence[target_index]
            )
            conditional_selective[target].append(
                None if output is None else output.selective_target_presence[target_index]
            )
            conditional_probabilities[target].append(
                None
                if output is None or output.target_probabilities is None
                else output.target_probabilities[target_index]
            )
    conditional_targets = _target_metric_surface(
        conditional_expected,
        conditional_full,
        conditional_probabilities,
    )
    conditional_targets["selective"] = _target_metric_surface(
        conditional_expected,
        conditional_selective,
        conditional_probabilities,
    )

    end_expected = {target: [] for target in TARGET_CLASSES}
    end_full = {target: [] for target in TARGET_CLASSES}
    end_selective = {target: [] for target in TARGET_CLASSES}
    end_probabilities = {target: [] for target in TARGET_CLASSES}
    for index in codable_indices:
        output = outputs[index]
        for target_index, target in enumerate(TARGET_CLASSES):
            end_expected[target].append(states[index].targets[target_index])
            if output is None:
                full_value = selective_value = probability = None
            else:
                full_value = (
                    output.full_target_presence[target_index]
                    if output.full_relevance == "material"
                    else False
                )
                if output.selective_relevance is None:
                    selective_value = None
                elif output.selective_relevance == "not_material":
                    selective_value = False
                else:
                    selective_value = output.selective_target_presence[target_index]
                probability = (
                    None
                    if output.relevance_probability is None
                    else output.relevance_probability * output.target_probabilities[target_index]
                )
            end_full[target].append(full_value)
            end_selective[target].append(selective_value)
            end_probabilities[target].append(probability)
    end_targets = _target_metric_surface(end_expected, end_full, end_probabilities)
    end_targets["selective"] = _target_metric_surface(
        end_expected,
        end_selective,
        end_probabilities,
    )

    stance_expected = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    stance_full = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    stance_selective = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    stance_probabilities = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    for index in material_indices:
        output = outputs[index]
        for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES):
            reference_stance = states[index].stances[target_index]
            if reference_stance is None or reference_stance in excluded_stance_states:
                continue
            stance_expected[target].append(reference_stance)
            stance_full[target].append(
                None if output is None else output.full_stances[target_index]
            )
            stance_selective[target].append(
                None if output is None else output.selective_stances[target_index]
            )
            stance_probabilities[target].append(
                None
                if output is None or output.stance_class_probabilities is None
                else output.stance_class_probabilities[target_index]
            )
    stance_per_target = {
        target: {
            **_categorical_metrics(stance_expected[target], stance_full[target]),
            "selective": _selective_summary(stance_expected[target], stance_selective[target]),
            "proper_scores": _categorical_probability_scores(
                stance_expected[target], stance_probabilities[target]
            ),
        }
        for target in ANALYTIC_TARGET_CLASSES
    }
    all_stance_expected = [
        value for target in ANALYTIC_TARGET_CLASSES for value in stance_expected[target]
    ]
    all_stance_full = [value for target in ANALYTIC_TARGET_CLASSES for value in stance_full[target]]
    all_stance_selective = [
        value for target in ANALYTIC_TARGET_CLASSES for value in stance_selective[target]
    ]
    all_stance_probabilities = [
        value for target in ANALYTIC_TARGET_CLASSES for value in stance_probabilities[target]
    ]
    conditional_stance = {
        "micro": {
            **_categorical_metrics(all_stance_expected, all_stance_full),
            "selective": _selective_summary(all_stance_expected, all_stance_selective),
            "proper_scores": _categorical_probability_scores(
                all_stance_expected, all_stance_probabilities
            ),
        },
        "per_target": stance_per_target,
    }

    expected_tuples: list[set[tuple[str, str]]] = []
    full_tuples: list[set[tuple[str, str]]] = []
    selective_tuples: list[set[tuple[str, str]]] = []
    reference_tuple_expected: list[str] = []
    reference_tuple_selected: list[str | None] = []
    for index in codable_indices:
        state = states[index]
        output = outputs[index]
        expected_set = {
            (ANALYTIC_TARGET_CLASSES[target_index], stance)
            for target_index, stance in enumerate(state.stances)
            if stance is not None and stance not in excluded_stance_states
        }
        full_set: set[tuple[str, str]] = set()
        selective_set: set[tuple[str, str]] = set()
        if output is not None and output.full_relevance == "material":
            full_set = {
                (target, output.full_stances[target_index])
                for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES)
                if output.full_target_presence[target_index]
                and state.stances[target_index] not in excluded_stance_states
            }
        if output is not None and output.selective_relevance == "material":
            selective_set = {
                (target, output.selective_stances[target_index])
                for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES)
                if output.selective_target_presence[target_index] is True
                and output.selective_stances[target_index] is not None
                and state.stances[target_index] not in excluded_stance_states
            }
        expected_tuples.append(expected_set)
        full_tuples.append(full_set)
        selective_tuples.append(selective_set)

        for target, expected_stance in expected_set:
            reference_tuple_expected.append(f"{target}:{expected_stance}")
            if output is None or output.selective_relevance is None:
                selected = None
            elif output.selective_relevance == "not_material":
                selected = "__absent__"
            else:
                target_index = ANALYTIC_TARGET_CLASSES.index(target)
                target_value = output.selective_target_presence[target_index]
                if target_value is None:
                    selected = None
                elif target_value is False:
                    selected = "__absent__"
                else:
                    stance_value = output.selective_stances[target_index]
                    selected = None if stance_value is None else f"{target}:{stance_value}"
            reference_tuple_selected.append(selected)

    full_tuple_surface = _tuple_surface(expected_tuples, full_tuples)
    selective_tuple_surface = _tuple_surface(expected_tuples, selective_tuples)
    selective_tuple_surface["reference_tuple_risk"] = _selective_summary(
        reference_tuple_expected,
        reference_tuple_selected,
    )

    full_conflicts = selective_conflicts = 0
    for index in codable_indices:
        output = outputs[index]
        if output is None:
            continue
        full_conflicts += output.full_relevance == "material" and not any(
            output.full_target_presence
        )
        selective_conflicts += output.selective_relevance == "material" and not any(
            value is True for value in output.selective_target_presence
        )

    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evidence_scope": "unweighted-diagnostic-only",
        "scientific_aggregate_eligible": False,
        "mixed_frame_aggregate_permitted": False,
        "config_digest": thresholds.digest(),
        "stance_representation": thresholds.stance_representation,
        "excluded_reference_stance_states": list(excluded_stance_states),
        "excluded_reference_stance_instances": excluded_stance_instances,
        "items": len(states),
        "reference_masks": {
            "codable": len(codable_indices),
            "not_codable": len(states) - len(codable_indices),
            "material": len(material_indices),
            "not_material": len(codable_indices) - len(material_indices),
        },
        "outputs": {
            "valid": sum(output is not None for output in outputs),
            "missing": sum(output is None for output in outputs),
        },
        "relevance": relevance_surface,
        "conditional_diagnostics": {
            "target_presence_on_reference_material": conditional_targets,
            "stance_on_reference_present_targets": conditional_stance,
        },
        "end_to_end": {
            "target_presence": end_targets,
            "target_stance_tuples": full_tuple_surface,
            "selective_target_stance_tuples": selective_tuple_surface,
        },
        "diagnostics": {
            "full_material_without_target": full_conflicts,
            "selective_material_without_accepted_target": selective_conflicts,
            "forced_target_selections": 0,
        },
    }
    assert_metadata_only(result, where="semantic_evaluation_v2")
    return result


def score_natural_probability_arm(
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, FactorisedPrediction | None],
    design: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: DecodeThresholds,
    weighting: NaturalArmWeightingConfig,
    excluded_stance_states: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return design-weighted aggregates for one exact natural probability arm.

    This entrypoint cannot score enrichment or a pooled development frame: the
    component, private membership set, stratum quotas and inclusion probabilities are
    exact-bound before any outcome is inspected.
    """

    unweighted = score_factorised_predictions(
        reference,
        predictions,
        thresholds=thresholds,
        excluded_stance_states=excluded_stance_states,
    )
    reference_keys = list(reference)
    row_weights, design_summary = _validate_natural_design(
        reference_keys,
        design,
        config=weighting,
    )
    states = [_reference_state(reference[item_key]) for item_key in reference_keys]
    outputs = [predictions.get(item_key) for item_key in reference_keys]
    codable_indices = [index for index, state in enumerate(states) if state.codable]
    if not codable_indices:
        raise ValueError("natural probability arm has no codable outcome support")
    material_indices = [index for index, state in enumerate(states) if state.material]
    codable_weights = [row_weights[index] for index in codable_indices]

    relevance_expected = [states[index].material for index in codable_indices]
    relevance_full = [
        None if outputs[index] is None else outputs[index].full_relevance == "material"
        for index in codable_indices
    ]
    relevance_selective = [
        None
        if output is None or output.selective_relevance is None
        else output.selective_relevance == "material"
        for index in codable_indices
        for output in (outputs[index],)
    ]
    relevance_probabilities = [
        None if outputs[index] is None else outputs[index].relevance_probability
        for index in codable_indices
    ]
    relevance_material = _weighted_binary_metrics(
        relevance_expected, relevance_full, codable_weights
    )
    relevance_not_material = _weighted_binary_metrics(
        [not value for value in relevance_expected],
        [None if value is None else not value for value in relevance_full],
        codable_weights,
    )
    relevance_surface = {
        "accuracy": relevance_material["accuracy"],
        "macro_f1": _mean([relevance_material["f1"], relevance_not_material["f1"]]),
        "material": relevance_material,
        "not_material": relevance_not_material,
        "selective": {
            **_weighted_binary_metrics(relevance_expected, relevance_selective, codable_weights),
            "coverage_risk": _weighted_selective_summary(
                relevance_expected, relevance_selective, codable_weights
            ),
        },
        "proper_scores": _weighted_binary_probability_scores(
            relevance_expected, relevance_probabilities, codable_weights
        ),
    }

    material_weights = [row_weights[index] for index in material_indices]
    conditional_expected = {target: [] for target in TARGET_CLASSES}
    conditional_full = {target: [] for target in TARGET_CLASSES}
    conditional_selective = {target: [] for target in TARGET_CLASSES}
    conditional_probabilities = {target: [] for target in TARGET_CLASSES}
    for index in material_indices:
        state = states[index]
        output = outputs[index]
        for target_index, target in enumerate(TARGET_CLASSES):
            conditional_expected[target].append(state.targets[target_index])
            conditional_full[target].append(
                None if output is None else output.full_target_presence[target_index]
            )
            conditional_selective[target].append(
                None if output is None else output.selective_target_presence[target_index]
            )
            conditional_probabilities[target].append(
                None
                if output is None or output.target_probabilities is None
                else output.target_probabilities[target_index]
            )
    conditional_targets = _weighted_target_metric_surface(
        conditional_expected,
        conditional_full,
        conditional_probabilities,
        material_weights,
    )
    conditional_targets["selective"] = _weighted_target_metric_surface(
        conditional_expected,
        conditional_selective,
        conditional_probabilities,
        material_weights,
    )

    end_expected = {target: [] for target in TARGET_CLASSES}
    end_full = {target: [] for target in TARGET_CLASSES}
    end_selective = {target: [] for target in TARGET_CLASSES}
    end_probabilities = {target: [] for target in TARGET_CLASSES}
    for index in codable_indices:
        state = states[index]
        output = outputs[index]
        for target_index, target in enumerate(TARGET_CLASSES):
            end_expected[target].append(state.targets[target_index])
            if output is None:
                full_value = selective_value = probability = None
            else:
                full_value = (
                    output.full_target_presence[target_index]
                    if output.full_relevance == "material"
                    else False
                )
                if output.selective_relevance is None:
                    selective_value = None
                elif output.selective_relevance == "not_material":
                    selective_value = False
                else:
                    selective_value = output.selective_target_presence[target_index]
                probability = (
                    None
                    if output.relevance_probability is None
                    else output.relevance_probability * output.target_probabilities[target_index]
                )
            end_full[target].append(full_value)
            end_selective[target].append(selective_value)
            end_probabilities[target].append(probability)
    end_targets = _weighted_target_metric_surface(
        end_expected,
        end_full,
        end_probabilities,
        codable_weights,
    )
    end_targets["selective"] = _weighted_target_metric_surface(
        end_expected,
        end_selective,
        end_probabilities,
        codable_weights,
    )

    stance_expected = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    stance_full = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    stance_selective = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    stance_probabilities = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    stance_weights = {target: [] for target in ANALYTIC_TARGET_CLASSES}
    for index in material_indices:
        state = states[index]
        output = outputs[index]
        for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES):
            reference_stance = state.stances[target_index]
            if reference_stance is None or reference_stance in excluded_stance_states:
                continue
            stance_expected[target].append(reference_stance)
            stance_full[target].append(
                None if output is None else output.full_stances[target_index]
            )
            stance_selective[target].append(
                None if output is None else output.selective_stances[target_index]
            )
            stance_probabilities[target].append(
                None
                if output is None or output.stance_class_probabilities is None
                else output.stance_class_probabilities[target_index]
            )
            stance_weights[target].append(row_weights[index])

    stance_per_target = {}
    for target in ANALYTIC_TARGET_CLASSES:
        full_metrics = _weighted_categorical_metrics(
            stance_expected[target], stance_full[target], stance_weights[target]
        )
        stance_per_target[target] = {
            **full_metrics,
            "selective": {
                **_weighted_categorical_metrics(
                    stance_expected[target],
                    stance_selective[target],
                    stance_weights[target],
                ),
                "coverage_risk": _weighted_selective_summary(
                    stance_expected[target],
                    stance_selective[target],
                    stance_weights[target],
                ),
            },
            "proper_scores": _weighted_categorical_probability_scores(
                stance_expected[target],
                stance_probabilities[target],
                stance_weights[target],
            ),
        }
    all_stance_expected = [
        value for target in ANALYTIC_TARGET_CLASSES for value in stance_expected[target]
    ]
    all_stance_full = [value for target in ANALYTIC_TARGET_CLASSES for value in stance_full[target]]
    all_stance_selective = [
        value for target in ANALYTIC_TARGET_CLASSES for value in stance_selective[target]
    ]
    all_stance_probabilities = [
        value for target in ANALYTIC_TARGET_CLASSES for value in stance_probabilities[target]
    ]
    all_stance_weights = [
        value for target in ANALYTIC_TARGET_CLASSES for value in stance_weights[target]
    ]
    conditional_stance = {
        "micro": {
            **_weighted_categorical_metrics(
                all_stance_expected, all_stance_full, all_stance_weights
            ),
            "selective": {
                **_weighted_categorical_metrics(
                    all_stance_expected,
                    all_stance_selective,
                    all_stance_weights,
                ),
                "coverage_risk": _weighted_selective_summary(
                    all_stance_expected,
                    all_stance_selective,
                    all_stance_weights,
                ),
            },
            "proper_scores": _weighted_categorical_probability_scores(
                all_stance_expected,
                all_stance_probabilities,
                all_stance_weights,
            ),
        },
        "per_target": stance_per_target,
    }

    expected_tuples: list[set[tuple[str, str]]] = []
    full_tuples: list[set[tuple[str, str]]] = []
    selective_tuples: list[set[tuple[str, str]]] = []
    reference_tuple_expected: list[str] = []
    reference_tuple_selected: list[str | None] = []
    reference_tuple_weights: list[float] = []
    for index in codable_indices:
        state = states[index]
        output = outputs[index]
        weight = row_weights[index]
        expected_set = {
            (ANALYTIC_TARGET_CLASSES[target_index], stance)
            for target_index, stance in enumerate(state.stances)
            if stance is not None and stance not in excluded_stance_states
        }
        full_set: set[tuple[str, str]] = set()
        selective_set: set[tuple[str, str]] = set()
        if output is not None and output.full_relevance == "material":
            full_set = {
                (target, output.full_stances[target_index])
                for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES)
                if output.full_target_presence[target_index]
                and state.stances[target_index] not in excluded_stance_states
            }
        if output is not None and output.selective_relevance == "material":
            selective_set = {
                (target, output.selective_stances[target_index])
                for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES)
                if output.selective_target_presence[target_index] is True
                and output.selective_stances[target_index] is not None
                and state.stances[target_index] not in excluded_stance_states
            }
        expected_tuples.append(expected_set)
        full_tuples.append(full_set)
        selective_tuples.append(selective_set)
        for target, expected_stance in expected_set:
            reference_tuple_expected.append(f"{target}:{expected_stance}")
            reference_tuple_weights.append(weight)
            if output is None or output.selective_relevance is None:
                selected = None
            elif output.selective_relevance == "not_material":
                selected = "__absent__"
            else:
                target_index = ANALYTIC_TARGET_CLASSES.index(target)
                target_value = output.selective_target_presence[target_index]
                if target_value is None:
                    selected = None
                elif target_value is False:
                    selected = "__absent__"
                else:
                    stance_value = output.selective_stances[target_index]
                    selected = None if stance_value is None else f"{target}:{stance_value}"
            reference_tuple_selected.append(selected)

    full_tuple_surface = _weighted_tuple_surface(expected_tuples, full_tuples, codable_weights)
    selective_tuple_surface = _weighted_tuple_surface(
        expected_tuples, selective_tuples, codable_weights
    )
    selective_tuple_surface["reference_tuple_risk"] = _weighted_selective_summary(
        reference_tuple_expected,
        reference_tuple_selected,
        reference_tuple_weights,
    )

    full_conflicts = selective_conflicts = 0.0
    for index in codable_indices:
        output = outputs[index]
        weight = row_weights[index]
        if output is None:
            continue
        full_conflicts += weight * (
            output.full_relevance == "material" and not any(output.full_target_presence)
        )
        selective_conflicts += weight * (
            output.selective_relevance == "material"
            and not any(value is True for value in output.selective_target_presence)
        )

    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evidence_scope": "design-weighted-natural-probability-arm",
        "scientific_aggregate_eligible": True,
        "mixed_frame_aggregate_permitted": False,
        "enrichment_included": False,
        "config_digest": thresholds.digest(),
        "stance_representation": thresholds.stance_representation,
        "excluded_reference_stance_states": list(excluded_stance_states),
        "excluded_reference_stance_instances": unweighted["excluded_reference_stance_instances"],
        "reference_masks": unweighted["reference_masks"],
        "probability_design": design_summary,
        "outputs": unweighted["outputs"],
        "design_weighted": {
            "relevance": relevance_surface,
            "conditional_diagnostics": {
                "target_presence_on_reference_material": conditional_targets,
                "stance_on_reference_present_targets": conditional_stance,
            },
            "end_to_end": {
                "target_presence": end_targets,
                "target_stance_tuples": full_tuple_surface,
                "selective_target_stance_tuples": selective_tuple_surface,
            },
            "diagnostics": {
                "estimated_full_material_without_target": _weighted_count(full_conflicts),
                "estimated_selective_material_without_accepted_target": _weighted_count(
                    selective_conflicts
                ),
                "forced_target_selections": 0,
            },
        },
        "unweighted_conditional_diagnostics": (
            unweighted["conditional_diagnostics"]
            if weighting.include_unweighted_conditional_diagnostics
            else None
        ),
        "evidence_boundary": (
            "natural-arm inverse-probability estimates only; enrichment is separate "
            "unweighted diagnostic evidence and pooling is prohibited"
        ),
    }
    assert_metadata_only(result, where="semantic_evaluation_v2.natural_arm")
    return result


def weighted_binary_threshold_grid(
    reference: Mapping[str, bool],
    probabilities: Mapping[str, float],
    design: Mapping[str, Mapping[str, Any]],
    *,
    weighting: NaturalArmWeightingConfig,
    decision_name: str,
    candidates: tuple[tuple[float, float], ...],
) -> dict[str, Any]:
    """Evaluate a frozen threshold/margin grid without selecting a post-hoc winner."""

    if not isinstance(decision_name, str) or not decision_name:
        raise ValueError("decision_name must be a non-empty registered name")
    if (
        not candidates
        or type(candidates) is not tuple
        or any(type(candidate) is not tuple or len(candidate) != 2 for candidate in candidates)
    ):
        raise ValueError("threshold candidates must be a non-empty immutable pair grid")
    if len(set(candidates)) != len(candidates) or tuple(sorted(candidates)) != candidates:
        raise ValueError("threshold candidates must be unique and lexicographically sorted")
    if set(probabilities) != set(reference):
        raise ValueError("threshold probabilities must exactly match the natural-arm reference")
    reference_keys = list(reference)
    weights, design_summary = _validate_natural_design(
        reference_keys,
        design,
        config=weighting,
    )
    expected: list[bool] = []
    probability_values: list[float] = []
    for private_identity in reference_keys:
        value = reference[private_identity]
        if type(value) is not bool:
            raise ValueError("threshold reference states must be boolean")
        expected.append(value)
        probability_values.append(
            _probability(
                probabilities[private_identity],
                where="threshold probability",
            )
        )

    candidate_metrics: list[dict[str, Any]] = []
    for raw_threshold, raw_margin in candidates:
        threshold = _probability(raw_threshold, where="candidate threshold")
        margin = _probability(raw_margin, where="candidate abstention margin")
        if not 0.0 < threshold < 1.0:
            raise ValueError("candidate threshold must be strictly inside (0, 1)")
        if margin >= min(threshold, 1.0 - threshold):
            raise ValueError("candidate abstention margin covers an entire decision side")
        full: list[bool] = []
        selective: list[bool | None] = []
        for probability in probability_values:
            full_value, selective_value = _binary_decode(
                probability,
                threshold,
                margin,
            )
            full.append(full_value)
            selective.append(selective_value)
        candidate_metrics.append(
            {
                "threshold": threshold,
                "abstention_margin": margin,
                "full_coverage": _weighted_binary_metrics(expected, full, weights),
                "selective": {
                    **_weighted_binary_metrics(expected, selective, weights),
                    "coverage_risk": _weighted_selective_summary(expected, selective, weights),
                },
            }
        )

    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "weighted-binary-threshold-grid-v1",
        "evidence_scope": "design-weighted-natural-probability-arm",
        "decision_name": decision_name,
        "probability_design": design_summary,
        "candidate_grid_digest": canonical_sha256(candidates),
        "candidate_count": len(candidates),
        "candidates": candidate_metrics,
        "proper_scores": _weighted_binary_probability_scores(expected, probability_values, weights),
        "automatic_selection_performed": False,
        "mixed_frame_aggregate_permitted": False,
    }
    assert_metadata_only(result, where="semantic_evaluation_v2.threshold_grid")
    return result
