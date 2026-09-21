from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation_v2 import (
    DecodeThresholds,
    FactorisedPrediction,
    NaturalArmWeightingConfig,
    decode_factorised_probabilities,
    natural_arm_design_digest,
    natural_arm_weighting_config_from_summary,
    score_factorised_predictions,
    score_natural_probability_arm,
    weighted_binary_threshold_grid,
)


def _label(
    relevance: str | None,
    targets: list[tuple[str, str | None]],
    *,
    codability: str = "codable",
) -> dict[str, object]:
    return {
        "codability": codability,
        "relevance": relevance,
        "targets": [
            {"target": target, "stance": stance} for target, stance in targets
        ],
    }


def _b4_state(state: str) -> list[float]:
    order = ("negative", "mixed", "no_directed_stance", "positive")
    return [float(value == state) for value in order]


def _b2_state(state: str) -> list[float]:
    return {
        "negative": [1.0, 0.0],
        "positive": [0.0, 1.0],
        "mixed": [1.0, 1.0],
        "no_directed_stance": [0.0, 0.0],
    }[state]


def _decode(
    thresholds: DecodeThresholds,
    *,
    relevance: float = 0.9,
    targets: tuple[float, ...] = (0.9, 0.1, 0.1, 0.1, 0.1, 0.1),
    stances: tuple[str, ...] = (
        "negative",
        "no_directed_stance",
        "positive",
        "mixed",
        "negative",
    ),
) -> FactorisedPrediction:
    builder = _b4_state if thresholds.stance_representation == "B4" else _b2_state
    return decode_factorised_probabilities(
        relevance_probability=relevance,
        target_probabilities=targets,
        stance_probabilities=[builder(state) for state in stances],
        thresholds=thresholds,
    )


def _natural_design(*, component: str = "calibration_probability") -> dict[str, dict[str, object]]:
    frame = "calibration" if component == "calibration_probability" else "development"
    result: dict[str, dict[str, object]] = {}
    for item_id, stratum, numerator, denominator in (
        ("a", '[2020,"large","comment"]', 2, 4),
        ("b", '[2020,"large","comment"]', 2, 4),
        ("c", '[2021,"small","submission"]', 2, 8),
        ("d", '[2021,"small","submission"]', 2, 8),
    ):
        result[item_id] = {
            "frame": frame,
            "selection_component": component,
            "selection_stratum": stratum,
            "inclusion_probability_numerator": numerator,
            "inclusion_probability_denominator": denominator,
            "inclusion_probability": numerator / denominator,
            "probability_scope": "conditional-within-frozen-metadata-stratum",
        }
    return result


def _weighting_config(
    design: dict[str, dict[str, object]],
    *,
    minimum_ess: float = 3.5,
    maximum_ratio: float = 2.0,
    probability_min: float = 0.25,
    probability_max: float = 0.5,
    include_unweighted: bool = False,
) -> NaturalArmWeightingConfig:
    return NaturalArmWeightingConfig(
        selection_component=str(design["a"]["selection_component"]),
        estimand="synthetic-natural-arm-population",
        design_digest=natural_arm_design_digest(design),
        expected_sample_rows=4,
        expected_population_rows=12,
        expected_strata=2,
        expected_probability_min=probability_min,
        expected_probability_max=probability_max,
        minimum_effective_sample_size=minimum_ess,
        maximum_inverse_weight_ratio=maximum_ratio,
        include_unweighted_conditional_diagnostics=include_unweighted,
    )


def test_b4_and_b2_decode_equivalent_states_without_forcing_targets() -> None:
    b4 = DecodeThresholds(stance_representation="B4")
    b2 = DecodeThresholds(stance_representation="B2")
    no_target_probabilities = (0.1,) * 6

    b4_output = _decode(b4, targets=no_target_probabilities)
    b2_output = _decode(b2, targets=no_target_probabilities)

    assert b4_output.full_relevance == b2_output.full_relevance == "material"
    assert b4_output.full_target_presence == b2_output.full_target_presence == (False,) * 6
    assert b4_output.full_stances == b2_output.full_stances
    assert b4_output.stance_class_probabilities == b2_output.stance_class_probabilities

    reference = {
        "synthetic": _label(
            "material", [("china_general", "negative")]
        )
    }
    b4_metrics = score_factorised_predictions(
        reference, {"synthetic": b4_output}, thresholds=b4
    )
    b2_metrics = score_factorised_predictions(
        reference, {"synthetic": b2_output}, thresholds=b2
    )
    assert b4_metrics["end_to_end"] == b2_metrics["end_to_end"]
    assert b4_metrics["conditional_diagnostics"] == b2_metrics["conditional_diagnostics"]
    assert b4_metrics["diagnostics"]["full_material_without_target"] == 1
    assert b4_metrics["diagnostics"]["forced_target_selections"] == 0


def test_metrics_separate_conditional_and_end_to_end_denominators() -> None:
    thresholds = DecodeThresholds(stance_representation="B4")
    reference = {
        "a": _label(
            "material",
            [
                ("china_general", "negative"),
                ("people_identity", "positive"),
            ],
        ),
        "b": _label("material", [("residual_other", None)]),
        "c": _label("not_material", []),
        "d": _label(None, [], codability="not_codable"),
    }
    outputs = {
        "a": _decode(
            thresholds,
            targets=(0.9, 0.1, 0.9, 0.1, 0.1, 0.1),
            stances=(
                "negative",
                "mixed",
                "positive",
                "no_directed_stance",
                "negative",
            ),
        ),
        "b": _decode(thresholds, targets=(0.1,) * 6),
        "c": _decode(
            thresholds,
            relevance=0.9,
            targets=(0.1, 0.1, 0.1, 0.1, 0.9, 0.1),
        ),
        "d": _decode(thresholds),
    }

    metrics = score_factorised_predictions(reference, outputs, thresholds=thresholds)

    assert metrics["reference_masks"] == {
        "codable": 3,
        "not_codable": 1,
        "material": 2,
        "not_material": 1,
    }
    assert metrics["relevance"]["support"] == 3
    assert (
        metrics["conditional_diagnostics"]["target_presence_on_reference_material"][
            "micro"
        ]["support"]
        == 12
    )
    assert (
        metrics["end_to_end"]["target_presence"]["micro"]["support"]
        == 18
    )
    assert (
        metrics["conditional_diagnostics"]["stance_on_reference_present_targets"][
            "micro"
        ]["support"]
        == 2
    )
    assert metrics["end_to_end"]["target_stance_tuples"]["micro"]["support"] == 2
    assert metrics["diagnostics"]["full_material_without_target"] == 1
    assert_metadata_only(metrics)


def test_selective_abstentions_remain_in_population_risk_and_tuple_recall() -> None:
    thresholds = DecodeThresholds(
        stance_representation="B4",
        relevance_abstention_margin=0.1,
        target_abstention_margins=(0.1,) * 6,
        stance_min_confidence=0.8,
    )
    reference = {
        "a": _label("material", [("china_general", "negative")]),
        "b": _label("not_material", []),
    }
    outputs = {
        "a": decode_factorised_probabilities(
            relevance_probability=0.5,
            target_probabilities=(0.5,) + (0.1,) * 5,
            stance_probabilities=[
                [0.4, 0.2, 0.2, 0.2],
                *[_b4_state("negative") for _ in range(4)],
            ],
            thresholds=thresholds,
        ),
        "b": _decode(thresholds, relevance=0.1, targets=(0.1,) * 6),
    }

    metrics = score_factorised_predictions(reference, outputs, thresholds=thresholds)
    selective_relevance = metrics["relevance"]["selective"]
    tuple_risk = metrics["end_to_end"]["selective_target_stance_tuples"][
        "reference_tuple_risk"
    ]

    assert selective_relevance["support"] == 2
    assert selective_relevance["accepted"] == 1
    assert selective_relevance["abstained"] == 1
    assert selective_relevance["population_errors_including_abstentions"] == 1
    assert selective_relevance["population_risk_including_abstentions"] == 0.5
    assert tuple_risk["support"] == 1
    assert tuple_risk["abstained"] == 1
    assert (
        metrics["end_to_end"]["selective_target_stance_tuples"]["micro"][
            "false_negative"
        ]
        == 1
    )


def test_probability_scores_are_aggregate_and_optional() -> None:
    thresholds = DecodeThresholds(stance_representation="B4")
    reference = {
        "a": _label("material", [("china_general", "negative")]),
        "b": _label("not_material", []),
    }
    metrics = score_factorised_predictions(
        reference,
        {
            "a": _decode(thresholds, relevance=1.0),
            "b": _decode(thresholds, relevance=0.0, targets=(0.1,) * 6),
        },
        thresholds=thresholds,
    )
    proper = metrics["relevance"]["proper_scores"]
    assert proper["available"] == 2
    assert proper["brier"] == 0.0
    assert proper["nll"] == 0.0
    assert "probabilities" not in str(metrics).casefold()


def test_missing_full_output_remains_in_every_relevant_denominator() -> None:
    thresholds = DecodeThresholds(stance_representation="B4")
    reference = {
        "a": _label("material", [("china_general", "negative")]),
        "b": _label("not_material", []),
    }
    metrics = score_factorised_predictions(
        reference,
        {"a": None, "b": _decode(thresholds, relevance=0.0, targets=(0.1,) * 6)},
        thresholds=thresholds,
    )

    assert metrics["outputs"] == {"valid": 1, "missing": 1}
    assert metrics["relevance"]["support"] == 2
    assert metrics["relevance"]["material"]["false_negative"] == 1
    assert metrics["relevance"]["accuracy"] == 0.5
    assert (
        metrics["end_to_end"]["target_stance_tuples"]["micro"]["false_negative"]
        == 1
    )


def test_natural_arm_uses_exact_inverse_probability_weights_and_reports_design_health() -> None:
    thresholds = DecodeThresholds(stance_representation="B4")
    design = _natural_design()
    weighting = _weighting_config(design, include_unweighted=True)
    reference = {
        "a": _label("material", [("china_general", "negative")]),
        "b": _label("not_material", []),
        "c": _label("material", [("people_identity", "positive")]),
        "d": _label("material", [("government_ccp", "negative")]),
    }
    outputs = {
        "a": _decode(thresholds),
        "b": _decode(thresholds, relevance=0.1, targets=(0.1,) * 6),
        "c": _decode(thresholds, relevance=0.1, targets=(0.1,) * 6),
        "d": _decode(thresholds, relevance=0.1, targets=(0.1,) * 6),
    }

    metrics = score_natural_probability_arm(
        reference,
        outputs,
        design,
        thresholds=thresholds,
        weighting=weighting,
    )

    health = metrics["probability_design"]
    relevance = metrics["design_weighted"]["relevance"]
    tuples = metrics["design_weighted"]["end_to_end"]["target_stance_tuples"]
    assert health["sample_rows"] == 4
    assert health["represented_population_rows"] == 12
    assert health["observed_strata"] == 2
    assert health["inclusion_probability_min"] == 0.25
    assert health["inclusion_probability_max"] == 0.5
    assert health["inverse_weight_min"] == 2.0
    assert health["inverse_weight_max"] == 4.0
    assert health["inverse_weight_ratio"] == 2.0
    assert health["effective_sample_size"] == 3.6
    assert relevance["accuracy"] == 0.333333
    assert tuples["micro"]["sample_support"] == 3
    assert tuples["micro"]["sample_true_positive"] == 1
    assert tuples["micro"]["sample_false_negative"] == 2
    assert tuples["per_target"]["china_general"]["sample_support"] == 1
    assert tuples["reference_cells"]["china_general"]["negative"] == {
        "sample_support": 1,
        "sample_correct": 1,
        "estimated_support": 2.0,
        "estimated_correct": 2.0,
        "accuracy": 1.0,
    }
    assert metrics["scientific_aggregate_eligible"] is True
    assert metrics["mixed_frame_aggregate_permitted"] is False
    assert metrics["enrichment_included"] is False
    assert metrics["unweighted_conditional_diagnostics"] is not None
    assert_metadata_only(metrics)

    from_summary = natural_arm_weighting_config_from_summary(
        {
            "component": "calibration_probability",
            "design_digest": natural_arm_design_digest(design),
            "expected_sample_rows": 4,
            "represented_population_rows": 12,
            "expected_strata": 2,
            "expected_probability_min": 0.25,
            "expected_probability_max": 0.5,
            "design_effective_sample_size": 3.6,
            "inverse_weight_ratio": 2.0,
        },
        estimand="synthetic-natural-arm-population",
    )
    assert from_summary.design_digest == weighting.design_digest
    assert from_summary.minimum_effective_sample_size == 3.6


@pytest.mark.parametrize(
    ("mutation", "config_overrides", "match"),
    [
        (
            lambda design: design["a"].__setitem__("inclusion_probability", 0.3),
            {},
            "disagrees with its fraction",
        ),
        (
            lambda design: design["a"].update(
                {
                    "inclusion_probability_numerator": 1,
                    "inclusion_probability": 0.25,
                }
            ),
            {},
            "inconsistent inclusion fractions",
        ),
        (
            lambda design: None,
            {"minimum_ess": 3.7},
            "effective sample size",
        ),
        (
            lambda design: None,
            {"maximum_ratio": 1.5},
            "inverse-weight ratio",
        ),
        (
            lambda design: None,
            {"probability_min": 0.2},
            "probability bounds drifted",
        ),
    ],
)
def test_natural_arm_weighting_gates_fail_closed(
    mutation: Callable[[dict[str, dict[str, object]]], None],
    config_overrides: dict[str, float],
    match: str,
) -> None:
    thresholds = DecodeThresholds(stance_representation="B4")
    design = _natural_design()
    mutation(design)
    weighting = _weighting_config(design, **config_overrides)
    reference = {
        item_id: _label("material", [("china_general", "negative")])
        for item_id in design
    }
    outputs = {item_id: _decode(thresholds) for item_id in design}
    with pytest.raises(ValueError, match=match):
        score_natural_probability_arm(
            reference,
            outputs,
            design,
            thresholds=thresholds,
            weighting=weighting,
        )


def test_natural_arm_rejects_enrichment_and_mixed_frame_aggregates() -> None:
    thresholds = DecodeThresholds(stance_representation="B4")
    design = _natural_design()
    design["d"]["selection_component"] = "development_context_available"
    weighting = _weighting_config(design)
    reference = {
        item_id: _label("material", [("china_general", "negative")])
        for item_id in design
    }
    outputs = {item_id: _decode(thresholds) for item_id in design}

    with pytest.raises(ValueError, match="mixes selection components"):
        score_natural_probability_arm(
            reference,
            outputs,
            design,
            thresholds=thresholds,
            weighting=weighting,
        )

    unweighted = score_factorised_predictions(reference, outputs, thresholds=thresholds)
    assert unweighted["evidence_scope"] == "unweighted-diagnostic-only"
    assert unweighted["scientific_aggregate_eligible"] is False
    assert unweighted["mixed_frame_aggregate_permitted"] is False


def test_weighted_threshold_grid_is_bound_and_does_not_select_post_hoc() -> None:
    design = _natural_design()
    weighting = _weighting_config(design)
    result = weighted_binary_threshold_grid(
        {"a": True, "b": False, "c": True, "d": False},
        {"a": 0.9, "b": 0.4, "c": 0.55, "d": 0.45},
        design,
        weighting=weighting,
        decision_name="relevance_material",
        candidates=((0.4, 0.0), (0.5, 0.1)),
    )

    assert result["candidate_count"] == 2
    assert result["automatic_selection_performed"] is False
    assert result["mixed_frame_aggregate_permitted"] is False
    assert result["proper_scores"]["coverage"] == 1.0
    assert result["candidates"][1]["selective"]["coverage_risk"]["estimated_abstained"] > 0
    assert_metadata_only(result)

    with pytest.raises(ValueError, match="lexicographically sorted"):
        weighted_binary_threshold_grid(
            {"a": True, "b": False, "c": True, "d": False},
            {"a": 0.9, "b": 0.4, "c": 0.55, "d": 0.45},
            design,
            weighting=weighting,
            decision_name="relevance_material",
            candidates=((0.5, 0.0), (0.4, 0.0)),
        )


def test_invalid_schema_probability_and_config_binding_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly B4 or B2"):
        DecodeThresholds(stance_representation="four-way")
    with pytest.raises(ValueError, match="entire decision side"):
        DecodeThresholds(
            stance_representation="B4", relevance_abstention_margin=0.5
        )
    thresholds = DecodeThresholds(stance_representation="B4")
    with pytest.raises(ValueError, match="sum to one"):
        decode_factorised_probabilities(
            relevance_probability=0.9,
            target_probabilities=(0.1,) * 6,
            stance_probabilities=[[0.5, 0.5, 0.5, 0.5]] * 5,
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="wrong target width"):
        decode_factorised_probabilities(
            relevance_probability=0.9,
            target_probabilities=(0.1,) * 6,
            stance_probabilities=[_b4_state("negative")],
            thresholds=thresholds,
        )

    output = _decode(thresholds)
    drifted = FactorisedPrediction(
        **{
            **deepcopy(asdict_prediction(output)),
            "config_digest": "0" * 64,
        }
    )
    reference = {"a": _label("material", [("china_general", "negative")])}
    with pytest.raises(ValueError, match="digest drifted"):
        score_factorised_predictions(reference, {"a": drifted}, thresholds=thresholds)
    with pytest.raises(ValueError, match="unexpected identifiers"):
        score_factorised_predictions(
            reference,
            {"a": output, "extra": output},
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="invalid ontology-v2"):
        score_factorised_predictions(
            {"a": _label("unclear", [])},
            {"a": output},
            thresholds=thresholds,
        )


def asdict_prediction(output: FactorisedPrediction) -> dict[str, object]:
    return {
        "config_digest": output.config_digest,
        "stance_representation": output.stance_representation,
        "full_relevance": output.full_relevance,
        "selective_relevance": output.selective_relevance,
        "full_target_presence": output.full_target_presence,
        "selective_target_presence": output.selective_target_presence,
        "full_stances": output.full_stances,
        "selective_stances": output.selective_stances,
        "relevance_probability": output.relevance_probability,
        "target_probabilities": output.target_probabilities,
        "stance_class_probabilities": output.stance_class_probabilities,
    }
