from __future__ import annotations

from reddit_china_stance import modernbert_probability_random_calibration_v1 as calibration
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    TARGET_CLASSES,
)


def _label(
    relevance: str | None,
    targets: list[tuple[str, str]],
    *,
    codability: str = "codable",
) -> dict:
    return {
        "codability": codability,
        "relevance": relevance,
        "targets": [{"target": target, "stance": stance} for target, stance in targets],
    }


def _rows() -> list[dict]:
    all_negative = [(target, "negative") for target in ANALYTIC_TARGET_CLASSES]
    mostly_positive = [
        (target, "mixed" if target == "china_general" else "positive")
        for target in ANALYTIC_TARGET_CLASSES
    ]
    labels = [
        _label("material", all_negative),
        _label("not_material", []),
        _label(None, [], codability="not_codable"),
        _label("material", mostly_positive),
    ]
    result = []
    for index, label in enumerate(labels):
        positive = index in (0, 3)
        result.append(
            {
                "item_id": f"item-{index}",
                "label": label,
                "inclusion_probability": 4 / candidate.REMAINING_POPULATION_ROWS,
                "relevance_logit": 2.0 if positive else -2.0,
                "target_presence_logits": [
                    2.0 if positive and target != "residual_other" else -2.0
                    for target in TARGET_CLASSES
                ],
                "stance_logits": [
                    ([2.0, -2.0, -1.0, 0.0] if index == 0 else [0.0, 2.0, -1.0, 1.0])
                    for _ in ANALYTIC_TARGET_CLASSES
                ],
            }
        )
    return result


def _membership() -> list[dict]:
    probability = 4 / candidate.REMAINING_POPULATION_ROWS
    return [
        {
            "opaque_id": f"item-{index}",
            "selection_component": "calibration_probability",
            "selection_stratum": '{"content_type":"comment","subreddit":"x","year":2024}',
            "inclusion_probability_numerator": 4,
            "inclusion_probability_denominator": candidate.REMAINING_POPULATION_ROWS,
            "inclusion_probability": probability,
            "probability_scope": "post-acquisition-eligible-candidate-frame",
        }
        for index in range(4)
    ]


def test_mean_aligned_logits_requires_three_seeds_and_uses_raw_arithmetic_mean() -> None:
    per_seed = [[{"item_id": "a", "relevance_logit": value}] for value in (1.0, 2.0, 6.0)]
    result = calibration.mean_aligned_logits(per_seed, component="relevance")
    assert result == [{"item_id": "a", "relevance_logit": 3.0}]


def test_weighted_calibration_and_validation_mask_noncodable_and_mixed() -> None:
    fitted = calibration.fit_weighted_calibration(
        _rows(),
        bindings={"sample_receipt_id": "a" * 64, "cohort_id": "b" * 64},
    )
    assert fitted["codable_relevance_rows"] == 3
    assert fitted["material_target_rows"] == 2
    assert fitted["excluded_mixed_stance_instances"] == 1
    assert fitted["abstention_margin"] == 0.0
    assert fitted["stance_min_confidence"] == 0.0
    assert set(fitted["stance"]) == set(ANALYTIC_TARGET_CLASSES)

    aggregate = calibration.build_aggregate_validation(_rows(), _membership(), fitted)
    score = aggregate["score"]
    assert score["scientific_aggregate_eligible"] is True
    assert score["reference_masks"] == {
        "codable": 3,
        "not_codable": 1,
        "material": 2,
        "not_material": 1,
    }
    assert score["excluded_reference_stance_states"] == ["mixed"]
    assert score["excluded_reference_stance_instances"] == 1
    assert score["probability_design"]["sample_rows"] == 4
    assert (
        score["probability_design"]["represented_population_rows"]
        == candidate.REMAINING_POPULATION_ROWS
    )


def test_projected_stance_probabilities_assign_zero_mass_to_mixed() -> None:
    fitted = calibration.fit_weighted_calibration(_rows(), bindings={"sample_receipt_id": "a" * 64})
    probabilities = calibration.calibrated_probabilities(_rows()[0], fitted)
    assert all(values[1] == 0.0 for values in probabilities["stance_probabilities"])
    assert all(abs(sum(values) - 1.0) < 1e-12 for values in probabilities["stance_probabilities"])
