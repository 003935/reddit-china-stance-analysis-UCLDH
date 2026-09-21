from __future__ import annotations

from decimal import Decimal
from itertools import pairwise

import pytest

from reddit_china_stance import modernbert_probability_random_corpus_inference_v1 as corpus
from reddit_china_stance.modernbert_factorised_data import ANALYTIC_TARGET_CLASSES, TARGET_CLASSES


def _fitted() -> dict:
    return {
        "relevance": {"temperature": 1.0, "threshold": 0.5},
        "target_presence": {
            target: {"temperature": 1.0, "threshold": 0.5} for target in TARGET_CLASSES
        },
        "stance": {target: {"temperature": 1.0} for target in ANALYTIC_TARGET_CLASSES},
    }


def _logits(*, relevant: bool = True) -> dict:
    return {
        "relevance_logit": 2.0 if relevant else -2.0,
        "target_presence_logits": [2.0] + [-2.0] * (len(TARGET_CLASSES) - 1),
        "stance_logits": [[3.0, 20.0, 1.0, -1.0] for _ in ANALYTIC_TARGET_CLASSES],
    }


def test_shard_plan_exactly_covers_the_post_acquisition_corpus() -> None:
    plan = corpus.shard_plan()
    assert len(plan) == 120
    assert plan[0] == {"shard_id": "000", "start": 0, "stop": 7568, "row_count": 7568}
    assert plan[-1] == {
        "shard_id": "119",
        "start": 900574,
        "stop": 908141,
        "row_count": 7567,
    }
    assert all(left["stop"] == right["start"] for left, right in pairwise(plan))


def test_compute_projection_is_below_both_registered_caps() -> None:
    projection = corpus.compute_projection()
    assert projection["projected_cost_usd"] == "15.986618"
    assert projection["conservative_cost_usd"] == "23.979928"
    assert projection["budgeted_base_attempt_cost_usd"] == "19.980000"
    assert projection["maximum_platform_execution_attempts_per_call"] == 2
    assert projection["queue_bounded_authorised_execution_cost_usd"] == "22.977000"
    assert projection["queue_bounded_cumulative_cost_projection_usd"] == "25.274870"
    authority = {
        **projection,
        "gpu_type": "L4",
        "gpu_fallback_allowed": False,
        "retry_authorised": False,
        "maximum_locked_test_rows": 0,
        "maximum_corpus_rows": corpus.CORPUS_ROWS,
        "full_phase_operational_monitoring_target_usd": "25.00",
        "cumulative_operational_monitoring_target_usd": "30.00",
        "provider_billing_hard_cap_enforced": False,
        "proceed_without_provider_hard_cap_authorised": True,
    }
    assert corpus.validate_compute_authority(authority) == authority
    with pytest.raises(RuntimeError, match="exceeds"):
        corpus.validate_compute_authority(
            {
                **authority,
                "full_phase_operational_monitoring_target_usd": str(Decimal("20.00")),
            }
        )


def test_private_prediction_is_relevance_gated_and_excludes_mixed() -> None:
    row = corpus.build_private_prediction_row(
        corpus_position=7,
        opaque_id="private-item",
        subreddit="China",
        year=2024,
        content_type="comment",
        retrieval_mode="target",
        calibration_member=True,
        logits=_logits(relevant=True),
        fitted_calibration=_fitted(),
    )
    assert row["relevance"] == "material"
    assert row["target_china_general_present"] is True
    assert row["stance_china_general"] == "negative"
    assert "stance_china_general_mixed_probability" not in row
    assert corpus.validate_private_prediction_rows([row], start=7, stop=8) == [row]

    irrelevant = corpus.build_private_prediction_row(
        corpus_position=8,
        opaque_id="private-item-2",
        subreddit="China",
        year=2024,
        content_type="comment",
        retrieval_mode="target",
        calibration_member=False,
        logits=_logits(relevant=False),
        fitted_calibration=_fitted(),
    )
    assert irrelevant["target_china_general_present"] is False
    assert irrelevant["stance_china_general"] is None


def test_prediction_validator_rejects_position_gaps_and_mixed() -> None:
    row = corpus.build_private_prediction_row(
        corpus_position=0,
        opaque_id="private-item",
        subreddit="China",
        year=2024,
        content_type="comment",
        retrieval_mode="target",
        calibration_member=False,
        logits=_logits(),
        fitted_calibration=_fitted(),
    )
    with pytest.raises(ValueError, match="positions"):
        corpus.validate_private_prediction_rows([{**row, "corpus_position": 1}], start=0, stop=1)
    with pytest.raises(ValueError, match="three-class"):
        corpus.validate_private_prediction_rows(
            [{**row, "stance_china_general": "mixed"}], start=0, stop=1
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_prediction_validator_rejects_non_finite_or_out_of_range_probabilities(
    bad: float,
) -> None:
    row = corpus.build_private_prediction_row(
        corpus_position=0,
        opaque_id="private-item",
        subreddit="China",
        year=2024,
        content_type="comment",
        retrieval_mode="target",
        calibration_member=False,
        logits=_logits(),
        fitted_calibration=_fitted(),
    )
    for key in (
        "relevance_probability",
        "target_china_general_probability",
        "stance_china_general_negative_probability",
    ):
        with pytest.raises(ValueError, match="probability"):
            corpus.validate_private_prediction_rows([{**row, key: bad}], start=0, stop=1)
