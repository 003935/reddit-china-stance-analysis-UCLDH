from __future__ import annotations

import pytest

from reddit_china_stance.corpus_analysis_v1 import (
    _sum_contributions,
    canonical_sha256,
    linear_trend,
    standardise_cells,
)


def _cell(year: int, subreddit: str, weight: float, score: float) -> dict[str, object]:
    return {
        "year": year,
        "subreddit": subreddit,
        "content_type": "comment",
        "target": "china_general",
        "target_weight": weight,
        "score": score,
    }


def test_standardisation_and_symmetric_decomposition_close_exactly() -> None:
    rows = [
        _cell(2022, "A", 0.75, -0.4),
        _cell(2022, "B", 0.25, 0.0),
        _cell(2025, "A", 0.25, -0.2),
        _cell(2025, "B", 0.75, 0.2),
    ]

    standardised, decomposition, contributions = standardise_cells(rows)

    assert [row["observed_score"] for row in standardised] == pytest.approx([-0.3, 0.1])
    assert decomposition["total_change"] == pytest.approx(0.4)
    assert decomposition["within_change"] == pytest.approx(0.2)
    assert decomposition["composition_change"] == pytest.approx(0.2)
    assert decomposition["within_share"] == pytest.approx(0.5)
    assert decomposition["composition_share"] == pytest.approx(0.5)
    assert sum(float(row["total_contribution"]) for row in contributions) == pytest.approx(
        decomposition["total_change"]
    )


def test_contribution_aggregation_preserves_components() -> None:
    rows = [
        {
            "target": "china_general",
            "within_contribution": 0.1,
            "composition_contribution": -0.02,
        },
        {
            "target": "china_general",
            "within_contribution": 0.03,
            "composition_contribution": 0.01,
        },
    ]

    result = _sum_contributions(rows, "target")

    assert result == [
        {
            "target": "china_general",
            "within_contribution": pytest.approx(0.13),
            "composition_contribution": pytest.approx(-0.01),
            "total_contribution": pytest.approx(0.12),
        }
    ]


def test_linear_trend_reports_monthly_and_annual_scales() -> None:
    result = linear_trend([{"score": 0.0}, {"score": 1.0}, {"score": 2.0}])

    assert result == {
        "slope_per_month": pytest.approx(1.0),
        "slope_per_year": pytest.approx(12.0),
        "r_squared": pytest.approx(1.0),
    }


def test_canonical_hash_is_mapping_order_independent() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
