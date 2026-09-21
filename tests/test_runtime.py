from __future__ import annotations

import pytest

from reddit_china_stance.runtime import estimate_runtime


def test_runtime_arithmetic_matches_documented_formula() -> None:
    result = estimate_runtime(
        rows=10_000,
        prompt_tokens=650,
        output_tokens=45,
        prefill_tokens_per_second=6_000,
        decode_tokens_per_second=400,
        overhead_fraction=0.25,
    )
    expected_seconds = 10_000 * (650 / 6_000 + 45 / 400) * 1.25
    assert result.seconds == pytest.approx(expected_seconds)
    assert result.hours == pytest.approx(0.7667824074)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"prompt_tokens": 0},
        {"output_tokens": -1},
        {"prefill_tokens_per_second": 0},
        {"decode_tokens_per_second": 0},
        {"overhead_fraction": -0.01},
        {"prompt_tokens": float("nan")},
        {"decode_tokens_per_second": float("inf")},
        {"overhead_fraction": float("nan")},
    ],
)
def test_runtime_rejects_invalid_inputs(kwargs: dict[str, float]) -> None:
    inputs = {
        "rows": 10,
        "prompt_tokens": 650,
        "output_tokens": 45,
        "prefill_tokens_per_second": 6_000,
        "decode_tokens_per_second": 400,
        "overhead_fraction": 0.25,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        estimate_runtime(**inputs)
