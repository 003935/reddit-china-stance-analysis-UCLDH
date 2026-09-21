from __future__ import annotations

from decimal import Decimal

import pytest

from reddit_china_stance import modal_modernbert_acquisition_benchmark as benchmark
from reddit_china_stance.privacy import assert_metadata_only


def test_evenly_spaced_indices_are_unique_and_bounded() -> None:
    observed = benchmark.evenly_spaced_indices(population_rows=101, sample_rows=10)
    assert observed == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert len(observed) == len(set(observed))


def test_evenly_spaced_indices_reject_oversized_sample() -> None:
    with pytest.raises(ValueError, match="must fit"):
        benchmark.evenly_spaced_indices(population_rows=9, sample_rows=10)


def test_benchmark_cost_and_confirmation_are_exact() -> None:
    run_id = "a" * 64
    assert benchmark.validate_approval("1") == Decimal("1")
    assert benchmark.confirmation_token(run_id) == (
        "BENCHMARK_MODERNBERT_ACQUISITION_SCORING_aaaaaaaaaaaa"
    )
    with pytest.raises(RuntimeError, match="exact \\$1"):
        benchmark.validate_approval("0.99")


def test_benchmark_measurement_shape_is_metadata_only() -> None:
    assert_metadata_only(
        {
            "measurements": [
                {
                    "batch_size": 8,
                    "row_count": 10_000,
                    "wall_seconds": 1.0,
                    "rows_per_second": 10_000.0,
                    "peak_cuda_memory_bytes": 1,
                }
            ]
        },
        where="benchmark-test",
    )
