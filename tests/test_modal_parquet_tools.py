from __future__ import annotations

import pytest

from reddit_china_stance.modal_parquet_batch import EXPECTED_SOURCE_FILES, _parse_sources
from reddit_china_stance.modal_parquet_benchmark import _parse_cpu_sequence


def test_durable_batch_requires_exact_pending_sources_and_preserves_order() -> None:
    ordered = [
        "AskReddit_comments.zst",
        "worldnews_comments.zst",
        "news_comments.zst",
        "funny_comments.zst",
        "gaming_comments.zst",
    ]

    assert _parse_sources(",".join(ordered)) == ordered
    assert set(ordered) == EXPECTED_SOURCE_FILES


def test_durable_batch_rejects_partial_or_duplicate_sources() -> None:
    with pytest.raises(ValueError, match="exactly the five pending sources"):
        _parse_sources("AskReddit_comments.zst,worldnews_comments.zst")
    with pytest.raises(ValueError, match="must be unique"):
        _parse_sources("AskReddit_comments.zst,AskReddit_comments.zst")


def test_benchmark_cpu_sequence_requires_all_tiers_and_preserves_repetitions() -> None:
    assert _parse_cpu_sequence("2,4,8,8,4,2") == [2.0, 4.0, 8.0, 8.0, 4.0, 2.0]
    with pytest.raises(ValueError, match="cover every allocation"):
        _parse_cpu_sequence("2,2")
    with pytest.raises(ValueError, match="use only"):
        _parse_cpu_sequence("2,4,16")
