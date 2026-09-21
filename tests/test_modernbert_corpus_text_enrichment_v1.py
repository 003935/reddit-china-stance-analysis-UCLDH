from __future__ import annotations

import pytest

from reddit_china_stance import modernbert_corpus_source_enrichment_v2 as parent
from reddit_china_stance import modernbert_corpus_text_enrichment_v1 as enrichment
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256


def test_text_column_is_append_only_and_follows_created_utc() -> None:
    columns = enrichment.enriched_prediction_columns()
    assert columns[:6] == (
        "corpus_position",
        "opaque_id",
        "source_id",
        "created_utc",
        "text",
        "subreddit",
    )
    assert tuple(column for column in columns if column != "text") == (
        parent.enriched_prediction_columns()
    )
    assert enrichment.ADDED_COLUMNS == ("text",)


def test_text_shard_plan_conserves_the_frozen_corpus() -> None:
    plan = enrichment.shard_plan()
    assert len(plan) == enrichment.SHARD_COUNT
    assert plan[0]["start"] == 0
    assert plan[-1]["stop"] == enrichment.CORPUS_ROWS
    assert sum(int(item["row_count"]) for item in plan) == enrichment.CORPUS_ROWS
    assert all(item["stop"] - item["start"] == item["row_count"] for item in plan)


def test_text_enrichment_authority_is_content_addressed_and_private() -> None:
    authority = enrichment.enrichment_authority(
        policy_sha256="a" * 64,
        source_bundle_id="b" * 64,
        canonical_inventory_id="c" * 64,
    )
    body = {key: value for key, value in authority.items() if key != "authority_id"}
    assert authority["authority_id"] == canonical_sha256(body)
    assert authority["added_columns"] == ["text"]
    assert authority["text_source_field"] == "canonical_record.text"
    assert authority["locked_test_rows_accessed"] == 0
    with pytest.raises(ValueError, match="source_bundle_id"):
        enrichment.enrichment_authority(
            policy_sha256="a" * 64,
            source_bundle_id="not-a-digest",
            canonical_inventory_id="c" * 64,
        )
