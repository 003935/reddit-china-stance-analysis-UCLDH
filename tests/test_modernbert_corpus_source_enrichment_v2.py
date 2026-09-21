from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from reddit_china_stance import modernbert_corpus_source_enrichment_v2 as enrichment
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance import modernbert_probability_random_corpus_inference_v1 as corpus
from reddit_china_stance import reconciliation
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256


def test_enriched_columns_preserve_v1_projection() -> None:
    columns = enrichment.enriched_prediction_columns()
    assert columns[0:5] == (
        "corpus_position",
        "opaque_id",
        "source_id",
        "created_utc",
        "subreddit",
    )
    assert tuple(item for item in columns if item not in enrichment.ADDED_COLUMNS) == (
        corpus.private_prediction_columns()
    )


def test_frozen_corpus_contract_matches_authoritative_inference_module() -> None:
    assert enrichment.CORPUS_ROWS == corpus.CORPUS_ROWS
    assert enrichment.SHARD_COUNT == corpus.SHARD_COUNT
    assert enrichment.CALIBRATION_ROWS == candidate.CALIBRATION_ROWS
    assert set(enrichment.YEARS) == reconciliation.EXPECTED_YEARS
    assert corpus.private_prediction_columns() == enrichment.PRIVATE_PREDICTION_COLUMNS
    assert enrichment.shard_plan() == corpus.shard_plan()


def test_created_utc_requires_utc_and_exact_year() -> None:
    value = datetime(2022, 4, 2, 12, 30, tzinfo=UTC)
    assert enrichment.validate_created_utc(value, expected_year=2022) is value
    with pytest.raises(ValueError, match="must use UTC"):
        enrichment.validate_created_utc(
            datetime(2022, 4, 2, tzinfo=timezone(timedelta(hours=1))),
            expected_year=2022,
        )
    with pytest.raises(ValueError, match="differs"):
        enrichment.validate_created_utc(value, expected_year=2021)
    with pytest.raises(ValueError, match="must be a datetime"):
        enrichment.validate_created_utc("2022-04-02T12:30:00Z", expected_year=2022)


def test_enrichment_authority_is_exactly_content_addressed() -> None:
    authority = enrichment.enrichment_authority(
        policy_sha256="a" * 64,
        source_bundle_id="b" * 64,
        canonical_inventory_id="c" * 64,
    )
    body = {key: value for key, value in authority.items() if key != "authority_id"}
    assert authority["authority_id"] == canonical_sha256(body)
    assert authority["locked_test_rows_accessed"] == 0
    with pytest.raises(ValueError, match="policy_sha256"):
        enrichment.enrichment_authority(
            policy_sha256="bad", source_bundle_id="b" * 64, canonical_inventory_id="c" * 64
        )
