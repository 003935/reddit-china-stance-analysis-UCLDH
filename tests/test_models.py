from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from reddit_china_stance.models import CanonicalRecord

ROOT = Path(__file__).parents[1]


def valid_canonical_record() -> dict[str, object]:
    return {
        "schema_version": "1.0.2",
        "record_id": "t1_example",
        "source_id": "t1_example",
        "content_type": "comment",
        "subreddit": "example",
        "stratum": "general_interest",
        "created_utc": "2023-04-05T12:30:00Z",
        "year": 2023,
        "month": 4,
        "language_status": "unclassified",
        "text": "A synthetic contract-test sentence.",
        "text_sha256": "a" * 64,
        "submission_id": "t3_parent",
        "parent_id": "t3_parent",
        "source_file": "example_comments.zst",
    }


def test_json_schema_accepts_valid_canonical_record() -> None:
    schema = json.loads((ROOT / "schemas" / "canonical-record.schema.json").read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert list(validator.iter_errors(valid_canonical_record())) == []


def test_model_accepts_valid_canonical_record() -> None:
    assert CanonicalRecord.model_validate(valid_canonical_record()).year == 2023


def test_canonical_model_rejects_timestamp_partition_mismatch() -> None:
    payload = valid_canonical_record()
    payload["month"] = 5
    with pytest.raises(ValidationError, match="year/month must match"):
        CanonicalRecord.model_validate(payload)


def test_canonical_model_rejects_non_utc_offset() -> None:
    payload = valid_canonical_record()
    payload["created_utc"] = "2023-04-05T12:30:00+01:00"
    with pytest.raises(ValidationError, match="must use a UTC offset"):
        CanonicalRecord.model_validate(payload)
