from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from reddit_china_stance.ingestion import (
    ExclusionReason,
    SourceRowExcluded,
    extract_target_text,
    infer_content_type,
    normalise_source_row,
)
from reddit_china_stance.models import ContentType

ROOT = Path(__file__).parents[1]


def test_source_manifest_matches_pinned_dataset_totals() -> None:
    manifest = json.loads((ROOT / "configs" / "source-files.json").read_text())
    dataset = tomllib.loads((ROOT / "configs" / "dataset.toml").read_text())["dataset"]
    assert len(manifest["files"]) == 20
    assert len({item["path"] for item in manifest["files"]}) == 20
    assert sum(item["size"] for item in manifest["files"]) == manifest["compressed_bytes"]
    assert manifest["compressed_bytes"] == 169_284_685_552
    assert manifest["dataset_id"] == dataset["dataset_id"]
    assert manifest["revision"] == dataset["revision"]
    assert manifest["compressed_bytes"] == dataset["compressed_bytes"]


def test_infer_content_type_fails_on_unknown_filename() -> None:
    assert infer_content_type("China_comments.zst") is ContentType.COMMENT
    assert infer_content_type("China_submissions.zst") is ContentType.SUBMISSION
    with pytest.raises(ValueError, match="unsupported archive filename"):
        infer_content_type("China.jsonl")


def test_normalises_comment_with_string_timestamp_and_fullnames() -> None:
    row = {
        "id": "abc123",
        "created_utc": "1680697800",
        "subreddit": "ChineseLanguage",
        "body": "A synthetic English fixture.",
        "link_id": "t3_submission",
        "parent_id": "t1_parent",
    }
    record = normalise_source_row(row, source_file="ChineseLanguage_comments.zst")
    assert record["record_id"] == "t1_abc123"
    assert record["submission_id"] == "t3_submission"
    assert record["parent_id"] == "t1_parent"
    assert record["year"] == 2023
    assert record["text_sha256"] != "0" * 64


def test_normalises_submission_and_ignores_removed_selftext() -> None:
    row = {
        "id": "submission",
        "created_utc": 1680697800,
        "subreddit": "geopolitics",
        "title": "Synthetic title",
        "selftext": "[removed]",
    }
    record = normalise_source_row(row, source_file="geopolitics_submissions.zst")
    assert record["record_id"] == "t3_submission"
    assert record["submission_id"] == record["record_id"]
    assert record["parent_id"] is None
    assert record["text"] == "Synthetic title"


@pytest.mark.parametrize(
    ("marker", "reason"),
    [
        ("[deleted]", ExclusionReason.DELETED_MARKER),
        ("[removed]", ExclusionReason.REMOVED_MARKER),
    ],
)
def test_submission_title_only_sentinel_is_explicitly_excluded(
    marker: str, reason: ExclusionReason
) -> None:
    with pytest.raises(SourceRowExcluded) as captured:
        extract_target_text({"title": marker, "selftext": ""}, ContentType.SUBMISSION)
    assert captured.value.reason is reason


@pytest.mark.parametrize(
    ("title", "selftext", "reason"),
    [
        ("[deleted]", "[deleted]", ExclusionReason.DELETED_MARKER),
        ("[removed]", "[removed]", ExclusionReason.REMOVED_MARKER),
        ("[deleted]", "[removed]", ExclusionReason.DELETED_MARKER),
        ("[removed]", "[deleted]", ExclusionReason.REMOVED_MARKER),
    ],
)
def test_submission_with_only_sentinels_is_explicitly_excluded(
    title: str, selftext: str, reason: ExclusionReason
) -> None:
    with pytest.raises(SourceRowExcluded) as captured:
        extract_target_text({"title": title, "selftext": selftext}, ContentType.SUBMISSION)
    assert captured.value.reason is reason


@pytest.mark.parametrize("marker", ["[deleted]", "[removed]"])
def test_submission_title_sentinel_is_dropped_when_body_is_substantive(
    marker: str,
) -> None:
    text = extract_target_text(
        {"title": marker, "selftext": "Synthetic body"}, ContentType.SUBMISSION
    )
    assert text == "Synthetic body"


def test_normalised_source_record_defers_language_and_drops_author() -> None:
    row = {
        "id": "submission",
        "created_utc": 1680697800,
        "subreddit": "geopolitics",
        "title": "Synthetic title",
        "selftext": "Synthetic body",
        "author": "must-not-survive",
    }

    record = normalise_source_row(row, source_file="geopolitics_submissions.zst")

    assert record["record_id"] == "t3_submission"
    assert record["schema_version"] == "1.0.2"
    assert record["language_status"] == "unclassified"
    assert record["year"] == 2023
    assert "author" not in record


def test_fullname_prefixes_are_strict_except_for_parent_ids() -> None:
    row = {
        "id": "t3_wrong-kind",
        "created_utc": 1680697800,
        "subreddit": "China",
        "body": "Synthetic body",
        "link_id": "t3_submission",
        "parent_id": "t3_submission",
    }
    with pytest.raises(SourceRowExcluded, match="contradicts expected prefix"):
        normalise_source_row(row, source_file="China_comments.zst")

    row["id"] = "t1_comment"
    record = normalise_source_row(row, source_file="China_comments.zst")
    assert record["parent_id"] == "t3_submission"


@pytest.mark.parametrize("bad_id", ["t2_user", "bad id", "t1_bad-id", "ABC123"])
def test_malformed_reddit_ids_are_excluded(bad_id: str) -> None:
    row = {
        "id": bad_id,
        "created_utc": 1680697800,
        "subreddit": "China",
        "body": "Synthetic body",
        "link_id": "t3_submission",
        "parent_id": "t3_submission",
    }
    with pytest.raises(SourceRowExcluded, match="base-36"):
        normalise_source_row(row, source_file="China_comments.zst")


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("[deleted]", ExclusionReason.DELETED_MARKER),
        ("[removed]", ExclusionReason.REMOVED_MARKER),
        ("", ExclusionReason.EMPTY_TEXT),
    ],
)
def test_comment_exclusions_are_explicit(body: str, reason: ExclusionReason) -> None:
    with pytest.raises(SourceRowExcluded) as captured:
        extract_target_text({"body": body}, ContentType.COMMENT)
    assert captured.value.reason is reason


def test_out_of_scope_rows_fail_explicitly() -> None:
    row = {
        "id": "abc123",
        "created_utc": 1514764800,
        "subreddit": "China",
        "body": "A synthetic fixture.",
        "link_id": "t3_submission",
        "parent_id": "t3_submission",
    }
    with pytest.raises(SourceRowExcluded) as captured:
        normalise_source_row(row, source_file="China_comments.zst")
    assert captured.value.reason is ExclusionReason.OUTSIDE_TIME_WINDOW
