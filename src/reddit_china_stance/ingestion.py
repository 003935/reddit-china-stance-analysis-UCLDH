"""Pure, locally testable canonicalisation for Reddit archive rows."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from reddit_china_stance.models import CanonicalRecord, ContentType, SubredditStratum

START_UTC = datetime(2020, 1, 1, tzinfo=UTC)
END_UTC_EXCLUSIVE = datetime(2026, 1, 1, tzinfo=UTC)
TEXT_SEPARATOR = "\n\n"
DELETION_MARKERS = {"[deleted]", "[removed]"}
REDDIT_BASE36_ID = re.compile(r"^[0-9a-z]+$")

SUBREDDIT_STRATA: dict[str, SubredditStratum] = {
    "china": SubredditStratum.DOMAIN,
    "chineselanguage": SubredditStratum.DOMAIN,
    "sino": SubredditStratum.DOMAIN,
    "geopolitics": SubredditStratum.NEWS_POLITICAL,
    "news": SubredditStratum.NEWS_POLITICAL,
    "worldnews": SubredditStratum.NEWS_POLITICAL,
    "askreddit": SubredditStratum.GENERAL_INTEREST,
    "funny": SubredditStratum.GENERAL_INTEREST,
    "gaming": SubredditStratum.GENERAL_INTEREST,
    "todayilearned": SubredditStratum.GENERAL_INTEREST,
}


class ExclusionReason(StrEnum):
    OUTSIDE_TIME_WINDOW = "outside_time_window"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    DELETED_MARKER = "deleted_marker"
    REMOVED_MARKER = "removed_marker"
    EMPTY_TEXT = "empty_after_encoding_normalisation"
    SOURCE_FILE_MISMATCH = "source_file_mismatch"


DELETION_MARKER_REASONS = {
    "[deleted]": ExclusionReason.DELETED_MARKER,
    "[removed]": ExclusionReason.REMOVED_MARKER,
}


class SourceRowExcluded(ValueError):
    """Expected row-level exclusion carrying exactly one primary reason."""

    def __init__(self, reason: ExclusionReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def infer_content_type(source_file: str) -> ContentType:
    name = PurePosixPath(source_file).name
    if name.endswith("_submissions.zst"):
        return ContentType.SUBMISSION
    if name.endswith("_comments.zst"):
        return ContentType.COMMENT
    raise ValueError(f"unsupported archive filename: {source_file}")


def infer_subreddit(source_file: str) -> str:
    name = PurePosixPath(source_file).name
    for suffix in ("_submissions.zst", "_comments.zst"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"unsupported archive filename: {source_file}")


def normalise_fullname(
    value: Any, expected_prefix: str, *, allow_either_prefix: bool = False
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceRowExcluded(
            ExclusionReason.MISSING_REQUIRED_FIELD,
            f"missing Reddit ID requiring prefix {expected_prefix}",
        )
    cleaned = value.strip()
    if cleaned.startswith("t1_") or cleaned.startswith("t3_"):
        if allow_either_prefix or cleaned.startswith(expected_prefix):
            payload = cleaned[3:]
            if REDDIT_BASE36_ID.fullmatch(payload):
                return cleaned
            raise SourceRowExcluded(
                ExclusionReason.MISSING_REQUIRED_FIELD,
                f"Reddit fullname {cleaned!r} has an invalid base-36 payload",
            )
        raise SourceRowExcluded(
            ExclusionReason.MISSING_REQUIRED_FIELD,
            f"Reddit fullname {cleaned!r} contradicts expected prefix {expected_prefix}",
        )
    if not REDDIT_BASE36_ID.fullmatch(cleaned):
        raise SourceRowExcluded(
            ExclusionReason.MISSING_REQUIRED_FIELD,
            f"Reddit ID {cleaned!r} is not a base-36 identifier",
        )
    return f"{expected_prefix}{cleaned}"


def parse_created_utc(value: Any) -> datetime:
    if isinstance(value, bool):
        raise SourceRowExcluded(
            ExclusionReason.MISSING_REQUIRED_FIELD, "created_utc cannot be boolean"
        )
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceRowExcluded(
            ExclusionReason.MISSING_REQUIRED_FIELD, "created_utc is not a Unix timestamp"
        ) from exc
    try:
        created = datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise SourceRowExcluded(
            ExclusionReason.MISSING_REQUIRED_FIELD, "created_utc is outside supported range"
        ) from exc
    if not START_UTC <= created < END_UTC_EXCLUSIVE:
        raise SourceRowExcluded(
            ExclusionReason.OUTSIDE_TIME_WINDOW,
            f"timestamp {created.isoformat()} is outside the registered scope",
        )
    return created


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SourceRowExcluded(
            ExclusionReason.MISSING_REQUIRED_FIELD, "text field is not a string"
        )
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_target_text(row: dict[str, Any], content_type: ContentType) -> str:
    if content_type is ContentType.COMMENT:
        body = _normalise_text(row.get("body"))
        if body == "[deleted]":
            raise SourceRowExcluded(ExclusionReason.DELETED_MARKER, "comment is deleted")
        if body == "[removed]":
            raise SourceRowExcluded(ExclusionReason.REMOVED_MARKER, "comment is removed")
        if not body:
            raise SourceRowExcluded(ExclusionReason.EMPTY_TEXT, "comment body is empty")
        return body

    title = _normalise_text(row.get("title"))
    selftext = _normalise_text(row.get("selftext"))
    fields = (title, selftext)
    parts = [part for part in fields if part and part not in DELETION_MARKERS]
    marker_reasons = [DELETION_MARKER_REASONS[part] for part in fields if part in DELETION_MARKERS]
    if not parts:
        if marker_reasons:
            reason = marker_reasons[0]
            state = "deleted" if reason is ExclusionReason.DELETED_MARKER else "removed"
            raise SourceRowExcluded(reason, f"submission is {state}")
        raise SourceRowExcluded(ExclusionReason.EMPTY_TEXT, "submission title and body are empty")
    return TEXT_SEPARATOR.join(parts)


def normalise_source_row(row: dict[str, Any], *, source_file: str) -> dict[str, Any]:
    """Normalise one in-scope row before candidate-scoped language classification."""

    content_type = infer_content_type(source_file)
    expected_subreddit = infer_subreddit(source_file)
    subreddit = row.get("subreddit")
    if not isinstance(subreddit, str) or not subreddit.strip():
        raise SourceRowExcluded(ExclusionReason.MISSING_REQUIRED_FIELD, "subreddit is missing")
    subreddit = subreddit.strip()
    if subreddit.casefold() != expected_subreddit.casefold():
        raise SourceRowExcluded(
            ExclusionReason.SOURCE_FILE_MISMATCH,
            f"row subreddit {subreddit!r} does not match archive {expected_subreddit!r}",
        )
    try:
        stratum = SUBREDDIT_STRATA[subreddit.casefold()]
    except KeyError as exc:
        raise SourceRowExcluded(
            ExclusionReason.SOURCE_FILE_MISMATCH,
            f"subreddit {subreddit!r} has no registered stratum",
        ) from exc

    created = parse_created_utc(row.get("created_utc"))
    text = extract_target_text(row, content_type)

    prefix = "t3_" if content_type is ContentType.SUBMISSION else "t1_"
    record_id = normalise_fullname(row.get("id"), prefix)
    if content_type is ContentType.SUBMISSION:
        submission_id = record_id
        parent_id = None
    else:
        submission_id = normalise_fullname(row.get("link_id"), "t3_")
        parent_id = normalise_fullname(row.get("parent_id"), "t1_", allow_either_prefix=True)

    record = CanonicalRecord(
        schema_version="1.0.2",
        record_id=record_id,
        source_id=record_id,
        content_type=content_type,
        subreddit=subreddit,
        stratum=stratum,
        created_utc=created,
        year=created.year,
        month=created.month,
        language_status="unclassified",
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        submission_id=submission_id,
        parent_id=parent_id,
        source_file=PurePosixPath(source_file).name,
    )
    return record.model_dump(mode="python")
