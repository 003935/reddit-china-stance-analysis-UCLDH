"""Strict canonical-corpus contracts shared by ingestion and validation."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    """Base contract: reject unknown fields instead of silently discarding them."""

    model_config = ConfigDict(extra="forbid")


class ContentType(StrEnum):
    SUBMISSION = "submission"
    COMMENT = "comment"


class SubredditStratum(StrEnum):
    DOMAIN = "domain"
    NEWS_POLITICAL = "news_political"
    GENERAL_INTEREST = "general_interest"


class CanonicalRecord(StrictModel):
    """One in-scope Reddit row before candidate-scoped language classification."""

    schema_version: Literal["1.0.2"]
    record_id: NonEmptyString
    source_id: NonEmptyString
    content_type: ContentType
    subreddit: NonEmptyString
    stratum: SubredditStratum
    created_utc: AwareDatetime
    year: Annotated[int, Field(ge=2020, le=2025)]
    month: Annotated[int, Field(ge=1, le=12)]
    language_status: Literal["unclassified"]
    text: NonEmptyString
    text_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    submission_id: NonEmptyString
    parent_id: str | None = None
    source_file: NonEmptyString

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> CanonicalRecord:
        timestamp: datetime = self.created_utc
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("created_utc must use a UTC offset")
        if (timestamp.year, timestamp.month) != (self.year, self.month):
            raise ValueError("year/month must match created_utc")
        if self.content_type is ContentType.SUBMISSION:
            if self.submission_id != self.source_id:
                raise ValueError("submission records must use their source_id as submission_id")
            if self.parent_id is not None:
                raise ValueError("submission records cannot have parent_id")
        elif self.parent_id is None:
            raise ValueError("comment records require parent_id")
        return self
