"""Pure contracts for adding the canonical Reddit text field to the private corpus export.

This is a new, append-only authority.  The preceding source-ID/timestamp enrichment remains
unchanged and is treated as the immutable parent for this text-bearing revision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reddit_china_stance import modernbert_corpus_source_enrichment_v2 as parent

POLICY_PATH = Path("configs/modernbert-corpus-text-enrichment-v1.toml")
DATASET_REVISION = parent.DATASET_REVISION
SOURCE_SCHEMA_VERSION = parent.SOURCE_SCHEMA_VERSION
CANONICAL_ROWS = parent.CANONICAL_ROWS
CORPUS_ROWS = parent.CORPUS_ROWS
SHARD_COUNT = parent.SHARD_COUNT
CALIBRATION_ROWS = parent.CALIBRATION_ROWS
YEARS = parent.YEARS

HF_REPO_ID = "aisafteycommons/reddit-china-stance-modernbert-corpus-v1"
HF_PREVIOUS_REVISION = "6e3466ccc56c1a846e30375c8b99924f6efa5723"
HF_PREVIOUS_EXPORT_ID = "60d0da5c61ad075a9f71458addb3761d42a315514b589deaec27297096f067af"
HF_PREVIOUS_MANIFEST_SHA256 = "a3d531c8f8719335cc35d862f0a6e1f712ad0b43e44ac325243ef88f2bbc13ac"

PARENT_ENRICHMENT_AUTHORITY_ID = "10707280a32795edac03d341cee5b2971d045d477f941b382f05ae5e88d470f9"
PARENT_ENRICHMENT_RECEIPT_ID = "084c84a7c16b00d74819c647bd4ec9fdf4fef9e0a5db9b36ad8b9321e5c2ead6"
PARENT_ENRICHMENT_SOURCE_BUNDLE_ID = (
    "563b71779f1ed7db0f9bf8caea04ad4fcb9f7a1c9a2710a5d21ff33bd474cbcb"
)
PARENT_CANONICAL_INVENTORY_ID = "5853e4ce4c61ce0995328912e23c77300510e2186aed6096f430e6f28f3439b2"

CONFIRMATION = "ADD_CANONICAL_TEXT_TO_PRIVATE_HF_CORPUS"
ADDED_COLUMNS = ("text",)
TIMESTAMP_UNIT = parent.TIMESTAMP_UNIT
TIMESTAMP_TIMEZONE = parent.TIMESTAMP_TIMEZONE


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def shard_plan() -> list[dict[str, int | str]]:
    base, remainder = divmod(CORPUS_ROWS, SHARD_COUNT)
    result = []
    start = 0
    for index in range(SHARD_COUNT):
        rows = base + (1 if index < remainder else 0)
        result.append(
            {"shard_id": f"{index:03d}", "start": start, "stop": start + rows, "row_count": rows}
        )
        start += rows
    return result


def parent_prediction_columns() -> tuple[str, ...]:
    return parent.enriched_prediction_columns()


def enriched_prediction_columns() -> tuple[str, ...]:
    """Append text immediately after the existing canonical source fields."""

    columns = list(parent_prediction_columns())
    columns.insert(columns.index("created_utc") + 1, "text")
    return tuple(columns)


def enrichment_authority(
    *, policy_sha256: str, source_bundle_id: str, canonical_inventory_id: str
) -> dict[str, Any]:
    for name, value in {
        "policy_sha256": policy_sha256,
        "source_bundle_id": source_bundle_id,
        "canonical_inventory_id": canonical_inventory_id,
    }.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest")
    body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-text-enrichment-authority-v1",
        "dataset_revision": DATASET_REVISION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "canonical_rows": CANONICAL_ROWS,
        "corpus_rows": CORPUS_ROWS,
        "shard_count": SHARD_COUNT,
        "calibration_member_rows": CALIBRATION_ROWS,
        "hf_repo_id": HF_REPO_ID,
        "hf_previous_revision": HF_PREVIOUS_REVISION,
        "hf_previous_export_id": HF_PREVIOUS_EXPORT_ID,
        "hf_previous_manifest_sha256": HF_PREVIOUS_MANIFEST_SHA256,
        "parent_enrichment_authority_id": PARENT_ENRICHMENT_AUTHORITY_ID,
        "parent_enrichment_receipt_id": PARENT_ENRICHMENT_RECEIPT_ID,
        "parent_enrichment_source_bundle_id": PARENT_ENRICHMENT_SOURCE_BUNDLE_ID,
        "parent_canonical_inventory_id": PARENT_CANONICAL_INVENTORY_ID,
        "added_columns": list(ADDED_COLUMNS),
        "preserved_columns": list(parent_prediction_columns()),
        "text_source_field": "canonical_record.text",
        "text_hash_field": "canonical_record.text_sha256",
        "text_join_key": "canonical_record.record_id = parent.source_id",
        "policy_sha256": policy_sha256,
        "source_bundle_id": source_bundle_id,
        "canonical_inventory_id": canonical_inventory_id,
        "locked_test_rows_accessed": 0,
    }
    return {**body, "authority_id": canonical_sha256(body)}
