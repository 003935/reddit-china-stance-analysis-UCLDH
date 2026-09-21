"""Pure contracts for the private corpus source-ID/timestamp enrichment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

POLICY_PATH = Path("configs/modernbert-corpus-source-enrichment-v2.toml")
DATASET_REVISION = "97627c42893bc479c6a4952b1db38bbc33a60ac8"
SOURCE_SCHEMA_VERSION = "1.0.2"
CANONICAL_ROWS = 548_963_310
INFERENCE_AUTHORITY_ID = "d755a6b4268adde78417b718c471670a987e48feb82f5a685cb876b97e0ee2d7"
INFERENCE_FINAL_RECEIPT_ID = "ce5117e89be73dcd613db8ff09850b379c984d2fa9c2d560a9cdb32583d97bea"
INFERENCE_SOURCE_BUNDLE_ID = "85e0fa6d2fb156fc0091928c20194758030ec863bbb85a77e85c63341703531e"
HF_REPO_ID = "aisafteycommons/reddit-china-stance-modernbert-corpus-v1"
HF_PREVIOUS_REVISION = "61c2498311e719795aa345eedd6d6dbdae239b00"
HF_PREVIOUS_EXPORT_ID = "432fdfa56b7f81d3b52c1d56f4247ed1f23409bde8a7a2b9d8a61cf3dae5c2c1"
HF_PREVIOUS_MANIFEST_SHA256 = "e44d8c7a41bff1a8c6aa67495029ac7baffa8ef5efa0aff46f60a5ab954de84d"
CONFIRMATION = "ADD_SOURCE_ID_AND_CREATED_UTC_TO_PRIVATE_HF_CORPUS"
ADDED_COLUMNS = ("source_id", "created_utc")
TIMESTAMP_UNIT = "us"
TIMESTAMP_TIMEZONE = "UTC"
CORPUS_ROWS = 908_141
SHARD_COUNT = 120
CALIBRATION_ROWS = 600
YEARS = tuple(range(2020, 2026))
PRIVATE_PREDICTION_COLUMNS = (
    "corpus_position",
    "opaque_id",
    "subreddit",
    "year",
    "content_type",
    "retrieval_mode",
    "calibration_member",
    "relevance_probability",
    "relevance",
    "target_china_general_probability",
    "target_china_general_present",
    "stance_china_general_negative_probability",
    "stance_china_general_no_directed_stance_probability",
    "stance_china_general_positive_probability",
    "stance_china_general",
    "target_government_ccp_probability",
    "target_government_ccp_present",
    "stance_government_ccp_negative_probability",
    "stance_government_ccp_no_directed_stance_probability",
    "stance_government_ccp_positive_probability",
    "stance_government_ccp",
    "target_people_identity_probability",
    "target_people_identity_present",
    "stance_people_identity_negative_probability",
    "stance_people_identity_no_directed_stance_probability",
    "stance_people_identity_positive_probability",
    "stance_people_identity",
    "target_culture_media_probability",
    "target_culture_media_present",
    "stance_culture_media_negative_probability",
    "stance_culture_media_no_directed_stance_probability",
    "stance_culture_media_positive_probability",
    "stance_culture_media",
    "target_company_tech_product_probability",
    "target_company_tech_product_present",
    "stance_company_tech_product_negative_probability",
    "stance_company_tech_product_no_directed_stance_probability",
    "stance_company_tech_product_positive_probability",
    "stance_company_tech_product",
    "target_residual_other_probability",
    "target_residual_other_present",
)


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
    """Return the exact contiguous predecessor shard plan."""

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


def enriched_prediction_columns() -> tuple[str, ...]:
    """Insert canonical source fields while preserving the complete v1 projection."""

    base = list(PRIVATE_PREDICTION_COLUMNS)
    index = base.index("opaque_id") + 1
    base[index:index] = ADDED_COLUMNS
    return tuple(base)


def validate_created_utc(value: Any, *, expected_year: int) -> datetime:
    """Require a timezone-aware canonical UTC timestamp matching the stored year."""

    if not isinstance(value, datetime):
        raise ValueError("created_utc must be a datetime")
    if value.utcoffset() != timedelta(0) or value.tzinfo is None:
        raise ValueError("created_utc must use UTC")
    if value.year != expected_year:
        raise ValueError("created_utc year differs from the prediction year")
    return value


def enrichment_authority(
    *, policy_sha256: str, source_bundle_id: str, canonical_inventory_id: str
) -> dict[str, Any]:
    """Freeze one exact enrichment authority from code, policy and source inventory."""

    for name, value in {
        "policy_sha256": policy_sha256,
        "source_bundle_id": source_bundle_id,
        "canonical_inventory_id": canonical_inventory_id,
    }.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest")
    body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-source-enrichment-authority-v2",
        "dataset_revision": DATASET_REVISION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "canonical_rows": CANONICAL_ROWS,
        "corpus_rows": CORPUS_ROWS,
        "shard_count": SHARD_COUNT,
        "inference_authority_id": INFERENCE_AUTHORITY_ID,
        "inference_final_receipt_id": INFERENCE_FINAL_RECEIPT_ID,
        "inference_source_bundle_id": INFERENCE_SOURCE_BUNDLE_ID,
        "hf_repo_id": HF_REPO_ID,
        "hf_previous_revision": HF_PREVIOUS_REVISION,
        "hf_previous_export_id": HF_PREVIOUS_EXPORT_ID,
        "hf_previous_manifest_sha256": HF_PREVIOUS_MANIFEST_SHA256,
        "added_columns": list(ADDED_COLUMNS),
        "preserved_columns": list(PRIVATE_PREDICTION_COLUMNS),
        "policy_sha256": policy_sha256,
        "source_bundle_id": source_bundle_id,
        "canonical_inventory_id": canonical_inventory_id,
        "locked_test_rows_accessed": 0,
    }
    return {**body, "authority_id": canonical_sha256(body)}
