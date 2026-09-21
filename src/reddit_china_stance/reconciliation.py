"""Pure validation and reconciliation for canonical Parquet receipts."""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reddit_china_stance.source_manifest import (
    EXPECTED_ARCHIVE_COUNT,
    EXPECTED_COMPRESSED_BYTES,
)

RECEIPT_SCHEMA_VERSION = "1.0.0"
SOURCE_SCHEMA_VERSION = "1.0.2"
REPORT_SCHEMA_VERSION = "1.0.0"
EXPECTED_YEARS = set(range(2020, 2026))
SOURCE_MANIFEST_KEYS = {
    "schema_version",
    "dataset_id",
    "revision",
    "compressed_bytes",
    "files",
}

RECEIPT_KEYS = {
    "schema_version",
    "status",
    "dataset_id",
    "revision",
    "source_file",
    "source_compressed_bytes",
    "source_sha256",
    "source_schema_version",
    "language_status",
    "physical_lines",
    "retained_rows",
    "excluded_rows",
    "exclusions",
    "year_counts",
    "month_counts",
    "partitions",
    "exclusion_partition",
    "code_sha256",
    "git_head",
    "git_dirty",
    "manifest_sha256",
    "source_schema_sha256",
    "python_version",
    "pyarrow_version",
    "zstandard_version",
    "completed_at",
    "wall_seconds",
}
PARTITION_KEYS = {"year", "relative_path", "rows", "bytes", "sha256"}
EXCLUSION_PARTITION_KEYS = {"relative_path", "rows", "bytes", "sha256"}

EXPECTED_THESIS_CLAIMS: dict[str, dict[str, Any]] = {
    "line-128-raw": {
        "line": 128,
        "stage": "raw_collected",
        "content_type": "all",
        "count": 73_020_520,
    },
    "line-128-filtered": {
        "line": 128,
        "stage": "legacy_keyword_filtered",
        "content_type": "all",
        "count": 949_624,
    },
    "line-261-submissions": {
        "line": 261,
        "stage": "legacy_analysis_corpus",
        "content_type": "submission",
        "count": 414_095,
    },
    "line-261-comments": {
        "line": 261,
        "stage": "legacy_analysis_corpus",
        "content_type": "comment",
        "count": 4_661_222,
    },
}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}, got {sorted(value)}")


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    result = _require_nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_nonempty_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validated_thesis_claims(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require_exact_keys(
        payload,
        {"schema_version", "source_document", "resolution", "claims"},
        "claims",
    )
    if payload["schema_version"] != "1.0.0":
        raise ValueError("unsupported thesis-count schema_version")
    _require_nonempty_string(payload["source_document"], "source_document")
    resolution = payload["resolution"]
    if not isinstance(resolution, dict):
        raise ValueError("thesis-count resolution must be a table")
    _require_exact_keys(
        resolution,
        {"decision_id", "decision", "accepted_at", "reason"},
        "thesis-count resolution",
    )
    for key in ("decision_id", "accepted_at", "reason"):
        _require_nonempty_string(resolution[key], f"resolution {key}")
    if resolution["decision"] != "supersede_with_canonical_receipt_counts":
        raise ValueError("unsupported thesis-count resolution decision")
    claims = payload["claims"]
    if not isinstance(claims, list) or len(claims) != len(EXPECTED_THESIS_CLAIMS):
        raise ValueError("thesis counts must contain exactly the four registered claims")

    observed: dict[str, dict[str, Any]] = {}
    claim_keys = {"claim_id", "line", "stage", "content_type", "count"}
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("each thesis claim must be a table")
        _require_exact_keys(claim, claim_keys, "thesis claim")
        claim_id = _require_nonempty_string(claim["claim_id"], "claim_id")
        if claim_id in observed:
            raise ValueError(f"duplicate thesis claim_id: {claim_id}")
        observed[claim_id] = {key: claim[key] for key in claim_keys - {"claim_id"}}

    if observed != EXPECTED_THESIS_CLAIMS:
        raise ValueError(
            "thesis claims do not match the four registered line-128/line-261 count claims"
        )
    return claims


def load_thesis_claims(path: Path) -> dict[str, Any]:
    """Load the four explicit legacy corpus-count claims used for reconciliation."""

    with path.open("rb") as source:
        payload = tomllib.load(source)
    _validated_thesis_claims(payload)
    return payload


def _validated_source_manifest(source_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require_exact_keys(source_manifest, SOURCE_MANIFEST_KEYS, "source manifest")
    if source_manifest["schema_version"] != "1.0.0":
        raise ValueError("unsupported source manifest schema_version")
    _require_nonempty_string(source_manifest["dataset_id"], "dataset_id")
    revision = _require_nonempty_string(source_manifest["revision"], "revision")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be a lowercase 40-character commit digest")
    compressed_bytes = _require_positive_int(
        source_manifest["compressed_bytes"], "compressed_bytes"
    )
    if compressed_bytes != EXPECTED_COMPRESSED_BYTES:
        raise ValueError(
            f"expected {EXPECTED_COMPRESSED_BYTES} compressed bytes, got {compressed_bytes}"
        )
    sources = source_manifest["files"]
    if not isinstance(sources, list) or len(sources) != EXPECTED_ARCHIVE_COUNT:
        raise ValueError(f"source manifest must contain exactly {EXPECTED_ARCHIVE_COUNT} sources")
    observed_bytes = 0
    observed_paths: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("every source manifest entry must be an object")
        _require_exact_keys(source, {"path", "size", "sha256"}, "source manifest entry")
        source_file = _require_nonempty_string(source["path"], "source path")
        if not source_file.endswith(("_comments.zst", "_submissions.zst")) or "/" in source_file:
            raise ValueError(f"invalid source path: {source_file}")
        if source_file in observed_paths:
            raise ValueError(f"duplicate source path: {source_file}")
        observed_paths.add(source_file)
        observed_bytes += _require_positive_int(source["size"], f"source size for {source_file}")
        _require_sha256(source["sha256"], f"source sha256 for {source_file}")
    if observed_bytes != compressed_bytes:
        raise ValueError(
            f"source sizes do not sum to compressed_bytes: {observed_bytes} != {compressed_bytes}"
        )
    return sources


def _validated_count_map(value: Any, *, label: str, key_validator: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key_validator(key):
            raise ValueError(f"invalid {label} key: {key!r}")
        result[key] = _require_positive_int(count, f"{label}[{key!r}]")
    return result


def _validate_partition(
    partition: Any, *, source_file: str, expected_year: int, expected_rows: int
) -> None:
    if not isinstance(partition, dict):
        raise ValueError(f"partition for {source_file} must be an object")
    _require_exact_keys(partition, PARTITION_KEYS, f"partition for {source_file}")
    if partition["year"] != expected_year:
        raise ValueError(f"partition year mismatch for {source_file}")
    if _require_positive_int(partition["rows"], "partition rows") != expected_rows:
        raise ValueError(f"partition row count mismatch for {source_file} year {expected_year}")
    _require_positive_int(partition["bytes"], "partition bytes")
    _require_sha256(partition["sha256"], "partition sha256")
    relative_path = _require_nonempty_string(partition["relative_path"], "partition path")
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError(f"partition path must be relative and traversal-free: {relative_path}")


def _validate_receipt(receipt: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    source_file = str(source["path"])
    _require_exact_keys(receipt, RECEIPT_KEYS, f"receipt for {source_file}")
    expected_scalars = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "converted",
        "source_file": source_file,
        "source_compressed_bytes": source["size"],
        "source_sha256": source["sha256"],
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "language_status": "unclassified",
    }
    for key, expected in expected_scalars.items():
        if receipt[key] != expected:
            raise ValueError(
                f"receipt {key} mismatch for {source_file}: {receipt[key]!r} != {expected!r}"
            )

    physical = _require_positive_int(receipt["physical_lines"], "physical_lines")
    retained = _require_positive_int(receipt["retained_rows"], "retained_rows")
    excluded = _require_nonnegative_int(receipt["excluded_rows"], "excluded_rows")
    if physical != retained + excluded:
        raise ValueError(
            f"count conservation failed for {source_file}: {physical} != {retained} + {excluded}"
        )

    exclusions = _validated_count_map(
        receipt["exclusions"],
        label="exclusions",
        key_validator=lambda key: bool(re.fullmatch(r"[a-z][a-z0-9_]*", key)),
    )
    if sum(exclusions.values()) != excluded:
        raise ValueError(f"excluded_rows does not equal summed exclusions for {source_file}")

    year_counts = _validated_count_map(
        receipt["year_counts"],
        label="year_counts",
        key_validator=lambda key: key.isdigit() and int(key) in EXPECTED_YEARS,
    )
    month_counts = _validated_count_map(
        receipt["month_counts"],
        label="month_counts",
        key_validator=lambda key: bool(re.fullmatch(r"202[0-5]-(0[1-9]|1[0-2])", key)),
    )
    if sum(year_counts.values()) != retained:
        raise ValueError(f"retained_rows does not equal summed year_counts for {source_file}")
    if sum(month_counts.values()) != retained:
        raise ValueError(f"retained_rows does not equal summed month_counts for {source_file}")
    month_year_counts: Counter[str] = Counter()
    for month, count in month_counts.items():
        month_year_counts[month[:4]] += count
    if dict(month_year_counts) != year_counts:
        raise ValueError(f"month_counts do not roll up to year_counts for {source_file}")

    partitions = receipt["partitions"]
    if not isinstance(partitions, list) or len(partitions) != len(year_counts):
        raise ValueError(f"partitions must have one entry per retained year for {source_file}")
    partitions_by_year: dict[int, Any] = {}
    for partition in partitions:
        if not isinstance(partition, dict) or not isinstance(partition.get("year"), int):
            raise ValueError(f"invalid partition entry for {source_file}")
        year = partition["year"]
        if year in partitions_by_year:
            raise ValueError(f"duplicate partition year {year} for {source_file}")
        partitions_by_year[year] = partition
    if set(partitions_by_year) != {int(year) for year in year_counts}:
        raise ValueError(f"partition years do not match year_counts for {source_file}")
    for year_text, rows in year_counts.items():
        year = int(year_text)
        _validate_partition(
            partitions_by_year[year],
            source_file=source_file,
            expected_year=year,
            expected_rows=rows,
        )

    exclusion_partition = receipt["exclusion_partition"]
    if excluded == 0:
        if exclusions or exclusion_partition is not None:
            raise ValueError(f"zero exclusions require no exclusion partition for {source_file}")
    else:
        if not isinstance(exclusion_partition, dict):
            raise ValueError(f"missing exclusion partition for {source_file}")
        _require_exact_keys(
            exclusion_partition,
            EXCLUSION_PARTITION_KEYS,
            f"exclusion partition for {source_file}",
        )
        if _require_positive_int(exclusion_partition["rows"], "exclusion rows") != excluded:
            raise ValueError(f"exclusion partition row mismatch for {source_file}")
        _require_positive_int(exclusion_partition["bytes"], "exclusion partition bytes")
        _require_sha256(exclusion_partition["sha256"], "exclusion partition sha256")
        exclusion_relative_path = _require_nonempty_string(
            exclusion_partition["relative_path"], "exclusion path"
        )
        if exclusion_relative_path.startswith("/") or ".." in Path(exclusion_relative_path).parts:
            raise ValueError(
                "exclusion partition path must be relative and traversal-free: "
                f"{exclusion_relative_path}"
            )

    for key in (
        "dataset_id",
        "revision",
        "git_head",
        "python_version",
        "pyarrow_version",
        "zstandard_version",
        "completed_at",
    ):
        _require_nonempty_string(receipt[key], f"receipt {key}")
    for key in ("code_sha256", "manifest_sha256", "source_schema_sha256"):
        _require_sha256(receipt[key], f"receipt {key}")
    if not isinstance(receipt["git_dirty"], bool):
        raise ValueError(f"receipt git_dirty must be boolean for {source_file}")
    wall_seconds = receipt["wall_seconds"]
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, int | float)
        or wall_seconds < 0
    ):
        raise ValueError(f"receipt wall_seconds must be non-negative for {source_file}")
    return dict(receipt)


def _canonical_value_for_claim(claim: Mapping[str, Any], totals: Mapping[str, Any]) -> int:
    if claim["stage"] == "raw_collected" and claim["content_type"] == "all":
        return int(totals["physical_lines"])
    if claim["content_type"] == "all":
        return int(totals["retained_rows"])
    return int(totals["by_content_type"][claim["content_type"]]["retained_rows"])


def _canonical_count_index(totals: Mapping[str, Any]) -> dict[str, int]:
    return {
        "physical_lines.all": int(totals["physical_lines"]),
        "physical_lines.submission": int(totals["by_content_type"]["submission"]["physical_lines"]),
        "physical_lines.comment": int(totals["by_content_type"]["comment"]["physical_lines"]),
        "retained_rows.all": int(totals["retained_rows"]),
        "retained_rows.submission": int(totals["by_content_type"]["submission"]["retained_rows"]),
        "retained_rows.comment": int(totals["by_content_type"]["comment"]["retained_rows"]),
    }


def reconcile_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    source_manifest: Mapping[str, Any],
    thesis_claims: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exactly 20 receipts and produce a metadata-only reconciliation report."""

    sources = _validated_source_manifest(source_manifest)
    if len(receipts) != EXPECTED_ARCHIVE_COUNT:
        raise ValueError(f"reconciliation requires exactly {EXPECTED_ARCHIVE_COUNT} receipts")
    claims = _validated_thesis_claims(thesis_claims)

    sources_by_file = {str(source["path"]): source for source in sources}
    if len(sources_by_file) != EXPECTED_ARCHIVE_COUNT:
        raise ValueError("source manifest paths must be unique")
    receipts_by_file: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("every receipt must be an object")
        source_file = receipt.get("source_file")
        if not isinstance(source_file, str):
            raise ValueError("every receipt must name source_file")
        if source_file in receipts_by_file:
            raise ValueError(f"duplicate receipt for {source_file}")
        receipts_by_file[source_file] = receipt
    expected_files = set(sources_by_file)
    observed_files = set(receipts_by_file)
    if observed_files != expected_files:
        raise ValueError(
            "receipt source set mismatch: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_files)}"
        )

    aggregate: dict[str, Any] = {
        "source_compressed_bytes": 0,
        "physical_lines": 0,
        "retained_rows": 0,
        "excluded_rows": 0,
        "by_content_type": {
            "submission": {
                "source_compressed_bytes": 0,
                "physical_lines": 0,
                "retained_rows": 0,
                "excluded_rows": 0,
            },
            "comment": {
                "source_compressed_bytes": 0,
                "physical_lines": 0,
                "retained_rows": 0,
                "excluded_rows": 0,
            },
        },
        "by_subreddit": {},
    }
    exclusions: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    month_counts: Counter[str] = Counter()
    provenance: dict[str, set[Any]] = {
        "code_sha256": set(),
        "git_head": set(),
        "git_dirty": set(),
        "manifest_sha256": set(),
        "source_schema_sha256": set(),
    }
    source_summaries = []

    dataset_id = source_manifest.get("dataset_id")
    revision = source_manifest.get("revision")
    for source_file in sorted(expected_files):
        receipt = _validate_receipt(receipts_by_file[source_file], sources_by_file[source_file])
        if receipt["dataset_id"] != dataset_id or receipt["revision"] != revision:
            raise ValueError(f"dataset identity mismatch for {source_file}")
        content_type = "comment" if source_file.endswith("_comments.zst") else "submission"
        subreddit = source_file.removesuffix(f"_{content_type}s.zst")
        subreddit_totals = aggregate["by_subreddit"].setdefault(
            subreddit,
            {
                "source_compressed_bytes": 0,
                "physical_lines": 0,
                "retained_rows": 0,
                "excluded_rows": 0,
            },
        )
        for metric in (
            "source_compressed_bytes",
            "physical_lines",
            "retained_rows",
            "excluded_rows",
        ):
            aggregate[metric] += receipt[metric]
            aggregate["by_content_type"][content_type][metric] += receipt[metric]
            subreddit_totals[metric] += receipt[metric]
        exclusions.update(receipt["exclusions"])
        year_counts.update(receipt["year_counts"])
        month_counts.update(receipt["month_counts"])
        for field in provenance:
            provenance[field].add(receipt[field])
        source_summaries.append(
            {
                "source_file": source_file,
                "subreddit": subreddit,
                "content_type": content_type,
                "source_compressed_bytes": receipt["source_compressed_bytes"],
                "physical_lines": receipt["physical_lines"],
                "retained_rows": receipt["retained_rows"],
                "excluded_rows": receipt["excluded_rows"],
            }
        )

    expected_bytes = source_manifest.get("compressed_bytes")
    if aggregate["source_compressed_bytes"] != expected_bytes:
        raise ValueError(
            "aggregate source bytes do not match manifest: "
            f"{aggregate['source_compressed_bytes']} != {expected_bytes}"
        )
    if aggregate["physical_lines"] != aggregate["retained_rows"] + aggregate["excluded_rows"]:
        raise ValueError("aggregate count conservation failed")

    aggregate["exclusions"] = dict(sorted(exclusions.items()))
    aggregate["year_counts_retained_only"] = dict(sorted(year_counts.items()))
    aggregate["month_counts_retained_only"] = dict(sorted(month_counts.items()))

    canonical_count_index = _canonical_count_index(aggregate)
    comparisons = []
    for claim in claims:
        canonical_count = _canonical_value_for_claim(claim, aggregate)
        claimed_count = int(claim["count"])
        numeric_match = canonical_count == claimed_count
        comparisons.append(
            {
                "claim_id": claim["claim_id"],
                "line": claim["line"],
                "legacy_stage": claim["stage"],
                "content_type": claim["content_type"],
                "thesis_count": claimed_count,
                "canonical_metric": (
                    "physical_lines" if claim["stage"] == "raw_collected" else "retained_rows"
                ),
                "canonical_count": canonical_count,
                "canonical_minus_thesis": canonical_count - claimed_count,
                "numeric_match": numeric_match,
                "exact_canonical_matches": sorted(
                    metric
                    for metric, count in canonical_count_index.items()
                    if count == claimed_count
                ),
                "scope_relation": "not_established",
                "status": (
                    "numeric_match_scope_unverified" if numeric_match else "unexplained_mismatch"
                ),
            }
        )

    legacy_filtered = next(
        int(claim["count"]) for claim in claims if claim["claim_id"] == "line-128-filtered"
    )
    legacy_analysis_total = sum(
        int(claim["count"]) for claim in claims if claim["stage"] == "legacy_analysis_corpus"
    )
    has_unexplained_mismatch = (
        any(comparison["status"] == "unexplained_mismatch" for comparison in comparisons)
        or legacy_filtered != legacy_analysis_total
    )
    resolution = dict(thesis_claims["resolution"])

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": ("corrected_legacy_claims" if has_unexplained_mismatch else "reconciled"),
        "acceptance_gate_passed": True,
        "canonical_scope": {
            "dataset_id": dataset_id,
            "revision": revision,
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "source_archives": EXPECTED_ARCHIVE_COUNT,
            "retained_time_window": "2020-01-01T00:00:00Z/2026-01-01T00:00:00Z",
            "language_status": "unclassified",
            "physical_lines_scope": "all decompressed archive lines before canonical exclusions",
            "retained_rows_scope": (
                "canonical rows retained after schema and time-window exclusions"
            ),
        },
        "totals": aggregate,
        "sources": source_summaries,
        "provenance_values": {key: sorted(values) for key, values in provenance.items()},
        "thesis_claim_comparisons": comparisons,
        "resolution": {
            **resolution,
            "status": "accepted_correction",
            "replacement_filtered_count": "pending_stage_a",
        },
        "thesis_internal_reconciliation": {
            "line_128_filtered": legacy_filtered,
            "line_261_analysis_total": legacy_analysis_total,
            "line_261_minus_line_128": legacy_analysis_total - legacy_filtered,
            "status": (
                "reconciled" if legacy_filtered == legacy_analysis_total else "unexplained_mismatch"
            ),
        },
        "unavailable_reconciliations": [
            {
                "metric": "excluded_rows_by_year_or_month",
                "reason": "receipts contain retained-row year/month counts only",
            },
            {
                "metric": "submission_deleted_or_removed_fraction",
                "reason": (
                    "receipts do not retain submission deletion state when title text survives"
                ),
            },
        ],
    }
