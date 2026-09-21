from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import modal_modernbert_corpus_source_enrichment_v2 as modal_job
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256


def test_source_bundle_validator_rejects_drift() -> None:
    files = [
        {"path": path, "sha256": str(index) * 64, "bytes": index + 1}
        for index, path in enumerate(modal_job.REQUIRED_SOURCE_FILES, start=1)
    ]
    body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-source-enrichment-source-bundle-v2",
        "files": files,
    }
    bundle = {**body, "source_bundle_id": canonical_sha256(body)}
    assert modal_job._validate_source_bundle(bundle) == bundle
    bundle["files"] = list(reversed(files))
    with pytest.raises(RuntimeError, match="source bundle binding drifted"):
        modal_job._validate_source_bundle(bundle)


def test_timestamp_array_requires_utc_microseconds_and_matching_year() -> None:
    table = pa.table(
        {
            "created_utc": pa.array(
                [datetime(2020, 1, 1, tzinfo=UTC)], type=pa.timestamp("us", tz="UTC")
            ),
            "year": pa.array([2020], type=pa.int32()),
        }
    )
    modal_job._validate_timestamp_array(table)
    with pytest.raises(RuntimeError, match="year differs"):
        modal_job._validate_timestamp_array(table.set_column(1, "year", pa.array([2021])))
    with pytest.raises(RuntimeError, match="Arrow type drifted"):
        modal_job._validate_timestamp_array(
            table.set_column(
                0,
                "created_utc",
                pa.array([datetime(2020, 1, 1)], type=pa.timestamp("us")),
            )
        )


def test_sql_paths_quotes_only_trusted_paths() -> None:
    assert modal_job._sql_paths([modal_job.Path("/data/a'b.parquet")]) == (
        "['/data/a''b.parquet']"
    )


def test_canonical_year_scope_is_exactly_2020_through_2025() -> None:
    assert modal_job.contract.YEARS == (2020, 2021, 2022, 2023, 2024, 2025)


def test_partition_payload_rejects_same_size_same_row_count_mutation(tmp_path) -> None:
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), path)
    descriptor = {
        "bytes": path.stat().st_size,
        "rows": 2,
        "sha256": modal_job._file_sha256(path),
    }
    modal_job._validate_partition_payload(path, descriptor)
    payload = bytearray(path.read_bytes())
    payload[10] ^= 1
    path.write_bytes(payload)
    assert path.stat().st_size == descriptor["bytes"]
    assert pq.ParquetFile(path).metadata.num_rows == descriptor["rows"]
    with pytest.raises(RuntimeError, match="bytes, rows, or hash drifted"):
        modal_job._validate_partition_payload(path, descriptor)


def test_staging_creation_creates_only_the_exact_new_prefix(tmp_path) -> None:
    output_root = tmp_path / "new-prefix" / "run=authority"
    staging = modal_job._create_staging_dir(output_root)
    assert staging.is_dir()
    assert staging.parent == output_root.parent
    assert staging.name.startswith(".run=authority.incomplete-")
    assert not output_root.exists()


def test_staging_tree_rejects_unreceipted_duckdb_state(tmp_path) -> None:
    staging = tmp_path / "staging"
    (staging / "shards/shard=000").mkdir(parents=True)
    (staging / "shards/shard=000/predictions.parquet").write_bytes(b"parquet")
    for name in ("authority.json", "source-bundle.json", "canonical-inventory.json"):
        (staging / name).write_text("{}")
    modal_job._assert_staging_files(staging, shard_ids=["000"], include_receipt=False)
    (staging / "join.duckdb").write_bytes(b"private temp state")
    with pytest.raises(RuntimeError, match="missing or unreceipted"):
        modal_job._assert_staging_files(staging, shard_ids=["000"], include_receipt=False)
    (staging / "join.duckdb").unlink()
    (staging / ".duckdb-spill").mkdir()
    with pytest.raises(RuntimeError, match="retains DuckDB spill state"):
        modal_job._assert_staging_files(staging, shard_ids=["000"], include_receipt=False)
