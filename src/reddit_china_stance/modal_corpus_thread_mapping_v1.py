"""Recover submission-thread IDs for the scored corpus from the canonical source corpus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_corpus_source_enrichment_v2 as contract
from reddit_china_stance.modernbert_corpus_source_enrichment_v2 import canonical_sha256

APP_NAME = "reddit-china-stance-corpus-thread-mapping-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
SOURCE_ENRICHMENT_AUTHORITY_ID = "10707280a32795edac03d341cee5b2971d045d477f941b382f05ae5e88d470f9"
SOURCE_ENRICHMENT_RECEIPT_ID = "084c84a7c16b00d74819c647bd4ec9fdf4fef9e0a5db9b36ad8b9321e5c2ead6"
CANONICAL_INVENTORY_ID = "5853e4ce4c61ce0995328912e23c77300510e2186aed6096f430e6f28f3439b2"
SOURCE_RUN_ROOT = (
    VOLUME_PATH
    / "student-modernbert-corpus-source-enrichment-v2"
    / f"run={SOURCE_ENRICHMENT_AUTHORITY_ID}"
)
OUTPUT_PREFIX = VOLUME_PATH / "student-modernbert-corpus-thread-mapping-v1"
MAPPING_COLUMNS = ("corpus_position", "thread_id")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("duckdb==1.4.4", "pyarrow==25.0.1", "pydantic==2.13.4")
    .add_local_python_source("reddit_china_stance")
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{where} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{where} must contain an object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sql_paths(paths: list[Path]) -> str:
    return "[" + ",".join("'" + str(path).replace("'", "''") + "'" for path in paths) + "]"


def _validate_source_receipts() -> tuple[dict[str, Any], dict[str, Any], list[Path], list[Path]]:
    import pyarrow.parquet as pq

    receipt_path = SOURCE_RUN_ROOT / "receipt.json"
    inventory_path = SOURCE_RUN_ROOT / "canonical-inventory.json"
    receipt = _read_object(receipt_path, where="source-enrichment receipt")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    output_files = receipt.get("output_files")
    if (
        receipt.get("receipt_id") != canonical_sha256(receipt_body)
        or receipt.get("receipt_id") != SOURCE_ENRICHMENT_RECEIPT_ID
        or receipt.get("authority_id") != SOURCE_ENRICHMENT_AUTHORITY_ID
        or receipt.get("canonical_inventory_id") != CANONICAL_INVENTORY_ID
        or receipt.get("corpus_rows") != contract.CORPUS_ROWS
        or receipt.get("shard_count") != contract.SHARD_COUNT
        or receipt.get("status") != "complete"
        or not isinstance(output_files, list)
        or len(output_files) != contract.SHARD_COUNT
    ):
        raise RuntimeError("source-enrichment receipt binding drifted")

    inventory = _read_object(inventory_path, where="canonical inventory")
    inventory_body = {
        key: value for key, value in inventory.items() if key != "canonical_inventory_id"
    }
    if (
        inventory.get("canonical_inventory_id") != canonical_sha256(inventory_body)
        or inventory.get("canonical_inventory_id") != CANONICAL_INVENTORY_ID
        or inventory.get("canonical_rows") != contract.CANONICAL_ROWS
        or inventory.get("partition_files") != 120
    ):
        raise RuntimeError("canonical inventory binding drifted")

    prediction_paths: list[Path] = []
    for shard, descriptor in zip(contract.shard_plan(), output_files, strict=True):
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("shard_id") != shard["shard_id"]
            or descriptor.get("row_count") != shard["row_count"]
        ):
            raise RuntimeError("source-enrichment output descriptor drifted")
        path = SOURCE_RUN_ROOT / str(descriptor["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
            or pq.ParquetFile(path).metadata.num_rows != descriptor.get("row_count")
        ):
            raise RuntimeError("source-enrichment output payload drifted")
        prediction_paths.append(path)

    canonical_paths: list[Path] = []
    for entry in inventory.get("entries", []):
        for descriptor in entry.get("partitions", []):
            path = VOLUME_PATH / str(descriptor["relative_path"])
            parquet = pq.ParquetFile(path)
            if (
                not path.is_file()
                or parquet.metadata.num_rows != descriptor.get("rows")
                or "submission_id" not in parquet.schema_arrow.names
            ):
                raise RuntimeError("canonical partition lacks the receipted thread field")
            canonical_paths.append(path)
    if len(canonical_paths) != 120:
        raise RuntimeError("canonical partition count drifted")
    return receipt, inventory, prediction_paths, canonical_paths


def _authority(*, script_sha256: str) -> dict[str, Any]:
    if len(script_sha256) != 64:
        raise ValueError("script SHA-256 is invalid")
    body = {
        "schema_version": "1.0.0",
        "kind": "corpus-thread-mapping-authority-v1",
        "dataset_revision": contract.DATASET_REVISION,
        "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
        "corpus_rows": contract.CORPUS_ROWS,
        "source_enrichment_authority_id": SOURCE_ENRICHMENT_AUTHORITY_ID,
        "source_enrichment_receipt_id": SOURCE_ENRICHMENT_RECEIPT_ID,
        "canonical_inventory_id": CANONICAL_INVENTORY_ID,
        "mapping_columns": list(MAPPING_COLUMNS),
        "script_sha256": script_sha256,
        "privacy": "private direct Reddit submission fullnames",
    }
    return {**body, "authority_id": canonical_sha256(body)}


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    timeout=60 * 60,
    cpu=8,
    memory=32768,
)
def build(script_sha256: str) -> dict[str, Any]:
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    runtime_script = Path(__file__)
    if _file_sha256(runtime_script) != script_sha256:
        raise RuntimeError("runtime thread-mapping source differs from the local authority")
    authority = _authority(script_sha256=script_sha256)
    output_root = OUTPUT_PREFIX / f"run={authority['authority_id']}"
    receipt_path = output_root / "receipt.json"
    if output_root.exists():
        if not receipt_path.is_file():
            raise RuntimeError("partial immutable thread-mapping namespace exists")
        existing = _read_object(receipt_path, where="existing thread-mapping receipt")
        body = {key: value for key, value in existing.items() if key != "receipt_id"}
        if (
            existing.get("receipt_id") != canonical_sha256(body)
            or existing.get("status") != "complete"
        ):
            raise RuntimeError("existing thread-mapping receipt is invalid")
        return {
            "status": "already_complete",
            "authority_id": authority["authority_id"],
            "receipt_id": existing["receipt_id"],
            "mapping_path": existing["mapping"]["relative_path"],
            "corpus_rows": existing["mapping"]["row_count"],
            "distinct_threads": existing["distinct_threads"],
        }

    volume.reload()
    source_receipt, inventory, prediction_paths, canonical_paths = _validate_source_receipts()
    OUTPUT_PREFIX.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".thread-mapping-incomplete-", dir=OUTPUT_PREFIX))
    try:
        connection = duckdb.connect(database=":memory:")
        connection.execute("SET TimeZone='UTC'")
        connection.execute("PRAGMA threads=8")
        spill = staging / ".duckdb-spill"
        spill.mkdir()
        connection.execute(f"PRAGMA temp_directory='{spill!s}'")
        prediction_sql = _sql_paths(prediction_paths)
        canonical_sql = _sql_paths(canonical_paths)
        connection.execute(
            f"""
            CREATE TEMP TABLE wanted AS
            SELECT corpus_position, opaque_id, content_type, subreddit, year
            FROM read_parquet({prediction_sql}, hive_partitioning=false)
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE mapping AS
            SELECT
                w.corpus_position,
                c.submission_id AS thread_id,
                w.opaque_id,
                w.content_type,
                c.record_id AS canonical_record_id,
                c.content_type AS canonical_content_type,
                w.subreddit,
                c.subreddit AS canonical_subreddit,
                w.year,
                c.year AS canonical_year
            FROM wanted AS w
            LEFT JOIN read_parquet({canonical_sql}, hive_partitioning=false) AS c
              ON c.record_id = w.opaque_id
            """
        )
        checks = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT corpus_position),
                count(DISTINCT opaque_id),
                count(DISTINCT thread_id),
                count(*) FILTER (WHERE canonical_record_id IS NULL),
                count(*) FILTER (WHERE thread_id IS NULL OR thread_id = ''),
                count(*) FILTER (WHERE NOT starts_with(thread_id, 't3_')),
                count(*) FILTER (
                    WHERE content_type = 'submission' AND thread_id != opaque_id
                ),
                count(*) FILTER (WHERE content_type != canonical_content_type),
                count(*) FILTER (WHERE subreddit != canonical_subreddit),
                count(*) FILTER (WHERE year != canonical_year)
            FROM mapping
            """
        ).fetchone()
        if checks[:3] != (contract.CORPUS_ROWS, contract.CORPUS_ROWS, contract.CORPUS_ROWS) or any(
            checks[index] for index in range(4, len(checks))
        ):
            raise RuntimeError(f"thread mapping join validation failed: {checks}")

        cluster_stats = connection.execute(
            """
            WITH sizes AS (
                SELECT thread_id, count(*)::BIGINT AS records
                FROM mapping GROUP BY thread_id
            )
            SELECT
                min(records),
                approx_quantile(records, 0.5),
                approx_quantile(records, 0.9),
                approx_quantile(records, 0.95),
                approx_quantile(records, 0.99),
                max(records),
                avg(records)
            FROM sizes
            """
        ).fetchone()
        mapping_path = staging / "thread-mapping.parquet"
        table = connection.execute(
            "SELECT corpus_position, thread_id FROM mapping ORDER BY corpus_position"
        ).fetch_arrow_table()
        if table.num_rows != contract.CORPUS_ROWS or tuple(table.column_names) != MAPPING_COLUMNS:
            raise RuntimeError("thread mapping projection drifted")
        table = table.set_column(
            table.schema.get_field_index("corpus_position"),
            "corpus_position",
            table["corpus_position"].cast(pa.int64()),
        )
        pq.write_table(table, mapping_path, compression="zstd")
        connection.close()
        shutil.rmtree(spill)

        output_descriptor = {
            "relative_path": (output_root.relative_to(VOLUME_PATH) / mapping_path.name).as_posix(),
            "sha256": _file_sha256(mapping_path),
            "bytes": mapping_path.stat().st_size,
            "row_count": table.num_rows,
            "columns": list(MAPPING_COLUMNS),
        }
        receipt_body = {
            "schema_version": "1.0.0",
            "kind": "corpus-thread-mapping-receipt-v1",
            "authority_id": authority["authority_id"],
            "source_enrichment_receipt_id": source_receipt["receipt_id"],
            "canonical_inventory_id": inventory["canonical_inventory_id"],
            "corpus_rows": contract.CORPUS_ROWS,
            "matched_rows": checks[0],
            "distinct_threads": checks[3],
            "cluster_size": {
                "minimum": cluster_stats[0],
                "median": cluster_stats[1],
                "p90": cluster_stats[2],
                "p95": cluster_stats[3],
                "p99": cluster_stats[4],
                "maximum": cluster_stats[5],
                "mean": cluster_stats[6],
            },
            "mapping": output_descriptor,
            "privacy": "private direct Reddit submission fullnames",
            "completed_at": datetime.now(UTC).isoformat(),
            "status": "complete",
        }
        receipt = {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}
        (staging / "authority.json").write_bytes(_json_bytes(authority))
        (staging / "receipt.json").write_bytes(_json_bytes(receipt))
        expected_files = {"thread-mapping.parquet", "authority.json", "receipt.json"}
        actual_files = {path.name for path in staging.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise RuntimeError("thread-mapping staging tree drifted")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, output_root)
        volume.commit()
        return {
            "status": "complete",
            "authority_id": authority["authority_id"],
            "receipt_id": receipt["receipt_id"],
            "mapping_path": output_descriptor["relative_path"],
            "corpus_rows": contract.CORPUS_ROWS,
            "distinct_threads": checks[3],
            "cluster_size": receipt["cluster_size"],
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


@app.local_entrypoint()
def main() -> None:
    script_sha256 = _file_sha256(Path(__file__))
    result = build.remote(script_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
