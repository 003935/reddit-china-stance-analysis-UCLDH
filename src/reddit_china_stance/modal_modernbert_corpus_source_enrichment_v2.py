"""Exact Modal join adding canonical source IDs and timestamps to corpus predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_corpus_source_enrichment_v2 as contract
from reddit_china_stance.modernbert_corpus_source_enrichment_v2 import canonical_sha256

APP_NAME = "reddit-china-stance-modernbert-corpus-source-enrichment-v2"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path("student-modernbert-corpus-source-enrichment-v2")
POLICY_REPO_PATH = "configs/modernbert-corpus-source-enrichment-v2.toml"
POLICY_RUNTIME_PATH = "/configs/modernbert-corpus-source-enrichment-v2.toml"
INFERENCE_RUN_ROOT = (
    VOLUME_PATH
    / "student-modernbert-probability-random-corpus-inference-v1"
    / f"run={contract.INFERENCE_AUTHORITY_ID}"
)
CANONICAL_ROOT = (
    VOLUME_PATH
    / "normalised"
    / contract.DATASET_REVISION
    / f"schema={contract.SOURCE_SCHEMA_VERSION}"
)
REQUIRED_SOURCE_FILES = (
    POLICY_REPO_PATH,
    "configs/source-files.json",
    "schemas/canonical-record.schema.json",
    "src/reddit_china_stance/ingestion.py",
    "src/reddit_china_stance/models.py",
    "src/reddit_china_stance/modal_parquet.py",
    "src/reddit_china_stance/modal_parquet_validate.py",
    "src/reddit_china_stance/reconciliation.py",
    "src/reddit_china_stance/modernbert_probability_random_corpus_inference_v1.py",
    "src/reddit_china_stance/export_private_hf_modernbert_corpus_v1.py",
    "src/reddit_china_stance/modernbert_corpus_source_enrichment_v2.py",
    "src/reddit_china_stance/modal_modernbert_corpus_source_enrichment_v2.py",
    "src/reddit_china_stance/export_private_hf_modernbert_corpus_source_v2.py",
)


def _repository_root(module_path: Path) -> Path:
    resolved = module_path.resolve()
    if resolved.parent.name == "reddit_china_stance" and resolved.parent.parent.name == "src":
        return resolved.parents[2]
    return resolved.parent


REPO_ROOT = _repository_root(Path(__file__))
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("duckdb==1.4.4", "pyarrow==25.0.1", "pydantic==2.13.4")
    .add_local_python_source("reddit_china_stance")
    .add_local_file(POLICY_REPO_PATH, remote_path=POLICY_RUNTIME_PATH)
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _read_object(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _source_bundle(root: Path) -> dict[str, Any]:
    files = []
    for relative in REQUIRED_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source file is missing: {relative}")
        files.append(
            {"path": relative, "sha256": _file_sha256(path), "bytes": path.stat().st_size}
        )
    body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-source-enrichment-source-bundle-v2",
        "files": files,
    }
    return {**body, "source_bundle_id": canonical_sha256(body)}


def _validate_source_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(bundle)
    body = {key: value for key, value in clean.items() if key != "source_bundle_id"}
    files = clean.get("files")
    if (
        clean.get("kind") != "modernbert-corpus-source-enrichment-source-bundle-v2"
        or clean.get("source_bundle_id") != canonical_sha256(body)
        or not isinstance(files, list)
        or [item.get("path") for item in files if isinstance(item, Mapping)]
        != list(REQUIRED_SOURCE_FILES)
    ):
        raise RuntimeError("source bundle binding drifted")
    return clean


def _validate_policy() -> tuple[dict[str, Any], str]:
    path = Path(POLICY_RUNTIME_PATH)
    raw = path.read_bytes()
    policy = tomllib.loads(raw.decode("utf-8"))
    exact = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-source-enrichment-policy-v2",
        "dataset_revision": contract.DATASET_REVISION,
        "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
        "canonical_rows": contract.CANONICAL_ROWS,
        "corpus_rows": contract.CORPUS_ROWS,
        "shard_count": contract.SHARD_COUNT,
        "inference_authority_id": contract.INFERENCE_AUTHORITY_ID,
        "inference_final_receipt_id": contract.INFERENCE_FINAL_RECEIPT_ID,
        "inference_source_bundle_id": contract.INFERENCE_SOURCE_BUNDLE_ID,
        "hf_repo_id": contract.HF_REPO_ID,
        "hf_previous_revision": contract.HF_PREVIOUS_REVISION,
        "hf_previous_export_id": contract.HF_PREVIOUS_EXPORT_ID,
        "hf_previous_manifest_sha256": contract.HF_PREVIOUS_MANIFEST_SHA256,
        "added_columns": list(contract.ADDED_COLUMNS),
        "preserve_columns": ["opaque_id"],
        "forbidden_columns": [
            "author",
            "text",
            "body",
            "target_text",
            "parent_context",
            "submission_context",
        ],
        "locked_test_rows_accessed": 0,
        "confirmation": contract.CONFIRMATION,
    }
    if policy != exact:
        raise RuntimeError("enrichment policy drifted")
    return policy, hashlib.sha256(raw).hexdigest()


def _validate_partition_payload(path: Path, descriptor: Mapping[str, Any]) -> None:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise RuntimeError("canonical partition is missing")
    if (
        path.stat().st_size != descriptor.get("bytes")
        or pq.ParquetFile(path).metadata.num_rows != descriptor.get("rows")
        or _file_sha256(path) != descriptor.get("sha256")
    ):
        raise RuntimeError("canonical partition bytes, rows, or hash drifted")


def _canonical_inventory() -> tuple[dict[str, Any], list[Path]]:
    import pyarrow.parquet as pq

    if not CANONICAL_ROOT.is_dir():
        raise FileNotFoundError("canonical schema root is unavailable")
    receipt_paths = sorted(CANONICAL_ROOT.glob("*/_receipt.json"))
    if len(receipt_paths) != 20:
        raise RuntimeError("canonical producer receipt count drifted")
    entries: list[dict[str, Any]] = []
    paths: list[Path] = []
    retained_rows = 0
    for receipt_path in receipt_paths:
        receipt = _read_object(receipt_path, where="canonical producer receipt")
        partitions = receipt.get("partitions")
        source_file = receipt.get("source_file")
        if (
            receipt.get("status") != "converted"
            or receipt.get("revision") != contract.DATASET_REVISION
            or receipt.get("source_schema_version") != contract.SOURCE_SCHEMA_VERSION
            or not isinstance(source_file, str)
            or receipt_path.parent.name != Path(source_file).stem
            or not isinstance(partitions, list)
            or len(partitions) != 6
        ):
            raise RuntimeError("canonical producer receipt binding drifted")
        if source_file.endswith("_comments.zst"):
            expected_content_type = "comment"
            expected_subreddit = source_file.removesuffix("_comments.zst")
        elif source_file.endswith("_submissions.zst"):
            expected_content_type = "submission"
            expected_subreddit = source_file.removesuffix("_submissions.zst")
        else:
            raise RuntimeError("canonical producer source filename drifted")
        producer_rows = 0
        partition_entries = []
        for descriptor in partitions:
            if not isinstance(descriptor, Mapping):
                raise RuntimeError("canonical partition descriptor is malformed")
            relative = descriptor.get("relative_path")
            rows = descriptor.get("rows")
            size = descriptor.get("bytes")
            digest = descriptor.get("sha256")
            if (
                not isinstance(relative, str)
                or not isinstance(rows, int)
                or rows < 0
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise RuntimeError("canonical partition descriptor fields drifted")
            path = receipt_path.parent / relative
            relative_parts = Path(relative).parts
            if (
                len(relative_parts) != 4
                or relative_parts[0] not in {f"year={year}" for year in contract.YEARS}
                or relative_parts[1] != f"subreddit={expected_subreddit}"
                or relative_parts[2] != f"content_type={expected_content_type}"
                or relative_parts[3] != "part-00000.parquet"
            ):
                raise RuntimeError("canonical partition path semantics drifted")
            _validate_partition_payload(path, descriptor)
            parquet = pq.ParquetFile(path)
            expected = {
                "record_id",
                "source_id",
                "created_utc",
                "year",
                "subreddit",
                "content_type",
            }
            if not expected.issubset(parquet.schema_arrow.names):
                raise RuntimeError("canonical partition schema lacks enrichment fields")
            paths.append(path)
            producer_rows += rows
            partition_entries.append(
                {
                    "relative_path": path.relative_to(VOLUME_PATH).as_posix(),
                    "sha256": digest,
                    "bytes": size,
                    "rows": rows,
                }
            )
        if producer_rows != receipt.get("retained_rows"):
            raise RuntimeError("canonical producer retained-row conservation failed")
        retained_rows += producer_rows
        entries.append(
            {
                "receipt_path": receipt_path.relative_to(VOLUME_PATH).as_posix(),
                "receipt_sha256": _file_sha256(receipt_path),
                "source_file": source_file,
                "retained_rows": producer_rows,
                "partitions": partition_entries,
            }
        )
    if len(paths) != 120 or retained_rows != contract.CANONICAL_ROWS:
        raise RuntimeError("canonical inventory does not conserve the frozen corpus")
    body = {
        "schema_version": "1.0.0",
        "kind": "canonical-source-enrichment-inventory-v2",
        "dataset_revision": contract.DATASET_REVISION,
        "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
        "producer_receipts": len(entries),
        "partition_files": len(paths),
        "canonical_rows": retained_rows,
        "entries": entries,
    }
    return {**body, "canonical_inventory_id": canonical_sha256(body)}, paths


def _context(source_bundle: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    bundle = _validate_source_bundle(source_bundle)
    _, policy_sha256 = _validate_policy()
    policy_entry = next(item for item in bundle["files"] if item["path"] == POLICY_REPO_PATH)
    if policy_entry["sha256"] != policy_sha256:
        raise RuntimeError("runtime policy differs from source bundle")
    inventory, canonical_paths = _canonical_inventory()
    authority = contract.enrichment_authority(
        policy_sha256=policy_sha256,
        source_bundle_id=bundle["source_bundle_id"],
        canonical_inventory_id=inventory["canonical_inventory_id"],
    )
    return authority, inventory, canonical_paths


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, timeout=60 * 60, cpu=2, memory=4096)
def inspect(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    volume.reload()
    authority, inventory, _ = _context(source_bundle)
    return {
        "authority_id": authority["authority_id"],
        "source_bundle_id": authority["source_bundle_id"],
        "canonical_inventory_id": authority["canonical_inventory_id"],
        "canonical_rows": inventory["canonical_rows"],
        "canonical_partitions": inventory["partition_files"],
        "corpus_rows": authority["corpus_rows"],
        "locked_test_rows_accessed": 0,
        "status": "ready",
    }


def _sql_paths(paths: Sequence[Path]) -> str:
    return "[" + ",".join("'" + str(path).replace("'", "''") + "'" for path in paths) + "]"


def _validate_timestamp_array(table: Any) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    field = table.schema.field("created_utc")
    if field.type != pa.timestamp(contract.TIMESTAMP_UNIT, tz=contract.TIMESTAMP_TIMEZONE):
        raise RuntimeError("created_utc Arrow type drifted")
    if table["created_utc"].null_count:
        raise RuntimeError("created_utc contains nulls")
    years = pc.year(table["created_utc"])
    expected = table["year"].cast(pa.int64())
    if not bool(pc.all(pc.equal(years, expected)).as_py()):
        raise RuntimeError("created_utc year differs from prediction year")


def _validate_inference_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    final_path = INFERENCE_RUN_ROOT / "final" / "receipt.json"
    final_receipt = _read_object(final_path, where="corpus-inference final receipt")
    final_body = {
        key: value for key, value in final_receipt.items() if key != "receipt_id"
    }
    index = final_receipt.get("prediction_index")
    if (
        final_receipt.get("receipt_id") != canonical_sha256(final_body)
        or final_receipt.get("receipt_id") != contract.INFERENCE_FINAL_RECEIPT_ID
        or final_receipt.get("kind")
        != "modernbert-probability-random-corpus-inference-receipt-v1"
        or final_receipt.get("authority_id") != contract.INFERENCE_AUTHORITY_ID
        or final_receipt.get("source_bundle_id") != contract.INFERENCE_SOURCE_BUNDLE_ID
        or final_receipt.get("corpus_rows") != contract.CORPUS_ROWS
        or final_receipt.get("calibration_member_rows") != contract.CALIBRATION_ROWS
        or final_receipt.get("label_unseen_rows")
        != contract.CORPUS_ROWS - contract.CALIBRATION_ROWS
        or final_receipt.get("prediction_shard_receipts") != contract.SHARD_COUNT
        or final_receipt.get("locked_test_rows_accessed") != 0
        or final_receipt.get("status") != "complete"
        or not isinstance(index, list)
        or len(index) != contract.SHARD_COUNT
    ):
        raise RuntimeError("corpus-inference final receipt binding drifted")
    seen: set[str] = set()
    calibration_members = 0
    validated = []
    for shard, entry in zip(contract.shard_plan(), index, strict=True):
        descriptor = (
            entry.get("private_predictions_artifact")
            if isinstance(entry, Mapping)
            else None
        )
        if (
            not isinstance(entry, Mapping)
            or entry.get("shard_id") != shard["shard_id"]
            or not isinstance(entry.get("receipt_id"), str)
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"relative_path", "sha256", "bytes", "row_count"}
            or descriptor.get("row_count") != shard["row_count"]
        ):
            raise RuntimeError("corpus-inference prediction descriptor drifted")
        path = (
            INFERENCE_RUN_ROOT
            / "full"
            / "prediction-shards"
            / f"shard={shard['shard_id']}"
            / "predictions.parquet"
        )
        expected_suffix = (
            f"run={contract.INFERENCE_AUTHORITY_ID}/full/prediction-shards/"
            f"shard={shard['shard_id']}/predictions.parquet"
        )
        if (
            not str(descriptor["relative_path"]).endswith(expected_suffix)
            or not path.is_file()
            or path.stat().st_size != descriptor["bytes"]
            or _file_sha256(path) != descriptor["sha256"]
        ):
            raise RuntimeError("corpus-inference prediction payload drifted")
        parquet = pq.ParquetFile(path)
        if (
            tuple(parquet.schema_arrow.names) != contract.PRIVATE_PREDICTION_COLUMNS
            or parquet.metadata.num_rows != shard["row_count"]
        ):
            raise RuntimeError("corpus-inference prediction schema drifted")
        identity = parquet.read(
            columns=["corpus_position", "opaque_id", "calibration_member"]
        ).to_pylist()
        if [row["corpus_position"] for row in identity] != list(
            range(int(shard["start"]), int(shard["stop"]))
        ):
            raise RuntimeError("corpus-inference prediction row order drifted")
        identities = [row["opaque_id"] for row in identity]
        if (
            any(not isinstance(value, str) or not value for value in identities)
            or len(identities) != len(set(identities))
            or seen.intersection(identities)
            or any(type(row["calibration_member"]) is not bool for row in identity)
        ):
            raise RuntimeError("corpus-inference private identities drifted")
        seen.update(identities)
        calibration_members += sum(row["calibration_member"] for row in identity)
        validated.append(
            {
                "shard_id": shard["shard_id"],
                "source": path,
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "row_count": shard["row_count"],
            }
        )
    if len(seen) != contract.CORPUS_ROWS or calibration_members != contract.CALIBRATION_ROWS:
        raise RuntimeError("corpus-inference prediction union drifted")
    return final_receipt, validated


def _create_staging_dir(output_root: Path) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.incomplete-", dir=output_root.parent
        )
    )


def _assert_staging_files(
    staging: Path, *, shard_ids: Sequence[str], include_receipt: bool
) -> None:
    if (staging / ".duckdb-spill").exists():
        raise RuntimeError("enrichment staging tree retains DuckDB spill state")
    expected = {
        *(f"shards/shard={shard_id}/predictions.parquet" for shard_id in shard_ids),
        "authority.json",
        "source-bundle.json",
        "canonical-inventory.json",
    }
    if include_receipt:
        expected.add("receipt.json")
    actual = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError("enrichment staging tree contains missing or unreceipted files")


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    timeout=6 * 60 * 60,
    cpu=8,
    memory=32768,
)
def enrich(
    source_bundle: Mapping[str, Any], *, expected_authority_id: str, confirmation: str
) -> dict[str, Any]:
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    if confirmation != contract.CONFIRMATION:
        raise RuntimeError("exact enrichment confirmation token is required")
    volume.reload()
    authority, inventory, canonical_paths = _context(source_bundle)
    if expected_authority_id != authority["authority_id"]:
        raise RuntimeError("expected enrichment authority differs")
    output_root = VOLUME_PATH / OUTPUT_PREFIX / f"run={authority['authority_id']}"
    receipt_path = output_root / "receipt.json"
    if output_root.exists():
        if not receipt_path.is_file():
            raise RuntimeError("partial immutable enrichment namespace exists")
        existing = _read_object(receipt_path, where="existing enrichment receipt")
        body = {key: value for key, value in existing.items() if key != "receipt_id"}
        if (
            existing.get("receipt_id") != canonical_sha256(body)
            or existing.get("status") != "complete"
        ):
            raise RuntimeError("existing enrichment receipt differs")
        return {
            "authority_id": authority["authority_id"],
            "receipt_id": existing["receipt_id"],
            "corpus_rows": existing["corpus_rows"],
            "shard_count": existing["shard_count"],
            "status": "already_complete",
            "locked_test_rows_accessed": 0,
        }

    _final_receipt, shards = _validate_inference_inputs()

    started = time.monotonic()
    staging = _create_staging_dir(output_root)
    try:
        wanted_tables = [
            pq.read_table(
                shard["source"],
                columns=["corpus_position", "opaque_id", "subreddit", "year", "content_type"],
            )
            for shard in shards
        ]
        wanted = pa.concat_tables(wanted_tables)
        if wanted.num_rows != contract.CORPUS_ROWS:
            raise RuntimeError("prediction row conservation failed before join")
        connection = duckdb.connect()
        connection.execute("SET TimeZone='UTC'")
        connection.execute("PRAGMA threads=8")
        spill = staging / ".duckdb-spill"
        spill.mkdir()
        connection.execute(
            f"PRAGMA temp_directory='{str(spill).replace(chr(39), chr(39) * 2)}'"
        )
        connection.register("wanted_arrow", wanted)
        connection.execute("CREATE TABLE wanted AS SELECT * FROM wanted_arrow")
        if connection.execute(
            "SELECT count(*), count(DISTINCT opaque_id) FROM wanted"
        ).fetchone() != (contract.CORPUS_ROWS, contract.CORPUS_ROWS):
            raise RuntimeError("prediction opaque IDs are missing or duplicated")
        canonical_sql = _sql_paths(canonical_paths)
        connection.execute(
            f"""
            CREATE TABLE mapping AS
            SELECT
              w.corpus_position,
              w.opaque_id,
              w.subreddit,
              w.year,
              w.content_type,
              c.record_id AS canonical_record_id,
              c.source_id,
              c.created_utc,
              c.subreddit AS canonical_subreddit,
              c.year AS canonical_year,
              c.content_type AS canonical_content_type
            FROM wanted AS w
            LEFT JOIN read_parquet({canonical_sql}, hive_partitioning=false) AS c
              ON c.record_id = w.opaque_id
            """
        )
        checks = connection.execute(
            """
            SELECT
              count(*), count(DISTINCT opaque_id), count(DISTINCT source_id),
              count(*) FILTER (WHERE canonical_record_id IS NULL),
              count(*) FILTER (WHERE source_id IS NULL OR source_id = ''),
              count(*) FILTER (WHERE source_id != canonical_record_id),
              count(*) FILTER (WHERE subreddit != canonical_subreddit),
              count(*) FILTER (WHERE year != canonical_year),
              count(*) FILTER (WHERE content_type != canonical_content_type),
              count(*) FILTER (WHERE created_utc IS NULL OR year(created_utc) != year)
            FROM mapping
            """
        ).fetchone()
        if checks != (
            contract.CORPUS_ROWS,
            contract.CORPUS_ROWS,
            contract.CORPUS_ROWS,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ):
            raise RuntimeError(f"canonical one-to-one join validation failed: {checks}")

        output_descriptors = []
        projection_digest = hashlib.sha256()
        for shard in shards:
            original = pq.read_table(shard["source"])
            start = int(original["corpus_position"][0].as_py())
            stop = start + original.num_rows
            mapping = connection.execute(
                """
                SELECT corpus_position, opaque_id, source_id, created_utc, year
                FROM mapping WHERE corpus_position >= ? AND corpus_position < ?
                ORDER BY corpus_position
                """,
                [start, stop],
            ).fetch_arrow_table()
            if (
                mapping.num_rows != original.num_rows
                or mapping["corpus_position"].to_pylist()
                != original["corpus_position"].to_pylist()
                or mapping["opaque_id"].to_pylist() != original["opaque_id"].to_pylist()
            ):
                raise RuntimeError("joined mapping order differs from prediction shard")
            mapping = mapping.set_column(
                mapping.schema.get_field_index("created_utc"),
                "created_utc",
                mapping["created_utc"].cast(
                    pa.timestamp(contract.TIMESTAMP_UNIT, tz=contract.TIMESTAMP_TIMEZONE)
                ),
            )
            _validate_timestamp_array(mapping)
            arrays = list(original.columns)
            names = list(original.column_names)
            insertion = names.index("opaque_id") + 1
            arrays[insertion:insertion] = [mapping["source_id"], mapping["created_utc"]]
            names[insertion:insertion] = list(contract.ADDED_COLUMNS)
            enriched = pa.Table.from_arrays(arrays, names=names)
            if tuple(enriched.column_names) != contract.enriched_prediction_columns():
                raise RuntimeError("enriched prediction schema drifted")
            _validate_timestamp_array(enriched)
            for row in mapping.select(
                ["corpus_position", "opaque_id", "source_id", "created_utc"]
            ).to_pylist():
                projection_digest.update(
                    json.dumps(
                        {
                            **row,
                            "created_utc": row["created_utc"].isoformat(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
            path = staging / "shards" / f"shard={shard['shard_id']}" / "predictions.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(enriched, path, compression="zstd")
            output_descriptors.append(
                {
                    "shard_id": shard["shard_id"],
                    "relative_path": path.relative_to(staging).as_posix(),
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                    "row_count": enriched.num_rows,
                    "source_prediction_sha256": shard["sha256"],
                }
            )
        connection.close()
        shutil.rmtree(spill)
        (staging / "authority.json").write_bytes(_json_bytes(authority))
        (staging / "source-bundle.json").write_bytes(_json_bytes(dict(source_bundle)))
        (staging / "canonical-inventory.json").write_bytes(_json_bytes(inventory))
        _assert_staging_files(
            staging,
            shard_ids=[str(item["shard_id"]) for item in output_descriptors],
            include_receipt=False,
        )
        receipt_body = {
            "schema_version": "1.0.0",
            "kind": "modernbert-corpus-source-enrichment-receipt-v2",
            "authority_id": authority["authority_id"],
            "source_bundle_id": authority["source_bundle_id"],
            "canonical_inventory_id": authority["canonical_inventory_id"],
            "dataset_revision": contract.DATASET_REVISION,
            "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
            "canonical_rows": contract.CANONICAL_ROWS,
            "inference_authority_id": contract.INFERENCE_AUTHORITY_ID,
            "inference_final_receipt_id": contract.INFERENCE_FINAL_RECEIPT_ID,
            "inference_source_bundle_id": contract.INFERENCE_SOURCE_BUNDLE_ID,
            "hf_previous_revision": contract.HF_PREVIOUS_REVISION,
            "hf_previous_export_id": contract.HF_PREVIOUS_EXPORT_ID,
            "corpus_rows": contract.CORPUS_ROWS,
            "shard_count": contract.SHARD_COUNT,
            "added_columns": list(contract.ADDED_COLUMNS),
            "timestamp_unit": contract.TIMESTAMP_UNIT,
            "timestamp_timezone": contract.TIMESTAMP_TIMEZONE,
            "matched_rows": checks[0],
            "distinct_source_ids": checks[2],
            "missing_rows": checks[3],
            "metadata_mismatch_rows": sum(checks[6:10]),
            "private_identity_projection_sha256": projection_digest.hexdigest(),
            "output_files": output_descriptors,
            "locked_test_rows_accessed": 0,
            "evidence_boundary": (
                "private canonical identifiers and timestamps joined to provisional "
                "model-assisted corpus predictions; not human-validated thesis truth"
            ),
            "wall_seconds": round(time.monotonic() - started, 3),
            "completed_at": datetime.now(UTC).isoformat(),
            "status": "complete",
        }
        receipt = {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}
        (staging / "receipt.json").write_bytes(_json_bytes(receipt))
        _assert_staging_files(
            staging,
            shard_ids=[str(item["shard_id"]) for item in output_descriptors],
            include_receipt=True,
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, output_root)
        volume.commit()
        return {
            "authority_id": authority["authority_id"],
            "receipt_id": receipt["receipt_id"],
            "corpus_rows": contract.CORPUS_ROWS,
            "shard_count": contract.SHARD_COUNT,
            "wall_seconds": receipt["wall_seconds"],
            "status": "complete",
            "locked_test_rows_accessed": 0,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("inspect", "enrich"), default="inspect")
    parser.add_argument("--authority-id")
    parser.add_argument("--confirmation")
    return parser


@app.local_entrypoint()
def main(
    action: str = "inspect", authority_id: str = "", confirmation: str = ""
) -> None:
    if action not in {"inspect", "enrich"}:
        raise ValueError("action must be inspect or enrich")
    bundle = _source_bundle(REPO_ROOT)
    if action == "inspect":
        result = inspect.remote(bundle)
    else:
        if not authority_id:
            raise ValueError("--authority-id is required for enrichment")
        result = enrich.remote(
            bundle,
            expected_authority_id=authority_id,
            confirmation=confirmation,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
