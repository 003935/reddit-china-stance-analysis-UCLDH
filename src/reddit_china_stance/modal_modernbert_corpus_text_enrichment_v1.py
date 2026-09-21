"""Exact Modal join adding canonical Reddit text to the private corpus predictions.

The job starts from the completed source-ID/timestamp enrichment on the shared volume and
joins only ``canonical_record.text`` by the canonical Reddit fullname.  It never opens the
locked test and writes to a new immutable namespace.
"""

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

from reddit_china_stance import modal_modernbert_corpus_source_enrichment_v2 as parent_job
from reddit_china_stance import modernbert_corpus_text_enrichment_v1 as contract
from reddit_china_stance.modernbert_corpus_source_enrichment_v2 import canonical_sha256

APP_NAME = "reddit-china-stance-modernbert-corpus-text-enrichment-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path("student-modernbert-corpus-text-enrichment-v1")
POLICY_REPO_PATH = "configs/modernbert-corpus-text-enrichment-v1.toml"
POLICY_RUNTIME_PATH = "/configs/modernbert-corpus-text-enrichment-v1.toml"
CANONICAL_ROOT = parent_job.CANONICAL_ROOT
PARENT_OUTPUT_ROOT = (
    VOLUME_PATH
    / parent_job.OUTPUT_PREFIX
    / f"run={contract.PARENT_ENRICHMENT_AUTHORITY_ID}"
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
    "src/reddit_china_stance/modernbert_corpus_source_enrichment_v2.py",
    "src/reddit_china_stance/modal_modernbert_corpus_source_enrichment_v2.py",
    "src/reddit_china_stance/modernbert_corpus_text_enrichment_v1.py",
    "src/reddit_china_stance/modal_modernbert_corpus_text_enrichment_v1.py",
    "src/reddit_china_stance/export_private_hf_modernbert_corpus_source_v2.py",
    "src/reddit_china_stance/export_private_hf_modernbert_corpus_text_v1.py",
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
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
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
        files.append({"path": relative, "sha256": _file_sha256(path), "bytes": path.stat().st_size})
    body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-text-enrichment-source-bundle-v1",
        "files": files,
    }
    return {**body, "source_bundle_id": canonical_sha256(body)}


def _validate_source_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(bundle)
    body = {key: value for key, value in clean.items() if key != "source_bundle_id"}
    files = clean.get("files")
    if (
        clean.get("kind") != "modernbert-corpus-text-enrichment-source-bundle-v1"
        or clean.get("source_bundle_id") != canonical_sha256(body)
        or not isinstance(files, list)
        or [item.get("path") for item in files if isinstance(item, Mapping)]
        != list(REQUIRED_SOURCE_FILES)
    ):
        raise RuntimeError("text-enrichment source bundle binding drifted")
    return clean


def _validate_policy() -> tuple[dict[str, Any], str]:
    path = Path(POLICY_RUNTIME_PATH)
    raw = path.read_bytes()
    policy = tomllib.loads(raw.decode("utf-8"))
    exact = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-text-enrichment-policy-v1",
        "dataset_revision": contract.DATASET_REVISION,
        "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
        "canonical_rows": contract.CANONICAL_ROWS,
        "corpus_rows": contract.CORPUS_ROWS,
        "shard_count": contract.SHARD_COUNT,
        "calibration_member_rows": contract.CALIBRATION_ROWS,
        "hf_repo_id": contract.HF_REPO_ID,
        "hf_previous_revision": contract.HF_PREVIOUS_REVISION,
        "hf_previous_export_id": contract.HF_PREVIOUS_EXPORT_ID,
        "hf_previous_manifest_sha256": contract.HF_PREVIOUS_MANIFEST_SHA256,
        "parent_enrichment_authority_id": contract.PARENT_ENRICHMENT_AUTHORITY_ID,
        "parent_enrichment_receipt_id": contract.PARENT_ENRICHMENT_RECEIPT_ID,
        "parent_enrichment_source_bundle_id": contract.PARENT_ENRICHMENT_SOURCE_BUNDLE_ID,
        "parent_canonical_inventory_id": contract.PARENT_CANONICAL_INVENTORY_ID,
        "added_columns": list(contract.ADDED_COLUMNS),
        "preserve_columns": ["corpus_position", "opaque_id", "source_id", "created_utc"],
        "forbidden_columns": [
            "author",
            "body",
            "target_text",
            "parent_context",
            "submission_context",
            "thread_id",
            "submission_id",
            "comment_id",
            "label_json",
            "logits",
        ],
        "contains_reddit_text": True,
        "text_source_field": "canonical_record.text",
        "text_hash_field": "canonical_record.text_sha256",
        "text_join_key": "canonical_record.record_id = parent.source_id",
        "locked_test_rows_accessed": 0,
        "confirmation": contract.CONFIRMATION,
    }
    if policy != exact:
        raise RuntimeError("text-enrichment policy drifted")
    return policy, hashlib.sha256(raw).hexdigest()


def _context(
    source_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    import pyarrow.parquet as pq

    bundle = _validate_source_bundle(source_bundle)
    _, policy_sha256 = _validate_policy()
    policy_entry = next(item for item in bundle["files"] if item["path"] == POLICY_REPO_PATH)
    if policy_entry["sha256"] != policy_sha256:
        raise RuntimeError("runtime policy differs from source bundle")
    inventory, canonical_paths = parent_job._canonical_inventory()
    if inventory.get("canonical_inventory_id") != contract.PARENT_CANONICAL_INVENTORY_ID:
        raise RuntimeError("parent canonical inventory binding drifted")
    for path in canonical_paths:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not {"text", "text_sha256"}.issubset(names):
            raise RuntimeError("canonical partition lacks the text or text hash field")
    authority = contract.enrichment_authority(
        policy_sha256=policy_sha256,
        source_bundle_id=bundle["source_bundle_id"],
        canonical_inventory_id=inventory["canonical_inventory_id"],
    )
    return authority, inventory, canonical_paths


def _validate_parent_receipt() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = PARENT_OUTPUT_ROOT / "receipt.json"
    receipt = _read_object(receipt_path, where="parent source-enrichment receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    descriptors = receipt.get("output_files")
    if (
        receipt.get("receipt_id") != canonical_sha256(body)
        or receipt.get("receipt_id") != contract.PARENT_ENRICHMENT_RECEIPT_ID
        or receipt.get("kind") != "modernbert-corpus-source-enrichment-receipt-v2"
        or receipt.get("authority_id") != contract.PARENT_ENRICHMENT_AUTHORITY_ID
        or receipt.get("source_bundle_id") != contract.PARENT_ENRICHMENT_SOURCE_BUNDLE_ID
        or receipt.get("canonical_inventory_id") != contract.PARENT_CANONICAL_INVENTORY_ID
        or receipt.get("dataset_revision") != contract.DATASET_REVISION
        or receipt.get("source_schema_version") != contract.SOURCE_SCHEMA_VERSION
        or receipt.get("canonical_rows") != contract.CANONICAL_ROWS
        or receipt.get("corpus_rows") != contract.CORPUS_ROWS
        or receipt.get("shard_count") != contract.SHARD_COUNT
        or receipt.get("added_columns") != ["source_id", "created_utc"]
        or receipt.get("matched_rows") != contract.CORPUS_ROWS
        or receipt.get("distinct_source_ids") != contract.CORPUS_ROWS
        or receipt.get("missing_rows") != 0
        or receipt.get("metadata_mismatch_rows") != 0
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("status") != "complete"
        or not isinstance(descriptors, list)
        or len(descriptors) != contract.SHARD_COUNT
    ):
        raise RuntimeError("parent source-enrichment receipt binding drifted")
    return receipt, descriptors


def _validate_parent_shards() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    _, descriptors = _validate_parent_receipt()
    plan = contract.shard_plan()
    validated = []
    seen: set[str] = set()
    for shard, descriptor in zip(plan, descriptors, strict=True):
        if not isinstance(descriptor, Mapping):
            raise RuntimeError("parent shard descriptor is malformed")
        relative = f"shards/shard={shard['shard_id']}/predictions.parquet"
        path = PARENT_OUTPUT_ROOT / relative
        if (
            descriptor.get("shard_id") != shard["shard_id"]
            or descriptor.get("relative_path") != relative
            or descriptor.get("row_count") != shard["row_count"]
            or not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
            or _file_sha256(path) != descriptor.get("sha256")
        ):
            raise RuntimeError("parent source-enrichment shard differs from its receipt")
        parquet = pq.ParquetFile(path)
        if (
            tuple(parquet.schema_arrow.names) != contract.parent_prediction_columns()
            or parquet.metadata.num_rows != shard["row_count"]
        ):
            raise RuntimeError("parent source-enrichment schema or row count drifted")
        identity = parquet.read(columns=["corpus_position", "opaque_id", "source_id"]).to_pylist()
        positions = [row["corpus_position"] for row in identity]
        source_ids = [row["source_id"] for row in identity]
        opaque_ids = [row["opaque_id"] for row in identity]
        if (
            positions != list(range(int(shard["start"]), int(shard["stop"])))
            or opaque_ids != source_ids
            or any(not isinstance(value, str) or not value for value in source_ids)
            or len(source_ids) != len(set(source_ids))
            or seen.intersection(source_ids)
        ):
            raise RuntimeError("parent source-enrichment identity order drifted")
        seen.update(source_ids)
        validated.append(
            {
                "shard_id": shard["shard_id"],
                "source": path,
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "row_count": descriptor["row_count"],
            }
        )
    if len(seen) != contract.CORPUS_ROWS:
        raise RuntimeError("parent source-enrichment shards do not conserve corpus IDs")
    return validated


def _sql_paths(paths: Sequence[Path]) -> str:
    return "[" + ",".join("'" + str(path).replace("'", "''") + "'" for path in paths) + "]"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _create_staging_dir(output_root: Path) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_root.name}.incomplete-", dir=output_root.parent))


def _assert_staging_files(
    staging: Path, *, shard_ids: Sequence[str], include_receipt: bool
) -> None:
    if (staging / ".duckdb-spill").exists():
        raise RuntimeError("text-enrichment staging tree retains DuckDB spill state")
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
        raise RuntimeError("text-enrichment staging tree contains missing or unreceipted files")


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, timeout=60 * 60, cpu=2, memory=4096)
def inspect(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    volume.reload()
    authority, inventory, _ = _context(source_bundle)
    parent_receipt, parent_descriptors = _validate_parent_receipt()
    return {
        "authority_id": authority["authority_id"],
        "source_bundle_id": authority["source_bundle_id"],
        "canonical_inventory_id": inventory["canonical_inventory_id"],
        "canonical_rows": inventory["canonical_rows"],
        "canonical_partitions": inventory["partition_files"],
        "parent_receipt_id": parent_receipt["receipt_id"],
        "parent_shards": len(parent_descriptors),
        "corpus_rows": authority["corpus_rows"],
        "added_columns": list(contract.ADDED_COLUMNS),
        "contains_reddit_text": True,
        "locked_test_rows_accessed": 0,
        "status": "ready",
    }


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
    import pyarrow.parquet as pq

    if confirmation != contract.CONFIRMATION:
        raise RuntimeError("exact text-enrichment confirmation token is required")
    volume.reload()
    authority, inventory, canonical_paths = _context(source_bundle)
    if expected_authority_id != authority["authority_id"]:
        raise RuntimeError("expected text-enrichment authority differs")
    output_root = VOLUME_PATH / OUTPUT_PREFIX / f"run={authority['authority_id']}"
    receipt_path = output_root / "receipt.json"
    if output_root.exists():
        if not receipt_path.is_file():
            raise RuntimeError("partial immutable text-enrichment namespace exists")
        existing = _read_object(receipt_path, where="existing text-enrichment receipt")
        body = {key: value for key, value in existing.items() if key != "receipt_id"}
        if (
            existing.get("receipt_id") != canonical_sha256(body)
            or existing.get("authority_id") != authority["authority_id"]
            or existing.get("status") != "complete"
        ):
            raise RuntimeError("existing text-enrichment receipt differs")
        return {
            "authority_id": authority["authority_id"],
            "receipt_id": existing["receipt_id"],
            "corpus_rows": existing["corpus_rows"],
            "shard_count": existing["shard_count"],
            "status": "already_complete",
            "locked_test_rows_accessed": 0,
        }

    parent_shards = _validate_parent_shards()
    started = time.monotonic()
    staging = _create_staging_dir(output_root)
    connection = None
    spill = staging / ".duckdb-spill"
    try:
        parent_sql = _sql_paths([item["source"] for item in parent_shards])
        canonical_sql = _sql_paths(canonical_paths)
        connection = duckdb.connect()
        connection.execute("SET TimeZone='UTC'")
        connection.execute("PRAGMA threads=8")
        spill.mkdir()
        connection.execute(f"PRAGMA temp_directory='{str(spill).replace(chr(39), chr(39) * 2)}'")
        connection.execute(
            f"""
            CREATE TABLE mapping AS
            SELECT
              p.*,
              c.record_id AS canonical_record_id,
              c.text AS canonical_text,
              c.text_sha256 AS canonical_text_sha256,
              c.subreddit AS canonical_subreddit,
              c.year AS canonical_year,
              c.content_type AS canonical_content_type
            FROM read_parquet({parent_sql}, hive_partitioning=false) AS p
            LEFT JOIN read_parquet({canonical_sql}, hive_partitioning=false) AS c
              ON c.record_id = p.source_id
            """
        )
        checks = connection.execute(
            """
            SELECT
              count(*),
              count(DISTINCT source_id),
              count(DISTINCT canonical_record_id),
              count(*) FILTER (WHERE canonical_record_id IS NULL),
              count(*) FILTER (WHERE source_id != canonical_record_id),
              count(*) FILTER (WHERE canonical_text IS NULL OR length(canonical_text) = 0),
              count(*) FILTER (WHERE canonical_text_sha256 IS NULL OR canonical_text_sha256 = ''),
              count(*) FILTER (WHERE canonical_text_sha256 != sha256(canonical_text)),
              count(*) FILTER (WHERE subreddit != canonical_subreddit),
              count(*) FILTER (WHERE year != canonical_year),
              count(*) FILTER (WHERE content_type != canonical_content_type)
            FROM mapping
            """
        ).fetchone()
        expected_checks = (
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
            0,
        )
        if checks != expected_checks:
            raise RuntimeError(f"canonical text join validation failed: {checks}")

        parent_columns = contract.parent_prediction_columns()
        output_columns = contract.enriched_prediction_columns()
        insertion = parent_columns.index("created_utc") + 1
        select_parent = ", ".join(
            [
                *(_quote_identifier(column) for column in parent_columns[:insertion]),
                "canonical_text AS text",
                *(_quote_identifier(column) for column in parent_columns[insertion:]),
            ]
        )
        output_descriptors = []
        text_projection_digest = hashlib.sha256()
        for shard in parent_shards:
            original = pq.read_table(shard["source"])
            start = int(original["corpus_position"][0].as_py())
            stop = start + original.num_rows
            joined_with_hash = connection.execute(
                f"""
                SELECT {select_parent}, canonical_text_sha256 AS _text_sha256
                FROM mapping
                WHERE corpus_position >= ? AND corpus_position < ?
                ORDER BY corpus_position
                """,
                [start, stop],
            ).fetch_arrow_table()
            if tuple(joined_with_hash.column_names) != (*output_columns, "_text_sha256"):
                raise RuntimeError("text-enriched prediction schema drifted")
            for row in joined_with_hash.select(
                ["corpus_position", "source_id", "_text_sha256"]
            ).to_pylist():
                text_projection_digest.update(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                )
            joined = joined_with_hash.drop(["_text_sha256"])
            if (
                joined.num_rows != original.num_rows
                or joined["corpus_position"].to_pylist() != original["corpus_position"].to_pylist()
                or not joined.select(parent_columns).equals(original)
                or joined["text"].null_count
            ):
                raise RuntimeError("text join changed the parent prediction projection")
            path = staging / "shards" / f"shard={shard['shard_id']}" / "predictions.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(joined, path, compression="zstd")
            output_descriptors.append(
                {
                    "shard_id": shard["shard_id"],
                    "relative_path": path.relative_to(staging).as_posix(),
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                    "row_count": joined.num_rows,
                    "parent_shard_sha256": shard["sha256"],
                }
            )
        connection.close()
        connection = None
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
            "kind": "modernbert-corpus-text-enrichment-receipt-v1",
            "authority_id": authority["authority_id"],
            "source_bundle_id": authority["source_bundle_id"],
            "canonical_inventory_id": authority["canonical_inventory_id"],
            "dataset_revision": contract.DATASET_REVISION,
            "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
            "canonical_rows": contract.CANONICAL_ROWS,
            "corpus_rows": contract.CORPUS_ROWS,
            "shard_count": contract.SHARD_COUNT,
            "calibration_member_rows": contract.CALIBRATION_ROWS,
            "hf_previous_revision": contract.HF_PREVIOUS_REVISION,
            "hf_previous_export_id": contract.HF_PREVIOUS_EXPORT_ID,
            "parent_enrichment_authority_id": contract.PARENT_ENRICHMENT_AUTHORITY_ID,
            "parent_enrichment_receipt_id": contract.PARENT_ENRICHMENT_RECEIPT_ID,
            "parent_enrichment_source_bundle_id": contract.PARENT_ENRICHMENT_SOURCE_BUNDLE_ID,
            "parent_canonical_inventory_id": contract.PARENT_CANONICAL_INVENTORY_ID,
            "added_columns": list(contract.ADDED_COLUMNS),
            "preserved_columns": list(contract.parent_prediction_columns()),
            "text_source_field": "canonical_record.text",
            "text_hash_field": "canonical_record.text_sha256",
            "text_join_key": "canonical_record.record_id = parent.source_id",
            "text_hash_algorithm": "sha256(UTF-8 canonical_record.text)",
            "matched_rows": checks[0],
            "distinct_source_ids": checks[1],
            "distinct_canonical_record_ids": checks[2],
            "missing_text_rows": checks[5],
            "missing_text_hash_rows": checks[6],
            "text_hash_mismatch_rows": checks[7],
            "metadata_mismatch_rows": sum(checks[8:11]),
            "text_projection_sha256": text_projection_digest.hexdigest(),
            "output_files": output_descriptors,
            "contains_reddit_text": True,
            "privacy": "private",
            "locked_test_rows_accessed": 0,
            "evidence_boundary": (
                "private canonical Reddit text joined to provisional model-assisted corpus "
                "predictions; not human-validated thesis truth"
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
            "text_hash_verified_rows": checks[0] - checks[7],
            "wall_seconds": receipt["wall_seconds"],
            "status": "complete",
            "locked_test_rows_accessed": 0,
        }
    except BaseException:
        if connection is not None:
            connection.close()
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
def main(action: str = "inspect", authority_id: str = "", confirmation: str = "") -> None:
    if action not in {"inspect", "enrich"}:
        raise ValueError("action must be inspect or enrich")
    bundle = _source_bundle(REPO_ROOT)
    if action == "inspect":
        result = inspect.remote(bundle)
    else:
        if not authority_id:
            raise ValueError("--authority-id is required for text enrichment")
        result = enrich.remote(
            bundle,
            expected_authority_id=authority_id,
            confirmation=confirmation,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
