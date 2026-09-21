"""Build the immutable private Hugging Face export for the Sol v2 10k ledger.

This command packages the already exact-validated v2 teacher ledger.  It does
not call a provider or upload anything.  The resulting directory contains
Reddit-derived text and row-level silver labels and must remain private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from reddit_china_stance import sol_teacher_10k_v2 as teacher
from reddit_china_stance.semantic_ontology_v2 import (
    ANALYTIC_TARGETS,
    TARGETS,
    canonical_sha256,
    file_sha256,
    validate_v2_label,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_PACKET_ID = "94c87ac1a9e68b3977fdf7332ec96017e60e77fde3647464a0454ff144e18797"
PRODUCTION_RUN_ID = "820ed0ab1f6b65ba6f79659eb7b7e7d9172d0900d8236fc9b80822d0b726d831"
DEFAULT_PACKET_ROOT = teacher.DEFAULT_PACKET_PARENT / f"packet={PRODUCTION_PACKET_ID}"
DEFAULT_RUN_ROOT = teacher.DEFAULT_PRIVATE_ROOT / f"run={PRODUCTION_RUN_ID}"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/private-hf-sol-teacher-10k-v2"
DEFAULT_EXPECTED_ROWS = teacher.EXPECTED_SOURCE_ROWS

EVIDENCE_BOUNDARY = "silver-model-assisted-teacher-labels-not-human-validation"
SOURCE_COLUMNS = (
    "sample_id",
    "thread_id",
    "target_text",
    "submission_context",
    "parent_context",
    "subreddit",
    "year",
    "month",
    "content_type",
    "retrieval_mode",
    "cell_id",
    "cell_rank",
    "population_threads_in_cell",
    "selected_threads_in_cell",
    "packet_order",
)
OUTPUT_COLUMNS = (
    "sample_id",
    "thread_group_id",
    "target_text",
    "submission_context",
    "parent_context",
    "subreddit",
    "year",
    "month",
    "content_type",
    "retrieval_mode",
    "cell_id",
    "cell_rank",
    "population_threads_in_cell",
    "selected_threads_in_cell",
    "packet_order",
    "codability",
    "relevance",
    *(f"has_target_{target}" for target in TARGETS),
    *(f"stance_{target}" for target in ANALYTIC_TARGETS),
    "label_json",
    "quality_tier",
    "primary_training_eligible",
    "teacher_run_id",
    "teacher_packet_id",
    "label_schema_sha256",
    "evidence_boundary",
)
FORBIDDEN_OUTPUT_COLUMNS = frozenset(
    {
        "author",
        "record_id",
        "thread_id",
        "canonical_sample_id",
        "opaque_id",
        "generation_opaque_id",
        "target_surface_sha256",
        "prompt",
        "reasoning",
        "provider_response",
    }
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _base_output_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("thread_group_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("submission_context", pa.string()),
            pa.field("parent_context", pa.string()),
            pa.field("subreddit", pa.string(), nullable=False),
            pa.field("year", pa.int64(), nullable=False),
            pa.field("month", pa.int64(), nullable=False),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("retrieval_mode", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("cell_rank", pa.int64(), nullable=False),
            pa.field("population_threads_in_cell", pa.int64(), nullable=False),
            pa.field("selected_threads_in_cell", pa.int64(), nullable=False),
            pa.field("packet_order", pa.int64(), nullable=False),
            pa.field("codability", pa.string(), nullable=False),
            pa.field("relevance", pa.string()),
            *(
                pa.field(f"has_target_{target}", pa.bool_(), nullable=False)
                for target in TARGETS
            ),
            *(pa.field(f"stance_{target}", pa.string()) for target in ANALYTIC_TARGETS),
            pa.field("label_json", pa.string(), nullable=False),
            pa.field("quality_tier", pa.string(), nullable=False),
            pa.field("primary_training_eligible", pa.bool_(), nullable=False),
            pa.field("teacher_run_id", pa.string(), nullable=False),
            pa.field("teacher_packet_id", pa.string(), nullable=False),
            pa.field("label_schema_sha256", pa.string(), nullable=False),
            pa.field("evidence_boundary", pa.string(), nullable=False),
        ]
    )


BASE_OUTPUT_SCHEMA = _base_output_schema()


def _output_schema(*, export_id: str, contract: Mapping[str, Any]) -> pa.Schema:
    return BASE_OUTPUT_SCHEMA.with_metadata(
        {
            b"kind": b"private-sol-teacher-hf-dataset-v2",
            b"export_id": export_id.encode("utf-8"),
            b"teacher_run_id": str(contract["teacher_run_id"]).encode("utf-8"),
            b"teacher_packet_id": str(contract["teacher_packet_id"]).encode("utf-8"),
            b"source_parquet_sha256": str(contract["source_parquet_sha256"]).encode(
                "utf-8"
            ),
            b"private_mapping_sha256": str(contract["private_mapping_sha256"]).encode(
                "utf-8"
            ),
            b"private_labels_parquet_sha256": str(
                contract["private_labels_parquet_sha256"]
            ).encode("utf-8"),
            b"label_schema_sha256": str(contract["label_schema_sha256"]).encode(
                "utf-8"
            ),
            b"evidence_boundary": EVIDENCE_BOUNDARY.encode("utf-8"),
            b"direct_reddit_ids": b"excluded",
        }
    )


def _require_parquet_columns(
    path: Path,
    expected: Sequence[str],
    *,
    exact: bool,
    where: str,
) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(f"{where} is missing: {path}")
    parquet = pq.ParquetFile(path)
    observed = tuple(parquet.schema_arrow.names)
    if (exact and observed != tuple(expected)) or (
        not exact and not set(expected) <= set(observed)
    ):
        raise ValueError(
            f"{where} columns drifted: expected "
            f"{'exactly ' if exact else 'at least '}{list(expected)}, got {list(observed)}"
        )
    return parquet


def _load_source_rows(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    parquet = _require_parquet_columns(
        path,
        SOURCE_COLUMNS,
        exact=False,
        where="frozen text/context source",
    )
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("frozen text/context source row count drifted")
    rows = parquet.read(columns=list(SOURCE_COLUMNS)).to_pylist()
    sample_ids: set[str] = set()
    thread_ids: set[str] = set()
    for index, row in enumerate(rows):
        for field in (
            "sample_id",
            "thread_id",
            "target_text",
            "subreddit",
            "content_type",
            "retrieval_mode",
            "cell_id",
        ):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"source row {index} has invalid {field}")
        for field in ("submission_context", "parent_context"):
            if row.get(field) is not None and not isinstance(row[field], str):
                raise ValueError(f"source row {index} has invalid {field}")
        for field in (
            "year",
            "month",
            "cell_rank",
            "population_threads_in_cell",
            "selected_threads_in_cell",
            "packet_order",
        ):
            if type(row.get(field)) is not int:
                raise ValueError(f"source row {index} has invalid {field}")
        if row["sample_id"] in sample_ids or row["thread_id"] in thread_ids:
            raise ValueError("source must contain unique samples and one row per thread")
        sample_ids.add(row["sample_id"])
        thread_ids.add(row["thread_id"])
    return rows


def _load_mapping_rows(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    parquet = _require_parquet_columns(
        path,
        teacher.PRIVATE_MAPPING_COLUMNS,
        exact=True,
        where="v2 private mapping",
    )
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("v2 private mapping row count drifted")
    rows = sorted(parquet.read().to_pylist(), key=lambda row: row["packet_order"])
    if [row.get("packet_order") for row in rows] != list(range(expected_rows)):
        raise ValueError("v2 private mapping packet order drifted")
    for field in ("opaque_id", "sample_id", "thread_id"):
        values = [row.get(field) for row in rows]
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"v2 private mapping has invalid {field}")
        if len(values) != len(set(values)):
            raise ValueError(f"v2 private mapping has duplicate {field}")
    return rows


def _expected_final_schema_metadata(run_manifest: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {
        b"kind": b"sol-teacher-10k-v2-final-labels-v1",
        b"run_id": str(run_manifest["run_id"]).encode("utf-8"),
        b"packet_id": str(run_manifest["packet_id"]).encode("utf-8"),
        b"label_schema_sha256": str(run_manifest["label_schema_sha256"]).encode(
            "utf-8"
        ),
        b"source_parquet_sha256": str(run_manifest["source_parquet_sha256"]).encode(
            "utf-8"
        ),
    }


def _load_final_rows(
    path: Path,
    *,
    run_manifest: Mapping[str, Any],
    expected_rows: int,
) -> list[dict[str, Any]]:
    parquet = _require_parquet_columns(
        path,
        teacher.FINAL_COLUMNS,
        exact=True,
        where="v2 final teacher labels",
    )
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("v2 final teacher label row count drifted")
    if parquet.schema_arrow.metadata != _expected_final_schema_metadata(run_manifest):
        raise ValueError("v2 final teacher label metadata binding drifted")
    rows = parquet.read().to_pylist()
    opaque_ids: set[str] = set()
    for index, row in enumerate(rows):
        opaque_id = row.get("opaque_id")
        if not isinstance(opaque_id, str) or not opaque_id or opaque_id in opaque_ids:
            raise ValueError("v2 final labels contain invalid or duplicate opaque IDs")
        opaque_ids.add(opaque_id)
        try:
            raw_label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"v2 final row {index} has invalid label_json") from exc
        label = validate_v2_label(raw_label)
        canonical_label_json = json.dumps(
            label,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if (
            row["label_json"] != canonical_label_json
            or row["codability"] != label["codability"]
            or row["relevance"] != label["relevance"]
        ):
            raise ValueError(f"v2 final row {index} flattened label drifted")
        tier = row.get("quality_tier")
        eligible = row.get("primary_training_eligible")
        if tier not in teacher.QUALITY_TIERS or type(eligible) is not bool:
            raise ValueError(f"v2 final row {index} has invalid quality metadata")
        expected_eligible = (
            tier in {"exact_consensus", "blind_majority"}
            and label["codability"] == "codable"
        )
        if eligible is not expected_eligible:
            raise ValueError(f"v2 final row {index} eligibility policy drifted")
    return rows


def _thread_group_id(*, packet_id: str, opaque_id: str) -> str:
    digest = hashlib.sha256(f"{packet_id}\0thread-group\0{opaque_id}".encode()).hexdigest()
    return f"T2G{digest[:24]}"


def _assemble_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    run_id: str,
    packet_id: str,
    label_schema_sha256: str,
) -> list[dict[str, Any]]:
    if not (len(source_rows) == len(mapping_rows) == len(label_rows)):
        raise ValueError("source, mapping and final-label row counts differ")
    assembled: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_thread_group_ids: set[str] = set()
    for index, (source, mapping, teacher_row) in enumerate(
        zip(source_rows, mapping_rows, label_rows, strict=True)
    ):
        if (
            mapping["packet_order"] != index
            or source["sample_id"] != mapping["sample_id"]
            or source["thread_id"] != mapping["thread_id"]
            or source["packet_order"] != index
            or teacher_row["opaque_id"] != mapping["opaque_id"]
        ):
            raise ValueError(f"v2 source/mapping/label join drifted at packet position {index}")
        label = validate_v2_label(json.loads(teacher_row["label_json"]))
        target_stances = {item["target"]: item["stance"] for item in label["targets"]}
        sample_id = mapping["opaque_id"]
        thread_group_id = _thread_group_id(packet_id=packet_id, opaque_id=sample_id)
        if sample_id in seen_sample_ids or thread_group_id in seen_thread_group_ids:
            raise RuntimeError("export pseudonym collision")
        seen_sample_ids.add(sample_id)
        seen_thread_group_ids.add(thread_group_id)
        output = {
            "sample_id": sample_id,
            "thread_group_id": thread_group_id,
            **{field: source[field] for field in SOURCE_COLUMNS[2:]},
            "codability": label["codability"],
            "relevance": label["relevance"],
            **{
                f"has_target_{target}": target in target_stances
                for target in TARGETS
            },
            **{
                f"stance_{target}": target_stances.get(target)
                for target in ANALYTIC_TARGETS
            },
            "label_json": teacher_row["label_json"],
            "quality_tier": teacher_row["quality_tier"],
            "primary_training_eligible": teacher_row["primary_training_eligible"],
            "teacher_run_id": run_id,
            "teacher_packet_id": packet_id,
            "label_schema_sha256": label_schema_sha256,
            "evidence_boundary": EVIDENCE_BOUNDARY,
        }
        if tuple(output) != OUTPUT_COLUMNS:
            raise RuntimeError("internal v2 export column contract drifted")
        assembled.append(output)
    return assembled


def _validate_upstream(
    *,
    source_parquet_path: Path,
    packet_root: Path,
    run_root: Path,
    public_root: Path,
    bridge_receipt_path: Path,
    expected_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if run_root.parent != teacher.DEFAULT_PRIVATE_ROOT and not run_root.parent.is_dir():
        raise FileNotFoundError(f"v2 private run parent is missing: {run_root.parent}")
    packet_receipt = teacher.validate_teacher_packet(
        packet_root,
        source_parquet_path=source_parquet_path,
        bridge_receipt_path=bridge_receipt_path,
        expected_rows=expected_rows,
    )
    final_receipt = teacher.validate_final_teacher_labels(
        packet_root=packet_root,
        private_root=run_root.parent,
        public_root=public_root,
        source_parquet_path=source_parquet_path,
        bridge_receipt_path=bridge_receipt_path,
        expected_rows=expected_rows,
    )
    if (
        run_root.name != f"run={final_receipt.get('run_id')}"
        or packet_receipt.get("packet_id") != final_receipt.get("packet_id")
        or packet_root.name != f"packet={final_receipt.get('packet_id')}"
        or final_receipt.get("evidence_boundary") != EVIDENCE_BOUNDARY
        or final_receipt.get("row_count") != expected_rows
    ):
        raise RuntimeError("v2 packet/run/public-receipt binding drifted")
    return packet_receipt, final_receipt


def _contract(
    *,
    packet_manifest_path: Path,
    mapping_path: Path,
    run_manifest_path: Path,
    labels_path: Path,
    source_parquet_path: Path,
    public_receipt_path: Path,
    packet_manifest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    public_receipt: Mapping[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "kind": "private-sol-teacher-hf-dataset-contract-v2",
        "row_count": expected_rows,
        "columns": list(OUTPUT_COLUMNS),
        "teacher_run_id": run_manifest["run_id"],
        "teacher_packet_id": packet_manifest["packet_id"],
        "source_parquet_sha256": file_sha256(source_parquet_path),
        "packet_manifest_sha256": file_sha256(packet_manifest_path),
        "private_mapping_sha256": file_sha256(mapping_path),
        "run_manifest_sha256": file_sha256(run_manifest_path),
        "private_labels_parquet_sha256": file_sha256(labels_path),
        "teacher_public_receipt_sha256": file_sha256(public_receipt_path),
        "label_schema_sha256": run_manifest["label_schema_sha256"],
        "rubric_sha256": run_manifest["rubric_sha256"],
        "quality_policy": run_manifest["primary_training_policy"],
        "sample_id_source": "v2-packet-opaque-id",
        "thread_group_policy": (
            "deterministic-export-pseudonym-from-packet-id-and-v2-opaque-id;"
            "valid-because-source-contract-is-one-row-per-thread"
        ),
        "parquet": {"compression": "zstd", "split": "train", "shard_count": 1},
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "upstream_receipt_id": public_receipt_path.stem.removeprefix("receipt-"),
        "upstream_status": public_receipt["status"],
    }


def _dataset_card(*, row_count: int, eligible_count: int, export_id: str) -> str:
    fields = "\n".join(f"- `{field.name}`: `{field.type}`" for field in BASE_OUTPUT_SCHEMA)
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Reddit China stance: private Sol v2 teacher labels

Private thesis-development dataset containing {row_count:,} model-assisted v2 labels and their
bounded source text/context. Exactly {eligible_count:,} rows satisfy the frozen primary-training
policy. Repository export ID: `{export_id}`.

## Access and evidence boundary

This dataset contains Reddit-derived text and row-level labels. Keep the Hub repository private
and restrict access to authorised thesis collaborators. These are **silver, model-assisted teacher
labels**, not human annotations. They do not establish construct validity or authorise thesis or
whole-corpus claims. Independent double-coded human evaluation remains required.

## Data fields

{fields}

`sample_id` is the v2 packet's opaque identifier. `thread_group_id` is a deterministic export
pseudonym used only for leakage-safe grouping. The frozen source contains one row per thread, so
this preserves the necessary grouping contract without exposing direct Reddit submission IDs.

The canonical v2 label is retained in `label_json`; flattened target-presence and stance columns
are deterministic projections for convenient model training. `primary_training_eligible` is true
only for codable exact-consensus or blind-majority rows. Informed-adjudication and non-codable rows
remain available for audit but are excluded from primary training.

Direct Reddit record/thread IDs, canonical private mapping IDs, author data, legacy v1 labels,
provider prompts, raw responses and reasoning traces are intentionally excluded. See
`manifest.json` for exact input/output hashes, run and packet bindings, quality counts, privacy
exclusions and the evidence boundary.
"""


def _directory_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _publish_directory(staging: Path, output_dir: Path) -> None:
    staged_hashes = _directory_file_hashes(staging)
    if output_dir.exists():
        if not output_dir.is_dir() or _directory_file_hashes(output_dir) != staged_hashes:
            raise RuntimeError(f"immutable dataset differs: {output_dir}")
        shutil.rmtree(staging)
        return
    try:
        os.rename(staging, output_dir)
    except OSError:
        if output_dir.is_dir() and _directory_file_hashes(output_dir) == staged_hashes:
            shutil.rmtree(staging)
            return
        raise
    directory_fd = os.open(output_dir.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def export_private_dataset(
    *,
    source_parquet_path: Path = teacher.DEFAULT_SOURCE_PARQUET,
    packet_root: Path = DEFAULT_PACKET_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
    public_root: Path = teacher.DEFAULT_PUBLIC_ROOT,
    bridge_receipt_path: Path = teacher.DEFAULT_BRIDGE_RECEIPT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
) -> dict[str, Any]:
    """Create or exact-validate the immutable private Hub-ready v2 dataset."""

    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("expected_rows must be a positive integer")
    _, validated_public_receipt = _validate_upstream(
        source_parquet_path=source_parquet_path,
        packet_root=packet_root,
        run_root=run_root,
        public_root=public_root,
        bridge_receipt_path=bridge_receipt_path,
        expected_rows=expected_rows,
    )
    packet_manifest_path = packet_root / "manifest.json"
    mapping_path = packet_root / "private-mapping.parquet"
    run_manifest_path = run_root / "run-manifest.json"
    labels_path = run_root / "final/labels.parquet"
    receipt_paths = sorted(
        (public_root / f"run={validated_public_receipt['run_id']}").glob("receipt-*.json")
    )
    if len(receipt_paths) != 1:
        raise RuntimeError("v2 teacher public output must contain exactly one receipt")
    public_receipt_path = receipt_paths[0]
    packet_manifest = _read_object(packet_manifest_path)
    run_manifest = _read_object(run_manifest_path)
    public_receipt = _read_object(public_receipt_path)
    if public_receipt != validated_public_receipt:
        raise RuntimeError("v2 validated public receipt changed before export")
    if (
        packet_manifest.get("packet_id") != run_manifest.get("packet_id")
        or run_manifest.get("packet_id") != public_receipt.get("packet_id")
        or run_manifest.get("run_id") != public_receipt.get("run_id")
        or packet_manifest.get("source_parquet_sha256")
        != run_manifest.get("source_parquet_sha256")
        or run_manifest.get("source_parquet_sha256")
        != public_receipt.get("source_parquet_sha256")
        or packet_manifest.get("private_mapping_sha256")
        != public_receipt.get("private_mapping_sha256")
        or public_receipt.get("private_labels_parquet_sha256") != file_sha256(labels_path)
        or public_receipt.get("packet_manifest_sha256") != file_sha256(packet_manifest_path)
    ):
        raise RuntimeError("v2 export provenance hashes or identities drifted")

    source_rows = _load_source_rows(source_parquet_path, expected_rows=expected_rows)
    mapping_rows = _load_mapping_rows(mapping_path, expected_rows=expected_rows)
    label_rows = _load_final_rows(
        labels_path,
        run_manifest=run_manifest,
        expected_rows=expected_rows,
    )
    rows = _assemble_rows(
        source_rows=source_rows,
        mapping_rows=mapping_rows,
        label_rows=label_rows,
        run_id=run_manifest["run_id"],
        packet_id=run_manifest["packet_id"],
        label_schema_sha256=run_manifest["label_schema_sha256"],
    )
    contract = _contract(
        packet_manifest_path=packet_manifest_path,
        mapping_path=mapping_path,
        run_manifest_path=run_manifest_path,
        labels_path=labels_path,
        source_parquet_path=source_parquet_path,
        public_receipt_path=public_receipt_path,
        packet_manifest=packet_manifest,
        run_manifest=run_manifest,
        public_receipt=public_receipt,
        expected_rows=expected_rows,
    )
    export_id = canonical_sha256(contract)
    table = pa.Table.from_pylist(
        rows,
        schema=_output_schema(export_id=export_id, contract=contract),
    )
    if (
        table.num_rows != expected_rows
        or tuple(table.column_names) != OUTPUT_COLUMNS
        or FORBIDDEN_OUTPUT_COLUMNS.intersection(table.column_names)
    ):
        raise RuntimeError("private v2 output row, column or privacy conservation failed")

    quality_counts = Counter(row["quality_tier"] for row in rows)
    eligible_count = sum(row["primary_training_eligible"] for row in rows)
    expected_quality = public_receipt.get("quality_tier_counts")
    if (
        dict(quality_counts) != expected_quality
        or eligible_count != public_receipt.get("primary_training_eligible_count")
    ):
        raise RuntimeError("private v2 export quality aggregates drifted")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        data_path = staging / "data/train-00000-of-00001.parquet"
        data_path.parent.mkdir()
        pq.write_table(table, data_path, compression="zstd")
        with data_path.open("rb") as handle:
            os.fsync(handle.fileno())
        card_path = staging / "README.md"
        card_path.write_text(
            _dataset_card(
                row_count=expected_rows,
                eligible_count=eligible_count,
                export_id=export_id,
            ),
            encoding="utf-8",
        )
        with card_path.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest = {
            **contract,
            "kind": "private-sol-teacher-hf-dataset-manifest-v2",
            "status": "complete",
            "export_id": export_id,
            "column_count": len(OUTPUT_COLUMNS),
            "quality_tier_counts": {
                tier: quality_counts[tier] for tier in teacher.QUALITY_TIERS
            },
            "primary_training_eligible_count": eligible_count,
            "primary_training_ineligible_count": expected_rows - eligible_count,
            "output_files": {
                "data/train-00000-of-00001.parquet": {
                    "sha256": file_sha256(data_path),
                    "bytes": data_path.stat().st_size,
                },
                "README.md": {
                    "sha256": file_sha256(card_path),
                    "bytes": card_path.stat().st_size,
                },
            },
            "privacy": {
                "access": "private",
                "contains_reddit_text": True,
                "contains_row_level_silver_labels": True,
                "contains_author_data": False,
                "contains_direct_reddit_ids": False,
                "contains_provider_prompts": False,
                "contains_provider_reasoning_or_responses": False,
                "excluded_columns": sorted(FORBIDDEN_OUTPUT_COLUMNS),
            },
            "claim_boundary": {
                "label_source": "model-assisted Sol v2 silver teacher labels",
                "human_validated": False,
                "intended_use": "student-model development and comparison",
                "not_sufficient_for": "construct validity, thesis claims or corpus promotion",
                "remaining_gate": "independent double-coded human evaluation",
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _read_object(output_dir / "manifest.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, default=teacher.DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--public-root", type=Path, default=teacher.DEFAULT_PUBLIC_ROOT)
    parser.add_argument(
        "--bridge-receipt", type=Path, default=teacher.DEFAULT_BRIDGE_RECEIPT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    args = parser.parse_args(argv)
    manifest = export_private_dataset(
        source_parquet_path=args.source_parquet,
        packet_root=args.packet_root,
        run_root=args.run_root,
        public_root=args.public_root,
        bridge_receipt_path=args.bridge_receipt,
        output_dir=args.output_dir,
        expected_rows=args.expected_rows,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "row_count": manifest["row_count"],
                "primary_training_eligible_count": manifest[
                    "primary_training_eligible_count"
                ],
                "export_id": manifest["export_id"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
