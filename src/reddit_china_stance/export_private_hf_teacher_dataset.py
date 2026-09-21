"""Build the private Hugging Face dataset for the direct-Sol teacher labels.

The export is deliberately a local, immutable packaging step.  Repository data
artefacts remain ignored, and uploading the resulting directory is a separate,
explicit operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from reddit_china_stance.human_seeded_consensus_v1 import (
    load_label_schema,
    validate_semantic_label,
)
from reddit_china_stance.sol_adjudicator import SCHEMA_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLINDED_INPUT = (
    REPO_ROOT / "data/private-sol-teacher-10k-v1/packet-corrected/blinded-input.json"
)
DEFAULT_TEACHER_LABELS = REPO_ROOT / (
    "data/private-sol-teacher-10k-v1/generation/"
    "run=bc7854a60140b3584d5cdeb5513b07ff413253fd59cb84877c278de21702afa4/"
    "reconciliation=e0ec646ec302309facbe9495eb2a142a571009f4c887cab1a109f9b1e9d41066/"
    "teacher-labels.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/private-hf-sol-teacher-10k-v1"
DEFAULT_EXPECTED_ROWS = 10_000

BLINDED_TOP_LEVEL_KEYS = frozenset(
    {
        "blindness",
        "consensus_artifact_sha256",
        "consensus_id",
        "kind",
        "label_schema_sha256",
        "rows",
        "rubric_sha256",
        "schema_version",
        "source_packet_sha256",
    }
)
BLINDED_ROW_KEYS = frozenset(
    {"source_sample_id", "target_text", "submission_context", "parent_context"}
)
TARGETS = ("china_general", "government_ccp", "people_culture", "other")
TEACHER_LABEL_COLUMNS = (
    "opaque_id",
    "record_id",
    "thread_id",
    "subreddit",
    "year",
    "month",
    "content_type",
    "retrieval_mode",
    "cell_id",
    "cell_rank",
    "population_threads_in_cell",
    "selected_threads_in_cell",
    "target_surface_sha256",
    "packet_order",
    "generation_opaque_id",
    "relevance",
    "has_target_china_general",
    "has_target_government_ccp",
    "has_target_people_culture",
    "has_target_other",
    "stance_china_general",
    "stance_government_ccp",
    "stance_people_culture",
    "stance_other",
    "label_json",
    "teacher_run_id",
    "teacher_model",
    "sampling_packet_id",
)
OUTPUT_COLUMNS = (
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
    "relevance",
    "has_target_china_general",
    "has_target_government_ccp",
    "has_target_people_culture",
    "has_target_other",
    "stance_china_general",
    "stance_government_ccp",
    "stance_people_culture",
    "stance_other",
    "label_json",
    "teacher_run_id",
    "teacher_model",
    "sampling_packet_id",
)
FORBIDDEN_OUTPUT_COLUMNS = frozenset(
    {"record_id", "generation_opaque_id", "target_surface_sha256", "author"}
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("thread_id", pa.string(), nullable=False),
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
        pa.field("relevance", pa.string(), nullable=False),
        *(pa.field(f"has_target_{target}", pa.bool_(), nullable=False) for target in TARGETS),
        *(pa.field(f"stance_{target}", pa.string()) for target in TARGETS),
        pa.field("label_json", pa.string(), nullable=False),
        pa.field("teacher_run_id", pa.string(), nullable=False),
        pa.field("teacher_model", pa.string(), nullable=False),
        pa.field("sampling_packet_id", pa.string(), nullable=False),
    ]
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, where: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise ValueError(
            f"{where} keys drifted: expected {sorted(expected)}, got {sorted(observed)}"
        )


def _read_blinded_rows(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("blinded input must be a JSON object")
    _require_exact_keys(value, BLINDED_TOP_LEVEL_KEYS, where="blinded input")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError(
            f"blinded input row count drifted: expected {expected_rows}, got "
            f"{len(rows) if isinstance(rows, list) else 'non-list'}"
        )
    output: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"blinded row {index} must be an object")
        _require_exact_keys(row, BLINDED_ROW_KEYS, where=f"blinded row {index}")
        sample_id = row["source_sample_id"]
        target_text = row["target_text"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"blinded row {index} has invalid source_sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"blinded input contains duplicate sample ID {sample_id!r}")
        if not isinstance(target_text, str) or not target_text:
            raise ValueError(f"blinded row {index} has invalid target_text")
        for context_name in ("submission_context", "parent_context"):
            context = row[context_name]
            if context is not None and not isinstance(context, str):
                raise ValueError(f"blinded row {index} has invalid {context_name}")
        sample_ids.add(sample_id)
        output.append(dict(row))
    return output


def _read_teacher_rows(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    observed_columns = tuple(parquet.schema_arrow.names)
    if observed_columns != TEACHER_LABEL_COLUMNS:
        raise ValueError(
            "teacher-label columns drifted: "
            f"expected {list(TEACHER_LABEL_COLUMNS)}, got {list(observed_columns)}"
        )
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"teacher-label row count drifted: expected {expected_rows}, "
            f"got {parquet.metadata.num_rows}"
        )
    return parquet.read().to_pylist()


def _validate_flattened_label(row: Mapping[str, Any], *, index: int) -> None:
    try:
        raw_label = json.loads(row["label_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"teacher row {index} has invalid label_json") from exc
    label = validate_semantic_label(raw_label, schema=load_label_schema(SCHEMA_PATH))
    target_stances = {item["target"]: item["stance"] for item in label["target_stances"]}
    if row["relevance"] != label["relevance"]:
        raise ValueError(f"teacher row {index} relevance disagrees with label_json")
    for target in TARGETS:
        if row[f"has_target_{target}"] is not (target in target_stances):
            raise ValueError(f"teacher row {index} target flags disagree with label_json")
        if row[f"stance_{target}"] != target_stances.get(target):
            raise ValueError(f"teacher row {index} stance columns disagree with label_json")


def _assemble_rows(
    blinded_rows: Sequence[Mapping[str, Any]],
    teacher_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(blinded_rows) != len(teacher_rows):
        raise ValueError("blinded and teacher row counts differ")
    teacher_ids = [row["opaque_id"] for row in teacher_rows]
    if any(not isinstance(value, str) or not value for value in teacher_ids):
        raise ValueError("teacher labels contain an invalid opaque_id")
    if len(set(teacher_ids)) != len(teacher_ids):
        raise ValueError("teacher labels contain duplicate opaque IDs")
    blinded_ids = [row["source_sample_id"] for row in blinded_rows]
    if set(teacher_ids) != set(blinded_ids):
        missing = len(set(blinded_ids) - set(teacher_ids))
        unexpected = len(set(teacher_ids) - set(blinded_ids))
        raise ValueError(
            f"teacher-label ID conservation failed: {missing} missing, {unexpected} unexpected"
        )
    rows: list[dict[str, Any]] = []
    for index, (blinded, teacher) in enumerate(zip(blinded_rows, teacher_rows, strict=True)):
        if teacher["packet_order"] != index or teacher["opaque_id"] != blinded["source_sample_id"]:
            raise ValueError(f"teacher-label ID/order drift at packet position {index}")
        _validate_flattened_label(teacher, index=index)
        row = {
            "sample_id": blinded["source_sample_id"],
            "thread_id": teacher["thread_id"],
            "target_text": blinded["target_text"],
            "submission_context": blinded["submission_context"],
            "parent_context": blinded["parent_context"],
            **{column: teacher[column] for column in OUTPUT_COLUMNS[5:]},
        }
        if set(row) != set(OUTPUT_COLUMNS):
            raise RuntimeError("internal output-column contract drifted")
        rows.append(row)
    return rows


def _dataset_card(*, row_count: int) -> str:
    fields = "\n".join(f"- `{field.name}`: `{field.type}`" for field in OUTPUT_SCHEMA)
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Reddit China stance: private Sol teacher labels

Private development dataset containing {row_count:,} model-generated teacher labels for
target-specific stance classification. The records contain Reddit-derived text and stable
thread identifiers, so access must remain private and restricted to the thesis project.

## Intended use

Train and compare downstream student classifiers with thread-aware splits. These are Sol
teacher labels, not human annotations. They do not independently validate the construct and
must not be presented as evidence that thesis findings are accurate. Independent double-coded
human evaluation remains required for the final model and thesis claims.

## Data fields

{fields}

The stable opaque `sample_id` is safe for joins within this private dataset. `thread_id` is
retained only because it is required to prevent thread leakage across train and evaluation
splits. Direct Reddit record IDs, generation-time opaque IDs, author data and target-surface
hashes are intentionally excluded.

## Provenance

See `manifest.json` for exact input and output hashes, row conservation, model/run binding,
privacy exclusions and the claim boundary. The Parquet file uses Zstandard compression.
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
    blinded_input_path: Path,
    teacher_labels_path: Path,
    output_dir: Path,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
) -> dict[str, Any]:
    """Join the corrected packet and labels into an immutable Hub-ready directory."""

    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if not blinded_input_path.is_file() or not teacher_labels_path.is_file():
        raise FileNotFoundError("blinded input and teacher labels must both exist")
    blinded_rows = _read_blinded_rows(blinded_input_path, expected_rows=expected_rows)
    teacher_rows = _read_teacher_rows(teacher_labels_path, expected_rows=expected_rows)
    rows = _assemble_rows(blinded_rows, teacher_rows)
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    if table.num_rows != expected_rows or tuple(table.column_names) != OUTPUT_COLUMNS:
        raise RuntimeError("output row/column conservation failed")
    if FORBIDDEN_OUTPUT_COLUMNS.intersection(table.column_names):
        raise RuntimeError("forbidden private identifiers entered the export")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        data_path = staging / "data/train-00000-of-00001.parquet"
        data_path.parent.mkdir()
        pq.write_table(table, data_path, compression="zstd")
        with data_path.open("rb") as handle:
            os.fsync(handle.fileno())
        card_path = staging / "README.md"
        card_path.write_text(_dataset_card(row_count=expected_rows), encoding="utf-8")
        with card_path.open("rb") as handle:
            os.fsync(handle.fileno())

        unique_teacher_runs = sorted({row["teacher_run_id"] for row in rows})
        unique_teacher_models = sorted({row["teacher_model"] for row in rows})
        unique_packets = sorted({row["sampling_packet_id"] for row in rows})
        if not all(
            len(values) == 1
            for values in (unique_teacher_runs, unique_teacher_models, unique_packets)
        ):
            raise ValueError("teacher provenance is not constant across the export")
        manifest = {
            "schema_version": "1.0.0",
            "kind": "private-sol-teacher-hf-dataset-manifest-v1",
            "status": "complete",
            "row_count": expected_rows,
            "column_count": len(OUTPUT_COLUMNS),
            "columns": list(OUTPUT_COLUMNS),
            "input_files": {
                "corrected_blinded_input": {
                    "sha256": file_sha256(blinded_input_path),
                    "bytes": blinded_input_path.stat().st_size,
                },
                "corrected_reconciled_teacher_labels": {
                    "sha256": file_sha256(teacher_labels_path),
                    "bytes": teacher_labels_path.stat().st_size,
                },
            },
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
            "parquet": {"compression": "zstd", "split": "train", "shard_count": 1},
            "provenance": {
                "teacher_run_id": unique_teacher_runs[0],
                "teacher_model": unique_teacher_models[0],
                "sampling_packet_id": unique_packets[0],
            },
            "privacy": {
                "access": "private",
                "contains_reddit_text": True,
                "contains_thread_ids": True,
                "contains_author_data": False,
                "excluded_columns": sorted(FORBIDDEN_OUTPUT_COLUMNS),
            },
            "claim_boundary": {
                "label_source": "model-generated Sol teacher labels",
                "human_validated": False,
                "intended_use": "student-model development and comparison",
                "not_sufficient_for": "construct validity or thesis outcome claims",
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
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blinded-input", type=Path, default=DEFAULT_BLINDED_INPUT)
    parser.add_argument("--teacher-labels", type=Path, default=DEFAULT_TEACHER_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    args = parser.parse_args(argv)
    manifest = export_private_dataset(
        blinded_input_path=args.blinded_input,
        teacher_labels_path=args.teacher_labels,
        output_dir=args.output_dir,
        expected_rows=args.expected_rows,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "row_count": manifest["row_count"],
                "column_count": manifest["column_count"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
