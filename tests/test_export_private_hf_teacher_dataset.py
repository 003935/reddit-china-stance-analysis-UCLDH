from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import export_private_hf_teacher_dataset as exporter


def _label(index: int, *, relevance: str = "material") -> dict[str, object]:
    semantic = (
        {
            "relevance": "material",
            "target_stances": [{"target": "government_ccp", "stance": "negative"}],
        }
        if relevance == "material"
        else {"relevance": "not_material", "target_stances": []}
    )
    return {
        "opaque_id": f"opaque-{index}",
        "record_id": f"record-{index}",
        "thread_id": f"thread-{index}",
        "subreddit": "news",
        "year": 2024,
        "month": 1,
        "content_type": "comment",
        "retrieval_mode": "lexical",
        "cell_id": "cell-1",
        "cell_rank": index + 1,
        "population_threads_in_cell": 100,
        "selected_threads_in_cell": 2,
        "target_surface_sha256": "a" * 64,
        "packet_order": index,
        "generation_opaque_id": f"generation-{index}",
        "relevance": relevance,
        "has_target_china_general": False,
        "has_target_government_ccp": relevance == "material",
        "has_target_people_culture": False,
        "has_target_other": False,
        "stance_china_general": None,
        "stance_government_ccp": "negative" if relevance == "material" else None,
        "stance_people_culture": None,
        "stance_other": None,
        "label_json": json.dumps(semantic, sort_keys=True, separators=(",", ":")),
        "teacher_run_id": "r" * 64,
        "teacher_model": "gpt-5.6-sol",
        "sampling_packet_id": "p" * 64,
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    blinded_path = tmp_path / "blinded-input.json"
    blinded_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "kind": "human-reference-blinded-adjudication-input-v1",
                "consensus_id": "c" * 64,
                "consensus_artifact_sha256": "d" * 64,
                "source_packet_sha256": "s" * 64,
                "rubric_sha256": "u" * 64,
                "label_schema_sha256": "l" * 64,
                "blindness": {"record_ids_removed": True},
                "rows": [
                    {
                        "source_sample_id": f"opaque-{index}",
                        "target_text": f"synthetic target {index}",
                        "submission_context": f"synthetic submission {index}",
                        "parent_context": None,
                    }
                    for index in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "teacher-labels.parquet"
    pq.write_table(
        pa.Table.from_pylist([_label(0), _label(1)], schema=pa.schema([
            pa.field(name, exporter.OUTPUT_SCHEMA.field(name).type)
            if name in exporter.OUTPUT_SCHEMA.names
            else pa.field(name, pa.string())
            for name in exporter.TEACHER_LABEL_COLUMNS
        ])),
        labels_path,
    )
    return blinded_path, labels_path


def _write_teacher_labels(path: Path, rows: list[dict[str, object]]) -> None:
    types = {
        "year": pa.int64(),
        "month": pa.int64(),
        "cell_rank": pa.int64(),
        "population_threads_in_cell": pa.int64(),
        "selected_threads_in_cell": pa.int64(),
        "packet_order": pa.int64(),
        **{f"has_target_{target}": pa.bool_() for target in exporter.TARGETS},
    }
    schema = pa.schema(
        [pa.field(name, types.get(name, pa.string())) for name in exporter.TEACHER_LABEL_COLUMNS]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_export_builds_private_hub_dataset_with_exact_conservation(tmp_path: Path) -> None:
    blinded_path, labels_path = _inputs(tmp_path)
    _write_teacher_labels(labels_path, [_label(0), _label(1)])
    output_dir = tmp_path / "hub-dataset"

    manifest = exporter.export_private_dataset(
        blinded_input_path=blinded_path,
        teacher_labels_path=labels_path,
        output_dir=output_dir,
        expected_rows=2,
    )

    parquet_path = output_dir / "data/train-00000-of-00001.parquet"
    table = pq.read_table(parquet_path)
    assert table.num_rows == 2
    assert tuple(table.column_names) == exporter.OUTPUT_COLUMNS
    assert not exporter.FORBIDDEN_OUTPUT_COLUMNS.intersection(table.column_names)
    assert table.column("sample_id").to_pylist() == ["opaque-0", "opaque-1"]
    assert table.column("thread_id").to_pylist() == ["thread-0", "thread-1"]
    parquet = pq.ParquetFile(parquet_path)
    assert {
        parquet.metadata.row_group(0).column(index).compression
        for index in range(parquet.metadata.num_columns)
    } == {"ZSTD"}
    assert manifest["row_count"] == 2
    assert manifest["column_count"] == len(exporter.OUTPUT_COLUMNS)
    assert manifest["input_files"]["corrected_blinded_input"]["sha256"] == (
        exporter.file_sha256(blinded_path)
    )
    assert manifest["output_files"]["data/train-00000-of-00001.parquet"]["sha256"] == (
        exporter.file_sha256(parquet_path)
    )
    assert manifest["claim_boundary"]["human_validated"] is False
    assert "not human annotations" in (output_dir / "README.md").read_text(encoding="utf-8")

    assert exporter.export_private_dataset(
        blinded_input_path=blinded_path,
        teacher_labels_path=labels_path,
        output_dir=output_dir,
        expected_rows=2,
    ) == manifest


def test_export_rejects_id_or_order_drift(tmp_path: Path) -> None:
    blinded_path, labels_path = _inputs(tmp_path)
    _write_teacher_labels(labels_path, [_label(1), _label(0)])
    with pytest.raises(ValueError, match="ID/order drift"):
        exporter.export_private_dataset(
            blinded_input_path=blinded_path,
            teacher_labels_path=labels_path,
            output_dir=tmp_path / "output",
            expected_rows=2,
        )


def test_export_rejects_unexpected_teacher_columns(tmp_path: Path) -> None:
    blinded_path, labels_path = _inputs(tmp_path)
    table = pq.read_table(labels_path).append_column("author", pa.array(["a", "b"]))
    pq.write_table(table, labels_path)
    with pytest.raises(ValueError, match="teacher-label columns drifted"):
        exporter.export_private_dataset(
            blinded_input_path=blinded_path,
            teacher_labels_path=labels_path,
            output_dir=tmp_path / "output",
            expected_rows=2,
        )


def test_export_refuses_to_overwrite_different_bytes(tmp_path: Path) -> None:
    blinded_path, labels_path = _inputs(tmp_path)
    _write_teacher_labels(labels_path, [_label(0), _label(1)])
    output_dir = tmp_path / "output"
    exporter.export_private_dataset(
        blinded_input_path=blinded_path,
        teacher_labels_path=labels_path,
        output_dir=output_dir,
        expected_rows=2,
    )
    rows = [_label(0, relevance="not_material"), _label(1)]
    _write_teacher_labels(labels_path, rows)
    with pytest.raises(RuntimeError, match="immutable dataset differs"):
        exporter.export_private_dataset(
            blinded_input_path=blinded_path,
            teacher_labels_path=labels_path,
            output_dir=output_dir,
            expected_rows=2,
        )
