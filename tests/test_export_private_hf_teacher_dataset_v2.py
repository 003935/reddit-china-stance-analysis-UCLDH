from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import export_private_hf_teacher_dataset_v2 as exporter
from reddit_china_stance import sol_teacher_10k_v2 as teacher
from reddit_china_stance.semantic_ontology_v2 import file_sha256


def _canonical_label(
    codability: str,
    relevance: str | None,
    targets: list[dict[str, str | None]],
) -> str:
    return json.dumps(
        {"codability": codability, "relevance": relevance, "targets": targets},
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_schema() -> pa.Schema:
    integer_fields = {
        "year",
        "month",
        "cell_rank",
        "population_threads_in_cell",
        "selected_threads_in_cell",
        "packet_order",
    }
    return pa.schema(
        [
            pa.field(
                name,
                pa.int64() if name in integer_fields else pa.string(),
                nullable=name in {"submission_context", "parent_context"},
            )
            for name in exporter.SOURCE_COLUMNS
        ]
    )


def _final_schema(run_manifest: dict[str, object]) -> pa.Schema:
    return pa.schema(
        [
            pa.field("opaque_id", pa.string(), nullable=False),
            pa.field("codability", pa.string(), nullable=False),
            pa.field("relevance", pa.string()),
            pa.field("label_json", pa.string(), nullable=False),
            pa.field("quality_tier", pa.string(), nullable=False),
            pa.field("primary_training_eligible", pa.bool_(), nullable=False),
        ],
        metadata=exporter._expected_final_schema_metadata(run_manifest),
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    noncanonical_label: bool = False,
) -> dict[str, object]:
    packet_id = "p" * 64
    run_id = "r" * 64
    source_path = tmp_path / "source.parquet"
    packet_root = tmp_path / f"packet={packet_id}"
    run_root = tmp_path / "private" / f"run={run_id}"
    public_root = tmp_path / "public"
    output_dir = tmp_path / "hub-dataset"
    packet_root.mkdir(parents=True)
    (run_root / "final").mkdir(parents=True)
    (public_root / f"run={run_id}").mkdir(parents=True)

    source_rows = [
        {
            "sample_id": f"L1sample-{index}",
            "thread_id": f"t3_direct{index}",
            "target_text": f"synthetic target {index}",
            "submission_context": f"synthetic submission {index}",
            "parent_context": None,
            "subreddit": "news",
            "year": 2024,
            "month": index + 1,
            "content_type": "comment",
            "retrieval_mode": "lexical",
            "cell_id": "cell-1",
            "cell_rank": index + 1,
            "population_threads_in_cell": 100,
            "selected_threads_in_cell": 3,
            "packet_order": index,
        }
        for index in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(source_rows, schema=_source_schema()), source_path)

    mapping_rows = [
        {
            "opaque_id": f"T2opaque-{index}",
            "sample_id": row["sample_id"],
            "thread_id": row["thread_id"],
            "packet_order": index,
        }
        for index, row in enumerate(source_rows)
    ]
    mapping_path = packet_root / "private-mapping.parquet"
    pq.write_table(
        pa.Table.from_pylist(mapping_rows, schema=teacher._mapping_schema()),
        mapping_path,
        compression="zstd",
    )

    run_manifest: dict[str, object] = {
        "run_id": run_id,
        "packet_id": packet_id,
        "source_parquet_sha256": file_sha256(source_path),
        "label_schema_sha256": "l" * 64,
        "rubric_sha256": "u" * 64,
        "primary_training_policy": (
            "eligible-only-if-exact-consensus-or-blind-majority-and-codable"
        ),
    }
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    labels = [
        {
            "opaque_id": "T2opaque-0",
            "codability": "codable",
            "relevance": "material",
            "label_json": _canonical_label(
                "codable",
                "material",
                [{"target": "government_ccp", "stance": "negative"}],
            ),
            "quality_tier": "exact_consensus",
            "primary_training_eligible": True,
        },
        {
            "opaque_id": "T2opaque-1",
            "codability": "codable",
            "relevance": "not_material",
            "label_json": _canonical_label("codable", "not_material", []),
            "quality_tier": "blind_majority",
            "primary_training_eligible": True,
        },
        {
            "opaque_id": "T2opaque-2",
            "codability": "not_codable",
            "relevance": None,
            "label_json": _canonical_label("not_codable", None, []),
            "quality_tier": "informed_adjudication",
            "primary_training_eligible": False,
        },
    ]
    if noncanonical_label:
        labels[0]["label_json"] = json.dumps(json.loads(labels[0]["label_json"]), indent=2)
    labels_path = run_root / "final/labels.parquet"
    pq.write_table(
        pa.Table.from_pylist(labels, schema=_final_schema(run_manifest)),
        labels_path,
        compression="zstd",
    )

    packet_manifest = {
        "kind": teacher.PACKET_KIND,
        "packet_id": packet_id,
        "source_parquet_sha256": file_sha256(source_path),
        "private_mapping_sha256": file_sha256(mapping_path),
    }
    packet_manifest_path = packet_root / "manifest.json"
    packet_manifest_path.write_text(json.dumps(packet_manifest), encoding="utf-8")
    public_receipt = {
        "kind": "sol-teacher-10k-v2-receipt-v1",
        "status": "complete",
        "run_id": run_id,
        "packet_id": packet_id,
        "row_count": 3,
        "source_parquet_sha256": file_sha256(source_path),
        "packet_manifest_sha256": file_sha256(packet_manifest_path),
        "private_mapping_sha256": file_sha256(mapping_path),
        "private_labels_parquet_sha256": file_sha256(labels_path),
        "label_schema_sha256": "l" * 64,
        "rubric_sha256": "u" * 64,
        "quality_tier_counts": {
            "exact_consensus": 1,
            "blind_majority": 1,
            "informed_adjudication": 1,
        },
        "primary_training_eligible_count": 2,
        "primary_training_ineligible_count": 1,
        "evidence_boundary": exporter.EVIDENCE_BOUNDARY,
    }
    public_receipt_path = public_root / f"run={run_id}" / "receipt-synthetic.json"
    public_receipt_path.write_text(json.dumps(public_receipt), encoding="utf-8")

    monkeypatch.setattr(
        teacher,
        "validate_teacher_packet",
        lambda *args, **kwargs: {"packet_id": packet_id},
    )
    monkeypatch.setattr(
        teacher,
        "validate_final_teacher_labels",
        lambda **kwargs: dict(public_receipt),
    )
    return {
        "source_path": source_path,
        "packet_root": packet_root,
        "run_root": run_root,
        "public_root": public_root,
        "output_dir": output_dir,
        "labels_path": labels_path,
        "public_receipt": public_receipt,
        "source_rows": source_rows,
    }


def _export(paths: dict[str, object]) -> dict[str, object]:
    return exporter.export_private_dataset(
        source_parquet_path=paths["source_path"],
        packet_root=paths["packet_root"],
        run_root=paths["run_root"],
        public_root=paths["public_root"],
        bridge_receipt_path=Path("unused-in-focused-test.json"),
        output_dir=paths["output_dir"],
        expected_rows=3,
    )


def test_export_builds_exact_private_hub_dataset_without_direct_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    manifest = _export(paths)

    output_dir = paths["output_dir"]
    parquet_path = output_dir / "data/train-00000-of-00001.parquet"
    parquet = pq.ParquetFile(parquet_path)
    table = parquet.read()
    assert table.num_rows == 3
    assert tuple(table.column_names) == exporter.OUTPUT_COLUMNS
    assert not exporter.FORBIDDEN_OUTPUT_COLUMNS.intersection(table.column_names)
    assert table.column("sample_id").to_pylist() == [
        "T2opaque-0",
        "T2opaque-1",
        "T2opaque-2",
    ]
    assert len(set(table.column("thread_group_id").to_pylist())) == 3
    assert not set(table.column("thread_group_id").to_pylist()) & {
        row["thread_id"] for row in paths["source_rows"]
    }
    assert table.column("has_target_government_ccp").to_pylist() == [True, False, False]
    assert table.column("stance_government_ccp").to_pylist() == [
        "negative",
        None,
        None,
    ]
    assert parquet.schema_arrow.metadata[b"direct_reddit_ids"] == b"excluded"
    assert {
        parquet.metadata.row_group(0).column(index).compression
        for index in range(parquet.metadata.num_columns)
    } == {"ZSTD"}
    assert manifest["row_count"] == 3
    assert manifest["primary_training_eligible_count"] == 2
    assert manifest["quality_tier_counts"] == paths["public_receipt"][
        "quality_tier_counts"
    ]
    assert manifest["privacy"]["contains_direct_reddit_ids"] is False
    assert manifest["claim_boundary"]["human_validated"] is False
    assert manifest["output_files"]["data/train-00000-of-00001.parquet"][
        "sha256"
    ] == file_sha256(parquet_path)
    card = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "silver, model-assisted teacher" in card
    assert "Direct Reddit record/thread IDs" in card

    assert _export(paths) == manifest


def test_export_rejects_noncanonical_final_label_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch, noncanonical_label=True)

    with pytest.raises(ValueError, match="flattened label drifted"):
        _export(paths)


def test_export_rejects_source_mapping_join_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    mapping_path = paths["packet_root"] / "private-mapping.parquet"
    table = pq.read_table(mapping_path)
    rows = table.to_pylist()
    rows[0]["sample_id"], rows[1]["sample_id"] = rows[1]["sample_id"], rows[0]["sample_id"]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), mapping_path)
    mapping_sha = file_sha256(mapping_path)
    packet_manifest_path = paths["packet_root"] / "manifest.json"
    packet_manifest = json.loads(packet_manifest_path.read_text(encoding="utf-8"))
    packet_manifest["private_mapping_sha256"] = mapping_sha
    packet_manifest_path.write_text(json.dumps(packet_manifest), encoding="utf-8")
    receipt_path = next((paths["public_root"] / f"run={'r' * 64}").glob("receipt-*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["private_mapping_sha256"] = mapping_sha
    receipt["packet_manifest_sha256"] = file_sha256(packet_manifest_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        teacher,
        "validate_final_teacher_labels",
        lambda **kwargs: dict(receipt),
    )

    with pytest.raises(ValueError, match="join drifted"):
        _export(paths)


def test_export_refuses_mutated_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _export(paths)
    (paths["output_dir"] / "README.md").write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable dataset differs"):
        _export(paths)


def test_export_rejects_unvalidated_run_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bad_receipt = dict(paths["public_receipt"])
    bad_receipt["row_count"] = 2
    monkeypatch.setattr(
        teacher,
        "validate_final_teacher_labels",
        lambda **kwargs: bad_receipt,
    )

    with pytest.raises(RuntimeError, match="packet/run/public-receipt binding drifted"):
        _export(paths)
