from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import export_private_hf_modernbert_corpus_source_v2 as exporter
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _addressed(kind: str, id_field: str, **values: object) -> dict[str, object]:
    body = {"schema_version": "1.0.0", "kind": kind, **values}
    return {**body, id_field: canonical_sha256(body)}


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, str]:
    base_dir = tmp_path / "v1"
    enrichment_root = tmp_path / "enrichment"
    plan = [{"shard_id": "000", "start": 0, "stop": 2, "row_count": 2}]
    base_columns = (
        "corpus_position",
        "opaque_id",
        "subreddit",
        "year",
        "content_type",
        "calibration_member",
    )
    monkeypatch.setattr(exporter.corpus, "CORPUS_ROWS", 2)
    monkeypatch.setattr(exporter.corpus, "SHARD_COUNT", 1)
    monkeypatch.setattr(exporter.candidate, "CALIBRATION_ROWS", 1)
    monkeypatch.setattr(exporter.corpus, "shard_plan", lambda: plan)
    monkeypatch.setattr(exporter.contract, "CORPUS_ROWS", 2)
    monkeypatch.setattr(exporter.contract, "SHARD_COUNT", 1)
    monkeypatch.setattr(exporter.contract, "CALIBRATION_ROWS", 1)
    monkeypatch.setattr(exporter.contract, "CANONICAL_ROWS", 8)
    monkeypatch.setattr(exporter.contract, "PRIVATE_PREDICTION_COLUMNS", base_columns)
    monkeypatch.setattr(exporter.contract, "shard_plan", lambda: plan)
    monkeypatch.setattr(exporter.contract, "HF_PREVIOUS_EXPORT_ID", "e" * 64)
    monkeypatch.setattr(exporter.contract, "HF_PREVIOUS_REVISION", "r" * 40)

    rows = [
        {
            "corpus_position": 0,
            "opaque_id": "t1_a",
            "subreddit": "China",
            "year": 2020,
            "content_type": "comment",
            "calibration_member": True,
        },
        {
            "corpus_position": 1,
            "opaque_id": "t3_b",
            "subreddit": "China",
            "year": 2021,
            "content_type": "submission",
            "calibration_member": False,
        },
    ]
    base_path = base_dir / "data/train-00000-of-00001.parquet"
    base_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), base_path)
    aggregate = base_dir / "aggregates/aggregate.json"
    inference_receipt = base_dir / "provenance/corpus-inference-receipt.json"
    _write_json(aggregate, {"aggregate": "frozen"})
    _write_json(inference_receipt, {"receipt": "frozen"})
    previous_manifest = {
        "schema_version": "1.0.0",
        "kind": "private-modernbert-probability-random-corpus-hf-export-manifest-v1",
        "repo_id": exporter.contract.HF_REPO_ID,
        "access": "private",
        "export_id": exporter.contract.HF_PREVIOUS_EXPORT_ID,
        "corpus_rows": 2,
        "shard_count": 1,
        "output_files": {
            "aggregates/aggregate.json": {
                "sha256": file_sha256(aggregate),
                "bytes": aggregate.stat().st_size,
            },
            "provenance/corpus-inference-receipt.json": {
                "sha256": file_sha256(inference_receipt),
                "bytes": inference_receipt.stat().st_size,
            },
        },
        "status": "complete",
    }
    _write_json(base_dir / "manifest.json", previous_manifest)
    monkeypatch.setattr(
        exporter.contract,
        "HF_PREVIOUS_MANIFEST_SHA256",
        file_sha256(base_dir / "manifest.json"),
    )

    source_bundle = _addressed(
        "modernbert-corpus-source-enrichment-source-bundle-v2",
        "source_bundle_id",
        files=[],
    )
    inventory = _addressed(
        "canonical-source-enrichment-inventory-v2",
        "canonical_inventory_id",
        canonical_rows=8,
    )
    authority = _addressed(
        "modernbert-corpus-source-enrichment-authority-v2",
        "authority_id",
        source_bundle_id=source_bundle["source_bundle_id"],
        canonical_inventory_id=inventory["canonical_inventory_id"],
    )
    enriched_rows = [
        {
            **row,
            "source_id": row["opaque_id"],
            "created_utc": datetime(row["year"], 1, 2, tzinfo=UTC),
        }
        for row in rows
    ]
    enriched_table = pa.Table.from_pylist(enriched_rows).select(
        exporter.contract.enriched_prediction_columns()
    )
    enriched_table = enriched_table.set_column(
        enriched_table.schema.get_field_index("created_utc"),
        "created_utc",
        enriched_table["created_utc"].cast(pa.timestamp("us", tz="UTC")),
    )
    shard_path = enrichment_root / "shards/shard=000/predictions.parquet"
    shard_path.parent.mkdir(parents=True)
    pq.write_table(enriched_table, shard_path)
    receipt_body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-source-enrichment-receipt-v2",
        "authority_id": authority["authority_id"],
        "source_bundle_id": source_bundle["source_bundle_id"],
        "canonical_inventory_id": inventory["canonical_inventory_id"],
        "dataset_revision": exporter.contract.DATASET_REVISION,
        "source_schema_version": exporter.contract.SOURCE_SCHEMA_VERSION,
        "canonical_rows": 8,
        "inference_authority_id": exporter.contract.INFERENCE_AUTHORITY_ID,
        "inference_final_receipt_id": exporter.contract.INFERENCE_FINAL_RECEIPT_ID,
        "inference_source_bundle_id": exporter.contract.INFERENCE_SOURCE_BUNDLE_ID,
        "hf_previous_revision": exporter.contract.HF_PREVIOUS_REVISION,
        "hf_previous_export_id": exporter.contract.HF_PREVIOUS_EXPORT_ID,
        "corpus_rows": 2,
        "shard_count": 1,
        "added_columns": list(exporter.contract.ADDED_COLUMNS),
        "timestamp_unit": "us",
        "timestamp_timezone": "UTC",
        "matched_rows": 2,
        "distinct_source_ids": 2,
        "missing_rows": 0,
        "metadata_mismatch_rows": 0,
        "private_identity_projection_sha256": "p" * 64,
        "output_files": [
            {
                "shard_id": "000",
                "relative_path": "shards/shard=000/predictions.parquet",
                "sha256": file_sha256(shard_path),
                "bytes": shard_path.stat().st_size,
                "row_count": 2,
                "source_prediction_sha256": file_sha256(base_path),
            }
        ],
        "locked_test_rows_accessed": 0,
        "evidence_boundary": exporter.EVIDENCE_BOUNDARY,
        "status": "complete",
    }
    receipt = {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}
    _write_json(enrichment_root / "source-bundle.json", source_bundle)
    _write_json(enrichment_root / "canonical-inventory.json", inventory)
    _write_json(enrichment_root / "authority.json", authority)
    _write_json(enrichment_root / "receipt.json", receipt)
    return base_dir, enrichment_root, str(authority["authority_id"]), str(receipt["receipt_id"])


def test_build_enriched_private_dataset_conserves_projection_and_declares_direct_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, enrichment, authority_id, receipt_id = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "v2"
    manifest = exporter.build_private_hf_dataset(
        enrichment_root=enrichment,
        base_dir=base,
        output_dir=output,
        expected_authority_id=authority_id,
        expected_receipt_id=receipt_id,
    )
    assert manifest["corpus_rows"] == 2
    assert manifest["privacy"]["contains_direct_reddit_ids"] is True
    assert manifest["privacy"]["opaque_id_is_direct_reddit_fullname"] is True
    assert pq.read_table(output / "data/train-00000-of-00001.parquet")[
        "source_id"
    ].to_pylist() == ["t1_a", "t3_b"]
    card = (output / "README.md").read_text(encoding="utf-8")
    assert "historically" in card and "described as a pseudonym" in card
    assert "timestamp[us, tz=UTC]" in card


@pytest.mark.parametrize(
    "relative",
    ["aggregates/aggregate.json", "provenance/corpus-inference-receipt.json"],
)
def test_build_rejects_mutated_predecessor_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    base, enrichment, authority_id, receipt_id = _fixture(tmp_path, monkeypatch)
    (base / relative).write_bytes((base / relative).read_bytes() + b"mutation")
    with pytest.raises(RuntimeError, match="previous private Hub payload drifted"):
        exporter.build_private_hf_dataset(
            enrichment_root=enrichment,
            base_dir=base,
            output_dir=tmp_path / "v2",
            expected_authority_id=authority_id,
            expected_receipt_id=receipt_id,
        )


def test_build_rejects_unreceipted_join_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, enrichment, authority_id, receipt_id = _fixture(tmp_path, monkeypatch)
    (enrichment / "join.duckdb").write_bytes(b"private unreceipted state")
    with pytest.raises(RuntimeError, match="missing or unreceipted state"):
        exporter.build_private_hf_dataset(
            enrichment_root=enrichment,
            base_dir=base,
            output_dir=tmp_path / "v2",
            expected_authority_id=authority_id,
            expected_receipt_id=receipt_id,
        )


def test_publish_requires_pinned_private_parent_and_exact_verifies_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, enrichment, authority_id, receipt_id = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "v2"
    manifest = exporter.build_private_hf_dataset(
        enrichment_root=enrichment,
        base_dir=base,
        output_dir=output,
        expected_authority_id=authority_id,
        expected_receipt_id=receipt_id,
    )
    calls: list[dict[str, object]] = []

    class Info:
        private = True
        sha = exporter.contract.HF_PREVIOUS_REVISION

    class Api:
        def dataset_info(self, **_kwargs: object) -> Info:
            return Info()

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(oid="new-revision")

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: Api())
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: str(base / "manifest.json"),
    )
    local_hashes = exporter.previous._directory_hashes(output)
    monkeypatch.setattr(
        exporter.previous,
        "_remote_file_hashes",
        lambda **_kwargs: {**local_hashes, ".gitattributes": "server-managed"},
    )
    receipt = exporter.publish_private_hf_dataset(
        output_dir=output,
        repo_id=exporter.contract.HF_REPO_ID,
        expected_export_id=manifest["export_id"],
    )
    assert calls[0]["parent_commit"] == exporter.contract.HF_PREVIOUS_REVISION
    assert receipt["revision"] == "new-revision"
    assert receipt["payload_file_count"] == len(local_hashes)
    assert receipt["receipt_id"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_id"}
    )


def test_publish_rejects_stale_remote_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, enrichment, authority_id, receipt_id = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "v2"
    manifest = exporter.build_private_hf_dataset(
        enrichment_root=enrichment,
        base_dir=base,
        output_dir=output,
        expected_authority_id=authority_id,
        expected_receipt_id=receipt_id,
    )

    class Info:
        private = True
        sha = "stale-revision"

    class Api:
        def dataset_info(self, **_kwargs: object) -> Info:
            return Info()

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: Api())
    with pytest.raises(RuntimeError, match="predecessor revision or visibility drifted"):
        exporter.publish_private_hf_dataset(
            output_dir=output,
            repo_id=exporter.contract.HF_REPO_ID,
            expected_export_id=manifest["export_id"],
        )
