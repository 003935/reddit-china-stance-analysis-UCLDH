from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import export_private_hf_modernbert_corpus_v1 as exporter
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256

AUTHORITY_ID = "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, dict[str, object]]:
    plan = [
        {"shard_id": "00", "start": 0, "stop": 2, "row_count": 2},
        {"shard_id": "01", "start": 2, "stop": 3, "row_count": 1},
    ]
    monkeypatch.setattr(exporter.corpus, "CORPUS_ROWS", 3)
    monkeypatch.setattr(exporter.corpus, "SHARD_COUNT", 2)
    monkeypatch.setattr(exporter.candidate, "CALIBRATION_ROWS", 1)
    monkeypatch.setattr(exporter.corpus, "shard_plan", lambda: list(plan))
    monkeypatch.setattr(
        exporter.corpus,
        "private_prediction_columns",
        lambda: ("corpus_position", "opaque_id", "calibration_member"),
    )
    run_root = tmp_path / f"run={AUTHORITY_ID}"
    index: list[dict[str, object]] = []
    for shard in plan:
        path = (
            run_root
            / "full"
            / "prediction-shards"
            / f"shard={shard['shard_id']}"
            / "predictions.parquet"
        )
        path.parent.mkdir(parents=True)
        rows = [
            {
                "corpus_position": position,
                "opaque_id": f"opaque-{position}",
                "calibration_member": position == 1,
            }
            for position in range(shard["start"], shard["stop"])
        ]
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        index.append(
            {
                "shard_id": shard["shard_id"],
                "receipt_id": str(shard["shard_id"]) * 32,
                "private_predictions_artifact": {
                    "relative_path": (
                        "student-modernbert-probability-random-corpus-inference-v1/"
                        f"run={AUTHORITY_ID}/full/prediction-shards/"
                        f"shard={shard['shard_id']}/predictions.parquet"
                    ),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                    "row_count": shard["row_count"],
                },
            }
        )
    aggregate = {"aggregate_id": "b" * 64, "counts": {"rows": 3}}
    aggregate_path = run_root / "final" / "aggregate.json"
    _write_json(aggregate_path, aggregate)
    body: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "modernbert-probability-random-corpus-inference-receipt-v1",
        "authority_id": AUTHORITY_ID,
        "source_bundle_id": "c" * 64,
        "plan_id": "d" * 64,
        "checkpoint_cohort_id": (
            "ad01414410538d5001ac5fb0070751ccfd251d31c5b75c5460bcd5a0dbe06f9b"
        ),
        "calibration_run_id": "e" * 64,
        "calibration_receipt_id": "f" * 64,
        "calibration_id": ("07b429e62653fb8513364327f67e7f620cf44496471c8443023767ad5f66526c"),
        "corpus_rows": 3,
        "calibration_member_rows": 1,
        "label_unseen_rows": 2,
        "ensemble_shard_receipts": 2,
        "prediction_shard_receipts": 2,
        "prediction_index": index,
        "aggregate_id": aggregate["aggregate_id"],
        "aggregate_artifact": {
            "relative_path": f"run={AUTHORITY_ID}/final/aggregate.json",
            "sha256": file_sha256(aggregate_path),
            "bytes": aggregate_path.stat().st_size,
        },
        "provider_billing_hard_cap_enforced": False,
        "locked_test_rows_accessed": 0,
        "status": "complete",
        "evidence_boundary": (
            "provisional model-assisted predictions for the post-acquisition eligible candidate "
            "corpus; default thesis aggregates exclude the 600 calibration members"
        ),
    }
    receipt = {**body, "receipt_id": canonical_sha256(body)}
    _write_json(run_root / "final" / "receipt.json", receipt)
    return run_root, str(receipt["receipt_id"]), receipt


def test_build_private_hf_dataset_exactly_conserves_shards_and_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, receipt_id, _receipt = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "private-hf"

    manifest = exporter.build_private_hf_dataset(
        run_root=run_root,
        output_dir=output,
        expected_authority_id=AUTHORITY_ID,
        expected_final_receipt_id=receipt_id,
    )

    assert manifest["corpus_rows"] == 3
    assert manifest["calibration_member_rows"] == 1
    assert manifest["default_label_unseen_rows"] == 2
    assert manifest["privacy"] == {
        "contains_reddit_text": False,
        "contains_author_data": False,
        "contains_direct_reddit_ids": False,
        "contains_opaque_candidate_ids": True,
        "contains_row_level_model_predictions": True,
        "contains_teacher_labels": False,
        "contains_logits": False,
        "forbidden_columns": sorted(exporter.FORBIDDEN_COLUMNS),
    }
    assert sorted((output / "data").glob("*.parquet")) == [
        output / "data/train-00000-of-00002.parquet",
        output / "data/train-00001-of-00002.parquet",
    ]
    assert "not human-validated labels" in (output / "README.md").read_text(encoding="utf-8")
    assert (
        exporter.build_private_hf_dataset(
            run_root=run_root,
            output_dir=output,
            expected_authority_id=AUTHORITY_ID,
            expected_final_receipt_id=receipt_id,
        )
        == manifest
    )


def test_build_private_hf_dataset_rejects_final_receipt_or_shard_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, _receipt_id, receipt = _fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="receipt binding drifted"):
        exporter.build_private_hf_dataset(
            run_root=run_root,
            output_dir=tmp_path / "bad-receipt",
            expected_authority_id=AUTHORITY_ID,
            expected_final_receipt_id="0" * 64,
        )

    first_path = run_root / "full/prediction-shards/shard=00/predictions.parquet"
    first_path.write_bytes(first_path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="differs from final receipt"):
        exporter.build_private_hf_dataset(
            run_root=run_root,
            output_dir=tmp_path / "bad-shard",
            expected_authority_id=AUTHORITY_ID,
            expected_final_receipt_id=str(receipt["receipt_id"]),
        )


def test_build_private_hf_dataset_refuses_mutated_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, receipt_id, _receipt = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "private-hf"
    exporter.build_private_hf_dataset(
        run_root=run_root,
        output_dir=output,
        expected_authority_id=AUTHORITY_ID,
        expected_final_receipt_id=receipt_id,
    )
    (output / "README.md").write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable private Hub export differs"):
        exporter.build_private_hf_dataset(
            run_root=run_root,
            output_dir=output,
            expected_authority_id=AUTHORITY_ID,
            expected_final_receipt_id=receipt_id,
        )


def test_private_hf_verifier_allows_only_server_managed_gitattributes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "export"
    output.mkdir()
    export_id = "e" * 64
    manifest = {
        "export_id": export_id,
        "repo_id": exporter.DEFAULT_REPO_ID,
        "access": "private",
        "status": "complete",
    }
    _write_json(output / "manifest.json", manifest)
    (output / "payload.txt").write_text("payload\n", encoding="utf-8")

    class Info:
        private = True
        sha = "revision"

    class Api:
        def create_repo(self, **_kwargs: object) -> None:
            return None

        def dataset_info(self, **_kwargs: object) -> Info:
            return Info()

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: Api())
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: str(output / "manifest.json"),
    )
    local_hashes = exporter._directory_hashes(output)
    monkeypatch.setattr(
        exporter,
        "_remote_file_hashes",
        lambda **_kwargs: {**local_hashes, ".gitattributes": "server-managed"},
    )

    receipt = exporter.publish_private_hf_dataset(
        output_dir=output,
        repo_id=exporter.DEFAULT_REPO_ID,
        expected_export_id=export_id,
    )

    assert receipt["payload_file_count"] == 2
    assert receipt["server_managed_files"] == [".gitattributes"]
    assert receipt["private"] is True
    assert receipt["receipt_id"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_id"}
    )
