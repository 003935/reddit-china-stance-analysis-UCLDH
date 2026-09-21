from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import reddit_china_stance.sol_teacher_generation as generation
from reddit_china_stance.privacy import assert_metadata_only


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _packet(tmp_path: Path, rows: int) -> Path:
    root = tmp_path / "packet"
    root.mkdir()
    blinded = {
        "source_packet_sha256": "s" * 64,
        "rows": [{"source_sample_id": f"opaque-{index}"} for index in range(rows)],
    }
    _write_json(root / "blinded-input.json", blinded)
    mapping = [
        {
            "opaque_id": f"opaque-{index}",
            "record_id": f"record-{index}",
            "thread_id": f"thread-{index}",
            "packet_order": index,
        }
        for index in range(rows)
    ]
    pq.write_table(pa.Table.from_pylist(mapping), root / "private-mapping.parquet")
    binding = {
        "item_count": rows,
        "blinded_input_sha256": generation.file_sha256(root / "blinded-input.json"),
        "private_mapping_sha256": generation.file_sha256(root / "private-mapping.parquet"),
    }
    manifest = {
        "kind": "sol-teacher-packet-manifest-v1",
        "packet_id": "p" * 64,
        "contract_id": "c" * 64,
        "packet_binding": binding,
    }
    _write_json(root / "manifest.json", manifest)
    receipt = {
        "kind": "sol-teacher-packet-receipt-v1",
        "status": "complete",
        "packet_id": manifest["packet_id"],
        "contract_id": manifest["contract_id"],
        "item_count": rows,
        "blinded_input_sha256": binding["blinded_input_sha256"],
        "private_mapping_sha256": binding["private_mapping_sha256"],
        "manifest_sha256": generation.file_sha256(root / "manifest.json"),
    }
    _write_json(root / "receipt-test.json", receipt)
    return root


def test_finalise_reconciles_exact_private_rows_and_emits_metadata_only(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(generation, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(generation, "SHARD_COUNT", 2)
    packet_root = _packet(tmp_path, 3)
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    contract = generation.make_run_contract(packet_root)
    run_id = generation.canonical_sha256(contract)
    run_root = private_root / f"run={run_id}"
    generation._write_immutable(
        run_root / "run-manifest.json", {**contract, "run_id": run_id}
    )
    execution = run_root / "execution" / f"input={contract['blinded_input_sha256']}"
    decisions = {
        "schema_version": "1.0.0",
        "kind": "human-reference-blinded-adjudication-output-v1",
        "adjudicator_id": generation.ADJUDICATOR_ID,
        "adjudicator_model": generation.SOL_ADJUDICATOR_MODEL,
        "input_sha256": contract["blinded_input_sha256"],
        "consensus_id": contract["packet_contract_id"],
        "source_packet_sha256": "s" * 64,
        "rubric_sha256": contract["rubric_sha256"],
        "label_schema_sha256": contract["label_schema_sha256"],
        "rows": [
            {
                "source_sample_id": f"opaque-{index}",
                "label": (
                    {
                        "relevance": "material",
                        "target_stances": [
                            {"target": "china_general", "stance": "mixed"},
                            {"target": "other", "stance": "negative"},
                        ],
                    }
                    if index == 0
                    else {"relevance": "not_material", "target_stances": []}
                ),
            }
            for index in range(3)
        ]
    }
    _write_json(execution / "adjudicator-output.json", decisions)
    for index in range(2):
        shard_rows = decisions["rows"][index::2]
        shard_root = execution / f"shard-{index:03d}"
        raw_shard_rows = json.loads(json.dumps(shard_rows))
        if index == 0:
            raw_shard_rows[0]["label"]["target_stances"].reverse()
        _write_json(shard_root / "last-message.json", {"rows": raw_shard_rows})
        _write_json(
            shard_root / "event.json",
            {
                "input_sha256": contract["blinded_input_sha256"],
                "shard_index": index,
                "model": generation.SOL_ADJUDICATOR_MODEL,
                "reasoning_effort": generation.REASONING_EFFORT,
                "output_sha256": generation.file_sha256(
                    shard_root / "last-message.json"
                ),
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 2,
                },
                "elapsed_seconds": 1.5,
            },
        )
        _write_json(
            shard_root / "fragment.json",
            {
                "input_sha256": contract["blinded_input_sha256"],
                "shard_index": index,
                "item_count": len(shard_rows),
                "event_sha256": generation.file_sha256(shard_root / "event.json"),
                "rows": shard_rows,
            },
        )
    receipt = generation.finalise(
        packet_root=packet_root,
        private_root=private_root,
        public_root=public_root,
    )
    assert receipt["item_count"] == 3
    assert receipt["invalid_item_count"] == 0
    assert receipt["usage_totals"]["input_tokens"] == 20
    assert_metadata_only(receipt)
    labels = pq.read_table(run_root / "teacher-labels.parquet")
    assert labels.num_rows == 3


def test_finalise_rejects_unbound_teacher_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generation, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(generation, "SHARD_COUNT", 1)
    packet_root = _packet(tmp_path, 1)
    private_root = tmp_path / "private"
    contract = generation.make_run_contract(packet_root)
    run_id = generation.canonical_sha256(contract)
    run_root = private_root / f"run={run_id}"
    generation._write_immutable(
        run_root / "run-manifest.json", {**contract, "run_id": run_id}
    )
    output_path = (
        run_root
        / "execution"
        / f"input={contract['blinded_input_sha256']}"
        / "adjudicator-output.json"
    )
    _write_json(
        output_path,
        {
            "rows": [
                {
                    "source_sample_id": "opaque-0",
                    "label": {"relevance": "not_material", "target_stances": []},
                }
            ]
        },
    )
    with pytest.raises(RuntimeError, match="provenance binding failed"):
        generation.finalise(
            packet_root=packet_root,
            private_root=private_root,
            public_root=tmp_path / "public",
        )


def test_parquet_publication_never_leaves_partial_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "labels.parquet"

    def fail_link(*_args: object) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(generation.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        generation._write_parquet_immutable(
            destination, pa.Table.from_pylist([{"label": "negative"}])
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
