from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import reddit_china_stance.sol_adjudicator as adjudicator
from reddit_china_stance.human_seeded_consensus_v1 import (
    ADJUDICATION_BLINDNESS_CONTRACT,
    ADJUDICATION_INPUT_KIND,
)


def _input(tmp_path: Path, *, rows: int = 5) -> Path:
    value = {
        "schema_version": "1.0.0",
        "kind": ADJUDICATION_INPUT_KIND,
        "consensus_id": "c" * 64,
        "consensus_artifact_sha256": "a" * 64,
        "source_packet_sha256": "s" * 64,
        "rubric_sha256": "r" * 64,
        "label_schema_sha256": "l" * 64,
        "blindness": ADJUDICATION_BLINDNESS_CONTRACT,
        "rows": [
            {
                "source_sample_id": f"opaque-{index}",
                "target_text": f"synthetic text {index}",
                "submission_context": None,
                "parent_context": None,
            }
            for index in range(rows)
        ],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_transport_schema_omits_unsupported_keywords_but_full_validation_remains() -> None:
    schema = adjudicator._output_schema(["opaque-0"])
    assert not _contains_key(schema, "allOf")
    assert not _contains_key(schema, "uniqueItems")

    with pytest.raises(ValueError, match="should be non-empty"):
        adjudicator._validate_rows(
            {
                "rows": [
                    {
                        "source_sample_id": "opaque-0",
                        "label": {"relevance": "material", "target_stances": []},
                    }
                ]
            },
            item_ids=["opaque-0"],
        )


def test_run_reassembles_shards_in_original_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _input(tmp_path)

    def fake_run_shard(**kwargs: Any) -> Path:
        rows = kwargs["rows"]
        shard_index = kwargs["shard_index"]
        path = tmp_path / f"fragment-{shard_index}.json"
        value = {
            "schema_version": "1.0.0",
            "kind": "human-reference-sol-adjudicator-fragment-v1",
            "input_sha256": adjudicator.file_sha256(input_path),
            "shard_index": shard_index,
            "item_count": len(rows),
            "event_sha256": "e" * 64,
            "rows": [
                {
                    "source_sample_id": row["source_sample_id"],
                    "label": {"relevance": "not_material", "target_stances": []},
                }
                for row in rows
            ],
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    monkeypatch.setattr(adjudicator, "_run_shard", fake_run_shard)
    result = adjudicator.run(
        input_path=input_path,
        private_root=tmp_path / "private",
        shard_count=2,
        jobs=2,
    )
    output = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    assert [row["source_sample_id"] for row in output["rows"]] == [
        f"opaque-{index}" for index in range(5)
    ]
    assert result["rows"] == 5


def test_run_accepts_production_shard_count_but_rejects_contract_drift(
    tmp_path: Path,
) -> None:
    input_path = _input(tmp_path, rows=5)
    with pytest.raises(ValueError, match="shard_count exceeds row count"):
        adjudicator.run(input_path=input_path, shard_count=264, jobs=8)
    with pytest.raises(ValueError, match="shard_count must be"):
        adjudicator.run(
            input_path=input_path,
            shard_count=adjudicator.MAX_SHARDS + 1,
            jobs=8,
        )


def test_immutable_write_never_publishes_a_partial_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence.json"

    def fail_link(*_args: object) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(adjudicator.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        adjudicator._write_immutable(destination, {"status": "complete"})
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_resume_rejects_fragment_without_bound_execution_event(tmp_path: Path) -> None:
    input_path = _input(tmp_path, rows=1)
    private_root = tmp_path / "private"
    input_sha = adjudicator.file_sha256(input_path)
    shard_root = private_root / f"input={input_sha}" / "shard-000"
    fragment = {
        "schema_version": "1.0.0",
        "kind": "human-reference-sol-adjudicator-fragment-v1",
        "input_sha256": input_sha,
        "shard_index": 0,
        "item_count": 1,
        "event_sha256": "e" * 64,
        "rows": [
            {
                "source_sample_id": "opaque-0",
                "label": {"relevance": "not_material", "target_stances": []},
            }
        ],
    }
    adjudicator._write_immutable(shard_root / "fragment.json", fragment)
    with pytest.raises(RuntimeError, match="event binding drifted"):
        adjudicator._run_shard(
            input_path=input_path,
            rows=json.loads(input_path.read_text())["rows"],
            shard_index=0,
            private_root=private_root,
            timeout_seconds=1,
        )


def test_resume_recovers_fragment_from_bound_event_and_output(tmp_path: Path) -> None:
    input_path = _input(tmp_path, rows=1)
    private_root = tmp_path / "private"
    input_sha = adjudicator.file_sha256(input_path)
    shard_root = private_root / f"input={input_sha}" / "shard-000"
    output = {
        "rows": [
            {
                "source_sample_id": "opaque-0",
                "label": {"relevance": "not_material", "target_stances": []},
            }
        ]
    }
    adjudicator._write_immutable(shard_root / "last-message.json", output)
    event = {
        "schema_version": adjudicator.SCHEMA_VERSION,
        "kind": "human-reference-sol-adjudicator-execution-v1",
        "input_sha256": input_sha,
        "shard_index": 0,
        "model": adjudicator.SOL_ADJUDICATOR_MODEL,
        "reasoning_effort": adjudicator.REASONING_EFFORT,
        "instruction_sha256": hashlib.sha256(
            adjudicator.INSTRUCTION.encode()
        ).hexdigest(),
        "rubric_sha256": adjudicator.file_sha256(adjudicator.RUBRIC_PATH),
        "label_schema_sha256": adjudicator.file_sha256(adjudicator.SCHEMA_PATH),
        "thread_id": "synthetic",
        "output_sha256": adjudicator.file_sha256(shard_root / "last-message.json"),
        "elapsed_seconds": 1.0,
        "usage": None,
    }
    adjudicator._write_immutable(shard_root / "event.json", event)

    fragment_path = adjudicator._run_shard(
        input_path=input_path,
        rows=json.loads(input_path.read_text())["rows"],
        shard_index=0,
        private_root=private_root,
        timeout_seconds=1,
    )

    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    assert fragment["rows"] == output["rows"]
    assert fragment["event_sha256"] == adjudicator.file_sha256(
        shard_root / "event.json"
    )
