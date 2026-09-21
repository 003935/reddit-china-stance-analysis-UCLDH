from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import sol_teacher_10k_v2 as teacher
from reddit_china_stance import sol_teacher_acquisition_v2 as acquisition
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256


def _label(target: str, stance: str) -> dict[str, object]:
    return {
        "codability": "codable",
        "relevance": "material",
        "targets": [{"target": target, "stance": stance}],
    }


def _not_codable() -> dict[str, object]:
    return {"codability": "not_codable", "relevance": None, "targets": []}


def _row(item_id: str, label: Mapping[str, object]) -> dict[str, object]:
    return {"source_sample_id": item_id, "label": dict(label)}


@pytest.fixture(autouse=True)
def small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acquisition, "EXPECTED_ROWS", 8)
    monkeypatch.setattr(acquisition, "ARM_ROWS", 4)
    monkeypatch.setattr(
        acquisition,
        "ACTIVE_BUCKET_COUNTS",
        {
            "rare_cell": 1,
            "boundary": 1,
            "multi_context": 1,
            "uncertainty_disagreement": 1,
        },
    )


@pytest.fixture
def cli_provenance(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    provenance = {
        "requested_model": "gpt-5.6-sol",
        "codex_cli_binary_sha256": "c" * 64,
        "codex_cli_version": "codex-cli test",
    }
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.codex_cli_provenance",
        lambda: dict(provenance),
    )
    return provenance


def _write_bridge_receipt(path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    receipt = {
        "schema_version": "1.0.0",
        "kind": "semantic-ontology-v2-bridge-receipt-v1",
        "status": "complete",
        "run_id": teacher.ACCEPTED_BRIDGE_RUN_ID,
        "packet_id": teacher.ACCEPTED_BRIDGE_PACKET_ID,
        "row_count": 480,
        "dual_exact_agreement_rate": 0.7541666666666667,
        "engineering_verdict": "pass",
        "gate_results": {"all_registered_gates": True},
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "evidence_boundary": "model-assisted-ontology-development-not-human-validation",
    }
    receipt_id = canonical_sha256(receipt)
    receipt_path = path / f"receipt-{receipt_id}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    monkeypatch.setattr(teacher, "ACCEPTED_BRIDGE_RECEIPT_ID", receipt_id)
    monkeypatch.setattr(teacher, "ACCEPTED_BRIDGE_RECEIPT_SHA256", file_sha256(receipt_path))
    return receipt_path


def _stratum() -> dict[str, object]:
    return {
        "subreddit": "synthetic",
        "year": 2024,
        "content_type": "comment",
        "retrieval_mode": "lexical",
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    probability_rows = [
        {
            "opaque_id": f"private-{index}",
            "thread_id": f"thread-{index}",
            "near_duplicate_cluster_id": f"cluster-{index}",
            "stratum": _stratum(),
            "inclusion_probability_numerator": 4,
            "inclusion_probability_denominator": 40,
            "inclusion_probability": 0.1,
            "selection_tiebreak_sha256": canonical_sha256(["random", index]),
        }
        for index in range(4)
    ]
    probability_rows.sort(key=lambda row: row["selection_tiebreak_sha256"])
    active_rows = [
        {
            "opaque_id": f"private-{index + 4}",
            "thread_id": f"thread-{index + 4}",
            "near_duplicate_cluster_id": f"cluster-{index + 4}",
            "bucket": bucket,
            "bucket_rank": 1,
            "score": 0.9 - index * 0.1,
            "selection_tiebreak_sha256": canonical_sha256(["active", index]),
        }
        for index, bucket in enumerate(acquisition.ACTIVE_BUCKETS)
    ]
    policy = {"seed": "synthetic", "query_rows_per_arm": 4}
    ledger: dict[str, object] = {
        "schema_version": acquisition.SCHEMA_VERSION,
        "kind": acquisition.LEDGER_KIND,
        "policy": policy,
        "policy_file_sha256": file_sha256(acquisition.DEFAULT_POLICY_PATH),
        "policy_contract_sha256": canonical_sha256(policy),
        "input_bindings": {
            "eligible_frame_sha256": "1" * 64,
            "source_inventory_sha256": "2" * 64,
            "exclusion_ledger_sha256": "3" * 64,
            "checkpoint_bundle_sha256": "4" * 64,
            "scoring_artifact_sha256": "5" * 64,
            "policy_file_sha256": file_sha256(acquisition.DEFAULT_POLICY_PATH),
        },
        "rare_cells": [{"target": "china_general", "stance": "positive"}],
        "rare_cell_list_sha256": canonical_sha256(
            [{"target": "china_general", "stance": "positive"}]
        ),
        "eligible_population_rows": 80,
        "candidate_score_digest": "6" * 64,
        "probability_arm": {
            "row_count": 4,
            "strata": [
                {"stratum": _stratum(), "population_rows": 40, "sample_rows": 4}
            ],
            "rows": probability_rows,
        },
        "active_arm": {
            "row_count": 4,
            "buckets": dict(acquisition.ACTIVE_BUCKET_COUNTS),
            "rows": active_rows,
        },
    }
    ledger["ledger_id"] = canonical_sha256(ledger)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True))
    source_path = tmp_path / "source.parquet"
    source_rows = [
        {
            "opaque_id": row["opaque_id"],
            "thread_id": row["thread_id"],
            "target_text": f"synthetic private text {index}",
            "submission_context": None if index % 2 else f"submission {index}",
            "parent_context": None if index % 3 else f"parent {index}",
        }
        for index, row in enumerate([*probability_rows, *active_rows])
    ]
    pq.write_table(pa.Table.from_pylist(source_rows), source_path, compression="zstd")
    return source_path, ledger_path, ledger


def test_packet_binds_design_and_blinds_balanced_interleave(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, ledger, frozen = _write_inputs(tmp_path)
    bridge = _write_bridge_receipt(tmp_path, monkeypatch)
    parent = tmp_path / "packet"
    receipt = acquisition.build_acquisition_packet(
        source_parquet_path=source,
        acquisition_ledger_path=ledger,
        bridge_receipt_path=bridge,
        output_parent=parent,
    )
    packet_root = next(parent.glob("packet=*"))
    provider_packet = acquisition._provider_packet_root(packet_root)
    blinded = json.loads((provider_packet / "blinded-input.json").read_text())["rows"]
    mapping = pq.read_table(packet_root / "private-mapping.parquet").to_pylist()

    assert receipt == acquisition.validate_acquisition_packet(
        packet_root,
        source_parquet_path=source,
        acquisition_ledger_path=ledger,
        bridge_receipt_path=bridge,
    )
    assert receipt["acquisition_id"] == frozen["ledger_id"]
    assert receipt["arm_counts"] == {"probability_random": 4, "active": 4}
    assert [row["packet_order"] for row in mapping] == list(range(8))
    assert all(
        {mapping[2 * index]["acquisition_arm"], mapping[2 * index + 1]["acquisition_arm"]}
        == set(acquisition.ARMS)
        for index in range(4)
    )
    assert all("acquisition_arm" not in row and "active_bucket" not in row for row in blinded)
    assert receipt["receipt_contains_row_ids"] is False
    assert receipt["row_replacement_count"] == 0


def test_ledger_rejects_extra_fields_probability_drift_and_cluster_overlap(
    tmp_path: Path,
) -> None:
    _, ledger_path, ledger = _write_inputs(tmp_path)
    ledger["unexpected"] = True
    ledger_path.write_text(json.dumps(ledger, sort_keys=True))
    with pytest.raises(ValueError, match="top-level fields drifted"):
        acquisition.validate_acquisition_ledger(ledger_path)

    _, ledger_path, ledger = _write_inputs(tmp_path)
    ledger["probability_arm"]["rows"][0]["inclusion_probability"] = 0.2
    body = {key: value for key, value in ledger.items() if key != "ledger_id"}
    ledger["ledger_id"] = canonical_sha256(body)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True))
    with pytest.raises(ValueError, match="inclusion probability is inconsistent"):
        acquisition.validate_acquisition_ledger(ledger_path)

    _, ledger_path, ledger = _write_inputs(tmp_path)
    ledger["active_arm"]["rows"][0]["near_duplicate_cluster_id"] = "cluster-0"
    body = {key: value for key, value in ledger.items() if key != "ledger_id"}
    ledger["ledger_id"] = canonical_sha256(body)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True))
    with pytest.raises(ValueError, match="overlapping near-duplicate"):
        acquisition.validate_acquisition_ledger(ledger_path)


def test_source_must_match_exact_ledger_order(tmp_path: Path) -> None:
    source, ledger, _ = _write_inputs(tmp_path)
    table = pq.read_table(source)
    rows = table.to_pylist()
    rows[0], rows[1] = rows[1], rows[0]
    pq.write_table(pa.Table.from_pylist(rows), source, compression="zstd")
    with pytest.raises(RuntimeError, match="frozen ledger order"):
        acquisition._validate_source_and_ledger(source, ledger)


def test_end_to_end_reuses_provider_engine_and_publishes_per_arm_aggregates(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, ledger, _ = _write_inputs(tmp_path)
    bridge = _write_bridge_receipt(tmp_path, monkeypatch)
    parent = tmp_path / "packet"
    acquisition.build_acquisition_packet(
        source_parquet_path=source,
        acquisition_ledger_path=ledger,
        bridge_receipt_path=bridge,
        output_parent=parent,
    )
    packet_root = next(parent.glob("packet=*"))
    provider_packet = acquisition._provider_packet_root(packet_root)
    opaque_ids = [
        row["source_sample_id"]
        for row in json.loads((provider_packet / "blinded-input.json").read_text())["rows"]
    ]
    a = [
        _row(item_id, _not_codable() if index == 7 else _label("china_general", "negative"))
        for index, item_id in enumerate(opaque_ids)
    ]
    b = list(a)
    b[1] = _row(opaque_ids[1], _label("china_general", "positive"))
    b[2] = _row(opaque_ids[2], _label("china_general", "positive"))
    c = [
        _row(opaque_ids[1], _label("china_general", "negative")),
        _row(opaque_ids[2], _label("china_general", "mixed")),
    ]
    adjudication = [_row(opaque_ids[2], _label("china_general", "positive"))]
    queued = [a, b, c, adjudication]
    call_index = 0
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal call_index
        output = queued[call_index]
        call_index += 1
        path = Path(command[command.index("--output-last-message") + 1])
        path.write_text(json.dumps({"rows": output}))
        stdout = "\n".join(
            (
                f'{{"type":"thread.started","thread_id":"task-{call_index}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":100,'
                '"cached_input_tokens":10,"output_tokens":20}}',
            )
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(teacher.subprocess, "run", fake_run)
    private = tmp_path / "private"
    public = tmp_path / "public"
    common = {
        "packet_root": packet_root,
        "private_root": private,
        "source_parquet_path": source,
        "acquisition_ledger_path": ledger,
        "bridge_receipt_path": bridge,
    }
    assert acquisition.run_dual_passes(**common)["pass_item_counts"] == {
        "review_a": 8,
        "review_b": 8,
    }
    assert acquisition.run_tie_break(**common)["tie_break_items"] == 2
    assert acquisition.run_adjudication(**common)["adjudication_items"] == 1
    receipt = acquisition.finalise_acquisition_labels(
        **common, public_root=public
    )
    validated = acquisition.validate_final_acquisition_labels(
        **common, public_root=public
    )

    assert receipt == validated
    assert call_index == 4
    assert receipt["row_count"] == 8
    assert set(receipt["arm_diagnostics"]) == set(acquisition.ARMS)
    assert sum(
        arm["primary_training_eligible_count"]
        for arm in receipt["arm_diagnostics"].values()
    ) == 6
    assert sum(
        arm["informed_adjudication_count"]
        for arm in receipt["arm_diagnostics"].values()
    ) == 1
    assert all(
        arm["provider_telemetry_attribution"]["available"] is False
        and arm["row_replacement_count"] == 0
        for arm in receipt["arm_diagnostics"].values()
    )
    assert receipt["global_provider_telemetry"]["execution_count"] == 4
    assert receipt["row_replacement_count"] == 0
    serialised = json.dumps(receipt)
    assert not any(item_id in serialised for item_id in opaque_ids)
    assert "target_text" not in serialised
    run_root, *_ = acquisition._ensure_run(
        packet_root,
        private,
        source_parquet_path=source,
        acquisition_ledger_path=ledger,
        bridge_receipt_path=bridge,
    )
    final = pq.read_table(run_root / "final/acquisition-labels.parquet")
    assert final.num_rows == 8
    assert final.schema.metadata[b"kind"] == b"sol-teacher-acquisition-v2-final-labels-v1"
    assert len(list(public.glob("run=*/receipt-*.json"))) == 1
