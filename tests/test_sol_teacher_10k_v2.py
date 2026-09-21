from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import sol_teacher_10k_v2 as teacher
from reddit_china_stance.semantic_ontology_v2 import (
    BLINDED_ROW_FIELDS,
    canonical_sha256,
    file_sha256,
)
from reddit_china_stance.sol_teacher_10k_v2 import (
    ADJUDICATION_PASS,
    MAX_JOBS,
    PASS_NAMES,
    SHARD_SIZE,
    TIE_BREAK_PASS,
    _adjudication_input,
    _ensure_run,
    _run_shard,
    authorise_failed_shard_recovery,
    build_teacher_packet,
    finalise_teacher_labels,
    reconcile_teacher_rows,
    recovery_confirmation_token,
    run_adjudication,
    run_dual_passes,
    run_tie_break,
    validate_final_teacher_labels,
    validate_teacher_packet,
)


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


def _write_source(path: Path, *, rows: int, suffix: str = "") -> None:
    records = [
        {
            "sample_id": f"sample-{index:03d}",
            "thread_id": f"thread-{index:03d}",
            "target_text": f"synthetic private text {index}{suffix}",
            "submission_context": None if index % 2 else f"submission {index}",
            "parent_context": None if index % 3 else f"parent {index}",
        }
        for index in range(rows)
    ]
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


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
    monkeypatch.setattr(
        teacher,
        "ACCEPTED_BRIDGE_RECEIPT_SHA256",
        file_sha256(receipt_path),
    )
    return receipt_path


def _rebind_packet_after_adversarial_rewrite(packet_root: Path) -> None:
    manifest_path = packet_root / "manifest.json"
    blinded_path = packet_root / "blinded-input.json"
    mapping_path = packet_root / "private-mapping.parquet"
    manifest = json.loads(manifest_path.read_text())
    manifest["blinded_input_sha256"] = file_sha256(blinded_path)
    manifest["private_mapping_sha256"] = file_sha256(mapping_path)
    manifest_path.write_bytes(teacher._json_bytes(manifest))
    receipt_path = next(packet_root.glob("receipt-*.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["blinded_input_sha256"] = manifest["blinded_input_sha256"]
    receipt["private_mapping_sha256"] = manifest["private_mapping_sha256"]
    receipt["manifest_sha256"] = file_sha256(manifest_path)
    receipt_path.unlink()
    rebound = packet_root / f"receipt-{canonical_sha256(receipt)}.json"
    rebound.write_bytes(teacher._json_bytes(receipt))


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


def test_packet_conserves_all_threads_and_blinds_v1_surface(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=6)
    bridge_receipt = _write_bridge_receipt(tmp_path, monkeypatch)
    parent = tmp_path / "packets"

    receipt = build_teacher_packet(
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        output_parent=parent,
        expected_rows=6,
    )
    packet_root = next(parent.glob("packet=*"))
    blinded = json.loads((packet_root / "blinded-input.json").read_text())
    mapping = pq.read_table(packet_root / "private-mapping.parquet").to_pylist()

    assert receipt == validate_teacher_packet(
        packet_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=6,
    )
    assert receipt["source_rows"] == receipt["selected_rows"] == receipt["unique_threads"] == 6
    assert receipt["shard_size"] == SHARD_SIZE == 30
    assert receipt["max_jobs"] == MAX_JOBS == 8
    assert all(set(row) == BLINDED_ROW_FIELDS for row in blinded["rows"])
    assert all("label" not in row and "metadata" not in row for row in blinded["rows"])
    assert len({row["thread_id"] for row in mapping}) == 6
    assert len({row["sample_id"] for row in mapping}) == 6
    assert receipt["receipt_contains_raw_text"] is False
    assert receipt["receipt_contains_row_ids"] is False
    assert receipt["receipt_contains_row_level_labels"] is False


def test_packet_validation_preserves_frozen_cli_provenance_across_cli_upgrade(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=3)
    bridge_receipt = _write_bridge_receipt(tmp_path, monkeypatch)
    parent = tmp_path / "packets"
    receipt = build_teacher_packet(
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        output_parent=parent,
        expected_rows=3,
    )
    packet_root = next(parent.glob("packet=*"))

    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.codex_cli_provenance",
        lambda: {
            **cli_provenance,
            "codex_cli_version": "codex-cli upgraded-after-generation",
        },
    )
    assert validate_teacher_packet(
        packet_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=3,
    ) == receipt


def test_packet_rejects_source_and_accepted_receipt_drift(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=4)
    bridge_receipt = _write_bridge_receipt(tmp_path, monkeypatch)
    parent = tmp_path / "packets"
    build_teacher_packet(
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        output_parent=parent,
        expected_rows=4,
    )
    packet_root = next(parent.glob("packet=*"))

    _write_source(source, rows=4, suffix=" drift")
    with pytest.raises(RuntimeError, match="manifest or current source binding"):
        validate_teacher_packet(
            packet_root,
            source_parquet_path=source,
            bridge_receipt_path=bridge_receipt,
            expected_rows=4,
        )
    _write_source(source, rows=4)
    bridge_receipt.write_text(bridge_receipt.read_text() + "\n")
    with pytest.raises(RuntimeError, match="exact authorised receipt"):
        validate_teacher_packet(
            packet_root,
            source_parquet_path=source,
            bridge_receipt_path=bridge_receipt,
            expected_rows=4,
        )


def test_packet_rejects_caller_substituted_generic_bridge_authority(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=2)
    accepted = _write_bridge_receipt(tmp_path, monkeypatch)
    generic = json.loads(accepted.read_text())
    generic["dual_exact_agreement_rate"] = 0.99
    generic_id = canonical_sha256(generic)
    generic_path = tmp_path / f"receipt-{generic_id}.json"
    generic_path.write_text(json.dumps(generic, sort_keys=True))

    with pytest.raises(RuntimeError, match="exact authorised receipt"):
        build_teacher_packet(
            source_parquet_path=source,
            bridge_receipt_path=generic_path,
            output_parent=tmp_path / "packets",
            expected_rows=2,
        )


def test_packet_rejects_coordinated_blinded_and_mapping_rewrite(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=3)
    bridge_receipt = _write_bridge_receipt(tmp_path, monkeypatch)
    parent = tmp_path / "packets"
    build_teacher_packet(
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        output_parent=parent,
        expected_rows=3,
    )
    packet_root = next(parent.glob("packet=*"))
    blinded_path = packet_root / "blinded-input.json"
    blinded = json.loads(blinded_path.read_text())
    blinded["rows"][0]["target_text"] = "coordinated rewritten private text"
    blinded_path.write_bytes(teacher._json_bytes(blinded))
    mapping_path = packet_root / "private-mapping.parquet"
    mapping = pq.read_table(mapping_path).to_pylist()
    mapping[0]["sample_id"] = "coordinated-forged-sample"
    mapping[0]["thread_id"] = "coordinated-forged-thread"
    pq.write_table(
        pa.Table.from_pylist(mapping, schema=teacher._mapping_schema()),
        mapping_path,
        compression="zstd",
    )
    _rebind_packet_after_adversarial_rewrite(packet_root)

    with pytest.raises(RuntimeError, match="exact frozen source reconstruction"):
        validate_teacher_packet(
            packet_root,
            source_parquet_path=source,
            bridge_receipt_path=bridge_receipt,
            expected_rows=3,
        )

def test_reconciliation_tiers_and_training_eligibility_are_exact() -> None:
    review_a = [
        _row("one", _label("china_general", "negative")),
        _row("two", _label("people_identity", "negative")),
        _row("three", _label("culture_media", "negative")),
        _row("four", _not_codable()),
    ]
    review_b = [
        _row("one", _label("china_general", "negative")),
        _row("two", _label("people_identity", "positive")),
        _row("three", _label("culture_media", "positive")),
        _row("four", _not_codable()),
    ]
    tie = [
        _row("two", _label("people_identity", "negative")),
        _row("three", _label("culture_media", "mixed")),
    ]
    adjudication = [_row("three", _label("culture_media", "positive"))]

    final, inventory = reconcile_teacher_rows(
        review_a,
        review_b,
        tie_break=tie,
        adjudication=adjudication,
    )

    assert [row["quality_tier"] for row in final] == [
        "exact_consensus",
        "blind_majority",
        "informed_adjudication",
        "exact_consensus",
    ]
    assert [row["primary_training_eligible"] for row in final] == [True, True, False, False]
    assert inventory["dual_disagreement_ids"] == ["two", "three"]
    assert inventory["blind_majority_ids"] == ["two"]
    assert inventory["three_way_disagreement_ids"] == ["three"]
    with pytest.raises(ValueError, match="every three-way"):
        reconcile_teacher_rows(
            review_a,
            review_b,
            tie_break=tie,
            adjudication=[],
        )


def test_adjudication_input_randomises_candidates_without_reviewer_identity() -> None:
    blinded = [
        {
            "source_sample_id": "three",
            "target_text": "private synthetic text",
            "submission_context": None,
            "parent_context": None,
        }
    ]
    review_a = [_row("three", _label("culture_media", "negative"))]
    review_b = [_row("three", _label("culture_media", "positive"))]
    tie = [_row("three", _label("culture_media", "mixed"))]

    first = _adjudication_input(blinded, review_a, review_b, tie)
    second = _adjudication_input(blinded, review_a, review_b, tie)

    assert first == second
    assert first["item_count"] == 1
    assert len(first["rows"][0]["candidate_labels"]) == 3
    serialised = json.dumps(first)
    assert "review_a" not in serialised
    assert "review_b" not in serialised
    assert len(first["candidate_order_digest"]) == 64


def _run_manifest(provenance: Mapping[str, str]) -> dict[str, object]:
    return {
        "run_id": "exact-run",
        "packet_id": "exact-packet",
        "runtime_source_bundle": teacher._source_bundle(),
        **provenance,
        "reasoning_effort": "high",
        "blinded_input_sha256": "d" * 64,
        "rubric_sha256": file_sha256(
            Path("docs/rubrics/target-stance-v2-pilot.md")
        ),
        "label_schema_sha256": file_sha256(
            Path("schemas/target-stance-v2-pilot.schema.json")
        ),
    }


def test_raw_execution_artefacts_are_transactional_bound_and_reused(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run=exact-run"
    run_root.mkdir()
    (run_root / "run-manifest.json").write_text(json.dumps(_run_manifest(cli_provenance)))
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps({"rows": [_row("opaque-one", _label("china_general", "negative"))]})
        )
        stdout = "\n".join(
            (
                '{"type":"thread.started","thread_id":"thread-raw"}',
                '{"type":"turn.completed","usage":{"input_tokens":11,'
                '"cached_input_tokens":2,"output_tokens":4}}',
            )
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="private stderr evidence")

    monkeypatch.setattr(
        "reddit_china_stance.sol_teacher_10k_v2.subprocess.run",
        fake_run,
    )
    rows = [
        {
            "source_sample_id": "opaque-one",
            "target_text": "private synthetic text",
            "submission_context": None,
            "parent_context": None,
        }
    ]
    fragment = _run_shard(
        run_root=run_root,
        pass_name=PASS_NAMES[0],
        rows=rows,
        shard_index=0,
        informed=False,
    )
    shard_root = fragment.parent

    assert calls == 1
    assert {path.name for path in shard_root.iterdir()} == {
        "event.json",
        "fragment.json",
        "last-message.json",
        "stdout.jsonl",
        "stderr.txt",
    }
    event = json.loads((shard_root / "event.json").read_text())
    assert event["raw_artifact_sha256"] == {
        "last_message": file_sha256(shard_root / "last-message.json"),
        "stdout": file_sha256(shard_root / "stdout.jsonl"),
        "stderr": file_sha256(shard_root / "stderr.txt"),
    }

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact shard reuse must not call the provider")

    monkeypatch.setattr(
        "reddit_china_stance.sol_teacher_10k_v2.subprocess.run",
        forbidden_run,
    )
    assert _run_shard(
        run_root=run_root,
        pass_name=PASS_NAMES[0],
        rows=rows,
        shard_index=0,
        informed=False,
    ) == fragment
    (shard_root / "stdout.jsonl").write_text("tampered private trace")
    with pytest.raises(RuntimeError, match="raw execution artefact hash drifted"):
        _run_shard(
            run_root=run_root,
            pass_name=PASS_NAMES[0],
            rows=rows,
            shard_index=0,
            informed=False,
        )


def test_timeout_preserves_raw_failure_bundle(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run=exact-run"
    run_root.mkdir()
    (run_root / "run-manifest.json").write_text(json.dumps(_run_manifest(cli_provenance)))
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )

    def timed_out(command: list[str], **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            command,
            timeout=123,
            output="private partial stdout",
            stderr="private partial stderr",
        )

    monkeypatch.setattr(
        "reddit_china_stance.sol_teacher_10k_v2.subprocess.run",
        timed_out,
    )
    rows = [
        {
            "source_sample_id": "opaque-one",
            "target_text": "private synthetic text",
            "submission_context": None,
            "parent_context": None,
        }
    ]
    with pytest.raises(RuntimeError, match="timed out; failed attempt preserved"):
        _run_shard(
            run_root=run_root,
            pass_name=PASS_NAMES[0],
            rows=rows,
            shard_index=0,
            informed=False,
        )
    attempt_root = (
        run_root / f"pass={PASS_NAMES[0]}/failures/shard-000/attempt-000"
    )
    marker = json.loads((attempt_root / "failure.json").read_text())
    assert marker["failure_type"] == "timeout"
    assert (attempt_root / "stdout.jsonl").read_text() == "private partial stdout"
    assert (attempt_root / "stderr.txt").read_text() == "private partial stderr"


@pytest.mark.parametrize(
    ("raised", "expected_type"),
    ((KeyboardInterrupt(), "interrupted"), (OSError("invocation failed"), "invocation_error")),
)
def test_invocation_exception_preserves_failure_and_blocks_rerun(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raised: BaseException,
    expected_type: str,
) -> None:
    run_root = tmp_path / "run=exact-run"
    run_root.mkdir()
    (run_root / "run-manifest.json").write_text(json.dumps(_run_manifest(cli_provenance)))
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )
    calls = 0

    def invocation_error(_command: list[str], **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise raised

    monkeypatch.setattr(
        "reddit_china_stance.sol_teacher_10k_v2.subprocess.run",
        invocation_error,
    )
    rows = [
        {
            "source_sample_id": "opaque-one",
            "target_text": "private synthetic text",
            "submission_context": None,
            "parent_context": None,
        }
    ]
    with pytest.raises(RuntimeError, match="failed attempt preserved"):
        _run_shard(
            run_root=run_root,
            pass_name=PASS_NAMES[0],
            rows=rows,
            shard_index=0,
            informed=False,
        )
    attempt_root = (
        run_root / f"pass={PASS_NAMES[0]}/failures/shard-000/attempt-000"
    )
    assert json.loads((attempt_root / "failure.json").read_text())[
        "failure_type"
    ] == expected_type
    with pytest.raises(RuntimeError, match="separately authorise"):
        _run_shard(
            run_root=run_root,
            pass_name=PASS_NAMES[0],
            rows=rows,
            shard_index=0,
            informed=False,
        )
    assert calls == 1


def test_failed_attempt_is_immutable_and_requires_explicit_recovery(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=1)
    bridge_receipt = _write_bridge_receipt(tmp_path, monkeypatch)
    packet_parent = tmp_path / "packets"
    build_teacher_packet(
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        output_parent=packet_parent,
        expected_rows=1,
    )
    packet_root = next(packet_parent.glob("packet=*"))
    private_root = tmp_path / "private"
    run_root, rows = _ensure_run(
        packet_root,
        private_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=1,
    )
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )
    calls = 0

    def failed_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=7,
            stdout='{"type":"thread.started","thread_id":"failed-thread"}',
            stderr="private durable failure evidence",
        )

    monkeypatch.setattr(
        "reddit_china_stance.sol_teacher_10k_v2.subprocess.run",
        failed_run,
    )
    with pytest.raises(RuntimeError, match="failed attempt preserved"):
        _run_shard(
            run_root=run_root,
            pass_name=PASS_NAMES[0],
            rows=rows,
            shard_index=0,
            informed=False,
        )
    attempt_root = (
        run_root / f"pass={PASS_NAMES[0]}/failures/shard-000/attempt-000"
    )
    assert calls == 1
    assert {path.name for path in attempt_root.iterdir()} == {
        "failure.json",
        "last-message.raw",
        "stdout.jsonl",
        "stderr.txt",
    }
    failure = json.loads((attempt_root / "failure.json").read_text())
    assert failure["failure_type"] == "nonzero_exit"
    assert failure["raw_artifact_sha256"] == {
        "last_message": file_sha256(attempt_root / "last-message.raw"),
        "stdout": file_sha256(attempt_root / "stdout.jsonl"),
        "stderr": file_sha256(attempt_root / "stderr.txt"),
    }

    entered = threading.Event()
    release = threading.Event()

    def successful_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "rows": [
                        _row(
                            rows[0]["source_sample_id"],
                            _label("china_general", "negative"),
                        )
                    ]
                }
            )
        )
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                (
                    '{"type":"thread.started","thread_id":"recovered-thread"}',
                    '{"type":"turn.completed","usage":{"input_tokens":9,'
                    '"cached_input_tokens":1,"output_tokens":3}}',
                )
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "reddit_china_stance.sol_teacher_10k_v2.subprocess.run",
        successful_run,
    )
    with pytest.raises(RuntimeError, match="separately authorise"):
        _run_shard(
            run_root=run_root,
            pass_name=PASS_NAMES[0],
            rows=rows,
            shard_index=0,
            informed=False,
        )
    assert calls == 1
    token = recovery_confirmation_token(
        run_id=run_root.name.removeprefix("run="),
        pass_name=PASS_NAMES[0],
        shard_index=0,
        attempt_index=0,
    )
    with pytest.raises(RuntimeError, match="confirmation is invalid"):
        authorise_failed_shard_recovery(
            packet_root=packet_root,
            pass_name=PASS_NAMES[0],
            shard_index=0,
            attempt_index=0,
            confirmation="wrong-token",
            private_root=private_root,
            source_parquet_path=source,
            bridge_receipt_path=bridge_receipt,
            expected_rows=1,
        )
    authorised = authorise_failed_shard_recovery(
        packet_root=packet_root,
        pass_name=PASS_NAMES[0],
        shard_index=0,
        attempt_index=0,
        confirmation=token,
        private_root=private_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=1,
    )
    assert authorised["provider_calls_made"] == 0
    call_kwargs = {
        "run_root": run_root,
        "pass_name": PASS_NAMES[0],
        "rows": rows,
        "shard_index": 0,
        "informed": False,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_run_shard, **call_kwargs)
        assert entered.wait(timeout=5)
        second = pool.submit(_run_shard, **call_kwargs)
        with pytest.raises(RuntimeError, match="active or orphaned provider-attempt claim"):
            second.result(timeout=5)
        release.set()
        fragment = first.result(timeout=5)
    assert fragment.is_file()
    assert calls == 2
    assert attempt_root.is_dir()


def test_end_to_end_small_run_finalises_private_tiers_and_public_metadata(
    cli_provenance: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source(source, rows=4)
    bridge_receipt = _write_bridge_receipt(tmp_path, monkeypatch)
    packet_parent = tmp_path / "packets"
    build_teacher_packet(
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        output_parent=packet_parent,
        expected_rows=4,
    )
    packet_root = next(packet_parent.glob("packet=*"))
    opaque_ids = [
        row["source_sample_id"]
        for row in json.loads((packet_root / "blinded-input.json").read_text())["rows"]
    ]
    a = [
        _row(opaque_ids[0], _label("china_general", "negative")),
        _row(opaque_ids[1], _label("people_identity", "negative")),
        _row(opaque_ids[2], _label("culture_media", "negative")),
        _row(opaque_ids[3], _not_codable()),
    ]
    b = [
        _row(opaque_ids[0], _label("china_general", "negative")),
        _row(opaque_ids[1], _label("people_identity", "positive")),
        _row(opaque_ids[2], _label("culture_media", "positive")),
        _row(opaque_ids[3], _not_codable()),
    ]
    c = [
        _row(opaque_ids[1], _label("people_identity", "negative")),
        _row(opaque_ids[2], _label("culture_media", "mixed")),
    ]
    adjudication = [_row(opaque_ids[2], _label("culture_media", "positive"))]
    queued = [a, b, c, adjudication]
    call_index = 0
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal call_index
        rows = queued[call_index]
        call_index += 1
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps({"rows": rows}))
        stdout = "\n".join(
            (
                f'{{"type":"thread.started","thread_id":"thread-{call_index}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":100,'
                '"cached_input_tokens":10,"output_tokens":20}}',
            )
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        "reddit_china_stance.sol_teacher_10k_v2.subprocess.run",
        fake_run,
    )
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"

    assert run_dual_passes(
        packet_root=packet_root,
        private_root=private_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=4,
    )["pass_item_counts"] == {PASS_NAMES[0]: 4, PASS_NAMES[1]: 4}
    assert run_tie_break(
        packet_root=packet_root,
        private_root=private_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=4,
    )["tie_break_items"] == 2
    adjudication_result = run_adjudication(
        packet_root=packet_root,
        private_root=private_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=4,
    )
    assert adjudication_result["adjudication_items"] == 1
    assert len(adjudication_result["candidate_order_digest"]) == 64
    assert call_index == 4

    receipt = finalise_teacher_labels(
        packet_root=packet_root,
        private_root=private_root,
        public_root=public_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=4,
    )
    repeated = finalise_teacher_labels(
        packet_root=packet_root,
        private_root=private_root,
        public_root=public_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=4,
    )
    validated = validate_final_teacher_labels(
        packet_root=packet_root,
        private_root=private_root,
        public_root=public_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=4,
    )

    assert receipt == repeated == validated
    assert call_index == 4
    assert receipt["quality_tier_counts"] == {
        "exact_consensus": 2,
        "blind_majority": 1,
        "informed_adjudication": 1,
    }
    assert receipt["primary_training_eligible_count"] == 2
    assert receipt["primary_training_ineligible_count"] == 2
    assert receipt["blind_reconciliation"]["three_way_disagreement_count"] == 1
    assert receipt["automatic_retry_count"] == 0
    serialised = json.dumps(receipt)
    assert not any(item_id in serialised for item_id in opaque_ids)
    assert "target_text" not in serialised
    assert receipt["receipt_contains_raw_text"] is False
    run_root, _ = _ensure_run(
        packet_root,
        private_root,
        source_parquet_path=source,
        bridge_receipt_path=bridge_receipt,
        expected_rows=4,
    )
    labels_path = run_root / "final/labels.parquet"
    assert file_sha256(labels_path) == receipt["private_labels_parquet_sha256"]
    final_rows = pq.read_table(labels_path).to_pylist()
    assert [row["quality_tier"] for row in final_rows] == [
        "exact_consensus",
        "blind_majority",
        "informed_adjudication",
        "exact_consensus",
    ]
    assert [row["primary_training_eligible"] for row in final_rows] == [
        True,
        True,
        False,
        False,
    ]
    assert len(list(public_root.glob("run=*/receipt-*.json"))) == 1
    assert len(list(run_root.glob("pass=*/shard-*/stdout.jsonl"))) == 4
    assert len(list(run_root.glob("pass=*/shard-*/last-message.json"))) == 4
    assert len(list(run_root.glob("pass=*/shard-*/stderr.txt"))) == 4
    assert (run_root / f"pass={TIE_BREAK_PASS}/pass-output.json").is_file()
    assert (run_root / f"pass={ADJUDICATION_PASS}/pass-output.json").is_file()
