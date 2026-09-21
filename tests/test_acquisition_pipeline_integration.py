from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import modernbert_acquisition as selector
from reddit_china_stance import modernbert_acquisition_training as training
from reddit_china_stance import sol_teacher_10k_v2 as teacher
from reddit_china_stance import sol_teacher_acquisition_v2 as acquisition_teacher
from reddit_china_stance.semantic_ontology_v2 import (
    ANALYTIC_TARGETS,
    canonical_sha256,
    file_sha256,
)


def _teacher_label() -> dict[str, object]:
    return {
        "codability": "codable",
        "relevance": "material",
        "targets": [{"target": "government_ccp", "stance": "negative"}],
    }


def _not_codable() -> dict[str, object]:
    return {"codability": "not_codable", "relevance": None, "targets": []}


def _rare_cells() -> list[dict[str, object]]:
    return [
        {
            "target": "company_tech_product",
            "stance": "no_directed_stance",
            "training_support": 9,
        },
        {"target": "government_ccp", "stance": "negative", "training_support": 7},
        {"target": "people_identity", "stance": "positive", "training_support": 8},
    ]


def _model_output(index: int, seed: int, *, target_only: bool) -> dict[str, object]:
    context_shift = 0.08 if target_only and index % 3 == 0 else 0.0
    relevance = min(0.95, 0.2 + (index % 6) * 0.11 + seed * 0.02 + context_shift)
    target_presence: dict[str, float] = {}
    stance: dict[str, dict[str, float]] = {}
    for target_index, target in enumerate(ANALYTIC_TARGETS):
        target_presence[target] = min(
            0.95,
            0.15 + ((index + target_index) % 5) * 0.14 + seed * 0.01,
        )
        negative = 0.2 + ((index + target_index + seed) % 3) * 0.1
        stance[target] = {
            "negative": negative,
            "positive": 0.2,
            "mixed": 0.15,
            "no_directed_stance": 0.65 - negative,
        }
    return {
        "relevance": relevance,
        "target_presence": target_presence,
        "stance": stance,
    }


def _candidate(index: int) -> dict[str, object]:
    return {
        "opaque_id": f"candidate-{index:03d}",
        "thread_id": f"candidate-thread-{index:03d}",
        "near_duplicate_cluster_id": f"candidate-cluster-{index:03d}",
        "subreddit": f"sub-{index % 2}",
        "year": 2023 + index % 2,
        "content_type": "comment" if index % 2 else "submission",
        "retrieval_mode": "lexical" if index % 3 else "anchor",
        "seed_outputs": {
            f"seed-{seed}": {
                "full": _model_output(index, seed, target_only=False),
                "target_only": _model_output(index, seed, target_only=True),
            }
            for seed in range(3)
        },
    }


def _frame_row(item_id: str, thread_id: str, *, frame: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "frame": frame,
        "thread_id": thread_id,
        "target_text": "synthetic bounded input",
        "parent_context": None,
        "submission_context": None,
        "label_json": json.dumps(_teacher_label()),
        "selection_component": "base",
        "selection_stratum": None,
        "inclusion_probability_numerator": None,
        "inclusion_probability_denominator": None,
        "inclusion_probability": None,
        "probability_scope": None,
    }


def _factorised_parent_provenance() -> dict[str, object]:
    def descriptor(name: str, digest: str, rows: int | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "relative_path": f"parent/{name}",
            "sha256": digest * 64,
            "bytes": 1,
        }
        if rows is not None:
            value["row_count"] = rows
        return value

    return {
        "experiment_run_id": "a" * 64,
        "phase_run_id": "b" * 64,
        "run_manifest": descriptor("run-manifest.json", "c"),
        "representation_gate": descriptor("representation-gate.json", "d"),
        "gate_receipt_id": "e" * 64,
        "split_manifest": descriptor("membership.json", "f", 10_000),
        "selected_representation": "B4",
        "verdict": "retain_b4",
    }


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
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(teacher, "ACCEPTED_BRIDGE_RECEIPT_ID", receipt_id)
    monkeypatch.setattr(teacher, "ACCEPTED_BRIDGE_RECEIPT_SHA256", file_sha256(receipt_path))
    return receipt_path


def test_selector_teacher_and_training_preparation_form_one_exact_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = acquisition_teacher.DEFAULT_POLICY_PATH
    config_sha = file_sha256(config_path)
    policy = selector.AcquisitionPolicy(
        probability_rows=4,
        rare_cell_rows=1,
        boundary_rows=1,
        multi_context_rows=1,
        uncertainty_disagreement_rows=1,
        probability_seed="integration-probability",
        active_seed="integration-active",
        expected_seed_count=3,
        policy_file_sha256=config_sha,
    )
    rare_cells = _rare_cells()
    bindings = {
        "eligible_frame_sha256": "1" * 64,
        "source_inventory_sha256": "2" * 64,
        "exclusion_ledger_sha256": "3" * 64,
        "checkpoint_bundle_sha256": "4" * 64,
        "scoring_artifact_sha256": "5" * 64,
        "policy_file_sha256": config_sha,
    }
    candidates = [_candidate(index) for index in range(24)]
    ledger, selector_receipt = selector.build_acquisition_ledger(
        candidates,
        rare_cells=rare_cells,
        input_bindings=bindings,
        policy=policy,
    )
    selector.validate_acquisition_ledger(
        ledger,
        selector_receipt,
        candidates,
        rare_cells=rare_cells,
        input_bindings=bindings,
        policy=policy,
    )

    monkeypatch.setattr(acquisition_teacher, "EXPECTED_ROWS", 8)
    monkeypatch.setattr(acquisition_teacher, "ARM_ROWS", 4)
    monkeypatch.setattr(
        acquisition_teacher,
        "ACTIVE_BUCKET_COUNTS",
        {
            "rare_cell": 1,
            "boundary": 1,
            "multi_context": 1,
            "uncertainty_disagreement": 1,
        },
    )
    ledger_path = tmp_path / "acquisition-ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    source_rows: list[dict[str, object]] = []
    for arm_key in ("probability_arm", "active_arm"):
        for row in ledger[arm_key]["rows"]:
            source_rows.append(
                {
                    "opaque_id": row["opaque_id"],
                    "thread_id": row["thread_id"],
                    "target_text": "synthetic bounded input",
                    "submission_context": None,
                    "parent_context": None,
                }
            )
    source_path = tmp_path / "acquisition-source.parquet"
    pq.write_table(pa.Table.from_pylist(source_rows), source_path, compression="zstd")

    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.codex_cli_provenance",
        lambda: {
            "requested_model": "gpt-5.6-sol",
            "codex_cli_binary_sha256": "c" * 64,
            "codex_cli_version": "codex-cli integration",
        },
    )
    bridge_receipt = _write_bridge_receipt(tmp_path, monkeypatch)
    packet_parent = tmp_path / "teacher-packet"
    acquisition_teacher.build_acquisition_packet(
        source_parquet_path=source_path,
        acquisition_ledger_path=ledger_path,
        output_parent=packet_parent,
        bridge_receipt_path=bridge_receipt,
    )
    packet_root = next(packet_parent.glob("packet=*"))
    acquisition_teacher.validate_acquisition_packet(
        packet_root,
        source_parquet_path=source_path,
        acquisition_ledger_path=ledger_path,
        bridge_receipt_path=bridge_receipt,
    )
    private_mapping = pq.read_table(
        packet_root / "private-mapping.parquet"
    ).to_pylist()
    labels = []
    active_seen = 0
    for row in private_mapping:
        label = _teacher_label()
        if row["acquisition_arm"] == "active":
            active_seen += 1
            if active_seen == 1:
                label = _not_codable()
        labels.append({"source_sample_id": row["provider_opaque_id"], "label": label})
    queued = [labels, labels]
    call_index = 0
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )

    def fake_provider(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal call_index
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps({"rows": queued[call_index]}), encoding="utf-8")
        call_index += 1
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                (
                    f'{{"type":"thread.started","thread_id":"integration-{call_index}"}}',
                    '{"type":"turn.completed","usage":{"input_tokens":80,'
                    '"cached_input_tokens":8,"output_tokens":16}}',
                )
            ),
            stderr="",
        )

    monkeypatch.setattr(teacher.subprocess, "run", fake_provider)
    private_root = tmp_path / "teacher-private"
    public_root = tmp_path / "teacher-public"
    teacher_common = {
        "packet_root": packet_root,
        "private_root": private_root,
        "source_parquet_path": source_path,
        "acquisition_ledger_path": ledger_path,
        "bridge_receipt_path": bridge_receipt,
    }
    acquisition_teacher.run_dual_passes(**teacher_common)
    assert acquisition_teacher.run_tie_break(**teacher_common)["tie_break_items"] == 0
    assert acquisition_teacher.run_adjudication(**teacher_common)["adjudication_items"] == 0
    teacher_receipt = acquisition_teacher.finalise_acquisition_labels(
        **teacher_common,
        public_root=public_root,
    )
    assert teacher_receipt == acquisition_teacher.validate_final_acquisition_labels(
        **teacher_common,
        public_root=public_root,
    )
    assert call_index == 2
    run_root, *_ = acquisition_teacher._ensure_run(
        packet_root,
        private_root,
        source_parquet_path=source_path,
        acquisition_ledger_path=ledger_path,
        bridge_receipt_path=bridge_receipt,
    )
    labels_path = run_root / "final/acquisition-labels.parquet"
    teacher_receipt_path = next(public_root.glob("run=*/receipt-*.json"))

    monkeypatch.setattr(training, "BASE_TRAINING_ROWS", 2)
    monkeypatch.setattr(training, "QUERIES_PER_ARM", 4)
    base_path = tmp_path / "base.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _frame_row("base-a", "base-thread-a", frame="training"),
                _frame_row("base-b", "base-thread-b", frame="training"),
            ]
        ),
        base_path,
    )
    checkpoint_rows = [
        _frame_row(
            f"checkpoint-{index:03d}",
            f"checkpoint-thread-{index:03d}",
            frame="development",
        )
        for index in range(training.EVALUATION_ROWS)
    ]
    pq.write_table(pa.Table.from_pylist(checkpoint_rows), checkpoint_path)
    evaluation = [
        {
            **_frame_row(
                f"evaluation-{index:03d}",
                f"evaluation-thread-{index:03d}",
                frame="calibration",
            ),
            "quality_tier": "exact_consensus" if index % 2 == 0 else "blind_majority",
        }
        for index in range(training.EVALUATION_ROWS)
    ]
    pq.write_table(pa.Table.from_pylist(evaluation), evaluation_path)
    prepared = training.prepare_acquisition_training_frames(
        base_training_path=base_path,
        acquisition_source_path=source_path,
        acquisition_labels_path=labels_path,
        acquisition_ledger_path=ledger_path,
        acquisition_teacher_receipt_path=teacher_receipt_path,
        checkpoint_selection_path=checkpoint_path,
        acquisition_evaluation_path=evaluation_path,
        acquisition_config_path=config_path,
        expected_config_sha256=config_sha,
        factorised_parent_provenance=_factorised_parent_provenance(),
        output_root=tmp_path / "prepared",
        descriptor_root=tmp_path,
    )
    assert prepared["primary_eligible_rows_by_arm"] == {"random": 4, "active": 3}
    assert prepared["training_rows_by_arm"] == {"random": 6, "active": 5}
    assert prepared["backfilled_rows"] == 0
    assert prepared["locked_test_rows_accessed"] == 0
    assert prepared["validated_teacher_inputs"]["acquisition_id"] == ledger["ledger_id"]

    preparation_receipt_body = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": "modernbert-acquisition-training-preparation-receipt-v1",
        "preparation_spec_sha256": canonical_sha256(
            {
                "source": file_sha256(source_path),
                "labels": file_sha256(labels_path),
                "ledger": file_sha256(ledger_path),
            }
        ),
        "preparation": prepared,
    }
    preparation_receipt = {
        **preparation_receipt_body,
        "receipt_id": canonical_sha256(preparation_receipt_body),
    }
    preparation_receipt_path = tmp_path / (
        f"preparation-receipt-{preparation_receipt['receipt_id']}.json"
    )
    preparation_receipt_path.write_text(
        json.dumps(preparation_receipt, sort_keys=True), encoding="utf-8"
    )
    validated_preparation_receipt = training.validate_preparation_receipt(
        json.loads(preparation_receipt_path.read_text(encoding="utf-8"))
    )
    assert validated_preparation_receipt["preparation"] == prepared

    _, gate_policy = selector.load_acquisition_policies(config_path)
    experiment = training.freeze_experiment_contract(
        preparation=validated_preparation_receipt["preparation"],
        acquisition_config={
            "repo_relative_path": "configs/modernbert-acquisition-v1.toml",
            "sha256": config_sha,
            "bytes": config_path.stat().st_size,
        },
        gate_policy=gate_policy,
        rare_cells=rare_cells,
        loss_contribution_counts_by_arm=prepared["loss_contribution_counts_by_arm"],
        source_bundle_sha256=canonical_sha256(
            [
                file_sha256(Path("src/reddit_china_stance/modernbert_acquisition.py")),
                file_sha256(
                    Path("src/reddit_china_stance/modernbert_acquisition_training.py")
                ),
                file_sha256(
                    Path("src/reddit_china_stance/sol_teacher_acquisition_v2.py")
                ),
            ]
        ),
        dependency_lock_sha256=file_sha256(Path("uv.lock")),
        rate_card_usd_per_gpu_second="0.000222",
        cumulative_measured_spend_usd="30",
        active_reservation_usd="0",
        planned_phase_upper_usd="25",
    )
    manifest = training.build_run_manifest(experiment)
    validated_manifest = training.validate_run_manifest(manifest)
    assert validated_manifest == manifest
    assert len(manifest["trials"]) == 12
    assert {
        (trial["arm"], trial["component"], trial["optimiser_seed"])
        for trial in manifest["trials"]
    } == {
        (arm, component, seed)
        for arm in training.ARMS
        for component in training.COMPONENTS
        for seed in training.SEEDS
    }
    assert manifest["locked_test_rows_accessed"] == 0
