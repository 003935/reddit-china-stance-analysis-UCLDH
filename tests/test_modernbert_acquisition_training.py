from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import modernbert_acquisition as acquisition
from reddit_china_stance import modernbert_acquisition_training as training
from reddit_china_stance import sol_teacher_acquisition_v2 as acquisition_teacher


def _descriptor(
    name: str,
    digest: str,
    *,
    rows: int | None = None,
    frame: str | None = None,
    thread_digest: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "relative_path": f"inputs/{name}",
        "sha256": digest * 64,
        "bytes": 123,
    }
    if rows is not None:
        result["row_count"] = rows
    if frame is not None:
        result["frame"] = frame
    if thread_digest is not None:
        result["thread_set_sha256"] = thread_digest
    return result


def _gate_policy() -> acquisition.AcquisitionGatePolicy:
    _, gate = acquisition.load_acquisition_policies(
        Path("configs/modernbert-acquisition-v1.toml")
    )
    return gate


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


def _factorised_parent_provenance() -> dict[str, object]:
    return {
        "experiment_run_id": "e" * 64,
        "phase_run_id": "f" * 64,
        "run_manifest": _descriptor("factorised-run.json", "a"),
        "representation_gate": _descriptor("representation-gate.json", "b"),
        "gate_receipt_id": "c" * 64,
        "split_manifest": _descriptor("membership.json", "d", rows=10_000),
        "selected_representation": "B4",
        "verdict": "retain_b4",
    }


def _preparation_and_contributions() -> tuple[dict, dict]:
    gate = _gate_policy()
    contributions = {
        arm: {
            "training_rows": 8_200 + index,
            "relevance_bce_rows": 8_100,
            "target_presence_bce_rows": 6_000,
            "stance_cross_entropy_rows_by_target": {
                target: 100 for target in training.ANALYTIC_TARGET_CLASSES
            },
            "stance_cross_entropy_total": 500,
        }
        for index, arm in enumerate(training.ARMS)
    }
    provenance = {
        "source_artifact": _descriptor("source.parquet", "5", rows=2_000),
        "label_artifact": _descriptor("labels.parquet", "6", rows=2_000),
        "ledger_artifact": _descriptor("ledger.json", "7", rows=2_000),
        "teacher_receipt_artifact": _descriptor("teacher.json", "8"),
        "teacher_receipt_canonical_sha256": "9" * 64,
        "teacher_run_id": "a" * 64,
        "teacher_packet_id": "b" * 64,
        "acquisition_id": "c" * 64,
        "policy_file_sha256": gate.policy_file_sha256,
        "policy_contract_sha256": "d" * 64,
        "rare_cells": _rare_cells(),
        "rare_cell_list_sha256": training.canonical_sha256(_rare_cells()),
    }
    preparation_body = {
        "schema_version": training.SCHEMA_VERSION,
        "kind": training.PREPARATION_KIND,
        "base_training_rows": training.BASE_TRAINING_ROWS,
        "queried_rows_by_arm": {arm: training.QUERIES_PER_ARM for arm in training.ARMS},
        "primary_eligible_rows_by_arm": {"random": 53, "active": 54},
        "training_rows_by_arm": {"random": 8_200, "active": 8_201},
        "quality_tier_counts_by_arm": {
            arm: {"exact_consensus": training.QUERIES_PER_ARM}
            for arm in training.ARMS
        },
        "loss_contribution_counts_by_arm": contributions,
        "training_frames": {
            "random": _descriptor(
                "random.parquet", "1", rows=8_200, frame="training",
                thread_digest="a" * 64,
            ),
            "active": _descriptor(
                "active.parquet", "2", rows=8_201, frame="training",
                thread_digest="b" * 64,
            ),
        },
        "checkpoint_selection_frame": _descriptor(
            "checkpoint.parquet", "3", rows=600, frame="development",
            thread_digest="c" * 64,
        ),
        "acquisition_evaluation_frame": _descriptor(
            "evaluation.parquet", "4", rows=600, frame="acquisition_evaluation",
            thread_digest="d" * 64,
        ),
        "validated_teacher_inputs": provenance,
        "factorised_parent_provenance": _factorised_parent_provenance(),
        "thread_overlap_counts": {
            "base_checkpoint_selection": 0,
            "base_acquisition_evaluation": 0,
            "checkpoint_selection_acquisition_evaluation": 0,
            "random_active_additions": 0,
            "additions_checkpoint_selection": 0,
            "additions_acquisition_evaluation": 0,
        },
        "backfilled_rows": 0,
        "acquisition_config_sha256": gate.policy_file_sha256,
        "locked_test_rows_accessed": 0,
    }
    preparation = {
        **preparation_body,
        "preparation_id": training.canonical_sha256(preparation_body),
    }
    return preparation, contributions


def _contract() -> dict:
    gate = _gate_policy()
    preparation, contributions = _preparation_and_contributions()
    return training.freeze_experiment_contract(
        preparation=preparation,
        acquisition_config={
            "repo_relative_path": "configs/modernbert-acquisition-v1.toml",
            "sha256": gate.policy_file_sha256,
            "bytes": Path("configs/modernbert-acquisition-v1.toml").stat().st_size,
        },
        gate_policy=gate,
        rare_cells=_rare_cells(),
        loss_contribution_counts_by_arm=contributions,
        source_bundle_sha256="6" * 64,
        dependency_lock_sha256="7" * 64,
        rate_card_usd_per_gpu_second="0.000222",
        cumulative_measured_spend_usd="30",
        active_reservation_usd="0",
        planned_phase_upper_usd="25",
    )


def _label(*, material: bool = True) -> dict:
    return {
        "codability": "codable",
        "relevance": "material" if material else "not_material",
        "targets": (
            [{"target": "government_ccp", "stance": "negative"}]
            if material
            else []
        ),
    }


def _base_row(item_id: str, thread_id: str, *, frame: str = "training") -> dict:
    return {
        "item_id": item_id,
        "frame": frame,
        "thread_id": thread_id,
        "target_text": "bounded target",
        "parent_context": None,
        "submission_context": None,
        "label_json": json.dumps(_label()),
        "selection_component": "base",
        "selection_stratum": None,
        "inclusion_probability_numerator": None,
        "inclusion_probability_denominator": None,
        "inclusion_probability": None,
        "probability_scope": None,
    }


def test_load_private_frame_accepts_exact_acquisition_evaluation_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "acquisition-evaluation.parquet"
    row = {
        **_base_row("eval-1", "eval-thread-1", frame="acquisition_evaluation"),
        "quality_tier": "exact_consensus",
    }
    pq.write_table(pa.Table.from_pylist([row]), path)
    descriptor = {
        "relative_path": path.name,
        "sha256": training.file_sha256(path),
        "bytes": path.stat().st_size,
        "row_count": 1,
        "frame": "acquisition_evaluation",
    }

    rows, reference = training.load_private_frame(
        path,
        descriptor,
        expected_frame="acquisition_evaluation",
    )

    assert rows[0]["quality_tier"] == "exact_consensus"
    assert reference == {"eval-1": _label()}
    with pytest.raises(
        training.AcquisitionTrainingContractError,
        match="descriptor identity drifted",
    ):
        training.load_private_frame(
            path,
            {**descriptor, "frame": "development"},
            expected_frame="acquisition_evaluation",
        )


def _teacher_receipt(
    *, source_path: Path, labels_path: Path, ledger_path: Path, config_path: Path
) -> dict:
    ledger_sha = training.file_sha256(ledger_path)
    config_sha = training.file_sha256(config_path)
    receipt = {key: "e" * 64 for key in training._TEACHER_RECEIPT_KEYS}
    receipt.update(
        {
            "schema_version": "1.0.0",
            "kind": "sol-teacher-acquisition-v2-receipt-v1",
            "status": "complete",
            "run_id": "1" * 64,
            "packet_id": "2" * 64,
            "acquisition_id": "3" * 64,
            "row_count": 4,
            "arm_counts": {"probability_random": 2, "active": 2},
            "arm_diagnostics": {},
            "blind_reconciliation": {},
            "global_provider_telemetry": {},
            "source_parquet_sha256": training.file_sha256(source_path),
            "acquisition_ledger_sha256": ledger_sha,
            "policy_file_sha256": config_sha,
            "policy_contract_sha256": "4" * 64,
            "eligible_frame_sha256": "5" * 64,
            "source_inventory_sha256": "6" * 64,
            "exclusion_ledger_sha256": "7" * 64,
            "checkpoint_bundle_sha256": "8" * 64,
            "scoring_artifact_sha256": "9" * 64,
            "candidate_score_digest": "a" * 64,
            "interleave_order_sha256": training.canonical_sha256(
                ["new-0", "new-2", "new-1", "new-3"]
            ),
            "private_labels_parquet_sha256": training.file_sha256(labels_path),
            "automatic_retry_count": 0,
            "row_replacement_count": 0,
            "receipt_contains_raw_text": False,
            "receipt_contains_row_ids": False,
            "receipt_contains_thread_ids": False,
            "receipt_contains_row_level_labels": False,
            "evidence_boundary": {},
        }
    )
    return receipt


def test_manifest_is_exact_two_arms_by_two_components_by_three_seeds() -> None:
    manifest = training.build_run_manifest(_contract())
    clean = training.validate_run_manifest(manifest)
    assert len(clean["trials"]) == 12
    assert {
        (trial["arm"], trial["component"], trial["optimiser_seed"])
        for trial in clean["trials"]
    } == {
        (arm, component, seed)
        for arm in training.ARMS
        for component in training.COMPONENTS
        for seed in training.SEEDS
    }
    assert all("target_stance_b2" not in json.dumps(trial) for trial in clean["trials"])
    assert all(
        trial["max_gpu_seconds"] == training.TRIAL_MAX_GPU_SECONDS
        and trial["checkpoint_selection_frame_sha256"] != trial[
            "acquisition_evaluation_frame_sha256"
        ]
        for trial in clean["trials"]
    )
    assert clean["locked_test_rows_accessed"] == 0


def test_manifest_binds_policy_digest_and_rejects_loss_count_drift() -> None:
    contract = _contract()
    assert (
        contract["registered_design"]["gate_policy"]["policy_file_sha256"]
        == contract["bindings"]["acquisition_config"]["sha256"]
    )
    contract["bindings"]["loss_contribution_counts_by_arm"]["active"][
        "training_rows"
    ] += 1
    with pytest.raises(training.AcquisitionTrainingContractError):
        training.validate_experiment_contract(contract)


def test_preparation_requires_exact_three_canonical_rare_cells() -> None:
    preparation, _ = _preparation_and_contributions()
    provenance = preparation["validated_teacher_inputs"]
    provenance["rare_cells"] = provenance["rare_cells"][:2]
    provenance["rare_cell_list_sha256"] = training.canonical_sha256(
        provenance["rare_cells"]
    )
    body = {
        key: value for key, value in preparation.items() if key != "preparation_id"
    }
    preparation["preparation_id"] = training.canonical_sha256(body)
    with pytest.raises(
        training.AcquisitionTrainingContractError,
        match="rare-cell",
    ):
        training.validate_preparation_summary(preparation)

    preparation, _ = _preparation_and_contributions()
    preparation["validated_teacher_inputs"]["rare_cell_list_sha256"] = "f" * 64
    body = {
        key: value for key, value in preparation.items() if key != "preparation_id"
    }
    preparation["preparation_id"] = training.canonical_sha256(body)
    with pytest.raises(
        training.AcquisitionTrainingContractError,
        match="rare-cell",
    ):
        training.validate_preparation_summary(preparation)


def test_loss_contribution_counts_are_per_objective() -> None:
    rows = [
        _base_row("a", "ta"),
        {**_base_row("b", "tb"), "label_json": json.dumps(_label(material=False))},
        {
            **_base_row("c", "tc"),
            "label_json": json.dumps(
                {"codability": "not_codable", "relevance": None, "targets": []}
            ),
        },
    ]
    counts = training.loss_contribution_counts(rows)
    assert counts["training_rows"] == 3
    assert counts["relevance_bce_rows"] == 2
    assert counts["target_presence_bce_rows"] == 1
    assert counts["stance_cross_entropy_total"] == 1
    assert counts["stance_cross_entropy_rows_by_target"]["government_ccp"] == 1


def test_prepare_joins_teacher_labels_keeps_unequal_usable_yield_and_never_backfills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "BASE_TRAINING_ROWS", 2)
    monkeypatch.setattr(training, "EVALUATION_ROWS", 2)
    monkeypatch.setattr(training, "QUERIES_PER_ARM", 2)
    monkeypatch.setattr(acquisition_teacher, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(acquisition_teacher, "ARM_ROWS", 2)
    base_rows = [_base_row("base-a", "thread-base-a"), _base_row("base-b", "thread-base-b")]
    source_rows = []
    for index in range(4):
        sample = f"new-{index}"
        source_rows.append(
            {
                "opaque_id": sample,
                "thread_id": f"thread-{sample}",
                "target_text": "bounded target",
                "parent_context": None,
                "submission_context": None,
            }
        )
    interleave_indices = (0, 2, 1, 3)
    label_rows = [
        {
            "source_sample_id": f"new-{index}",
            "thread_id": f"thread-new-{index}",
            "acquisition_arm": (
                "probability_random" if index < 2 else "active"
            ),
            "arm_order": index % 2,
            "packet_order": packet_order,
            "codability": "codable",
            "relevance": "material",
            "label_json": json.dumps(_label()),
            "quality_tier": "exact_consensus",
            "primary_training_eligible": index in {0, 2, 3},
        }
        for packet_order, index in enumerate(interleave_indices)
    ]
    evaluation_rows = [
        {
            **_base_row("eval-a", "thread-eval-a", frame="calibration"),
            "quality_tier": "exact_consensus",
        },
        {
            **_base_row("eval-b", "thread-eval-b", frame="calibration"),
            "quality_tier": "blind_majority",
        },
    ]
    base_path = tmp_path / "base.parquet"
    source_path = tmp_path / "source.parquet"
    labels_path = tmp_path / "labels.parquet"
    ledger_path = tmp_path / "ledger.json"
    evaluation_path = tmp_path / "evaluation.parquet"
    checkpoint_path = tmp_path / "checkpoint.parquet"
    pq.write_table(pa.Table.from_pylist(base_rows), base_path)
    pq.write_table(pa.Table.from_pylist(source_rows), source_path)
    ledger_path.write_text("{}\n", encoding="utf-8")
    label_schema = acquisition_teacher._combined_schema(
        {
            "run_id": "1" * 64,
            "packet_id": "2" * 64,
            "acquisition_id": "3" * 64,
            "acquisition_ledger_sha256": training.file_sha256(ledger_path),
        }
    )
    pq.write_table(pa.Table.from_pylist(label_rows, schema=label_schema), labels_path)
    pq.write_table(pa.Table.from_pylist(evaluation_rows), evaluation_path)
    checkpoint_rows = [
        _base_row("checkpoint-a", "thread-checkpoint-a", frame="development"),
        _base_row("checkpoint-b", "thread-checkpoint-b", frame="development"),
    ]
    pq.write_table(pa.Table.from_pylist(checkpoint_rows), checkpoint_path)
    config = Path("configs/modernbert-acquisition-v1.toml")
    ledger = {
        "ledger_id": "3" * 64,
        "policy_file_sha256": training.file_sha256(config),
        "policy_contract_sha256": "4" * 64,
        "candidate_score_digest": "a" * 64,
        "rare_cells": _rare_cells(),
        "input_bindings": {
            "eligible_frame_sha256": "5" * 64,
            "source_inventory_sha256": "6" * 64,
            "exclusion_ledger_sha256": "7" * 64,
            "checkpoint_bundle_sha256": "8" * 64,
            "scoring_artifact_sha256": "9" * 64,
        },
        "probability_arm": {
            "rows": [
                {
                    "opaque_id": f"new-{index}",
                    "stratum": {"cell": "synthetic"},
                    "inclusion_probability_numerator": 1,
                    "inclusion_probability_denominator": 100,
                    "inclusion_probability": 0.01,
                }
                for index in range(2)
            ]
        },
        "active_arm": {
            "rows": [
                {"opaque_id": f"new-{index}", "bucket": "boundary"}
                for index in range(2, 4)
            ]
        },
    }
    interleaved = [
        {
            "source_sample_id": f"new-{index}",
            "thread_id": f"thread-new-{index}",
            "acquisition_arm": (
                "probability_random" if index < 2 else "active"
            ),
            "arm_order": index % 2,
        }
        for index in interleave_indices
    ]
    monkeypatch.setattr(
        acquisition_teacher,
        "_validate_source_and_ledger",
        lambda *_: (source_rows, ledger, interleaved),
    )
    teacher_receipt = _teacher_receipt(
        source_path=source_path,
        labels_path=labels_path,
        ledger_path=ledger_path,
        config_path=config,
    )
    teacher_receipt_path = tmp_path / (
        f"receipt-{training.canonical_sha256(teacher_receipt)}.json"
    )
    teacher_receipt_path.write_text(
        json.dumps(teacher_receipt, sort_keys=True), encoding="utf-8"
    )
    result = training.prepare_acquisition_training_frames(
        base_training_path=base_path,
        acquisition_source_path=source_path,
        acquisition_labels_path=labels_path,
        acquisition_ledger_path=ledger_path,
        acquisition_teacher_receipt_path=teacher_receipt_path,
        checkpoint_selection_path=checkpoint_path,
        acquisition_evaluation_path=evaluation_path,
        acquisition_config_path=config,
        expected_config_sha256=training.file_sha256(config),
        factorised_parent_provenance=_factorised_parent_provenance(),
        output_root=tmp_path / "out",
        descriptor_root=tmp_path,
    )
    assert result["primary_eligible_rows_by_arm"] == {"random": 1, "active": 2}
    assert result["training_rows_by_arm"] == {"random": 3, "active": 4}
    assert result["backfilled_rows"] == 0
    assert result["checkpoint_selection_frame"]["sha256"] != result[
        "acquisition_evaluation_frame"
    ]["sha256"]
    assert result["validated_teacher_inputs"]["source_artifact"]["sha256"] == (
        teacher_receipt["source_parquet_sha256"]
    )
    assert result["locked_test_rows_accessed"] == 0

    source_path.write_bytes(source_path.read_bytes() + b"tamper")
    with pytest.raises(training.AcquisitionTrainingContractError):
        training.validate_teacher_acquisition_inputs(
            source_path=source_path,
            labels_path=labels_path,
            ledger_path=ledger_path,
            teacher_receipt_path=teacher_receipt_path,
            acquisition_config_path=config,
            descriptor_root=tmp_path,
        )


def test_private_prediction_contract_rejects_shape_drift() -> None:
    manifest = training.build_run_manifest(_contract())
    trial = next(
        row for row in manifest["trials"] if row["component"] == "target_stance_b4"
    )
    with pytest.raises((ValueError, TypeError)):
        training._build_private_predictions(
            manifest,
            trial,
            [
                {
                    "item_id": "x",
                    "target_presence_logits": [0.0],
                    "stance_logits": [[0.0] * 4 for _ in range(5)],
                }
            ],
        )
