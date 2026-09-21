from __future__ import annotations

import inspect
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import modernbert_inference_candidate_v1 as candidate
from reddit_china_stance.modernbert_factorised_model import (
    compute_factorised_target_stance_losses,
)


def _descriptor(
    name: str,
    digest: str,
    *,
    rows: int | None = None,
    frame: str | None = None,
    thread_digest: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "relative_path": f"candidate/{name}",
        "sha256": digest * 64,
        "bytes": 123,
    }
    if rows is not None:
        value["row_count"] = rows
    if frame is not None:
        value["frame"] = frame
    if thread_digest is not None:
        value["thread_set_sha256"] = thread_digest
    return value


def _contract() -> dict:
    return candidate.freeze_experiment_contract(
        training_frame=_descriptor(
            "training.parquet",
            "1",
            rows=9_000,
            frame="training",
            thread_digest="a" * 64,
        ),
        development_frame=_descriptor(
            "development.parquet",
            "2",
            rows=600,
            frame="development",
            thread_digest="b" * 64,
        ),
        acquisition_labels=_descriptor("labels.parquet", "3", rows=2_000),
        acquisition_receipt=_descriptor("receipt.json", "4"),
        acquisition_run_id="5" * 64,
        source_bundle_sha256="6" * 64,
        dependency_lock_sha256="7" * 64,
        rate_card_usd_per_gpu_second="0.000222",
        cumulative_measured_spend_usd="30",
        active_reservation_usd="0",
        planned_phase_upper_usd="5",
    )


class _Tokenizer:
    def pad(self, rows: list[dict], *, padding: bool, return_tensors: str | None) -> dict:
        assert padding is True
        assert return_tensors is None
        return {
            "input_ids": [row["input_ids"] for row in rows],
            "attention_mask": [row["attention_mask"] for row in rows],
        }


class _TorchTokenizer:
    def pad(self, rows: list[dict], *, padding: bool, return_tensors: str) -> dict:
        torch = pytest.importorskip("torch")
        assert padding is True
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in rows]),
            "attention_mask": torch.tensor([row["attention_mask"] for row in rows]),
        }


def _mixed_feature() -> dict:
    return {
        "item_id": "private-1",
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
        "target_presence_labels": [1, 1, 0, 0, 0, 0],
        "target_presence_mask": [1, 1, 1, 1, 1, 1],
        "stance_b4_labels": [1, 0, -100, -100, -100],
        "stance_mask": [1, 1, 0, 0, 0],
    }


def _calibration_rows() -> list[dict]:
    return [
        {
            "item_id": "development-1",
            "target_presence_logits": [2.0, 1.0, -2.0, -1.0, -3.0, -4.0],
            "target_presence_labels": [1, 1, 0, 0, 0, 0],
            "reference_material": True,
            "stance_logits": [
                [0.0, 9.0, -1.0, 1.0],
                [3.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            "stance_labels": [1, 0, -100, -100, -100],
            "stance_mask": [True, True, False, False, False],
        }
    ]


def _frame_row(
    *,
    item_id: str,
    thread_id: str,
    frame: str,
    label_json: str,
    acquisition_source_sample_id: str | None = None,
    acquisition_label_sha256: str | None = None,
    primary_training_eligible: bool | None = None,
) -> dict:
    return {
        "item_id": item_id,
        "frame": frame,
        "thread_id": thread_id,
        "target_text": f"text for {item_id}",
        "parent_context": None,
        "submission_context": None,
        "label_json": label_json,
        "selection_component": "candidate",
        "selection_stratum": None,
        "inclusion_probability_numerator": None,
        "inclusion_probability_denominator": None,
        "inclusion_probability": None,
        "probability_scope": None,
        "acquisition_source_sample_id": acquisition_source_sample_id,
        "acquisition_label_sha256": acquisition_label_sha256,
        "primary_training_eligible": primary_training_eligible,
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _bound_descriptor(
    path: Path,
    *,
    rows: int | None = None,
    frame: str | None = None,
    threads: list[str] | None = None,
) -> dict:
    descriptor: dict[str, object] = {
        "relative_path": path.name,
        "sha256": candidate.file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        descriptor["row_count"] = rows
    if frame is not None:
        descriptor["frame"] = frame
    if threads is not None:
        descriptor["thread_set_sha256"] = candidate.thread_set_sha256(sorted(threads))
    return descriptor


def _provenance_bound_inputs(
    tmp_path: Path,
    *,
    missing_acquisition_index: int | None = None,
    development_overlap: bool = False,
) -> dict:
    label = {
        "codability": "codable",
        "relevance": "material",
        "targets": [{"target": "government_ccp", "stance": "negative"}],
    }
    label_json = json.dumps(label, sort_keys=True)
    label_sha256 = candidate.canonical_sha256(label)
    raw_labels = []
    acquisition_rows = []
    for index in range(candidate.ACQUISITION_LABEL_ROWS):
        item_id = f"acquired-{index}"
        thread_id = f"acquired-thread-{index}"
        eligible = index % 2 == 0
        raw_labels.append(
            {
                "source_sample_id": item_id,
                "thread_id": thread_id,
                "acquisition_arm": ("active" if index % 2 == 0 else "probability_random"),
                "label_json": label_json,
                "primary_training_eligible": eligible,
            }
        )
        if index != missing_acquisition_index:
            acquisition_rows.append(
                _frame_row(
                    item_id=item_id,
                    thread_id=thread_id,
                    frame="training",
                    label_json=label_json,
                    acquisition_source_sample_id=item_id,
                    acquisition_label_sha256=label_sha256,
                    primary_training_eligible=eligible,
                )
            )
    base_rows = [
        _frame_row(
            item_id=f"base-{index}",
            thread_id=f"base-thread-{index}",
            frame="training",
            label_json=label_json,
        )
        for index in range(2)
    ]
    training_rows = [*base_rows, *acquisition_rows]
    development_rows = [
        _frame_row(
            item_id=f"development-{index}",
            thread_id=(
                "acquired-thread-0"
                if development_overlap and index == 0
                else f"development-thread-{index}"
            ),
            frame="development",
            label_json=label_json,
        )
        for index in range(candidate.DEVELOPMENT_ROWS)
    ]
    training_path = tmp_path / "combined-training.parquet"
    development_path = tmp_path / "development.parquet"
    labels_path = tmp_path / "acquisition-labels.parquet"
    receipt_path = tmp_path / "acquisition-receipt.json"
    _write_parquet(training_path, training_rows)
    _write_parquet(development_path, development_rows)
    _write_parquet(labels_path, raw_labels)
    acquisition_run_id = "a" * 64
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": candidate.SCHEMA_VERSION,
                "kind": "sol-teacher-acquisition-v2-receipt-v1",
                "status": "complete",
                "run_id": acquisition_run_id,
                "row_count": candidate.ACQUISITION_LABEL_ROWS,
                "private_labels_parquet_sha256": candidate.file_sha256(labels_path),
                "automatic_retry_count": 0,
                "row_replacement_count": 0,
            }
        ),
        encoding="utf-8",
    )
    return {
        "training_frame": _bound_descriptor(
            training_path,
            rows=len(training_rows),
            frame="training",
            threads=[row["thread_id"] for row in training_rows],
        ),
        "development_frame": _bound_descriptor(
            development_path,
            rows=len(development_rows),
            frame="development",
            threads=[row["thread_id"] for row in development_rows],
        ),
        "acquisition_labels": _bound_descriptor(
            labels_path,
            rows=len(raw_labels),
        ),
        "acquisition_receipt": _bound_descriptor(receipt_path),
        "acquisition_run_id": acquisition_run_id,
    }


def test_registers_exactly_one_seed_and_retained_b4_one_pass_recipe() -> None:
    config = candidate.registered_candidate_config()
    optimisation = candidate.candidate_optimisation_config()
    assert candidate.REGISTERED_SEEDS == (47,)
    assert candidate.EXPECTED_CANDIDATES == 1
    assert config["candidate_count"] == 1
    assert config["encoder_architecture"] == "single_modernbert_large_one_pass"
    assert config["model_id"] == "answerdotai/ModernBERT-large"
    assert config["model_revision"] == "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
    assert config["component"] == "target_stance_b4"
    assert config["stance_head"]["checkpoint_logits_per_target"] == 4
    assert config["calibration_uses"] == "development_only"
    assert "threshold_or_calibration_selection" not in config["optimisation"]
    assert config["checkpoint_score"] == "non_mixed_conditional_stance_accuracy"
    assert config["combined_training_frame"] == {
        "acquisition_label_rows_required": 2_000,
        "acquisition_row_identity": "item_id_equals_source_sample_id",
        "acquisition_label_binding": "exact_label_json_and_canonical_sha256",
        "acquisition_thread_binding": "exact_thread_id",
        "eligibility_binding": "exact_primary_training_eligible_boolean",
        "training_use": "base_plus_primary_training_eligible_acquisition_rows",
        "development_thread_overlap_allowed": False,
    }
    assert optimisation.effective_batch_size == 32
    assert optimisation.use_bf16 is True


def test_collator_masks_only_mixed_stance_ce_and_preserves_presence_and_label() -> None:
    source = _mixed_feature()
    masked = candidate.mask_mixed_stance_feature(source)
    assert source["stance_mask"] == [1, 1, 0, 0, 0]
    assert masked["stance_mask"] == [0, 1, 0, 0, 0]
    assert masked["stance_b4_labels"] == source["stance_b4_labels"]
    assert masked["target_presence_labels"] == source["target_presence_labels"]
    assert masked["target_presence_mask"] == source["target_presence_mask"]

    batch = candidate.InferenceCandidateB4Collator(_Tokenizer(), return_tensors=None)([source])
    assert batch["target_presence_labels"] == [[1, 1, 0, 0, 0, 0]]
    assert batch["reference_material_mask"] == [True]
    assert batch["stance_labels"] == [[1, 0, -100, -100, -100]]
    assert batch["stance_known_mask"] == [[False, True, False, False, False]]


def test_mixed_target_has_presence_gradient_but_no_stance_ce_gradient() -> None:
    torch = pytest.importorskip("torch")
    batch = candidate.InferenceCandidateB4Collator(_TorchTokenizer())([_mixed_feature()])
    presence_logits = torch.zeros((1, 6), requires_grad=True)
    stance_logits = torch.zeros((1, 5, 4), requires_grad=True)
    losses = compute_factorised_target_stance_losses(
        target_presence_logits=presence_logits,
        stance_logits=stance_logits,
        target_presence_labels=batch["target_presence_labels"],
        stance_labels=batch["stance_labels"],
        reference_material_mask=batch["reference_material_mask"],
        stance_known_mask=batch["stance_known_mask"],
        stance_variant="b4",
    )
    losses["loss"].backward()
    assert bool((presence_logits.grad[0] != 0).all().item())
    assert bool((stance_logits.grad[0, 0] == 0).all().item())
    assert bool((stance_logits.grad[0, 1] != 0).any().item())


def test_three_class_decode_explicitly_excludes_the_mixed_logit() -> None:
    high_mixed = candidate.decode_three_class_stance_logits([0.0, 1_000.0, 0.0, 1.0])
    low_mixed = candidate.decode_three_class_stance_logits([0.0, -1_000.0, 0.0, 1.0])
    assert high_mixed == low_mixed
    assert high_mixed["stance"] == "positive"
    assert set(high_mixed["probabilities"]) == {
        "negative",
        "no_directed_stance",
        "positive",
    }
    assert high_mixed["source_logit_indices"] == [0, 2, 3]


def test_calibration_is_development_only_and_excludes_mixed_without_relabelling() -> None:
    development = _descriptor(
        "development.parquet",
        "a",
        rows=1,
        frame="development",
        thread_digest="c" * 64,
    )
    calibration = candidate.fit_development_calibration(
        development,
        _calibration_rows(),
    )
    clean = candidate.validate_development_calibration(calibration)
    assert clean["source_frame_role"] == "development_only"
    assert clean["excluded_mixed_stance_instances"] == 1
    assert clean["stance_support"] == 1
    assert clean["decode_logit_indices"] == [0, 2, 3]

    training = {**development, "frame": "training"}
    with pytest.raises(candidate.InferenceCandidateContractError, match="development"):
        candidate.fit_development_calibration(training, _calibration_rows())


def test_manifest_has_one_job_no_retry_fallback_locked_or_corpus_authority() -> None:
    contract = candidate.validate_experiment_contract(_contract())
    manifest = candidate.validate_run_manifest(candidate.build_run_manifest(contract))
    assert len(manifest["training_jobs"]) == 1
    assert manifest["training_jobs"][0]["optimiser_seed"] == 47
    assert manifest["training_jobs"][0]["max_attempts"] == 1
    assert contract["compute"]["allowed_gpus"] == ["L4"]
    assert contract["compute"]["max_concurrent_candidates"] == 1
    assert contract["compute"]["gpu_fallback_allowed"] is False
    assert contract["compute"]["retry_authorised"] is False
    assert contract["evidence_boundary"] == {
        "development_calibration_only": True,
        "locked_test_authorised": False,
        "locked_test_rows_accessed": 0,
        "corpus_inference_authorised": False,
        "corpus_rows_accessed": 0,
        "human_validation_claim_authorised": False,
    }
    assert manifest["cuda_preflight"]["max_gpu_seconds"] == 1_800
    assert manifest["cuda_preflight"]["max_attempts"] == 1
    assert manifest["throughput_smoke"]["rows"] == 128
    assert manifest["throughput_smoke"]["batch_size"] == 32
    assert manifest["reserved_cost_usd"] == "2.397600"


def test_manifest_rejects_second_seed_or_overlapping_development() -> None:
    manifest = candidate.build_run_manifest(_contract())
    manifest["training_jobs"].append({**manifest["training_jobs"][0], "optimiser_seed": 61})
    with pytest.raises(candidate.InferenceCandidateContractError):
        candidate.validate_run_manifest(manifest)

    with pytest.raises(candidate.InferenceCandidateContractError, match="overlap"):
        candidate.freeze_experiment_contract(
            training_frame=_descriptor(
                "training.parquet",
                "1",
                rows=9_000,
                frame="training",
                thread_digest="a" * 64,
            ),
            development_frame=_descriptor(
                "development.parquet",
                "2",
                rows=600,
                frame="development",
                thread_digest="a" * 64,
            ),
            acquisition_labels=_descriptor("labels.parquet", "3", rows=2_000),
            acquisition_receipt=_descriptor("receipt.json", "4"),
            acquisition_run_id="5" * 64,
            source_bundle_sha256="6" * 64,
            dependency_lock_sha256="7" * 64,
            rate_card_usd_per_gpu_second="0.000222",
            cumulative_measured_spend_usd="30",
            active_reservation_usd="0",
            planned_phase_upper_usd="5",
        )


def test_combined_frame_conserves_all_labels_and_uses_only_exact_eligible_subset(
    tmp_path: Path,
) -> None:
    bindings = _provenance_bound_inputs(tmp_path)
    training_rows, development_rows, reference, summary = (
        candidate.load_provenance_bound_training_data(
            volume_root=tmp_path,
            bindings=bindings,
        )
    )
    training_ids = {row["item_id"] for row in training_rows}
    assert len(development_rows) == 600
    assert len(reference) == 600
    assert training_ids >= {"base-0", "base-1", "acquired-0"}
    assert "acquired-1" not in training_ids
    assert len(training_rows) == 1_002
    assert summary == {
        "combined_training_rows": 2_002,
        "base_training_rows": 2,
        "acquisition_label_rows": 2_000,
        "acquisition_rows_represented_exactly_once": 2_000,
        "primary_training_eligible_acquisition_rows": 1_000,
        "authorised_training_rows": 1_002,
        "acquisition_lineage_sha256": summary["acquisition_lineage_sha256"],
        "training_thread_set_sha256": bindings["training_frame"]["thread_set_sha256"],
        "development_thread_set_sha256": bindings["development_frame"]["thread_set_sha256"],
        "training_development_thread_overlap": 0,
    }


def test_materialiser_builds_and_revalidates_exact_all_label_lineage(
    tmp_path: Path,
) -> None:
    bindings = _provenance_bound_inputs(tmp_path)
    combined = pq.read_table(tmp_path / "combined-training.parquet").to_pylist()
    base_columns = (
        "item_id",
        "frame",
        "thread_id",
        "target_text",
        "parent_context",
        "submission_context",
        "label_json",
        "selection_component",
        "selection_stratum",
        "inclusion_probability_numerator",
        "inclusion_probability_denominator",
        "inclusion_probability",
        "probability_scope",
    )
    base_rows = [
        {key: row[key] for key in base_columns}
        for row in combined
        if row["acquisition_source_sample_id"] is None
    ]
    acquisition_rows = [row for row in combined if row["acquisition_source_sample_id"] is not None]
    base_path = tmp_path / "base-training.parquet"
    source_path = tmp_path / "acquisition-source.parquet"
    output_path = tmp_path / "materialised-training.parquet"
    _write_parquet(base_path, base_rows)
    _write_parquet(
        source_path,
        [
            {
                "opaque_id": row["item_id"],
                "source_sample_id": row["item_id"],
                "thread_id": row["thread_id"],
                "target_text": row["target_text"],
                "parent_context": row["parent_context"],
                "submission_context": row["submission_context"],
            }
            for row in acquisition_rows
        ],
    )
    base_descriptor = _bound_descriptor(
        base_path,
        rows=len(base_rows),
        frame="training",
        threads=[row["thread_id"] for row in base_rows],
    )
    source_descriptor = _bound_descriptor(
        source_path,
        rows=candidate.ACQUISITION_LABEL_ROWS,
    )
    summary = candidate.materialise_combined_training_frame(
        base_training_path=base_path,
        base_training_descriptor=base_descriptor,
        development_path=tmp_path / "development.parquet",
        development_descriptor=bindings["development_frame"],
        acquisition_source_path=source_path,
        acquisition_source_descriptor=source_descriptor,
        acquisition_labels_path=tmp_path / "acquisition-labels.parquet",
        acquisition_labels_descriptor=bindings["acquisition_labels"],
        acquisition_receipt_path=tmp_path / "acquisition-receipt.json",
        acquisition_receipt_descriptor=bindings["acquisition_receipt"],
        acquisition_run_id=bindings["acquisition_run_id"],
        output_path=output_path,
        descriptor_root=tmp_path,
    )
    assert summary["validation"]["combined_training_rows"] == 2_002
    assert summary["validation"]["acquisition_rows_represented_exactly_once"] == 2_000
    assert summary["validation"]["primary_training_eligible_acquisition_rows"] == 1_000
    assert summary["validation"]["authorised_training_rows"] == 1_002
    assert summary["locked_test_rows_accessed"] == 0
    assert summary["corpus_rows_accessed"] == 0


def test_combined_frame_rejects_valid_sidecar_when_one_acquired_label_is_absent(
    tmp_path: Path,
) -> None:
    bindings = _provenance_bound_inputs(
        tmp_path,
        missing_acquisition_index=1_999,
    )
    with pytest.raises(
        candidate.InferenceCandidateContractError,
        match="all 2,000 acquired labels exactly once",
    ):
        candidate.load_provenance_bound_training_data(
            volume_root=tmp_path,
            bindings=bindings,
        )


def test_combined_frame_rejects_same_size_unauthorised_label_substitution(
    tmp_path: Path,
) -> None:
    bindings = _provenance_bound_inputs(tmp_path)
    labels_path = tmp_path / "acquisition-labels.parquet"
    rows = pq.read_table(labels_path).to_pylist()
    replacement = {
        "codability": "codable",
        "relevance": "material",
        "targets": [{"target": "government_ccp", "stance": "positive"}],
    }
    rows[0]["label_json"] = json.dumps(replacement, sort_keys=True)
    _write_parquet(labels_path, rows)
    bindings["acquisition_labels"] = _bound_descriptor(
        labels_path,
        rows=candidate.ACQUISITION_LABEL_ROWS,
    )

    with pytest.raises(
        candidate.InferenceCandidateContractError,
        match="exact completed 2,000-label artefact",
    ):
        candidate.load_provenance_bound_training_data(
            volume_root=tmp_path,
            bindings=bindings,
        )


def test_combined_frame_rejects_actual_development_thread_overlap(
    tmp_path: Path,
) -> None:
    bindings = _provenance_bound_inputs(tmp_path, development_overlap=True)
    with pytest.raises(
        candidate.InferenceCandidateContractError,
        match="thread sets overlap",
    ):
        candidate.load_provenance_bound_training_data(
            volume_root=tmp_path,
            bindings=bindings,
        )


def test_training_rejects_combined_lineage_before_output_or_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = {"sentinel": "bound inputs"}
    manifest = {"experiment_contract": {"bindings": bindings}}
    monkeypatch.setattr(candidate, "validate_run_manifest", lambda _value: manifest)

    def reject_lineage(*, volume_root: Path, bindings: dict) -> None:
        assert volume_root == tmp_path
        assert bindings == {"sentinel": "bound inputs"}
        raise candidate.InferenceCandidateContractError("combined lineage rejected")

    monkeypatch.setattr(candidate, "load_provenance_bound_training_data", reject_lineage)
    monkeypatch.setattr(
        candidate,
        "_require_torch_runtime",
        lambda: pytest.fail("CUDA must not be touched before lineage validation"),
    )
    output_root = tmp_path / "candidate-output"
    with pytest.raises(
        candidate.InferenceCandidateContractError,
        match="combined lineage rejected",
    ):
        candidate.execute_registered_training(
            manifest,
            volume_root=tmp_path,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_preflight_and_training_receipts_bind_exact_manifest_and_lineage() -> None:
    manifest = candidate.build_run_manifest(_contract())
    bindings = manifest["experiment_contract"]["bindings"]
    provenance = {
        "combined_training_rows": 9_000,
        "base_training_rows": 7_000,
        "acquisition_label_rows": 2_000,
        "acquisition_rows_represented_exactly_once": 2_000,
        "primary_training_eligible_acquisition_rows": 1_000,
        "authorised_training_rows": 8_000,
        "acquisition_lineage_sha256": "f" * 64,
        "training_thread_set_sha256": bindings["training_frame"]["thread_set_sha256"],
        "development_thread_set_sha256": bindings["development_frame"]["thread_set_sha256"],
        "training_development_thread_overlap": 0,
    }
    preflight_body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "modernbert-inference-candidate-cuda-preflight-v1",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "model_id": candidate.MODEL_ID,
        "model_revision": candidate.MODEL_REVISION,
        "gpu_type": candidate.GPU_TYPE,
        "seed_count": 1,
        "stance_logits_per_target": 4,
        "mixed_stance_ce_mask_checked": True,
        "training_provenance": provenance,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    preflight = {
        **preflight_body,
        "receipt_id": candidate.canonical_sha256(preflight_body),
    }
    assert candidate.validate_cuda_preflight_receipt(preflight, manifest=manifest) == preflight

    history = [
        {
            "epoch": epoch,
            "train_mean_loss": 1.0 / epoch,
            "development_three_class_stance_accuracy": epoch / 10,
        }
        for epoch in range(1, 9)
    ]
    calibration = _descriptor("calibration.json", "d")
    receipt_body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": candidate.TRAINING_RECEIPT_KIND,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "optimiser_seed": candidate.REGISTERED_SEED,
        "selected_epoch": 8,
        "development_three_class_stance_accuracy": 0.8,
        "history": history,
        "training_provenance": provenance,
        "checkpoint": _descriptor("checkpoint.pt", "c"),
        "development_calibration": calibration,
        "wall_seconds": 100.0,
        "gpu_seconds": 100.0,
        "attempt_count": 1,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    receipt = {
        **receipt_body,
        "receipt_id": candidate.canonical_sha256(receipt_body),
    }
    assert candidate.validate_training_receipt(receipt, manifest=manifest) == receipt
    drifted = {**receipt, "attempt_count": 2}
    with pytest.raises(candidate.InferenceCandidateContractError, match="receipt drifted"):
        candidate.validate_training_receipt(drifted, manifest=manifest)


def test_bounded_decode_has_no_locked_test_or_corpus_scope_and_no_forced_target() -> None:
    calibration = candidate.fit_development_calibration(
        _descriptor(
            "development.parquet",
            "a",
            rows=1,
            frame="development",
            thread_digest="c" * 64,
        ),
        _calibration_rows(),
    )
    decoded = candidate.decode_candidate_logits(
        target_presence_logits=[-10.0] * 6,
        stance_logits=[[0.0, 999.0, 0.0, 1.0] for _ in range(5)],
        calibration=calibration,
        scope="development",
    )
    assert decoded["present_targets"] == []
    assert decoded["stances"] == {}
    assert decoded["forced_target_selections"] == 0
    for forbidden_scope in ("locked_test", "corpus", "production"):
        with pytest.raises(candidate.InferenceCandidateContractError):
            candidate.validate_inference_scope(forbidden_scope)


def test_throughput_smoke_has_no_injectable_or_unpinned_model_path() -> None:
    signature = inspect.signature(candidate.measure_real_pinned_throughput)
    source = inspect.getsource(candidate.measure_real_pinned_throughput)
    assert "model_factory" not in signature.parameters
    assert "tokenizer_loader" not in signature.parameters
    assert "load_pinned_tokenizer()" in source
    assert "create_candidate_model()" in source
    assert "real_pinned_model_path_measured" in source
