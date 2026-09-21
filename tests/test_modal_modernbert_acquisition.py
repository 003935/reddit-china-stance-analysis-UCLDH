from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import reddit_china_stance.modal_modernbert_acquisition as launcher
from reddit_china_stance import modernbert_acquisition as selector
from reddit_china_stance import semantic_ontology_v2 as semantic_v2
from reddit_china_stance import sol_teacher_acquisition_v2 as acquisition_teacher


def _compute_contract() -> dict[str, object]:
    ledger = {
        "schema_version": "1.0.0",
        "kind": launcher.COMPUTE_LEDGER_KIND,
        "cumulative_measured_spend_usd": "25",
        "active_reservation_usd": "0",
        "phase_upper_cost_usd": {
            "prepare": "5",
            "cuda-preflight": "1",
            "launch-scoring": "40",
            "closeout": "4",
        },
        "hard_cost_cap_usd": "200",
    }
    compute = {
        "allowed_gpus": ["L4"],
        "account_gpu_limit": 10,
        "max_concurrent_checkpoints": 6,
        "gpu_fallback_allowed": False,
        "rate_card_usd_per_gpu_second": {"L4": "0.000222"},
        "cumulative_measured_spend_usd": ledger["cumulative_measured_spend_usd"],
        "active_reservation_usd": ledger["active_reservation_usd"],
        "phase_upper_cost_usd": ledger["phase_upper_cost_usd"],
        "hard_cost_cap_usd": ledger["hard_cost_cap_usd"],
        "remaining_after_plan_usd": "125",
        "ledger_sha256": launcher._canonical_sha256(ledger),
    }
    return {"compute": compute}


def _spec(*, prepared_run_id: str | None = None) -> dict[str, object]:
    _policy, policy_sha256 = launcher.load_frozen_policy()
    manifest, source_bundle = launcher._validate_retained_evidence()
    return {
        "schema_version": "1.0.0",
        "kind": launcher.INPUT_SPEC_KIND,
        "policy_sha256": policy_sha256,
        "base_factorised_run_id": launcher.BASE_FACTORISED_RUN_ID,
        "retained_manifest_sha256": launcher._canonical_sha256(manifest),
        "retained_source_bundle_sha256": launcher._canonical_sha256(source_bundle),
        **_compute_contract(),
        "prepared_run_id": prepared_run_id,
    }


def _checkpoint_trials() -> list[dict[str, object]]:
    return [
        {
            "component": component,
            "optimiser_seed": seed,
            "trial_id": f"{component}-{seed}",
            "sha256": hashlib.sha256(f"{component}-{seed}".encode()).hexdigest(),
        }
        for component in ("relevance", "target_stance_b4")
        for seed in (47, 61, 89)
    ]


def _file_descriptor(path: Path, *, root: Path, row_count: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": launcher._file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        value["row_count"] = row_count
    return value


def _label(target: str | None = None, stance: str | None = None) -> str:
    value: dict[str, object] = {
        "codability": "codable",
        "relevance": "not_material" if target is None else "material",
        "targets": [],
    }
    if target is not None:
        value["targets"] = [{"target": target, "stance": stance}]
    return json.dumps(value, sort_keys=True)


def _synthetic_scoring_shards(
    tmp_path: Path,
    *,
    omit_shard: str | None = None,
    duplicate_shard: str | None = None,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    run_id = "a" * 64
    runtime_digest = "b" * 64
    rows = [
        {
            "record_id": f"opaque-{index:03d}",
            "thread_id": f"thread-{index:03d}",
            "near_duplicate_cluster_id": f"cluster-{index:03d}",
            "subreddit": f"sub-{index % 2}",
            "year": 2020 + index % 6,
            "content_type": "comment",
            "retrieval_mode": "direct",
            "target_text": f"text {index}",
            "submission_context": None,
            "parent_context": None,
        }
        for index in range(20)
    ]
    prepared_root = launcher._prepared_root(run_id, volume_root=tmp_path)
    prepared_root.mkdir(parents=True)
    frame_path = prepared_root / "eligible-frame.parquet"
    pq.write_table(pa.Table.from_pylist(rows), frame_path)
    frame_descriptor = _file_descriptor(frame_path, root=prepared_root, row_count=len(rows))
    prepared = {
        "prepared_frame": frame_descriptor,
        "runtime_source_bundle_sha256": runtime_digest,
    }
    launcher._materialise_prepared_frame_shards(
        run_id=run_id,
        prepared_frame_path=frame_path,
        prepared_frame=frame_descriptor,
        volume_root=tmp_path,
    )
    checkpoint_path = tmp_path / "private" / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"synthetic-checkpoint")
    checkpoint = {
        "component": "relevance",
        "optimiser_seed": 47,
        "trial_id": "relevance-47",
        "relative_path": checkpoint_path.relative_to(tmp_path).as_posix(),
        "sha256": launcher._file_sha256(checkpoint_path),
    }
    checkpoints = [checkpoint]
    checkpoint_bundle_sha256 = launcher._canonical_sha256(checkpoints)
    spec = _spec(prepared_run_id=run_id)
    phase = launcher._scoring_phase_approval_claim(
        spec=spec,
        prepared=prepared,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        estimated_cost_usd=Decimal("40"),
        approved_cost_usd=Decimal("40"),
    )
    scoring_root = launcher._checkpoint_root(run_id, volume_root=tmp_path)
    launcher._write_immutable_json(scoring_root / "phase-approval.json", phase)
    plan = launcher._scoring_plan_contract(len(rows))
    launcher._write_immutable_json(scoring_root / "scoring-plan.json", plan)
    claim = launcher._checkpoint_launch_claim(
        prepared_run_id=run_id,
        checkpoint=checkpoint,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        phase_claim=phase,
    )
    checkpoint_root = launcher._checkpoint_scoring_root(
        run_id=run_id,
        checkpoint=checkpoint,
        volume_root=tmp_path,
    )
    launcher._write_immutable_json(checkpoint_root / "launch-claim.json", claim)
    jobs: list[dict[str, object]] = []
    for shard in plan["shards"]:
        attempt_id = f"attempt-{shard['shard_id']}"
        dispatch_id = hashlib.sha256(attempt_id.encode()).hexdigest()
        job = launcher._scoring_shard_job(
            prepared_run_id=run_id,
            checkpoint=checkpoint,
            checkpoint_bundle_sha256=checkpoint_bundle_sha256,
            phase_claim=phase,
            checkpoint_claim=claim,
            scoring_plan=plan,
            shard=shard,
            attempt_id=attempt_id,
            dispatch_id=dispatch_id,
        )
        jobs.append(job)
        if shard["shard_id"] == omit_shard:
            continue
        source = rows[shard["start"] : shard["stop"]]
        if shard["shard_id"] == duplicate_shard:
            source = [source[0], source[0]]
        logits = [
            {
                "opaque_id": row["record_id"],
                "render": render,
                "relevance_logit": float(index),
            }
            for index, row in enumerate(source)
            for render in launcher.SCORING_RENDERS
        ]
        root = launcher._scoring_shard_root(
            run_id=run_id,
            checkpoint=checkpoint,
            shard=shard,
            volume_root=tmp_path,
        )
        attempt = root / "attempts" / f"attempt={attempt_id}.incomplete"
        attempt.mkdir(parents=True)
        logits_path = attempt / "logits.parquet"
        pq.write_table(pa.Table.from_pylist(logits), logits_path)
        descriptor = _file_descriptor(logits_path, root=root, row_count=len(logits))
        approval = launcher._validate_scoring_shard_job(
            job,
            prepared=prepared,
            volume_root=tmp_path,
        )
        body = {
            "kind": launcher.SCORING_SHARD_KIND,
            "schema_version": "1.0.0",
            "prepared_run_id": run_id,
            "prepared_frame_sha256": frame_descriptor["sha256"],
            "checkpoint": checkpoint,
            "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
            "checkpoint_sha256": checkpoint["sha256"],
            **approval,
            "text_renderer": "factorised-target-parent-submission-v2",
            "token_truncation": "manual-retain-final-token-max-768",
            "padding": "dynamic-per-length-bucket-batch",
            "precision": "bf16-autocast",
            "row_count": shard["row_count"],
            "render_count": 2,
            "logit_rows": shard["row_count"] * 2,
            "logits": descriptor,
            "wall_seconds": 1.0,
            "rows_per_worker_second": float(shard["row_count"]),
            "locked_test_rows_accessed": 0,
        }
        receipt = {**body, "receipt_id": launcher._canonical_sha256(body)}
        launcher._write_immutable_json(attempt / "receipt.json", receipt)
        launcher._write_immutable_json(root / "receipt.json", receipt)
    return prepared, spec, jobs


def test_policy_is_the_single_source_for_frozen_decisions() -> None:
    policy, digest = launcher.load_frozen_policy()

    assert policy["source"]["language_eligibility"] == "provisional_english"
    assert policy["source"]["human_language_gate_accepted"] is False
    assert policy["active"]["bucket_rows"] == {
        "rare_cell": 200,
        "boundary": 300,
        "multi_context": 100,
        "uncertainty_disagreement": 400,
    }
    assert policy["student"]["representation"] == "B4"
    assert digest == launcher._file_sha256(launcher._policy_path())


def test_policy_path_prefers_modal_runtime_mount_with_shallow_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_policy = tmp_path / "modernbert-acquisition-v1.toml"
    runtime_policy.write_bytes(launcher._policy_path().read_bytes())
    monkeypatch.setattr(launcher, "POLICY_RUNTIME_PATH", runtime_policy)
    monkeypatch.setattr(launcher, "__file__", "/root/modal_modernbert_acquisition.py")

    assert launcher._policy_path() == runtime_policy


def test_retained_evidence_prefers_runtime_mount_with_shallow_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_manifest, local_bundle = launcher._retained_evidence_paths()
    runtime_manifest = tmp_path / "run-manifest.json"
    runtime_bundle = tmp_path / "source-bundle.json"
    runtime_manifest.write_bytes(local_manifest.read_bytes())
    runtime_bundle.write_bytes(local_bundle.read_bytes())
    monkeypatch.setattr(launcher, "RETAINED_MANIFEST_RUNTIME_PATH", runtime_manifest)
    monkeypatch.setattr(launcher, "RETAINED_SOURCE_BUNDLE_RUNTIME_PATH", runtime_bundle)
    monkeypatch.setattr(launcher, "__file__", "/root/modal_modernbert_acquisition.py")

    assert launcher._retained_evidence_paths() == (runtime_manifest, runtime_bundle)


def test_semantic_v2_schema_is_mounted_at_modal_runtime_resolution_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mounts = dict(launcher.RUNTIME_FILE_MOUNTS)
    assert mounts[launcher.SEMANTIC_V2_SCHEMA_REPO_PATH] == launcher.SEMANTIC_V2_SCHEMA_RUNTIME_PATH
    modal_module_path = Path("/root/reddit_china_stance/semantic_ontology_v2.py")
    assert (
        modal_module_path.resolve().parents[2] / launcher.SEMANTIC_V2_SCHEMA_REPO_PATH
        == launcher.SEMANTIC_V2_SCHEMA_RUNTIME_PATH
    )

    runtime_schema = tmp_path / "schemas" / "target-stance-v2-pilot.schema.json"
    runtime_schema.parent.mkdir()
    runtime_schema.write_bytes(semantic_v2.SCHEMA_PATH.read_bytes())
    monkeypatch.setattr(launcher, "SEMANTIC_V2_SCHEMA_RUNTIME_PATH", runtime_schema)
    monkeypatch.setattr(launcher, "__file__", "/root/reddit_china_stance/modal.py")

    assert launcher._semantic_v2_schema_path() == runtime_schema
    assert semantic_v2.load_v2_label_schema(runtime_schema)["$schema"].endswith(
        "draft/2020-12/schema"
    )


def test_registered_near_duplicate_clustering_is_deterministic_and_respects_lengths() -> None:
    packet = launcher._packet_module()
    text = "registered near duplicate token sequence " * 12
    rows = [
        {
            "record_id": record_id,
            "simhash": packet._simhash(value),
            "normalised_chars": len(launcher._normalise_surface(value)),
        }
        for record_id, value in (("a", text), ("b", text), ("short", "x" * 79))
    ]
    first = launcher._cluster_near_duplicates([dict(row) for row in rows])
    second = launcher._cluster_near_duplicates([dict(row) for row in reversed(rows)])
    assert {row["record_id"] for row in first} == {row["record_id"] for row in second}
    assert len(first) == 2  # exact duplicate collapses; sub-minimum surface does not.


def test_compute_guardrail_fails_closed_at_shared_cap() -> None:
    launcher.validate_compute_contract(_compute_contract())
    launcher.enforce_cost_guardrail(
        estimated_cost_usd=Decimal("50"), approved_cost_usd=Decimal("50")
    )

    over = _compute_contract()
    over["compute"]["phase_upper_cost_usd"]["launch-scoring"] = "166"  # type: ignore[index]
    over["compute"]["remaining_after_plan_usd"] = "0"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="total cost cap"):
        launcher.validate_compute_contract(over)
    with pytest.raises(ValueError, match="<= 200"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("1"), approved_cost_usd=Decimal("201")
        )


def test_local_builder_is_deterministic_and_requires_authoritative_ledger(
    tmp_path: Path,
) -> None:
    ledger = {
        "schema_version": "1.0.0",
        "kind": launcher.COMPUTE_LEDGER_KIND,
        "cumulative_measured_spend_usd": "25",
        "active_reservation_usd": "0",
        "phase_upper_cost_usd": {
            "prepare": "5",
            "cuda-preflight": "1",
            "launch-scoring": "40",
            "closeout": "4",
        },
        "hard_cost_cap_usd": "200",
    }
    ledger_path = tmp_path / "compute-ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")

    first = launcher.build_local_input_spec(compute_ledger_path=ledger_path)
    second = launcher.build_local_input_spec(compute_ledger_path=ledger_path)

    assert first == second
    assert first["prepared_run_id"] is None
    assert first["compute"]["remaining_after_plan_usd"] == "125"
    assert first["compute"]["ledger_sha256"] == launcher._canonical_sha256(ledger)
    launcher.validate_input_spec(first, action="prepare")

    malformed = {**first, "unexpected": True}
    with pytest.raises(ValueError, match="top-level schema"):
        launcher.validate_input_spec(malformed)
    ledger["cumulative_measured_spend_usd"] = "1"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="retained manifest's measured floor"):
        launcher.build_local_input_spec(compute_ledger_path=ledger_path)
    with pytest.raises(FileNotFoundError):
        launcher.build_local_input_spec(compute_ledger_path=tmp_path / "missing.json")


def test_phase_approval_and_confirmation_tokens_are_exact() -> None:
    prepared_run_id = "a" * 64
    spec = _spec(prepared_run_id=prepared_run_id)

    estimate, approved = launcher._validate_phase_approval(
        spec, action="launch-scoring", approved_cost_usd="40"
    )
    assert (estimate, approved) == (Decimal("40"), Decimal("40"))
    with pytest.raises(ValueError, match="exactly match"):
        launcher._validate_phase_approval(spec, action="launch-scoring", approved_cost_usd="41")

    assert launcher.required_confirmation("prepare", _spec()) == "PREPARE_MODERNBERT_ACQUISITION_V1"
    assert (
        launcher.required_confirmation("cuda-preflight", spec)
        == "CUDA_PREFLIGHT_MODERNBERT_ACQUISITION_aaaaaaaaaaaa"
    )
    assert (
        launcher.required_confirmation("launch-scoring", spec)
        == "LAUNCH_SCORING_MODERNBERT_ACQUISITION_aaaaaaaaaaaa"
    )
    assert (
        launcher.required_confirmation("closeout", spec)
        == "CLOSEOUT_MODERNBERT_ACQUISITION_aaaaaaaaaaaa"
    )


def test_public_prepare_receipt_exposes_and_binds_run_id_without_private_inventory() -> None:
    contract = {
        "kind": launcher.PREPARED_KIND,
        "schema_version": "1.0.0",
        "source_inventory": {"private": [{"relative_path": "private/source.parquet"}]},
        "source_inventory_sha256": "a" * 64,
        "locked_test_rows_accessed": 0,
    }
    descriptor = {
        "relative_path": "eligible-frame.parquet",
        "sha256": "b" * 64,
        "bytes": 1,
        "row_count": 2600,
    }
    evaluation = {
        "relative_path": "acquisition-evaluation.parquet",
        "sha256": "c" * 64,
        "bytes": 1,
        "row_count": 600,
    }
    receipt = launcher._public_preparation_receipt(
        contract=contract,
        prepared_frame=descriptor,
        acquisition_evaluation_frame=evaluation,
    )

    assert receipt["prepared_run_id"] == launcher._canonical_sha256(contract)
    assert "source_inventory" not in receipt
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    assert receipt["receipt_id"] == launcher._canonical_sha256(body)


def test_post_prepare_promotion_is_bound_and_immutable(tmp_path: Path) -> None:
    input_spec = _spec()
    run_id = "d" * 64
    receipt_body = {
        "kind": launcher.PREPARED_KIND,
        "schema_version": "1.0.0",
        "prepared_run_id": run_id,
        "launch_spec_sha256": launcher._launch_spec_sha256(input_spec),
        "policy_sha256": input_spec["policy_sha256"],
        "base_factorised_run_id": input_spec["base_factorised_run_id"],
        "locked_test_rows_accessed": 0,
    }
    receipt = {
        **receipt_body,
        "receipt_id": launcher._canonical_sha256(receipt_body),
    }
    input_path = tmp_path / "input.json"
    receipt_path = tmp_path / "prepare-receipt.json"
    output_path = tmp_path / "prepared-input.json"
    launcher._write_immutable_json(input_path, input_spec)
    launcher._write_immutable_json(receipt_path, receipt)

    promoted = launcher.promote_prepared_input_file(
        input_spec_path=input_path,
        prepare_receipt_path=receipt_path,
        output_spec_path=output_path,
        prepared_run_id=run_id,
    )

    assert promoted["prepared_run_id"] == run_id
    assert json.loads(input_path.read_text())["prepared_run_id"] is None
    assert json.loads(output_path.read_text()) == promoted
    assert (
        launcher.promote_prepared_input_file(
            input_spec_path=input_path,
            prepare_receipt_path=receipt_path,
            output_spec_path=output_path,
            prepared_run_id=run_id,
        )
        == promoted
    )
    with pytest.raises(ValueError, match="new output path"):
        launcher.promote_prepared_input_file(
            input_spec_path=input_path,
            prepare_receipt_path=receipt_path,
            output_spec_path=input_path,
            prepared_run_id=run_id,
        )
    mismatching_output = tmp_path / "mismatching-prepared-input.json"
    launcher._write_immutable_json(mismatching_output, {**promoted, "prepared_run_id": "f" * 64})
    with pytest.raises(RuntimeError, match="existing immutable output differs"):
        launcher.promote_prepared_input_file(
            input_spec_path=input_path,
            prepare_receipt_path=receipt_path,
            output_spec_path=mismatching_output,
            prepared_run_id=run_id,
        )
    with pytest.raises(RuntimeError, match="receipt/input-spec binding"):
        launcher.promote_prepared_input_spec(
            input_spec=input_spec,
            prepare_receipt=receipt,
            prepared_run_id="e" * 64,
        )


def test_post_prepare_promotion_binds_compute_contract() -> None:
    spec = _spec()
    changed = json.loads(json.dumps(spec))
    changed["compute"]["rate_card_usd_per_gpu_second"]["L4"] = "0.000223"

    assert launcher._launch_spec_sha256(spec) != launcher._launch_spec_sha256(changed)


def test_scoring_phase_claim_binds_approval_spend_and_runtime() -> None:
    spec = _spec(prepared_run_id="a" * 64)
    claim = launcher._scoring_phase_approval_claim(
        spec=spec,
        prepared={"runtime_source_bundle_sha256": "b" * 64},
        checkpoint_bundle_sha256="c" * 64,
        estimated_cost_usd=Decimal("40"),
        approved_cost_usd=Decimal("40"),
    )

    assert claim["approved_cost_usd"] == "40"
    assert claim["estimated_upper_cost_usd"] == "40"
    assert claim["cumulative_measured_spend_usd"] == "25"
    assert claim["runtime_source_bundle_sha256"] == "b" * 64
    body = {key: value for key, value in claim.items() if key != "claim_id"}
    assert claim["claim_id"] == launcher._canonical_sha256(body)


def test_scoring_job_requires_exact_persisted_phase_and_trial_claims(
    tmp_path: Path,
) -> None:
    run_id = "a" * 64
    runtime_digest = "b" * 64
    checkpoint_bundle_sha256 = "c" * 64
    checkpoint = {
        "component": "relevance",
        "optimiser_seed": 47,
        "trial_id": "relevance-47",
        "relative_path": "private/checkpoint.pt",
        "sha256": "d" * 64,
    }
    spec = _spec(prepared_run_id=run_id)
    prepared = {"runtime_source_bundle_sha256": runtime_digest}
    phase = launcher._scoring_phase_approval_claim(
        spec=spec,
        prepared=prepared,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        estimated_cost_usd=Decimal("40"),
        approved_cost_usd=Decimal("40"),
    )
    scoring_root = launcher._checkpoint_root(run_id, volume_root=tmp_path)
    launcher._write_immutable_json(scoring_root / "phase-approval.json", phase)
    launch_body = {
        "kind": "modernbert-acquisition-checkpoint-launch-claim-v1",
        "schema_version": "1.0.0",
        "action": "score-checkpoint",
        "prepared_run_id": run_id,
        "component": "relevance",
        "optimiser_seed": 47,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_descriptor_sha256": launcher._canonical_sha256(checkpoint),
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "phase_approval_claim_id": phase["claim_id"],
        "estimated_upper_cost_usd": "40",
        "approved_cost_usd": "40",
        "runtime_source_bundle_sha256": runtime_digest,
        "status": "claimed_before_spawn",
        "locked_test_rows_accessed": 0,
    }
    launch = {
        **launch_body,
        "claim_id": launcher._canonical_sha256(launch_body),
    }
    trial_root = scoring_root / "component=relevance" / "seed=47"
    launcher._write_immutable_json(trial_root / "launch-claim.json", launch)
    job = {
        "prepared_run_id": run_id,
        "checkpoint": checkpoint,
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "phase_approval_claim_id": phase["claim_id"],
        "checkpoint_launch_claim_id": launch["claim_id"],
        "approved_cost_usd": "40",
        "estimated_upper_cost_usd": "40",
        "runtime_source_bundle_sha256": runtime_digest,
    }

    approval = launcher._validate_scoring_job_approval(job, prepared=prepared, volume_root=tmp_path)
    assert approval == {
        "phase_approval_claim_id": phase["claim_id"],
        "checkpoint_launch_claim_id": launch["claim_id"],
        "approved_cost_usd": "40",
        "estimated_upper_cost_usd": "40",
        "runtime_source_bundle_sha256": runtime_digest,
    }
    direct = {
        "prepared_run_id": run_id,
        "checkpoint": checkpoint,
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
    }
    with pytest.raises(ValueError, match="approval schema"):
        launcher._validate_scoring_job_approval(direct, prepared=prepared, volume_root=tmp_path)
    with pytest.raises(RuntimeError, match="phase approval claim"):
        launcher._validate_scoring_job_approval(
            {**job, "phase_approval_claim_id": "f" * 64},
            prepared=prepared,
            volume_root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="launch claim"):
        launcher._validate_scoring_job_approval(
            {**job, "checkpoint_launch_claim_id": "f" * 64},
            prepared=prepared,
            volume_root=tmp_path,
        )
    with pytest.raises(ValueError, match="exactly match"):
        launcher._validate_scoring_job_approval(
            {**job, "approved_cost_usd": "41"},
            prepared=prepared,
            volume_root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="runtime binding"):
        launcher._validate_scoring_job_approval(
            {**job, "runtime_source_bundle_sha256": "f" * 64},
            prepared=prepared,
            volume_root=tmp_path,
        )


def test_remote_read_only_action_schema_boundaries() -> None:
    launcher.validate_input_spec(_spec(), action="validate-volume")
    launcher.validate_input_spec(_spec(prepared_run_id="a" * 64), action="validate-scoring")
    with pytest.raises(ValueError, match="null"):
        launcher.validate_input_spec(_spec(prepared_run_id="a" * 64), action="validate-volume")
    with pytest.raises(ValueError, match="requires a prepared_run_id"):
        launcher.validate_input_spec(_spec(), action="validate-scoring")


def test_scoring_validation_returns_only_bound_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = {
        "prepared_run_id": "a" * 64,
        "preflight": {"receipt_id": "b" * 64},
        "checkpoints": _checkpoint_trials(),
        "checkpoint_bundle_sha256": "c" * 64,
        "scoring_artifact_sha256": "d" * 64,
        "ledger": {"ledger_id": "e" * 64},
        "public": {"receipt_id": "f" * 64},
        "teacher_validation": {
            "teacher_packet_rows": 2_000,
            "teacher_source_projection_sha256": "1" * 64,
            "teacher_interleave_sha256": "2" * 64,
        },
        "prepared": {"runtime_source_bundle_sha256": "3" * 64},
    }
    monkeypatch.setattr(
        launcher,
        "_reconstruct_scoring_material",
        lambda *_args, **_kwargs: material,
    )

    result = launcher.validate_scoring_artifacts(
        _spec(prepared_run_id="a" * 64), volume_root=Path("/unused")
    )

    assert result["checkpoint_bundles"] == 6
    assert result["teacher_packet_rows"] == 2_000
    assert result["runtime_source_bundle_sha256"] == "3" * 64
    assert "ledger" not in result
    assert "source" not in result


def test_checkpoint_inventory_is_resolved_from_bound_run_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "private" / "run-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    trials = _checkpoint_trials()
    manifest = {
        "experiment_run_id": launcher.BASE_FACTORISED_RUN_ID,
        "phase_run_id": "p" * 64,
        "experiment_contract": {
            "bindings": {
                "development_frame": {
                    "relative_path": "private/development.parquet",
                    "sha256": "d" * 64,
                }
            }
        },
        "trials": trials,
    }

    class Training:
        @staticmethod
        def validate_run_manifest(_value: object) -> dict[str, object]:
            return manifest

        @staticmethod
        def canonical_sha256(_value: object) -> str:
            return "m" * 64

        @staticmethod
        def _trial_roots(**kwargs: object) -> tuple[Path, Path]:
            final = tmp_path / "final" / str(kwargs["trial_id"])
            final.mkdir(parents=True, exist_ok=True)
            checkpoint = final / "checkpoint.pt"
            prediction = final / "development-predictions.json"
            checkpoint.write_bytes(str(kwargs["trial_id"]).encode())
            prediction.write_text("{}", encoding="utf-8")
            (final / "receipt.json").write_text("{}", encoding="utf-8")
            return final, tmp_path / "attempts"

        @staticmethod
        def validate_trial_artifacts(
            final: Path, *_args: object, **_kwargs: object
        ) -> dict[str, object]:
            return {
                "artifacts": {
                    "checkpoint": {
                        "relative_path": "checkpoint.pt",
                        "sha256": launcher._file_sha256(final / "checkpoint.pt"),
                    },
                    "private_development_predictions": {
                        "relative_path": "development-predictions.json",
                        "sha256": launcher._file_sha256(final / "development-predictions.json"),
                    },
                }
            }

    monkeypatch.setattr(launcher, "_training_module", lambda: Training)
    source_bundle = {"kind": "synthetic-retained-source"}
    monkeypatch.setattr(
        launcher,
        "_validate_retained_evidence",
        lambda **_kwargs: (manifest, source_bundle),
    )
    spec = _spec()
    spec["retained_manifest_sha256"] = launcher._canonical_sha256(manifest)
    spec["retained_source_bundle_sha256"] = launcher._canonical_sha256(source_bundle)
    clean = launcher._checkpoint_specs(
        spec,
        volume_root=tmp_path,
    )
    assert len(clean) == 6
    assert {row["component"] for row in clean} == {"relevance", "target_stance_b4"}
    validation = launcher.validate_volume_bindings(spec, volume_root=tmp_path)
    assert validation["checkpoint_bundles"] == 6
    assert validation["checkpoint_bundle_sha256"] == launcher._canonical_sha256(clean)
    assert validation["locked_test_rows_accessed"] == 0


def test_local_preflight_validates_policy_retained_bindings_and_six_trial_bundles() -> None:
    result = launcher.preflight_local(_spec())

    assert result["status"] == "validated"
    assert result["checkpoint_trial_bundles"] == 6
    assert result["checkpoint_artifacts_resolved"] == 0
    assert result["checkpoint_resolution"] == "trial-specifications-only"
    assert result["retained_volume_binding_count"] > 0
    assert result["locked_test_rows_accessed"] == 0


def test_prepared_contract_binds_policy_and_explicit_language_boundary() -> None:
    _policy, digest = launcher.load_frozen_policy()
    spec = _spec()
    universe = [
        {
            "record_id": "private-record",
            "thread_id": "private-thread",
            "near_duplicate_cluster_id": "private-cluster",
            "subreddit": "news",
            "year": 2024,
            "content_type": "comment",
            "retrieval_mode": "direct",
            "exact_surface_sha256": "a" * 64,
            "normalised_surface_sha256": "b" * 64,
        }
    ]
    contract = launcher._prepared_contract(
        spec,
        inventories={"candidates": [], "languages": [], "canonical": []},
        exclusions={
            "thread_sha256": set(),
            "record_id_sha256": set(),
            "exact_surface_sha256": set(),
            "normalised_surface_sha256": set(),
            "near_duplicate_references": [],
        },
        universe=universe,
        rare_cells=[
            {"target": "china_general", "stance": "negative", "training_support": 1},
            {"target": "government_ccp", "stance": "negative", "training_support": 2},
            {"target": "people_identity", "stance": "positive", "training_support": 3},
        ],
        rare_cell_support_summary={
            "retained_cell_count": 8,
            "supported_cell_count": 3,
            "summary_id": "e" * 64,
        },
        calibration_derivation={
            "base_factorised_run_id": launcher.BASE_FACTORISED_RUN_ID,
            "row_count": 600,
            "derivation_id": "c" * 64,
            "thread_set_sha256": "d" * 64,
            "row_projection_sha256": "f" * 64,
        },
        runtime_source_bundle=launcher._runtime_source_bundle(),
        policy_sha256=digest,
    )
    assert contract["policy_sha256"] == digest
    assert contract["analysis_language"] == {
        "main_corpus_status": "provisional_english",
        "human_language_gate_accepted": False,
    }
    assert contract["locked_test_rows_accessed"] == 0
    assert len(contract["rare_cells"]) == 3
    assert contract["rare_cell_list_sha256"] == launcher._canonical_sha256(contract["rare_cells"])
    assert (
        contract["runtime_source_bundle_sha256"]
        == contract["runtime_source_bundle"]["source_bundle_id"]
    )

    with pytest.raises(ValueError, match="exactly three"):
        launcher._prepared_contract(
            spec,
            inventories={"candidates": [], "languages": [], "canonical": []},
            exclusions={
                "thread_sha256": set(),
                "record_id_sha256": set(),
                "exact_surface_sha256": set(),
                "normalised_surface_sha256": set(),
                "near_duplicate_references": [],
            },
            universe=universe,
            rare_cells=[
                {"target": "china_general", "stance": "negative", "training_support": 1},
                {"target": "government_ccp", "stance": "negative", "training_support": 2},
            ],
            rare_cell_support_summary={
                "retained_cell_count": 8,
                "supported_cell_count": 2,
            },
            calibration_derivation={
                "base_factorised_run_id": launcher.BASE_FACTORISED_RUN_ID,
                "row_count": 600,
                "derivation_id": "c" * 64,
                "thread_set_sha256": "d" * 64,
                "row_projection_sha256": "f" * 64,
            },
            runtime_source_bundle=launcher._runtime_source_bundle(),
            policy_sha256=digest,
        )

    with pytest.raises(ValueError, match="evaluation frame"):
        launcher._prepared_contract(
            spec,
            inventories={"candidates": [], "languages": [], "canonical": []},
            exclusions={
                "thread_sha256": set(),
                "record_id_sha256": set(),
                "exact_surface_sha256": set(),
                "normalised_surface_sha256": set(),
                "near_duplicate_references": [],
            },
            universe=universe,
            rare_cells=[
                {"target": "china_general", "stance": "negative", "training_support": 1},
                {"target": "government_ccp", "stance": "negative", "training_support": 2},
                {"target": "people_identity", "stance": "positive", "training_support": 3},
            ],
            rare_cell_support_summary={"retained_cell_count": 8, "supported_cell_count": 3},
            calibration_derivation={
                "base_factorised_run_id": launcher.BASE_FACTORISED_RUN_ID,
                "row_count": 222,
                "derivation_id": "c" * 64,
            },
            runtime_source_bundle=launcher._runtime_source_bundle(),
            policy_sha256=digest,
        )


def test_runtime_source_bundle_is_deterministic_and_exact() -> None:
    first = launcher._runtime_source_bundle()
    second = launcher._runtime_source_bundle()

    assert first == second
    assert tuple(sorted(first["files"])) == tuple(sorted(launcher.REQUIRED_SOURCE_FILES))
    assert first["runtime_dependencies"]["jsonschema"] == "4.26.0"
    assert first["runtime_dependencies"]["pydantic"] == "2.13.4"
    assert first["files"]["schemas/target-stance-v2-pilot.schema.json"] == launcher._file_sha256(
        semantic_v2.SCHEMA_PATH
    )
    assert "src/reddit_china_stance/modernbert_factorised_experiment.py" in first["files"]
    assert "src/reddit_china_stance/semantic_ontology_v2.py" in first["files"]
    body = {key: value for key, value in first.items() if key != "source_bundle_id"}
    assert first["source_bundle_id"] == launcher._canonical_sha256(body)


def test_runtime_source_bundle_changes_with_experiment_and_ontology_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = launcher._runtime_source_bundle()
    experiment = tmp_path / "modernbert_factorised_experiment.py"
    ontology = tmp_path / "semantic_ontology_v2.py"
    experiment.write_text("EXPERIMENT_REVISION = 2\n", encoding="utf-8")
    ontology.write_text("ONTOLOGY_REVISION = 2\n", encoding="utf-8")
    original_import = launcher.importlib.import_module

    class ExperimentModule:
        __file__ = str(experiment)

    class OntologyModule:
        __file__ = str(ontology)

    monkeypatch.setattr(launcher, "_experiment_module", lambda: ExperimentModule)
    experiment_bound = launcher._runtime_source_bundle()

    def import_module(name: str) -> object:
        if name == "reddit_china_stance.semantic_ontology_v2":
            return OntologyModule
        return original_import(name)

    monkeypatch.setattr(launcher.importlib, "import_module", import_module)
    both_bound = launcher._runtime_source_bundle()

    assert experiment_bound["source_bundle_id"] != baseline["source_bundle_id"]
    assert both_bound["source_bundle_id"] != experiment_bound["source_bundle_id"]
    assert experiment_bound["files"][
        "src/reddit_china_stance/modernbert_factorised_experiment.py"
    ] == launcher._file_sha256(experiment)
    assert both_bound["files"][
        "src/reddit_china_stance/semantic_ontology_v2.py"
    ] == launcher._file_sha256(ontology)


def test_receipt_backed_inventory_does_not_rehash_parquet_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = tmp_path / "candidates" / "subreddit=news" / "year=2024" / "unit=abc"
    cell.mkdir(parents=True)
    frame = cell / "candidates-data.parquet"
    pq.write_table(pa.Table.from_pylist([{"record_id": "row"}]), frame)
    output = {
        "file": frame.name,
        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        "bytes": frame.stat().st_size,
        "rows": 1,
    }
    receipt = {"status": "complete", "output": output}
    encoded = json.dumps(receipt, sort_keys=True).encode()
    receipt_path = cell / f"receipt-{hashlib.sha256(encoded).hexdigest()}.json"
    receipt_path.write_bytes(encoded)
    original = launcher._file_sha256

    def guarded(path: Path) -> str:
        if path.suffix == ".parquet":
            raise AssertionError("inventory must not hash receipt-bound Parquet payloads")
        return original(path)

    monkeypatch.setattr(launcher, "_file_sha256", guarded)
    inventory = launcher._receipt_backed_inventory(
        [frame], producer="candidates", volume_root=tmp_path
    )

    assert inventory[0]["sha256"] == output["sha256"]
    assert inventory[0]["row_count"] == 1


def test_prepare_shard_authority_rejects_any_path_or_spec_drift() -> None:
    spec = _spec()
    shard = {
        "shard_id": "a" * 64,
        "cells": ["subreddit=news/year=2024"],
        "candidates": ["/volume/candidates.parquet"],
        "languages": ["/volume/languages.parquet"],
        "canonical": ["/volume/comments.parquet", "/volume/submissions.parquet"],
    }
    support = {
        "launch_spec_sha256": launcher._launch_spec_sha256(spec),
        "shards": [shard],
    }

    launcher._authorise_prepare_shard(support=support, shard=shard, spec=spec)
    with pytest.raises(RuntimeError, match="not support-authorised"):
        launcher._authorise_prepare_shard(
            support=support,
            shard={**shard, "canonical": ["/volume/other.parquet"]},
            spec=spec,
        )
    changed = json.loads(json.dumps(spec))
    changed["compute"]["rate_card_usd_per_gpu_second"]["L4"] = "0.001"
    with pytest.raises(RuntimeError, match="launch spec"):
        launcher._authorise_prepare_shard(support=support, shard=shard, spec=changed)


def _stub_prepare_shard_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> type[object]:
    class Volume:
        commits = 0

        @staticmethod
        def reload() -> None:
            return None

        @classmethod
        def commit(cls) -> None:
            cls.commits += 1

    monkeypatch.setattr(launcher, "VOLUME_PATH", tmp_path)
    monkeypatch.setattr(launcher, "volume", Volume)
    monkeypatch.setattr(
        launcher,
        "validate_input_spec",
        lambda spec, *, action: dict(spec) if action == "prepare" else pytest.fail(action),
    )
    monkeypatch.setattr(launcher, "_load_prepare_support", lambda _run_id: {"exclusions": {}})
    monkeypatch.setattr(launcher, "_authorise_prepare_shard", lambda **_kwargs: None)
    return Volume


def test_prepare_shard_ignores_sibling_publishing_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume = _stub_prepare_shard_runtime(tmp_path, monkeypatch)
    pre_prepare_id = "a" * 64
    shard_id = "b" * 64
    sibling_id = "c" * 64
    shard = {
        "shard_id": shard_id,
        "cells": ["subreddit=China/year=2024"],
        "candidates": [],
        "languages": [],
        "canonical": [],
    }
    parent = launcher._prepare_stage_root(pre_prepare_id) / "shards"
    sibling = parent / f".publishing-{sibling_id}"
    sibling.mkdir(parents=True)
    monkeypatch.setattr(launcher, "_prepare_sql_universe", lambda **_kwargs: [])

    receipt = launcher.prepare_shard.get_raw_f()(pre_prepare_id, shard, {})

    assert receipt["shard_id"] == shard_id
    assert (parent / f"shard={shard_id}" / "receipt.json").is_file()
    assert sibling.is_dir()
    assert volume.commits == 1


@pytest.mark.parametrize(
    ("collision", "message"),
    [
        ("root", "namespace is incomplete"),
        ("staging", "publishing namespace requires reconciliation"),
    ],
)
def test_prepare_shard_rejects_only_its_own_incomplete_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
    message: str,
) -> None:
    _stub_prepare_shard_runtime(tmp_path, monkeypatch)
    pre_prepare_id = "a" * 64
    shard_id = "b" * 64
    shard = {
        "shard_id": shard_id,
        "cells": ["subreddit=China/year=2024"],
        "candidates": [],
        "languages": [],
        "canonical": [],
    }
    parent = launcher._prepare_stage_root(pre_prepare_id) / "shards"
    target = (
        parent / f"shard={shard_id}"
        if collision == "root"
        else parent / f".publishing-{shard_id}"
    )
    target.mkdir(parents=True)
    monkeypatch.setattr(
        launcher,
        "_prepare_sql_universe",
        lambda **_kwargs: pytest.fail("collision must fail before source reconstruction"),
    )

    with pytest.raises(FileExistsError, match=message):
        launcher.prepare_shard.get_raw_f()(pre_prepare_id, shard, {})


def test_pre_prepare_id_binds_runtime_source_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    inventories = {"candidates": [], "languages": [], "canonical": []}
    monkeypatch.setattr(
        launcher, "_runtime_source_bundle", lambda: {"source_bundle_id": "a" * 64}
    )
    first = launcher._pre_prepare_id(spec, inventories)
    monkeypatch.setattr(
        launcher, "_runtime_source_bundle", lambda: {"source_bundle_id": "b" * 64}
    )

    assert launcher._pre_prepare_id(spec, inventories) != first


def test_calibration_cache_is_immutable_exact_and_resumable(tmp_path: Path) -> None:
    pre_prepare_id = "a" * 64
    rows = [
        {
            "item_id": "private-item",
            "thread_id": "private-thread",
            "target_text": "private text",
            "label_json": {"relevance": "material"},
        }
    ]
    derivation = {
        "derivation_id": "b" * 64,
        "row_projection_sha256": launcher._canonical_sha256(rows),
    }

    first = launcher._write_prepare_calibration_cache(
        pre_prepare_id=pre_prepare_id,
        rows=rows,
        derivation=derivation,
        volume_root=tmp_path,
    )
    second = launcher._write_prepare_calibration_cache(
        pre_prepare_id=pre_prepare_id,
        rows=rows,
        derivation=derivation,
        volume_root=tmp_path,
    )
    loaded, receipt = launcher._load_prepare_calibration_cache(
        pre_prepare_id, volume_root=tmp_path
    )

    assert first == second == receipt
    assert loaded == rows
    assert receipt["frame"]["row_count"] == 1


def test_near_duplicate_reduction_reports_bounded_progress() -> None:
    progress: list[tuple[int, int]] = []
    rows = [
        {
            "record_id": f"row-{index}",
            "simhash": index,
            "normalised_chars": 100,
        }
        for index in range(5)
    ]

    clustered = launcher._cluster_near_duplicates(
        rows,
        progress=lambda completed, expected, _elapsed: progress.append(
            (completed, expected)
        ),
        progress_interval=2,
    )

    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert 1 <= len(clustered) <= len(rows)


def test_prepare_source_table_preserves_full_unsigned_simhash_range(
    tmp_path: Path,
) -> None:
    row = {
        "record_id": "record",
        "thread_id": "thread",
        "subreddit": "China",
        "year": 2020,
        "content_type": "comment",
        "retrieval_mode": "direct",
        "target_text": "bounded private text",
        "parent_id": None,
        "submission_context": None,
        "parent_context": None,
        "exact_surface_sha256": "a" * 64,
        "normalised_surface_sha256": "b" * 64,
        "simhash": (1 << 64) - 1,
        "normalised_chars": 20,
        "thread_tiebreak_sha256": "c" * 64,
        "surface_tiebreak_sha256": "d" * 64,
    }
    path = tmp_path / "source.parquet"

    pq.write_table(launcher._prepare_source_table([row]), path)
    restored = pq.read_table(path).to_pylist()

    assert restored[0]["simhash"] == (1 << 64) - 1
    assert pq.read_schema(path).field("simhash").type == pa.uint64()


def test_progress_event_is_metadata_only() -> None:
    event = launcher._progress_event(
        stage="source-shards",
        expected_shards=60,
        completed_shards=2,
        started=time.monotonic(),
        rows=10,
    )

    assert event["row_count"] == 10
    launcher.assert_metadata_only(event, where="test progress")


def test_context_requests_publish_empty_expected_subreddit_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "VOLUME_PATH", tmp_path)
    run_id = "a" * 64
    rows = [
        {"subreddit": "news", "thread_id": "thread", "parent_id": "parent"},
    ]

    first = launcher._write_context_requests(
        pre_prepare_id=run_id,
        universe=rows,
        expected_subreddits={"news", "worldnews"},
    )
    second = launcher._write_context_requests(
        pre_prepare_id=run_id,
        universe=rows,
        expected_subreddits={"news", "worldnews"},
    )

    assert first == second
    assert first["news"]["row_count"] == 2
    assert first["worldnews"]["row_count"] == 0
    empty = tmp_path / first["worldnews"]["relative_path"]
    assert pq.ParquetFile(empty).metadata.num_rows == 0


def test_empty_context_partition_is_typed_readable_and_hydratable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Volume:
        @staticmethod
        def reload() -> None:
            return None

        @staticmethod
        def commit() -> None:
            return None

    monkeypatch.setattr(launcher, "VOLUME_PATH", tmp_path)
    monkeypatch.setattr(launcher, "volume", Volume())
    pre_prepare_id = "a" * 64
    canonical_path = tmp_path / "canonical.parquet"
    request_path = tmp_path / "requests.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "unrequested",
                    "submission_id": "unrequested",
                    "content_type": "submission",
                    "text": "not selected",
                }
            ]
        ),
        canonical_path,
    )
    request_schema = pa.schema([pa.field("record_id", pa.string(), nullable=False)])
    pq.write_table(pa.Table.from_pylist([], schema=request_schema), request_path)
    canonical_descriptor = {
        **launcher._descriptor(canonical_path, root=tmp_path, row_count=1),
        "producer_receipt_sha256": "b" * 64,
    }
    request_descriptor = launcher._descriptor(request_path, root=tmp_path, row_count=0)
    monkeypatch.setattr(
        launcher,
        "_load_prepare_support",
        lambda observed: {
            "inventories": {"canonical": [canonical_descriptor]}
        }
        if observed == pre_prepare_id
        else pytest.fail("unexpected prepare ID"),
    )

    receipt = launcher.prepare_context_shard.get_raw_f()(
        pre_prepare_id,
        canonical_descriptor,
        request_descriptor,
    )
    context_root = (
        launcher._prepare_stage_root(pre_prepare_id)
        / "contexts"
        / f"partition={receipt['partition_id']}"
    )
    context_path = context_root / receipt["frame"]["relative_path"]
    assert pq.read_schema(context_path) == pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("submission_id", pa.string(), nullable=False),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("bounded_text", pa.string()),
        ]
    )
    assert pq.read_table(context_path).to_pylist() == []

    near_path = tmp_path / "near.parquet"
    near_schema = pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
                pa.field("parent_id", pa.string()),
                pa.field("content_type", pa.string(), nullable=False),
                pa.field("target_text", pa.string(), nullable=False),
                pa.field("submission_context", pa.string()),
            pa.field("parent_context", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "submission",
                    "thread_id": "submission",
                    "parent_id": None,
                    "content_type": "submission",
                    "target_text": "submission target",
                    "submission_context": None,
                    "parent_context": None,
                }
            ],
            schema=near_schema,
        ),
        near_path,
    )
    hydrated = tmp_path / "hydrated.parquet"
    summary = launcher._hydrate_context_frames(
        near_path=near_path,
        context_paths=[context_path],
        output=hydrated,
        threads=4,
    )
    assert summary["output_rows"] == 1
    assert pq.ParquetFile(hydrated).metadata.num_rows == 1


def test_streamed_parquet_projection_hash_matches_canonical_list(tmp_path: Path) -> None:
    rows = [{"a": "one", "b": 1}, {"a": "two", "b": 2}]
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    digest, count = launcher._canonical_parquet_projection_sha256(
        path, columns=("a", "b")
    )

    assert count == 2
    assert digest == launcher._canonical_sha256(rows)


def test_internal_parquet_descriptor_uses_internal_not_upstream_contract(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    path = stage / "frame.parquet"
    pq.write_table(pa.Table.from_pylist([{"value": 1}]), path)
    descriptor = launcher._descriptor(path, root=stage, row_count=1)

    assert launcher._bound_internal_parquet(
        descriptor,
        volume_root=stage,
        where="test stage frame",
    ) == path
    with pytest.raises(ValueError, match="receipt-backed descriptor is incomplete"):
        launcher._bound_receipt_parquet(
            descriptor,
            volume_root=stage,
            where="test upstream frame",
        )


def test_receipt_backed_parquet_rejects_same_size_payload_tamper(tmp_path: Path) -> None:
    path = tmp_path / "upstream.parquet"
    pq.write_table(pa.Table.from_pylist([{"value": 1}]), path)
    descriptor = {
        **launcher._descriptor(path, root=tmp_path, row_count=1),
        "producer_receipt_sha256": "c" * 64,
    }
    payload = bytearray(path.read_bytes())
    payload[8] ^= 1
    path.write_bytes(payload)

    assert path.stat().st_size == descriptor["bytes"]
    with pytest.raises(RuntimeError, match="producer-receipt binding drifted"):
        launcher._bound_receipt_parquet(
            descriptor,
            volume_root=tmp_path,
            where="tampered upstream frame",
        )


def test_exact_reducer_accepts_all_internal_source_shard_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "VOLUME_PATH", tmp_path)
    pre_prepare_id = "a" * 64
    shards = [
        {
            "shard_id": hashlib.sha256(f"shard-{index}".encode()).hexdigest(),
            "cells": [f"subreddit=test{index}/year=2020"],
        }
        for index in range(60)
    ]
    monkeypatch.setattr(
        launcher,
        "_load_prepare_support",
        lambda observed: {"shards": shards}
        if observed == pre_prepare_id
        else pytest.fail("unexpected preparation run"),
    )
    for index, shard in enumerate(shards):
        root = (
            launcher._prepare_stage_root(pre_prepare_id)
            / "shards"
            / f"shard={shard['shard_id']}"
        )
        root.mkdir(parents=True)
        frame = root / "source.parquet"
        pq.write_table(pa.Table.from_pylist([{"value": index}]), frame)
        body = {
            "kind": "modernbert-acquisition-prepare-shard-v1",
            "schema_version": "1.0.0",
            "pre_prepare_id": pre_prepare_id,
            "shard_id": shard["shard_id"],
            "cells": shard["cells"],
            "frame": launcher._descriptor(frame, root=root, row_count=1),
            "row_count": 1,
            "wall_seconds": 1.0,
            "locked_test_rows_accessed": 0,
        }
        launcher._write_immutable_json(
            root / "receipt.json",
            {**body, "receipt_id": launcher._canonical_sha256(body)},
        )

    frames, receipts, rows = launcher._validated_prepare_shard_frames(pre_prepare_id)

    assert len(frames) == len(receipts) == rows == 60


def test_cross_year_context_hydration_is_exact_and_fails_closed(tmp_path: Path) -> None:
    near = tmp_path / "near.parquet"
    context_submission = tmp_path / "submission.parquet"
    context_parent = tmp_path / "parent.parquet"
    output = tmp_path / "hydrated.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "t1_target-2024",
                    "thread_id": "t3_submission-2023",
                    "parent_id": "t1_parent-2023",
                    "content_type": "comment",
                    "target_text": "target context",
                    "submission_context": None,
                    "parent_context": None,
                }
            ]
        ),
        near,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "t3_submission-2023",
                    "submission_id": "t3_submission-2023",
                    "content_type": "submission",
                    "bounded_text": "submission context",
                }
            ]
        ),
        context_submission,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "t1_parent-2023",
                    "submission_id": "t3_submission-2023",
                    "content_type": "comment",
                    "bounded_text": "parent context",
                }
            ]
        ),
        context_parent,
    )

    coverage = launcher._hydrate_context_frames(
        near_path=near,
        context_paths=[context_submission, context_parent],
        output=output,
        threads=4,
    )
    hydrated = pq.read_table(output).to_pylist()

    assert coverage == {
        "universe_rows": 1,
        "comment_rows": 1,
        "missing_submission_contexts": 0,
        "missing_parent_contexts": 0,
        "packet_bound_violations": 0,
        "top_level_parent_contexts": 0,
        "output_rows": 1,
    }
    assert hydrated[0]["submission_context"] == "submission context"
    assert hydrated[0]["parent_context"] == "parent context"
    bad_parent = tmp_path / "bad-parent.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "t1_parent-2023",
                    "submission_id": "t3_other-thread",
                    "content_type": "comment",
                    "bounded_text": "wrong parent",
                }
            ]
        ),
        bad_parent,
    )
    with pytest.raises(RuntimeError, match="exact thread validation"):
        launcher._hydrate_context_frames(
            near_path=near,
            context_paths=[context_submission, bad_parent],
            output=tmp_path / "invalid.parquet",
            threads=4,
        )


def test_cross_year_context_hydration_retains_missing_external_context(
    tmp_path: Path,
) -> None:
    near = tmp_path / "near.parquet"
    unrelated = tmp_path / "unrelated.parquet"
    output = tmp_path / "hydrated.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "t1_target-2020",
                    "thread_id": "t3_submission-2019",
                    "parent_id": "t1_parent-2019",
                    "content_type": "comment",
                    "target_text": "target context",
                    "submission_context": None,
                    "parent_context": None,
                }
            ]
        ),
        near,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "unrelated",
                    "submission_id": "unrelated",
                    "content_type": "submission",
                    "bounded_text": "unrelated context",
                }
            ]
        ),
        unrelated,
    )

    coverage = launcher._hydrate_context_frames(
        near_path=near,
        context_paths=[unrelated],
        output=output,
        threads=4,
    )

    assert coverage["missing_submission_contexts"] == 1
    assert coverage["missing_parent_contexts"] == 1
    assert coverage["output_rows"] == 1
    hydrated = pq.read_table(output).to_pylist()[0]
    assert hydrated["submission_context"] is None
    assert hydrated["parent_context"] is None


def test_cross_year_context_hydration_does_not_duplicate_top_level_submission_parent(
    tmp_path: Path,
) -> None:
    near = tmp_path / "near.parquet"
    context = tmp_path / "context.parquet"
    output = tmp_path / "hydrated.parquet"
    submission_text = "s" * launcher._packet_module().MAX_CONTEXT_CHARS
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "t1_target",
                    "thread_id": "t3_submission",
                    "parent_id": "t3_submission",
                    "content_type": "comment",
                    "target_text": "target context",
                    "submission_context": None,
                    "parent_context": None,
                }
            ]
        ),
        near,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": "t3_submission",
                    "submission_id": "t3_submission",
                    "content_type": "submission",
                    "bounded_text": submission_text,
                }
            ]
        ),
        context,
    )

    coverage = launcher._hydrate_context_frames(
        near_path=near,
        context_paths=[context],
        output=output,
        threads=4,
    )
    hydrated = pq.read_table(output).to_pylist()[0]

    assert coverage["missing_submission_contexts"] == 0
    assert coverage["missing_parent_contexts"] == 0
    assert coverage["packet_bound_violations"] == 0
    assert coverage["top_level_parent_contexts"] == 0
    assert hydrated["submission_context"] == submission_text
    assert hydrated["parent_context"] is None
    assert launcher._packet_bounded_fields(hydrated) == {
        "target_text": "target context",
        "submission_context": submission_text,
        "parent_context": None,
    }


def test_exposure_ledger_reads_real_schemas_and_hashes_packet_bounded_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = launcher._training_module()
    teacher_path = tmp_path / "source.parquet"
    development_path = tmp_path / "development.parquet"
    legacy_path = tmp_path / "legacy.parquet"
    bridge_path = tmp_path / "bridge.json"
    long_text = "x" * 3_000
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": f"teacher-{index}",
                    "thread_id": f"teacher-thread-{index}",
                    "target_text": long_text if index == 0 else f"teacher text {index}",
                }
                for index in range(10_000)
            ]
        ),
        teacher_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "item_id": f"development-{index}",
                    "thread_id": f"development-thread-{index}",
                    "target_text": f"development text {index}",
                }
                for index in range(600)
            ]
        ),
        development_path,
    )
    legacy_schema = pa.schema(
        [
            pa.field("source_sample_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("parent_context", pa.string()),
            pa.field("submission_context", pa.string()),
            pa.field("split", pa.string(), nullable=False),
            pa.field("resolution", pa.string(), nullable=False),
            pa.field("label_json", pa.string(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_sample_id": f"legacy-{index}",
                    "target_text": f"legacy text {index}",
                    "parent_context": None,
                    "submission_context": None,
                    "split": "development" if index < 222 else "locked_test_candidate",
                    "resolution": "exact_consensus",
                    "label_json": _label(),
                }
                for index in range(452)
            ],
            schema=legacy_schema,
        ),
        legacy_path,
    )
    bridge = training._build_exposure_register(
        scope="bridge",
        thread_ids=[f"bridge-thread-{index}" for index in range(480)],
        source_artifacts={
            "synthetic": {
                "relative_path": "synthetic.json",
                "sha256": "a" * 64,
                "bytes": 1,
            }
        },
    )
    bridge_path.write_text(json.dumps(bridge, sort_keys=True), encoding="utf-8")
    bindings = {
        "source_parquet": _file_descriptor(teacher_path, root=tmp_path, row_count=10_000),
        "development_frame": _file_descriptor(development_path, root=tmp_path, row_count=600),
        "legacy_proxy": _file_descriptor(legacy_path, root=tmp_path, row_count=452),
        "bridge_exposure_register": _file_descriptor(bridge_path, root=tmp_path),
    }
    monkeypatch.setattr(
        launcher,
        "_retained_manifest",
        lambda *_args, **_kwargs: {"experiment_contract": {"bindings": bindings}},
    )
    calibration = [
        {
            "item_id": f"calibration-{index}",
            "thread_id": f"calibration-thread-{index}",
            "target_text": f"calibration text {index}",
        }
        for index in range(600)
    ]

    ledger = launcher._build_exposure_ledger({}, calibration_rows=calibration, volume_root=tmp_path)

    bounded = launcher._packet_bounded_fields(
        {"target_text": long_text, "submission_context": None, "parent_context": None}
    )["target_text"]
    assert isinstance(bounded, str)
    assert hashlib.sha256(bounded.encode()).hexdigest() in ledger["exact_surface_sha256"]
    assert hashlib.sha256(b"legacy-0").hexdigest() in ledger["record_id_sha256"]
    assert hashlib.sha256(b"bridge-thread-0").hexdigest() in ledger["thread_sha256"]
    assert hashlib.sha256(long_text.encode()).hexdigest() not in ledger["exact_surface_sha256"]


def _three_row_thread_sources(
    tmp_path: Path,
) -> tuple[dict[str, list[Path]], str, str, dict[str, str]]:
    record_ids = ["candidate-000", "candidate-001", "candidate-005"]
    exposed_id = min(
        record_ids,
        key=lambda value: launcher._canonical_sha256(["acquisition-thread-v1", value]),
    )
    eligible_ids = [value for value in record_ids if value != exposed_id]
    fallback_id = min(
        eligible_ids,
        key=lambda value: launcher._canonical_sha256(["acquisition-thread-v1", value]),
    )
    assert exposed_id == min(
        record_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest()
    )
    texts = {
        exposed_id: "exposed preliminary representative " * 12,
        eligible_ids[0]: "eligible fallback discussion with unrelated vocabulary " * 12,
        eligible_ids[1]: "secondary eligible row about a separate ordinary topic " * 12,
    }
    candidates = tmp_path / "candidates.parquet"
    languages = tmp_path / "languages.parquet"
    canonical = tmp_path / "canonical.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": record_id,
                    "content_type": "comment",
                    "retrieval_channels": ["direct_lexical"],
                }
                for record_id in record_ids
            ]
        ),
        candidates,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": record_id,
                    "provisional_status": "provisional_english",
                }
                for record_id in record_ids
            ]
        ),
        languages,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "record_id": record_id,
                    "submission_id": "shared-thread",
                    "subreddit": "China",
                    "year": 2024,
                    "content_type": "comment",
                    "text": texts[record_id],
                    "parent_id": None,
                }
                for record_id in record_ids
            ]
        ),
        canonical,
    )
    return (
        {"candidates": [candidates], "languages": [languages], "canonical": [canonical]},
        exposed_id,
        fallback_id,
        texts,
    )


@pytest.mark.parametrize("row_exclusion", ["record", "exact", "normalised", "near"])
def test_row_exposure_is_removed_before_thread_representative_ranking(
    tmp_path: Path, row_exclusion: str
) -> None:
    paths, exposed_id, fallback_id, texts = _three_row_thread_sources(tmp_path)
    exposed_text = texts[exposed_id]
    fallback_text = texts[fallback_id]
    packet = launcher._packet_module()
    exposed_bounded = launcher._packet_bounded_fields({"target_text": exposed_text})[
        "target_text"
    ]
    fallback_bounded = launcher._packet_bounded_fields({"target_text": fallback_text})[
        "target_text"
    ]
    assert isinstance(exposed_bounded, str)
    assert isinstance(fallback_bounded, str)
    exposed_normalised = launcher._normalise_surface(exposed_bounded)
    fallback_normalised = launcher._normalise_surface(fallback_bounded)
    exposed_simhash = packet._simhash(exposed_bounded)
    fallback_simhash = packet._simhash(fallback_bounded)
    assert (exposed_simhash ^ fallback_simhash).bit_count() > packet.NEAR_DUPLICATE_MAX_HAMMING
    exclusions = {
        "record_id_sha256": set(),
        "thread_sha256": set(),
        "exact_surface_sha256": set(),
        "normalised_surface_sha256": set(),
        "near_duplicate_references": [],
    }
    if row_exclusion == "record":
        exclusions["record_id_sha256"].add(hashlib.sha256(exposed_id.encode()).hexdigest())
    elif row_exclusion == "exact":
        exclusions["exact_surface_sha256"].add(
            hashlib.sha256(exposed_bounded.encode()).hexdigest()
        )
    elif row_exclusion == "normalised":
        exclusions["normalised_surface_sha256"].add(
            hashlib.sha256(exposed_normalised.encode()).hexdigest()
        )
    else:
        exclusions["near_duplicate_references"] = [
            (exposed_simhash, len(exposed_normalised))
        ]

    rows = launcher._prepare_sql_universe(
        paths=paths,
        exclusions=exclusions,
        spec=_spec(),
        reduce=False,
        include_context=False,
        threads=4,
    )

    assert [row["record_id"] for row in rows] == [fallback_id]
    assert rows[0]["thread_id"] == "shared-thread"
    assert rows[0]["target_text"] == fallback_bounded
    assert hashlib.sha256(fallback_normalised.encode()).hexdigest() not in exclusions[
        "normalised_surface_sha256"
    ]


def test_thread_exposure_removes_every_row_before_representative_ranking(
    tmp_path: Path,
) -> None:
    paths, _exposed_id, _fallback_id, _texts = _three_row_thread_sources(tmp_path)
    exclusions = {
        "record_id_sha256": set(),
        "thread_sha256": {hashlib.sha256(b"shared-thread").hexdigest()},
        "exact_surface_sha256": set(),
        "normalised_surface_sha256": set(),
        "near_duplicate_references": [],
    }

    assert (
        launcher._prepare_sql_universe(
            paths=paths,
            exclusions=exclusions,
            spec=_spec(),
            reduce=False,
            include_context=False,
            threads=4,
        )
        == []
    )


def test_calibration_is_derived_from_exact_retained_membership_not_caller_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_training = launcher._training_module()
    joined = [
        {
            "item_id": f"item-{index:05d}",
            "thread_id": f"thread-{index:05d}",
            "target_text": f"synthetic target {index}",
            "parent_context": None,
            "submission_context": None,
            "label_json": _label(),
            "quality_tier": "exact_consensus",
            "primary_training_eligible": True,
            "label_sha256": f"{index:064x}"[-64:],
            "source_row_sha256": f"{index + 10_000:064x}"[-64:],
        }
        for index in range(10_000)
    ]
    memberships: list[dict[str, object]] = []
    for index, row in enumerate(joined):
        if index < 600:
            memberships.append(
                {
                    "item_id": row["item_id"],
                    "frame": "calibration",
                    "thread_id": row["thread_id"],
                    "bridge_exposed": False,
                    "quality_tier": row["quality_tier"],
                    "primary_training_eligible": row["primary_training_eligible"],
                    "label_sha256": row["label_sha256"],
                    "source_row_sha256": row["source_row_sha256"],
                    "selection_component": "natural",
                    "selection_stratum": "synthetic",
                    "inclusion_probability_numerator": 3,
                    "inclusion_probability_denominator": 50,
                    "inclusion_probability": 0.06,
                    "probability_scope": "synthetic",
                }
            )
        else:
            memberships.append({"item_id": row["item_id"], "frame": "training"})
    membership_body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-factorised-private-membership-v1",
        "contract_digest": "1" * 64,
        "source_count": 10_000,
        "probability_designs": {},
        "rare_cell_support": [],
        "memberships": memberships,
    }
    membership = {
        **membership_body,
        "membership_id": launcher._canonical_sha256(membership_body),
    }
    split_path = tmp_path / "membership.json"
    split_path.write_text(json.dumps(membership, sort_keys=True), encoding="utf-8")
    bridge = actual_training._build_exposure_register(
        scope="bridge",
        thread_ids=[f"bridge-{index}" for index in range(480)],
        source_artifacts={
            "synthetic": {
                "relative_path": "synthetic.json",
                "sha256": "a" * 64,
                "bytes": 1,
            }
        },
    )
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(json.dumps(bridge, sort_keys=True), encoding="utf-8")
    blinded_path = tmp_path / "blinded.json"
    blinded_path.write_text("{}", encoding="utf-8")
    parquet_paths: dict[str, Path] = {}
    for name in ("teacher_ledger", "teacher_private_mapping", "source_parquet"):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table({"synthetic": pa.array(range(10_000), type=pa.int64())}), path)
        parquet_paths[name] = path
    bindings: dict[str, object] = {
        name: _file_descriptor(path, root=tmp_path, row_count=10_000)
        for name, path in parquet_paths.items()
    }
    bindings.update(
        {
            "teacher_blinded_input": _file_descriptor(blinded_path, root=tmp_path),
            "split_manifest": _file_descriptor(split_path, root=tmp_path),
            "bridge_exposure_register": _file_descriptor(bridge_path, root=tmp_path),
            "teacher_run_id": "teacher-run",
        }
    )

    class Training:
        canonical_sha256 = staticmethod(launcher._canonical_sha256)
        validate_private_exposure_register = staticmethod(
            actual_training.validate_private_exposure_register
        )

        @staticmethod
        def _joined_rows_from_teacher_artifacts(
            **_kwargs: object,
        ) -> list[dict[str, object]]:
            return joined

    monkeypatch.setattr(launcher, "_training_module", lambda: Training)
    monkeypatch.setattr(
        launcher,
        "_retained_manifest",
        lambda *_args, **_kwargs: {"experiment_contract": {"bindings": bindings}},
    )

    rows, derivation = launcher._derive_retained_calibration(
        {
            "acquisition_evaluation": {
                "relative_path": "caller-controlled-input-must-not-be-read.parquet"
            }
        },
        volume_root=tmp_path,
    )

    assert len(rows) == 600
    assert {row["item_id"] for row in rows} == {f"item-{index:05d}" for index in range(600)}
    assert derivation["membership_id"] == membership["membership_id"]
    assert derivation["row_projection_sha256"] == launcher._canonical_sha256(rows)


def test_rare_cell_freeze_recomputes_support_and_filters_only_by_frozen_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = [
        ("china_general", "negative"),
        ("china_general", "positive"),
        ("government_ccp", "negative"),
        ("government_ccp", "mixed"),
        ("people_identity", "positive"),
        ("people_identity", "mixed"),
        ("culture_media", "negative"),
        ("company_tech_product", "positive"),
    ]
    evaluation_counts = (14, 12, 12, 5, 4, 3, 2, 1)
    training_rows = [
        {"label_json": _label(target, stance)} for target, stance in cells for _ in range(20)
    ]
    training_rows.extend({"label_json": _label()} for _ in range(8_147 - len(training_rows)))
    calibration_rows = [
        {"label_json": _label(target, stance)}
        for (target, stance), count in zip(cells, evaluation_counts, strict=True)
        for _ in range(count)
    ]
    calibration_rows.extend({"label_json": _label()} for _ in range(600 - len(calibration_rows)))
    split_path = tmp_path / "membership.json"
    split_path.write_text(
        json.dumps(
            {
                "rare_cell_support": [
                    {"target": target, "stance": stance, "support": 20} for target, stance in cells
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bindings = {
        "training_frame": {"relative_path": "training.parquet"},
        "split_manifest": _file_descriptor(split_path, root=tmp_path),
    }

    class Training:
        @staticmethod
        def load_private_frame(
            *_args: object, **_kwargs: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return training_rows, {}

    monkeypatch.setattr(launcher, "_training_module", lambda: Training)
    monkeypatch.setattr(
        launcher,
        "_retained_manifest",
        lambda *_args, **_kwargs: {"experiment_contract": {"bindings": bindings}},
    )

    rare, summary = launcher._freeze_rare_cells(
        {}, calibration_rows=calibration_rows, volume_root=tmp_path
    )

    assert [(row["target"], row["stance"]) for row in rare] == sorted(cells[:3])
    assert {row["training_support"] for row in rare} == {20}
    assert summary["supported_cell_count"] == 3
    assert summary["excluded_unsupported_cell_count"] == 5


def test_candidate_assembly_uses_canonical_ontology_and_handoff_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame_rows = [
        {
            "record_id": f"opaque-{index:03d}",
            "thread_id": f"thread-{index:03d}",
            "near_duplicate_cluster_id": f"cluster-{index:03d}",
            "subreddit": f"sub-{index % 3}",
            "year": 2020 + index % 6,
            "content_type": "comment" if index % 2 else "submission",
            "retrieval_mode": "direct" if index % 3 else "expanded_only",
            "target_text": f"synthetic target {index}",
            "submission_context": None,
            "parent_context": None,
        }
        for index in range(30)
    ]
    frame_path = tmp_path / "eligible-frame.parquet"
    pq.write_table(pa.Table.from_pylist(frame_rows), frame_path)
    frame_descriptor = _file_descriptor(frame_path, root=tmp_path, row_count=30)
    runtime_digest = "9" * 64
    prepared = {
        "prepared_frame": frame_descriptor,
        "runtime_source_bundle_sha256": runtime_digest,
    }
    checkpoints = _checkpoint_trials()
    checkpoint_bundle_sha256 = launcher._canonical_sha256(checkpoints)
    for checkpoint in checkpoints:
        root = (
            launcher._checkpoint_root("a" * 64, volume_root=tmp_path)
            / f"component={checkpoint['component']}"
            / f"seed={checkpoint['optimiser_seed']}"
        )
        root.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        for index, source in enumerate(frame_rows):
            for render in ("full", "target_only"):
                common = {"opaque_id": source["record_id"], "render": render}
                if checkpoint["component"] == "relevance":
                    rows.append(
                        {
                            **common,
                            "relevance_logit": (index % 9 - 4) / 3
                            + int(checkpoint["optimiser_seed"]) / 10_000,
                        }
                    )
                else:
                    rows.append(
                        {
                            **common,
                            "target_presence_logits": [
                                (index + target_index) % 7 / 3 - 1 for target_index in range(6)
                            ],
                            "stance_logits": [
                                [
                                    (index + target_index + stance_index) % 5 / 2 - 1
                                    for stance_index in range(4)
                                ]
                                for target_index in range(5)
                            ],
                        }
                    )
        logits_path = root / "logits.parquet"
        pq.write_table(pa.Table.from_pylist(rows), logits_path)
        receipt_body = {
            "kind": launcher.SCORING_KIND,
            "schema_version": "1.0.0",
            "prepared_run_id": "a" * 64,
            "checkpoint": checkpoint,
            "prepared_frame_sha256": frame_descriptor["sha256"],
            "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
            "checkpoint_sha256": checkpoint.get("sha256", "8" * 64),
            "phase_approval_claim_id": "9" * 64,
            "checkpoint_launch_claim_id": "7" * 64,
            "approved_cost_usd": "40",
            "estimated_upper_cost_usd": "40",
            "logit_rows": 60,
            "runtime_source_bundle_sha256": runtime_digest,
            "text_renderer": "factorised-target-parent-submission-v2",
            "token_truncation": "manual-retain-final-token-max-768",
            "padding": "dynamic-per-length-bucket-batch",
            "precision": "bf16-autocast",
            "row_count": 30,
            "render_count": 2,
            "logits": _file_descriptor(logits_path, root=root, row_count=60),
            "locked_test_rows_accessed": 0,
        }
        receipt = {
            **receipt_body,
            "receipt_id": launcher._canonical_sha256(receipt_body),
        }
        (root / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))

    monkeypatch.setattr(
        launcher,
        "_load_prepared_contract",
        lambda *_args, **_kwargs: (prepared, frame_path),
    )
    monkeypatch.setattr(
        launcher,
        "_checkpoint_specs",
        lambda *_args, **_kwargs: checkpoints,
    )
    candidates = launcher._assemble_candidates(
        prepared_run_id="a" * 64,
        spec=_spec(prepared_run_id="a" * 64),
        volume_root=tmp_path,
    )
    analytic_targets = set(launcher._data_module().ANALYTIC_TARGET_CLASSES)
    for candidate in candidates:
        for seed in candidate["seed_outputs"].values():
            for render in seed.values():
                assert set(render["target_presence"]) == analytic_targets
                assert set(render["stance"]) == analytic_targets
                assert "residual_other" not in render["target_presence"]

    _policy, policy_digest = launcher.load_frozen_policy()
    small_policy = selector.AcquisitionPolicy(
        probability_rows=4,
        rare_cell_rows=1,
        boundary_rows=1,
        multi_context_rows=1,
        uncertainty_disagreement_rows=1,
        probability_seed="synthetic-probability",
        active_seed="synthetic-active",
        expected_seed_count=3,
        policy_file_sha256=policy_digest,
    )
    bindings = {
        "eligible_frame_sha256": "1" * 64,
        "source_inventory_sha256": "2" * 64,
        "exclusion_ledger_sha256": "3" * 64,
        "checkpoint_bundle_sha256": "4" * 64,
        "scoring_artifact_sha256": "5" * 64,
        "policy_file_sha256": policy_digest,
    }
    ledger, _public = selector.build_acquisition_ledger(
        candidates,
        rare_cells=[
            {"target": "china_general", "stance": "negative", "training_support": 12},
            {"target": "government_ccp", "stance": "negative", "training_support": 12},
            {"target": "people_identity", "stance": "positive", "training_support": 12},
        ],
        input_bindings=bindings,
        policy=small_policy,
    )
    ledger_path = tmp_path / "acquisition-ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    by_id = {row["record_id"]: row for row in frame_rows}
    ordered_ids = [
        row["opaque_id"] for arm in ("probability_arm", "active_arm") for row in ledger[arm]["rows"]
    ]
    source_path = tmp_path / "acquisition-source.parquet"
    source_rows = [
        {
            "opaque_id": opaque_id,
            "source_sample_id": opaque_id,
            "thread_id": by_id[opaque_id]["thread_id"],
            "target_text": by_id[opaque_id]["target_text"],
            "submission_context": None,
            "parent_context": None,
        }
        for opaque_id in ordered_ids
    ]
    pq.write_table(pa.Table.from_pylist(source_rows), source_path)
    monkeypatch.setattr(acquisition_teacher, "EXPECTED_ROWS", 8)
    monkeypatch.setattr(acquisition_teacher, "ARM_ROWS", 4)
    monkeypatch.setattr(
        acquisition_teacher,
        "ACTIVE_BUCKET_COUNTS",
        {bucket: 1 for bucket in acquisition_teacher.ACTIVE_BUCKETS},
    )

    validated_source, validated_ledger, _interleaved = (
        acquisition_teacher._validate_source_and_ledger(source_path, ledger_path)
    )
    assert [row["opaque_id"] for row in validated_source] == ordered_ids
    assert validated_ledger == ledger
    read_only_validation = launcher._validate_teacher_handoff_read_only(
        ledger=ledger, acquisition_rows=source_rows
    )
    assert read_only_validation["teacher_packet_rows"] == 8
    assert len(read_only_validation["teacher_source_projection_sha256"]) == 64
    persisted_aliases = pq.read_table(
        source_path, columns=["opaque_id", "source_sample_id"]
    ).to_pylist()
    assert all(row["opaque_id"] == row["source_sample_id"] for row in persisted_aliases)


def test_discovery_requires_exact_60_60_120_inventory(tmp_path: Path) -> None:
    candidate_root = tmp_path / launcher.STAGE_ROOT / "candidates"
    language_root = tmp_path / launcher.LANGUAGE_ROOT / "decisions"
    canonical_root = (
        tmp_path
        / "normalised"
        / launcher.DATASET_REVISION
        / f"schema={launcher.SOURCE_SCHEMA_VERSION}"
    )
    for index in range(60):
        (candidate_root / f"subreddit=s{index}" / "year=2024" / "unit=comment").mkdir(parents=True)
        (
            candidate_root
            / f"subreddit=s{index}"
            / "year=2024"
            / "unit=comment"
            / f"candidates-{'a' * 63}{index % 10}.parquet"
        ).touch()
        (language_root / f"subreddit=s{index}" / "year=2024" / "unit=comment").mkdir(parents=True)
        (
            language_root
            / f"subreddit=s{index}"
            / "year=2024"
            / "unit=comment"
            / f"language-decisions-{'b' * 63}{index % 10}.parquet"
        ).touch()
    for index in range(120):
        path = canonical_root / f"x={index}" / "year=2024" / "subreddit=s" / "content_type=comment"
        path.mkdir(parents=True)
        (path / "part-00000.parquet").touch()

    paths = launcher._discover_inputs(volume_root=tmp_path)
    assert {key: len(value) for key, value in paths.items()} == {
        "candidates": 60,
        "languages": 60,
        "canonical": 120,
    }


def test_preflight_validator_rejects_policy_digest_drift(tmp_path: Path) -> None:
    _policy, digest = launcher.load_frozen_policy()
    run_id = "a" * 64
    spec = _spec(prepared_run_id=run_id)
    spec["policy_sha256"] = digest
    receipt_path = launcher._checkpoint_root(run_id, volume_root=tmp_path) / "cuda-preflight.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "kind": launcher.PREFLIGHT_KIND,
                "prepared_run_id": run_id,
                "policy_sha256": "0" * 64,
                "checkpoint_bundle_sha256": "a" * 64,
                "gpu_type": "L4",
                "locked_test_rows_accessed": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher.validate_cuda_preflight(spec, volume_root=tmp_path)


def test_scoring_plan_and_prepared_frame_shards_are_exact_and_reusable(
    tmp_path: Path,
) -> None:
    run_id = "a" * 64
    root = launcher._prepared_root(run_id, volume_root=tmp_path)
    root.mkdir(parents=True)
    rows = [{"record_id": f"opaque-{index:03d}", "value": index} for index in range(23)]
    frame_path = root / "eligible-frame.parquet"
    pq.write_table(pa.Table.from_pylist(rows), frame_path)
    descriptor = _file_descriptor(frame_path, root=root, row_count=len(rows))

    first = launcher._materialise_prepared_frame_shards(
        run_id=run_id,
        prepared_frame_path=frame_path,
        prepared_frame=descriptor,
        volume_root=tmp_path,
    )
    second = launcher._materialise_prepared_frame_shards(
        run_id=run_id,
        prepared_frame_path=frame_path,
        prepared_frame=descriptor,
        volume_root=tmp_path,
    )

    assert first == second
    assert [row["row_count"] for row in first["frame_shards"]] == [3, 3, 3, *([2] * 7)]
    assert sum(row["row_count"] for row in first["frame_shards"]) == 23
    assert first["record_id_sequence_sha256"] == selector.canonical_sequence_sha256(
        row["record_id"] for row in rows
    )
    launcher.assert_metadata_only(first, where="synthetic prepared shard receipt")


def test_scoring_shards_bind_batch_32_and_publish_metadata_only_index(
    tmp_path: Path,
) -> None:
    prepared, _specification, jobs = _synthetic_scoring_shards(tmp_path)
    approval = launcher._validate_scoring_shard_job(
        jobs[0], prepared=prepared, volume_root=tmp_path
    )
    assert approval["batch_size"] == 32
    with pytest.raises(RuntimeError, match="plan binding"):
        launcher._validate_scoring_shard_job(
            {**jobs[0], "batch_size": 16},
            prepared=prepared,
            volume_root=tmp_path,
        )

    checkpoint = jobs[0]["checkpoint"]
    index = launcher._finalise_checkpoint_scoring_index(
        prepared_run_id="a" * 64,
        prepared=prepared,
        checkpoint=checkpoint,
        checkpoint_bundle_sha256=jobs[0]["checkpoint_bundle_sha256"],
        volume_root=tmp_path,
    )
    assert index["kind"] == launcher.SCORING_INDEX_KIND
    assert index["shard_count"] == 10
    assert index["logit_rows"] == 40
    assert "logits" not in index
    launcher.assert_metadata_only(index, where="synthetic checkpoint index")


def test_scoring_shard_union_rejects_missing_and_duplicate_rows(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    prepared, _specification, jobs = _synthetic_scoring_shards(
        missing_root, omit_shard="09"
    )
    with pytest.raises(FileNotFoundError):
        launcher._validated_checkpoint_shard_union(
            prepared_run_id="a" * 64,
            prepared=prepared,
            checkpoint=jobs[0]["checkpoint"],
            checkpoint_bundle_sha256=jobs[0]["checkpoint_bundle_sha256"],
            volume_root=missing_root,
        )

    duplicate_root = tmp_path / "duplicate"
    prepared, _specification, jobs = _synthetic_scoring_shards(
        duplicate_root, duplicate_shard="03"
    )
    with pytest.raises(RuntimeError, match="conserve identities"):
        launcher._validated_checkpoint_shard_union(
            prepared_run_id="a" * 64,
            prepared=prepared,
            checkpoint=jobs[0]["checkpoint"],
            checkpoint_bundle_sha256=jobs[0]["checkpoint_bundle_sha256"],
            volume_root=duplicate_root,
        )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("active", "active"),
        ("completed", "completed"),
        ("execution_failed", "failed"),
        ("function_timeout", "failed"),
        ("output_expired", "failed"),
        ("remote_failed", "failed"),
        ("user_code_failed", "failed"),
    ],
)
def test_persisted_function_call_state_is_exact(
    monkeypatch: pytest.MonkeyPatch, outcome: str, expected: str
) -> None:
    class Call:
        def get(self, *, timeout: int) -> object:
            assert timeout == 0
            if outcome == "active":
                raise launcher.modal.exception.TimeoutError()
            if outcome == "execution_failed":
                raise launcher.modal.exception.ExecutionError()
            if outcome == "function_timeout":
                raise launcher.modal.exception.FunctionTimeoutError()
            if outcome == "output_expired":
                raise launcher.modal.exception.OutputExpiredError()
            if outcome == "remote_failed":
                raise launcher.modal.exception.RemoteError()
            if outcome == "user_code_failed":
                raise launcher.modal.exception.UserCodeException(RuntimeError("synthetic"))
            return {"status": "complete"}

    class FunctionCall:
        @staticmethod
        def from_id(function_call_id: str) -> Call:
            assert function_call_id == "fc-synthetic"
            return Call()

    monkeypatch.setattr(launcher.modal, "FunctionCall", FunctionCall)
    assert launcher._function_call_state("fc-synthetic") == expected


def test_persisted_function_call_state_propagates_service_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Call:
        @staticmethod
        def get(*, timeout: int) -> object:
            assert timeout == 0
            raise launcher.modal.exception.ServiceError("synthetic outage")

    class FunctionCall:
        @staticmethod
        def from_id(_function_call_id: str) -> Call:
            return Call()

    monkeypatch.setattr(launcher.modal, "FunctionCall", FunctionCall)
    with pytest.raises(launcher.modal.exception.ServiceError, match="synthetic outage"):
        launcher._function_call_state("fc-synthetic")


def test_persisted_dispatch_state_checks_every_immutable_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def state(function_call_id: object) -> str:
        assert isinstance(function_call_id, str)
        observed.append(function_call_id)
        return {"fc-failed": "failed", "fc-active": "active"}[function_call_id]

    monkeypatch.setattr(launcher, "_function_call_state", state)
    assert launcher._persisted_dispatch_call_states(
        [
            {"function_call_id": "fc-failed"},
            {"function_call_id": "fc-active"},
        ]
    ) == {"failed", "active"}
    assert observed == ["fc-failed", "fc-active"]


@pytest.mark.parametrize(
    "stage",
    ["score-checkpoint", "reduce-candidate-scores", "finalise-scoring"],
)
def test_committed_dispatch_recovers_spawn_gap_without_duplicate_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    class Volume:
        commits = 0

        @classmethod
        def commit(cls) -> None:
            cls.commits += 1

    class Function:
        calls: ClassVar[list[dict[str, object]]] = []

        @classmethod
        def spawn(cls, job: dict[str, object]) -> object:
            intent_path = next(tmp_path.glob("dispatch=*/intent.json"))
            assert intent_path.is_file()
            assert Volume.commits >= 1
            cls.calls.append(job)
            return type("Call", (), {"object_id": f"fc-{len(cls.calls)}"})()

    class Leases:
        values: ClassVar[dict[str, object]] = {}

        @classmethod
        def put(cls, key: str, value: object, *, skip_if_exists: bool) -> bool:
            assert skip_if_exists is True
            if key in cls.values:
                return False
            cls.values[key] = value
            return True

        @classmethod
        def get(cls, key: str) -> object | None:
            return cls.values.get(key)

    monkeypatch.setattr(launcher, "volume", Volume)
    monkeypatch.setattr(launcher, "dispatch_leases", Leases)
    attempt_id = "attempt-0"
    dispatch_id = hashlib.sha256(stage.encode()).hexdigest()
    job = {
        "attempt_id": attempt_id,
        "dispatch_id": dispatch_id,
        "synthetic_stage": stage,
    }
    intent = launcher._dispatch_intent(
        stage=stage,
        generation=0,
        attempt_id=attempt_id,
        dispatch_id=dispatch_id,
        job=job,
    )
    record_call = launcher._record_dispatch_call

    def crash_after_spawn(**_kwargs: object) -> None:
        raise RuntimeError("injected crash after spawn")

    monkeypatch.setattr(launcher, "_record_dispatch_call", crash_after_spawn)
    with pytest.raises(RuntimeError, match="injected crash after spawn"):
        launcher._spawn_committed_dispatch(
            dispatch_root=tmp_path,
            intent=intent,
            job=job,
            function=Function,
        )
    assert launcher._load_dispatch_intents(dispatch_root=tmp_path, stage=stage) == [intent]
    assert launcher._dispatch_attempt_state(dispatch_root=tmp_path, intent=intent) == "reconcile"

    monkeypatch.setattr(launcher, "_record_dispatch_call", record_call)
    recovery_call_id = launcher._spawn_committed_dispatch(
        dispatch_root=tmp_path,
        intent=intent,
        job=job,
        function=Function,
    )
    assert recovery_call_id == "fc-2"
    assert Function.calls == [job, job]
    call_id = "fc-1"
    monkeypatch.setattr(launcher.modal, "current_function_call_id", lambda: call_id)
    owner = launcher._claim_dispatch_lease(job, stage=stage)
    assert owner is not None
    call_id = "fc-2"
    assert launcher._claim_dispatch_lease(job, stage=stage) is None
    monkeypatch.setattr(
        launcher,
        "_function_call_state",
        lambda observed: "active"
        if observed == "fc-1"
        else pytest.fail("polled non-owner call"),
    )
    assert launcher._dispatch_attempt_state(dispatch_root=tmp_path, intent=intent) == "active"
    assert launcher._load_dispatch_intents(dispatch_root=tmp_path, stage=stage) == [intent]


def test_candidate_score_attempt_recovers_without_deleting_crashed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "a" * 64
    shard = {"shard_id": "00", "start": 0, "stop": 2, "row_count": 2}
    root = launcher._candidate_score_root(run_id, volume_root=tmp_path) / "shard=00"
    crashed = root / "attempts" / "attempt=crashed.incomplete"
    crashed.mkdir(parents=True)
    (crashed / "partial").write_text("retained", encoding="utf-8")
    rows = [
        {
            "opaque_id": f"opaque-{index}",
            "thread_id": f"thread-{index}",
            "near_duplicate_cluster_id": f"cluster-{index}",
            "subreddit": "China",
            "year": 2024,
            "content_type": "comment",
            "retrieval_mode": "direct",
            "uncertainty_disagreement": 0.1 + index / 10,
            "boundary": 0.2,
            "rare_cell": 0.3,
            "multi_context": 0.4,
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        launcher,
        "_compact_candidate_score_rows",
        lambda **_kwargs: (rows, [str(index) * 64 for index in range(1, 7)]),
    )
    prepared = {"prepared_frame": {"sha256": "b" * 64, "row_count": 20}}
    rare_cells = [
        {"target": "china_general", "stance": "negative", "training_support": 12},
        {"target": "government_ccp", "stance": "negative", "training_support": 12},
        {"target": "people_identity", "stance": "positive", "training_support": 12},
    ]
    receipt = launcher._materialise_candidate_score_shard(
        prepared_run_id=run_id,
        prepared=prepared,
        checkpoint_bundle_sha256="c" * 64,
        checkpoints=[],
        rare_cells=rare_cells,
        policy=object(),
        shard=shard,
        attempt_id="resumed",
        dispatch_id="d" * 64,
        volume_root=tmp_path,
    )
    assert crashed.is_dir()
    assert receipt["attempt_id"] == "resumed"
    assert (root / "receipt.json").is_file()
    reused = launcher._materialise_candidate_score_shard(
        prepared_run_id=run_id,
        prepared=prepared,
        checkpoint_bundle_sha256="c" * 64,
        checkpoints=[],
        rare_cells=rare_cells,
        policy=object(),
        shard=shard,
        attempt_id="unused",
        dispatch_id="e" * 64,
        volume_root=tmp_path,
    )
    assert reused == receipt


def test_bounded_compact_selection_is_exactly_equal_to_selector_oracle(
    tmp_path: Path,
) -> None:
    rare_cells = [
        {"target": "china_general", "stance": "negative", "training_support": 12},
        {"target": "government_ccp", "stance": "negative", "training_support": 15},
        {"target": "people_identity", "stance": "positive", "training_support": 18},
    ]
    candidates: list[dict[str, object]] = []
    for index in range(40):
        seed_outputs: dict[str, object] = {}
        for seed_index, seed in enumerate((47, 61, 89)):
            renders: dict[str, object] = {}
            for render_index, render in enumerate(("full", "target_only")):
                relevance = 0.1 + 0.8 * ((index + seed_index + render_index) % 11) / 10
                renders[render] = {
                    "relevance": min(relevance, 0.9),
                    "target_presence": {
                        target: 0.1
                        + 0.8
                        * ((index + target_index + seed_index + render_index) % 9)
                        / 8
                        for target_index, target in enumerate(semantic_v2.ANALYTIC_TARGETS)
                    },
                    "stance": {
                        target: {
                            stance: [0.1, 0.2, 0.3, 0.4][
                                (stance_index + target_index + index) % 4
                            ]
                            for stance_index, stance in enumerate(semantic_v2.STANCES)
                        }
                        for target_index, target in enumerate(semantic_v2.ANALYTIC_TARGETS)
                    },
                }
            seed_outputs[str(seed)] = renders
        candidates.append(
            {
                "opaque_id": f"opaque-{index:03d}",
                "thread_id": f"thread-{index:03d}",
                "near_duplicate_cluster_id": f"cluster-{index:03d}",
                "subreddit": f"sub-{index % 3}",
                "year": 2020 + index % 6,
                "content_type": "comment" if index % 2 else "submission",
                "retrieval_mode": "direct" if index % 4 else "expanded_only",
                "seed_outputs": seed_outputs,
            }
        )
    _policy, policy_digest = launcher.load_frozen_policy()
    policy = selector.AcquisitionPolicy(
        probability_rows=4,
        rare_cell_rows=1,
        boundary_rows=1,
        multi_context_rows=1,
        uncertainty_disagreement_rows=1,
        probability_seed="synthetic-probability",
        active_seed="synthetic-active",
        expected_seed_count=3,
        policy_file_sha256=policy_digest,
    )
    bindings = {
        "eligible_frame_sha256": "1" * 64,
        "source_inventory_sha256": "2" * 64,
        "exclusion_ledger_sha256": "3" * 64,
        "checkpoint_bundle_sha256": "4" * 64,
        "scoring_artifact_sha256": "5" * 64,
        "policy_file_sha256": policy_digest,
    }
    expected = selector.build_acquisition_ledger(
        candidates,
        rare_cells=rare_cells,
        input_bindings=bindings,
        policy=policy,
    )
    compact = []
    for candidate in candidates:
        row = selector.compute_compact_score_record(candidate, rare_cells=rare_cells)
        ranking_source = dict(row)
        row["probability_tiebreak_sha256"] = selector.build_probability_selection_row(
            ranking_source, quota=1, population_rows=1, policy=policy
        )["selection_tiebreak_sha256"]
        for bucket in policy.bucket_quotas:
            row[f"{bucket}_tiebreak_sha256"] = selector.active_rank_key(
                ranking_source, bucket=bucket, policy=policy
            )[1]
        compact.append(row)
    score_paths = []
    for shard_index in range(10):
        path = tmp_path / f"scores-{shard_index:02d}.parquet"
        chunk = list(reversed(compact[shard_index * 4 : (shard_index + 1) * 4]))
        pq.write_table(pa.Table.from_pylist(chunk), path)
        score_paths.append(path)
    with pytest.MonkeyPatch.context():
        observed = launcher._bounded_acquisition_ledger(
            score_paths=list(reversed(score_paths)),
            rare_cells=rare_cells,
            input_bindings=bindings,
            policy=policy,
            temporary_root=tmp_path,
        )
    assert observed == expected
