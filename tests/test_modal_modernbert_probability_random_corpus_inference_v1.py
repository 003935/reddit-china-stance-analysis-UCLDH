from __future__ import annotations

from pathlib import Path

import pytest

from reddit_china_stance import modal_modernbert_probability_random_corpus_inference_v1 as launcher
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance import modernbert_probability_random_corpus_inference_v1 as corpus
from reddit_china_stance.privacy import assert_metadata_only


def _local_policy(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(
        launcher,
        "POLICY_RUNTIME_PATH",
        str(launcher.REPO_ROOT / launcher.POLICY_REPO_PATH),
    )
    bundle = launcher.build_source_bundle()
    return launcher._runtime_policy(bundle)


def test_source_bundle_and_separate_corpus_authority_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local_policy(monkeypatch)
    bundle = launcher.build_source_bundle()
    assert launcher.validate_source_bundle(bundle) == bundle
    authority = launcher._authority(bundle)
    assert authority["post_acquisition_corpus_rows"] == 908_141
    assert authority["calibration_members_in_scored_corpus"] == 600
    assert authority["default_label_unseen_analysis_rows"] == 907_541
    assert authority["teacher_receipt_id"].startswith("809fe852")
    assert authority["teacher_receipt_file_sha256"].startswith("d3a7bf0c")
    assert authority["locked_test_authorised"] is False
    assert authority["corpus_inference_authorised"] is True
    assert authority["provider_budget_gate_verified"] is False
    assert authority["proceed_without_provider_hard_cap_authorised"] is True
    assert "src/reddit_china_stance/human_seeded_consensus_v1.py" in launcher.REQUIRED_SOURCE_FILES
    assert_metadata_only(authority)


def test_full_launch_requires_explicit_operational_cost_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _local_policy(monkeypatch)
    launcher._require_operational_cost_authority(policy)
    with pytest.raises(RuntimeError, match="operational cost authority"):
        launcher._require_operational_cost_authority(
            {
                **policy,
                "execution": {
                    **policy["execution"],
                    "proceed_without_provider_hard_cap_authorised": False,
                },
            }
        )


def test_policy_rejects_a_changed_cost_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_policy(monkeypatch)
    original = launcher.POLICY_RUNTIME_PATH
    policy = Path(original).read_text(encoding="utf-8")
    changed = policy.replace('conservative_cost_usd = "23.979928"', 'conservative_cost_usd = "29"')
    temporary = tmp_path / "changed-policy.toml"
    temporary.write_text(changed, encoding="utf-8")
    bundle = launcher.build_source_bundle()
    files = dict(bundle["files"])
    files[launcher.POLICY_REPO_PATH] = launcher._file_sha256(temporary)
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "listed-source-bundle-v1",
        "files": files,
        "code_sha256": candidate.canonical_sha256(dict(sorted(files.items()))),
    }
    changed_bundle = {
        **body,
        "source_bundle_id": candidate.canonical_sha256(body),
    }
    monkeypatch.setattr(launcher, "POLICY_RUNTIME_PATH", str(temporary))
    with pytest.raises(RuntimeError, match="authority drifted"):
        launcher._runtime_policy(changed_bundle)


def test_gpu_and_reducer_jobs_are_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local_policy(monkeypatch)
    authority = launcher._authority(launcher.build_source_bundle())
    assert launcher._plan()["gpu_job_count"] == 120
    assert launcher._plan()["maximum_concurrent_l4s"] == 10
    assert launcher._plan()["forward_pass_shard_count"] == 720
    assert launcher._plan()["launch_parent_timeout_seconds"] == 86_400
    assert launcher._plan()["launch_parent_timeout_seconds"] > float(
        authority["compute"]["projected_gpu_seconds"]
    )
    shard = launcher._plan()["shards"][0]
    job = launcher._score_job(authority, shard)
    assert launcher._validate_score_job(job, authority) == shard
    assert job["checkpoint_count"] == len(candidate.CHECKPOINT_SPECS)
    assert job["platform_preemption_recovery_authorised"] is True
    assert job["maximum_platform_preemption_recoveries_per_call"] == 1
    assert job["maximum_platform_execution_attempts_per_call"] == 2
    attempt_tokens = launcher._attempt_tokens(job)
    assert len(attempt_tokens) == 2
    assert len(set(attempt_tokens)) == 2
    assert all(candidate.is_sha256(value) for value in attempt_tokens)
    tokens = launcher._recovery_tokens(authority)
    assert len(tokens) == 18
    assert len(set(tokens)) == 18
    assert all(candidate.is_sha256(value) for value in tokens)
    with pytest.raises(RuntimeError, match="authority drifted"):
        launcher._validate_score_job({**job, "batch_size": 32}, authority)

    reducer = launcher._reducer_job(authority, shard)
    assert launcher._validate_reducer_job(reducer, authority) == shard
    assert (
        launcher._launch_claim(authority, stage="full-scoring", jobs=[job])["retry_authorised"]
        is False
    )


def test_emitted_ensemble_receipt_schema_round_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_policy(monkeypatch)
    authority = launcher._authority(launcher.build_source_bundle())
    shard = launcher._plan()["shards"][0]
    job = launcher._score_job(authority, shard)
    descriptor = {
        "relative_path": "private/ensemble-logits.parquet",
        "sha256": "a" * 64,
        "bytes": 1,
        "row_count": shard["row_count"],
    }
    body = {
        "schema_version": corpus.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-corpus-ensemble-shard-receipt-v1",
        **{
            key: job[key]
            for key in (
                "authority_id",
                "source_bundle_id",
                "plan_id",
                "dispatch_id",
                "job_id",
                "scoring_shard",
                "checkpoint_cohort_id",
                "checkpoint_count",
                "ensemble",
                "batch_size",
                "gpu_type",
                "platform_preemption_recovery_authorised",
                "maximum_platform_preemption_recoveries_per_call",
                "maximum_platform_preemption_recoveries_global",
                "maximum_platform_execution_attempts_per_call",
                "execution_attempt_permit_gate",
            )
        },
        "precision": "bf16-autocast",
        "row_count": shard["row_count"],
        "forward_pass_rows": shard["row_count"] * len(candidate.CHECKPOINT_SPECS),
        "private_identity_sequence_sha256": "b" * 64,
        "private_logits_artifact": descriptor,
        "started_at_unix_seconds": 1.0,
        "completed_at_unix_seconds": 2.0,
        "wall_seconds": 1.0,
        "rows_per_second": float(shard["row_count"]),
        "estimated_cost_usd": "0.000222",
        "corpus_rows_accessed": shard["row_count"],
        "locked_test_rows_accessed": 0,
        "status": "complete",
        "platform_attempt_index": 0,
        "platform_preemption_recovery_count": 0,
    }
    receipt = {**body, "receipt_id": candidate.canonical_sha256(body)}
    monkeypatch.setattr(launcher, "_read_json", lambda *_args, **_kwargs: receipt)
    monkeypatch.setattr(
        launcher, "_validate_descriptor", lambda *_args, **_kwargs: tmp_path / "unused"
    )
    validated, rows = launcher._validate_score_receipt(job, authority, read_rows=False)
    assert validated == receipt
    assert rows is None
