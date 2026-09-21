from __future__ import annotations

from copy import deepcopy

import pytest

from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.privacy import assert_metadata_only


def _checkpoint_receipts() -> list[dict[str, object]]:
    return [
        {
            "receipt_id": f"{index + 1:064x}",
            "experiment_run_id": candidate.ACQUISITION_TRAINING_RUN_ID,
            "phase_run_id": candidate.ACQUISITION_TRAINING_PHASE_ID,
            "trial_id": spec.trial_id,
            "arm": "random",
            "component": spec.component,
            "optimiser_seed": spec.seed,
            "training_frame_sha256": candidate.RANDOM_TRAINING_FRAME_SHA256,
            "invalid_outputs": 0,
            "locked_test_rows_accessed": 0,
            "aggregate_metrics": {"selected_epoch": spec.selected_epoch},
            "artifacts": {
                "checkpoint": {
                    "relative_path": "checkpoint.pt",
                    "sha256": spec.checkpoint_sha256,
                    "bytes": 1_500_000_000 + index,
                }
            },
        }
        for index, spec in enumerate(candidate.CHECKPOINT_SPECS)
    ]


def _membership_rows() -> list[dict[str, object]]:
    return [
        candidate.build_membership_row(
            {
                "opaque_id": f"opaque-{index:04d}",
                "thread_id": f"thread-{index:04d}",
                "near_duplicate_cluster_id": f"cluster-{index:04d}",
                "year": 2022,
                "subreddit": "China",
                "content_type": "comment",
                "retrieval_mode": "direct",
                "target_text": f"synthetic text {index}",
                "parent_context": None,
                "submission_context": "synthetic submission",
            },
            population_rows=1_000,
            sample_rows=candidate.CALIBRATION_ROWS,
            rank=index + 1,
        )
        for index in range(candidate.CALIBRATION_ROWS)
    ]


def test_minimum_one_largest_remainder_allocation_is_exact_and_deterministic() -> None:
    strata = [
        {
            "stratum": {
                "year": 2022,
                "subreddit": "China",
                "content_type": "comment",
            },
            "population_rows": 900,
        },
        {
            "stratum": {
                "year": 2023,
                "subreddit": "Sino",
                "content_type": "submission",
            },
            "population_rows": 100,
        },
    ]
    allocated = candidate.allocate_minimum_one_quotas(strata)
    assert allocated == candidate.allocate_minimum_one_quotas(list(reversed(strata)))
    assert sum(row["sample_rows"] for row in allocated) == candidate.CALIBRATION_ROWS
    assert all(0 < row["sample_rows"] <= row["population_rows"] for row in allocated)
    assert {tuple(row["stratum"]) for row in allocated} == {
        ("year", "subreddit", "content_type")
    }


def test_membership_validation_preserves_three_field_strata_and_exact_probabilities() -> None:
    rows = _membership_rows()
    clean = candidate.validate_membership_rows(rows)
    assert len(clean) == candidate.CALIBRATION_ROWS
    assert clean[0]["inclusion_probability"] == pytest.approx(0.6)
    assert '"retrieval_mode"' not in clean[0]["selection_stratum"]

    broken = deepcopy(rows)
    broken[-1]["thread_id"] = broken[0]["thread_id"]
    with pytest.raises(ValueError, match="duplicate thread_id"):
        candidate.validate_membership_rows(broken)


def test_checkpoint_cohort_requires_every_exact_random_seed_without_seed_selection() -> None:
    cohort = candidate.freeze_checkpoint_cohort(_checkpoint_receipts())
    assert len(cohort["members"]) == 6
    assert cohort["combination"] == (
        "unweighted-arithmetic-mean-aligned-raw-logits-per-component"
    )
    assert "no-post-hoc-seed-selection" in cohort["seed_selection"]

    broken = _checkpoint_receipts()
    broken[0]["arm"] = "active"
    with pytest.raises(candidate.ProbabilityRandomCandidateContractError):
        candidate.freeze_checkpoint_cohort(broken)


def test_sampling_receipt_is_metadata_only_and_binds_closeout_and_cohort() -> None:
    receipt = candidate.sampling_public_receipt(
        membership_rows=_membership_rows(),
        membership_artifact={
            "relative_path": "private/calibration-membership.parquet",
            "sha256": "1" * 64,
            "bytes": 100,
            "row_count": candidate.CALIBRATION_ROWS,
        },
        teacher_source_artifact={
            "relative_path": "private/calibration-source.parquet",
            "sha256": "2" * 64,
            "bytes": 100,
            "row_count": candidate.CALIBRATION_ROWS,
        },
        acquisition_ledger_sha256=candidate.ACQUISITION_LEDGER_FILE_SHA256,
        source_bundle_sha256="3" * 64,
        checkpoint_cohort_id="4" * 64,
        run_id="5" * 64,
    )
    assert receipt["acquisition_closeout_id"] == candidate.ACQUISITION_CLOSEOUT_ID
    assert receipt["checkpoint_cohort_id"] == "4" * 64
    assert receipt["locked_test_access_count"] == 0
    assert_metadata_only(receipt, where="test calibration receipt")


def test_parent_closeout_validates_immutable_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "kind": "modernbert-acquisition-gate-v1",
        "gate_id": candidate.ACQUISITION_GATE_ID,
        "experiment_run_id": candidate.ACQUISITION_TRAINING_RUN_ID,
        "phase_run_id": candidate.ACQUISITION_TRAINING_PHASE_ID,
        "verdict": candidate.ACQUISITION_VERDICT,
        "passed": False,
        "locked_test_rows_accessed": 0,
        "corpus_inference_authorised": False,
        "invalid_outputs": 0,
    }
    closeout_id = candidate.canonical_sha256(body)
    monkeypatch.setattr(candidate, "ACQUISITION_CLOSEOUT_ID", closeout_id)
    closeout = {**body, "closeout_id": closeout_id}
    assert candidate.validate_parent_closeout(closeout) == closeout

    closeout["verdict"] = "retain_active"
    with pytest.raises(candidate.ProbabilityRandomCandidateContractError):
        candidate.validate_parent_closeout(closeout)
