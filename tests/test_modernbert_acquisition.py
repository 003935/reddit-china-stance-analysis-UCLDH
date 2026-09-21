from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from reddit_china_stance.modernbert_acquisition import (
    DEFAULT_POLICY_PATH,
    AcquisitionGatePolicy,
    AcquisitionPolicy,
    active_rank_key,
    allocate_probability_strata,
    allocate_probability_stratum_counts,
    binary_entropy,
    build_acquisition_ledger,
    build_active_selection_row,
    build_probability_selection_row,
    canonical_sequence_sha256,
    canonical_sha256,
    categorical_entropy,
    compute_candidate_scores,
    compute_compact_score_record,
    evaluate_acquisition_gate,
    finalise_acquisition_ledger,
    jensen_shannon,
    load_acquisition_policies,
    validate_acquisition_ledger,
    validate_rare_cells,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_ontology_v2 import ANALYTIC_TARGETS


def _rare_cells() -> list[dict[str, object]]:
    return [
        {"target": "government_ccp", "stance": "negative", "training_support": 5},
        {"target": "people_identity", "stance": "positive", "training_support": 20},
        {"target": "culture_media", "stance": "mixed", "training_support": 10},
    ]


def _output(index: int, seed: int, *, target_only: bool) -> dict[str, object]:
    shift = 0.04 * seed + (0.12 if target_only and index % 4 == 0 else 0.0)
    relevance = min(0.95, max(0.05, 0.25 + (index % 7) * 0.09 + shift))
    presence: dict[str, float] = {}
    stance: dict[str, dict[str, float]] = {}
    for target_index, target in enumerate(ANALYTIC_TARGETS):
        probability = min(
            0.95,
            max(0.05, 0.15 + ((index + target_index) % 6) * 0.12 + shift / 2),
        )
        presence[target] = probability
        negative = min(0.75, 0.15 + ((index + target_index + seed) % 4) * 0.1)
        positive = 0.2
        mixed = 0.15
        no_stance = 1.0 - negative - positive - mixed
        stance[target] = {
            "negative": negative,
            "positive": positive,
            "mixed": mixed,
            "no_directed_stance": no_stance,
        }
    return {
        "relevance": relevance,
        "target_presence": presence,
        "stance": stance,
    }


def _candidate(index: int) -> dict[str, object]:
    return {
        "opaque_id": f"opaque-{index:03d}",
        "thread_id": f"thread-{index:03d}",
        "near_duplicate_cluster_id": f"cluster-{index:03d}",
        "subreddit": f"sub-{index % 3}",
        "year": 2020 + (index % 2),
        "content_type": "comment" if index % 2 else "submission",
        "retrieval_mode": "lexical" if index % 3 else "anchor",
        "seed_outputs": {
            f"seed-{seed}": {
                "full": _output(index, seed, target_only=False),
                "target_only": _output(index, seed, target_only=True),
            }
            for seed in range(3)
        },
    }


def _bindings() -> dict[str, str]:
    return {
        "eligible_frame_sha256": "1" * 64,
        "source_inventory_sha256": "2" * 64,
        "exclusion_ledger_sha256": "3" * 64,
        "checkpoint_bundle_sha256": "4" * 64,
        "scoring_artifact_sha256": "5" * 64,
        "policy_file_sha256": "6" * 64,
    }


def _policy() -> AcquisitionPolicy:
    return AcquisitionPolicy(
        probability_rows=4,
        rare_cell_rows=2,
        boundary_rows=3,
        multi_context_rows=1,
        uncertainty_disagreement_rows=4,
        probability_seed="test-probability",
        active_seed="test-active",
        expected_seed_count=3,
        policy_file_sha256="6" * 64,
    )


def test_entropy_and_jensen_shannon_are_bounded_and_exact_at_extremes() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert categorical_entropy([0.25] * 4) == pytest.approx(1.0)
    assert jensen_shannon([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1.0)
    assert jensen_shannon([[0.4, 0.6], [0.4, 0.6]]) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="sum to one"):
        categorical_entropy([0.4, 0.4])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        binary_entropy(float("nan"))


def test_candidate_scores_are_deterministic_bounded_and_use_context_render() -> None:
    candidate = _candidate(4)
    score = compute_candidate_scores(candidate, rare_cells=_rare_cells())
    assert set(score) == {
        "uncertainty_disagreement",
        "boundary",
        "rare_cell",
        "multi_context",
    }
    assert all(0.0 <= value <= 1.0 for value in score.values())
    assert score == compute_candidate_scores(candidate, rare_cells=_rare_cells())
    assert score["multi_context"] >= 0.1

    broken = deepcopy(candidate)
    del broken["seed_outputs"]["seed-0"]["full"]["target_presence"][
        "government_ccp"
    ]
    with pytest.raises(ValueError, match="target set drifted"):
        compute_candidate_scores(broken, rare_cells=_rare_cells())


def test_rare_cell_contract_requires_exact_three_and_has_stable_digest() -> None:
    canonical = validate_rare_cells(_rare_cells())
    reversed_canonical = validate_rare_cells(list(reversed(_rare_cells())))
    assert canonical == reversed_canonical
    assert canonical_sha256(canonical) == canonical_sha256(reversed_canonical)
    assert len(canonical) == 3

    for invalid in (_rare_cells()[:1], _rare_cells()[:2], [*_rare_cells(), _rare_cells()[0]]):
        with pytest.raises(ValueError, match="exactly three canonical rare cells"):
            validate_rare_cells(invalid)


def test_build_and_validate_ledger_is_exact_disjoint_and_metadata_only() -> None:
    candidates = [_candidate(index) for index in range(30)]
    private, public = build_acquisition_ledger(
        candidates,
        rare_cells=_rare_cells(),
        input_bindings=_bindings(),
        policy=_policy(),
    )
    assert private["probability_arm"]["row_count"] == 4
    assert private["active_arm"]["row_count"] == 10
    assert private["active_arm"]["buckets"] == {
        "rare_cell": 2,
        "boundary": 3,
        "multi_context": 1,
        "uncertainty_disagreement": 4,
    }
    random_threads = {row["thread_id"] for row in private["probability_arm"]["rows"]}
    active_threads = {row["thread_id"] for row in private["active_arm"]["rows"]}
    assert random_threads.isdisjoint(active_threads)
    assert sum(
        row["sample_rows"] for row in private["probability_arm"]["strata"]
    ) == 4
    for row in private["probability_arm"]["rows"]:
        assert row["inclusion_probability"] == pytest.approx(
            row["inclusion_probability_numerator"]
            / row["inclusion_probability_denominator"]
        )
    assert_metadata_only(public)
    validate_acquisition_ledger(
        private,
        public,
        candidates,
        rare_cells=_rare_cells(),
        input_bindings=_bindings(),
        policy=_policy(),
    )
    private_2, public_2 = build_acquisition_ledger(
        list(reversed(candidates)),
        rare_cells=list(reversed(_rare_cells())),
        input_bindings=_bindings(),
        policy=_policy(),
    )
    assert private_2 == private
    assert public_2 == public
    assert private["ledger_id"] == (
        "3284b4f87e01c33489be9993b2cd006bcee8e172c1b826f5c5649acc98a55c70"
    )
    assert public["receipt_id"] == (
        "65f787fcb4d60c106374dd2d573c8c6cd3f02c04a138ebd6008549514849e1d8"
    )
    assert private["candidate_score_digest"] == (
        "c0a231e2a490cab71db36c892f13c0ed7ee7724b8e62b322116913b89b5f05d2"
    )
    assert canonical_sha256(private) == (
        "8809df5680d0d68571a6b26cb2974d124f5baed92704c8a236478503145bf4f3"
    )

    tampered = deepcopy(private)
    tampered["active_arm"]["rows"][0]["score"] += 0.01
    with pytest.raises(ValueError, match="private acquisition ledger drifted"):
        validate_acquisition_ledger(
            tampered,
            public,
            candidates,
            rare_cells=_rare_cells(),
            input_bindings=_bindings(),
            policy=_policy(),
        )

    cluster_overlap = deepcopy(private)
    cluster_overlap["active_arm"]["rows"][0]["near_duplicate_cluster_id"] = private[
        "probability_arm"
    ]["rows"][0]["near_duplicate_cluster_id"]
    with pytest.raises(ValueError, match="arms overlap by near-duplicate cluster"):
        validate_acquisition_ledger(
            cluster_overlap,
            public,
            candidates,
            rare_cells=_rare_cells(),
            input_bindings=_bindings(),
            policy=_policy(),
        )


def test_compact_primitives_reproduce_the_ledger_without_model_outputs() -> None:
    policy = _policy()
    rare = _rare_cells()
    compact = [
        compute_compact_score_record(
            _candidate(index),
            rare_cells=rare,
            expected_seed_count=policy.expected_seed_count,
        )
        for index in range(30)
    ]
    probability, strata = allocate_probability_strata(compact, policy)
    random_ids = {row["opaque_id"] for row in probability}
    random_clusters = {row["near_duplicate_cluster_id"] for row in probability}
    pool = [
        row
        for row in compact
        if row["opaque_id"] not in random_ids
        and row["near_duplicate_cluster_id"] not in random_clusters
    ]
    active: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selected_clusters = set(random_clusters)
    for bucket, quota in policy.bucket_quotas.items():
        accepted = 0
        ranking = sorted(
            pool, key=lambda row: active_rank_key(row, bucket=bucket, policy=policy)
        )
        for rank, row in enumerate(ranking, start=1):
            if row["opaque_id"] in selected_ids:
                continue
            if row["near_duplicate_cluster_id"] in selected_clusters:
                continue
            active.append(
                build_active_selection_row(
                    row, bucket=bucket, bucket_rank=rank, policy=policy
                )
            )
            selected_ids.add(row["opaque_id"])
            selected_clusters.add(row["near_duplicate_cluster_id"])
            accepted += 1
            if accepted == quota:
                break

    ordered = sorted(compact, key=lambda row: row["opaque_id"])
    score_digest = canonical_sequence_sha256(
        {
            "opaque_id": row["opaque_id"],
            **{bucket: row[bucket] for bucket in policy.bucket_quotas},
        }
        for row in ordered
    )
    primitive_private, primitive_public = finalise_acquisition_ledger(
        eligible_population_rows=len(compact),
        candidate_score_digest=score_digest,
        probability_rows=probability,
        probability_strata=strata,
        active_rows=active,
        rare_cells=rare,
        input_bindings=_bindings(),
        policy=policy,
    )
    oracle_private, oracle_public = build_acquisition_ledger(
        [_candidate(index) for index in range(30)],
        rare_cells=rare,
        input_bindings=_bindings(),
        policy=policy,
    )
    assert primitive_private == oracle_private
    assert primitive_public == oracle_public


def test_probability_allocation_handles_largest_remainder_ties_and_zero_quota() -> None:
    policy = replace(_policy(), probability_rows=2)
    compact = [
        compute_compact_score_record(_candidate(index), rare_cells=_rare_cells())
        for index in range(6)
    ]
    for index, row in enumerate(compact):
        row["subreddit"] = f"unique-{index}"
    selected, strata = allocate_probability_strata(compact, policy)
    assert len(selected) == 2
    assert sum(row["sample_rows"] for row in strata) == 2
    assert sum(row["sample_rows"] == 0 for row in strata) == 4

    expected = sorted(
        compact,
        key=lambda row: canonical_sha256(
            [policy.probability_seed, canonical_sha256({
                "subreddit": row["subreddit"],
                "year": row["year"],
                "content_type": row["content_type"],
                "retrieval_mode": row["retrieval_mode"],
            })]
        ),
    )[:2]
    assert {row["opaque_id"] for row in selected} == {
        row["opaque_id"] for row in expected
    }
    counts = [
        {
            "stratum": {
                "subreddit": row["subreddit"],
                "year": row["year"],
                "content_type": row["content_type"],
                "retrieval_mode": row["retrieval_mode"],
            },
            "population_rows": 1,
        }
        for row in compact
    ]
    assert allocate_probability_stratum_counts(counts, policy) == strata


def test_probability_and_active_row_builders_preserve_exact_rank_semantics() -> None:
    policy = _policy()
    first = compute_compact_score_record(_candidate(0), rare_cells=_rare_cells())
    second = compute_compact_score_record(_candidate(1), rare_cells=_rare_cells())
    first["boundary"] = second["boundary"] = 0.5
    ordered = sorted(
        [first, second],
        key=lambda row: active_rank_key(row, bucket="boundary", policy=policy),
    )
    expected = sorted(
        [first, second],
        key=lambda row: canonical_sha256(
            [policy.active_seed, "boundary", row["opaque_id"]]
        ),
    )
    assert [row["opaque_id"] for row in ordered] == [
        row["opaque_id"] for row in expected
    ]
    active = build_active_selection_row(
        ordered[1], bucket="boundary", bucket_rank=7, policy=policy
    )
    assert active["bucket_rank"] == 7
    probability = build_probability_selection_row(
        first, quota=2, population_rows=5, policy=policy
    )
    assert probability["inclusion_probability"] == pytest.approx(0.4)

    private, _ = build_acquisition_ledger(
        [_candidate(index) for index in range(30)],
        rare_cells=_rare_cells(),
        input_bindings=_bindings(),
        policy=policy,
    )
    uncertainty_ranks = [
        row["bucket_rank"]
        for row in private["active_arm"]["rows"]
        if row["bucket"] == "uncertainty_disagreement"
    ]
    assert uncertainty_ranks == [3, 4, 5, 6]


def test_canonical_sequence_digest_matches_materialised_canonical_digest() -> None:
    values = [{"z": index, "a": [index / 10]} for index in range(5)]
    assert canonical_sequence_sha256(iter(values)) == canonical_sha256(values)
    assert canonical_sequence_sha256(iter(())) == canonical_sha256([])
    with pytest.raises(ValueError, match="finite JSON"):
        canonical_sequence_sha256(iter([{"value": float("nan")}]))


def test_acquisition_ledger_fails_instead_of_topping_up_or_coercing() -> None:
    with pytest.raises(ValueError, match="active pool cannot satisfy"):
        build_acquisition_ledger(
            [_candidate(index) for index in range(13)],
            rare_cells=_rare_cells(),
            input_bindings=_bindings(),
            policy=_policy(),
        )
    duplicate = [_candidate(index) for index in range(30)]
    duplicate[1]["thread_id"] = duplicate[0]["thread_id"]
    with pytest.raises(ValueError, match="duplicate thread_id"):
        build_acquisition_ledger(
            duplicate,
            rare_cells=_rare_cells(),
            input_bindings=_bindings(),
            policy=_policy(),
        )


def _evaluation_inputs(
    *, active_is_better: bool
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, list[dict[str, object]]]],
]:
    references: list[dict[str, object]] = []
    for index in range(20):
        references.append(
            {
                "thread_id": f"eval-{index:02d}",
                "quality_tier": "exact_consensus",
                "material": True,
                "target_stances": (
                    {
                        "government_ccp": "negative",
                        "people_identity": "positive",
                    }
                    if index < 10
                    else {"culture_media": "mixed"}
                ),
            }
        )

    def arm_rows(good: bool) -> list[dict[str, object]]:
        return [
            {
                "thread_id": f"eval-{index:02d}",
                "material": True,
                "target_stances": (
                    (
                        {
                            "government_ccp": "negative",
                            "people_identity": "positive",
                        }
                        if index < 10
                        else {"culture_media": "mixed"}
                    )
                    if good
                    else {}
                ),
            }
            for index in range(20)
        ]

    predictions = {
        "random": {f"seed-{seed}": arm_rows(not active_is_better) for seed in range(3)},
        "active": {f"seed-{seed}": arm_rows(active_is_better) for seed in range(3)},
    }
    return references, predictions


def test_paired_bootstrap_gate_promotes_only_when_every_gate_passes() -> None:
    reference, predictions = _evaluation_inputs(active_is_better=True)
    rare = [
        {"target": "government_ccp", "stance": "negative", "training_support": 5},
        {"target": "people_identity", "stance": "positive", "training_support": 20},
        {"target": "culture_media", "stance": "mixed", "training_support": 10},
    ]
    policy = AcquisitionGatePolicy(
        bootstrap_replicates=500,
        bootstrap_seed="test-bootstrap",
        minimum_cell_support=10,
        minimum_mean_rare_gain=0.03,
        maximum_tuple_decline=0.01,
        maximum_material_recall_decline=0.01,
        maximum_supported_target_decline=0.05,
        policy_file_sha256="6" * 64,
    )
    gate = evaluate_acquisition_gate(
        reference,
        predictions,
        rare_cells=rare,
        policy=policy,
    )
    assert gate["verdict"] == "promote_active"
    assert gate["passed"] is True
    assert gate["mean_paired_rare_cell_macro_f1_gain"] == pytest.approx(1.0)
    assert gate["bootstrap"]["interval"]["lower"] > 0.0
    assert_metadata_only(gate)
    assert gate["gate_id"] == canonical_sha256(
        {key: value for key, value in gate.items() if key != "gate_id"}
    )

    reverse_reference, reverse_predictions = _evaluation_inputs(active_is_better=False)
    reverse = evaluate_acquisition_gate(
        reverse_reference,
        reverse_predictions,
        rare_cells=rare,
        policy=policy,
    )
    assert reverse["verdict"] == "retain_probability_random"
    assert reverse["passed"] is False
    assert not reverse["criteria"]["mean_rare_gain_at_least_0_03"]


def test_gate_rejects_identity_seed_and_support_drift() -> None:
    reference, predictions = _evaluation_inputs(active_is_better=True)
    rare = [
        {"target": "government_ccp", "stance": "negative", "training_support": 5},
        {"target": "people_identity", "stance": "positive", "training_support": 20},
        {"target": "culture_media", "stance": "mixed", "training_support": 10},
    ]
    broken = deepcopy(predictions)
    broken["active"]["seed-x"] = broken["active"].pop("seed-2")
    with pytest.raises(ValueError, match="identical optimiser seeds"):
        evaluate_acquisition_gate(
            reference,
            broken,
            rare_cells=rare,
            policy=AcquisitionGatePolicy(
                bootstrap_replicates=10,
                bootstrap_seed="test-bootstrap",
                minimum_cell_support=10,
                minimum_mean_rare_gain=0.03,
                maximum_tuple_decline=0.01,
                maximum_material_recall_decline=0.01,
                maximum_supported_target_decline=0.05,
                policy_file_sha256="6" * 64,
            ),
        )

    sparse = deepcopy(reference)
    sparse[9]["target_stances"] = {}
    with pytest.raises(ValueError, match="minimum evaluation-frame support"):
        evaluate_acquisition_gate(
            sparse,
            predictions,
            rare_cells=rare,
            policy=AcquisitionGatePolicy(
                bootstrap_replicates=10,
                bootstrap_seed="test-bootstrap",
                minimum_cell_support=10,
                minimum_mean_rare_gain=0.03,
                maximum_tuple_decline=0.01,
                maximum_material_recall_decline=0.01,
                maximum_supported_target_decline=0.05,
                policy_file_sha256="6" * 64,
            ),
        )


def test_registered_toml_is_the_default_policy_authority(tmp_path) -> None:
    acquisition, gate = load_acquisition_policies()
    assert acquisition.probability_rows == 1_000
    assert acquisition.bucket_quotas == {
        "rare_cell": 200,
        "boundary": 300,
        "multi_context": 100,
        "uncertainty_disagreement": 400,
    }
    assert acquisition.policy_file_sha256 == gate.policy_file_sha256
    assert gate.bootstrap_replicates == 10_000
    assert acquisition.as_contract()["probability_sampling_unit"] == (
        "near-duplicate-deduplicated-submission-thread"
    )

    old_unit = tmp_path / "old-unit.toml"
    old_unit.write_text(
        DEFAULT_POLICY_PATH.read_text(encoding="utf-8").replace(
            "near-duplicate-deduplicated-submission-thread", "submission_thread"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sampling unit drifted"):
        load_acquisition_policies(old_unit)

    source = DEFAULT_POLICY_PATH.read_text(encoding="utf-8")
    frame_drifts = (
        (
            "checkpoint_selection_rows = 600",
            "checkpoint_selection_rows = 599",
            "checkpoint-selection frame contract drifted",
        ),
        (
            'checkpoint_selection_source = "retained-factorised-v2-development"',
            'checkpoint_selection_source = "unused-factorised-v2-calibration-membership"',
            "checkpoint-selection frame contract drifted",
        ),
        (
            "evaluation_rows = 600",
            "evaluation_rows = 599",
            "evaluation frame contract drifted",
        ),
        (
            'evaluation_source = "unused-factorised-v2-calibration-membership"',
            'evaluation_source = "retained-factorised-v2-development"',
            "evaluation frame contract drifted",
        ),
    )
    for index, (old, new, error) in enumerate(frame_drifts):
        drifted = tmp_path / f"frame-drift-{index}.toml"
        drifted.write_text(source.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            load_acquisition_policies(drifted)
