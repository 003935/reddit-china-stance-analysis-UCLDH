from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import reddit_china_stance.modernbert_factorised_splits as splits
from reddit_china_stance.modernbert_factorised_splits import (
    PRODUCTION_POLICY_DIGEST,
    EvidenceFramePolicy,
    build_evidence_frames,
    canonical_sha256,
    validate_evidence_frames,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_ontology_v2 import validate_v2_label


def _policy() -> EvidenceFramePolicy:
    return EvidenceFramePolicy(
        expected_source_rows=400,
        calibration_rows=12,
        development_rows=36,
        development_probability_rows=6,
        rare_target_stance_rows=20,
        multi_target_rows=5,
        context_available_rows=5,
        rare_cell_count=2,
        minimum_rare_cell_support=15,
        minimum_rare_cell_selected=10,
        minimum_training_rows=300,
        expected_bridge_exposure_threads=2,
        legacy_development_proxy_rows=7,
        legacy_locked_proxy_rows=9,
        calibration_seed="test-calibration",
        development_probability_seed="test-development-random",
        development_enrichment_seed="test-development-enrichment",
    )


def _label(index: int, *, not_codable: bool = False) -> dict[str, object]:
    if not_codable:
        return {"codability": "not_codable", "relevance": None, "targets": []}
    targets = (
        "china_general",
        "government_ccp",
        "people_identity",
        "culture_media",
        "company_tech_product",
    )
    stances = ("negative", "positive", "mixed", "no_directed_stance")
    selected = [
        {"target": targets[index % len(targets)], "stance": stances[index % 4]}
    ]
    if index % 11 == 0:
        selected.append(
            {
                "target": targets[(index + 1) % len(targets)],
                "stance": stances[(index + 1) % 4],
            }
        )
    return {"codability": "codable", "relevance": "material", "targets": selected}


def _refresh_bindings(row: dict[str, object]) -> None:
    label = validate_v2_label(json.loads(str(row["label_json"])))
    row["label_sha256"] = canonical_sha256(label)
    row["source_row_sha256"] = canonical_sha256(
        {
            "sample_id": row["item_id"],
            "thread_id": row["thread_id"],
            "target_text": row["target_text"],
            "submission_context": row["submission_context"],
            "parent_context": row["parent_context"],
            "subreddit": row["subreddit"],
            "year": row["year"],
            "content_type": row["content_type"],
            "retrieval_mode": row["retrieval_mode"],
        }
    )
    row["mapping_row_sha256"] = canonical_sha256(
        {
            "sample_id": row["item_id"],
            "thread_id": row["thread_id"],
            "opaque_id": row["teacher_opaque_id"],
        }
    )
    row["teacher_row_sha256"] = canonical_sha256(
        {
            "opaque_id": row["teacher_opaque_id"],
            "label_sha256": row["label_sha256"],
            "quality_tier": row["quality_tier"],
            "primary_training_eligible": row["primary_training_eligible"],
        }
    )
    row["joined_row_sha256"] = canonical_sha256(
        {
            "source_row_sha256": row["source_row_sha256"],
            "mapping_row_sha256": row["mapping_row_sha256"],
            "teacher_row_sha256": row["teacher_row_sha256"],
        }
    )


def _surface_sha256(row: dict[str, object]) -> str:
    return canonical_sha256(
        {
            "target_text": row["target_text"],
            "parent_context": row["parent_context"],
            "submission_context": row["submission_context"],
        }
    )


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(400):
        not_codable = index in {396, 397}
        label = validate_v2_label(_label(index, not_codable=not_codable))
        tier = (
            "informed_adjudication"
            if index in {394, 395}
            else ("blind_majority" if index % 3 else "exact_consensus")
        )
        primary = not not_codable and tier != "informed_adjudication" and index != 393
        row: dict[str, object] = {
            "item_id": f"item-{index:03d}",
            "teacher_opaque_id": f"opaque-{index:03d}",
            "thread_id": f"thread-{index:03d}",
            "year": 2020 + (index % 2),
            "subreddit": f"sub-{index % 2}",
            "content_type": "comment" if index % 2 else "submission",
            "retrieval_mode": "lexical" if index % 3 else "anchor",
            "target_text": f"synthetic target {index}",
            "submission_context": (
                f"synthetic submission {index}" if index % 7 == 0 else None
            ),
            "parent_context": None,
            "label_json": json.dumps(label),
            "quality_tier": tier,
            "primary_training_eligible": primary,
        }
        _refresh_bindings(row)
        rows.append(row)
    return rows


def _bridge() -> list[str]:
    return ["thread-000", "thread-001"]


def _proxy_surfaces(
    policy: EvidenceFramePolicy | None = None,
) -> dict[str, list[str]]:
    active = policy or _policy()

    def surfaces(prefix: str, count: int) -> list[str]:
        return [
            canonical_sha256(
                {
                    "target_text": f"external {prefix} target {index}",
                    "parent_context": None,
                    "submission_context": None,
                }
            )
            for index in range(count)
        ]

    return {
        "legacy_development": surfaces(
            "development", active.legacy_development_proxy_rows
        ),
        "legacy_locked": surfaces("locked", active.legacy_locked_proxy_rows),
    }


def _proxy_sample_ids(
    policy: EvidenceFramePolicy | None = None,
) -> dict[str, list[str]]:
    active = policy or _policy()
    return {
        "legacy_development": [
            f"external-development-{index}"
            for index in range(active.legacy_development_proxy_rows)
        ],
        "legacy_locked": [
            f"external-locked-{index}"
            for index in range(active.legacy_locked_proxy_rows)
        ],
    }


def _bindings(
    bridge: list[str],
    proxy_sample_ids: dict[str, list[str]],
    proxy_surfaces: dict[str, list[str]],
) -> dict[str, str]:
    return {
        "source_receipt_sha256": "1" * 64,
        "source_metadata_mapping_sha256": "2" * 64,
        "teacher_receipt_sha256": "3" * 64,
        "teacher_final_ledger_sha256": "4" * 64,
        "bridge_receipt_sha256": "5" * 64,
        "bridge_exposure_register_file_sha256": "6" * 64,
        "bridge_exposure_register_content_sha256": canonical_sha256(sorted(bridge)),
        "legacy_development_proxy_file_sha256": "7" * 64,
        "legacy_development_proxy_schema_metadata_sha256": "8" * 64,
        "legacy_development_proxy_sample_id_set_sha256": canonical_sha256(
            sorted(proxy_sample_ids["legacy_development"])
        ),
        "legacy_development_proxy_surface_set_sha256": canonical_sha256(
            sorted(proxy_surfaces["legacy_development"])
        ),
        "legacy_locked_proxy_file_sha256": "9" * 64,
        "legacy_locked_proxy_schema_metadata_sha256": "a" * 64,
        "legacy_locked_proxy_sample_id_set_sha256": canonical_sha256(
            sorted(proxy_sample_ids["legacy_locked"])
        ),
        "legacy_locked_proxy_surface_set_sha256": canonical_sha256(
            sorted(proxy_surfaces["legacy_locked"])
        ),
    }


def _build(
    rows: list[dict[str, object]] | None = None,
    *,
    bridge: list[str] | None = None,
    proxy_sample_ids: dict[str, list[str]] | None = None,
    proxy_surfaces: dict[str, list[str]] | None = None,
    bindings: dict[str, str] | None = None,
    policy: EvidenceFramePolicy | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    active_policy = policy or _policy()
    active_bridge = bridge or _bridge()
    active_proxy_ids = proxy_sample_ids or _proxy_sample_ids(active_policy)
    active_surfaces = proxy_surfaces or _proxy_surfaces(active_policy)
    return build_evidence_frames(
        _rows() if rows is None else rows,
        input_bindings=bindings
        or _bindings(active_bridge, active_proxy_ids, active_surfaces),
        bridge_exposed_thread_ids=active_bridge,
        legacy_development_proxy_sample_ids=active_proxy_ids["legacy_development"],
        legacy_development_proxy_surface_hashes=active_surfaces[
            "legacy_development"
        ],
        legacy_locked_proxy_sample_ids=active_proxy_ids["legacy_locked"],
        legacy_locked_proxy_surface_hashes=active_surfaces["legacy_locked"],
        policy=active_policy,
    )


def test_builds_exact_conserved_disjoint_frames_with_probabilities() -> None:
    rows = _rows()
    private, public = _build(rows)
    assert public["kind"] == splits.TEST_PUBLIC_KIND
    assert private["kind"] == splits.TEST_PRIVATE_KIND
    assert public["contract"]["policy_scope"] == "test-only-miniature-in-memory"
    assert public["frames"]["calibration"]["row_count"] == 12
    assert public["frames"]["development"]["row_count"] == 36
    assert public["frames"]["training"]["row_count"] == 347
    assert public["frames"]["audit_excluded"]["row_count"] == 5
    assert public["frames"]["development"]["selection_component_counts"] == {
        "development_context_available": 5,
        "development_multi_target": 5,
        "development_probability": 6,
        "development_rare_target_stance": 20,
    }
    memberships = private["memberships"]
    assert len(memberships) == len(rows)
    assert len({row["item_id"] for row in memberships}) == len(rows)
    assert len({row["thread_id"] for row in memberships}) == len(rows)
    for row in memberships:
        if row["inclusion_probability"] is not None:
            assert row["inclusion_probability"] == (
                row["inclusion_probability_numerator"]
                / row["inclusion_probability_denominator"]
            )
    assert_metadata_only(public)


def test_sparse_strata_receive_minimum_one_then_residual_proportional_quota() -> None:
    rows: list[dict[str, object]] = []
    for subreddit, size in (("large", 50), ("small", 2), ("singleton", 1)):
        rows.extend(
            {
                "item_id": f"{subreddit}-{index}",
                "thread_id": f"thread-{subreddit}-{index}",
                "year": 2020,
                "subreddit": subreddit,
                "content_type": "comment",
            }
            for index in range(size)
        )

    selected, design = splits._stratified_probability_draw(
        rows,
        total=6,
        seed="sparse-stratum-test",
        component="calibration_probability",
    )
    selected_counts = Counter(row["subreddit"] for row in selected)
    assert selected_counts == {"large": 4, "small": 1, "singleton": 1}
    population_counts = Counter(row["subreddit"] for row in rows)
    for row in selected:
        record = design[str(row["item_id"])]
        expected_numerator = selected_counts[str(row["subreddit"])]
        expected_denominator = population_counts[str(row["subreddit"])]
        assert record["inclusion_probability_numerator"] == expected_numerator
        assert record["inclusion_probability_denominator"] == expected_denominator
        assert record["inclusion_probability"] == (
            expected_numerator / expected_denominator
        )
        assert expected_numerator <= expected_denominator

    summary = splits._probability_design_summary(
        rows,
        selected,
        design,
        frame="calibration",
        component="calibration_probability",
    )
    assert summary["forced_one_stratum_count"] == 2
    assert summary["minimum_only_stratum_count"] == 2
    assert summary["represented_population_rows"] == 53
    assert summary["expected_strata"] == 3
    assert 0 < summary["design_effective_sample_size_ratio"] <= 1
    assert summary["inverse_weight_ratio"] >= 1

    groups = {
        (2020, f"sub-{index}", "comment"): [{"item_id": f"item-{index}"}]
        for index in range(4)
    }
    with pytest.raises(ValueError, match="at least the observed metadata-stratum count"):
        splits._proportional_quotas(groups, total=3)
    with pytest.raises(ValueError, match="must leave rows outside the draw"):
        splits._proportional_quotas(groups, total=4)


def test_natural_arm_design_digests_bind_exact_private_membership_records() -> None:
    private, public = _build()
    assert private["probability_designs"] == public["probability_designs"]
    for component, summary in public["probability_designs"].items():
        memberships = [
            row
            for row in private["memberships"]
            if row["selection_component"] == component
        ]
        design = {
            str(row["item_id"]): {
                field: row[field] for field in splits.NATURAL_DESIGN_DIGEST_FIELDS
            }
            for row in memberships
        }
        assert splits.natural_arm_design_digest(design) == summary["design_digest"]
        assert summary["expected_sample_rows"] == len(memberships)
        stratum_population: dict[str, int] = {}
        for row in memberships:
            stratum = str(row["selection_stratum"])
            denominator = int(row["inclusion_probability_denominator"])
            assert stratum_population.setdefault(stratum, denominator) == denominator
        assert summary["represented_population_rows"] == sum(
            stratum_population.values()
        )
        assert summary["expected_strata"] == len(stratum_population)
        probabilities = [float(row["inclusion_probability"]) for row in memberships]
        assert summary["expected_probability_min"] == min(probabilities)
        assert summary["expected_probability_max"] == max(probabilities)
        assert 0 < summary["design_effective_sample_size"] <= len(memberships)
        assert 0 < summary["design_effective_sample_size_ratio"] <= 1
        assert summary["forced_one_stratum_count"] >= 0
        assert summary["minimum_only_stratum_count"] >= (
            summary["forced_one_stratum_count"]
        )
        assert summary["inverse_weight_min"] > 0
        assert summary["inverse_weight_max"] >= summary["inverse_weight_min"]
        assert summary["inverse_weight_ratio"] >= 1.0


def test_is_deterministic_and_exactly_validates_against_frozen_inputs() -> None:
    rows = _rows()
    bridge = _bridge()
    proxy_ids = _proxy_sample_ids()
    surfaces = _proxy_surfaces()
    bindings = _bindings(bridge, proxy_ids, surfaces)
    left = _build(
        rows,
        bridge=bridge,
        proxy_sample_ids=proxy_ids,
        proxy_surfaces=surfaces,
        bindings=bindings,
    )
    right = _build(
        list(reversed(rows)),
        bridge=list(reversed(bridge)),
        proxy_sample_ids={
            key: list(reversed(value)) for key, value in proxy_ids.items()
        },
        proxy_surfaces={key: list(reversed(value)) for key, value in surfaces.items()},
        bindings=bindings,
    )
    assert left == right
    assert validate_evidence_frames(
        *left,
        rows=rows,
        input_bindings=bindings,
        bridge_exposed_thread_ids=bridge,
        legacy_development_proxy_sample_ids=proxy_ids["legacy_development"],
        legacy_development_proxy_surface_hashes=surfaces["legacy_development"],
        legacy_locked_proxy_sample_ids=proxy_ids["legacy_locked"],
        legacy_locked_proxy_surface_hashes=surfaces["legacy_locked"],
        policy=_policy(),
    ) == left


def test_internal_join_digest_and_exact_row_binding_chain_reject_mutations() -> None:
    rows = _rows()
    _, original = _build(rows)
    drifted = deepcopy(rows)
    drifted[100]["retrieval_mode"] = "changed-but-still-valid"
    _refresh_bindings(drifted[100])
    _, changed = _build(drifted)
    assert original["normalised_joined_input_digest"] != changed[
        "normalised_joined_input_digest"
    ]
    identity = original["contract"]["identity_contract"]
    assert identity["membership_item_id"] == "canonical-private-mapping.sample_id"
    assert identity["teacher_join_key"] == "private-mapping.opaque_id"

    broken_source = deepcopy(rows)
    broken_source[100]["target_text"] = "mutated without rebinding"
    with pytest.raises(ValueError, match="source_row_sha256 does not bind"):
        _build(broken_source)

    broken_mapping = deepcopy(rows)
    broken_mapping[100]["teacher_opaque_id"] = "new-opaque-100"
    with pytest.raises(ValueError, match="mapping_row_sha256 does not bind"):
        _build(broken_mapping)

    broken_teacher = deepcopy(rows)
    broken_teacher[100]["quality_tier"] = "exact_consensus"
    with pytest.raises(ValueError, match="teacher_row_sha256 does not bind"):
        _build(broken_teacher)

    broken_join = deepcopy(rows)
    broken_join[100]["retrieval_mode"] = "rebound-source-only"
    broken_join[100]["source_row_sha256"] = canonical_sha256(
        {
            "sample_id": broken_join[100]["item_id"],
            "thread_id": broken_join[100]["thread_id"],
            "target_text": broken_join[100]["target_text"],
            "submission_context": broken_join[100]["submission_context"],
            "parent_context": broken_join[100]["parent_context"],
            "subreddit": broken_join[100]["subreddit"],
            "year": broken_join[100]["year"],
            "content_type": broken_join[100]["content_type"],
            "retrieval_mode": broken_join[100]["retrieval_mode"],
        }
    )
    with pytest.raises(ValueError, match="joined_row_sha256 does not bind"):
        _build(broken_join)


def test_decision_budget_calibration_estimand_and_production_digest_are_frozen() -> None:
    _, public = _build()
    contract = public["contract"]
    assert contract["algorithms"]["probability_allocation"] == (
        splits.PROBABILITY_ALLOCATION_ALGORITHM
    )
    assert contract["probability_design"] == {
        "minimum_per_observed_metadata_stratum": 1,
        "remaining_allocation_basis": "post-minimum-residual-capacity",
        "exact_row_inclusion_probability": "stratum-quota/stratum-population",
        "allocation_may_be_disproportionate": True,
        "frame_level_analysis_requirement": (
            "inverse-recorded-inclusion-probability-weighting"
        ),
        "design_digest_schema_version": splits.EVALUATION_SCHEMA_VERSION,
        "design_digest_fields": list(splits.NATURAL_DESIGN_DIGEST_FIELDS),
        "weighting_estimator": splits.WEIGHTING_ESTIMATOR,
        "forced_one_diagnostic": (
            "counterfactual-pure-proportional-largest-remainder-zero-quota-v1"
        ),
        "effective_sample_size_diagnostic": (
            "kish-on-inverse-inclusion-probability-v1"
        ),
    }
    budget = contract["decision_budget"]
    assert budget["development"] == {
        "representation_selections": 1,
        "allowed_comparison": "B4-vs-B2",
        "checkpoint_rule_applications": 1,
        "reporting": "component-specific-no-unweighted-mixed-frame-aggregate",
    }
    assert budget["calibration"] == {
        "probability_calibration_and_threshold_freezes": 1,
        "timing": "after-representation-selection",
        "estimand": "unexposed-primary-eligible-10k-engineering-frame",
        "excluded_tier_sensitivity": "separate-analysis",
    }
    assert contract["decision_budget_enforcement"] == (
        "runtime-ledger-must-consume-and-freeze-access-receipts"
    )
    assert canonical_sha256(EvidenceFramePolicy().as_contract()) == (
        PRODUCTION_POLICY_DIGEST
    )
    assert contract["policy_digest"] != PRODUCTION_POLICY_DIGEST


def test_bridge_is_only_in_source_exclusion_and_external_proxies_have_zero_overlap() -> None:
    rows = _rows()
    bridge = _bridge()
    proxy_ids = _proxy_sample_ids()
    surfaces = _proxy_surfaces()
    private, public = _build(
        rows,
        bridge=bridge,
        proxy_sample_ids=proxy_ids,
        proxy_surfaces=surfaces,
    )
    by_thread = {row["thread_id"]: row for row in private["memberships"]}
    assert {by_thread[value]["frame"] for value in bridge} == {"training"}
    assert all(by_thread[value]["bridge_exposed"] for value in bridge)
    assert public["exposure_intersections"]["bridge"] == {
        "source_intersection_count": 2,
        "calibration_intersection_count": 0,
        "development_intersection_count": 0,
        "training_intersection_count": 2,
        "audit_intersection_count": 0,
    }
    assert set(public["exposure_intersections"]) == {"bridge"}
    proxy_audit = public["external_proxy_overlap_audit"]
    assert proxy_audit["legacy_development"] == {
        "source_row_count": 7,
        "unique_sample_id_count": 7,
        "canonical_sample_id_overlap_count": 0,
        "unique_surface_hash_count": 7,
        "exact_surface_overlap_count": 0,
    }
    assert proxy_audit["legacy_locked"] == {
        "source_row_count": 9,
        "unique_sample_id_count": 9,
        "canonical_sample_id_overlap_count": 0,
        "unique_surface_hash_count": 9,
        "exact_surface_overlap_count": 0,
    }
    assert all(
        "legacy_development_exposed" not in row
        and "legacy_locked_exposed" not in row
        for row in private["memberships"]
    )


def test_external_proxy_surface_overlap_and_digest_drift_fail_closed() -> None:
    rows = _rows()
    bridge = _bridge()
    proxy_ids = _proxy_sample_ids()
    surfaces = _proxy_surfaces()
    id_overlap = deepcopy(proxy_ids)
    id_overlap["legacy_locked"][0] = str(rows[100]["item_id"])
    id_rebound = _bindings(bridge, id_overlap, surfaces)
    with pytest.raises(ValueError, match="canonical sample-ID overlaps"):
        _build(
            rows,
            bridge=bridge,
            proxy_sample_ids=id_overlap,
            proxy_surfaces=surfaces,
            bindings=id_rebound,
        )

    omitted = deepcopy(proxy_ids)
    omitted["legacy_development"].pop()
    omitted_rebound = _bindings(bridge, omitted, surfaces)
    with pytest.raises(ValueError, match="exactly 7 unique non-empty strings"):
        _build(
            rows,
            bridge=bridge,
            proxy_sample_ids=omitted,
            proxy_surfaces=surfaces,
            bindings=omitted_rebound,
        )

    overlap = deepcopy(surfaces)
    overlap["legacy_development"][0] = _surface_sha256(rows[100])
    rebound = _bindings(bridge, proxy_ids, overlap)
    with pytest.raises(ValueError, match="exact surface overlaps"):
        _build(
            rows,
            bridge=bridge,
            proxy_sample_ids=proxy_ids,
            proxy_surfaces=overlap,
            bindings=rebound,
        )

    stale = _bindings(bridge, proxy_ids, surfaces)
    stale["legacy_locked_proxy_surface_set_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="proxy surface-set digest drifted"):
        _build(
            rows,
            bridge=bridge,
            proxy_sample_ids=proxy_ids,
            proxy_surfaces=surfaces,
            bindings=stale,
        )


def test_enrichment_rows_match_registered_exclusive_components_and_guarantees() -> None:
    rows = _rows()
    source_by_id = {row["item_id"]: row for row in rows}
    private, public = _build(rows)
    enriched = [
        row
        for row in private["memberships"]
        if row["selection_component"].startswith("development_")
        and row["selection_component"] != "development_probability"
    ]
    contexts = [
        row
        for row in enriched
        if row["selection_component"].endswith("context_available")
    ]
    assert all(
        source_by_id[row["item_id"]]["submission_context"]
        or source_by_id[row["item_id"]]["parent_context"]
        for row in contexts
    )
    assert Counter(row["selection_component"] for row in enriched) == {
        "development_context_available": 5,
        "development_multi_target": 5,
        "development_rare_target_stance": 20,
    }
    assert len(private["rare_cell_support"]) == 2
    assert all(row["source_support"] >= 15 for row in private["rare_cell_support"])
    assert all(
        row["remaining_support"] >= row["candidate_support"] >= 10
        for row in private["rare_cell_support"]
    )
    assert all(row["selected_support"] >= 10 for row in private["rare_cell_support"])
    assert sum(row["selected_support"] for row in private["rare_cell_support"]) == 20
    selected_by_anchor = Counter(
        row["selection_stratum"]
        for row in enriched
        if row["selection_component"] == "development_rare_target_stance"
    )
    assert sorted(selected_by_anchor.values()) == sorted(
        row["selected_support"] for row in private["rare_cell_support"]
    )
    assert public["rare_enrichment_support_summary"]["selected_support_min"] >= 10
    assert public["contract"]["policy"]["enrichment_priority"] == [
        "rare_target_stance",
        "context_available",
        "multi_target",
    ]
    assert (
        public["contract"]["algorithms"]["rare_enrichment"]
        == splits.RARE_ENRICHMENT_ALGORITHM
    )
    support = public["enrichment_candidate_counts"]
    assert support["rare_target_stance"] >= 20
    assert support["context_available"] >= 5
    assert support["multi_target"] >= 5
    post_natural_rows = (
        public["probability_designs"]["development_probability"][
            "represented_population_rows"
        ]
        - 6
    )
    assert support["post_rare_remaining"] == (
        post_natural_rows - support["rare_target_stance"]
    )
    assert support["post_context_remaining"] == (
        support["post_rare_remaining"] - support["context_available"]
    )
    assert "rare_cell_set_digest" not in public


def test_rare_cell_selection_uses_post_natural_exclusive_support() -> None:
    cell_a = ("china_general", "negative")
    cell_b = ("government_ccp", "positive")
    cell_c = ("people_identity", "mixed")
    source_rows = [
        {"target_cells": (cell_a,)} for _ in range(20)
    ] + [
        {"target_cells": (cell_b,)} for _ in range(20)
    ] + [
        {"target_cells": (cell_c,)} for _ in range(25)
    ]
    candidate_rows = [
        {"target_cells": (cell_a,)} for _ in range(9)
    ] + [
        {"target_cells": (cell_b,)} for _ in range(10)
    ] + [
        {"target_cells": (cell_c,)} for _ in range(11)
    ]

    chosen, source, remaining, exclusive = splits._rare_cells(
        source_rows,
        candidate_rows,
        policy=_policy(),
    )

    assert chosen == (cell_b, cell_c)
    assert source[cell_b] == 20 and source[cell_c] == 25
    assert remaining[cell_a] == 9
    assert [exclusive[cell] for cell in chosen] == [10, 11]


def test_rare_anchor_uses_remaining_then_source_then_cell_identity() -> None:
    cell_a = ("china_general", "negative")
    cell_b = ("government_ccp", "positive")
    row = {"target_cells": (cell_a, cell_b)}
    rare = frozenset({cell_a, cell_b})

    assert splits._rare_anchor(
        row,
        rare_cells=rare,
        remaining_support={cell_a: 11, cell_b: 10},
        source_support={cell_a: 15, cell_b: 30},
    ) == cell_b
    assert splits._rare_anchor(
        row,
        rare_cells=rare,
        remaining_support={cell_a: 10, cell_b: 10},
        source_support={cell_a: 15, cell_b: 30},
    ) == cell_a
    assert splits._rare_anchor(
        row,
        rare_cells=rare,
        remaining_support={cell_a: 10, cell_b: 10},
        source_support={cell_a: 15, cell_b: 15},
    ) == cell_a


def test_overlapping_occurrence_support_cannot_fake_rare_capacity() -> None:
    cell_a = ("china_general", "negative")
    cell_b = ("government_ccp", "positive")
    source_rows = [
        {"target_cells": (cell_a, cell_b)} for _ in range(20)
    ]
    candidate_rows = [
        {"target_cells": (cell_a, cell_b)} for _ in range(10)
    ]

    with pytest.raises(ValueError, match="exclusive rare-cell capacity"):
        splits._rare_cells(source_rows, candidate_rows, policy=_policy())


def test_quality_mask_routes_not_codable_adjudicated_and_ineligible_to_audit() -> None:
    private, _ = _build()
    audit = {
        row["item_id"]: row["selection_component"]
        for row in private["memberships"]
        if row["frame"] == "audit_excluded"
    }
    assert set(audit.values()) == {
        "not_codable",
        "informed_adjudication",
        "upstream_primary_ineligible",
    }


def test_rejects_global_binding_drift_missing_hashes_and_duplicate_threads() -> None:
    rows = _rows()
    bridge = _bridge()
    proxy_ids = _proxy_sample_ids()
    surfaces = _proxy_surfaces()
    private, public = _build(
        rows,
        bridge=bridge,
        proxy_sample_ids=proxy_ids,
        proxy_surfaces=surfaces,
    )
    drifted = _bindings(bridge, proxy_ids, surfaces)
    drifted["teacher_receipt_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="drifted from the frozen inputs"):
        validate_evidence_frames(
            private,
            public,
            rows=rows,
            input_bindings=drifted,
            bridge_exposed_thread_ids=bridge,
            legacy_development_proxy_sample_ids=proxy_ids[
                "legacy_development"
            ],
            legacy_development_proxy_surface_hashes=surfaces[
                "legacy_development"
            ],
            legacy_locked_proxy_sample_ids=proxy_ids["legacy_locked"],
            legacy_locked_proxy_surface_hashes=surfaces["legacy_locked"],
            policy=_policy(),
        )

    missing_hash = deepcopy(rows)
    missing_hash[0]["label_sha256"] = None
    with pytest.raises(ValueError, match="label_sha256"):
        _build(missing_hash)

    duplicate_group = deepcopy(rows)
    duplicate_group[2]["thread_id"] = duplicate_group[3]["thread_id"]
    _refresh_bindings(duplicate_group[2])
    with pytest.raises(ValueError, match="one-row-per-thread"):
        _build(duplicate_group)

    duplicate_teacher_key = deepcopy(rows)
    duplicate_teacher_key[2]["teacher_opaque_id"] = duplicate_teacher_key[3][
        "teacher_opaque_id"
    ]
    _refresh_bindings(duplicate_teacher_key[2])
    with pytest.raises(ValueError, match="duplicate teacher_opaque_id"):
        _build(duplicate_teacher_key)


def test_rejects_unsupported_strata_and_insufficient_enrichment_support() -> None:
    unsupported = _rows()
    unsupported[5]["content_type"] = "poll"
    with pytest.raises(ValueError, match="outside the registered source strata"):
        _build(unsupported)

    no_context = _rows()
    for row in no_context:
        row["submission_context"] = None
        row["parent_context"] = None
        _refresh_bindings(row)
    with pytest.raises(ValueError, match="insufficient context_available support"):
        _build(no_context)

    no_rare_capacity = _rows()
    residual_label = {
        "codability": "codable",
        "relevance": "material",
        "targets": [
            {"target": "residual_other", "stance": None}
        ],
    }
    validate_v2_label(residual_label)
    for row in no_rare_capacity:
        row["label_json"] = json.dumps(residual_label, sort_keys=True)
        _refresh_bindings(row)
    with pytest.raises(ValueError, match="post-natural analytical"):
        _build(no_rare_capacity)

    no_multi = _rows()
    for row in no_multi:
        label = validate_v2_label(json.loads(str(row["label_json"])))
        label["targets"] = label["targets"][:1]
        row["label_json"] = json.dumps(label, sort_keys=True)
        _refresh_bindings(row)
    with pytest.raises(ValueError, match="insufficient multi_target support"):
        _build(no_multi)


def test_test_only_writer_enforces_registered_ignored_disjoint_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_parent = tmp_path / "registered-private"
    public_parent = tmp_path / "registered-public"
    monkeypatch.setattr(splits, "REGISTERED_TEST_PRIVATE_PARENT", private_parent)
    monkeypatch.setattr(splits, "REGISTERED_TEST_PUBLIC_PARENT", public_parent)
    monkeypatch.setattr(splits, "_git_check_ignored", lambda _: True)

    with pytest.raises(ValueError, match="outside the registered git-ignored parent"):
        splits._validate_test_writer_roots(
            private_root=tmp_path / "wrong-private",
            public_root=public_parent / "run",
        )

    monkeypatch.setattr(splits, "REGISTERED_TEST_PRIVATE_PARENT", tmp_path)
    monkeypatch.setattr(splits, "REGISTERED_TEST_PUBLIC_PARENT", tmp_path)
    with pytest.raises(ValueError, match="strictly disjoint"):
        splits._validate_test_writer_roots(
            private_root=tmp_path / "same",
            public_root=tmp_path / "same",
        )

    monkeypatch.setattr(splits, "REGISTERED_TEST_PRIVATE_PARENT", private_parent)
    monkeypatch.setattr(splits, "REGISTERED_TEST_PUBLIC_PARENT", public_parent)
    monkeypatch.setattr(splits, "_git_check_ignored", lambda _: False)
    with pytest.raises(ValueError, match="not git-ignored"):
        splits._validate_test_writer_roots(
            private_root=private_parent / "run",
            public_root=public_parent / "run",
        )

    monkeypatch.setattr(splits, "_git_check_ignored", lambda _: True)
    bridge = _bridge()
    proxy_ids = _proxy_sample_ids()
    surfaces = _proxy_surfaces()
    written = splits._write_evidence_frames_test_only(
        private_root=private_parent / "run",
        public_root=public_parent / "run",
        rows=_rows(),
        input_bindings=_bindings(bridge, proxy_ids, surfaces),
        bridge_exposed_thread_ids=bridge,
        legacy_development_proxy_sample_ids=proxy_ids["legacy_development"],
        legacy_development_proxy_surface_hashes=surfaces["legacy_development"],
        legacy_locked_proxy_sample_ids=proxy_ids["legacy_locked"],
        legacy_locked_proxy_surface_hashes=surfaces["legacy_locked"],
        policy=_policy(),
    )
    assert written["private_membership"].is_file()
    assert written["public_manifest"].is_file()


def test_test_only_writer_rejects_production_policy() -> None:
    with pytest.raises(ValueError, match="sole production prepare boundary"):
        splits._write_evidence_frames_test_only(
            private_root=Path("unused-private"),
            public_root=Path("unused-public"),
            rows=[],
            input_bindings={},
            bridge_exposed_thread_ids=[],
            legacy_development_proxy_sample_ids=[],
            legacy_development_proxy_surface_hashes=[],
            legacy_locked_proxy_sample_ids=[],
            legacy_locked_proxy_surface_hashes=[],
            policy=EvidenceFramePolicy(),
        )
