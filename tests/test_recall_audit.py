from __future__ import annotations

from decimal import Decimal

import pytest

from reddit_china_stance.recall_audit import (
    HASH_SPACE,
    HashProbability,
    assert_id_only_rows,
    make_audit_contract,
    make_cell_plan,
    merge_selected_rows,
    probability_for_decimal,
    probability_for_expected_count,
    score_band,
    selected_by_hash,
    validate_score_bands,
)


def _bands() -> list[dict[str, object]]:
    return [
        {
            "name": "low",
            "min_score": -1,
            "max_score": 0.30,
            "phase2_probability": "0.001",
        },
        {
            "name": "medium",
            "min_score": 0.30,
            "max_score": 0.45,
            "phase2_probability": "0.005",
        },
        {
            "name": "high",
            "min_score": 0.45,
            "max_score": 1.0000001,
            "phase2_probability": "0.02",
        },
    ]


def test_hash_probability_is_exact_and_deterministic() -> None:
    probability = probability_for_expected_count(population_rows=100, expected_rows=10)
    assert probability.threshold == (10 * HASH_SPACE) // 100
    assert probability.probability == probability.threshold / HASH_SPACE
    first = selected_by_hash(
        record_id="t1_abc", seed="audit-v1", arm="outside", probability=probability
    )
    second = selected_by_hash(
        record_id="t1_abc", seed="audit-v1", arm="outside", probability=probability
    )
    assert first is second
    assert not selected_by_hash(
        record_id="t1_abc",
        seed="audit-v1",
        arm="outside",
        probability=HashProbability(0),
    )
    assert selected_by_hash(
        record_id="t1_abc",
        seed="audit-v1",
        arm="outside",
        probability=HashProbability(HASH_SPACE),
    )


def test_cell_plan_conserves_population_and_exposes_probabilities() -> None:
    plan = make_cell_plan(
        subreddit="news",
        year=2024,
        content_type="comment",
        canonical_rows=1_000,
        candidate_rows=100,
        expected_inside_rows=10,
        expected_outside_rows=9,
        expected_challenger_screen_rows=90,
    )
    assert plan["noncandidate_rows"] == 900
    assert plan["inside_probability_arm"]["probability"] == pytest.approx(0.1)
    assert plan["outside_probability_arm"]["probability"] == pytest.approx(0.01)
    assert plan["challenger_phase1"]["probability"] == pytest.approx(0.1)


def test_score_bands_are_exhaustive_and_phase_two_is_explicit() -> None:
    bands = validate_score_bands(_bands())
    assert score_band(0.5, bands)["name"] == "high"
    assert bands[1]["phase2"]["probability"] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="contiguous"):
        validate_score_bands([_bands()[0], _bands()[2]])
    with pytest.raises(ValueError, match="between zero and one"):
        probability_for_decimal(Decimal("1.1"))


def test_contract_requires_exact_120_cell_accounting_and_inside_arm() -> None:
    cells = [
        make_cell_plan(
            subreddit=f"subreddit-{index // 12}",
            year=2020 + ((index // 2) % 6),
            content_type=("submission", "comment")[index % 2],
            canonical_rows=100,
            candidate_rows=10,
            expected_inside_rows=1,
            expected_outside_rows=1,
            expected_challenger_screen_rows=10,
        )
        for index in range(120)
    ]
    contract = make_audit_contract(
        stage_a_run_id="a" * 64,
        dataset_revision="b" * 40,
        retrieval_policy_digest="c" * 64,
        cells=cells,
        seed="recall-audit-v1",
        expected_inside_rows_per_cell=1,
        expected_outside_rows_per_cell=1,
        expected_challenger_screen_rows_per_cell=10,
        challenger={"model_id": "example/model", "revision": "d" * 40},
        score_bands=_bands(),
        stop_rules={"pass_recall_lower_bound": 0.9},
        cost_guardrail={"estimated_cost_usd": "10", "approved_cost_usd": "20"},
        code_state={"code_sha256": "e" * 64},
    )
    assert contract["design"]["candidate_probability_arm"]["purpose"].startswith(
        "estimate relevant Stage A"
    )
    assert len(contract["cells"]) == 120
    with pytest.raises(ValueError, match="exactly 120"):
        make_audit_contract(
            stage_a_run_id="a" * 64,
            dataset_revision="b" * 40,
            retrieval_policy_digest="c" * 64,
            cells=cells[:-1],
            seed="recall-audit-v1",
            expected_inside_rows_per_cell=1,
            expected_outside_rows_per_cell=1,
            expected_challenger_screen_rows_per_cell=10,
            challenger={},
            score_bands=_bands(),
            stop_rules={},
            cost_guardrail={},
            code_state={},
        )


def test_packet_blinds_channels_while_ledger_preserves_design_weights() -> None:
    cell = {"subreddit": "news", "year": 2024, "content_type": "comment"}
    packet, ledger = merge_selected_rows(
        audit_run_id="a" * 64,
        cell=cell,
        probability_rows=[
            {
                "record_id": "t1_abc",
                "is_stage_a_candidate": False,
                "outside_probability_inclusion_probability": 0.01,
            }
        ],
        challenger_rows=[
            {
                "record_id": "t1_abc",
                "is_stage_a_candidate": False,
                "challenger_phase1_inclusion_probability": 0.1,
                "challenger_phase2_inclusion_probability": 0.02,
                "challenger_combined_inclusion_probability": 0.002,
                "challenger_score": 0.61,
                "challenger_score_band": "high",
            }
        ],
    )
    assert len(packet) == len(ledger) == 1
    assert "selection_channels" not in packet[0]
    assert "challenger_score" not in packet[0]
    assert ledger[0]["selection_channels"] == [
        "noncandidate_probability",
        "challenger_two_phase",
    ]
    assert ledger[0]["challenger_combined_inclusion_probability"] == 0.002
    assert ledger[0]["record_id"] == packet[0]["record_id"]


def test_publishable_rows_reject_raw_text_fields() -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        assert_id_only_rows([{"record_id": "t1_abc", "text": "not allowed"}])
