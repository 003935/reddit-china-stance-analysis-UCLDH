from __future__ import annotations

from decimal import Decimal

import pytest

from reddit_china_stance.modal_recall_audit import (
    HARD_MAX_APPROVED_COST_USD,
    MAX_CONTAINERS,
    _audit_unit_contract,
    _reconstruct_stage_a_contract,
    app,
    audit_output_dirs,
    audit_run_id,
    enforce_cost_guardrail,
    estimate_audit_cost_usd,
    resolve_stage_a_contract,
)
from reddit_china_stance.modal_stage_a import make_production_contract, production_run_id
from reddit_china_stance.retrieval import load_retrieval_policy


def test_cost_estimate_is_conservative_and_below_track_cap() -> None:
    estimate = estimate_audit_cost_usd(
        canonical_rows=548_963_310,
        expected_challenger_screen_rows=120_000,
        units=60,
    )
    assert estimate == Decimal("12.18")
    assert estimate < HARD_MAX_APPROVED_COST_USD


def test_gpu_concurrency_respects_modal_plan_ceiling() -> None:
    assert MAX_CONTAINERS <= 10


def test_cost_guardrail_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="exceeds approved"):
        enforce_cost_guardrail(estimated_cost_usd=Decimal("12.18"), approved_cost_usd=Decimal("10"))
    with pytest.raises(ValueError, match="<= 25"):
        enforce_cost_guardrail(
            estimated_cost_usd=Decimal("12.18"), approved_cost_usd=Decimal("25.01")
        )


def test_audit_unit_paths_are_content_addressed_and_cell_specific() -> None:
    contract = {"stage_a_run_id": "a" * 64}
    unit = _audit_unit_contract(run_id="b" * 64, subreddit="news", year=2024)
    assert unit["cell"] == {"subreddit": "news", "year": 2024}
    first, first_staging = audit_output_dirs(
        contract=contract,
        run_id="b" * 64,
        subreddit="news",
        year=2024,
    )
    second, _ = audit_output_dirs(
        contract=contract,
        run_id="b" * 64,
        subreddit="news",
        year=2025,
    )
    assert first != second
    assert first.name.startswith("unit=")
    assert first_staging.name.endswith(".incomplete")


def test_audit_run_id_is_stable() -> None:
    left = audit_run_id({"b": 2, "a": 1})
    right = audit_run_id({"a": 1, "b": 2})
    assert left == right
    assert len(left) == 64


def test_stage_a_resolver_belongs_to_recall_app_boundary() -> None:
    assert resolve_stage_a_contract.app is app


def test_recall_resolver_uses_exact_stage_a_contract_algorithm() -> None:
    policy = load_retrieval_policy(__import__("pathlib").Path("configs/retrieval-policy-v1.toml"))
    partitions = [
        {
            "subreddit": subreddit,
            "year": year,
            "content_type": content_type,
            "source_file": f"{subreddit}_{content_type}s.zst",
            "source_sha256": "1" * 64,
            "partition_relative_path": f"x/{subreddit}/{year}/{content_type}.parquet",
            "partition_rows": 10 if content_type == "comment" else 2,
            "partition_bytes": 100,
            "partition_sha256": "2" * 64,
            "conversion_code_sha256": "3" * 64,
        }
        for subreddit in (
            "AskReddit",
            "China",
            "ChineseLanguage",
            "Sino",
            "funny",
            "gaming",
            "geopolitics",
            "news",
            "todayilearned",
            "worldnews",
        )
        for year in range(2020, 2026)
        for content_type in ("submission", "comment")
    ]
    kwargs = {
        "manifest": {"dataset_id": "awogies/7719723843r", "revision": "9" * 40},
        "policy": policy,
        "code_state": {"code_sha256": "4" * 64},
        "input_partitions": partitions,
    }
    expected = make_production_contract(**kwargs)
    reconstructed = _reconstruct_stage_a_contract(**kwargs)
    assert reconstructed == expected
    assert production_run_id(reconstructed) == production_run_id(expected)
