from __future__ import annotations

from decimal import Decimal

import pytest

from reddit_china_stance.modal_stage_a import (
    CONTENT_TYPES,
    SUBREDDITS,
    enforce_cost_guardrail,
    estimate_production_cost_usd,
    make_production_contract,
    production_run_id,
)
from reddit_china_stance.retrieval import load_retrieval_policy


def _partitions() -> list[dict[str, object]]:
    return [
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
        for subreddit in SUBREDDITS
        for year in range(2020, 2026)
        for content_type in CONTENT_TYPES
    ]


def _manifest() -> dict[str, object]:
    return {
        "dataset_id": "awogies/7719723843r",
        "revision": "9" * 40,
    }


def test_cost_estimate_matches_frozen_formula() -> None:
    estimate = estimate_production_cost_usd(retained_rows=548_963_310, comment_rows=520_523_632)
    assert estimate == Decimal("18.85")


def test_cost_guardrail_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="exceeds approved"):
        enforce_cost_guardrail(estimated_cost_usd=Decimal("18.85"), approved_usd=Decimal("10"))
    with pytest.raises(ValueError, match="<= 20"):
        enforce_cost_guardrail(estimated_cost_usd=Decimal("18.85"), approved_usd=Decimal("21"))


def test_production_contract_pins_exact_grid() -> None:
    policy = load_retrieval_policy(__import__("pathlib").Path("configs/retrieval-policy-v1.toml"))
    contract = make_production_contract(
        manifest=_manifest(),
        policy=policy,
        code_state={"code_sha256": "4" * 64},
        input_partitions=_partitions(),
    )
    assert contract["counts"] == {
        "partitions": 120,
        "retained_rows": 720,
        "comment_rows": 600,
        "planned_row_visits": 1320,
    }
    assert len(production_run_id(contract)) == 64
    assert contract["compute_plan"]["hard_max_estimated_cost_usd"] == "20"


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_production_contract_rejects_incomplete_or_duplicate_grid(mutation: str) -> None:
    partitions = _partitions()
    if mutation == "missing":
        partitions.pop()
    else:
        partitions[-1] = dict(partitions[0])
    policy = load_retrieval_policy(__import__("pathlib").Path("configs/retrieval-policy-v1.toml"))
    with pytest.raises(ValueError, match="exactly 120"):
        make_production_contract(
            manifest=_manifest(),
            policy=policy,
            code_state={"code_sha256": "4" * 64},
            input_partitions=partitions,
        )
