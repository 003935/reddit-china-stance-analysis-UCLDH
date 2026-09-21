from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

import reddit_china_stance.modal_stage_a as stage_a_module
import reddit_china_stance.modal_stage_a_language as language_module
from reddit_china_stance.modal_stage_a_language import (
    FORBIDDEN_OUTPUT_FIELDS,
    agreement_bucket,
    allocate_stratified_quotas,
    benchmark_output_field_names,
    enforce_cost_guardrail,
    enforce_smoke_cost_guardrail,
    estimate_benchmark_incremental_cost_usd,
    estimate_cell_cost_usd,
    estimate_language_cost_usd,
    language_output_field_names,
    load_language_policy,
    provisional_status,
    retrieval_channel_bucket,
    select_bottom_k,
)


@pytest.fixture
def policy() -> dict[str, object]:
    return load_language_policy(Path("configs/language-policy-v1.toml"))


def test_language_runner_is_bound_to_completed_stage_a_v3() -> None:
    assert language_module.FROZEN_STAGE_A_RUN_ID == (
        "07556d5f8f472560e5e09008b97fd138bfb2bc6312d018fd8662c76ea3a873f7"
    )
    assert language_module.FROZEN_RETRIEVAL_POLICY_DIGEST == (
        "5b7ced047d90407ed8849bcec8744078ed768e74eb78851624c67c009b684d46"
    )
    assert language_module.EXPECTED_STAGE_A_CANDIDATE_RECEIPTS == 60
    assert language_module.EXPECTED_STAGE_A_CANDIDATE_ROWS == 14_487_562
    source = Path("src/reddit_china_stance/modal_stage_a_language.py").read_text()
    assert 'retrieval_policy_path: str = "configs/retrieval-policy-v3.toml"' in source


def test_language_resolver_rejects_older_stage_a_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_a_contract = {
        "retrieval_policy": {"policy_digest": language_module.FROZEN_RETRIEVAL_POLICY_DIGEST}
    }
    monkeypatch.setattr(language_module, "production_run_id", lambda _: "v2-run")
    with pytest.raises(RuntimeError, match="frozen completed v3"):
        language_module._require_frozen_stage_a_binding(
            stage_a_contract=stage_a_contract,
            stage_a_run_manifest_id="v2-run",
        )

    monkeypatch.setattr(
        language_module,
        "production_run_id",
        lambda _: language_module.FROZEN_STAGE_A_RUN_ID,
    )
    stage_a_contract["retrieval_policy"]["policy_digest"] = (
        "9e8c9050b9512fd83b2a844d4259bd3df8e3f84dc213460fc19407238266769a"
    )
    with pytest.raises(RuntimeError, match="frozen v3 policy"):
        language_module._require_frozen_stage_a_binding(
            stage_a_contract=stage_a_contract,
            stage_a_run_manifest_id=language_module.FROZEN_STAGE_A_RUN_ID,
        )


def test_policy_pins_models_and_prohibits_unaudited_exclusion(
    policy: dict[str, object],
) -> None:
    assert len(policy["policy_digest"]) == 64  # type: ignore[arg-type]
    assert policy["primary"]["model_sha256"] == (  # type: ignore[index]
        "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
    )
    assert policy["challenger"]["package_version"] == "2.2.0"  # type: ignore[index]
    assert policy["triage"]["exclude_without_human_audit"] is False  # type: ignore[index]
    assert policy["compute"]["hard_max_benchmark_cost_usd"] == 2  # type: ignore[index]
    assert policy["compute"]["hard_max_estimated_cost_usd"] == 25  # type: ignore[index]


def test_channel_buckets_are_mutually_exclusive() -> None:
    assert retrieval_channel_bucket(["direct_lexical"]) == "direct_lexical"
    assert retrieval_channel_bucket(["submission_expansion", "direct_lexical"]) == "multi_channel"
    with pytest.raises(ValueError, match="non-empty subset"):
        retrieval_channel_bucket([])


def test_stratified_quotas_are_exact_deterministic_and_bounded() -> None:
    counts = {"a": 100, "b": 20, "c": 5, "empty": 0}
    first = allocate_stratified_quotas(counts, sample_size=60, minimum_per_nonempty_stratum=5)
    second = allocate_stratified_quotas(
        dict(reversed(list(counts.items()))),
        sample_size=60,
        minimum_per_nonempty_stratum=5,
    )
    assert first == second
    assert sum(first.values()) == 60
    assert first["c"] == 5
    assert all(first[key] <= counts[key] for key in first)


def test_stratified_quotas_fail_when_minimums_exceed_sample() -> None:
    with pytest.raises(ValueError, match="minimums exceed"):
        allocate_stratified_quotas(
            {"a": 10, "b": 10}, sample_size=5, minimum_per_nonempty_stratum=3
        )


def test_bottom_hash_sample_is_order_invariant_and_exact() -> None:
    rows = [(f"t1_{index}", "a" if index % 2 else "b") for index in range(40)]
    quotas = {"a": 5, "b": 7}
    first = select_bottom_k(rows, quotas, seed="fixed")
    second = select_bottom_k(list(reversed(rows)), quotas, seed="fixed")
    assert first == second
    assert len(first) == 12
    assert sum(key == "a" for key, _ in first.values()) == 5
    assert sum(key == "b" for key, _ in first.values()) == 7


def test_language_schemas_are_metadata_only() -> None:
    assert not FORBIDDEN_OUTPUT_FIELDS.intersection(language_output_field_names())
    assert not FORBIDDEN_OUTPUT_FIELDS.intersection(benchmark_output_field_names())
    assert "record_id" in language_output_field_names()
    assert "exclusion_allowed" in language_output_field_names()
    assert "primary_top_predictions" in language_output_field_names()
    assert "primary_top_codes" not in language_output_field_names()


def test_language_schema_structurally_pairs_ranked_code_and_confidence() -> None:
    schema = json.loads(Path("schemas/language-decision.schema.json").read_text())
    valid = {
        "schema_version": "1.0.0",
        "record_id": "t1_example",
        "content_type": "comment",
        "primary_language_code": "en",
        "primary_confidence": 0.98,
        "primary_english_confidence": 0.98,
        "primary_top_predictions": [
            {"language_code": "en", "confidence": 0.98},
            {"language_code": "de", "confidence": 0.01},
        ],
        "provisional_status": "provisional_english",
        "exclusion_allowed": False,
        "detector_id": "fasttext-lid.176.bin",
        "language_policy_digest": "1" * 64,
        "stage_a_run_manifest_id": "2" * 64,
    }
    validate(valid, schema)
    unpaired = {
        **valid,
        "primary_top_codes": ["en", "de"],
        "primary_top_confidences": [0.98],
    }
    unpaired.pop("primary_top_predictions")
    with pytest.raises(ValidationError):
        validate(unpaired, schema)
    malformed_pair = {
        **valid,
        "primary_top_predictions": [{"language_code": "en"}],
    }
    with pytest.raises(ValidationError):
        validate(malformed_pair, schema)


@pytest.mark.parametrize(
    ("code", "confidence", "expected"),
    [
        ("en", 0.9, "provisional_english"),
        ("de", 0.9, "provisional_non_english"),
        ("en", 0.79, "uncertain"),
        ("und", 1.0, "uncertain"),
    ],
)
def test_provisional_status_never_implies_accepted_exclusion(
    code: str, confidence: float, expected: str
) -> None:
    assert provisional_status(code, confidence, high_confidence_threshold=0.8) == expected


def test_agreement_buckets_prioritise_disagreement() -> None:
    assert (
        agreement_bucket(
            primary_code="en",
            primary_confidence=0.99,
            challenger_code="de",
            challenger_confidence=0.99,
            high_confidence_threshold=0.8,
        )
        == "language_disagreement"
    )
    assert (
        agreement_bucket(
            primary_code="en",
            primary_confidence=0.99,
            challenger_code="en",
            challenger_confidence=0.7,
            high_confidence_threshold=0.8,
        )
        == "agreement_low_confidence"
    )


def test_cost_estimates_and_both_guardrails(policy: dict[str, object]) -> None:
    benchmark = estimate_benchmark_incremental_cost_usd(benchmark_rows=100_000, policy=policy)
    estimate = estimate_language_cost_usd(
        canonical_rows=548_963_310,
        candidate_rows=12_766_513,
        benchmark_rows=100_000,
        policy=policy,
    )
    assert benchmark <= Decimal("2")
    assert estimate <= Decimal("25")
    enforce_cost_guardrail(estimated_cost_usd=estimate, approved_usd=Decimal("25"), policy=policy)
    with pytest.raises(RuntimeError, match="exceeds approved"):
        enforce_cost_guardrail(
            estimated_cost_usd=estimate,
            approved_usd=Decimal("1"),
            policy=policy,
        )
    with pytest.raises(ValueError, match="<= 25"):
        enforce_cost_guardrail(
            estimated_cost_usd=estimate,
            approved_usd=Decimal("26"),
            policy=policy,
        )


def test_smoke_estimate_uses_one_exact_cell_and_two_dollar_guard(
    policy: dict[str, object],
) -> None:
    ref = {
        "subreddit": "ChineseLanguage",
        "year": 2025,
        "rows": 12_000,
        "canonical_inputs": {
            "submission": {"partition_rows": 2_000},
            "comment": {"partition_rows": 80_000},
        },
    }
    estimate = estimate_cell_cost_usd(
        ref=ref,
        sample_quotas={
            "ChineseLanguage|2025|submission|direct_lexical": 100,
            "ChineseLanguage|2025|comment|multi_channel": 500,
            "China|2025|comment|direct_lexical": 9_000,
        },
        policy=policy,
    )
    assert estimate < Decimal("2")
    enforce_smoke_cost_guardrail(estimated_cost_usd=estimate, approved_usd=Decimal("2"))
    with pytest.raises(RuntimeError, match="exceeds approved"):
        enforce_smoke_cost_guardrail(
            estimated_cost_usd=Decimal("0.50"), approved_usd=Decimal("0.10")
        )
    with pytest.raises(ValueError, match="<= 2"):
        enforce_smoke_cost_guardrail(estimated_cost_usd=estimate, approved_usd=Decimal("2.01"))


def test_language_preparation_never_calls_foreign_stage_a_modal_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RemoteStub:
        def __init__(self, result: dict[str, object], *, fail: bool = False) -> None:
            self.result = result
            self.fail = fail
            self.calls = 0

        def remote(self, **_: object) -> dict[str, object]:
            self.calls += 1
            if self.fail:
                raise AssertionError("foreign Stage A Modal function was called")
            return self.result

    foreign = RemoteStub({}, fail=True)
    owned_stage_resolver = RemoteStub({"owned": True})
    owned_language_resolver = RemoteStub({"language_contract": True})
    monkeypatch.setattr(stage_a_module, "resolve_production_contract", foreign)
    monkeypatch.setattr(
        language_module,
        "resolve_stage_a_contract_for_language",
        owned_stage_resolver,
    )
    monkeypatch.setattr(
        language_module,
        "resolve_language_contract",
        owned_language_resolver,
    )
    monkeypatch.setattr(
        language_module,
        "production_run_id",
        lambda _: language_module.FROZEN_STAGE_A_RUN_ID,
    )

    contract, policy = language_module._prepare_language_contract(
        manifest_path="configs/source-files.json",
        retrieval_policy_path="configs/retrieval-policy-v3.toml",
        language_policy_path="configs/language-policy-v1.toml",
    )

    assert contract == {"language_contract": True}
    assert len(policy["policy_digest"]) == 64
    assert foreign.calls == 0
    assert owned_stage_resolver.calls == 1
    assert owned_language_resolver.calls == 1
