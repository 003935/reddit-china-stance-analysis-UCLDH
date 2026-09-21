from __future__ import annotations

from pathlib import Path

import pytest

from reddit_china_stance.modal_stage_a_proof import (
    DEFAULT_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE,
    DEFAULT_MAX_INPUT_ROWS_PER_SOURCE,
    FORBIDDEN_OUTPUT_FIELDS,
    _canonical_sha256,
    _scan_columnar_rows,
    acquire_run_claim,
    candidate_output_field_names,
    enforce_anchor_scan_bounds,
    enforce_input_bounds,
    make_run_contract,
    pin_input_partitions,
    select_stage_a_candidates,
    update_run_claim,
)
from reddit_china_stance.retrieval import RetrievalChannel, load_retrieval_policy

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def policy():
    return load_retrieval_policy(ROOT / "configs" / "retrieval-policy-v1.toml")


def _dump(candidates):
    return [candidate.model_dump(mode="json") for candidate in candidates]


class FakeClaimRegistry:
    def __init__(self):
        self.rows = {}

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self.rows:
            return False
        self.rows[key] = value
        return True

    def get(self, key, default=None):
        return self.rows.get(key, default)


def test_atomic_claim_allows_exactly_one_owner() -> None:
    registry = FakeClaimRegistry()
    first = acquire_run_claim(
        registry,
        run_manifest_id="run-001",
        owner_call_id="call-a",
        claimed_at="2026-08-24T10:00:00Z",
    )
    assert first["same_owner_retry"] is False
    assert first["claim"]["attempt"] == 1

    with pytest.raises(RuntimeError, match="different Modal function call"):
        acquire_run_claim(
            registry,
            run_manifest_id="run-001",
            owner_call_id="call-b",
            claimed_at="2026-08-24T10:01:00Z",
        )
    assert registry.rows["run-001"]["owner_call_id"] == "call-a"


def test_same_call_retry_keeps_owner_and_increments_attempt() -> None:
    registry = FakeClaimRegistry()
    acquire_run_claim(
        registry,
        run_manifest_id="run-001",
        owner_call_id="call-a",
        claimed_at="2026-08-24T10:00:00Z",
    )
    update_run_claim(
        registry,
        run_manifest_id="run-001",
        owner_call_id="call-a",
        status="failed",
        updated_at="2026-08-24T10:01:00Z",
        metadata={"failed_at": "2026-08-24T10:01:00Z", "failure_type": "Preemption"},
    )
    with pytest.raises(RuntimeError, match="different Modal function call"):
        acquire_run_claim(
            registry,
            run_manifest_id="run-001",
            owner_call_id="call-b",
            claimed_at="2026-08-24T10:01:30Z",
        )

    retry = acquire_run_claim(
        registry,
        run_manifest_id="run-001",
        owner_call_id="call-a",
        claimed_at="2026-08-24T10:02:00Z",
    )
    assert retry["same_owner_retry"] is True
    assert retry["claim"]["attempt"] == 2
    assert retry["claim"]["status"] == "active"
    assert retry["claim"]["owner_call_id"] == "call-a"
    assert "failed_at" not in retry["claim"]


def test_complete_cell_bound_fails_instead_of_truncating() -> None:
    assert DEFAULT_MAX_INPUT_ROWS_PER_SOURCE == 3_000_000
    assert DEFAULT_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE == 25_000_000
    enforce_input_bounds(
        submission_rows=25,
        comment_rows=100,
        max_input_rows_per_source=100,
    )
    with pytest.raises(RuntimeError, match="refusing to truncate"):
        enforce_input_bounds(
            submission_rows=25,
            comment_rows=101,
            max_input_rows_per_source=100,
        )
    with pytest.raises(ValueError, match="between 1"):
        enforce_input_bounds(
            submission_rows=0,
            comment_rows=0,
            max_input_rows_per_source=0,
        )
    with pytest.raises(RuntimeError, match="all-year comment anchor scan"):
        enforce_anchor_scan_bounds(
            submission_rows=5,
            comment_rows=101,
            max_anchor_scan_rows_per_source=100,
        )


def test_expansion_preserves_all_paths_and_is_deterministic(policy) -> None:
    submissions = [
        {"record_id": "t3_s2", "text": "Nothing relevant"},
        {"record_id": "t3_s1", "text": "China policy"},
    ]
    comments = [
        {
            "record_id": "t1_c4",
            "submission_id": "t3_s2",
            "parent_id": "t3_s2",
            "text": "Beijing",
        },
        {
            "record_id": "t1_c2",
            "submission_id": "t3_s1",
            "parent_id": "t1_c1",
            "text": "Nothing relevant",
        },
        {
            "record_id": "t1_c1",
            "submission_id": "t3_s1",
            "parent_id": "t3_s1",
            "text": "CCP",
        },
        {
            "record_id": "t1_c3",
            "submission_id": "t3_s2",
            "parent_id": "t1_c1",
            "text": "Nothing relevant",
        },
    ]

    first = select_stage_a_candidates(
        submissions=submissions,
        comments=comments,
        policy=policy,
        manifest_id="proof-001",
        max_input_rows_per_source=10,
    )
    second = select_stage_a_candidates(
        submissions=reversed(submissions),
        comments=reversed(comments),
        policy=policy,
        manifest_id="proof-001",
        max_input_rows_per_source=10,
    )
    assert _dump(first) == _dump(second)
    by_id = {candidate.record_id: candidate for candidate in first}
    assert set(by_id) == {"t3_s1", "t1_c1", "t1_c2", "t1_c3", "t1_c4"}
    assert by_id["t1_c1"].retrieval_channels == [
        RetrievalChannel.DIRECT_LEXICAL,
        RetrievalChannel.SUBMISSION_EXPANSION,
    ]
    assert by_id["t1_c2"].retrieval_channels == [
        RetrievalChannel.SUBMISSION_EXPANSION,
        RetrievalChannel.DIRECT_REPLY_EXPANSION,
    ]
    assert [item.anchor_record_id for item in by_id["t1_c2"].selection_evidence] == [
        "t3_s1",
        "t1_c1",
    ]
    assert by_id["t1_c3"].retrieval_channels == [RetrievalChannel.DIRECT_REPLY_EXPANSION]


def test_overlapping_routes_deduplicate_to_one_metadata_only_row(policy) -> None:
    candidates = select_stage_a_candidates(
        submissions=[{"record_id": "t3_s1", "text": "China"}],
        comments=[
            {
                "record_id": "t1_c1",
                "submission_id": "t3_s1",
                "parent_id": "t3_s1",
                "text": "Chinese",
            }
        ],
        policy=policy,
        manifest_id="proof-001",
        max_input_rows_per_source=1,
    )
    assert [row.record_id for row in candidates].count("t1_c1") == 1
    row = next(row.model_dump(mode="json") for row in candidates if row.record_id == "t1_c1")
    assert not FORBIDDEN_OUTPUT_FIELDS.intersection(row)
    assert row["language_decision"] == {
        "status": "unclassified",
        "language_code": None,
        "model_run_manifest_id": None,
    }


def test_target_year_comment_uses_other_year_submission_and_parent_anchors(policy) -> None:
    candidates = select_stage_a_candidates(
        submissions=[{"record_id": "t3_2025", "text": "Nothing relevant"}],
        comments=[
            {
                "record_id": "t1_2025",
                "submission_id": "t3_2024",
                "parent_id": "t1_2024",
                "text": "Nothing relevant",
            }
        ],
        other_year_submissions=[{"record_id": "t3_2024", "text": "China"}],
        other_year_comments=[{"record_id": "t1_2024", "text": "CCP"}],
        policy=policy,
        manifest_id="proof-cross-year",
        max_input_rows_per_source=10,
        max_anchor_scan_rows_per_source=10,
    )

    assert [candidate.record_id for candidate in candidates] == ["t1_2025"]
    candidate = candidates[0]
    assert candidate.retrieval_channels == [
        RetrievalChannel.SUBMISSION_EXPANSION,
        RetrievalChannel.DIRECT_REPLY_EXPANSION,
    ]
    assert [evidence.anchor_record_id for evidence in candidate.selection_evidence] == [
        "t3_2024",
        "t1_2024",
    ]


def test_candidate_parquet_contract_excludes_source_text_and_identifiers() -> None:
    fields = set(candidate_output_field_names())
    assert not FORBIDDEN_OUTPUT_FIELDS.intersection(fields)
    assert "selection_evidence" in fields
    assert "retrieval_policy_digest" in fields
    assert "manifest_id" in fields


def test_columnar_scanner_emits_comment_identifiers_and_term_ids_without_text(policy) -> None:
    results = _scan_columnar_rows(
        data={
            "record_id": ["t1_comment"],
            "content_type": ["comment"],
            "subreddit": ["China"],
            "year": [2025],
            "text": ["China and the CCP"],
            "submission_id": ["t3_submission"],
            "parent_id": ["t1_parent"],
        },
        row_count=1,
        content_type="comment",
        subreddit="China",
        year=2025,
        policy=policy,
    )

    assert results == [("t1_comment", "t3_submission", "t1_parent", ("ccp", "china"))]
    assert all("China and the CCP" not in item for row in results for item in row)


def test_other_year_columnar_scanner_keeps_only_direct_anchor_metadata(policy) -> None:
    results = _scan_columnar_rows(
        data={
            "record_id": ["t1_direct", "t1_irrelevant"],
            "content_type": ["comment", "comment"],
            "subreddit": ["China", "China"],
            "year": [2024, 2024],
            "text": ["China", "Nothing relevant"],
        },
        row_count=2,
        content_type="comment",
        subreddit="China",
        year=2024,
        target_year=2025,
        policy=policy,
    )

    assert results == [("t1_direct", ("china",))]


def test_duplicate_input_id_is_rejected(policy) -> None:
    with pytest.raises(ValueError, match="duplicate input record_id"):
        select_stage_a_candidates(
            submissions=[
                {"record_id": "t3_duplicate", "text": "China"},
                {"record_id": "t3_duplicate", "text": "China"},
            ],
            comments=[],
            policy=policy,
            manifest_id="proof-001",
            max_input_rows_per_source=10,
        )


def test_run_contract_is_content_addressed_and_pins_cell(policy) -> None:
    sources = [
        {"path": "China_comments.zst", "size": 1, "sha256": "a" * 64},
        {"path": "China_submissions.zst", "size": 2, "sha256": "b" * 64},
    ]
    kwargs = {
        "dataset_id": "dataset/id",
        "revision": "c" * 40,
        "subreddit": "China",
        "year": 2025,
        "max_input_rows_per_source": 3_000_000,
        "max_anchor_scan_rows_per_source": 25_000_000,
        "sources": sources,
        "policy": policy,
        "code_state": {"code_sha256": "d" * 64},
    }
    base_contract = make_run_contract(**kwargs)

    def input_partitions(digest: str):
        return {
            content_type: [
                {
                    "year": year,
                    "content_type": content_type,
                    "partition_sha256": digest,
                }
                for year in range(2020, 2026)
            ]
            for content_type in ("submission", "comment")
        }

    contract = pin_input_partitions(base_contract, input_partitions("e" * 64))
    first = _canonical_sha256(contract)
    assert first == _canonical_sha256(
        pin_input_partitions(make_run_contract(**kwargs), input_partitions("e" * 64))
    )
    changed_cell = pin_input_partitions(
        make_run_contract(**{**kwargs, "year": 2024}), input_partitions("e" * 64)
    )
    changed_input = pin_input_partitions(base_contract, input_partitions("f" * 64))
    assert first != _canonical_sha256(changed_cell)
    assert first != _canonical_sha256(changed_input)
