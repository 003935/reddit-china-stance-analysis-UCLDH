from __future__ import annotations

from copy import deepcopy

import pytest

from reddit_china_stance.context_assembly import (
    TRUNCATION_MARKER,
    ContextLimits,
    assemble_context_record,
    deterministic_head_tail,
    sample_calibration_items,
)


def _canonical(
    record_id: str,
    *,
    content_type: str,
    submission_id: str,
    parent_id: str | None,
    text: str,
    year: int = 2023,
    stratum: str = "news_political",
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "content_type": content_type,
        "submission_id": submission_id,
        "parent_id": parent_id,
        "text": text,
        "text_sha256": "a" * 64,
        "subreddit": "worldnews",
        "stratum": stratum,
        "year": year,
        "month": 6,
    }


def _candidate(record_id: str, content_type: str, *, direct: bool = True) -> dict[str, object]:
    return {
        "record_id": record_id,
        "content_type": content_type,
        "retrieval_channels": ["direct_lexical" if direct else "direct_reply_expansion"],
        "manifest_id": "stage-a-manifest",
    }


def test_deterministic_head_tail_is_bounded_and_preserves_boundaries() -> None:
    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    result = deterministic_head_tail(source, max_chars=len(TRUNCATION_MARKER) + 8)

    assert result.truncated is True
    assert len(result.text) == len(TRUNCATION_MARKER) + 8
    assert result.text.startswith("ABCD")
    assert result.text.endswith("wxyz")
    assert result.original_chars == len(source)
    assert result.retained_source_chars == 8
    assert deterministic_head_tail(source, max_chars=len(TRUNCATION_MARKER) + 8) == result


def test_submission_target_is_not_duplicated_as_context() -> None:
    target = _canonical(
        "t3_submission",
        content_type="submission",
        submission_id="t3_submission",
        parent_id=None,
        text="Target stance belongs here only.",
    )
    assembled = assemble_context_record(
        candidate=_candidate("t3_submission", "submission"),
        target=target,
        submission=target,
        parent=None,
        language_run_manifest_id="language-run",
    )

    assert assembled["target_text"]["text"] == "Target stance belongs here only."
    assert assembled["submission_context"] is None
    assert assembled["submission_join_status"] == "same_as_target"
    assert assembled["parent_context"] is None
    assert assembled["parent_join_status"] == "not_applicable"
    assert assembled["context_role_rule"] == "reference_disambiguation_only"


def test_comment_context_fields_are_separate_and_truncation_is_explicit() -> None:
    target = _canonical(
        "t1_target",
        content_type="comment",
        submission_id="t3_submission",
        parent_id="t1_parent",
        text="Target text",
    )
    submission = _canonical(
        "t3_submission",
        content_type="submission",
        submission_id="t3_submission",
        parent_id=None,
        text="S" * 100,
    )
    parent = _canonical(
        "t1_parent",
        content_type="comment",
        submission_id="t3_submission",
        parent_id="t3_submission",
        text="Parent text",
    )
    limits = ContextLimits(
        submission_chars=len(TRUNCATION_MARKER) + 10,
        parent_chars=100,
        target_chars=100,
    )

    assembled = assemble_context_record(
        candidate=_candidate("t1_target", "comment", direct=False),
        target=target,
        submission=submission,
        parent=parent,
        language_run_manifest_id="language-run",
        limits=limits,
    )

    assert assembled["target_text"]["text"] == "Target text"
    assert assembled["submission_context"]["record_id"] == "t3_submission"
    assert assembled["submission_context"]["truncated"] is True
    assert assembled["parent_context"]["record_id"] == "t1_parent"
    assert assembled["parent_context"]["text"] == "Parent text"
    assert assembled["submission_join_status"] == "present"
    assert assembled["parent_join_status"] == "present"


def test_missing_and_cross_thread_context_are_not_silently_used() -> None:
    target = _canonical(
        "t1_target",
        content_type="comment",
        submission_id="t3_submission",
        parent_id="t1_parent",
        text="Target text",
    )
    wrong_parent = _canonical(
        "t1_parent",
        content_type="comment",
        submission_id="t3_other",
        parent_id="t3_other",
        text="This must never leak across threads.",
    )
    assembled = assemble_context_record(
        candidate=_candidate("t1_target", "comment"),
        target=target,
        submission=None,
        parent=wrong_parent,
        language_run_manifest_id="language-run",
    )

    assert assembled["submission_join_status"] == "missing"
    assert assembled["submission_context"] is None
    assert assembled["parent_join_status"] == "invalid_relation"
    assert assembled["parent_context"] is None


def _context_record(
    index: int,
    *,
    thread_id: str | None = None,
    direct: bool | None = None,
) -> dict[str, object]:
    is_submission = index % 2 == 0
    return {
        "record_id": f"t1_record{index}",
        "thread_id": thread_id or f"t3_thread{index}",
        "submission_id": thread_id or f"t3_thread{index}",
        "year": 2020 + index % 6,
        "stratum": ("domain", "news_political", "general_interest")[index % 3],
        "content_type": "submission" if is_submission else "comment",
        "retrieval_channels": [
            "direct_lexical"
            if (index % 2 == 0 if direct is None else direct)
            else "submission_expansion"
        ],
        "target_text": {"truncated": index % 13 == 0},
        "submission_context": None,
        "parent_context": None,
        "submission_join_status": "present",
        "parent_join_status": "present",
    }


def test_calibration_sample_is_exact_reproducible_and_thread_safe() -> None:
    records = [_context_record(index) for index in range(240)]
    # More than one candidate can belong to a thread; only one may be sampled.
    records.append(_context_record(1000, thread_id="t3_thread0", direct=False))

    first = sample_calibration_items(records, sample_size=150, seed=17)
    second = sample_calibration_items(deepcopy(records), sample_size=150, seed=17)

    assert first == second
    assert first["calibration_manifest_id"] == second["calibration_manifest_id"]
    assert first["contains_raw_text"] is False
    assert len(first["items"]) == 150
    threads = [row["thread_id"] for row in first["items"]]
    assert len(threads) == len(set(threads))
    assert all(row["double_coding_required"] is True for row in first["items"])
    assert all(row["coder_slots"] == ["coder_1", "coder_2"] for row in first["items"])
    assert all("/" in row["thread_inclusion_fraction"] for row in first["items"])
    assert all(
        set(row["stratum"]) == {"year", "stratum", "content_type", "retrieval_mode"}
        for row in first["items"]
    )


def test_calibration_sampler_honours_prior_thread_exclusions() -> None:
    records = [_context_record(index) for index in range(180)]
    excluded = {f"t3_thread{index}" for index in range(10)}
    sample = sample_calibration_items(
        records,
        sample_size=150,
        seed=123,
        excluded_thread_ids=excluded,
    )

    assert excluded.isdisjoint(row["thread_id"] for row in sample["items"])
    assert sample["population"]["excluded_threads"] == 10


def test_calibration_sampler_fails_when_unique_threads_are_insufficient() -> None:
    records = [_context_record(index, thread_id="t3_one") for index in range(3)]
    with pytest.raises(ValueError, match="eligible unique threads"):
        sample_calibration_items(records, sample_size=2)
