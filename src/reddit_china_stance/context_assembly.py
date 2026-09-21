"""Deterministic context assembly and thread-safe calibration sampling.

Raw text enters this module only as in-memory canonical rows.  The calibration
manifest deliberately contains IDs and design metadata only; text belongs in
access-controlled context artefacts on the Modal Volume.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CONTEXT_SCHEMA_VERSION = "1.0.0"
CALIBRATION_SCHEMA_VERSION = "1.0.0"
DEFAULT_CALIBRATION_SEED = 20250823
DEFAULT_CALIBRATION_SIZE = 150
TRUNCATION_MARKER = "\n...[TRUNCATED]...\n"

JoinStatus = Literal[
    "present",
    "missing",
    "not_applicable",
    "same_as_target",
    "same_as_submission",
    "invalid_relation",
]


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """Character limits for reproducible, tokenizer-independent assembly."""

    submission_chars: int = 1_536
    parent_chars: int = 768
    target_chars: int = 2_048

    def __post_init__(self) -> None:
        for field, value in (
            ("submission_chars", self.submission_chars),
            ("parent_chars", self.parent_chars),
            ("target_chars", self.target_chars),
        ):
            if type(value) is not int or value < len(TRUNCATION_MARKER) + 2:
                raise ValueError(f"{field} must be an integer >= {len(TRUNCATION_MARKER) + 2}")


@dataclass(frozen=True, slots=True)
class BoundedText:
    text: str
    original_chars: int
    retained_source_chars: int
    truncated: bool


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping using the repository's canonical form."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def deterministic_head_tail(text: str, *, max_chars: int) -> BoundedText:
    """Bound text deterministically while retaining both discourse boundaries."""

    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if type(max_chars) is not int or max_chars < len(TRUNCATION_MARKER) + 2:
        raise ValueError(f"max_chars must be an integer >= {len(TRUNCATION_MARKER) + 2}")
    if len(text) <= max_chars:
        return BoundedText(
            text=text,
            original_chars=len(text),
            retained_source_chars=len(text),
            truncated=False,
        )

    source_budget = max_chars - len(TRUNCATION_MARKER)
    head_chars = (source_budget + 1) // 2
    tail_chars = source_budget - head_chars
    bounded = f"{text[:head_chars]}{TRUNCATION_MARKER}{text[-tail_chars:]}"
    if len(bounded) != max_chars:
        raise RuntimeError("deterministic truncation did not conserve its character budget")
    return BoundedText(
        text=bounded,
        original_chars=len(text),
        retained_source_chars=source_budget,
        truncated=True,
    )


def _required_string(row: Mapping[str, Any], field: str, *, record_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string for {record_id}")
    return value


def _optional_string(row: Mapping[str, Any], field: str, *, record_id: str) -> str | None:
    value = row.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{field} must be null or a non-empty string for {record_id}")
    return value


def _bounded_payload(row: Mapping[str, Any], *, max_chars: int) -> dict[str, Any]:
    record_id = _required_string(row, "record_id", record_id="unknown")
    bounded = deterministic_head_tail(
        _required_string(row, "text", record_id=record_id), max_chars=max_chars
    )
    return {
        "record_id": record_id,
        "text": bounded.text,
        "text_sha256": _required_string(row, "text_sha256", record_id=record_id),
        "original_chars": bounded.original_chars,
        "retained_source_chars": bounded.retained_source_chars,
        "truncated": bounded.truncated,
    }


def _join_payload(
    row: Mapping[str, Any] | None,
    *,
    expected_id: str,
    expected_content_type: str,
    expected_submission_id: str,
    max_chars: int,
) -> tuple[JoinStatus, dict[str, Any] | None]:
    if row is None:
        return "missing", None
    record_id = _required_string(row, "record_id", record_id=expected_id)
    content_type = _required_string(row, "content_type", record_id=record_id)
    submission_id = _required_string(row, "submission_id", record_id=record_id)
    if (
        record_id != expected_id
        or content_type != expected_content_type
        or submission_id != expected_submission_id
    ):
        return "invalid_relation", None
    return "present", _bounded_payload(row, max_chars=max_chars)


def assemble_context_record(
    *,
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
    submission: Mapping[str, Any] | None,
    parent: Mapping[str, Any] | None,
    language_run_manifest_id: str,
    limits: ContextLimits | None = None,
) -> dict[str, Any]:
    """Assemble one target while keeping context roles separate and auditable."""

    limits = limits or ContextLimits()
    record_id = _required_string(candidate, "record_id", record_id="candidate")
    if _required_string(target, "record_id", record_id=record_id) != record_id:
        raise ValueError("candidate target join returned a different record_id")
    content_type = _required_string(target, "content_type", record_id=record_id)
    if content_type not in {"submission", "comment"}:
        raise ValueError(f"unsupported content_type for {record_id}: {content_type}")
    candidate_content_type = _required_string(candidate, "content_type", record_id=record_id)
    if candidate_content_type != content_type:
        raise ValueError("candidate and canonical content_type disagree")
    submission_id = _required_string(target, "submission_id", record_id=record_id)
    parent_id = _optional_string(target, "parent_id", record_id=record_id)
    target_payload = _bounded_payload(target, max_chars=limits.target_chars)

    submission_status: JoinStatus
    submission_payload: dict[str, Any] | None
    parent_status: JoinStatus
    parent_payload: dict[str, Any] | None
    if content_type == "submission":
        if submission_id != record_id or parent_id is not None:
            raise ValueError("submission target has inconsistent thread identifiers")
        submission_status, submission_payload = "same_as_target", None
        parent_status, parent_payload = "not_applicable", None
    else:
        if parent_id is None:
            raise ValueError("comment target requires parent_id")
        submission_status, submission_payload = _join_payload(
            submission,
            expected_id=submission_id,
            expected_content_type="submission",
            expected_submission_id=submission_id,
            max_chars=limits.submission_chars,
        )
        if parent_id.startswith("t3_"):
            if parent_id == submission_id:
                parent_status, parent_payload = "same_as_submission", None
            else:
                parent_status, parent_payload = "invalid_relation", None
        elif parent_id == record_id:
            parent_status, parent_payload = "invalid_relation", None
        elif parent_id.startswith("t1_"):
            parent_status, parent_payload = _join_payload(
                parent,
                expected_id=parent_id,
                expected_content_type="comment",
                expected_submission_id=submission_id,
                max_chars=limits.parent_chars,
            )
        else:
            parent_status, parent_payload = "invalid_relation", None

    retrieval_channels = candidate.get("retrieval_channels")
    if not isinstance(retrieval_channels, Sequence) or isinstance(retrieval_channels, (str, bytes)):
        raise ValueError(f"retrieval_channels must be a sequence for {record_id}")
    channels = sorted({_required_channel(value, record_id) for value in retrieval_channels})
    if not channels:
        raise ValueError(f"retrieval_channels cannot be empty for {record_id}")
    manifest_id = _required_string(candidate, "manifest_id", record_id=record_id)
    if not language_run_manifest_id:
        raise ValueError("language_run_manifest_id must be non-empty")

    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "record_id": record_id,
        "thread_id": submission_id,
        "submission_id": submission_id,
        "parent_id": parent_id,
        "content_type": content_type,
        "subreddit": _required_string(target, "subreddit", record_id=record_id),
        "stratum": _required_string(target, "stratum", record_id=record_id),
        "year": _required_year(target.get("year"), record_id),
        "month": _required_month(target.get("month"), record_id),
        "retrieval_channels": channels,
        "stage_a_run_manifest_id": manifest_id,
        "language_run_manifest_id": language_run_manifest_id,
        "target_join_status": "present",
        "target_text": target_payload,
        "submission_join_status": submission_status,
        "submission_context": submission_payload,
        "parent_join_status": parent_status,
        "parent_context": parent_payload,
        "context_role_rule": "reference_disambiguation_only",
    }


def _required_channel(value: Any, record_id: str) -> str:
    if value not in {
        "direct_lexical",
        "submission_expansion",
        "direct_reply_expansion",
    }:
        raise ValueError(f"invalid retrieval channel for {record_id}: {value!r}")
    return str(value)


def _required_year(value: Any, record_id: str) -> int:
    if type(value) is not int or not 2020 <= value <= 2025:
        raise ValueError(f"year must be 2020..2025 for {record_id}")
    return value


def _required_month(value: Any, record_id: str) -> int:
    if type(value) is not int or not 1 <= value <= 12:
        raise ValueError(f"month must be 1..12 for {record_id}")
    return value


def _retrieval_mode(record: Mapping[str, Any]) -> str:
    channels = record.get("retrieval_channels")
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
        raise ValueError("context record retrieval_channels must be a sequence")
    observed = {_required_channel(value, str(record.get("record_id"))) for value in channels}
    return "direct" if "direct_lexical" in observed else "expanded_only"


def _context_condition(record: Mapping[str, Any]) -> str:
    statuses = {
        record.get("submission_join_status"),
        record.get("parent_join_status"),
    }
    missing = bool(statuses.intersection({"missing", "invalid_relation"}))
    truncated = any(
        isinstance(record.get(field), Mapping) and record[field].get("truncated") is True
        for field in ("target_text", "submission_context", "parent_context")
    )
    if missing and truncated:
        return "missing_and_truncated"
    if missing:
        return "missing"
    if truncated:
        return "truncated"
    return "ordinary"


def calibration_stratum(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the preregistered first-round calibration dimensions."""

    record_id = _required_string(record, "record_id", record_id="context")
    content_type = _required_string(record, "content_type", record_id=record_id)
    if content_type not in {"submission", "comment"}:
        raise ValueError(f"invalid content_type for {record_id}")
    return {
        "year": _required_year(record.get("year"), record_id),
        "stratum": _required_string(record, "stratum", record_id=record_id),
        "content_type": content_type,
        "retrieval_mode": _retrieval_mode(record),
    }


def _stratum_id(stratum: Mapping[str, Any]) -> str:
    return "|".join(f"{key}={stratum[key]}" for key in sorted(stratum))


def _selection_key(*, seed: int, namespace: str, value: str) -> str:
    payload = f"{seed}\0{namespace}\0{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def _allocate_stratified_sample(
    sizes: Mapping[str, int], *, sample_size: int, seed: int
) -> dict[str, int]:
    if not sizes or any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("stratum sizes must be non-empty positive integers")
    population = sum(sizes.values())
    if not 1 <= sample_size <= population:
        raise ValueError("sample_size must not exceed eligible unique threads")

    allocation = {key: 0 for key in sizes}
    ordered = sorted(
        sizes,
        key=lambda key: _selection_key(seed=seed, namespace="stratum-coverage", value=key),
    )
    for key in ordered[: min(sample_size, len(ordered))]:
        allocation[key] = 1

    remaining = sample_size - sum(allocation.values())
    while remaining:
        eligible = [key for key, size in sizes.items() if allocation[key] < size]
        if not eligible:
            raise RuntimeError("sample allocation exhausted eligible threads")
        # Deficit against proportional allocation; keyed hash breaks exact ties reproducibly.
        key = max(
            eligible,
            key=lambda item: (
                sample_size * sizes[item] / population - allocation[item],
                _selection_key(seed=seed, namespace="allocation-tie", value=item),
            ),
        )
        allocation[key] += 1
        remaining -= 1
    return {key: value for key, value in allocation.items() if value}


def sample_calibration_items(
    records: Iterable[Mapping[str, Any]],
    *,
    sample_size: int = DEFAULT_CALIBRATION_SIZE,
    seed: int = DEFAULT_CALIBRATION_SEED,
    excluded_thread_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Select one item per thread for the independently double-coded first round."""

    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    excluded = set(excluded_thread_ids)
    if any(not isinstance(value, str) or not value for value in excluded):
        raise ValueError("excluded_thread_ids must contain non-empty strings")

    by_thread: dict[str, dict[str, Any]] = {}
    seen_records: set[str] = set()
    eligible_records = 0
    for source in records:
        record = dict(source)
        record_id = _required_string(record, "record_id", record_id="context")
        if record_id in seen_records:
            raise ValueError(f"duplicate context record_id: {record_id}")
        seen_records.add(record_id)
        thread_id = _required_string(record, "thread_id", record_id=record_id)
        if thread_id != _required_string(record, "submission_id", record_id=record_id):
            raise ValueError(f"thread_id must equal submission_id for {record_id}")
        if thread_id in excluded:
            continue
        eligible_records += 1
        previous = by_thread.get(thread_id)
        if previous is None or _selection_key(
            seed=seed,
            namespace=f"thread-representative:{thread_id}",
            value=record_id,
        ) < _selection_key(
            seed=seed,
            namespace=f"thread-representative:{thread_id}",
            value=str(previous["record_id"]),
        ):
            by_thread[thread_id] = record
    if len(by_thread) < sample_size:
        raise ValueError(f"need {sample_size} eligible unique threads, found {len(by_thread)}")

    representatives: list[dict[str, Any]] = []
    for thread_id, representative in by_thread.items():
        stratum = calibration_stratum(representative)
        representatives.append(
            {
                "record": representative,
                "thread_id": thread_id,
                "stratum": stratum,
                "stratum_id": _stratum_id(stratum),
            }
        )

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for representative in representatives:
        grouped[representative["stratum_id"]].append(representative)
    sizes = {key: len(value) for key, value in grouped.items()}
    allocation = _allocate_stratified_sample(sizes, sample_size=sample_size, seed=seed)

    selected: list[dict[str, Any]] = []
    for stratum_id, selected_threads in allocation.items():
        ordered = sorted(
            grouped[stratum_id],
            key=lambda item: _selection_key(
                seed=seed,
                namespace=f"within-stratum:{stratum_id}",
                value=item["thread_id"],
            ),
        )
        for rank, item in enumerate(ordered[:selected_threads], start=1):
            record = item["record"]
            population_threads = sizes[stratum_id]
            selected.append(
                {
                    "schema_version": CALIBRATION_SCHEMA_VERSION,
                    "record_id": record["record_id"],
                    "thread_id": item["thread_id"],
                    "sampling_unit": "submission_thread",
                    "stratum": item["stratum"],
                    "stratum_id": stratum_id,
                    "context_condition": _context_condition(record),
                    "population_threads_in_stratum": population_threads,
                    "selected_threads_in_stratum": selected_threads,
                    "thread_inclusion_fraction": f"{selected_threads}/{population_threads}",
                    "selection_rank_within_stratum": rank,
                    "double_coding_required": True,
                    "coder_slots": ["coder_1", "coder_2"],
                }
            )
    selected.sort(key=lambda row: (row["stratum_id"], row["selection_rank_within_stratum"]))
    if len(selected) != sample_size:
        raise RuntimeError("calibration sampler did not produce the requested size")
    selected_threads = [row["thread_id"] for row in selected]
    if len(selected_threads) != len(set(selected_threads)):
        raise RuntimeError("submission thread leaked into multiple calibration items")

    identity = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "design": {
            "round": "calibration",
            "sampling_unit": "submission_thread",
            "representative_per_thread": 1,
            "allocation": "one-per-observed-stratum-then-proportional-deficit",
            "stratum_dimensions": [
                "year",
                "stratum",
                "content_type",
                "retrieval_mode",
            ],
            "double_coded": True,
            "seed": seed,
            "requested_items": sample_size,
        },
        "population": {
            "eligible_records": eligible_records,
            "eligible_threads": len(by_thread),
            "excluded_threads": len(excluded),
            "stratum_thread_counts": dict(sorted(sizes.items())),
        },
        "selected": [
            {
                "record_id": row["record_id"],
                "thread_id": row["thread_id"],
                "stratum_id": row["stratum_id"],
                "selection_rank_within_stratum": row["selection_rank_within_stratum"],
            }
            for row in selected
        ],
    }
    manifest_id = canonical_sha256(identity)
    for row in selected:
        row["calibration_manifest_id"] = manifest_id
    return {
        **identity,
        "calibration_manifest_id": manifest_id,
        "items": selected,
        "contains_raw_text": False,
    }
