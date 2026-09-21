"""Bounded, immutable Modal proof of the frozen Stage A retrieval policy."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import modal

from reddit_china_stance.modal_parquet import SOURCE_SCHEMA_VERSION
from reddit_china_stance.retrieval import (
    AnchorRelation,
    CandidateRecord,
    RetrievalChannel,
    RetrievalPolicy,
    SelectionEvidence,
    make_candidate,
    match_direct_lexical,
    merge_candidates,
)

APP_NAME = "reddit-china-stance-stage-a-proof"
VOLUME_NAME = "reddit-china-stance-data"
CLAIM_REGISTRY_NAME = "reddit-china-stance-stage-a-proof-claims"
ENVIRONMENT_NAME = "main"
VOLUME_PATH = Path("/data")
PROOF_PREFIX = Path("proofs") / "stage-a"
PROOF_SCHEMA_VERSION = "1.0.0"
CONFIRMATION = "RUN_ONE_STAGE_A_PROOF_CELL"
DEFAULT_MAX_INPUT_ROWS_PER_SOURCE = 3_000_000
HARD_MAX_INPUT_ROWS_PER_SOURCE = 5_000_000
DEFAULT_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE = 25_000_000
HARD_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE = 50_000_000
MIN_YEAR = 2020
MAX_YEAR = 2025
ANCHOR_YEARS = tuple(range(MIN_YEAR, MAX_YEAR + 1))
WORKER_COUNT = 8

CANDIDATE_PARQUET_FIELD_NAMES = (
    "schema_version",
    "record_id",
    "content_type",
    "retrieval_channels",
    "selection_evidence",
    "retrieval_policy_version",
    "retrieval_policy_digest",
    "manifest_id",
    "language_decision",
)
FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "text",
        "raw_text",
        "body",
        "title",
        "selftext",
        "author",
        "source_id",
        "text_sha256",
    }
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
claim_registry = modal.Dict.from_name(
    CLAIM_REGISTRY_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "pyarrow==25.0.1",
        "pydantic>=2.11.0,<3",
        "pyahocorasick>=2.2,<3",
    )
    .add_local_python_source("reddit_china_stance")
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_run_claim(
    registry: Any,
    *,
    run_manifest_id: str,
    owner_call_id: str,
    claimed_at: str,
) -> dict[str, Any]:
    """Atomically establish one logical owner, allowing only same-call restarts."""

    if not run_manifest_id or not owner_call_id or not claimed_at:
        raise ValueError("claim identifiers and timestamp must be non-empty")
    claim = {
        "schema_version": "1.0.0",
        "run_manifest_id": run_manifest_id,
        "owner_call_id": owner_call_id,
        "status": "active",
        "attempt": 1,
        "claimed_at": claimed_at,
        "last_started_at": claimed_at,
    }
    if registry.put(run_manifest_id, claim, skip_if_exists=True):
        return {"claim": claim, "same_owner_retry": False}

    existing = registry.get(run_manifest_id)
    if not isinstance(existing, dict) or existing.get("run_manifest_id") != run_manifest_id:
        raise RuntimeError("existing run claim is malformed")
    existing_owner = existing.get("owner_call_id")
    if existing_owner != owner_call_id:
        raise RuntimeError(
            f"run is already claimed by a different Modal function call: {existing_owner}"
        )
    if existing.get("status") == "complete":
        raise RuntimeError("claim is complete but immutable final output was not found")
    if existing.get("status") not in {"active", "failed"}:
        raise RuntimeError(
            f"existing same-owner claim has invalid status: {existing.get('status')}"
        )
    retry = {
        **existing,
        "status": "active",
        "attempt": int(existing.get("attempt", 0)) + 1,
        "last_started_at": claimed_at,
        "updated_at": claimed_at,
    }
    retry.pop("failed_at", None)
    retry.pop("failure_type", None)
    registry.put(run_manifest_id, retry)
    return {"claim": retry, "same_owner_retry": True}


def update_run_claim(
    registry: Any,
    *,
    run_manifest_id: str,
    owner_call_id: str,
    status: Literal["active", "complete", "failed"],
    updated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deliberately transition claim metadata without changing logical ownership."""

    existing = registry.get(run_manifest_id)
    if not isinstance(existing, dict) or existing.get("owner_call_id") != owner_call_id:
        raise RuntimeError("cannot update a run claim owned by another function call")
    if existing.get("run_manifest_id") != run_manifest_id:
        raise RuntimeError("cannot update a malformed run claim")
    updated = {**existing, "status": status, "updated_at": updated_at}
    if metadata:
        overlap = set(metadata).intersection(
            {"schema_version", "run_manifest_id", "owner_call_id", "status", "attempt"}
        )
        if overlap:
            raise ValueError(f"claim metadata cannot replace identity fields: {sorted(overlap)}")
        updated.update(metadata)
    registry.put(run_manifest_id, updated)
    return updated


def enforce_input_bounds(
    *, submission_rows: int, comment_rows: int, max_input_rows_per_source: int
) -> None:
    """Fail before retrieval if either complete source cell exceeds the explicit bound."""

    if not 1 <= max_input_rows_per_source <= HARD_MAX_INPUT_ROWS_PER_SOURCE:
        raise ValueError(
            f"max_input_rows_per_source must be between 1 and {HARD_MAX_INPUT_ROWS_PER_SOURCE}"
        )
    counts = {"submission": submission_rows, "comment": comment_rows}
    for content_type, count in counts.items():
        if type(count) is not int or count < 0:
            raise ValueError(f"{content_type} row count must be a non-negative integer")
        if count > max_input_rows_per_source:
            raise RuntimeError(
                f"complete {content_type} cell has {count} rows, exceeding explicit "
                f"max_input_rows_per_source={max_input_rows_per_source}; refusing to truncate"
            )


def enforce_anchor_scan_bounds(
    *, submission_rows: int, comment_rows: int, max_anchor_scan_rows_per_source: int
) -> None:
    """Fail if either complete all-year anchor scan exceeds its explicit bound."""

    if not 1 <= max_anchor_scan_rows_per_source <= HARD_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE:
        raise ValueError(
            "max_anchor_scan_rows_per_source must be between 1 and "
            f"{HARD_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE}"
        )
    counts = {"submission": submission_rows, "comment": comment_rows}
    for content_type, count in counts.items():
        if type(count) is not int or count < 0:
            raise ValueError(f"{content_type} anchor row count must be a non-negative integer")
        if count > max_anchor_scan_rows_per_source:
            raise RuntimeError(
                f"complete all-year {content_type} anchor scan has {count} rows, exceeding "
                f"explicit max_anchor_scan_rows_per_source={max_anchor_scan_rows_per_source}; "
                "refusing to truncate"
            )


def candidate_output_field_names() -> tuple[str, ...]:
    """Return the fixed metadata-only Parquet contract without importing PyArrow."""

    if FORBIDDEN_OUTPUT_FIELDS.intersection(CANDIDATE_PARQUET_FIELD_NAMES):
        raise RuntimeError("candidate output contract contains a forbidden source field")
    return CANDIDATE_PARQUET_FIELD_NAMES


def _required_row_value(row: Mapping[str, Any], field: str, *, record_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string for record {record_id}")
    return value


def _optional_row_value(row: Mapping[str, Any], field: str, *, record_id: str) -> str | None:
    value = row.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{field} must be null or a non-empty string for record {record_id}")
    return value


def _assemble_candidates(
    *,
    direct_submissions: Mapping[str, tuple[str, ...]],
    target_direct_submission_ids: frozenset[str],
    direct_comments: Mapping[str, tuple[str, ...]],
    comments: Sequence[tuple[str, str, str | None, tuple[str, ...]]],
    policy: RetrievalPolicy,
    manifest_id: str,
) -> list[CandidateRecord]:
    candidates: defaultdict[str, list[CandidateRecord]] = defaultdict(list)

    for record_id in sorted(target_direct_submission_ids):
        term_ids = direct_submissions[record_id]
        candidates[record_id].append(
            make_candidate(
                record_id=record_id,
                content_type="submission",
                evidence=SelectionEvidence(
                    channel=RetrievalChannel.DIRECT_LEXICAL,
                    anchor_record_id=record_id,
                    anchor_relation=AnchorRelation.SELF,
                    expansion_depth=0,
                    match_term_ids=list(term_ids),
                ),
                policy=policy,
                manifest_id=manifest_id,
            )
        )

    for record_id, submission_id, parent_id, term_ids in sorted(comments):
        if term_ids:
            candidates[record_id].append(
                make_candidate(
                    record_id=record_id,
                    content_type="comment",
                    evidence=SelectionEvidence(
                        channel=RetrievalChannel.DIRECT_LEXICAL,
                        anchor_record_id=record_id,
                        anchor_relation=AnchorRelation.SELF,
                        expansion_depth=0,
                        match_term_ids=list(term_ids),
                    ),
                    policy=policy,
                    manifest_id=manifest_id,
                )
            )
        submission_terms = direct_submissions.get(submission_id)
        if submission_terms:
            candidates[record_id].append(
                make_candidate(
                    record_id=record_id,
                    content_type="comment",
                    evidence=SelectionEvidence(
                        channel=RetrievalChannel.SUBMISSION_EXPANSION,
                        anchor_record_id=submission_id,
                        anchor_relation=AnchorRelation.SUBMISSION,
                        expansion_depth=1,
                        match_term_ids=list(submission_terms),
                    ),
                    policy=policy,
                    manifest_id=manifest_id,
                )
            )
        parent_terms = direct_comments.get(parent_id) if parent_id is not None else None
        if parent_terms:
            candidates[record_id].append(
                make_candidate(
                    record_id=record_id,
                    content_type="comment",
                    evidence=SelectionEvidence(
                        channel=RetrievalChannel.DIRECT_REPLY_EXPANSION,
                        anchor_record_id=parent_id,
                        anchor_relation=AnchorRelation.DIRECT_PARENT,
                        expansion_depth=1,
                        match_term_ids=list(parent_terms),
                    ),
                    policy=policy,
                    manifest_id=manifest_id,
                )
            )

    return [merge_candidates(candidates[record_id]) for record_id in sorted(candidates)]


def select_stage_a_candidates(
    *,
    submissions: Iterable[Mapping[str, Any]],
    comments: Iterable[Mapping[str, Any]],
    other_year_submissions: Iterable[Mapping[str, Any]] = (),
    other_year_comments: Iterable[Mapping[str, Any]] = (),
    policy: RetrievalPolicy,
    manifest_id: str,
    max_input_rows_per_source: int,
    max_anchor_scan_rows_per_source: int = DEFAULT_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE,
) -> list[CandidateRecord]:
    """Emit one target cell using direct anchors scanned across every included year."""

    submission_rows = list(submissions)
    comment_rows = list(comments)
    other_submission_rows = list(other_year_submissions)
    other_comment_rows = list(other_year_comments)
    enforce_input_bounds(
        submission_rows=len(submission_rows),
        comment_rows=len(comment_rows),
        max_input_rows_per_source=max_input_rows_per_source,
    )
    enforce_anchor_scan_bounds(
        submission_rows=len(submission_rows) + len(other_submission_rows),
        comment_rows=len(comment_rows) + len(other_comment_rows),
        max_anchor_scan_rows_per_source=max_anchor_scan_rows_per_source,
    )

    seen: set[str] = set()
    direct_submissions: dict[str, tuple[str, ...]] = {}
    target_direct_submission_ids: set[str] = set()
    for is_target, rows in ((True, submission_rows), (False, other_submission_rows)):
        for row in sorted(rows, key=lambda item: str(item.get("record_id", ""))):
            record_id = _required_row_value(row, "record_id", record_id="<unknown>")
            if record_id in seen:
                raise ValueError(f"duplicate input record_id: {record_id}")
            seen.add(record_id)
            text = _required_row_value(row, "text", record_id=record_id)
            term_ids = match_direct_lexical(text, policy)
            if term_ids:
                direct_submissions[record_id] = term_ids
                if is_target:
                    target_direct_submission_ids.add(record_id)

    direct_comments: dict[str, tuple[str, ...]] = {}
    for row in sorted(other_comment_rows, key=lambda item: str(item.get("record_id", ""))):
        record_id = _required_row_value(row, "record_id", record_id="<unknown>")
        if record_id in seen:
            raise ValueError(f"duplicate input record_id: {record_id}")
        seen.add(record_id)
        text = _required_row_value(row, "text", record_id=record_id)
        term_ids = match_direct_lexical(text, policy)
        if term_ids:
            direct_comments[record_id] = term_ids

    observed_comments: list[tuple[str, str, str | None, tuple[str, ...]]] = []
    for row in sorted(comment_rows, key=lambda item: str(item.get("record_id", ""))):
        record_id = _required_row_value(row, "record_id", record_id="<unknown>")
        if record_id in seen:
            raise ValueError(f"duplicate input record_id: {record_id}")
        seen.add(record_id)
        submission_id = _required_row_value(row, "submission_id", record_id=record_id)
        parent_id = _optional_row_value(row, "parent_id", record_id=record_id)
        text = _required_row_value(row, "text", record_id=record_id)
        term_ids = match_direct_lexical(text, policy)
        observed_comments.append((record_id, submission_id, parent_id, term_ids))
        if term_ids:
            direct_comments[record_id] = term_ids

    return _assemble_candidates(
        direct_submissions=direct_submissions,
        target_direct_submission_ids=frozenset(target_direct_submission_ids),
        direct_comments=direct_comments,
        comments=observed_comments,
        policy=policy,
        manifest_id=manifest_id,
    )


def make_run_contract(
    *,
    dataset_id: str,
    revision: str,
    subreddit: str,
    year: int,
    max_input_rows_per_source: int,
    max_anchor_scan_rows_per_source: int,
    sources: Sequence[Mapping[str, Any]],
    policy: RetrievalPolicy,
    code_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the content-addressed invocation contract for exactly one cell."""

    if not subreddit or "/" in subreddit or ".." in subreddit:
        raise ValueError("subreddit must be one exact, path-safe name")
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError(f"year must be between {MIN_YEAR} and {MAX_YEAR}")
    enforce_input_bounds(
        submission_rows=0,
        comment_rows=0,
        max_input_rows_per_source=max_input_rows_per_source,
    )
    enforce_anchor_scan_bounds(
        submission_rows=0,
        comment_rows=0,
        max_anchor_scan_rows_per_source=max_anchor_scan_rows_per_source,
    )
    expected_paths = {f"{subreddit}_submissions.zst", f"{subreddit}_comments.zst"}
    if {str(source.get("path")) for source in sources} != expected_paths or len(sources) != 2:
        raise ValueError(f"sources must be exactly {sorted(expected_paths)}")
    return {
        "schema_version": PROOF_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "revision": revision,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "cell": {"subreddit": subreddit, "year": year},
        "anchor_years": list(ANCHOR_YEARS),
        "max_input_rows_per_source": max_input_rows_per_source,
        "max_anchor_scan_rows_per_source": max_anchor_scan_rows_per_source,
        "retrieval_policy": {
            "schema_version": policy.schema_version,
            "policy_version": policy.policy_version,
            "policy_digest": policy.policy_digest,
            "lexicon_version": policy.lexicon_version,
        },
        "sources": sorted((dict(source) for source in sources), key=lambda row: row["path"]),
        "code_state": dict(code_state),
    }


def _partition_input(
    *,
    contract: Mapping[str, Any],
    source: Mapping[str, Any],
    content_type: str,
    year: int,
) -> tuple[Path, dict[str, Any]]:
    revision = str(contract["revision"])
    subreddit = str(contract["cell"]["subreddit"])
    source_file = str(source["path"])
    source_dir = (
        VOLUME_PATH
        / "normalised"
        / revision
        / f"schema={SOURCE_SCHEMA_VERSION}"
        / Path(source_file).stem
    )
    receipt_path = source_dir / "_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"canonical source receipt does not exist: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt = {
        "dataset_id": contract["dataset_id"],
        "revision": revision,
        "source_file": source_file,
        "source_compressed_bytes": source["size"],
        "source_sha256": source["sha256"],
        "source_schema_version": SOURCE_SCHEMA_VERSION,
    }
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"canonical receipt mismatch for {source_file}: {key}")
    if receipt.get("status") != "converted":
        raise RuntimeError(f"canonical receipt is not final for {source_file}")

    expected_relative = (
        Path(f"year={year}")
        / f"subreddit={subreddit}"
        / f"content_type={content_type}"
        / "part-00000.parquet"
    )
    partitions = [row for row in receipt["partitions"] if int(row["year"]) == year]
    if len(partitions) != 1 or partitions[0]["relative_path"] != str(expected_relative):
        raise RuntimeError(f"canonical receipt lacks exact {subreddit}/{year}/{content_type} cell")
    partition = partitions[0]
    path = source_dir / expected_relative
    if not path.exists() or path.stat().st_size != int(partition["bytes"]):
        raise RuntimeError(f"canonical partition size mismatch: {path}")
    if _sha256_file(path) != partition["sha256"]:
        raise RuntimeError(f"canonical partition digest mismatch: {path}")
    return path, {
        "year": year,
        "content_type": content_type,
        "source_file": source_file,
        "source_sha256": source["sha256"],
        "source_compressed_bytes": source["size"],
        "conversion_code_sha256": receipt["code_sha256"],
        "partition_relative_path": str(path.relative_to(VOLUME_PATH)),
        "partition_rows": int(partition["rows"]),
        "partition_bytes": int(partition["bytes"]),
        "partition_sha256": partition["sha256"],
    }


def pin_input_partitions(
    contract: Mapping[str, Any], input_partitions: Mapping[str, Any]
) -> dict[str, Any]:
    """Add exact all-year canonical partition receipts to the immutable run contract."""

    if "input_partitions" in contract:
        raise ValueError("run contract already contains pinned input partitions")
    expected_types = {"submission", "comment"}
    if set(input_partitions) != expected_types:
        raise ValueError(f"input partitions must contain exactly {sorted(expected_types)}")
    expected_years = list(contract["anchor_years"])
    pinned: dict[str, list[dict[str, Any]]] = {}
    for content_type in sorted(expected_types):
        rows = [dict(row) for row in input_partitions[content_type]]
        if [row.get("year") for row in rows] != expected_years:
            raise ValueError(
                f"{content_type} input partitions must pin anchor years {expected_years}"
            )
        if any(row.get("content_type") != content_type for row in rows):
            raise ValueError(f"{content_type} input partition contract has wrong content type")
        pinned[content_type] = rows
    return {**dict(contract), "input_partitions": pinned}


def _resolve_input_partitions(contract: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    subreddit = str(contract["cell"]["subreddit"])
    sources = {str(source["path"]): source for source in contract["sources"]}
    resolved: dict[str, list[dict[str, Any]]] = {"submission": [], "comment": []}
    for content_type in ("submission", "comment"):
        source = sources[f"{subreddit}_{content_type}s.zst"]
        for year in contract["anchor_years"]:
            _, metadata = _partition_input(
                contract=contract,
                source=source,
                content_type=content_type,
                year=int(year),
            )
            resolved[content_type].append(metadata)
    return resolved


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=1.0,
    memory=1024,
    timeout=3_600,
    max_containers=4,
)
def resolve_stage_a_inputs(*, contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Resolve and hash every 2020-2025 partition before assigning the run ID."""

    if "input_partitions" in contract:
        raise ValueError("input-resolution contract must not already contain input partitions")
    if contract.get("anchor_years") != list(ANCHOR_YEARS):
        raise ValueError("input-resolution contract must pin all configured anchor years")
    volume.reload()
    return _resolve_input_partitions(contract)


def _row_group_chunks(row_group_count: int, target_chunks: int) -> list[tuple[int, ...]]:
    if row_group_count <= 0:
        return []
    chunk_count = min(row_group_count, target_chunks)
    buckets: list[list[int]] = [[] for _ in range(chunk_count)]
    for index in range(row_group_count):
        buckets[index % chunk_count].append(index)
    return [tuple(bucket) for bucket in buckets]


def _scan_row_groups(
    *,
    path: str,
    content_type: Literal["submission", "comment"],
    row_groups: tuple[int, ...],
    subreddit: str,
    year: int,
    target_year: int,
    policy: RetrievalPolicy,
) -> dict[str, Any]:
    """Worker scan that returns identifiers and policy term IDs, never source text."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ["record_id", "content_type", "subreddit", "year", "text"]
    is_target = year == target_year
    if content_type == "comment" and is_target:
        columns.extend(["submission_id", "parent_id"])
    table = parquet.read_row_groups(list(row_groups), columns=columns, use_threads=True)
    data = table.to_pydict()
    results = _scan_columnar_rows(
        data=data,
        row_count=table.num_rows,
        content_type=content_type,
        subreddit=subreddit,
        year=year,
        policy=policy,
        target_year=target_year,
    )
    return {
        "content_type": content_type,
        "year": year,
        "is_target": is_target,
        "rows": table.num_rows,
        "results": results,
    }


def _scan_columnar_rows(
    *,
    data: Mapping[str, Sequence[Any]],
    row_count: int,
    content_type: Literal["submission", "comment"],
    subreddit: str,
    year: int,
    policy: RetrievalPolicy,
    target_year: int | None = None,
) -> list[tuple[Any, ...]]:
    """Validate one columnar batch and discard text after deriving term IDs."""

    results: list[tuple[Any, ...]] = []
    is_target = target_year is None or year == target_year
    for index in range(row_count):
        record_id = data["record_id"][index]
        if not isinstance(record_id, str) or not record_id:
            raise RuntimeError("canonical row has invalid record_id")
        if (
            data["content_type"][index] != content_type
            or data["subreddit"][index] != subreddit
            or data["year"][index] != year
        ):
            raise RuntimeError(f"canonical row escaped selected cell: {record_id}")
        text = data["text"][index]
        if not isinstance(text, str):
            raise RuntimeError(f"canonical row has non-string text: {record_id}")
        term_ids = match_direct_lexical(text, policy)
        if content_type == "submission":
            if term_ids:
                results.append((record_id, term_ids))
        elif is_target:
            submission_id = data["submission_id"][index]
            parent_id = data["parent_id"][index]
            if not isinstance(submission_id, str) or not submission_id:
                raise RuntimeError(f"canonical comment has invalid submission_id: {record_id}")
            if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
                raise RuntimeError(f"canonical comment has invalid parent_id: {record_id}")
            results.append((record_id, submission_id, parent_id, term_ids))
        elif term_ids:
            results.append((record_id, term_ids))
    return results


def _candidate_arrow_schema() -> Any:
    import pyarrow as pa

    evidence = pa.struct(
        [
            ("channel", pa.string()),
            ("anchor_record_id", pa.string()),
            ("anchor_relation", pa.string()),
            ("expansion_depth", pa.int8()),
            ("match_term_ids", pa.list_(pa.string())),
        ]
    )
    language = pa.struct(
        [
            ("status", pa.string()),
            ("language_code", pa.string()),
            ("model_run_manifest_id", pa.string()),
        ]
    )
    schema = pa.schema(
        [
            ("schema_version", pa.string()),
            ("record_id", pa.string()),
            ("content_type", pa.string()),
            ("retrieval_channels", pa.list_(pa.string())),
            ("selection_evidence", pa.list_(evidence)),
            ("retrieval_policy_version", pa.string()),
            ("retrieval_policy_digest", pa.string()),
            ("manifest_id", pa.string()),
            ("language_decision", language),
        ]
    )
    if tuple(schema.names) != candidate_output_field_names():
        raise RuntimeError("candidate PyArrow schema drifted from the fixed output contract")
    return schema


def _candidate_rows(candidates: Sequence[CandidateRecord]) -> list[dict[str, Any]]:
    rows = [candidate.model_dump(mode="json") for candidate in candidates]
    for row in rows:
        if FORBIDDEN_OUTPUT_FIELDS.intersection(row):
            raise RuntimeError("candidate row contains a forbidden source field")
    return rows


def _count_output(candidates: Sequence[CandidateRecord]) -> dict[str, Any]:
    channel_candidates: Counter[str] = Counter({channel.value: 0 for channel in RetrievalChannel})
    channel_evidence_paths: Counter[str] = Counter(
        {channel.value: 0 for channel in RetrievalChannel}
    )
    content_types: Counter[str] = Counter({"submission": 0, "comment": 0})
    multi_channel = 0
    for candidate in candidates:
        content_types[candidate.content_type] += 1
        if len(candidate.retrieval_channels) > 1:
            multi_channel += 1
        channel_candidates.update(channel.value for channel in candidate.retrieval_channels)
        channel_evidence_paths.update(
            evidence.channel.value for evidence in candidate.selection_evidence
        )
    return {
        "candidate_rows": len(candidates),
        "candidate_rows_by_content_type": dict(sorted(content_types.items())),
        "candidate_rows_by_channel": dict(sorted(channel_candidates.items())),
        "selection_evidence_paths_by_channel": dict(sorted(channel_evidence_paths.items())),
        "selection_evidence_paths": sum(channel_evidence_paths.values()),
        "multi_channel_candidate_rows": multi_channel,
    }


def _verify_existing_run(final_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    receipts = sorted(final_dir.glob("receipt-*.json"))
    candidates = sorted(final_dir.glob("candidates-*.parquet"))
    unexpected = [path.name for path in final_dir.iterdir() if path not in {*receipts, *candidates}]
    if len(receipts) != 1 or len(candidates) != 1 or unexpected:
        raise RuntimeError(
            "existing proof output is not an exact immutable pair: "
            f"receipts={len(receipts)} candidates={len(candidates)} unexpected={unexpected}"
        )
    receipt_path = receipts[0]
    candidate_path = candidates[0]
    receipt_sha256 = _sha256_file(receipt_path)
    candidate_sha256 = _sha256_file(candidate_path)
    if receipt_path.name != f"receipt-{receipt_sha256}.json":
        raise RuntimeError("existing proof receipt is not content-addressed")
    if candidate_path.name != f"candidates-{candidate_sha256}.parquet":
        raise RuntimeError("existing candidate output is not content-addressed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("run_contract") != contract:
        raise RuntimeError("existing proof receipt run contract mismatch")
    output = receipt.get("output", {})
    if (
        output.get("candidate_sha256") != candidate_sha256
        or output.get("candidate_bytes") != candidate_path.stat().st_size
        or output.get("candidate_file") != candidate_path.name
        or pq.ParquetFile(candidate_path).metadata.num_rows != output.get("candidate_rows")
    ):
        raise RuntimeError("existing candidate output does not match immutable receipt")
    return {
        "status": "already_complete",
        "run_manifest_id": receipt["run_manifest_id"],
        "receipt_path": str(receipt_path.relative_to(VOLUME_PATH)),
        "receipt_sha256": receipt_sha256,
        "candidate_path": str(candidate_path.relative_to(VOLUME_PATH)),
        "counts": receipt["counts"],
        "timings_seconds": receipt["timings_seconds"],
    }


def _validate_proof_job(
    *, job: Mapping[str, Any], policy: RetrievalPolicy
) -> tuple[Mapping[str, Any], str]:
    if set(job) != {"run_contract", "run_manifest_id"}:
        raise ValueError("proof job requires exactly run_contract and run_manifest_id")
    contract = job["run_contract"]
    run_manifest_id = str(job["run_manifest_id"])
    if _canonical_sha256(contract) != run_manifest_id:
        raise RuntimeError("run manifest digest does not match its contract")
    policy_contract = contract["retrieval_policy"]
    if (
        policy.policy_version != policy_contract["policy_version"]
        or policy.policy_digest != policy_contract["policy_digest"]
        or policy.schema_version != policy_contract["schema_version"]
        or policy.lexicon_version != policy_contract["lexicon_version"]
    ):
        raise RuntimeError("loaded policy does not match pinned run contract")
    return contract, run_manifest_id


def _proof_output_dirs(
    *, contract: Mapping[str, Any], policy_digest: str, run_manifest_id: str
) -> tuple[Path, Path, Path]:
    subreddit = str(contract["cell"]["subreddit"])
    year = int(contract["cell"]["year"])
    proof_root = (
        VOLUME_PATH
        / PROOF_PREFIX
        / str(contract["revision"])
        / f"source-schema={SOURCE_SCHEMA_VERSION}"
        / f"policy={policy_digest}"
        / f"subreddit={subreddit}"
        / f"year={year}"
    )
    return (
        proof_root,
        proof_root / f"run={run_manifest_id}",
        proof_root / f".run={run_manifest_id}.incomplete",
    )


def _execute_stage_a_proof(*, job: dict[str, Any], policy: RetrievalPolicy) -> dict[str, Any]:
    """Execute an already-claimed proof while preserving immutable Volume outputs."""

    import pyarrow.parquet as pq

    contract, run_manifest_id = _validate_proof_job(job=job, policy=policy)

    subreddit = str(contract["cell"]["subreddit"])
    year = int(contract["cell"]["year"])
    max_rows = int(contract["max_input_rows_per_source"])
    max_anchor_rows = int(contract["max_anchor_scan_rows_per_source"])
    if contract.get("anchor_years") != list(ANCHOR_YEARS):
        raise RuntimeError("run contract does not pin the complete anchor-year range")
    if "input_partitions" not in contract:
        raise RuntimeError("run contract lacks exact canonical input partition digests")
    _, final_dir, staging_dir = _proof_output_dirs(
        contract=contract,
        policy_digest=policy.policy_digest,
        run_manifest_id=run_manifest_id,
    )

    volume.reload()
    if staging_dir.exists():
        raise FileExistsError(f"stale incomplete proof requires explicit inspection: {staging_dir}")
    if final_dir.exists():
        return _verify_existing_run(final_dir, contract)

    started = time.monotonic()
    input_started = time.monotonic()
    observed_inputs = _resolve_input_partitions(contract)
    if observed_inputs != contract["input_partitions"]:
        raise RuntimeError("canonical input partitions changed after run contract resolution")

    parquet_inputs: list[tuple[Path, str, int, Any]] = []
    rows_by_source_year: dict[tuple[str, int], int] = {}
    for content_type in ("submission", "comment"):
        for metadata in observed_inputs[content_type]:
            partition_year = int(metadata["year"])
            path = VOLUME_PATH / metadata["partition_relative_path"]
            parquet = pq.ParquetFile(path)
            rows = parquet.metadata.num_rows
            if rows != int(metadata["partition_rows"]):
                raise RuntimeError(f"Parquet row count changed after input resolution: {path}")
            parquet_inputs.append((path, content_type, partition_year, parquet))
            rows_by_source_year[(content_type, partition_year)] = rows

    submission_rows = rows_by_source_year[("submission", year)]
    comment_rows = rows_by_source_year[("comment", year)]
    anchor_submission_rows = sum(
        rows
        for (content_type, _), rows in rows_by_source_year.items()
        if content_type == "submission"
    )
    anchor_comment_rows = sum(
        rows for (content_type, _), rows in rows_by_source_year.items() if content_type == "comment"
    )
    enforce_input_bounds(
        submission_rows=submission_rows,
        comment_rows=comment_rows,
        max_input_rows_per_source=max_rows,
    )
    enforce_anchor_scan_bounds(
        submission_rows=anchor_submission_rows,
        comment_rows=anchor_comment_rows,
        max_anchor_scan_rows_per_source=max_anchor_rows,
    )
    input_validation_seconds = time.monotonic() - input_started

    retrieval_started = time.monotonic()
    tasks: list[dict[str, Any]] = []
    for path, content_type, partition_year, parquet in parquet_inputs:
        for row_groups in _row_group_chunks(parquet.metadata.num_row_groups, WORKER_COUNT):
            tasks.append(
                {
                    "path": str(path),
                    "content_type": content_type,
                    "row_groups": row_groups,
                    "subreddit": subreddit,
                    "year": partition_year,
                    "target_year": year,
                    "policy": policy,
                }
            )

    direct_submissions: dict[str, tuple[str, ...]] = {}
    target_direct_submission_ids: set[str] = set()
    direct_comments: dict[str, tuple[str, ...]] = {}
    observed_comments: list[tuple[str, str, str | None, tuple[str, ...]]] = []
    scanned_rows: Counter[tuple[str, int]] = Counter()
    workers = min(WORKER_COUNT, len(tasks), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_scan_row_groups, **task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            content_type = result["content_type"]
            partition_year = int(result["year"])
            scanned_rows[(content_type, partition_year)] += int(result["rows"])
            if content_type == "submission":
                for record_id, term_ids in result["results"]:
                    if record_id in direct_submissions:
                        raise RuntimeError(f"duplicate direct submission anchor: {record_id}")
                    direct_submissions[record_id] = tuple(term_ids)
                    if result["is_target"]:
                        target_direct_submission_ids.add(record_id)
            elif result["is_target"]:
                for record_id, submission_id, parent_id, term_ids in result["results"]:
                    observed_comments.append((record_id, submission_id, parent_id, term_ids))
                    if term_ids:
                        if record_id in direct_comments:
                            raise RuntimeError(f"duplicate direct comment anchor: {record_id}")
                        direct_comments[record_id] = tuple(term_ids)
            else:
                for record_id, term_ids in result["results"]:
                    if record_id in direct_comments:
                        raise RuntimeError(f"duplicate direct comment anchor: {record_id}")
                    direct_comments[record_id] = tuple(term_ids)
    if scanned_rows != Counter(rows_by_source_year):
        raise RuntimeError("parallel scan row counts do not match complete all-year inputs")
    if len({row[0] for row in observed_comments}) != len(observed_comments):
        raise RuntimeError("duplicate canonical comment ID in selected cell")
    if set(direct_submissions).intersection(row[0] for row in observed_comments):
        raise RuntimeError("canonical record ID appears in both content types")

    candidates = _assemble_candidates(
        direct_submissions=direct_submissions,
        target_direct_submission_ids=frozenset(target_direct_submission_ids),
        direct_comments=direct_comments,
        comments=sorted(observed_comments),
        policy=policy,
        manifest_id=run_manifest_id,
    )
    retrieval_seconds = time.monotonic() - retrieval_started

    write_started = time.monotonic()
    staging_dir.mkdir(parents=True)
    try:
        temporary_candidate = staging_dir / "candidates.parquet.incomplete"
        pq.write_table(
            __import__("pyarrow").Table.from_pylist(
                _candidate_rows(candidates), schema=_candidate_arrow_schema()
            ),
            temporary_candidate,
            compression="zstd",
            use_dictionary=True,
        )
        candidate_sha256 = _sha256_file(temporary_candidate)
        candidate_path = staging_dir / f"candidates-{candidate_sha256}.parquet"
        temporary_candidate.replace(candidate_path)
        counts = {
            "input_submission_rows": submission_rows,
            "input_comment_rows": comment_rows,
            "input_total_rows": submission_rows + comment_rows,
            "anchor_scan_submission_rows": anchor_submission_rows,
            "anchor_scan_comment_rows": anchor_comment_rows,
            "anchor_scan_total_rows": anchor_submission_rows + anchor_comment_rows,
            "all_year_direct_submission_anchor_rows": len(direct_submissions),
            "all_year_direct_comment_anchor_rows": len(direct_comments),
            "target_direct_submission_rows": len(target_direct_submission_ids),
            "target_direct_comment_rows": sum(bool(row[3]) for row in observed_comments),
            **_count_output(candidates),
        }
        output_write_seconds = time.monotonic() - write_started
        receipt = {
            "schema_version": PROOF_SCHEMA_VERSION,
            "status": "complete",
            "run_manifest_id": run_manifest_id,
            "run_contract": contract,
            "inputs": contract["input_partitions"],
            "counts": counts,
            "output": {
                "candidate_file": candidate_path.name,
                "candidate_rows": len(candidates),
                "candidate_bytes": candidate_path.stat().st_size,
                "candidate_sha256": candidate_sha256,
                "candidate_schema_fields": list(candidate_output_field_names()),
                "contains_raw_text": False,
                "language_status": "unclassified",
            },
            "timings_seconds": {
                "input_validation": round(input_validation_seconds, 3),
                "retrieval": round(retrieval_seconds, 3),
                "output_write": round(output_write_seconds, 3),
                "total": round(time.monotonic() - started, 3),
            },
            "runtime": {
                "requested_cpus": 8.0,
                "worker_processes": workers,
                "python_version": platform.python_version(),
                "pyarrow_version": __import__("pyarrow").__version__,
            },
            "completed_at": datetime.now(UTC).isoformat(),
        }
        receipt_json = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        receipt_sha256 = hashlib.sha256(receipt_json.encode()).hexdigest()
        receipt_path = staging_dir / f"receipt-{receipt_sha256}.json"
        receipt_path.write_text(receipt_json, encoding="utf-8")
        staging_dir.replace(final_dir)
        volume.commit()
        return {
            "status": "complete",
            "run_manifest_id": run_manifest_id,
            "receipt_path": str((final_dir / receipt_path.name).relative_to(VOLUME_PATH)),
            "receipt_sha256": receipt_sha256,
            "candidate_path": str((final_dir / candidate_path.name).relative_to(VOLUME_PATH)),
            "counts": counts,
            "timings_seconds": receipt["timings_seconds"],
        }
    except BaseException:
        if staging_dir.exists():
            with suppress(Exception):
                volume.commit()
        raise


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=8.0,
    memory=32_768,
    timeout=14_400,
    max_containers=4,
)
def run_stage_a_proof(*, job: dict[str, Any], policy: RetrievalPolicy) -> dict[str, Any]:
    """Claim and run one cell, allowing only a restart of the same Modal call."""

    contract, run_manifest_id = _validate_proof_job(job=job, policy=policy)
    proof_root, final_dir, staging_dir = _proof_output_dirs(
        contract=contract,
        policy_digest=policy.policy_digest,
        run_manifest_id=run_manifest_id,
    )
    volume.reload()
    if final_dir.exists():
        return _verify_existing_run(final_dir, contract)

    owner_call_id = modal.current_function_call_id()
    if not owner_call_id:
        raise RuntimeError("Modal did not provide a current function-call ID")
    now = datetime.now(UTC).isoformat()
    acquisition = acquire_run_claim(
        claim_registry,
        run_manifest_id=run_manifest_id,
        owner_call_id=owner_call_id,
        claimed_at=now,
    )
    try:
        volume.reload()
        if final_dir.exists():
            result = _verify_existing_run(final_dir, contract)
        else:
            if staging_dir.exists():
                if not acquisition["same_owner_retry"]:
                    raise FileExistsError(
                        f"unclaimed stale incomplete proof requires inspection: {staging_dir}"
                    )
                previous_attempt = int(acquisition["claim"]["attempt"]) - 1
                abandoned_root = proof_root / ".abandoned"
                abandoned_path = (
                    abandoned_root / f"run={run_manifest_id}.attempt={previous_attempt}.incomplete"
                )
                if abandoned_path.exists():
                    raise FileExistsError(
                        f"same-call abandoned staging already exists: {abandoned_path}"
                    )
                abandoned_root.mkdir(parents=True, exist_ok=True)
                staging_dir.replace(abandoned_path)
                volume.commit()
                update_run_claim(
                    claim_registry,
                    run_manifest_id=run_manifest_id,
                    owner_call_id=owner_call_id,
                    status="active",
                    updated_at=datetime.now(UTC).isoformat(),
                    metadata={
                        "last_abandoned_staging": str(abandoned_path.relative_to(VOLUME_PATH))
                    },
                )
            result = _execute_stage_a_proof(job=job, policy=policy)
    except BaseException as exc:
        failed_at = datetime.now(UTC).isoformat()
        try:
            update_run_claim(
                claim_registry,
                run_manifest_id=run_manifest_id,
                owner_call_id=owner_call_id,
                status="failed",
                updated_at=failed_at,
                metadata={"failed_at": failed_at, "failure_type": type(exc).__name__},
            )
        except BaseException as claim_exc:
            raise BaseExceptionGroup(
                "Stage A proof and claim-state update both failed", [exc, claim_exc]
            ) from None
        raise

    completed_at = datetime.now(UTC).isoformat()
    update_run_claim(
        claim_registry,
        run_manifest_id=run_manifest_id,
        owner_call_id=owner_call_id,
        status="complete",
        updated_at=completed_at,
        metadata={
            "completed_at": completed_at,
            "receipt_path": result["receipt_path"],
            "receipt_sha256": result["receipt_sha256"],
            "candidate_path": result["candidate_path"],
        },
    )
    return result


def _code_state(root: Path, manifest_path: Path, policy_path: Path) -> dict[str, Any]:
    tracked_inputs = [
        Path("src/reddit_china_stance/modal_stage_a_proof.py"),
        Path("src/reddit_china_stance/retrieval.py"),
        Path("schemas/candidate-record.schema.json"),
        Path("uv.lock"),
    ]
    digest = hashlib.sha256()
    for relative in tracked_inputs:
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "code_sha256": digest.hexdigest(),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "retrieval_policy_file_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }


@app.local_entrypoint()
def main(
    subreddit: str = "",
    year: int = 0,
    max_input_rows_per_source: int = DEFAULT_MAX_INPUT_ROWS_PER_SOURCE,
    max_anchor_scan_rows_per_source: int = DEFAULT_MAX_ANCHOR_SCAN_ROWS_PER_SOURCE,
    manifest_path: str = "configs/source-files.json",
    policy_path: str = "configs/retrieval-policy-v1.toml",
    confirm: str = "",
) -> None:
    """Launch exactly one bounded subreddit/year proof after explicit confirmation."""

    if confirm != CONFIRMATION:
        raise ValueError(f"refusing Stage A proof: pass --confirm {CONFIRMATION}")
    from reddit_china_stance.retrieval import load_retrieval_policy
    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest_path = root / manifest_path
    resolved_policy_path = root / policy_path
    manifest = load_source_manifest(resolved_manifest_path)
    policy = load_retrieval_policy(resolved_policy_path)
    expected_paths = {f"{subreddit}_submissions.zst", f"{subreddit}_comments.zst"}
    sources = [source for source in manifest["files"] if source["path"] in expected_paths]
    code_state = _code_state(root, resolved_manifest_path, resolved_policy_path)
    contract = make_run_contract(
        dataset_id=manifest["dataset_id"],
        revision=manifest["revision"],
        subreddit=subreddit,
        year=year,
        max_input_rows_per_source=max_input_rows_per_source,
        max_anchor_scan_rows_per_source=max_anchor_scan_rows_per_source,
        sources=sources,
        policy=policy,
        code_state=code_state,
    )
    input_partitions = resolve_stage_a_inputs.remote(contract=contract)
    contract = pin_input_partitions(contract, input_partitions)
    run_manifest_id = _canonical_sha256(contract)
    print(
        f"Launching one bounded Stage A proof cell subreddit={subreddit} year={year} "
        f"max_input_rows_per_source={max_input_rows_per_source} "
        f"max_anchor_scan_rows_per_source={max_anchor_scan_rows_per_source} "
        f"run_manifest_id={run_manifest_id}",
        flush=True,
    )
    result = run_stage_a_proof.remote(
        job={"run_contract": contract, "run_manifest_id": run_manifest_id},
        policy=policy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
