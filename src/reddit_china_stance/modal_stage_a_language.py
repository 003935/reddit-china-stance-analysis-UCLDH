"""Candidate-scoped, metadata-only language evidence for frozen Stage A."""

from __future__ import annotations

import hashlib
import heapq
import json
import platform
import subprocess
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance.modal_stage_a import (
    ANCHOR_YEARS,
    CONTENT_TYPES,
    ENVIRONMENT_NAME,
    SUBREDDITS,
    _candidate_output_dirs,
    _receipt_partition_metadata,
    _run_root,
    _sha256_file,
    _validate_partition_file,
    _verify_pair,
    make_production_contract,
    production_run_id,
)
from reddit_china_stance.modal_stage_a import (
    _code_state as _stage_a_code_state,
)
from reddit_china_stance.modal_stage_a_proof import (
    VOLUME_NAME,
    VOLUME_PATH,
    acquire_run_claim,
    candidate_output_field_names,
    update_run_claim,
)
from reddit_china_stance.retrieval import RetrievalPolicy

APP_NAME = "reddit-china-stance-stage-a-language"
CLAIM_REGISTRY_NAME = "reddit-china-stance-stage-a-language-claims"
LANGUAGE_PREFIX = Path("language")
LANGUAGE_CONTRACT_SCHEMA_VERSION = "1.0.0"
LANGUAGE_OUTPUT_SCHEMA_VERSION = "1.0.0"
CONFIRMATION = "LAUNCH_ALL_60_STAGE_A_LANGUAGE_CELLS"
SMOKE_CONFIRMATION = "RUN_ONE_STAGE_A_LANGUAGE_SMOKE"
FROZEN_STAGE_A_RUN_ID = "07556d5f8f472560e5e09008b97fd138bfb2bc6312d018fd8662c76ea3a873f7"
FROZEN_RETRIEVAL_POLICY_DIGEST = "5b7ced047d90407ed8849bcec8744078ed768e74eb78851624c67c009b684d46"
EXPECTED_STAGE_A_CANDIDATE_RECEIPTS = 60
EXPECTED_STAGE_A_CANDIDATE_ROWS = 14_487_562
CPU_HOUR_COST = Decimal("0.04730")
MEM_GIB_HOUR_COST = Decimal("0.00800")
MODEL_PATH = Path("/opt/language-models/lid.176.bin")
MODEL_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
MODEL_BYTES = 131_266_198
SMOKE_HARD_MAX_COST_USD = Decimal("2")
DECISION_BATCH_ROWS = 20_000
CANONICAL_BATCH_ROWS = 50_000
FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {"text", "raw_text", "body", "title", "selftext", "author", "source_id", "text_sha256"}
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
    .apt_install("build-essential", "curl", "ca-certificates")
    .uv_pip_install(
        "fasttext-wheel==0.9.2",
        "lingua-language-detector==2.2.0",
        "numpy==1.26.4",
        "pyahocorasick>=2.2,<3",
        "pyarrow==25.0.1",
        "pydantic>=2.11.0,<3",
    )
    .run_commands(
        "mkdir -p /opt/language-models",
        (
            "curl -fsSL https://dl.fbaipublicfiles.com/fasttext/"
            "supervised-models/lid.176.bin -o /opt/language-models/lid.176.bin"
        ),
        f"printf '{MODEL_SHA256}  /opt/language-models/lid.176.bin\\n' | sha256sum -c -",
    )
    .add_local_python_source("reddit_china_stance")
)

_FASTTEXT_MODEL: Any | None = None
_LINGUA_DETECTOR: Any | None = None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}")


def load_language_policy(path: Path) -> dict[str, Any]:
    """Load the closed language policy and attach its canonical digest."""

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "policy_version",
            "output_schema_version",
            "primary",
            "challenger",
            "sampling",
            "triage",
            "compute",
        },
        "language policy",
    )
    _require_exact_keys(
        raw["primary"],
        {
            "detector_id",
            "package",
            "package_version",
            "model_url",
            "model_path",
            "model_bytes",
            "model_sha256",
            "top_k",
        },
        "primary detector",
    )
    _require_exact_keys(
        raw["challenger"],
        {
            "detector_id",
            "package",
            "package_version",
            "language_set",
            "low_accuracy_mode",
        },
        "challenger detector",
    )
    _require_exact_keys(
        raw["sampling"],
        {"sample_size", "seed", "minimum_per_nonempty_stratum", "stratify_by"},
        "sampling policy",
    )
    _require_exact_keys(
        raw["triage"],
        {"high_confidence_threshold", "exclude_without_human_audit"},
        "triage policy",
    )
    _require_exact_keys(
        raw["compute"],
        {
            "requested_cpus",
            "requested_memory_mib",
            "max_containers",
            "conservative_scan_rows_per_second",
            "conservative_primary_rows_per_second",
            "conservative_challenger_rows_per_second",
            "cost_overhead_factor",
            "hard_max_benchmark_cost_usd",
            "hard_max_estimated_cost_usd",
        },
        "compute policy",
    )
    if raw["schema_version"] != LANGUAGE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported language policy schema version")
    if raw["output_schema_version"] != LANGUAGE_OUTPUT_SCHEMA_VERSION:
        raise ValueError("unsupported language output schema version")
    primary = raw["primary"]
    if (
        primary["detector_id"] != "fasttext-lid.176.bin"
        or primary["package"] != "fasttext-wheel"
        or primary["package_version"] != "0.9.2"
        or primary["model_path"] != str(MODEL_PATH)
        or primary["model_sha256"] != MODEL_SHA256
        or primary["model_bytes"] != MODEL_BYTES
        or primary["top_k"] != 5
    ):
        raise ValueError("primary detector does not match the pinned runtime artifact")
    challenger = raw["challenger"]
    if (
        challenger["detector_id"] != "lingua-all-languages"
        or challenger["package"] != "lingua-language-detector"
        or challenger["package_version"] != "2.2.0"
        or challenger["language_set"] != "all"
        or challenger["low_accuracy_mode"] is not False
    ):
        raise ValueError("challenger detector does not match the pinned runtime")
    sampling = raw["sampling"]
    if type(sampling["sample_size"]) is not int or sampling["sample_size"] <= 0:
        raise ValueError("sample_size must be a positive integer")
    if (
        type(sampling["minimum_per_nonempty_stratum"]) is not int
        or sampling["minimum_per_nonempty_stratum"] < 0
    ):
        raise ValueError("minimum_per_nonempty_stratum must be a non-negative integer")
    if sampling["stratify_by"] != [
        "subreddit",
        "year",
        "content_type",
        "retrieval_channel_bucket",
    ]:
        raise ValueError("sampling strata are not the frozen four-way design")
    threshold = raw["triage"]["high_confidence_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("high_confidence_threshold must be strictly between zero and one")
    if raw["triage"]["exclude_without_human_audit"] is not False:
        raise ValueError("language policy must prohibit exclusion before human audit")
    if raw["compute"]["hard_max_benchmark_cost_usd"] > 2:
        raise ValueError("language benchmark hard cost ceiling cannot exceed $2")
    if raw["compute"]["hard_max_estimated_cost_usd"] > 25:
        raise ValueError("language full-wave hard cost ceiling cannot exceed $25")
    canonical = json.loads(json.dumps(raw, sort_keys=True))
    return {**canonical, "policy_digest": _canonical_sha256(canonical)}


def _code_state(root: Path, policy_path: Path) -> dict[str, Any]:
    tracked = (
        root / "src/reddit_china_stance/modal_stage_a_language.py",
        policy_path,
        root / "schemas/language-decision.schema.json",
        root / "uv.lock",
    )
    files = {str(path.relative_to(root)): _sha256_file(path) for path in tracked}
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "files": files,
        "code_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "git_commit": git_commit,
    }


def retrieval_channel_bucket(channels: Sequence[str]) -> str:
    """Collapse selection channels into one deterministic sampling bucket."""

    values = tuple(sorted(set(channels)))
    allowed = {"direct_lexical", "submission_expansion", "direct_reply_expansion"}
    if not values or not set(values).issubset(allowed):
        raise ValueError("retrieval channels must be a non-empty subset of the frozen channels")
    return values[0] if len(values) == 1 else "multi_channel"


def stratum_key(*, subreddit: str, year: int, content_type: str, channel_bucket: str) -> str:
    return f"{subreddit}|{year}|{content_type}|{channel_bucket}"


def allocate_stratified_quotas(
    counts: Mapping[str, int], *, sample_size: int, minimum_per_nonempty_stratum: int
) -> dict[str, int]:
    """Allocate an exact sample using minimums plus proportional largest remainder."""

    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if type(minimum_per_nonempty_stratum) is not int or minimum_per_nonempty_stratum < 0:
        raise ValueError("minimum_per_nonempty_stratum must be non-negative")
    clean: dict[str, int] = {}
    for key, count in counts.items():
        if not key or type(count) is not int or count < 0:
            raise ValueError("stratum counts require non-empty keys and non-negative integers")
        if count:
            clean[key] = count
    population = sum(clean.values())
    if sample_size > population:
        raise ValueError("sample_size cannot exceed the non-empty population")
    quotas = {key: min(count, minimum_per_nonempty_stratum) for key, count in clean.items()}
    if sum(quotas.values()) > sample_size:
        raise ValueError("stratum minimums exceed the requested sample size")
    remaining = sample_size - sum(quotas.values())
    while remaining:
        capacity = {key: clean[key] - quotas[key] for key in clean if clean[key] > quotas[key]}
        capacity_total = sum(capacity.values())
        if not capacity_total:
            raise RuntimeError("sampling allocation exhausted before reaching sample_size")
        exact = {
            key: Decimal(remaining) * value / capacity_total for key, value in capacity.items()
        }
        floor_add = {key: min(capacity[key], int(value)) for key, value in exact.items()}
        added = sum(floor_add.values())
        for key, value in floor_add.items():
            quotas[key] += value
        remaining -= added
        if not remaining:
            break
        order = sorted(
            capacity,
            key=lambda key: (-(exact[key] - int(exact[key])), key),
        )
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if quotas[key] < clean[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("sampling largest-remainder allocation made no progress")
    if sum(quotas.values()) != sample_size or any(quotas[key] > clean[key] for key in quotas):
        raise RuntimeError("sampling allocation failed conservation checks")
    return dict(sorted(quotas.items()))


def stable_sample_rank(seed: str, record_id: str) -> int:
    if not seed or not record_id:
        raise ValueError("sample seed and record_id must be non-empty")
    return int.from_bytes(hashlib.sha256(f"{seed}\0{record_id}".encode()).digest(), byteorder="big")


def select_bottom_k(
    rows: Sequence[tuple[str, str]], quotas: Mapping[str, int], *, seed: str
) -> dict[str, tuple[str, int]]:
    """Select stable bottom-hash record IDs per stratum for tests and small inputs."""

    heaps: dict[str, list[tuple[int, str]]] = {key: [] for key in quotas}
    seen: set[str] = set()
    for record_id, key in rows:
        if record_id in seen:
            raise ValueError(f"duplicate sample candidate record_id: {record_id}")
        seen.add(record_id)
        quota = quotas.get(key, 0)
        if quota <= 0:
            continue
        rank = stable_sample_rank(seed, record_id)
        heap = heaps[key]
        item = (-rank, record_id)
        if len(heap) < quota:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected: dict[str, tuple[str, int]] = {}
    for key, heap in heaps.items():
        if len(heap) != quotas[key]:
            raise RuntimeError(f"stratum did not contain its allocated quota: {key}")
        for negative_rank, record_id in heap:
            selected[record_id] = (key, -negative_rank)
    return selected


def _count_candidate_strata(path: Path, *, subreddit: str, year: int) -> Counter[str]:
    import pyarrow.parquet as pq

    counts: Counter[str] = Counter()
    parquet = pq.ParquetFile(path)
    expected_fields = set(candidate_output_field_names())
    if set(parquet.schema_arrow.names) != expected_fields:
        raise RuntimeError(f"candidate output schema mismatch: {path}")
    for batch in parquet.iter_batches(
        batch_size=100_000, columns=["content_type", "retrieval_channels"]
    ):
        data = batch.to_pydict()
        for content_type, channels in zip(
            data["content_type"], data["retrieval_channels"], strict=True
        ):
            if content_type not in CONTENT_TYPES:
                raise RuntimeError("candidate has invalid content type")
            bucket = retrieval_channel_bucket(channels)
            counts[
                stratum_key(
                    subreddit=subreddit,
                    year=year,
                    content_type=content_type,
                    channel_bucket=bucket,
                )
            ] += 1
    if sum(counts.values()) != parquet.metadata.num_rows:
        raise RuntimeError("candidate stratum count does not conserve rows")
    return counts


def estimate_language_cost_usd(
    *, canonical_rows: int, candidate_rows: int, benchmark_rows: int, policy: Mapping[str, Any]
) -> Decimal:
    """Estimate one verified canonical scan, full primary pass, and challenger sample."""

    for name, value in (
        ("canonical_rows", canonical_rows),
        ("candidate_rows", candidate_rows),
        ("benchmark_rows", benchmark_rows),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    compute = policy["compute"]
    scan_seconds = Decimal(canonical_rows * 2) / Decimal(
        compute["conservative_scan_rows_per_second"]
    )
    primary_seconds = Decimal(candidate_rows) / Decimal(
        compute["conservative_primary_rows_per_second"]
    )
    challenger_seconds = Decimal(benchmark_rows) / Decimal(
        compute["conservative_challenger_rows_per_second"]
    )
    hours = (scan_seconds + primary_seconds + challenger_seconds) / Decimal(3600)
    hourly = (
        Decimal(compute["requested_cpus"]) * CPU_HOUR_COST
        + Decimal(compute["requested_memory_mib"]) / Decimal(1024) * MEM_GIB_HOUR_COST
    )
    return (hours * hourly * Decimal(str(compute["cost_overhead_factor"]))).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )


def estimate_benchmark_incremental_cost_usd(
    *, benchmark_rows: int, policy: Mapping[str, Any]
) -> Decimal:
    """Price the challenger inference added to the shared single canonical pass."""

    if type(benchmark_rows) is not int or benchmark_rows < 0:
        raise ValueError("benchmark_rows must be a non-negative integer")
    compute = policy["compute"]
    hours = (
        Decimal(benchmark_rows)
        / Decimal(compute["conservative_challenger_rows_per_second"])
        / Decimal(3600)
    )
    hourly = (
        Decimal(compute["requested_cpus"]) * CPU_HOUR_COST
        + Decimal(compute["requested_memory_mib"]) / Decimal(1024) * MEM_GIB_HOUR_COST
    )
    return (hours * hourly * Decimal(str(compute["cost_overhead_factor"]))).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )


def enforce_cost_guardrail(
    *, estimated_cost_usd: Decimal, approved_usd: Decimal, policy: Mapping[str, Any]
) -> None:
    hard_max = Decimal(str(policy["compute"]["hard_max_estimated_cost_usd"]))
    if approved_usd <= 0 or approved_usd > hard_max:
        raise ValueError(f"approved_usd must be > 0 and <= {hard_max}")
    if estimated_cost_usd > approved_usd:
        raise RuntimeError(
            f"estimated language cost ${estimated_cost_usd} exceeds approved ${approved_usd}"
        )


def estimate_cell_cost_usd(
    *, ref: Mapping[str, Any], sample_quotas: Mapping[str, int], policy: Mapping[str, Any]
) -> Decimal:
    """Estimate one exact production cell using the same conservative model."""

    subreddit = str(ref["subreddit"])
    year = int(ref["year"])
    canonical_rows = sum(
        int(partition["partition_rows"]) for partition in ref["canonical_inputs"].values()
    )
    benchmark_rows = sum(
        int(quota) for key, quota in sample_quotas.items() if key.startswith(f"{subreddit}|{year}|")
    )
    return estimate_language_cost_usd(
        canonical_rows=canonical_rows,
        candidate_rows=int(ref["rows"]),
        benchmark_rows=benchmark_rows,
        policy=policy,
    )


def enforce_smoke_cost_guardrail(*, estimated_cost_usd: Decimal, approved_usd: Decimal) -> None:
    if approved_usd <= 0 or approved_usd > SMOKE_HARD_MAX_COST_USD:
        raise ValueError(f"smoke approved_usd must be > 0 and <= {SMOKE_HARD_MAX_COST_USD}")
    if estimated_cost_usd > approved_usd:
        raise RuntimeError(
            f"estimated language smoke cost ${estimated_cost_usd} exceeds approved ${approved_usd}"
        )


def _require_frozen_stage_a_binding(
    *, stage_a_contract: Mapping[str, Any], stage_a_run_manifest_id: str
) -> None:
    """Reject every Stage A contract except the completed v3 production run."""

    if stage_a_run_manifest_id != FROZEN_STAGE_A_RUN_ID:
        raise RuntimeError("Stage A run ID is not the frozen completed v3 production run")
    if production_run_id(stage_a_contract) != stage_a_run_manifest_id:
        raise RuntimeError("Stage A run ID does not match its contract")
    retrieval_policy = stage_a_contract.get("retrieval_policy")
    if not isinstance(retrieval_policy, Mapping) or retrieval_policy.get("policy_digest") != (
        FROZEN_RETRIEVAL_POLICY_DIGEST
    ):
        raise RuntimeError("Stage A retrieval policy is not the frozen v3 policy")


def make_language_contract(
    *,
    stage_a_contract: Mapping[str, Any],
    stage_a_run_manifest_id: str,
    policy: Mapping[str, Any],
    code_state: Mapping[str, Any],
    candidate_refs: Sequence[Mapping[str, Any]],
    stratum_counts: Mapping[str, int],
) -> dict[str, Any]:
    _require_frozen_stage_a_binding(
        stage_a_contract=stage_a_contract,
        stage_a_run_manifest_id=stage_a_run_manifest_id,
    )
    expected_cells = {(subreddit, year) for subreddit in SUBREDDITS for year in ANCHOR_YEARS}
    refs = [dict(ref) for ref in candidate_refs]
    observed_cells = {(ref.get("subreddit"), ref.get("year")) for ref in refs}
    if observed_cells != expected_cells or len(refs) != EXPECTED_STAGE_A_CANDIDATE_RECEIPTS:
        raise ValueError(
            "language contract must pin exactly "
            f"{EXPECTED_STAGE_A_CANDIDATE_RECEIPTS} unique candidate outputs"
        )
    refs.sort(key=lambda row: (row["subreddit"], row["year"]))
    candidate_rows = sum(int(ref["rows"]) for ref in refs)
    if candidate_rows != EXPECTED_STAGE_A_CANDIDATE_ROWS:
        raise ValueError(
            f"language contract must pin exactly {EXPECTED_STAGE_A_CANDIDATE_ROWS:,} candidate rows"
        )
    if sum(stratum_counts.values()) != candidate_rows:
        raise ValueError("language strata do not conserve candidate rows")
    sample_size = int(policy["sampling"]["sample_size"])
    quotas = allocate_stratified_quotas(
        stratum_counts,
        sample_size=sample_size,
        minimum_per_nonempty_stratum=int(policy["sampling"]["minimum_per_nonempty_stratum"]),
    )
    canonical_rows = sum(int(row["partition_rows"]) for row in stage_a_contract["input_partitions"])
    estimated = estimate_language_cost_usd(
        canonical_rows=canonical_rows,
        candidate_rows=candidate_rows,
        benchmark_rows=sample_size,
        policy=policy,
    )
    benchmark_estimated = estimate_benchmark_incremental_cost_usd(
        benchmark_rows=sample_size, policy=policy
    )
    if benchmark_estimated > Decimal(str(policy["compute"]["hard_max_benchmark_cost_usd"])):
        raise RuntimeError("estimated language benchmark exceeds its $2 hard ceiling")
    return {
        "schema_version": LANGUAGE_CONTRACT_SCHEMA_VERSION,
        "stage_a_run_manifest_id": stage_a_run_manifest_id,
        "stage_a_contract_sha256": _canonical_sha256(stage_a_contract),
        "dataset_id": stage_a_contract["dataset_id"],
        "revision": stage_a_contract["revision"],
        "source_schema_version": stage_a_contract["source_schema_version"],
        "retrieval_policy_digest": stage_a_contract["retrieval_policy"]["policy_digest"],
        "language_policy": dict(policy),
        "candidate_inputs": refs,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "sample_quotas": quotas,
        "counts": {
            "cells": len(refs),
            "canonical_rows": canonical_rows,
            "candidate_rows": candidate_rows,
            "benchmark_rows": sample_size,
            "canonical_row_visits_including_digest_verification": canonical_rows * 2,
        },
        "compute_plan": {
            **dict(policy["compute"]),
            "estimated_benchmark_incremental_cost_usd": str(benchmark_estimated),
            "estimated_full_run_cost_usd": str(estimated),
        },
        "code_state": dict(code_state),
    }


def language_run_id(contract: Mapping[str, Any]) -> str:
    return _canonical_sha256(contract)


def _language_root(contract: Mapping[str, Any], run_id: str) -> Path:
    stage_root = _run_root(
        {
            "revision": contract["revision"],
            "retrieval_policy": {"policy_digest": contract["retrieval_policy_digest"]},
        },
        str(contract["stage_a_run_manifest_id"]),
    )
    return (
        stage_root
        / LANGUAGE_PREFIX
        / f"policy={contract['language_policy']['policy_digest']}"
        / f"run={run_id}"
    )


def language_output_dirs(
    *, contract: Mapping[str, Any], run_id: str, subreddit: str, year: int
) -> tuple[Path, Path]:
    unit_contract = {
        "schema_version": LANGUAGE_CONTRACT_SCHEMA_VERSION,
        "language_run_id": run_id,
        "cell": {"subreddit": subreddit, "year": year},
    }
    unit_id = _canonical_sha256(unit_contract)
    root = (
        _language_root(contract, run_id) / "decisions" / f"subreddit={subreddit}" / f"year={year}"
    )
    return root / f"unit={unit_id}", root / f".unit={unit_id}.incomplete"


def _decision_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("record_id", pa.string()),
            ("content_type", pa.string()),
            ("primary_language_code", pa.string()),
            ("primary_confidence", pa.float32()),
            ("primary_english_confidence", pa.float32()),
            (
                "primary_top_predictions",
                pa.list_(
                    pa.struct(
                        [
                            ("language_code", pa.string()),
                            ("confidence", pa.float32()),
                        ]
                    )
                ),
            ),
            ("provisional_status", pa.string()),
            ("exclusion_allowed", pa.bool_()),
            ("detector_id", pa.string()),
            ("language_policy_digest", pa.string()),
            ("stage_a_run_manifest_id", pa.string()),
        ]
    )


def _benchmark_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("record_id", pa.string()),
            ("content_type", pa.string()),
            ("sample_stratum", pa.string()),
            ("sample_rank_sha256", pa.string()),
            ("inclusion_probability", pa.float64()),
            ("text_length_chars", pa.int32()),
            ("primary_language_code", pa.string()),
            ("primary_confidence", pa.float32()),
            ("primary_english_confidence", pa.float32()),
            ("challenger_language_code", pa.string()),
            ("challenger_confidence", pa.float32()),
            ("challenger_english_confidence", pa.float32()),
            ("agreement_bucket", pa.string()),
            ("primary_detector_id", pa.string()),
            ("challenger_detector_id", pa.string()),
            ("language_policy_digest", pa.string()),
            ("stage_a_run_manifest_id", pa.string()),
        ]
    )


def language_output_field_names() -> tuple[str, ...]:
    names = tuple(_decision_schema().names)
    if FORBIDDEN_OUTPUT_FIELDS.intersection(names):
        raise RuntimeError("language output contains a forbidden source field")
    return names


def benchmark_output_field_names() -> tuple[str, ...]:
    names = tuple(_benchmark_schema().names)
    if FORBIDDEN_OUTPUT_FIELDS.intersection(names):
        raise RuntimeError("language benchmark output contains a forbidden source field")
    return names


def provisional_status(
    language_code: str, confidence: float, *, high_confidence_threshold: float
) -> str:
    if not 0 <= confidence <= 1:
        raise ValueError("language confidence must be between zero and one")
    if confidence < high_confidence_threshold or language_code == "und":
        return "uncertain"
    return "provisional_english" if language_code == "en" else "provisional_non_english"


def agreement_bucket(
    *,
    primary_code: str,
    primary_confidence: float,
    challenger_code: str,
    challenger_confidence: float,
    high_confidence_threshold: float,
) -> str:
    if primary_code != challenger_code:
        return "language_disagreement"
    if min(primary_confidence, challenger_confidence) < high_confidence_threshold:
        return "agreement_low_confidence"
    return "agreement_high_english" if primary_code == "en" else "agreement_high_non_english"


def _verify_model_artifact() -> None:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size != MODEL_BYTES:
        raise RuntimeError("pinned fastText language model size mismatch")
    if _sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("pinned fastText language model digest mismatch")


def _load_fasttext() -> Any:
    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is None:
        import fasttext

        _verify_model_artifact()
        _FASTTEXT_MODEL = fasttext.load_model(str(MODEL_PATH))
    return _FASTTEXT_MODEL


def _load_lingua() -> Any:
    global _LINGUA_DETECTOR
    if _LINGUA_DETECTOR is None:
        from lingua import LanguageDetectorBuilder

        _LINGUA_DETECTOR = (
            LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        )
    return _LINGUA_DETECTOR


def _clean_text(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").replace("\x00", " ").strip()


def _primary_predictions(texts: Sequence[str], *, top_k: int) -> list[dict[str, Any]]:
    cleaned = [_clean_text(text) for text in texts]
    nonempty_positions = [index for index, text in enumerate(cleaned) if text]
    results: list[dict[str, Any]] = [
        {
            "language_code": "und",
            "confidence": 0.0,
            "english_confidence": None,
            "top_predictions": [],
        }
        for _ in texts
    ]
    if not nonempty_positions:
        return results
    model = _load_fasttext()
    labels, probabilities = model.predict([cleaned[index] for index in nonempty_positions], k=top_k)
    for position, row_labels, row_probabilities in zip(
        nonempty_positions, labels, probabilities, strict=True
    ):
        codes = [str(label).removeprefix("__label__") for label in row_labels]
        scores = [min(1.0, max(0.0, float(value))) for value in row_probabilities]
        english = scores[codes.index("en")] if "en" in codes else None
        results[position] = {
            "language_code": codes[0],
            "confidence": scores[0],
            "english_confidence": english,
            "top_predictions": [
                {"language_code": code, "confidence": score}
                for code, score in zip(codes, scores, strict=True)
            ],
        }
    return results


def _challenger_prediction(text: str) -> dict[str, Any]:
    from lingua import Language

    cleaned = _clean_text(text)
    if not cleaned:
        return {"language_code": "und", "confidence": 0.0, "english_confidence": 0.0}
    values = _load_lingua().compute_language_confidence_values(cleaned)
    if not values:
        return {"language_code": "und", "confidence": 0.0, "english_confidence": 0.0}
    top = values[0]
    english = next((value.value for value in values if value.language == Language.ENGLISH), 0.0)
    return {
        "language_code": top.language.iso_code_639_1.name.lower(),
        "confidence": min(1.0, max(0.0, float(top.value))),
        "english_confidence": min(1.0, max(0.0, float(english))),
    }


def _peak_rss_mib() -> float:
    """Return process peak resident memory using platform-specific ru_maxrss units."""

    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return peak / divisor


def _write_json_receipt(staging: Path, receipt: Mapping[str, Any]) -> tuple[Path, str]:
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    path = staging / f"receipt-{digest}.json"
    path.write_text(payload, encoding="utf-8")
    return path, digest


def _verify_language_output(
    *, contract: Mapping[str, Any], run_id: str, subreddit: str, year: int
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    final, _ = language_output_dirs(
        contract=contract, run_id=run_id, subreddit=subreddit, year=year
    )
    files = list(final.iterdir()) if final.exists() else []
    receipts = [path for path in files if path.name.startswith("receipt-")]
    decisions = [path for path in files if path.name.startswith("language-decisions-")]
    benchmarks = [path for path in files if path.name.startswith("language-benchmark-")]
    if len(files) != 3 or len(receipts) != 1 or len(decisions) != 1 or len(benchmarks) != 1:
        raise RuntimeError(f"language unit is not an exact immutable output trio: {final}")
    receipt_path, decision_path, benchmark_path = receipts[0], decisions[0], benchmarks[0]
    receipt_sha = _sha256_file(receipt_path)
    decision_sha = _sha256_file(decision_path)
    benchmark_sha = _sha256_file(benchmark_path)
    if receipt_path.name != f"receipt-{receipt_sha}.json":
        raise RuntimeError("language receipt is not content-addressed")
    if decision_path.name != f"language-decisions-{decision_sha}.parquet":
        raise RuntimeError("language decision output is not content-addressed")
    if benchmark_path.name != f"language-benchmark-{benchmark_sha}.parquet":
        raise RuntimeError("language benchmark output is not content-addressed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_unit = {
        "schema_version": LANGUAGE_CONTRACT_SCHEMA_VERSION,
        "language_run_id": run_id,
        "cell": {"subreddit": subreddit, "year": year},
    }
    if receipt.get("unit_contract") != expected_unit:
        raise RuntimeError("language receipt unit contract mismatch")
    for key, path, digest, expected_schema in (
        ("decisions", decision_path, decision_sha, _decision_schema()),
        ("benchmark", benchmark_path, benchmark_sha, _benchmark_schema()),
    ):
        fields = tuple(expected_schema.names)
        output = receipt.get("outputs", {}).get(key, {})
        parquet = pq.ParquetFile(path)
        if (
            output.get("sha256") != digest
            or output.get("bytes") != path.stat().st_size
            or output.get("rows") != parquet.metadata.num_rows
            or output.get("schema_fields") != list(fields)
            or not parquet.schema_arrow.equals(expected_schema)
            or output.get("contains_raw_text") is not False
        ):
            raise RuntimeError(f"language {key} output does not match receipt")
    return {
        "receipt": receipt,
        "receipt_sha256": receipt_sha,
        "receipt_relative_path": str(receipt_path.relative_to(VOLUME_PATH)),
        "final_relative_path": str(final.relative_to(VOLUME_PATH)),
    }


def _resolve_candidate_ref(
    *, stage_a_contract: Mapping[str, Any], stage_a_run_manifest_id: str, subreddit: str, year: int
) -> tuple[dict[str, Any], Counter[str]]:
    final, _ = _candidate_output_dirs(
        contract=stage_a_contract,
        run_manifest_id=stage_a_run_manifest_id,
        subreddit=subreddit,
        year=year,
    )
    unit = {
        "schema_version": stage_a_contract["schema_version"],
        "run_manifest_id": stage_a_run_manifest_id,
        "phase": "candidate",
        "cell": {"subreddit": subreddit, "year": year},
    }
    verified = _verify_pair(final, receipt_contract=unit, parquet_prefix="candidates")
    strata = _count_candidate_strata(verified["output_path"], subreddit=subreddit, year=year)
    partitions = {
        row["content_type"]: dict(row)
        for row in stage_a_contract["input_partitions"]
        if row["subreddit"] == subreddit and row["year"] == year
    }
    if set(partitions) != set(CONTENT_TYPES):
        raise RuntimeError("language cell lacks exact canonical input partitions")
    receipt = verified["receipt"]
    if sum(strata.values()) != receipt["output"]["rows"]:
        raise RuntimeError("candidate receipt row count differs from stratum scan")
    return (
        {
            "subreddit": subreddit,
            "year": year,
            "candidate_receipt_relative_path": str(
                verified["receipt_path"].relative_to(VOLUME_PATH)
            ),
            "candidate_receipt_sha256": verified["receipt_sha256"],
            "candidate_output_relative_path": str(verified["output_path"].relative_to(VOLUME_PATH)),
            "candidate_output_sha256": verified["output_sha256"],
            "candidate_output_bytes": verified["output_path"].stat().st_size,
            "rows": int(receipt["output"]["rows"]),
            "canonical_inputs": partitions,
        },
        strata,
    )


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=8192,
    timeout=3_600,
    max_containers=1,
)
def resolve_stage_a_contract_for_language(
    *,
    manifest: dict[str, Any],
    retrieval_policy: RetrievalPolicy,
    code_state: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the exact Stage A contract inside this app from canonical receipts."""

    if retrieval_policy.policy_digest != FROZEN_RETRIEVAL_POLICY_DIGEST:
        raise RuntimeError("Stage A retrieval policy is not the frozen v3 policy")
    volume.reload()
    sources = sorted(manifest["files"], key=lambda row: row["path"])
    partitions = [
        _receipt_partition_metadata(manifest=manifest, source=source, year=year)
        for source in sources
        for year in ANCHOR_YEARS
    ]
    stage_a_contract = make_production_contract(
        manifest=manifest,
        policy=retrieval_policy,
        code_state=code_state,
        input_partitions=partitions,
    )
    if production_run_id(stage_a_contract) != FROZEN_STAGE_A_RUN_ID:
        raise RuntimeError(
            "computed Stage A run ID does not match the frozen completed v3 production run"
        )
    return stage_a_contract


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=8192,
    timeout=3_600,
    max_containers=1,
)
def resolve_language_contract(
    *,
    stage_a_contract: dict[str, Any],
    stage_a_run_manifest_id: str,
    policy: dict[str, Any],
    code_state: dict[str, Any],
) -> dict[str, Any]:
    volume.reload()
    _require_frozen_stage_a_binding(
        stage_a_contract=stage_a_contract,
        stage_a_run_manifest_id=stage_a_run_manifest_id,
    )
    refs: list[dict[str, Any]] = []
    strata: Counter[str] = Counter()
    for subreddit in SUBREDDITS:
        for year in ANCHOR_YEARS:
            ref, counts = _resolve_candidate_ref(
                stage_a_contract=stage_a_contract,
                stage_a_run_manifest_id=stage_a_run_manifest_id,
                subreddit=subreddit,
                year=year,
            )
            refs.append(ref)
            strata.update(counts)
    return make_language_contract(
        stage_a_contract=stage_a_contract,
        stage_a_run_manifest_id=stage_a_run_manifest_id,
        policy=policy,
        code_state=code_state,
        candidate_refs=refs,
        stratum_counts=strata,
    )


def _validate_language_job(job: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if set(job) != {"contract", "language_run_id", "cell"}:
        raise ValueError("language job has unexpected fields")
    contract = dict(job["contract"])
    run_id = str(job["language_run_id"])
    if language_run_id(contract) != run_id:
        raise RuntimeError("language run ID does not match contract")
    cell = dict(job["cell"])
    if set(cell) != {"subreddit", "year"}:
        raise ValueError("language cell must contain exactly subreddit and year")
    if cell["subreddit"] not in SUBREDDITS or cell["year"] not in ANCHOR_YEARS:
        raise ValueError("language cell is outside the frozen Stage A grid")
    refs = [
        ref
        for ref in contract["candidate_inputs"]
        if ref["subreddit"] == cell["subreddit"] and ref["year"] == cell["year"]
    ]
    if len(refs) != 1:
        raise RuntimeError("language cell does not resolve one exact candidate input")
    return contract, run_id, dict(refs[0])


def _cell_sample(
    candidate_path: Path,
    *,
    subreddit: str,
    year: int,
    quotas: Mapping[str, int],
    seed: str,
) -> tuple[dict[str, set[str]], dict[str, tuple[str, int]]]:
    import pyarrow.parquet as pq

    ids: dict[str, set[str]] = {content_type: set() for content_type in CONTENT_TYPES}
    heaps: dict[str, list[tuple[int, str]]] = {
        key: [] for key in quotas if key.startswith(f"{subreddit}|{year}|") and quotas[key] > 0
    }
    parquet = pq.ParquetFile(candidate_path)
    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["record_id", "content_type", "retrieval_channels"],
    ):
        data = batch.to_pydict()
        for record_id, content_type, channels in zip(
            data["record_id"], data["content_type"], data["retrieval_channels"], strict=True
        ):
            if not isinstance(record_id, str) or not record_id or content_type not in ids:
                raise RuntimeError("candidate has invalid identity fields")
            if record_id in ids[content_type] or any(
                record_id in ids[other] for other in CONTENT_TYPES if other != content_type
            ):
                raise RuntimeError(f"duplicate candidate record ID: {record_id}")
            ids[content_type].add(record_id)
            key = stratum_key(
                subreddit=subreddit,
                year=year,
                content_type=content_type,
                channel_bucket=retrieval_channel_bucket(channels),
            )
            quota = int(quotas.get(key, 0))
            if quota:
                rank = stable_sample_rank(seed, record_id)
                heap = heaps[key]
                if len(heap) < quota:
                    heapq.heappush(heap, (-rank, record_id))
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, (-rank, record_id))
    selected: dict[str, tuple[str, int]] = {}
    for key, heap in heaps.items():
        if len(heap) != quotas[key]:
            raise RuntimeError(f"candidate cell did not fill benchmark quota: {key}")
        for negative_rank, record_id in heap:
            selected[record_id] = (key, -negative_rank)
    if sum(len(values) for values in ids.values()) != parquet.metadata.num_rows:
        raise RuntimeError("candidate ID scan did not conserve rows")
    return ids, selected


def _execute_language(job: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract, run_id, ref = _validate_language_job(job)
    subreddit, year = str(ref["subreddit"]), int(ref["year"])
    final, staging = language_output_dirs(
        contract=contract, run_id=run_id, subreddit=subreddit, year=year
    )
    unit_contract = {
        "schema_version": LANGUAGE_CONTRACT_SCHEMA_VERSION,
        "language_run_id": run_id,
        "cell": {"subreddit": subreddit, "year": year},
    }
    unit_id = _canonical_sha256(unit_contract)
    volume.reload()
    if final.exists():
        verified = _verify_language_output(
            contract=contract, run_id=run_id, subreddit=subreddit, year=year
        )
        receipt = verified["receipt"]
        return {
            "status": "already_complete",
            "unit_id": unit_id,
            "candidate_rows": receipt["outputs"]["decisions"]["rows"],
            "benchmark_rows": receipt["outputs"]["benchmark"]["rows"],
            **receipt["runtime"],
        }
    owner = modal.current_function_call_id()
    if not owner:
        raise RuntimeError("Modal did not provide a current function-call ID")
    acquisition = acquire_run_claim(
        claim_registry,
        run_manifest_id=unit_id,
        owner_call_id=owner,
        claimed_at=datetime.now(UTC).isoformat(),
    )
    try:
        volume.reload()
        if final.exists():
            verified = _verify_language_output(
                contract=contract, run_id=run_id, subreddit=subreddit, year=year
            )
            receipt = verified["receipt"]
            return {
                "status": "already_complete",
                "unit_id": unit_id,
                "candidate_rows": receipt["outputs"]["decisions"]["rows"],
                "benchmark_rows": receipt["outputs"]["benchmark"]["rows"],
                **receipt["runtime"],
            }
        if staging.exists():
            if not acquisition["same_owner_retry"]:
                raise FileExistsError(f"unclaimed language staging exists: {staging}")
            attempt = acquisition["claim"]["attempt"] - 1
            abandoned = staging.parent / f".abandoned-{unit_id}-attempt={attempt}"
            if abandoned.exists():
                raise FileExistsError(f"language abandoned staging already exists: {abandoned}")
            staging.replace(abandoned)
            volume.commit()

        job_started = time.monotonic()
        candidate_path = VOLUME_PATH / ref["candidate_output_relative_path"]
        if (
            not candidate_path.exists()
            or candidate_path.stat().st_size != ref["candidate_output_bytes"]
            or _sha256_file(candidate_path) != ref["candidate_output_sha256"]
        ):
            raise RuntimeError("candidate input does not match the frozen language contract")
        policy = contract["language_policy"]
        candidate_ids, sample = _cell_sample(
            candidate_path,
            subreddit=subreddit,
            year=year,
            quotas=contract["sample_quotas"],
            seed=policy["sampling"]["seed"],
        )
        if sum(len(values) for values in candidate_ids.values()) != ref["rows"]:
            raise RuntimeError("candidate input row count differs from its receipt")

        staging.mkdir(parents=True)
        decision_tmp = staging / "language-decisions.parquet.incomplete"
        benchmark_tmp = staging / "language-benchmark.parquet.incomplete"
        decision_writer = pq.ParquetWriter(decision_tmp, _decision_schema(), compression="zstd")
        benchmark_writer = pq.ParquetWriter(benchmark_tmp, _benchmark_schema(), compression="zstd")
        decision_buffer: list[dict[str, Any]] = []
        benchmark_buffer: list[dict[str, Any]] = []
        decision_counts: Counter[str] = Counter()
        benchmark_counts: Counter[str] = Counter()
        primary_seconds = 0.0
        challenger_seconds = 0.0

        def flush_decisions() -> None:
            if decision_buffer:
                decision_writer.write_table(
                    pa.Table.from_pylist(decision_buffer, schema=_decision_schema())
                )
                decision_buffer.clear()

        def flush_benchmark() -> None:
            if benchmark_buffer:
                benchmark_writer.write_table(
                    pa.Table.from_pylist(benchmark_buffer, schema=_benchmark_schema())
                )
                benchmark_buffer.clear()

        try:
            for content_type in CONTENT_TYPES:
                partition = ref["canonical_inputs"][content_type]
                parquet = _validate_partition_file(partition)
                remaining = candidate_ids[content_type]
                for batch in parquet.iter_batches(
                    batch_size=CANONICAL_BATCH_ROWS,
                    columns=["record_id", "text"],
                    use_threads=True,
                ):
                    data = batch.to_pydict()
                    matched: list[tuple[str, str]] = []
                    for record_id, text in zip(data["record_id"], data["text"], strict=True):
                        if record_id not in remaining:
                            continue
                        if not isinstance(text, str):
                            raise RuntimeError(
                                f"canonical candidate has non-string text: {record_id}"
                            )
                        remaining.remove(record_id)
                        matched.append((record_id, text))
                    for start in range(0, len(matched), DECISION_BATCH_ROWS):
                        chunk = matched[start : start + DECISION_BATCH_ROWS]
                        primary_started = time.monotonic()
                        predictions = _primary_predictions(
                            [text for _, text in chunk], top_k=int(policy["primary"]["top_k"])
                        )
                        primary_seconds += time.monotonic() - primary_started
                        for (record_id, text), prediction in zip(chunk, predictions, strict=True):
                            status = provisional_status(
                                prediction["language_code"],
                                prediction["confidence"],
                                high_confidence_threshold=float(
                                    policy["triage"]["high_confidence_threshold"]
                                ),
                            )
                            decision_buffer.append(
                                {
                                    "schema_version": LANGUAGE_OUTPUT_SCHEMA_VERSION,
                                    "record_id": record_id,
                                    "content_type": content_type,
                                    "primary_language_code": prediction["language_code"],
                                    "primary_confidence": prediction["confidence"],
                                    "primary_english_confidence": prediction["english_confidence"],
                                    "primary_top_predictions": prediction["top_predictions"],
                                    "provisional_status": status,
                                    "exclusion_allowed": False,
                                    "detector_id": policy["primary"]["detector_id"],
                                    "language_policy_digest": policy["policy_digest"],
                                    "stage_a_run_manifest_id": contract["stage_a_run_manifest_id"],
                                }
                            )
                            decision_counts[status] += 1
                            if record_id in sample:
                                key, rank = sample[record_id]
                                challenger_started = time.monotonic()
                                challenger = _challenger_prediction(text)
                                challenger_seconds += time.monotonic() - challenger_started
                                bucket = agreement_bucket(
                                    primary_code=prediction["language_code"],
                                    primary_confidence=prediction["confidence"],
                                    challenger_code=challenger["language_code"],
                                    challenger_confidence=challenger["confidence"],
                                    high_confidence_threshold=float(
                                        policy["triage"]["high_confidence_threshold"]
                                    ),
                                )
                                benchmark_buffer.append(
                                    {
                                        "schema_version": LANGUAGE_OUTPUT_SCHEMA_VERSION,
                                        "record_id": record_id,
                                        "content_type": content_type,
                                        "sample_stratum": key,
                                        "sample_rank_sha256": f"{rank:064x}",
                                        "inclusion_probability": contract["sample_quotas"][key]
                                        / contract["stratum_counts"][key],
                                        "text_length_chars": len(text),
                                        "primary_language_code": prediction["language_code"],
                                        "primary_confidence": prediction["confidence"],
                                        "primary_english_confidence": prediction[
                                            "english_confidence"
                                        ],
                                        "challenger_language_code": challenger["language_code"],
                                        "challenger_confidence": challenger["confidence"],
                                        "challenger_english_confidence": challenger[
                                            "english_confidence"
                                        ],
                                        "agreement_bucket": bucket,
                                        "primary_detector_id": policy["primary"]["detector_id"],
                                        "challenger_detector_id": policy["challenger"][
                                            "detector_id"
                                        ],
                                        "language_policy_digest": policy["policy_digest"],
                                        "stage_a_run_manifest_id": contract[
                                            "stage_a_run_manifest_id"
                                        ],
                                    }
                                )
                                benchmark_counts[bucket] += 1
                            if len(decision_buffer) >= DECISION_BATCH_ROWS:
                                flush_decisions()
                            if len(benchmark_buffer) >= DECISION_BATCH_ROWS:
                                flush_benchmark()
                if remaining:
                    raise RuntimeError(
                        f"canonical join missed {len(remaining)} {content_type} candidates"
                    )
            flush_decisions()
            flush_benchmark()
        finally:
            decision_writer.close()
            benchmark_writer.close()

        decision_rows = sum(decision_counts.values())
        benchmark_rows = sum(benchmark_counts.values())
        expected_benchmark_rows = sum(
            quota
            for key, quota in contract["sample_quotas"].items()
            if key.startswith(f"{subreddit}|{year}|")
        )
        if decision_rows != ref["rows"] or benchmark_rows != expected_benchmark_rows:
            raise RuntimeError("language outputs failed exact row conservation")
        decision_sha = _sha256_file(decision_tmp)
        benchmark_sha = _sha256_file(benchmark_tmp)
        decision_path = staging / f"language-decisions-{decision_sha}.parquet"
        benchmark_path = staging / f"language-benchmark-{benchmark_sha}.parquet"
        decision_tmp.replace(decision_path)
        benchmark_tmp.replace(benchmark_path)
        wall_seconds = time.monotonic() - job_started
        canonical_rows = sum(
            int(partition["partition_rows"]) for partition in ref["canonical_inputs"].values()
        )
        peak_rss_mib = _peak_rss_mib()
        requested_memory_mib = int(policy["compute"]["requested_memory_mib"])
        if peak_rss_mib > requested_memory_mib:
            raise RuntimeError("language cell exceeded its requested memory before commit")
        receipt = {
            "schema_version": LANGUAGE_OUTPUT_SCHEMA_VERSION,
            "status": "complete_pending_human_language_audit",
            "unit_contract": unit_contract,
            "inputs": ref,
            "model_manifest": {
                "primary": policy["primary"],
                "challenger": policy["challenger"],
            },
            "sampling": {
                "seed": policy["sampling"]["seed"],
                "stratify_by": policy["sampling"]["stratify_by"],
                "expected_rows": expected_benchmark_rows,
            },
            "counts": {
                "candidate_rows": decision_rows,
                "benchmark_rows": benchmark_rows,
                "provisional_status": dict(sorted(decision_counts.items())),
                "agreement_bucket": dict(sorted(benchmark_counts.items())),
            },
            "outputs": {
                "decisions": {
                    "file": decision_path.name,
                    "rows": decision_rows,
                    "bytes": decision_path.stat().st_size,
                    "sha256": decision_sha,
                    "schema_fields": list(language_output_field_names()),
                    "contains_raw_text": False,
                    "exclusion_allowed": False,
                },
                "benchmark": {
                    "file": benchmark_path.name,
                    "rows": benchmark_rows,
                    "bytes": benchmark_path.stat().st_size,
                    "sha256": benchmark_sha,
                    "schema_fields": list(benchmark_output_field_names()),
                    "contains_raw_text": False,
                },
            },
            "runtime": {
                "requested_cpus": policy["compute"]["requested_cpus"],
                "requested_memory_mib": policy["compute"]["requested_memory_mib"],
                "python_version": platform.python_version(),
                "pyarrow_version": pa.__version__,
                "primary_inference_seconds": round(primary_seconds, 3),
                "challenger_inference_seconds": round(challenger_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                "canonical_rows_scanned": canonical_rows,
                "canonical_rows_per_second": round(canonical_rows / wall_seconds, 3),
                "decision_rows_per_second": round(decision_rows / wall_seconds, 3),
                "peak_rss_mib": round(peak_rss_mib, 3),
                "memory_safe_completion": True,
            },
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _, receipt_sha = _write_json_receipt(staging, receipt)
        staging.replace(final)
        volume.commit()
        update_run_claim(
            claim_registry,
            run_manifest_id=unit_id,
            owner_call_id=owner,
            status="complete",
            updated_at=datetime.now(UTC).isoformat(),
            metadata={"receipt_sha256": receipt_sha},
        )
        return {
            "status": "complete_pending_human_language_audit",
            "unit_id": unit_id,
            "candidate_rows": decision_rows,
            "benchmark_rows": benchmark_rows,
            **receipt["runtime"],
        }
    except BaseException as exc:
        with suppress(Exception):
            update_run_claim(
                claim_registry,
                run_manifest_id=unit_id,
                owner_call_id=owner,
                status="failed",
                updated_at=datetime.now(UTC).isoformat(),
                metadata={"failure_type": type(exc).__name__},
            )
        raise


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=8192,
    timeout=14_400,
    max_containers=16,
)
def run_language_cell(job: dict[str, Any]) -> dict[str, Any]:
    return _execute_language(job)


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=8192,
    timeout=3_600,
    max_containers=1,
)
def validate_all_language_outputs(
    *, contract: dict[str, Any], language_run_manifest_id: str
) -> dict[str, Any]:
    volume.reload()
    if language_run_id(contract) != language_run_manifest_id:
        raise RuntimeError("language run ID does not match contract")
    decisions = 0
    benchmark = 0
    cell_refs: list[dict[str, Any]] = []
    for ref in contract["candidate_inputs"]:
        verified = _verify_language_output(
            contract=contract,
            run_id=language_run_manifest_id,
            subreddit=ref["subreddit"],
            year=ref["year"],
        )
        decisions += int(verified["receipt"]["outputs"]["decisions"]["rows"])
        benchmark += int(verified["receipt"]["outputs"]["benchmark"]["rows"])
        cell_refs.append(
            {
                "subreddit": ref["subreddit"],
                "year": ref["year"],
                "final_relative_path": verified["final_relative_path"],
                "receipt_relative_path": verified["receipt_relative_path"],
                "receipt_sha256": verified["receipt_sha256"],
                "decision_rows": verified["receipt"]["outputs"]["decisions"]["rows"],
                "decision_sha256": verified["receipt"]["outputs"]["decisions"]["sha256"],
                "benchmark_rows": verified["receipt"]["outputs"]["benchmark"]["rows"],
                "benchmark_sha256": verified["receipt"]["outputs"]["benchmark"]["sha256"],
            }
        )
    if decisions != contract["counts"]["candidate_rows"]:
        raise RuntimeError("validated language decisions do not conserve candidates")
    if benchmark != contract["counts"]["benchmark_rows"]:
        raise RuntimeError("validated benchmark outputs do not conserve sample rows")
    run_root = _language_root(contract, language_run_manifest_id)
    incomplete = sorted(
        str(path.relative_to(VOLUME_PATH)) for path in run_root.rglob(".unit=*.incomplete")
    )
    if incomplete:
        raise RuntimeError(f"language run has {len(incomplete)} incomplete outputs")
    reconciliation = {
        "schema_version": LANGUAGE_OUTPUT_SCHEMA_VERSION,
        "status": "technically_complete_pending_human_language_audit",
        "technical_complete": True,
        "language_gate_accepted": False,
        "eligible_provisional_statuses": [],
        "language_run_manifest_id": language_run_manifest_id,
        "stage_a_run_manifest_id": contract["stage_a_run_manifest_id"],
        "language_policy_digest": contract["language_policy"]["policy_digest"],
        "model_manifest": {
            "primary": contract["language_policy"]["primary"],
            "challenger": contract["language_policy"]["challenger"],
        },
        "counts": {
            "cells": len(cell_refs),
            "decision_rows": decisions,
            "benchmark_rows": benchmark,
            "incomplete_outputs": 0,
        },
        "cells": sorted(cell_refs, key=lambda row: (row["subreddit"], row["year"])),
    }
    reconciliation_payload = json.dumps(reconciliation, indent=2, sort_keys=True) + "\n"
    reconciliation_sha = hashlib.sha256(reconciliation_payload.encode()).hexdigest()
    reconciliation_dir = run_root / "reconciliation"
    reconciliation_path = reconciliation_dir / f"reconciliation-{reconciliation_sha}.json"
    if reconciliation_dir.exists():
        existing = list(reconciliation_dir.iterdir())
        if (
            len(existing) != 1
            or existing[0] != reconciliation_path
            or _sha256_file(reconciliation_path) != reconciliation_sha
        ):
            raise RuntimeError("language reconciliation directory is not one exact immutable file")
    else:
        reconciliation_dir.mkdir(parents=True)
        reconciliation_path.write_text(reconciliation_payload, encoding="utf-8")
        volume.commit()
    return {
        "status": reconciliation["status"],
        "receipts": 60,
        "decision_rows": decisions,
        "benchmark_rows": benchmark,
        "exclusion_allowed": False,
        "language_gate_accepted": False,
        "reconciliation_relative_path": str(reconciliation_path.relative_to(VOLUME_PATH)),
        "reconciliation_sha256": reconciliation_sha,
    }


def _prepare_language_contract(
    *, manifest_path: str, retrieval_policy_path: str, language_policy_path: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    from reddit_china_stance.retrieval import load_retrieval_policy
    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest_path = root / manifest_path
    resolved_retrieval_policy_path = root / retrieval_policy_path
    manifest = load_source_manifest(resolved_manifest_path)
    retrieval_policy = load_retrieval_policy(resolved_retrieval_policy_path)
    stage_a_contract = resolve_stage_a_contract_for_language.remote(
        manifest=manifest,
        retrieval_policy=retrieval_policy,
        code_state=_stage_a_code_state(
            root, resolved_manifest_path, resolved_retrieval_policy_path
        ),
    )
    stage_a_run_manifest_id = production_run_id(stage_a_contract)
    if stage_a_run_manifest_id != FROZEN_STAGE_A_RUN_ID:
        raise RuntimeError(
            "computed Stage A run ID does not match the frozen completed production run"
        )
    resolved_policy_path = root / language_policy_path
    policy = load_language_policy(resolved_policy_path)
    code_state = _code_state(root, resolved_policy_path)
    contract = resolve_language_contract.remote(
        stage_a_contract=stage_a_contract,
        stage_a_run_manifest_id=stage_a_run_manifest_id,
        policy=policy,
        code_state=code_state,
    )
    return contract, policy


@app.local_entrypoint()
def main(
    phase: str = "plan",
    manifest_path: str = "configs/source-files.json",
    retrieval_policy_path: str = "configs/retrieval-policy-v3.toml",
    language_policy_path: str = "configs/language-policy-v1.toml",
    approved_cost_usd: str = "25",
    confirm: str = "",
    smoke_subreddit: str = "ChineseLanguage",
    smoke_year: int = 2025,
) -> None:
    """Plan, smoke, submit, or validate the exact combined language pass."""

    if phase not in {"plan", "smoke", "run", "validate"}:
        raise ValueError("phase must be exactly plan, smoke, run, or validate")
    contract, policy = _prepare_language_contract(
        manifest_path=manifest_path,
        retrieval_policy_path=retrieval_policy_path,
        language_policy_path=language_policy_path,
    )
    run_id = language_run_id(contract)
    estimated = Decimal(contract["compute_plan"]["estimated_full_run_cost_usd"])
    approved = Decimal(approved_cost_usd)
    if phase != "smoke":
        enforce_cost_guardrail(
            estimated_cost_usd=estimated,
            approved_usd=approved,
            policy=policy,
        )
    summary: dict[str, Any] = {
        "status": "planned" if phase == "plan" else "submitted",
        "phase": phase,
        "stage_a_run_manifest_id": contract["stage_a_run_manifest_id"],
        "language_run_manifest_id": run_id,
        "counts": contract["counts"],
        "compute_plan": contract["compute_plan"],
        "exclusion_allowed": False,
    }
    if phase == "plan":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if phase == "smoke":
        if confirm != SMOKE_CONFIRMATION:
            raise ValueError(f"refusing language smoke: pass --confirm {SMOKE_CONFIRMATION}")
        matches = [
            ref
            for ref in contract["candidate_inputs"]
            if ref["subreddit"] == smoke_subreddit and ref["year"] == smoke_year
        ]
        if len(matches) != 1:
            raise ValueError("language smoke must resolve exactly one frozen production cell")
        smoke_estimated = estimate_cell_cost_usd(
            ref=matches[0],
            sample_quotas=contract["sample_quotas"],
            policy=policy,
        )
        enforce_smoke_cost_guardrail(estimated_cost_usd=smoke_estimated, approved_usd=approved)
        summary["smoke"] = {
            "cell": {"subreddit": smoke_subreddit, "year": smoke_year},
            "estimated_cost_usd": str(smoke_estimated),
            "hard_max_cost_usd": str(SMOKE_HARD_MAX_COST_USD),
            "result": run_language_cell.remote(
                {
                    "contract": contract,
                    "language_run_id": run_id,
                    "cell": {"subreddit": smoke_subreddit, "year": smoke_year},
                }
            ),
        }
        summary["status"] = "smoke_complete"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if phase == "validate":
        summary["validation"] = validate_all_language_outputs.remote(
            contract=contract, language_run_manifest_id=run_id
        )
        summary["status"] = "validated"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if confirm != CONFIRMATION:
        raise ValueError(f"refusing language launch: pass --confirm {CONFIRMATION}")
    jobs = [
        {
            "contract": contract,
            "language_run_id": run_id,
            "cell": {"subreddit": ref["subreddit"], "year": ref["year"]},
        }
        for ref in contract["candidate_inputs"]
    ]
    summary["submitted_function_call_ids"] = [
        run_language_cell.spawn(job).object_id for job in jobs
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
