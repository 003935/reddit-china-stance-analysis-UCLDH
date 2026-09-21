"""Production, two-wave Modal execution for frozen Stage A retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import modal

from reddit_china_stance.modal_parquet import SOURCE_SCHEMA_VERSION
from reddit_china_stance.modal_stage_a_proof import (
    ANCHOR_YEARS,
    VOLUME_NAME,
    VOLUME_PATH,
    _assemble_candidates,
    _candidate_arrow_schema,
    _candidate_rows,
    _canonical_sha256,
    _count_output,
    _row_group_chunks,
    _scan_row_groups,
    _sha256_file,
    acquire_run_claim,
    candidate_output_field_names,
    update_run_claim,
)
from reddit_china_stance.retrieval import RetrievalPolicy

APP_NAME = "reddit-china-stance-stage-a"
ENVIRONMENT_NAME = "main"
CLAIM_REGISTRY_NAME = "reddit-china-stance-stage-a-claims"
PRODUCTION_PREFIX = Path("derived") / "stage-a"
PRODUCTION_SCHEMA_VERSION = "1.0.0"
ANCHOR_SCHEMA_VERSION = "1.0.0"
CONFIRM_ANCHORS = "LAUNCH_ALL_120_STAGE_A_ANCHORS"
CONFIRM_CANDIDATES = "LAUNCH_ALL_60_STAGE_A_CANDIDATES"
CONFIRM_ANCHOR_SMOKE = "RUN_ONE_STAGE_A_ANCHOR_SMOKE"
MIN_YEAR = 2020
MAX_YEAR = 2025
SUBREDDITS = (
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
CONTENT_TYPES = ("submission", "comment")
ANCHOR_WORKERS = 8
MAX_CONTAINERS = 16
ANCHOR_BATCH_ROWS = 50_000
CANDIDATE_BATCH_ROWS = 20_000
CPU_HOUR_COST = Decimal("0.04730")
MEM_GIB_HOUR_COST = Decimal("0.00800")
REQUESTED_CPUS = Decimal("8")
REQUESTED_MEMORY_GIB = Decimal("32")
CONSERVATIVE_ROWS_PER_SECOND = Decimal("15000")
COST_OVERHEAD_FACTOR = Decimal("1.5")
HARD_MAX_ESTIMATED_COST_USD = Decimal("20")

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


def estimate_production_cost_usd(*, retained_rows: int, comment_rows: int) -> Decimal:
    """Conservatively price one anchor pass plus one relation-only comment pass."""

    for name, value in (("retained_rows", retained_rows), ("comment_rows", comment_rows)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    row_visits = Decimal(retained_rows + comment_rows)
    hours = row_visits / CONSERVATIVE_ROWS_PER_SECOND / Decimal(3600)
    hourly = REQUESTED_CPUS * CPU_HOUR_COST + REQUESTED_MEMORY_GIB * MEM_GIB_HOUR_COST
    return (hours * hourly * COST_OVERHEAD_FACTOR).quantize(Decimal("0.01"))


def enforce_cost_guardrail(*, estimated_cost_usd: Decimal, approved_usd: Decimal) -> None:
    """Fail locally before either durable wave if its full-run estimate is not approved."""

    if approved_usd <= 0 or approved_usd > HARD_MAX_ESTIMATED_COST_USD:
        raise ValueError(f"approved_usd must be > 0 and <= {HARD_MAX_ESTIMATED_COST_USD}")
    if estimated_cost_usd > approved_usd:
        raise RuntimeError(
            f"estimated Stage A cost ${estimated_cost_usd} exceeds approved ${approved_usd}"
        )


def _code_state(root: Path, manifest_path: Path, policy_path: Path) -> dict[str, Any]:
    tracked = (
        root / "src/reddit_china_stance/modal_stage_a.py",
        root / "src/reddit_china_stance/modal_stage_a_proof.py",
        root / "src/reddit_china_stance/retrieval.py",
        manifest_path,
        policy_path,
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


def _source_dir(*, revision: str, source_file: str) -> Path:
    return (
        VOLUME_PATH
        / "normalised"
        / revision
        / f"schema={SOURCE_SCHEMA_VERSION}"
        / Path(source_file).stem
    )


def _receipt_partition_metadata(
    *, manifest: Mapping[str, Any], source: Mapping[str, Any], year: int
) -> dict[str, Any]:
    source_file = str(source["path"])
    content_type = "submission" if source_file.endswith("_submissions.zst") else "comment"
    subreddit = source_file.removesuffix(f"_{content_type}s.zst")
    source_dir = _source_dir(revision=str(manifest["revision"]), source_file=source_file)
    receipt_path = source_dir / "_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"canonical receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "status": "converted",
        "dataset_id": manifest["dataset_id"],
        "revision": manifest["revision"],
        "source_file": source_file,
        "source_compressed_bytes": source["size"],
        "source_sha256": source["sha256"],
        "source_schema_version": SOURCE_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"canonical receipt mismatch for {source_file}: {key}")
    expected_relative = (
        Path(f"year={year}")
        / f"subreddit={subreddit}"
        / f"content_type={content_type}"
        / "part-00000.parquet"
    )
    partitions = [row for row in receipt.get("partitions", []) if row.get("year") == year]
    if len(partitions) != 1 or partitions[0].get("relative_path") != str(expected_relative):
        raise RuntimeError(f"canonical receipt lacks exact partition: {source_file}/{year}")
    partition = partitions[0]
    return {
        "subreddit": subreddit,
        "year": year,
        "content_type": content_type,
        "source_file": source_file,
        "source_sha256": source["sha256"],
        "partition_relative_path": str((source_dir / expected_relative).relative_to(VOLUME_PATH)),
        "partition_rows": int(partition["rows"]),
        "partition_bytes": int(partition["bytes"]),
        "partition_sha256": partition["sha256"],
        "conversion_code_sha256": receipt["code_sha256"],
    }


def make_production_contract(
    *,
    manifest: Mapping[str, Any],
    policy: RetrievalPolicy,
    code_state: Mapping[str, Any],
    input_partitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the complete 120-partition production invocation."""

    partitions = [dict(row) for row in input_partitions]
    expected = {
        (subreddit, year, content_type)
        for subreddit in SUBREDDITS
        for year in ANCHOR_YEARS
        for content_type in CONTENT_TYPES
    }
    observed = {
        (str(row.get("subreddit")), row.get("year"), str(row.get("content_type")))
        for row in partitions
    }
    if observed != expected or len(partitions) != len(expected):
        raise ValueError("production contract must pin exactly 120 unique canonical partitions")
    partitions.sort(key=lambda row: (row["subreddit"], row["year"], row["content_type"]))
    retained_rows = sum(int(row["partition_rows"]) for row in partitions)
    comment_rows = sum(
        int(row["partition_rows"]) for row in partitions if row["content_type"] == "comment"
    )
    estimated_cost = estimate_production_cost_usd(
        retained_rows=retained_rows, comment_rows=comment_rows
    )
    return {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "dataset_id": manifest["dataset_id"],
        "revision": manifest["revision"],
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "retrieval_policy": {
            "schema_version": policy.schema_version,
            "policy_version": policy.policy_version,
            "policy_digest": policy.policy_digest,
            "lexicon_version": policy.lexicon_version,
        },
        "input_partitions": partitions,
        "counts": {
            "partitions": len(partitions),
            "retained_rows": retained_rows,
            "comment_rows": comment_rows,
            "planned_row_visits": retained_rows + comment_rows,
        },
        "compute_plan": {
            "requested_cpus": int(REQUESTED_CPUS),
            "requested_memory_gib": int(REQUESTED_MEMORY_GIB),
            "max_containers": MAX_CONTAINERS,
            "conservative_rows_per_second": int(CONSERVATIVE_ROWS_PER_SECOND),
            "overhead_factor": str(COST_OVERHEAD_FACTOR),
            "cpu_hour_cost_usd": str(CPU_HOUR_COST),
            "memory_gib_hour_cost_usd": str(MEM_GIB_HOUR_COST),
            "estimated_full_run_cost_usd": str(estimated_cost),
            "hard_max_estimated_cost_usd": str(HARD_MAX_ESTIMATED_COST_USD),
        },
        "code_state": dict(code_state),
    }


def production_run_id(contract: Mapping[str, Any]) -> str:
    return _canonical_sha256(contract)


def _unit_contract(
    *, run_manifest_id: str, phase: Literal["anchor", "candidate"], cell: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "run_manifest_id": run_manifest_id,
        "phase": phase,
        "cell": dict(cell),
    }


def _unit_id(unit_contract: Mapping[str, Any]) -> str:
    return _canonical_sha256(unit_contract)


def _run_root(contract: Mapping[str, Any], run_manifest_id: str) -> Path:
    return (
        VOLUME_PATH
        / PRODUCTION_PREFIX
        / str(contract["revision"])
        / f"source-schema={SOURCE_SCHEMA_VERSION}"
        / f"policy={contract['retrieval_policy']['policy_digest']}"
        / f"run={run_manifest_id}"
    )


def anchor_output_dirs(
    *, contract: Mapping[str, Any], run_manifest_id: str, partition: Mapping[str, Any]
) -> tuple[Path, Path]:
    unit = _unit_contract(
        run_manifest_id=run_manifest_id,
        phase="anchor",
        cell={key: partition[key] for key in ("subreddit", "year", "content_type")},
    )
    root = (
        _run_root(contract, run_manifest_id)
        / "anchors"
        / f"subreddit={partition['subreddit']}"
        / f"year={partition['year']}"
        / f"content_type={partition['content_type']}"
    )
    final = root / f"unit={_unit_id(unit)}"
    return final, root / f".unit={_unit_id(unit)}.incomplete"


def _candidate_output_dirs(
    *, contract: Mapping[str, Any], run_manifest_id: str, subreddit: str, year: int
) -> tuple[Path, Path]:
    unit = _unit_contract(
        run_manifest_id=run_manifest_id,
        phase="candidate",
        cell={"subreddit": subreddit, "year": year},
    )
    root = (
        _run_root(contract, run_manifest_id)
        / "candidates"
        / f"subreddit={subreddit}"
        / f"year={year}"
    )
    final = root / f"unit={_unit_id(unit)}"
    return final, root / f".unit={_unit_id(unit)}.incomplete"


def _anchor_arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema([("record_id", pa.string()), ("match_term_ids", pa.list_(pa.string()))])


def _validate_partition_file(partition: Mapping[str, Any]) -> Any:
    import pyarrow.parquet as pq

    path = VOLUME_PATH / str(partition["partition_relative_path"])
    if not path.exists() or path.stat().st_size != int(partition["partition_bytes"]):
        raise RuntimeError(f"canonical partition size mismatch: {path}")
    if _sha256_file(path) != partition["partition_sha256"]:
        raise RuntimeError(f"canonical partition digest mismatch: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != int(partition["partition_rows"]):
        raise RuntimeError(f"canonical partition row mismatch: {path}")
    required = {
        "record_id",
        "content_type",
        "subreddit",
        "year",
        "text",
        "submission_id",
        "parent_id",
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise RuntimeError(f"canonical partition schema mismatch: {path}")
    return parquet


def _write_json_receipt(staging: Path, receipt: Mapping[str, Any]) -> tuple[Path, str]:
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    path = staging / f"receipt-{digest}.json"
    path.write_text(payload, encoding="utf-8")
    return path, digest


def _verify_pair(
    final_dir: Path, *, receipt_contract: Mapping[str, Any], parquet_prefix: str
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    receipts = list(final_dir.glob("receipt-*.json"))
    outputs = list(final_dir.glob(f"{parquet_prefix}-*.parquet"))
    if len(receipts) != 1 or len(outputs) != 1 or len(list(final_dir.iterdir())) != 2:
        raise RuntimeError(f"immutable unit is not an exact receipt/output pair: {final_dir}")
    receipt_path, output_path = receipts[0], outputs[0]
    receipt_sha = _sha256_file(receipt_path)
    output_sha = _sha256_file(output_path)
    if receipt_path.name != f"receipt-{receipt_sha}.json":
        raise RuntimeError("unit receipt is not content-addressed")
    if output_path.name != f"{parquet_prefix}-{output_sha}.parquet":
        raise RuntimeError("unit Parquet output is not content-addressed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("unit_contract") != receipt_contract:
        raise RuntimeError("unit receipt contract mismatch")
    output = receipt.get("output", {})
    if (
        output.get("sha256") != output_sha
        or output.get("bytes") != output_path.stat().st_size
        or output.get("rows") != pq.ParquetFile(output_path).metadata.num_rows
    ):
        raise RuntimeError("unit output does not match receipt")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha,
        "output_path": output_path,
        "output_sha256": output_sha,
    }


def verify_anchor_output(
    *, contract: Mapping[str, Any], run_manifest_id: str, partition: Mapping[str, Any]
) -> dict[str, Any]:
    unit = _unit_contract(
        run_manifest_id=run_manifest_id,
        phase="anchor",
        cell={key: partition[key] for key in ("subreddit", "year", "content_type")},
    )
    final, _ = anchor_output_dirs(
        contract=contract, run_manifest_id=run_manifest_id, partition=partition
    )
    result = _verify_pair(final, receipt_contract=unit, parquet_prefix="anchors")
    if result["receipt"].get("input") != dict(partition):
        raise RuntimeError("anchor receipt input partition mismatch")
    return result


def _claim_unit(*, unit_id: str) -> tuple[str, dict[str, Any]]:
    owner = modal.current_function_call_id()
    if not owner:
        raise RuntimeError("Modal did not provide a current function-call ID")
    acquisition = acquire_run_claim(
        claim_registry,
        run_manifest_id=unit_id,
        owner_call_id=owner,
        claimed_at=datetime.now(UTC).isoformat(),
    )
    return owner, acquisition


def _validate_job(job: Mapping[str, Any], *, phase: str) -> tuple[dict[str, Any], str]:
    if set(job) != {"contract", "run_manifest_id", "cell"}:
        raise ValueError(f"{phase} job has unexpected fields")
    contract = dict(job["contract"])
    run_manifest_id = str(job["run_manifest_id"])
    if production_run_id(contract) != run_manifest_id:
        raise RuntimeError("production run ID does not match contract")
    return contract, run_manifest_id


def _validate_policy_contract(contract: Mapping[str, Any], policy: RetrievalPolicy) -> None:
    expected = contract["retrieval_policy"]
    observed = {
        "schema_version": policy.schema_version,
        "policy_version": policy.policy_version,
        "policy_digest": policy.policy_digest,
        "lexicon_version": policy.lexicon_version,
    }
    if observed != expected:
        raise RuntimeError("loaded retrieval policy does not match production contract")


def _execute_anchor(job: Mapping[str, Any], policy: RetrievalPolicy) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract, run_manifest_id = _validate_job(job, phase="anchor")
    _validate_policy_contract(contract, policy)
    partition = dict(job["cell"])
    if partition not in contract["input_partitions"]:
        raise RuntimeError("anchor cell is not an exact pinned input partition")
    unit_contract = _unit_contract(
        run_manifest_id=run_manifest_id,
        phase="anchor",
        cell={key: partition[key] for key in ("subreddit", "year", "content_type")},
    )
    unit_id = _unit_id(unit_contract)
    final_dir, staging_dir = anchor_output_dirs(
        contract=contract, run_manifest_id=run_manifest_id, partition=partition
    )
    volume.reload()
    if final_dir.exists():
        verified = verify_anchor_output(
            contract=contract, run_manifest_id=run_manifest_id, partition=partition
        )
        return {
            "status": "already_complete",
            "unit_id": unit_id,
            "rows": verified["receipt"]["output"]["rows"],
        }
    owner, acquisition = _claim_unit(unit_id=unit_id)
    try:
        volume.reload()
        if final_dir.exists():
            verified = verify_anchor_output(
                contract=contract, run_manifest_id=run_manifest_id, partition=partition
            )
            return {
                "status": "already_complete",
                "unit_id": unit_id,
                "rows": verified["receipt"]["output"]["rows"],
            }
        if staging_dir.exists():
            if not acquisition["same_owner_retry"]:
                raise FileExistsError(f"unclaimed anchor staging exists: {staging_dir}")
            previous_attempt = acquisition["claim"]["attempt"] - 1
            abandoned = staging_dir.parent / (f".abandoned-{unit_id}-attempt={previous_attempt}")
            if abandoned.exists():
                raise FileExistsError(f"anchor abandoned staging already exists: {abandoned}")
            staging_dir.replace(abandoned)
            volume.commit()

        started = time.monotonic()
        parquet = _validate_partition_file(partition)
        path = VOLUME_PATH / partition["partition_relative_path"]
        tasks = [
            {
                "path": str(path),
                "content_type": partition["content_type"],
                "row_groups": row_groups,
                "subreddit": partition["subreddit"],
                "year": partition["year"],
                "target_year": -1 if partition["content_type"] == "comment" else partition["year"],
                "policy": policy,
            }
            for row_groups in _row_group_chunks(parquet.metadata.num_row_groups, ANCHOR_WORKERS)
        ]
        anchors: list[tuple[str, tuple[str, ...]]] = []
        scanned_rows = 0
        workers = min(ANCHOR_WORKERS, len(tasks), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_scan_row_groups, **task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                scanned_rows += int(result["rows"])
                anchors.extend((row[0], tuple(row[-1])) for row in result["results"])
        if scanned_rows != partition["partition_rows"]:
            raise RuntimeError("anchor scan did not conserve input rows")
        anchors.sort()
        if len(anchors) != len({record_id for record_id, _ in anchors}):
            raise RuntimeError("duplicate record ID in anchor shard")

        staging_dir.mkdir(parents=True)
        temporary = staging_dir / "anchors.parquet.incomplete"
        writer = pq.ParquetWriter(temporary, _anchor_arrow_schema(), compression="zstd")
        try:
            for start in range(0, len(anchors), ANCHOR_BATCH_ROWS):
                rows = [
                    {"record_id": record_id, "match_term_ids": list(term_ids)}
                    for record_id, term_ids in anchors[start : start + ANCHOR_BATCH_ROWS]
                ]
                writer.write_table(pa.Table.from_pylist(rows, schema=_anchor_arrow_schema()))
        finally:
            writer.close()
        output_sha = _sha256_file(temporary)
        output_path = staging_dir / f"anchors-{output_sha}.parquet"
        temporary.replace(output_path)
        receipt = {
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "status": "complete",
            "unit_contract": unit_contract,
            "input": partition,
            "output": {
                "file": output_path.name,
                "rows": len(anchors),
                "bytes": output_path.stat().st_size,
                "sha256": output_sha,
                "schema_fields": ["record_id", "match_term_ids"],
                "contains_raw_text": False,
            },
            "scanned_rows": scanned_rows,
            "runtime": {
                "requested_cpus": int(REQUESTED_CPUS),
                "worker_processes": workers,
                "python_version": platform.python_version(),
                "pyarrow_version": pa.__version__,
            },
            "wall_seconds": round(time.monotonic() - started, 3),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _, receipt_sha = _write_json_receipt(staging_dir, receipt)
        staging_dir.replace(final_dir)
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
            "status": "complete",
            "unit_id": unit_id,
            "anchor_rows": len(anchors),
            "scanned_rows": scanned_rows,
            "wall_seconds": receipt["wall_seconds"],
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
    cpu=8.0,
    memory=32_768,
    timeout=14_400,
    max_containers=MAX_CONTAINERS,
)
def run_anchor(job: dict[str, Any]) -> dict[str, Any]:
    policy = job.get("policy")
    if not isinstance(policy, RetrievalPolicy):
        raise ValueError("anchor job requires a validated RetrievalPolicy")
    payload = {key: value for key, value in job.items() if key != "policy"}
    return _execute_anchor(payload, policy)


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=1.0,
    memory=2048,
    timeout=3_600,
    max_containers=1,
)
def resolve_production_contract(
    *, manifest: dict[str, Any], policy: RetrievalPolicy, code_state: dict[str, Any]
) -> dict[str, Any]:
    """Read exactly 20 canonical receipts and freeze all 120 partition references."""

    volume.reload()
    sources = sorted(manifest["files"], key=lambda row: row["path"])
    partitions = [
        _receipt_partition_metadata(manifest=manifest, source=source, year=year)
        for source in sources
        for year in ANCHOR_YEARS
    ]
    return make_production_contract(
        manifest=manifest,
        policy=policy,
        code_state=code_state,
        input_partitions=partitions,
    )


def _load_anchor_map(paths: Sequence[Path]) -> dict[str, tuple[str, ...]]:
    import pyarrow.parquet as pq

    result: dict[str, tuple[str, ...]] = {}
    for path in paths:
        table = pq.read_table(path, columns=["record_id", "match_term_ids"])
        data = table.to_pydict()
        for record_id, term_ids in zip(data["record_id"], data["match_term_ids"], strict=True):
            if record_id in result:
                raise RuntimeError(f"duplicate cross-year anchor record ID: {record_id}")
            result[record_id] = tuple(term_ids)
    return result


def _merge_counts(total: Counter[str], batch: Mapping[str, Any]) -> None:
    total["candidate_rows"] += int(batch["candidate_rows"])
    total["selection_evidence_paths"] += int(batch["selection_evidence_paths"])
    total["multi_channel_candidate_rows"] += int(batch["multi_channel_candidate_rows"])
    for prefix in (
        "candidate_rows_by_content_type",
        "candidate_rows_by_channel",
        "selection_evidence_paths_by_channel",
    ):
        for key, value in batch[prefix].items():
            total[f"{prefix}.{key}"] += int(value)


def _counts_payload(total: Counter[str]) -> dict[str, Any]:
    return {
        "candidate_rows": total["candidate_rows"],
        "candidate_rows_by_content_type": {
            key: total[f"candidate_rows_by_content_type.{key}"] for key in CONTENT_TYPES
        },
        "candidate_rows_by_channel": {
            key: total[f"candidate_rows_by_channel.{key}"]
            for key in ("direct_lexical", "submission_expansion", "direct_reply_expansion")
        },
        "selection_evidence_paths_by_channel": {
            key: total[f"selection_evidence_paths_by_channel.{key}"]
            for key in ("direct_lexical", "submission_expansion", "direct_reply_expansion")
        },
        "selection_evidence_paths": total["selection_evidence_paths"],
        "multi_channel_candidate_rows": total["multi_channel_candidate_rows"],
    }


def _execute_candidate(job: Mapping[str, Any], policy: RetrievalPolicy) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract, run_manifest_id = _validate_job(job, phase="candidate")
    _validate_policy_contract(contract, policy)
    cell = dict(job["cell"])
    subreddit, year = str(cell["subreddit"]), int(cell["year"])
    if subreddit not in SUBREDDITS or year not in ANCHOR_YEARS:
        raise ValueError("candidate cell is outside the frozen production grid")
    unit_contract = _unit_contract(
        run_manifest_id=run_manifest_id,
        phase="candidate",
        cell={"subreddit": subreddit, "year": year},
    )
    unit_id = _unit_id(unit_contract)
    final_dir, staging_dir = _candidate_output_dirs(
        contract=contract, run_manifest_id=run_manifest_id, subreddit=subreddit, year=year
    )
    volume.reload()
    if final_dir.exists():
        verified = _verify_pair(
            final_dir,
            receipt_contract=unit_contract,
            parquet_prefix="candidates",
        )
        return {
            "status": "already_complete",
            "unit_id": unit_id,
            "rows": verified["receipt"]["output"]["rows"],
        }
    owner, acquisition = _claim_unit(unit_id=unit_id)
    try:
        volume.reload()
        if staging_dir.exists():
            if not acquisition["same_owner_retry"]:
                raise FileExistsError(f"unclaimed candidate staging exists: {staging_dir}")
            previous_attempt = acquisition["claim"]["attempt"] - 1
            abandoned = staging_dir.parent / (f".abandoned-{unit_id}-attempt={previous_attempt}")
            if abandoned.exists():
                raise FileExistsError(f"candidate abandoned staging already exists: {abandoned}")
            staging_dir.replace(abandoned)
            volume.commit()

        partitions = {
            (row["year"], row["content_type"]): row
            for row in contract["input_partitions"]
            if row["subreddit"] == subreddit
        }
        if set(partitions) != {
            (anchor_year, content_type)
            for anchor_year in ANCHOR_YEARS
            for content_type in CONTENT_TYPES
        }:
            raise RuntimeError("candidate cell lacks exact all-year canonical inputs")

        anchor_refs: list[dict[str, Any]] = []
        anchor_paths: dict[str, list[Path]] = {content_type: [] for content_type in CONTENT_TYPES}
        for content_type in CONTENT_TYPES:
            for anchor_year in ANCHOR_YEARS:
                partition = partitions[(anchor_year, content_type)]
                verified = verify_anchor_output(
                    contract=contract,
                    run_manifest_id=run_manifest_id,
                    partition=partition,
                )
                anchor_paths[content_type].append(verified["output_path"])
                anchor_refs.append(
                    {
                        "subreddit": subreddit,
                        "year": anchor_year,
                        "content_type": content_type,
                        "receipt_sha256": verified["receipt_sha256"],
                        "output_sha256": verified["output_sha256"],
                    }
                )
        direct_submissions = _load_anchor_map(anchor_paths["submission"])
        direct_comments = _load_anchor_map(anchor_paths["comment"])

        target_submission_verified = verify_anchor_output(
            contract=contract,
            run_manifest_id=run_manifest_id,
            partition=partitions[(year, "submission")],
        )
        target_submission_ids = set(
            pq.read_table(target_submission_verified["output_path"], columns=["record_id"])
            .column("record_id")
            .to_pylist()
        )
        submission_candidates = _assemble_candidates(
            direct_submissions=direct_submissions,
            target_direct_submission_ids=frozenset(target_submission_ids),
            direct_comments=direct_comments,
            comments=[],
            policy=policy,
            manifest_id=run_manifest_id,
        )

        comment_partition = partitions[(year, "comment")]
        comment_parquet = _validate_partition_file(comment_partition)
        staging_dir.mkdir(parents=True)
        temporary = staging_dir / "candidates.parquet.incomplete"
        writer = pq.ParquetWriter(temporary, _candidate_arrow_schema(), compression="zstd")
        totals: Counter[str] = Counter()
        scanned_comments = 0
        try:
            if submission_candidates:
                writer.write_table(
                    pa.Table.from_pylist(
                        _candidate_rows(submission_candidates), schema=_candidate_arrow_schema()
                    )
                )
                _merge_counts(totals, _count_output(submission_candidates))
            for batch in comment_parquet.iter_batches(
                batch_size=CANDIDATE_BATCH_ROWS,
                columns=["record_id", "submission_id", "parent_id"],
                use_threads=True,
            ):
                data = batch.to_pydict()
                rows: list[tuple[str, str, str | None, tuple[str, ...]]] = []
                for record_id, submission_id, parent_id in zip(
                    data["record_id"],
                    data["submission_id"],
                    data["parent_id"],
                    strict=True,
                ):
                    if not isinstance(record_id, str) or not record_id:
                        raise RuntimeError("canonical comment has invalid record_id")
                    if not isinstance(submission_id, str) or not submission_id:
                        raise RuntimeError(
                            f"canonical comment has invalid submission_id: {record_id}"
                        )
                    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
                        raise RuntimeError(f"canonical comment has invalid parent_id: {record_id}")
                    rows.append(
                        (
                            record_id,
                            submission_id,
                            parent_id,
                            direct_comments.get(record_id, ()),
                        )
                    )
                candidates = _assemble_candidates(
                    direct_submissions=direct_submissions,
                    target_direct_submission_ids=frozenset(),
                    direct_comments=direct_comments,
                    comments=rows,
                    policy=policy,
                    manifest_id=run_manifest_id,
                )
                if candidates:
                    writer.write_table(
                        pa.Table.from_pylist(
                            _candidate_rows(candidates), schema=_candidate_arrow_schema()
                        )
                    )
                    _merge_counts(totals, _count_output(candidates))
                scanned_comments += batch.num_rows
        finally:
            writer.close()
        if scanned_comments != comment_partition["partition_rows"]:
            raise RuntimeError("candidate join did not conserve target comment rows")
        output_sha = _sha256_file(temporary)
        output_path = staging_dir / f"candidates-{output_sha}.parquet"
        temporary.replace(output_path)
        counts = _counts_payload(totals)
        receipt = {
            "schema_version": PRODUCTION_SCHEMA_VERSION,
            "status": "complete",
            "unit_contract": unit_contract,
            "target_inputs": {
                "submission": partitions[(year, "submission")],
                "comment": comment_partition,
            },
            "anchor_inputs": sorted(
                anchor_refs,
                key=lambda row: (row["year"], row["content_type"]),
            ),
            "counts": {
                "scanned_comment_rows": scanned_comments,
                "all_year_direct_submission_anchors": len(direct_submissions),
                "all_year_direct_comment_anchors": len(direct_comments),
                **counts,
            },
            "output": {
                "file": output_path.name,
                "rows": counts["candidate_rows"],
                "bytes": output_path.stat().st_size,
                "sha256": output_sha,
                "schema_fields": list(candidate_output_field_names()),
                "contains_raw_text": False,
                "language_status": "unclassified",
            },
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _, receipt_sha = _write_json_receipt(staging_dir, receipt)
        staging_dir.replace(final_dir)
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
            "status": "complete",
            "unit_id": unit_id,
            "candidate_rows": counts["candidate_rows"],
            "scanned_comment_rows": scanned_comments,
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
    cpu=8.0,
    memory=32_768,
    timeout=14_400,
    max_containers=MAX_CONTAINERS,
)
def run_candidate(job: dict[str, Any]) -> dict[str, Any]:
    policy = job.get("policy")
    if not isinstance(policy, RetrievalPolicy):
        raise ValueError("candidate job requires a validated RetrievalPolicy")
    payload = {key: value for key, value in job.items() if key != "policy"}
    return _execute_candidate(payload, policy)


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=4096,
    timeout=3_600,
    max_containers=1,
)
def validate_all_anchors(*, contract: dict[str, Any], run_manifest_id: str) -> dict[str, Any]:
    volume.reload()
    if production_run_id(contract) != run_manifest_id:
        raise RuntimeError("production run ID does not match contract")
    rows = 0
    for partition in contract["input_partitions"]:
        result = verify_anchor_output(
            contract=contract, run_manifest_id=run_manifest_id, partition=partition
        )
        rows += int(result["receipt"]["output"]["rows"])
    return {"status": "complete", "anchor_receipts": 120, "anchor_rows": rows}


def _prepare_local_contract(
    *, manifest_path: str, policy_path: str
) -> tuple[dict[str, Any], RetrievalPolicy]:
    from reddit_china_stance.retrieval import load_retrieval_policy
    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest = root / manifest_path
    resolved_policy = root / policy_path
    manifest = load_source_manifest(resolved_manifest)
    policy = load_retrieval_policy(resolved_policy)
    code_state = _code_state(root, resolved_manifest, resolved_policy)
    contract = resolve_production_contract.remote(
        manifest=manifest, policy=policy, code_state=code_state
    )
    return contract, policy


@app.local_entrypoint()
def main(
    phase: str = "",
    manifest_path: str = "configs/source-files.json",
    policy_path: str = "configs/retrieval-policy-v1.toml",
    approved_cost_usd: str = "20",
    confirm: str = "",
    smoke_subreddit: str = "todayilearned",
    smoke_year: int = 2025,
    smoke_content_type: str = "comment",
) -> None:
    """Plan or durably submit one exact production wave."""

    if phase not in {"plan", "anchor-smoke", "anchors", "candidates"}:
        raise ValueError("phase must be exactly plan, anchor-smoke, anchors, or candidates")
    contract, policy = _prepare_local_contract(manifest_path=manifest_path, policy_path=policy_path)
    run_manifest_id = production_run_id(contract)
    estimated = Decimal(contract["compute_plan"]["estimated_full_run_cost_usd"])
    approved = Decimal(approved_cost_usd)
    enforce_cost_guardrail(estimated_cost_usd=estimated, approved_usd=approved)
    summary = {
        "status": "planned" if phase == "plan" else "submitted",
        "phase": phase,
        "run_manifest_id": run_manifest_id,
        "counts": contract["counts"],
        "compute_plan": contract["compute_plan"],
    }
    if phase == "plan":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if phase == "anchor-smoke":
        if confirm != CONFIRM_ANCHOR_SMOKE:
            raise ValueError(f"refusing anchor smoke: pass --confirm {CONFIRM_ANCHOR_SMOKE}")
        matches = [
            partition
            for partition in contract["input_partitions"]
            if partition["subreddit"] == smoke_subreddit
            and partition["year"] == smoke_year
            and partition["content_type"] == smoke_content_type
        ]
        if len(matches) != 1:
            raise ValueError("anchor smoke must resolve exactly one production partition")
        result = run_anchor.remote(
            {
                "contract": contract,
                "run_manifest_id": run_manifest_id,
                "cell": matches[0],
                "policy": policy,
            }
        )
        summary["smoke_result"] = result
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if phase == "anchors":
        if confirm != CONFIRM_ANCHORS:
            raise ValueError(f"refusing anchor launch: pass --confirm {CONFIRM_ANCHORS}")
        jobs = [
            {
                "contract": contract,
                "run_manifest_id": run_manifest_id,
                "cell": partition,
                "policy": policy,
            }
            for partition in contract["input_partitions"]
        ]
        summary["submitted_function_call_ids"] = [run_anchor.spawn(job).object_id for job in jobs]
    else:
        if confirm != CONFIRM_CANDIDATES:
            raise ValueError(f"refusing candidate launch: pass --confirm {CONFIRM_CANDIDATES}")
        validation = validate_all_anchors.remote(contract=contract, run_manifest_id=run_manifest_id)
        if validation.get("anchor_receipts") != 120:
            raise RuntimeError("candidate wave requires exactly 120 validated anchor receipts")
        jobs = [
            {
                "contract": contract,
                "run_manifest_id": run_manifest_id,
                "cell": {"subreddit": subreddit, "year": year},
                "policy": policy,
            }
            for subreddit in SUBREDDITS
            for year in ANCHOR_YEARS
        ]
        summary["submitted_function_call_ids"] = [
            run_candidate.spawn(job).object_id for job in jobs
        ]
    print(json.dumps(summary, indent=2, sort_keys=True))
