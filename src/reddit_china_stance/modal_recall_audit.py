"""Bounded Modal production runner for the design-weighted Stage A recall audit."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance.modal_stage_a import (
    ANCHOR_YEARS,
    CONTENT_TYPES,
    SUBREDDITS,
    _candidate_output_dirs,
    _receipt_partition_metadata,
    _unit_contract,
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
    _sha256_file,
    acquire_run_claim,
    update_run_claim,
)
from reddit_china_stance.recall_audit import (
    AUDIT_SCHEMA_VERSION,
    HashProbability,
    canonical_sha256,
    make_audit_contract,
    make_cell_plan,
    merge_selected_rows,
    score_band,
    selected_by_hash,
)

APP_NAME = "reddit-china-stance-recall-audit"
ENVIRONMENT_NAME = "main"
CLAIM_REGISTRY_NAME = "reddit-china-stance-recall-audit-claims"
AUDIT_PREFIX = Path("derived") / "recall-audit"
CONFIRMATION = "LAUNCH_RECALL_AUDIT_FIRST_WAVE"
SMOKE_CONFIRMATION = "RUN_ONE_RECALL_AUDIT_SMOKE"
DEFAULT_SEED = "recall-audit-first-wave-v1"
EXPECTED_INSIDE_PER_CELL = 1
EXPECTED_OUTSIDE_PER_CELL = 1
EXPECTED_CHALLENGER_SCREEN_PER_CELL = 1_000
MAX_SCREENED_ROWS_PER_CONTENT_TYPE = 5_000
MAX_PACKET_ROWS_PER_UNIT = 500
# The current Modal plan permits at most ten concurrent GPU containers account-wide.
MAX_CONTAINERS = 10
REQUESTED_CPUS = Decimal("8")
REQUESTED_MEMORY_GIB = Decimal("32")
CPU_HOUR_COST = Decimal("0.04730")
MEM_GIB_HOUR_COST = Decimal("0.00800")
L4_HOUR_COST = Decimal("0.80")
CONSERVATIVE_SCAN_ROWS_PER_SECOND = Decimal("15000")
CONSERVATIVE_EMBEDDING_ROWS_PER_SECOND = Decimal("200")
MODEL_STARTUP_SECONDS_PER_UNIT = Decimal("60")
COST_OVERHEAD_FACTOR = Decimal("1.5")
HARD_MAX_APPROVED_COST_USD = Decimal("25")
CHALLENGER_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHALLENGER_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
CHALLENGER_QUERIES = (
    "The text materially discusses China or Chinese affairs.",
    "The text discusses the Chinese government, the CCP, or Chinese politics.",
    "The text discusses Chinese people, society, language, or culture.",
    "The text discusses a Chinese company, technology, economy, or international actor.",
)
SCORE_BANDS = (
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
)
STOP_RULES = {
    "confidence_level": 0.95,
    "pass_point_recall": 0.95,
    "pass_recall_lower_bound": 0.90,
    "revise_retrieval_if_recall_upper_bound_below": 0.90,
    "expand_if_recall_interval_width_above": 0.10,
    "expand_if_any_relevant_challenger_only_items": True,
    "minimum_completed_probability_labels": 200,
    "decision_requires_both_probability_arms": True,
    "challenger_is_discovery_only": True,
}

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
        "pyahocorasick>=2.2,<3",
        "pyarrow==25.0.1",
        "pydantic>=2.11.0,<3",
        "sentence-transformers==5.1.2",
    )
    .add_local_python_source("reddit_china_stance")
)

_MODEL: Any | None = None
_QUERY_EMBEDDINGS: Any | None = None


def estimate_audit_cost_usd(
    *, canonical_rows: int, expected_challenger_screen_rows: int, units: int
) -> Decimal:
    """Conservatively price one full metadata/text scan and bounded embedding screen."""

    for name, value in (
        ("canonical_rows", canonical_rows),
        ("expected_challenger_screen_rows", expected_challenger_screen_rows),
        ("units", units),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    cpu_mem_hourly = REQUESTED_CPUS * CPU_HOUR_COST + REQUESTED_MEMORY_GIB * MEM_GIB_HOUR_COST
    scan_hours = Decimal(canonical_rows) / CONSERVATIVE_SCAN_ROWS_PER_SECOND / Decimal(3600)
    embedding_hours = (
        Decimal(expected_challenger_screen_rows)
        / CONSERVATIVE_EMBEDDING_ROWS_PER_SECOND
        / Decimal(3600)
    )
    startup_hours = Decimal(units) * MODEL_STARTUP_SECONDS_PER_UNIT / Decimal(3600)
    total = (
        scan_hours * cpu_mem_hourly
        + (embedding_hours + startup_hours) * (cpu_mem_hourly + L4_HOUR_COST)
    ) * COST_OVERHEAD_FACTOR
    return total.quantize(Decimal("0.01"))


def enforce_cost_guardrail(*, estimated_cost_usd: Decimal, approved_cost_usd: Decimal) -> None:
    if approved_cost_usd <= 0 or approved_cost_usd > HARD_MAX_APPROVED_COST_USD:
        raise ValueError(f"approved_cost_usd must be > 0 and <= {HARD_MAX_APPROVED_COST_USD}")
    if estimated_cost_usd > approved_cost_usd:
        raise RuntimeError(
            f"estimated recall-audit cost ${estimated_cost_usd} exceeds approved "
            f"${approved_cost_usd}"
        )


def _code_state(root: Path) -> dict[str, Any]:
    paths = (
        root / "src/reddit_china_stance/recall_audit.py",
        root / "src/reddit_china_stance/modal_recall_audit.py",
        root / "uv.lock",
    )
    files = {str(path.relative_to(root)): _sha256_file(path) for path in paths}
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
        "code_sha256": canonical_sha256(files),
        "git_commit": git_commit,
    }


def audit_run_id(contract: Mapping[str, Any]) -> str:
    return canonical_sha256(contract)


def _audit_root(contract: Mapping[str, Any], run_id: str) -> Path:
    return (
        VOLUME_PATH
        / AUDIT_PREFIX
        / f"stage-a-run={contract['stage_a_run_id']}"
        / f"schema={AUDIT_SCHEMA_VERSION}"
        / f"run={run_id}"
    )


def _audit_unit_contract(*, run_id: str, subreddit: str, year: int) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_run_id": run_id,
        "cell": {"subreddit": subreddit, "year": year},
    }


def audit_output_dirs(
    *, contract: Mapping[str, Any], run_id: str, subreddit: str, year: int
) -> tuple[Path, Path]:
    unit = _audit_unit_contract(run_id=run_id, subreddit=subreddit, year=year)
    unit_id = canonical_sha256(unit)
    root = _audit_root(contract, run_id) / "units" / f"subreddit={subreddit}" / f"year={year}"
    return root / f"unit={unit_id}", root / f".unit={unit_id}.incomplete"


def _verified_stage_a_candidate(
    *, stage_contract: Mapping[str, Any], stage_run_id: str, subreddit: str, year: int
) -> dict[str, Any]:
    unit = _unit_contract(
        run_manifest_id=stage_run_id,
        phase="candidate",
        cell={"subreddit": subreddit, "year": year},
    )
    final, _ = _candidate_output_dirs(
        contract=stage_contract,
        run_manifest_id=stage_run_id,
        subreddit=subreddit,
        year=year,
    )
    verified = _verify_pair(final, receipt_contract=unit, parquet_prefix="candidates")
    return {
        "receipt": verified["receipt"],
        "receipt_sha256": verified["receipt_sha256"],
        "receipt_relative_path": str(verified["receipt_path"].relative_to(VOLUME_PATH)),
        "output_sha256": verified["output_sha256"],
        "output_relative_path": str(verified["output_path"].relative_to(VOLUME_PATH)),
    }


def _make_contract(
    *,
    stage_contract: Mapping[str, Any],
    stage_run_id: str,
    seed: str,
    approved_cost_usd: Decimal,
    code_state: Mapping[str, Any],
) -> dict[str, Any]:
    if production_run_id(stage_contract) != stage_run_id:
        raise RuntimeError("Stage A run ID does not match its frozen contract")
    partition_index = {
        (row["subreddit"], row["year"], row["content_type"]): row
        for row in stage_contract["input_partitions"]
    }
    candidate_inputs: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for subreddit in SUBREDDITS:
        for year in ANCHOR_YEARS:
            verified = _verified_stage_a_candidate(
                stage_contract=stage_contract,
                stage_run_id=stage_run_id,
                subreddit=subreddit,
                year=year,
            )
            receipt = verified.pop("receipt")
            counts = receipt["counts"]["candidate_rows_by_content_type"]
            candidate_inputs.append(
                {
                    "subreddit": subreddit,
                    "year": year,
                    **verified,
                }
            )
            for content_type in CONTENT_TYPES:
                partition = partition_index[(subreddit, year, content_type)]
                plan = make_cell_plan(
                    subreddit=subreddit,
                    year=year,
                    content_type=content_type,
                    canonical_rows=int(partition["partition_rows"]),
                    candidate_rows=int(counts[content_type]),
                    expected_inside_rows=EXPECTED_INSIDE_PER_CELL,
                    expected_outside_rows=EXPECTED_OUTSIDE_PER_CELL,
                    expected_challenger_screen_rows=EXPECTED_CHALLENGER_SCREEN_PER_CELL,
                )
                plan["canonical_input"] = dict(partition)
                cells.append(plan)
    canonical_rows = sum(int(cell["canonical_rows"]) for cell in cells)
    expected_screened = sum(
        min(EXPECTED_CHALLENGER_SCREEN_PER_CELL, int(cell["noncandidate_rows"])) for cell in cells
    )
    estimated = estimate_audit_cost_usd(
        canonical_rows=canonical_rows,
        expected_challenger_screen_rows=expected_screened,
        units=len(SUBREDDITS) * len(ANCHOR_YEARS),
    )
    enforce_cost_guardrail(estimated_cost_usd=estimated, approved_cost_usd=approved_cost_usd)
    contract = make_audit_contract(
        stage_a_run_id=stage_run_id,
        dataset_revision=str(stage_contract["revision"]),
        retrieval_policy_digest=str(stage_contract["retrieval_policy"]["policy_digest"]),
        cells=cells,
        seed=seed,
        expected_inside_rows_per_cell=EXPECTED_INSIDE_PER_CELL,
        expected_outside_rows_per_cell=EXPECTED_OUTSIDE_PER_CELL,
        expected_challenger_screen_rows_per_cell=EXPECTED_CHALLENGER_SCREEN_PER_CELL,
        challenger={
            "kind": "semantic_embedding_cosine_challenger",
            "model_id": CHALLENGER_MODEL_ID,
            "revision": CHALLENGER_MODEL_REVISION,
            "queries": list(CHALLENGER_QUERIES),
            "maximum_scored_rows_per_content_type": MAX_SCREENED_ROWS_PER_CONTENT_TYPE,
            "bounded_scope": "phase-one hash sample only",
            "interpretation": "diagnostic challenger, not an oracle",
        },
        score_bands=SCORE_BANDS,
        stop_rules=STOP_RULES,
        cost_guardrail={
            "estimated_cost_usd": str(estimated),
            "approved_cost_usd": str(approved_cost_usd),
            "hard_max_approved_cost_usd": str(HARD_MAX_APPROVED_COST_USD),
            "cost_overhead_factor": str(COST_OVERHEAD_FACTOR),
        },
        code_state=code_state,
    )
    contract["stage_a_candidate_inputs"] = candidate_inputs
    return contract


def _reconstruct_stage_a_contract(
    *,
    manifest: Mapping[str, Any],
    policy: Any,
    code_state: Mapping[str, Any],
    input_partitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use Stage A's exact pure contract algorithm at the recall-app boundary."""

    return make_production_contract(
        manifest=manifest,
        policy=policy,
        code_state=code_state,
        input_partitions=input_partitions,
    )


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, cpu=2.0, memory=4096)
def resolve_stage_a_contract(
    *, manifest: dict[str, Any], policy: Any, code_state: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct the frozen Stage A contract inside this Modal app boundary."""

    volume.reload()
    sources = sorted(manifest["files"], key=lambda row: row["path"])
    partitions = [
        _receipt_partition_metadata(manifest=manifest, source=source, year=year)
        for source in sources
        for year in ANCHOR_YEARS
    ]
    return _reconstruct_stage_a_contract(
        manifest=manifest,
        policy=policy,
        code_state=code_state,
        input_partitions=partitions,
    )


def _prepare_stage_a_contract(*, manifest_path: str, policy_path: str) -> dict[str, Any]:
    """Load Stage A inputs locally, then resolve receipts through this app's function."""

    from reddit_china_stance.retrieval import load_retrieval_policy
    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest = root / manifest_path
    resolved_policy = root / policy_path
    manifest = load_source_manifest(resolved_manifest)
    policy = load_retrieval_policy(resolved_policy)
    code_state = _stage_a_code_state(root, resolved_manifest, resolved_policy)
    return resolve_stage_a_contract.remote(
        manifest=manifest,
        policy=policy,
        code_state=code_state,
    )


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, cpu=2.0, memory=4096)
def resolve_audit_contract(
    *,
    stage_contract: dict[str, Any],
    stage_run_id: str,
    seed: str,
    approved_cost_usd: str,
    code_state: dict[str, Any],
) -> dict[str, Any]:
    volume.reload()
    return _make_contract(
        stage_contract=stage_contract,
        stage_run_id=stage_run_id,
        seed=seed,
        approved_cost_usd=Decimal(approved_cost_usd),
        code_state=code_state,
    )


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, cpu=1.0, memory=2048)
def persist_audit_manifest(contract: dict[str, Any], run_id: str) -> dict[str, Any]:
    volume.reload()
    if audit_run_id(contract) != run_id:
        raise RuntimeError("audit run ID does not match contract")
    root = _audit_root(contract, run_id)
    path = root / f"manifest-{run_id}.json"
    payload = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if (
        hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != run_id
    ):
        raise RuntimeError("manifest canonical digest mismatch")
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError("existing audit manifest differs from frozen contract")
        return {"status": "already_complete", "manifest": str(path.relative_to(VOLUME_PATH))}
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    volume.commit()
    return {"status": "complete", "manifest": str(path.relative_to(VOLUME_PATH))}


def _probability_from_contract(value: Mapping[str, Any]) -> HashProbability:
    if int(value.get("hash_space", -1)) != 1 << 64:
        raise RuntimeError("sampling hash-space contract mismatch")
    probability = HashProbability(int(value["threshold"]))
    if probability.probability != float(value["probability"]):
        raise RuntimeError("sampling probability contract mismatch")
    return probability


def _score_challenger(texts: Sequence[str]) -> list[float]:
    global _MODEL, _QUERY_EMBEDDINGS
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if _MODEL is None:
        _MODEL = SentenceTransformer(
            CHALLENGER_MODEL_ID,
            revision=CHALLENGER_MODEL_REVISION,
            trust_remote_code=False,
        )
    if _QUERY_EMBEDDINGS is None:
        _QUERY_EMBEDDINGS = _MODEL.encode(
            list(CHALLENGER_QUERIES),
            batch_size=len(CHALLENGER_QUERIES),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    if not texts:
        return []
    embeddings = _MODEL.encode(
        list(texts),
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = np.max(embeddings @ _QUERY_EMBEDDINGS.T, axis=1)
    return [float(score) for score in scores]


def _packet_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("audit_item_id", pa.string()),
            ("record_id", pa.string()),
            ("subreddit", pa.string()),
            ("year", pa.int16()),
            ("content_type", pa.string()),
            ("display_order", pa.int32()),
        ]
    )


def _ledger_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("record_id", pa.string()),
            ("is_stage_a_candidate", pa.bool_()),
            ("inside_probability_inclusion_probability", pa.float64()),
            ("outside_probability_inclusion_probability", pa.float64()),
            ("challenger_phase1_inclusion_probability", pa.float64()),
            ("challenger_phase2_inclusion_probability", pa.float64()),
            ("challenger_combined_inclusion_probability", pa.float64()),
            ("challenger_score", pa.float32()),
            ("challenger_score_band", pa.string()),
            ("audit_item_id", pa.string()),
            ("subreddit", pa.string()),
            ("year", pa.int16()),
            ("content_type", pa.string()),
            ("selection_channels", pa.list_(pa.string())),
        ]
    )


def _verify_audit_unit(
    *, contract: Mapping[str, Any], run_id: str, subreddit: str, year: int
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    final, _ = audit_output_dirs(contract=contract, run_id=run_id, subreddit=subreddit, year=year)
    unit = _audit_unit_contract(run_id=run_id, subreddit=subreddit, year=year)
    receipts = list(final.glob("receipt-*.json"))
    packets = list(final.glob("annotation-packet-*.parquet"))
    ledgers = list(final.glob("selection-ledger-*.parquet"))
    if (
        len(receipts) != 1
        or len(packets) != 1
        or len(ledgers) != 1
        or len(list(final.iterdir())) != 3
    ):
        raise RuntimeError(f"audit unit is not an exact receipt/output triple: {final}")
    receipt_path, packet_path, ledger_path = receipts[0], packets[0], ledgers[0]
    receipt_sha = _sha256_file(receipt_path)
    packet_sha = _sha256_file(packet_path)
    ledger_sha = _sha256_file(ledger_path)
    if receipt_path.name != f"receipt-{receipt_sha}.json":
        raise RuntimeError("audit receipt is not content-addressed")
    if packet_path.name != f"annotation-packet-{packet_sha}.parquet":
        raise RuntimeError("audit packet is not content-addressed")
    if ledger_path.name != f"selection-ledger-{ledger_sha}.parquet":
        raise RuntimeError("audit ledger is not content-addressed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("unit_contract") != unit:
        raise RuntimeError("audit unit contract mismatch")
    for name, path, digest in (
        ("annotation_packet", packet_path, packet_sha),
        ("selection_ledger", ledger_path, ledger_sha),
    ):
        output = receipt["outputs"][name]
        if (
            output["sha256"] != digest
            or output["bytes"] != path.stat().st_size
            or output["rows"] != pq.ParquetFile(path).metadata.num_rows
            or output.get("contains_raw_text") is not False
        ):
            raise RuntimeError(f"audit {name} output mismatch")
    return {
        "receipt": receipt,
        "receipt_sha256": receipt_sha,
        "packet_path": packet_path,
        "ledger_path": ledger_path,
    }


def _canonical_parquet(cell: Mapping[str, Any]) -> Any:
    import pyarrow.parquet as pq

    source = cell["canonical_input"]
    path = VOLUME_PATH / source["partition_relative_path"]
    if not path.exists() or path.stat().st_size != int(source["partition_bytes"]):
        raise RuntimeError(f"canonical partition size mismatch: {path}")
    if _sha256_file(path) != source["partition_sha256"]:
        raise RuntimeError(f"canonical partition digest mismatch: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != int(source["partition_rows"]):
        raise RuntimeError(f"canonical partition row mismatch: {path}")
    if not {"record_id", "text"}.issubset(parquet.schema_arrow.names):
        raise RuntimeError(f"canonical partition lacks recall-audit fields: {path}")
    return parquet


def _execute_audit_unit(job: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if set(job) != {"contract", "audit_run_id", "cell"}:
        raise ValueError("recall-audit job has unexpected fields")
    contract = dict(job["contract"])
    run_id = str(job["audit_run_id"])
    if audit_run_id(contract) != run_id:
        raise RuntimeError("audit run ID does not match contract")
    cell = dict(job["cell"])
    subreddit, year = str(cell["subreddit"]), int(cell["year"])
    if subreddit not in SUBREDDITS or year not in ANCHOR_YEARS:
        raise ValueError("recall-audit unit is outside the frozen grid")
    unit_contract = _audit_unit_contract(run_id=run_id, subreddit=subreddit, year=year)
    unit_id = canonical_sha256(unit_contract)
    final_dir, staging_dir = audit_output_dirs(
        contract=contract, run_id=run_id, subreddit=subreddit, year=year
    )
    volume.reload()
    if final_dir.exists():
        verified = _verify_audit_unit(
            contract=contract, run_id=run_id, subreddit=subreddit, year=year
        )
        return {
            "status": "already_complete",
            "unit_id": unit_id,
            "rows": verified["receipt"]["outputs"]["annotation_packet"]["rows"],
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
        if final_dir.exists():
            verified = _verify_audit_unit(
                contract=contract, run_id=run_id, subreddit=subreddit, year=year
            )
            return {
                "status": "already_complete",
                "unit_id": unit_id,
                "rows": verified["receipt"]["outputs"]["annotation_packet"]["rows"],
            }
        if staging_dir.exists():
            if not acquisition["same_owner_retry"]:
                raise FileExistsError(f"unclaimed recall-audit staging exists: {staging_dir}")
            raise FileExistsError(
                "same-owner recall-audit staging exists; inspect it before an explicit retry"
            )

        inputs = [
            row
            for row in contract["stage_a_candidate_inputs"]
            if row["subreddit"] == subreddit and row["year"] == year
        ]
        if len(inputs) != 1:
            raise RuntimeError("audit unit lacks one exact Stage A candidate input")
        candidate_input = inputs[0]
        candidate_path = VOLUME_PATH / candidate_input["output_relative_path"]
        if _sha256_file(candidate_path) != candidate_input["output_sha256"]:
            raise RuntimeError("Stage A candidate input digest mismatch")
        candidate_table = pq.read_table(
            candidate_path, columns=["record_id", "content_type"]
        ).to_pydict()
        candidate_ids: dict[str, set[str]] = {kind: set() for kind in CONTENT_TYPES}
        for record_id, content_type in zip(
            candidate_table["record_id"], candidate_table["content_type"], strict=True
        ):
            if content_type not in candidate_ids or not isinstance(record_id, str):
                raise RuntimeError("Stage A candidate row is malformed")
            if record_id in candidate_ids[content_type]:
                raise RuntimeError(f"duplicate Stage A candidate record ID: {record_id}")
            candidate_ids[content_type].add(record_id)

        started = time.monotonic()
        all_packets: list[dict[str, Any]] = []
        all_ledgers: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for content_type in CONTENT_TYPES:
            plans = [
                row
                for row in contract["cells"]
                if row["subreddit"] == subreddit
                and row["year"] == year
                and row["content_type"] == content_type
            ]
            if len(plans) != 1:
                raise RuntimeError("audit unit lacks one exact content-type plan")
            plan = plans[0]
            if len(candidate_ids[content_type]) != int(plan["candidate_rows"]):
                raise RuntimeError("Stage A candidate count differs from frozen audit plan")
            inside_probability = _probability_from_contract(plan["inside_probability_arm"])
            outside_probability = _probability_from_contract(plan["outside_probability_arm"])
            challenger_phase1 = _probability_from_contract(plan["challenger_phase1"])
            probability_rows: list[dict[str, Any]] = []
            screened: list[tuple[str, str]] = []
            seen_candidates: set[str] = set()
            scanned_rows = 0
            parquet = _canonical_parquet(plan)
            for batch in parquet.iter_batches(
                batch_size=50_000, columns=["record_id", "text"], use_threads=True
            ):
                data = batch.to_pydict()
                for record_id, text in zip(data["record_id"], data["text"], strict=True):
                    if not isinstance(record_id, str) or not record_id:
                        raise RuntimeError("canonical recall-audit row has invalid record_id")
                    in_candidate = record_id in candidate_ids[content_type]
                    if in_candidate:
                        seen_candidates.add(record_id)
                        if selected_by_hash(
                            record_id=record_id,
                            seed=contract["seed"],
                            arm="candidate_probability",
                            probability=inside_probability,
                        ):
                            probability_rows.append(
                                {
                                    "record_id": record_id,
                                    "is_stage_a_candidate": True,
                                    "inside_probability_inclusion_probability": (
                                        inside_probability.probability
                                    ),
                                }
                            )
                    else:
                        if selected_by_hash(
                            record_id=record_id,
                            seed=contract["seed"],
                            arm="noncandidate_probability",
                            probability=outside_probability,
                        ):
                            probability_rows.append(
                                {
                                    "record_id": record_id,
                                    "is_stage_a_candidate": False,
                                    "outside_probability_inclusion_probability": (
                                        outside_probability.probability
                                    ),
                                }
                            )
                        if selected_by_hash(
                            record_id=record_id,
                            seed=contract["seed"],
                            arm="challenger_phase1",
                            probability=challenger_phase1,
                        ):
                            if not isinstance(text, str) or not text:
                                raise RuntimeError(
                                    f"canonical recall-audit row has invalid text: {record_id}"
                                )
                            screened.append((record_id, text))
                    scanned_rows += 1
            if scanned_rows != int(plan["canonical_rows"]):
                raise RuntimeError("recall-audit scan did not conserve canonical rows")
            if seen_candidates != candidate_ids[content_type]:
                raise RuntimeError("not every Stage A candidate exists in its canonical cell")
            if len(screened) > MAX_SCREENED_ROWS_PER_CONTENT_TYPE:
                raise RuntimeError(
                    f"challenger phase-one sample has {len(screened)} rows, exceeding "
                    f"hard maximum {MAX_SCREENED_ROWS_PER_CONTENT_TYPE}"
                )
            scores = _score_challenger([text for _, text in screened])
            if len(scores) != len(screened):
                raise RuntimeError("challenger scorer did not conserve screened rows")
            challenger_rows: list[dict[str, Any]] = []
            bands = contract["design"]["challenger_two_phase_arm"]["score_bands"]
            for (record_id, _), score in zip(screened, scores, strict=True):
                band = score_band(score, bands)
                phase2 = _probability_from_contract(band["phase2"])
                if selected_by_hash(
                    record_id=record_id,
                    seed=contract["seed"],
                    arm=f"challenger_phase2:{band['name']}",
                    probability=phase2,
                ):
                    challenger_rows.append(
                        {
                            "record_id": record_id,
                            "is_stage_a_candidate": False,
                            "challenger_phase1_inclusion_probability": (
                                challenger_phase1.probability
                            ),
                            "challenger_phase2_inclusion_probability": phase2.probability,
                            "challenger_combined_inclusion_probability": (
                                challenger_phase1.probability * phase2.probability
                            ),
                            "challenger_score": score,
                            "challenger_score_band": band["name"],
                        }
                    )
            packet, ledger = merge_selected_rows(
                audit_run_id=run_id,
                cell=plan,
                probability_rows=probability_rows,
                challenger_rows=challenger_rows,
            )
            all_packets.extend(packet)
            all_ledgers.extend(ledger)
            counts[f"{content_type}.canonical_rows"] = scanned_rows
            counts[f"{content_type}.candidate_rows"] = len(candidate_ids[content_type])
            counts[f"{content_type}.challenger_phase1_rows"] = len(screened)
            counts[f"{content_type}.packet_rows"] = len(packet)
        if len(all_packets) > MAX_PACKET_ROWS_PER_UNIT:
            raise RuntimeError(
                f"audit packet has {len(all_packets)} rows, exceeding hard maximum "
                f"{MAX_PACKET_ROWS_PER_UNIT}"
            )
        all_packets.sort(key=lambda row: row["audit_item_id"])
        for index, row in enumerate(all_packets):
            row["display_order"] = index
        all_ledgers.sort(key=lambda row: row["audit_item_id"])
        staging_dir.mkdir(parents=True)
        packet_temporary = staging_dir / "annotation-packet.parquet.incomplete"
        ledger_temporary = staging_dir / "selection-ledger.parquet.incomplete"
        pq.write_table(
            pa.Table.from_pylist(all_packets, schema=_packet_schema()),
            packet_temporary,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(all_ledgers, schema=_ledger_schema()),
            ledger_temporary,
            compression="zstd",
        )
        packet_sha = _sha256_file(packet_temporary)
        ledger_sha = _sha256_file(ledger_temporary)
        packet_path = staging_dir / f"annotation-packet-{packet_sha}.parquet"
        ledger_path = staging_dir / f"selection-ledger-{ledger_sha}.parquet"
        packet_temporary.replace(packet_path)
        ledger_temporary.replace(ledger_path)
        receipt = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": "complete",
            "unit_contract": unit_contract,
            "stage_a_candidate_input": candidate_input,
            "counts": dict(sorted(counts.items())),
            "outputs": {
                "annotation_packet": {
                    "file": packet_path.name,
                    "rows": len(all_packets),
                    "bytes": packet_path.stat().st_size,
                    "sha256": packet_sha,
                    "schema_fields": _packet_schema().names,
                    "contains_raw_text": False,
                    "selection_channel_blinded": True,
                },
                "selection_ledger": {
                    "file": ledger_path.name,
                    "rows": len(all_ledgers),
                    "bytes": ledger_path.stat().st_size,
                    "sha256": ledger_sha,
                    "schema_fields": _ledger_schema().names,
                    "contains_raw_text": False,
                    "selection_channel_blinded": False,
                },
            },
            "runtime": {
                "requested_cpus": int(REQUESTED_CPUS),
                "requested_memory_gib": int(REQUESTED_MEMORY_GIB),
                "gpu": "L4",
                "python_version": platform.python_version(),
                "pyarrow_version": pa.__version__,
                "challenger_model_id": CHALLENGER_MODEL_ID,
                "challenger_model_revision": CHALLENGER_MODEL_REVISION,
            },
            "wall_seconds": round(time.monotonic() - started, 3),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        receipt_payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        receipt_sha = hashlib.sha256(receipt_payload.encode()).hexdigest()
        (staging_dir / f"receipt-{receipt_sha}.json").write_text(receipt_payload, encoding="utf-8")
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
            "packet_rows": len(all_packets),
            "counts": dict(sorted(counts.items())),
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
    gpu="L4",
    timeout=14_400,
    max_containers=MAX_CONTAINERS,
)
def run_audit_unit(job: dict[str, Any]) -> dict[str, Any]:
    return _execute_audit_unit(job)


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, cpu=2.0, memory=4096)
def validate_audit_run(contract: dict[str, Any], run_id: str) -> dict[str, Any]:
    volume.reload()
    if audit_run_id(contract) != run_id:
        raise RuntimeError("audit run ID does not match contract")
    receipts = 0
    rows = 0
    for subreddit in SUBREDDITS:
        for year in ANCHOR_YEARS:
            result = _verify_audit_unit(
                contract=contract, run_id=run_id, subreddit=subreddit, year=year
            )
            receipts += 1
            rows += int(result["receipt"]["outputs"]["annotation_packet"]["rows"])
    return {"status": "complete", "receipts": receipts, "packet_rows": rows}


@app.local_entrypoint()
def main(
    phase: str = "",
    manifest_path: str = "configs/source-files.json",
    policy_path: str = "configs/retrieval-policy-v1.toml",
    seed: str = DEFAULT_SEED,
    approved_cost_usd: str = "25",
    confirm: str = "",
    smoke_subreddit: str = "todayilearned",
    smoke_year: int = 2025,
) -> None:
    """Plan, durably launch, or validate the first recall-audit wave."""

    if phase not in {"plan", "smoke", "launch", "validate"}:
        raise ValueError("phase must be exactly plan, smoke, launch, or validate")
    root = Path(__file__).resolve().parents[2]
    stage_contract = _prepare_stage_a_contract(manifest_path=manifest_path, policy_path=policy_path)
    stage_run_id = production_run_id(stage_contract)
    contract = resolve_audit_contract.remote(
        stage_contract=stage_contract,
        stage_run_id=stage_run_id,
        seed=seed,
        approved_cost_usd=approved_cost_usd,
        code_state=_code_state(root),
    )
    run_id = audit_run_id(contract)
    summary: dict[str, Any] = {
        "status": "planned" if phase == "plan" else "submitted",
        "phase": phase,
        "stage_a_run_id": stage_run_id,
        "audit_run_id": run_id,
        "cost_guardrail": contract["cost_guardrail"],
        "expected_cells": len(contract["cells"]),
        "expected_units": len(SUBREDDITS) * len(ANCHOR_YEARS),
    }
    if phase == "plan":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if phase == "smoke" and confirm != SMOKE_CONFIRMATION:
        raise ValueError(f"refusing audit smoke: pass --confirm {SMOKE_CONFIRMATION}")
    if phase == "launch" and confirm != CONFIRMATION:
        raise ValueError(f"refusing audit launch: pass --confirm {CONFIRMATION}")
    persist_audit_manifest.remote(contract, run_id)
    if phase == "smoke":
        if smoke_subreddit not in SUBREDDITS or smoke_year not in ANCHOR_YEARS:
            raise ValueError("recall-audit smoke cell is outside the frozen grid")
        summary["smoke_result"] = run_audit_unit.remote(
            {
                "contract": contract,
                "audit_run_id": run_id,
                "cell": {"subreddit": smoke_subreddit, "year": smoke_year},
            }
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if phase == "validate":
        summary["validation"] = validate_audit_run.remote(contract, run_id)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    summary["submitted_function_call_ids"] = [
        run_audit_unit.spawn(
            {
                "contract": contract,
                "audit_run_id": run_id,
                "cell": {"subreddit": subreddit, "year": year},
            }
        ).object_id
        for subreddit in SUBREDDITS
        for year in ANCHOR_YEARS
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
