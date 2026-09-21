"""Acquisition-specific Sol-v2 labelling over one frozen two-arm packet.

This module deliberately wraps, rather than changes, the historical 10k Sol-v2
engine.  The wrapper binds the acquisition design and private arm ledger, builds
one deterministically interleaved arm-blinded provider packet, and publishes an
acquisition-specific final mapping plus aggregate-only arm diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from reddit_china_stance import sol_teacher_10k_v2 as teacher
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_ontology_v2 import (
    canonical_sha256,
    file_sha256,
    validate_v2_label,
)

SCHEMA_VERSION = "1.0.0"
LEDGER_KIND = "modernbert-acquisition-ledger-v1"
PACKET_KIND = "sol-teacher-acquisition-v2-packet-v1"
RUN_KIND = "sol-teacher-acquisition-v2-run-contract-v1"
EXPECTED_ROWS = 2_000
ARM_ROWS = 1_000
ARMS = ("probability_random", "active")
ACTIVE_BUCKETS = (
    "rare_cell",
    "boundary",
    "multi_context",
    "uncertainty_disagreement",
)
ACTIVE_BUCKET_COUNTS = {
    "rare_cell": 200,
    "boundary": 300,
    "multi_context": 100,
    "uncertainty_disagreement": 400,
}
INPUT_BINDING_FIELDS = {
    "eligible_frame_sha256",
    "source_inventory_sha256",
    "exclusion_ledger_sha256",
    "checkpoint_bundle_sha256",
    "scoring_artifact_sha256",
    "policy_file_sha256",
}
SOURCE_COLUMNS = (
    "opaque_id",
    "thread_id",
    "target_text",
    "submission_context",
    "parent_context",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "configs/modernbert-acquisition-v1.toml"
DEFAULT_PACKET_PARENT = REPO_ROOT / "data/private-sol-teacher-acquisition-v2/packet"
DEFAULT_PRIVATE_ROOT = REPO_ROOT / "data/private-sol-teacher-acquisition-v2/generation"
DEFAULT_PUBLIC_ROOT = REPO_ROOT / "outputs/sol-teacher-acquisition-v2"
INTERLEAVE_ALGORITHM = "paired-arm-order-hash-flip-v1"
RUNNER_SOURCE_PATHS = (
    "configs/modernbert-acquisition-v1.toml",
    "src/reddit_china_stance/sol_teacher_acquisition_v2.py",
    "src/reddit_china_stance/sol_teacher_10k_v2.py",
    "src/reddit_china_stance/sol_ontology_bridge_v2.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
    "src/reddit_china_stance/privacy.py",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    teacher._write_immutable_json(path, value)


def _source_bundle() -> dict[str, Any]:
    files = []
    for relative in RUNNER_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"acquisition runtime source is missing: {relative}")
        files.append({"path": relative, "sha256": file_sha256(path)})
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *RUNNER_SOURCE_PATHS,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_head": head,
        "scoped_git_status": status,
        "files": files,
        "digest": canonical_sha256(files),
    }


def _validate_source_bundle(
    bundle: Mapping[str, Any], *, require_current_files: bool = False
) -> dict[str, Any]:
    files = bundle.get("files")
    if (
        not isinstance(bundle.get("git_head"), str)
        or not bundle["git_head"]
        or not isinstance(bundle.get("scoped_git_status"), list)
        or not isinstance(files, list)
        or any(not isinstance(entry, Mapping) for entry in files)
        or [entry.get("path") for entry in files] != list(RUNNER_SOURCE_PATHS)
        or bundle.get("digest") != canonical_sha256(files)
    ):
        raise RuntimeError("acquisition runtime source bundle is invalid")
    if require_current_files:
        for entry in files:
            path = REPO_ROOT / str(entry["path"])
            if not path.is_file() or file_sha256(path) != entry.get("sha256"):
                raise RuntimeError("acquisition runtime listed-file digest drifted")
    return dict(bundle)


def _validate_probability(value: object, *, numerator: object, denominator: object) -> float:
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator <= 0
        or denominator <= 0
        or numerator > denominator
        or not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise ValueError("probability-arm inclusion probability is invalid")
    expected = numerator / denominator
    if abs(float(value) - expected) > 1e-12:
        raise ValueError("probability-arm inclusion probability is inconsistent")
    return float(value)


def validate_acquisition_ledger(path: Path) -> dict[str, Any]:
    """Validate the exact private sampler ledger without exposing row data."""

    if not path.is_file():
        raise FileNotFoundError(f"private acquisition ledger is missing: {path}")
    ledger = _read_object(path)
    expected_top = {
        "schema_version",
        "kind",
        "policy",
        "policy_file_sha256",
        "policy_contract_sha256",
        "input_bindings",
        "rare_cells",
        "rare_cell_list_sha256",
        "eligible_population_rows",
        "candidate_score_digest",
        "probability_arm",
        "active_arm",
        "ledger_id",
    }
    if set(ledger) != expected_top:
        raise ValueError("acquisition ledger top-level fields drifted")
    if ledger.get("schema_version") != SCHEMA_VERSION or ledger.get("kind") != LEDGER_KIND:
        raise ValueError("acquisition ledger kind or schema drifted")
    if not isinstance(ledger.get("policy"), Mapping):
        raise ValueError("acquisition policy is invalid")
    if (
        not _is_sha256(ledger.get("policy_file_sha256"))
        or ledger.get("policy_contract_sha256") != canonical_sha256(ledger["policy"])
    ):
        raise ValueError("acquisition policy digest drifted")
    if file_sha256(DEFAULT_POLICY_PATH) != ledger["policy_file_sha256"]:
        raise ValueError("acquisition policy file digest drifted")
    bindings = ledger.get("input_bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != INPUT_BINDING_FIELDS
        or any(not _is_sha256(value) for value in bindings.values())
    ):
        raise ValueError("acquisition input bindings are invalid")
    if bindings["policy_file_sha256"] != ledger["policy_file_sha256"]:
        raise ValueError("acquisition input policy binding drifted")
    rare_cells = ledger.get("rare_cells")
    if not isinstance(rare_cells, list) or ledger.get("rare_cell_list_sha256") != canonical_sha256(
        rare_cells
    ):
        raise ValueError("acquisition rare-cell binding drifted")
    if type(ledger.get("eligible_population_rows")) is not int or ledger[
        "eligible_population_rows"
    ] < EXPECTED_ROWS:
        raise ValueError("acquisition eligible-population count is invalid")
    if not _is_sha256(ledger.get("candidate_score_digest")):
        raise ValueError("acquisition candidate-score digest is invalid")

    probability = ledger.get("probability_arm")
    active = ledger.get("active_arm")
    if not isinstance(probability, Mapping) or set(probability) != {
        "row_count",
        "strata",
        "rows",
    }:
        raise ValueError("probability arm contract is invalid")
    if not isinstance(active, Mapping) or set(active) != {"row_count", "buckets", "rows"}:
        raise ValueError("active arm contract is invalid")
    if probability.get("row_count") != ARM_ROWS or active.get("row_count") != ARM_ROWS:
        raise ValueError("acquisition ledger must contain exactly 1,000 rows per arm")
    if active.get("buckets") != ACTIVE_BUCKET_COUNTS:
        raise ValueError("active acquisition bucket counts drifted")
    if not isinstance(probability.get("strata"), list):
        raise ValueError("probability-arm strata are invalid")
    stratum_total = 0
    stratum_contracts: dict[str, Mapping[str, Any]] = {}
    for stratum in probability["strata"]:
        if not isinstance(stratum, Mapping) or set(stratum) != {
            "stratum",
            "population_rows",
            "sample_rows",
        }:
            raise ValueError("probability-arm stratum contract drifted")
        if (
            not isinstance(stratum["stratum"], Mapping)
            or set(stratum["stratum"])
            != {"subreddit", "year", "content_type", "retrieval_mode"}
            or type(stratum["population_rows"]) is not int
            or type(stratum["sample_rows"]) is not int
            or not 0 <= stratum["sample_rows"] <= stratum["population_rows"]
        ):
            raise ValueError("probability-arm stratum is invalid")
        key = canonical_sha256(stratum["stratum"])
        if key in stratum_contracts:
            raise ValueError("probability-arm contains a duplicate stratum")
        stratum_contracts[key] = stratum
        stratum_total += stratum["sample_rows"]
    if stratum_total != ARM_ROWS:
        raise ValueError("probability-arm stratum samples do not conserve rows")

    probability_rows = probability.get("rows")
    active_rows = active.get("rows")
    if not isinstance(probability_rows, list) or len(probability_rows) != ARM_ROWS:
        raise ValueError("probability-arm rows do not conserve the frozen count")
    if not isinstance(active_rows, list) or len(active_rows) != ARM_ROWS:
        raise ValueError("active-arm rows do not conserve the frozen count")
    seen_ids: set[str] = set()
    seen_threads: set[str] = set()
    seen_clusters: set[str] = set()
    probability_schema = {
        "opaque_id",
        "thread_id",
        "near_duplicate_cluster_id",
        "stratum",
        "inclusion_probability_numerator",
        "inclusion_probability_denominator",
        "inclusion_probability",
        "selection_tiebreak_sha256",
    }
    active_schema = {
        "opaque_id",
        "thread_id",
        "near_duplicate_cluster_id",
        "bucket",
        "bucket_rank",
        "score",
        "selection_tiebreak_sha256",
    }
    bucket_counts: Counter[str] = Counter()
    stratum_observed: Counter[str] = Counter()
    for arm, rows, schema in (
        (ARMS[0], probability_rows, probability_schema),
        (ARMS[1], active_rows, active_schema),
    ):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != schema:
                raise ValueError(f"{arm} row schema drifted")
            for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"{arm} row {index} has invalid {field}")
            if not _is_sha256(row.get("selection_tiebreak_sha256")):
                raise ValueError(f"{arm} row {index} has invalid selection tie-break")
            if row["opaque_id"] in seen_ids or row["thread_id"] in seen_threads:
                raise ValueError("acquisition arms contain duplicate IDs or threads")
            if row["near_duplicate_cluster_id"] in seen_clusters:
                raise ValueError("acquisition arms contain overlapping near-duplicate clusters")
            seen_ids.add(row["opaque_id"])
            seen_threads.add(row["thread_id"])
            seen_clusters.add(row["near_duplicate_cluster_id"])
            if arm == ARMS[0]:
                _validate_probability(
                    row["inclusion_probability"],
                    numerator=row["inclusion_probability_numerator"],
                    denominator=row["inclusion_probability_denominator"],
                )
                if not isinstance(row["stratum"], Mapping) or set(row["stratum"]) != {
                    "subreddit",
                    "year",
                    "content_type",
                    "retrieval_mode",
                }:
                    raise ValueError("probability-arm row stratum drifted")
                stratum_key = canonical_sha256(row["stratum"])
                stratum = stratum_contracts.get(stratum_key)
                if (
                    stratum is None
                    or row["inclusion_probability_numerator"] != stratum["sample_rows"]
                    or row["inclusion_probability_denominator"]
                    != stratum["population_rows"]
                ):
                    raise ValueError("probability-arm row differs from its stratum contract")
                stratum_observed[stratum_key] += 1
            else:
                if row.get("bucket") not in ACTIVE_BUCKETS:
                    raise ValueError("active-arm bucket is invalid")
                if type(row.get("bucket_rank")) is not int or row["bucket_rank"] <= 0:
                    raise ValueError("active-arm bucket rank is invalid")
                if not isinstance(row.get("score"), (int, float)) or isinstance(
                    row.get("score"), bool
                ) or not math.isfinite(float(row["score"])):
                    raise ValueError("active-arm score is invalid")
                bucket_counts[row["bucket"]] += 1
    if dict(bucket_counts) != ACTIVE_BUCKET_COUNTS:
        raise ValueError("active-arm realised bucket counts drifted")
    if any(
        stratum_observed.get(key, 0) != contract["sample_rows"]
        for key, contract in stratum_contracts.items()
    ):
        raise ValueError("probability-arm row counts differ from strata")
    if probability_rows != sorted(
        probability_rows,
        key=lambda row: (
            canonical_sha256(row["stratum"]),
            row["selection_tiebreak_sha256"],
        ),
    ):
        raise ValueError("probability-arm row order drifted")
    expected_active_order = sorted(
        active_rows,
        key=lambda row: (ACTIVE_BUCKETS.index(row["bucket"]), row["bucket_rank"]),
    )
    if active_rows != expected_active_order:
        raise ValueError("active-arm row order drifted")
    body = {key: value for key, value in ledger.items() if key != "ledger_id"}
    if ledger.get("ledger_id") != canonical_sha256(body):
        raise ValueError("acquisition ledger ID drifted")
    return ledger


def _load_source_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"private acquisition source Parquet is missing: {path}")
    parquet = pq.ParquetFile(path)
    missing = set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(
            f"private acquisition source is missing required columns: {sorted(missing)}"
        )
    rows = parquet.read(columns=list(SOURCE_COLUMNS)).to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"private acquisition source row count drifted: expected {EXPECTED_ROWS}, "
            f"got {len(rows)}"
        )
    ids: set[str] = set()
    threads: set[str] = set()
    for index, row in enumerate(rows):
        for field in ("opaque_id", "thread_id", "target_text"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"private acquisition source row {index} has invalid {field}")
        for field in ("submission_context", "parent_context"):
            if row.get(field) is not None and not isinstance(row[field], str):
                raise ValueError(f"private acquisition source row {index} has invalid {field}")
        if row["opaque_id"] in ids or row["thread_id"] in threads:
            raise ValueError("private acquisition source contains duplicate IDs or threads")
        ids.add(row["opaque_id"])
        threads.add(row["thread_id"])
    return [dict(row) for row in rows]


def _interleaved_rows(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_arm = {
        ARMS[0]: list(ledger["probability_arm"]["rows"]),
        ARMS[1]: list(ledger["active_arm"]["rows"]),
    }
    result: list[dict[str, Any]] = []
    for arm_order in range(ARM_ROWS):
        digest = hashlib.sha256(
            f"{INTERLEAVE_ALGORITHM}\0{ledger['ledger_id']}\0{arm_order}".encode()
        ).digest()
        order = ARMS if digest[0] % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            source = by_arm[arm][arm_order]
            result.append(
                {
                    "source_sample_id": source["opaque_id"],
                    "thread_id": source["thread_id"],
                    "acquisition_arm": arm,
                    "arm_order": arm_order,
                    "packet_order": len(result),
                    "inclusion_probability": (
                        float(source["inclusion_probability"])
                        if arm == ARMS[0]
                        else None
                    ),
                    "active_bucket": source.get("bucket") if arm == ARMS[1] else None,
                    "near_duplicate_cluster_id": source["near_duplicate_cluster_id"],
                }
            )
    return result


def _validate_source_and_ledger(
    source_path: Path, ledger_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_rows = _load_source_rows(source_path)
    ledger = validate_acquisition_ledger(ledger_path)
    ledger_order = [
        row["opaque_id"]
        for arm_key in ("probability_arm", "active_arm")
        for row in ledger[arm_key]["rows"]
    ]
    source_ids = [row["opaque_id"] for row in source_rows]
    if source_ids != ledger_order:
        raise RuntimeError("source Parquet rows do not conserve the frozen ledger order")
    thread_by_id = {row["opaque_id"]: row["thread_id"] for row in source_rows}
    for arm_key in ("probability_arm", "active_arm"):
        for row in ledger[arm_key]["rows"]:
            if thread_by_id.get(row["opaque_id"]) != row["thread_id"]:
                raise RuntimeError("source Parquet thread binding differs from acquisition ledger")
    return source_rows, ledger, _interleaved_rows(ledger)


def _packet_contract(
    *,
    source_path: Path,
    ledger_path: Path,
    runtime_source_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_rows, ledger, interleaved = _validate_source_and_ledger(source_path, ledger_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-teacher-acquisition-v2-packet-contract-v1",
        "acquisition_id": ledger["ledger_id"],
        "acquisition_ledger_sha256": file_sha256(ledger_path),
        "policy_file_sha256": ledger["policy_file_sha256"],
        "policy_contract_sha256": ledger["policy_contract_sha256"],
        "eligible_frame_sha256": ledger["input_bindings"]["eligible_frame_sha256"],
        "source_inventory_sha256": ledger["input_bindings"]["source_inventory_sha256"],
        "exclusion_ledger_sha256": ledger["input_bindings"]["exclusion_ledger_sha256"],
        "checkpoint_bundle_sha256": ledger["input_bindings"]["checkpoint_bundle_sha256"],
        "scoring_artifact_sha256": ledger["input_bindings"]["scoring_artifact_sha256"],
        "candidate_score_digest": ledger["candidate_score_digest"],
        "rare_cell_list_sha256": ledger["rare_cell_list_sha256"],
        "source_parquet_name": source_path.name,
        "source_parquet_sha256": file_sha256(source_path),
        "source_projection_sha256": canonical_sha256(source_rows),
        "source_rows": EXPECTED_ROWS,
        "unique_threads": EXPECTED_ROWS,
        "arm_counts": {arm: ARM_ROWS for arm in ARMS},
        "interleave_algorithm": INTERLEAVE_ALGORITHM,
        "interleave_order_sha256": canonical_sha256(
            [row["source_sample_id"] for row in interleaved]
        ),
        "runtime_source_bundle": (
            _source_bundle()
            if runtime_source_bundle is None
            else _validate_source_bundle(runtime_source_bundle)
        ),
        "provider_engine": {
            "module": "reddit_china_stance.sol_teacher_10k_v2",
            "historical_implementation_unchanged": True,
            "expected_rows": EXPECTED_ROWS,
        },
        "row_replacement_policy": "prohibited-after-ledger-freeze",
        "public_output_contract": "aggregate-metadata-only",
        "evidence_boundary": "silver-model-assisted-acquisition-labels-not-human-validation",
    }


def _mapping_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("provider_opaque_id", pa.string(), nullable=False),
            pa.field("source_sample_id", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
            pa.field("acquisition_arm", pa.string(), nullable=False),
            pa.field("arm_order", pa.int64(), nullable=False),
            pa.field("packet_order", pa.int64(), nullable=False),
            pa.field("inclusion_probability", pa.float64(), nullable=True),
            pa.field("active_bucket", pa.string(), nullable=True),
            pa.field("near_duplicate_cluster_id", pa.string(), nullable=False),
        ]
    )


def _write_provider_source(
    path: Path,
    *,
    source_rows: Sequence[Mapping[str, Any]],
    interleaved: Sequence[Mapping[str, Any]],
) -> None:
    by_id = {row["opaque_id"]: row for row in source_rows}
    records = [
        {
            "sample_id": source["opaque_id"],
            "thread_id": source["thread_id"],
            "target_text": source["target_text"],
            "submission_context": source["submission_context"],
            "parent_context": source["parent_context"],
        }
        for row in interleaved
        for source in (by_id[row["source_sample_id"]],)
    ]
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def _provider_packet_root(packet_root: Path) -> Path:
    roots = sorted((packet_root / "provider-packet").glob("packet=*"))
    if len(roots) != 1:
        raise RuntimeError("acquisition packet must contain exactly one provider packet")
    return roots[0]


def build_acquisition_packet(
    *,
    source_parquet_path: Path,
    acquisition_ledger_path: Path,
    output_parent: Path = DEFAULT_PACKET_PARENT,
    bridge_receipt_path: Path = teacher.DEFAULT_BRIDGE_RECEIPT,
) -> dict[str, Any]:
    """Build one exact arm-blinded packet and acquisition-private mapping."""

    source_rows, ledger, interleaved = _validate_source_and_ledger(
        source_parquet_path, acquisition_ledger_path
    )
    contract = _packet_contract(
        source_path=source_parquet_path,
        ledger_path=acquisition_ledger_path,
    )
    packet_id = canonical_sha256(contract)
    packet_root = output_parent / f"packet={packet_id}"
    if packet_root.exists():
        return validate_acquisition_packet(
            packet_root,
            source_parquet_path=source_parquet_path,
            acquisition_ledger_path=acquisition_ledger_path,
            bridge_receipt_path=bridge_receipt_path,
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{packet_root.name}.incomplete-", dir=output_parent)
    )
    try:
        ledger_copy = staging / "acquisition-ledger.json"
        shutil.copyfile(acquisition_ledger_path, ledger_copy)
        provider_source = staging / "provider-source.parquet"
        _write_provider_source(
            provider_source,
            source_rows=source_rows,
            interleaved=interleaved,
        )
        teacher.build_teacher_packet(
            source_parquet_path=provider_source,
            bridge_receipt_path=bridge_receipt_path,
            output_parent=staging / "provider-packet",
            expected_rows=EXPECTED_ROWS,
        )
        provider_packet = _provider_packet_root(staging)
        provider_mapping = pq.read_table(
            provider_packet / "private-mapping.parquet"
        ).to_pylist()
        if [row["sample_id"] for row in provider_mapping] != [
            row["source_sample_id"] for row in interleaved
        ]:
            raise RuntimeError("provider packet did not conserve acquisition interleave order")
        provider_by_source = {
            row["sample_id"]: row["opaque_id"] for row in provider_mapping
        }
        mapping_records = [
            {
                "provider_opaque_id": provider_by_source[row["source_sample_id"]],
                **row,
            }
            for row in interleaved
        ]
        mapping_path = staging / "private-mapping.parquet"
        pq.write_table(
            pa.Table.from_pylist(mapping_records, schema=_mapping_schema()),
            mapping_path,
            compression="zstd",
        )
        provider_manifest = _read_object(provider_packet / "manifest.json")
        manifest = {
            **contract,
            "kind": PACKET_KIND,
            "packet_id": packet_id,
            "acquisition_ledger_copy_sha256": file_sha256(ledger_copy),
            "provider_source_parquet_sha256": file_sha256(provider_source),
            "provider_packet_id": provider_manifest["packet_id"],
            "provider_packet_manifest_sha256": file_sha256(provider_packet / "manifest.json"),
            "private_mapping_sha256": file_sha256(mapping_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(teacher._json_bytes(manifest))
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sol-teacher-acquisition-v2-packet-receipt-v1",
            "status": "complete",
            "packet_id": packet_id,
            "acquisition_id": ledger["ledger_id"],
            "source_rows": EXPECTED_ROWS,
            "unique_threads": EXPECTED_ROWS,
            "arm_counts": {arm: ARM_ROWS for arm in ARMS},
            "source_parquet_sha256": contract["source_parquet_sha256"],
            "acquisition_ledger_sha256": contract["acquisition_ledger_sha256"],
            "policy_file_sha256": contract["policy_file_sha256"],
            "policy_contract_sha256": contract["policy_contract_sha256"],
            "eligible_frame_sha256": contract["eligible_frame_sha256"],
            "source_inventory_sha256": contract["source_inventory_sha256"],
            "exclusion_ledger_sha256": contract["exclusion_ledger_sha256"],
            "checkpoint_bundle_sha256": contract["checkpoint_bundle_sha256"],
            "scoring_artifact_sha256": contract["scoring_artifact_sha256"],
            "candidate_score_digest": contract["candidate_score_digest"],
            "interleave_order_sha256": contract["interleave_order_sha256"],
            "provider_packet_manifest_sha256": manifest[
                "provider_packet_manifest_sha256"
            ],
            "private_mapping_sha256": manifest["private_mapping_sha256"],
            "manifest_sha256": file_sha256(manifest_path),
            "runtime_source_bundle_digest": contract["runtime_source_bundle"]["digest"],
            "row_replacement_count": 0,
            "receipt_contains_raw_text": False,
            "receipt_contains_row_ids": False,
            "receipt_contains_thread_ids": False,
            "receipt_contains_row_level_labels": False,
            "evidence_boundary": contract["evidence_boundary"],
        }
        assert_metadata_only(receipt, where="sol-teacher-acquisition-v2-packet-receipt")
        receipt_id = canonical_sha256(receipt)
        (staging / f"receipt-{receipt_id}.json").write_bytes(teacher._json_bytes(receipt))
        os.replace(staging, packet_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_acquisition_packet(
        packet_root,
        source_parquet_path=source_parquet_path,
        acquisition_ledger_path=acquisition_ledger_path,
        bridge_receipt_path=bridge_receipt_path,
    )


def validate_acquisition_packet(
    packet_root: Path,
    *,
    source_parquet_path: Path,
    acquisition_ledger_path: Path,
    bridge_receipt_path: Path = teacher.DEFAULT_BRIDGE_RECEIPT,
) -> dict[str, Any]:
    manifest_path = packet_root / "manifest.json"
    mapping_path = packet_root / "private-mapping.parquet"
    ledger_copy = packet_root / "acquisition-ledger.json"
    provider_source = packet_root / "provider-source.parquet"
    required = (manifest_path, mapping_path, ledger_copy, provider_source)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("acquisition packet is incomplete")
    manifest = _read_object(manifest_path)
    contract = _packet_contract(
        source_path=source_parquet_path,
        ledger_path=acquisition_ledger_path,
        runtime_source_bundle=manifest.get("runtime_source_bundle"),
    )
    packet_id = canonical_sha256(contract)
    expected_manifest_contract = {key: manifest.get(key) for key in contract}
    expected_manifest_contract["kind"] = contract["kind"]
    if (
        manifest.get("kind") != PACKET_KIND
        or expected_manifest_contract != contract
        or manifest.get("packet_id") != packet_id
        or packet_root.name != f"packet={packet_id}"
        or file_sha256(ledger_copy) != file_sha256(acquisition_ledger_path)
        or manifest.get("acquisition_ledger_copy_sha256") != file_sha256(ledger_copy)
        or manifest.get("provider_source_parquet_sha256") != file_sha256(provider_source)
        or manifest.get("private_mapping_sha256") != file_sha256(mapping_path)
    ):
        raise RuntimeError("acquisition packet manifest or input binding failed")
    source_rows, ledger, interleaved = _validate_source_and_ledger(
        source_parquet_path, acquisition_ledger_path
    )
    provider_rows = teacher._load_source_rows(provider_source, expected_rows=EXPECTED_ROWS)
    source_by_id = {row["opaque_id"]: row for row in source_rows}
    expected_provider_rows = [
        {
            "sample_id": source["opaque_id"],
            "thread_id": source["thread_id"],
            "target_text": source["target_text"],
            "submission_context": source["submission_context"],
            "parent_context": source["parent_context"],
        }
        for row in interleaved
        for source in (source_by_id[row["source_sample_id"]],)
    ]
    if provider_rows != expected_provider_rows:
        raise RuntimeError("provider source differs from deterministic acquisition interleave")
    provider_packet = _provider_packet_root(packet_root)
    provider_receipt = teacher.validate_teacher_packet(
        provider_packet,
        source_parquet_path=provider_source,
        bridge_receipt_path=bridge_receipt_path,
        expected_rows=EXPECTED_ROWS,
    )
    if (
        manifest.get("provider_packet_id") != provider_receipt["packet_id"]
        or manifest.get("provider_packet_manifest_sha256")
        != file_sha256(provider_packet / "manifest.json")
    ):
        raise RuntimeError("provider packet binding drifted")
    provider_mapping = pq.read_table(
        provider_packet / "private-mapping.parquet"
    ).to_pylist()
    mapping = pq.read_table(mapping_path)
    if tuple(mapping.column_names) != tuple(field.name for field in _mapping_schema()):
        raise RuntimeError("acquisition private mapping columns drifted")
    provider_by_source = {row["sample_id"]: row["opaque_id"] for row in provider_mapping}
    expected_mapping = [
        {"provider_opaque_id": provider_by_source[row["source_sample_id"]], **row}
        for row in interleaved
    ]
    if mapping.to_pylist() != expected_mapping:
        raise RuntimeError("acquisition private mapping or order drifted")
    receipt_paths = sorted(packet_root.glob("receipt-*.json"))
    if len(receipt_paths) != 1:
        raise RuntimeError("acquisition packet must contain exactly one public receipt")
    receipt = _read_object(receipt_paths[0])
    if (
        receipt.get("kind") != "sol-teacher-acquisition-v2-packet-receipt-v1"
        or receipt.get("packet_id") != packet_id
        or receipt.get("acquisition_id") != ledger["ledger_id"]
        or receipt.get("manifest_sha256") != file_sha256(manifest_path)
        or receipt_paths[0].stem != f"receipt-{canonical_sha256(receipt)}"
        or receipt.get("arm_counts") != {arm: ARM_ROWS for arm in ARMS}
        or receipt.get("row_replacement_count") != 0
    ):
        raise RuntimeError("acquisition packet receipt binding failed")
    assert_metadata_only(receipt, where="sol-teacher-acquisition-v2-packet-receipt")
    return receipt


def make_run_contract(
    packet_root: Path,
    *,
    source_parquet_path: Path,
    acquisition_ledger_path: Path,
    bridge_receipt_path: Path = teacher.DEFAULT_BRIDGE_RECEIPT,
) -> dict[str, Any]:
    validate_acquisition_packet(
        packet_root,
        source_parquet_path=source_parquet_path,
        acquisition_ledger_path=acquisition_ledger_path,
        bridge_receipt_path=bridge_receipt_path,
    )
    manifest = _read_object(packet_root / "manifest.json")
    provider_packet = _provider_packet_root(packet_root)
    provider_source = packet_root / "provider-source.parquet"
    provider_contract = teacher.make_run_contract(
        provider_packet,
        source_parquet_path=provider_source,
        bridge_receipt_path=bridge_receipt_path,
        expected_rows=EXPECTED_ROWS,
    )
    provider_run_id = canonical_sha256(provider_contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "packet_id": manifest["packet_id"],
        "acquisition_id": manifest["acquisition_id"],
        "packet_manifest_sha256": file_sha256(packet_root / "manifest.json"),
        "acquisition_ledger_sha256": manifest["acquisition_ledger_sha256"],
        "policy_file_sha256": manifest["policy_file_sha256"],
        "policy_contract_sha256": manifest["policy_contract_sha256"],
        "eligible_frame_sha256": manifest["eligible_frame_sha256"],
        "source_inventory_sha256": manifest["source_inventory_sha256"],
        "exclusion_ledger_sha256": manifest["exclusion_ledger_sha256"],
        "checkpoint_bundle_sha256": manifest["checkpoint_bundle_sha256"],
        "scoring_artifact_sha256": manifest["scoring_artifact_sha256"],
        "candidate_score_digest": manifest["candidate_score_digest"],
        "source_parquet_sha256": manifest["source_parquet_sha256"],
        "interleave_order_sha256": manifest["interleave_order_sha256"],
        "private_mapping_sha256": manifest["private_mapping_sha256"],
        "provider_packet_id": manifest["provider_packet_id"],
        "provider_packet_manifest_sha256": manifest["provider_packet_manifest_sha256"],
        "provider_run_id": provider_run_id,
        "provider_run_contract_sha256": canonical_sha256(provider_contract),
        "arm_counts": {arm: ARM_ROWS for arm in ARMS},
        "runtime_source_bundle": manifest["runtime_source_bundle"],
        "row_replacement_policy": manifest["row_replacement_policy"],
        "public_output_contract": manifest["public_output_contract"],
        "evidence_boundary": manifest["evidence_boundary"],
    }


def _ensure_run(
    packet_root: Path,
    private_root: Path,
    *,
    source_parquet_path: Path,
    acquisition_ledger_path: Path,
    bridge_receipt_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    contract = make_run_contract(
        packet_root,
        source_parquet_path=source_parquet_path,
        acquisition_ledger_path=acquisition_ledger_path,
        bridge_receipt_path=bridge_receipt_path,
    )
    _validate_source_bundle(contract["runtime_source_bundle"], require_current_files=True)
    run_id = canonical_sha256(contract)
    run_root = private_root / f"run={run_id}"
    _write_immutable_json(run_root / "run-manifest.json", {**contract, "run_id": run_id})
    return (
        run_root,
        _provider_packet_root(packet_root),
        packet_root / "provider-source.parquet",
        {**contract, "run_id": run_id},
    )


def _provider_kwargs(
    packet_root: Path,
    private_root: Path,
    *,
    source_parquet_path: Path,
    acquisition_ledger_path: Path,
    bridge_receipt_path: Path,
) -> tuple[Path, dict[str, Any]]:
    run_root, provider_packet, provider_source, run_manifest = _ensure_run(
        packet_root,
        private_root,
        source_parquet_path=source_parquet_path,
        acquisition_ledger_path=acquisition_ledger_path,
        bridge_receipt_path=bridge_receipt_path,
    )
    kwargs = {
        "packet_root": provider_packet,
        "private_root": run_root / "provider-generation",
        "source_parquet_path": provider_source,
        "bridge_receipt_path": bridge_receipt_path,
        "expected_rows": EXPECTED_ROWS,
    }
    return run_root, {"run_manifest": run_manifest, "provider": kwargs}


def _validate_provider_result(result: Mapping[str, Any], run_manifest: Mapping[str, Any]) -> None:
    if result.get("run_id") != run_manifest["provider_run_id"]:
        raise RuntimeError("provider engine run ID differs from acquisition run binding")


def run_dual_passes(
    *,
    packet_root: Path,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    source_parquet_path: Path,
    acquisition_ledger_path: Path,
    bridge_receipt_path: Path = teacher.DEFAULT_BRIDGE_RECEIPT,
) -> dict[str, Any]:
    _, state = _provider_kwargs(
        packet_root,
        private_root,
        source_parquet_path=source_parquet_path,
        acquisition_ledger_path=acquisition_ledger_path,
        bridge_receipt_path=bridge_receipt_path,
    )
    result = teacher.run_dual_passes(**state["provider"])
    _validate_provider_result(result, state["run_manifest"])
    return {
        "run_id": state["run_manifest"]["run_id"],
        "pass_item_counts": result["pass_item_counts"],
        "automatic_retry_count": 0,
    }


def run_tie_break(**kwargs: Any) -> dict[str, Any]:
    packet_root = kwargs.pop("packet_root")
    private_root = kwargs.pop("private_root", DEFAULT_PRIVATE_ROOT)
    run_root, state = _provider_kwargs(packet_root, private_root, **kwargs)
    result = teacher.run_tie_break(**state["provider"])
    _validate_provider_result(result, state["run_manifest"])
    return {
        "run_id": run_root.name.removeprefix("run="),
        "tie_break_items": result["tie_break_items"],
        "automatic_retry_count": 0,
    }


def run_adjudication(**kwargs: Any) -> dict[str, Any]:
    packet_root = kwargs.pop("packet_root")
    private_root = kwargs.pop("private_root", DEFAULT_PRIVATE_ROOT)
    run_root, state = _provider_kwargs(packet_root, private_root, **kwargs)
    result = teacher.run_adjudication(**state["provider"])
    _validate_provider_result(result, state["run_manifest"])
    return {
        "run_id": run_root.name.removeprefix("run="),
        "adjudication_items": result["adjudication_items"],
        "candidate_order_digest": result["candidate_order_digest"],
        "automatic_retry_count": 0,
    }


def _provider_run_root(run_root: Path, run_manifest: Mapping[str, Any]) -> Path:
    path = run_root / "provider-generation" / f"run={run_manifest['provider_run_id']}"
    if not path.is_dir():
        raise FileNotFoundError("bound provider run output is missing")
    return path


def _combined_schema(run_manifest: Mapping[str, Any]) -> pa.Schema:
    metadata = {
        b"kind": b"sol-teacher-acquisition-v2-final-labels-v1",
        b"run_id": str(run_manifest["run_id"]).encode(),
        b"packet_id": str(run_manifest["packet_id"]).encode(),
        b"acquisition_id": str(run_manifest["acquisition_id"]).encode(),
        b"acquisition_ledger_sha256": str(
            run_manifest["acquisition_ledger_sha256"]
        ).encode(),
    }
    return pa.schema(
        [
            pa.field("source_sample_id", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
            pa.field("acquisition_arm", pa.string(), nullable=False),
            pa.field("arm_order", pa.int64(), nullable=False),
            pa.field("packet_order", pa.int64(), nullable=False),
            pa.field("codability", pa.string(), nullable=False),
            pa.field("relevance", pa.string(), nullable=True),
            pa.field("label_json", pa.string(), nullable=False),
            pa.field("quality_tier", pa.string(), nullable=False),
            pa.field("primary_training_eligible", pa.bool_(), nullable=False),
        ],
        metadata=metadata,
    )


def _final_material(
    packet_root: Path,
    run_root: Path,
    run_manifest: Mapping[str, Any],
    *,
    bridge_receipt_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    provider_run = _provider_run_root(run_root, run_manifest)
    provider_packet = _provider_packet_root(packet_root)
    _, blinded_rows = teacher._load_packet(
        provider_packet,
        source_parquet_path=packet_root / "provider-source.parquet",
        bridge_receipt_path=bridge_receipt_path,
        expected_rows=EXPECTED_ROWS,
    )
    first, second, final_rows, reconciliation = teacher._reconciled_outputs(
        provider_run, blinded_rows
    )
    mapping = pq.read_table(packet_root / "private-mapping.parquet").to_pylist()
    by_provider = {row["provider_opaque_id"]: row for row in mapping}
    first_by_id = {row["source_sample_id"]: row["label"] for row in first["rows"]}
    second_by_id = {row["source_sample_id"]: row["label"] for row in second["rows"]}
    combined: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    per_arm_rows: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        arm: [] for arm in ARMS
    }
    for final in final_rows:
        item_id = final["source_sample_id"]
        mapped = by_provider.get(item_id)
        if mapped is None:
            raise RuntimeError("final provider label is absent from acquisition mapping")
        label = validate_v2_label(final["label"])
        record = {
            "source_sample_id": mapped["source_sample_id"],
            "thread_id": mapped["thread_id"],
            "acquisition_arm": mapped["acquisition_arm"],
            "arm_order": mapped["arm_order"],
            "packet_order": mapped["packet_order"],
            "codability": label["codability"],
            "relevance": label["relevance"],
            "label_json": json.dumps(
                label,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "quality_tier": final["quality_tier"],
            "primary_training_eligible": final["primary_training_eligible"],
        }
        combined.append(record)
        per_arm_rows[mapped["acquisition_arm"]].append((record, {"provider_id": item_id}))
    if len(combined) != EXPECTED_ROWS or [row["packet_order"] for row in combined] != list(
        range(EXPECTED_ROWS)
    ):
        raise RuntimeError("acquisition final labels do not conserve packet rows or order")
    for arm in ARMS:
        rows = per_arm_rows[arm]
        if len(rows) != ARM_ROWS:
            raise RuntimeError("acquisition final labels do not conserve arm rows")
        codability = Counter(record["codability"] for record, _ in rows)
        quality = Counter(record["quality_tier"] for record, _ in rows)
        exact = sum(
            first_by_id[private["provider_id"]] == second_by_id[private["provider_id"]]
            for _, private in rows
        )
        eligible = sum(record["primary_training_eligible"] for record, _ in rows)
        diagnostics[arm] = {
            "query_count": ARM_ROWS,
            "dual_exact_agreement_count": exact,
            "dual_exact_agreement_rate": exact / ARM_ROWS,
            "codability_counts": dict(sorted(codability.items())),
            "quality_tier_counts": {
                tier: quality.get(tier, 0) for tier in teacher.QUALITY_TIERS
            },
            "blind_tie_break_count": ARM_ROWS - exact,
            "informed_adjudication_count": quality.get("informed_adjudication", 0),
            "primary_training_eligible_count": eligible,
            "primary_training_eligible_rate": eligible / ARM_ROWS,
            "row_replacement_count": 0,
            "provider_telemetry_attribution": {
                "available": False,
                "calls": None,
                "input_tokens": None,
                "cached_input_tokens": None,
                "output_tokens": None,
                "elapsed_seconds": None,
                "reason": "provider telemetry is emitted per mixed-arm shard",
            },
        }
    return combined, diagnostics, reconciliation


def _write_combined_labels(
    path: Path, rows: Sequence[Mapping[str, Any]], run_manifest: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=_combined_schema(run_manifest))
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        pq.write_table(table, temporary, compression="zstd")
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise RuntimeError("immutable acquisition final labels differ")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_combined_labels(
    path: Path, rows: Sequence[Mapping[str, Any]], run_manifest: Mapping[str, Any]
) -> None:
    if not path.is_file():
        raise FileNotFoundError("private acquisition final labels are missing")
    table = pq.read_table(path)
    expected_schema = _combined_schema(run_manifest)
    if table.schema != expected_schema or table.to_pylist() != list(rows):
        raise RuntimeError("private acquisition final labels binding drifted")


def _public_receipt(
    *,
    packet_root: Path,
    run_root: Path,
    run_manifest: Mapping[str, Any],
    labels_path: Path,
    diagnostics_path: Path,
    diagnostics: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    provider_receipt: Mapping[str, Any],
    provider_receipt_path: Path,
) -> dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-teacher-acquisition-v2-receipt-v1",
        "status": "complete",
        "run_id": run_manifest["run_id"],
        "packet_id": run_manifest["packet_id"],
        "acquisition_id": run_manifest["acquisition_id"],
        "row_count": EXPECTED_ROWS,
        "arm_counts": {arm: ARM_ROWS for arm in ARMS},
        "arm_diagnostics": dict(diagnostics),
        "blind_reconciliation": dict(reconciliation),
        "global_provider_telemetry": {
            "execution_count": provider_receipt["execution_count"],
            "usage_totals": provider_receipt["usage_totals"],
            "elapsed_seconds_sum": provider_receipt["elapsed_seconds_sum"],
        },
        "source_parquet_sha256": run_manifest["source_parquet_sha256"],
        "acquisition_ledger_sha256": run_manifest["acquisition_ledger_sha256"],
        "policy_file_sha256": run_manifest["policy_file_sha256"],
        "policy_contract_sha256": run_manifest["policy_contract_sha256"],
        "eligible_frame_sha256": run_manifest["eligible_frame_sha256"],
        "source_inventory_sha256": run_manifest["source_inventory_sha256"],
        "exclusion_ledger_sha256": run_manifest["exclusion_ledger_sha256"],
        "checkpoint_bundle_sha256": run_manifest["checkpoint_bundle_sha256"],
        "scoring_artifact_sha256": run_manifest["scoring_artifact_sha256"],
        "candidate_score_digest": run_manifest["candidate_score_digest"],
        "interleave_order_sha256": run_manifest["interleave_order_sha256"],
        "packet_manifest_sha256": run_manifest["packet_manifest_sha256"],
        "private_mapping_sha256": run_manifest["private_mapping_sha256"],
        "private_labels_parquet_sha256": file_sha256(labels_path),
        "private_diagnostics_sha256": file_sha256(diagnostics_path),
        "provider_packet_id": run_manifest["provider_packet_id"],
        "provider_run_id": run_manifest["provider_run_id"],
        "provider_receipt_sha256": file_sha256(provider_receipt_path),
        "runtime_source_bundle_digest": run_manifest["runtime_source_bundle"]["digest"],
        "automatic_retry_count": 0,
        "row_replacement_count": 0,
        "receipt_contains_raw_text": False,
        "receipt_contains_row_ids": False,
        "receipt_contains_thread_ids": False,
        "receipt_contains_row_level_labels": False,
        "evidence_boundary": run_manifest["evidence_boundary"],
    }
    assert_metadata_only(receipt, where="sol-teacher-acquisition-v2-public-receipt")
    return receipt


def _finalise_or_validate(
    *,
    packet_root: Path,
    private_root: Path,
    public_root: Path,
    source_parquet_path: Path,
    acquisition_ledger_path: Path,
    bridge_receipt_path: Path,
    publish: bool,
) -> dict[str, Any]:
    run_root, state = _provider_kwargs(
        packet_root,
        private_root,
        source_parquet_path=source_parquet_path,
        acquisition_ledger_path=acquisition_ledger_path,
        bridge_receipt_path=bridge_receipt_path,
    )
    run_manifest = state["run_manifest"]
    provider_public = run_root / "provider-public-private"
    if publish:
        provider_receipt = teacher.finalise_teacher_labels(
            **state["provider"], public_root=provider_public
        )
    else:
        provider_receipt = teacher.validate_final_teacher_labels(
            **state["provider"], public_root=provider_public
        )
    provider_receipt_paths = sorted(
        (provider_public / f"run={run_manifest['provider_run_id']}").glob("receipt-*.json")
    )
    if len(provider_receipt_paths) != 1:
        raise RuntimeError("bound private provider receipt is missing or duplicated")
    combined, diagnostics, reconciliation = _final_material(
        packet_root,
        run_root,
        run_manifest,
        bridge_receipt_path=bridge_receipt_path,
    )
    labels_path = run_root / "final/acquisition-labels.parquet"
    diagnostics_path = run_root / "final/arm-diagnostics.json"
    if publish:
        _write_combined_labels(labels_path, combined, run_manifest)
        _write_immutable_json(
            diagnostics_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "sol-teacher-acquisition-v2-private-arm-diagnostics-v1",
                "run_id": run_manifest["run_id"],
                "packet_id": run_manifest["packet_id"],
                "acquisition_id": run_manifest["acquisition_id"],
                "arms": diagnostics,
                "row_replacement_count": 0,
            },
        )
    _validate_combined_labels(labels_path, combined, run_manifest)
    expected_diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-teacher-acquisition-v2-private-arm-diagnostics-v1",
        "run_id": run_manifest["run_id"],
        "packet_id": run_manifest["packet_id"],
        "acquisition_id": run_manifest["acquisition_id"],
        "arms": diagnostics,
        "row_replacement_count": 0,
    }
    if _read_object(diagnostics_path) != expected_diagnostics:
        raise RuntimeError("private acquisition arm diagnostics drifted")
    receipt = _public_receipt(
        packet_root=packet_root,
        run_root=run_root,
        run_manifest=run_manifest,
        labels_path=labels_path,
        diagnostics_path=diagnostics_path,
        diagnostics=diagnostics,
        reconciliation=reconciliation,
        provider_receipt=provider_receipt,
        provider_receipt_path=provider_receipt_paths[0],
    )
    output_root = public_root / f"run={run_manifest['run_id']}"
    if publish:
        _write_immutable_json(
            output_root / f"receipt-{canonical_sha256(receipt)}.json", receipt
        )
    receipt_paths = sorted(output_root.glob("receipt-*.json"))
    if len(receipt_paths) != 1:
        raise RuntimeError("acquisition public output must contain exactly one receipt")
    observed = _read_object(receipt_paths[0])
    if observed != receipt or receipt_paths[0].stem != f"receipt-{canonical_sha256(observed)}":
        raise RuntimeError("acquisition public receipt binding drifted")
    assert_metadata_only(observed, where="sol-teacher-acquisition-v2-public-receipt")
    return observed


def finalise_acquisition_labels(**kwargs: Any) -> dict[str, Any]:
    return _finalise_or_validate(publish=True, **kwargs)


def validate_final_acquisition_labels(**kwargs: Any) -> dict[str, Any]:
    return _finalise_or_validate(publish=False, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        required=True,
        choices=(
            "prepare-packet",
            "validate-packet",
            "run-dual",
            "run-tie-break",
            "run-adjudication",
            "finalise",
            "validate-final",
        ),
    )
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--acquisition-ledger", type=Path, required=True)
    parser.add_argument("--packet-parent", type=Path, default=DEFAULT_PACKET_PARENT)
    parser.add_argument("--packet-root", type=Path)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--bridge-receipt", type=Path, default=teacher.DEFAULT_BRIDGE_RECEIPT)
    args = parser.parse_args(argv)
    common = {
        "source_parquet_path": args.source_parquet,
        "acquisition_ledger_path": args.acquisition_ledger,
        "bridge_receipt_path": args.bridge_receipt,
    }
    if args.action == "prepare-packet":
        result = build_acquisition_packet(output_parent=args.packet_parent, **common)
    else:
        if args.packet_root is None:
            parser.error("--packet-root is required for this action")
        if args.action == "validate-packet":
            result = validate_acquisition_packet(args.packet_root, **common)
        elif args.action == "run-dual":
            result = run_dual_passes(
                packet_root=args.packet_root, private_root=args.private_root, **common
            )
        elif args.action == "run-tie-break":
            result = run_tie_break(
                packet_root=args.packet_root, private_root=args.private_root, **common
            )
        elif args.action == "run-adjudication":
            result = run_adjudication(
                packet_root=args.packet_root, private_root=args.private_root, **common
            )
        elif args.action == "finalise":
            result = finalise_acquisition_labels(
                packet_root=args.packet_root,
                private_root=args.private_root,
                public_root=args.public_root,
                **common,
            )
        else:
            result = validate_final_acquisition_labels(
                packet_root=args.packet_root,
                private_root=args.private_root,
                public_root=args.public_root,
                **common,
            )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PACKET_PARENT",
    "DEFAULT_PRIVATE_ROOT",
    "DEFAULT_PUBLIC_ROOT",
    "build_acquisition_packet",
    "finalise_acquisition_labels",
    "make_run_contract",
    "run_adjudication",
    "run_dual_passes",
    "run_tie_break",
    "validate_acquisition_ledger",
    "validate_acquisition_packet",
    "validate_final_acquisition_labels",
]
