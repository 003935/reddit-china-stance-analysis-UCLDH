"""Candidate-v2 ontology and deterministic Phase-0 pilot packet.

The packet builder reads only the existing private 10k teacher Parquet and
publishes a blinded 480-row surface plus a private selection mapping.  Public
receipts contain aggregate metadata only.  No provider execution lives here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from jsonschema import Draft202012Validator

from reddit_china_stance.human_seeded_consensus_v1 import (
    load_label_schema as load_v1_label_schema,
)
from reddit_china_stance.human_seeded_consensus_v1 import (
    validate_semantic_label as validate_v1_semantic_label,
)
from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
PACKET_KIND = "semantic-ontology-v2-pilot-packet-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas/target-stance-v2-pilot.schema.json"
RUBRIC_PATH = REPO_ROOT / "docs/rubrics/target-stance-v2-pilot.md"
DEFAULT_SOURCE_PARQUET = (
    REPO_ROOT / "data/private-hf-sol-teacher-10k-v1/data/train-00000-of-00001.parquet"
)
DEFAULT_PRIVATE_ROOT = REPO_ROOT / "data/private-semantic-ontology-v2/pilot"

TARGETS = (
    "china_general",
    "government_ccp",
    "people_identity",
    "culture_media",
    "company_tech_product",
    "residual_other",
)
ANALYTIC_TARGETS = TARGETS[:-1]
STANCES = ("negative", "positive", "mixed", "no_directed_stance")
CODABILITY = ("codable", "not_codable")
RELEVANCE = ("material", "not_material")
LEGACY_TARGETS = ("china_general", "government_ccp", "people_culture", "other")
STRATUM_QUOTAS = {
    "people_only": 160,
    "other_only": 160,
    "both": 80,
    "core_controls": 80,
}
PILOT_ROWS = sum(STRATUM_QUOTAS.values())
SAMPLING_SEED = "semantic-ontology-v2-bridge-phase0"
SELECTION_METHOD = "proportional-largest-remainder-within-metadata-stance-cells-v1"
BLINDNESS_CONTRACT = {
    "legacy_v1_labels_hidden": True,
    "source_metadata_hidden": True,
    "selection_stratum_hidden": True,
    "selection_reason_hidden": True,
    "canonical_sample_ids_hidden": True,
    "thread_ids_hidden": True,
    "repository_context_not_provided": True,
    "web_use_prohibited": True,
}

SOURCE_REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "thread_id",
        "target_text",
        "submission_context",
        "parent_context",
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
        "relevance",
        "label_json",
        *(f"has_target_{target}" for target in LEGACY_TARGETS),
        *(f"stance_{target}" for target in LEGACY_TARGETS),
    }
)
BLINDED_ROW_FIELDS = frozenset(
    {"source_sample_id", "target_text", "submission_context", "parent_context"}
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"stale immutable temporary exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_v2_label_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    schema = _read_json_object(path)
    Draft202012Validator.check_schema(schema)
    return schema


def load_v2_label_validator(path: Path = SCHEMA_PATH) -> Draft202012Validator:
    """Load and compile the pinned v2 label schema once for batch validation."""

    return Draft202012Validator(load_v2_label_schema(path))


def validate_v2_label(
    value: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
    validator: Draft202012Validator | None = None,
) -> dict[str, Any]:
    """Validate and canonicalise one candidate-v2 label without coercion."""

    if not isinstance(value, Mapping):
        raise ValueError("v2 label must be an object")
    clean = deepcopy(dict(value))
    if schema is not None and validator is not None:
        raise ValueError("provide either schema or validator, not both")
    active_validator = validator or Draft202012Validator(
        dict(schema or load_v2_label_schema())
    )
    errors = sorted(
        active_validator.iter_errors(clean),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"invalid v2 label: {errors[0].message}")
    if clean["codability"] not in CODABILITY:
        raise ValueError("invalid v2 label: unsupported codability")
    relevance = clean["relevance"]
    targets = clean["targets"]
    names = [item["target"] for item in targets]
    if len(names) != len(set(names)):
        raise ValueError("invalid v2 label: duplicate target entries")
    if clean["codability"] == "not_codable":
        if relevance is not None or targets:
            raise ValueError(
                "invalid v2 label: not_codable requires null relevance and no targets"
            )
    elif relevance not in RELEVANCE:
        raise ValueError("invalid v2 label: codable requires a relevance decision")
    elif relevance == "material" and not targets:
        raise ValueError("invalid v2 label: codable material requires at least one target")
    elif relevance == "not_material" and targets:
        raise ValueError("invalid v2 label: codable not_material requires no targets")
    order = {target: index for index, target in enumerate(TARGETS)}
    clean["targets"] = sorted(
        (dict(item) for item in targets),
        key=lambda item: order[item["target"]],
    )
    return clean


def _required_text(row: Mapping[str, Any], field: str, *, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"source row {index} has invalid {field}")
    return value


def _validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_source_rows: int | None,
) -> list[dict[str, Any]]:
    if expected_source_rows is not None and len(rows) != expected_source_rows:
        raise ValueError(
            f"source row count drifted: expected {expected_source_rows}, got {len(rows)}"
        )
    if not rows:
        raise ValueError("source rows cannot be empty")
    v1_schema = load_v1_label_schema()
    clean: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    thread_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or not set(raw) >= SOURCE_REQUIRED_COLUMNS:
            raise ValueError(f"source row {index} is missing required private 10k fields")
        row = dict(raw)
        sample_id = _required_text(row, "sample_id", index=index)
        thread_id = _required_text(row, "thread_id", index=index)
        _required_text(row, "target_text", index=index)
        for field in ("submission_context", "parent_context"):
            if row[field] is not None and not isinstance(row[field], str):
                raise ValueError(f"source row {index} has invalid {field}")
        for field in ("subreddit", "content_type", "retrieval_mode"):
            _required_text(row, field, index=index)
        if type(row["year"]) is not int:
            raise ValueError(f"source row {index} has invalid year")
        if sample_id in sample_ids:
            raise ValueError("source contains duplicate sample IDs")
        if thread_id in thread_ids:
            raise ValueError("source must remain one-row-per-thread")
        sample_ids.add(sample_id)
        thread_ids.add(thread_id)
        try:
            parsed_label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"source row {index} has invalid label_json") from exc
        label = validate_v1_semantic_label(parsed_label, schema=v1_schema)
        if row["relevance"] != label["relevance"]:
            raise ValueError(f"source row {index} relevance disagrees with label_json")
        target_stances = {item["target"]: item["stance"] for item in label["target_stances"]}
        for target in LEGACY_TARGETS:
            present = row[f"has_target_{target}"]
            if type(present) is not bool or present is not (target in target_stances):
                raise ValueError(f"source row {index} target flags disagree with label_json")
            if row[f"stance_{target}"] != target_stances.get(target):
                raise ValueError(f"source row {index} stance fields disagree with label_json")
        clean.append(row)
    return clean


def _legacy_stratum(row: Mapping[str, Any]) -> str | None:
    if row["relevance"] != "material":
        return None
    people = row["has_target_people_culture"]
    other = row["has_target_other"]
    if people and other:
        return "both"
    if people:
        return "people_only"
    if other:
        return "other_only"
    if row["has_target_china_general"] or row["has_target_government_ccp"]:
        return "core_controls"
    return None


def _stance_signature(row: Mapping[str, Any], *, stratum: str) -> str:
    fields = {
        "people_only": ("people_culture",),
        "other_only": ("other",),
        "both": ("people_culture", "other"),
        "core_controls": ("china_general", "government_ccp"),
    }[stratum]
    return "|".join(f"{target}:{row[f'stance_{target}'] or 'absent'}" for target in fields)


def _selection_cell(row: Mapping[str, Any], *, stratum: str) -> str:
    values = (
        row["subreddit"],
        row["year"],
        row["content_type"],
        row["retrieval_mode"],
        _stance_signature(row, stratum=stratum),
    )
    return canonical_sha256(values)


def _allocate_proportionally(
    cell_sizes: Mapping[str, int],
    *,
    quota: int,
    seed: str,
    stratum: str,
) -> dict[str, int]:
    total = sum(cell_sizes.values())
    if quota <= 0 or total < quota:
        raise ValueError(f"stratum {stratum} cannot satisfy quota {quota} from population {total}")
    exact = {cell: quota * size / total for cell, size in cell_sizes.items()}
    allocated = {cell: math.floor(value) for cell, value in exact.items()}
    remaining = quota - sum(allocated.values())
    candidates = sorted(
        cell_sizes,
        key=lambda cell: (
            -(exact[cell] - allocated[cell]),
            hashlib.sha256(f"{seed}\0{stratum}\0{cell}".encode()).hexdigest(),
        ),
    )
    for cell in candidates[:remaining]:
        allocated[cell] += 1
    if sum(allocated.values()) != quota or any(
        allocation > cell_sizes[cell] for cell, allocation in allocated.items()
    ):
        raise RuntimeError("deterministic stratum allocation failed")
    return allocated


def select_pilot_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    seed: str = SAMPLING_SEED,
    quotas: Mapping[str, int] = STRATUM_QUOTAS,
    expected_source_rows: int | None = 10_000,
) -> list[dict[str, Any]]:
    """Select the exact deterministic, thread-disjoint Phase-0 sample."""

    if not isinstance(seed, str) or not seed:
        raise ValueError("sampling seed must be a non-empty string")
    if set(quotas) != set(STRATUM_QUOTAS):
        raise ValueError("quotas must contain exactly the four registered legacy strata")
    if any(type(value) is not int or value <= 0 for value in quotas.values()):
        raise ValueError("all stratum quotas must be positive integers")
    rows = _validate_source_rows(source_rows, expected_source_rows=expected_source_rows)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = _legacy_stratum(row)
        if stratum is not None:
            by_stratum[stratum].append(row)

    selected: list[dict[str, Any]] = []
    for stratum in STRATUM_QUOTAS:
        population_rows = by_stratum[stratum]
        quota = int(quotas[stratum])
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in population_rows:
            by_cell[_selection_cell(row, stratum=stratum)].append(row)
        allocations = _allocate_proportionally(
            {cell: len(cell_rows) for cell, cell_rows in by_cell.items()},
            quota=quota,
            seed=seed,
            stratum=stratum,
        )
        for cell in sorted(by_cell):
            cell_rows = sorted(
                by_cell[cell],
                key=lambda row: hashlib.sha256(
                    f"{seed}\0{stratum}\0{cell}\0{row['sample_id']}".encode()
                ).hexdigest(),
            )
            allocation = allocations[cell]
            for row in cell_rows[:allocation]:
                selected.append(
                    {
                        **row,
                        "selection_stratum": stratum,
                        "selection_cell": cell,
                        "selection_cell_population": len(cell_rows),
                        "selection_cell_selected": allocation,
                        "inclusion_probability_numerator": allocation,
                        "inclusion_probability_denominator": len(cell_rows),
                        "inclusion_probability": allocation / len(cell_rows),
                    }
                )
    expected_selected = sum(quotas.values())
    if len(selected) != expected_selected:
        raise RuntimeError("selected pilot row count drifted")
    if len({row["thread_id"] for row in selected}) != len(selected):
        raise RuntimeError("selected pilot is not thread-disjoint")
    stratum_counts = Counter(row["selection_stratum"] for row in selected)
    if stratum_counts != Counter(quotas):
        raise RuntimeError("selected pilot stratum quotas drifted")
    selected.sort(
        key=lambda row: (
            tuple(STRATUM_QUOTAS).index(row["selection_stratum"]),
            hashlib.sha256(f"{seed}\0packet\0{row['sample_id']}".encode()).hexdigest(),
        )
    )
    return selected


def _packet_contract(
    *,
    source_path: Path,
    seed: str,
    expected_source_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "semantic-ontology-v2-pilot-contract-v1",
        "source_parquet_sha256": file_sha256(source_path),
        "source_rows": expected_source_rows,
        "target_rows": PILOT_ROWS,
        "stratum_quotas": dict(STRATUM_QUOTAS),
        "sampling_seed": seed,
        "selection_method": SELECTION_METHOD,
        "rubric_sha256": file_sha256(RUBRIC_PATH),
        "label_schema_sha256": file_sha256(SCHEMA_PATH),
        "sampling_unit": "submission_thread",
        "balance_fields": [
            "subreddit",
            "year",
            "content_type",
            "retrieval_mode",
            "legacy_stance_signature",
        ],
    }


def build_pilot_packet(
    *,
    source_parquet_path: Path = DEFAULT_SOURCE_PARQUET,
    output_root: Path = DEFAULT_PRIVATE_ROOT,
    seed: str = SAMPLING_SEED,
    expected_source_rows: int = 10_000,
) -> dict[str, Any]:
    """Build or exact-validate the immutable private Phase-0 packet."""

    if not source_parquet_path.is_file():
        raise FileNotFoundError(f"private 10k Parquet is missing: {source_parquet_path}")
    parquet = pq.ParquetFile(source_parquet_path)
    missing = SOURCE_REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"private 10k Parquet is missing required columns: {sorted(missing)}")
    source_rows = parquet.read(columns=sorted(SOURCE_REQUIRED_COLUMNS)).to_pylist()
    contract = _packet_contract(
        source_path=source_parquet_path,
        seed=seed,
        expected_source_rows=expected_source_rows,
    )
    contract_id = canonical_sha256(contract)
    packet_root = output_root / f"packet={contract_id}"
    if packet_root.exists():
        return validate_pilot_packet(packet_root)
    selected = select_pilot_rows(
        source_rows,
        seed=seed,
        expected_source_rows=expected_source_rows,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{packet_root.name}.incomplete-", dir=output_root))
    try:
        blinded_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        for packet_order, row in enumerate(selected):
            opaque_id = "O2" + hashlib.sha256(
                f"{contract_id}\0{row['sample_id']}".encode()
            ).hexdigest()[:24]
            blinded_rows.append(
                {
                    "source_sample_id": opaque_id,
                    "target_text": row["target_text"],
                    "submission_context": row["submission_context"],
                    "parent_context": row["parent_context"],
                }
            )
            mapping_rows.append(
                {
                    "opaque_id": opaque_id,
                    "sample_id": row["sample_id"],
                    "thread_id": row["thread_id"],
                    "packet_order": packet_order,
                    "selection_stratum": row["selection_stratum"],
                    "selection_cell": row["selection_cell"],
                    "selection_cell_population": row["selection_cell_population"],
                    "selection_cell_selected": row["selection_cell_selected"],
                    "inclusion_probability_numerator": row[
                        "inclusion_probability_numerator"
                    ],
                    "inclusion_probability_denominator": row[
                        "inclusion_probability_denominator"
                    ],
                    "inclusion_probability": row["inclusion_probability"],
                    "subreddit": row["subreddit"],
                    "year": row["year"],
                    "content_type": row["content_type"],
                    "retrieval_mode": row["retrieval_mode"],
                    "legacy_label_json": row["label_json"],
                }
            )
        blinded = {
            "schema_version": SCHEMA_VERSION,
            "kind": "semantic-ontology-v2-blinded-input-v1",
            "packet_id": contract_id,
            "rubric_sha256": contract["rubric_sha256"],
            "label_schema_sha256": contract["label_schema_sha256"],
            "blindness": dict(BLINDNESS_CONTRACT),
            "rows": blinded_rows,
        }
        blinded_path = staging / "blinded-input.json"
        blinded_path.write_bytes(_json_bytes(blinded))
        mapping_path = staging / "private-mapping.parquet"
        pq.write_table(pa.Table.from_pylist(mapping_rows), mapping_path, compression="zstd")
        manifest = {
            **contract,
            "kind": PACKET_KIND,
            "packet_id": contract_id,
            "blinded_input_sha256": file_sha256(blinded_path),
            "private_mapping_sha256": file_sha256(mapping_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        stratum_population = Counter(
            stratum
            for row in source_rows
            if (stratum := _legacy_stratum(row)) is not None
        )
        probabilities = [row["inclusion_probability"] for row in mapping_rows]
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "semantic-ontology-v2-pilot-packet-receipt-v1",
            "status": "complete",
            "packet_id": contract_id,
            "source_rows": expected_source_rows,
            "selected_rows": len(mapping_rows),
            "unique_threads": len({row["thread_id"] for row in mapping_rows}),
            "stratum_population_counts": dict(sorted(stratum_population.items())),
            "stratum_selected_counts": dict(
                sorted(Counter(row["selection_stratum"] for row in mapping_rows).items())
            ),
            "selection_cell_count": len({row["selection_cell"] for row in mapping_rows}),
            "inclusion_probability_min": min(probabilities),
            "inclusion_probability_max": max(probabilities),
            "blinded_input_sha256": manifest["blinded_input_sha256"],
            "private_mapping_sha256": manifest["private_mapping_sha256"],
            "manifest_sha256": file_sha256(manifest_path),
            "receipt_contains_raw_text": False,
            "receipt_contains_row_ids": False,
            "receipt_contains_thread_ids": False,
            "receipt_contains_row_level_labels": False,
        }
        assert_metadata_only(receipt, where="semantic-ontology-v2-pilot-receipt")
        receipt_id = canonical_sha256(receipt)
        (staging / f"receipt-{receipt_id}.json").write_bytes(_json_bytes(receipt))
        packet_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, packet_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_pilot_packet(packet_root)


def validate_pilot_packet(packet_root: Path) -> dict[str, Any]:
    """Validate exact row conservation, blindness and immutable packet bindings."""

    manifest_path = packet_root / "manifest.json"
    blinded_path = packet_root / "blinded-input.json"
    mapping_path = packet_root / "private-mapping.parquet"
    receipt_paths = sorted(packet_root.glob("receipt-*.json"))
    if not all(path.is_file() for path in (manifest_path, blinded_path, mapping_path)):
        raise FileNotFoundError("pilot packet is incomplete")
    if len(receipt_paths) != 1:
        raise RuntimeError("pilot packet must contain exactly one receipt")
    manifest = _read_json_object(manifest_path)
    blinded = _read_json_object(blinded_path)
    receipt = _read_json_object(receipt_paths[0])
    frozen_contract = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "packet_id",
            "blinded_input_sha256",
            "private_mapping_sha256",
        }
    }
    frozen_contract["kind"] = "semantic-ontology-v2-pilot-contract-v1"
    if (
        manifest.get("kind") != PACKET_KIND
        or canonical_sha256(frozen_contract) != manifest.get("packet_id")
        or packet_root.name != f"packet={manifest.get('packet_id')}"
        or manifest.get("rubric_sha256") != file_sha256(RUBRIC_PATH)
        or manifest.get("label_schema_sha256") != file_sha256(SCHEMA_PATH)
        or manifest.get("blinded_input_sha256") != file_sha256(blinded_path)
        or manifest.get("private_mapping_sha256") != file_sha256(mapping_path)
    ):
        raise RuntimeError("pilot packet manifest binding failed")
    rows = blinded.get("rows")
    if (
        blinded.get("kind") != "semantic-ontology-v2-blinded-input-v1"
        or blinded.get("packet_id") != manifest["packet_id"]
        or blinded.get("rubric_sha256") != manifest["rubric_sha256"]
        or blinded.get("label_schema_sha256") != manifest["label_schema_sha256"]
        or blinded.get("blindness") != BLINDNESS_CONTRACT
        or not isinstance(rows, list)
        or len(rows) != PILOT_ROWS
        or any(not isinstance(row, Mapping) or set(row) != BLINDED_ROW_FIELDS for row in rows)
    ):
        raise RuntimeError("pilot blinded-input contract failed")
    ids = [row["source_sample_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise RuntimeError("pilot blinded opaque IDs are invalid")
    mapping = pq.read_table(mapping_path).to_pylist()
    mapping.sort(key=lambda row: row["packet_order"])
    if (
        len(mapping) != PILOT_ROWS
        or [row["opaque_id"] for row in mapping] != ids
        or len({row["thread_id"] for row in mapping}) != PILOT_ROWS
        or Counter(row["selection_stratum"] for row in mapping) != Counter(STRATUM_QUOTAS)
    ):
        raise RuntimeError("pilot private mapping row conservation failed")
    for row in mapping:
        numerator = row["inclusion_probability_numerator"]
        denominator = row["inclusion_probability_denominator"]
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or not 0 < numerator <= denominator
            or not math.isclose(row["inclusion_probability"], numerator / denominator)
        ):
            raise RuntimeError("pilot inclusion-probability binding failed")
    if (
        receipt.get("kind") != "semantic-ontology-v2-pilot-packet-receipt-v1"
        or receipt.get("status") != "complete"
        or receipt.get("packet_id") != manifest["packet_id"]
        or receipt.get("selected_rows") != PILOT_ROWS
        or receipt.get("unique_threads") != PILOT_ROWS
        or receipt.get("manifest_sha256") != file_sha256(manifest_path)
        or receipt.get("blinded_input_sha256") != file_sha256(blinded_path)
        or receipt.get("private_mapping_sha256") != file_sha256(mapping_path)
        or receipt_paths[0].stem != f"receipt-{canonical_sha256(receipt)}"
    ):
        raise RuntimeError("pilot metadata-only receipt binding failed")
    assert_metadata_only(receipt, where="semantic-ontology-v2-pilot-receipt")
    return receipt


__all__ = [
    "ANALYTIC_TARGETS",
    "BLINDNESS_CONTRACT",
    "CODABILITY",
    "DEFAULT_PRIVATE_ROOT",
    "DEFAULT_SOURCE_PARQUET",
    "PILOT_ROWS",
    "RELEVANCE",
    "RUBRIC_PATH",
    "SAMPLING_SEED",
    "SCHEMA_PATH",
    "STANCES",
    "STRATUM_QUOTAS",
    "TARGETS",
    "build_pilot_packet",
    "canonical_sha256",
    "file_sha256",
    "load_v2_label_schema",
    "select_pilot_rows",
    "validate_pilot_packet",
    "validate_v2_label",
]
