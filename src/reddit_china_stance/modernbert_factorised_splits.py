"""Prepare deterministic private evidence frames for the factorised v2 students.

This module is deliberately data- and model-runtime independent.  It consumes a
private, already joined view of the exact 10k v2 teacher ledger and source mapping,
then emits a private membership ledger plus a metadata-only public manifest.

No real manifest is built at import time.  In particular, this module does not
authorise use of the legacy 222/230-row proxies or construction of the separate
future human-evaluation frame.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation_v2 import (
    EVALUATION_SCHEMA_VERSION,
    NATURAL_DESIGN_DIGEST_FIELDS,
    WEIGHTING_ESTIMATOR,
    natural_arm_design_digest,
)
from reddit_china_stance.semantic_ontology_v2 import ANALYTIC_TARGETS, validate_v2_label

SCHEMA_VERSION = "1.0.0"
PUBLIC_KIND = "modernbert-factorised-evidence-frames-v1"
PRIVATE_KIND = "modernbert-factorised-private-membership-v1"
TEST_PUBLIC_KIND = "modernbert-factorised-evidence-frames-test-only-v1"
TEST_PRIVATE_KIND = "modernbert-factorised-private-membership-test-only-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTERED_TEST_PRIVATE_PARENT = REPO_ROOT / "data/private-modernbert-factorised-v2/tests"
REGISTERED_TEST_PUBLIC_PARENT = REPO_ROOT / "outputs/modernbert-factorised-v2/tests"

QUALITY_TIERS = ("exact_consensus", "blind_majority", "informed_adjudication")
PRIMARY_QUALITY_TIERS = frozenset({"exact_consensus", "blind_majority"})
SUPPORTED_YEARS = frozenset(range(2020, 2026))
SUPPORTED_CONTENT_TYPES = frozenset({"comment", "submission"})
METADATA_STRATUM_FIELDS = ("year", "subreddit", "content_type")
PROBABILITY_ALLOCATION_ALGORITHM = (
    "minimum-one-per-observed-stratum-residual-capacity-largest-remainder-srswor-v1"
)
RARE_ENRICHMENT_ALGORITHM = (
    "post-natural-source15-exclusive10-rare-first-residual-capacity-v2"
)
SOURCE_ROW_HASH_FIELDS = (
    "sample_id",
    "thread_id",
    "target_text",
    "submission_context",
    "parent_context",
    "subreddit",
    "year",
    "content_type",
    "retrieval_mode",
)
PROXY_SURFACE_HASH_FIELDS = (
    "target_text",
    "parent_context",
    "submission_context",
)
MAPPING_ROW_HASH_FIELDS = ("sample_id", "thread_id", "opaque_id")
TEACHER_ROW_HASH_FIELDS = (
    "opaque_id",
    "label_sha256",
    "quality_tier",
    "primary_training_eligible",
)
JOINED_ROW_HASH_FIELDS = (
    "source_row_sha256",
    "mapping_row_sha256",
    "teacher_row_sha256",
)

REQUIRED_INPUT_BINDINGS = frozenset(
    {
        "source_receipt_sha256",
        "source_metadata_mapping_sha256",
        "teacher_receipt_sha256",
        "teacher_final_ledger_sha256",
        "bridge_receipt_sha256",
        "bridge_exposure_register_file_sha256",
        "bridge_exposure_register_content_sha256",
        "legacy_development_proxy_file_sha256",
        "legacy_development_proxy_schema_metadata_sha256",
        "legacy_development_proxy_sample_id_set_sha256",
        "legacy_development_proxy_surface_set_sha256",
        "legacy_locked_proxy_file_sha256",
        "legacy_locked_proxy_schema_metadata_sha256",
        "legacy_locked_proxy_sample_id_set_sha256",
        "legacy_locked_proxy_surface_set_sha256",
    }
)

REQUIRED_ROW_FIELDS = frozenset(
    {
        "item_id",
        "teacher_opaque_id",
        "thread_id",
        "year",
        "subreddit",
        "content_type",
        "retrieval_mode",
        "target_text",
        "submission_context",
        "parent_context",
        "label_json",
        "label_sha256",
        "source_row_sha256",
        "mapping_row_sha256",
        "teacher_row_sha256",
        "joined_row_sha256",
        "quality_tier",
        "primary_training_eligible",
    }
)


def canonical_sha256(value: Any) -> str:
    """Return the repository's canonical finite-JSON digest."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_text(row: Mapping[str, Any], field: str, *, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}.{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceFramePolicy:
    """Frozen sampling policy for the 10k v2 engineering frames."""

    expected_source_rows: int = 10_000
    calibration_rows: int = 600
    development_rows: int = 600
    development_probability_rows: int = 300
    rare_target_stance_rows: int = 150
    multi_target_rows: int = 75
    context_available_rows: int = 75
    rare_cell_count: int = 8
    minimum_rare_cell_support: int = 15
    minimum_rare_cell_selected: int = 10
    minimum_training_rows: int = 1_000
    expected_bridge_exposure_threads: int = 480
    legacy_development_proxy_rows: int = 222
    legacy_locked_proxy_rows: int = 230
    calibration_seed: str = "factorised-v2-calibration-probability-v1"
    development_probability_seed: str = "factorised-v2-development-probability-v1"
    development_enrichment_seed: str = "factorised-v2-development-enrichment-v1"

    def __post_init__(self) -> None:
        integer_fields = (
            self.expected_source_rows,
            self.calibration_rows,
            self.development_rows,
            self.development_probability_rows,
            self.rare_target_stance_rows,
            self.multi_target_rows,
            self.context_available_rows,
            self.rare_cell_count,
            self.minimum_rare_cell_support,
            self.minimum_rare_cell_selected,
            self.minimum_training_rows,
            self.expected_bridge_exposure_threads,
            self.legacy_development_proxy_rows,
            self.legacy_locked_proxy_rows,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise ValueError("all evidence-frame counts must be positive integers")
        enrichment_rows = (
            self.rare_target_stance_rows
            + self.multi_target_rows
            + self.context_available_rows
        )
        if self.development_probability_rows + enrichment_rows != self.development_rows:
            raise ValueError("development probability and enrichment quotas must sum exactly")
        if self.minimum_rare_cell_support < 15:
            raise ValueError("minimum_rare_cell_support must remain at least 15")
        if self.minimum_rare_cell_selected < 10:
            raise ValueError("minimum_rare_cell_selected must remain at least 10")
        if self.rare_cell_count > 8:
            raise ValueError("rare_cell_count cannot exceed the registered maximum of eight")
        if self.rare_target_stance_rows < (
            self.rare_cell_count * self.minimum_rare_cell_selected
        ):
            raise ValueError(
                "rare-target quota cannot guarantee the minimum for every selected rare cell"
            )
        for field in (
            "calibration_seed",
            "development_probability_seed",
            "development_enrichment_seed",
        ):
            if not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")

    @property
    def enrichment_quotas(self) -> dict[str, int]:
        return {
            "context_available": self.context_available_rows,
            "multi_target": self.multi_target_rows,
            "rare_target_stance": self.rare_target_stance_rows,
        }

    def as_contract(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "metadata_stratum_fields": list(METADATA_STRATUM_FIELDS),
            "probability_allocation_algorithm": PROBABILITY_ALLOCATION_ALGORITHM,
            "rare_enrichment_algorithm": RARE_ENRICHMENT_ALGORITHM,
            "enrichment_priority": [
                "rare_target_stance",
                "context_available",
                "multi_target",
            ],
            "rare_source_support_scope": (
                "unexposed-primary-eligible-evaluation-pool"
            ),
            "rare_remaining_support_scope": (
                "after-calibration-and-development-probability-draws"
            ),
        }


PRODUCTION_POLICY = EvidenceFramePolicy()
PRODUCTION_POLICY_DIGEST = canonical_sha256(PRODUCTION_POLICY.as_contract())


def _is_production_policy(policy: EvidenceFramePolicy) -> bool:
    return policy == PRODUCTION_POLICY


def _validate_bindings(bindings: Mapping[str, Any]) -> dict[str, str]:
    if set(bindings) != REQUIRED_INPUT_BINDINGS:
        raise ValueError(
            "input_bindings must contain exactly " f"{sorted(REQUIRED_INPUT_BINDINGS)}"
        )
    clean: dict[str, str] = {}
    for field in sorted(REQUIRED_INPUT_BINDINGS):
        value = bindings[field]
        if not _is_sha256(value):
            raise ValueError(f"input_bindings.{field} must be a lowercase SHA-256")
        clean[field] = value
    return clean


def _validate_exposure_register(
    values: Sequence[str],
    *,
    name: str,
    expected_count: int,
    expected_content_sha256: str,
    source_threads: frozenset[str],
) -> frozenset[str]:
    register = list(values)
    if (
        len(register) != expected_count
        or len(set(register)) != len(register)
        or any(not isinstance(value, str) or not value for value in register)
    ):
        raise ValueError(f"{name} exposure register has the wrong size or duplicate group IDs")
    result = frozenset(register)
    if not result <= source_threads:
        raise ValueError(f"{name} exposure register contains groups outside the 10k source")
    if canonical_sha256(sorted(result)) != expected_content_sha256:
        raise ValueError(f"{name} exposure register content digest drifted")
    return result


def _validate_external_proxy_surface_set(
    values: Sequence[str],
    *,
    name: str,
    expected_source_rows: int,
    expected_set_sha256: str,
    source_surface_hashes: frozenset[str],
) -> tuple[frozenset[str], dict[str, int]]:
    surface_hashes = list(values)
    if (
        not surface_hashes
        or len(surface_hashes) > expected_source_rows
        or len(set(surface_hashes)) != len(surface_hashes)
        or any(not _is_sha256(value) for value in surface_hashes)
    ):
        raise ValueError(
            f"{name} proxy surface set must contain one to {expected_source_rows} "
            "unique lowercase SHA-256 values"
        )
    result = frozenset(surface_hashes)
    if canonical_sha256(sorted(result)) != expected_set_sha256:
        raise ValueError(f"{name} proxy surface-set digest drifted")
    overlap_count = len(result & source_surface_hashes)
    if overlap_count:
        raise ValueError(
            f"{name} proxy has {overlap_count} exact surface overlaps with the 10k source"
        )
    return result, {
        "source_row_count": expected_source_rows,
        "unique_surface_hash_count": len(result),
        "exact_surface_overlap_count": overlap_count,
    }


def _validate_external_proxy_sample_id_set(
    values: Sequence[str],
    *,
    name: str,
    expected_source_rows: int,
    expected_set_sha256: str,
    source_item_ids: frozenset[str],
) -> tuple[frozenset[str], dict[str, int]]:
    sample_ids = list(values)
    if (
        len(sample_ids) != expected_source_rows
        or len(set(sample_ids)) != len(sample_ids)
        or any(not isinstance(value, str) or not value for value in sample_ids)
    ):
        raise ValueError(
            f"{name} proxy sample-ID set must contain exactly {expected_source_rows} "
            "unique non-empty strings, one per proxy row"
        )
    result = frozenset(sample_ids)
    if canonical_sha256(sorted(result)) != expected_set_sha256:
        raise ValueError(f"{name} proxy sample-ID-set digest drifted")
    overlap_count = len(result & source_item_ids)
    if overlap_count:
        raise ValueError(
            f"{name} proxy has {overlap_count} canonical sample-ID overlaps with the "
            "10k source"
        )
    return result, {
        "unique_sample_id_count": len(result),
        "canonical_sample_id_overlap_count": overlap_count,
    }


def _normalise_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("joined v2 evidence rows cannot be empty")
    clean_rows: list[dict[str, Any]] = []
    item_ids: set[str] = set()
    thread_ids: set[str] = set()
    source_hashes: set[str] = set()
    mapping_hashes: set[str] = set()
    teacher_hashes: set[str] = set()
    joined_hashes: set[str] = set()
    teacher_opaque_ids: set[str] = set()
    for index, raw in enumerate(rows):
        where = f"rows[{index}]"
        if not isinstance(raw, Mapping) or not set(raw) >= REQUIRED_ROW_FIELDS:
            raise ValueError(f"{where} is missing the joined v2 evidence fields")
        item_id = _required_text(raw, "item_id", where=where)
        teacher_opaque_id = _required_text(raw, "teacher_opaque_id", where=where)
        thread_id = _required_text(raw, "thread_id", where=where)
        if item_id in item_ids:
            raise ValueError("joined v2 evidence rows contain a duplicate item_id")
        if thread_id in thread_ids:
            raise ValueError(
                "joined v2 evidence rows must remain one-row-per-thread for exact quotas"
            )
        item_ids.add(item_id)
        thread_ids.add(thread_id)
        if teacher_opaque_id in teacher_opaque_ids:
            raise ValueError("joined v2 evidence rows contain a duplicate teacher_opaque_id")
        teacher_opaque_ids.add(teacher_opaque_id)

        year = raw.get("year")
        if type(year) is not int or year not in SUPPORTED_YEARS:
            raise ValueError(f"{where}.year is outside the registered 2020--2025 frame")
        subreddit = _required_text(raw, "subreddit", where=where)
        content_type = _required_text(raw, "content_type", where=where)
        if content_type not in SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"{where}.content_type is outside the registered source strata")
        retrieval_mode = _required_text(raw, "retrieval_mode", where=where)
        target_text = _required_text(raw, "target_text", where=where)
        contexts: dict[str, str | None] = {}
        for field in ("submission_context", "parent_context"):
            value = raw.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{where}.{field} must be a string or null")
            contexts[field] = value

        label_json = raw.get("label_json")
        if not isinstance(label_json, str) or not label_json:
            raise ValueError(f"{where}.label_json must be a non-empty JSON string")
        try:
            parsed = json.loads(label_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}.label_json is invalid JSON") from exc
        label = validate_v2_label(parsed)
        label_sha256 = raw.get("label_sha256")
        if not _is_sha256(label_sha256):
            raise ValueError(f"{where}.label_sha256 is missing or invalid")
        if label_sha256 != canonical_sha256(label):
            raise ValueError(f"{where}.label_sha256 does not bind the canonical v2 label")
        source_row_sha256 = raw.get("source_row_sha256")
        if not _is_sha256(source_row_sha256):
            raise ValueError(f"{where}.source_row_sha256 is missing or invalid")
        source_record = {
            "sample_id": item_id,
            "thread_id": thread_id,
            "target_text": target_text,
            "submission_context": contexts["submission_context"],
            "parent_context": contexts["parent_context"],
            "subreddit": subreddit,
            "year": year,
            "content_type": content_type,
            "retrieval_mode": retrieval_mode,
        }
        if tuple(source_record) != SOURCE_ROW_HASH_FIELDS:
            raise RuntimeError("source-row hash field ordering drifted")
        if source_row_sha256 != canonical_sha256(source_record):
            raise ValueError(f"{where}.source_row_sha256 does not bind the canonical source row")
        if source_row_sha256 in source_hashes:
            raise ValueError("source_row_sha256 values must uniquely bind source rows")
        source_hashes.add(source_row_sha256)
        surface_record = {
            "target_text": target_text,
            "parent_context": contexts["parent_context"],
            "submission_context": contexts["submission_context"],
        }
        if tuple(surface_record) != PROXY_SURFACE_HASH_FIELDS:
            raise RuntimeError("proxy-surface hash field ordering drifted")
        surface_sha256 = canonical_sha256(surface_record)

        quality_tier = raw.get("quality_tier")
        if quality_tier not in QUALITY_TIERS:
            raise ValueError(f"{where}.quality_tier is unsupported")
        primary = raw.get("primary_training_eligible")
        if type(primary) is not bool:
            raise ValueError(f"{where}.primary_training_eligible must be boolean")
        context_available = any(
            isinstance(contexts[field], str) and bool(contexts[field].strip())
            for field in ("submission_context", "parent_context")
        )
        excluded_by_policy = (
            label["codability"] == "not_codable"
            or quality_tier == "informed_adjudication"
        )
        if primary and excluded_by_policy:
            raise ValueError(
                f"{where}.primary_training_eligible conflicts with the registered quality policy"
            )
        if primary and quality_tier not in PRIMARY_QUALITY_TIERS:
            raise ValueError(f"{where} uses an unsupported primary-training quality tier")

        mapping_record = {
            "sample_id": item_id,
            "thread_id": thread_id,
            "opaque_id": teacher_opaque_id,
        }
        if tuple(mapping_record) != MAPPING_ROW_HASH_FIELDS:
            raise RuntimeError("mapping-row hash field ordering drifted")
        mapping_row_sha256 = raw.get("mapping_row_sha256")
        if not _is_sha256(mapping_row_sha256):
            raise ValueError(f"{where}.mapping_row_sha256 is missing or invalid")
        if mapping_row_sha256 != canonical_sha256(mapping_record):
            raise ValueError(
                f"{where}.mapping_row_sha256 does not bind sample/thread/opaque identity"
            )
        if mapping_row_sha256 in mapping_hashes:
            raise ValueError("mapping_row_sha256 values must uniquely bind mapping rows")
        mapping_hashes.add(mapping_row_sha256)

        teacher_record = {
            "opaque_id": teacher_opaque_id,
            "label_sha256": label_sha256,
            "quality_tier": quality_tier,
            "primary_training_eligible": primary,
        }
        if tuple(teacher_record) != TEACHER_ROW_HASH_FIELDS:
            raise RuntimeError("teacher-row hash field ordering drifted")
        teacher_row_sha256 = raw.get("teacher_row_sha256")
        if not _is_sha256(teacher_row_sha256):
            raise ValueError(f"{where}.teacher_row_sha256 is missing or invalid")
        if teacher_row_sha256 != canonical_sha256(teacher_record):
            raise ValueError(
                f"{where}.teacher_row_sha256 does not bind opaque/label/tier/eligibility"
            )
        if teacher_row_sha256 in teacher_hashes:
            raise ValueError("teacher_row_sha256 values must uniquely bind teacher rows")
        teacher_hashes.add(teacher_row_sha256)

        joined_record = {
            "source_row_sha256": source_row_sha256,
            "mapping_row_sha256": mapping_row_sha256,
            "teacher_row_sha256": teacher_row_sha256,
        }
        if tuple(joined_record) != JOINED_ROW_HASH_FIELDS:
            raise RuntimeError("joined-row hash field ordering drifted")
        joined_row_sha256 = raw.get("joined_row_sha256")
        if not _is_sha256(joined_row_sha256):
            raise ValueError(f"{where}.joined_row_sha256 is missing or invalid")
        if joined_row_sha256 != canonical_sha256(joined_record):
            raise ValueError(
                f"{where}.joined_row_sha256 does not bind source/mapping/teacher rows"
            )
        if joined_row_sha256 in joined_hashes:
            raise ValueError("joined_row_sha256 values must uniquely bind joined rows")
        joined_hashes.add(joined_row_sha256)

        target_cells = tuple(
            sorted(
                (entry["target"], entry["stance"])
                for entry in label["targets"]
                if entry["target"] in ANALYTIC_TARGETS
            )
        )
        clean_rows.append(
            {
                "item_id": item_id,
                "teacher_opaque_id": teacher_opaque_id,
                "thread_id": thread_id,
                "year": year,
                "subreddit": subreddit,
                "content_type": content_type,
                "retrieval_mode": retrieval_mode,
                "label_sha256": label_sha256,
                "source_row_sha256": source_row_sha256,
                "mapping_row_sha256": mapping_row_sha256,
                "teacher_row_sha256": teacher_row_sha256,
                "joined_row_sha256": joined_row_sha256,
                "surface_sha256": surface_sha256,
                "quality_tier": quality_tier,
                "primary_training_eligible": primary,
                "context_available": context_available,
                "codability": label["codability"],
                "target_count": len(label["targets"]),
                "target_cells": target_cells,
            }
        )
    clean_rows.sort(key=lambda row: row["item_id"])
    return clean_rows


def _metadata_stratum(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in METADATA_STRATUM_FIELDS)


def _stratum_text(stratum: Sequence[Any]) -> str:
    return json.dumps(list(stratum), ensure_ascii=True, separators=(",", ":"))


def _rank(seed: str, namespace: str, item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{item_id}".encode()).hexdigest()
    return digest, item_id


def _proportional_quotas(
    groups: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
    *,
    total: int,
) -> dict[tuple[Any, ...], int]:
    population = sum(len(values) for values in groups.values())
    if population <= total:
        raise ValueError("probability frame must leave rows outside the draw")
    stratum_count = len(groups)
    if total < stratum_count:
        raise ValueError(
            "probability draw size must be at least the observed metadata-stratum count "
            "to guarantee positive inclusion probability"
        )
    quotas: dict[tuple[Any, ...], int] = {stratum: 1 for stratum in groups}
    remaining = total - stratum_count
    if remaining == 0:
        return quotas
    capacities = {stratum: len(values) - 1 for stratum, values in groups.items()}
    capacity_total = sum(capacities.values())
    if remaining > capacity_total:
        raise ValueError("probability draw exceeds residual metadata-stratum capacity")
    remainders: list[tuple[int, str, tuple[Any, ...]]] = []
    assigned = 0
    for stratum, capacity in capacities.items():
        numerator = remaining * capacity
        allocation, remainder = divmod(numerator, capacity_total)
        quotas[stratum] += allocation
        assigned += allocation
        remainders.append((-remainder, _stratum_text(stratum), stratum))
    for _, _, stratum in sorted(remainders):
        if assigned == remaining:
            break
        if quotas[stratum] < len(groups[stratum]):
            quotas[stratum] += 1
            assigned += 1
    if sum(quotas.values()) != total:
        raise RuntimeError("proportional allocation did not conserve its exact quota")
    if any(quota > len(groups[stratum]) for stratum, quota in quotas.items()):
        raise RuntimeError("proportional allocation exceeded a metadata stratum")
    return quotas


def _stratified_probability_draw(
    rows: Sequence[Mapping[str, Any]],
    *,
    total: int,
    seed: str,
    component: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_metadata_stratum(row)].append(row)
    quotas = _proportional_quotas(groups, total=total)
    selected: list[dict[str, Any]] = []
    design: dict[str, dict[str, Any]] = {}
    for stratum in sorted(groups, key=_stratum_text):
        population = groups[stratum]
        quota = quotas[stratum]
        ranked = sorted(
            population,
            key=lambda row: _rank(seed, _stratum_text(stratum), row["item_id"]),
        )
        for row in ranked[:quota]:
            selected.append(dict(row))
            design[row["item_id"]] = {
                "selection_stratum": _stratum_text(stratum),
                "inclusion_probability_numerator": quota,
                "inclusion_probability_denominator": len(population),
                "inclusion_probability": quota / len(population),
                "probability_scope": "conditional-within-frozen-metadata-stratum",
                "selection_component": component,
            }
    selected.sort(key=lambda row: row["item_id"])
    if len(selected) != total or len(design) != total:
        raise RuntimeError("probability draw did not conserve its exact quota")
    return selected, design


def _probability_design_summary(
    population_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    design: Mapping[str, Mapping[str, Any]],
    *,
    frame: str,
    component: str,
) -> dict[str, Any]:
    population_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in population_rows:
        population_groups[_stratum_text(_metadata_stratum(row))].append(row)
    selected_groups: Counter[str] = Counter()
    digest_design: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        item_id = row["item_id"]
        if item_id not in design:
            raise RuntimeError("probability design is missing a selected membership identity")
        record = {
            "frame": frame,
            **{
                field: design[item_id][field]
                for field in NATURAL_DESIGN_DIGEST_FIELDS
                if field != "frame"
            },
        }
        if record["selection_component"] != component:
            raise RuntimeError("probability design component drifted")
        stratum = record["selection_stratum"]
        if stratum not in population_groups:
            raise RuntimeError("probability design references an absent population stratum")
        selected_groups[stratum] += 1
        digest_design[item_id] = record
    if len(digest_design) != len(design) or not digest_design:
        raise RuntimeError("probability design does not bind exactly the selected rows")
    if set(selected_groups) != set(population_groups):
        raise RuntimeError("probability design did not represent every observed metadata stratum")
    for record in digest_design.values():
        stratum = record["selection_stratum"]
        if record["inclusion_probability_numerator"] != selected_groups[stratum]:
            raise RuntimeError("probability design numerator differs from its stratum quota")
        if record["inclusion_probability_denominator"] != len(
            population_groups[stratum]
        ):
            raise RuntimeError(
                "probability design denominator differs from its stratum population"
            )
    probabilities = [
        record["inclusion_probability_numerator"]
        / record["inclusion_probability_denominator"]
        for record in digest_design.values()
    ]
    inverse_weights = [1.0 / probability for probability in probabilities]
    quotas = list(selected_groups.values())
    population_sizes = [len(values) for values in population_groups.values()]
    counterfactual_quotas: dict[str, int] = {}
    counterfactual_remainders: list[tuple[int, str]] = []
    counterfactual_assigned = 0
    for stratum, values in population_groups.items():
        numerator = len(digest_design) * len(values)
        quota, remainder = divmod(numerator, len(population_rows))
        counterfactual_quotas[stratum] = quota
        counterfactual_assigned += quota
        counterfactual_remainders.append((-remainder, stratum))
    for _, stratum in sorted(counterfactual_remainders)[
        : len(digest_design) - counterfactual_assigned
    ]:
        counterfactual_quotas[stratum] += 1
    forced_one_stratum_count = sum(
        quota == 0 for quota in counterfactual_quotas.values()
    )
    effective_sample_size = sum(inverse_weights) ** 2 / sum(
        weight**2 for weight in inverse_weights
    )
    return {
        "component": component,
        "frame": frame,
        "design_digest": natural_arm_design_digest(digest_design),
        "expected_sample_rows": len(digest_design),
        "represented_population_rows": sum(population_sizes),
        "expected_strata": len(population_groups),
        "population_stratum_size_min": min(population_sizes),
        "population_stratum_size_max": max(population_sizes),
        "selected_thread_count": len({row["thread_id"] for row in selected_rows}),
        "selected_stratum_count": len(selected_groups),
        "forced_one_stratum_count": forced_one_stratum_count,
        "minimum_only_stratum_count": sum(quota == 1 for quota in quotas),
        "stratum_quota_min": min(quotas),
        "stratum_quota_max": max(quotas),
        "expected_probability_min": min(probabilities),
        "expected_probability_max": max(probabilities),
        "design_effective_sample_size": effective_sample_size,
        "design_effective_sample_size_ratio": (
            effective_sample_size / len(digest_design)
        ),
        "inverse_weight_min": min(inverse_weights),
        "inverse_weight_max": max(inverse_weights),
        "inverse_weight_ratio": max(inverse_weights) / min(inverse_weights),
    }


def _cell_support(
    rows: Sequence[Mapping[str, Any]],
) -> Counter[tuple[str, str]]:
    support: Counter[tuple[str, str]] = Counter()
    for row in rows:
        support.update(row["target_cells"])
    return support


def _exclusive_rare_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    rare_cells: frozenset[tuple[str, str]],
    remaining_support: Mapping[tuple[str, str], int],
    source_support: Mapping[tuple[str, str], int],
) -> Counter[tuple[str, str]]:
    support: Counter[tuple[str, str]] = Counter()
    for row in rows:
        anchor = _rare_anchor(
            row,
            rare_cells=rare_cells,
            remaining_support=remaining_support,
            source_support=source_support,
        )
        if anchor is not None:
            support[anchor] += 1
    return support


def _rare_cells(
    source_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    policy: EvidenceFramePolicy,
) -> tuple[
    tuple[tuple[str, str], ...],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
]:
    source_support = _cell_support(source_rows)
    remaining_support = _cell_support(candidate_rows)
    occurrence_eligible = frozenset(
        cell
        for cell, count in remaining_support.items()
        if source_support[cell] >= policy.minimum_rare_cell_support
        and count >= policy.minimum_rare_cell_selected
    )
    if not occurrence_eligible:
        raise ValueError(
            "insufficient post-natural analytical target-by-stance support for "
            "registered enrichment"
        )

    # Anchor against every occurrence-eligible cell before choosing the lowest-support
    # cells.  This makes the >=10 requirement a property of disjoint candidate pools,
    # not of overlapping label occurrences.  Removing unchosen cells can only add rows
    # to a retained cell under the same total ordering.
    all_exclusive_support = _exclusive_rare_support(
        candidate_rows,
        rare_cells=occurrence_eligible,
        remaining_support=remaining_support,
        source_support=source_support,
    )
    exclusive_eligible = [
        cell
        for cell in occurrence_eligible
        if all_exclusive_support[cell] >= policy.minimum_rare_cell_selected
    ]
    if not exclusive_eligible:
        raise ValueError(
            "no post-natural exclusive rare-cell pool satisfies the selected minimum"
        )
    ranked = sorted(
        exclusive_eligible,
        key=lambda cell: (
            remaining_support[cell],
            source_support[cell],
            cell[0],
            cell[1],
        ),
    )
    chosen = tuple(ranked[: policy.rare_cell_count])
    if policy.rare_target_stance_rows < (
        len(chosen) * policy.minimum_rare_cell_selected
    ):
        raise ValueError("rare-target quota cannot satisfy the per-cell selected minimum")
    chosen_exclusive_support = _exclusive_rare_support(
        candidate_rows,
        rare_cells=frozenset(chosen),
        remaining_support=remaining_support,
        source_support=source_support,
    )
    if any(
        chosen_exclusive_support[cell] < policy.minimum_rare_cell_selected
        for cell in chosen
    ):
        raise RuntimeError(
            "post-natural rare-cell re-anchoring violated the selected minimum"
        )
    if sum(chosen_exclusive_support.values()) < policy.rare_target_stance_rows:
        raise ValueError(
            "post-natural exclusive rare-cell capacity cannot satisfy the registered quota"
        )
    return chosen, source_support, remaining_support, chosen_exclusive_support


def _rare_anchor(
    row: Mapping[str, Any],
    *,
    rare_cells: frozenset[tuple[str, str]],
    remaining_support: Mapping[tuple[str, str], int],
    source_support: Mapping[tuple[str, str], int],
) -> tuple[str, str] | None:
    candidates = [cell for cell in row["target_cells"] if cell in rare_cells]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda cell: (
            remaining_support[cell],
            source_support[cell],
            cell[0],
            cell[1],
        ),
    )


def _minimum_proportional_quotas(
    support: Mapping[tuple[str, str], int],
    *,
    total: int,
    minimum: int,
) -> dict[tuple[str, str], int]:
    if not support:
        raise ValueError("rare-cell candidate pools are empty")
    if any(count < minimum for count in support.values()):
        raise ValueError("a rare-cell candidate pool cannot satisfy the selected minimum")
    base_total = len(support) * minimum
    if total < base_total or sum(support.values()) < total:
        raise ValueError("rare-cell support cannot satisfy the registered total quota")
    quotas = {cell: minimum for cell in support}
    remaining = total - base_total
    capacities = {cell: support[cell] - minimum for cell in support}
    capacity_total = sum(capacities.values())
    if remaining == 0:
        return quotas
    if capacity_total < remaining:
        raise ValueError("rare-cell residual capacity cannot satisfy the registered quota")
    remainders: list[tuple[int, str, str, tuple[str, str]]] = []
    assigned = 0
    for cell, capacity in capacities.items():
        numerator = remaining * capacity
        allocation, remainder = divmod(numerator, capacity_total)
        quotas[cell] += allocation
        assigned += allocation
        remainders.append((-remainder, cell[0], cell[1], cell))
    for _, _, _, cell in sorted(remainders):
        if assigned == remaining:
            break
        if quotas[cell] < support[cell]:
            quotas[cell] += 1
            assigned += 1
    if assigned != remaining or sum(quotas.values()) != total:
        raise RuntimeError("rare-cell minimum/proportional allocation did not conserve quota")
    return quotas


def _enrichment_draw(
    rows: Sequence[Mapping[str, Any]],
    *,
    rare_cells: Sequence[tuple[str, str]],
    rare_source_support: Mapping[tuple[str, str], int],
    rare_remaining_support: Mapping[tuple[str, str], int],
    expected_rare_exclusive_support: Mapping[tuple[str, str], int],
    policy: EvidenceFramePolicy,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, int],
    list[dict[str, Any]],
]:
    pools: dict[str, list[Mapping[str, Any]]] = {
        "context_available": [],
        "multi_target": [],
    }
    rare_pools: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        cell: [] for cell in rare_cells
    }
    rare_set = frozenset(rare_cells)
    for row in rows:
        anchor = _rare_anchor(
            row,
            rare_cells=rare_set,
            remaining_support=rare_remaining_support,
            source_support=rare_source_support,
        )
        if anchor is not None:
            rare_pools[anchor].append(row)
            continue
        if row["context_available"]:
            pools["context_available"].append(row)
            continue
        if row["target_count"] >= 2:
            pools["multi_target"].append(row)
    selected: list[dict[str, Any]] = []
    design: dict[str, dict[str, Any]] = {}
    support = {component: len(values) for component, values in pools.items()}
    for component, quota in (
        ("context_available", policy.context_available_rows),
        ("multi_target", policy.multi_target_rows),
    ):
        population = pools[component]
        if len(population) < quota:
            raise ValueError(
                f"insufficient {component} support: need {quota}, found {len(population)}"
            )
        ranked = sorted(
            population,
            key=lambda row: _rank(
                policy.development_enrichment_seed,
                component,
                row["item_id"],
            ),
        )
        for row in ranked[:quota]:
            selected.append(dict(row))
            design[row["item_id"]] = {
                "selection_stratum": f"enrichment:{component}",
                "inclusion_probability_numerator": quota,
                "inclusion_probability_denominator": len(population),
                "inclusion_probability": quota / len(population),
                "probability_scope": "conditional-on-prior-draws-and-exclusive-enrichment-pool",
                "selection_component": f"development_{component}",
            }

    rare_candidate_support = {cell: len(values) for cell, values in rare_pools.items()}
    if rare_candidate_support != {
        cell: expected_rare_exclusive_support[cell] for cell in rare_cells
    }:
        raise RuntimeError("rare-first exclusive candidate support drifted")
    rare_quotas = _minimum_proportional_quotas(
        rare_candidate_support,
        total=policy.rare_target_stance_rows,
        minimum=policy.minimum_rare_cell_selected,
    )
    rare_support_records: list[dict[str, Any]] = []
    for cell in rare_cells:
        population = rare_pools[cell]
        quota = rare_quotas[cell]
        namespace = json.dumps(list(cell), ensure_ascii=True, separators=(",", ":"))
        ranked = sorted(
            population,
            key=lambda row: _rank(
                policy.development_enrichment_seed,
                f"rare-target-stance:{namespace}",
                row["item_id"],
            ),
        )
        for row in ranked[:quota]:
            selected.append(dict(row))
            design[row["item_id"]] = {
                "selection_stratum": f"enrichment:rare_target_stance:{namespace}",
                "inclusion_probability_numerator": quota,
                "inclusion_probability_denominator": len(population),
                "inclusion_probability": quota / len(population),
                "probability_scope": (
                    "conditional-on-prior-draws-and-exclusive-anchored-rare-cell-pool"
                ),
                "selection_component": "development_rare_target_stance",
            }
        rare_support_records.append(
            {
                "target": cell[0],
                "stance": cell[1],
                "source_support": rare_source_support[cell],
                "remaining_support": rare_remaining_support[cell],
                "candidate_support": len(population),
                "selected_support": quota,
            }
        )
    support["rare_target_stance"] = sum(rare_candidate_support.values())
    support["post_rare_remaining"] = len(rows) - support["rare_target_stance"]
    support["post_context_remaining"] = (
        support["post_rare_remaining"] - support["context_available"]
    )
    selected.sort(key=lambda row: row["item_id"])
    if len(selected) != sum(policy.enrichment_quotas.values()):
        raise RuntimeError("enrichment draw did not conserve its exact quota")
    if len(design) != len(selected):
        raise RuntimeError("enrichment pools overlap after deterministic rare-cell anchoring")
    return selected, design, support, rare_support_records


def _audit_reason(row: Mapping[str, Any]) -> str:
    if row["codability"] == "not_codable":
        return "not_codable"
    if row["quality_tier"] == "informed_adjudication":
        return "informed_adjudication"
    return "upstream_primary_ineligible"


def _membership_row(
    row: Mapping[str, Any],
    *,
    frame: str,
    design: Mapping[str, Any] | None,
    exposure_flags: Mapping[str, bool],
) -> dict[str, Any]:
    if design is None:
        component = "training_remainder" if frame == "training" else _audit_reason(row)
        selection = {
            "selection_component": component,
            "selection_stratum": None,
            "inclusion_probability_numerator": None,
            "inclusion_probability_denominator": None,
            "inclusion_probability": None,
            "probability_scope": None,
        }
    else:
        selection = dict(design)
    return {
        "item_id": row["item_id"],
        "teacher_opaque_id": row["teacher_opaque_id"],
        "thread_id": row["thread_id"],
        "frame": frame,
        **selection,
        "source_row_sha256": row["source_row_sha256"],
        "label_sha256": row["label_sha256"],
        "mapping_row_sha256": row["mapping_row_sha256"],
        "teacher_row_sha256": row["teacher_row_sha256"],
        "joined_row_sha256": row["joined_row_sha256"],
        "quality_tier": row["quality_tier"],
        "primary_training_eligible": row["primary_training_eligible"],
        **dict(exposure_flags),
    }


def _set_digest(values: Sequence[str]) -> str:
    return canonical_sha256(sorted(values))


def _frame_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = [
        row["inclusion_probability"]
        for row in rows
        if row["inclusion_probability"] is not None
    ]
    scopes = Counter(
        row["probability_scope"]
        for row in rows
        if row["probability_scope"] is not None
    )
    return {
        "row_count": len(rows),
        "group_count": len({row["thread_id"] for row in rows}),
        "item_set_digest": _set_digest([row["item_id"] for row in rows]),
        "group_set_digest": _set_digest([row["thread_id"] for row in rows]),
        "source_binding_set_digest": _set_digest(
            [row["source_row_sha256"] for row in rows]
        ),
        "label_binding_set_digest": _set_digest([row["label_sha256"] for row in rows]),
        "mapping_binding_set_digest": _set_digest(
            [row["mapping_row_sha256"] for row in rows]
        ),
        "teacher_binding_set_digest": _set_digest(
            [row["teacher_row_sha256"] for row in rows]
        ),
        "joined_binding_set_digest": _set_digest(
            [row["joined_row_sha256"] for row in rows]
        ),
        "selection_component_counts": dict(sorted(Counter(
            row["selection_component"] for row in rows
        ).items())),
        "probability_scope_counts": dict(sorted(scopes.items())),
        "inclusion_probability_min": min(probabilities) if probabilities else None,
        "inclusion_probability_max": max(probabilities) if probabilities else None,
    }


def _validate_membership_invariants(
    membership_rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    policy: EvidenceFramePolicy,
) -> None:
    expected_items = {row["item_id"] for row in source_rows}
    observed_items = [row["item_id"] for row in membership_rows]
    if len(observed_items) != len(set(observed_items)):
        raise RuntimeError("evidence frames overlap or duplicate a row")
    if set(observed_items) != expected_items:
        raise RuntimeError("evidence frames lost or invented rows")
    observed_threads = [row["thread_id"] for row in membership_rows]
    if len(observed_threads) != len(set(observed_threads)):
        raise RuntimeError("evidence frames overlap at thread level")
    counts = Counter(row["frame"] for row in membership_rows)
    if counts["calibration"] != policy.calibration_rows:
        raise RuntimeError("calibration quota drifted")
    if counts["development"] != policy.development_rows:
        raise RuntimeError("development quota drifted")
    if counts["training"] < policy.minimum_training_rows:
        raise ValueError("insufficient primary-eligible rows remain for training")
    for row in membership_rows:
        probability = row["inclusion_probability"]
        numerator = row["inclusion_probability_numerator"]
        denominator = row["inclusion_probability_denominator"]
        if probability is None:
            if numerator is not None or denominator is not None:
                raise RuntimeError("non-probability membership has a partial probability")
        elif (
            type(numerator) is not int
            or type(denominator) is not int
            or numerator <= 0
            or denominator < numerator
            or not math.isclose(probability, numerator / denominator)
        ):
            raise RuntimeError("membership inclusion probability is not exact")
        if row["frame"] in {"calibration", "development"} and row[
            "bridge_exposed"
        ]:
            raise RuntimeError("an exposed thread entered an evaluation frame")


def build_evidence_frames(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_bindings: Mapping[str, Any],
    bridge_exposed_thread_ids: Sequence[str],
    legacy_development_proxy_sample_ids: Sequence[str],
    legacy_development_proxy_surface_hashes: Sequence[str],
    legacy_locked_proxy_sample_ids: Sequence[str],
    legacy_locked_proxy_surface_hashes: Sequence[str],
    policy: EvidenceFramePolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build private/public frames from an exact-validated joined view in memory.

    Production code must construct ``rows`` by exact-validating and internally joining
    the source, mapping and teacher artefacts.  This pure sampler independently checks
    every per-row cryptographic join binding; it is not a path-loading entrypoint.
    """

    active = policy or EvidenceFramePolicy()
    bindings = _validate_bindings(input_bindings)
    clean_rows = _normalise_rows(rows)
    if len(clean_rows) != active.expected_source_rows:
        raise ValueError(
            "joined v2 evidence row count drifted: "
            f"expected {active.expected_source_rows}, got {len(clean_rows)}"
        )
    normalised_joined_input_digest = canonical_sha256(clean_rows)
    source_item_ids = frozenset(row["item_id"] for row in clean_rows)
    source_threads = frozenset(row["thread_id"] for row in clean_rows)
    source_surface_hashes = frozenset(row["surface_sha256"] for row in clean_rows)
    bridge_exposure = _validate_exposure_register(
        bridge_exposed_thread_ids,
        name="bridge",
        expected_count=active.expected_bridge_exposure_threads,
        expected_content_sha256=bindings["bridge_exposure_register_content_sha256"],
        source_threads=source_threads,
    )
    _, legacy_development_id_audit = _validate_external_proxy_sample_id_set(
        legacy_development_proxy_sample_ids,
        name="legacy development",
        expected_source_rows=active.legacy_development_proxy_rows,
        expected_set_sha256=bindings[
            "legacy_development_proxy_sample_id_set_sha256"
        ],
        source_item_ids=source_item_ids,
    )
    _, legacy_development_surface_audit = _validate_external_proxy_surface_set(
        legacy_development_proxy_surface_hashes,
        name="legacy development",
        expected_source_rows=active.legacy_development_proxy_rows,
        expected_set_sha256=bindings["legacy_development_proxy_surface_set_sha256"],
        source_surface_hashes=source_surface_hashes,
    )
    _, legacy_locked_id_audit = _validate_external_proxy_sample_id_set(
        legacy_locked_proxy_sample_ids,
        name="legacy locked",
        expected_source_rows=active.legacy_locked_proxy_rows,
        expected_set_sha256=bindings["legacy_locked_proxy_sample_id_set_sha256"],
        source_item_ids=source_item_ids,
    )
    _, legacy_locked_surface_audit = _validate_external_proxy_surface_set(
        legacy_locked_proxy_surface_hashes,
        name="legacy locked",
        expected_source_rows=active.legacy_locked_proxy_rows,
        expected_set_sha256=bindings["legacy_locked_proxy_surface_set_sha256"],
        source_surface_hashes=source_surface_hashes,
    )

    primary = [row for row in clean_rows if row["primary_training_eligible"]]
    audit = [row for row in clean_rows if not row["primary_training_eligible"]]
    evaluation_pool = [row for row in primary if row["thread_id"] not in bridge_exposure]
    required_evaluation = active.calibration_rows + active.development_rows
    if len(evaluation_pool) <= required_evaluation:
        raise ValueError("insufficient unexposed primary rows for calibration and development")

    calibration, calibration_design = _stratified_probability_draw(
        evaluation_pool,
        total=active.calibration_rows,
        seed=active.calibration_seed,
        component="calibration_probability",
    )
    calibration_probability_summary = _probability_design_summary(
        evaluation_pool,
        calibration,
        calibration_design,
        frame="calibration",
        component="calibration_probability",
    )
    calibration_ids = {row["item_id"] for row in calibration}
    after_calibration = [row for row in evaluation_pool if row["item_id"] not in calibration_ids]

    development_probability, development_probability_design = _stratified_probability_draw(
        after_calibration,
        total=active.development_probability_rows,
        seed=active.development_probability_seed,
        component="development_probability",
    )
    development_probability_summary = _probability_design_summary(
        after_calibration,
        development_probability,
        development_probability_design,
        frame="development",
        component="development_probability",
    )
    probability_ids = {row["item_id"] for row in development_probability}
    after_probability = [
        row for row in after_calibration if row["item_id"] not in probability_ids
    ]
    (
        rare_cells,
        rare_source_support,
        rare_remaining_support,
        rare_exclusive_support,
    ) = _rare_cells(
        evaluation_pool,
        after_probability,
        policy=active,
    )
    (
        development_enrichment,
        enrichment_design,
        enrichment_support,
        rare_support_records,
    ) = _enrichment_draw(
        after_probability,
        rare_cells=rare_cells,
        rare_source_support=rare_source_support,
        rare_remaining_support=rare_remaining_support,
        expected_rare_exclusive_support=rare_exclusive_support,
        policy=active,
    )
    enrichment_ids = {row["item_id"] for row in development_enrichment}
    development_ids = probability_ids | enrichment_ids
    selected_evaluation_ids = calibration_ids | development_ids
    training = [row for row in primary if row["item_id"] not in selected_evaluation_ids]

    membership_rows: list[dict[str, Any]] = []
    def exposure_flags(row: Mapping[str, Any]) -> dict[str, bool]:
        return {"bridge_exposed": row["thread_id"] in bridge_exposure}

    for row in calibration:
        membership_rows.append(
            _membership_row(
                row,
                frame="calibration",
                design=calibration_design[row["item_id"]],
                exposure_flags=exposure_flags(row),
            )
        )
    merged_development_design = {
        **development_probability_design,
        **enrichment_design,
    }
    for row in [*development_probability, *development_enrichment]:
        membership_rows.append(
            _membership_row(
                row,
                frame="development",
                design=merged_development_design[row["item_id"]],
                exposure_flags=exposure_flags(row),
            )
        )
    for row in training:
        membership_rows.append(
            _membership_row(
                row,
                frame="training",
                design=None,
                exposure_flags=exposure_flags(row),
            )
        )
    for row in audit:
        membership_rows.append(
            _membership_row(
                row,
                frame="audit_excluded",
                design=None,
                exposure_flags=exposure_flags(row),
            )
        )
    membership_rows.sort(key=lambda row: row["item_id"])
    _validate_membership_invariants(
        membership_rows,
        source_rows=clean_rows,
        policy=active,
    )

    production_policy = _is_production_policy(active)
    policy_digest = canonical_sha256(active.as_contract())
    contract = {
        "schema_version": SCHEMA_VERSION,
        "input_bindings": bindings,
        "policy": active.as_contract(),
        "policy_digest": policy_digest,
        "policy_scope": (
            "production-exact-10k"
            if production_policy
            else "test-only-miniature-in-memory"
        ),
        "algorithms": {
            "probability_allocation": PROBABILITY_ALLOCATION_ALGORITHM,
            "probability_rank": "sha256-seed-namespace-item-v1",
            "rare_enrichment": RARE_ENRICHMENT_ALGORITHM,
            "rare_cell_definition": (
                "post-natural-lowest-remaining-support-source15-exclusive10-v2"
            ),
            "enrichment_assignment": "exclusive-priority-rare-context-multi-v2",
            "frame_order": [
                "calibration_probability",
                "development_probability",
                "development_enrichment",
                "training_remainder",
                "audit_excluded",
            ],
            "membership_order": "lexicographic-item-id-v1",
        },
        "probability_design": {
            "minimum_per_observed_metadata_stratum": 1,
            "remaining_allocation_basis": "post-minimum-residual-capacity",
            "exact_row_inclusion_probability": "stratum-quota/stratum-population",
            "allocation_may_be_disproportionate": True,
            "frame_level_analysis_requirement": (
                "inverse-recorded-inclusion-probability-weighting"
            ),
            "design_digest_schema_version": EVALUATION_SCHEMA_VERSION,
            "design_digest_fields": list(NATURAL_DESIGN_DIGEST_FIELDS),
            "weighting_estimator": WEIGHTING_ESTIMATOR,
            "forced_one_diagnostic": (
                "counterfactual-pure-proportional-largest-remainder-zero-quota-v1"
            ),
            "effective_sample_size_diagnostic": (
                "kish-on-inverse-inclusion-probability-v1"
            ),
        },
        "identity_contract": {
            "membership_item_id": "canonical-private-mapping.sample_id",
            "teacher_join_key": "private-mapping.opaque_id",
            "group_id": "canonical-private-mapping.thread_id",
            "opaque_id_is_not_membership_identity": True,
        },
        "source_row_hash_contract": {
            "algorithm": "canonical-json-sha256-v1",
            "fields": list(SOURCE_ROW_HASH_FIELDS),
            "sample_id_source": "canonical-private-mapping.sample_id",
            "context_available": (
                "submission_context-or-parent_context-is-non-empty-after-strip"
            ),
        },
        "joined_row_binding_contract": {
            "algorithm": "canonical-json-sha256-v1",
            "mapping_fields": list(MAPPING_ROW_HASH_FIELDS),
            "teacher_fields": list(TEACHER_ROW_HASH_FIELDS),
            "joined_fields": list(JOINED_ROW_HASH_FIELDS),
            "required_chain": (
                "sample-thread-opaque-to-label-tier-eligibility-to-source-binding"
            ),
        },
        "external_proxy_surface_contract": {
            "algorithm": "canonical-json-sha256-v1",
            "fields": list(PROXY_SURFACE_HASH_FIELDS),
            "required_exact_overlap": 0,
            "proxies_are_external_to_10k": True,
        },
        "external_proxy_sample_id_contract": {
            "algorithm": "canonical-json-sha256-v1",
            "required_exact_overlap": 0,
            "proxies_are_external_to_10k": True,
        },
        "quality_policy": {
            "primary_tiers": sorted(PRIMARY_QUALITY_TIERS),
            "audit_excluded_if": [
                "not_codable",
                "informed_adjudication",
                "primary_training_eligible=false",
            ],
            "not_codable_is_mask_not_class": True,
            "informed_adjudication_is_audit_only": True,
        },
        "exposure_policy": {
            "evaluation_excludes_bridge_groups": True,
            "fresh_full_run_bridge_groups_permitted_in_training": True,
            "legacy_proxies_are_external_not_in_source_registers": True,
            "registered_counts": {
                "bridge": active.expected_bridge_exposure_threads,
                "legacy_development_proxy_rows": active.legacy_development_proxy_rows,
                "legacy_locked_proxy_rows": active.legacy_locked_proxy_rows,
            },
            "content_digests": {
                "bridge": canonical_sha256(sorted(bridge_exposure)),
                "legacy_development_proxy_surface_set": bindings[
                    "legacy_development_proxy_surface_set_sha256"
                ],
                "legacy_development_proxy_sample_id_set": bindings[
                    "legacy_development_proxy_sample_id_set_sha256"
                ],
                "legacy_locked_proxy_surface_set": bindings[
                    "legacy_locked_proxy_surface_set_sha256"
                ],
                "legacy_locked_proxy_sample_id_set": bindings[
                    "legacy_locked_proxy_sample_id_set_sha256"
                ],
            },
        },
        "decision_budget": {
            "development": {
                "representation_selections": 1,
                "allowed_comparison": "B4-vs-B2",
                "checkpoint_rule_applications": 1,
                "reporting": "component-specific-no-unweighted-mixed-frame-aggregate",
            },
            "calibration": {
                "probability_calibration_and_threshold_freezes": 1,
                "timing": "after-representation-selection",
                "estimand": "unexposed-primary-eligible-10k-engineering-frame",
                "excluded_tier_sensitivity": "separate-analysis",
            },
        },
        "normalised_joined_input_digest": normalised_joined_input_digest,
        "external_artifact_requirement": (
            "production-runtime-exact-validates-and-internally-joins-all-external-artifacts"
        ),
        "decision_budget_enforcement": (
            "runtime-ledger-must-consume-and-freeze-access-receipts"
        ),
        "evidence_boundary": "model-assisted-engineering-frames-not-human-validation",
    }
    contract_digest = canonical_sha256(contract)
    private_base = {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_KIND if production_policy else TEST_PRIVATE_KIND,
        "contract_digest": contract_digest,
        "source_count": len(clean_rows),
        "probability_designs": {
            "calibration_probability": calibration_probability_summary,
            "development_probability": development_probability_summary,
        },
        "rare_cell_support": rare_support_records,
        "memberships": membership_rows,
    }
    membership_id = canonical_sha256(private_base)
    private_membership = {**private_base, "membership_id": membership_id}
    private_bytes = _json_bytes(private_membership)

    frames: dict[str, dict[str, Any]] = {}
    for frame in ("calibration", "development", "training", "audit_excluded"):
        frames[frame] = _frame_summary(
            [row for row in membership_rows if row["frame"] == frame]
        )

    def exposure_summary(register: frozenset[str], field: str) -> dict[str, int]:
        counts = Counter(
            row["frame"] for row in membership_rows if row[field]
        )
        return {
            "source_intersection_count": len(register & source_threads),
            "calibration_intersection_count": counts["calibration"],
            "development_intersection_count": counts["development"],
            "training_intersection_count": counts["training"],
            "audit_intersection_count": counts["audit_excluded"],
        }

    rare_source_values = [row["source_support"] for row in rare_support_records]
    rare_remaining_values = [row["remaining_support"] for row in rare_support_records]
    rare_candidate_values = [row["candidate_support"] for row in rare_support_records]
    rare_selected_values = [row["selected_support"] for row in rare_support_records]
    rare_support_summary = {
        "cell_count": len(rare_support_records),
        "source_support_min": min(rare_source_values),
        "source_support_max": max(rare_source_values),
        "source_support_total": sum(rare_source_values),
        "source_cells_at_least_30": sum(value >= 30 for value in rare_source_values),
        "remaining_support_min": min(rare_remaining_values),
        "remaining_support_max": max(rare_remaining_values),
        "remaining_support_total": sum(rare_remaining_values),
        "candidate_support_min": min(rare_candidate_values),
        "candidate_support_max": max(rare_candidate_values),
        "candidate_support_total": sum(rare_candidate_values),
        "selected_support_min": min(rare_selected_values),
        "selected_support_max": max(rare_selected_values),
        "selected_support_total": sum(rare_selected_values),
        "selected_cells_at_least_30": sum(value >= 30 for value in rare_selected_values),
    }
    public_base = {
        "schema_version": SCHEMA_VERSION,
        "kind": PUBLIC_KIND if production_policy else TEST_PUBLIC_KIND,
        "status": "prepared",
        "contract": contract,
        "contract_digest": contract_digest,
        "source_count": len(clean_rows),
        "primary_eligible_count": len(primary),
        "audit_excluded_count": len(audit),
        "input_item_set_digest": _set_digest([row["item_id"] for row in clean_rows]),
        "input_group_set_digest": _set_digest([row["thread_id"] for row in clean_rows]),
        "input_source_binding_set_digest": _set_digest(
            [row["source_row_sha256"] for row in clean_rows]
        ),
        "input_label_binding_set_digest": _set_digest(
            [row["label_sha256"] for row in clean_rows]
        ),
        "input_mapping_binding_set_digest": _set_digest(
            [row["mapping_row_sha256"] for row in clean_rows]
        ),
        "input_teacher_binding_set_digest": _set_digest(
            [row["teacher_row_sha256"] for row in clean_rows]
        ),
        "input_joined_binding_set_digest": _set_digest(
            [row["joined_row_sha256"] for row in clean_rows]
        ),
        "normalised_joined_input_digest": normalised_joined_input_digest,
        "private_membership_id": membership_id,
        "private_membership_sha256": _bytes_sha256(private_bytes),
        "frames": frames,
        "rare_enrichment_support_summary": rare_support_summary,
        "enrichment_candidate_counts": dict(sorted(enrichment_support.items())),
        "probability_designs": {
            "calibration_probability": calibration_probability_summary,
            "development_probability": development_probability_summary,
        },
        "exposure_intersections": {
            "bridge": exposure_summary(bridge_exposure, "bridge_exposed"),
        },
        "external_proxy_overlap_audit": {
            "legacy_development": {
                **legacy_development_id_audit,
                **legacy_development_surface_audit,
            },
            "legacy_locked": {
                **legacy_locked_id_audit,
                **legacy_locked_surface_audit,
            },
        },
        "row_conservation": True,
        "group_disjointness": True,
        "future_human_frame": {
            "row_count": 600,
            "source": "outside-all-10k-groups",
            "status": "not-sampled-separate-future-manifest",
        },
        "evidence_boundary": contract["evidence_boundary"],
    }
    manifest_id = canonical_sha256(public_base)
    public_manifest = {**public_base, "manifest_id": manifest_id}
    assert_metadata_only(public_manifest, where="factorised-v2-evidence-frame-manifest")
    return private_membership, public_manifest


def validate_evidence_frames(
    private_membership: Mapping[str, Any],
    public_manifest: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, Any],
    bridge_exposed_thread_ids: Sequence[str],
    legacy_development_proxy_sample_ids: Sequence[str],
    legacy_development_proxy_surface_hashes: Sequence[str],
    legacy_locked_proxy_sample_ids: Sequence[str],
    legacy_locked_proxy_surface_hashes: Sequence[str],
    policy: EvidenceFramePolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild and exact-validate a private/public evidence-frame pair."""

    expected_private, expected_public = build_evidence_frames(
        rows,
        input_bindings=input_bindings,
        bridge_exposed_thread_ids=bridge_exposed_thread_ids,
        legacy_development_proxy_sample_ids=legacy_development_proxy_sample_ids,
        legacy_development_proxy_surface_hashes=(
            legacy_development_proxy_surface_hashes
        ),
        legacy_locked_proxy_sample_ids=legacy_locked_proxy_sample_ids,
        legacy_locked_proxy_surface_hashes=legacy_locked_proxy_surface_hashes,
        policy=policy,
    )
    if dict(private_membership) != expected_private:
        raise ValueError("private evidence-frame membership drifted from the frozen inputs")
    if dict(public_manifest) != expected_public:
        raise ValueError("public evidence-frame manifest drifted from the frozen inputs")
    if public_manifest.get("manifest_id") != canonical_sha256(
        {key: value for key, value in public_manifest.items() if key != "manifest_id"}
    ):
        raise ValueError("public evidence-frame manifest_id is invalid")
    assert_metadata_only(public_manifest, where="factorised-v2-evidence-frame-manifest")
    return expected_private, expected_public


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable evidence-frame output differs: {path}")
        return
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"stale evidence-frame temporary exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _git_check_ignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "--quiet", str(path)],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _validate_test_writer_roots(*, private_root: Path, public_root: Path) -> None:
    private = private_root.resolve()
    public = public_root.resolve()
    registered_private = REGISTERED_TEST_PRIVATE_PARENT.resolve()
    registered_public = REGISTERED_TEST_PUBLIC_PARENT.resolve()
    if not _is_within(private, registered_private):
        raise ValueError("test private root is outside the registered git-ignored parent")
    if not _is_within(public, registered_public):
        raise ValueError("test public root is outside the registered public parent")
    if private == public or _is_within(private, public) or _is_within(public, private):
        raise ValueError("private and public evidence-frame roots must be strictly disjoint")
    if not _git_check_ignored(private):
        raise ValueError("registered test private root is not git-ignored")


def _write_evidence_frames_test_only(
    *,
    private_root: Path,
    public_root: Path,
    rows: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, Any],
    bridge_exposed_thread_ids: Sequence[str],
    legacy_development_proxy_sample_ids: Sequence[str],
    legacy_development_proxy_surface_hashes: Sequence[str],
    legacy_locked_proxy_sample_ids: Sequence[str],
    legacy_locked_proxy_surface_hashes: Sequence[str],
    policy: EvidenceFramePolicy,
) -> dict[str, Path]:
    """Write a miniature test pair under registered roots; never use in production."""

    if _is_production_policy(policy):
        raise ValueError(
            "the test-only writer rejects the exact production policy; use "
            "modernbert_factorised_training as the sole production prepare boundary"
        )
    _validate_test_writer_roots(private_root=private_root, public_root=public_root)

    private_membership, public_manifest = build_evidence_frames(
        rows,
        input_bindings=input_bindings,
        bridge_exposed_thread_ids=bridge_exposed_thread_ids,
        legacy_development_proxy_sample_ids=legacy_development_proxy_sample_ids,
        legacy_development_proxy_surface_hashes=(
            legacy_development_proxy_surface_hashes
        ),
        legacy_locked_proxy_sample_ids=legacy_locked_proxy_sample_ids,
        legacy_locked_proxy_surface_hashes=legacy_locked_proxy_surface_hashes,
        policy=policy,
    )
    validate_evidence_frames(
        private_membership,
        public_manifest,
        rows=rows,
        input_bindings=input_bindings,
        bridge_exposed_thread_ids=bridge_exposed_thread_ids,
        legacy_development_proxy_sample_ids=legacy_development_proxy_sample_ids,
        legacy_development_proxy_surface_hashes=(
            legacy_development_proxy_surface_hashes
        ),
        legacy_locked_proxy_sample_ids=legacy_locked_proxy_sample_ids,
        legacy_locked_proxy_surface_hashes=legacy_locked_proxy_surface_hashes,
        policy=policy,
    )
    private_path = private_root / (
        f"membership-{private_membership['membership_id']}.json"
    )
    public_path = public_root / f"manifest-{public_manifest['manifest_id']}.json"
    _write_immutable(private_path, _json_bytes(private_membership))
    _write_immutable(public_path, _json_bytes(public_manifest))
    return {"private_membership": private_path, "public_manifest": public_path}


__all__ = [
    "JOINED_ROW_HASH_FIELDS",
    "MAPPING_ROW_HASH_FIELDS",
    "PROBABILITY_ALLOCATION_ALGORITHM",
    "PRODUCTION_POLICY_DIGEST",
    "PROXY_SURFACE_HASH_FIELDS",
    "QUALITY_TIERS",
    "REQUIRED_INPUT_BINDINGS",
    "SOURCE_ROW_HASH_FIELDS",
    "TEACHER_ROW_HASH_FIELDS",
    "EvidenceFramePolicy",
    "build_evidence_frames",
    "canonical_sha256",
    "validate_evidence_frames",
]
