"""Deterministic, design-weighted sampling contracts for the Stage A recall audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

AUDIT_SCHEMA_VERSION = "1.0.0"
HASH_SPACE = 1 << 64
CONTENT_TYPES = ("submission", "comment")
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


@dataclass(frozen=True)
class HashProbability:
    """An exact Bernoulli probability implemented on a 64-bit hash space."""

    threshold: int

    @property
    def probability(self) -> float:
        return self.threshold / HASH_SPACE

    def as_contract(self) -> dict[str, int | float]:
        return {
            "hash_space": HASH_SPACE,
            "threshold": self.threshold,
            "probability": self.probability,
        }


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def probability_for_expected_count(*, population_rows: int, expected_rows: int) -> HashProbability:
    """Return an exact hash threshold targeting an expected Bernoulli sample size."""

    if type(population_rows) is not int or population_rows < 0:
        raise ValueError("population_rows must be a non-negative integer")
    if type(expected_rows) is not int or expected_rows < 0:
        raise ValueError("expected_rows must be a non-negative integer")
    if population_rows == 0 or expected_rows == 0:
        return HashProbability(0)
    if expected_rows >= population_rows:
        return HashProbability(HASH_SPACE)
    return HashProbability((expected_rows * HASH_SPACE) // population_rows)


def probability_for_decimal(value: Decimal) -> HashProbability:
    if not Decimal(0) <= value <= Decimal(1):
        raise ValueError("probability must be between zero and one")
    return HashProbability(int(value * HASH_SPACE))


def selected_by_hash(*, record_id: str, seed: str, arm: str, probability: HashProbability) -> bool:
    """Apply an arm-specific deterministic Bernoulli draw to one record ID."""

    if not record_id or not seed or not arm:
        raise ValueError("record_id, seed, and arm must be non-empty")
    digest = hashlib.sha256(f"{seed}\x1f{arm}\x1f{record_id}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big")
    return draw < probability.threshold


def audit_item_id(*, audit_run_id: str, record_id: str) -> str:
    if not audit_run_id or not record_id:
        raise ValueError("audit_run_id and record_id must be non-empty")
    return hashlib.sha256(f"{audit_run_id}\x1f{record_id}".encode()).hexdigest()


def make_cell_plan(
    *,
    subreddit: str,
    year: int,
    content_type: str,
    canonical_rows: int,
    candidate_rows: int,
    expected_inside_rows: int,
    expected_outside_rows: int,
    expected_challenger_screen_rows: int,
) -> dict[str, Any]:
    """Freeze all sampling probabilities for one canonical population cell."""

    if not subreddit or content_type not in CONTENT_TYPES:
        raise ValueError("invalid audit cell")
    if type(year) is not int:
        raise ValueError("year must be an integer")
    if type(canonical_rows) is not int or canonical_rows < 0:
        raise ValueError("canonical_rows must be a non-negative integer")
    if type(candidate_rows) is not int or not 0 <= candidate_rows <= canonical_rows:
        raise ValueError("candidate_rows must be between zero and canonical_rows")
    noncandidate_rows = canonical_rows - candidate_rows
    return {
        "subreddit": subreddit,
        "year": year,
        "content_type": content_type,
        "canonical_rows": canonical_rows,
        "candidate_rows": candidate_rows,
        "noncandidate_rows": noncandidate_rows,
        "inside_probability_arm": probability_for_expected_count(
            population_rows=candidate_rows, expected_rows=expected_inside_rows
        ).as_contract(),
        "outside_probability_arm": probability_for_expected_count(
            population_rows=noncandidate_rows, expected_rows=expected_outside_rows
        ).as_contract(),
        "challenger_phase1": probability_for_expected_count(
            population_rows=noncandidate_rows,
            expected_rows=expected_challenger_screen_rows,
        ).as_contract(),
    }


def validate_score_bands(bands: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate fixed, exhaustive challenger score bands and exact phase-two draws."""

    result: list[dict[str, Any]] = []
    expected_min = Decimal("-1")
    names: set[str] = set()
    for band in bands:
        if set(band) != {"name", "min_score", "max_score", "phase2_probability"}:
            raise ValueError("challenger score band has unexpected fields")
        name = str(band["name"])
        if not name or name in names:
            raise ValueError("challenger score-band names must be non-empty and unique")
        names.add(name)
        minimum = Decimal(str(band["min_score"]))
        maximum = Decimal(str(band["max_score"]))
        if minimum != expected_min or not minimum < maximum:
            raise ValueError("challenger score bands must be contiguous and ordered")
        if not Decimal("-1") <= minimum < maximum <= Decimal("1.0000001"):
            raise ValueError("challenger score bands must stay within cosine range")
        probability = probability_for_decimal(Decimal(str(band["phase2_probability"])))
        result.append(
            {
                "name": name,
                "min_score": float(minimum),
                "max_score": float(maximum),
                "phase2": probability.as_contract(),
            }
        )
        expected_min = maximum
    if expected_min != Decimal("1.0000001"):
        raise ValueError("challenger score bands must exhaust the registered score range")
    return result


def score_band(score: float, bands: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for band in bands:
        if float(band["min_score"]) <= score < float(band["max_score"]):
            return band
    raise ValueError(f"challenger score {score} is outside the frozen score bands")


def make_audit_contract(
    *,
    stage_a_run_id: str,
    dataset_revision: str,
    retrieval_policy_digest: str,
    cells: Sequence[Mapping[str, Any]],
    seed: str,
    expected_inside_rows_per_cell: int,
    expected_outside_rows_per_cell: int,
    expected_challenger_screen_rows_per_cell: int,
    challenger: Mapping[str, Any],
    score_bands: Sequence[Mapping[str, Any]],
    stop_rules: Mapping[str, Any],
    cost_guardrail: Mapping[str, Any],
    code_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the immutable 120-cell recall-audit contract."""

    for name, value in (
        ("stage_a_run_id", stage_a_run_id),
        ("retrieval_policy_digest", retrieval_policy_digest),
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if not dataset_revision or not seed:
        raise ValueError("dataset_revision and seed must be non-empty")
    plans = [dict(cell) for cell in cells]
    keys = [
        (str(cell.get("subreddit")), cell.get("year"), str(cell.get("content_type")))
        for cell in plans
    ]
    if len(plans) != 120 or len(set(keys)) != 120:
        raise ValueError("recall audit must pin exactly 120 unique population cells")
    for cell in plans:
        if int(cell["candidate_rows"]) + int(cell["noncandidate_rows"]) != int(
            cell["canonical_rows"]
        ):
            raise ValueError("candidate and non-candidate counts do not conserve the cell")
    plans.sort(key=lambda row: (row["subreddit"], row["year"], row["content_type"]))
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage_a_run_id": stage_a_run_id,
        "dataset_revision": dataset_revision,
        "retrieval_policy_digest": retrieval_policy_digest,
        "seed": seed,
        "design": {
            "candidate_probability_arm": {
                "purpose": "estimate relevant Stage A candidate count",
                "expected_rows_per_nonempty_cell": expected_inside_rows_per_cell,
            },
            "noncandidate_probability_arm": {
                "purpose": "estimate relevant records missed by Stage A",
                "expected_rows_per_nonempty_cell": expected_outside_rows_per_cell,
            },
            "challenger_two_phase_arm": {
                "purpose": "bounded miss discovery; not an oracle or recall denominator",
                "expected_phase1_rows_per_nonempty_cell": (
                    expected_challenger_screen_rows_per_cell
                ),
                "phase1_design": "independent deterministic hash Bernoulli sample",
                "phase2_design": "independent hash Bernoulli sample within fixed score bands",
                "model": dict(challenger),
                "score_bands": validate_score_bands(score_bands),
            },
            "estimator": {
                "inside_total": "Horvitz-Thompson over candidate_probability_arm",
                "outside_total": "Horvitz-Thompson over noncandidate_probability_arm",
                "recall": "inside_total / (inside_total + outside_total)",
                "challenger_use": "diagnostic and adaptive-design evidence only",
            },
        },
        "cells": plans,
        "stop_rules": dict(stop_rules),
        "cost_guardrail": dict(cost_guardrail),
        "code_state": dict(code_state),
    }


def assert_id_only_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    """Fail if a publishable audit row contains source text or author material."""

    for row in rows:
        forbidden = FORBIDDEN_OUTPUT_FIELDS.intersection(row)
        if forbidden:
            raise RuntimeError(f"audit output contains forbidden fields: {sorted(forbidden)}")


def merge_selected_rows(
    *,
    audit_run_id: str,
    cell: Mapping[str, Any],
    probability_rows: Iterable[Mapping[str, Any]],
    challenger_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate arms into a blinded packet and a separate design ledger."""

    merged: dict[str, dict[str, Any]] = {}
    for source in (probability_rows, challenger_rows):
        for raw in source:
            row = dict(raw)
            record_id = str(row["record_id"])
            existing = merged.setdefault(
                record_id,
                {
                    "record_id": record_id,
                    "is_stage_a_candidate": bool(row["is_stage_a_candidate"]),
                    "inside_probability_inclusion_probability": None,
                    "outside_probability_inclusion_probability": None,
                    "challenger_phase1_inclusion_probability": None,
                    "challenger_phase2_inclusion_probability": None,
                    "challenger_combined_inclusion_probability": None,
                    "challenger_score": None,
                    "challenger_score_band": None,
                },
            )
            if existing["is_stage_a_candidate"] != bool(row["is_stage_a_candidate"]):
                raise RuntimeError("record cannot be both inside and outside Stage A")
            for key, value in row.items():
                if key not in {"record_id", "is_stage_a_candidate"} and value is not None:
                    if existing.get(key) not in {None, value}:
                        raise RuntimeError(f"conflicting audit selection metadata for {record_id}")
                    existing[key] = value

    ledger: list[dict[str, Any]] = []
    for record_id, row in sorted(merged.items()):
        channels: list[str] = []
        if row["inside_probability_inclusion_probability"] is not None:
            channels.append("candidate_probability")
        if row["outside_probability_inclusion_probability"] is not None:
            channels.append("noncandidate_probability")
        if row["challenger_combined_inclusion_probability"] is not None:
            channels.append("challenger_two_phase")
        row.update(
            {
                "audit_item_id": audit_item_id(audit_run_id=audit_run_id, record_id=record_id),
                "subreddit": cell["subreddit"],
                "year": cell["year"],
                "content_type": cell["content_type"],
                "selection_channels": channels,
            }
        )
        ledger.append(row)
    ledger.sort(key=lambda row: row["audit_item_id"])
    packet = [
        {
            "audit_item_id": row["audit_item_id"],
            "record_id": row["record_id"],
            "subreddit": row["subreddit"],
            "year": row["year"],
            "content_type": row["content_type"],
            "display_order": index,
        }
        for index, row in enumerate(ledger)
    ]
    assert_id_only_rows(packet)
    assert_id_only_rows(ledger)
    return packet, ledger
