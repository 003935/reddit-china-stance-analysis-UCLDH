"""Pure contracts for calibrated probability-random full-corpus inference.

The runtime module owns private inputs and outputs.  This module keeps the
deterministic shard, cost and decoded prediction contracts independently
testable without Modal, Torch or private Reddit data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from reddit_china_stance import modernbert_probability_random_calibration_v1 as calibration
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    STANCE_CLASSES_B4,
    TARGET_CLASSES,
)
from reddit_china_stance.semantic_evaluation_v2 import decode_factorised_probabilities

SCHEMA_VERSION = "1.0.0"
SOURCE_ROWS = candidate.ELIGIBLE_POPULATION_ROWS
CORPUS_ROWS = candidate.REMAINING_POPULATION_ROWS
SHARD_COUNT = 120
MAXIMUM_CONCURRENT_L4S = 10
SMOKE_ROWS = 128
GPU_TYPE = "L4"
GPU_RATE_USD_PER_SECOND = Decimal("0.000222")
MEASURED_ROWS_PER_SECOND = Decimal("13.23")
MODEL_LOAD_SECONDS_PER_WORKER = Decimal("28.077")
CONSERVATIVE_MULTIPLIER = Decimal("1.5")
MAXIMUM_WORKER_SECONDS = 750
MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL = 2
MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES = 18
FULL_PHASE_MONITORING_TARGET_USD = Decimal("25.00")
CUMULATIVE_MONITORING_TARGET_USD = Decimal("30.00")
PRIOR_FAILED_GPU_SECONDS_ESTIMATE = Decimal("9889.000000")
PRIOR_FAILED_COST_ESTIMATE_USD = Decimal("2.195358")
PRIOR_SMOKE_COST_USD = Decimal("0.052512")
NEW_SMOKE_COST_MONITORING_TARGET_USD = Decimal("0.05")


def private_prediction_columns() -> tuple[str, ...]:
    columns = [
        "corpus_position",
        "opaque_id",
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
        "calibration_member",
        "relevance_probability",
        "relevance",
    ]
    for target in TARGET_CLASSES:
        columns.extend((f"target_{target}_probability", f"target_{target}_present"))
        if target in ANALYTIC_TARGET_CLASSES:
            columns.extend(
                f"stance_{target}_{state}_probability"
                for state in calibration.STANCE_ESTIMAND_CLASSES
            )
            columns.append(f"stance_{target}")
    return tuple(columns)


def shard_plan(
    *, row_count: int = CORPUS_ROWS, shard_count: int = SHARD_COUNT
) -> list[dict[str, int | str]]:
    """Build exact contiguous zero-based corpus-position shards."""

    if (
        type(row_count) is not int
        or type(shard_count) is not int
        or row_count <= 0
        or shard_count <= 0
        or shard_count > row_count
    ):
        raise ValueError("corpus shard dimensions are invalid")
    quotient, remainder = divmod(row_count, shard_count)
    shard_id_width = max(2, len(str(shard_count - 1)))
    result: list[dict[str, int | str]] = []
    start = 0
    for index in range(shard_count):
        size = quotient + (1 if index < remainder else 0)
        stop = start + size
        result.append(
            {
                "shard_id": f"{index:0{shard_id_width}d}",
                "start": start,
                "stop": stop,
                "row_count": size,
            }
        )
        start = stop
    if start != row_count or sum(int(row["row_count"]) for row in result) != row_count:
        raise RuntimeError("corpus shard plan does not conserve rows")
    return result


def compute_projection(
    *,
    row_count: int = CORPUS_ROWS,
    shard_count: int = SHARD_COUNT,
    measured_rows_per_second: Decimal = MEASURED_ROWS_PER_SECOND,
    model_load_seconds_per_worker: Decimal = MODEL_LOAD_SECONDS_PER_WORKER,
    rate_usd_per_gpu_second: Decimal = GPU_RATE_USD_PER_SECOND,
    conservative_multiplier: Decimal = CONSERVATIVE_MULTIPLIER,
) -> dict[str, str | int]:
    """Project aggregate L4 use from the measured six-checkpoint benchmark."""

    values = (
        measured_rows_per_second,
        model_load_seconds_per_worker,
        rate_usd_per_gpu_second,
        conservative_multiplier,
    )
    if (
        type(row_count) is not int
        or type(shard_count) is not int
        or row_count <= 0
        or shard_count <= 0
        or any(not isinstance(value, Decimal) or not value.is_finite() for value in values)
        or measured_rows_per_second <= 0
        or model_load_seconds_per_worker < 0
        or rate_usd_per_gpu_second <= 0
        or conservative_multiplier < 1
    ):
        raise ValueError("compute projection inputs are invalid")
    steady_seconds = Decimal(row_count) / measured_rows_per_second
    model_load_seconds = Decimal(shard_count) * model_load_seconds_per_worker
    projected_seconds = steady_seconds + model_load_seconds
    projected_cost = projected_seconds * rate_usd_per_gpu_second
    conservative_seconds = projected_seconds * conservative_multiplier
    conservative_cost = conservative_seconds * rate_usd_per_gpu_second
    budgeted_base_attempt_seconds = Decimal(shard_count * MAXIMUM_WORKER_SECONDS)
    budgeted_base_attempt_cost = budgeted_base_attempt_seconds * rate_usd_per_gpu_second
    budgeted_recovery_attempt_seconds = Decimal(
        MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES * MAXIMUM_WORKER_SECONDS
    )
    budgeted_recovery_attempt_cost = (
        budgeted_recovery_attempt_seconds * rate_usd_per_gpu_second
    )
    queue_bounded_authorised_seconds = (
        budgeted_base_attempt_seconds + budgeted_recovery_attempt_seconds
    )
    queue_bounded_authorised_cost = (
        queue_bounded_authorised_seconds * rate_usd_per_gpu_second
    )
    queue_bounded_cumulative_cost_projection = (
        PRIOR_FAILED_COST_ESTIMATE_USD
        + PRIOR_SMOKE_COST_USD
        + NEW_SMOKE_COST_MONITORING_TARGET_USD
        + queue_bounded_authorised_cost
    )
    return {
        "row_count": row_count,
        "worker_count": shard_count,
        "measured_rows_per_second": format(measured_rows_per_second, "f"),
        "projected_steady_state_gpu_seconds": format(steady_seconds, ".6f"),
        "projected_model_load_gpu_seconds": format(model_load_seconds, ".6f"),
        "projected_gpu_seconds": format(projected_seconds, ".6f"),
        "projected_cost_usd": format(projected_cost, ".6f"),
        "conservative_multiplier": format(conservative_multiplier, "f"),
        "conservative_gpu_seconds": format(conservative_seconds, ".6f"),
        "conservative_cost_usd": format(conservative_cost, ".6f"),
        "maximum_worker_seconds": MAXIMUM_WORKER_SECONDS,
        "maximum_platform_execution_attempts_per_call": (
            MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL
        ),
        "budgeted_base_attempt_gpu_seconds": format(budgeted_base_attempt_seconds, ".6f"),
        "budgeted_base_attempt_cost_usd": format(budgeted_base_attempt_cost, ".6f"),
        "maximum_platform_preemption_recoveries": MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES,
        "budgeted_recovery_attempt_gpu_seconds": format(
            budgeted_recovery_attempt_seconds, ".6f"
        ),
        "budgeted_recovery_attempt_cost_usd": format(
            budgeted_recovery_attempt_cost, ".6f"
        ),
        "queue_bounded_authorised_execution_gpu_seconds": format(
            queue_bounded_authorised_seconds, ".6f"
        ),
        "queue_bounded_authorised_execution_cost_usd": format(
            queue_bounded_authorised_cost, ".6f"
        ),
        "prior_failed_gpu_seconds_estimate": format(PRIOR_FAILED_GPU_SECONDS_ESTIMATE, ".6f"),
        "prior_failed_cost_estimate_usd": format(PRIOR_FAILED_COST_ESTIMATE_USD, ".6f"),
        "prior_smoke_cost_usd": format(PRIOR_SMOKE_COST_USD, ".6f"),
        "new_smoke_cost_monitoring_target_usd": format(
            NEW_SMOKE_COST_MONITORING_TARGET_USD, ".2f"
        ),
        "queue_bounded_cumulative_cost_projection_usd": format(
            queue_bounded_cumulative_cost_projection, ".6f"
        ),
    }


def validate_compute_authority(compute: Mapping[str, Any]) -> dict[str, Any]:
    """Require the frozen projection and fail above either operational monitoring target."""

    expected = compute_projection()
    for key, value in expected.items():
        if compute.get(key) != value:
            raise ValueError(f"compute authority drifted at {key}")
    try:
        phase_target = Decimal(
            str(compute.get("full_phase_operational_monitoring_target_usd"))
        )
        cumulative_target = Decimal(
            str(compute.get("cumulative_operational_monitoring_target_usd"))
        )
        conservative = Decimal(str(compute["conservative_cost_usd"]))
        budgeted_base = Decimal(str(compute["budgeted_base_attempt_cost_usd"]))
        queue_bounded = Decimal(
            str(compute["queue_bounded_authorised_execution_cost_usd"])
        )
        cumulative = Decimal(
            str(compute["queue_bounded_cumulative_cost_projection_usd"])
        )
    except (InvalidOperation, KeyError) as exc:
        raise ValueError("compute cost authority is malformed") from exc
    if (
        compute.get("gpu_type") != GPU_TYPE
        or compute.get("gpu_fallback_allowed") is not False
        or compute.get("retry_authorised") is not False
        or compute.get("maximum_locked_test_rows") != 0
        or compute.get("maximum_corpus_rows") != CORPUS_ROWS
        or compute.get("provider_billing_hard_cap_enforced") is not False
        or compute.get("proceed_without_provider_hard_cap_authorised") is not True
        or phase_target != FULL_PHASE_MONITORING_TARGET_USD
        or cumulative_target != CUMULATIVE_MONITORING_TARGET_USD
        or conservative > phase_target
        or conservative > cumulative_target
        or budgeted_base > phase_target
        or budgeted_base > cumulative_target
        or queue_bounded > phase_target
        or queue_bounded > cumulative_target
        or cumulative > cumulative_target
    ):
        raise RuntimeError("compute authority exceeds or changes the registered boundary")
    return dict(compute)


def build_private_prediction_row(
    *,
    corpus_position: int,
    opaque_id: str,
    subreddit: str,
    year: int,
    content_type: str,
    retrieval_mode: str,
    calibration_member: bool,
    logits: Mapping[str, Any],
    fitted_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibrate and decode one private row under the three-class stance estimand."""

    if (
        type(corpus_position) is not int
        or corpus_position < 0
        or not isinstance(opaque_id, str)
        or not opaque_id
        or not isinstance(subreddit, str)
        or not subreddit
        or type(year) is not int
        or not isinstance(content_type, str)
        or not content_type
        or not isinstance(retrieval_mode, str)
        or not retrieval_mode
        or type(calibration_member) is not bool
    ):
        raise ValueError("private corpus row metadata is invalid")
    probabilities = calibration.calibrated_probabilities(logits, fitted_calibration)
    decoded = decode_factorised_probabilities(
        relevance_probability=probabilities["relevance_probability"],
        target_probabilities=probabilities["target_probabilities"],
        stance_probabilities=probabilities["stance_probabilities"],
        thresholds=calibration.thresholds_from_calibration(fitted_calibration),
    )
    material = decoded.full_relevance == "material"
    row: dict[str, Any] = {
        "corpus_position": corpus_position,
        "opaque_id": opaque_id,
        "subreddit": subreddit,
        "year": year,
        "content_type": content_type,
        "retrieval_mode": retrieval_mode,
        "calibration_member": calibration_member,
        "relevance_probability": decoded.relevance_probability,
        "relevance": decoded.full_relevance,
    }
    for index, target in enumerate(TARGET_CLASSES):
        raw_present = decoded.full_target_presence[index]
        present = material and raw_present
        row[f"target_{target}_probability"] = decoded.target_probabilities[index]
        row[f"target_{target}_present"] = present
        if target not in ANALYTIC_TARGET_CLASSES:
            continue
        stance_probabilities = decoded.stance_class_probabilities[index]
        by_state = dict(zip(STANCE_CLASSES_B4, stance_probabilities, strict=True))
        for state in calibration.STANCE_ESTIMAND_CLASSES:
            row[f"stance_{target}_{state}_probability"] = by_state[state]
        row[f"stance_{target}"] = decoded.full_stances[index] if present else None
    if tuple(row) != private_prediction_columns():
        raise RuntimeError("private prediction column order drifted")
    return row


def validate_private_prediction_rows(
    rows: Sequence[Mapping[str, Any]], *, start: int, stop: int
) -> list[dict[str, Any]]:
    """Validate exact in-shard position and identity conservation without exposing IDs."""

    if type(start) is not int or type(stop) is not int or start < 0 or stop <= start:
        raise ValueError("prediction shard bounds are invalid")
    clean = [dict(row) for row in rows]
    if len(clean) != stop - start:
        raise ValueError("prediction shard row count drifted")
    positions = [row.get("corpus_position") for row in clean]
    identities = [row.get("opaque_id") for row in clean]
    if positions != list(range(start, stop)):
        raise ValueError("prediction shard positions are not exact and ordered")
    if any(not isinstance(value, str) or not value for value in identities) or len(
        set(identities)
    ) != len(identities):
        raise ValueError("prediction shard identities are invalid or duplicated")
    allowed_stances = set(calibration.STANCE_ESTIMAND_CLASSES)
    for row in clean:
        if tuple(row) != private_prediction_columns():
            raise ValueError("private prediction columns drifted")
        material = row.get("relevance") == "material"
        if row.get("relevance") not in {"material", "not_material"}:
            raise ValueError("decoded relevance is invalid")
        relevance_probability = row.get("relevance_probability")
        if (
            not isinstance(relevance_probability, float)
            or not math.isfinite(relevance_probability)
            or not 0 <= relevance_probability <= 1
        ):
            raise ValueError("relevance probability is invalid")
        for target in TARGET_CLASSES:
            present = row.get(f"target_{target}_present")
            if type(present) is not bool or (present and not material):
                raise ValueError("decoded target presence is not relevance-gated")
            probability = row.get(f"target_{target}_probability")
            if (
                not isinstance(probability, float)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("target probability is invalid")
            if target in ANALYTIC_TARGET_CLASSES:
                stance = row.get(f"stance_{target}")
                if stance is not None and stance not in allowed_stances:
                    raise ValueError("decoded stance is outside the three-class estimand")
                if present != (stance is not None):
                    raise ValueError("decoded stance is not target-gated")
                stance_probabilities = [
                    row[f"stance_{target}_{state}_probability"]
                    for state in calibration.STANCE_ESTIMAND_CLASSES
                ]
                if any(
                    not isinstance(value, float) or not math.isfinite(value) or not 0 <= value <= 1
                    for value in stance_probabilities
                ):
                    raise ValueError("three-class stance probability is invalid")
                total = sum(stance_probabilities)
                if abs(total - 1.0) > 1e-6:
                    raise ValueError("three-class stance probabilities do not sum to one")
    return clean


__all__ = [
    "CONSERVATIVE_MULTIPLIER",
    "CORPUS_ROWS",
    "CUMULATIVE_MONITORING_TARGET_USD",
    "FULL_PHASE_MONITORING_TARGET_USD",
    "GPU_RATE_USD_PER_SECOND",
    "MAXIMUM_CONCURRENT_L4S",
    "MAXIMUM_PLATFORM_EXECUTION_ATTEMPTS_PER_CALL",
    "MAXIMUM_PLATFORM_PREEMPTION_RECOVERIES",
    "MAXIMUM_WORKER_SECONDS",
    "MEASURED_ROWS_PER_SECOND",
    "MODEL_LOAD_SECONDS_PER_WORKER",
    "NEW_SMOKE_COST_MONITORING_TARGET_USD",
    "PRIOR_FAILED_COST_ESTIMATE_USD",
    "PRIOR_FAILED_GPU_SECONDS_ESTIMATE",
    "PRIOR_SMOKE_COST_USD",
    "SHARD_COUNT",
    "SMOKE_ROWS",
    "SOURCE_ROWS",
    "build_private_prediction_row",
    "compute_projection",
    "private_prediction_columns",
    "shard_plan",
    "validate_compute_authority",
    "validate_private_prediction_rows",
]
