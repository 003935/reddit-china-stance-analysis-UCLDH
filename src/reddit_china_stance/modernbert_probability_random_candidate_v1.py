"""Frozen post-acquisition probability-random ModernBERT candidate.

This module contains only storage-agnostic contracts.  It freezes the retained
probability-random checkpoint cohort, the fresh 600-row calibration design and
the deterministic calibration grids.  Reddit text, row identities, logits and
labels remain private; builders for public receipts reject row-level fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
NAMESPACE = "student-modernbert-probability-random-candidate-v1"

ELIGIBLE_PREPARED_RUN_ID = (
    "20aa5baeb55ac09ae57a1e681ebe6886aac247ece48c7aaa7ad05b2c65861d9e"
)
ELIGIBLE_POPULATION_ROWS = 910_141
ELIGIBLE_FRAME_SHA256 = (
    "d9544843792536a144f61a8015e082b9a690023e19c842ddb7c4f1cfe660ce57"
)
ELIGIBLE_PROJECTION_SHA256 = (
    "6ebe5e5d8aa4734b2ce455719ded69db2ae1f610309ba7abc959adb3af5b09b6"
)
ACQUISITION_ROWS = 2_000
REMAINING_POPULATION_ROWS = 908_141
CALIBRATION_ROWS = 600
CALIBRATION_SEED = "modernbert-post-acquisition-calibration-v1-probability-srswor"
# Match the already registered natural-frame design. Retrieval mode remains in
# the private membership ledger for diagnostics, but is not a sampling stratum.
STRATUM_FIELDS = ("year", "subreddit", "content_type")

ACQUISITION_TRAINING_RUN_ID = (
    "4fb74290fe82a65291b0689b74de137fd1c5c905c0f8d9a5e097926df43568b0"
)
ACQUISITION_TRAINING_PHASE_ID = (
    "35f1b38ea600695cbf75ee89aed4848485b0faaca7f1bcd49b07683dd462d12e"
)
ACQUISITION_GATE_ID = (
    "b9d7d66399b7e4e9205f00b0912bdb357aefb868310cabecd1a87e369cce33a1"
)
ACQUISITION_CLOSEOUT_ID = (
    "9e4f803ba828db1207f8675d9e167fd99c28f83fdaab2222272dc1ee3e6e641d"
)
ACQUISITION_LEDGER_ID = (
    "532062f085d6ac6ed41b603b7f987157f32f45f06d9e48e58af40e4ddcfa3f74"
)
ACQUISITION_LEDGER_FILE_SHA256 = (
    "10c2555efad8d87d7ba79a30fa6a4547c5584ad600dca79c011585aac2d7dc16"
)
ACQUISITION_VERDICT = "retain_probability_random"
RANDOM_TRAINING_FRAME_SHA256 = (
    "f09fe87cb4bcb00dd8feb6454a774e897882abc3de39a8d1478f54bf247cd94a"
)
SEEDS = (47, 61, 89)
COMPONENTS = ("relevance", "target_stance_b4")
TEMPERATURE_GRID = (0.50, 0.67, 0.80, 1.00, 1.25, 1.50, 2.00)
ABSTENTION_MARGIN_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)
STANCE_CONFIDENCE_GRID = (0.0, 0.40, 0.50, 0.60, 0.70, 0.80)


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    component: str
    seed: int
    selected_epoch: int
    trial_id: str
    checkpoint_sha256: str


CHECKPOINT_SPECS = (
    CheckpointSpec(
        "relevance",
        47,
        6,
        "aef38d0c31da6e44887f5464984e180a1fa7f0aaf29e4bd1b0584a9819247627",
        "5d8bdb41daad71f8efbaf3b960f99e0e52ea4c7abd4ddbc3a298dadd209dab84",
    ),
    CheckpointSpec(
        "relevance",
        61,
        4,
        "2396b9bd895cb73d05786ae1611fdfb318d8259462d75df7bddf2ddd292798b3",
        "a2f2cb2f60ec12f4590f6e130c13af393a7a4e490a64ce2d3933adf1c70ac5af",
    ),
    CheckpointSpec(
        "relevance",
        89,
        5,
        "7eda155ca88eb2ca494ae4639ec8a9b3343cb153e31c3f4d358fb3b0f5c31596",
        "cc6769b6a5860880253e0b906099e6f5d00d40c453dfda11fc879f3f10898ddf",
    ),
    CheckpointSpec(
        "target_stance_b4",
        47,
        7,
        "033e81ca52e47e7962808087371186c66c51464f69977371e346d37fb1dabe9c",
        "bfd8c08ce83ea86fbdbb7a427422a30adb13348da15101b74b6d81de751d3c38",
    ),
    CheckpointSpec(
        "target_stance_b4",
        61,
        7,
        "50b58700074018b8d6b5df63e9e5d211ebac8117a12256a0eae9d23d51e0f624",
        "c028882e33d7c01491dfad478625845890e9644f5ec22cb86f26f1f9505dd2b6",
    ),
    CheckpointSpec(
        "target_stance_b4",
        89,
        6,
        "41c972ffb2f90c367ec07b3f02094f500ca16bb2d4fb5814b930dabeddf555ea",
        "7e363fb8808e9f3a9d955344abd8b9513035ee97ee0235c1de43971271d44381",
    ),
)


class ProbabilityRandomCandidateContractError(RuntimeError):
    """Raised when a frozen candidate binding or design drifts."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def stratum_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: row.get(field) for field in STRATUM_FIELDS}
    if (
        not isinstance(result["subreddit"], str)
        or not result["subreddit"]
        or type(result["year"]) is not int
        or result["year"] not in range(2020, 2026)
        or not isinstance(result["content_type"], str)
        or not result["content_type"]
    ):
        raise ValueError("calibration stratum is invalid")
    return result


def stratum_key(stratum: Mapping[str, Any]) -> str:
    return canonical_sha256(stratum_from_row(stratum))


def selection_tiebreak_sha256(
    opaque_id: str,
    *,
    stratum: Mapping[str, Any],
    seed: str = CALIBRATION_SEED,
) -> str:
    if not isinstance(opaque_id, str) or not opaque_id:
        raise ValueError("opaque_id must be a non-empty string")
    if not isinstance(seed, str) or not seed:
        raise ValueError("selection seed must be a non-empty string")
    return canonical_sha256([seed, stratum_from_row(stratum), opaque_id])


def allocate_minimum_one_quotas(
    strata_counts: Sequence[Mapping[str, Any]],
    *,
    sample_rows: int = CALIBRATION_ROWS,
    seed: str = CALIBRATION_SEED,
) -> list[dict[str, Any]]:
    """Allocate minimum-one, then proportional residual capacity by largest remainder."""

    if type(sample_rows) is not int or sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer")
    if isinstance(strata_counts, (str, bytes)) or not strata_counts:
        raise ValueError("strata_counts must be a non-empty sequence")
    clean: dict[str, dict[str, Any]] = {}
    for raw in strata_counts:
        if not isinstance(raw, Mapping) or set(raw) != {"stratum", "population_rows"}:
            raise ValueError("stratum count fields drifted")
        stratum = stratum_from_row(raw["stratum"])
        population = raw["population_rows"]
        if type(population) is not int or population <= 0:
            raise ValueError("stratum population must be positive")
        key = stratum_key(stratum)
        if key in clean:
            raise ValueError("duplicate calibration stratum")
        clean[key] = {"stratum": stratum, "population_rows": population}
    if len(clean) > sample_rows:
        raise ValueError("sample cannot allocate one row per observed stratum")
    total = sum(row["population_rows"] for row in clean.values())
    if sample_rows > total:
        raise ValueError("sample quota exceeds remaining population")

    quota = {key: 1 for key in clean}
    residual_quota = sample_rows - len(clean)
    residual_total = total - len(clean)
    remainders: list[tuple[float, str, str]] = []
    allocated_residual = 0
    if residual_quota:
        if residual_total <= 0:
            raise RuntimeError("positive residual quota has no residual capacity")
        for key, row in clean.items():
            capacity = row["population_rows"] - 1
            exact = residual_quota * capacity / residual_total
            base = min(capacity, math.floor(exact))
            quota[key] += base
            allocated_residual += base
            remainders.append(
                (exact - base, canonical_sha256([seed, "quota", key]), key)
            )
        remaining = residual_quota - allocated_residual
        for _, _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if not remaining:
                break
            if quota[key] < clean[key]["population_rows"]:
                quota[key] += 1
                remaining -= 1
        if remaining:
            raise RuntimeError("largest-remainder allocation did not conserve quota")

    result = [
        {
            "stratum": clean[key]["stratum"],
            "population_rows": clean[key]["population_rows"],
            "sample_rows": quota[key],
        }
        for key in sorted(clean)
    ]
    if sum(row["sample_rows"] for row in result) != sample_rows:
        raise RuntimeError("calibration quota did not conserve sample rows")
    if any(row["sample_rows"] > row["population_rows"] for row in result):
        raise RuntimeError("calibration quota exceeds a stratum population")
    return result


def build_membership_row(
    row: Mapping[str, Any],
    *,
    population_rows: int,
    sample_rows: int,
    rank: int,
) -> dict[str, Any]:
    required = (
        "opaque_id",
        "thread_id",
        "near_duplicate_cluster_id",
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
        "target_text",
        "parent_context",
        "submission_context",
    )
    if not set(required) <= set(row):
        raise ValueError("selected calibration row omits required fields")
    if any(not isinstance(row[field], str) or not row[field] for field in (
        "opaque_id",
        "thread_id",
        "near_duplicate_cluster_id",
        "target_text",
    )):
        raise ValueError("selected calibration identity or text is invalid")
    for field in ("parent_context", "submission_context"):
        if row[field] is not None and not isinstance(row[field], str):
            raise ValueError("selected calibration context is invalid")
    if (
        type(population_rows) is not int
        or type(sample_rows) is not int
        or type(rank) is not int
        or not 0 < sample_rows <= population_rows
        or not 0 < rank <= sample_rows
    ):
        raise ValueError("calibration design counts are invalid")
    stratum = stratum_from_row(row)
    return {
        **{key: row[key] for key in required},
        "selection_component": "calibration_probability",
        "selection_stratum": json.dumps(
            stratum, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "selection_rank": rank,
        "selection_tiebreak_sha256": selection_tiebreak_sha256(
            row["opaque_id"], stratum=stratum
        ),
        "inclusion_probability_numerator": sample_rows,
        "inclusion_probability_denominator": population_rows,
        "inclusion_probability": sample_rows / population_rows,
        "probability_scope": "post-acquisition-eligible-candidate-frame",
    }


def validate_membership_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or len(rows) != CALIBRATION_ROWS:
        raise ValueError(f"calibration membership must contain {CALIBRATION_ROWS} rows")
    clean = [dict(row) for row in rows]
    for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id"):
        values = [row.get(field) for row in clean]
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"calibration membership has invalid {field}")
        if len(values) != len(set(values)):
            raise ValueError(f"calibration membership contains duplicate {field}")
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in clean:
        stratum = stratum_from_row(row)
        key = stratum_key(stratum)
        expected_stratum = json.dumps(
            stratum, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if row.get("selection_component") != "calibration_probability":
            raise ValueError("calibration selection component drifted")
        if row.get("selection_stratum") != expected_stratum:
            raise ValueError("calibration stored stratum drifted")
        if row.get("selection_tiebreak_sha256") != selection_tiebreak_sha256(
            row["opaque_id"], stratum=stratum
        ):
            raise ValueError("calibration tiebreak drifted")
        numerator = row.get("inclusion_probability_numerator")
        denominator = row.get("inclusion_probability_denominator")
        probability = row.get("inclusion_probability")
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or type(row.get("selection_rank")) is not int
            or not 0 < numerator <= denominator
            or not isinstance(probability, float)
            or not math.isclose(probability, numerator / denominator, abs_tol=1e-15)
            or row.get("probability_scope")
            != "post-acquisition-eligible-candidate-frame"
        ):
            raise ValueError("calibration inclusion probability drifted")
        by_stratum.setdefault(key, []).append(row)
    for members in by_stratum.values():
        numerators = {row["inclusion_probability_numerator"] for row in members}
        denominators = {row["inclusion_probability_denominator"] for row in members}
        ranks = sorted(row["selection_rank"] for row in members)
        if len(numerators) != 1 or len(denominators) != 1:
            raise ValueError("calibration stratum design counts disagree")
        if next(iter(numerators)) != len(members) or ranks != list(range(1, len(members) + 1)):
            raise ValueError("calibration stratum ranks or quota drifted")
    return sorted(
        clean,
        key=lambda row: (stratum_key(row), row["selection_tiebreak_sha256"]),
    )


def sampling_public_receipt(
    *,
    membership_rows: Sequence[Mapping[str, Any]],
    membership_artifact: Mapping[str, Any],
    teacher_source_artifact: Mapping[str, Any],
    acquisition_ledger_sha256: str,
    source_bundle_sha256: str,
    checkpoint_cohort_id: str,
    run_id: str,
) -> dict[str, Any]:
    clean = validate_membership_rows(membership_rows)
    if not all(is_sha256(value) for value in (
        acquisition_ledger_sha256,
        source_bundle_sha256,
        checkpoint_cohort_id,
        run_id,
    )):
        raise ValueError("sampling input binding is not a SHA-256")
    probabilities = [row["inclusion_probability"] for row in clean]
    weights = [1.0 / value for value in probabilities]
    kish = sum(weights) ** 2 / sum(value * value for value in weights)
    strata = {row["selection_stratum"] for row in clean}
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-probability-random-calibration-sample-receipt-v1",
        "status": "complete",
        "run_id": run_id,
        "eligible_prepared_run_id": ELIGIBLE_PREPARED_RUN_ID,
        "eligible_population_count": ELIGIBLE_POPULATION_ROWS,
        "eligible_frame_sha256": ELIGIBLE_FRAME_SHA256,
        "eligible_projection_sha256": ELIGIBLE_PROJECTION_SHA256,
        "acquisition_exclusion_count": ACQUISITION_ROWS,
        "post_acquisition_population_count": REMAINING_POPULATION_ROWS,
        "sample_count": CALIBRATION_ROWS,
        "observed_stratum_count": len(strata),
        "sampling_seed_sha256": hashlib.sha256(CALIBRATION_SEED.encode()).hexdigest(),
        "sampling_design": "minimum-one-residual-capacity-largest-remainder-srswor-v1",
        "minimum_inclusion_probability": min(probabilities),
        "maximum_inclusion_probability": max(probabilities),
        "inverse_weight_ratio": max(weights) / min(weights),
        "kish_effective_sample_size": kish,
        "identity_overlap_count": 0,
        "thread_overlap_count": 0,
        "near_duplicate_overlap_count": 0,
        "membership_artifact": dict(membership_artifact),
        "teacher_source_artifact": dict(teacher_source_artifact),
        "acquisition_ledger_sha256": acquisition_ledger_sha256,
        "acquisition_ledger_id": ACQUISITION_LEDGER_ID,
        "acquisition_gate_id": ACQUISITION_GATE_ID,
        "acquisition_closeout_id": ACQUISITION_CLOSEOUT_ID,
        "checkpoint_cohort_id": checkpoint_cohort_id,
        "source_bundle_sha256": source_bundle_sha256,
        "locked_test_access_count": 0,
        "raw_text_in_receipt": False,
        "row_identity_in_receipt": False,
        "evidence_boundary": "model-assisted-calibration-selection-not-human-validation",
    }
    assert_metadata_only(body, where="probability-random calibration sample receipt")
    return {**body, "receipt_id": canonical_sha256(body)}


def checkpoint_relative_path(spec: CheckpointSpec) -> str:
    return (
        "student-modernbert-acquisition-training-v1/"
        f"run={ACQUISITION_TRAINING_RUN_ID}/phase=paired-acquisition-comparison/"
        f"arm=random/component={spec.component}/trial={spec.trial_id}/checkpoint.pt"
    )


def checkpoint_receipt_relative_path(spec: CheckpointSpec) -> str:
    return checkpoint_relative_path(spec).removesuffix("checkpoint.pt") + "receipt.json"


def freeze_checkpoint_cohort(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the six completed random-arm receipts and freeze mean-logit ensembling."""

    if isinstance(receipts, (str, bytes)) or len(receipts) != len(CHECKPOINT_SPECS):
        raise ValueError("checkpoint cohort requires exactly six receipts")
    by_trial = {receipt.get("trial_id"): receipt for receipt in receipts}
    if len(by_trial) != len(CHECKPOINT_SPECS):
        raise ValueError("checkpoint receipts contain duplicate or missing trials")
    members: list[dict[str, Any]] = []
    for spec in CHECKPOINT_SPECS:
        receipt = by_trial.get(spec.trial_id)
        if not isinstance(receipt, Mapping):
            raise ValueError("checkpoint receipt is missing")
        artifacts = receipt.get("artifacts")
        checkpoint = artifacts.get("checkpoint") if isinstance(artifacts, Mapping) else None
        aggregate = receipt.get("aggregate_metrics")
        if (
            receipt.get("experiment_run_id") != ACQUISITION_TRAINING_RUN_ID
            or receipt.get("phase_run_id") != ACQUISITION_TRAINING_PHASE_ID
            or receipt.get("arm") != "random"
            or receipt.get("component") != spec.component
            or receipt.get("optimiser_seed") != spec.seed
            or receipt.get("training_frame_sha256") != RANDOM_TRAINING_FRAME_SHA256
            or receipt.get("invalid_outputs") != 0
            or receipt.get("locked_test_rows_accessed") != 0
            or not isinstance(aggregate, Mapping)
            or aggregate.get("selected_epoch") != spec.selected_epoch
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("sha256") != spec.checkpoint_sha256
            or checkpoint.get("relative_path") != "checkpoint.pt"
            or type(checkpoint.get("bytes")) is not int
            or checkpoint["bytes"] <= 0
            or not is_sha256(receipt.get("receipt_id"))
        ):
            raise ProbabilityRandomCandidateContractError(
                f"retained checkpoint receipt drifted for {spec.component} seed {spec.seed}"
            )
        members.append(
            {
                "component": spec.component,
                "seed": spec.seed,
                "selected_epoch": spec.selected_epoch,
                "trial_id": spec.trial_id,
                "checkpoint": {
                    "relative_path": checkpoint_relative_path(spec),
                    "sha256": spec.checkpoint_sha256,
                    "bytes": checkpoint["bytes"],
                },
                "receipt_id": receipt.get("receipt_id"),
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "modernbert-probability-random-checkpoint-cohort-v1",
        "acquisition_gate_id": ACQUISITION_GATE_ID,
        "acquisition_verdict": ACQUISITION_VERDICT,
        "training_run_id": ACQUISITION_TRAINING_RUN_ID,
        "training_phase_id": ACQUISITION_TRAINING_PHASE_ID,
        "training_frame_sha256": RANDOM_TRAINING_FRAME_SHA256,
        "members": members,
        "combination": "unweighted-arithmetic-mean-aligned-raw-logits-per-component",
        "stance_representation": "B4-four-class-including-mixed",
        "seed_selection": "all-three-preregistered-paired-seeds-no-post-hoc-seed-selection",
        "locked_test_access_count": 0,
    }
    return {**body, "cohort_id": canonical_sha256(body)}


def validate_parent_closeout(closeout: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact completed acquisition decision that retained random."""

    if not isinstance(closeout, Mapping):
        raise ValueError("acquisition closeout must be an object")
    body = {key: value for key, value in closeout.items() if key != "closeout_id"}
    if (
        closeout.get("kind") != "modernbert-acquisition-gate-v1"
        or closeout.get("closeout_id") != ACQUISITION_CLOSEOUT_ID
        or canonical_sha256(body) != ACQUISITION_CLOSEOUT_ID
        or closeout.get("gate_id") != ACQUISITION_GATE_ID
        or closeout.get("experiment_run_id") != ACQUISITION_TRAINING_RUN_ID
        or closeout.get("phase_run_id") != ACQUISITION_TRAINING_PHASE_ID
        or closeout.get("verdict") != ACQUISITION_VERDICT
        or closeout.get("passed") is not False
        or closeout.get("locked_test_rows_accessed") != 0
        or closeout.get("corpus_inference_authorised") is not False
        or closeout.get("invalid_outputs") != 0
    ):
        raise ProbabilityRandomCandidateContractError(
            "acquisition closeout differs from the exact retained-random decision"
        )
    return json.loads(json.dumps(closeout, sort_keys=True, allow_nan=False))


__all__ = [
    "ABSTENTION_MARGIN_GRID",
    "ACQUISITION_CLOSEOUT_ID",
    "ACQUISITION_GATE_ID",
    "ACQUISITION_LEDGER_FILE_SHA256",
    "ACQUISITION_LEDGER_ID",
    "ACQUISITION_ROWS",
    "CALIBRATION_ROWS",
    "CALIBRATION_SEED",
    "CHECKPOINT_SPECS",
    "COMPONENTS",
    "ELIGIBLE_FRAME_SHA256",
    "ELIGIBLE_POPULATION_ROWS",
    "ELIGIBLE_PREPARED_RUN_ID",
    "NAMESPACE",
    "REMAINING_POPULATION_ROWS",
    "SEEDS",
    "STANCE_CONFIDENCE_GRID",
    "STRATUM_FIELDS",
    "TEMPERATURE_GRID",
    "ProbabilityRandomCandidateContractError",
    "allocate_minimum_one_quotas",
    "build_membership_row",
    "canonical_sha256",
    "checkpoint_receipt_relative_path",
    "checkpoint_relative_path",
    "freeze_checkpoint_cohort",
    "sampling_public_receipt",
    "selection_tiebreak_sha256",
    "stratum_from_row",
    "stratum_key",
    "validate_membership_rows",
    "validate_parent_closeout",
]
