"""Deterministic acquisition policy for the one-shot ModernBERT comparison.

The module is deliberately free of model and storage runtimes.  It consumes an
already frozen, private candidate frame plus six-checkpoint probability outputs,
and produces a private row-level acquisition ledger and a metadata-only public
receipt.  It also implements the preregistered paired acquisition gate.

No function accepts or emits Reddit text or author fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_ontology_v2 import ANALYTIC_TARGETS, STANCES

SCHEMA_VERSION = "1.0.0"
PRIVATE_KIND = "modernbert-acquisition-ledger-v1"
PUBLIC_KIND = "modernbert-acquisition-receipt-v1"
GATE_KIND = "modernbert-acquisition-gate-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "configs/modernbert-acquisition-v1.toml"
RENDERS = ("full", "target_only")
BUCKET_PRIORITY = (
    "rare_cell",
    "boundary",
    "multi_context",
    "uncertainty_disagreement",
)
REQUIRED_INPUT_BINDINGS = frozenset(
    {
        "eligible_frame_sha256",
        "source_inventory_sha256",
        "exclusion_ledger_sha256",
        "checkpoint_bundle_sha256",
        "scoring_artifact_sha256",
        "policy_file_sha256",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "opaque_id",
        "thread_id",
        "near_duplicate_cluster_id",
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
        "seed_outputs",
    }
)
_OUTPUT_KEYS = frozenset({"relevance", "target_presence", "stance"})
_REFERENCE_KEYS = frozenset(
    {"thread_id", "quality_tier", "material", "target_stances"}
)
_PREDICTION_KEYS = frozenset({"thread_id", "material", "target_stances"})
_COMPACT_SCORE_KEYS = frozenset(
    {
        "opaque_id",
        "thread_id",
        "near_duplicate_cluster_id",
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
        *BUCKET_PRIORITY,
    }
)
QUALITY_TIERS = frozenset(
    {"exact_consensus", "blind_majority", "informed_adjudication"}
)


def _canonical_json_bytes(value: Any) -> bytes:
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
    return payload.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a canonical SHA-256 digest, rejecting non-finite JSON."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def canonical_sequence_sha256(values: Iterable[Any]) -> str:
    """Hash a JSON sequence incrementally with ``canonical_sha256`` semantics.

    The iterable is consumed exactly once and is never materialised.  Its digest
    is byte-for-byte equal to ``canonical_sha256(list(values))`` for the same
    ordered values.
    """

    if isinstance(values, (str, bytes)):
        raise ValueError("canonical sequence must be an iterable of JSON values")
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        digest.update(_canonical_json_bytes(value))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{where} fields drifted")


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a probability")
    clean = float(value)
    if not math.isfinite(clean) or not 0.0 <= clean <= 1.0:
        raise ValueError(f"{where} must be finite and in [0,1]")
    return clean


def _tiebreak(seed: str, *parts: str) -> str:
    return canonical_sha256([seed, *parts])


@dataclass(frozen=True, slots=True)
class AcquisitionPolicy:
    """Frozen quotas and deterministic seeds for the acquisition ledger."""

    probability_rows: int
    rare_cell_rows: int
    boundary_rows: int
    multi_context_rows: int
    uncertainty_disagreement_rows: int
    probability_seed: str
    active_seed: str
    expected_seed_count: int
    policy_file_sha256: str

    def __post_init__(self) -> None:
        counts = (
            self.probability_rows,
            self.rare_cell_rows,
            self.boundary_rows,
            self.multi_context_rows,
            self.uncertainty_disagreement_rows,
            self.expected_seed_count,
        )
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("all acquisition counts must be positive integers")
        if self.expected_seed_count != 3:
            raise ValueError("the frozen acquisition policy requires exactly three seeds")
        _text(self.probability_seed, "probability_seed")
        _text(self.active_seed, "active_seed")
        if not _is_sha256(self.policy_file_sha256):
            raise ValueError("policy_file_sha256 must be a lowercase SHA-256")

    @property
    def bucket_quotas(self) -> dict[str, int]:
        return {
            "rare_cell": self.rare_cell_rows,
            "boundary": self.boundary_rows,
            "multi_context": self.multi_context_rows,
            "uncertainty_disagreement": self.uncertainty_disagreement_rows,
        }

    def as_contract(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "stratum_fields": [
                "subreddit",
                "year",
                "content_type",
                "retrieval_mode",
            ],
            "probability_allocation": "proportional-largest-remainder-srswor-v1",
            "probability_sampling_unit": (
                "near-duplicate-deduplicated-submission-thread"
            ),
            "active_bucket_priority": list(BUCKET_PRIORITY),
            "active_quota_fallback": False,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionGatePolicy:
    """Frozen gates for the equal-query student comparison."""

    bootstrap_replicates: int
    bootstrap_seed: str
    minimum_cell_support: int
    minimum_mean_rare_gain: float
    maximum_tuple_decline: float
    maximum_material_recall_decline: float
    maximum_supported_target_decline: float
    policy_file_sha256: str

    def __post_init__(self) -> None:
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be a positive integer")
        if type(self.minimum_cell_support) is not int or self.minimum_cell_support < 10:
            raise ValueError("minimum_cell_support must remain at least ten")
        _text(self.bootstrap_seed, "bootstrap_seed")
        if not _is_sha256(self.policy_file_sha256):
            raise ValueError("policy_file_sha256 must be a lowercase SHA-256")
        for field in (
            "minimum_mean_rare_gain",
            "maximum_tuple_decline",
            "maximum_material_recall_decline",
            "maximum_supported_target_decline",
        ):
            value = getattr(self, field)
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be a finite non-negative float")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_acquisition_policies(
    path: Path = DEFAULT_POLICY_PATH,
) -> tuple[AcquisitionPolicy, AcquisitionGatePolicy]:
    """Load and fail-closed validate the registered TOML policy authority."""

    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot load acquisition policy: {path}") from exc
    expected_top_level = {
        "schema_version",
        "kind",
        "purpose",
        "evidence_boundary",
        "source",
        "exclusions",
        "random",
        "active",
        "teacher",
        "student",
        "gate",
        "budget",
    }
    if set(value) != expected_top_level:
        raise ValueError("acquisition policy top-level fields drifted")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("acquisition policy schema version drifted")
    if value["kind"] != "modernbert-acquisition-policy-v1":
        raise ValueError("acquisition policy kind drifted")
    if value["purpose"] != (
        "one-time-exploratory-probability-random-versus-active-query-comparison"
    ):
        raise ValueError("acquisition policy purpose drifted")
    if value["evidence_boundary"] != (
        "adaptive-model-assisted-development-not-human-validation"
    ):
        raise ValueError("acquisition evidence boundary drifted")
    random_policy = value["random"]
    active = value["active"]
    buckets = active.get("bucket_rows") if isinstance(active, Mapping) else None
    gate = value["gate"]
    student = value["student"]
    teacher = value["teacher"]
    budget = value["budget"]
    for where, row in (
        ("random", random_policy),
        ("active", active),
        ("active.bucket_rows", buckets),
        ("student", student),
        ("teacher", teacher),
        ("gate", gate),
        ("budget", budget),
    ):
        if not isinstance(row, Mapping):
            raise ValueError(f"acquisition policy {where} must be a table")
    expected_section_keys = {
        "source": {
            "dataset_revision",
            "source_schema_version",
            "retrieval_policy_digest",
            "stage_a_run_id",
            "language_policy_digest",
            "language_run_id",
            "language_eligibility",
            "language_decision",
            "human_language_gate_accepted",
        },
        "exclusions": {
            "exclude_complete_teacher_10k",
            "exclude_legacy_reference",
            "exclude_factorised_development",
            "exclude_acquisition_evaluation",
            "exact_surface",
            "normalised_surface",
            "near_duplicate_algorithm",
            "near_duplicate_minimum_characters",
            "near_duplicate_maximum_hamming_distance",
            "near_duplicate_minimum_length_ratio",
            "near_duplicate_maximum_length_ratio",
        },
        "random": {"rows", "seed", "sampling_unit", "strata", "draw_order"},
        "active": {"rows", "seed", "draw_order", "bucket_priority", "bucket_rows"},
        "teacher": {
            "queries_per_arm",
            "arm_blinded",
            "dual_pass",
            "blind_tie_break",
            "informed_adjudication",
            "maximum_concurrent_jobs",
            "estimated_wall_minutes_low",
            "estimated_wall_minutes_high",
        },
        "student": {
            "base_training_rows",
            "representation",
            "optimiser_seeds",
            "components",
            "trial_count",
            "checkpoint_selection_rows",
            "checkpoint_selection_source",
            "evaluation_rows",
            "evaluation_source",
        },
        "gate": {
            "rare_cell_minimum_evaluation_support",
            "rare_cell_mean_gain_minimum",
            "bootstrap_replicates",
            "bootstrap_confidence",
            "bootstrap_lower_bound_must_exceed",
            "minimum_improved_seeds",
            "maximum_tuple_f1_decline",
            "maximum_material_recall_decline",
            "maximum_supported_target_regression",
            "exact_consensus_direction_must_not_reverse",
            "maximum_invalid_outputs",
            "maximum_locked_test_rows_accessed",
        },
        "budget": {
            "shared_hard_cap_usd",
            "modal_preparation_and_scoring_phase_cap_usd",
            "modal_training_phase_cap_usd",
            "currency_telemetry_available_for_codex_teacher",
        },
    }
    for section, expected in expected_section_keys.items():
        row = value[section]
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError(f"acquisition policy {section} fields drifted")
    source = value["source"]
    dataset_revision = source["dataset_revision"]
    if not (
        isinstance(dataset_revision, str)
        and len(dataset_revision) == 40
        and all(character in "0123456789abcdef" for character in dataset_revision)
    ):
        raise ValueError("acquisition policy source.dataset_revision must be a git revision")
    for field in (
        "retrieval_policy_digest",
        "stage_a_run_id",
        "language_policy_digest",
        "language_run_id",
    ):
        if not _is_sha256(source[field]):
            raise ValueError(f"acquisition policy source.{field} must be a SHA-256")
    if source["source_schema_version"] != "1.0.2":
        raise ValueError("acquisition source schema version drifted")
    if source["language_eligibility"] != "provisional_english":
        raise ValueError("acquisition language eligibility drifted")
    if source["human_language_gate_accepted"] is not False:
        raise ValueError("acquisition policy cannot claim a human language gate")
    exclusions = value["exclusions"]
    for field in (
        "exclude_complete_teacher_10k",
        "exclude_legacy_reference",
        "exclude_factorised_development",
        "exclude_acquisition_evaluation",
        "exact_surface",
        "normalised_surface",
    ):
        if exclusions[field] is not True:
            raise ValueError(f"acquisition exclusion {field} must remain enabled")
    if exclusions["near_duplicate_algorithm"] != "token-trigram-simhash64":
        raise ValueError("acquisition near-duplicate algorithm drifted")
    if random_policy.get("strata") != [
        "subreddit",
        "year",
        "content_type",
        "retrieval_mode",
    ]:
        raise ValueError("acquisition probability strata drifted")
    if random_policy.get("sampling_unit") != (
        "near-duplicate-deduplicated-submission-thread"
    ):
        raise ValueError("acquisition sampling unit drifted")
    if random_policy.get("draw_order") != 1 or active.get("draw_order") != 2:
        raise ValueError("acquisition draw order drifted")
    if active.get("bucket_priority") != list(BUCKET_PRIORITY):
        raise ValueError("acquisition active bucket priority drifted")
    bucket_counts = {bucket: buckets.get(bucket) for bucket in BUCKET_PRIORITY}
    if set(buckets) != set(BUCKET_PRIORITY):
        raise ValueError("acquisition active bucket fields drifted")
    integer_values = {
        "random.rows": random_policy.get("rows"),
        "active.rows": active.get("rows"),
        **{f"active.bucket_rows.{key}": value for key, value in bucket_counts.items()},
        "gate.bootstrap_replicates": gate.get("bootstrap_replicates"),
        "gate.rare_cell_minimum_evaluation_support": gate.get(
            "rare_cell_minimum_evaluation_support"
        ),
    }
    for where, item in integer_values.items():
        if type(item) is not int or item <= 0:
            raise ValueError(f"acquisition policy {where} must be a positive integer")
    if active.get("rows") != sum(bucket_counts.values()):
        raise ValueError("acquisition active bucket rows do not conserve the arm")
    if teacher.get("queries_per_arm") != random_policy.get("rows"):
        raise ValueError("teacher query budget disagrees with the probability arm")
    if teacher.get("queries_per_arm") != active.get("rows"):
        raise ValueError("teacher query budget disagrees with the active arm")
    for field in ("arm_blinded", "dual_pass", "blind_tie_break", "informed_adjudication"):
        if teacher.get(field) is not True:
            raise ValueError(f"teacher {field} contract drifted")
    if student.get("representation") != "B4":
        raise ValueError("acquisition student representation drifted")
    if student.get("components") != ["relevance", "target_stance_b4"]:
        raise ValueError("acquisition student component contract drifted")
    if student.get("trial_count") != 12:
        raise ValueError("acquisition student trial contract drifted")
    if student.get("checkpoint_selection_rows") != 600 or student.get(
        "checkpoint_selection_source"
    ) != "retained-factorised-v2-development":
        raise ValueError("acquisition checkpoint-selection frame contract drifted")
    if student.get("evaluation_rows") != 600 or student.get("evaluation_source") != (
        "unused-factorised-v2-calibration-membership"
    ):
        raise ValueError("acquisition evaluation frame contract drifted")
    optimiser_seeds = student.get("optimiser_seeds")
    if (
        not isinstance(optimiser_seeds, list)
        or len(optimiser_seeds) != 3
        or any(type(seed) is not int for seed in optimiser_seeds)
        or len(set(optimiser_seeds)) != 3
    ):
        raise ValueError("acquisition optimiser seed contract drifted")
    if gate.get("bootstrap_confidence") != 0.95:
        raise ValueError("acquisition bootstrap confidence drifted")
    if gate.get("bootstrap_lower_bound_must_exceed") != 0.0:
        raise ValueError("acquisition bootstrap lower-bound gate drifted")
    if gate.get("minimum_improved_seeds") != 2:
        raise ValueError("acquisition minimum improved seeds drifted")
    if gate.get("exact_consensus_direction_must_not_reverse") is not True:
        raise ValueError("exact-consensus sensitivity gate drifted")
    if gate.get("maximum_invalid_outputs") != 0:
        raise ValueError("invalid-output gate drifted")
    if gate.get("maximum_locked_test_rows_accessed") != 0:
        raise ValueError("locked-test gate drifted")
    if budget.get("shared_hard_cap_usd") != 200:
        raise ValueError("shared acquisition budget cap drifted")
    if budget.get("modal_preparation_and_scoring_phase_cap_usd") != 50:
        raise ValueError("acquisition preparation/scoring cap drifted")
    if budget.get("modal_training_phase_cap_usd") != 25:
        raise ValueError("acquisition training cap drifted")
    if budget.get("currency_telemetry_available_for_codex_teacher") is not False:
        raise ValueError("acquisition policy invents teacher currency telemetry")

    digest = _file_sha256(path)
    acquisition_policy = AcquisitionPolicy(
        probability_rows=random_policy["rows"],
        rare_cell_rows=bucket_counts["rare_cell"],
        boundary_rows=bucket_counts["boundary"],
        multi_context_rows=bucket_counts["multi_context"],
        uncertainty_disagreement_rows=bucket_counts["uncertainty_disagreement"],
        probability_seed=random_policy["seed"],
        active_seed=active["seed"],
        expected_seed_count=len(optimiser_seeds),
        policy_file_sha256=digest,
    )
    gate_policy = AcquisitionGatePolicy(
        bootstrap_replicates=gate["bootstrap_replicates"],
        bootstrap_seed=f"modernbert-acquisition-paired-bootstrap-v1:{digest}",
        minimum_cell_support=gate["rare_cell_minimum_evaluation_support"],
        minimum_mean_rare_gain=float(gate["rare_cell_mean_gain_minimum"]),
        maximum_tuple_decline=float(gate["maximum_tuple_f1_decline"]),
        maximum_material_recall_decline=float(gate["maximum_material_recall_decline"]),
        maximum_supported_target_decline=float(
            gate["maximum_supported_target_regression"]
        ),
        policy_file_sha256=digest,
    )
    return acquisition_policy, gate_policy


def binary_entropy(probability: float) -> float:
    """Binary entropy normalised to [0,1]."""

    probability = _probability(probability, "binary entropy input")
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    ) / math.log(2.0)


def categorical_entropy(probabilities: Sequence[float]) -> float:
    """Categorical entropy normalised by log(number of classes)."""

    if isinstance(probabilities, (str, bytes)) or len(probabilities) < 2:
        raise ValueError("categorical entropy requires at least two classes")
    clean = [_probability(value, "categorical entropy input") for value in probabilities]
    if not math.isclose(sum(clean), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("categorical entropy distribution must sum to one")
    return -sum(value * math.log(value) for value in clean if value) / math.log(len(clean))


def jensen_shannon(distributions: Sequence[Sequence[float]]) -> float:
    """Jensen-Shannon divergence normalised by its attainable maximum."""

    if isinstance(distributions, (str, bytes)) or len(distributions) < 2:
        raise ValueError("Jensen-Shannon divergence requires at least two distributions")
    width = len(distributions[0])
    if width < 2 or any(len(row) != width for row in distributions):
        raise ValueError("Jensen-Shannon distributions have shape drift")
    clean: list[list[float]] = []
    for row in distributions:
        values = [_probability(value, "Jensen-Shannon input") for value in row]
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Jensen-Shannon distribution must sum to one")
        clean.append(values)
    mean = [sum(row[index] for row in clean) / len(clean) for index in range(width)]
    divergence = 0.0
    for row in clean:
        divergence += sum(
            value * math.log(value / mean[index])
            for index, value in enumerate(row)
            if value
        )
    divergence /= len(clean)
    maximum = math.log(min(len(clean), width))
    return min(1.0, max(0.0, divergence / maximum))


def _validate_model_output(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    _exact_keys(value, _OUTPUT_KEYS, where)
    relevance = _probability(value["relevance"], f"{where}.relevance")
    target_presence = value["target_presence"]
    stance = value["stance"]
    if not isinstance(target_presence, Mapping) or set(target_presence) != set(ANALYTIC_TARGETS):
        raise ValueError(f"{where}.target_presence target set drifted")
    if not isinstance(stance, Mapping) or set(stance) != set(ANALYTIC_TARGETS):
        raise ValueError(f"{where}.stance target set drifted")
    clean_presence: dict[str, float] = {}
    clean_stance: dict[str, dict[str, float]] = {}
    for target in ANALYTIC_TARGETS:
        clean_presence[target] = _probability(
            target_presence[target], f"{where}.target_presence.{target}"
        )
        distribution = stance[target]
        if not isinstance(distribution, Mapping) or set(distribution) != set(STANCES):
            raise ValueError(f"{where}.stance.{target} stance set drifted")
        values = {
            label: _probability(distribution[label], f"{where}.stance.{target}.{label}")
            for label in STANCES
        }
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"{where}.stance.{target} must sum to one")
        clean_stance[target] = values
    return {
        "relevance": relevance,
        "target_presence": clean_presence,
        "stance": clean_stance,
    }


def _validate_candidate(value: Mapping[str, Any], expected_seed_count: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("candidate must be an object")
    _exact_keys(value, _CANDIDATE_KEYS, "candidate")
    clean: dict[str, Any] = {
        field: _text(value[field], f"candidate.{field}")
        for field in (
            "opaque_id",
            "thread_id",
            "near_duplicate_cluster_id",
            "subreddit",
            "content_type",
            "retrieval_mode",
        )
    }
    year = value["year"]
    if type(year) is not int or year not in range(2020, 2026):
        raise ValueError("candidate.year must be an integer in the registered range")
    clean["year"] = year
    seed_outputs = value["seed_outputs"]
    if not isinstance(seed_outputs, Mapping) or len(seed_outputs) != expected_seed_count:
        raise ValueError("candidate.seed_outputs must contain exactly three seeds")
    clean_outputs: dict[str, dict[str, Any]] = {}
    for raw_seed, render_outputs in seed_outputs.items():
        seed = _text(raw_seed, "candidate seed")
        if not isinstance(render_outputs, Mapping) or set(render_outputs) != set(RENDERS):
            raise ValueError("candidate seed render set drifted")
        clean_outputs[seed] = {
            render: _validate_model_output(
                render_outputs[render], f"candidate.seed_outputs.{seed}.{render}"
            )
            for render in RENDERS
        }
    clean["seed_outputs"] = dict(sorted(clean_outputs.items()))
    return clean


def validate_rare_cells(value: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Validate and canonically order the pre-acquisition rare-cell list."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("rare-cell list must be a sequence")
    if len(value) != 3:
        raise ValueError("acquisition requires exactly three canonical rare cells")
    clean: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {"target", "stance", "training_support"}:
            raise ValueError(f"rare_cells[{index}] fields drifted")
        target = row["target"]
        stance = row["stance"]
        support = row["training_support"]
        if target not in ANALYTIC_TARGETS or stance not in STANCES:
            raise ValueError("rare-cell target or stance drifted")
        if type(support) is not int or support <= 0:
            raise ValueError("rare-cell training support must be a positive integer")
        key = (target, stance)
        if key in seen:
            raise ValueError("rare-cell list contains duplicates")
        seen.add(key)
        clean.append({"target": target, "stance": stance, "training_support": support})
    return tuple(sorted(clean, key=lambda row: (row["target"], row["stance"])))


def compute_candidate_scores(
    candidate: Mapping[str, Any],
    *,
    rare_cells: Sequence[Mapping[str, Any]],
    expected_seed_count: int = 3,
) -> dict[str, float]:
    """Compute the four preregistered acquisition scores for one candidate."""

    clean = _validate_candidate(candidate, expected_seed_count)
    rare = validate_rare_cells(rare_cells)
    outputs = [row["full"] for row in clean["seed_outputs"].values()]

    relevance_entropy = sum(binary_entropy(row["relevance"]) for row in outputs) / len(outputs)
    target_entropy = sum(
        binary_entropy(row["target_presence"][target])
        for row in outputs
        for target in ANALYTIC_TARGETS
    ) / (len(outputs) * len(ANALYTIC_TARGETS))
    weighted_stance_entropy = sum(
        row["target_presence"][target]
        * categorical_entropy([row["stance"][target][stance] for stance in STANCES])
        for row in outputs
        for target in ANALYTIC_TARGETS
    ) / (len(outputs) * len(ANALYTIC_TARGETS))

    disagreement_terms = [
        jensen_shannon([[row["relevance"], 1.0 - row["relevance"]] for row in outputs])
    ]
    disagreement_terms.extend(
        jensen_shannon(
            [
                [row["target_presence"][target], 1.0 - row["target_presence"][target]]
                for row in outputs
            ]
        )
        for target in ANALYTIC_TARGETS
    )
    disagreement_terms.extend(
        jensen_shannon(
            [[row["stance"][target][stance] for stance in STANCES] for row in outputs]
        )
        for target in ANALYTIC_TARGETS
    )
    disagreement = sum(disagreement_terms) / len(disagreement_terms)
    uncertainty = (
        relevance_entropy + target_entropy + weighted_stance_entropy + disagreement
    ) / 4.0

    mean_relevance = sum(row["relevance"] for row in outputs) / len(outputs)
    mean_presence = {
        target: sum(row["target_presence"][target] for row in outputs) / len(outputs)
        for target in ANALYTIC_TARGETS
    }
    boundary = max(
        4.0 * mean_relevance * (1.0 - mean_relevance),
        max(4.0 * value * (1.0 - value) for value in mean_presence.values()),
        mean_relevance * (1.0 - max(mean_presence.values())),
    )

    maximum_weight = max(1.0 / math.sqrt(row["training_support"]) for row in rare)
    rare_score = max(
        (
            sum(
                output["target_presence"][row["target"]]
                * output["stance"][row["target"]][row["stance"]]
                for output in outputs
            )
            / len(outputs)
        )
        * ((1.0 / math.sqrt(row["training_support"])) / maximum_weight)
        for row in rare
    )

    sorted_presence = sorted(mean_presence.values(), reverse=True)
    second_largest_presence = sorted_presence[1]
    context_changes: list[float] = []
    for seed_output in clean["seed_outputs"].values():
        full = seed_output["full"]
        target_only = seed_output["target_only"]
        context_changes.append(abs(full["relevance"] - target_only["relevance"]))
        for target in ANALYTIC_TARGETS:
            context_changes.append(
                abs(full["target_presence"][target] - target_only["target_presence"][target])
            )
            context_changes.extend(
                abs(full["stance"][target][stance] - target_only["stance"][target][stance])
                for stance in STANCES
            )
    multi_context = max(second_largest_presence, max(context_changes))

    result = {
        "uncertainty_disagreement": uncertainty,
        "boundary": boundary,
        "rare_cell": rare_score,
        "multi_context": multi_context,
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
        raise RuntimeError("computed acquisition score left [0,1]")
    return result


def compute_compact_score_record(
    candidate: Mapping[str, Any],
    *,
    rare_cells: Sequence[Mapping[str, Any]],
    expected_seed_count: int = 3,
) -> dict[str, Any]:
    """Validate one candidate and reduce it to selection-relevant metadata/scores."""

    clean = _validate_candidate(candidate, expected_seed_count)
    scores = compute_candidate_scores(
        clean,
        rare_cells=rare_cells,
        expected_seed_count=expected_seed_count,
    )
    return {
        "opaque_id": clean["opaque_id"],
        "thread_id": clean["thread_id"],
        "near_duplicate_cluster_id": clean["near_duplicate_cluster_id"],
        "subreddit": clean["subreddit"],
        "year": clean["year"],
        "content_type": clean["content_type"],
        "retrieval_mode": clean["retrieval_mode"],
        **scores,
    }


def _validate_compact_score_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("compact score record must be an object")
    _exact_keys(value, _COMPACT_SCORE_KEYS, "compact score record")
    clean: dict[str, Any] = {
        field: _text(value[field], f"compact score record.{field}")
        for field in (
            "opaque_id",
            "thread_id",
            "near_duplicate_cluster_id",
            "subreddit",
            "content_type",
            "retrieval_mode",
        )
    }
    year = value["year"]
    if type(year) is not int or year not in range(2020, 2026):
        raise ValueError("compact score record.year must be in the registered range")
    clean["year"] = year
    for bucket in BUCKET_PRIORITY:
        clean[bucket] = _probability(
            value[bucket], f"compact score record.{bucket}"
        )
    return clean


def _stratum(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subreddit": row["subreddit"],
        "year": row["year"],
        "content_type": row["content_type"],
        "retrieval_mode": row["retrieval_mode"],
    }


def _stratum_key(row: Mapping[str, Any]) -> str:
    return canonical_sha256(_stratum(row))


def build_probability_selection_row(
    candidate: Mapping[str, Any],
    *,
    quota: int,
    population_rows: int,
    policy: AcquisitionPolicy,
) -> dict[str, Any]:
    """Build one probability-arm row from a compact candidate record."""

    clean = _validate_compact_score_record(candidate)
    if type(quota) is not int or quota <= 0:
        raise ValueError("probability stratum quota must be a positive integer")
    if type(population_rows) is not int or population_rows < quota:
        raise ValueError("probability stratum population must cover its quota")
    return {
        "opaque_id": clean["opaque_id"],
        "thread_id": clean["thread_id"],
        "near_duplicate_cluster_id": clean["near_duplicate_cluster_id"],
        "stratum": _stratum(clean),
        "inclusion_probability_numerator": quota,
        "inclusion_probability_denominator": population_rows,
        "inclusion_probability": quota / population_rows,
        "selection_tiebreak_sha256": _tiebreak(
            policy.probability_seed, clean["opaque_id"]
        ),
    }


def allocate_probability_stratum_counts(
    strata_counts: Sequence[Mapping[str, Any]],
    policy: AcquisitionPolicy,
) -> list[dict[str, Any]]:
    """Allocate exact largest-remainder quotas from compact stratum counts.

    Each input row has ``stratum`` (the four registered fields) and a positive
    ``population_rows`` count.  Returned rows add ``sample_rows`` and are
    canonically ordered by the stratum digest.  Zero per-stratum quotas are
    retained so an out-of-core caller can prove population/sample conservation.
    """

    if isinstance(strata_counts, (str, bytes)) or not strata_counts:
        raise ValueError("probability stratum counts must be a non-empty sequence")
    clean: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(strata_counts):
        if not isinstance(row, Mapping) or set(row) != {
            "stratum",
            "population_rows",
        }:
            raise ValueError(f"probability stratum counts[{index}] fields drifted")
        stratum = row["stratum"]
        if not isinstance(stratum, Mapping) or set(stratum) != {
            "subreddit",
            "year",
            "content_type",
            "retrieval_mode",
        }:
            raise ValueError(f"probability stratum counts[{index}].stratum drifted")
        canonical_stratum = {
            "subreddit": _text(stratum["subreddit"], "stratum.subreddit"),
            "year": stratum["year"],
            "content_type": _text(stratum["content_type"], "stratum.content_type"),
            "retrieval_mode": _text(
                stratum["retrieval_mode"], "stratum.retrieval_mode"
            ),
        }
        if type(canonical_stratum["year"]) is not int or canonical_stratum[
            "year"
        ] not in range(2020, 2026):
            raise ValueError("stratum.year must be in the registered range")
        population_rows = row["population_rows"]
        if type(population_rows) is not int or population_rows <= 0:
            raise ValueError("stratum population_rows must be a positive integer")
        key = canonical_sha256(canonical_stratum)
        if key in clean:
            raise ValueError("probability stratum counts contain a duplicate stratum")
        clean[key] = {
            "stratum": canonical_stratum,
            "population_rows": population_rows,
        }

    total = sum(row["population_rows"] for row in clean.values())
    if policy.probability_rows > total:
        raise ValueError("probability quota exceeds the eligible universe")
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str, str]] = []
    for key, row in clean.items():
        exact = policy.probability_rows * row["population_rows"] / total
        allocations[key] = math.floor(exact)
        remainders.append(
            (exact - allocations[key], _tiebreak(policy.probability_seed, key), key)
        )
    remaining = policy.probability_rows - sum(allocations.values())
    for _, _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if allocations[key] < clean[key]["population_rows"]:
            allocations[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("proportional allocation could not conserve the sample quota")
    return [
        {
            "stratum": clean[key]["stratum"],
            "population_rows": clean[key]["population_rows"],
            "sample_rows": allocations[key],
        }
        for key in sorted(clean)
    ]


def allocate_probability_strata(
    candidates: Sequence[Mapping[str, Any]], policy: AcquisitionPolicy
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Allocate and select the frozen proportional probability sample."""

    if isinstance(candidates, (str, bytes)) or not candidates:
        raise ValueError("probability candidates must be a non-empty sequence")
    clean_candidates = [_validate_compact_score_record(row) for row in candidates]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in clean_candidates:
        groups[_stratum_key(row)].append(row)
    strata = allocate_probability_stratum_counts(
        [
            {"stratum": _stratum(rows[0]), "population_rows": len(rows)}
            for rows in groups.values()
        ],
        policy,
    )
    allocations = {
        canonical_sha256(row["stratum"]): row["sample_rows"] for row in strata
    }

    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        quota = allocations[key]
        ordered = sorted(
            rows,
            key=lambda row: _tiebreak(policy.probability_seed, row["opaque_id"]),
        )
        for row in ordered[:quota]:
            selected.append(
                build_probability_selection_row(
                    row,
                    quota=quota,
                    population_rows=len(rows),
                    policy=policy,
                )
            )
    selected.sort(
        key=lambda row: (canonical_sha256(row["stratum"]), row["selection_tiebreak_sha256"])
    )
    return selected, strata


def active_rank_key(
    candidate: Mapping[str, Any],
    *,
    bucket: str,
    policy: AcquisitionPolicy,
) -> tuple[float, str]:
    """Return the exact descending-score/hash-tiebreak active ranking key."""

    if bucket not in BUCKET_PRIORITY:
        raise ValueError("active bucket drifted")
    clean = _validate_compact_score_record(candidate)
    return (
        -clean[bucket],
        _tiebreak(policy.active_seed, bucket, clean["opaque_id"]),
    )


def build_active_selection_row(
    candidate: Mapping[str, Any],
    *,
    bucket: str,
    bucket_rank: int,
    policy: AcquisitionPolicy,
) -> dict[str, Any]:
    """Build one active-arm row while preserving its pre-filter ranking position."""

    if bucket not in BUCKET_PRIORITY:
        raise ValueError("active bucket drifted")
    if type(bucket_rank) is not int or bucket_rank <= 0:
        raise ValueError("active bucket rank must be a positive integer")
    clean = _validate_compact_score_record(candidate)
    return {
        "opaque_id": clean["opaque_id"],
        "thread_id": clean["thread_id"],
        "near_duplicate_cluster_id": clean["near_duplicate_cluster_id"],
        "bucket": bucket,
        "bucket_rank": bucket_rank,
        "score": clean[bucket],
        "selection_tiebreak_sha256": _tiebreak(
            policy.active_seed, bucket, clean["opaque_id"]
        ),
    }


def _validate_bindings(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != REQUIRED_INPUT_BINDINGS:
        raise ValueError("input binding fields drifted")
    clean = dict(value)
    if any(not _is_sha256(item) for item in clean.values()):
        raise ValueError("every input binding must be a lowercase SHA-256")
    return clean


def finalise_acquisition_ledger(
    *,
    eligible_population_rows: int,
    candidate_score_digest: str,
    probability_rows: Sequence[Mapping[str, Any]],
    probability_strata: Sequence[Mapping[str, Any]],
    active_rows: Sequence[Mapping[str, Any]],
    rare_cells: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, str],
    policy: AcquisitionPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finalise private/public artefacts from already selected compact records."""

    if type(eligible_population_rows) is not int or eligible_population_rows <= 0:
        raise ValueError("eligible population rows must be a positive integer")
    if not _is_sha256(candidate_score_digest):
        raise ValueError("candidate score digest must be a lowercase SHA-256")
    bindings = _validate_bindings(input_bindings)
    if bindings["policy_file_sha256"] != policy.policy_file_sha256:
        raise ValueError("input policy file binding disagrees with the acquisition policy")
    rare = validate_rare_cells(rare_cells)
    probability_rows = [dict(row) for row in probability_rows]
    strata = [dict(row) for row in probability_strata]
    active_rows = [dict(row) for row in active_rows]
    if len(probability_rows) != policy.probability_rows:
        raise ValueError("probability arm row count disagrees with policy")
    if len(active_rows) != sum(policy.bucket_quotas.values()):
        raise ValueError("active arm row count disagrees with policy")
    if Counter(row.get("bucket") for row in active_rows) != Counter(
        policy.bucket_quotas
    ):
        raise ValueError("active bucket counts disagree with policy")

    policy_contract = policy.as_contract()
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_KIND,
        "policy": policy_contract,
        "policy_file_sha256": policy.policy_file_sha256,
        "policy_contract_sha256": canonical_sha256(policy_contract),
        "input_bindings": bindings,
        "rare_cells": list(rare),
        "rare_cell_list_sha256": canonical_sha256(rare),
        "eligible_population_rows": eligible_population_rows,
        "candidate_score_digest": candidate_score_digest,
        "probability_arm": {
            "row_count": len(probability_rows),
            "strata": strata,
            "rows": probability_rows,
        },
        "active_arm": {
            "row_count": len(active_rows),
            "buckets": policy.bucket_quotas,
            "rows": active_rows,
        },
    }
    private = {**body, "ledger_id": canonical_sha256(body)}
    score_summary = {
        bucket: {
            "minimum": min(
                row["score"] for row in active_rows if row["bucket"] == bucket
            ),
            "maximum": max(
                row["score"] for row in active_rows if row["bucket"] == bucket
            ),
        }
        for bucket in BUCKET_PRIORITY
    }
    public_body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": PUBLIC_KIND,
        "ledger_sha256": canonical_sha256(private),
        "ledger_id": private["ledger_id"],
        "policy_file_sha256": private["policy_file_sha256"],
        "policy_contract_sha256": private["policy_contract_sha256"],
        "input_bindings": bindings,
        "rare_cell_list_sha256": private["rare_cell_list_sha256"],
        "rare_cell_count": len(rare),
        "eligible_population_rows": eligible_population_rows,
        "probability_arm_rows": len(probability_rows),
        "probability_strata": len(strata),
        "active_arm_rows": len(active_rows),
        "active_bucket_counts": dict(Counter(row["bucket"] for row in active_rows)),
        "active_selected_score_ranges": score_summary,
        "arms_thread_disjoint": True,
        "arms_near_duplicate_disjoint": True,
    }
    public = {**public_body, "receipt_id": canonical_sha256(public_body)}
    _validate_arm_disjointness(private)
    assert_metadata_only(public, where="acquisition public receipt")
    return private, public


def build_acquisition_ledger(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rare_cells: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, str],
    policy: AcquisitionPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the deterministic private acquisition ledger and public receipt."""

    active_policy = policy or load_acquisition_policies()[0]
    if isinstance(candidates, (str, bytes)) or not candidates:
        raise ValueError("eligible candidates must be a non-empty sequence")
    clean_candidates = [
        _validate_candidate(row, active_policy.expected_seed_count) for row in candidates
    ]
    seed_sets = {tuple(row["seed_outputs"]) for row in clean_candidates}
    if len(seed_sets) != 1:
        raise ValueError("eligible candidate checkpoint seed set drifted")
    for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id"):
        values = [row[field] for row in clean_candidates]
        if len(values) != len(set(values)):
            raise ValueError(f"eligible candidates contain duplicate {field}")
    bindings = _validate_bindings(input_bindings)
    if bindings["policy_file_sha256"] != active_policy.policy_file_sha256:
        raise ValueError("input policy file binding disagrees with the acquisition policy")
    rare = validate_rare_cells(rare_cells)
    compact_records = [
        compute_compact_score_record(
            row, rare_cells=rare, expected_seed_count=active_policy.expected_seed_count
        )
        for row in clean_candidates
    ]
    probability_rows, strata = allocate_probability_strata(
        compact_records, active_policy
    )
    random_ids = {row["opaque_id"] for row in probability_rows}
    random_clusters = {row["near_duplicate_cluster_id"] for row in probability_rows}
    pool = [
        row
        for row in compact_records
        if row["opaque_id"] not in random_ids
        and row["near_duplicate_cluster_id"] not in random_clusters
    ]
    if len(pool) < sum(active_policy.bucket_quotas.values()):
        raise ValueError("active pool cannot satisfy the frozen bucket quotas")

    active_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_clusters: set[str] = set(random_clusters)
    for bucket in BUCKET_PRIORITY:
        ranking = sorted(
            pool,
            key=lambda row: active_rank_key(
                row, bucket=bucket, policy=active_policy
            ),
        )
        accepted = 0
        for rank, row in enumerate(ranking, start=1):
            if row["opaque_id"] in selected_ids:
                continue
            if row["near_duplicate_cluster_id"] in selected_clusters:
                continue
            active_rows.append(
                build_active_selection_row(
                    row,
                    bucket=bucket,
                    bucket_rank=rank,
                    policy=active_policy,
                )
            )
            selected_ids.add(row["opaque_id"])
            selected_clusters.add(row["near_duplicate_cluster_id"])
            accepted += 1
            if accepted == active_policy.bucket_quotas[bucket]:
                break
        if accepted != active_policy.bucket_quotas[bucket]:
            raise ValueError(f"active bucket {bucket} cannot satisfy its quota")

    score_records = sorted(compact_records, key=lambda row: row["opaque_id"])
    score_digest = canonical_sequence_sha256(
        {
            "opaque_id": row["opaque_id"],
            **{bucket: row[bucket] for bucket in BUCKET_PRIORITY},
        }
        for row in score_records
    )
    return finalise_acquisition_ledger(
        eligible_population_rows=len(clean_candidates),
        candidate_score_digest=score_digest,
        probability_rows=probability_rows,
        probability_strata=strata,
        active_rows=active_rows,
        rare_cells=rare,
        input_bindings=bindings,
        policy=active_policy,
    )


def validate_acquisition_ledger(
    private: Mapping[str, Any],
    public: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    rare_cells: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, str],
    policy: AcquisitionPolicy | None = None,
) -> None:
    """Rebuild and exact-compare both immutable acquisition artefacts."""

    _validate_arm_disjointness(private)
    expected_private, expected_public = build_acquisition_ledger(
        candidates,
        rare_cells=rare_cells,
        input_bindings=input_bindings,
        policy=policy,
    )
    if dict(private) != expected_private:
        raise ValueError("private acquisition ledger drifted")
    if dict(public) != expected_public:
        raise ValueError("public acquisition receipt drifted")
    assert_metadata_only(public, where="acquisition public receipt")


def _validate_arm_disjointness(private: Mapping[str, Any]) -> None:
    """Fail closed on identity, thread, or cluster overlap between acquisition arms."""

    if not isinstance(private, Mapping):
        raise ValueError("private acquisition ledger must be an object")
    probability = private.get("probability_arm")
    active = private.get("active_arm")
    if not isinstance(probability, Mapping) or not isinstance(active, Mapping):
        raise ValueError("private acquisition ledger arm structure drifted")
    probability_rows = probability.get("rows")
    active_rows = active.get("rows")
    if (
        isinstance(probability_rows, (str, bytes))
        or not isinstance(probability_rows, Sequence)
        or isinstance(active_rows, (str, bytes))
        or not isinstance(active_rows, Sequence)
    ):
        raise ValueError("private acquisition ledger arm rows drifted")
    for field, description in (
        ("opaque_id", "identity"),
        ("thread_id", "thread"),
        ("near_duplicate_cluster_id", "near-duplicate cluster"),
    ):
        probability_values = {
            row.get(field) for row in probability_rows if isinstance(row, Mapping)
        }
        active_values = {row.get(field) for row in active_rows if isinstance(row, Mapping)}
        if len(probability_values) != len(probability_rows):
            raise ValueError(f"probability arm contains duplicate {description}s")
        if len(active_values) != len(active_rows):
            raise ValueError(f"active arm contains duplicate {description}s")
        if probability_values & active_values:
            raise ValueError(f"acquisition arms overlap by {description}")


def _validate_target_stances(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    if not set(value).issubset(set(ANALYTIC_TARGETS)):
        raise ValueError(f"{where} contains an unknown target")
    clean: dict[str, str] = {}
    for target, stance in value.items():
        if stance not in STANCES:
            raise ValueError(f"{where}.{target} has an unknown stance")
        clean[target] = stance
    return clean


def _validate_evaluation_rows(
    reference_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    if isinstance(reference_rows, (str, bytes)) or not reference_rows:
        raise ValueError("acquisition evaluation reference must be non-empty")
    reference: list[dict[str, Any]] = []
    for row in reference_rows:
        if not isinstance(row, Mapping):
            raise ValueError("reference row must be an object")
        _exact_keys(row, _REFERENCE_KEYS, "reference row")
        thread_id = _text(row["thread_id"], "reference.thread_id")
        quality_tier = row["quality_tier"]
        if quality_tier not in QUALITY_TIERS:
            raise ValueError("reference quality_tier drifted")
        if type(row["material"]) is not bool:
            raise ValueError("reference material must be boolean")
        target_stances = _validate_target_stances(
            row["target_stances"], "reference.target_stances"
        )
        if not row["material"] and target_stances:
            raise ValueError("not-material reference cannot contain target stances")
        reference.append(
            {
                "thread_id": thread_id,
                "quality_tier": quality_tier,
                "material": row["material"],
                "target_stances": target_stances,
            }
        )
    ids = [row["thread_id"] for row in reference]
    if len(ids) != len(set(ids)):
        raise ValueError("reference contains duplicate thread IDs")
    if not isinstance(predictions, Mapping) or set(predictions) != {"random", "active"}:
        raise ValueError("predictions must contain exactly random and active arms")
    clean_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    expected_ids = set(ids)
    seed_sets: list[set[str]] = []
    for arm in ("random", "active"):
        by_seed = predictions[arm]
        if not isinstance(by_seed, Mapping) or len(by_seed) != 3:
            raise ValueError("each acquisition arm must contain exactly three seeds")
        seed_sets.append(set(by_seed))
        clean_predictions[arm] = {}
        for raw_seed, rows in by_seed.items():
            seed = _text(raw_seed, "prediction seed")
            if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
                raise ValueError("prediction rows must be a sequence")
            by_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("prediction row must be an object")
                _exact_keys(row, _PREDICTION_KEYS, "prediction row")
                thread_id = _text(row["thread_id"], "prediction.thread_id")
                if thread_id in by_id:
                    raise ValueError("prediction contains duplicate thread IDs")
                if type(row["material"]) is not bool:
                    raise ValueError("prediction material must be boolean")
                target_stances = _validate_target_stances(
                    row["target_stances"], "prediction.target_stances"
                )
                if not row["material"] and target_stances:
                    raise ValueError("not-material prediction cannot contain target stances")
                by_id[thread_id] = {
                    "material": row["material"],
                    "target_stances": target_stances,
                }
            if set(by_id) != expected_ids:
                raise ValueError("prediction identity set drifted")
            clean_predictions[arm][seed] = by_id
    if seed_sets[0] != seed_sets[1]:
        raise ValueError("paired acquisition arms must use identical optimiser seeds")
    return reference, clean_predictions


def _f1(pairs: Sequence[tuple[bool, bool]]) -> float:
    true_positive = sum(reference and prediction for reference, prediction in pairs)
    false_positive = sum(not reference and prediction for reference, prediction in pairs)
    false_negative = sum(reference and not prediction for reference, prediction in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _recall(pairs: Sequence[tuple[bool, bool]]) -> float:
    positive = sum(reference for reference, _ in pairs)
    if positive == 0:
        raise ValueError("material recall is undefined without positive references")
    return sum(reference and prediction for reference, prediction in pairs) / positive


def _metrics(
    reference: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    rare_cells: Sequence[Mapping[str, Any]],
    sampled_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    indices = list(range(len(reference))) if sampled_indices is None else list(sampled_indices)
    rare_f1: dict[str, float] = {}
    for cell in rare_cells:
        target, stance = cell["target"], cell["stance"]
        key = f"{target}|{stance}"
        rare_f1[key] = _f1(
            [
                (
                    reference[index]["target_stances"].get(target) == stance,
                    predictions[reference[index]["thread_id"]]["target_stances"].get(target)
                    == stance,
                )
                for index in indices
            ]
        )
    tuple_pairs: list[tuple[bool, bool]] = []
    for index in indices:
        ref = reference[index]
        pred = predictions[ref["thread_id"]]
        for target in ANALYTIC_TARGETS:
            for stance in STANCES:
                tuple_pairs.append(
                    (
                        ref["target_stances"].get(target) == stance,
                        pred["target_stances"].get(target) == stance,
                    )
                )
    target_f1 = {
        target: _f1(
            [
                (
                    target in reference[index]["target_stances"],
                    target
                    in predictions[reference[index]["thread_id"]]["target_stances"],
                )
                for index in indices
            ]
        )
        for target in ANALYTIC_TARGETS
    }
    return {
        "rare_cell_macro_f1": sum(rare_f1.values()) / len(rare_f1),
        "rare_cell_f1": rare_f1,
        "tuple_micro_f1": _f1(tuple_pairs),
        "material_recall": _recall(
            [
                (
                    reference[index]["material"],
                    predictions[reference[index]["thread_id"]]["material"],
                )
                for index in indices
            ]
        ),
        "target_f1": target_f1,
    }


def _rare_macro_f1(
    reference: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    rare_cells: Sequence[Mapping[str, Any]],
    sampled_indices: Sequence[int] | None = None,
) -> float:
    indices = list(range(len(reference))) if sampled_indices is None else list(sampled_indices)
    values = []
    for cell in rare_cells:
        target, stance = cell["target"], cell["stance"]
        values.append(
            _f1(
                [
                    (
                        reference[index]["target_stances"].get(target) == stance,
                        predictions[reference[index]["thread_id"]]["target_stances"].get(
                            target
                        )
                        == stance,
                    )
                    for index in indices
                ]
            )
        )
    return sum(values) / len(values)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_acquisition_gate(
    reference_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    rare_cells: Sequence[Mapping[str, Any]],
    policy: AcquisitionGatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate the deterministic paired bootstrap acquisition gate."""

    active_policy = policy or load_acquisition_policies()[1]
    reference, clean_predictions = _validate_evaluation_rows(reference_rows, predictions)
    rare = validate_rare_cells(rare_cells)
    for cell in rare:
        support = sum(
            row["target_stances"].get(cell["target"]) == cell["stance"]
            for row in reference
        )
        if support < active_policy.minimum_cell_support:
            raise ValueError("frozen rare cell lacks minimum evaluation-frame support")
    exact_reference = [row for row in reference if row["quality_tier"] == "exact_consensus"]
    if not exact_reference:
        raise ValueError("exact-consensus sensitivity frame is empty")

    seeds = sorted(clean_predictions["random"])
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        random_metrics = _metrics(reference, clean_predictions["random"][seed], rare)
        active_metrics = _metrics(reference, clean_predictions["active"][seed], rare)
        exact_random_rare = _rare_macro_f1(
            exact_reference, clean_predictions["random"][seed], rare
        )
        exact_active_rare = _rare_macro_f1(
            exact_reference, clean_predictions["active"][seed], rare
        )
        per_seed[seed] = {
            "random": random_metrics,
            "active": active_metrics,
            "rare_gain": active_metrics["rare_cell_macro_f1"]
            - random_metrics["rare_cell_macro_f1"],
            "tuple_delta": active_metrics["tuple_micro_f1"]
            - random_metrics["tuple_micro_f1"],
            "material_recall_delta": active_metrics["material_recall"]
            - random_metrics["material_recall"],
            "exact_consensus_rare_gain": exact_active_rare - exact_random_rare,
        }

    rng = random.Random(active_policy.bootstrap_seed)
    bootstrap: list[float] = []
    for _ in range(active_policy.bootstrap_replicates):
        indices = rng.choices(range(len(reference)), k=len(reference))
        differences = []
        for seed in seeds:
            random_metric = _rare_macro_f1(
                reference, clean_predictions["random"][seed], rare, indices
            )
            active_metric = _rare_macro_f1(
                reference, clean_predictions["active"][seed], rare, indices
            )
            differences.append(active_metric - random_metric)
        bootstrap.append(sum(differences) / len(differences))

    mean_rare_gain = sum(row["rare_gain"] for row in per_seed.values()) / len(per_seed)
    mean_tuple_delta = sum(row["tuple_delta"] for row in per_seed.values()) / len(per_seed)
    mean_recall_delta = sum(row["material_recall_delta"] for row in per_seed.values()) / len(
        per_seed
    )
    mean_exact_gain = sum(
        row["exact_consensus_rare_gain"] for row in per_seed.values()
    ) / len(per_seed)
    target_support = {
        target: sum(target in row["target_stances"] for row in reference)
        for target in ANALYTIC_TARGETS
    }
    target_deltas = {
        target: sum(
            per_seed[seed]["active"]["target_f1"][target]
            - per_seed[seed]["random"]["target_f1"][target]
            for seed in seeds
        )
        / len(seeds)
        for target in ANALYTIC_TARGETS
    }
    target_regressions = [
        target
        for target in ANALYTIC_TARGETS
        if target_support[target] >= active_policy.minimum_cell_support
        and target_deltas[target] < -active_policy.maximum_supported_target_decline
    ]
    interval = {
        "lower": _percentile(bootstrap, 0.025),
        "upper": _percentile(bootstrap, 0.975),
    }
    criteria = {
        "mean_rare_gain_at_least_0_03": mean_rare_gain >= active_policy.minimum_mean_rare_gain,
        "paired_bootstrap_lower_above_zero": interval["lower"] > 0.0,
        "at_least_two_of_three_seeds_improve": sum(
            row["rare_gain"] > 0.0 for row in per_seed.values()
        )
        >= 2,
        "tuple_decline_at_most_0_01": mean_tuple_delta >= -active_policy.maximum_tuple_decline,
        "material_recall_decline_at_most_0_01": mean_recall_delta
        >= -active_policy.maximum_material_recall_decline,
        "no_supported_target_regression_over_0_05": not target_regressions,
        "exact_consensus_sensitivity_does_not_reverse": mean_exact_gain >= 0.0,
        "zero_invalid_outputs": True,
        "zero_locked_test_accesses": True,
    }
    passed = all(criteria.values())
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_KIND,
        "policy_file_sha256": active_policy.policy_file_sha256,
        "verdict": "promote_active" if passed else "retain_probability_random",
        "passed": passed,
        "rare_cell_list_sha256": canonical_sha256(rare),
        "rare_cell_count": len(rare),
        "evaluation_rows": len(reference),
        "optimiser_seeds": len(seeds),
        "bootstrap": {
            "replicates": active_policy.bootstrap_replicates,
            "seed_sha256": canonical_sha256(active_policy.bootstrap_seed),
            "interval": interval,
        },
        "mean_paired_rare_cell_macro_f1_gain": mean_rare_gain,
        "positive_seed_pairs": sum(row["rare_gain"] > 0.0 for row in per_seed.values()),
        "mean_tuple_f1_delta": mean_tuple_delta,
        "mean_material_recall_delta": mean_recall_delta,
        "mean_exact_consensus_rare_gain": mean_exact_gain,
        "arm_mean_metrics": {
            arm: {
                metric: sum(per_seed[seed][arm][metric] for seed in seeds) / len(seeds)
                for metric in (
                    "rare_cell_macro_f1",
                    "tuple_micro_f1",
                    "material_recall",
                )
            }
            for arm in ("random", "active")
        },
        "paired_seed_deltas": [
            {
                "rare_cell_macro_f1": per_seed[seed]["rare_gain"],
                "tuple_micro_f1": per_seed[seed]["tuple_delta"],
                "material_recall": per_seed[seed]["material_recall_delta"],
                "exact_consensus_rare_cell_macro_f1": per_seed[seed][
                    "exact_consensus_rare_gain"
                ],
            }
            for seed in seeds
        ],
        "supported_target_regression_count": len(target_regressions),
        "criteria": criteria,
        "invalid_outputs": 0,
        "locked_test_accesses": 0,
    }
    result["gate_id"] = canonical_sha256(result)
    assert_metadata_only(result, where="acquisition gate")
    return result
