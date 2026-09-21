"""Exact paired B4 training contract for the acquisition-policy experiment.

This module deliberately reuses the frozen factorised-v2 model, tokenizer and
optimisation recipe while giving the acquisition comparison its own immutable
manifest and publication namespace.  It has no B2, calibration, abstention,
locked-test or corpus-inference path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from reddit_china_stance import sol_teacher_acquisition_v2 as acquisition_teacher
from reddit_china_stance.modernbert_acquisition import (
    AcquisitionGatePolicy,
    canonical_sha256,
    evaluate_acquisition_gate,
    load_acquisition_policies,
    validate_rare_cells,
)
from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    STANCE_CLASSES_B4,
    TARGET_CLASSES,
    build_factorised_record,
)
from reddit_china_stance.modernbert_factorised_experiment import (
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    artifact_descriptor,
    frozen_component_config,
)
from reddit_china_stance.modernbert_factorised_training import (
    FactorisedDynamicPaddingCollator,
    build_optimisation_config,
    collect_development_logits,
    create_adamw,
    create_component_model,
    load_pinned_tokenizer,
    tokenise_factorised_record,
    train_epoch,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_ontology_v2 import (
    load_v2_label_validator,
    validate_v2_label,
)

SCHEMA_VERSION = "1.0.0"
NAMESPACE = "student-modernbert-acquisition-training-v1"
EXPERIMENT_KIND = "modernbert-acquisition-training-experiment-v1"
RUN_MANIFEST_KIND = "modernbert-acquisition-training-run-manifest-v1"
TRIAL_SPEC_KIND = "modernbert-acquisition-training-trial-spec-v1"
TRIAL_RECEIPT_KIND = "modernbert-acquisition-training-trial-receipt-v1"
PRIVATE_PREDICTIONS_KIND = "modernbert-acquisition-private-evaluation-logits-v1"
TRIAL_METRICS_KIND = "modernbert-acquisition-training-metrics-v1"
ATTEMPT_KIND = "modernbert-acquisition-training-attempt-v1"
PREPARATION_KIND = "modernbert-acquisition-training-preparation-v1"
ARMS = ("random", "active")
COMPONENTS = ("relevance", "target_stance_b4")
SEEDS = (47, 61, 89)
EXPECTED_TRIALS = 12
EXPECTED_RARE_CELLS = 3
BASE_TRAINING_ROWS = 8_147
EVALUATION_ROWS = 600
QUERIES_PER_ARM = 1_000
GPU_TYPE = "L4"
MAX_CONCURRENT_TRIALS = 10
HARD_COST_CAP_USD = Decimal("200")
MAX_LENGTH = 768
TRIAL_MAX_GPU_SECONDS = 7_200

Arm = Literal["random", "active"]
Component = Literal["relevance", "target_stance_b4"]

_FRAME_REQUIRED = {
    "item_id",
    "frame",
    "thread_id",
    "target_text",
    "parent_context",
    "submission_context",
    "label_json",
    "selection_component",
    "selection_stratum",
    "inclusion_probability_numerator",
    "inclusion_probability_denominator",
    "inclusion_probability",
    "probability_scope",
}
_ACQUISITION_SOURCE_REQUIRED = set(acquisition_teacher.SOURCE_COLUMNS)
_ACQUISITION_LABEL_REQUIRED = {
    "source_sample_id",
    "thread_id",
    "acquisition_arm",
    "arm_order",
    "packet_order",
    "codability",
    "relevance",
    "label_json",
    "quality_tier",
    "primary_training_eligible",
}


class AcquisitionTrainingContractError(RuntimeError):
    """Raised when a frozen acquisition-training binding drifts."""


def _validate_exact_rare_cells(
    value: Sequence[Mapping[str, Any]], *, where: str
) -> list[dict[str, Any]]:
    """Translate lower-level rare-cell failures into this contract's public error."""

    try:
        rare = list(validate_rare_cells(value))
    except (TypeError, ValueError) as exc:
        raise AcquisitionTrainingContractError(
            f"{where} rare-cell contract drifted"
        ) from exc
    if len(rare) != EXPECTED_RARE_CELLS:
        raise AcquisitionTrainingContractError(
            f"{where} rare-cell contract drifted"
        )
    return rare


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _safe_relative(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{where} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"{where} must be a safe relative path")
    return value


def validate_factorised_parent_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the frozen B4 decision and calibration membership used by this run."""

    expected = {
        "experiment_run_id",
        "phase_run_id",
        "run_manifest",
        "representation_gate",
        "gate_receipt_id",
        "split_manifest",
        "selected_representation",
        "verdict",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("factorised parent provenance schema drifted")
    for name in ("experiment_run_id", "phase_run_id", "gate_receipt_id"):
        _sha(value[name], where=f"factorised parent {name}")
    for name in ("run_manifest", "representation_gate", "split_manifest"):
        artifact_descriptor(value[name], where=f"factorised parent {name}")
    if (
        value["selected_representation"] != "B4"
        or value["verdict"] not in {"retain_b4", "scrap_b2_keep_b4"}
    ):
        raise AcquisitionTrainingContractError(
            "factorised parent did not retain the registered B4 representation"
        )
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _arm(value: Any) -> Arm:
    if value == "probability_random":
        return "random"
    if value not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return value


def _component(value: Any) -> Component:
    if value not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    return value


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise AcquisitionTrainingContractError(
                f"existing immutable JSON differs: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise AcquisitionTrainingContractError(f"stale incomplete JSON exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _descriptor(path: Path, *, root: Path, **extra: Any) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _validate_descriptor(root: Path, value: Mapping[str, Any], *, where: str) -> Path:
    clean = artifact_descriptor(value, where=where)
    path = root / clean["relative_path"]
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or file_sha256(path) != clean["sha256"]
    ):
        raise AcquisitionTrainingContractError(f"{where} is missing or corrupt")
    return path


def loss_contribution_counts(
    rows: Sequence[Mapping[str, Any]], *, label_validator: Any | None = None
) -> dict[str, Any]:
    """Count exact examples contributing to every proper-loss term."""

    relevance = 0
    target_presence = 0
    stance = {target: 0 for target in ANALYTIC_TARGET_CLASSES}
    for index, row in enumerate(rows):
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"training row {index} has an invalid item_id")
        try:
            label = json.loads(row["label_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"training row {index} contains invalid label JSON") from exc
        clean = validate_v2_label(label, validator=label_validator)
        record = build_factorised_record(
            item_id=item_id,
            row=row,
            label=clean,
            separator="[SEP]",
        )
        feature = record.encoding.as_feature()
        relevance += int(bool(feature["codability_mask"]))
        material = bool(feature["target_presence_mask"][0])
        target_presence += int(material)
        for target, known in zip(
            ANALYTIC_TARGET_CLASSES, feature["stance_mask"], strict=True
        ):
            stance[target] += int(bool(known))
    return {
        "training_rows": len(rows),
        "relevance_bce_rows": relevance,
        "target_presence_bce_rows": target_presence,
        "stance_cross_entropy_rows_by_target": stance,
        "stance_cross_entropy_total": sum(stance.values()),
    }


def _validate_loss_contribution_counts(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "training_rows",
        "relevance_bce_rows",
        "target_presence_bce_rows",
        "stance_cross_entropy_rows_by_target",
        "stance_cross_entropy_total",
    }
    if set(value) != expected:
        raise ValueError("loss-contribution schema drifted")
    counts = {
        key: _positive_int(value[key], where=f"loss contributions.{key}")
        for key in ("training_rows", "relevance_bce_rows", "target_presence_bce_rows")
    }
    stances = value["stance_cross_entropy_rows_by_target"]
    if not isinstance(stances, Mapping) or set(stances) != set(ANALYTIC_TARGET_CLASSES):
        raise ValueError("stance contribution target inventory drifted")
    stance_counts: dict[str, int] = {}
    for target in ANALYTIC_TARGET_CLASSES:
        count = stances[target]
        if type(count) is not int or count < 0:
            raise ValueError("stance contribution count must be non-negative")
        stance_counts[target] = count
    total = value["stance_cross_entropy_total"]
    if type(total) is not int or total < 0 or total != sum(stance_counts.values()):
        raise ValueError("stance contribution total drifted")
    if not (
        counts["relevance_bce_rows"] <= counts["training_rows"]
        and counts["target_presence_bce_rows"] <= counts["relevance_bce_rows"]
    ):
        raise ValueError("loss-contribution masks violate expected nesting")
    return {
        **counts,
        "stance_cross_entropy_rows_by_target": stance_counts,
        "stance_cross_entropy_total": total,
    }


def _read_parquet_rows(path: Path, *, required: set[str], where: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - Modal dependency path
        raise RuntimeError("acquisition preparation requires PyArrow") from exc
    table = pq.read_table(path)
    if not required <= set(table.column_names):
        missing = sorted(required - set(table.column_names))
        raise ValueError(f"{where} omits required columns: {missing}")
    return table.select(sorted(required)).to_pylist()


def _validate_frame_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame: str,
    where: str,
    label_validator: Any | None = None,
) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    threads: set[str] = set()
    for index, row in enumerate(rows):
        item_id = row.get("item_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in ids
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in threads
            or row.get("frame") != frame
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
        ):
            raise ValueError(f"{where} row {index} violates ID/thread/frame/text contract")
        if any(
            row.get(field) is not None and not isinstance(row[field], str)
            for field in ("parent_context", "submission_context")
        ):
            raise ValueError(f"{where} row {index} has invalid context")
        label = json.loads(row["label_json"])
        validate_v2_label(label, validator=label_validator)
        ids.add(item_id)
        threads.add(thread_id)
    return ids, threads


_TEACHER_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "run_id",
    "packet_id",
    "acquisition_id",
    "row_count",
    "arm_counts",
    "arm_diagnostics",
    "blind_reconciliation",
    "global_provider_telemetry",
    "source_parquet_sha256",
    "acquisition_ledger_sha256",
    "policy_file_sha256",
    "policy_contract_sha256",
    "eligible_frame_sha256",
    "source_inventory_sha256",
    "exclusion_ledger_sha256",
    "checkpoint_bundle_sha256",
    "scoring_artifact_sha256",
    "candidate_score_digest",
    "interleave_order_sha256",
    "packet_manifest_sha256",
    "private_mapping_sha256",
    "private_labels_parquet_sha256",
    "private_diagnostics_sha256",
    "provider_packet_id",
    "provider_run_id",
    "provider_receipt_sha256",
    "runtime_source_bundle_digest",
    "automatic_retry_count",
    "row_replacement_count",
    "receipt_contains_raw_text",
    "receipt_contains_row_ids",
    "receipt_contains_thread_ids",
    "receipt_contains_row_level_labels",
    "evidence_boundary",
}


def validate_teacher_acquisition_inputs(
    *,
    source_path: Path,
    labels_path: Path,
    ledger_path: Path,
    teacher_receipt_path: Path,
    acquisition_config_path: Path,
    descriptor_root: Path,
    label_validator: Any | None = None,
) -> dict[str, Any]:
    """Fail closed unless selector, teacher and training inputs are one chain."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - Modal dependency path
        raise RuntimeError("teacher input validation requires PyArrow") from exc
    receipt = _read_json(teacher_receipt_path, where="acquisition teacher receipt")
    receipt_digest = canonical_sha256(receipt)
    if (
        set(receipt) != _TEACHER_RECEIPT_KEYS
        or receipt.get("schema_version") != acquisition_teacher.SCHEMA_VERSION
        or receipt.get("kind") != "sol-teacher-acquisition-v2-receipt-v1"
        or receipt.get("status") != "complete"
        or teacher_receipt_path.name != f"receipt-{receipt_digest}.json"
    ):
        raise AcquisitionTrainingContractError(
            "acquisition teacher public receipt identity drifted"
        )
    prior_policy_path = acquisition_teacher.DEFAULT_POLICY_PATH
    acquisition_teacher.DEFAULT_POLICY_PATH = acquisition_config_path
    try:
        source_rows, ledger, interleaved = acquisition_teacher._validate_source_and_ledger(
            source_path, ledger_path
        )
    finally:
        acquisition_teacher.DEFAULT_POLICY_PATH = prior_policy_path
    source_sha = file_sha256(source_path)
    ledger_sha = file_sha256(ledger_path)
    labels_sha = file_sha256(labels_path)
    config_sha = file_sha256(acquisition_config_path)
    binding_fields = {
        "eligible_frame_sha256",
        "source_inventory_sha256",
        "exclusion_ledger_sha256",
        "checkpoint_bundle_sha256",
        "scoring_artifact_sha256",
    }
    if (
        receipt["row_count"] != acquisition_teacher.EXPECTED_ROWS
        or receipt["arm_counts"]
        != {arm: acquisition_teacher.ARM_ROWS for arm in acquisition_teacher.ARMS}
        or receipt["source_parquet_sha256"] != source_sha
        or receipt["acquisition_ledger_sha256"] != ledger_sha
        or receipt["private_labels_parquet_sha256"] != labels_sha
        or receipt["acquisition_id"] != ledger["ledger_id"]
        or receipt["policy_file_sha256"] != config_sha
        or receipt["policy_file_sha256"] != ledger["policy_file_sha256"]
        or receipt["policy_contract_sha256"] != ledger["policy_contract_sha256"]
        or receipt["candidate_score_digest"] != ledger["candidate_score_digest"]
        or any(
            receipt[field] != ledger["input_bindings"][field]
            for field in binding_fields
        )
        or receipt["interleave_order_sha256"]
        != canonical_sha256([row["source_sample_id"] for row in interleaved])
        or receipt["automatic_retry_count"] != 0
        or receipt["row_replacement_count"] != 0
        or any(
            receipt[field] is not False
            for field in (
                "receipt_contains_raw_text",
                "receipt_contains_row_ids",
                "receipt_contains_thread_ids",
                "receipt_contains_row_level_labels",
            )
        )
    ):
        raise AcquisitionTrainingContractError(
            "acquisition teacher receipt does not bind the exact source/ledger/labels"
        )
    labels_table = pq.read_table(labels_path)
    if labels_table.schema != acquisition_teacher._combined_schema(receipt):
        raise AcquisitionTrainingContractError("acquisition label Parquet schema drifted")
    label_rows = labels_table.to_pylist()
    if len(label_rows) != acquisition_teacher.EXPECTED_ROWS:
        raise AcquisitionTrainingContractError("acquisition label row count drifted")
    source_ids = {row["opaque_id"] for row in source_rows}
    expected_label_order = [row["source_sample_id"] for row in interleaved]
    if [row["source_sample_id"] for row in label_rows] != expected_label_order:
        raise AcquisitionTrainingContractError(
            "acquisition labels do not conserve the teacher packet order"
        )
    for index, row in enumerate(label_rows):
        if (
            row["source_sample_id"] not in source_ids
            or row["thread_id"] != interleaved[index]["thread_id"]
            or row["acquisition_arm"] != interleaved[index]["acquisition_arm"]
            or row["arm_order"] != interleaved[index]["arm_order"]
            or row["packet_order"] != index
            or row["quality_tier"]
            not in {"exact_consensus", "blind_majority", "informed_adjudication"}
            or type(row["primary_training_eligible"]) is not bool
        ):
            raise AcquisitionTrainingContractError(
                "acquisition label row binding drifted"
            )
        clean_label = validate_v2_label(
            json.loads(row["label_json"]), validator=label_validator
        )
        if (
            row["codability"] != clean_label["codability"]
            or row["relevance"] != clean_label["relevance"]
        ):
            raise AcquisitionTrainingContractError(
                "acquisition label projection disagrees with label_json"
            )
    rare_cells = _validate_exact_rare_cells(
        ledger["rare_cells"], where="teacher ledger"
    )
    result = {
        "source_artifact": _descriptor(
            source_path,
            root=descriptor_root,
            row_count=len(source_rows),
        ),
        "label_artifact": _descriptor(
            labels_path,
            root=descriptor_root,
            row_count=len(label_rows),
        ),
        "ledger_artifact": _descriptor(
            ledger_path,
            root=descriptor_root,
            row_count=len(interleaved),
        ),
        "teacher_receipt_artifact": _descriptor(
            teacher_receipt_path,
            root=descriptor_root,
        ),
        "teacher_receipt_canonical_sha256": receipt_digest,
        "teacher_run_id": receipt["run_id"],
        "teacher_packet_id": receipt["packet_id"],
        "acquisition_id": receipt["acquisition_id"],
        "policy_file_sha256": receipt["policy_file_sha256"],
        "policy_contract_sha256": receipt["policy_contract_sha256"],
        "rare_cells": rare_cells,
        "rare_cell_list_sha256": canonical_sha256(rare_cells),
    }
    assert_metadata_only(result, where="validated acquisition teacher inputs")
    return result


def _materialise_acquisition_source_rows(
    *,
    source_path: Path,
    ledger_path: Path,
    acquisition_config_path: Path,
) -> list[dict[str, Any]]:
    """Join private source text to acquisition design metadata from the ledger.

    The acquisition closeout intentionally stores text-bearing source columns in
    Parquet and policy/selection metadata in the immutable ledger.  The ledger is
    authoritative for arm membership and probability-design fields.
    """

    prior_policy_path = acquisition_teacher.DEFAULT_POLICY_PATH
    acquisition_teacher.DEFAULT_POLICY_PATH = acquisition_config_path
    try:
        source_rows, ledger, _interleaved = acquisition_teacher._validate_source_and_ledger(
            source_path, ledger_path
        )
    finally:
        acquisition_teacher.DEFAULT_POLICY_PATH = prior_policy_path

    design_by_id: dict[str, dict[str, Any]] = {}
    for arm_key, acquisition_arm in (
        ("probability_arm", "probability_random"),
        ("active_arm", "active"),
    ):
        random_arm = acquisition_arm == "probability_random"
        for row in ledger[arm_key]["rows"]:
            sample_id = row["opaque_id"]
            if sample_id in design_by_id:
                raise AcquisitionTrainingContractError(
                    "acquisition ledger repeats a source sample across arms"
                )
            design_by_id[sample_id] = {
                "source_sample_id": sample_id,
                "acquisition_arm": acquisition_arm,
                "selection_component": "probability" if random_arm else row["bucket"],
                "selection_stratum": (
                    canonical_sha256(row["stratum"]) if random_arm else None
                ),
                "inclusion_probability_numerator": (
                    row["inclusion_probability_numerator"] if random_arm else None
                ),
                "inclusion_probability_denominator": (
                    row["inclusion_probability_denominator"] if random_arm else None
                ),
                "inclusion_probability": (
                    float(row["inclusion_probability"]) if random_arm else None
                ),
                "probability_scope": (
                    "conditional-within-frozen-metadata-stratum" if random_arm else None
                ),
            }

    source_ids = {row["opaque_id"] for row in source_rows}
    if source_ids != set(design_by_id):
        raise AcquisitionTrainingContractError(
            "acquisition source and ledger IDs do not conserve"
        )
    return [{**row, **design_by_id[row["opaque_id"]]} for row in source_rows]


def prepare_acquisition_training_frames(
    *,
    base_training_path: Path,
    acquisition_source_path: Path,
    acquisition_labels_path: Path,
    acquisition_ledger_path: Path,
    acquisition_teacher_receipt_path: Path,
    checkpoint_selection_path: Path,
    acquisition_evaluation_path: Path,
    acquisition_config_path: Path,
    expected_config_sha256: str,
    factorised_parent_provenance: Mapping[str, Any],
    output_root: Path,
    descriptor_root: Path,
) -> dict[str, Any]:
    """Materialise base+usable-arm frames without post-label backfill.

    The additions file must contain every queried row.  Rows enter training iff
    ``primary_training_eligible`` is true; the observed arm yields are retained
    exactly and are never equalised by replacement.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - Modal dependency path
        raise RuntimeError("acquisition preparation requires PyArrow") from exc
    label_validator = load_v2_label_validator()
    provenance = validate_teacher_acquisition_inputs(
        source_path=acquisition_source_path,
        labels_path=acquisition_labels_path,
        ledger_path=acquisition_ledger_path,
        teacher_receipt_path=acquisition_teacher_receipt_path,
        acquisition_config_path=acquisition_config_path,
        descriptor_root=descriptor_root,
        label_validator=label_validator,
    )
    base = _read_parquet_rows(
        base_training_path, required=_FRAME_REQUIRED, where="base training frame"
    )
    sources = _materialise_acquisition_source_rows(
        source_path=acquisition_source_path,
        ledger_path=acquisition_ledger_path,
        acquisition_config_path=acquisition_config_path,
    )
    labels = _read_parquet_rows(
        acquisition_labels_path,
        required=_ACQUISITION_LABEL_REQUIRED,
        where="acquisition labels",
    )
    checkpoint_selection = _read_parquet_rows(
        checkpoint_selection_path,
        required=_FRAME_REQUIRED,
        where="checkpoint-selection frame",
    )
    evaluation = _read_parquet_rows(
        acquisition_evaluation_path,
        required=_FRAME_REQUIRED | {"quality_tier"},
        where="acquisition evaluation frame",
    )
    parent_provenance = validate_factorised_parent_provenance(
        factorised_parent_provenance
    )
    if (
        not acquisition_config_path.is_file()
        or file_sha256(acquisition_config_path)
        != _sha(expected_config_sha256, where="expected_config_sha256")
    ):
        raise AcquisitionTrainingContractError("acquisition config digest drifted")
    if len(base) != BASE_TRAINING_ROWS:
        raise AcquisitionTrainingContractError(
            f"base training frame must contain exactly {BASE_TRAINING_ROWS} rows"
        )
    expected_queries = QUERIES_PER_ARM * len(ARMS)
    if len(sources) != expected_queries or len(labels) != expected_queries:
        raise AcquisitionTrainingContractError(
            "acquisition source/labels must conserve 2,000 queries"
        )
    if len(checkpoint_selection) != EVALUATION_ROWS or len(evaluation) != EVALUATION_ROWS:
        raise AcquisitionTrainingContractError(
            "checkpoint-selection and acquisition-evaluation frames must each contain "
            f"exactly {EVALUATION_ROWS} rows"
        )
    base_ids, base_threads = _validate_frame_rows(
        base,
        frame="training",
        where="base training",
        label_validator=label_validator,
    )
    _validate_frame_rows(
        evaluation,
        frame="calibration",
        where="frozen calibration membership",
        label_validator=label_validator,
    )
    evaluation = [{**row, "frame": "acquisition_evaluation"} for row in evaluation]
    evaluation_ids, evaluation_threads = _validate_frame_rows(
        evaluation,
        frame="acquisition_evaluation",
        where="acquisition evaluation",
        label_validator=label_validator,
    )
    checkpoint_ids, checkpoint_threads = _validate_frame_rows(
        checkpoint_selection,
        frame="development",
        where="checkpoint selection",
        label_validator=label_validator,
    )
    if (
        base_ids & evaluation_ids
        or base_threads & evaluation_threads
        or base_ids & checkpoint_ids
        or base_threads & checkpoint_threads
        or checkpoint_ids & evaluation_ids
        or checkpoint_threads & evaluation_threads
    ):
        raise AcquisitionTrainingContractError(
            "training, checkpoint-selection and acquisition-evaluation frames overlap"
        )
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        sample_id = source.get("source_sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or source.get("opaque_id") != sample_id
            or sample_id in source_by_id
        ):
            raise ValueError("acquisition source contains invalid or duplicate source_sample_id")
        source_by_id[sample_id] = source
    label_by_id: dict[str, dict[str, Any]] = {}
    for label in labels:
        sample_id = label.get("source_sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in label_by_id:
            raise ValueError("acquisition labels contain invalid or duplicate source_sample_id")
        label_by_id[sample_id] = label
    if set(source_by_id) != set(label_by_id):
        raise AcquisitionTrainingContractError("acquisition source/label IDs do not conserve")
    additions: list[dict[str, Any]] = []
    for sample_id in sorted(source_by_id):
        source = source_by_id[sample_id]
        label = label_by_id[sample_id]
        if (
            source["thread_id"] != label["thread_id"]
            or source["acquisition_arm"] != label["acquisition_arm"]
        ):
            raise AcquisitionTrainingContractError(
                "acquisition source and label bindings disagree"
            )
        additions.append(
            {
                "item_id": sample_id,
                "frame": "acquisition",
                "thread_id": source["thread_id"],
                "target_text": source["target_text"],
                "parent_context": source["parent_context"],
                "submission_context": source["submission_context"],
                "label_json": label["label_json"],
                "selection_component": source["selection_component"],
                "selection_stratum": source["selection_stratum"],
                "inclusion_probability_numerator": source[
                    "inclusion_probability_numerator"
                ],
                "inclusion_probability_denominator": source[
                    "inclusion_probability_denominator"
                ],
                "inclusion_probability": source["inclusion_probability"],
                "probability_scope": source["probability_scope"],
                "arm": _arm(label["acquisition_arm"]),
                "quality_tier": label["quality_tier"],
                "primary_training_eligible": label["primary_training_eligible"],
            }
        )
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    queried_counts = {arm: 0 for arm in ARMS}
    seen_addition_ids: set[str] = set()
    seen_addition_threads: set[str] = set()
    tier_counts: dict[str, dict[str, int]] = {arm: {} for arm in ARMS}
    for index, row in enumerate(additions):
        arm = _arm(row.get("arm"))
        queried_counts[arm] += 1
        item_id = row.get("item_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen_addition_ids
            or item_id in base_ids
            or item_id in evaluation_ids
            or item_id in checkpoint_ids
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in seen_addition_threads
            or thread_id in base_threads
            or thread_id in evaluation_threads
            or thread_id in checkpoint_threads
            or row.get("frame") != "acquisition"
            or type(row.get("primary_training_eligible")) is not bool
        ):
            raise AcquisitionTrainingContractError(
                f"acquisition addition row {index} violates disjointness or eligibility contract"
            )
        tier = row.get("quality_tier")
        if tier not in {"exact_consensus", "blind_majority", "informed_adjudication"}:
            raise ValueError("acquisition addition quality tier drifted")
        label = validate_v2_label(
            json.loads(row["label_json"]), validator=label_validator
        )
        build_factorised_record(item_id=item_id, row=row, label=label, separator="[SEP]")
        seen_addition_ids.add(item_id)
        seen_addition_threads.add(thread_id)
        tier_counts[arm][tier] = tier_counts[arm].get(tier, 0) + 1
        if row["primary_training_eligible"]:
            training_row = {field: row.get(field) for field in _FRAME_REQUIRED}
            training_row["frame"] = "training"
            by_arm[arm].append(training_row)
    if queried_counts != {arm: QUERIES_PER_ARM for arm in ARMS}:
        raise AcquisitionTrainingContractError(
            f"each arm must conserve exactly {QUERIES_PER_ARM} queries"
        )

    schema = pa.schema(
        [
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("frame", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("parent_context", pa.string()),
            pa.field("submission_context", pa.string()),
            pa.field("label_json", pa.string(), nullable=False),
            pa.field("selection_component", pa.string(), nullable=False),
            pa.field("selection_stratum", pa.string()),
            pa.field("inclusion_probability_numerator", pa.int64()),
            pa.field("inclusion_probability_denominator", pa.int64()),
            pa.field("inclusion_probability", pa.float64()),
            pa.field("probability_scope", pa.string()),
        ]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    descriptors: dict[str, dict[str, Any]] = {}
    contribution_counts: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [*base, *by_arm[arm]]
        _validate_frame_rows(
            rows,
            frame="training",
            where=f"{arm} training",
            label_validator=label_validator,
        )
        path = output_root / f"training-{arm}.parquet"
        temporary = path.with_suffix(".parquet.incomplete")
        table = pa.Table.from_pylist(rows, schema=schema)
        if path.exists():
            if not pq.read_table(path).equals(table):
                raise AcquisitionTrainingContractError(
                    f"existing immutable {arm} training frame differs"
                )
        else:
            if temporary.exists():
                raise AcquisitionTrainingContractError(
                    f"stale incomplete {arm} training frame exists"
                )
            pq.write_table(table, temporary, compression="zstd")
            os.replace(temporary, path)
        threads = sorted(row["thread_id"] for row in rows)
        descriptors[arm] = _descriptor(
            path,
            root=descriptor_root,
            row_count=len(rows),
            thread_set_sha256=canonical_sha256(threads),
            frame="training",
        )
        contribution_counts[arm] = loss_contribution_counts(
            rows, label_validator=label_validator
        )

    evaluation_schema = pa.schema(
        [*schema, pa.field("quality_tier", pa.string(), nullable=False)]
    )
    evaluation_rows = [{**row, "frame": "acquisition_evaluation"} for row in evaluation]
    evaluation_path = output_root / "acquisition-evaluation.parquet"
    temporary = evaluation_path.with_suffix(".parquet.incomplete")
    evaluation_table = pa.Table.from_pylist(evaluation_rows, schema=evaluation_schema)
    if evaluation_path.exists():
        if not pq.read_table(evaluation_path).equals(evaluation_table):
            raise AcquisitionTrainingContractError(
                "existing immutable acquisition evaluation frame differs"
            )
    else:
        if temporary.exists():
            raise AcquisitionTrainingContractError(
                "stale incomplete acquisition evaluation frame exists"
            )
        pq.write_table(evaluation_table, temporary, compression="zstd")
        os.replace(temporary, evaluation_path)
    evaluation_descriptor = _descriptor(
        evaluation_path,
        root=descriptor_root,
        row_count=len(evaluation_rows),
        thread_set_sha256=canonical_sha256(sorted(evaluation_threads)),
        frame="acquisition_evaluation",
    )
    checkpoint_descriptor = _descriptor(
        checkpoint_selection_path,
        root=descriptor_root,
        row_count=len(checkpoint_selection),
        thread_set_sha256=canonical_sha256(sorted(checkpoint_threads)),
        frame="development",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREPARATION_KIND,
        "base_training_rows": len(base),
        "queried_rows_by_arm": queried_counts,
        "primary_eligible_rows_by_arm": {
            arm: len(by_arm[arm]) for arm in ARMS
        },
        "training_rows_by_arm": {
            arm: descriptors[arm]["row_count"] for arm in ARMS
        },
        "quality_tier_counts_by_arm": tier_counts,
        "loss_contribution_counts_by_arm": contribution_counts,
        "training_frames": descriptors,
        "checkpoint_selection_frame": checkpoint_descriptor,
        "acquisition_evaluation_frame": evaluation_descriptor,
        "validated_teacher_inputs": provenance,
        "factorised_parent_provenance": parent_provenance,
        "thread_overlap_counts": {
            "base_checkpoint_selection": 0,
            "base_acquisition_evaluation": 0,
            "checkpoint_selection_acquisition_evaluation": 0,
            "random_active_additions": 0,
            "additions_checkpoint_selection": 0,
            "additions_acquisition_evaluation": 0,
        },
        "backfilled_rows": 0,
        "acquisition_config_sha256": expected_config_sha256,
        "locked_test_rows_accessed": 0,
    }
    summary["preparation_id"] = canonical_sha256(summary)
    assert_metadata_only(summary, where="acquisition training preparation")
    return summary


def validate_preparation_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "base_training_rows",
        "queried_rows_by_arm",
        "primary_eligible_rows_by_arm",
        "training_rows_by_arm",
        "quality_tier_counts_by_arm",
        "loss_contribution_counts_by_arm",
        "training_frames",
        "checkpoint_selection_frame",
        "acquisition_evaluation_frame",
        "validated_teacher_inputs",
        "factorised_parent_provenance",
        "thread_overlap_counts",
        "backfilled_rows",
        "acquisition_config_sha256",
        "locked_test_rows_accessed",
        "preparation_id",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != PREPARATION_KIND
        or value.get("base_training_rows") != BASE_TRAINING_ROWS
        or value.get("queried_rows_by_arm")
        != {arm: QUERIES_PER_ARM for arm in ARMS}
        or value.get("backfilled_rows") != 0
        or value.get("locked_test_rows_accessed") != 0
    ):
        raise AcquisitionTrainingContractError("preparation summary identity drifted")
    body = {key: value[key] for key in expected - {"preparation_id"}}
    if value.get("preparation_id") != canonical_sha256(body):
        raise AcquisitionTrainingContractError("preparation summary digest drifted")
    frames = value.get("training_frames")
    if not isinstance(frames, Mapping) or set(frames) != set(ARMS):
        raise ValueError("preparation training-frame inventory drifted")
    for arm in ARMS:
        frame = artifact_descriptor(frames[arm], where=f"prepared {arm} frame")
        if frame.get("frame") != "training":
            raise AcquisitionTrainingContractError("prepared training frame drifted")
    checkpoint = artifact_descriptor(
        value["checkpoint_selection_frame"], where="checkpoint_selection_frame"
    )
    evaluation = artifact_descriptor(
        value["acquisition_evaluation_frame"], where="acquisition_evaluation_frame"
    )
    if (
        checkpoint.get("frame") != "development"
        or evaluation.get("frame") != "acquisition_evaluation"
        or checkpoint.get("row_count") != 600
        or evaluation.get("row_count") != 600
    ):
        raise AcquisitionTrainingContractError(
            "checkpoint-selection and acquisition-evaluation frames must bind 600 rows"
        )
    if value["checkpoint_selection_frame"]["sha256"] == value[
        "acquisition_evaluation_frame"
    ]["sha256"]:
        raise AcquisitionTrainingContractError(
            "checkpoint-selection and acquisition-evaluation frames must differ"
        )
    provenance = value.get("validated_teacher_inputs")
    expected_provenance = {
        "source_artifact",
        "label_artifact",
        "ledger_artifact",
        "teacher_receipt_artifact",
        "teacher_receipt_canonical_sha256",
        "teacher_run_id",
        "teacher_packet_id",
        "acquisition_id",
        "policy_file_sha256",
        "policy_contract_sha256",
        "rare_cells",
        "rare_cell_list_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance:
        raise ValueError("validated teacher provenance schema drifted")
    for name in (
        "source_artifact",
        "label_artifact",
        "ledger_artifact",
        "teacher_receipt_artifact",
    ):
        artifact_descriptor(provenance[name], where=f"teacher provenance {name}")
    for name in (
        "teacher_receipt_canonical_sha256",
        "teacher_run_id",
        "teacher_packet_id",
        "acquisition_id",
        "policy_file_sha256",
        "policy_contract_sha256",
        "rare_cell_list_sha256",
    ):
        _sha(provenance[name], where=f"teacher provenance {name}")
    rare = _validate_exact_rare_cells(
        provenance["rare_cells"], where="teacher provenance"
    )
    if (
        provenance["rare_cell_list_sha256"] != canonical_sha256(rare)
    ):
        raise AcquisitionTrainingContractError("teacher rare-cell digest drifted")
    if (
        provenance["source_artifact"]["row_count"] != 2 * QUERIES_PER_ARM
        or provenance["label_artifact"]["row_count"] != 2 * QUERIES_PER_ARM
        or provenance["ledger_artifact"]["row_count"] != 2 * QUERIES_PER_ARM
        or provenance["policy_file_sha256"] != value["acquisition_config_sha256"]
    ):
        raise AcquisitionTrainingContractError("validated teacher provenance drifted")
    validate_factorised_parent_provenance(value["factorised_parent_provenance"])
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_preparation_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "preparation_spec_sha256",
        "preparation",
        "receipt_id",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind")
        != "modernbert-acquisition-training-preparation-receipt-v1"
    ):
        raise ValueError("acquisition preparation receipt schema drifted")
    _sha(value.get("preparation_spec_sha256"), where="preparation spec digest")
    preparation = validate_preparation_summary(value["preparation"])
    body = {key: value[key] for key in expected - {"receipt_id"}}
    if value.get("receipt_id") != canonical_sha256(body):
        raise AcquisitionTrainingContractError("preparation receipt digest drifted")
    return {
        **json.loads(json.dumps(body, sort_keys=True, allow_nan=False)),
        "preparation": preparation,
        "receipt_id": value["receipt_id"],
    }


def freeze_experiment_contract(
    *,
    preparation: Mapping[str, Any],
    acquisition_config: Mapping[str, Any],
    gate_policy: AcquisitionGatePolicy,
    rare_cells: Sequence[Mapping[str, Any]],
    loss_contribution_counts_by_arm: Mapping[str, Mapping[str, Any]],
    source_bundle_sha256: str,
    dependency_lock_sha256: str,
    rate_card_usd_per_gpu_second: str | float,
    cumulative_measured_spend_usd: str | float,
    active_reservation_usd: str | float,
    planned_phase_upper_usd: str | float,
    hard_cost_cap_usd: str | float = "200",
) -> dict[str, Any]:
    prepared = validate_preparation_summary(preparation)
    if acquisition_config.get("sha256") != prepared["acquisition_config_sha256"]:
        raise AcquisitionTrainingContractError(
            "prepared and experiment acquisition-config digests differ"
        )
    frames = {
        arm: artifact_descriptor(
            prepared["training_frames"][arm], where=f"{arm}_training_frame"
        )
        for arm in ARMS
    }
    checkpoint_selection = artifact_descriptor(
        prepared["checkpoint_selection_frame"], where="checkpoint_selection_frame"
    )
    evaluation = artifact_descriptor(
        prepared["acquisition_evaluation_frame"], where="acquisition_evaluation_frame"
    )
    if any(frame.get("frame") != "training" for frame in frames.values()):
        raise ValueError("arm training frame identity drifted")
    if (
        checkpoint_selection.get("frame") != "development"
        or evaluation.get("frame") != "acquisition_evaluation"
        or checkpoint_selection.get("row_count") != 600
        or evaluation.get("row_count") != 600
    ):
        raise ValueError("both evaluation frames must bind exactly 600 rows")
    if frames["random"].get("thread_set_sha256") == frames["active"].get(
        "thread_set_sha256"
    ):
        raise AcquisitionTrainingContractError("arm training frames must differ")
    rare = _validate_exact_rare_cells(rare_cells, where="experiment")
    if rare != prepared["validated_teacher_inputs"]["rare_cells"]:
        raise AcquisitionTrainingContractError(
            "experiment rare cells do not match preparation evidence"
        )
    if (
        loss_contribution_counts_by_arm
        != prepared["loss_contribution_counts_by_arm"]
    ):
        raise AcquisitionTrainingContractError(
            "experiment loss contributions do not match preparation evidence"
        )
    if set(loss_contribution_counts_by_arm) != set(ARMS):
        raise ValueError("loss-contribution inventory must contain both arms")
    contributions = {
        arm: _validate_loss_contribution_counts(loss_contribution_counts_by_arm[arm])
        for arm in ARMS
    }
    for arm in ARMS:
        if contributions[arm].get("training_rows") != frames[arm].get("row_count"):
            raise AcquisitionTrainingContractError(
                f"{arm} contribution counts do not conserve its training frame"
            )
    rate = Decimal(str(rate_card_usd_per_gpu_second))
    measured = Decimal(str(cumulative_measured_spend_usd))
    active = Decimal(str(active_reservation_usd))
    planned = Decimal(str(planned_phase_upper_usd))
    cap = Decimal(str(hard_cost_cap_usd))
    if (
        any(not value.is_finite() or value < 0 for value in (measured, active, planned))
        or not rate.is_finite()
        or rate <= 0
        or not cap.is_finite()
        or cap <= 0
        or cap > HARD_COST_CAP_USD
        or measured + active + planned > cap
    ):
        raise AcquisitionTrainingContractError("acquisition training cost contract is invalid")
    bindings = {
        "preparation_id": prepared["preparation_id"],
        "training_frames": frames,
        "checkpoint_selection_frame": checkpoint_selection,
        "acquisition_evaluation_frame": evaluation,
        "acquisition_source": prepared["validated_teacher_inputs"]["source_artifact"],
        "acquisition_labels": prepared["validated_teacher_inputs"]["label_artifact"],
        "acquisition_ledger": prepared["validated_teacher_inputs"]["ledger_artifact"],
        "acquisition_teacher_receipt": prepared["validated_teacher_inputs"][
            "teacher_receipt_artifact"
        ],
        "teacher_provenance": prepared["validated_teacher_inputs"],
        "factorised_parent_provenance": prepared[
            "factorised_parent_provenance"
        ],
        "acquisition_config": {
            "repo_relative_path": _safe_relative(
                acquisition_config.get("repo_relative_path"),
                where="acquisition_config.repo_relative_path",
            ),
            "sha256": _sha(
                acquisition_config.get("sha256"), where="acquisition_config.sha256"
            ),
            "bytes": _positive_int(
                acquisition_config.get("bytes"), where="acquisition_config.bytes"
            ),
        },
        "rare_cells": rare,
        "rare_cells_sha256": canonical_sha256(rare),
        "loss_contribution_counts_by_arm": contributions,
        "source_bundle_sha256": _sha(
            source_bundle_sha256, where="source_bundle_sha256"
        ),
        "dependency_lock_sha256": _sha(
            dependency_lock_sha256, where="dependency_lock_sha256"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPERIMENT_KIND,
        "namespace": NAMESPACE,
        "bindings": bindings,
        "registered_design": {
            "arms": list(ARMS),
            "components": list(COMPONENTS),
            "seeds": list(SEEDS),
            "expected_trials": EXPECTED_TRIALS,
            "configs": {
                component: frozen_component_config(component) for component in COMPONENTS
            },
            "evaluation_use": (
                "retained_development_checkpoint_selection_then_one_time_"
                "acquisition_evaluation"
            ),
            "full_coverage_threshold": 0.5,
            "calibration_or_abstention_selection": False,
            "gate_policy": asdict(gate_policy),
        },
        "compute": {
            "allowed_gpus": [GPU_TYPE],
            "account_gpu_limit": 10,
            "max_concurrent_trials": MAX_CONCURRENT_TRIALS,
            "gpu_fallback_allowed": False,
            "rate_card_usd_per_gpu_second": {GPU_TYPE: format(rate, "f")},
            "cumulative_measured_spend_usd": format(measured, "f"),
            "active_reservation_usd": format(active, "f"),
            "planned_phase_upper_usd": format(planned, "f"),
            "hard_cost_cap_usd": format(cap, "f"),
            "remaining_after_plan_usd": format(cap - measured - active - planned, "f"),
        },
        "evidence_boundary": {
            "acquisition_evaluation_only": True,
            "calibration_authorised": False,
            "locked_test_authorised": False,
            "locked_test_rows_accessed": 0,
            "human_validation_claim_authorised": False,
            "corpus_inference_authorised": False,
        },
    }
    return {**body, "experiment_run_id": canonical_sha256(body)}


def validate_experiment_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "namespace",
        "bindings",
        "registered_design",
        "compute",
        "evidence_boundary",
        "experiment_run_id",
    }
    if set(value) != expected:
        raise ValueError("acquisition training experiment schema drifted")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != EXPERIMENT_KIND
        or value.get("namespace") != NAMESPACE
    ):
        raise AcquisitionTrainingContractError("acquisition experiment identity drifted")
    bindings = value.get("bindings")
    design = value.get("registered_design")
    boundary = value.get("evidence_boundary")
    if not all(isinstance(item, Mapping) for item in (bindings, design, boundary)):
        raise ValueError("acquisition experiment nested contract is invalid")
    expected_binding_keys = {
        "preparation_id", "training_frames", "checkpoint_selection_frame",
        "acquisition_evaluation_frame", "acquisition_source", "acquisition_labels",
        "acquisition_ledger", "acquisition_teacher_receipt", "teacher_provenance",
        "factorised_parent_provenance",
        "acquisition_config", "rare_cells",
        "rare_cells_sha256", "loss_contribution_counts_by_arm",
        "source_bundle_sha256", "dependency_lock_sha256", "model_id",
        "model_revision", "tokenizer_revision",
    }
    if set(bindings) != expected_binding_keys:
        raise ValueError("acquisition input binding schema drifted")
    if design.get("arms") != list(ARMS) or design.get("components") != list(COMPONENTS):
        raise AcquisitionTrainingContractError("acquisition trial design drifted")
    if design.get("seeds") != list(SEEDS) or design.get("expected_trials") != 12:
        raise AcquisitionTrainingContractError("acquisition seed/trial design drifted")
    if design.get("configs") != {
        component: frozen_component_config(component) for component in COMPONENTS
    }:
        raise AcquisitionTrainingContractError("frozen B4 optimisation recipe drifted")
    gate_policy = design.get("gate_policy")
    if not isinstance(gate_policy, Mapping):
        raise ValueError("acquisition gate policy binding is missing")
    clean_gate_policy = AcquisitionGatePolicy(**dict(gate_policy))
    if bindings.get("model_id") != MODEL_ID or bindings.get("model_revision") != MODEL_REVISION:
        raise AcquisitionTrainingContractError("pinned model binding drifted")
    if bindings.get("tokenizer_revision") != TOKENIZER_REVISION:
        raise AcquisitionTrainingContractError("pinned tokenizer binding drifted")
    frames = bindings.get("training_frames")
    if not isinstance(frames, Mapping) or set(frames) != set(ARMS):
        raise ValueError("training frame inventory drifted")
    for arm in ARMS:
        frame = artifact_descriptor(frames[arm], where=f"{arm}_training_frame")
        if frame.get("frame") != "training" or "row_count" not in frame:
            raise AcquisitionTrainingContractError("arm training frame binding drifted")
    _sha(bindings.get("preparation_id"), where="preparation_id")
    checkpoint_selection = artifact_descriptor(
        bindings.get("checkpoint_selection_frame", {}),
        where="checkpoint_selection_frame",
    )
    evaluation = artifact_descriptor(
        bindings.get("acquisition_evaluation_frame", {}),
        where="acquisition_evaluation_frame",
    )
    if (
        checkpoint_selection.get("frame") != "development"
        or evaluation.get("frame") != "acquisition_evaluation"
        or checkpoint_selection.get("row_count") != 600
        or evaluation.get("row_count") != 600
        or checkpoint_selection["sha256"] == evaluation["sha256"]
    ):
        raise AcquisitionTrainingContractError("evaluation frame binding drifted")
    for name in (
        "acquisition_source",
        "acquisition_labels",
        "acquisition_ledger",
        "acquisition_teacher_receipt",
    ):
        artifact_descriptor(bindings.get(name, {}), where=name)
    provenance = bindings.get("teacher_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("teacher provenance binding is missing")
    expected_provenance_keys = {
        "source_artifact", "label_artifact", "ledger_artifact",
        "teacher_receipt_artifact",
        "teacher_receipt_canonical_sha256", "teacher_run_id", "teacher_packet_id",
        "acquisition_id", "policy_file_sha256", "policy_contract_sha256",
        "rare_cells", "rare_cell_list_sha256",
    }
    if set(provenance) != expected_provenance_keys:
        raise ValueError("teacher provenance binding schema drifted")
    for name, binding_name in (
        ("source_artifact", "acquisition_source"),
        ("label_artifact", "acquisition_labels"),
        ("ledger_artifact", "acquisition_ledger"),
        ("teacher_receipt_artifact", "acquisition_teacher_receipt"),
    ):
        if artifact_descriptor(provenance[name], where=name) != bindings[binding_name]:
            raise AcquisitionTrainingContractError("teacher provenance descriptor drifted")
    for name in (
        "teacher_receipt_canonical_sha256", "teacher_run_id", "teacher_packet_id",
        "acquisition_id", "policy_file_sha256", "policy_contract_sha256",
        "rare_cell_list_sha256",
    ):
        _sha(provenance[name], where=f"teacher provenance {name}")
    config = bindings.get("acquisition_config")
    if not isinstance(config, Mapping) or set(config) != {
        "repo_relative_path", "sha256", "bytes"
    }:
        raise ValueError("acquisition config binding schema drifted")
    _safe_relative(config.get("repo_relative_path"), where="acquisition config path")
    _sha(config.get("sha256"), where="acquisition config digest")
    _positive_int(config.get("bytes"), where="acquisition config bytes")
    if clean_gate_policy.policy_file_sha256 != config.get("sha256"):
        raise AcquisitionTrainingContractError(
            "gate policy and acquisition config digests differ"
        )
    if provenance["policy_file_sha256"] != config.get("sha256"):
        raise AcquisitionTrainingContractError(
            "teacher provenance and acquisition config digests differ"
        )
    validate_factorised_parent_provenance(
        bindings["factorised_parent_provenance"]
    )
    rare = _validate_exact_rare_cells(
        bindings.get("rare_cells", []), where="experiment binding"
    )
    if (
        bindings.get("rare_cells_sha256") != canonical_sha256(rare)
    ):
        raise AcquisitionTrainingContractError("rare-cell digest drifted")
    if (
        provenance["rare_cells"] != rare
        or provenance["rare_cell_list_sha256"] != canonical_sha256(rare)
    ):
        raise AcquisitionTrainingContractError(
            "teacher provenance and experiment rare cells differ"
        )
    _sha(bindings.get("source_bundle_sha256"), where="source_bundle_sha256")
    _sha(bindings.get("dependency_lock_sha256"), where="dependency_lock_sha256")
    contributions = bindings.get("loss_contribution_counts_by_arm")
    if not isinstance(contributions, Mapping) or set(contributions) != set(ARMS):
        raise ValueError("loss-contribution binding drifted")
    for arm in ARMS:
        counts = _validate_loss_contribution_counts(contributions[arm])
        if counts["training_rows"] != frames[arm].get("row_count"):
            raise AcquisitionTrainingContractError("loss-contribution count drifted")
    expected_design_keys = {
        "arms", "components", "seeds", "expected_trials", "configs",
        "evaluation_use", "full_coverage_threshold",
        "calibration_or_abstention_selection", "gate_policy",
    }
    if set(design) != expected_design_keys:
        raise ValueError("registered acquisition design schema drifted")
    if (
        design.get("evaluation_use")
        != "retained_development_checkpoint_selection_then_one_time_acquisition_evaluation"
        or design.get("full_coverage_threshold") != 0.5
        or design.get("calibration_or_abstention_selection") is not False
    ):
        raise AcquisitionTrainingContractError("evaluation-use contract drifted")
    compute = value.get("compute")
    expected_compute_keys = {
        "allowed_gpus", "account_gpu_limit", "max_concurrent_trials",
        "gpu_fallback_allowed", "rate_card_usd_per_gpu_second",
        "cumulative_measured_spend_usd", "active_reservation_usd",
        "planned_phase_upper_usd", "hard_cost_cap_usd",
        "remaining_after_plan_usd",
    }
    if not isinstance(compute, Mapping) or set(compute) != expected_compute_keys:
        raise ValueError("acquisition compute schema drifted")
    if (
        compute.get("allowed_gpus") != [GPU_TYPE]
        or compute.get("account_gpu_limit") != 10
        or compute.get("max_concurrent_trials") != MAX_CONCURRENT_TRIALS
        or compute.get("gpu_fallback_allowed") is not False
    ):
        raise AcquisitionTrainingContractError("acquisition compute inventory drifted")
    rates = compute.get("rate_card_usd_per_gpu_second")
    if not isinstance(rates, Mapping) or set(rates) != {GPU_TYPE}:
        raise ValueError("acquisition rate-card schema drifted")
    try:
        rate = Decimal(str(rates[GPU_TYPE]))
        measured = Decimal(str(compute["cumulative_measured_spend_usd"]))
        active = Decimal(str(compute["active_reservation_usd"]))
        planned = Decimal(str(compute["planned_phase_upper_usd"]))
        cap = Decimal(str(compute["hard_cost_cap_usd"]))
        remaining = Decimal(str(compute["remaining_after_plan_usd"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("acquisition compute values are invalid") from exc
    if (
        not rate.is_finite() or rate <= 0
        or any(not item.is_finite() or item < 0 for item in (measured, active, planned))
        or not cap.is_finite() or cap <= 0 or cap > HARD_COST_CAP_USD
        or measured + active + planned > cap
        or remaining != cap - measured - active - planned
    ):
        raise AcquisitionTrainingContractError("acquisition cost contract drifted")
    expected_boundary = {
        "acquisition_evaluation_only": True,
        "calibration_authorised": False,
        "locked_test_authorised": False,
        "locked_test_rows_accessed": 0,
        "human_validation_claim_authorised": False,
        "corpus_inference_authorised": False,
    }
    if dict(boundary) != expected_boundary:
        raise AcquisitionTrainingContractError("acquisition evidence boundary drifted")
    body = {key: value[key] for key in expected - {"experiment_run_id"}}
    if value.get("experiment_run_id") != canonical_sha256(body):
        raise AcquisitionTrainingContractError("acquisition experiment run ID drifted")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def freeze_trial_spec(
    experiment: Mapping[str, Any], *, arm: str, component: str, seed: int
) -> dict[str, Any]:
    clean = validate_experiment_contract(experiment)
    arm = _arm(arm)
    component = _component(component)
    if seed not in SEEDS:
        raise ValueError("seed is not registered")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_SPEC_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "arm": arm,
        "component": component,
        "optimiser_seed": seed,
        "pairing_key": f"seed={seed}",
        "gpu_type": GPU_TYPE,
        "max_gpu_seconds": TRIAL_MAX_GPU_SECONDS,
        "config": frozen_component_config(component),
        "training_frame_sha256": clean["bindings"]["training_frames"][arm]["sha256"],
        "checkpoint_selection_frame_sha256": clean["bindings"][
            "checkpoint_selection_frame"
        ]["sha256"],
        "acquisition_evaluation_frame_sha256": clean["bindings"][
            "acquisition_evaluation_frame"
        ]["sha256"],
        "acquisition_ledger_sha256": clean["bindings"]["acquisition_ledger"]["sha256"],
        "loss_contribution_counts": clean["bindings"]["loss_contribution_counts_by_arm"][arm],
        "locked_test_rows_accessed": 0,
    }
    return {**body, "trial_id": canonical_sha256(body)}


def validate_trial_spec(
    value: Mapping[str, Any], *, experiment: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version", "kind", "experiment_run_id", "arm", "component",
        "optimiser_seed", "pairing_key", "gpu_type", "max_gpu_seconds", "config",
        "training_frame_sha256", "checkpoint_selection_frame_sha256",
        "acquisition_evaluation_frame_sha256",
        "acquisition_ledger_sha256", "loss_contribution_counts",
        "locked_test_rows_accessed", "trial_id",
    }
    if set(value) != required:
        raise ValueError("acquisition trial schema drifted")
    expected = freeze_trial_spec(
        experiment,
        arm=str(value.get("arm")),
        component=str(value.get("component")),
        seed=value.get("optimiser_seed"),
    )
    if dict(value) != expected:
        raise AcquisitionTrainingContractError("acquisition trial binding drifted")
    return expected


def build_run_manifest(experiment: Mapping[str, Any]) -> dict[str, Any]:
    clean = validate_experiment_contract(experiment)
    trials = [
        freeze_trial_spec(
            clean, arm=arm, component=component, seed=seed,
        )
        for arm in ARMS for component in COMPONENTS for seed in SEEDS
    ]
    rate = Decimal(clean["compute"]["rate_card_usd_per_gpu_second"][GPU_TYPE])
    reserved = sum(rate * Decimal(trial["max_gpu_seconds"]) for trial in trials)
    planned = Decimal(clean["compute"]["planned_phase_upper_usd"])
    if reserved > planned:
        raise AcquisitionTrainingContractError("registered trials exceed phase reservation")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_MANIFEST_KIND,
        "experiment_run_id": clean["experiment_run_id"],
        "phase": "paired-acquisition-comparison",
        "experiment_contract": clean,
        "trials": trials,
        "reserved_cost_usd": format(reserved, "f"),
        "locked_test_rows_accessed": 0,
    }
    return {**body, "phase_run_id": canonical_sha256(body)}


def validate_run_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version", "kind", "experiment_run_id", "phase",
        "experiment_contract", "trials", "reserved_cost_usd",
        "locked_test_rows_accessed", "phase_run_id",
    }
    if set(value) != expected or value.get("kind") != RUN_MANIFEST_KIND:
        raise ValueError("acquisition run manifest schema drifted")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("phase") != "paired-acquisition-comparison"
        or value.get("locked_test_rows_accessed") != 0
    ):
        raise AcquisitionTrainingContractError("acquisition run identity drifted")
    experiment = validate_experiment_contract(value["experiment_contract"])
    if value.get("experiment_run_id") != experiment["experiment_run_id"]:
        raise AcquisitionTrainingContractError("run and experiment IDs differ")
    trials = value.get("trials")
    if not isinstance(trials, list):
        raise ValueError("acquisition trials must be a list")
    clean_trials = [validate_trial_spec(trial, experiment=experiment) for trial in trials]
    expected_keys = {
        (arm, component, seed)
        for arm in ARMS for component in COMPONENTS for seed in SEEDS
    }
    observed = {
        (trial["arm"], trial["component"], trial["optimiser_seed"])
        for trial in clean_trials
    }
    if len(clean_trials) != EXPECTED_TRIALS or observed != expected_keys:
        raise AcquisitionTrainingContractError("run must contain exact 2x2x3 trials")
    if any(trial["max_gpu_seconds"] != TRIAL_MAX_GPU_SECONDS for trial in clean_trials):
        raise AcquisitionTrainingContractError("trial runtime reservation drifted")
    rate = Decimal(experiment["compute"]["rate_card_usd_per_gpu_second"][GPU_TYPE])
    reserved = sum(
        rate * Decimal(trial["max_gpu_seconds"]) for trial in clean_trials
    )
    if value.get("reserved_cost_usd") != format(reserved, "f"):
        raise AcquisitionTrainingContractError("reserved trial cost drifted")
    body = {key: value[key] for key in expected - {"phase_run_id"}}
    if value.get("phase_run_id") != canonical_sha256(body):
        raise AcquisitionTrainingContractError("phase run ID drifted")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def build_trial_job(manifest: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    clean = validate_run_manifest(manifest)
    by_id = {row["trial_id"]: row for row in clean["trials"]}
    if trial.get("trial_id") not in by_id or dict(trial) != by_id[trial["trial_id"]]:
        raise ValueError("trial is not registered in the manifest")
    body = {
        "schema_version": SCHEMA_VERSION,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "run_manifest_sha256": canonical_sha256(clean),
        "run_manifest": clean,
        "experiment_contract": clean["experiment_contract"],
        "trial_spec": dict(trial),
        "trial_spec_sha256": canonical_sha256(trial),
        "locked_test_rows_accessed": 0,
    }
    return body


def _validate_job(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        "schema_version", "experiment_run_id", "phase_run_id",
        "run_manifest_sha256", "run_manifest", "experiment_contract",
        "trial_spec", "trial_spec_sha256", "locked_test_rows_accessed",
    }
    if set(job) != expected or job.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("acquisition training job schema drifted")
    manifest = validate_run_manifest(job["run_manifest"])
    trial = validate_trial_spec(
        job["trial_spec"], experiment=manifest["experiment_contract"]
    )
    if (
        job["experiment_run_id"] != manifest["experiment_run_id"]
        or job["phase_run_id"] != manifest["phase_run_id"]
        or job["run_manifest_sha256"] != canonical_sha256(manifest)
        or job["experiment_contract"] != manifest["experiment_contract"]
        or job["trial_spec_sha256"] != canonical_sha256(trial)
        or job["locked_test_rows_accessed"] != 0
    ):
        raise AcquisitionTrainingContractError("acquisition training job binding drifted")
    return manifest, trial


def load_private_frame(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_frame: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load an exact private train, development, or acquisition-evaluation frame."""

    clean = artifact_descriptor(descriptor, where="private_frame")
    if expected_frame not in {"training", "development", "acquisition_evaluation"}:
        raise ValueError("unsupported private frame identity")
    if clean.get("frame") != expected_frame:
        raise AcquisitionTrainingContractError("private frame descriptor identity drifted")
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or file_sha256(path) != clean["sha256"]
    ):
        raise AcquisitionTrainingContractError("private frame is missing or corrupt")
    required = set(_FRAME_REQUIRED)
    if expected_frame == "acquisition_evaluation":
        required.add("quality_tier")
    rows = _read_parquet_rows(path, required=required, where="private frame")
    if len(rows) != clean.get("row_count"):
        raise AcquisitionTrainingContractError("private frame row count drifted")
    ids, _ = _validate_frame_rows(rows, frame=expected_frame, where="private frame")
    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        reference[row["item_id"]] = validate_v2_label(json.loads(row["label_json"]))
    if len(reference) != len(ids):
        raise AcquisitionTrainingContractError("private frame identity conservation failed")
    return rows, reference


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _reference_projection(label: Mapping[str, Any]) -> tuple[bool, dict[str, str]]:
    clean = validate_v2_label(label)
    material = clean.get("codability") == "codable" and clean.get("relevance") == "material"
    if not material:
        return False, {}
    targets = clean.get("targets")
    if not isinstance(targets, list):
        return True, {}
    projected: dict[str, str] = {}
    for target_value in targets:
        if not isinstance(target_value, Mapping):
            continue
        target = target_value.get("target")
        stance = target_value.get("stance")
        if target in ANALYTIC_TARGET_CLASSES and stance in STANCE_CLASSES_B4:
            projected[target] = stance
    return True, projected


def _binary_f1(pairs: Sequence[tuple[bool, bool]]) -> float:
    tp = sum(reference and prediction for reference, prediction in pairs)
    fp = sum(not reference and prediction for reference, prediction in pairs)
    fn = sum(reference and not prediction for reference, prediction in pairs)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _score_checkpoint(
    *,
    component: str,
    reference: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    selected_epoch: int,
    loss_counts: Mapping[str, Any],
) -> dict[str, Any]:
    component = _component(component)
    by_id = {row.get("item_id"): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(reference):
        raise AcquisitionTrainingContractError(
            "checkpoint predictions do not conserve acquisition evaluation IDs"
        )
    invalid = 0
    if component == "relevance":
        pairs = []
        for item_id, label in reference.items():
            material, _ = _reference_projection(label)
            logit = by_id[item_id].get("relevance_logit")
            if not isinstance(logit, (int, float)) or not math.isfinite(float(logit)):
                invalid += 1
                predicted = False
            else:
                predicted = _sigmoid(float(logit)) >= 0.5
            pairs.append((material, predicted))
        positives = sum(reference_value for reference_value, _ in pairs)
        recall = (
            0.0
            if positives == 0
            else sum(reference_value and prediction for reference_value, prediction in pairs)
            / positives
        )
        negative_pairs = [
            (not reference_value, not prediction)
            for reference_value, prediction in pairs
        ]
        macro = 0.5 * (_binary_f1(pairs) + _binary_f1(negative_pairs))
        return {
            "selected_epoch": selected_epoch,
            "evaluation_rows": len(reference),
            "relevance_macro_f1": macro,
            "material_recall": recall,
            "checkpoint_score": 0.5 * macro + 0.5 * recall,
            "invalid_outputs": invalid,
            "loss_contribution_counts": dict(loss_counts),
        }
    tuple_pairs: list[tuple[bool, bool]] = []
    for item_id, label in reference.items():
        material, target_stances = _reference_projection(label)
        row = by_id[item_id]
        presence = row.get("target_presence_logits")
        stance_logits = row.get("stance_logits")
        if (
            not isinstance(presence, Sequence)
            or len(presence) != len(TARGET_CLASSES)
            or not isinstance(stance_logits, Sequence)
            or len(stance_logits) != len(ANALYTIC_TARGET_CLASSES)
        ):
            raise AcquisitionTrainingContractError("target/stance prediction shape drifted")
        predicted: dict[str, str] = {}
        for target_index, target in enumerate(ANALYTIC_TARGET_CLASSES):
            raw_presence = float(presence[target_index])
            raw_stance = stance_logits[target_index]
            if (
                not math.isfinite(raw_presence)
                or not isinstance(raw_stance, Sequence)
                or len(raw_stance) != len(STANCE_CLASSES_B4)
                or any(not math.isfinite(float(value)) for value in raw_stance)
            ):
                invalid += 1
                continue
            if material and _sigmoid(raw_presence) >= 0.5:
                probabilities = _softmax([float(value) for value in raw_stance])
                predicted[target] = STANCE_CLASSES_B4[max(range(4), key=probabilities.__getitem__)]
        for target in ANALYTIC_TARGET_CLASSES:
            for stance in STANCE_CLASSES_B4:
                tuple_pairs.append(
                    (
                        target_stances.get(target) == stance,
                        predicted.get(target) == stance,
                    )
                )
    score = _binary_f1(tuple_pairs)
    return {
        "selected_epoch": selected_epoch,
        "evaluation_rows": len(reference),
        "conditional_analytic_tuple_micro_f1": score,
        "checkpoint_score": score,
        "invalid_outputs": invalid,
        "loss_contribution_counts": dict(loss_counts),
    }


def execute_registered_gpu_trial(
    job: Mapping[str, Any], volume_root: Path, attempt_work_root: Path
) -> dict[str, Any]:
    """Execute one fresh L4 attempt using the frozen factorised B4 recipe."""

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - Modal-only path
        raise RuntimeError("acquisition training runtime dependencies are incomplete") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("acquisition training requires the requested L4")
    from reddit_china_stance.modernbert_trainer import (
        build_length_bucket_batches,
        seed_everything,
    )

    manifest, trial = _validate_job(job)
    bindings = manifest["experiment_contract"]["bindings"]
    train_descriptor = bindings["training_frames"][trial["arm"]]
    checkpoint_selection_descriptor = bindings["checkpoint_selection_frame"]
    evaluation_descriptor = bindings["acquisition_evaluation_frame"]
    train_rows, _ = load_private_frame(
        volume_root / train_descriptor["relative_path"],
        train_descriptor,
        expected_frame="training",
    )
    checkpoint_selection_rows, checkpoint_reference = load_private_frame(
        volume_root / checkpoint_selection_descriptor["relative_path"],
        checkpoint_selection_descriptor,
        expected_frame="development",
    )
    evaluation_rows, _ = load_private_frame(
        volume_root / evaluation_descriptor["relative_path"],
        evaluation_descriptor,
        expected_frame="acquisition_evaluation",
    )
    train_threads = {row["thread_id"] for row in train_rows}
    checkpoint_threads = {row["thread_id"] for row in checkpoint_selection_rows}
    evaluation_threads = {row["thread_id"] for row in evaluation_rows}
    if (
        train_threads & checkpoint_threads
        or train_threads & evaluation_threads
        or checkpoint_threads & evaluation_threads
    ):
        raise AcquisitionTrainingContractError("private training/evaluation threads overlap")
    component = trial["component"]
    seed = trial["optimiser_seed"]
    optimisation = build_optimisation_config(trial)
    seed_everything(seed)
    tokenizer = load_pinned_tokenizer()

    def encode_training(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        for row in rows:
            feature = tokenise_factorised_record(
                tokenizer,
                item_id=row["item_id"],
                row=row,
                label=json.loads(row["label_json"]),
            )
            if component != "relevance" and not feature["target_presence_mask"][0]:
                continue
            encoded.append(feature)
        if not encoded:
            raise AcquisitionTrainingContractError("component training frame is empty")
        return encoded

    train_encoded = encode_training(train_rows)
    checkpoint_selection_encoded = [
        tokenise_factorised_record(
            tokenizer,
            item_id=row["item_id"],
            row=row,
            label=json.loads(row["label_json"]),
        )
        for row in checkpoint_selection_rows
    ]
    evaluation_encoded = [
        tokenise_factorised_record(
            tokenizer,
            item_id=row["item_id"],
            row=row,
            label=json.loads(row["label_json"]),
        )
        for row in evaluation_rows
    ]
    collator = FactorisedDynamicPaddingCollator(tokenizer, component=component)
    checkpoint_selection_loader = DataLoader(
        checkpoint_selection_encoded,
        batch_sampler=build_length_bucket_batches(
            [len(row["input_ids"]) for row in checkpoint_selection_encoded],
            batch_size=optimisation.per_device_batch_size,
            seed=0,
            epoch=0,
        ),
        collate_fn=collator,
        num_workers=0,
    )
    model = create_component_model(component=component, config=optimisation).to("cuda")
    optimizer = create_adamw(model, optimisation)
    max_epochs = trial["config"]["max_epochs"]
    updates_per_epoch = math.ceil(
        math.ceil(len(train_encoded) / optimisation.per_device_batch_size)
        / optimisation.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * optimisation.warmup_ratio),
        num_training_steps=total_updates,
    )
    train_lengths = [len(row["input_ids"]) for row in train_encoded]
    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    checkpoint = attempt_work_root / "checkpoint.pt"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        train_loader = DataLoader(
            train_encoded,
            batch_sampler=build_length_bucket_batches(
                train_lengths,
                batch_size=optimisation.per_device_batch_size,
                seed=seed,
                epoch=epoch,
            ),
            collate_fn=collator,
            num_workers=0,
        )
        trained = train_epoch(
            model,
            train_loader,
            optimizer,
            config=optimisation,
            device="cuda",
            scheduler=scheduler,
        )
        checkpoint_rows = collect_development_logits(
            model,
            checkpoint_selection_loader,
            component=component,
            device="cuda",
            use_bf16=True,
        )
        metrics = _score_checkpoint(
            component=component,
            reference=checkpoint_reference,
            rows=checkpoint_rows,
            selected_epoch=epoch,
            loss_counts=trial["loss_contribution_counts"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_mean_loss": round(float(trained["mean_loss"]), 8),
                "checkpoint_score": round(float(metrics["checkpoint_score"]), 8),
            }
        )
        if float(metrics["checkpoint_score"]) > best_score:
            best_score = float(metrics["checkpoint_score"])
            best_metrics = metrics
            temporary = checkpoint.with_suffix(".pt.new")
            if temporary.exists():
                temporary.unlink()
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "modernbert-acquisition-checkpoint-v1",
                    "experiment_run_id": manifest["experiment_run_id"],
                    "phase_run_id": manifest["phase_run_id"],
                    "trial_id": trial["trial_id"],
                    "arm": trial["arm"],
                    "component": component,
                    "optimiser_seed": seed,
                    "config_sha256": trial["config"]["config_sha256"],
                    "training_frame_sha256": trial["training_frame_sha256"],
                    "checkpoint_selection_frame_sha256": trial[
                        "checkpoint_selection_frame_sha256"
                    ],
                    "acquisition_evaluation_frame_sha256": trial[
                        "acquisition_evaluation_frame_sha256"
                    ],
                    "source_bundle_sha256": bindings["source_bundle_sha256"],
                    "dependency_lock_sha256": bindings["dependency_lock_sha256"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "selected_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
    if best_metrics is None:
        raise RuntimeError("acquisition trial failed to select a checkpoint")
    checkpoint_payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    evaluation_loader = DataLoader(
        evaluation_encoded,
        batch_sampler=build_length_bucket_batches(
            [len(row["input_ids"]) for row in evaluation_encoded],
            batch_size=optimisation.per_device_batch_size,
            seed=0,
            epoch=0,
        ),
        collate_fn=collator,
        num_workers=0,
    )
    evaluation_prediction_rows = collect_development_logits(
        model,
        evaluation_loader,
        component=component,
        device="cuda",
        use_bf16=True,
    )
    if (
        len(evaluation_prediction_rows) != EVALUATION_ROWS
        or {row["item_id"] for row in evaluation_prediction_rows}
        != {row["item_id"] for row in evaluation_rows}
    ):
        raise AcquisitionTrainingContractError(
            "post-selection acquisition-evaluation predictions do not conserve rows"
        )
    aggregate_metrics = {
        **best_metrics,
        "checkpoint_selection_rows": len(checkpoint_selection_rows),
        "acquisition_evaluation_rows": len(evaluation_rows),
        "acquisition_evaluation_used_for_checkpoint_selection": False,
    }
    elapsed = time.monotonic() - started
    if elapsed > trial["max_gpu_seconds"]:
        raise AcquisitionTrainingContractError("trial exceeded its registered GPU seconds")
    return {
        "checkpoint_path": str(checkpoint),
        "private_prediction_rows": evaluation_prediction_rows,
        "aggregate_metrics": aggregate_metrics,
        "epoch_history": history,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": elapsed,
        "gpu_seconds": elapsed,
    }


def _trial_roots(
    *, volume_root: Path, manifest: Mapping[str, Any], trial: Mapping[str, Any]
) -> tuple[Path, Path]:
    phase = (
        volume_root
        / NAMESPACE
        / f"run={manifest['experiment_run_id']}"
        / "phase=paired-acquisition-comparison"
    )
    relative = (
        Path(f"arm={trial['arm']}")
        / f"component={trial['component']}"
        / f"trial={trial['trial_id']}"
    )
    return phase / relative, phase / ".attempts" / relative


def _attempt_binding(
    manifest: Mapping[str, Any], trial: Mapping[str, Any], attempt_id: str
) -> dict[str, Any]:
    if not (
        len(attempt_id) == 32
        and all(character in "0123456789abcdef" for character in attempt_id)
    ):
        raise ValueError("attempt ID must be 32 lowercase hex characters")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ATTEMPT_KIND,
        "attempt_id": attempt_id,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "run_manifest_sha256": canonical_sha256(manifest),
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": canonical_sha256(trial),
        "arm": trial["arm"],
        "component": trial["component"],
        "locked_test_rows_accessed": 0,
    }


def _inspect_attempts(
    root: Path, manifest: Mapping[str, Any], trial: Mapping[str, Any]
) -> int:
    if not root.exists():
        return 0
    if not root.is_dir():
        raise AcquisitionTrainingContractError("attempt root is not a directory")
    count = 0
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith("attempt="):
            raise AcquisitionTrainingContractError("attempt inventory drifted")
        attempt_id = path.name.removeprefix("attempt=")
        marker = path / "attempt.json"
        if marker.exists() and _read_json(marker, where="attempt") != _attempt_binding(
            manifest, trial, attempt_id
        ):
            raise AcquisitionTrainingContractError("attempt binding drifted")
        count += 1
    return count


def _build_private_predictions(
    manifest: Mapping[str, Any], trial: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    clean_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("private predictions contain invalid or duplicate item_id")
        seen.add(item_id)
        if trial["component"] == "relevance":
            if set(row) != {"item_id", "relevance_logit"}:
                raise ValueError("relevance prediction schema drifted")
            clean_rows.append(
                {"item_id": item_id, "relevance_logit": float(row["relevance_logit"])}
            )
        else:
            if set(row) != {"item_id", "target_presence_logits", "stance_logits"}:
                raise ValueError("B4 prediction schema drifted")
            if (
                not isinstance(row["target_presence_logits"], Sequence)
                or len(row["target_presence_logits"]) != len(TARGET_CLASSES)
                or not isinstance(row["stance_logits"], Sequence)
                or len(row["stance_logits"]) != len(ANALYTIC_TARGET_CLASSES)
                or any(
                    not isinstance(values, Sequence)
                    or len(values) != len(STANCE_CLASSES_B4)
                    for values in row["stance_logits"]
                )
            ):
                raise ValueError("B4 prediction shape drifted")
            numeric = [
                *row["target_presence_logits"],
                *(value for values in row["stance_logits"] for value in values),
            ]
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in numeric
            ):
                raise ValueError("B4 prediction contains non-finite logits")
            clean_rows.append(
                {
                    "item_id": item_id,
                    "target_presence_logits": [
                        float(value) for value in row["target_presence_logits"]
                    ],
                    "stance_logits": [
                        [float(value) for value in values] for values in row["stance_logits"]
                    ],
                }
            )
    if len(clean_rows) != EVALUATION_ROWS:
        raise AcquisitionTrainingContractError(
            "private acquisition-evaluation predictions must contain exactly 600 rows"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": PRIVATE_PREDICTIONS_KIND,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "trial_id": trial["trial_id"],
        "arm": trial["arm"],
        "component": trial["component"],
        "optimiser_seed": trial["optimiser_seed"],
        "acquisition_evaluation_frame_sha256": trial[
            "acquisition_evaluation_frame_sha256"
        ],
        "rows": clean_rows,
        "locked_test_rows_accessed": 0,
    }
    return payload


def _build_trial_receipt(
    manifest: Mapping[str, Any],
    trial: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    wall_seconds: float,
    gpu_seconds: float,
) -> dict[str, Any]:
    rate = Decimal(
        manifest["experiment_contract"]["compute"]["rate_card_usd_per_gpu_second"][GPU_TYPE]
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRIAL_RECEIPT_KIND,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "run_manifest_sha256": canonical_sha256(manifest),
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": canonical_sha256(trial),
        "arm": trial["arm"],
        "component": trial["component"],
        "optimiser_seed": trial["optimiser_seed"],
        "training_frame_sha256": trial["training_frame_sha256"],
        "checkpoint_selection_frame_sha256": trial[
            "checkpoint_selection_frame_sha256"
        ],
        "acquisition_evaluation_frame_sha256": trial[
            "acquisition_evaluation_frame_sha256"
        ],
        "loss_contribution_counts": trial["loss_contribution_counts"],
        "artifacts": {key: dict(value) for key, value in sorted(artifacts.items())},
        "aggregate_metrics": dict(metrics),
        "wall_seconds": round(float(wall_seconds), 6),
        "gpu_seconds": round(float(gpu_seconds), 6),
        "estimated_cost_usd": format(rate * Decimal(str(gpu_seconds)), ".6f"),
        "invalid_outputs": int(metrics.get("invalid_outputs", -1)),
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": canonical_sha256(body)}
    assert_metadata_only(receipt, where="acquisition trial receipt")
    return receipt


def validate_trial_artifacts(
    trial_root: Path,
    *,
    manifest: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    clean_manifest = validate_run_manifest(manifest)
    clean_trial = validate_trial_spec(trial, experiment=clean_manifest["experiment_contract"])
    receipt = _read_json(trial_root / "receipt.json", where="acquisition trial receipt")
    expected_keys = {
        "schema_version", "kind", "experiment_run_id", "phase_run_id",
        "run_manifest_sha256", "trial_id", "trial_spec_sha256", "arm", "component",
        "optimiser_seed", "training_frame_sha256",
        "checkpoint_selection_frame_sha256", "acquisition_evaluation_frame_sha256",
        "loss_contribution_counts", "artifacts", "aggregate_metrics", "wall_seconds",
        "gpu_seconds", "estimated_cost_usd", "invalid_outputs",
        "locked_test_rows_accessed", "receipt_id",
    }
    if set(receipt) != expected_keys or receipt.get("kind") != TRIAL_RECEIPT_KIND:
        raise ValueError("acquisition trial receipt schema drifted")
    if (
        receipt["experiment_run_id"] != clean_manifest["experiment_run_id"]
        or receipt["phase_run_id"] != clean_manifest["phase_run_id"]
        or receipt["run_manifest_sha256"] != canonical_sha256(clean_manifest)
        or receipt["trial_id"] != clean_trial["trial_id"]
        or receipt["trial_spec_sha256"] != canonical_sha256(clean_trial)
        or receipt["arm"] != clean_trial["arm"]
        or receipt["component"] != clean_trial["component"]
        or receipt["optimiser_seed"] != clean_trial["optimiser_seed"]
        or receipt["training_frame_sha256"] != clean_trial["training_frame_sha256"]
        or receipt["checkpoint_selection_frame_sha256"]
        != clean_trial["checkpoint_selection_frame_sha256"]
        or receipt["acquisition_evaluation_frame_sha256"]
        != clean_trial["acquisition_evaluation_frame_sha256"]
        or receipt["loss_contribution_counts"] != clean_trial["loss_contribution_counts"]
        or receipt["invalid_outputs"] != 0
        or receipt["locked_test_rows_accessed"] != 0
        or float(receipt["gpu_seconds"]) > clean_trial["max_gpu_seconds"]
    ):
        raise AcquisitionTrainingContractError("acquisition trial receipt binding drifted")
    body = {key: receipt[key] for key in expected_keys - {"receipt_id"}}
    if receipt["receipt_id"] != canonical_sha256(body):
        raise AcquisitionTrainingContractError("acquisition trial receipt digest drifted")
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "checkpoint", "metrics", "private_evaluation_predictions"
    }:
        raise ValueError("acquisition trial artifact inventory drifted")
    paths = {
        name: _validate_descriptor(trial_root, descriptor, where=name)
        for name, descriptor in artifacts.items()
    }
    predictions = _read_json(paths["private_evaluation_predictions"], where="private predictions")
    if predictions != _build_private_predictions(
        clean_manifest, clean_trial, predictions.get("rows", [])
    ):
        raise AcquisitionTrainingContractError("private prediction binding drifted")
    metrics = _read_json(paths["metrics"], where="trial metrics")
    if (
        metrics.get("kind") != TRIAL_METRICS_KIND
        or metrics.get("aggregate_metrics") != receipt["aggregate_metrics"]
        or metrics.get("loss_contribution_counts") != clean_trial["loss_contribution_counts"]
        or receipt["aggregate_metrics"].get("checkpoint_selection_rows")
        != EVALUATION_ROWS
        or receipt["aggregate_metrics"].get("acquisition_evaluation_rows")
        != EVALUATION_ROWS
        or receipt["aggregate_metrics"].get(
            "acquisition_evaluation_used_for_checkpoint_selection"
        )
        is not False
    ):
        raise AcquisitionTrainingContractError("trial metrics and receipt disagree")
    observed = {
        str(path.relative_to(trial_root)) for path in trial_root.rglob("*") if path.is_file()
    }
    expected_files = {"receipt.json", *(value["relative_path"] for value in artifacts.values())}
    if observed != expected_files:
        raise AcquisitionTrainingContractError("final trial inventory drifted")
    return receipt


def run_training_trial(
    *, job: Mapping[str, Any], volume_root: Path, trial_executor: Any
) -> dict[str, Any]:
    manifest, trial = _validate_job(job)
    final, attempts = _trial_roots(volume_root=volume_root, manifest=manifest, trial=trial)
    attempt_count = _inspect_attempts(attempts, manifest, trial)
    if final.exists():
        receipt = validate_trial_artifacts(final, manifest=manifest, trial=trial)
        return {"status": "already_complete", "receipt_id": receipt["receipt_id"]}
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    attempt_root = attempts / f"attempt={attempt_id}"
    attempt_root.mkdir()
    work = attempt_root / "work"
    publication = attempt_root / "publication"
    phase = "initialisation"
    try:
        _write_json_atomic(
            attempt_root / "attempt.json",
            _attempt_binding(manifest, trial, attempt_id),
        )
        work.mkdir()
        phase = "training"
        result = trial_executor(job, volume_root, work)
        checkpoint = Path(result["checkpoint_path"])
        if not checkpoint.is_file() or checkpoint.resolve().parent != work.resolve():
            raise RuntimeError("checkpoint must be an attempt-local file")
        phase = "publication"
        publication.mkdir()
        published_checkpoint = publication / "checkpoint.pt"
        os.replace(checkpoint, published_checkpoint)
        predictions = _build_private_predictions(
            manifest, trial, result["private_prediction_rows"]
        )
        prediction_path = publication / "evaluation-predictions.json"
        _write_json_atomic(prediction_path, predictions)
        aggregate_metrics = dict(result["aggregate_metrics"])
        if aggregate_metrics.get("invalid_outputs") != 0:
            raise AcquisitionTrainingContractError("trial has invalid outputs")
        metrics_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": TRIAL_METRICS_KIND,
            "arm": trial["arm"],
            "component": trial["component"],
            "aggregate_metrics": aggregate_metrics,
            "loss_contribution_counts": trial["loss_contribution_counts"],
            "epoch_history": list(result.get("epoch_history", [])),
            "peak_gpu_bytes": int(result.get("peak_gpu_bytes", 0)),
            "locked_test_rows_accessed": 0,
        }
        assert_metadata_only(metrics_payload, where="acquisition trial metrics")
        metrics_path = publication / "metrics.json"
        _write_json_atomic(metrics_path, metrics_payload)
        artifacts = {
            "checkpoint": _descriptor(published_checkpoint, root=publication),
            "metrics": _descriptor(metrics_path, root=publication),
            "private_evaluation_predictions": _descriptor(prediction_path, root=publication),
        }
        receipt = _build_trial_receipt(
            manifest,
            trial,
            artifacts=artifacts,
            metrics=aggregate_metrics,
            wall_seconds=float(result["wall_seconds"]),
            gpu_seconds=float(result["gpu_seconds"]),
        )
        _write_json_atomic(publication / "receipt.json", receipt)
        validate_trial_artifacts(publication, manifest=manifest, trial=trial)
        phase = "promotion"
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(publication, final)
        validate_trial_artifacts(final, manifest=manifest, trial=trial)
        _write_json_atomic(
            attempt_root / "promoted.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "modernbert-acquisition-attempt-outcome-v1",
                "status": "promoted",
                "locked_test_rows_accessed": 0,
            },
        )
        return {
            "status": "complete",
            "receipt_id": receipt["receipt_id"],
            "estimated_cost_usd": receipt["estimated_cost_usd"],
            "prior_attempts": attempt_count,
        }
    except Exception as exc:
        with suppress(Exception):
            _write_json_atomic(
                attempt_root / "failed.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "modernbert-acquisition-attempt-outcome-v1",
                    "status": "failed",
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "locked_test_rows_accessed": 0,
                },
            )
        raise


def inspect_trial_output(
    *, manifest: Mapping[str, Any], trial_id: str, volume_root: Path
) -> dict[str, Any]:
    clean = validate_run_manifest(manifest)
    by_id = {trial["trial_id"]: trial for trial in clean["trials"]}
    if trial_id not in by_id:
        raise ValueError("trial_id is not registered")
    trial = by_id[trial_id]
    final, attempts = _trial_roots(volume_root=volume_root, manifest=clean, trial=trial)
    attempt_count = _inspect_attempts(attempts, clean, trial)
    if not final.exists():
        return {
            "trial_id": trial_id,
            "arm": trial["arm"],
            "component": trial["component"],
            "status": "incomplete" if attempt_count else "missing",
            "attempt_count": attempt_count,
        }
    receipt = validate_trial_artifacts(final, manifest=clean, trial=trial)
    result = {
        "trial_id": trial_id,
        "arm": trial["arm"],
        "component": trial["component"],
        "status": "complete",
        "receipt_id": receipt["receipt_id"],
        "estimated_cost_usd": receipt["estimated_cost_usd"],
        "attempt_count": attempt_count,
    }
    assert_metadata_only(result, where="acquisition trial inspection")
    return result


def aggregate_trial_inspections(
    *, manifest: Mapping[str, Any], inspections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    clean = validate_run_manifest(manifest)
    expected = {trial["trial_id"] for trial in clean["trials"]}
    by_id = {inspection.get("trial_id"): inspection for inspection in inspections}
    if set(by_id) != expected or len(by_id) != len(inspections):
        raise ValueError("inspection set does not exactly cover registered trials")
    complete = [value for value in by_id.values() if value.get("status") == "complete"]
    incomplete = [value for value in by_id.values() if value.get("status") == "incomplete"]
    missing = [value for value in by_id.values() if value.get("status") == "missing"]
    result = {
        "status": "complete" if len(complete) == EXPECTED_TRIALS else "incomplete",
        "expected_trials": EXPECTED_TRIALS,
        "complete_trials": len(complete),
        "incomplete_trials": len(incomplete),
        "missing_trials": len(missing),
        "estimated_cost_usd": f"{sum(float(row['estimated_cost_usd']) for row in complete):.6f}",
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition run inspection")
    return result


def _load_trial_predictions(
    *, manifest: Mapping[str, Any], trial: Mapping[str, Any], volume_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    final, _ = _trial_roots(volume_root=volume_root, manifest=manifest, trial=trial)
    receipt = validate_trial_artifacts(final, manifest=manifest, trial=trial)
    descriptor = receipt["artifacts"]["private_evaluation_predictions"]
    payload = _read_json(final / descriptor["relative_path"], where="private predictions")
    return receipt, payload


def _decode_paired_predictions(
    *,
    evaluation_rows: Sequence[Mapping[str, Any]],
    relevance: Mapping[str, Any],
    target_stance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    relevance_by_id = {row["item_id"]: row for row in relevance["rows"]}
    target_by_id = {row["item_id"]: row for row in target_stance["rows"]}
    expected = {row["item_id"] for row in evaluation_rows}
    if set(relevance_by_id) != expected or set(target_by_id) != expected:
        raise AcquisitionTrainingContractError("paired predictions do not conserve evaluation IDs")
    decoded: list[dict[str, Any]] = []
    for row in evaluation_rows:
        item_id = row["item_id"]
        material = _sigmoid(float(relevance_by_id[item_id]["relevance_logit"])) >= 0.5
        target_stances: dict[str, str] = {}
        if material:
            target_row = target_by_id[item_id]
            for index, target in enumerate(ANALYTIC_TARGET_CLASSES):
                if _sigmoid(float(target_row["target_presence_logits"][index])) >= 0.5:
                    probabilities = _softmax(
                        [float(value) for value in target_row["stance_logits"][index]]
                    )
                    target_stances[target] = STANCE_CLASSES_B4[
                        max(range(4), key=probabilities.__getitem__)
                    ]
        decoded.append(
            {
                "thread_id": row["thread_id"],
                "material": material,
                "target_stances": target_stances,
            }
        )
    return decoded


def closeout_comparison(
    *, manifest: Mapping[str, Any], volume_root: Path, policy_path: Path
) -> dict[str, Any]:
    """Publish the one-time metadata-only random-vs-active gate."""

    clean = validate_run_manifest(manifest)
    _, gate_policy = load_acquisition_policies(policy_path)
    config = clean["experiment_contract"]["bindings"]["acquisition_config"]
    if (
        file_sha256(policy_path) != config["sha256"]
        or gate_policy.policy_file_sha256 != config["sha256"]
        or asdict(gate_policy)
        != clean["experiment_contract"]["registered_design"]["gate_policy"]
    ):
        raise AcquisitionTrainingContractError("closeout policy/config binding drifted")
    evaluation_descriptor = clean["experiment_contract"]["bindings"][
        "acquisition_evaluation_frame"
    ]
    evaluation_rows, reference_labels = load_private_frame(
        volume_root / evaluation_descriptor["relative_path"],
        evaluation_descriptor,
        expected_frame="acquisition_evaluation",
    )
    reference_rows = []
    for row in evaluation_rows:
        material, target_stances = _reference_projection(reference_labels[row["item_id"]])
        reference_rows.append(
            {
                "thread_id": row["thread_id"],
                "quality_tier": row["quality_tier"],
                "material": material,
                "target_stances": target_stances,
            }
        )
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    receipt_ids: list[str] = []
    total_cost = 0.0
    for trial in clean["trials"]:
        receipt, payload = _load_trial_predictions(
            manifest=clean, trial=trial, volume_root=volume_root
        )
        by_key[(trial["arm"], trial["component"], trial["optimiser_seed"])] = payload
        receipt_ids.append(receipt["receipt_id"])
        total_cost += float(receipt["estimated_cost_usd"])
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {} for arm in ARMS
    }
    for arm in ARMS:
        for seed in SEEDS:
            predictions[arm][str(seed)] = _decode_paired_predictions(
                evaluation_rows=evaluation_rows,
                relevance=by_key[(arm, "relevance", seed)],
                target_stance=by_key[(arm, "target_stance_b4", seed)],
            )
    gate = evaluate_acquisition_gate(
        reference_rows,
        predictions,
        rare_cells=clean["experiment_contract"]["bindings"]["rare_cells"],
        policy=gate_policy,
    )
    closeout = {
        **gate,
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "run_manifest_sha256": canonical_sha256(clean),
        "trial_receipt_set_sha256": canonical_sha256(sorted(receipt_ids)),
        "loss_contribution_counts_by_arm": clean["experiment_contract"]["bindings"][
            "loss_contribution_counts_by_arm"
        ],
        "estimated_training_cost_usd": f"{total_cost:.6f}",
        "evidence_scope": "adaptive_model_assisted_acquisition_evaluation_only",
        "calibration_selected": False,
        "locked_test_rows_accessed": 0,
        "corpus_inference_authorised": False,
    }
    closeout["closeout_id"] = canonical_sha256(closeout)
    assert_metadata_only(closeout, where="acquisition training closeout")
    root = (
        volume_root
        / NAMESPACE
        / f"run={clean['experiment_run_id']}"
        / "closeout"
    )
    _write_json_atomic(root / "acquisition-gate.json", closeout)
    return closeout
