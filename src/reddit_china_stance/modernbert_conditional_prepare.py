"""Prepare the immutable first rung of the conditional ModernBERT experiment.

This module is intentionally local-only and metadata-only.  It reads the
already frozen split and confirmatory manifests plus three downloaded public
trial receipts, but it never reads teacher rows, development rows, predictions,
or Reddit text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from reddit_china_stance.modernbert_conditional_experiment import (
    build_run_manifest,
    freeze_experiment_contract,
    freeze_trial_spec,
    frozen_trial_configs,
    validate_run_manifest,
)
from reddit_china_stance.modernbert_conditional_model import (
    TARGET_STATE_LABELS,
    encode_conditional_semantic_label,
)
from reddit_china_stance.modernbert_model import RELEVANCE_LABELS, TARGET_LABELS
from reddit_china_stance.modernbert_training import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    PROXY_JSON_SHA256,
    REFERENCE_ROWS_JSON_SHA256,
    REGISTERED_LADDERS,
    TEACHER_GENERATION_RUN_ID,
    canonical_sha256,
    file_sha256,
)
from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
SOURCE_BUNDLE_KIND = "modernbert-conditional-source-bundle-v1"
MATCHED_BASELINE_KIND = "modernbert-conditional-matched-baseline-v1"
TARGET_STATE_SUPPORT_KIND = "modernbert-conditional-target-state-support-v1"

CONFIRMATORY_MANIFEST_RELATIVE = Path(
    "data/private-modernbert-v1/confirmatory/run-manifest.json"
)
SPLIT_MANIFEST_DIRECTORY_RELATIVE = Path("data/private-modernbert-v1/splits")
BASELINE_RECEIPTS_RELATIVE = Path(
    "data/private-modernbert-conditional-v1/baseline-receipts"
)
OUTPUT_DIRECTORY_RELATIVE = Path(
    "data/private-modernbert-conditional-v1/sweep-rung2-v5"
)
BASELINE_MANIFEST_OUTPUT_RELATIVE = OUTPUT_DIRECTORY_RELATIVE / "baseline-manifest.json"
SOURCE_DIRECTORY_RELATIVE = Path("src/reddit_china_stance")
DEPENDENCY_LOCK_RELATIVE = Path("uv.lock")
TEACHER_PARQUET_RELATIVE = Path(
    "data/private-hf-sol-teacher-10k-v1/data/train-00000-of-00001.parquet"
)
DEVELOPMENT_PROXY_RELATIVE = Path(
    "data/private-modernbert-v1/inputs/development-proxy.parquet"
)

GPU_TYPE = "L4"
GPU_RATE_USD_PER_SECOND = Decimal("0.000222")
PLANNED_UPPER_COST_USD = Decimal("50")
APPROVED_HARD_COST_USD = Decimal("200")
TRIAL_COUNT = 8
ASHA_RUNG_EPOCHS = 2
LABEL_BUDGET = 5_000
LADDER_SEED = 101
OPTIMISER_SEED = 7
MAX_GPU_SECONDS_PER_TRIAL = 5_400
FULL_FAMILY_TRIAL_COUNT = 8 + 4 + 2 + 6
FULL_FAMILY_RESERVED_COST_USD = (
    GPU_RATE_USD_PER_SECOND * MAX_GPU_SECONDS_PER_TRIAL * FULL_FAMILY_TRIAL_COUNT
)

_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "experiment_run_id",
    "trial_id",
    "trial_spec_sha256",
    "bindings",
    "artifacts",
    "aggregate_metrics",
    "compute",
    "receipt_id",
}


class ConditionalPreparationError(RuntimeError):
    """Raised when preparation inputs or immutable outputs drift."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _read_json_object(path: Path, *, where: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConditionalPreparationError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ConditionalPreparationError(f"{where} must contain a JSON object")
    return value


def _require_sha256(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ConditionalPreparationError(f"{where} must be a lowercase SHA-256")
    return value


def _source_bundle(repo_root: Path) -> tuple[dict[str, Any], bytes]:
    source_root = repo_root / SOURCE_DIRECTORY_RELATIVE
    files = sorted(path for path in source_root.glob("*.py") if path.is_file())
    if not files:
        raise ConditionalPreparationError("conditional source bundle has no Python files")
    file_hashes = {
        path.relative_to(repo_root).as_posix(): file_sha256(path) for path in files
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_BUNDLE_KIND,
        "source_glob": "src/reddit_china_stance/*.py",
        "file_count": len(file_hashes),
        "files": file_hashes,
        "code_sha256": canonical_sha256(file_hashes),
    }
    bundle = {**body, "source_bundle_id": canonical_sha256(body)}
    assert_metadata_only(bundle, where="conditional source bundle")
    return bundle, _json_bytes(bundle)


def _confirmatory_inputs(
    repo_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = repo_root / CONFIRMATORY_MANIFEST_RELATIVE
    manifest = _read_json_object(manifest_path, where="confirmatory manifest")
    contract = manifest.get("experiment_contract")
    confirmatory = manifest.get("confirmatory")
    trials = manifest.get("trials")
    if not (
        isinstance(contract, Mapping)
        and isinstance(confirmatory, Mapping)
        and isinstance(trials, list)
    ):
        raise ConditionalPreparationError("confirmatory manifest is incomplete")
    bindings = contract.get("bindings")
    split_binding = confirmatory.get("split_binding")
    if not isinstance(bindings, Mapping) or not isinstance(split_binding, Mapping):
        raise ConditionalPreparationError("confirmatory input bindings are incomplete")
    if confirmatory.get("locked_test") != {
        "authorised": False,
        "rows_accessed": 0,
        "predictions_authorised": False,
    }:
        raise ConditionalPreparationError("confirmatory manifest has locked-test access")
    if (
        bindings.get("dataset_id") != DATASET_ID
        or bindings.get("dataset_revision") != DATASET_REVISION
        or bindings.get("dataset_parquet_sha256") != DATASET_SHA256
    ):
        raise ConditionalPreparationError("confirmatory teacher binding drifted")
    development_binding = _require_sha256(
        bindings.get("development_proxy_sha256"),
        where="confirmatory development binding",
    )
    split_file_sha256 = _require_sha256(
        split_binding.get("split_manifest_file_sha256"),
        where="confirmatory split file binding",
    )
    if bindings.get("split_manifest_sha256") != split_file_sha256:
        raise ConditionalPreparationError("confirmatory split bindings disagree")
    split_id = _require_sha256(
        split_binding.get("split_manifest_id"), where="confirmatory split manifest ID"
    )
    split_path = (
        repo_root
        / SPLIT_MANIFEST_DIRECTORY_RELATIVE
        / f"manifest-{split_id}.json"
    )
    if file_sha256(split_path) != split_file_sha256:
        raise ConditionalPreparationError("split manifest file hash drifted")
    split_manifest = _read_json_object(split_path, where="split manifest")
    split_body = {key: value for key, value in split_manifest.items() if key != "manifest_id"}
    if (
        split_manifest.get("manifest_id") != split_id
        or canonical_sha256(split_body) != split_id
        or split_manifest.get("contains_reddit_text") is not False
    ):
        raise ConditionalPreparationError("split manifest identity or privacy binding drifted")
    return (
        manifest,
        split_path,
        {
            "bindings": dict(bindings),
            "development_binding": development_binding,
            "split_file_sha256": split_file_sha256,
            "split_manifest": split_manifest,
            "confirmatory_manifest_sha256": file_sha256(manifest_path),
        },
    )


def _matched_baseline(
    *,
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    baseline_directory: Path,
    target_state_support: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    receipt_paths = sorted(baseline_directory.glob("*.json"))
    if len(receipt_paths) != 3 or any(not path.is_file() for path in receipt_paths):
        raise ConditionalPreparationError(
            "matched baseline requires exactly three JSON receipt files"
        )
    trials = manifest.get("trials")
    if not isinstance(trials, list):
        raise ConditionalPreparationError("confirmatory trial inventory is missing")
    confirmatory = manifest.get("confirmatory")
    design = confirmatory.get("design") if isinstance(confirmatory, Mapping) else None
    raw_conditions = design.get("paired_ladders") if isinstance(design, Mapping) else None
    frozen_conditions = [dict(row) for row in REGISTERED_LADDERS]
    if raw_conditions != frozen_conditions:
        raise ConditionalPreparationError("confirmatory baseline condition design drifted")
    condition_pairs = [
        {
            "ladder_seed": row["ladder_seed"],
            "optimiser_seed": row["optimiser_seed"],
        }
        for row in frozen_conditions
    ]
    expected_conditions = {
        (row["ladder_seed"], row["optimiser_seed"]) for row in condition_pairs
    }
    expected: dict[str, dict[str, Any]] = {}
    observed_trial_conditions: set[tuple[int, int]] = set()
    registered_config_ids: set[str] = set()
    for raw_trial in trials:
        if not isinstance(raw_trial, Mapping):
            raise ConditionalPreparationError("confirmatory trial must be an object")
        config = raw_trial.get("config")
        if not isinstance(config, Mapping) or config.get("label_budget") != 10_000:
            continue
        registered = config.get("registered_config")
        if not isinstance(registered, Mapping):
            raise ConditionalPreparationError("10k baseline trial lacks registered config")
        registered_body = {
            key: value for key, value in registered.items() if key != "config_sha256"
        }
        config_sha256 = _require_sha256(
            registered.get("config_sha256"), where="baseline config SHA-256"
        )
        if canonical_sha256(registered_body) != config_sha256:
            raise ConditionalPreparationError("baseline registered config digest drifted")
        condition = (config.get("ladder_seed"), config.get("optimiser_seed"))
        if (
            condition not in expected_conditions
            or config.get("seed") != config.get("optimiser_seed")
            or condition in observed_trial_conditions
        ):
            raise ConditionalPreparationError(
                "10k baseline trials do not cover exact frozen conditions"
            )
        trial = dict(raw_trial)
        trial_id = _require_sha256(trial.get("trial_id"), where="baseline trial ID")
        if canonical_sha256({k: v for k, v in trial.items() if k != "trial_id"}) != trial_id:
            raise ConditionalPreparationError("baseline trial content address drifted")
        expected[trial_id] = {
            "trial": trial,
            "config_sha256": config_sha256,
            "ladder_seed": config.get("ladder_seed"),
            "optimiser_seed": config.get("optimiser_seed"),
        }
        observed_trial_conditions.add(condition)
        registered_config_ids.add(config_sha256)
    if (
        len(expected) != 3
        or observed_trial_conditions != expected_conditions
        or len(registered_config_ids) != 1
    ):
        raise ConditionalPreparationError(
            "confirmatory baselines require one shared config across exact frozen conditions"
        )
    registered_config_sha256 = next(iter(registered_config_ids))

    entries: list[dict[str, Any]] = []
    observed: set[str] = set()
    for path in receipt_paths:
        receipt = _read_json_object(path, where=f"baseline receipt {path.name}")
        if set(receipt) != _RECEIPT_KEYS:
            raise ConditionalPreparationError("baseline receipt field schema drifted")
        assert_metadata_only(receipt, where=f"baseline receipt {path.name}")
        receipt_id = _require_sha256(receipt.get("receipt_id"), where="receipt ID")
        if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_id"}) != (
            receipt_id
        ):
            raise ConditionalPreparationError("baseline receipt content address drifted")
        trial_id = _require_sha256(receipt.get("trial_id"), where="receipt trial ID")
        source = expected.get(trial_id)
        if source is None or trial_id in observed:
            raise ConditionalPreparationError("baseline receipts do not match three unique trials")
        trial = source["trial"]
        if (
            receipt.get("status") != "complete"
            or receipt.get("experiment_run_id") != manifest.get("experiment_run_id")
            or receipt.get("trial_spec_sha256") != canonical_sha256(trial)
            or receipt.get("bindings") != binding["bindings"]
        ):
            raise ConditionalPreparationError("baseline receipt binding drifted")
        observed.add(trial_id)
        entries.append(
            {
                "config_sha256": source["config_sha256"],
                "ladder_seed": source["ladder_seed"],
                "optimiser_seed": source["optimiser_seed"],
                "trial_id": trial_id,
                "trial_spec_sha256": receipt["trial_spec_sha256"],
                "receipt_id": receipt_id,
                "receipt_file": path.name,
                "receipt_file_sha256": file_sha256(path),
            }
        )
    if observed != set(expected):
        raise ConditionalPreparationError("matched baseline receipt inventory is incomplete")
    entries.sort(key=lambda row: (row["ladder_seed"], row["optimiser_seed"]))
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": MATCHED_BASELINE_KIND,
        "source_confirmatory_manifest_sha256": binding[
            "confirmatory_manifest_sha256"
        ],
        "source_experiment_run_id": manifest.get("experiment_run_id"),
        "source_phase_run_id": manifest.get("phase_run_id"),
        "label_budget": 10_000,
        "registered_config_sha256": registered_config_sha256,
        "condition_pairs": condition_pairs,
        "target_state_support_receipt_id": target_state_support[
            "support_receipt_id"
        ],
        "target_state_support_receipt_sha256": canonical_sha256(
            target_state_support
        ),
        "target_state_support": dict(target_state_support),
        "receipt_count": 3,
        "entries": entries,
    }
    aggregate = {**body, "matched_baseline_id": canonical_sha256(body)}
    assert_metadata_only(aggregate, where="matched baseline binding")
    return aggregate, _json_bytes(aggregate)


def _subset_binding(split_manifest: Mapping[str, Any]) -> tuple[str, int]:
    ladders = split_manifest.get("ladders")
    ladder = ladders.get(str(LADDER_SEED)) if isinstance(ladders, Mapping) else None
    budgets = ladder.get("budgets") if isinstance(ladder, Mapping) else None
    subset = budgets.get("5k") if isinstance(budgets, Mapping) else None
    if not isinstance(subset, Mapping):
        raise ConditionalPreparationError("split manifest lacks train_5k_101")
    row_count = subset.get("row_count")
    if row_count != LABEL_BUDGET:
        raise ConditionalPreparationError("train_5k_101 row count drifted")
    return canonical_sha256(dict(subset)), row_count


def _load_teacher_label_rows(
    repo_root: Path, *, expected_sha256: str
) -> list[dict[str, Any]]:
    teacher_path = repo_root / TEACHER_PARQUET_RELATIVE
    if file_sha256(teacher_path) != expected_sha256:
        raise ConditionalPreparationError("private teacher Parquet hash drifted")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised in the repository environment
        raise RuntimeError("PyArrow is required for the private support preflight") from exc
    table = pq.read_table(teacher_path, columns=["sample_id", "label_json"])
    if table.num_rows != 10_000 or table.column_names != ["sample_id", "label_json"]:
        raise ConditionalPreparationError("private teacher label projection drifted")
    rows = table.to_pylist()
    if any(
        not isinstance(row.get("sample_id"), str)
        or not row["sample_id"]
        or not isinstance(row.get("label_json"), str)
        for row in rows
    ):
        raise ConditionalPreparationError("private teacher label projection is invalid")
    return rows


def _development_reference_digest(
    repo_root: Path, *, expected_proxy_id: str
) -> dict[str, Any]:
    if expected_proxy_id != DEVELOPMENT_PROXY_ID:
        raise ConditionalPreparationError("development proxy identity drifted")
    path = repo_root / DEVELOPMENT_PROXY_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised in the repository environment
        raise RuntimeError("PyArrow is required for development-reference preparation") from exc
    parquet = pq.ParquetFile(path)
    expected_columns = {
        "source_sample_id",
        "target_text",
        "parent_context",
        "submission_context",
        "split",
        "resolution",
        "label_json",
    }
    if parquet.metadata.num_rows != 452 or set(parquet.schema_arrow.names) != expected_columns:
        raise ConditionalPreparationError("development proxy schema or row count drifted")
    metadata = parquet.schema_arrow.metadata or {}
    expected_metadata = {
        b"proxy_id": expected_proxy_id.encode(),
        b"proxy_json_sha256": PROXY_JSON_SHA256.encode(),
        b"reference_rows_json_sha256": REFERENCE_ROWS_JSON_SHA256.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ConditionalPreparationError("development proxy embedded metadata drifted")
    rows = pq.read_table(
        path,
        columns=["source_sample_id", "label_json"],
        filters=[("split", "=", "development")],
    ).to_pylist()
    if len(rows) != 222:
        raise ConditionalPreparationError("development proxy must have 222 development rows")
    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("source_sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in reference:
            raise ConditionalPreparationError("development reference IDs are invalid or duplicated")
        try:
            label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ConditionalPreparationError(
                "development reference label JSON is invalid"
            ) from exc
        if not isinstance(label, dict):
            raise ConditionalPreparationError("development reference label must be an object")
        encode_conditional_semantic_label(label)
        reference[sample_id] = label
    binding = {
        "development_proxy_id": expected_proxy_id,
        "split": "development",
        "labels": [
            {"source_sample_id": sample_id, "label": reference[sample_id]}
            for sample_id in sorted(reference)
        ],
    }
    return {
        "development_proxy_id": expected_proxy_id,
        "development_proxy_file_sha256": file_sha256(path),
        "development_reference_sha256": canonical_sha256(binding),
        "development_reference_rows": len(reference),
    }


def _target_state_support_receipt(
    *,
    repo_root: Path,
    split_manifest: Mapping[str, Any],
    split_manifest_sha256: str,
    subset_manifest_sha256: str,
    expected_teacher_sha256: str,
    configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = split_manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 10_000:
        raise ConditionalPreparationError("split manifest must contain 10,000 private rows")
    selected_ids: set[str] = set()
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ConditionalPreparationError("split manifest row is invalid")
        sample_id = row.get("sample_id")
        folds = row.get("folds")
        fold = folds.get(str(LADDER_SEED)) if isinstance(folds, Mapping) else None
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen_ids
            or type(fold) is not int
            or not 0 <= fold < 10
        ):
            raise ConditionalPreparationError("split manifest private ID/fold binding drifted")
        seen_ids.add(sample_id)
        if fold in set(range(5)):
            selected_ids.add(sample_id)
    subset = split_manifest["ladders"][str(LADDER_SEED)]["budgets"]["5k"]
    if (
        len(selected_ids) != LABEL_BUDGET
        or canonical_sha256(sorted(selected_ids)) != subset.get("sample_ids_sha256")
        or canonical_sha256(dict(subset)) != subset_manifest_sha256
    ):
        raise ConditionalPreparationError("train_5k_101 private ID selection drifted")

    teacher_rows = _load_teacher_label_rows(
        repo_root, expected_sha256=expected_teacher_sha256
    )
    by_id: dict[str, str] = {}
    for row in teacher_rows:
        sample_id = row["sample_id"]
        if sample_id in by_id:
            raise ConditionalPreparationError("private teacher IDs are not unique")
        by_id[sample_id] = row["label_json"]
    if len(by_id) != 10_000 or not selected_ids <= set(by_id):
        raise ConditionalPreparationError("private teacher/split join is incomplete")

    counts = {
        target: {state: 0 for state in TARGET_STATE_LABELS} for target in TARGET_LABELS
    }
    material_rows = 0
    for sample_id in sorted(selected_ids):
        try:
            label = json.loads(by_id[sample_id])
        except json.JSONDecodeError as exc:
            raise ConditionalPreparationError("private teacher label JSON is invalid") from exc
        encoded = encode_conditional_semantic_label(label)
        if RELEVANCE_LABELS[encoded["relevance_labels"]] != "material":
            continue
        material_rows += 1
        for target, state_index in zip(
            TARGET_LABELS, encoded["target_state_labels"], strict=True
        ):
            counts[target][TARGET_STATE_LABELS[state_index]] += 1
    state_totals = {
        state: sum(counts[target][state] for target in TARGET_LABELS)
        for state in TARGET_STATE_LABELS
    }
    material_slots = material_rows * len(TARGET_LABELS)
    if material_rows == 0 or sum(state_totals.values()) != material_slots:
        raise ConditionalPreparationError("target-state support did not conserve material slots")
    weighted_config_ids = sorted(
        config["config_sha256"]
        for config in configs
        if config.get("target_state_class_weights") != "none"
    )
    if not weighted_config_ids:
        raise ConditionalPreparationError("frozen sweep contains no weighted trial")
    unsupported_weighted_states = [
        state for state in TARGET_STATE_LABELS if state_totals[state] == 0
    ]
    if unsupported_weighted_states:
        raise ConditionalPreparationError(
            "weighted target-state class lacks selected-5k support: "
            + ", ".join(unsupported_weighted_states)
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TARGET_STATE_SUPPORT_KIND,
        "teacher_parquet_sha256": expected_teacher_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "subset_manifest_sha256": subset_manifest_sha256,
        "ladder_seed": LADDER_SEED,
        "label_budget": LABEL_BUDGET,
        "selected_row_count": len(selected_ids),
        "material_row_count": material_rows,
        "material_slot_count": material_slots,
        "targets": list(TARGET_LABELS),
        "states": list(TARGET_STATE_LABELS),
        "counts": counts,
        "state_totals": state_totals,
        "weighted_config_ids": weighted_config_ids,
        "zero_support_weighted_states": unsupported_weighted_states,
        "gate": "pass",
    }
    receipt = {**body, "support_receipt_id": canonical_sha256(body)}
    assert_metadata_only(receipt, where="conditional target-state support receipt")
    return receipt


def _write_immutable_set(output_directory: Path, outputs: Mapping[str, bytes]) -> None:
    expected_names = {
        "baseline-manifest.json",
        "source-bundle.json",
        "matched-baseline.json",
        "run-manifest.json",
    }
    if set(outputs) != expected_names:
        raise AssertionError("conditional preparation output inventory drifted")
    if output_directory.exists() and not output_directory.is_dir():
        raise ConditionalPreparationError("conditional output path is not a directory")
    if output_directory.is_dir():
        unexpected = {
            path.name for path in output_directory.iterdir() if path.name not in expected_names
        }
        if unexpected:
            raise ConditionalPreparationError("conditional output directory is not exact")
        for name, payload in outputs.items():
            path = output_directory / name
            if path.exists() and (not path.is_file() or path.read_bytes() != payload):
                raise ConditionalPreparationError(f"immutable output differs: {name}")
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = output_directory / name
        if path.exists():
            continue
        incomplete = output_directory / f".{name}.incomplete-{os.getpid()}"
        try:
            with incomplete.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(incomplete, path)
        finally:
            incomplete.unlink(missing_ok=True)


def prepare_conditional_rung2(
    repo_root: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Build or dry-run the exact metadata-only conditional rung-2 manifest."""

    root = repo_root.resolve()
    source_bundle, source_bytes = _source_bundle(root)
    manifest, _, binding = _confirmatory_inputs(root)
    assert_metadata_only(manifest, where="source baseline confirmatory manifest")
    baseline_manifest_bytes = (root / CONFIRMATORY_MANIFEST_RELATIVE).read_bytes()
    baseline_manifest_sha256 = hashlib.sha256(baseline_manifest_bytes).hexdigest()
    if baseline_manifest_sha256 != binding["confirmatory_manifest_sha256"]:
        raise ConditionalPreparationError("source baseline manifest changed during preparation")
    baseline_manifest_descriptor = {
        "repo_relative_path": BASELINE_MANIFEST_OUTPUT_RELATIVE.as_posix(),
        "sha256": baseline_manifest_sha256,
        "bytes": len(baseline_manifest_bytes),
    }
    split_sha256 = binding["split_file_sha256"]
    subset_sha256, training_row_count = _subset_binding(binding["split_manifest"])
    configs = frozen_trial_configs()
    target_state_support = _target_state_support_receipt(
        repo_root=root,
        split_manifest=binding["split_manifest"],
        split_manifest_sha256=split_sha256,
        subset_manifest_sha256=subset_sha256,
        expected_teacher_sha256=binding["bindings"]["dataset_parquet_sha256"],
        configs=configs,
    )
    baseline, baseline_bytes = _matched_baseline(
        manifest=manifest,
        binding=binding,
        baseline_directory=root / BASELINE_RECEIPTS_RELATIVE,
        target_state_support=target_state_support,
    )
    development_reference = _development_reference_digest(
        root,
        expected_proxy_id=binding["development_binding"],
    )
    source_bundle_sha256 = hashlib.sha256(source_bytes).hexdigest()
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    dependency_lock_path = root / DEPENDENCY_LOCK_RELATIVE
    dependency_lock_sha256 = file_sha256(dependency_lock_path)
    experiment = freeze_experiment_contract(
        dataset_revision=binding["bindings"]["dataset_revision"],
        dataset_sha256=binding["bindings"]["dataset_parquet_sha256"],
        split_manifest_sha256=split_sha256,
        source_bundle_sha256=source_bundle_sha256,
        code_sha256=source_bundle["code_sha256"],
        dependency_lock_sha256=dependency_lock_sha256,
        development_reference_sha256=development_reference[
            "development_reference_sha256"
        ],
        matched_baseline_receipt_sha256=baseline_sha256,
        teacher_generation_run_id=TEACHER_GENERATION_RUN_ID,
        rate_card_usd_per_gpu_second={GPU_TYPE: str(GPU_RATE_USD_PER_SECOND)},
        hard_cost_cap_usd=str(APPROVED_HARD_COST_USD),
    )
    trials = [
        freeze_trial_spec(
            experiment,
            phase="asha",
            config=config,
            subset_manifest_sha256=subset_sha256,
            label_budget=LABEL_BUDGET,
            training_row_count=training_row_count,
            ladder_seed=LADDER_SEED,
            optimiser_seed=OPTIMISER_SEED,
            target_epochs=ASHA_RUNG_EPOCHS,
            gpu_type=GPU_TYPE,
            max_gpu_seconds=MAX_GPU_SECONDS_PER_TRIAL,
        )
        for config in configs
    ]
    run_manifest = build_run_manifest(
        experiment,
        phase="sweep",
        trials=trials,
        asha_rung_epochs=ASHA_RUNG_EPOCHS,
        baseline_manifest=baseline_manifest_descriptor,
    )
    validate_run_manifest(run_manifest)
    assert_metadata_only(run_manifest, where="conditional rung-2 run manifest")
    manifest_bytes = _json_bytes(run_manifest)
    outputs = {
        "baseline-manifest.json": baseline_manifest_bytes,
        "source-bundle.json": source_bytes,
        "matched-baseline.json": baseline_bytes,
        "run-manifest.json": manifest_bytes,
    }
    if not dry_run:
        _write_immutable_set(root / OUTPUT_DIRECTORY_RELATIVE, outputs)
    total_gpu_seconds = MAX_GPU_SECONDS_PER_TRIAL * TRIAL_COUNT
    planned_cost = GPU_RATE_USD_PER_SECOND * total_gpu_seconds
    if FULL_FAMILY_RESERVED_COST_USD > PLANNED_UPPER_COST_USD:
        raise AssertionError("conditional experiment family exceeds the planned upper cost")
    summary = {
        "status": "dry-run" if dry_run else "prepared",
        "output_directory": OUTPUT_DIRECTORY_RELATIVE.as_posix(),
        "source_file_count": source_bundle["file_count"],
        "baseline_manifest": baseline_manifest_descriptor,
        "code_sha256": source_bundle["code_sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "matched_baseline_receipt_count": baseline["receipt_count"],
        "matched_baseline_sha256": baseline_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "split_manifest_sha256": split_sha256,
        "subset_manifest_sha256": subset_sha256,
        **development_reference,
        "target_state_support_receipt_id": target_state_support[
            "support_receipt_id"
        ],
        "target_state_material_row_count": target_state_support[
            "material_row_count"
        ],
        "target_state_material_slot_count": target_state_support[
            "material_slot_count"
        ],
        "zero_support_weighted_state_count": len(
            target_state_support["zero_support_weighted_states"]
        ),
        "training_row_count": training_row_count,
        "experiment_run_id": experiment["experiment_run_id"],
        "phase_run_id": run_manifest["phase_run_id"],
        "trial_count": len(trials),
        "gpu_type": GPU_TYPE,
        "gpu_rate_usd_per_second": str(GPU_RATE_USD_PER_SECOND),
        "max_gpu_seconds_per_trial": MAX_GPU_SECONDS_PER_TRIAL,
        "reserved_gpu_seconds": total_gpu_seconds,
        "planned_reserved_cost_usd": format(planned_cost, "f"),
        "full_family_trial_count": FULL_FAMILY_TRIAL_COUNT,
        "full_family_reserved_cost_usd": format(
            FULL_FAMILY_RESERVED_COST_USD, "f"
        ),
        "planned_upper_cost_usd": format(PLANNED_UPPER_COST_USD, "f"),
        "approved_hard_cost_usd": format(APPROVED_HARD_COST_USD, "f"),
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(summary, where="conditional preparation summary")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the exact metadata-only conditional ModernBERT rung-2 run."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the installed source checkout).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the metadata-only plan without writing outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = prepare_conditional_rung2(args.repo_root, dry_run=args.dry_run)
    except (ConditionalPreparationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
