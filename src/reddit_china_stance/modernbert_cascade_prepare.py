"""Prepare the immutable six-trial separate-model cascade experiment.

Preparation is metadata-only.  It validates the existing frozen confirmatory and split
manifests, derives the exact 10k training profile from their aggregate support counts, and never
opens teacher rows, development rows, predictions, Reddit text, or the locked test.
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

from reddit_china_stance.modernbert_cascade_experiment import (
    BASELINE_CONFIG_SHA256,
    COMPONENTS,
    DEVELOPMENT_ROWS,
    EXCLUDED_NON_MATERIAL_TARGET_SLOTS,
    MATERIAL_TRAINING_ROWS,
    PAIRED_CONDITIONS,
    RELEVANCE_TRAINING_ROWS,
    TARGET_STATE_COUNTS,
    TARGET_TRAINING_ROWS,
    TARGETS,
    build_run_manifest,
    freeze_experiment_contract,
    freeze_trial_spec,
    validate_run_manifest,
)
from reddit_china_stance.modernbert_training import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    MODEL_ID,
    MODEL_REVISION,
    TEACHER_GENERATION_RUN_ID,
    canonical_sha256,
    file_sha256,
)
from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
SOURCE_BUNDLE_KIND = "modernbert-cascade-source-bundle-v1"
DATASET_PROFILE_KIND = "modernbert-cascade-dataset-profile-v1"

CONFIRMATORY_MANIFEST_RELATIVE = Path("data/private-modernbert-v1/confirmatory/run-manifest.json")
SPLIT_MANIFEST_DIRECTORY_RELATIVE = Path("data/private-modernbert-v1/splits")
MATCHED_BASELINE_RELATIVE = Path(
    "data/private-modernbert-conditional-v1/sweep-rung2-v5/matched-baseline.json"
)
OUTPUT_DIRECTORY_RELATIVE = Path("data/private-modernbert-cascade-v1/confirmation-v3")
BASELINE_OUTPUT_RELATIVE = OUTPUT_DIRECTORY_RELATIVE / "baseline-manifest.json"
MATCHED_BASELINE_OUTPUT_RELATIVE = OUTPUT_DIRECTORY_RELATIVE / "matched-baseline.json"
SOURCE_DIRECTORY_RELATIVE = Path("src/reddit_china_stance")
DEPENDENCY_LOCK_RELATIVE = Path("uv.lock")

GPU_TYPE = "L4"
GPU_RATE_USD_PER_SECOND = Decimal("0.000222")
APPROVED_HARD_COST_USD = Decimal("20")
PLANNED_UPPER_COST_USD = Decimal("20")
RELEVANCE_MAX_GPU_SECONDS = 6_000
TARGET_MAX_GPU_SECONDS = 18_000
TRIAL_COUNT = 6


class CascadePreparationError(RuntimeError):
    """Raised when preparation inputs or immutable outputs drift."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CascadePreparationError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CascadePreparationError(f"{where} must contain a JSON object")
    return value


def _sha(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise CascadePreparationError(f"{where} must be a lowercase SHA-256")
    return value


def _source_bundle(repo_root: Path) -> tuple[dict[str, Any], bytes]:
    paths = sorted(
        path for path in (repo_root / SOURCE_DIRECTORY_RELATIVE).glob("*.py") if path.is_file()
    )
    if not paths:
        raise CascadePreparationError("cascade source bundle has no Python files")
    files = {path.relative_to(repo_root).as_posix(): file_sha256(path) for path in paths}
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_BUNDLE_KIND,
        "source_glob": "src/reddit_china_stance/*.py",
        "file_count": len(files),
        "files": files,
        "code_sha256": canonical_sha256(files),
    }
    bundle = {**body, "source_bundle_id": canonical_sha256(body)}
    assert_metadata_only(bundle, where="cascade source bundle")
    return bundle, _json_bytes(bundle)


def _confirmatory_inputs(repo_root: Path) -> dict[str, Any]:
    manifest_path = repo_root / CONFIRMATORY_MANIFEST_RELATIVE
    manifest = _read_json(manifest_path, where="frozen confirmatory manifest")
    contract = manifest.get("experiment_contract")
    confirmatory = manifest.get("confirmatory")
    if not isinstance(contract, Mapping) or not isinstance(confirmatory, Mapping):
        raise CascadePreparationError("frozen confirmatory manifest is incomplete")
    bindings = contract.get("bindings")
    split_binding = confirmatory.get("split_binding")
    threshold = confirmatory.get("threshold_receipt")
    if not all(isinstance(value, Mapping) for value in (bindings, split_binding, threshold)):
        raise CascadePreparationError("frozen confirmatory bindings are incomplete")
    if manifest.get("phase") != "confirmatory" or confirmatory.get("locked_test") != {
        "authorised": False,
        "rows_accessed": 0,
        "predictions_authorised": False,
    }:
        raise CascadePreparationError("source confirmatory run is not a locked-test-free phase")
    if (
        bindings.get("dataset_id") != DATASET_ID
        or bindings.get("dataset_revision") != DATASET_REVISION
        or bindings.get("dataset_parquet_sha256") != DATASET_SHA256
        or bindings.get("development_proxy_sha256") != DEVELOPMENT_PROXY_ID
        or bindings.get("model_id") != MODEL_ID
        or bindings.get("model_revision") != MODEL_REVISION
    ):
        raise CascadePreparationError("source dataset, development, or model binding drifted")
    development_reference_sha256 = _sha(
        threshold.get("development_reference_sha256"),
        where="development reference SHA-256",
    )
    if threshold.get("development_rows") != DEVELOPMENT_ROWS:
        raise CascadePreparationError("development reference row count drifted")
    split_file_sha256 = _sha(
        split_binding.get("split_manifest_file_sha256"),
        where="split manifest file SHA-256",
    )
    split_id = _sha(split_binding.get("split_manifest_id"), where="split manifest ID")
    if bindings.get("split_manifest_sha256") != split_file_sha256:
        raise CascadePreparationError("source split bindings disagree")
    split_path = repo_root / SPLIT_MANIFEST_DIRECTORY_RELATIVE / f"manifest-{split_id}.json"
    if file_sha256(split_path) != split_file_sha256:
        raise CascadePreparationError("split manifest file hash drifted")
    split_manifest = _read_json(split_path, where="frozen split manifest")
    body = {key: value for key, value in split_manifest.items() if key != "manifest_id"}
    if (
        split_manifest.get("manifest_id") != split_id
        or canonical_sha256(body) != split_id
        or split_manifest.get("contains_reddit_text") is not False
    ):
        raise CascadePreparationError("split manifest identity or privacy binding drifted")
    assert_metadata_only(manifest, where="source confirmatory manifest")
    return {
        "manifest": manifest,
        "manifest_bytes": manifest_path.read_bytes(),
        "manifest_sha256": file_sha256(manifest_path),
        "split_manifest": split_manifest,
        "split_manifest_sha256": split_file_sha256,
        "development_reference_sha256": development_reference_sha256,
    }


def _dataset_profile(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    ladders = split_manifest.get("ladders")
    if not isinstance(ladders, Mapping):
        raise CascadePreparationError("split manifest lacks ladders")
    expected_conditions = {(row["ladder_seed"], row["optimiser_seed"]) for row in PAIRED_CONDITIONS}
    subset_digests: dict[str, str] = {}
    profiles: list[Mapping[str, Any]] = []
    observed_conditions: set[tuple[int, int]] = set()
    for condition in PAIRED_CONDITIONS:
        ladder = ladders.get(str(condition["ladder_seed"]))
        budgets = ladder.get("budgets") if isinstance(ladder, Mapping) else None
        subset = budgets.get("10k") if isinstance(budgets, Mapping) else None
        if not isinstance(subset, Mapping):
            raise CascadePreparationError("split manifest lacks a registered 10k subset")
        optimiser_seed = ladder.get("optimiser_seed")
        observed_conditions.add((condition["ladder_seed"], optimiser_seed))
        if subset.get("row_count") != RELEVANCE_TRAINING_ROWS:
            raise CascadePreparationError("10k subset row count drifted")
        support = subset.get("feature_support")
        if not isinstance(support, Mapping):
            raise CascadePreparationError("10k subset lacks aggregate feature support")
        profiles.append(support)
        subset_digests[str(condition["ladder_seed"])] = canonical_sha256(dict(subset))
    if observed_conditions != expected_conditions or any(
        profile != profiles[0] for profile in profiles[1:]
    ):
        raise CascadePreparationError("paired 10k subsets or aggregate profiles disagree")
    support = profiles[0]
    if support.get('relevance="material"') != MATERIAL_TRAINING_ROWS:
        raise CascadePreparationError("material-row support drifted")
    present_counts = {target: support.get(f'target="{target}"') for target in TARGETS}
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in present_counts.values()
    ):
        raise CascadePreparationError("target support is incomplete")
    state_counts = {state: 0 for state in TARGET_STATE_COUNTS if state != "absent"}
    for target in TARGETS:
        for state in state_counts:
            value = support.get(f'target_stance=["{target}","{state}"]', 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CascadePreparationError("target-by-stance support is invalid")
            state_counts[state] += value
    state_counts["absent"] = MATERIAL_TRAINING_ROWS * len(TARGETS) - sum(present_counts.values())
    ordered_state_counts = {state: state_counts[state] for state in TARGET_STATE_COUNTS}
    if (
        ordered_state_counts != TARGET_STATE_COUNTS
        or sum(ordered_state_counts.values()) != TARGET_TRAINING_ROWS
        or RELEVANCE_TRAINING_ROWS * len(TARGETS) - TARGET_TRAINING_ROWS
        != EXCLUDED_NON_MATERIAL_TARGET_SLOTS
    ):
        raise CascadePreparationError("target-conditioned expansion profile drifted")
    body = {
        "relevance_training_rows": RELEVANCE_TRAINING_ROWS,
        "material_training_rows": MATERIAL_TRAINING_ROWS,
        "target_conditioned_training_rows": TARGET_TRAINING_ROWS,
        "excluded_non_material_target_slots": EXCLUDED_NON_MATERIAL_TARGET_SLOTS,
        "target_state_counts": ordered_state_counts,
        "development_rows": DEVELOPMENT_ROWS,
    }
    receipt_body = {
        "schema_version": SCHEMA_VERSION,
        "kind": DATASET_PROFILE_KIND,
        "data": body,
        "subset_manifest_sha256_by_ladder": subset_digests,
        "source": "frozen_split_manifest_aggregate_feature_support",
    }
    receipt = {**receipt_body, "profile_id": canonical_sha256(receipt_body)}
    assert_metadata_only(receipt, where="cascade dataset profile receipt")
    return receipt


def _matched_baseline(
    repo_root: Path, *, source_manifest_sha256: str
) -> tuple[dict[str, Any], bytes]:
    path = repo_root / MATCHED_BASELINE_RELATIVE
    value = _read_json(path, where="matched three-head baseline")
    entries = value.get("entries")
    conditions = value.get("condition_pairs")
    expected_conditions = [dict(row) for row in PAIRED_CONDITIONS]
    if (
        value.get("kind") != "modernbert-conditional-matched-baseline-v1"
        or value.get("source_confirmatory_manifest_sha256") != source_manifest_sha256
        or value.get("label_budget") != RELEVANCE_TRAINING_ROWS
        or value.get("registered_config_sha256") != BASELINE_CONFIG_SHA256
        or conditions != expected_conditions
        or value.get("receipt_count") != 3
        or not isinstance(entries, list)
        or len(entries) != 3
    ):
        raise CascadePreparationError("matched baseline contract drifted")
    observed = {(entry.get("ladder_seed"), entry.get("optimiser_seed")) for entry in entries}
    expected = {(row["ladder_seed"], row["optimiser_seed"]) for row in PAIRED_CONDITIONS}
    if observed != expected or any(
        entry.get("config_sha256") != BASELINE_CONFIG_SHA256 for entry in entries
    ):
        raise CascadePreparationError("matched baseline paired inventory drifted")
    body = {key: item for key, item in value.items() if key != "matched_baseline_id"}
    if value.get("matched_baseline_id") != canonical_sha256(body):
        raise CascadePreparationError("matched baseline content address drifted")
    assert_metadata_only(value, where="matched baseline")
    return value, path.read_bytes()


def _descriptor(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "repo_relative_path": path.relative_to(repo_root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_immutable_set(output_directory: Path, outputs: Mapping[str, bytes]) -> None:
    expected = {
        "baseline-manifest.json",
        "matched-baseline.json",
        "dataset-profile.json",
        "source-bundle.json",
        "run-manifest.json",
    }
    if set(outputs) != expected:
        raise AssertionError("cascade preparation output inventory drifted")
    if output_directory.exists() and not output_directory.is_dir():
        raise CascadePreparationError("cascade output path is not a directory")
    if output_directory.is_dir():
        unexpected = {path.name for path in output_directory.iterdir()} - expected
        if unexpected:
            raise CascadePreparationError("cascade output directory is not exact")
        for name, payload in outputs.items():
            path = output_directory / name
            if path.exists() and (not path.is_file() or path.read_bytes() != payload):
                raise CascadePreparationError(f"immutable cascade output differs: {name}")
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


def prepare_cascade_confirmation(repo_root: Path, *, dry_run: bool) -> dict[str, Any]:
    """Build or dry-run the exact metadata-only six-trial cascade manifest."""

    root = repo_root.resolve()
    inputs = _confirmatory_inputs(root)
    source_bundle, source_bytes = _source_bundle(root)
    profile = _dataset_profile(inputs["split_manifest"])
    matched, matched_bytes = _matched_baseline(
        root, source_manifest_sha256=inputs["manifest_sha256"]
    )
    baseline_bytes = inputs["manifest_bytes"]
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    matched_sha256 = hashlib.sha256(matched_bytes).hexdigest()
    dependency_lock_sha256 = file_sha256(root / DEPENDENCY_LOCK_RELATIVE)
    source_bundle_sha256 = hashlib.sha256(source_bytes).hexdigest()
    experiment = freeze_experiment_contract(
        dataset_revision=DATASET_REVISION,
        dataset_sha256=DATASET_SHA256,
        split_manifest_sha256=inputs["split_manifest_sha256"],
        source_bundle_sha256=source_bundle_sha256,
        code_sha256=source_bundle["code_sha256"],
        dependency_lock_sha256=dependency_lock_sha256,
        development_proxy_sha256=DEVELOPMENT_PROXY_ID,
        development_reference_sha256=inputs["development_reference_sha256"],
        baseline_manifest_sha256=baseline_sha256,
        matched_baseline_sha256=matched_sha256,
        teacher_generation_run_id=TEACHER_GENERATION_RUN_ID,
        rate_card_usd_per_gpu_second={GPU_TYPE: str(GPU_RATE_USD_PER_SECOND)},
        hard_cost_cap_usd=str(APPROVED_HARD_COST_USD),
    )
    subset_digests = profile["subset_manifest_sha256_by_ladder"]
    trials = [
        freeze_trial_spec(
            experiment,
            component=component,
            subset_manifest_sha256=subset_digests[str(condition["ladder_seed"])],
            ladder_seed=condition["ladder_seed"],
            optimiser_seed=condition["optimiser_seed"],
            gpu_type=GPU_TYPE,
            max_gpu_seconds=(
                RELEVANCE_MAX_GPU_SECONDS if component == "relevance" else TARGET_MAX_GPU_SECONDS
            ),
        )
        for component in COMPONENTS
        for condition in PAIRED_CONDITIONS
    ]
    output_root = root / OUTPUT_DIRECTORY_RELATIVE
    baseline_descriptor = {
        "repo_relative_path": BASELINE_OUTPUT_RELATIVE.as_posix(),
        "sha256": baseline_sha256,
        "bytes": len(baseline_bytes),
    }
    matched_descriptor = {
        "repo_relative_path": MATCHED_BASELINE_OUTPUT_RELATIVE.as_posix(),
        "sha256": matched_sha256,
        "bytes": len(matched_bytes),
    }
    run_manifest = build_run_manifest(
        experiment,
        trials=trials,
        baseline_manifest=baseline_descriptor,
        matched_baseline=matched_descriptor,
        dataset_profile=profile["data"],
    )
    validate_run_manifest(run_manifest)
    profile_bytes = _json_bytes(profile)
    manifest_bytes = _json_bytes(run_manifest)
    outputs = {
        "baseline-manifest.json": baseline_bytes,
        "matched-baseline.json": matched_bytes,
        "dataset-profile.json": profile_bytes,
        "source-bundle.json": source_bytes,
        "run-manifest.json": manifest_bytes,
    }
    if not dry_run:
        _write_immutable_set(output_root, outputs)
        # Verify the copied immutable inputs rather than trusting write completion.
        if (
            _descriptor(output_root / "baseline-manifest.json", repo_root=root)
            != baseline_descriptor
        ):
            raise CascadePreparationError("published baseline descriptor drifted")
        if _descriptor(output_root / "matched-baseline.json", repo_root=root) != matched_descriptor:
            raise CascadePreparationError("published matched-baseline descriptor drifted")
    relevance_reserved_seconds = RELEVANCE_MAX_GPU_SECONDS * 3
    target_reserved_seconds = TARGET_MAX_GPU_SECONDS * 3
    total_reserved_seconds = relevance_reserved_seconds + target_reserved_seconds
    reserved_cost = GPU_RATE_USD_PER_SECOND * total_reserved_seconds
    if reserved_cost > PLANNED_UPPER_COST_USD:
        raise CascadePreparationError("six-trial reservation exceeds the $20 hard cap")
    summary = {
        "status": "dry-run" if dry_run else "prepared",
        "output_directory": OUTPUT_DIRECTORY_RELATIVE.as_posix(),
        "source_file_count": source_bundle["file_count"],
        "code_sha256": source_bundle["code_sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "split_manifest_sha256": inputs["split_manifest_sha256"],
        "development_reference_sha256": inputs["development_reference_sha256"],
        "baseline_manifest_sha256": baseline_sha256,
        "matched_baseline_id": matched["matched_baseline_id"],
        "matched_baseline_sha256": matched_sha256,
        "dataset_profile_id": profile["profile_id"],
        "experiment_run_id": experiment["experiment_run_id"],
        "phase_run_id": run_manifest["phase_run_id"],
        "trial_count": len(trials),
        "component_trial_counts": {component: 3 for component in COMPONENTS},
        "gpu_type": GPU_TYPE,
        "gpu_rate_usd_per_second": str(GPU_RATE_USD_PER_SECOND),
        "reserved_gpu_seconds": total_reserved_seconds,
        "reserved_cost_usd": format(reserved_cost, "f"),
        "planned_upper_cost_usd": format(PLANNED_UPPER_COST_USD, "f"),
        "approved_hard_cost_usd": format(APPROVED_HARD_COST_USD, "f"),
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(summary, where="cascade preparation summary")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the exact metadata-only ModernBERT cascade confirmation run."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = prepare_cascade_confirmation(args.repo_root, dry_run=args.dry_run)
    except (CascadePreparationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
