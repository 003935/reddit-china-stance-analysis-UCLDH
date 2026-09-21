from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from reddit_china_stance import modernbert_conditional_prepare as prepare
from reddit_china_stance.modernbert_training import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    PROXY_JSON_SHA256,
    REFERENCE_ROWS_JSON_SHA256,
    canonical_sha256,
)
from reddit_china_stance.privacy import assert_metadata_only


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prepare._json_bytes(value))


def _trial(*, ladder_seed: int, optimiser_seed: int) -> dict:
    registered_body = {
        "encoder_learning_rate": 5e-5,
    }
    body = {
        "schema_version": "1.0.0",
        "phase": "confirmatory",
        "name": f"baseline-{ladder_seed}-{optimiser_seed}",
        "gpu_type": "L4",
        "config": {
            "label_budget": 10_000,
            "training_row_count": 10_000,
            "ladder_seed": ladder_seed,
            "optimiser_seed": optimiser_seed,
            "seed": optimiser_seed,
            "registered_config": {
                **registered_body,
                "config_sha256": canonical_sha256(registered_body),
            },
        },
    }
    return {**body, "trial_id": canonical_sha256(body)}


def _receipt(*, trial: dict, run_id: str, bindings: dict) -> dict:
    body = {
        "schema_version": "1.0.0",
        "status": "complete",
        "experiment_run_id": run_id,
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": canonical_sha256(trial),
        "bindings": deepcopy(bindings),
        "artifacts": {
            "metrics": {
                "relative_path": "metrics.json",
                "sha256": "b" * 64,
                "bytes": 10,
            }
        },
        "aggregate_metrics": {
            "development_rows": 222,
            "training_rows": 10_000,
            "composite": 0.5,
            "material_recall": 0.9,
            "invalid_outputs": 0,
        },
        "compute": {
            "gpu_type": "L4",
            "wall_seconds": 10,
            "gpu_seconds": 10,
            "estimated_cost_usd": 0.00222,
        },
    }
    return {**body, "receipt_id": canonical_sha256(body)}


def _synthetic_teacher_rows(*, all_states: bool) -> list[dict[str, str]]:
    targets = ("china_general", "government_ccp", "people_culture", "other")
    stances = (
        "negative",
        "mixed",
        "no_directed_stance",
        "positive",
        "unclear",
    )
    rows = []
    for index in range(10_000):
        stance = stances[index % len(stances)] if all_states else "negative"
        label = {
            "relevance": "material",
            "target_stances": [
                {
                    "target": targets[(index // len(stances)) % len(targets)],
                    "stance": stance,
                }
            ],
        }
        rows.append(
            {
                "sample_id": f"sample-{index:05d}",
                "label_json": json.dumps(label, sort_keys=True),
            }
        )
    return rows


def _write_development_proxy(repo_root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    label = json.dumps(
        {
            "relevance": "material",
            "target_stances": [
                {"target": "china_general", "stance": "negative"}
            ],
        },
        sort_keys=True,
    )
    rows = [
        {
            "source_sample_id": f"development-{index:03d}",
            "target_text": "private",
            "parent_context": None,
            "submission_context": None,
            "split": "development",
            "resolution": "synthetic",
            "label_json": label,
        }
        for index in range(222)
    ]
    rows.extend(
        {
            "source_sample_id": f"locked-{index:03d}",
            "target_text": "private",
            "parent_context": None,
            "submission_context": None,
            "split": "locked_test_candidate",
            "resolution": "synthetic",
            "label_json": label,
        }
        for index in range(230)
    )
    schema = pa.schema(
        [
            pa.field("source_sample_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("parent_context", pa.string()),
            pa.field("submission_context", pa.string()),
            pa.field("split", pa.string(), nullable=False),
            pa.field("resolution", pa.string(), nullable=False),
            pa.field("label_json", pa.string(), nullable=False),
        ],
        metadata={
            b"proxy_id": DEVELOPMENT_PROXY_ID.encode(),
            b"proxy_json_sha256": PROXY_JSON_SHA256.encode(),
            b"reference_rows_json_sha256": REFERENCE_ROWS_JSON_SHA256.encode(),
        },
    )
    path = repo_root / prepare.DEVELOPMENT_PROXY_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


@pytest.fixture
def synthetic_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "src/reddit_china_stance"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("\n", encoding="utf-8")
    (source / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "beta.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _write_development_proxy(tmp_path)

    split_rows = [
        {
            "sample_id": f"sample-{index:05d}",
            "folds": {"101": index % 10},
        }
        for index in range(10_000)
    ]
    selected_ids = [
        row["sample_id"] for row in split_rows if row["folds"]["101"] in set(range(5))
    ]
    subset = {
        "row_count": 5_000,
        "sample_ids_sha256": canonical_sha256(sorted(selected_ids)),
        "folds": [0, 1, 2, 3, 4],
        "feature_support": {"material": 3_000},
    }
    split_body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-private-split-manifest-v1",
        "private": True,
        "contains_reddit_text": False,
        "ladders": {
            "101": {
                "acquisition_seed": 101,
                "optimiser_seed": 47,
                "budgets": {"5k": subset},
            }
        },
        "rows": split_rows,
    }
    split = {**split_body, "manifest_id": canonical_sha256(split_body)}
    split_path = (
        tmp_path
        / prepare.SPLIT_MANIFEST_DIRECTORY_RELATIVE
        / f"manifest-{split['manifest_id']}.json"
    )
    _write_json(split_path, split)
    split_file_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()

    bindings = {
        "model_id": "answerdotai/ModernBERT-large",
        "model_revision": "d" * 64,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_parquet_sha256": DATASET_SHA256,
        "development_proxy_sha256": DEVELOPMENT_PROXY_ID,
        "split_manifest_sha256": split_file_sha256,
        "trainer_code_sha256": "f" * 64,
        "dependency_lock_sha256": "1" * 64,
    }
    run_id = "2" * 64
    trials = [
        _trial(ladder_seed=101, optimiser_seed=47),
        _trial(ladder_seed=202, optimiser_seed=61),
        _trial(ladder_seed=303, optimiser_seed=89),
    ]
    manifest = {
        "schema_version": "1.0.0",
        "kind": "modernbert-phase-run-v1",
        "experiment_run_id": run_id,
        "phase_run_id": "3" * 64,
        "phase": "confirmatory",
        "experiment_contract": {"bindings": bindings},
        "confirmatory": {
            "design": {
                "paired_ladders": [
                    {"ladder_seed": 101, "optimiser_seed": 47},
                    {"ladder_seed": 202, "optimiser_seed": 61},
                    {"ladder_seed": 303, "optimiser_seed": 89},
                ]
            },
            "split_binding": {
                "split_manifest_file_sha256": split_file_sha256,
                "split_manifest_id": split["manifest_id"],
            },
            "locked_test": {
                "authorised": False,
                "rows_accessed": 0,
                "predictions_authorised": False,
            },
        },
        "trials": trials,
    }
    _write_json(tmp_path / prepare.CONFIRMATORY_MANIFEST_RELATIVE, manifest)

    receipt_root = tmp_path / prepare.BASELINE_RECEIPTS_RELATIVE
    for trial in trials:
        seed = trial["config"]["optimiser_seed"]
        _write_json(
            receipt_root / f"seed-{seed}.json",
            _receipt(trial=trial, run_id=run_id, bindings=bindings),
        )
    teacher_rows = _synthetic_teacher_rows(all_states=True)
    monkeypatch.setattr(
        prepare,
        "_load_teacher_label_rows",
        lambda repo_root, *, expected_sha256: deepcopy(teacher_rows),
    )
    return tmp_path


def test_dry_run_freezes_exact_inventory_hashes_cost_and_privacy(
    synthetic_repo: Path,
) -> None:
    summary = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)
    assert summary["status"] == "dry-run"
    assert summary["source_file_count"] == 3
    assert summary["trial_count"] == 8
    assert summary["training_row_count"] == 5_000
    assert summary["max_gpu_seconds_per_trial"] == 5_400
    assert summary["reserved_gpu_seconds"] == 43_200
    assert summary["planned_reserved_cost_usd"] == "9.590400"
    assert summary["full_family_trial_count"] == 20
    assert summary["full_family_reserved_cost_usd"] == "23.976000"
    assert prepare.FULL_FAMILY_RESERVED_COST_USD <= prepare.PLANNED_UPPER_COST_USD
    assert summary["planned_upper_cost_usd"] == "50"
    assert summary["approved_hard_cost_usd"] == "200"
    assert summary["locked_test_rows_accessed"] == 0
    assert summary["target_state_material_row_count"] == 5_000
    assert summary["target_state_material_slot_count"] == 20_000
    assert summary["zero_support_weighted_state_count"] == 0
    assert summary["development_proxy_id"] == DEVELOPMENT_PROXY_ID
    assert summary["development_reference_rows"] == 222
    assert summary["development_reference_sha256"] != DEVELOPMENT_PROXY_ID
    assert not (synthetic_repo / prepare.OUTPUT_DIRECTORY_RELATIVE).exists()
    assert_metadata_only(summary)


def test_prepare_writes_only_exact_outputs_and_is_idempotent(synthetic_repo: Path) -> None:
    first = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=False)
    output = synthetic_repo / prepare.OUTPUT_DIRECTORY_RELATIVE
    assert {path.name for path in output.iterdir()} == {
        "baseline-manifest.json",
        "source-bundle.json",
        "matched-baseline.json",
        "run-manifest.json",
    }
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    second = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=False)
    assert first == second
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original

    bundle = json.loads((output / "source-bundle.json").read_text())
    expected_files = sorted(
        path.relative_to(synthetic_repo).as_posix()
        for path in (synthetic_repo / prepare.SOURCE_DIRECTORY_RELATIVE).glob("*.py")
    )
    assert sorted(bundle["files"]) == expected_files
    assert bundle["code_sha256"] == canonical_sha256(bundle["files"])
    assert first["source_bundle_sha256"] == hashlib.sha256(
        (output / "source-bundle.json").read_bytes()
    ).hexdigest()

    baseline = json.loads((output / "matched-baseline.json").read_text())
    assert baseline["receipt_count"] == 3
    assert baseline["condition_pairs"] == [
        {"ladder_seed": 101, "optimiser_seed": 47},
        {"ladder_seed": 202, "optimiser_seed": 61},
        {"ladder_seed": 303, "optimiser_seed": 89},
    ]
    assert {row["config_sha256"] for row in baseline["entries"]} == {
        baseline["registered_config_sha256"]
    }
    assert [(row["ladder_seed"], row["optimiser_seed"]) for row in baseline["entries"]] == [
        (101, 47),
        (202, 61),
        (303, 89),
    ]
    assert len({row["receipt_file_sha256"] for row in baseline["entries"]}) == 3
    support = baseline["target_state_support"]
    assert support["support_receipt_id"] == canonical_sha256(
        {key: value for key, value in support.items() if key != "support_receipt_id"}
    )
    assert baseline["target_state_support_receipt_id"] == support["support_receipt_id"]
    assert baseline["target_state_support_receipt_sha256"] == canonical_sha256(support)
    assert set(support["counts"]) == {
        "china_general",
        "government_ccp",
        "people_culture",
        "other",
    }
    assert all(
        set(states)
        == {"absent", "negative", "mixed", "no_directed_stance", "positive", "unclear"}
        for states in support["counts"].values()
    )
    assert all(value > 0 for value in support["state_totals"].values())
    manifest_sha = hashlib.sha256(
        (output / "matched-baseline.json").read_bytes()
    ).hexdigest()
    assert len(manifest_sha) == 64
    assert (
        json.loads((output / "run-manifest.json").read_text())["experiment_contract"][
            "bindings"
        ]["matched_baseline_receipt_sha256"]
        == manifest_sha
    )
    manifest = json.loads((output / "run-manifest.json").read_text())
    baseline_descriptor = manifest["experiment_artefacts"]["baseline_manifest"]
    baseline_copy = output / "baseline-manifest.json"
    source_baseline = synthetic_repo / prepare.CONFIRMATORY_MANIFEST_RELATIVE
    assert baseline_copy.read_bytes() == source_baseline.read_bytes()
    assert baseline_descriptor == {
        "repo_relative_path": prepare.BASELINE_MANIFEST_OUTPUT_RELATIVE.as_posix(),
        "sha256": hashlib.sha256(baseline_copy.read_bytes()).hexdigest(),
        "bytes": len(baseline_copy.read_bytes()),
    }
    assert manifest["experiment_contract"]["compute"]["planned_upper_cost_usd"] == "50"
    assert manifest["experiment_contract"]["compute"]["approved_cost_usd"] == "200"
    assert manifest["experiment_contract"]["budget"]["hard_cost_cap_usd"] == "200"
    assert manifest["experiment_contract"]["bindings"][
        "development_reference_sha256"
    ] == first["development_reference_sha256"]
    assert manifest["asha"] == {
        "rung_epochs": 2,
        "candidate_count": 8,
        "promotion_count": 4,
    }
    assert len(manifest["trials"]) == 8
    assert {row["ladder_seed"] for row in manifest["trials"]} == {101}
    assert {row["optimiser_seed"] for row in manifest["trials"]} == {7}
    assert {row["training_row_count"] for row in manifest["trials"]} == {5_000}
    assert {row["max_gpu_seconds"] for row in manifest["trials"]} == {5_400}
    for value in (bundle, baseline, manifest):
        assert_metadata_only(value)


def test_receipt_inventory_and_binding_fail_closed(synthetic_repo: Path) -> None:
    receipts = synthetic_repo / prepare.BASELINE_RECEIPTS_RELATIVE
    (receipts / "seed-47.json").unlink()
    with pytest.raises(prepare.ConditionalPreparationError, match="exactly three"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)


def test_receipt_content_drift_fails_closed(synthetic_repo: Path) -> None:
    path = synthetic_repo / prepare.BASELINE_RECEIPTS_RELATIVE / "seed-47.json"
    receipt = json.loads(path.read_text())
    receipt["aggregate_metrics"]["composite"] = 0.9
    _write_json(path, receipt)
    with pytest.raises(prepare.ConditionalPreparationError, match="content address"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)


def test_source_and_dependency_drift_change_plan_and_cannot_overwrite(
    synthetic_repo: Path,
) -> None:
    initial = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=False)
    source = synthetic_repo / prepare.SOURCE_DIRECTORY_RELATIVE / "alpha.py"
    source.write_text("VALUE = 99\n", encoding="utf-8")
    changed_source = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)
    assert changed_source["code_sha256"] != initial["code_sha256"]
    assert changed_source["phase_run_id"] != initial["phase_run_id"]
    with pytest.raises(prepare.ConditionalPreparationError, match="source-bundle"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=False)

    source.write_text("VALUE = 1\n", encoding="utf-8")
    lock = synthetic_repo / prepare.DEPENDENCY_LOCK_RELATIVE
    lock.write_text("version = 2\n", encoding="utf-8")
    changed_lock = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)
    assert changed_lock["dependency_lock_sha256"] != initial["dependency_lock_sha256"]
    assert changed_lock["phase_run_id"] != initial["phase_run_id"]
    with pytest.raises(prepare.ConditionalPreparationError, match="run-manifest"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=False)


def test_source_baseline_manifest_drift_cannot_overwrite_copy(
    synthetic_repo: Path,
) -> None:
    prepare.prepare_conditional_rung2(synthetic_repo, dry_run=False)
    source = synthetic_repo / prepare.CONFIRMATORY_MANIFEST_RELATIVE
    manifest = json.loads(source.read_text())
    manifest["metadata_note"] = "changed"
    _write_json(source, manifest)
    changed = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)
    original_copy = (
        synthetic_repo
        / prepare.OUTPUT_DIRECTORY_RELATIVE
        / "baseline-manifest.json"
    )
    assert changed["baseline_manifest"]["sha256"] != hashlib.sha256(
        original_copy.read_bytes()
    ).hexdigest()
    with pytest.raises(prepare.ConditionalPreparationError, match="baseline-manifest"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=False)

def test_split_subset_and_locked_test_drift_fail_closed(synthetic_repo: Path) -> None:
    manifest_path = synthetic_repo / prepare.CONFIRMATORY_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text())
    manifest["confirmatory"]["locked_test"]["rows_accessed"] = 1
    _write_json(manifest_path, manifest)
    with pytest.raises(prepare.ConditionalPreparationError, match="locked-test"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)


def test_baseline_design_drift_and_duplicate_condition_fail_closed(
    synthetic_repo: Path,
) -> None:
    manifest_path = synthetic_repo / prepare.CONFIRMATORY_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text())
    manifest["confirmatory"]["design"]["paired_ladders"][2]["optimiser_seed"] = 90
    _write_json(manifest_path, manifest)
    with pytest.raises(prepare.ConditionalPreparationError, match="design drifted"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)

    manifest["confirmatory"]["design"]["paired_ladders"][2]["optimiser_seed"] = 89
    old_trial = manifest["trials"][2]
    duplicate = deepcopy(old_trial)
    duplicate["config"]["ladder_seed"] = 202
    duplicate["config"]["optimiser_seed"] = 61
    duplicate["config"]["seed"] = 61
    duplicate["trial_id"] = canonical_sha256(
        {key: value for key, value in duplicate.items() if key != "trial_id"}
    )
    manifest["trials"][2] = duplicate
    _write_json(manifest_path, manifest)
    bindings = manifest["experiment_contract"]["bindings"]
    _write_json(
        synthetic_repo / prepare.BASELINE_RECEIPTS_RELATIVE / "seed-89.json",
        _receipt(
            trial=duplicate,
            run_id=manifest["experiment_run_id"],
            bindings=bindings,
        ),
    )
    with pytest.raises(prepare.ConditionalPreparationError, match="exact frozen conditions"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)


def test_baseline_registered_config_mismatch_fails_closed(synthetic_repo: Path) -> None:
    manifest_path = synthetic_repo / prepare.CONFIRMATORY_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text())
    changed = deepcopy(manifest["trials"][2])
    registered_body = {"encoder_learning_rate": 3e-5}
    changed["config"]["registered_config"] = {
        **registered_body,
        "config_sha256": canonical_sha256(registered_body),
    }
    changed["trial_id"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "trial_id"}
    )
    manifest["trials"][2] = changed
    _write_json(manifest_path, manifest)
    _write_json(
        synthetic_repo / prepare.BASELINE_RECEIPTS_RELATIVE / "seed-89.json",
        _receipt(
            trial=changed,
            run_id=manifest["experiment_run_id"],
            bindings=manifest["experiment_contract"]["bindings"],
        ),
    )
    with pytest.raises(prepare.ConditionalPreparationError, match="one shared config"):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)


def test_weighted_state_without_selected_5k_support_fails_before_manifest(
    synthetic_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsupported_rows = _synthetic_teacher_rows(all_states=False)
    monkeypatch.setattr(
        prepare,
        "_load_teacher_label_rows",
        lambda repo_root, *, expected_sha256: deepcopy(unsupported_rows),
    )
    with pytest.raises(
        prepare.ConditionalPreparationError,
        match="weighted target-state class lacks selected-5k support",
    ):
        prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)
    assert not (synthetic_repo / prepare.OUTPUT_DIRECTORY_RELATIVE).exists()


def test_development_reference_digest_matches_runtime_canonical_recomputation(
    synthetic_repo: Path,
) -> None:
    import pyarrow.parquet as pq

    summary = prepare.prepare_conditional_rung2(synthetic_repo, dry_run=True)
    rows = pq.read_table(
        synthetic_repo / prepare.DEVELOPMENT_PROXY_RELATIVE,
        columns=["source_sample_id", "label_json"],
        filters=[("split", "=", "development")],
    ).to_pylist()
    reference = {
        row["source_sample_id"]: json.loads(row["label_json"]) for row in rows
    }
    runtime_binding = {
        "development_proxy_id": DEVELOPMENT_PROXY_ID,
        "split": "development",
        "labels": [
            {"source_sample_id": sample_id, "label": reference[sample_id]}
            for sample_id in sorted(reference)
        ],
    }
    assert summary["development_reference_sha256"] == canonical_sha256(runtime_binding)
    assert summary["development_reference_sha256"] != summary["development_proxy_id"]
