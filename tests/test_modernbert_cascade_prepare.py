from __future__ import annotations

import json
from pathlib import Path

import pytest

from reddit_china_stance import modernbert_cascade_prepare as prepare
from reddit_china_stance.modernbert_cascade_experiment import (
    BASELINE_CONFIG_SHA256,
    PAIRED_CONDITIONS,
    TARGET_STATE_COUNTS,
    TARGETS,
)
from reddit_china_stance.modernbert_training import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    MODEL_ID,
    MODEL_REVISION,
    canonical_sha256,
    file_sha256,
)


def write_json(path: Path, value: dict) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def feature_support() -> dict[str, int]:
    support = {
        'relevance="material"': 6_436,
        'relevance="not_material"': 3_490,
        'relevance="unclear"': 74,
        'target="china_general"': 1_356,
        'target="government_ccp"': 2_398,
        'target="people_culture"': 1_923,
        'target="other"': 2_171,
    }
    target_stance = {
        "china_general": {
            "negative": 351,
            "mixed": 24,
            "no_directed_stance": 753,
            "positive": 220,
            "unclear": 8,
        },
        "government_ccp": {
            "negative": 1_060,
            "mixed": 33,
            "no_directed_stance": 1_096,
            "positive": 201,
            "unclear": 8,
        },
        "people_culture": {
            "negative": 264,
            "mixed": 44,
            "no_directed_stance": 1_108,
            "positive": 507,
            "unclear": 0,
        },
        "other": {
            "negative": 560,
            "mixed": 71,
            "no_directed_stance": 1_085,
            "positive": 450,
            "unclear": 5,
        },
    }
    for target, states in target_stance.items():
        for state, count in states.items():
            if count:
                support[f'target_stance=["{target}","{state}"]'] = count
    return support


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src/reddit_china_stance").mkdir(parents=True)
    (root / "src/reddit_china_stance/example.py").write_text("VALUE = 1\n")
    (root / "uv.lock").write_text("frozen = true\n")

    ladders = {}
    for condition in PAIRED_CONDITIONS:
        subset = {
            "row_count": 10_000,
            "folds": list(range(10)),
            "sample_ids_sha256": "a" * 64,
            "feature_support": feature_support(),
        }
        ladders[str(condition["ladder_seed"])] = {
            "optimiser_seed": condition["optimiser_seed"],
            "budgets": {"10k": subset},
        }
    split_body = {
        "schema_version": "1.0.0",
        "kind": "synthetic-split",
        "contains_reddit_text": False,
        "ladders": ladders,
    }
    split = {**split_body, "manifest_id": canonical_sha256(split_body)}
    split_path = (
        root / prepare.SPLIT_MANIFEST_DIRECTORY_RELATIVE / f"manifest-{split['manifest_id']}.json"
    )
    write_json(split_path, split)
    split_sha = file_sha256(split_path)

    confirmatory = {
        "schema_version": "1.0.0",
        "kind": "modernbert-phase-run-v1",
        "phase": "confirmatory",
        "experiment_contract": {
            "bindings": {
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "dataset_parquet_sha256": DATASET_SHA256,
                "development_proxy_sha256": DEVELOPMENT_PROXY_ID,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "split_manifest_sha256": split_sha,
            }
        },
        "confirmatory": {
            "locked_test": {
                "authorised": False,
                "rows_accessed": 0,
                "predictions_authorised": False,
            },
            "split_binding": {
                "split_manifest_file_sha256": split_sha,
                "split_manifest_id": split["manifest_id"],
            },
            "threshold_receipt": {
                "development_reference_sha256": "b" * 64,
                "development_rows": 222,
            },
        },
    }
    manifest_path = root / prepare.CONFIRMATORY_MANIFEST_RELATIVE
    write_json(manifest_path, confirmatory)

    entries = [
        {
            "config_sha256": BASELINE_CONFIG_SHA256,
            "ladder_seed": row["ladder_seed"],
            "optimiser_seed": row["optimiser_seed"],
            "trial_id": str(index) * 64,
        }
        for index, row in enumerate(PAIRED_CONDITIONS, start=1)
    ]
    baseline_body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-conditional-matched-baseline-v1",
        "source_confirmatory_manifest_sha256": file_sha256(manifest_path),
        "label_budget": 10_000,
        "registered_config_sha256": BASELINE_CONFIG_SHA256,
        "condition_pairs": [dict(row) for row in PAIRED_CONDITIONS],
        "receipt_count": 3,
        "entries": entries,
    }
    matched = {**baseline_body, "matched_baseline_id": canonical_sha256(baseline_body)}
    write_json(root / prepare.MATCHED_BASELINE_RELATIVE, matched)
    return root


def test_dataset_profile_uses_metadata_only_10k_aggregate(synthetic_repo: Path) -> None:
    inputs = prepare._confirmatory_inputs(synthetic_repo)
    profile = prepare._dataset_profile(inputs["split_manifest"])

    assert profile["data"]["material_training_rows"] == 6_436
    assert profile["data"]["target_conditioned_training_rows"] == 25_744
    assert profile["data"]["target_state_counts"] == TARGET_STATE_COUNTS
    assert set(profile["subset_manifest_sha256_by_ladder"]) == {"101", "202", "303"}


def test_prepare_dry_run_freezes_exact_six_trial_manifest(synthetic_repo: Path) -> None:
    summary = prepare.prepare_cascade_confirmation(synthetic_repo, dry_run=True)

    assert summary["status"] == "dry-run"
    assert summary["trial_count"] == 6
    assert summary["component_trial_counts"] == {
        "relevance": 3,
        "target_conditioned": 3,
    }
    assert summary["reserved_cost_usd"] == "15.984000"
    assert summary["approved_hard_cost_usd"] == "20"
    assert summary["locked_test_rows_accessed"] == 0
    assert not (synthetic_repo / prepare.OUTPUT_DIRECTORY_RELATIVE).exists()


def test_prepare_writes_exact_idempotent_immutable_set(synthetic_repo: Path) -> None:
    first = prepare.prepare_cascade_confirmation(synthetic_repo, dry_run=False)
    second = prepare.prepare_cascade_confirmation(synthetic_repo, dry_run=False)
    output = synthetic_repo / prepare.OUTPUT_DIRECTORY_RELATIVE

    assert first == second
    assert {path.name for path in output.iterdir()} == {
        "baseline-manifest.json",
        "matched-baseline.json",
        "dataset-profile.json",
        "source-bundle.json",
        "run-manifest.json",
    }
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["phase_run_id"] == first["phase_run_id"]
    assert len(manifest["trials"]) == 6
    assert manifest["locked_test"]["rows_accessed"] == 0


def test_prepare_rejects_locked_test_access(synthetic_repo: Path) -> None:
    path = synthetic_repo / prepare.CONFIRMATORY_MANIFEST_RELATIVE
    manifest = json.loads(path.read_text())
    manifest["confirmatory"]["locked_test"]["rows_accessed"] = 1
    write_json(path, manifest)

    with pytest.raises(prepare.CascadePreparationError, match="locked-test-free"):
        prepare.prepare_cascade_confirmation(synthetic_repo, dry_run=True)


def test_prepare_rejects_target_profile_drift(synthetic_repo: Path) -> None:
    inputs = prepare._confirmatory_inputs(synthetic_repo)
    split = inputs["split_manifest"]
    split["ladders"]["101"]["budgets"]["10k"]["feature_support"]['target="china_general"'] += 1

    with pytest.raises(prepare.CascadePreparationError):
        prepare._dataset_profile(split)


def test_prepare_rejects_immutable_output_drift(synthetic_repo: Path) -> None:
    prepare.prepare_cascade_confirmation(synthetic_repo, dry_run=False)
    output = synthetic_repo / prepare.OUTPUT_DIRECTORY_RELATIVE / "run-manifest.json"
    output.write_text("{}\n")

    with pytest.raises(prepare.CascadePreparationError, match="immutable cascade output"):
        prepare.prepare_cascade_confirmation(synthetic_repo, dry_run=False)


def test_profile_state_counts_conserve_expansion() -> None:
    assert sum(TARGET_STATE_COUNTS.values()) == 6_436 * len(TARGETS)
