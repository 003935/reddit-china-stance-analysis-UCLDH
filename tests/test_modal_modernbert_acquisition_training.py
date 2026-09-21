from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from reddit_china_stance import modal_modernbert_acquisition_training as launcher
from reddit_china_stance import modernbert_acquisition as acquisition
from reddit_china_stance import modernbert_acquisition_training as training


def _preparation_spec() -> dict:
    _, gate = acquisition.load_acquisition_policies(
        Path("configs/modernbert-acquisition-v1.toml")
    )
    return {
        "schema_version": "1.0.0",
        "kind": "modernbert-acquisition-training-preparation-spec-v1",
        "base_training_frame": {
            "relative_path": "a",
            "sha256": "1" * 64,
            "bytes": 1,
        },
        "acquisition_source": {
            "relative_path": "b",
            "sha256": "2" * 64,
            "bytes": 1,
        },
        "acquisition_labels": {
            "relative_path": "c",
            "sha256": "3" * 64,
            "bytes": 1,
        },
        "acquisition_ledger": {
            "relative_path": "ledger",
            "sha256": "5" * 64,
            "bytes": 1,
        },
        "acquisition_teacher_receipt": {
            "relative_path": "receipt",
            "sha256": "6" * 64,
            "bytes": 1,
        },
        "checkpoint_selection_frame": {
            "relative_path": "checkpoint",
            "sha256": "7" * 64,
            "bytes": 1,
        },
        "factorised_parent_manifest": {
            "relative_path": "parent-manifest",
            "sha256": "4" * 64,
            "bytes": 1,
        },
        "factorised_representation_gate": {
            "relative_path": "parent-gate",
            "sha256": "a" * 64,
            "bytes": 1,
        },
        "acquisition_config_sha256": gate.policy_file_sha256,
        "source_files": list(launcher.REQUIRED_SOURCE_FILES),
        "source_bundle_sha256": "8" * 64,
        "dependency_lock_sha256": "9" * 64,
        "rate_card_usd_per_gpu_second": "0.000222",
        "cumulative_measured_spend_usd": "30",
        "active_reservation_usd": "0",
        "planned_phase_upper_usd": "25",
        "hard_cost_cap_usd": "200",
    }


def test_launcher_pins_ten_concurrent_l4s_and_exact_runtime() -> None:
    assert launcher._resolve_local_repo_root(Path("/root/launcher.py")) is None
    assert launcher.REPO_ROOT is not None
    assert launcher.ALLOWED_GPUS == ("L4",)
    assert launcher.MAX_CONCURRENT_TRIALS == 10
    assert launcher.ACCOUNT_GPU_LIMIT == 10
    assert training.EXPECTED_TRIALS == 12
    assert training.COMPONENTS == ("relevance", "target_stance_b4")
    assert launcher.PREPARE_TIMEOUT_SECONDS == 1_800
    assert launcher.TRAIN_TIMEOUT_SECONDS == training.TRIAL_MAX_GPU_SECONDS
    assert "schemas/target-stance-v2-pilot.schema.json" in launcher.REQUIRED_SOURCE_FILES
    assert launcher.RUNTIME_DEPENDENCIES == {
        "accelerate": "1.10.1",
        "huggingface-hub": "0.36.2",
        "jsonschema": "4.26.0",
        "pyarrow": "25.0.1",
        "pydantic": "2.13.4",
        "safetensors": "0.8.0",
        "scikit-learn": "1.9.0",
        "torch": "2.8.0",
        "transformers": "4.57.6",
    }


def test_cost_guardrail_fails_closed_above_phase_or_shared_cap() -> None:
    launcher.enforce_cost_guardrail(
        estimated_cost_usd=Decimal("24.99"), approved_cost_usd=Decimal("25")
    )
    with pytest.raises(ValueError):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("1"), approved_cost_usd=Decimal("25.01")
        )
    with pytest.raises(RuntimeError):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("25"), approved_cost_usd=Decimal("24")
        )


def test_preparation_identity_binds_exact_policy_digest() -> None:
    spec = _preparation_spec()
    identity = launcher._preparation_id(spec)
    spec["acquisition_config_sha256"] = "f" * 64
    assert launcher._preparation_id(spec) != identity


def test_preparation_spec_rejects_source_inventory_or_budget_drift() -> None:
    spec = _preparation_spec()
    spec["source_files"] = spec["source_files"][:-1]
    with pytest.raises(ValueError, match="source-file inventory"):
        launcher.validate_preparation_spec(spec)
    spec = _preparation_spec()
    spec["planned_phase_upper_usd"] = "24"
    with pytest.raises(ValueError, match="cost contract"):
        launcher.validate_preparation_spec(spec)


def test_prepared_manifest_builder_verifies_source_locks_and_freezes_three_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _preparation_spec()
    bundle = launcher.factorised_experiment.build_source_bundle(
        launcher.REPO_ROOT, spec["source_files"]
    )
    spec["source_bundle_sha256"] = training.canonical_sha256(bundle)
    spec["dependency_lock_sha256"] = training.file_sha256(
        launcher.REPO_ROOT / "uv.lock"
    )
    rare_cells = [
        {
            "target": "company_tech_product",
            "stance": "no_directed_stance",
            "training_support": 9,
        },
        {"target": "government_ccp", "stance": "negative", "training_support": 7},
        {"target": "people_identity", "stance": "positive", "training_support": 8},
    ]
    receipt = {
        "preparation_spec_sha256": training.canonical_sha256(spec),
        "preparation": {
            "validated_teacher_inputs": {
                "rare_cells": rare_cells,
                "rare_cell_list_sha256": training.canonical_sha256(rare_cells),
            },
            "loss_contribution_counts_by_arm": {"random": {}, "active": {}},
        },
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        training,
        "validate_preparation_receipt",
        lambda value: value,
    )

    def freeze(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {"frozen": True}

    monkeypatch.setattr(training, "freeze_experiment_contract", freeze)
    manifest = {
        "experiment_contract": {},
        "reserved_cost_usd": "19.180800",
    }
    monkeypatch.setattr(training, "build_run_manifest", lambda _: manifest)
    monkeypatch.setattr(launcher, "validate_compute_contract", lambda _: None)
    observed, observed_bundle = launcher.freeze_prepared_manifest(
        spec=spec,
        preparation_receipt=receipt,
    )
    assert observed == manifest
    assert observed_bundle == bundle
    assert captured["rare_cells"] == rare_cells

    drifted = {**spec, "source_bundle_sha256": "f" * 64}
    drifted_receipt = {
        **receipt,
        "preparation_spec_sha256": training.canonical_sha256(drifted),
    }
    with pytest.raises(RuntimeError, match="source bundle"):
        launcher.freeze_prepared_manifest(
            spec=drifted,
            preparation_receipt=drifted_receipt,
        )


def test_prepare_manifest_executes_prepare_freeze_publish_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _preparation_spec()
    spec_path = tmp_path / "input.json"
    launcher._write_immutable_json(spec_path, spec)
    receipt = {"receipt_id": "a" * 64}
    manifest = {"reserved_cost_usd": "19.180800"}
    bundle = {"source_bundle_id": "b" * 64}
    observed: list[tuple[str, object]] = []
    monkeypatch.setattr(
        launcher,
        "prepare_frames",
        SimpleNamespace(
            remote=lambda value: observed.append(("prepare", value)) or receipt
        ),
    )
    monkeypatch.setattr(
        training,
        "validate_preparation_receipt",
        lambda value: value,
    )
    monkeypatch.setattr(
        launcher,
        "freeze_prepared_manifest",
        lambda **kwargs: (manifest, bundle),
    )
    monkeypatch.setattr(
        launcher,
        "publish_prepared_manifest",
        SimpleNamespace(
            remote=lambda *values: observed.append(("publish", values))
            or {"status": "published"}
        ),
    )
    manifest_path = tmp_path / "authority" / "run-manifest.json"
    bundle_path = manifest_path.parent / "source-bundle.json"
    receipt_path = manifest_path.parent / "preparation-receipt.json"
    result = launcher.prepare_manifest(
        preparation_spec_path=spec_path,
        manifest_path=manifest_path,
        source_bundle_path=bundle_path,
        preparation_receipt_path=receipt_path,
        approved_cost_usd=Decimal("25"),
    )
    assert result["status"] == "prepared_and_published"
    assert [name for name, _ in observed] == ["prepare", "publish"]
    assert launcher._read_json(manifest_path, where="manifest") == manifest
    assert launcher._read_json(bundle_path, where="source bundle") == bundle
    assert launcher._read_json(receipt_path, where="receipt") == receipt
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        launcher.prepare_manifest(
            preparation_spec_path=spec_path,
            manifest_path=manifest_path,
            source_bundle_path=bundle_path,
            preparation_receipt_path=receipt_path,
            approved_cost_usd=Decimal("25"),
        )


def test_cuda_preflight_receipt_is_phase_and_policy_bound() -> None:
    manifest = {
        "experiment_run_id": "1" * 64,
        "phase_run_id": "2" * 64,
        "experiment_contract": {
            "bindings": {"acquisition_config": {"sha256": "3" * 64}}
        },
    }
    body = {
        "schema_version": "1.0.0",
        "kind": launcher.CUDA_PREFLIGHT_KIND,
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "run_manifest_sha256": training.canonical_sha256(manifest),
        "policy_file_sha256": "3" * 64,
        "gpu_type": "L4",
        "cuda_available": True,
        "components_checked": list(training.COMPONENTS),
        "checkpoint_selection_rows_checked": 600,
        "acquisition_evaluation_rows_checked": 600,
        "evaluation_frames_distinct": True,
        "private_prediction_contract_checked": True,
        "publication_contract_checked": True,
        "invalid_outputs": 0,
        "locked_test_rows_accessed": 0,
    }
    receipt = {**body, "receipt_id": training.canonical_sha256(body)}
    assert launcher._validate_preflight_receipt(manifest, receipt) == receipt
    drifted = {**receipt, "policy_file_sha256": "4" * 64}
    with pytest.raises(RuntimeError):
        launcher._validate_preflight_receipt(manifest, drifted)


def test_cuda_preflight_accepts_exact_finite_loss_mapping() -> None:
    torch = pytest.importorskip("torch")
    loss = torch.tensor(1.0)

    assert (
        launcher._validate_cuda_model_output(
            {"loss": loss},
            torch_module=torch,
        )
        is loss
    )
    for output in (
        SimpleNamespace(loss=loss),
        {},
        {"loss": torch.tensor(float("nan"))},
        {"loss": torch.ones(2)},
    ):
        with pytest.raises(RuntimeError, match="loss"):
            launcher._validate_cuda_model_output(output, torch_module=torch)


def test_durable_launch_claims_are_exactly_once_and_timeout_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = {
        "trial_id": "4" * 64,
        "max_gpu_seconds": training.TRIAL_MAX_GPU_SECONDS,
    }
    manifest = {
        "experiment_run_id": "1" * 64,
        "phase_run_id": "2" * 64,
        "experiment_contract": {},
        "reserved_cost_usd": "1.00",
        "trials": [trial],
    }
    monkeypatch.setattr(training, "validate_run_manifest", lambda value: value)
    monkeypatch.setattr(launcher, "validate_compute_contract", lambda _: None)
    claims = launcher.create_launch_claims(
        manifest,
        approved_cost_usd=Decimal("2"),
        volume_root=tmp_path,
    )
    assert claims[trial["trial_id"]]["max_gpu_seconds"] == (
        launcher.TRAIN_TIMEOUT_SECONDS
    )
    with pytest.raises(RuntimeError, match="launch claim already exists"):
        launcher.create_launch_claims(
            manifest,
            approved_cost_usd=Decimal("2"),
            volume_root=tmp_path,
        )


def test_dispatch_receipt_exactly_binds_every_registered_function_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trials = [{"trial_id": f"{index:064x}"} for index in range(12)]
    manifest = {
        "experiment_run_id": "1" * 64,
        "phase_run_id": "2" * 64,
        "trials": trials,
    }
    monkeypatch.setattr(training, "validate_run_manifest", lambda value: value)
    submissions = [
        {
            "trial_id": trial["trial_id"],
            "function_call_id": f"fc-call-{index}",
        }
        for index, trial in enumerate(trials)
    ]

    receipt = launcher._dispatch_receipt(manifest, submissions)

    assert receipt["dispatch_mode"] == "detached_ephemeral_app"
    assert launcher._validate_dispatch_receipt(manifest, receipt) == receipt
    with pytest.raises(RuntimeError, match="exactly cover"):
        launcher._dispatch_receipt(manifest, submissions[:-1])
    drifted = {**receipt, "dispatch_mode": "ephemeral_app"}
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher._validate_dispatch_receipt(manifest, drifted)


def test_cli_has_no_b2_calibration_locked_or_retry_action() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "target_stance_b2" not in source
    assert "gpu_fallback_allowed\") is not False" in source
    assert 'action == "retry"' not in source
    assert 'action == "locked"' not in source
    assert 'action == "calibrate"' not in source
