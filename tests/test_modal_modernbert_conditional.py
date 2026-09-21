from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import reddit_china_stance.modal_modernbert_conditional as launcher


def _contract(*, approved: str = "200", estimate: str = "15") -> dict[str, object]:
    return {
        "bindings": {
            "source_bundle_sha256": "c" * 64,
            "code_sha256": "d" * 64,
            "dependency_lock_sha256": "e" * 64,
            "model_id": "answerdotai/ModernBERT-large",
            "model_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        },
        "compute": {
            "allowed_gpus": ["L4"],
            "account_gpu_limit": 10,
            "max_concurrent_trials": 8,
            "gpu_fallback_allowed": False,
            "planned_upper_cost_usd": estimate,
            "approved_cost_usd": approved,
        }
    }


def _manifest(*, phase: str = "sweep", rung: int = 2) -> dict[str, object]:
    count = launcher.ASHA_COUNTS.get(rung, 8) if phase == "sweep" else 6
    return {
        "experiment_run_id": "a" * 64,
        "phase_run_id": "b" * 64,
        "experiment_contract": _contract(),
        "phase": phase,
        "asha": {"rung_epochs": rung} if phase == "sweep" else None,
        "trials": [
            {"trial_id": f"{index:064x}", "gpu_type": "L4"} for index in range(count)
        ],
    }


def test_launcher_pins_separate_app_namespace_runtime_and_gpu_boundary() -> None:
    assert launcher.APP_NAME == "reddit-china-stance-modernbert-conditional-v1"
    assert Path("student-modernbert-conditional-v1") == launcher.OUTPUT_PREFIX
    assert launcher.VOLUME_NAME == "reddit-china-stance-data"
    assert launcher.MAX_CONCURRENT_TRIALS == 8
    assert launcher.ACCOUNT_GPU_LIMIT == 10
    assert launcher.MAX_CONCURRENT_TRIALS < launcher.ACCOUNT_GPU_LIMIT
    assert launcher.ALLOWED_GPUS == ("L4",)
    assert launcher.RUNTIME_DEPENDENCIES["torch"] == "2.8.0"
    assert launcher.RUNTIME_DEPENDENCIES["transformers"] == "4.57.6"
    assert "sweep-rung2-v5/run-manifest.json" in Path(launcher.__file__).read_text(
        encoding="utf-8"
    )


def test_cost_guardrail_fails_closed_at_the_200_dollar_boundary() -> None:
    launcher.enforce_cost_guardrail(
        estimated_cost_usd=Decimal("200"), approved_cost_usd=Decimal("200")
    )
    with pytest.raises(RuntimeError, match="exceeds approved"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("15"), approved_cost_usd=Decimal("14")
        )
    with pytest.raises(ValueError, match="<= 200"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("1"), approved_cost_usd=Decimal("201")
        )


def test_compute_contract_rejects_concurrency_or_gpu_drift() -> None:
    launcher.validate_compute_contract(_contract())
    over = _contract()
    over["compute"]["max_concurrent_trials"] = 11  # type: ignore[index]
    with pytest.raises(ValueError, match="concurrency limit drifted"):
        launcher.validate_compute_contract(over)
    fallback = _contract()
    fallback["compute"]["gpu_fallback_allowed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="fallback"):
        launcher.validate_compute_contract(fallback)


def test_manifest_requires_a_content_addressed_source_bundle() -> None:
    manifest = _manifest()
    manifest["experiment_contract"]["bindings"] = {"source_bundle_sha256": "bad"}  # type: ignore[index]
    # _load_manifest exercises this with the contract validator; the direct guard
    # remains deliberately local so no private manifest fixture is needed here.
    with pytest.raises(ValueError, match="SHA-256"):
        launcher._require_sha256("bad", where="source bundle binding")


def test_source_bundle_verification_fails_closed_on_source_or_lock_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    source = repo / "src/reddit_china_stance/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    lock = repo / "uv.lock"
    lock.write_text("lock-v1\n", encoding="utf-8")
    manifest_path = repo / "data/private-modernbert-conditional-v1/sweep/run-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    files = {"src/reddit_china_stance/example.py": launcher._file_sha256(source)}
    bundle = {
        "files": files,
        "code_sha256": launcher._canonical_sha256(files),
        "source_glob": "src/reddit_china_stance/*.py",
    }
    bundle_path = manifest_path.parent / "source-bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    manifest = _manifest()
    manifest["experiment_contract"]["bindings"].update(  # type: ignore[index]
        {
            "source_bundle_sha256": launcher._file_sha256(bundle_path),
            "code_sha256": bundle["code_sha256"],
            "dependency_lock_sha256": launcher._file_sha256(lock),
        }
    )
    monkeypatch.setattr(launcher, "_repo_root", lambda: repo)
    launcher.verify_frozen_source_bundle(manifest_path=manifest_path, manifest=manifest)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen source changed"):
        launcher.verify_frozen_source_bundle(manifest_path=manifest_path, manifest=manifest)


def test_source_bundle_verification_rejects_added_or_removed_matching_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    source = repo / "src/reddit_china_stance/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    lock = repo / "uv.lock"
    lock.write_text("lock-v1\n", encoding="utf-8")
    manifest_path = repo / "data/private-modernbert-conditional-v1/sweep/run-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    files = {"src/reddit_china_stance/example.py": launcher._file_sha256(source)}
    bundle = {
        "files": files,
        "code_sha256": launcher._canonical_sha256(files),
        "source_glob": "src/reddit_china_stance/*.py",
    }
    bundle_path = manifest_path.parent / "source-bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    manifest = _manifest()
    manifest["experiment_contract"]["bindings"].update(  # type: ignore[index]
        {
            "source_bundle_sha256": launcher._file_sha256(bundle_path),
            "code_sha256": bundle["code_sha256"],
            "dependency_lock_sha256": launcher._file_sha256(lock),
        }
    )
    (source.parent / "added.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_repo_root", lambda: repo)
    with pytest.raises(RuntimeError, match="inventory changed"):
        launcher.verify_frozen_source_bundle(manifest_path=manifest_path, manifest=manifest)


def test_training_gate_requires_an_exact_persisted_cuda_preflight_receipt(tmp_path: Path) -> None:
    manifest = _manifest()
    with pytest.raises(RuntimeError, match="requires an exact CUDA preflight"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)
    bindings = manifest["experiment_contract"]["bindings"]
    receipt = {
        "status": "passed",
        "kind": "modernbert-conditional-cuda-preflight-v1",
        "gpu_type": "L4",
        "synthetic_rows": 2,
        "synthetic_sequence_length": 16,
        "relevance_shape": [2, 3],
        "target_state_shape": [2, 4, 6],
        "head_gradients": {"relevance": True, "target_state": True},
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "source_bundle_sha256": bindings["source_bundle_sha256"],
        "code_sha256": bindings["code_sha256"],
        "dependency_lock_sha256": bindings["dependency_lock_sha256"],
        "model_id": bindings["model_id"],
        "model_revision": bindings["model_revision"],
    }
    receipt_path = launcher._cuda_preflight_receipt_path(manifest, volume_root=tmp_path)
    next_phase = _manifest()
    next_phase["phase_run_id"] = "9" * 64
    assert launcher._cuda_preflight_receipt_path(
        next_phase, volume_root=tmp_path
    ) != receipt_path
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path) == receipt
    receipt["target_state_shape"] = [2, 4, 5]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)


def test_manifest_inventory_is_exact_8_to_4_to_2_and_three_seed_confirmation() -> None:
    for rung, count in launcher.ASHA_COUNTS.items():
        manifest = _manifest(rung=rung)
        launcher._validate_manifest_inventory(manifest)
        assert len(manifest["trials"]) == count
    launcher._validate_manifest_inventory(_manifest(phase="confirmation"))
    malformed = _manifest(rung=2)
    malformed["trials"] = malformed["trials"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="count drifted"):
        launcher._validate_manifest_inventory(malformed)


def test_confirmation_requires_exact_two_recipe_by_three_seed_cross_product() -> None:
    manifest = _manifest(phase="confirmation")
    for index, trial in enumerate(manifest["trials"]):
        trial["config"] = {
            "registered_config": {"config_sha256": f"{index // 3 + 1:064x}"},
            "seed": (13, 29, 47)[index % 3],
        }
    launcher._validate_manifest_inventory(manifest)
    manifest["trials"][-1]["config"]["seed"] = 13
    with pytest.raises(ValueError, match="two-recipe by three-seed"):
        launcher._validate_manifest_inventory(manifest)


def test_manifest_inventory_rejects_duplicate_or_non_l4_trial() -> None:
    duplicate = _manifest()
    duplicate["trials"][1]["trial_id"] = duplicate["trials"][0]["trial_id"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unique"):
        launcher._validate_manifest_inventory(duplicate)
    wrong_gpu = _manifest()
    wrong_gpu["trials"][0]["gpu_type"] = "A100"  # type: ignore[index]
    with pytest.raises(ValueError, match="L4"):
        launcher._validate_manifest_inventory(wrong_gpu)


def test_confirmation_tokens_bind_action_and_immutable_phase_run() -> None:
    manifest = _manifest()
    token = launcher._confirmation("train", manifest)
    assert token == "TRAIN_MODERNBERT_CONDITIONAL_SWEEP_bbbbbbbbbbbb"
    assert launcher._confirmation("promote", manifest) != token


def test_promotion_authorises_only_the_first_two_frozen_asha_rungs() -> None:
    for rung in (2, 4):
        launcher._validate_action_phase("promote", _manifest(rung=rung))
    for rung in (1, 3, 8):
        with pytest.raises(ValueError, match="non-final ASHA"):
            launcher._validate_action_phase("promote", _manifest(rung=rung))


def test_closeout_is_authorised_only_for_the_exact_confirmation_phase() -> None:
    launcher._validate_action_phase("closeout-confirmation", _manifest(phase="confirmation"))
    with pytest.raises(ValueError, match="confirmation phase"):
        launcher._validate_action_phase("closeout-confirmation", _manifest())


def test_closeout_fails_closed_without_an_exact_baseline_manifest_binding() -> None:
    with pytest.raises(RuntimeError, match="does not bind an exact baseline manifest path"):
        launcher._load_bound_baseline_manifest(_manifest(phase="confirmation"))


def test_closeout_loads_only_the_hash_bound_repository_relative_baseline_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "data/private-modernbert-v1/confirmatory/run-manifest.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text("{}", encoding="utf-8")
    manifest = _manifest(phase="confirmation")
    manifest["experiment_artefacts"] = {
        "baseline_manifest": {
            "repo_relative_path": "data/private-modernbert-v1/confirmatory/run-manifest.json",
            "sha256": launcher._file_sha256(baseline_path),
            "bytes": baseline_path.stat().st_size,
        }
    }
    monkeypatch.setattr(launcher, "_repo_root", lambda: tmp_path)
    assert launcher._load_bound_baseline_manifest(manifest) == {}
    manifest["experiment_artefacts"]["baseline_manifest"]["repo_relative_path"] = "../bad.json"
    with pytest.raises(RuntimeError, match="unsafe"):
        launcher._load_bound_baseline_manifest(manifest)


def test_closeout_rejects_a_baseline_manifest_byte_size_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "data/private-modernbert-v1/confirmatory/run-manifest.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text("{}", encoding="utf-8")
    manifest = _manifest(phase="confirmation")
    manifest["experiment_artefacts"] = {
        "baseline_manifest": {
            "repo_relative_path": "data/private-modernbert-v1/confirmatory/run-manifest.json",
            "sha256": launcher._file_sha256(baseline_path),
            "bytes": baseline_path.stat().st_size + 1,
        }
    }
    monkeypatch.setattr(launcher, "_repo_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher._load_bound_baseline_manifest(manifest)


def test_confirmation_preparation_uses_a_volume_mounted_remote_wrapper() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    start = source.index("def prepare_confirmation(")
    end = source.index("def _confirmation(", start)
    wrapper_source = source[start:end]
    assert "volume_root=VOLUME_PATH" in wrapper_source
    assert "volume.commit()" in wrapper_source
    assert "prepare_confirmation.remote(manifest)" in source


def test_operational_guards_use_short_timeout_coordinator_and_synthetic_preflight() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert launcher.TRAIN_TIMEOUT_SECONDS == 5_400
    assert "timeout=TRAIN_TIMEOUT_SECONDS" in source
    assert "def coordinate_training(" in source
    assert "max_containers=1" in source
    assert "claimed_before_spawn" in source
    assert "existing launch claim requires manual reconciliation" in source
    assert "coordinate_training.remote(manifest)" in source
    assert "def cuda_preflight(" in source
    assert "cuda_preflight.remote(manifest)" in source
    assert "validate_cuda_preflight_receipt(manifest)" in source
    assert "cuda-preflight-receipts" in source
    assert "closeout_confirmation.remote(manifest, baseline_manifest)" in source
    assert "def closeout_confirmation(" in source


def test_public_launcher_has_no_evaluation_entrypoint_or_authority() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8").lower()
    assert "modal_modernbert_locked" not in source
    assert "modernbert_locked_test" not in source
    assert "locked_test" not in source
    assert "evaluation" not in {name.lower() for name in dir(launcher)}
