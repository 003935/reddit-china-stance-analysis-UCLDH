from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import reddit_china_stance.modal_modernbert_cascade as launcher


def _contract(*, estimate: str = "15", approved: str = "20") -> dict[str, object]:
    return {
        "bindings": {
            "source_bundle_sha256": "a" * 64,
            "code_sha256": "b" * 64,
            "dependency_lock_sha256": "c" * 64,
            "model_id": "answerdotai/ModernBERT-large",
            "model_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        },
        "compute": {
            "allowed_gpus": ["L4"],
            "account_gpu_limit": 10,
            "max_concurrent_trials": 6,
            "gpu_fallback_allowed": False,
            "planned_upper_cost_usd": estimate,
            "approved_cost_usd": approved,
        },
    }


def _manifest() -> dict[str, object]:
    trials = []
    for component in ("relevance", "target_conditioned"):
        for index, seed in enumerate((47, 61, 89)):
            trials.append(
                {
                    "trial_id": f"{len(trials) + 1:064x}",
                    "component": component,
                    "optimiser_seed": seed,
                    "ladder_seed": (101, 202, 303)[index],
                    "gpu_type": "L4",
                }
            )
    return {
        "experiment_run_id": "d" * 64,
        "phase_run_id": "e" * 64,
        "experiment_contract": _contract(),
        "phase": "confirmation",
        "trials": trials,
        "locked_test": {"rows_accessed": 0},
    }


def test_launcher_has_exact_cascade_namespace_cost_and_gpu_boundary() -> None:
    assert launcher.APP_NAME == "reddit-china-stance-modernbert-cascade-v1"
    assert Path("student-modernbert-cascade-v1") == launcher.OUTPUT_PREFIX
    assert launcher.MAX_CONCURRENT_TRIALS == 6
    assert launcher.ACCOUNT_GPU_LIMIT == 10
    assert Decimal("20") == launcher.HARD_MAX_APPROVAL_USD
    assert launcher.ALLOWED_GPUS == ("L4",)
    assert launcher.MAX_CONCURRENT_TRIALS <= launcher.ACCOUNT_GPU_LIMIT


def test_cost_guardrail_fails_closed_at_twenty_dollars() -> None:
    launcher.enforce_cost_guardrail(
        estimated_cost_usd=Decimal("20"), approved_cost_usd=Decimal("20")
    )
    with pytest.raises(RuntimeError, match="exceeds approved"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("15.01"), approved_cost_usd=Decimal("15")
        )
    with pytest.raises(ValueError, match="<= 20"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("1"), approved_cost_usd=Decimal("20.01")
        )


def test_compute_contract_requires_six_l4_workers_and_no_fallback() -> None:
    launcher.validate_compute_contract(_contract())
    wrong = _contract()
    wrong["compute"]["max_concurrent_trials"] = 7  # type: ignore[index]
    with pytest.raises(ValueError, match="binding drifted"):
        launcher.validate_compute_contract(wrong)
    too_expensive = _contract(estimate="20.01")
    with pytest.raises(RuntimeError, match="exceeds approved"):
        launcher.validate_compute_contract(too_expensive)


def test_manifest_inventory_is_exact_component_by_seed_cross_product() -> None:
    manifest = _manifest()
    launcher._validate_manifest_inventory(manifest)
    malformed = _manifest()
    malformed["trials"] = malformed["trials"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly six"):
        launcher._validate_manifest_inventory(malformed)
    duplicate = _manifest()
    duplicate["trials"][-1]["optimiser_seed"] = 61  # type: ignore[index]
    with pytest.raises(ValueError, match="two-component by three-seed"):
        launcher._validate_manifest_inventory(duplicate)


def test_manifest_inventory_rejects_gpu_or_locked_test_drift() -> None:
    wrong_gpu = _manifest()
    wrong_gpu["trials"][0]["gpu_type"] = "A100"  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly L4"):
        launcher._validate_manifest_inventory(wrong_gpu)
    locked = _manifest()
    locked["locked_test"]["rows_accessed"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="zero locked-test"):
        launcher._validate_manifest_inventory(locked)


def test_source_bundle_verification_detects_inventory_and_lock_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    source = root / "src/reddit_china_stance/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    lock = root / "uv.lock"
    lock.write_text("lock-v1\n", encoding="utf-8")
    manifest_path = root / "data/private-modernbert-cascade-v1/run-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    files = {"src/reddit_china_stance/example.py": launcher._file_sha256(source)}
    bundle = {
        "source_glob": "src/reddit_china_stance/*.py",
        "files": files,
        "code_sha256": launcher._canonical_sha256(files),
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
    monkeypatch.setattr(launcher, "_repo_root", lambda: root)
    launcher.verify_frozen_source_bundle(manifest_path=manifest_path, manifest=manifest)
    (source.parent / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory changed"):
        launcher.verify_frozen_source_bundle(manifest_path=manifest_path, manifest=manifest)


def test_cuda_preflight_receipt_is_phase_and_source_bound(tmp_path: Path) -> None:
    manifest = _manifest()
    with pytest.raises(RuntimeError, match="requires an exact CUDA preflight"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)
    bindings = manifest["experiment_contract"]["bindings"]
    receipt = {
        "status": "passed",
        "kind": "modernbert-cascade-cuda-preflight-v1",
        "gpu_type": "L4",
        "synthetic_rows": 2,
        "synthetic_sequence_length": 16,
        "relevance_shape": [2, 3],
        "target_conditioned_shape": [2, 6],
        "component_gradients": {"relevance": True, "target_conditioned": True},
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        **{key: bindings[key] for key in (
            "source_bundle_sha256",
            "code_sha256",
            "dependency_lock_sha256",
            "model_id",
            "model_revision",
        )},
    }
    path = launcher._cuda_preflight_receipt_path(manifest, volume_root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path) == receipt
    receipt["target_conditioned_shape"] = [2, 5]
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)


def test_launcher_has_no_locked_test_or_retry_authority() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8").lower()
    assert "modal_modernbert_locked" not in source
    assert "modernbert_locked_test" not in source
    assert "retry" not in source
    assert "max_containers=max_concurrent_trials" in source
    assert "coordinate_training.remote(dict(manifest))" in source
    assert 'summary["submission"] = _submit_training(manifest)' in source
    assert "closeout_confirmation.remote(manifest, baseline)" in source


def test_mutating_confirmation_token_is_phase_bound() -> None:
    manifest = _manifest()
    assert launcher._confirmation("train", manifest) == (
        "TRAIN_MODERNBERT_CASCADE_eeeeeeeeeeee"
    )
    assert launcher._confirmation("closeout", manifest) != launcher._confirmation(
        "train", manifest
    )


def test_local_training_dispatch_does_not_read_the_remote_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    observed: list[dict[str, object]] = []

    class RemoteCoordinator:
        @staticmethod
        def remote(value: dict[str, object]) -> dict[str, object]:
            observed.append(value)
            return {"status": "submitted", "submitted_function_call_ids": []}

    def fail_local_receipt_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local dispatch must not read the Modal Volume")

    monkeypatch.setattr(launcher, "coordinate_training", RemoteCoordinator())
    monkeypatch.setattr(
        launcher,
        "validate_cuda_preflight_receipt",
        fail_local_receipt_read,
    )

    result = launcher._submit_training(manifest)

    assert result["status"] == "submitted"
    assert observed == [manifest]
