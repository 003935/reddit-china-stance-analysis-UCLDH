from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from reddit_china_stance import modal_modernbert_inference_candidate_v1 as launcher
from reddit_china_stance import modernbert_inference_candidate_v1 as candidate


def _descriptor(
    name: str,
    digest: str,
    *,
    rows: int | None = None,
    frame: str | None = None,
    thread_digest: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "relative_path": f"candidate/{name}",
        "sha256": digest * 64,
        "bytes": 123,
    }
    if rows is not None:
        value["row_count"] = rows
    if frame is not None:
        value["frame"] = frame
    if thread_digest is not None:
        value["thread_set_sha256"] = thread_digest
    return value


def _spec() -> dict:
    return {
        "schema_version": "1.0.0",
        "kind": "modernbert-inference-candidate-preparation-spec-v1",
        "training_frame": _descriptor(
            "training.parquet",
            "1",
            rows=9_000,
            frame="training",
            thread_digest="a" * 64,
        ),
        "development_frame": _descriptor(
            "development.parquet",
            "2",
            rows=600,
            frame="development",
            thread_digest="b" * 64,
        ),
        "acquisition_labels": _descriptor("labels.parquet", "3", rows=2_000),
        "acquisition_receipt": _descriptor("receipt.json", "4"),
        "acquisition_run_id": "5" * 64,
        "source_files": list(launcher.REQUIRED_SOURCE_FILES),
        "source_bundle_sha256": "6" * 64,
        "dependency_lock_sha256": "7" * 64,
        "rate_card_usd_per_gpu_second": "0.000222",
        "cumulative_measured_spend_usd": "30",
        "active_reservation_usd": "0",
        "planned_phase_upper_usd": "5",
        "hard_cost_cap_usd": "200",
    }


def test_launcher_is_one_l4_candidate_with_exact_runtime() -> None:
    assert launcher.ALLOWED_GPUS == ("L4",)
    assert launcher.MAX_CONCURRENT_CANDIDATES == 1
    assert candidate.REGISTERED_SEEDS == (47,)
    assert launcher.RUNTIME_DEPENDENCIES == {
        "accelerate": "1.10.1",
        "huggingface-hub": "0.36.2",
        "jsonschema": "4.26.0",
        "pyarrow": "25.0.1",
        "pydantic": "2.13.4",
        "safetensors": "0.8.0",
        "torch": "2.8.0",
        "transformers": "4.57.6",
    }
    assert launcher._repository_root(
        Path("/root/modal_modernbert_inference_candidate_v1.py")
    ) == Path("/root")
    assert launcher.REPO_ROOT.name == "reddit-china-stance"
    assert launcher.SCHEMA_REPO_PATH in launcher.REQUIRED_SOURCE_FILES
    assert launcher.SCHEMA_RUNTIME_PATH == "/schemas/target-stance-v2-pilot.schema.json"


def test_runtime_schema_must_match_frozen_bundle_and_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_relative = Path(launcher.SCHEMA_REPO_PATH)
    schema_path = tmp_path / schema_relative
    schema_path.parent.mkdir(parents=True)
    schema_path.write_bytes((launcher.REPO_ROOT / schema_relative).read_bytes())
    bundle = launcher.factorised_experiment.build_source_bundle(
        tmp_path,
        [launcher.SCHEMA_REPO_PATH],
    )
    monkeypatch.setattr(launcher, "SCHEMA_RUNTIME_PATH", str(schema_path))

    result = launcher._validate_runtime_schema(bundle)
    assert result == {
        "status": "exact-valid",
        "runtime_path": str(schema_path),
        "sha256": bundle["files"][launcher.SCHEMA_REPO_PATH],
    }

    schema_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or corrupt"):
        launcher._validate_runtime_schema(bundle)


def test_cost_guardrail_fails_closed_above_candidate_phase_cap() -> None:
    launcher.enforce_cost_guardrail(
        estimated_cost_usd=Decimal("1.998"),
        approved_cost_usd=Decimal("5"),
    )
    with pytest.raises(ValueError, match="<= 5"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("1"),
            approved_cost_usd=Decimal("5.01"),
        )
    with pytest.raises(RuntimeError, match="exceeds"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("5"),
            approved_cost_usd=Decimal("4.99"),
        )


def test_preparation_spec_rejects_source_or_cost_drift() -> None:
    spec = _spec()
    assert launcher.validate_preparation_spec(spec) == spec
    spec["source_files"] = spec["source_files"][:-1]
    with pytest.raises(ValueError, match="source-file inventory"):
        launcher.validate_preparation_spec(spec)
    spec = _spec()
    spec["planned_phase_upper_usd"] = "5.01"
    with pytest.raises(ValueError, match="cost contract"):
        launcher.validate_preparation_spec(spec)


def test_freeze_manifest_binds_owned_source_and_one_job() -> None:
    spec = _spec()
    bundle = launcher.factorised_experiment.build_source_bundle(
        launcher.REPO_ROOT,
        spec["source_files"],
    )
    spec["source_bundle_sha256"] = candidate.canonical_sha256(bundle)
    spec["dependency_lock_sha256"] = candidate.file_sha256(launcher.REPO_ROOT / "uv.lock")
    manifest, observed_bundle = launcher.freeze_prepared_manifest(spec)
    assert observed_bundle == bundle
    assert len(manifest["training_jobs"]) == 1
    assert manifest["training_jobs"][0]["optimiser_seed"] == 47
    assert manifest["experiment_contract"]["compute"]["allowed_gpus"] == ["L4"]

    drifted = {**spec, "source_bundle_sha256": "f" * 64}
    with pytest.raises(RuntimeError, match="source bundle"):
        launcher.freeze_prepared_manifest(drifted)


def test_prepare_writes_immutable_local_authority_then_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "spec.json"
    launcher._write_immutable_json(spec_path, _spec())
    manifest = {"reserved_cost_usd": "1.998"}
    bundle = {"source_bundle_id": "a" * 64}
    monkeypatch.setattr(
        launcher,
        "freeze_prepared_manifest",
        lambda _spec_value: (manifest, bundle),
    )
    calls: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        launcher,
        "publish_manifest",
        SimpleNamespace(remote=lambda *args: calls.append(args) or {"status": "published"}),
    )
    manifest_path = tmp_path / "authority" / "run-manifest.json"
    bundle_path = manifest_path.parent / "source-bundle.json"
    result = launcher.prepare_manifest(
        preparation_spec_path=spec_path,
        manifest_path=manifest_path,
        source_bundle_path=bundle_path,
        approved_cost_usd=Decimal("5"),
    )
    assert result["status"] == "prepared_and_published"
    assert calls == [(manifest, bundle)]
    assert launcher._read_json(manifest_path, where="manifest") == manifest
    assert launcher._read_json(bundle_path, where="bundle") == bundle
    with pytest.raises(RuntimeError, match="overwrite"):
        launcher.prepare_manifest(
            preparation_spec_path=spec_path,
            manifest_path=manifest_path,
            source_bundle_path=bundle_path,
            approved_cost_usd=Decimal("5"),
        )


def test_modal_actions_expose_no_retry_fallback_sweep_locked_or_corpus_path() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert 'action == "retry"' not in source
    assert 'action == "sweep"' not in source
    assert 'action == "locked-test"' not in source
    assert 'action == "corpus-inference"' not in source
    assert 'gpu="L4"' in source
    assert "gpu_fallback_allowed" in source
    assert "max_containers=MAX_CONCURRENT_CANDIDATES" in source


def test_every_paid_l4_action_uses_one_shared_exact_run_confirmation() -> None:
    manifest = {"phase_run_id": "a" * 64}
    assert frozenset({"cuda-preflight", "train", "throughput-smoke"}) == launcher.PAID_ACTIONS
    confirmations = set()
    for action in launcher.PAID_ACTIONS:
        with pytest.raises(ValueError, match="requires --confirm"):
            launcher._require_phase_confirmation(action, manifest, "")
        confirmations.add(launcher._confirmation(action, manifest))
        launcher._require_phase_confirmation(
            action,
            manifest,
            launcher._confirmation(action, manifest),
        )
    assert confirmations == {"RUN_INFERENCE_CANDIDATE_aaaaaaaaaaaa"}


def test_cuda_preflight_dispatch_requires_confirmation_before_remote_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "run-manifest.json"
    spec = _spec()
    manifest_path.write_text(
        json.dumps(
            candidate.build_run_manifest(
                candidate.freeze_experiment_contract(
                    training_frame=spec["training_frame"],
                    development_frame=spec["development_frame"],
                    acquisition_labels=spec["acquisition_labels"],
                    acquisition_receipt=spec["acquisition_receipt"],
                    acquisition_run_id=spec["acquisition_run_id"],
                    source_bundle_sha256=spec["source_bundle_sha256"],
                    dependency_lock_sha256=spec["dependency_lock_sha256"],
                    rate_card_usd_per_gpu_second=spec["rate_card_usd_per_gpu_second"],
                    cumulative_measured_spend_usd=spec["cumulative_measured_spend_usd"],
                    active_reservation_usd=spec["active_reservation_usd"],
                    planned_phase_upper_usd=spec["planned_phase_upper_usd"],
                    hard_cost_cap_usd=spec["hard_cost_cap_usd"],
                )
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        launcher,
        "cuda_preflight",
        SimpleNamespace(
            remote=lambda *_args: pytest.fail(
                "unauthorised CUDA preflight must not reserve remote compute"
            )
        ),
    )

    with pytest.raises(ValueError, match="cuda-preflight requires --confirm"):
        launcher.main(
            action="cuda-preflight",
            manifest_path=str(manifest_path),
            approved_cost_usd="5",
            confirm="",
        )


def test_throughput_action_calls_real_pinned_measurement_contract() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "measure_real_pinned_throughput" in source
    assert "validate_training_receipt" in source
    assert "validate_development_calibration" in source
    assert "development_rows[: candidate.THROUGHPUT_SMOKE_ROWS]" in source
    assert "model_factory" not in source
