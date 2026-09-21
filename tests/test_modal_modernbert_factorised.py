from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import reddit_china_stance.modal_modernbert_factorised as launcher
from reddit_china_stance import modernbert_factorised_experiment as experiment


def _compute_contract() -> dict[str, object]:
    return {
        "compute": {
            "allowed_gpus": ["L4"],
            "account_gpu_limit": 10,
            "max_concurrent_trials": 9,
            "gpu_fallback_allowed": False,
            "rate_card_usd_per_gpu_second": {"L4": "0.000222"},
            "cumulative_measured_spend_usd": "40",
            "active_reservation_usd": "10",
            "planned_phase_upper_usd": "50",
            "hard_cost_cap_usd": "200",
            "remaining_after_plan_usd": "100",
        }
    }


def _inventory_manifest() -> dict[str, object]:
    return {
        "phase": "fixed-representation-comparison",
        "trials": [
            {
                "trial_id": f"{index:064x}",
                "component": component,
                "optimiser_seed": seed,
                "gpu_type": "L4",
            }
            for index, (component, seed) in enumerate(
                (
                    (component, seed)
                    for component in (
                        "relevance",
                        "target_stance_b4",
                        "target_stance_b2",
                    )
                    for seed in (47, 61, 89)
                ),
                start=1,
            )
        ],
        "locked_test_rows_accessed": 0,
    }


def _scorer_public_metrics(component: str) -> dict[str, object]:
    counts = {
        "development_context_available": 75,
        "development_multi_target": 75,
        "development_probability": 300,
        "development_rare_target_stance": 150,
    }
    diagnostics: dict[str, dict[str, object]] = {}
    for selection, row_count in counts.items():
        surface: dict[str, object] = {
            "row_count": row_count,
            "evidence_scope": (
                "design-weighted-natural-probability-arm"
                if selection == "development_probability"
                else "unweighted-diagnostic-only"
            ),
        }
        if component == "relevance":
            surface |= {"relevance_macro_f1": 0.5, "material_recall": 0.5}
        else:
            surface |= {"conditional_tuple_micro_f1": 0.5}
        diagnostics[selection] = surface
    metrics: dict[str, object] = {
        "selected_epoch": 1,
        "development_rows": 600,
        "primary_probability_rows": 300,
        "invalid_outputs": 0,
        "checkpoint_score": 0.5,
        "natural_arm_weighting_config_digest": "2" * 64,
        "probability_design": {"design_digest": "1" * 64},
        "development_component_metrics": diagnostics,
    }
    if component == "relevance":
        metrics |= {"relevance_macro_f1": 0.5, "material_recall": 0.5}
    else:
        metrics |= {"conditional_tuple_micro_f1": 0.5}
    return metrics


def test_launcher_pins_nine_l4_jobs_and_exact_runtime_dependencies() -> None:
    assert launcher.APP_NAME == "reddit-china-stance-modernbert-factorised-v2"
    assert launcher.VOLUME_NAME == "reddit-china-stance-data"
    assert launcher.ENVIRONMENT_NAME == "main"
    assert Path("student-modernbert-factorised-v2") == launcher.OUTPUT_PREFIX
    assert launcher.MAX_CONCURRENT_TRIALS == 9
    assert launcher.ACCOUNT_GPU_LIMIT == 10
    assert launcher.MAX_CONCURRENT_TRIALS < launcher.ACCOUNT_GPU_LIMIT
    assert launcher.ALLOWED_GPUS == ("L4",)
    assert launcher.RUNTIME_DEPENDENCIES["jsonschema"] == "4.26.0"
    assert launcher.RUNTIME_DEPENDENCIES["pyarrow"] == "25.0.1"
    assert launcher.RUNTIME_DEPENDENCIES["torch"] == "2.8.0"
    assert launcher.RUNTIME_DEPENDENCIES["transformers"] == "4.57.6"
    assert launcher.RUNTIME_FILE_MOUNTS == {
        "docs/rubrics/target-stance-v2-pilot.md": "/docs/rubrics/target-stance-v2-pilot.md",
        "schemas/target-stance-v2-pilot.schema.json": "/schemas/target-stance-v2-pilot.schema.json",
    }


def test_cost_and_compute_contract_fail_closed_at_200_dollars() -> None:
    launcher.enforce_cost_guardrail(
        estimated_cost_usd=Decimal("200"), approved_cost_usd=Decimal("200")
    )
    launcher.validate_compute_contract(_compute_contract())

    with pytest.raises(RuntimeError, match="exceeds approved"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("51"), approved_cost_usd=Decimal("50")
        )
    with pytest.raises(ValueError, match="<= 200"):
        launcher.enforce_cost_guardrail(
            estimated_cost_usd=Decimal("1"), approved_cost_usd=Decimal("201")
        )
    drifted = _compute_contract()
    drifted["compute"]["max_concurrent_trials"] = 10  # type: ignore[index]
    with pytest.raises(ValueError, match="concurrency binding drifted"):
        launcher.validate_compute_contract(drifted)
    over_cap = _compute_contract()
    over_cap["compute"]["cumulative_measured_spend_usd"] = "151"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="total cost cap"):
        launcher.validate_compute_contract(over_cap)

    negative = _compute_contract()
    negative["compute"]["cumulative_measured_spend_usd"] = "-500"  # type: ignore[index]
    negative["compute"]["remaining_after_plan_usd"] = "640"  # type: ignore[index]
    with pytest.raises(ValueError, match="finite and non-negative"):
        launcher.validate_compute_contract(negative)

    nonfinite = _compute_contract()
    nonfinite["compute"]["rate_card_usd_per_gpu_second"]["L4"] = "NaN"  # type: ignore[index]
    with pytest.raises(ValueError, match="positive and finite"):
        launcher.validate_compute_contract(nonfinite)

    mismatched_remaining = _compute_contract()
    mismatched_remaining["compute"]["remaining_after_plan_usd"] = "999"  # type: ignore[index]
    with pytest.raises(ValueError, match="remaining cost binding drifted"):
        launcher.validate_compute_contract(mismatched_remaining)


def test_manifest_inventory_is_exact_three_components_by_three_seeds() -> None:
    manifest = _inventory_manifest()
    launcher._validate_manifest_inventory(manifest)

    duplicate = _inventory_manifest()
    duplicate["trials"][0]["optimiser_seed"] = 61  # type: ignore[index]
    with pytest.raises(ValueError, match="exact L4"):
        launcher._validate_manifest_inventory(duplicate)
    wrong_gpu = _inventory_manifest()
    wrong_gpu["trials"][0]["gpu_type"] = "A100"  # type: ignore[index]
    with pytest.raises(ValueError, match="exact L4"):
        launcher._validate_manifest_inventory(wrong_gpu)
    locked = _inventory_manifest()
    locked["locked_test_rows_accessed"] = 1
    with pytest.raises(ValueError, match="zero locked-test"):
        launcher._validate_manifest_inventory(locked)


def test_listed_source_bundle_tolerates_unrelated_extra_files_but_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    for relative in launcher.REQUIRED_SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen {relative}\n", encoding="utf-8")
    (repo / "src/reddit_china_stance/unrelated.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    lock = repo / "uv.lock"
    lock.write_text("lock-v1\n", encoding="utf-8")
    bundle = experiment.build_source_bundle(repo, launcher.REQUIRED_SOURCE_FILES)
    manifest = {
        "experiment_contract": {
            "bindings": {
                "source_bundle": bundle,
                "source_bundle_sha256": launcher._canonical_sha256(bundle),
                "dependency_lock_sha256": launcher._file_sha256(lock),
            }
        }
    }
    manifest_path = repo / "data/private/run-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    monkeypatch.setattr(launcher, "_repo_root", lambda: repo)

    launcher.verify_frozen_source_bundle(
        manifest_path=manifest_path, manifest=manifest
    )
    changed = repo / launcher.REQUIRED_SOURCE_FILES[0]
    changed.write_text("# drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen source changed"):
        launcher.verify_frozen_source_bundle(
            manifest_path=manifest_path, manifest=manifest
        )


def test_preparation_identity_binds_source_bundle_and_dependency_lock(
    tmp_path: Path,
) -> None:
    spec = {
        field: {"binding": field}
        for field in launcher.PREPARATION_IDENTITY_FIELDS
    } | {
        "source_bundle_sha256": "a" * 64,
        "dependency_lock_sha256": "b" * 64,
    }
    original = launcher._preparation_id_from_spec(spec)
    source_changed = launcher._preparation_id_from_spec(
        {**spec, "source_bundle_sha256": "c" * 64}
    )
    lock_changed = launcher._preparation_id_from_spec(
        {**spec, "dependency_lock_sha256": "d" * 64}
    )
    rubric_changed = launcher._preparation_id_from_spec(
        {**spec, "rubric": {"binding": "changed-rubric"}}
    )
    schema_changed = launcher._preparation_id_from_spec(
        {**spec, "schema": {"binding": "changed-schema"}}
    )
    assert len(
        {original, source_changed, lock_changed, rubric_changed, schema_changed}
    ) == 5

    stale_root = launcher._prepared_output_root(original, volume_root=tmp_path)
    stale_root.mkdir(parents=True)
    (stale_root / "stale-marker.json").write_text("{}\n", encoding="utf-8")
    assert launcher._prepared_output_root(
        source_changed, volume_root=tmp_path
    ) != stale_root
    assert not launcher._prepared_output_root(
        source_changed, volume_root=tmp_path
    ).exists()
    assert launcher._prepared_output_root(lock_changed, volume_root=tmp_path) != stale_root


def test_launch_requires_phase_bound_cuda_preflight_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "experiment_run_id": "a" * 64,
        "phase_run_id": "b" * 64,
        "experiment_contract": {
            "bindings": {
                "source_bundle_sha256": "c" * 64,
                "dependency_lock_sha256": "d" * 64,
                "model_id": experiment.MODEL_ID,
                "model_revision": experiment.MODEL_REVISION,
                "development_frame": {
                    "relative_path": "prepared/development.parquet",
                    "sha256": "f" * 64,
                    "bytes": 123,
                    "row_count": 600,
                    "frame": "development",
                },
            }
        },
    }
    real_evidence = {
        "development_frame_sha256": "f" * 64,
        "development_frame_value": "development",
        "development_rows": 600,
        "natural_design_digest": "1" * 64,
        "natural_arm_weighting_config_digest": "2" * 64,
        "component_checkpoint_scoring": {},
        "publication_contract_validation": {
            "status": "passed",
            "validator": "build_trial_receipt+validate_trial_artifacts",
            "components": {
                component: {
                    "status": "passed",
                    "validator": "build_trial_receipt+validate_trial_artifacts",
                    "public_metrics_sha256": character * 64,
                    "trial_spec_sha256": "6" * 64,
                    "receipt_id": "7" * 64,
                    "artifact_bindings_sha256": "8" * 64,
                }
                for component, character in (
                    ("relevance", "3"),
                    ("target_stance_b4", "4"),
                    ("target_stance_b2", "5"),
                )
            },
        },
    }
    monkeypatch.setattr(
        launcher,
        "_real_development_preflight_evidence",
        lambda *_args, **_kwargs: real_evidence,
    )
    with pytest.raises(RuntimeError, match="requires an exact CUDA preflight"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)
    receipt = {
        "status": "passed",
        "kind": "modernbert-factorised-cuda-preflight-v3",
        "gpu_type": "L4",
        "experiment_run_id": manifest["experiment_run_id"],
        "phase_run_id": manifest["phase_run_id"],
        "source_bundle_sha256": "c" * 64,
        "dependency_lock_sha256": "d" * 64,
        "model_id": experiment.MODEL_ID,
        "model_revision": experiment.MODEL_REVISION,
        "synthetic_rows": 2,
        "synthetic_sequence_length": 16,
        "output_shapes": {
            "relevance": [2],
            "target_stance_b4": {"target": [2, 6], "stance": [2, 5, 4]},
            "target_stance_b2": {"target": [2, 6], "stance": [2, 5, 2]},
        },
        "component_gradients": {
            "relevance": True,
            "target_stance_b4": True,
            "target_stance_b2": True,
        },
        "real_development_scoring": json.loads(json.dumps(real_evidence)),
        "locked_test_rows_accessed": 0,
    }
    path = launcher._preflight_path(manifest, volume_root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)
    receipt["kind"] = launcher.CUDA_PREFLIGHT_RECEIPT_KIND
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert launcher.validate_cuda_preflight_receipt(
        manifest, volume_root=tmp_path
    ) == receipt
    receipt["real_development_scoring"]["publication_contract_validation"][  # type: ignore[index]
        "components"
    ].pop("target_stance_b2")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)
    receipt["real_development_scoring"] = real_evidence
    receipt["phase_run_id"] = "e" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding drifted"):
        launcher.validate_cuda_preflight_receipt(manifest, volume_root=tmp_path)


def test_real_frame_preflight_loads_development_and_scores_all_ipw_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    for index in range(600):
        selection = (
            "development_probability"
            if index < 300
            else "development_context_available"
        )
        rows.append(
            {
                "item_id": f"item-{index:03d}",
                "frame": "development",
                "selection_component": selection,
            }
        )
    reference = {row["item_id"]: {} for row in rows}
    natural_design = {
        row["item_id"]: {"frame": "development"} for row in rows[:300]
    }
    calls: list[tuple[str, int, int]] = []
    publication_calls: list[tuple[str, dict[str, object]]] = []
    receipt_calls: list[str] = []

    class Weighting:
        def digest(self) -> str:
            return "2" * 64

    def load_private_frame(
        path: Path, descriptor: dict, *, expected_frame: str
    ) -> tuple[list[dict], dict[str, dict]]:
        assert path == tmp_path / "prepared/development.parquet"
        assert descriptor["frame"] == "development"
        assert expected_frame == "development"
        return rows, reference

    def score_checkpoint(**kwargs: object) -> dict:
        component = str(kwargs["component"])
        calls.append(
            (
                component,
                len(kwargs["private_rows"]),  # type: ignore[arg-type]
                len(kwargs["natural_design"]),  # type: ignore[arg-type]
            )
        )
        assert kwargs["selected_epoch"] == 1
        return _scorer_public_metrics(component)

    def validate_public_trial_metrics(
        value: dict[str, object], *, component: str
    ) -> dict[str, object]:
        publication_calls.append((component, value))
        assert all(
            "evidence_scope" in surface
            for surface in value["development_component_metrics"].values()  # type: ignore[union-attr]
        )
        return dict(value)

    def write_json(path: Path, value: dict[str, object]) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def build_trial_receipt(
        _contract: dict[str, object],
        trial: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        receipt_calls.append(str(trial["component"]))
        return {
            "receipt_id": str(trial["component"])[0] * 64,
            "artifacts": kwargs["artifacts"],
        }

    fake_training = SimpleNamespace(
        load_private_frame=load_private_frame,
        _load_natural_development_weighting=lambda **_kwargs: (
            Weighting(),
            {"design_digest": "1" * 64},
        ),
        _natural_design_from_development_rows=lambda value: (
            natural_design if value is rows else (_ for _ in ()).throw(AssertionError())
        ),
        _score_checkpoint=score_checkpoint,
        build_private_development_predictions=lambda *_args, rows: {"rows": rows},
        _write_json_atomic=write_json,
        TRIAL_METRICS_KIND="trial-metrics",
        validate_trial_artifacts=lambda _root, receipt, **_kwargs: receipt,
    )
    monkeypatch.setattr(launcher, "_training_module", lambda: fake_training)
    monkeypatch.setattr(
        launcher,
        "_experiment_module",
        lambda: SimpleNamespace(
            validate_public_trial_metrics=validate_public_trial_metrics,
            build_trial_receipt=build_trial_receipt,
        ),
    )
    manifest = {
        "phase_run_id": "a" * 64,
        "experiment_contract": {
            "bindings": {
                "development_frame": {
                    "relative_path": "prepared/development.parquet",
                    "sha256": "f" * 64,
                    "bytes": 123,
                    "row_count": 600,
                    "frame": "development",
                }
            }
        },
        "trials": [
            {"component": component, "trial_id": component}
            for component in ("relevance", "target_stance_b4", "target_stance_b2")
        ],
    }
    evidence = launcher._real_development_preflight_evidence(
        manifest, volume_root=tmp_path
    )
    assert calls == [
        ("relevance", 600, 300),
        ("target_stance_b4", 600, 300),
        ("target_stance_b2", 600, 300),
    ]
    assert [component for component, _ in publication_calls] == [
        "relevance",
        "target_stance_b4",
        "target_stance_b2",
    ]
    assert receipt_calls == [
        "relevance",
        "target_stance_b4",
        "target_stance_b2",
    ]
    assert evidence["development_frame_value"] == "development"
    assert evidence["development_rows"] == 600
    assert set(evidence["component_checkpoint_scoring"]) == {
        "relevance",
        "target_stance_b4",
        "target_stance_b2",
    }
    publication = evidence["publication_contract_validation"]
    assert publication["status"] == "passed"
    assert publication["validator"] == "build_trial_receipt+validate_trial_artifacts"
    assert set(publication["components"]) == {
        "relevance",
        "target_stance_b4",
        "target_stance_b2",
    }


@pytest.mark.parametrize(
    "component", ("relevance", "target_stance_b4", "target_stance_b2")
)
def test_cpu_preflight_round_trips_exact_scorer_metrics_through_publication_contract(
    component: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _scorer_public_metrics(component)
    receipt = {
        "receipt_id": "a" * 64,
        "artifacts": {
            "checkpoint": {"relative_path": "checkpoint.pt"},
            "metrics": {"relative_path": "metrics.json"},
            "private_development_predictions": {
                "relative_path": "development-predictions.json"
            },
        },
    }
    monkeypatch.setattr(
        launcher,
        "_training_module",
        lambda: SimpleNamespace(
            build_private_development_predictions=lambda *_args, rows: {"rows": rows},
            _write_json_atomic=lambda path, value: path.write_text(
                json.dumps(value), encoding="utf-8"
            ),
            TRIAL_METRICS_KIND="trial-metrics",
            validate_trial_artifacts=lambda _root, value, **_kwargs: value,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_experiment_module",
        lambda: SimpleNamespace(
            validate_public_trial_metrics=lambda value, **_kwargs: dict(value),
            build_trial_receipt=lambda *_args, **_kwargs: receipt,
        ),
    )
    manifest = {"experiment_contract": {}, "phase_run_id": "b" * 64}
    trial = {"component": component, "trial_id": component}
    evidence = launcher._validate_preflight_publication_contract(
        metrics,
        component=component,
        manifest=manifest,
        trial=trial,
        private_rows=[{"item_id": "private"}],
    )
    assert evidence["status"] == "passed"
    assert evidence["validator"] == "build_trial_receipt+validate_trial_artifacts"
    assert evidence["public_metrics_sha256"] == launcher._canonical_sha256(metrics)
    assert evidence["receipt_id"] == "a" * 64

    prior_mismatched_shape = json.loads(json.dumps(metrics))
    del prior_mismatched_shape["development_component_metrics"][
        "development_probability"
    ]["evidence_scope"]
    with pytest.raises(ValueError, match="metric schema drifted"):
        experiment.validate_public_trial_metrics(
            prior_mismatched_shape, component=component
        )


def test_immutable_claim_writer_is_idempotent_and_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "claim.json"
    payload = {"trial_id": "a" * 64, "status": "claimed_before_spawn"}
    launcher._write_immutable_json(path, payload)
    launcher._write_immutable_json(path, payload)
    with pytest.raises(RuntimeError, match="immutable state differs"):
        launcher._write_immutable_json(
            path, {**payload, "status": "different"}
        )


def test_modal_wrappers_commit_writes_and_reload_before_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class SpyVolume:
        def commit(self) -> None:
            events.append("commit")

        def reload(self) -> None:
            events.append("reload")

    def run_training_trial(**_kwargs: object) -> dict[str, str]:
        events.append("train")
        return {"status": "complete"}

    def closeout_fixed_comparison(**_kwargs: object) -> dict[str, str]:
        events.append("closeout")
        return {"status": "closed"}

    monkeypatch.setattr(launcher, "volume", SpyVolume())
    monkeypatch.setattr(
        launcher,
        "_training_module",
        lambda: SimpleNamespace(
            run_training_trial=run_training_trial,
            execute_registered_gpu_trial=lambda: None,
            closeout_fixed_comparison=closeout_fixed_comparison,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_inspect_exact_trial",
        lambda _manifest, trial_id: events.append(f"inspect:{trial_id}")
        or {"trial_id": trial_id, "status": "missing"},
    )
    monkeypatch.setattr(
        launcher,
        "_aggregate_inspections",
        lambda _manifest, _inspections: events.append("aggregate")
        or {"status": "incomplete"},
    )

    with pytest.warns(UserWarning):
        assert launcher.train_l4.local({}) == {"status": "complete"}
    assert events == ["train", "commit"]

    events.clear()
    manifest = {"trials": [{"trial_id": "one"}]}
    with pytest.warns(UserWarning):
        assert launcher.inspect_run.local(manifest) == {"status": "incomplete"}
    assert events == ["reload", "inspect:one", "aggregate"]

    events.clear()
    with pytest.warns(UserWarning):
        assert launcher.inspect_trial.local(manifest, "one")["status"] == "missing"
    assert events == ["reload", "inspect:one"]

    events.clear()
    with pytest.warns(UserWarning):
        assert launcher.closeout_comparison.local({}) == {"status": "closed"}
    assert events == ["reload", "closeout", "commit"]


def test_cli_exposes_only_phase_one_boundaries_and_no_legacy_frame_path() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    for action in (
        '"prepare"',
        '"validate"',
        '"cuda-preflight"',
        '"launch"',
        '"inspect"',
        '"validate-sharded"',
        '"closeout"',
    ):
        assert action in source
    assert "development-proxy.parquet" not in source
    assert "locked_test_candidate" not in source
    assert '"calibrate"' not in source
    assert "calibration.parquet" not in source
    assert 'gpu="L4"' in source
