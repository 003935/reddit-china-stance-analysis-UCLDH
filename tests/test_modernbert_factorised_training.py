from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reddit_china_stance import modal_modernbert_factorised as launcher
from reddit_china_stance import modernbert_factorised_experiment as experiment
from reddit_china_stance import modernbert_factorised_training as training
from reddit_china_stance import semantic_evaluation_v2 as evaluation


def _artifact(
    name: str,
    digest: str,
    *,
    row_count: int | None = None,
    thread_set_sha256: str | None = None,
    frame: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "relative_path": f"inputs/{name}",
        "sha256": digest * 64,
        "bytes": 123,
    }
    if row_count is not None:
        value["row_count"] = row_count
    if thread_set_sha256 is not None:
        value["thread_set_sha256"] = thread_set_sha256
    if frame is not None:
        value["frame"] = frame
    return value


def _contract(*, development_rows: int = 600) -> dict:
    files = {"src/reddit_china_stance/frozen.py": "1" * 64}
    bundle_body = {
        "schema_version": experiment.SCHEMA_VERSION,
        "kind": experiment.SOURCE_BUNDLE_KIND,
        "files": files,
        "code_sha256": experiment.canonical_sha256(files),
    }
    bundle = {
        **bundle_body,
        "source_bundle_id": experiment.canonical_sha256(bundle_body),
    }
    return experiment.freeze_experiment_contract(
        teacher_run_id="a" * 64,
        teacher_receipt=_artifact("teacher-receipt.json", "2"),
        teacher_ledger=_artifact("labels.parquet", "3", row_count=10_000),
        teacher_blinded_input=_artifact("blinded.json", "4", row_count=10_000),
        teacher_private_mapping=_artifact("mapping.parquet", "5", row_count=10_000),
        source_parquet=_artifact("source.parquet", "6", row_count=10_000),
        source_receipt=_artifact("source-receipt.json", "3"),
        source_metadata_mapping=_artifact("mapping.parquet", "5", row_count=10_000),
        split_manifest=_artifact("membership.json", "7", row_count=10_000),
        split_public_manifest=_artifact("public.json", "8", row_count=10_000),
        bridge_exposure_register=_artifact("bridge.json", "9", row_count=480),
        legacy_proxy=_artifact("legacy.parquet", "b", row_count=452),
        legacy_overlap_audit=_artifact("legacy-audit.json", "c"),
        training_frame=_artifact(
            "training.parquet",
            "d",
            row_count=8_100,
            thread_set_sha256="d" * 64,
            frame="training",
        ),
        development_frame=_artifact(
            "development.parquet",
            "e",
            row_count=development_rows,
            thread_set_sha256="e" * 64,
            frame="development",
        )
        | {
            "primary_probability_row_count": 300,
            "selection_component_counts": {
                "development_context_available": 75,
                "development_multi_target": 75,
                "development_probability": 300,
                "development_rare_target_stance": 150,
            },
        },
        development_probability_design={
            "component": "development_probability",
            "design_digest": "6" * 64,
            "expected_sample_rows": 300,
            "represented_population_rows": 8_000,
            "expected_strata": 20,
            "expected_probability_min": 0.02,
            "expected_probability_max": 0.08,
            "design_effective_sample_size": 240.0,
            "inverse_weight_ratio": 4.0,
        },
        rubric={
            "repo_relative_path": "docs/rubrics/target-stance-v2-pilot.md",
            "sha256": "f" * 64,
            "bytes": 10,
        },
        schema={
            "repo_relative_path": "schemas/target-stance-v2-pilot.schema.json",
            "sha256": "0" * 64,
            "bytes": 10,
        },
        bridge_authorisation=_artifact("bridge-receipt.json", "1"),
        source_bundle=bundle,
        dependency_lock_sha256="2" * 64,
        rate_card_usd_per_gpu_second={"L4": "0.000222"},
        cumulative_measured_spend_usd="10",
        active_reservation_usd="5",
        planned_phase_upper_usd="30",
        hard_cost_cap_usd="200",
    )


def _manifest() -> dict:
    contract = _contract()
    trials = [
        experiment.freeze_trial_spec(
            contract,
            component=component,
            optimiser_seed=seed,
            max_gpu_seconds=3_600,
        )
        for component in experiment.COMPONENTS
        for seed in experiment.REGISTERED_SEEDS
    ]
    return experiment.build_run_manifest(contract, trials=trials)


class _Tokenizer:
    sep_token = "[SEP]"

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {
            "input_ids": [101, *([7] * min(len(text), 8)), 102],
            "attention_mask": [1] * (min(len(text), 8) + 2),
        }

    def pad(
        self,
        rows: list[dict[str, list[int]]],
        **_kwargs: object,
    ) -> dict[str, list[list[int]]]:
        width = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in rows],
            "attention_mask": [
                row["attention_mask"] + [0] * (width - len(row["attention_mask"]))
                for row in rows
            ],
        }


def _feature(item_id: str, label: dict) -> dict:
    return training.tokenise_factorised_record(
        _Tokenizer(),
        item_id=item_id,
        row={
            "target_text": "China policy statement",
            "parent_context": None,
            "submission_context": "Discussion context",
        },
        label=label,
    )


def test_collators_preserve_upstream_masks_and_component_boundaries() -> None:
    not_codable = _feature(
        "not-codable",
        {"codability": "not_codable", "relevance": None, "targets": []},
    )
    material = _feature(
        "material",
        {
            "codability": "codable",
            "relevance": "material",
            "targets": [{"target": "government_ccp", "stance": "mixed"}],
        },
    )

    relevance = training.FactorisedDynamicPaddingCollator(
        _Tokenizer(), component="relevance", return_tensors=None
    )([not_codable, material])
    assert relevance["relevance_labels"] == [-100, 1]
    assert relevance["codable_mask"] == [False, True]
    assert "target_presence_labels" not in relevance

    b4 = training.FactorisedDynamicPaddingCollator(
        _Tokenizer(), component="target_stance_b4", return_tensors=None
    )([not_codable, material])
    assert b4["reference_material_mask"] == [False, True]
    assert b4["target_presence_labels"][1] == [0, 1, 0, 0, 0, 0]
    assert b4["stance_known_mask"][1] == [False, True, False, False, False]
    assert b4["stance_labels"][1][1] == 1
    assert "relevance_labels" not in b4

    b2 = training.FactorisedDynamicPaddingCollator(
        _Tokenizer(), component="target_stance_b2", return_tensors=None
    )([material])
    assert b2["stance_labels"][0][1] == [1, 1]


def test_optimisation_config_is_exact_and_rejects_registered_drift() -> None:
    manifest = _manifest()
    relevance = next(
        trial for trial in manifest["trials"] if trial["component"] == "relevance"
    )
    config = training.build_optimisation_config(relevance)
    assert config.encoder_learning_rate == 5e-5
    assert config.head_learning_rate_multiplier == 5.0
    assert config.effective_batch_size == 32
    assert config.use_bf16 is True

    drifted = deepcopy(relevance)
    drifted["config"]["encoder_learning_rate"] = 1e-4
    with pytest.raises(experiment.FactorisedExperimentContractError, match="configuration"):
        training.build_optimisation_config(drifted)


def test_source_and_label_digests_bind_text_context_and_semantics() -> None:
    source = {
        "sample_id": "sample-1",
        "thread_id": "thread-1",
        "target_text": "target",
        "submission_context": None,
        "parent_context": "parent",
        "subreddit": "news",
        "year": 2024,
        "content_type": "comment",
        "retrieval_mode": "lexical",
    }
    source_digest = training.source_row_sha256(source)
    changed_source = {**source, "parent_context": "different parent"}
    assert training.source_row_sha256(changed_source) != source_digest

    label = {
        "codability": "codable",
        "relevance": "material",
        "targets": [{"target": "china_general", "stance": "negative"}],
    }
    changed_label = {
        **label,
        "targets": [{"target": "china_general", "stance": "positive"}],
    }
    assert training.label_sha256(changed_label) != training.label_sha256(label)

    with pytest.raises(ValueError, match="schema drifted"):
        training.source_row_sha256({**source, "unbound": "value"})


def test_label_digest_reuses_supplied_validated_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {"type": "object"}
    observed: list[object] = []

    def validate(
        value: object, *, schema: object = None, validator: object = None
    ) -> dict:
        assert validator is None
        observed.append(schema)
        return dict(value)  # type: ignore[arg-type]

    monkeypatch.setattr(training, "validate_v2_label", validate)
    training.label_sha256({"codability": "not_codable"}, schema=schema)
    assert observed == [schema]


def test_prepared_frame_descriptors_include_frozen_calibration() -> None:
    descriptors = {
        frame: {"frame": frame, "sha256": str(index) * 64}
        for index, frame in enumerate(
            ("training", "development", "calibration"), start=1
        )
    }
    assert training._prepared_frame_descriptors(descriptors) == descriptors
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="descriptor inventory",
    ):
        training._prepared_frame_descriptors(
            {key: value for key, value in descriptors.items() if key != "calibration"}
        )


def test_teacher_receipt_binds_fresh_labels_mapping_source_and_rubric() -> None:
    receipt = {
        "kind": "sol-teacher-10k-v2-receipt-v1",
        "status": "complete",
        "run_id": "a" * 64,
        "packet_id": "9" * 64,
        "row_count": 10_000,
        "private_labels_parquet_sha256": "b" * 64,
        "private_mapping_sha256": "c" * 64,
        "source_parquet_sha256": "d" * 64,
        "rubric_sha256": "e" * 64,
        "label_schema_sha256": "f" * 64,
        "evidence_boundary": "silver-model-assisted-teacher-labels-not-human-validation",
    }
    assert training._validate_teacher_receipt_binding(
        receipt,
        expected_teacher_run_id="a" * 64,
        teacher_labels_sha256="b" * 64,
        private_mapping_sha256="c" * 64,
        source_parquet_sha256="d" * 64,
        rubric_sha256="e" * 64,
        schema_sha256="f" * 64,
    ) == receipt
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="exact fresh-v2 inputs",
    ):
        training._validate_teacher_receipt_binding(
            {**receipt, "label_schema_sha256": "0" * 64},
            expected_teacher_run_id="a" * 64,
            teacher_labels_sha256="b" * 64,
            private_mapping_sha256="c" * 64,
            source_parquet_sha256="d" * 64,
            rubric_sha256="e" * 64,
            schema_sha256="f" * 64,
        )


def test_source_packet_receipt_exactly_binds_source_mapping_and_blinded_input() -> None:
    receipt = {
        "kind": "sol-teacher-10k-v2-packet-receipt-v1",
        "status": "complete",
        "packet_id": "a" * 64,
        "source_parquet_sha256": "b" * 64,
        "source_rows": 10_000,
        "selected_rows": 10_000,
        "unique_threads": 10_000,
        "blinded_input_sha256": "c" * 64,
        "private_mapping_sha256": "d" * 64,
        "evidence_boundary": "silver-model-assisted-teacher-labels-not-human-validation",
    }
    path = Path(f"receipt-{experiment.canonical_sha256(receipt)}.json")
    assert training._validate_source_provenance_binding(
        receipt,
        receipt_path=path,
        expected_packet_id="a" * 64,
        source_parquet_sha256="b" * 64,
        private_mapping_sha256="d" * 64,
        blinded_input_sha256="c" * 64,
    ) == receipt

    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="exact source, mapping and blinded input",
    ):
        training._validate_source_provenance_binding(
            receipt,
            receipt_path=path,
            expected_packet_id="a" * 64,
            source_parquet_sha256="b" * 64,
            private_mapping_sha256="e" * 64,
            blinded_input_sha256="c" * 64,
        )
    arbitrary = {"status": "complete", "contains_private_data": False}
    with pytest.raises(experiment.FactorisedExperimentContractError):
        training._validate_source_provenance_binding(
            arbitrary,
            receipt_path=Path(
                f"receipt-{experiment.canonical_sha256(arbitrary)}.json"
            ),
            expected_packet_id="a" * 64,
            source_parquet_sha256="b" * 64,
            private_mapping_sha256="d" * 64,
            blinded_input_sha256="c" * 64,
        )


def test_source_packet_receipt_requires_canonical_content_address() -> None:
    receipt = {
        "kind": "sol-teacher-10k-v2-packet-receipt-v1",
        "status": "complete",
        "packet_id": "a" * 64,
        "source_parquet_sha256": "b" * 64,
        "source_rows": 10_000,
        "selected_rows": 10_000,
        "unique_threads": 10_000,
        "blinded_input_sha256": "c" * 64,
        "private_mapping_sha256": "d" * 64,
        "evidence_boundary": "silver-model-assisted-teacher-labels-not-human-validation",
    }
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="canonical content address",
    ):
        training._validate_source_provenance_binding(
            receipt,
            receipt_path=Path("receipt-wrong.json"),
            expected_packet_id="a" * 64,
            source_parquet_sha256="b" * 64,
            private_mapping_sha256="d" * 64,
            blinded_input_sha256="c" * 64,
        )


def _successful_relevance_result(work_root: Path) -> dict:
    checkpoint = work_root / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    component_metrics = {}
    for selection, row_count in {
        "development_context_available": 75,
        "development_multi_target": 75,
        "development_probability": 300,
        "development_rare_target_stance": 150,
    }.items():
        component_metrics[selection] = {
            "row_count": row_count,
            "evidence_scope": (
                "design-weighted-natural-probability-arm"
                if selection == "development_probability"
                else "unweighted-diagnostic-only"
            ),
            "relevance_macro_f1": 0.8,
            "material_recall": 0.8,
        }
    return {
        "checkpoint_path": str(checkpoint),
        "private_prediction_rows": [
            {"item_id": f"development-{index:03d}", "relevance_logit": 1.25}
            for index in range(600)
        ],
        "aggregate_metrics": {
            "selected_epoch": 1,
            "development_rows": 600,
            "primary_probability_rows": 300,
            "invalid_outputs": 0,
            "checkpoint_score": 0.8,
            "relevance_macro_f1": 0.8,
            "material_recall": 0.8,
            "development_component_metrics": component_metrics,
        },
        "epoch_history": [{"epoch": 1, "checkpoint_score": 0.8}],
        "peak_gpu_bytes": 123,
        "wall_seconds": 10.0,
        "gpu_seconds": 10.0,
    }


def test_trial_publication_is_immutable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    trial = next(
        trial for trial in manifest["trials"] if trial["component"] == "relevance"
    )
    job = experiment.make_trial_job(manifest, trial)
    monkeypatch.setattr(training, "_new_attempt_id", lambda: "a" * 32)

    calls = 0

    def executor(_job: dict, _root: Path, work_root: Path) -> dict:
        nonlocal calls
        calls += 1
        return _successful_relevance_result(work_root)

    first = training.run_training_trial(
        job=job, volume_root=tmp_path, trial_executor=executor
    )
    second = training.run_training_trial(
        job=job, volume_root=tmp_path, trial_executor=executor
    )
    assert first["status"] == "complete"
    assert second["status"] == "already_complete"
    assert calls == 1
    inspection = training.inspect_trial_output(
        manifest=manifest, trial_id=trial["trial_id"], volume_root=tmp_path
    )
    assert inspection["status"] == "complete"
    assert inspection["attempt_count"] == 1
    final, attempts = training._trial_roots(
        volume_root=tmp_path,
        experiment_run_id=manifest["experiment_run_id"],
        component=trial["component"],
        trial_id=trial["trial_id"],
    )
    attempt_root = attempts / f"attempt={'a' * 32}"
    assert (final / "checkpoint.pt").is_file()
    assert not (attempt_root / "work" / "checkpoint.pt").exists()


def test_cuda_preflight_round_trips_real_receipt_and_artifact_contract(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    trial = next(
        trial for trial in manifest["trials"] if trial["component"] == "relevance"
    )
    result = _successful_relevance_result(tmp_path)
    evidence = launcher._validate_preflight_publication_contract(
        result["aggregate_metrics"],
        component="relevance",
        manifest=manifest,
        trial=trial,
        private_rows=result["private_prediction_rows"],
    )
    assert evidence["status"] == "passed"
    assert evidence["validator"] == "build_trial_receipt+validate_trial_artifacts"
    assert len(evidence["receipt_id"]) == 64
    assert len(evidence["artifact_bindings_sha256"]) == 64


def test_training_interruption_retains_attempt_and_clean_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    trial = next(
        trial for trial in manifest["trials"] if trial["component"] == "relevance"
    )
    job = experiment.make_trial_job(manifest, trial)
    attempt_ids = iter(("b" * 32, "c" * 32))
    monkeypatch.setattr(training, "_new_attempt_id", lambda: next(attempt_ids))
    calls = 0

    def executor(_job: dict, _root: Path, work_root: Path) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            (work_root / "partial.txt").write_text("retained", encoding="utf-8")
            raise RuntimeError("synthetic training interruption")
        return _successful_relevance_result(work_root)

    with pytest.raises(RuntimeError, match="synthetic training interruption"):
        training.run_training_trial(
            job=job, volume_root=tmp_path, trial_executor=executor
        )
    interrupted = training.inspect_trial_output(
        manifest=manifest, trial_id=trial["trial_id"], volume_root=tmp_path
    )
    assert interrupted["status"] == "incomplete"
    assert interrupted["attempt_count"] == 1

    completed = training.run_training_trial(
        job=job, volume_root=tmp_path, trial_executor=executor
    )
    already_complete = training.run_training_trial(
        job=job, volume_root=tmp_path, trial_executor=executor
    )
    assert completed["status"] == "complete"
    assert already_complete["status"] == "already_complete"
    assert calls == 2

    final, attempts = training._trial_roots(
        volume_root=tmp_path,
        experiment_run_id=manifest["experiment_run_id"],
        component=trial["component"],
        trial_id=trial["trial_id"],
    )
    assert final.is_dir()
    assert (attempts / f"attempt={'b' * 32}" / "work" / "partial.txt").is_file()
    assert len(list(attempts.glob("attempt=*"))) == 2
    final_trials = list(final.parent.glob("trial=*"))
    assert final_trials == [final]
    final_inspection = training.inspect_trial_output(
        manifest=manifest, trial_id=trial["trial_id"], volume_root=tmp_path
    )
    assert final_inspection["status"] == "complete"
    assert final_inspection["attempt_count"] == 2


def test_publication_interruption_never_promotes_partial_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    trial = next(
        trial for trial in manifest["trials"] if trial["component"] == "relevance"
    )
    job = experiment.make_trial_job(manifest, trial)
    attempt_ids = iter(("d" * 32, "e" * 32))
    monkeypatch.setattr(training, "_new_attempt_id", lambda: next(attempt_ids))
    original_write = training._write_json_atomic
    injected = False

    def interrupt_publication(path: Path, value: dict) -> None:
        nonlocal injected
        if not injected and path.parent.name == "publication" and path.name == "metrics.json":
            injected = True
            raise RuntimeError("synthetic publication interruption")
        original_write(path, value)

    monkeypatch.setattr(training, "_write_json_atomic", interrupt_publication)

    with pytest.raises(RuntimeError, match="synthetic publication interruption"):
        training.run_training_trial(
            job=job,
            volume_root=tmp_path,
            trial_executor=lambda _job, _root, work: _successful_relevance_result(work),
        )

    final, attempts = training._trial_roots(
        volume_root=tmp_path,
        experiment_run_id=manifest["experiment_run_id"],
        component=trial["component"],
        trial_id=trial["trial_id"],
    )
    failed_publication = attempts / f"attempt={'d' * 32}" / "publication"
    assert not final.exists()
    assert (failed_publication / "checkpoint.pt").is_file()
    assert (failed_publication / "development-predictions.json").is_file()
    assert not (failed_publication / "metrics.json").exists()

    completed = training.run_training_trial(
        job=job,
        volume_root=tmp_path,
        trial_executor=lambda _job, _root, work: _successful_relevance_result(work),
    )
    assert completed["status"] == "complete"
    assert final.is_dir()
    assert len(list(attempts.glob("attempt=*"))) == 2
    assert list(final.parent.glob("trial=*")) == [final]


def test_exposure_register_is_exact_and_metadata_only() -> None:
    register = training._build_exposure_register(
        scope="bridge",
        thread_ids=[f"thread-{index:03d}" for index in range(480)],
        source_artifacts={"mapping": _artifact("mapping.parquet", "a", row_count=480)},
    )
    assert training.validate_private_exposure_register(register, scope="bridge") == register
    assert register["thread_count"] == 480
    with pytest.raises(experiment.FactorisedExperimentContractError, match="480"):
        training._build_exposure_register(
            scope="bridge",
            thread_ids=["one-thread"],
            source_artifacts={"mapping": _artifact("mapping.parquet", "a", row_count=1)},
        )


def test_checkpoint_selection_uses_only_probability_primary_rows() -> None:
    reference: dict[str, dict] = {}
    rows = []
    components: dict[str, str] = {}
    component_plan = [
        ("development_probability", 300),
        ("development_context_available", 75),
        ("development_multi_target", 75),
        ("development_rare_target_stance", 150),
    ]
    index = 0
    for selection, count in component_plan:
        for local_index in range(count):
            item_id = f"item-{index:03d}"
            material = local_index % 2 == 0
            reference[item_id] = (
                {
                    "codability": "codable",
                    "relevance": "material",
                    "targets": [
                        {"target": "china_general", "stance": "negative"}
                    ],
                }
                if material
                else {
                    "codability": "codable",
                    "relevance": "not_material",
                    "targets": [],
                }
            )
            correct = selection == "development_probability"
            predicted_material = material if correct else not material
            rows.append(
                {
                    "item_id": item_id,
                    "relevance_logit": 20.0 if predicted_material else -20.0,
                }
            )
            components[item_id] = selection
            index += 1

    natural_design = {
        item_id: {
            "frame": "development",
            "selection_component": "development_probability",
            "selection_stratum": "all",
            "inclusion_probability_numerator": 300,
            "inclusion_probability_denominator": 300,
            "inclusion_probability": 1.0,
            "probability_scope": evaluation.PROBABILITY_SCOPE,
        }
        for item_id, selection in components.items()
        if selection == "development_probability"
    }
    weighting = evaluation.natural_arm_weighting_config_from_summary(
        {
            "component": "development_probability",
            "design_digest": evaluation.natural_arm_design_digest(natural_design),
            "expected_sample_rows": 300,
            "represented_population_rows": 300,
            "expected_strata": 1,
            "expected_probability_min": 1.0,
            "expected_probability_max": 1.0,
            "design_effective_sample_size": 300.0,
            "inverse_weight_ratio": 1.0,
        },
        estimand=training.NATURAL_DEVELOPMENT_ESTIMAND,
    )

    metrics = training._score_checkpoint(
        component="relevance",
        reference=reference,
        private_rows=rows,
        selection_component_by_item=components,
        natural_design=natural_design,
        natural_weighting=weighting,
        selected_epoch=1,
    )
    assert metrics["primary_probability_rows"] == 300
    assert metrics["relevance_macro_f1"] == 1.0
    assert metrics["checkpoint_score"] == 1.0
    assert metrics["development_component_metrics"][
        "development_context_available"
    ]["relevance_macro_f1"] == 0.0
    assert metrics["natural_arm_weighting_config_digest"] == weighting.digest()


def test_private_frame_loader_requires_explicit_frame_column_and_identity(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "development.parquet"
    base = {
        "item_id": "item-1",
        "target_text": "bounded text",
        "parent_context": None,
        "submission_context": None,
        "label_json": json.dumps(
            {"codability": "codable", "relevance": "not_material", "targets": []}
        ),
        "selection_component": "development_probability",
        "selection_stratum": "all",
        "inclusion_probability_numerator": 1,
        "inclusion_probability_denominator": 1,
        "inclusion_probability": 1.0,
        "probability_scope": evaluation.PROBABILITY_SCOPE,
    }
    parquet.write_table(pyarrow.Table.from_pylist([base]), path)
    descriptor = {
        "relative_path": "development.parquet",
        "sha256": training.file_sha256(path),
        "bytes": path.stat().st_size,
        "row_count": 1,
        "frame": "development",
    }
    with pytest.raises(ValueError, match="omits required columns"):
        training.load_private_frame(
            path, descriptor, expected_frame="development"
        )

    parquet.write_table(
        pyarrow.Table.from_pylist([{**base, "frame": "training"}]), path
    )
    descriptor = {
        **descriptor,
        "sha256": training.file_sha256(path),
        "bytes": path.stat().st_size,
    }
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="not in the expected development frame",
    ):
        training.load_private_frame(
            path, descriptor, expected_frame="development"
        )


def test_natural_design_extraction_fails_clearly_without_development_frame() -> None:
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="missing or drifted frame identity",
    ):
        training._natural_design_from_development_rows(
            [
                {
                    "item_id": "item-1",
                    "selection_component": "development_probability",
                }
            ]
        )


def test_representation_gate_rejects_unweighted_primary_metrics() -> None:
    with pytest.raises(
        training.FactorisedExperimentContractError,
        match="must be design-weighted",
    ):
        training.gate_metric_surface(
            {"scientific_aggregate_eligible": False, "end_to_end": {}}
        )


def test_closeout_weights_only_probability_arm_and_keeps_enrichment_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    plan = [
        ("development_probability", 300),
        ("development_context_available", 75),
        ("development_multi_target", 75),
        ("development_rare_target_stance", 150),
    ]
    development_rows = []
    reference = {}
    index = 0
    for component, count in plan:
        for _ in range(count):
            item_id = f"item-{index:03d}"
            development_rows.append(
                {
                    "item_id": item_id,
                    "selection_component": component,
                    "frame": "development",
                    "selection_stratum": "all",
                    "inclusion_probability_numerator": 300,
                    "inclusion_probability_denominator": 300,
                    "inclusion_probability": 1.0,
                    "probability_scope": evaluation.PROBABILITY_SCOPE,
                }
            )
            reference[item_id] = {
                "codability": "codable",
                "relevance": "not_material",
                "targets": [],
            }
            index += 1
    natural_design = training._natural_design_from_development_rows(
        development_rows
    )
    summary = {
        "component": "development_probability",
        "design_digest": evaluation.natural_arm_design_digest(natural_design),
        "expected_sample_rows": 300,
        "represented_population_rows": 300,
        "expected_strata": 1,
        "expected_probability_min": 1.0,
        "expected_probability_max": 1.0,
        "design_effective_sample_size": 300.0,
        "inverse_weight_ratio": 1.0,
    }
    weighting = evaluation.natural_arm_weighting_config_from_summary(
        summary, estimand=training.NATURAL_DEVELOPMENT_ESTIMAND
    )
    monkeypatch.setattr(
        training,
        "load_private_frame",
        lambda *_args, **_kwargs: (development_rows, reference),
    )
    monkeypatch.setattr(
        training,
        "_load_natural_development_weighting",
        lambda **_kwargs: (weighting, summary),
    )
    monkeypatch.setattr(
        training,
        "_load_trial_payload",
        lambda *, trial, **_kwargs: (
            {"receipt_id": trial["trial_id"]},
            {"rows": [{"item_id": item_id} for item_id in reference]},
        ),
    )
    calls: list[tuple[int, bool]] = []

    def fake_combine(**kwargs: object) -> dict:
        weighted = kwargs.get("natural_design") is not None
        calls.append((len(kwargs["reference"]), weighted))  # type: ignore[arg-type]
        if weighted:
            return {
                "scientific_aggregate_eligible": True,
                "design_weighted": {
                    "end_to_end": {
                        "target_stance_tuples": {"micro": {"f1": 0.5}}
                    }
                },
            }
        return {
            "scientific_aggregate_eligible": False,
            "evidence_scope": "unweighted-diagnostic-only",
            "end_to_end": {"target_stance_tuples": {"micro": {"f1": 0.5}}},
        }

    monkeypatch.setattr(training, "combine_component_predictions", fake_combine)
    monkeypatch.setattr(
        training,
        "gate_metric_surface",
        lambda metrics: {
            "tuple_micro_f1": 0.5,
            "target_presence_macro_f1": 0.5,
            "calibration_error": 0.5,
            "retained_coverage_risk": 0.5,
            "per_target": {},
            "per_target_stance_cell": {},
            "invalid_outputs": 0,
        }
        if metrics["scientific_aggregate_eligible"] is True
        else (_ for _ in ()).throw(AssertionError("unweighted primary")),
    )
    monkeypatch.setattr(
        training,
        "evaluate_representation_gate",
        lambda *_args, **_kwargs: {
            "selected_representation": "B4",
            "verdict": "retain_b4",
            "aggregate_evidence": {},
            "gate_receipt_id": "f" * 64,
        },
    )

    result = training.closeout_fixed_comparison(
        manifest=manifest, volume_root=tmp_path
    )
    assert sum(weighted for _, weighted in calls) == 6
    assert all(size == 300 for size, weighted in calls if weighted)
    assert all(size in {75, 150} for size, weighted in calls if not weighted)
    assert result["development_design"]["weighting_config_digest"] == weighting.digest()
    for component in (
        "development_context_available",
        "development_multi_target",
        "development_rare_target_stance",
    ):
        assert all(
            value["evidence_scope"] == "unweighted-enrichment-diagnostic-only"
            for value in result["development_design"][
                "selection_component_diagnostics"
            ][component].values()
        )


def test_exact_teacher_join_rejects_cross_joined_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    run_id = "a" * 64
    label = {
        "codability": "codable",
        "relevance": "material",
        "targets": [{"target": "china_general", "stance": "negative"}],
    }
    label_json = json.dumps(label, separators=(",", ":"), sort_keys=True)
    row_count = 400
    monkeypatch.setattr(training, "EXPECTED_SOURCE_ROWS", row_count)
    opaque_ids = [f"opaque-{index:05d}" for index in range(row_count)]
    sample_ids = [f"sample-{index:05d}" for index in range(row_count)]
    thread_ids = [f"thread-{index:05d}" for index in range(row_count)]
    labels = pyarrow.table(
        {
            "opaque_id": opaque_ids,
            "codability": ["codable"] * row_count,
            "relevance": ["material"] * row_count,
            "label_json": [label_json] * row_count,
            "quality_tier": ["exact_consensus"] * row_count,
            "primary_training_eligible": [True] * row_count,
        }
    ).replace_schema_metadata({b"run_id": run_id.encode()})
    labels_path = tmp_path / "labels.parquet"
    parquet.write_table(labels, labels_path)
    mapping_path = tmp_path / "mapping.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "opaque_id": opaque_ids,
                "sample_id": sample_ids,
                "thread_id": thread_ids,
                "packet_order": list(range(row_count)),
            }
        ),
        mapping_path,
    )
    source_path = tmp_path / "source.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "sample_id": sample_ids,
                "thread_id": thread_ids,
                "target_text": [f"target {index}" for index in range(row_count)],
                "submission_context": [None] * row_count,
                "parent_context": [None] * row_count,
                "subreddit": ["news"] * row_count,
                "year": [2024] * row_count,
                "content_type": ["comment"] * row_count,
                "retrieval_mode": ["lexical"] * row_count,
            }
        ),
        source_path,
    )
    blinded_path = tmp_path / "blinded.json"
    blinded_rows = [
        {
            "source_sample_id": opaque_id,
            "target_text": f"target {index}",
            "submission_context": None,
            "parent_context": None,
        }
        for index, opaque_id in enumerate(opaque_ids)
    ]
    blinded_rows[321]["target_text"] = "cross-joined mutation"
    blinded_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "kind": "sol-teacher-10k-v2-blinded-input-v1",
                "packet_id": "b" * 64,
                "rubric_sha256": "c" * 64,
                "label_schema_sha256": "d" * 64,
                "rows": blinded_rows,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        experiment.FactorisedExperimentContractError,
        match="mapping, blinded text/context and source",
    ):
        training._joined_rows_from_teacher_artifacts(
            teacher_labels_parquet_path=labels_path,
            blinded_input_json_path=blinded_path,
            private_mapping_parquet_path=mapping_path,
            source_parquet_path=source_path,
            expected_teacher_run_id=run_id,
        )


def test_bound_input_and_prepare_roots_fail_closed_on_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "bound.json"
    artifact.write_text("{}\n", encoding="utf-8")
    descriptor = {
        "relative_path": artifact.relative_to(tmp_path).as_posix(),
        "sha256": training.file_sha256(artifact),
        "bytes": artifact.stat().st_size,
    }
    artifact.write_text('{"drift":true}\n', encoding="utf-8")
    with pytest.raises(experiment.FactorisedExperimentContractError, match="file binding"):
        training._validate_bound_input_file(
            artifact,
            descriptor,
            descriptor_root=tmp_path,
            where="synthetic input",
        )

    with pytest.raises(ValueError, match="inside descriptor_root"):
        training.prepare_evidence_frames_from_artifacts(
            teacher_receipt_path=artifact,
            teacher_receipt_descriptor=descriptor,
            teacher_labels_parquet_path=artifact,
            teacher_labels_descriptor=descriptor,
            blinded_input_json_path=artifact,
            blinded_input_descriptor=descriptor,
            private_mapping_parquet_path=artifact,
            private_mapping_descriptor=descriptor,
            source_parquet_path=artifact,
            source_parquet_descriptor=descriptor,
            source_receipt_path=artifact,
            source_receipt_descriptor=descriptor,
            source_metadata_mapping_path=artifact,
            source_metadata_mapping_descriptor=descriptor,
            bridge_receipt_path=artifact,
            bridge_receipt_descriptor=descriptor,
            bridge_exposure_register_json_path=artifact,
            bridge_exposure_register_descriptor=descriptor,
            legacy_proxy_parquet_path=artifact,
            legacy_proxy_descriptor=descriptor,
            private_split_root=tmp_path / "private",
            public_split_root=tmp_path / "public",
            frame_output_root=tmp_path.parent / "outside",
            descriptor_root=tmp_path,
            expected_teacher_run_id="a" * 64,
            rubric_sha256="b" * 64,
            schema_sha256="c" * 64,
        )

    remote_parent = (
        tmp_path
        / training.VOLUME_PREPARATION_ROOT_NAME
        / "prepared"
        / "input=synthetic"
    )
    with pytest.raises(ValueError, match="sibling private-split/public-split/frames"):
        training._validate_preparation_output_roots(
            private_split_root=remote_parent / "private-leak",
            public_split_root=remote_parent / "public-split",
            frame_output_root=remote_parent / "frames",
            descriptor_root=tmp_path,
        )


def test_local_private_preparation_roots_are_git_ignored_and_public_is_disjoint() -> None:
    run = "test-layout"
    training._validate_preparation_output_roots(
        private_split_root=training.LOCAL_PRIVATE_PREPARATION_ROOT / run / "private-split",
        public_split_root=training.LOCAL_PUBLIC_PREPARATION_ROOT / run / "public-split",
        frame_output_root=training.LOCAL_PRIVATE_PREPARATION_ROOT / run / "frames",
        descriptor_root=training.REPO_ROOT,
    )
