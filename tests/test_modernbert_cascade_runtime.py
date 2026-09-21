from __future__ import annotations

from pathlib import Path

import pytest

from reddit_china_stance.modernbert_cascade_experiment import (
    TARGET_STATES,
    TARGETS,
    freeze_experiment_contract,
    freeze_trial_spec,
)
from reddit_china_stance.modernbert_cascade_runtime import (
    _evaluate_gate,
    _reconcile_resume_staging,
    _relevance_metrics,
    _target_metrics,
    _trial_roots,
    _validate_job,
    aggregate_trial_inspections,
    build_epoch_zero_resume_marker,
)
from reddit_china_stance.modernbert_training import (
    DATASET_REVISION,
    DATASET_SHA256,
    DEVELOPMENT_PROXY_ID,
    TEACHER_GENERATION_RUN_ID,
    canonical_sha256,
)


def _contract() -> dict[str, object]:
    return freeze_experiment_contract(
        dataset_revision=DATASET_REVISION,
        dataset_sha256=DATASET_SHA256,
        split_manifest_sha256="1" * 64,
        source_bundle_sha256="2" * 64,
        code_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        development_proxy_sha256=DEVELOPMENT_PROXY_ID,
        development_reference_sha256="5" * 64,
        baseline_manifest_sha256="6" * 64,
        matched_baseline_sha256="7" * 64,
        teacher_generation_run_id=TEACHER_GENERATION_RUN_ID,
        rate_card_usd_per_gpu_second={"L4": "0.000222"},
        hard_cost_cap_usd="20",
    )


def _trial(contract: dict[str, object], *, component: str = "relevance") -> dict[str, object]:
    return freeze_trial_spec(
        contract,
        component=component,
        subset_manifest_sha256="8" * 64,
        ladder_seed=101,
        optimiser_seed=47,
        gpu_type="L4",
        max_gpu_seconds=10_000,
    )


def _job(contract: dict[str, object], trial: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_run_id": contract["experiment_run_id"],
        "phase_run_id": "9" * 64,
        "experiment_contract": contract,
        "trial_spec": trial,
        "trial_spec_sha256": canonical_sha256(trial),
        "run_manifest_sha256": "a" * 64,
    }


def _reference() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for index in range(222):
        if index % 3 == 0:
            label: dict[str, object] = {
                "relevance": "material",
                "target_stances": [
                    {"target": "government_ccp", "stance": "negative"}
                ],
            }
        elif index % 3 == 1:
            label = {"relevance": "not_material", "target_stances": []}
        else:
            label = {"relevance": "unclear", "target_stances": []}
        rows[f"row-{index:03d}"] = label
    return rows


def test_job_and_epoch_zero_marker_bind_exact_contract_and_trial() -> None:
    contract = _contract()
    trial = _trial(contract)
    job = _job(contract, trial)
    assert _validate_job(job) == job
    marker = build_epoch_zero_resume_marker(job)
    assert marker["completed_epochs"] == 0
    assert marker["state"] == "claimed_no_checkpoint"
    assert marker["trial_id"] == trial["trial_id"]
    drifted = dict(job)
    drifted["run_manifest_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_job(drifted)


def test_trial_roots_isolate_final_and_crash_safe_staging(tmp_path: Path) -> None:
    root, final, incomplete = _trial_roots(
        volume_root=tmp_path,
        experiment_run_id="b" * 64,
        component="relevance",
        trial_id="c" * 64,
    )
    assert root == tmp_path / "student-modernbert-cascade-v1" / f"run={'b' * 64}"
    assert final == (
        root / "phase=confirmation" / "component=relevance" / f"trial={'c' * 64}"
    )
    assert incomplete == (
        root
        / "phase=confirmation.incomplete"
        / "component=relevance"
        / f"trial={'c' * 64}"
    )


def test_explicit_resume_discards_only_uncommitted_later_epoch_files(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "trial"
    checkpoints = staging / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "epoch-01.pt").write_bytes(b"committed")
    (checkpoints / "epoch-02.pt").write_bytes(b"orphan")
    (staging / "epoch-01-predictions.json").write_text("{}", encoding="utf-8")
    (staging / "epoch-02-predictions.json").write_text("{}", encoding="utf-8")
    (staging / "history.json").write_text(
        '{"epochs":[{"epoch":1},{"epoch":2}]}', encoding="utf-8"
    )
    (staging / "resume.json.new").write_text("partial", encoding="utf-8")

    _reconcile_resume_staging(staging, completed_epochs=1)

    assert (checkpoints / "epoch-01.pt").is_file()
    assert not (checkpoints / "epoch-02.pt").exists()
    assert (staging / "epoch-01-predictions.json").is_file()
    assert not (staging / "epoch-02-predictions.json").exists()
    assert not (staging / "resume.json.new").exists()
    assert '"epoch": 1' in (staging / "history.json").read_text(encoding="utf-8")
    assert '"epoch": 2' not in (staging / "history.json").read_text(encoding="utf-8")


def test_relevance_metrics_score_exact_222_rows() -> None:
    reference = _reference()
    rows = []
    for item_id, label in reference.items():
        index = ("material", "not_material", "unclear").index(label["relevance"])
        logits = [-1.0, -1.0, -1.0]
        logits[index] = 1.0
        rows.append({"source_sample_id": item_id, "logits": logits})
    metrics = _relevance_metrics(reference, rows)
    assert metrics["relevance_macro_f1"] == 1.0
    assert metrics["material_recall"] == 1.0
    assert metrics["checkpoint_score"] == 1.0
    with pytest.raises(Exception, match="conserve"):
        _relevance_metrics(reference, rows[:-1])


def test_target_metrics_score_gold_relevance_component_diagnostic() -> None:
    reference = _reference()
    rows = []
    for item_id, label in reference.items():
        expected = {
            row["target"]: row["stance"] for row in label["target_stances"]
        }
        for target in TARGETS:
            state = expected.get(target, "absent")
            logits = [-1.0] * len(TARGET_STATES)
            logits[TARGET_STATES.index(state)] = 1.0
            rows.append(
                {"source_sample_id": item_id, "target": target, "logits": logits}
            )
    metrics = _target_metrics(reference, rows)
    assert metrics["target_presence_f1_gold_relevance"] == 1.0
    assert metrics["stance_accuracy_present_targets"] == 1.0
    assert metrics["core_target_stance_tuple_micro_f1_gold_relevance"] == 1.0


def test_aggregate_inspections_conserves_six_trials_and_cost() -> None:
    trials = [{"trial_id": f"{index:064x}"} for index in range(6)]
    inspections = [
        {
            "trial_id": trial["trial_id"],
            "state": "complete",
            "receipt": {"estimated_cost_usd": 1.25},
        }
        for trial in trials
    ]
    result = aggregate_trial_inspections(
        manifest={"trials": trials}, inspections=inspections
    )
    assert result["status"] == "complete"
    assert result["estimated_cost_usd"] == "7.500000"
    with pytest.raises(ValueError, match="conserve"):
        aggregate_trial_inspections(manifest={"trials": trials}, inspections=inspections[:-1])


def test_aggregate_six_missing_trials_has_decimal_zero_cost() -> None:
    trials = [{"trial_id": f"{index:064x}"} for index in range(6)]
    inspections = [
        {"trial_id": trial["trial_id"], "state": "missing", "receipt": None}
        for trial in trials
    ]

    result = aggregate_trial_inspections(
        manifest={"trials": trials}, inspections=inspections
    )

    assert result["status"] == "incomplete"
    assert result["complete_trial_ids"] == []
    assert result["missing_trial_ids"] == sorted(trial["trial_id"] for trial in trials)
    assert result["estimated_cost_usd"] == "0.000000"


def _diagnostics(*, f1: float = 0.9) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        target: {
            state: {
                "reference_support": 12,
                "precision": f1,
                "recall": f1,
                "f1": f1,
            }
            for state in TARGET_STATES
        }
        for target in TARGETS
    }


def test_gate_passes_only_material_paired_gain_and_all_guards() -> None:
    experiment = _contract()
    baseline = []
    candidate = []
    for ladder, optimiser in ((101, 47), (202, 61), (303, 89)):
        condition = {"ladder_seed": ladder, "optimiser_seed": optimiser}
        baseline.append(
            {
                "condition": condition,
                "metric_payload": {
                    "core_target_stance_tuple_micro_f1": 0.60,
                    "material_recall": 0.90,
                    "relevance_macro_f1": 0.80,
                    "core_target_f1": {target: 0.80 for target in TARGETS[:3]},
                    "core_target_reference_support": {
                        target: 12 for target in TARGETS[:3]
                    },
                    "target_stance_diagnostics": _diagnostics(),
                    "invalid_outputs": 0,
                },
            }
        )
        candidate.append(
            {
                "condition": condition,
                "core_tuple_f1": 0.64,
                "material_recall": 0.90,
                "relevance_macro_f1": 0.80,
                "core_target_f1": {target: 0.82 for target in TARGETS[:3]},
                "core_target_support": {target: 12 for target in TARGETS[:3]},
                "target_stance_diagnostics": _diagnostics(f1=0.91),
                "invalid_outputs": 0,
                "forced_target_selections": 0,
            }
        )
    gate = _evaluate_gate(
        experiment=experiment,
        phase_run_id="d" * 64,
        baseline=baseline,
        candidate=candidate,
        candidate_receipt_ids=[f"{index:064x}" for index in range(6)],
    )
    assert gate["verdict"] == "continue_to_independent_evaluation"
    assert all(gate["guards"].values())
    candidate[0]["invalid_outputs"] = 1
    rejected = _evaluate_gate(
        experiment=experiment,
        phase_run_id="d" * 64,
        baseline=baseline,
        candidate=candidate,
        candidate_receipt_ids=[f"{index:064x}" for index in range(6)],
    )
    assert rejected["verdict"] == "scrap"
    assert rejected["guards"]["invalid_outputs"] is False


def test_runtime_has_no_locked_test_loader_or_inference_entrypoint() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/reddit_china_stance/modernbert_cascade_runtime.py"
    ).read_text(encoding="utf-8").lower()
    assert "modal_modernbert_locked" not in source
    assert "modernbert_locked_test" not in source
    assert "run_corpus" not in source
