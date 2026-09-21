"""Run and score the frozen blinded Sol direct-teacher evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reddit_china_stance.human_seeded_consensus_v1 import (
    ADJUDICATION_BLINDNESS_CONTRACT,
    ADJUDICATION_INPUT_KIND,
    SCHEMA_VERSION,
    file_sha256,
)
from reddit_china_stance.semantic_evaluation import (
    CORE_TARGETS,
    evaluate_capability_gate,
    score_semantic_labels,
)
from reddit_china_stance.sol_adjudicator import (
    ADJUDICATOR_ID,
    INSTRUCTION,
    REASONING_EFFORT,
    RUBRIC_PATH,
    SCHEMA_PATH,
    SOL_ADJUDICATOR_MODEL,
    _json_bytes,
    _read_object,
    _write_immutable,
)
from reddit_china_stance.sol_adjudicator import (
    run as run_sol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPO_ROOT / (
    "data/private-human-reference-v1/"
    "run=1e720f32ef6c51d1fbb4c1acc62ff4f1ddaf7027e3a652339d2ad92be7020987/"
    "reference-rows.json"
)
PROXY_PATH = REPO_ROOT / (
    "data/private-human-reference-v1/consensus/final/"
    "proxy-4ba40fcd0eb33dc2c7ca1c0b9ac75b06f42dc120bed470f9da7f713194a3a66d.json"
)
PRIVATE_ROOT = REPO_ROOT / "data/private-sol-direct-teacher-eval-v1"
PUBLIC_ROOT = REPO_ROOT / "outputs/sol-direct-teacher-eval-v1"
EXPERIMENT_PATH = REPO_ROOT / "docs/experiments/sol-direct-teacher-eval-v1.md"
EXPECTED_ROWS = 230
SHARD_COUNT = 6
JOBS = 6
RUNNER_ID = "sol-direct-teacher-eval-v1"
HUMAN_SUPPORTED_RESOLUTION = "strict_human_supported_majority"
SOL_RESOLUTION = "blinded_sol_adjudication"

HUMAN_SUPPORTED_THRESHOLDS = {
    "invalid_outputs_max": 0,
    "material_recall_min": 0.85,
    "core_target_micro_f1_min": 0.75,
    "fixed_reference_core_target_stance_accuracy_min": 0.75,
    "core_target_stance_tuple_micro_f1_min": 0.70,
}
CORE_TARGET_RECALL_MIN = 0.60


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = _read_object(path)
    rows = value.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"rows are missing or invalid: {path}")
    return [dict(row) for row in rows]


def _test_id(source_sample_id: str) -> str:
    digest = _sha256_bytes(f"{RUNNER_ID}:{source_sample_id}".encode())
    return f"T{digest[:20]}"


def _prepared_paths(run_id: str) -> tuple[Path, Path]:
    root = PRIVATE_ROOT / f"run={run_id}"
    return root / "blinded-input.json", root / "private-reference.json"


def _execution_root(run_id: str) -> Path:
    return PRIVATE_ROOT / f"run={run_id}" / "execution"


def _public_paths(run_id: str) -> tuple[Path, Path]:
    root = PUBLIC_ROOT / f"run={run_id}"
    return root / "summary-v3.json", root / "receipt-v3.json"


def _private_metrics_path(run_id: str) -> Path:
    return PRIVATE_ROOT / f"run={run_id}" / "scoring" / "metrics-v2.json"


def _input_binding() -> dict[str, Any]:
    prompt_template = f"{INSTRUCTION}\n\nFROZEN_RUBRIC\n{RUBRIC_PATH.read_text(encoding='utf-8')}"
    gate_contract = {
        "overall_gate": "development-proxy-capability-v1",
        "human_supported_thresholds": HUMAN_SUPPORTED_THRESHOLDS,
        "core_target_recall_min": CORE_TARGET_RECALL_MIN,
    }
    return {
        "runner_id": RUNNER_ID,
        "experiment_path": str(EXPERIMENT_PATH.relative_to(REPO_ROOT)),
        "gate_contract_sha256": _canonical_sha256(gate_contract),
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "proxy_sha256": file_sha256(PROXY_PATH),
        "rubric_sha256": file_sha256(RUBRIC_PATH),
        "label_schema_sha256": file_sha256(SCHEMA_PATH),
        "prompt_template_sha256": _sha256_bytes(prompt_template.encode()),
        "model": SOL_ADJUDICATOR_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "adjudicator_id": ADJUDICATOR_ID,
        "rows": EXPECTED_ROWS,
        "shards": SHARD_COUNT,
        "jobs": JOBS,
        "split": "locked_test_candidate",
        "few_shots": "frozen-rubric-synthetic-boundary-examples-only",
    }


def prepare() -> dict[str, Any]:
    reference_rows = _load_rows(REFERENCE_PATH)
    proxy_value = _read_object(PROXY_PATH)
    proxy_rows = proxy_value.get("rows")
    if not isinstance(proxy_rows, list):
        raise ValueError("final proxy rows are missing")
    proxy_by_id = {
        row["source_sample_id"]: row
        for row in proxy_rows
        if isinstance(row, Mapping) and isinstance(row.get("source_sample_id"), str)
    }
    locked = [row for row in reference_rows if row.get("split") == "locked_test_candidate"]
    if len(locked) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} locked rows, found {len(locked)}")
    if len({row["source_sample_id"] for row in locked}) != EXPECTED_ROWS:
        raise ValueError("locked test rows contain duplicate source IDs")
    if any(row["source_sample_id"] not in proxy_by_id for row in locked):
        raise ValueError("final proxy does not cover the locked test set")

    binding = _input_binding()
    run_id = _canonical_sha256(binding)
    blinded_rows = []
    private_rows = []
    for row in locked:
        source_id = row["source_sample_id"]
        item_id = _test_id(source_id)
        proxy_row = proxy_by_id[source_id]
        blinded_rows.append(
            {
                "source_sample_id": item_id,
                "target_text": row["target_text"],
                "submission_context": row["submission_context"],
                "parent_context": row["parent_context"],
            }
        )
        private_rows.append(
            {
                "test_item_id": item_id,
                "source_sample_id": source_id,
                "proxy_label": proxy_row["label"],
                "human_label": row["label"],
                "resolution": proxy_row["resolution"],
            }
        )

    blinded = {
        "schema_version": SCHEMA_VERSION,
        "kind": ADJUDICATION_INPUT_KIND,
        "consensus_id": proxy_value["proxy_id"],
        "consensus_artifact_sha256": file_sha256(PROXY_PATH),
        "source_packet_sha256": file_sha256(REFERENCE_PATH),
        "rubric_sha256": file_sha256(RUBRIC_PATH),
        "label_schema_sha256": file_sha256(SCHEMA_PATH),
        "blindness": ADJUDICATION_BLINDNESS_CONTRACT,
        "rows": blinded_rows,
    }
    private_reference = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-direct-teacher-private-reference-v1",
        "run_id": run_id,
        "input_binding": binding,
        "rows": private_rows,
    }
    input_path, private_reference_path = _prepared_paths(run_id)
    _write_immutable(input_path, blinded)
    _write_immutable(private_reference_path, private_reference)
    return {
        "status": "prepared",
        "run_id": run_id,
        "rows": len(blinded_rows),
        "human_supported_rows": sum(
            row["resolution"] == HUMAN_SUPPORTED_RESOLUTION for row in private_rows
        ),
        "sol_adjudicated_rows": sum(row["resolution"] == SOL_RESOLUTION for row in private_rows),
        "input_sha256": file_sha256(input_path),
        "private_reference_sha256": file_sha256(private_reference_path),
        "input_path": str(input_path),
        "private_reference_path": str(private_reference_path),
    }


def execute() -> dict[str, Any]:
    prepared = prepare()
    result = run_sol(
        input_path=Path(prepared["input_path"]),
        private_root=_execution_root(prepared["run_id"]),
        shard_count=SHARD_COUNT,
        jobs=JOBS,
        timeout_seconds=1800,
    )
    return {"run_id": prepared["run_id"], **result}


def _criteria(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "invalid_outputs": metrics["invalid_outputs"],
        "material_recall": metrics["relevance"]["material_recall"],
        "core_target_micro_f1": metrics["targets"]["core"]["micro"]["f1"],
        "fixed_reference_core_target_stance_accuracy": metrics["stance"][
            "fixed_reference_target_core"
        ]["accuracy"],
        "core_target_stance_tuple_micro_f1": metrics["end_to_end_core_target_stance"][
            "micro"
        ]["f1"],
    }
    checks = {
        "invalid_outputs": values["invalid_outputs"]
        <= HUMAN_SUPPORTED_THRESHOLDS["invalid_outputs_max"],
        "material_recall": values["material_recall"] is not None
        and values["material_recall"] >= HUMAN_SUPPORTED_THRESHOLDS["material_recall_min"],
        "core_target_micro_f1": values["core_target_micro_f1"] is not None
        and values["core_target_micro_f1"]
        >= HUMAN_SUPPORTED_THRESHOLDS["core_target_micro_f1_min"],
        "fixed_reference_core_target_stance_accuracy": values[
            "fixed_reference_core_target_stance_accuracy"
        ]
        is not None
        and values["fixed_reference_core_target_stance_accuracy"]
        >= HUMAN_SUPPORTED_THRESHOLDS["fixed_reference_core_target_stance_accuracy_min"],
        "core_target_stance_tuple_micro_f1": values["core_target_stance_tuple_micro_f1"]
        is not None
        and values["core_target_stance_tuple_micro_f1"]
        >= HUMAN_SUPPORTED_THRESHOLDS["core_target_stance_tuple_micro_f1_min"],
    }
    return {
        "gate": "strict-human-supported-sol-direct-v1",
        "passed": all(checks.values()),
        "values": values,
        "thresholds": HUMAN_SUPPORTED_THRESHOLDS,
        "criteria_passed": checks,
        "failed_criteria": [name for name, passed in checks.items() if not passed],
    }


def _core_target_recall_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    per_class = metrics["targets"]["per_class"]
    criteria = {}
    for target in CORE_TARGETS:
        support = per_class[target]["support"]
        recall = per_class[target]["recall"]
        criteria[target] = {
            "support": support,
            "recall": recall,
            "threshold": CORE_TARGET_RECALL_MIN,
            "passed": support == 0 or (recall is not None and recall >= CORE_TARGET_RECALL_MIN),
        }
    return {
        "gate": "supported-core-target-recall-v1",
        "passed": all(value["passed"] for value in criteria.values()),
        "criteria": criteria,
    }


def _prediction_path(run_id: str, input_sha256: str) -> Path:
    return _execution_root(run_id) / f"input={input_sha256}" / "adjudicator-output.json"


def _execution_summary(run_id: str, input_sha256: str) -> dict[str, Any]:
    root = _execution_root(run_id) / f"input={input_sha256}"
    events = [_read_object(path) for path in sorted(root.glob("shard-*/event.json"))]
    if len(events) != SHARD_COUNT:
        raise ValueError(f"expected {SHARD_COUNT} execution events, found {len(events)}")
    usage_fields = ("input_tokens", "cached_input_tokens", "output_tokens")
    return {
        "shards": len(events),
        "wall_seconds": max(event["elapsed_seconds"] for event in events),
        "aggregate_shard_seconds": round(sum(event["elapsed_seconds"] for event in events), 6),
        "usage": {
            field: sum((event.get("usage") or {}).get(field, 0) for event in events)
            for field in usage_fields
        },
    }


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "items": metrics["items"],
        "invalid_outputs": metrics["invalid_outputs"],
        "relevance_accuracy": metrics["relevance"]["accuracy"],
        "relevance_macro_f1": metrics["relevance"]["macro_f1"],
        "material_precision": metrics["relevance"]["material_precision"],
        "material_recall": metrics["relevance"]["material_recall"],
        "predicted_unclear_rate": metrics["relevance"]["predicted_unclear_rate"],
        "core_target_micro_f1": metrics["targets"]["core"]["micro"]["f1"],
        "core_target_macro_f1": metrics["targets"]["core"]["macro"]["f1"],
        "fixed_core_stance_accuracy": metrics["stance"]["fixed_reference_target_core"][
            "accuracy"
        ],
        "fixed_core_stance_kappa": metrics["stance"]["fixed_reference_target_core"][
            "cohen_kappa"
        ],
        "core_target_stance_tuple_micro_f1": metrics["end_to_end_core_target_stance"][
            "micro"
        ]["f1"],
        "whole_item_exact_accuracy": metrics["diagnostics"]["exact_whole_row"]["accuracy"],
    }


def score() -> dict[str, Any]:
    prepared = prepare()
    run_id = prepared["run_id"]
    input_sha256 = prepared["input_sha256"]
    prediction_path = _prediction_path(run_id, input_sha256)
    if not prediction_path.is_file():
        raise FileNotFoundError("frozen Sol prediction output does not exist")
    output = _read_object(prediction_path)
    if (
        output.get("input_sha256") != input_sha256
        or output.get("adjudicator_model") != SOL_ADJUDICATOR_MODEL
        or output.get("adjudicator_id") != ADJUDICATOR_ID
    ):
        raise ValueError("Sol prediction binding drifted")
    prediction_rows = output.get("rows")
    if not isinstance(prediction_rows, list) or len(prediction_rows) != EXPECTED_ROWS:
        raise ValueError("Sol prediction row conservation failed")
    predictions = {row["source_sample_id"]: row["label"] for row in prediction_rows}

    _, private_reference_path = _prepared_paths(run_id)
    private_reference = _read_object(private_reference_path)
    rows = private_reference["rows"]
    proxy_reference = {row["test_item_id"]: row["proxy_label"] for row in rows}
    human_reference = {row["test_item_id"]: row["human_label"] for row in rows}
    by_resolution = {
        resolution: {
            row["test_item_id"]: row["proxy_label"]
            for row in rows
            if row["resolution"] == resolution
        }
        for resolution in (HUMAN_SUPPORTED_RESOLUTION, SOL_RESOLUTION)
    }
    overall = score_semantic_labels(proxy_reference, predictions)
    human_supported_predictions = {
        item_id: predictions[item_id] for item_id in by_resolution[HUMAN_SUPPORTED_RESOLUTION]
    }
    sol_adjudicated_predictions = {
        item_id: predictions[item_id] for item_id in by_resolution[SOL_RESOLUTION]
    }
    human_supported = score_semantic_labels(
        by_resolution[HUMAN_SUPPORTED_RESOLUTION], human_supported_predictions
    )
    sol_adjudicated = score_semantic_labels(
        by_resolution[SOL_RESOLUTION], sol_adjudicated_predictions
    )
    original_human_sensitivity = score_semantic_labels(human_reference, predictions)
    overall_gate = evaluate_capability_gate(overall)
    human_supported_gate = _criteria(human_supported)
    target_recall_gate = _core_target_recall_gate(overall)
    passed = (
        overall_gate["passed"] and human_supported_gate["passed"] and target_recall_gate["passed"]
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-direct-teacher-evaluation-metrics-v1",
        "run_id": run_id,
        "evidence_status": "model-assisted development evidence; not independent human validation",
        "input_binding": _input_binding(),
        "prediction_sha256": file_sha256(prediction_path),
        "reference_counts": {
            "all": EXPECTED_ROWS,
            "strict_human_supported_majority": len(by_resolution[HUMAN_SUPPORTED_RESOLUTION]),
            "blinded_sol_adjudication": len(by_resolution[SOL_RESOLUTION]),
        },
        "metrics": {
            "all_against_final_proxy": overall,
            "strict_human_supported_majority": human_supported,
            "blinded_sol_adjudication": sol_adjudicated,
            "all_against_original_human_sensitivity": original_human_sensitivity,
        },
        "gates": {
            "all_proxy": overall_gate,
            "strict_human_supported": human_supported_gate,
            "supported_core_target_recall": target_recall_gate,
            "passed": passed,
        },
        "execution": _execution_summary(run_id, input_sha256),
        "operational_verdict": (
            "select_sol_direct_teacher" if passed else "do_not_select_sol_direct_teacher"
        ),
        "claim_boundary": (
            "Operational teacher-selection evidence only; the later independent double-coded "
            "human evaluation remains required for thesis measurement claims."
        ),
    }
    private_metrics_path = _private_metrics_path(run_id)
    _write_immutable(private_metrics_path, metrics)
    summary_path, receipt_path = _public_paths(run_id)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-direct-teacher-evaluation-summary-v3",
        "run_id": run_id,
        "evidence_status": metrics["evidence_status"],
        "input_binding": {
            ("item_count" if key == "rows" else key): value
            for key, value in metrics["input_binding"].items()
        },
        "private_metrics_sha256": file_sha256(private_metrics_path),
        "aggregate_metrics": {
            name: _compact_metrics(value) for name, value in metrics["metrics"].items()
        },
        "gates": metrics["gates"],
        "execution": metrics["execution"],
        "operational_verdict": metrics["operational_verdict"],
        "claim_boundary": metrics["claim_boundary"],
    }
    _write_immutable(summary_path, summary)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-direct-teacher-evaluation-receipt-v3",
        "run_id": run_id,
        "summary_sha256": file_sha256(summary_path),
        "private_metrics_sha256": file_sha256(private_metrics_path),
        "prediction_sha256": file_sha256(prediction_path),
        "item_count": EXPECTED_ROWS,
        "passed": passed,
        "operational_verdict": metrics["operational_verdict"],
        "evidence_status": metrics["evidence_status"],
        "contains_raw_text": False,
        "contains_record_ids": False,
        "contains_row_level_labels": False,
        "human_gold": False,
        "independent_human_evaluation": False,
    }
    _write_immutable(receipt_path, receipt)
    return {
        "status": "scored",
        "run_id": run_id,
        "passed": passed,
        "operational_verdict": metrics["operational_verdict"],
        "summary_path": str(summary_path),
        "receipt_path": str(receipt_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "score", "all"), required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare()
    elif args.action == "run":
        result = execute()
    elif args.action == "score":
        result = score()
    else:
        execute()
        result = score()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
