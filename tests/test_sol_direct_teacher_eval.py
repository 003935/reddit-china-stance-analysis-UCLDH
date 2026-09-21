from __future__ import annotations

import json
from pathlib import Path

import pytest

import reddit_china_stance.sol_direct_teacher_eval as direct


def test_prepared_blinded_input_exposes_no_labels_or_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    reference = {
        "rows": [
            {
                "source_sample_id": f"source-{index}",
                "split": "locked_test_candidate",
                "target_text": f"text {index}",
                "submission_context": None,
                "parent_context": None,
                "label": {"relevance": "not_material", "target_stances": []},
            }
            for index in range(2)
        ]
    }
    proxy = {
        "proxy_id": "p" * 64,
        "rows": [
            {
                "source_sample_id": f"source-{index}",
                "split": "locked_test_candidate",
                "resolution": direct.HUMAN_SUPPORTED_RESOLUTION,
                "label": {"relevance": "not_material", "target_stances": []},
            }
            for index in range(2)
        ],
    }
    reference_path = tmp_path / "reference.json"
    proxy_path = tmp_path / "proxy.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    proxy_path.write_text(json.dumps(proxy), encoding="utf-8")
    monkeypatch.setattr(direct, "REFERENCE_PATH", reference_path)
    monkeypatch.setattr(direct, "PROXY_PATH", proxy_path)
    monkeypatch.setattr(direct, "PRIVATE_ROOT", tmp_path / "private")
    monkeypatch.setattr(direct, "EXPECTED_ROWS", 2)

    result = direct.prepare()
    blinded = json.loads(Path(result["input_path"]).read_text(encoding="utf-8"))
    assert all(
        set(row) == {
            "source_sample_id",
            "target_text",
            "submission_context",
            "parent_context",
        }
        for row in blinded["rows"]
    )
    serialised_rows = json.dumps(blinded["rows"])
    assert "label" not in serialised_rows
    assert "resolution" not in serialised_rows
    assert "source-0" not in serialised_rows
    assert len(blinded["rows"]) == 2


def test_human_supported_gate_is_conjunctive() -> None:
    metrics = {
        "invalid_outputs": 0,
        "relevance": {"material_recall": 0.9},
        "targets": {"core": {"micro": {"f1": 0.8}}},
        "stance": {"fixed_reference_target_core": {"accuracy": 0.8}},
        "end_to_end_core_target_stance": {"micro": {"f1": 0.69}},
    }
    result = direct._criteria(metrics)
    assert result["passed"] is False
    assert result["failed_criteria"] == ["core_target_stance_tuple_micro_f1"]


def test_core_target_recall_gate_skips_only_unsupported_targets() -> None:
    metrics = {
        "targets": {
            "per_class": {
                "china_general": {"support": 3, "recall": 0.666667},
                "government_ccp": {"support": 2, "recall": 0.5},
                "people_culture": {"support": 0, "recall": None},
            }
        }
    }
    result = direct._core_target_recall_gate(metrics)
    assert result["passed"] is False
    assert result["criteria"]["people_culture"]["passed"] is True
    assert result["criteria"]["government_ccp"]["passed"] is False


def test_score_requires_frozen_predictions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        direct,
        "prepare",
        lambda: {"run_id": "r" * 64, "input_sha256": "i" * 64},
    )
    monkeypatch.setattr(direct, "PRIVATE_ROOT", tmp_path / "private")
    with pytest.raises(FileNotFoundError, match="prediction output"):
        direct.score()
