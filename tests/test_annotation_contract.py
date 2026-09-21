from __future__ import annotations

import pytest

from reddit_china_stance.annotation_contract import (
    compose_label,
    model_input,
    validate_phase_output,
)


def test_model_input_exposes_only_bounded_text() -> None:
    value = model_input(
        {
            "target_text": "Synthetic target text.",
            "submission_context": "Synthetic submission context.",
            "parent_context": None,
        },
        target="government_ccp",
    )
    assert "TARGET_CLASS\ngovernment_ccp" in value
    assert "record_id" not in value
    with pytest.raises(ValueError, match="only target text"):
        model_input({"target_text": "x", "record_id": "private"})


def test_phase_outputs_are_strict_and_targets_are_canonical() -> None:
    assert validate_phase_output(
        "targets", {"targets": ["other", "china_general"]}
    ) == {"targets": ["china_general", "other"]}
    with pytest.raises(ValueError, match="invalid target"):
        validate_phase_output("targets", {"targets": ["china_general", "china_general"]})
    with pytest.raises(ValueError, match="invalid relevance"):
        validate_phase_output("relevance", {"relevance": "direct"})


def test_compose_label_enforces_phase_dependencies() -> None:
    assert compose_label({"relevance": "not_material"}) == {
        "relevance": "not_material",
        "target_stances": [],
    }
    assert compose_label(
        {"relevance": "material"},
        {"targets": ["people_culture", "government_ccp"]},
        {
            "government_ccp": {"stance": "negative"},
            "people_culture": {"stance": "positive"},
        },
    ) == {
        "relevance": "material",
        "target_stances": [
            {"target": "government_ccp", "stance": "negative"},
            {"target": "people_culture", "stance": "positive"},
        ],
    }
    with pytest.raises(ValueError, match="exactly one stance"):
        compose_label(
            {"relevance": "material"},
            {"targets": ["government_ccp"]},
            {},
        )
