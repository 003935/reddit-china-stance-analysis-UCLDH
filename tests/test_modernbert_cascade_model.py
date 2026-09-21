from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import pytest

from reddit_china_stance.modernbert_cascade_model import (
    ABSENT_TARGET_STATE_INDEX,
    TARGET_STATE_LABELS,
    ModernBertRelevanceModel,
    ModernBertTargetConditionedModel,
    compute_relevance_loss,
    compute_target_conditioned_loss,
    decode_cascade_logits,
    encode_relevance_semantic_label,
    encode_target_conditioned_semantic_label,
    expand_target_conditioned_labels,
    render_target_conditioned_text,
)
from reddit_china_stance.modernbert_model import STANCE_LABELS, TARGET_LABELS


def _semantic_label() -> dict[str, object]:
    return {
        "relevance": "material",
        "target_stances": [
            {"target": "government_ccp", "stance": "negative"},
            {"target": "people_culture", "stance": "positive"},
        ],
    }


def _all_absent_rows(*, absent: float = 5.0) -> list[list[float]]:
    return [[absent, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in TARGET_LABELS]


def test_target_conditioned_rendering_is_explicit_and_target_text_first() -> None:
    rendered = render_target_conditioned_text(
        {
            "target_text": "main text",
            "parent_context": "parent",
            "submission_context": "submission",
        },
        target="government_ccp",
        separator="<SEP>",
    )

    assert rendered == (
        "[TARGET=government_ccp] <SEP> main text <SEP> parent <SEP> submission"
    )
    with pytest.raises(ValueError, match="target must be"):
        render_target_conditioned_text(
            {"target_text": "text"},
            target="government",
            separator="<SEP>",
        )
    with pytest.raises(ValueError, match="parent_context"):
        render_target_conditioned_text(
            {"target_text": "text", "parent_context": 3},
            target="other",
            separator="<SEP>",
        )


def test_label_expansion_encodes_absent_and_stance_for_every_target() -> None:
    label = _semantic_label()

    assert encode_relevance_semantic_label(label) == 0
    assert encode_target_conditioned_semantic_label(
        label,
        target="government_ccp",
    ) == 1 + STANCE_LABELS.index("negative")
    assert encode_target_conditioned_semantic_label(
        label,
        target="china_general",
    ) == ABSENT_TARGET_STATE_INDEX
    assert expand_target_conditioned_labels(label) == (
        {"target": "china_general", "target_state_labels": 0},
        {"target": "government_ccp", "target_state_labels": 1},
        {"target": "people_culture", "target_state_labels": 4},
        {"target": "other", "target_state_labels": 0},
    )


@pytest.mark.parametrize(
    ("label", "match"),
    [
        ({"relevance": "material", "target_stances": []}, "require at least one target"),
        (
            {
                "relevance": "not_material",
                "target_stances": [{"target": "other", "stance": "mixed"}],
            },
            "cannot contain targets",
        ),
        (
            {
                "relevance": "material",
                "target_stances": [
                    {"target": "other", "stance": "mixed"},
                    {"target": "other", "stance": "positive"},
                ],
            },
            "duplicate target",
        ),
    ],
)
def test_target_label_encoding_fails_closed(label: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        encode_target_conditioned_semantic_label(label, target="other")


def test_decoder_gates_non_material_and_emits_non_absent_targets() -> None:
    material = _all_absent_rows()
    material[1] = [0.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    material[2] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]

    decoded = decode_cascade_logits(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        [material, material],
    )

    assert decoded.forced_target_selections == 0
    assert decoded.predictions == (
        {
            "relevance": "material",
            "target_stances": [
                {"target": "government_ccp", "stance": "negative"},
                {"target": "people_culture", "stance": "positive"},
            ],
        },
        {"relevance": "not_material", "target_stances": []},
    )


def test_decoder_forces_largest_non_absent_margin_with_stable_ties() -> None:
    rows = _all_absent_rows()
    rows[0][2] = 4.0
    rows[1][1] = 4.5

    decoded = decode_cascade_logits([[4.0, 0.0, 0.0]], [rows])

    assert decoded.forced_target_selections == 1
    assert decoded.predictions == (
        {
            "relevance": "material",
            "target_stances": [{"target": "government_ccp", "stance": "negative"}],
        },
    )


@pytest.mark.parametrize(
    ("relevance", "states", "match"),
    [
        ([[1.0, 0.0]], [_all_absent_rows()], "exactly 3"),
        ([[1.0, 0.0, 0.0]], [[[]] * 4], "exactly 6"),
        ([[math.nan, 0.0, 0.0]], [_all_absent_rows()], "finite"),
        ([[1.0, 0.0, 0.0]], [], "same batch size"),
    ],
)
def test_decoder_rejects_invalid_logits(
    relevance: list[list[float]],
    states: list[list[list[float]]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        decode_cascade_logits(relevance, states)


def test_base_environment_dependency_guard_is_explicit() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("base-environment guard only applies when torch is absent")
    with pytest.raises(ModuleNotFoundError, match="optional ML dependencies"):
        ModernBertRelevanceModel(object())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_models_use_masked_mean_pooling_and_have_exact_head_shapes() -> None:
    import torch
    from torch import nn

    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)

        def forward(self, *, input_ids: object, attention_mask: object) -> object:
            del input_ids, attention_mask
            return SimpleNamespace(last_hidden_state=hidden)

    relevance = ModernBertRelevanceModel(Encoder(), dropout=0.0)
    target = ModernBertTargetConditionedModel(Encoder(), dropout=0.0)
    relevance.relevance_head.weight.data.fill_(1.0)
    relevance.relevance_head.bias.data.zero_()
    target.target_state_head.weight.data.fill_(1.0)
    target.target_state_head.bias.data.zero_()

    relevance_output = relevance(input_ids=torch.ones((1, 3)), attention_mask=mask)
    target_output = target(input_ids=torch.ones((1, 3)), attention_mask=mask)

    assert relevance_output["relevance_logits"].shape == (1, 3)
    assert target_output["target_state_logits"].shape == (1, len(TARGET_STATE_LABELS))
    assert relevance_output["relevance_logits"].tolist() == [[6.0, 6.0, 6.0]]
    assert target_output["target_state_logits"].tolist() == [[6.0] * 6]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_component_losses_are_weighted_cross_entropy_and_fail_closed() -> None:
    import torch
    import torch.nn.functional as functional

    relevance_logits = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    relevance_labels = torch.tensor([0, 1])
    target_logits = torch.zeros((2, 6))
    target_labels = torch.tensor([0, 4])

    relevance_loss = compute_relevance_loss(
        relevance_logits=relevance_logits,
        relevance_labels=relevance_labels,
        class_weights=(1.0, 3.0, 1.0),
    )
    target_loss = compute_target_conditioned_loss(
        target_state_logits=target_logits,
        target_state_labels=target_labels,
        class_weights=(1.0, 1.0, 1.0, 1.0, 2.0, 1.0),
    )

    assert relevance_loss.item() == pytest.approx(
        functional.cross_entropy(
            relevance_logits,
            relevance_labels,
            weight=torch.tensor([1.0, 3.0, 1.0]),
        ).item()
    )
    assert target_loss.item() == pytest.approx(math.log(6))
    with pytest.raises(ValueError, match="out of range"):
        compute_target_conditioned_loss(
            target_state_logits=target_logits,
            target_state_labels=torch.tensor([0, 6]),
        )
    with pytest.raises(ValueError, match="zero weight"):
        compute_relevance_loss(
            relevance_logits=relevance_logits,
            relevance_labels=relevance_labels,
            class_weights=(1.0, 0.0, 1.0),
        )
