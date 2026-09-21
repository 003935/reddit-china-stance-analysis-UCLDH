from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import pytest

from reddit_china_stance.modernbert_conditional_model import (
    ABSENT_TARGET_STATE_INDEX,
    TARGET_STATE_LABELS,
    ModernBertConditionalStateModel,
    compute_conditional_losses,
    decode_conditional_logits,
    encode_conditional_semantic_label,
)
from reddit_china_stance.modernbert_model import STANCE_LABELS


def _all_absent_rows(*, absent_logit: float = 2.0) -> list[list[float]]:
    return [[absent_logit, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(4)]


def test_label_construction_maps_absent_and_stance_states() -> None:
    encoded = encode_conditional_semantic_label(
        {
            "relevance": "material",
            "target_stances": [
                {"target": "government_ccp", "stance": "negative"},
                {"target": "people_culture", "stance": "positive"},
            ],
        }
    )

    assert encoded == {
        "relevance_labels": 0,
        "target_state_labels": [
            ABSENT_TARGET_STATE_INDEX,
            1 + STANCE_LABELS.index("negative"),
            1 + STANCE_LABELS.index("positive"),
            ABSENT_TARGET_STATE_INDEX,
        ],
    }
    assert encode_conditional_semantic_label(
        {"relevance": "not_material", "target_stances": []}
    ) == {
        "relevance_labels": 1,
        "target_state_labels": [ABSENT_TARGET_STATE_INDEX] * 4,
    }


@pytest.mark.parametrize(
    ("label", "match"),
    [
        (
            {"relevance": "material", "target_stances": []},
            "require at least one target",
        ),
        (
            {
                "relevance": "unclear",
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
        (
            {"relevance": "material", "target_stances": "not-a-list"},
            "must be a list",
        ),
    ],
)
def test_label_construction_fails_closed(label: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        encode_conditional_semantic_label(label)


def test_decoder_emits_every_non_absent_state_and_gates_on_relevance() -> None:
    material_states = _all_absent_rows()
    material_states[0] = [0.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    material_states[2] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]

    decoded = decode_conditional_logits(
        relevance_logits=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        target_state_logits=[material_states, material_states],
    )

    assert decoded.forced_target_selections == 0
    assert decoded.predictions == (
        {
            "relevance": "material",
            "target_stances": [
                {"target": "china_general", "stance": "negative"},
                {"target": "people_culture", "stance": "positive"},
            ],
        },
        {"relevance": "not_material", "target_stances": []},
    )


def test_decoder_forces_largest_non_absent_vs_absent_margin() -> None:
    rows = _all_absent_rows(absent_logit=5.0)
    rows[0][1] = 4.0  # margin -1.0
    rows[1][3] = 4.5  # margin -0.5: force this target and state
    rows[2][2] = 3.0
    rows[3][5] = 2.0

    decoded = decode_conditional_logits([[3.0, 0.0, 0.0]], [rows])

    assert decoded.forced_target_selections == 1
    assert decoded.predictions == (
        {
            "relevance": "material",
            "target_stances": [{"target": "government_ccp", "stance": "no_directed_stance"}],
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
def test_decoder_fails_closed_on_shape_and_logit_errors(
    relevance: list[list[float]],
    states: list[list[list[float]]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        decode_conditional_logits(relevance, states)


def test_base_environment_import_guard_is_explicit() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("base-environment guard only applies when torch is absent")
    with pytest.raises(ModuleNotFoundError, match="optional ML dependencies"):
        ModernBertConditionalStateModel(object())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_model_shapes_and_label_pair_contract() -> None:
    import torch
    from torch import nn

    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)

        def forward(self, *, input_ids: object, attention_mask: object) -> object:
            del input_ids, attention_mask
            return SimpleNamespace(last_hidden_state=hidden)

    model = ModernBertConditionalStateModel(Encoder(), dropout=0.0)
    outputs = model(input_ids=torch.ones((2, 3)), attention_mask=mask)

    assert outputs["relevance_logits"].shape == (2, 3)
    assert outputs["target_state_logits"].shape == (2, 4, len(TARGET_STATE_LABELS))
    with pytest.raises(ValueError, match="both label tensors"):
        model(
            input_ids=torch.ones((2, 3)),
            attention_mask=mask,
            relevance_labels=torch.tensor([0, 1]),
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_losses_mask_non_material_rows_and_apply_task_weights() -> None:
    import torch
    import torch.nn.functional as functional

    relevance_logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        requires_grad=True,
    )
    state_logits = torch.zeros((2, 4, 6), requires_grad=True)
    relevance_labels = torch.tensor([0, 1])
    state_labels = torch.tensor([[1, 0, 4, 0], [0, 0, 0, 0]])

    losses = compute_conditional_losses(
        relevance_logits=relevance_logits,
        target_state_logits=state_logits,
        relevance_labels=relevance_labels,
        target_state_labels=state_labels,
        relevance_loss_weight=0.5,
        target_state_loss_weight=1.5,
    )

    assert losses["relevance_loss"].item() == pytest.approx(
        functional.cross_entropy(relevance_logits, relevance_labels).item()
    )
    assert losses["target_state_loss"].item() == pytest.approx(math.log(6))
    assert losses["loss"].item() == pytest.approx(
        0.5 * losses["relevance_loss"].item() + 1.5 * losses["target_state_loss"].item()
    )

    perturbed = state_logits.detach().clone()
    perturbed[1] = 100.0
    masked = compute_conditional_losses(
        relevance_logits=relevance_logits,
        target_state_logits=perturbed,
        relevance_labels=relevance_labels,
        target_state_labels=state_labels,
    )
    assert masked["target_state_loss"].item() == pytest.approx(losses["target_state_loss"].item())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_class_weights_change_target_state_loss_and_zero_observed_weights_fail() -> None:
    import torch

    relevance_logits = torch.zeros((1, 3))
    state_logits = torch.zeros((1, 4, 6))
    relevance_labels = torch.tensor([0])
    state_labels = torch.tensor([[1, 0, 0, 0]])

    unweighted = compute_conditional_losses(
        relevance_logits=relevance_logits,
        target_state_logits=state_logits,
        relevance_labels=relevance_labels,
        target_state_labels=state_labels,
    )
    weighted = compute_conditional_losses(
        relevance_logits=relevance_logits,
        target_state_logits=state_logits,
        relevance_labels=relevance_labels,
        target_state_labels=state_labels,
        target_state_class_weights=(2.0, 3.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert weighted["target_state_loss"].item() == pytest.approx(
        2.25 * unweighted["target_state_loss"].item()
    )

    with pytest.raises(ValueError, match="observed target-state class"):
        compute_conditional_losses(
            relevance_logits=relevance_logits,
            target_state_logits=state_logits,
            relevance_labels=relevance_labels,
            target_state_labels=state_labels,
            target_state_class_weights=(1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_relevance_class_weights_use_weighted_cross_entropy() -> None:
    import torch
    import torch.nn.functional as functional

    relevance_logits = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    relevance_labels = torch.tensor([0, 1])
    state_logits = torch.zeros((2, 4, 6))
    state_labels = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
    weights = torch.tensor([1.0, 3.0, 1.0])

    losses = compute_conditional_losses(
        relevance_logits=relevance_logits,
        target_state_logits=state_logits,
        relevance_labels=relevance_labels,
        target_state_labels=state_labels,
        relevance_class_weights=weights,
    )

    assert losses["relevance_loss"].item() == pytest.approx(
        functional.cross_entropy(relevance_logits, relevance_labels, weight=weights).item()
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_all_non_material_batch_has_zero_target_state_loss() -> None:
    import torch

    state_logits = torch.randn((2, 4, 6), requires_grad=True)
    losses = compute_conditional_losses(
        relevance_logits=torch.zeros((2, 3), requires_grad=True),
        target_state_logits=state_logits,
        relevance_labels=torch.tensor([1, 2]),
        target_state_labels=torch.zeros((2, 4), dtype=torch.long),
    )

    assert losses["target_state_loss"].item() == 0.0
    losses["loss"].backward()
    assert state_logits.grad is not None
    assert torch.count_nonzero(state_logits.grad).item() == 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_losses_fail_closed_on_invalid_shapes_labels_and_weights() -> None:
    import torch

    valid = {
        "relevance_logits": torch.zeros((2, 3)),
        "target_state_logits": torch.zeros((2, 4, 6)),
        "relevance_labels": torch.tensor([0, 1]),
        "target_state_labels": torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]),
    }
    with pytest.raises(ValueError, match="shape"):
        compute_conditional_losses(**{**valid, "target_state_logits": torch.zeros((2, 4, 5))})
    with pytest.raises(ValueError, match="integer dtypes"):
        compute_conditional_losses(
            **{**valid, "target_state_labels": valid["target_state_labels"].float()}
        )
    with pytest.raises(ValueError, match="require targets"):
        compute_conditional_losses(
            **{**valid, "target_state_labels": torch.zeros((2, 4), dtype=torch.long)}
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        compute_conditional_losses(**valid, relevance_loss_weight=math.nan)
    with pytest.raises(ValueError, match="at least one"):
        compute_conditional_losses(
            **valid,
            relevance_loss_weight=0.0,
            target_state_loss_weight=0.0,
        )
