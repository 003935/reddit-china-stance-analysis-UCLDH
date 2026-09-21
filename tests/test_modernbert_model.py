from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import pytest

from reddit_china_stance.modernbert_model import (
    ABSENT_STANCE_INDEX,
    STANCE_LABELS,
    TARGET_THRESHOLD_GRID,
    ModernBertThreeHeadModel,
    capped_inverse_sqrt_class_weights,
    compute_multitask_losses,
    decode_probabilities,
    masked_mean_pool,
    select_global_target_threshold,
    target_positive_weights,
)


def _stance_rows(index: int = 0) -> list[list[float]]:
    rows = []
    for _ in range(4):
        row = [0.0] * len(STANCE_LABELS)
        row[index] = 1.0
        rows.append(row)
    return rows


def test_class_weights_follow_registered_formula_and_ignore_unsupported_classes() -> None:
    weights = capped_inverse_sqrt_class_weights((100, 25, 0))

    assert weights[2] == 0.0
    assert weights[0] == pytest.approx(2 / 3)
    assert weights[1] == pytest.approx(4 / 3)
    assert target_positive_weights((25, 5, 0, 20), material_count=25) == pytest.approx(
        (1.0, 2.0, 1.0, 1.0)
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: capped_inverse_sqrt_class_weights((0, 0)), "supported class"),
        (lambda: capped_inverse_sqrt_class_weights((1, -1)), "non-negative"),
        (
            lambda: target_positive_weights((1, 2, 3), material_count=3),
            "exactly 4",
        ),
        (
            lambda: target_positive_weights((1, 2, 3, 4), material_count=3),
            "between zero",
        ),
    ],
)
def test_weight_helpers_fail_closed(call: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()  # type: ignore[operator]


def test_decoder_enforces_relevance_target_and_stance_constraints() -> None:
    decoded = decode_probabilities(
        relevance_probabilities=[
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.6, 0.2, 0.2],
        ],
        target_probabilities=[
            [0.2, 0.8, 0.7, 0.1],
            [0.9, 0.9, 0.9, 0.9],
            [0.4, 0.3, 0.2, 0.1],
        ],
        stance_probabilities=[_stance_rows(3), _stance_rows(0), _stance_rows(2)],
        target_threshold=0.5,
    )

    assert decoded.forced_target_selections == 1
    assert decoded.predictions == (
        {
            "relevance": "material",
            "target_stances": [
                {"target": "government_ccp", "stance": "positive"},
                {"target": "people_culture", "stance": "positive"},
            ],
        },
        {"relevance": "not_material", "target_stances": []},
        {
            "relevance": "material",
            "target_stances": [{"target": "china_general", "stance": "no_directed_stance"}],
        },
    )


def test_decoder_rejects_bad_shapes_probabilities_and_threshold() -> None:
    with pytest.raises(ValueError, match="same batch size"):
        decode_probabilities([[1, 0, 0]], [], [], target_threshold=0.5)
    with pytest.raises(ValueError, match="sum to one"):
        decode_probabilities(
            [[0.8, 0.8, 0.0]],
            [[0.5] * 4],
            [_stance_rows()],
            target_threshold=0.5,
        )
    with pytest.raises(ValueError, match="finite probability"):
        decode_probabilities([], [], [], target_threshold=math.nan)


def test_threshold_selection_uses_exact_grid_and_breaks_ties_upward() -> None:
    scores = dict.fromkeys(TARGET_THRESHOLD_GRID, 0.5)
    scores[0.4] = 0.8
    scores[0.6] = 0.8

    selected = select_global_target_threshold(scores)

    assert selected.threshold == 0.6
    assert selected.score == 0.8
    with pytest.raises(ValueError, match="exactly"):
        select_global_target_threshold({0.5: 1.0})
    with pytest.raises(ValueError, match="finite numbers"):
        select_global_target_threshold({value: math.nan for value in TARGET_THRESHOLD_GRID})


def test_base_environment_import_guard_is_explicit() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("base-environment guard only applies when torch is absent")
    with pytest.raises(ModuleNotFoundError, match="optional ML dependencies"):
        ModernBertThreeHeadModel(object())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is an optional extra")
def test_masked_pool_losses_and_model_shapes() -> None:
    import torch
    from torch import nn

    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert torch.allclose(masked_mean_pool(hidden, mask), torch.tensor([[2.0, 3.0], [4.0, 8.0]]))

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)

        def forward(self, *, input_ids: object, attention_mask: object) -> object:
            del input_ids, attention_mask
            return SimpleNamespace(last_hidden_state=hidden)

    model = ModernBertThreeHeadModel(Encoder(), dropout=0.0)
    outputs = model(input_ids=torch.ones((2, 3)), attention_mask=mask)
    assert outputs["relevance_logits"].shape == (2, 3)
    assert outputs["target_logits"].shape == (2, 4)
    assert outputs["stance_logits"].shape == (2, 4, 5)

    relevance_logits = torch.zeros((2, 3), requires_grad=True)
    target_logits = torch.zeros((2, 4), requires_grad=True)
    stance_logits = torch.zeros((2, 4, 5), requires_grad=True)
    losses = compute_multitask_losses(
        relevance_logits=relevance_logits,
        target_logits=target_logits,
        stance_logits=stance_logits,
        relevance_labels=torch.tensor([1, 2]),
        target_labels=torch.zeros((2, 4), dtype=torch.long),
        stance_labels=torch.full((2, 4), ABSENT_STANCE_INDEX, dtype=torch.long),
    )
    assert losses["target_loss"].item() == 0.0
    assert losses["stance_loss"].item() == 0.0
    losses["loss"].backward()
    assert target_logits.grad is not None
    assert stance_logits.grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is an optional extra")
def test_exact_masked_loss_reductions_and_label_validation() -> None:
    import torch
    import torch.nn.functional as functional

    relevance_logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True)
    target_logits = torch.zeros((2, 4), requires_grad=True)
    stance_logits = torch.zeros((2, 4, 5), requires_grad=True)
    relevance_labels = torch.tensor([0, 1])
    target_labels = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]])
    stance_labels = torch.tensor(
        [[3, ABSENT_STANCE_INDEX, 0, ABSENT_STANCE_INDEX], [ABSENT_STANCE_INDEX] * 4]
    )

    losses = compute_multitask_losses(
        relevance_logits=relevance_logits,
        target_logits=target_logits,
        stance_logits=stance_logits,
        relevance_labels=relevance_labels,
        target_labels=target_labels,
        stance_labels=stance_labels,
        task_loss_weights=(0.5, 1.0, 1.5),
    )

    assert losses["relevance_loss"].item() == pytest.approx(
        functional.cross_entropy(relevance_logits, relevance_labels).item()
    )
    assert losses["target_loss"].item() == pytest.approx(math.log(2))
    assert losses["stance_loss"].item() == pytest.approx(math.log(5))
    assert losses["loss"].item() == pytest.approx(
        0.5 * losses["relevance_loss"].item()
        + losses["target_loss"].item()
        + 1.5 * losses["stance_loss"].item()
    )

    bad_stances = stance_labels.clone()
    bad_stances[0, 1] = 0
    with pytest.raises(ValueError, match="absent-target"):
        compute_multitask_losses(
            relevance_logits=relevance_logits,
            target_logits=target_logits,
            stance_logits=stance_logits,
            relevance_labels=relevance_labels,
            target_labels=target_labels,
            stance_labels=bad_stances,
        )

    with pytest.raises(ValueError, match="integer dtypes"):
        compute_multitask_losses(
            relevance_logits=relevance_logits,
            target_logits=target_logits,
            stance_logits=stance_logits,
            relevance_labels=relevance_labels.float(),
            target_labels=target_labels,
            stance_labels=stance_labels,
        )

    with pytest.raises(ValueError, match="zero class weight"):
        compute_multitask_losses(
            relevance_logits=relevance_logits,
            target_logits=target_logits,
            stance_logits=stance_logits,
            relevance_labels=relevance_labels,
            target_labels=target_labels,
            stance_labels=stance_labels,
            relevance_class_weights=(1.0, 0.0, 1.0),
        )
