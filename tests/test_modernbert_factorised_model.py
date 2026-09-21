from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from reddit_china_stance.modernbert_factorised_model import (
    ANALYTIC_TARGETS,
    B4_STANCE_LABELS,
    STANCES,
    TARGETS,
    FactorisedHeadConfig,
    ModernBertBinaryRelevanceModel,
    ModernBertFactorisedTargetStanceModel,
    assert_independent_encoders,
    b2_bits_for_stance,
    b2_stance_from_bits,
    compute_factorised_target_stance_losses,
    compute_masked_b2_stance_loss,
    compute_masked_b4_stance_loss,
    compute_masked_relevance_loss,
    compute_masked_target_presence_loss,
)


def test_frozen_v2_orders_and_b2_mapping_are_exact() -> None:
    assert TARGETS == (
        "china_general",
        "government_ccp",
        "people_identity",
        "culture_media",
        "company_tech_product",
        "residual_other",
    )
    assert TARGETS[:-1] == ANALYTIC_TARGETS
    assert B4_STANCE_LABELS == (
        "negative",
        "mixed",
        "no_directed_stance",
        "positive",
    )
    assert frozenset(B4_STANCE_LABELS) == frozenset(STANCES)
    expected = {
        "negative": (1, 0),
        "positive": (0, 1),
        "mixed": (1, 1),
        "no_directed_stance": (0, 0),
    }
    for stance, bits in expected.items():
        assert b2_bits_for_stance(stance) == bits
        assert b2_stance_from_bits(bits) == stance
    with pytest.raises(ValueError, match="stance must be"):
        b2_bits_for_stance("unclear")
    with pytest.raises(ValueError, match="two binary"):
        b2_stance_from_bits((1,))
    with pytest.raises(ValueError, match="two binary"):
        b2_stance_from_bits((1, 0.0))


def test_frozen_configs_fail_closed() -> None:
    assert FactorisedHeadConfig("b4").stance_variant == "b4"
    with pytest.raises(ValueError, match="stance_variant"):
        FactorisedHeadConfig("combined")
    with pytest.raises(ValueError, match="dropout"):
        FactorisedHeadConfig("b4", dropout=1.0)


def test_base_environment_import_guard_is_explicit() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("base-environment guard only applies when torch is absent")
    with pytest.raises(ModuleNotFoundError, match="optional ML dependencies"):
        ModernBertBinaryRelevanceModel(object())
    with pytest.raises(ModuleNotFoundError, match="optional ML dependencies"):
        ModernBertFactorisedTargetStanceModel(
            object(),
            config=FactorisedHeadConfig("b4"),
        )


def test_module_imports_when_torch_and_transformers_are_unavailable() -> None:
    code = """
import sys
sys.modules['torch'] = None
sys.modules['transformers'] = None
import reddit_china_stance.modernbert_factorised_model as model
assert model.torch is None
try:
    model.ModernBertBinaryRelevanceModel(object())
except ModuleNotFoundError as exc:
    assert 'optional ML dependencies' in str(exc)
else:
    raise AssertionError('missing dependency guard did not fail')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_masked_losses_equal_standard_unweighted_bce_and_ce() -> None:
    import torch
    import torch.nn.functional as functional

    relevance_logits = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    relevance_labels = torch.tensor([1, -100, 0])
    codable = torch.tensor([True, False, True])
    relevance_loss = compute_masked_relevance_loss(
        relevance_logits=relevance_logits,
        relevance_labels=relevance_labels,
        codable_mask=codable,
    )
    assert relevance_loss.item() == pytest.approx(
        functional.binary_cross_entropy_with_logits(
            relevance_logits[codable],
            relevance_labels[codable].float(),
        ).item()
    )

    target_logits = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10
    target_labels = torch.tensor(
        [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1]]
    )
    material = torch.tensor([True, False, True])
    target_loss = compute_masked_target_presence_loss(
        target_presence_logits=target_logits,
        target_presence_labels=target_labels,
        reference_material_mask=material,
    )
    assert target_loss.item() == pytest.approx(
        functional.binary_cross_entropy_with_logits(
            target_logits[material],
            target_labels[material].float(),
        ).item()
    )

    b4_logits = torch.arange(60, dtype=torch.float32).reshape(3, 5, 4) / 10
    b4_labels = torch.tensor(
        [
            [0, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100],
            [-100, 3, -100, -100, -100],
        ]
    )
    known = torch.ones((3, 5), dtype=torch.bool)
    b4_loss = compute_masked_b4_stance_loss(
        stance_logits=b4_logits,
        stance_labels=b4_labels,
        stance_known_mask=known,
        target_presence_labels=target_labels,
        reference_material_mask=material,
    )
    selected = known & target_labels[:, :5].bool() & material.unsqueeze(1)
    assert b4_loss.item() == pytest.approx(
        functional.cross_entropy(b4_logits[selected], b4_labels[selected]).item()
    )

    b2_logits = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2) / 10
    b2_labels = torch.zeros((3, 5, 2), dtype=torch.long)
    b2_labels[0, 0] = torch.tensor([1, 0])
    b2_labels[2, 1] = torch.tensor([0, 1])
    b2_loss = compute_masked_b2_stance_loss(
        stance_logits=b2_logits,
        stance_labels=b2_labels,
        stance_known_mask=known,
        target_presence_labels=target_labels,
        reference_material_mask=material,
    )
    assert b2_loss.item() == pytest.approx(
        functional.binary_cross_entropy_with_logits(
            b2_logits[selected],
            b2_labels[selected].float(),
        ).item()
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_absent_targets_and_non_material_rows_have_no_stance_loss() -> None:
    import torch

    relevance_logits = torch.randn(2, requires_grad=True)
    relevance_loss = compute_masked_relevance_loss(
        relevance_logits=relevance_logits,
        relevance_labels=torch.tensor([-100, -100]),
        codable_mask=torch.tensor([False, False]),
    )
    assert relevance_loss.item() == 0.0
    relevance_loss.backward()
    assert relevance_logits.grad is not None
    assert torch.count_nonzero(relevance_logits.grad).item() == 0

    stance_logits = torch.randn((2, 5, 4), requires_grad=True)
    target_logits = torch.randn((2, 6), requires_grad=True)
    target_labels = torch.zeros((2, 6), dtype=torch.long)
    known = torch.ones((2, 5), dtype=torch.bool)
    material = torch.tensor([False, False])
    losses = compute_factorised_target_stance_losses(
        target_presence_logits=target_logits,
        stance_logits=stance_logits,
        target_presence_labels=target_labels,
        stance_labels=torch.full((2, 5), -100),
        reference_material_mask=material,
        stance_known_mask=known,
        stance_variant="b4",
    )

    assert losses["target_presence_loss"].item() == 0.0
    assert losses["stance_loss"].item() == 0.0
    losses["loss"].backward()
    assert target_logits.grad is not None
    assert stance_logits.grad is not None
    assert torch.count_nonzero(target_logits.grad).item() == 0
    assert torch.count_nonzero(stance_logits.grad).item() == 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_absent_analytic_target_is_excluded_even_when_marked_known() -> None:
    import torch

    logits = torch.zeros((1, 5, 4), requires_grad=True)
    logits.data[0, 1] = torch.tensor([100.0, -100.0, -100.0, -100.0])
    labels = torch.tensor([[0, 3, -100, -100, -100]])
    targets = torch.tensor([[1, 0, 0, 0, 0, 0]])
    loss = compute_masked_b4_stance_loss(
        stance_logits=logits,
        stance_labels=labels,
        stance_known_mask=torch.tensor([[True, True, False, False, False]]),
        target_presence_labels=targets,
        reference_material_mask=torch.tensor([True]),
    )
    assert loss.item() == pytest.approx(math.log(4))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_loss_helpers_reject_wrong_shapes_and_non_boolean_masks() -> None:
    import torch

    with pytest.raises(ValueError, match="shape"):
        compute_masked_relevance_loss(
            relevance_logits=torch.zeros(2),
            relevance_labels=torch.zeros(1),
            codable_mask=torch.ones(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="boolean"):
        compute_masked_target_presence_loss(
            target_presence_logits=torch.zeros((2, 6)),
            target_presence_labels=torch.zeros((2, 6)),
            reference_material_mask=torch.ones(2, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="shape"):
        compute_masked_b2_stance_loss(
            stance_logits=torch.zeros((2, 5, 4)),
            stance_labels=torch.zeros((2, 5, 2)),
            stance_known_mask=torch.ones((2, 5), dtype=torch.bool),
            target_presence_labels=torch.zeros((2, 6)),
            reference_material_mask=torch.ones(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="integer dtype"):
        compute_masked_b4_stance_loss(
            stance_logits=torch.zeros((1, 5, 4)),
            stance_labels=torch.zeros((1, 5), dtype=torch.float32),
            stance_known_mask=torch.ones((1, 5), dtype=torch.bool),
            target_presence_labels=torch.ones((1, 6)),
            reference_material_mask=torch.ones(1, dtype=torch.bool),
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_models_have_exact_shapes_and_independent_gradient_paths() -> None:
    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, *, input_ids: object, attention_mask: object) -> object:
            del input_ids
            hidden = torch.ones((2, 3, 2), device=self.scale.device) * self.scale
            return SimpleNamespace(last_hidden_state=hidden)

    mask = torch.ones((2, 3), dtype=torch.long)
    relevance = ModernBertBinaryRelevanceModel(Encoder(), dropout=0.0)
    b4 = ModernBertFactorisedTargetStanceModel(
        Encoder(),
        config=FactorisedHeadConfig("b4", dropout=0.0),
    )
    assert_independent_encoders(relevance, b4)

    relevance_output = relevance(
        input_ids=torch.ones((2, 3)),
        attention_mask=mask,
        relevance_labels=torch.tensor([1, 0]),
        codable_mask=torch.tensor([True, True]),
    )
    b4_output = b4(input_ids=torch.ones((2, 3)), attention_mask=mask)
    assert relevance_output["relevance_logits"].shape == (2,)
    assert b4_output["target_presence_logits"].shape == (2, 6)
    assert b4_output["stance_logits"].shape == (2, 5, 4)

    relevance_output["loss"].backward()
    assert relevance.encoder.scale.grad is not None
    assert b4.encoder.scale.grad is None

    b2 = ModernBertFactorisedTargetStanceModel(
        Encoder(),
        config=FactorisedHeadConfig("b2", dropout=0.0),
    )
    b2_output = b2(input_ids=torch.ones((2, 3)), attention_mask=mask)
    assert b2_output["target_presence_logits"].shape == (2, 6)
    assert b2_output["stance_logits"].shape == (2, 5, 2)

    shared = Encoder()
    with pytest.raises(ValueError, match="independent encoders"):
        assert_independent_encoders(
            ModernBertBinaryRelevanceModel(shared),
            ModernBertFactorisedTargetStanceModel(
                shared,
                config=FactorisedHeadConfig("b4"),
            ),
        )
