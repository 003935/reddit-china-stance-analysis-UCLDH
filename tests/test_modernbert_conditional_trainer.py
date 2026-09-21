from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from reddit_china_stance.modernbert_conditional_model import STANCE_LABELS
from reddit_china_stance.modernbert_conditional_trainer import (
    ConditionalDynamicPaddingCollator,
    ConditionalOptimisationConfig,
    aggregate_conditional_metrics,
    conditional_model_loss_kwargs,
    create_modernbert_conditional_state_model,
    encode_conditional_label,
    evaluate_epoch,
    train_epoch,
)
from reddit_china_stance.modernbert_trainer import MODEL_ID, MODEL_REVISION


class FakeTokenizer:
    sep_token = "<SEP>"

    def pad(self, features, *, padding, return_tensors):
        assert padding is True
        width = max(len(feature["input_ids"]) for feature in features)
        result = {
            "input_ids": [
                feature["input_ids"] + [0] * (width - len(feature["input_ids"]))
                for feature in features
            ],
            "attention_mask": [
                feature["attention_mask"] + [0] * (width - len(feature["attention_mask"]))
                for feature in features
            ],
        }
        if return_tensors == "pt":
            import torch

            return {key: torch.tensor(value) for key, value in result.items()}
        return result


def _config(**overrides: object) -> ConditionalOptimisationConfig:
    values = {
        "encoder_learning_rate": 3e-5,
        "use_bf16": False,
        "lambda_relevance": 2.0,
        "lambda_target_state": 1.0,
        "relevance_class_weights": (1.0, 2.0, 3.0),
        "target_state_class_weights": (0.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    }
    values.update(overrides)
    return ConditionalOptimisationConfig(**values)


def test_encoding_covers_all_target_slots_on_material_rows() -> None:
    encoded = encode_conditional_label(
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
            0,
            1 + STANCE_LABELS.index("negative"),
            1 + STANCE_LABELS.index("positive"),
            0,
        ],
    }


def test_collator_preserves_exact_conditional_contract_without_torch() -> None:
    collator = ConditionalDynamicPaddingCollator(FakeTokenizer(), return_tensors=None)
    batch = collator(
        [
            {
                "item_id": "b",
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "relevance_labels": 0,
                "target_state_labels": [1, 0, 0, 0],
                "target_labels": [1, 0, 0, 0],
            },
            {
                "item_id": "a",
                "input_ids": [3],
                "attention_mask": [1],
                "relevance_labels": 1,
                "target_state_labels": [0, 0, 0, 0],
                "stance_labels": [-100, -100, -100, -100],
            },
        ]
    )
    assert batch == {
        "input_ids": [[1, 2], [3, 0]],
        "attention_mask": [[1, 1], [1, 0]],
        "item_ids": ["b", "a"],
        "relevance_labels": [0, 1],
        "target_state_labels": [[1, 0, 0, 0], [0, 0, 0, 0]],
    }

    with pytest.raises(ValueError, match="every row"):
        collator(
            [
                {"input_ids": [1], "attention_mask": [1], "relevance_labels": 0},
                {"input_ids": [2], "attention_mask": [1]},
            ]
        )


def test_config_and_model_factory_bind_exact_weights_and_pinned_encoder() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Encoder:
        def __init__(self) -> None:
            self.config = SimpleNamespace(hidden_size=4, _commit_hash=MODEL_REVISION)

    def loader(model_id: str, **kwargs: object) -> Encoder:
        calls.append((model_id, kwargs))
        return Encoder()

    config = _config()
    model = create_modernbert_conditional_state_model(
        optimisation_config=config,
        encoder_loader=loader,
    )

    assert model.encoder.config.reference_compile is False
    assert calls == [
        (
            MODEL_ID,
            {"revision": MODEL_REVISION, "attn_implementation": "sdpa"},
        )
    ]
    assert conditional_model_loss_kwargs(config) == {
        "relevance_loss_weight": 2.0,
        "target_state_loss_weight": 1.0,
        "relevance_class_weights": (1.0, 2.0, 3.0),
        "target_state_class_weights": (0.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    }
    with pytest.raises(ValueError, match="at least one task"):
        _config(lambda_relevance=0.0, lambda_target_state=0.0)


def test_aggregate_metrics_use_conditional_decoder_and_registered_scorer() -> None:
    reference = {
        "one": {
            "relevance": "material",
            "target_stances": [{"target": "government_ccp", "stance": "negative"}],
        },
        "two": {"relevance": "not_material", "target_stances": []},
    }
    target_states = [
        [
            [5, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0],
        ],
        [[0, 5, 0, 0, 0, 0]] * 4,
    ]
    metrics = aggregate_conditional_metrics(
        reference,
        item_ids=["one", "two"],
        relevance_logits=[[5, 0, 0], [0, 5, 0]],
        target_state_logits=target_states,
    )
    assert metrics["diagnostics"]["exact_whole_row"]["accuracy"] == 1.0
    assert metrics["decoding"] == {"forced_target_selections": 0}


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_train_and_evaluate_forward_exact_labels_weights_and_private_predictions() -> None:
    import torch
    from torch import nn

    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.calls: list[dict[str, object]] = []

        def forward(self, input_ids, attention_mask, **kwargs):
            del attention_mask
            self.calls.append(kwargs)
            batch_size = input_ids.shape[0]
            loss = self.weight.square()
            relevance = torch.tensor([[5.0, -5.0, -5.0]]).repeat(batch_size, 1)
            target_states = torch.tensor(
                [[[0.0, 5.0, -5.0, -5.0, -5.0, -5.0]] + [[5.0, -5.0, -5.0, -5.0, -5.0, -5.0]] * 3]
            ).repeat(batch_size, 1, 1)
            return {
                "loss": loss,
                "relevance_loss": loss,
                "target_state_loss": loss,
                "relevance_logits": relevance,
                "target_state_logits": target_states,
            }

    config = _config()
    model = RecordingModel()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "relevance_labels": torch.tensor([0]),
        "target_state_labels": torch.tensor([[1, 0, 0, 0]]),
    }
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    trained = train_epoch(model, [batch], optimiser, config=config, device="cpu")
    evaluated = evaluate_epoch(
        model,
        [{**batch, "item_ids": ["one"]}],
        device="cpu",
        reference={
            "one": {
                "relevance": "material",
                "target_stances": [{"target": "china_general", "stance": "negative"}],
            }
        },
        use_bf16=False,
        config=config,
    )

    assert trained["optimizer_steps"] == 1
    expected = conditional_model_loss_kwargs(config)
    assert all({key: call[key] for key in expected} == expected for call in model.calls)
    assert evaluated["metrics"]["diagnostics"]["exact_whole_row"]["accuracy"] == 1.0
    private = evaluated["prediction_payload"]
    assert private["kind"] == "modernbert-conditional-private-development-predictions-v1"
    assert private["row_count"] == 1
    assert set(private["rows"][0]) == {
        "source_sample_id",
        "relevance_logits",
        "target_state_logits",
        "decoded_label",
    }
    assert len(private["rows"][0]["target_state_logits"]) == 4


def test_optional_runtime_is_required_only_for_tensor_execution(monkeypatch) -> None:
    import reddit_china_stance.modernbert_conditional_trainer as trainer

    def unavailable() -> object:
        raise RuntimeError("PyTorch is required")

    monkeypatch.setattr(trainer, "_require_torch", unavailable)
    collator = trainer.ConditionalDynamicPaddingCollator(FakeTokenizer(), return_tensors=None)
    assert collator(
        [
            {
                "input_ids": [1],
                "attention_mask": [1],
                "relevance_labels": 1,
                "target_state_labels": [0, 0, 0, 0],
            }
        ]
    )["target_state_labels"] == [[0, 0, 0, 0]]
    with pytest.raises(RuntimeError, match="PyTorch"):
        trainer.create_adamw(object(), _config())
