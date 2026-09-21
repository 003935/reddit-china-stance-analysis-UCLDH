from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from reddit_china_stance.modernbert_cascade_trainer import (
    CascadeOptimisationConfig,
    RelevanceDynamicPaddingCollator,
    TargetConditionedDynamicPaddingCollator,
    aggregate_cascade_metrics,
    component_loss_kwargs,
    create_modernbert_cascade_component,
    create_modernbert_cascade_models,
    encode_relevance_label,
    encode_target_conditioned_label,
    evaluate_component_epoch,
    tokenise_relevance_record,
    tokenise_target_conditioned_record,
    train_component_epoch,
)
from reddit_china_stance.modernbert_trainer import MAX_LENGTH, MODEL_ID, MODEL_REVISION


class FakeTokenizer:
    sep_token = "<SEP>"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text, **kwargs):
        assert kwargs == {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "return_attention_mask": True,
            "return_token_type_ids": False,
        }
        self.texts.append(text)
        return {
            "input_ids": list(range(len(text.split()) + 2)),
            "attention_mask": [1] * (len(text.split()) + 2),
        }

    def pad(self, features, *, padding, return_tensors):
        assert padding is True
        width = max(len(feature["input_ids"]) for feature in features)
        values = {
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

            return {key: torch.tensor(value) for key, value in values.items()}
        return values


def _config(**overrides: object) -> CascadeOptimisationConfig:
    values = {
        "encoder_learning_rate": 3e-5,
        "use_bf16": False,
        "relevance_class_weights": (1.0, 2.0, 3.0),
        "target_state_class_weights": (0.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    }
    values.update(overrides)
    return CascadeOptimisationConfig(**values)


def test_config_loss_kwargs_and_component_validation() -> None:
    config = _config()

    assert component_loss_kwargs(config, component="relevance") == {
        "relevance_class_weights": (1.0, 2.0, 3.0)
    }
    assert component_loss_kwargs(config, component="target_conditioned") == {
        "target_state_class_weights": (0.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    }
    assert len(config.digest()) == 64
    with pytest.raises(ValueError, match="component must be"):
        component_loss_kwargs(config, component="stance")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        _config(gradient_accumulation_steps=True)


def test_encoding_and_collators_keep_only_each_component_contract() -> None:
    label = {
        "relevance": "material",
        "target_stances": [{"target": "government_ccp", "stance": "negative"}],
    }
    assert encode_relevance_label(label) == {"relevance_labels": 0}
    assert encode_target_conditioned_label(label, target="government_ccp") == {
        "target": "government_ccp",
        "target_state_labels": 1,
    }

    relevance = RelevanceDynamicPaddingCollator(FakeTokenizer(), return_tensors=None)(
        [
            {
                "item_id": "b",
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "relevance_labels": 0,
                "target_state_labels": 1,
            },
            {
                "item_id": "a",
                "input_ids": [3],
                "attention_mask": [1],
                "relevance_labels": 1,
            },
        ]
    )
    target = TargetConditionedDynamicPaddingCollator(FakeTokenizer(), return_tensors=None)(
        [
            {
                "item_id": "a",
                "target": "china_general",
                "input_ids": [1],
                "attention_mask": [1],
                "target_state_labels": 0,
                "relevance_labels": 0,
            }
        ]
    )

    assert relevance == {
        "input_ids": [[1, 2], [3, 0]],
        "attention_mask": [[1, 1], [1, 0]],
        "item_ids": ["b", "a"],
        "relevance_labels": [0, 1],
    }
    assert target == {
        "input_ids": [[1]],
        "attention_mask": [[1]],
        "item_ids": ["a"],
        "targets": ["china_general"],
        "target_state_labels": [0],
    }


def test_tokenisers_use_distinct_registered_renderings_and_strict_outputs() -> None:
    tokenizer = FakeTokenizer()
    row = {
        "target_text": "main",
        "parent_context": "parent",
        "submission_context": "submission",
    }

    relevance = tokenise_relevance_record(tokenizer, row)
    tokenise_target_conditioned_record(tokenizer, row, target="other")

    assert relevance["token_count"] == len(tokenizer.texts[0].split()) + 2
    assert tokenizer.texts[0] == "main <SEP> parent <SEP> submission"
    assert tokenizer.texts[1] == "[TARGET=other] <SEP> main <SEP> parent <SEP> submission"
    with pytest.raises(ValueError, match="remain frozen"):
        tokenise_relevance_record(tokenizer, row, max_length=MAX_LENGTH - 1)

    class BrokenTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            del text, kwargs
            return {"input_ids": [1]}

    with pytest.raises(RuntimeError, match="input_ids and attention_mask"):
        tokenise_target_conditioned_record(BrokenTokenizer(), row, target="other")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_factories_load_independent_pinned_encoders_and_single_component() -> None:
    import torch
    from torch import nn

    calls: list[tuple[str, dict[str, object]]] = []

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(hidden_size=2, _commit_hash=MODEL_REVISION)

        def forward(self, *, input_ids, attention_mask):
            del input_ids
            return SimpleNamespace(
                last_hidden_state=attention_mask.unsqueeze(-1).repeat(1, 1, 2).float()
            )

    def loader(model_id: str, **kwargs: object) -> Encoder:
        calls.append((model_id, kwargs))
        return Encoder()

    config = _config()
    models = create_modernbert_cascade_models(
        optimisation_config=config,
        encoder_loader=loader,
    )
    component = create_modernbert_cascade_component(
        component="relevance",
        optimisation_config=config,
        encoder_loader=loader,
    )

    assert models.relevance.encoder is not models.target_conditioned.encoder
    assert component.encoder is not models.relevance.encoder
    assert calls == [
        (MODEL_ID, {"revision": MODEL_REVISION, "attn_implementation": "sdpa"}),
        (MODEL_ID, {"revision": MODEL_REVISION, "attn_implementation": "sdpa"}),
        (MODEL_ID, {"revision": MODEL_REVISION, "attn_implementation": "sdpa"}),
    ]
    assert all(call_model == MODEL_ID for call_model, _ in calls)

    shared = Encoder()
    with pytest.raises((RuntimeError, ValueError), match=r"share|shared"):
        create_modernbert_cascade_models(
            optimisation_config=config,
            encoder_loader=lambda *_args, **_kwargs: shared,
        )


def test_aggregate_metrics_uses_end_to_end_cascade_merger() -> None:
    reference = {
        "one": {
            "relevance": "material",
            "target_stances": [{"target": "government_ccp", "stance": "negative"}],
        },
        "two": {"relevance": "not_material", "target_stances": []},
    }
    absent = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    first = [absent.copy() for _ in range(4)]
    first[1] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]

    metrics = aggregate_cascade_metrics(
        reference,
        item_ids=["one", "two"],
        relevance_logits=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        target_state_logits=[first, [absent.copy() for _ in range(4)]],
    )

    assert metrics["diagnostics"]["exact_whole_row"]["accuracy"] == 1.0
    assert metrics["decoding"] == {"forced_target_selections": 0}


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_train_and_evaluate_forward_exact_component_contracts() -> None:
    import torch
    from torch import nn

    class RecordingModel(nn.Module):
        def __init__(self, component: str) -> None:
            super().__init__()
            self.component = component
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.calls: list[dict[str, object]] = []

        def forward(self, input_ids, attention_mask, **kwargs):
            del attention_mask
            self.calls.append(kwargs)
            batch = input_ids.shape[0]
            result = {"loss": self.weight.square()}
            if self.component == "relevance":
                result["relevance_logits"] = torch.tensor([[5.0, 0.0, 0.0]]).repeat(batch, 1)
            else:
                result["target_state_logits"] = torch.tensor(
                    [[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]]
                ).repeat(batch, 1)
            return result

    relevance_config = _config(target_state_class_weights=None)
    relevance_model = RecordingModel("relevance")
    relevance_batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "relevance_labels": torch.tensor([0]),
    }
    optimiser = torch.optim.SGD(relevance_model.parameters(), lr=0.01)
    trained = train_component_epoch(
        relevance_model,
        [relevance_batch],
        optimiser,
        component="relevance",
        config=relevance_config,
        device="cpu",
    )
    evaluated = evaluate_component_epoch(
        relevance_model,
        [{**relevance_batch, "item_ids": ["one"]}],
        component="relevance",
        device="cpu",
        use_bf16=False,
        config=relevance_config,
        collect_predictions=True,
    )

    assert trained["optimizer_steps"] == 1
    assert trained["component"] == "relevance"
    assert all(
        call["relevance_class_weights"] == (1.0, 2.0, 3.0)
        for call in relevance_model.calls
    )
    assert evaluated["prediction_payload"] == {
        "schema_version": "1.0.0",
        "kind": "modernbert-cascade-relevance-private-development-predictions-v1",
        "component": "relevance",
        "row_count": 1,
        "rows": [{"source_sample_id": "one", "relevance_logits": [5.0, 0.0, 0.0]}],
    }

    target_config = _config(relevance_class_weights=None)
    target_model = RecordingModel("target")
    target_batch = {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
        "target_state_labels": torch.tensor([1]),
        "item_ids": ["one"],
        "targets": ["government_ccp"],
    }
    target_evaluated = evaluate_component_epoch(
        target_model,
        [target_batch],
        component="target_conditioned",
        device="cpu",
        use_bf16=False,
        config=target_config,
        collect_predictions=True,
    )
    assert target_evaluated["prediction_payload"]["rows"] == [
        {
            "source_sample_id": "one",
            "target_state_logits": [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            "target": "government_ccp",
        }
    ]
