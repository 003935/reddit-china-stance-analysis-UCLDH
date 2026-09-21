from __future__ import annotations

import importlib.util
from dataclasses import asdict
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from reddit_china_stance.modernbert_trainer import (
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    DynamicPaddingCollator,
    ModelConfig,
    OptimisationConfig,
    adamw_parameter_groups,
    aggregate_semantic_metrics,
    build_length_bucket_batches,
    canonical_sha256,
    create_modernbert_three_head_model,
    decode_predictions,
    encode_semantic_label,
    evaluate_epoch,
    load_pinned_tokenizer,
    model_loss_kwargs,
    render_model_text,
    restore_rng_state,
    tokenise_record,
    train_epoch,
)


class FakeTokenizer:
    sep_token = "<SEP>"
    init_kwargs: ClassVar[dict[str, str]] = {"_commit_hash": MODEL_REVISION}

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        assert kwargs == {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "return_attention_mask": True,
            "return_token_type_ids": False,
        }
        tokens = [101, *range(1000, 1000 + len(text.split())), 102]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def pad(
        self,
        features: list[dict[str, list[int]]],
        *,
        padding: bool,
        return_tensors: str | None,
    ) -> dict[str, object]:
        assert padding is True
        width = max(len(feature["input_ids"]) for feature in features)
        padded = {
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

            return {key: torch.tensor(value) for key, value in padded.items()}
        assert return_tensors is None
        return padded


def _optimisation_config() -> OptimisationConfig:
    return OptimisationConfig(encoder_learning_rate=3e-5)


def test_model_and_optimisation_configs_are_frozen_and_content_addressed() -> None:
    model = ModelConfig()
    optimiser = _optimisation_config()

    assert model.model_id == MODEL_ID
    assert model.model_revision == MODEL_REVISION
    assert model.max_length == MAX_LENGTH
    assert model.attention_implementation == "sdpa"
    assert model.reference_compile is False
    assert model.digest() == canonical_sha256(asdict(model))
    assert optimiser.digest() == canonical_sha256(asdict(optimiser))

    with pytest.raises(ValueError, match="model_revision"):
        ModelConfig(model_revision="main")
    with pytest.raises(ValueError, match="sdpa"):
        ModelConfig(attention_implementation="flash_attention_2")
    with pytest.raises(ValueError, match="compilation"):
        ModelConfig(reference_compile=True)
    with pytest.raises(ValueError, match="exactly 4"):
        OptimisationConfig(encoder_learning_rate=1e-5, target_positive_weights=(1.0,))


def test_render_and_tokenise_preserve_target_parent_submission_order_and_final_token() -> None:
    tokenizer = FakeTokenizer()
    row = {
        "target_text": "target words",
        "parent_context": "parent words",
        "submission_context": "submission words",
    }
    assert render_model_text(row, separator=tokenizer.sep_token) == (
        "target words <SEP> parent words <SEP> submission words"
    )

    long_row = {"target_text": " ".join(f"word-{index}" for index in range(MAX_LENGTH + 10))}
    encoded = tokenise_record(tokenizer, long_row)
    assert len(encoded["input_ids"]) == MAX_LENGTH
    assert encoded["input_ids"][-1] == 102
    assert encoded["token_count"] == MAX_LENGTH + 14
    assert encoded["truncated_tokens"] == 14

    with pytest.raises(ValueError, match="target_text"):
        render_model_text({"target_text": ""}, separator="<SEP>")


def test_pinned_tokenizer_loader_receives_exact_revision_and_rejects_drift() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def loader(model_id: str, **kwargs: object) -> FakeTokenizer:
        calls.append((model_id, kwargs))
        return FakeTokenizer()

    assert load_pinned_tokenizer(tokenizer_loader=loader).sep_token == "<SEP>"
    assert calls == [(MODEL_ID, {"revision": MODEL_REVISION, "use_fast": True})]

    class DriftedTokenizer(FakeTokenizer):
        init_kwargs: ClassVar[dict[str, str]] = {"_commit_hash": "0" * 40}

    with pytest.raises(RuntimeError, match="revision"):
        load_pinned_tokenizer(tokenizer_loader=lambda *_args, **_kwargs: DriftedTokenizer())


def test_dynamic_padding_collator_keeps_labels_and_ids_without_importing_torch() -> None:
    collator = DynamicPaddingCollator(FakeTokenizer(), return_tensors=None)
    batch = collator(
        [
            {
                "item_id": "one",
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "relevance_labels": 0,
                "target_labels": [1, 0, 0, 0],
            },
            {
                "item_id": "two",
                "input_ids": [1],
                "attention_mask": [1],
                "relevance_labels": 1,
                "target_labels": [0, 0, 0, 0],
            },
        ]
    )
    assert batch["input_ids"] == [[1, 2], [1, 0]]
    assert batch["item_ids"] == ["one", "two"]
    assert batch["relevance_labels"] == [0, 1]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_dynamic_padding_uses_authoritative_integer_label_contract() -> None:
    import torch

    collator = DynamicPaddingCollator(FakeTokenizer())
    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                **encode_semantic_label(
                    {
                        "relevance": "material",
                        "target_stances": [
                            {"target": "government_ccp", "stance": "negative"}
                        ],
                    }
                ),
            }
        ]
    )
    assert batch["relevance_labels"].dtype == torch.int64
    assert batch["target_labels"].dtype == torch.int64
    assert batch["stance_labels"].dtype == torch.int64
    assert batch["stance_labels"].tolist() == [[-100, 0, -100, -100]]


def test_semantic_label_encoding_builds_authoritative_three_head_labels() -> None:
    assert encode_semantic_label(
        {
            "relevance": "material",
            "target_stances": [
                {"target": "government_ccp", "stance": "negative"},
                {"target": "people_culture", "stance": "positive"},
            ],
        }
    ) == {
        "relevance_labels": 0,
        "target_labels": [0, 1, 1, 0],
        "stance_labels": [-100, 0, 3, -100],
    }
    with pytest.raises(ValueError, match="non-material"):
        encode_semantic_label(
            {
                "relevance": "not_material",
                "target_stances": [{"target": "china_general", "stance": "negative"}],
            }
        )


def test_model_factory_composes_authoritative_model_and_sdpa_binding(monkeypatch) -> None:
    constructed: list[tuple[object, float]] = []

    class FakeAuthoritativeModel:
        def __init__(self, encoder: object, *, dropout: float) -> None:
            constructed.append((encoder, dropout))

    fake_module = ModuleType("reddit_china_stance.modernbert_model")
    fake_module.ModernBertThreeHeadModel = FakeAuthoritativeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, fake_module.__name__, fake_module)
    loader_calls: list[tuple[str, dict[str, object]]] = []
    encoder = SimpleNamespace(config=SimpleNamespace(_commit_hash=MODEL_REVISION))

    def loader(model_id: str, **kwargs: object) -> object:
        loader_calls.append((model_id, kwargs))
        return encoder

    config = OptimisationConfig(
        encoder_learning_rate=1e-5,
        lambda_relevance=0.5,
        lambda_targets=1.0,
        lambda_stance=1.5,
        relevance_class_weights=(1.0, 2.0, 3.0),
        target_positive_weights=(1.0, 2.0, 3.0, 4.0),
        stance_class_weights=(1.0, 1.5, 2.0, 2.5, 3.0),
    )
    result = create_modernbert_three_head_model(
        optimisation_config=config,
        encoder_loader=loader,
    )

    assert isinstance(result, FakeAuthoritativeModel)
    assert constructed == [(encoder, 0.1)]
    assert loader_calls == [
        (
            MODEL_ID,
            {"revision": MODEL_REVISION, "attn_implementation": "sdpa"},
        )
    ]
    assert encoder.config.reference_compile is False
    assert model_loss_kwargs(config) == {
        "task_loss_weights": (0.5, 1.0, 1.5),
        "relevance_class_weights": (1.0, 2.0, 3.0),
        "target_pos_weights": (1.0, 2.0, 3.0, 4.0),
        "stance_class_weights": (1.0, 1.5, 2.0, 2.5, 3.0),
    }


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_train_and_eval_forward_configured_loss_bindings() -> None:
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
            targets = torch.tensor([[5.0, -5.0, -5.0, -5.0]]).repeat(batch_size, 1)
            stance = torch.tensor(
                [[[5.0, -5.0, -5.0, -5.0, -5.0]] * 4]
            ).repeat(batch_size, 1, 1)
            return {
                "loss": loss,
                "relevance_loss": loss,
                "target_loss": loss,
                "stance_loss": loss,
                "relevance_logits": relevance,
                "target_logits": targets,
                "stance_logits": stance,
            }

    config = OptimisationConfig(
        encoder_learning_rate=1e-5,
        use_bf16=False,
        lambda_relevance=0.5,
        lambda_targets=1.0,
        lambda_stance=1.5,
        relevance_class_weights=(1.0, 2.0, 3.0),
        target_positive_weights=(1.0, 2.0, 3.0, 4.0),
        stance_class_weights=(1.0, 1.5, 2.0, 2.5, 3.0),
    )
    model = RecordingModel()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "relevance_labels": torch.tensor([0]),
        "target_labels": torch.tensor([[1, 0, 0, 0]]),
        "stance_labels": torch.tensor([[0, -100, -100, -100]]),
    }
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    train_epoch(model, [batch], optimiser, config=config, device="cpu")
    evaluation_batch = {**batch, "item_ids": ["one"]}
    evaluated = evaluate_epoch(
        model,
        [evaluation_batch],
        device="cpu",
        target_threshold=0.5,
        reference={
            "one": {
                "relevance": "material",
                "target_stances": [{"target": "china_general", "stance": "negative"}],
            }
        },
        use_bf16=False,
        config=config,
    )

    expected_loss_kwargs = model_loss_kwargs(config)
    for call in model.calls:
        assert {key: call[key] for key in expected_loss_kwargs} == expected_loss_kwargs
    assert evaluated["metrics"]["diagnostics"]["exact_whole_row"]["accuracy"] == 1.0
    private = evaluated["prediction_payload"]
    assert private["kind"] == "modernbert-private-development-predictions-v1"
    assert private["row_count"] == 1
    assert private["rows"][0]["source_sample_id"] == "one"
    assert len(private["rows"][0]["target_logits"]) == 4


def test_length_bucketing_is_deterministic_complete_and_epoch_sensitive() -> None:
    lengths = [10 + index * 3 for index in range(23)]
    first = build_length_bucket_batches(lengths, batch_size=4, seed=17, epoch=0)
    repeated = build_length_bucket_batches(lengths, batch_size=4, seed=17, epoch=0)
    next_epoch = build_length_bucket_batches(lengths, batch_size=4, seed=17, epoch=1)

    assert first == repeated
    assert first != next_epoch
    assert sorted(index for batch in first for index in batch) == list(range(len(lengths)))
    assert all(1 <= len(batch) <= 4 for batch in first)


def test_restore_rng_state_moves_map_location_cuda_tensors_back_to_cpu(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class MappedTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def cpu(self) -> str:
            observed[f"cpu:{self.name}"] = True
            return f"cpu-{self.name}"

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_rng_state_all(values: list[object]) -> None:
            observed["torch_cuda"] = values

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def set_rng_state(value: object) -> None:
            observed["torch_cpu"] = value

    import reddit_china_stance.modernbert_trainer as trainer_module

    monkeypatch.setattr(trainer_module, "_require_torch", lambda: FakeTorch())
    restore_rng_state(
        {
            "python": __import__("random").getstate(),
            "torch_cpu": MappedTensor("torch-cpu"),
            "torch_cuda": [MappedTensor("cuda-0")],
        }
    )

    assert observed == {
        "cpu:torch-cpu": True,
        "torch_cpu": "cpu-torch-cpu",
        "cpu:cuda-0": True,
        "torch_cuda": ["cpu-cuda-0"],
    }


def test_decoder_applies_relevance_gate_threshold_fallback_and_stance_mapping() -> None:
    relevance = [
        [9.0, 0.0, 0.0],
        [0.0, 9.0, 0.0],
        [9.0, 0.0, 0.0],
    ]
    targets = [
        [1.0, -2.0, 2.0, -3.0],
        [9.0, 9.0, 9.0, 9.0],
        [-3.0, -2.0, -1.0, -4.0],
    ]
    stances = [
        [[0, 0, 0, 4, 0], [0, 0, 4, 0, 0], [4, 0, 0, 0, 0], [0, 4, 0, 0, 0]],
        [[0, 0, 0, 4, 0]] * 4,
        [[0, 0, 0, 4, 0], [0, 0, 0, 4, 0], [0, 4, 0, 0, 0], [0, 0, 0, 4, 0]],
    ]

    decoded = decode_predictions(relevance, targets, stances, target_threshold=0.7)

    assert decoded.forced_target_selections == 1
    assert decoded.labels[0] == {
        "relevance": "material",
        "target_stances": [
            {"target": "china_general", "stance": "positive"},
            {"target": "people_culture", "stance": "negative"},
        ],
    }
    assert decoded.labels[1] == {"relevance": "not_material", "target_stances": []}
    assert decoded.labels[2] == {
        "relevance": "material",
        "target_stances": [{"target": "people_culture", "stance": "mixed"}],
    }


def test_aggregate_metrics_delegate_to_registered_semantic_scorer() -> None:
    reference = {
        "one": {
            "relevance": "material",
            "target_stances": [{"target": "government_ccp", "stance": "negative"}],
        },
        "two": {"relevance": "not_material", "target_stances": []},
    }
    metrics = aggregate_semantic_metrics(
        reference,
        item_ids=["one", "two"],
        relevance_logits=[[5, 0, 0], [0, 5, 0]],
        target_logits=[[-5, 5, -5, -5], [5, 5, 5, 5]],
        stance_logits=[
            [[0, 0, 0, 0, 0], [5, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0]] * 4,
        ],
        target_threshold=0.5,
    )
    assert metrics["diagnostics"]["exact_whole_row"]["accuracy"] == 1.0
    assert metrics["decoding"] == {"target_threshold": 0.5, "forced_target_selections": 0}


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class FakeModel:
    def __init__(self) -> None:
        self.parameters = {
            "encoder.layer.weight": FakeParameter(),
            "encoder.layer.bias": FakeParameter(),
            "relevance_head.weight": FakeParameter(),
            "relevance_head.bias": FakeParameter(),
        }

    def named_parameters(self):
        return self.parameters.items()


def test_adamw_groups_separate_encoder_heads_and_decay() -> None:
    groups = adamw_parameter_groups(FakeModel(), _optimisation_config())
    summary = {
        group["group_name"]: (group["lr"], group["weight_decay"], len(group["params"]))
        for group in groups
    }
    assert summary == {
        "encoder_decay": (3e-5, 0.01, 1),
        "encoder_no_decay": (3e-5, 0.0, 1),
        "heads_decay": (pytest.approx(0.00015), 0.01, 1),
        "heads_no_decay": (pytest.approx(0.00015), 0.0, 1),
    }
