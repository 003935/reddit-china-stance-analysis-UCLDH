"""Training primitives for the one-pass conditional-state ModernBERT student.

The conditional architecture retains the frozen tokenizer, input rendering, encoder, batching,
seeding, optimiser grouping, and checkpoint machinery from :mod:`modernbert_trainer`.  This module
owns only the tensor contract that differs from the three-head baseline: relevance labels with
shape ``[B]`` and target-state labels with shape ``[B, 4]``.

PyTorch and Transformers remain optional dependencies.  Importing this module and using the
metadata-only helpers does not require either package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from reddit_china_stance.modernbert_conditional_model import (
    TARGET_STATE_LABELS,
    ModernBertConditionalStateModel,
    decode_conditional_logits,
    encode_conditional_semantic_label,
)
from reddit_china_stance.modernbert_trainer import (
    ModelConfig,
    _require_torch,
    _verify_loaded_revision,
    adamw_parameter_groups,
    build_checkpoint_payload,
    build_length_bucket_batches,
    canonical_sha256,
    load_checkpoint_exact,
    load_pinned_tokenizer,
    render_model_text,
    restore_training_state,
    save_checkpoint_atomic,
    seed_dataloader_worker,
    seed_everything,
    tokenise_record,
)
from reddit_china_stance.semantic_evaluation import score_semantic_labels

ConditionalModelConfig = ModelConfig
_LABEL_KEYS = {"relevance_labels", "target_state_labels"}
_LOSS_KEYS = ("loss", "relevance_loss", "target_state_loss")


@dataclass(frozen=True, slots=True)
class ConditionalOptimisationConfig:
    """Optimiser and exact conditional-task loss settings for one trial."""

    encoder_learning_rate: float
    head_learning_rate_multiplier: float = 5.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.06
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 32
    use_bf16: bool = True
    gradient_checkpointing: bool = False
    lambda_relevance: float = 1.0
    lambda_target_state: float = 1.0
    relevance_class_weights: tuple[float, float, float] | None = None
    target_state_class_weights: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        positive_values = {
            "encoder_learning_rate": self.encoder_learning_rate,
            "head_learning_rate_multiplier": self.head_learning_rate_multiplier,
            "adam_epsilon": self.adam_epsilon,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not math.isfinite(self.warmup_ratio) or not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.effective_batch_size < 1:
            raise ValueError("effective_batch_size must be positive")
        if not (0 < self.adam_beta1 < 1 and 0 < self.adam_beta2 < 1):
            raise ValueError("Adam betas must be in (0, 1)")
        task_weights = (self.lambda_relevance, self.lambda_target_state)
        if any(not math.isfinite(value) or value < 0 for value in task_weights):
            raise ValueError("task loss weights must be finite and non-negative")
        if not any(task_weights):
            raise ValueError("at least one task loss weight must be positive")
        self._validate_weights("relevance_class_weights", self.relevance_class_weights, 3)
        self._validate_weights(
            "target_state_class_weights",
            self.target_state_class_weights,
            len(TARGET_STATE_LABELS),
        )

    @staticmethod
    def _validate_weights(name: str, weights: tuple[float, ...] | None, size: int) -> None:
        if weights is None:
            return
        if len(weights) != size or any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError(f"{name} must contain exactly {size} finite non-negative values")
        if not any(weights):
            raise ValueError(f"{name} must contain at least one supported class")

    def digest(self) -> str:
        """Return the canonical trial-config digest used in checkpoint bindings."""

        return canonical_sha256(asdict(self))


def encode_conditional_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Encode the existing semantic schema for the conditional-state model."""

    return encode_conditional_semantic_label(label)


class ConditionalDynamicPaddingCollator:
    """Dynamically pad inputs while retaining only conditional labels and private item IDs."""

    def __init__(self, tokenizer: Any, *, return_tensors: str | None = "pt") -> None:
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        model_features = [
            {key: value for key, value in feature.items() if key in {"input_ids", "attention_mask"}}
            for feature in features
        ]
        batch = dict(
            self.tokenizer.pad(
                model_features,
                padding=True,
                return_tensors=self.return_tensors,
            )
        )

        item_ids = [feature.get("item_id") for feature in features]
        if any(item_id is not None for item_id in item_ids):
            if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
                raise ValueError("item_id must be present and non-empty for every row or no rows")
            batch["item_ids"] = item_ids

        for key in sorted(_LABEL_KEYS):
            values = [feature.get(key) for feature in features]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                raise ValueError(f"{key} must be present for every row in a batch")
            if self.return_tensors == "pt":
                torch = _require_torch()
                batch[key] = torch.tensor(values, dtype=torch.long)
            else:
                batch[key] = values
        return batch


def create_modernbert_conditional_state_model(
    *,
    model_config: ConditionalModelConfig | None = None,
    optimisation_config: ConditionalOptimisationConfig,
    encoder_loader: Any | None = None,
) -> Any:
    """Compose the conditional-state model around the exact pinned ModernBERT encoder."""

    model_config = model_config or ConditionalModelConfig()
    if encoder_loader is None:
        try:
            from transformers import ModernBertModel
        except ImportError as error:  # pragma: no cover - exercised in the Modal image
            raise RuntimeError("Transformers is required to load ModernBERT") from error
        encoder_loader = ModernBertModel.from_pretrained
    encoder = encoder_loader(
        model_config.model_id,
        revision=model_config.model_revision,
        attn_implementation=model_config.attention_implementation,
    )
    _verify_loaded_revision(encoder, model_config.model_revision, component="model")
    encoder.config.reference_compile = model_config.reference_compile
    if optimisation_config.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    return ModernBertConditionalStateModel(encoder, dropout=model_config.dropout)


def conditional_model_loss_kwargs(config: ConditionalOptimisationConfig) -> dict[str, Any]:
    """Translate the trial config into the conditional model's exact loss interface."""

    return {
        "relevance_loss_weight": config.lambda_relevance,
        "target_state_loss_weight": config.lambda_target_state,
        "relevance_class_weights": config.relevance_class_weights,
        "target_state_class_weights": config.target_state_class_weights,
    }


def create_adamw(model: Any, config: ConditionalOptimisationConfig) -> Any:
    """Build AdamW using the frozen encoder/head grouping rules."""

    torch = _require_torch()
    return torch.optim.AdamW(
        adamw_parameter_groups(model, config),
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def _move_batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key != "item_ids"
    }


def _step_optimizer(
    *, model: Any, optimizer: Any, scheduler: Any | None, gradient_clip_norm: float
) -> None:
    torch = _require_torch()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _rescale_gradients(model: Any, factor: float) -> None:
    for parameter in model.parameters():
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            gradient.mul_(factor)


def train_epoch(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Any,
    *,
    config: ConditionalOptimisationConfig,
    device: str,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Train one conditional epoch with exact gradient-accumulation accounting."""

    torch = _require_torch()
    resolved_device = torch.device(device)
    if config.use_bf16 and resolved_device.type != "cuda":
        raise ValueError("BF16 training is registered only for CUDA devices")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sums = {key: None for key in _LOSS_KEYS}
    batches = optimizer_steps = examples = 0
    for batches, batch in enumerate(dataloader, start=1):
        model_batch = _move_batch_to_device(batch, resolved_device)
        batch_size = int(model_batch["input_ids"].shape[0])
        examples += batch_size
        with torch.autocast(
            device_type=resolved_device.type,
            dtype=torch.bfloat16,
            enabled=config.use_bf16,
        ):
            output = model(**model_batch, **conditional_model_loss_kwargs(config))
            scaled_loss = output["loss"] / config.gradient_accumulation_steps
        scaled_loss.backward()
        for key in loss_sums:
            contribution = output[key].detach().float() * batch_size
            loss_sums[key] = (
                contribution if loss_sums[key] is None else loss_sums[key] + contribution
            )
        if batches % config.gradient_accumulation_steps == 0:
            _step_optimizer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            optimizer_steps += 1
    if batches == 0:
        raise ValueError("training dataloader yielded no batches")
    if batches % config.gradient_accumulation_steps:
        remainder = batches % config.gradient_accumulation_steps
        _rescale_gradients(model, config.gradient_accumulation_steps / remainder)
        _step_optimizer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        optimizer_steps += 1
    return {
        "batches": batches,
        "examples": examples,
        "optimizer_steps": optimizer_steps,
        "mean_losses": {
            key: float(value.cpu()) / examples for key, value in loss_sums.items()
        },
    }


def evaluate_epoch(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    device: str,
    reference: Mapping[str, Mapping[str, Any]] | None = None,
    use_bf16: bool = True,
    config: ConditionalOptimisationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one epoch and optionally score conditional predictions by private item ID."""

    torch = _require_torch()
    resolved_device = torch.device(device)
    if use_bf16 and resolved_device.type != "cuda":
        raise ValueError("BF16 evaluation is registered only for CUDA devices")
    model.eval()
    loss_sums = {key: None for key in _LOSS_KEYS}
    examples = batches = 0
    item_ids: list[str] = []
    relevance_logits: list[Any] = []
    target_state_logits: list[Any] = []
    observed_loss = False
    with torch.no_grad():
        for batch_number, batch in enumerate(dataloader, start=1):
            batches = batch_number
            batch_item_ids = batch.get("item_ids")
            if reference is not None and not batch_item_ids:
                raise ValueError("evaluation batches require item_ids when a reference is supplied")
            model_batch = _move_batch_to_device(batch, resolved_device)
            has_labels = any(key in model_batch for key in _LABEL_KEYS)
            if has_labels and config is None:
                raise ValueError("labelled evaluation requires the trial optimisation config")
            batch_size = int(model_batch["input_ids"].shape[0])
            examples += batch_size
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                loss_kwargs = {} if config is None else conditional_model_loss_kwargs(config)
                output = model(**model_batch, **loss_kwargs)
            if "loss" in output:
                observed_loss = True
                for key in loss_sums:
                    contribution = output[key].detach().float() * batch_size
                    loss_sums[key] = (
                        contribution
                        if loss_sums[key] is None
                        else loss_sums[key] + contribution
                    )
            if batch_item_ids:
                item_ids.extend(batch_item_ids)
                relevance_logits.extend(output["relevance_logits"].detach().float().cpu().tolist())
                target_state_logits.extend(
                    output["target_state_logits"].detach().float().cpu().tolist()
                )
    if batches == 0:
        raise ValueError("evaluation dataloader yielded no batches")
    result: dict[str, Any] = {
        "batches": batches,
        "examples": examples,
        "mean_losses": {
            key: float(value.cpu()) / examples for key, value in loss_sums.items()
        }
        if observed_loss
        else None,
    }
    if reference is not None:
        decoded = decode_conditional_logits(relevance_logits, target_state_logits)
        if len(item_ids) != len(decoded.predictions) or len(set(item_ids)) != len(item_ids):
            raise ValueError("item_ids must be unique and align exactly with decoded rows")
        predictions = dict(zip(item_ids, decoded.predictions, strict=True))
        metrics = score_semantic_labels(reference, predictions)
        metrics["decoding"] = {
            "forced_target_selections": decoded.forced_target_selections,
        }
        result["metrics"] = metrics
        prediction_rows = sorted(
            (
                {
                    "source_sample_id": item_id,
                    "relevance_logits": relevance,
                    "target_state_logits": target_states,
                    "decoded_label": label,
                }
                for item_id, relevance, target_states, label in zip(
                    item_ids,
                    relevance_logits,
                    target_state_logits,
                    decoded.predictions,
                    strict=True,
                )
            ),
            key=lambda row: row["source_sample_id"],
        )
        result["prediction_payload"] = {
            "schema_version": "1.0.0",
            "kind": "modernbert-conditional-private-development-predictions-v1",
            "row_count": len(prediction_rows),
            "rows": prediction_rows,
        }
    return result


def aggregate_conditional_metrics(
    reference: Mapping[str, Mapping[str, Any]],
    *,
    item_ids: Sequence[str],
    relevance_logits: Any,
    target_state_logits: Any,
) -> dict[str, Any]:
    """Decode a complete frame and delegate all aggregate metrics to the registered scorer."""

    decoded = decode_conditional_logits(relevance_logits, target_state_logits)
    if len(item_ids) != len(decoded.predictions) or len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids must be unique and align exactly with decoded rows")
    predictions = dict(zip(item_ids, decoded.predictions, strict=True))
    metrics = score_semantic_labels(reference, predictions)
    metrics["decoding"] = {
        "forced_target_selections": decoded.forced_target_selections,
    }
    return metrics


__all__ = [
    "ConditionalDynamicPaddingCollator",
    "ConditionalModelConfig",
    "ConditionalOptimisationConfig",
    "adamw_parameter_groups",
    "aggregate_conditional_metrics",
    "build_checkpoint_payload",
    "build_length_bucket_batches",
    "conditional_model_loss_kwargs",
    "create_adamw",
    "create_modernbert_conditional_state_model",
    "encode_conditional_label",
    "evaluate_epoch",
    "load_checkpoint_exact",
    "load_pinned_tokenizer",
    "render_model_text",
    "restore_training_state",
    "save_checkpoint_atomic",
    "seed_dataloader_worker",
    "seed_everything",
    "tokenise_record",
    "train_epoch",
]
