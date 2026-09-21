"""Training primitives for the separate-model ModernBERT cascade.

The relevance and target-conditioned components are trained, checkpointed and
evaluated independently.  This module reuses the repository's frozen model,
optimiser, batching, seeding and checkpoint conventions without introducing a
permissive compatibility path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from reddit_china_stance.modernbert_cascade_model import (
    TARGET_STATE_LABELS,
    ModernBertRelevanceModel,
    ModernBertTargetConditionedModel,
    decode_cascade_logits,
    encode_relevance_semantic_label,
    encode_target_conditioned_semantic_label,
    render_target_conditioned_text,
)
from reddit_china_stance.modernbert_trainer import (
    MAX_LENGTH,
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
)
from reddit_china_stance.semantic_evaluation import (
    RELEVANCE_LABELS,
    TARGET_LABELS,
    score_semantic_labels,
)

CascadeComponent = Literal["relevance", "target_conditioned"]
CASCADE_COMPONENTS: tuple[CascadeComponent, ...] = ("relevance", "target_conditioned")
CascadeModelConfig = ModelConfig


def _component(value: str) -> CascadeComponent:
    if value not in CASCADE_COMPONENTS:
        raise ValueError(f"component must be one of {CASCADE_COMPONENTS}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CascadeOptimisationConfig:
    """Single-component optimiser settings frozen into one cascade trial."""

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
    relevance_class_weights: tuple[float, float, float] | None = None
    target_state_class_weights: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        positive = {
            "encoder_learning_rate": self.encoder_learning_rate,
            "head_learning_rate_multiplier": self.head_learning_rate_multiplier,
            "adam_epsilon": self.adam_epsilon,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.warmup_ratio) or not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not (0 < self.adam_beta1 < 1 and 0 < self.adam_beta2 < 1):
            raise ValueError("Adam betas must be in (0, 1)")
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if type(self.effective_batch_size) is not int or self.effective_batch_size < 1:
            raise ValueError("effective_batch_size must be a positive integer")
        self._validate_weights(
            "relevance_class_weights",
            self.relevance_class_weights,
            len(RELEVANCE_LABELS),
        )
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

    def validate_for_component(self, component: CascadeComponent) -> None:
        """Validate a component before selecting its task-specific class weights."""

        _component(component)

    def digest(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class CascadeModels:
    """The two separately instantiated cascade components."""

    relevance: Any
    target_conditioned: Any

    def __post_init__(self) -> None:
        if self.relevance is self.target_conditioned:
            raise ValueError("cascade components must be separate model instances")
        if getattr(self.relevance, "encoder", None) is getattr(
            self.target_conditioned,
            "encoder",
            None,
        ):
            raise ValueError("cascade components must not share an encoder instance")


def encode_relevance_label(label: Mapping[str, Any]) -> dict[str, int]:
    return {"relevance_labels": encode_relevance_semantic_label(label)}


def encode_target_conditioned_label(
    label: Mapping[str, Any],
    *,
    target: str,
) -> dict[str, Any]:
    return {
        "target": target,
        "target_state_labels": encode_target_conditioned_semantic_label(label, target=target),
    }


def _tokenise_text(tokenizer: Any, text: str, *, max_length: int) -> dict[str, Any]:
    if max_length != MAX_LENGTH:
        raise ValueError(f"max_length must remain frozen at {MAX_LENGTH}")
    raw = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    if not isinstance(raw, Mapping) or "input_ids" not in raw or "attention_mask" not in raw:
        raise RuntimeError("tokenizer must return input_ids and attention_mask")
    input_ids = list(raw["input_ids"])
    attention_mask = list(raw["attention_mask"])
    if not input_ids or len(input_ids) != len(attention_mask):
        raise RuntimeError("tokenizer returned empty or mismatched inputs")
    original_tokens = len(input_ids)
    if original_tokens > max_length:
        input_ids = [*input_ids[: max_length - 1], input_ids[-1]]
        attention_mask = [*attention_mask[: max_length - 1], attention_mask[-1]]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_count": original_tokens,
        "truncated_tokens": max(0, original_tokens - max_length),
    }


def tokenise_relevance_record(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Tokenise the relevance component's registered text/context rendering."""

    text = render_model_text(row, separator=tokenizer.sep_token)
    return _tokenise_text(tokenizer, text, max_length=max_length)


def tokenise_target_conditioned_record(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    target: str,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Tokenise an explicit target-conditioned rendering."""

    text = render_target_conditioned_text(row, target=target, separator=tokenizer.sep_token)
    return _tokenise_text(tokenizer, text, max_length=max_length)


class _CascadeDynamicPaddingCollator:
    label_key: str
    preserve_target: bool = False

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
        if self.preserve_target:
            targets = [feature.get("target") for feature in features]
            if any(
                not isinstance(target, str) or target not in TARGET_LABELS for target in targets
            ):
                raise ValueError("every target-conditioned row requires a registered target")
            batch["targets"] = targets
        labels = [feature.get(self.label_key) for feature in features]
        if not all(label is None for label in labels):
            if any(label is None for label in labels):
                raise ValueError(f"{self.label_key} must be present for every row in a batch")
            if self.return_tensors == "pt":
                torch = _require_torch()
                batch[self.label_key] = torch.tensor(labels, dtype=torch.long)
            else:
                batch[self.label_key] = labels
        return batch


class RelevanceDynamicPaddingCollator(_CascadeDynamicPaddingCollator):
    label_key = "relevance_labels"


class TargetConditionedDynamicPaddingCollator(_CascadeDynamicPaddingCollator):
    label_key = "target_state_labels"
    preserve_target = True


def _load_encoder(
    *,
    model_config: CascadeModelConfig,
    optimisation_config: CascadeOptimisationConfig,
    encoder_loader: Any,
) -> Any:
    encoder = encoder_loader(
        model_config.model_id,
        revision=model_config.model_revision,
        attn_implementation=model_config.attention_implementation,
    )
    _verify_loaded_revision(encoder, model_config.model_revision, component="model")
    encoder.config.reference_compile = model_config.reference_compile
    if optimisation_config.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    return encoder


def create_modernbert_cascade_models(
    *,
    optimisation_config: CascadeOptimisationConfig | None = None,
    relevance_optimisation_config: CascadeOptimisationConfig | None = None,
    target_optimisation_config: CascadeOptimisationConfig | None = None,
    model_config: CascadeModelConfig | None = None,
    encoder_loader: Any | None = None,
) -> CascadeModels:
    """Load two independent pinned ModernBERT-large encoders and attach single-task heads."""

    if optimisation_config is not None:
        if relevance_optimisation_config is not None or target_optimisation_config is not None:
            raise ValueError(
                "supply either optimisation_config or both component-specific configs, not both"
            )
        relevance_optimisation_config = optimisation_config
        target_optimisation_config = optimisation_config
    if relevance_optimisation_config is None or target_optimisation_config is None:
        raise ValueError(
            "optimisation_config or both component-specific optimisation configs are required"
        )
    relevance = create_modernbert_cascade_component(
        component="relevance",
        optimisation_config=relevance_optimisation_config,
        model_config=model_config,
        encoder_loader=encoder_loader,
    )
    target_conditioned = create_modernbert_cascade_component(
        component="target_conditioned",
        optimisation_config=target_optimisation_config,
        model_config=model_config,
        encoder_loader=encoder_loader,
    )
    return CascadeModels(
        relevance=relevance,
        target_conditioned=target_conditioned,
    )


def create_modernbert_cascade_component(
    *,
    component: CascadeComponent,
    optimisation_config: CascadeOptimisationConfig,
    model_config: CascadeModelConfig | None = None,
    encoder_loader: Any | None = None,
) -> Any:
    """Load exactly one pinned encoder for one independently trained component."""

    component = _component(component)
    optimisation_config.validate_for_component(component)
    model_config = model_config or CascadeModelConfig()
    if encoder_loader is None:
        try:
            from transformers import ModernBertModel
        except ImportError as error:  # pragma: no cover - exercised in Modal
            raise RuntimeError("Transformers is required to load ModernBERT") from error
        encoder_loader = ModernBertModel.from_pretrained
    encoder = _load_encoder(
        model_config=model_config,
        optimisation_config=optimisation_config,
        encoder_loader=encoder_loader,
    )
    if component == "relevance":
        return ModernBertRelevanceModel(encoder, dropout=model_config.dropout)
    return ModernBertTargetConditionedModel(encoder, dropout=model_config.dropout)


def component_loss_kwargs(
    config: CascadeOptimisationConfig,
    *,
    component: CascadeComponent,
) -> dict[str, Any]:
    component = _component(component)
    config.validate_for_component(component)
    if component == "relevance":
        return {"relevance_class_weights": config.relevance_class_weights}
    return {"target_state_class_weights": config.target_state_class_weights}


def create_adamw(model: Any, config: CascadeOptimisationConfig) -> Any:
    """Build AdamW with the existing encoder/head and decay/no-decay grouping."""

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
        if key not in {"item_ids", "targets"}
    }


def _step_optimizer(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    gradient_clip_norm: float,
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


def train_component_epoch(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Any,
    *,
    component: CascadeComponent,
    config: CascadeOptimisationConfig,
    device: str,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Train exactly one cascade component for one epoch."""

    component = _component(component)
    config.validate_for_component(component)
    torch = _require_torch()
    resolved_device = torch.device(device)
    if config.use_bf16 and resolved_device.type != "cuda":
        raise ValueError("BF16 training is registered only for CUDA devices")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = None
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
            output = model(
                **model_batch,
                **component_loss_kwargs(config, component=component),
            )
            if "loss" not in output:
                raise ValueError("training model output must contain loss")
            scaled_loss = output["loss"] / config.gradient_accumulation_steps
        scaled_loss.backward()
        contribution = output["loss"].detach().float() * batch_size
        loss_sum = contribution if loss_sum is None else loss_sum + contribution
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
        "component": component,
        "batches": batches,
        "examples": examples,
        "optimizer_steps": optimizer_steps,
        "mean_loss": float(loss_sum.cpu()) / examples,
    }


def evaluate_component_epoch(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    component: CascadeComponent,
    device: str,
    use_bf16: bool = True,
    config: CascadeOptimisationConfig | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate one component and optionally return its private row-level logits."""

    component = _component(component)
    if config is not None:
        config.validate_for_component(component)
    torch = _require_torch()
    resolved_device = torch.device(device)
    if use_bf16 and resolved_device.type != "cuda":
        raise ValueError("BF16 evaluation is registered only for CUDA devices")
    model.eval()
    examples = batches = 0
    loss_sum = None
    prediction_rows: list[dict[str, Any]] = []
    observed_loss = False
    logit_key = "relevance_logits" if component == "relevance" else "target_state_logits"
    with torch.no_grad():
        for batch_number, batch in enumerate(dataloader, start=1):
            batches = batch_number
            item_ids = batch.get("item_ids")
            targets = batch.get("targets")
            if collect_predictions:
                if not item_ids or len(item_ids) != len(set(item_ids)):
                    if component == "relevance":
                        raise ValueError("prediction collection requires unique item_ids per batch")
                    if not item_ids:
                        raise ValueError("prediction collection requires item_ids")
                if component == "target_conditioned" and (
                    not targets or len(targets) != len(item_ids)
                ):
                    raise ValueError("target prediction collection requires aligned targets")
            model_batch = _move_batch_to_device(batch, resolved_device)
            has_labels = any(key.endswith("_labels") for key in model_batch)
            if has_labels and config is None:
                raise ValueError("labelled evaluation requires the component optimisation config")
            batch_size = int(model_batch["input_ids"].shape[0])
            examples += batch_size
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                kwargs = (
                    {} if config is None else component_loss_kwargs(config, component=component)
                )
                output = model(**model_batch, **kwargs)
            if logit_key not in output:
                raise ValueError(f"{component} model output must contain {logit_key}")
            if "loss" in output:
                observed_loss = True
                contribution = output["loss"].detach().float() * batch_size
                loss_sum = contribution if loss_sum is None else loss_sum + contribution
            if collect_predictions:
                logits = output[logit_key].detach().float().cpu().tolist()
                for index, (item_id, row_logits) in enumerate(zip(item_ids, logits, strict=True)):
                    row = {"source_sample_id": item_id, logit_key: row_logits}
                    if component == "target_conditioned":
                        row["target"] = targets[index]
                    prediction_rows.append(row)
    if batches == 0:
        raise ValueError("evaluation dataloader yielded no batches")
    result: dict[str, Any] = {
        "component": component,
        "batches": batches,
        "examples": examples,
        "mean_loss": float(loss_sum.cpu()) / examples if observed_loss else None,
    }
    if collect_predictions:
        sort_key = (
            (lambda row: (row["source_sample_id"], row["target"]))
            if component == "target_conditioned"
            else (lambda row: row["source_sample_id"])
        )
        prediction_rows.sort(key=sort_key)
        unique_keys = {
            (row["source_sample_id"], row.get("target")) for row in prediction_rows
        }
        if len(unique_keys) != len(prediction_rows):
            raise ValueError("prediction rows must have unique item/target keys")
        result["prediction_payload"] = {
            "schema_version": "1.0.0",
            "kind": f"modernbert-cascade-{component}-private-development-predictions-v1",
            "component": component,
            "row_count": len(prediction_rows),
            "rows": prediction_rows,
        }
    return result


def aggregate_cascade_metrics(
    reference: Mapping[str, Mapping[str, Any]],
    *,
    item_ids: Sequence[str],
    relevance_logits: Sequence[Sequence[float]],
    target_state_logits: Sequence[Sequence[Sequence[float]]],
) -> dict[str, Any]:
    """Merge both components and call the registered aggregate semantic scorer."""

    decoded = decode_cascade_logits(relevance_logits, target_state_logits)
    if len(item_ids) != len(decoded.predictions) or len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids must be unique and align exactly with decoded rows")
    predictions = dict(zip(item_ids, decoded.predictions, strict=True))
    metrics = score_semantic_labels(reference, predictions)
    metrics["decoding"] = {"forced_target_selections": decoded.forced_target_selections}
    return metrics


__all__ = [
    "CASCADE_COMPONENTS",
    "CascadeComponent",
    "CascadeModelConfig",
    "CascadeModels",
    "CascadeOptimisationConfig",
    "RelevanceDynamicPaddingCollator",
    "TargetConditionedDynamicPaddingCollator",
    "adamw_parameter_groups",
    "aggregate_cascade_metrics",
    "build_checkpoint_payload",
    "build_length_bucket_batches",
    "component_loss_kwargs",
    "create_adamw",
    "create_modernbert_cascade_component",
    "create_modernbert_cascade_models",
    "encode_relevance_label",
    "encode_target_conditioned_label",
    "evaluate_component_epoch",
    "load_checkpoint_exact",
    "load_pinned_tokenizer",
    "restore_training_state",
    "save_checkpoint_atomic",
    "seed_dataloader_worker",
    "seed_everything",
    "tokenise_relevance_record",
    "tokenise_target_conditioned_record",
    "train_component_epoch",
]
