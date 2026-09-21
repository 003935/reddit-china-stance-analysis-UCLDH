"""ModernBERT three-head training primitives.

The module intentionally imports neither PyTorch nor Transformers at import time.  Local contract
tests and orchestration code can therefore inspect, construct, and validate training plans without
installing the GPU stack.  Functions that execute the model fail explicitly when their optional
runtime dependencies are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reddit_china_stance.semantic_evaluation import (
    RELEVANCE_LABELS,
    TARGET_LABELS,
    score_semantic_labels,
)

MODEL_ID = "answerdotai/ModernBERT-large"
MODEL_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
TOKENIZER_REVISION = MODEL_REVISION
MAX_LENGTH = 768

# This order is frozen by the fine-tuning experiment, and deliberately differs from display order.
STANCE_LABELS = ("negative", "mixed", "no_directed_stance", "positive", "unclear")

CHECKPOINT_SCHEMA_VERSION = "modernbert-checkpoint-v1"
ABSENT_STANCE_INDEX = -100
_DIGEST_FIELDS = (
    "dataset_sha256",
    "split_manifest_sha256",
    "trial_manifest_sha256",
    "code_sha256",
)
_LABEL_KEYS = {
    "relevance_labels",
    "target_labels",
    "stance_labels",
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using the repository's canonical representation."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hex_digest(value: str, *, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Frozen architecture and input contract."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = TOKENIZER_REVISION
    max_length: int = MAX_LENGTH
    dropout: float = 0.1
    pooling: str = "masked_mean"
    attention_implementation: str = "sdpa"
    reference_compile: bool = False

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError(f"model_id must remain pinned to {MODEL_ID}")
        if self.model_revision != MODEL_REVISION:
            raise ValueError(f"model_revision must remain pinned to {MODEL_REVISION}")
        if self.tokenizer_revision != TOKENIZER_REVISION:
            raise ValueError(f"tokenizer_revision must remain pinned to {TOKENIZER_REVISION}")
        if self.max_length != MAX_LENGTH:
            raise ValueError(f"max_length must remain frozen at {MAX_LENGTH}")
        if self.pooling != "masked_mean":
            raise ValueError("only the registered masked_mean pooling is supported")
        if self.attention_implementation != "sdpa":
            raise ValueError("the executable attention implementation is frozen to sdpa")
        if self.reference_compile is not False:
            raise ValueError("per-container ModernBERT reference compilation must remain disabled")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def digest(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class OptimisationConfig:
    """Optimiser and execution settings frozen into every trial manifest."""

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
    lambda_targets: float = 1.0
    lambda_stance: float = 1.0
    relevance_class_weights: tuple[float, float, float] | None = None
    target_positive_weights: tuple[float, float, float, float] | None = None
    stance_class_weights: tuple[float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        positive_values = {
            "encoder_learning_rate": self.encoder_learning_rate,
            "head_learning_rate_multiplier": self.head_learning_rate_multiplier,
            "adam_epsilon": self.adam_epsilon,
            "gradient_clip_norm": self.gradient_clip_norm,
            "lambda_relevance": self.lambda_relevance,
            "lambda_targets": self.lambda_targets,
            "lambda_stance": self.lambda_stance,
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
        self._validate_weights("relevance_class_weights", self.relevance_class_weights, 3)
        self._validate_weights("target_positive_weights", self.target_positive_weights, 4)
        self._validate_weights("stance_class_weights", self.stance_class_weights, 5)

    @staticmethod
    def _validate_weights(name: str, weights: tuple[float, ...] | None, size: int) -> None:
        if weights is None:
            return
        if len(weights) != size or any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError(f"{name} must contain exactly {size} finite non-negative values")
        if not any(weights):
            raise ValueError(f"{name} must contain at least one supported class")

    def digest(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class DecodeResult:
    labels: tuple[dict[str, Any], ...]
    forced_target_selections: int


def render_model_text(row: Mapping[str, Any], *, separator: str) -> str:
    """Render target, parent, and submission segments in the registered truncation order."""

    if not isinstance(separator, str) or not separator:
        raise ValueError("separator must be a non-empty string")
    target = row.get("target_text")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target_text must be a non-empty string")
    contexts: list[str] = []
    for field in ("parent_context", "submission_context"):
        value = row.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
        contexts.append(value or "")
    return f"{target} {separator} {contexts[0]} {separator} {contexts[1]}"


def load_pinned_tokenizer(
    config: ModelConfig | None = None, *, tokenizer_loader: Any | None = None
) -> Any:
    """Load the exact tokenizer revision, with an injectable loader for local tests."""

    config = config or ModelConfig()
    if tokenizer_loader is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - exercised in the Modal image
            raise RuntimeError(
                "Transformers is required to load the ModernBERT tokenizer"
            ) from error
        tokenizer_loader = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_loader(
        config.model_id,
        revision=config.tokenizer_revision,
        use_fast=True,
    )
    _verify_loaded_revision(tokenizer, config.tokenizer_revision, component="tokenizer")
    if not getattr(tokenizer, "sep_token", None):
        raise RuntimeError("pinned tokenizer does not expose a separator token")
    return tokenizer


def _verify_loaded_revision(component_value: Any, expected: str, *, component: str) -> None:
    """Reject an explicitly reported commit that differs from the requested revision."""

    init_kwargs = getattr(component_value, "init_kwargs", {}) or {}
    component_config = getattr(component_value, "config", None)
    reported = (
        init_kwargs.get("_commit_hash")
        or getattr(component_value, "_commit_hash", None)
        or getattr(component_config, "_commit_hash", None)
    )
    if reported is not None and reported != expected:
        raise RuntimeError(f"loaded {component} revision does not match the frozen revision")


def tokenise_record(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Tokenise without padding and report exact truncation at the registered length."""

    if max_length != MAX_LENGTH:
        raise ValueError(f"max_length must remain frozen at {MAX_LENGTH}")
    text = render_model_text(row, separator=tokenizer.sep_token)
    raw = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    input_ids = list(raw["input_ids"])
    attention_mask = list(raw.get("attention_mask", [1] * len(input_ids)))
    if len(input_ids) != len(attention_mask):
        raise RuntimeError("tokenizer returned mismatched input IDs and attention mask")
    original_tokens = len(input_ids)
    if original_tokens > max_length:
        # Preserve the tokenizer's final special token while keeping the target-first prefix.
        input_ids = [*input_ids[: max_length - 1], input_ids[-1]]
        attention_mask = [*attention_mask[: max_length - 1], attention_mask[-1]]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_count": original_tokens,
        "truncated_tokens": max(0, original_tokens - max_length),
    }


def encode_semantic_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one frozen semantic label into the three heads' labels and masks."""

    if set(label) != {"relevance", "target_stances"}:
        raise ValueError("semantic label must contain exactly relevance and target_stances")
    relevance = label.get("relevance")
    if relevance not in RELEVANCE_LABELS:
        raise ValueError("semantic label has an unsupported relevance")
    raw_target_stances = label.get("target_stances")
    if not isinstance(raw_target_stances, list):
        raise ValueError("target_stances must be a list")
    target_labels = [0] * len(TARGET_LABELS)
    stance_labels = [ABSENT_STANCE_INDEX] * len(TARGET_LABELS)
    target_presence_mask = [False] * len(TARGET_LABELS)
    for item in raw_target_stances:
        if not isinstance(item, Mapping) or set(item) != {"target", "stance"}:
            raise ValueError("each target stance must contain exactly target and stance")
        target = item.get("target")
        stance = item.get("stance")
        if target not in TARGET_LABELS or stance not in STANCE_LABELS:
            raise ValueError("semantic label has an unsupported target or stance")
        target_index = TARGET_LABELS.index(target)
        if target_presence_mask[target_index]:
            raise ValueError("semantic label contains a duplicate target")
        target_labels[target_index] = 1
        stance_labels[target_index] = STANCE_LABELS.index(stance)
        target_presence_mask[target_index] = True
    if relevance == "material" and not any(target_presence_mask):
        raise ValueError("material semantic labels require at least one target")
    if relevance != "material" and any(target_presence_mask):
        raise ValueError("non-material semantic labels cannot contain targets")
    return {
        "relevance_labels": RELEVANCE_LABELS.index(relevance),
        "target_labels": target_labels,
        "stance_labels": stance_labels,
    }


class DynamicPaddingCollator:
    """Pad each batch to its longest member while retaining training labels and item IDs."""

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


def build_length_bucket_batches(
    lengths: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    drop_last: bool = False,
    bucket_size_multiplier: int = 50,
) -> list[list[int]]:
    """Return deterministic shuffled batches with low within-batch padding variance."""

    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("lengths must be a non-empty sequence of positive integers")
    if batch_size < 1 or bucket_size_multiplier < 1 or epoch < 0:
        raise ValueError("batch size and multiplier must be positive; epoch must be non-negative")
    rng = random.Random(f"{seed}:{epoch}:modernbert-length-buckets")
    indices = list(range(len(lengths)))
    rng.shuffle(indices)
    bucket_size = batch_size * bucket_size_multiplier
    batches: list[list[int]] = []
    for offset in range(0, len(indices), bucket_size):
        bucket = sorted(indices[offset : offset + bucket_size], key=lambda index: lengths[index])
        batches.extend(
            bucket[start : start + batch_size] for start in range(0, len(bucket), batch_size)
        )
    if drop_last:
        batches = [batch for batch in batches if len(batch) == batch_size]
    rng.shuffle(batches)
    return batches


def seed_everything(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Seed Python and the installed Torch runtime, including DataLoader workers."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from Torch's deterministic initial seed."""

    torch = _require_torch()
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in the Modal image
        raise RuntimeError("PyTorch is required for ModernBERT training") from error
    return torch


def create_modernbert_three_head_model(
    *,
    model_config: ModelConfig | None = None,
    optimisation_config: OptimisationConfig,
    encoder_loader: Any | None = None,
) -> Any:
    """Compose the authoritative three-head model around the exact pinned encoder."""

    model_config = model_config or ModelConfig()
    # Importing the optional model module is delayed until construction so the base package remains
    # usable without the ML dependency group.
    from reddit_china_stance.modernbert_model import ModernBertThreeHeadModel

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
    return ModernBertThreeHeadModel(encoder, dropout=model_config.dropout)


def model_loss_kwargs(config: OptimisationConfig) -> dict[str, Any]:
    """Translate the trial config to the authoritative model's exact loss interface."""

    return {
        "task_loss_weights": (
            config.lambda_relevance,
            config.lambda_targets,
            config.lambda_stance,
        ),
        "relevance_class_weights": config.relevance_class_weights,
        "target_pos_weights": config.target_positive_weights,
        "stance_class_weights": config.stance_class_weights,
    }


def adamw_parameter_groups(model: Any, config: OptimisationConfig) -> list[dict[str, Any]]:
    """Build deterministic encoder/head and decay/no-decay AdamW parameter groups."""

    grouped: dict[tuple[str, bool], list[Any]] = {
        (scope, decay): [] for scope in ("encoder", "heads") for decay in (False, True)
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        if id(parameter) in seen:
            raise ValueError("a trainable parameter appears under multiple names")
        seen.add(id(parameter))
        scope = "encoder" if name.startswith("encoder.") else "heads"
        lowered = name.lower()
        no_decay = (
            name.endswith(".bias")
            or "layernorm" in lowered
            or "layer_norm" in lowered
            or ".norm." in lowered
        )
        grouped[(scope, not no_decay)].append(parameter)
    if not seen:
        raise ValueError("model has no trainable parameters")
    result: list[dict[str, Any]] = []
    for scope in ("encoder", "heads"):
        for decay in (True, False):
            parameters = grouped[(scope, decay)]
            if not parameters:
                continue
            learning_rate = config.encoder_learning_rate * (
                config.head_learning_rate_multiplier if scope == "heads" else 1.0
            )
            result.append(
                {
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": config.weight_decay if decay else 0.0,
                    "group_name": f"{scope}_{'decay' if decay else 'no_decay'}",
                }
            )
    return result


def create_adamw(model: Any, config: OptimisationConfig) -> Any:
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
    config: OptimisationConfig,
    device: str,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Train one epoch with BF16 autocast and exact gradient accumulation accounting."""

    torch = _require_torch()
    resolved_device = torch.device(device)
    if config.use_bf16 and resolved_device.type != "cuda":
        raise ValueError("BF16 training is registered only for CUDA devices")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sums = {
        key: None for key in ("loss", "relevance_loss", "target_loss", "stance_loss")
    }
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
            output = model(**model_batch, **model_loss_kwargs(config))
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
    target_threshold: float,
    reference: Mapping[str, Mapping[str, Any]] | None = None,
    use_bf16: bool = True,
    config: OptimisationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one epoch and optionally score semantic predictions by item ID."""

    torch = _require_torch()
    resolved_device = torch.device(device)
    if use_bf16 and resolved_device.type != "cuda":
        raise ValueError("BF16 evaluation is registered only for CUDA devices")
    model.eval()
    loss_sums = {
        key: None for key in ("loss", "relevance_loss", "target_loss", "stance_loss")
    }
    examples = batches = 0
    item_ids: list[str] = []
    relevance_logits: list[Any] = []
    target_logits: list[Any] = []
    stance_logits: list[Any] = []
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
                loss_kwargs = {} if config is None else model_loss_kwargs(config)
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
                target_logits.extend(output["target_logits"].detach().float().cpu().tolist())
                stance_logits.extend(output["stance_logits"].detach().float().cpu().tolist())
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
        decoded = decode_predictions(
            relevance_logits,
            target_logits,
            stance_logits,
            target_threshold=target_threshold,
        )
        predictions = dict(zip(item_ids, decoded.labels, strict=True))
        metrics = score_semantic_labels(reference, predictions)
        metrics["decoding"] = {
            "target_threshold": target_threshold,
            "forced_target_selections": decoded.forced_target_selections,
        }
        result["metrics"] = metrics
        prediction_rows = sorted(
            (
                {
                    "source_sample_id": item_id,
                    "relevance_logits": relevance,
                    "target_logits": targets,
                    "stance_logits": stances,
                    "decoded_label": label,
                }
                for item_id, relevance, targets, stances, label in zip(
                    item_ids,
                    relevance_logits,
                    target_logits,
                    stance_logits,
                    decoded.labels,
                    strict=True,
                )
            ),
            key=lambda row: row["source_sample_id"],
        )
        result["prediction_payload"] = {
            "schema_version": "1.0.0",
            "kind": "modernbert-private-development-predictions-v1",
            "row_count": len(prediction_rows),
            "decoder_target_threshold": target_threshold,
            "rows": prediction_rows,
        }
    return result


def _as_nested_lists(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("logits must be a tensor-like object or nested list")
    return value


def _argmax(values: Sequence[float]) -> int:
    if not values or any(
        not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
    ):
        raise ValueError("each logit row must be a non-empty finite numeric sequence")
    return max(range(len(values)), key=values.__getitem__)


def _sigmoid(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("target logits must be finite numbers")
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def decode_predictions(
    relevance_logits: Any,
    target_logits: Any,
    stance_logits: Any,
    *,
    target_threshold: float,
) -> DecodeResult:
    """Apply the frozen constrained decoder and return semantic-evaluation labels."""

    if not math.isfinite(target_threshold) or not 0 < target_threshold < 1:
        raise ValueError("target_threshold must be in (0, 1)")
    relevance_rows = _as_nested_lists(relevance_logits)
    target_rows = _as_nested_lists(target_logits)
    stance_rows = _as_nested_lists(stance_logits)
    if not (len(relevance_rows) == len(target_rows) == len(stance_rows)):
        raise ValueError("all three logit tensors must have the same batch dimension")
    labels: list[dict[str, Any]] = []
    forced = 0
    for relevance_row, target_row, stance_row in zip(
        relevance_rows, target_rows, stance_rows, strict=True
    ):
        if len(relevance_row) != len(RELEVANCE_LABELS):
            raise ValueError("relevance logits have the wrong class dimension")
        if len(target_row) != len(TARGET_LABELS):
            raise ValueError("target logits have the wrong class dimension")
        if len(stance_row) != len(TARGET_LABELS) or any(
            len(per_target) != len(STANCE_LABELS) for per_target in stance_row
        ):
            raise ValueError("stance logits have the wrong target or class dimension")
        relevance = RELEVANCE_LABELS[_argmax(relevance_row)]
        if relevance != "material":
            labels.append({"relevance": relevance, "target_stances": []})
            continue
        selected = [
            index for index, logit in enumerate(target_row) if _sigmoid(logit) >= target_threshold
        ]
        if not selected:
            selected = [_argmax(target_row)]
            forced += 1
        labels.append(
            {
                "relevance": "material",
                "target_stances": [
                    {
                        "target": TARGET_LABELS[index],
                        "stance": STANCE_LABELS[_argmax(stance_row[index])],
                    }
                    for index in selected
                ],
            }
        )
    return DecodeResult(labels=tuple(labels), forced_target_selections=forced)


def aggregate_semantic_metrics(
    reference: Mapping[str, Mapping[str, Any]],
    *,
    item_ids: Sequence[str],
    relevance_logits: Any,
    target_logits: Any,
    stance_logits: Any,
    target_threshold: float,
) -> dict[str, Any]:
    """Decode a complete evaluation frame and call the repository's aggregate scorer."""

    decoded = decode_predictions(
        relevance_logits,
        target_logits,
        stance_logits,
        target_threshold=target_threshold,
    )
    if len(item_ids) != len(decoded.labels) or len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids must be unique and align exactly with decoded rows")
    predictions = dict(zip(item_ids, decoded.labels, strict=True))
    metrics = score_semantic_labels(reference, predictions)
    metrics["decoding"] = {
        "target_threshold": target_threshold,
        "forced_target_selections": decoded.forced_target_selections,
    }
    return metrics


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor state deterministically by name, dtype, shape, and raw CPU bytes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not hasattr(tensor, "detach"):
            raise TypeError("state_dict values must be tensor-like")
        value = tensor.detach().contiguous().cpu()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = bytes(value.untyped_storage())
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def nested_state_sha256(value: Any) -> str:
    """Hash a nested optimiser, scheduler, or RNG state without serialisation metadata."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if hasattr(item, "detach") and hasattr(item, "untyped_storage"):
            tensor = item.detach().contiguous().cpu()
            digest.update(b"tensor\0")
            update(str(tensor.dtype))
            update(list(tensor.shape))
            raw = bytes(tensor.untyped_storage())
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            ordered = sorted(item.items(), key=lambda pair: canonical_sha256(pair[0]))
            for key, nested in ordered:
                update(key)
                update(nested)
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(len(item).to_bytes(8, "big"))
            for nested in item:
                update(nested)
        elif item is None or isinstance(item, (bool, int, float, str)):
            encoded = json.dumps(item, allow_nan=False, separators=(",", ":")).encode()
            digest.update(b"scalar\0")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        else:
            raise TypeError(f"unsupported checkpoint-state value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture Python and Torch RNGs needed for deterministic local/Modal resume."""

    torch = _require_torch()
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    if set(state) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("RNG state is malformed")
    torch = _require_torch()
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def build_checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    model_config: ModelConfig,
    optimisation_config: OptimisationConfig,
    binding: Mapping[str, Any],
    epoch: int,
    global_step: int,
    optimizer_step: int,
    seed: int,
) -> dict[str, Any]:
    """Build a resumable checkpoint with exact data, code, config, and state bindings."""

    for field in _DIGEST_FIELDS:
        value = binding.get(field)
        if not isinstance(value, str):
            raise ValueError(f"checkpoint binding is missing {field}")
        _require_hex_digest(value, field=field)
    counters = (epoch, global_step, optimizer_step, seed)
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("checkpoint counters and seed must be non-negative integers")
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    scheduler_state = None if scheduler is None else scheduler.state_dict()
    rng_state = capture_rng_state()
    checkpoint_binding = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_id": model_config.model_id,
        "model_revision": model_config.model_revision,
        "tokenizer_revision": model_config.tokenizer_revision,
        "model_config_sha256": model_config.digest(),
        "optimisation_config_sha256": optimisation_config.digest(),
        **{field: binding[field] for field in _DIGEST_FIELDS},
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "seed": seed,
        "state_integrity": "checkpoint_file_sha256",
    }
    return {
        "binding": checkpoint_binding,
        "binding_sha256": canonical_sha256(checkpoint_binding),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "rng_state": rng_state,
    }


def save_checkpoint_atomic(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Atomically write one Torch checkpoint and return its public file descriptor."""

    torch = _require_torch()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete = path.with_name(f"{path.name}.incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete checkpoint already exists: {incomplete}")
    try:
        torch.save(dict(payload), incomplete)
        os.replace(incomplete, path)
    except BaseException:
        incomplete.unlink(missing_ok=True)
        raise
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def load_checkpoint_exact(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_binding_sha256: str,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load only a checkpoint whose complete file and metadata binding both match."""

    _require_hex_digest(expected_file_sha256, field="expected_file_sha256")
    _require_hex_digest(expected_binding_sha256, field="expected_binding_sha256")
    if file_sha256(path) != expected_file_sha256:
        raise ValueError("checkpoint file SHA-256 mismatch")
    torch = _require_torch()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("binding"), dict):
        raise ValueError("checkpoint payload is malformed")
    binding = payload["binding"]
    if payload.get("binding_sha256") != canonical_sha256(binding):
        raise ValueError("checkpoint binding digest is internally inconsistent")
    if payload["binding_sha256"] != expected_binding_sha256:
        raise ValueError("checkpoint binding digest mismatch")
    if binding.get("state_integrity") != "checkpoint_file_sha256":
        raise ValueError("checkpoint state-integrity contract mismatch")
    required_states = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
    }
    if not required_states <= set(payload):
        raise ValueError("checkpoint payload is missing resumable state")
    return payload


def restore_training_state(
    payload: Mapping[str, Any],
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    restore_rng: bool = True,
) -> None:
    """Restore already hash-validated training state without permissive key matching."""

    expected_scheduler = payload.get("scheduler_state_dict")
    if (scheduler is None) != (expected_scheduler is None):
        raise ValueError("checkpoint scheduler presence does not match the active trial")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(expected_scheduler)
    if restore_rng:
        restore_rng_state(payload["rng_state"])
