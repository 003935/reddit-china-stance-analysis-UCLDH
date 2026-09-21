"""ModernBERT three-head student model and deterministic decoding helpers.

PyTorch and Transformers are deliberately optional dependencies.  Importing this
module is safe in the base package; constructing or training the model requires
the ML environment to provide them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised in the optional ML test environment
    import torch
    import torch.nn.functional as torch_functional
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - normal for the base environment
    torch = None
    torch_functional = None
    nn = None

RELEVANCE_LABELS = ("material", "not_material", "unclear")
TARGET_LABELS = ("china_general", "government_ccp", "people_culture", "other")
STANCE_LABELS = ("negative", "mixed", "no_directed_stance", "positive", "unclear")

MATERIAL_INDEX = RELEVANCE_LABELS.index("material")
ABSENT_STANCE_INDEX = -100
TARGET_THRESHOLD_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True)
class DecodedBatch:
    """Schema-valid semantic labels plus decoder diagnostics."""

    predictions: tuple[dict[str, Any], ...]
    forced_target_selections: int


@dataclass(frozen=True)
class ThresholdSelection:
    """Frozen global target threshold selected from the registered grid."""

    threshold: float
    score: float


def _require_torch() -> None:
    if torch is None or nn is None or torch_functional is None:
        raise ModuleNotFoundError(
            "ModernBERT training requires the optional ML dependencies torch and transformers"
        )


def _validate_counts(counts: Sequence[int], *, expected: int, name: str) -> tuple[int, ...]:
    values = tuple(counts)
    if len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} counts")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"{name} counts must be non-negative integers")
    if not any(values):
        raise ValueError(f"{name} must contain at least one supported class")
    return values


def capped_inverse_sqrt_class_weights(
    counts: Sequence[int],
    *,
    minimum: float = 0.5,
    maximum: float = 4.0,
) -> tuple[float, ...]:
    """Return the registered normalised ``sqrt(N / class_count)`` weights.

    Unsupported classes receive a zero weight and are excluded from the
    normalisation.  Callers must still reject evaluation or training batches
    that unexpectedly contain such a class.
    """

    values = _validate_counts(counts, expected=len(counts), name="class")
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0:
        raise ValueError("class-weight bounds must be finite and positive")
    if maximum < minimum:
        raise ValueError("maximum class weight must be at least the minimum")

    total = sum(values)
    raw = [math.sqrt(total / count) for count in values if count > 0]
    normaliser = sum(raw) / len(raw)
    result: list[float] = []
    supported_index = 0
    for count in values:
        if count == 0:
            result.append(0.0)
            continue
        normalised = raw[supported_index] / normaliser
        supported_index += 1
        result.append(min(max(normalised, minimum), maximum))
    return tuple(result)


def target_positive_weights(
    positive_counts: Sequence[int],
    *,
    material_count: int,
    minimum: float = 1.0,
    maximum: float = 4.0,
) -> tuple[float, ...]:
    """Return registered BCE positive weights for the four target labels."""

    values = tuple(positive_counts)
    if len(values) != len(TARGET_LABELS):
        raise ValueError(f"positive_counts must contain exactly {len(TARGET_LABELS)} counts")
    if (
        isinstance(material_count, bool)
        or not isinstance(material_count, int)
        or material_count <= 0
    ):
        raise ValueError("material_count must be a positive integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > material_count
        for value in values
    ):
        raise ValueError("positive counts must be integers between zero and material_count")
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0:
        raise ValueError("positive-weight bounds must be finite and positive")
    if maximum < minimum:
        raise ValueError("maximum positive weight must be at least the minimum")

    weights = []
    for positive in values:
        if positive == 0:
            weights.append(1.0)
            continue
        negative = material_count - positive
        raw = math.sqrt(negative / positive)
        weights.append(min(max(raw, minimum), maximum))
    return tuple(weights)


def _as_probability_row(
    values: Sequence[float],
    *,
    expected: int,
    name: str,
    sum_to_one: bool,
) -> tuple[float, ...]:
    row = tuple(float(value) for value in values)
    if len(row) != expected:
        raise ValueError(f"{name} must contain exactly {expected} probabilities")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in row):
        raise ValueError(f"{name} probabilities must be finite and within [0, 1]")
    if sum_to_one and not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} probabilities must sum to one")
    return row


def decode_probabilities(
    relevance_probabilities: Sequence[Sequence[float]],
    target_probabilities: Sequence[Sequence[float]],
    stance_probabilities: Sequence[Sequence[Sequence[float]]],
    *,
    target_threshold: float,
) -> DecodedBatch:
    """Apply the frozen constrained decoder to already-normalised probabilities."""

    if (
        isinstance(target_threshold, bool)
        or not isinstance(target_threshold, (int, float))
        or not math.isfinite(target_threshold)
        or not 0.0 <= target_threshold <= 1.0
    ):
        raise ValueError("target_threshold must be a finite probability")
    batch_size = len(relevance_probabilities)
    if len(target_probabilities) != batch_size or len(stance_probabilities) != batch_size:
        raise ValueError("all probability tensors must have the same batch size")

    predictions: list[dict[str, Any]] = []
    forced = 0
    for row_index in range(batch_size):
        relevance = _as_probability_row(
            relevance_probabilities[row_index],
            expected=len(RELEVANCE_LABELS),
            name="relevance row",
            sum_to_one=True,
        )
        targets = _as_probability_row(
            target_probabilities[row_index],
            expected=len(TARGET_LABELS),
            name="target row",
            sum_to_one=False,
        )
        raw_stances = stance_probabilities[row_index]
        if len(raw_stances) != len(TARGET_LABELS):
            raise ValueError(f"stance row must contain exactly {len(TARGET_LABELS)} targets")
        stances = tuple(
            _as_probability_row(
                row,
                expected=len(STANCE_LABELS),
                name="stance target row",
                sum_to_one=True,
            )
            for row in raw_stances
        )

        relevance_index = max(range(len(relevance)), key=relevance.__getitem__)
        relevance_label = RELEVANCE_LABELS[relevance_index]
        if relevance_label != "material":
            predictions.append({"relevance": relevance_label, "target_stances": []})
            continue

        selected = [
            index for index, probability in enumerate(targets) if probability >= target_threshold
        ]
        if not selected:
            selected = [max(range(len(targets)), key=targets.__getitem__)]
            forced += 1
        target_stances = []
        for target_index in selected:
            stance = stances[target_index]
            stance_index = max(range(len(stance)), key=stance.__getitem__)
            target_stances.append(
                {
                    "target": TARGET_LABELS[target_index],
                    "stance": STANCE_LABELS[stance_index],
                }
            )
        predictions.append({"relevance": relevance_label, "target_stances": target_stances})

    return DecodedBatch(predictions=tuple(predictions), forced_target_selections=forced)


def select_global_target_threshold(
    scores_by_threshold: Mapping[float, float],
    *,
    grid: Sequence[float] = TARGET_THRESHOLD_GRID,
) -> ThresholdSelection:
    """Select the best registered threshold, breaking score ties upward."""

    registered = tuple(float(value) for value in grid)
    if not registered or len(set(registered)) != len(registered):
        raise ValueError("threshold grid must be non-empty and unique")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in registered):
        raise ValueError("threshold grid values must be finite probabilities")
    observed = {float(key): value for key, value in scores_by_threshold.items()}
    if set(observed) != set(registered):
        raise ValueError("scores must contain exactly the registered threshold grid")
    if any(
        isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        for score in observed.values()
    ):
        raise ValueError("threshold scores must be finite numbers")
    threshold = max(registered, key=lambda value: (float(observed[value]), value))
    return ThresholdSelection(threshold=threshold, score=float(observed[threshold]))


def masked_mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool non-padding token states and reject empty sequences."""

    _require_torch()
    if last_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError(
            "hidden states must be [batch, sequence, hidden] and mask [batch, sequence]"
        )
    if tuple(last_hidden_state.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError("attention mask shape must match hidden-state batch and sequence axes")
    if attention_mask.numel() == 0:
        raise ValueError("attention mask cannot be empty")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
        raise ValueError("attention mask must be binary")
    denominator = attention_mask.sum(dim=1)
    if bool((denominator == 0).any().item()):
        raise ValueError("every row must contain at least one non-padding token")
    expanded = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    return (last_hidden_state * expanded).sum(dim=1) / denominator.to(
        dtype=last_hidden_state.dtype
    ).unsqueeze(-1)


def _validate_loss_inputs(
    relevance_logits: Any,
    target_logits: Any,
    stance_logits: Any,
    relevance_labels: Any,
    target_labels: Any,
    stance_labels: Any,
) -> None:
    batch_size = relevance_logits.shape[0]
    if batch_size <= 0:
        raise ValueError("loss inputs must contain at least one row")
    expected_shapes = {
        "relevance_logits": (batch_size, len(RELEVANCE_LABELS)),
        "target_logits": (batch_size, len(TARGET_LABELS)),
        "stance_logits": (batch_size, len(TARGET_LABELS), len(STANCE_LABELS)),
        "relevance_labels": (batch_size,),
        "target_labels": (batch_size, len(TARGET_LABELS)),
        "stance_labels": (batch_size, len(TARGET_LABELS)),
    }
    actual = {
        "relevance_logits": tuple(relevance_logits.shape),
        "target_logits": tuple(target_logits.shape),
        "stance_logits": tuple(stance_logits.shape),
        "relevance_labels": tuple(relevance_labels.shape),
        "target_labels": tuple(target_labels.shape),
        "stance_labels": tuple(stance_labels.shape),
    }
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual[name]}")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if any(
        labels.dtype not in integer_dtypes
        for labels in (relevance_labels, target_labels, stance_labels)
    ):
        raise ValueError("all label tensors must use integer dtypes")
    if any(
        not bool(torch.isfinite(logits).all().item())
        for logits in (relevance_logits, target_logits, stance_logits)
    ):
        raise ValueError("all logits must be finite")
    if not bool(
        ((relevance_labels >= 0) & (relevance_labels < len(RELEVANCE_LABELS))).all().item()
    ):
        raise ValueError("relevance labels are out of range")
    if not bool(((target_labels == 0) | (target_labels == 1)).all().item()):
        raise ValueError("target labels must be binary")

    material_mask = relevance_labels == MATERIAL_INDEX
    target_count = target_labels.sum(dim=1)
    if bool(
        ((material_mask & (target_count == 0)) | (~material_mask & (target_count != 0)))
        .any()
        .item()
    ):
        raise ValueError("material rows require targets and non-material rows forbid targets")
    present = target_labels.bool()
    if bool(
        ((stance_labels[present] < 0) | (stance_labels[present] >= len(STANCE_LABELS))).any().item()
    ):
        raise ValueError("present-target stance labels are out of range")
    if bool((stance_labels[~present] != ABSENT_STANCE_INDEX).any().item()):
        raise ValueError(f"absent-target stance labels must be {ABSENT_STANCE_INDEX}")


def _optional_weight_tensor(
    value: Any, *, expected: int, name: str, device: Any, dtype: Any
) -> Any:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(tensor.shape) != (expected,):
        raise ValueError(f"{name} must contain exactly {expected} values")
    if not bool(torch.isfinite(tensor).all().item()) or bool((tensor < 0).any().item()):
        raise ValueError(f"{name} must contain finite non-negative values")
    return tensor


def compute_multitask_losses(
    *,
    relevance_logits: Any,
    target_logits: Any,
    stance_logits: Any,
    relevance_labels: Any,
    target_labels: Any,
    stance_labels: Any,
    task_loss_weights: Sequence[float] = (1.0, 1.0, 1.0),
    relevance_class_weights: Any = None,
    target_pos_weights: Any = None,
    stance_class_weights: Any = None,
) -> dict[str, Any]:
    """Compute exact registered reductions for all three task losses."""

    _require_torch()
    _validate_loss_inputs(
        relevance_logits,
        target_logits,
        stance_logits,
        relevance_labels,
        target_labels,
        stance_labels,
    )
    lambdas = tuple(float(value) for value in task_loss_weights)
    if len(lambdas) != 3 or any(not math.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("task_loss_weights must contain three finite non-negative values")
    relevance_weights = _optional_weight_tensor(
        relevance_class_weights,
        expected=len(RELEVANCE_LABELS),
        name="relevance_class_weights",
        device=relevance_logits.device,
        dtype=relevance_logits.dtype,
    )
    positive_weights = _optional_weight_tensor(
        target_pos_weights,
        expected=len(TARGET_LABELS),
        name="target_pos_weights",
        device=target_logits.device,
        dtype=target_logits.dtype,
    )
    stance_weights = _optional_weight_tensor(
        stance_class_weights,
        expected=len(STANCE_LABELS),
        name="stance_class_weights",
        device=stance_logits.device,
        dtype=stance_logits.dtype,
    )
    if relevance_weights is not None and bool(
        (relevance_weights[relevance_labels.long()] == 0).any().item()
    ):
        raise ValueError("an observed relevance class has zero class weight")
    present = target_labels.bool()
    if stance_weights is not None and bool(
        (stance_weights[stance_labels[present].long()] == 0).any().item()
    ):
        raise ValueError("an observed stance class has zero class weight")

    relevance_loss = torch_functional.cross_entropy(
        relevance_logits,
        relevance_labels.long(),
        weight=relevance_weights,
        reduction="mean",
    )
    material_mask = relevance_labels == MATERIAL_INDEX
    material_count = int(material_mask.sum().item())
    if material_count:
        target_unreduced = torch_functional.binary_cross_entropy_with_logits(
            target_logits[material_mask],
            target_labels[material_mask].to(dtype=target_logits.dtype),
            pos_weight=positive_weights,
            reduction="none",
        )
        target_loss = target_unreduced.sum() / (len(TARGET_LABELS) * material_count)
    else:  # Defensive: the strict label contract normally permits all-non-material batches.
        target_loss = target_logits.sum() * 0

    present_count = int(present.sum().item())
    if present_count:
        stance_loss = (
            torch_functional.cross_entropy(
                stance_logits[present],
                stance_labels[present].long(),
                weight=stance_weights,
                reduction="sum",
            )
            / present_count
        )
    else:
        stance_loss = stance_logits.sum() * 0

    total_loss = lambdas[0] * relevance_loss + lambdas[1] * target_loss + lambdas[2] * stance_loss
    return {
        "loss": total_loss,
        "relevance_loss": relevance_loss,
        "target_loss": target_loss,
        "stance_loss": stance_loss,
    }


if nn is not None:  # pragma: no branch - class body depends on the optional dependency

    class ModernBertThreeHeadModel(nn.Module):
        """One shared ModernBERT encoder with relevance, target and stance heads."""

        def __init__(self, encoder: Any, *, hidden_size: int | None = None, dropout: float = 0.1):
            super().__init__()
            if hidden_size is None:
                hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
            if (
                isinstance(hidden_size, bool)
                or not isinstance(hidden_size, int)
                or hidden_size <= 0
            ):
                raise ValueError(
                    "hidden_size must be a positive integer or available on encoder.config"
                )
            if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be finite and within [0, 1)")
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.relevance_head = nn.Linear(hidden_size, len(RELEVANCE_LABELS))
            self.target_head = nn.Linear(hidden_size, len(TARGET_LABELS))
            self.stance_head = nn.Linear(hidden_size, len(TARGET_LABELS) * len(STANCE_LABELS))

        @classmethod
        def from_pretrained(
            cls,
            model_name: str,
            *,
            revision: str,
            dropout: float = 0.1,
            **encoder_kwargs: Any,
        ) -> ModernBertThreeHeadModel:
            if not model_name.strip() or not revision.strip():
                raise ValueError("model_name and exact revision are required")
            try:
                from transformers import ModernBertModel
            except (ImportError, ModuleNotFoundError) as exc:
                raise ModuleNotFoundError(
                    "from_pretrained requires the optional transformers dependency"
                ) from exc
            encoder = ModernBertModel.from_pretrained(
                model_name,
                revision=revision,
                **encoder_kwargs,
            )
            return cls(encoder, dropout=dropout)

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            relevance_labels: Any = None,
            target_labels: Any = None,
            stance_labels: Any = None,
            task_loss_weights: Sequence[float] = (1.0, 1.0, 1.0),
            relevance_class_weights: Any = None,
            target_pos_weights: Any = None,
            stance_class_weights: Any = None,
        ) -> dict[str, Any]:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if not hasattr(encoded, "last_hidden_state"):
                raise ValueError("encoder output must expose last_hidden_state")
            pooled = self.dropout(masked_mean_pool(encoded.last_hidden_state, attention_mask))
            relevance_logits = self.relevance_head(pooled)
            target_logits = self.target_head(pooled)
            stance_logits = self.stance_head(pooled).reshape(
                pooled.shape[0], len(TARGET_LABELS), len(STANCE_LABELS)
            )
            result = {
                "relevance_logits": relevance_logits,
                "target_logits": target_logits,
                "stance_logits": stance_logits,
            }
            supplied = tuple(
                value is not None for value in (relevance_labels, target_labels, stance_labels)
            )
            if any(supplied) and not all(supplied):
                raise ValueError("all three label tensors are required when computing loss")
            if all(supplied):
                result.update(
                    compute_multitask_losses(
                        relevance_logits=relevance_logits,
                        target_logits=target_logits,
                        stance_logits=stance_logits,
                        relevance_labels=relevance_labels,
                        target_labels=target_labels,
                        stance_labels=stance_labels,
                        task_loss_weights=task_loss_weights,
                        relevance_class_weights=relevance_class_weights,
                        target_pos_weights=target_pos_weights,
                        stance_class_weights=stance_class_weights,
                    )
                )
            return result

else:

    class ModernBertThreeHeadModel:  # pragma: no cover - simple missing-dependency guard
        """Missing-dependency guard for the optional ModernBERT student model."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> ModernBertThreeHeadModel:
            _require_torch()
            raise AssertionError("unreachable")
