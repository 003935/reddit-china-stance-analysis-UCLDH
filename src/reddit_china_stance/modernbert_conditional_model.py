"""Conditional-state ModernBERT student primitive.

This module keeps the registered shared ModernBERT encoder and masked-mean
pooling, while replacing the separate target-presence and stance heads with one
six-state head per target: ``absent`` plus the five registered stance labels.

PyTorch and Transformers remain optional dependencies. Importing this module is
safe in the base package; constructing or training the model requires the ML
environment to provide them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reddit_china_stance.modernbert_model import (
    MATERIAL_INDEX,
    RELEVANCE_LABELS,
    STANCE_LABELS,
    TARGET_LABELS,
    masked_mean_pool,
)

try:  # pragma: no cover - exercised in the optional ML test environment
    import torch
    import torch.nn.functional as torch_functional
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - normal for the base environment
    torch = None
    torch_functional = None
    nn = None

TARGET_STATE_LABELS = ("absent", *STANCE_LABELS)
ABSENT_TARGET_STATE_INDEX = TARGET_STATE_LABELS.index("absent")


@dataclass(frozen=True)
class ConditionalDecodedBatch:
    """Schema-compatible semantic labels plus constrained-decoder diagnostics."""

    predictions: tuple[dict[str, Any], ...]
    forced_target_selections: int


def _require_torch() -> None:
    if torch is None or nn is None or torch_functional is None:
        raise ModuleNotFoundError(
            "conditional ModernBERT training requires the optional ML dependencies "
            "torch and transformers"
        )


def encode_conditional_semantic_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one semantic label into relevance and per-target state labels."""

    if not isinstance(label, Mapping) or set(label) != {"relevance", "target_stances"}:
        raise ValueError("semantic label must contain exactly relevance and target_stances")
    relevance = label.get("relevance")
    if relevance not in RELEVANCE_LABELS:
        raise ValueError("semantic label has an unsupported relevance")
    raw_target_stances = label.get("target_stances")
    if not isinstance(raw_target_stances, list):
        raise ValueError("target_stances must be a list")

    target_state_labels = [ABSENT_TARGET_STATE_INDEX] * len(TARGET_LABELS)
    observed_targets: set[str] = set()
    for item in raw_target_stances:
        if not isinstance(item, Mapping) or set(item) != {"target", "stance"}:
            raise ValueError("each target stance must contain exactly target and stance")
        target = item.get("target")
        stance = item.get("stance")
        if target not in TARGET_LABELS or stance not in STANCE_LABELS:
            raise ValueError("semantic label has an unsupported target or stance")
        if target in observed_targets:
            raise ValueError("semantic label contains a duplicate target")
        observed_targets.add(target)
        target_index = TARGET_LABELS.index(target)
        target_state_labels[target_index] = 1 + STANCE_LABELS.index(stance)

    if relevance == "material" and not observed_targets:
        raise ValueError("material semantic labels require at least one target")
    if relevance != "material" and observed_targets:
        raise ValueError("non-material semantic labels cannot contain targets")
    return {
        "relevance_labels": RELEVANCE_LABELS.index(relevance),
        "target_state_labels": target_state_labels,
    }


def _as_finite_row(values: Sequence[float], *, expected: int, name: str) -> tuple[float, ...]:
    try:
        row = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric logits") from exc
    if len(row) != expected:
        raise ValueError(f"{name} must contain exactly {expected} logits")
    if any(not math.isfinite(value) for value in row):
        raise ValueError(f"{name} logits must be finite")
    return row


def decode_conditional_logits(
    relevance_logits: Sequence[Sequence[float]],
    target_state_logits: Sequence[Sequence[Sequence[float]]],
) -> ConditionalDecodedBatch:
    """Decode conditional-state logits into the existing semantic-label schema.

    A material prediction normally emits every target whose highest-scoring
    state is not ``absent``. If all targets predict ``absent``, exactly one
    target is forced using the largest best-non-absent minus absent logit
    margin. Ties resolve deterministically in registered target/state order.
    """

    batch_size = len(relevance_logits)
    if len(target_state_logits) != batch_size:
        raise ValueError("all logit tensors must have the same batch size")

    predictions: list[dict[str, Any]] = []
    forced = 0
    for row_index in range(batch_size):
        relevance_row = _as_finite_row(
            relevance_logits[row_index],
            expected=len(RELEVANCE_LABELS),
            name="relevance row",
        )
        raw_target_rows = target_state_logits[row_index]
        if len(raw_target_rows) != len(TARGET_LABELS):
            raise ValueError(f"target-state row must contain exactly {len(TARGET_LABELS)} targets")
        target_rows = tuple(
            _as_finite_row(
                row,
                expected=len(TARGET_STATE_LABELS),
                name="target-state target row",
            )
            for row in raw_target_rows
        )

        relevance_index = max(range(len(relevance_row)), key=relevance_row.__getitem__)
        relevance_label = RELEVANCE_LABELS[relevance_index]
        if relevance_label != "material":
            predictions.append({"relevance": relevance_label, "target_stances": []})
            continue

        selected: list[tuple[int, int]] = []
        for target_index, state_row in enumerate(target_rows):
            state_index = max(range(len(state_row)), key=state_row.__getitem__)
            if state_index != ABSENT_TARGET_STATE_INDEX:
                selected.append((target_index, state_index))

        if not selected:
            best_by_target: list[tuple[float, int]] = []
            for state_row in target_rows:
                best_non_absent = max(
                    range(1, len(state_row)),
                    key=state_row.__getitem__,
                )
                margin = state_row[best_non_absent] - state_row[ABSENT_TARGET_STATE_INDEX]
                best_by_target.append((margin, best_non_absent))
            target_index = max(
                range(len(best_by_target)),
                key=lambda index: best_by_target[index][0],
            )
            selected = [(target_index, best_by_target[target_index][1])]
            forced += 1

        target_stances = [
            {
                "target": TARGET_LABELS[target_index],
                "stance": STANCE_LABELS[state_index - 1],
            }
            for target_index, state_index in selected
        ]
        predictions.append({"relevance": relevance_label, "target_stances": target_stances})

    return ConditionalDecodedBatch(
        predictions=tuple(predictions),
        forced_target_selections=forced,
    )


def _validate_tensor(name: str, value: Any, *, shape: tuple[int, ...]) -> None:
    if not hasattr(value, "shape") or tuple(value.shape) != shape:
        actual = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} must have shape {shape}, got {actual}")


def _validate_loss_inputs(
    *,
    relevance_logits: Any,
    target_state_logits: Any,
    relevance_labels: Any,
    target_state_labels: Any,
) -> None:
    if not hasattr(relevance_logits, "shape") or relevance_logits.ndim != 2:
        raise ValueError("relevance_logits must be a rank-two tensor")
    batch_size = relevance_logits.shape[0]
    if batch_size <= 0:
        raise ValueError("loss inputs must contain at least one row")
    _validate_tensor(
        "relevance_logits",
        relevance_logits,
        shape=(batch_size, len(RELEVANCE_LABELS)),
    )
    _validate_tensor(
        "target_state_logits",
        target_state_logits,
        shape=(batch_size, len(TARGET_LABELS), len(TARGET_STATE_LABELS)),
    )
    _validate_tensor("relevance_labels", relevance_labels, shape=(batch_size,))
    _validate_tensor(
        "target_state_labels",
        target_state_labels,
        shape=(batch_size, len(TARGET_LABELS)),
    )

    if not relevance_logits.is_floating_point() or not target_state_logits.is_floating_point():
        raise ValueError("all logits must use floating-point dtypes")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        relevance_labels.dtype not in integer_dtypes
        or target_state_labels.dtype not in integer_dtypes
    ):
        raise ValueError("all label tensors must use integer dtypes")
    if relevance_logits.device != relevance_labels.device:
        raise ValueError("relevance logits and labels must use the same device")
    if target_state_logits.device != target_state_labels.device:
        raise ValueError("target-state logits and labels must use the same device")
    if relevance_logits.device != target_state_logits.device:
        raise ValueError("all loss tensors must use the same device")
    if not bool(torch.isfinite(relevance_logits).all().item()) or not bool(
        torch.isfinite(target_state_logits).all().item()
    ):
        raise ValueError("all logits must be finite")
    if not bool(
        ((relevance_labels >= 0) & (relevance_labels < len(RELEVANCE_LABELS))).all().item()
    ):
        raise ValueError("relevance labels are out of range")
    if not bool(
        ((target_state_labels >= 0) & (target_state_labels < len(TARGET_STATE_LABELS))).all().item()
    ):
        raise ValueError("target-state labels are out of range")

    material_mask = relevance_labels == MATERIAL_INDEX
    has_target = (target_state_labels != ABSENT_TARGET_STATE_INDEX).any(dim=1)
    if bool(((material_mask & ~has_target) | (~material_mask & has_target)).any().item()):
        raise ValueError("material rows require targets and non-material rows forbid targets")


def _optional_weight_tensor(
    value: Any,
    *,
    expected: int,
    name: str,
    device: Any,
    dtype: Any,
) -> Any:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(tensor.shape) != (expected,):
        raise ValueError(f"{name} must contain exactly {expected} values")
    if not bool(torch.isfinite(tensor).all().item()) or bool((tensor < 0).any().item()):
        raise ValueError(f"{name} must contain finite non-negative values")
    return tensor


def _loss_weight(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def compute_conditional_losses(
    *,
    relevance_logits: Any,
    target_state_logits: Any,
    relevance_labels: Any,
    target_state_labels: Any,
    relevance_loss_weight: float = 1.0,
    target_state_loss_weight: float = 1.0,
    relevance_class_weights: Any = None,
    target_state_class_weights: Any = None,
) -> dict[str, Any]:
    """Compute relevance CE and material-row target-state CE.

    Target-state loss is reduced over all four target slots of material rows.
    Non-material rows are deliberately masked from that task.
    """

    _require_torch()
    _validate_loss_inputs(
        relevance_logits=relevance_logits,
        target_state_logits=target_state_logits,
        relevance_labels=relevance_labels,
        target_state_labels=target_state_labels,
    )
    relevance_lambda = _loss_weight(relevance_loss_weight, name="relevance_loss_weight")
    state_lambda = _loss_weight(target_state_loss_weight, name="target_state_loss_weight")
    if relevance_lambda == 0 and state_lambda == 0:
        raise ValueError("at least one task loss weight must be positive")

    relevance_weights = _optional_weight_tensor(
        relevance_class_weights,
        expected=len(RELEVANCE_LABELS),
        name="relevance_class_weights",
        device=relevance_logits.device,
        dtype=relevance_logits.dtype,
    )
    state_weights = _optional_weight_tensor(
        target_state_class_weights,
        expected=len(TARGET_STATE_LABELS),
        name="target_state_class_weights",
        device=target_state_logits.device,
        dtype=target_state_logits.dtype,
    )
    if relevance_weights is not None and bool(
        (relevance_weights[relevance_labels.long()] == 0).any().item()
    ):
        raise ValueError("an observed relevance class has zero class weight")

    material_mask = relevance_labels == MATERIAL_INDEX
    material_count = int(material_mask.sum().item())
    if (
        state_weights is not None
        and material_count
        and bool((state_weights[target_state_labels[material_mask].long()] == 0).any().item())
    ):
        raise ValueError("an observed target-state class has zero class weight")

    relevance_loss = torch_functional.cross_entropy(
        relevance_logits,
        relevance_labels.long(),
        weight=relevance_weights,
        reduction="mean",
    )
    if material_count:
        target_state_loss = torch_functional.cross_entropy(
            target_state_logits[material_mask].reshape(-1, len(TARGET_STATE_LABELS)),
            target_state_labels[material_mask].reshape(-1).long(),
            weight=state_weights,
            reduction="sum",
        ) / (material_count * len(TARGET_LABELS))
    else:
        target_state_loss = target_state_logits.sum() * 0

    total_loss = relevance_lambda * relevance_loss + state_lambda * target_state_loss
    return {
        "loss": total_loss,
        "relevance_loss": relevance_loss,
        "target_state_loss": target_state_loss,
    }


if nn is not None:  # pragma: no branch - class body depends on the optional dependency

    class ModernBertConditionalStateModel(nn.Module):
        """One shared ModernBERT encoder with relevance and target-state heads."""

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
            if (
                isinstance(dropout, bool)
                or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout)
                or not 0.0 <= dropout < 1.0
            ):
                raise ValueError("dropout must be finite and within [0, 1)")
            self.encoder = encoder
            self.dropout = nn.Dropout(float(dropout))
            self.relevance_head = nn.Linear(hidden_size, len(RELEVANCE_LABELS))
            self.target_state_head = nn.Linear(
                hidden_size,
                len(TARGET_LABELS) * len(TARGET_STATE_LABELS),
            )

        @classmethod
        def from_pretrained(
            cls,
            model_name: str,
            *,
            revision: str,
            dropout: float = 0.1,
            **encoder_kwargs: Any,
        ) -> ModernBertConditionalStateModel:
            if (
                not isinstance(model_name, str)
                or not model_name.strip()
                or not isinstance(revision, str)
                or not revision.strip()
            ):
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
            target_state_labels: Any = None,
            relevance_loss_weight: float = 1.0,
            target_state_loss_weight: float = 1.0,
            relevance_class_weights: Any = None,
            target_state_class_weights: Any = None,
        ) -> dict[str, Any]:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if not hasattr(encoded, "last_hidden_state"):
                raise ValueError("encoder output must expose last_hidden_state")
            pooled = self.dropout(masked_mean_pool(encoded.last_hidden_state, attention_mask))
            relevance_logits = self.relevance_head(pooled)
            target_state_logits = self.target_state_head(pooled).reshape(
                pooled.shape[0],
                len(TARGET_LABELS),
                len(TARGET_STATE_LABELS),
            )
            result = {
                "relevance_logits": relevance_logits,
                "target_state_logits": target_state_logits,
            }
            supplied = (relevance_labels is not None, target_state_labels is not None)
            if any(supplied) and not all(supplied):
                raise ValueError("both label tensors are required when computing loss")
            if all(supplied):
                result.update(
                    compute_conditional_losses(
                        relevance_logits=relevance_logits,
                        target_state_logits=target_state_logits,
                        relevance_labels=relevance_labels,
                        target_state_labels=target_state_labels,
                        relevance_loss_weight=relevance_loss_weight,
                        target_state_loss_weight=target_state_loss_weight,
                        relevance_class_weights=relevance_class_weights,
                        target_state_class_weights=target_state_class_weights,
                    )
                )
            return result


else:

    class ModernBertConditionalStateModel:  # pragma: no cover - dependency guard
        """Missing-dependency guard for the optional conditional-state model."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

        @classmethod
        def from_pretrained(
            cls,
            *_args: Any,
            **_kwargs: Any,
        ) -> ModernBertConditionalStateModel:
            _require_torch()
            raise AssertionError("unreachable")


__all__ = [
    "ABSENT_TARGET_STATE_INDEX",
    "TARGET_STATE_LABELS",
    "ConditionalDecodedBatch",
    "ModernBertConditionalStateModel",
    "compute_conditional_losses",
    "decode_conditional_logits",
    "encode_conditional_semantic_label",
]
