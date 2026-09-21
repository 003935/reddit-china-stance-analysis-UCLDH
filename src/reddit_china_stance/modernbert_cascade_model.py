"""Separate-model ModernBERT cascade primitives.

The cascade deliberately uses two independent encoders.  The first answers only
the three-way relevance question.  The second is run once per registered target
and answers one six-way target-presence-and-stance question.  Keeping these
modules separate prevents relevance gradients from changing the representation
used by the target-conditioned task.

PyTorch and Transformers are optional package dependencies.  Metadata-only
helpers remain importable without either runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reddit_china_stance.modernbert_model import (
    RELEVANCE_LABELS,
    STANCE_LABELS,
    TARGET_LABELS,
    masked_mean_pool,
)

try:  # pragma: no cover - exercised in the optional ML environment
    import torch
    import torch.nn.functional as torch_functional
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - normal for the base environment
    torch = None
    torch_functional = None
    nn = None

TARGET_STATE_LABELS = ("absent", *STANCE_LABELS)
ABSENT_TARGET_STATE_INDEX = TARGET_STATE_LABELS.index("absent")


@dataclass(frozen=True, slots=True)
class CascadeDecodedBatch:
    """Schema-valid cascade predictions plus merger diagnostics."""

    predictions: tuple[dict[str, Any], ...]
    forced_target_selections: int


def _require_torch() -> None:
    if torch is None or nn is None or torch_functional is None:
        raise ModuleNotFoundError(
            "cascade ModernBERT execution requires the optional ML dependencies "
            "torch and transformers"
        )


def _validate_target(target: Any) -> str:
    if not isinstance(target, str) or target not in TARGET_LABELS:
        raise ValueError(f"target must be one of {TARGET_LABELS}")
    return target


def render_target_conditioned_text(
    row: Mapping[str, Any],
    *,
    target: str,
    separator: str,
) -> str:
    """Render one explicit target-conditioned example in target-text-first order."""

    target = _validate_target(target)
    if not isinstance(separator, str) or not separator:
        raise ValueError("separator must be a non-empty string")
    target_text = row.get("target_text")
    if not isinstance(target_text, str) or not target_text.strip():
        raise ValueError("target_text must be a non-empty string")
    contexts: dict[str, str] = {}
    for field in ("parent_context", "submission_context"):
        value = row.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
        contexts[field] = value or ""
    return (
        f"[TARGET={target}] {separator} "
        f"{target_text} {separator} "
        f"{contexts['parent_context']} {separator} "
        f"{contexts['submission_context']}"
    )


def encode_relevance_semantic_label(label: Mapping[str, Any]) -> int:
    """Encode only the registered three-way relevance decision."""

    if not isinstance(label, Mapping) or set(label) != {"relevance", "target_stances"}:
        raise ValueError("semantic label must contain exactly relevance and target_stances")
    relevance = label.get("relevance")
    if relevance not in RELEVANCE_LABELS:
        raise ValueError("semantic label has an unsupported relevance")
    target_stances = label.get("target_stances")
    if not isinstance(target_stances, list):
        raise ValueError("target_stances must be a list")
    return RELEVANCE_LABELS.index(relevance)


def encode_target_conditioned_semantic_label(
    label: Mapping[str, Any],
    *,
    target: str,
) -> int:
    """Encode one target as ``absent`` or its registered stance state."""

    target = _validate_target(target)
    # Validate the complete schema and relevance value before selecting one target.
    encode_relevance_semantic_label(label)
    relevance = label["relevance"]
    target_stances = label["target_stances"]
    observed: dict[str, str] = {}
    for item in target_stances:
        if not isinstance(item, Mapping) or set(item) != {"target", "stance"}:
            raise ValueError("each target stance must contain exactly target and stance")
        item_target = item.get("target")
        stance = item.get("stance")
        if item_target not in TARGET_LABELS or stance not in STANCE_LABELS:
            raise ValueError("semantic label has an unsupported target or stance")
        if item_target in observed:
            raise ValueError("semantic label contains a duplicate target")
        observed[item_target] = stance
    if relevance == "material" and not observed:
        raise ValueError("material semantic labels require at least one target")
    if relevance != "material" and observed:
        raise ValueError("non-material semantic labels cannot contain targets")
    stance = observed.get(target)
    return ABSENT_TARGET_STATE_INDEX if stance is None else 1 + STANCE_LABELS.index(stance)


def expand_target_conditioned_labels(label: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand one semantic row into four deterministic target-conditioned labels."""

    return tuple(
        {
            "target": target,
            "target_state_labels": encode_target_conditioned_semantic_label(
                label,
                target=target,
            ),
        }
        for target in TARGET_LABELS
    )


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


def decode_cascade_logits(
    relevance_logits: Sequence[Sequence[float]],
    target_state_logits: Sequence[Sequence[Sequence[float]]],
) -> CascadeDecodedBatch:
    """Merge both models into the existing semantic-label schema.

    Non-material relevance decisions stop the cascade.  A material row emits
    every target whose six-way decision is not ``absent``.  If all four target
    decisions are absent, the merger deterministically forces the target and
    stance with the largest best-non-absent versus absent logit margin.  This is
    required by the repository schema, where material rows must name a target.
    """

    batch_size = len(relevance_logits)
    if len(target_state_logits) != batch_size:
        raise ValueError("relevance and target-state logits must have the same batch size")

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
        relevance = RELEVANCE_LABELS[relevance_index]
        if relevance != "material":
            predictions.append({"relevance": relevance, "target_stances": []})
            continue

        selected: list[tuple[int, int]] = []
        for target_index, state_row in enumerate(target_rows):
            state_index = max(range(len(state_row)), key=state_row.__getitem__)
            if state_index != ABSENT_TARGET_STATE_INDEX:
                selected.append((target_index, state_index))

        if not selected:
            alternatives: list[tuple[float, int]] = []
            for state_row in target_rows:
                state_index = max(range(1, len(state_row)), key=state_row.__getitem__)
                alternatives.append(
                    (
                        state_row[state_index] - state_row[ABSENT_TARGET_STATE_INDEX],
                        state_index,
                    )
                )
            target_index = max(range(len(alternatives)), key=lambda index: alternatives[index][0])
            selected = [(target_index, alternatives[target_index][1])]
            forced += 1

        predictions.append(
            {
                "relevance": "material",
                "target_stances": [
                    {
                        "target": TARGET_LABELS[target_index],
                        "stance": STANCE_LABELS[state_index - 1],
                    }
                    for target_index, state_index in selected
                ],
            }
        )
    return CascadeDecodedBatch(
        predictions=tuple(predictions),
        forced_target_selections=forced,
    )


def _validate_logits_and_labels(
    *,
    logits: Any,
    labels: Any,
    classes: int,
    logit_name: str,
    label_name: str,
) -> None:
    if (
        not hasattr(labels, "shape")
        or getattr(labels, "ndim", None) != 1
        or labels.shape[0] <= 0
    ):
        raise ValueError(f"{label_name} must be a non-empty rank-one tensor")
    batch_size = int(labels.shape[0])
    if not hasattr(logits, "shape") or tuple(logits.shape) != (batch_size, classes):
        actual = tuple(logits.shape) if hasattr(logits, "shape") else type(logits).__name__
        raise ValueError(f"{logit_name} must have shape ({batch_size}, {classes}), got {actual}")
    if not logits.is_floating_point():
        raise ValueError(f"{logit_name} must use a floating-point dtype")
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if labels.dtype not in integer_dtypes:
        raise ValueError(f"{label_name} must use an integer dtype")
    if logits.device != labels.device:
        raise ValueError("logits and labels must use the same device")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"{logit_name} must be finite")
    if not bool(((labels >= 0) & (labels < classes)).all().item()):
        raise ValueError(f"{label_name} are out of range")


def _optional_class_weights(
    weights: Any,
    *,
    classes: int,
    logits: Any,
    labels: Any,
    name: str,
) -> Any:
    if weights is None:
        return None
    value = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    if tuple(value.shape) != (classes,):
        raise ValueError(f"{name} must contain exactly {classes} values")
    if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
        raise ValueError(f"{name} must contain finite non-negative values")
    if bool((value[labels.long()] == 0).any().item()):
        raise ValueError(f"an observed class has zero weight in {name}")
    return value


def compute_relevance_loss(
    *,
    relevance_logits: Any,
    relevance_labels: Any,
    class_weights: Any = None,
) -> Any:
    """Compute the relevance model's exact weighted cross-entropy loss."""

    _require_torch()
    _validate_logits_and_labels(
        logits=relevance_logits,
        labels=relevance_labels,
        classes=len(RELEVANCE_LABELS),
        logit_name="relevance_logits",
        label_name="relevance_labels",
    )
    weights = _optional_class_weights(
        class_weights,
        classes=len(RELEVANCE_LABELS),
        logits=relevance_logits,
        labels=relevance_labels,
        name="relevance_class_weights",
    )
    return torch_functional.cross_entropy(
        relevance_logits,
        relevance_labels.long(),
        weight=weights,
        reduction="mean",
    )


def compute_target_conditioned_loss(
    *,
    target_state_logits: Any,
    target_state_labels: Any,
    class_weights: Any = None,
) -> Any:
    """Compute the target-conditioned model's exact weighted cross-entropy loss."""

    _require_torch()
    _validate_logits_and_labels(
        logits=target_state_logits,
        labels=target_state_labels,
        classes=len(TARGET_STATE_LABELS),
        logit_name="target_state_logits",
        label_name="target_state_labels",
    )
    weights = _optional_class_weights(
        class_weights,
        classes=len(TARGET_STATE_LABELS),
        logits=target_state_logits,
        labels=target_state_labels,
        name="target_state_class_weights",
    )
    return torch_functional.cross_entropy(
        target_state_logits,
        target_state_labels.long(),
        weight=weights,
        reduction="mean",
    )


def _hidden_size(encoder: Any, hidden_size: int | None) -> int:
    if hidden_size is None:
        hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer or available on encoder.config")
    return hidden_size


def _dropout(dropout: float) -> float:
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0.0 <= dropout < 1.0
    ):
        raise ValueError("dropout must be finite and within [0, 1)")
    return float(dropout)


if nn is not None:  # pragma: no branch - class definitions require optional Torch

    class ModernBertRelevanceModel(nn.Module):
        """Independent ModernBERT encoder with one three-way relevance head."""

        def __init__(self, encoder: Any, *, hidden_size: int | None = None, dropout: float = 0.1):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(_dropout(dropout))
            self.relevance_head = nn.Linear(
                _hidden_size(encoder, hidden_size),
                len(RELEVANCE_LABELS),
            )

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            relevance_labels: Any = None,
            relevance_class_weights: Any = None,
        ) -> dict[str, Any]:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if not hasattr(encoded, "last_hidden_state"):
                raise ValueError("encoder output must expose last_hidden_state")
            pooled = self.dropout(masked_mean_pool(encoded.last_hidden_state, attention_mask))
            logits = self.relevance_head(pooled)
            result = {"relevance_logits": logits}
            if relevance_labels is not None:
                result["loss"] = compute_relevance_loss(
                    relevance_logits=logits,
                    relevance_labels=relevance_labels,
                    class_weights=relevance_class_weights,
                )
            return result


    class ModernBertTargetConditionedModel(nn.Module):
        """Independent ModernBERT encoder with one shared six-way target-state head."""

        def __init__(self, encoder: Any, *, hidden_size: int | None = None, dropout: float = 0.1):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(_dropout(dropout))
            self.target_state_head = nn.Linear(
                _hidden_size(encoder, hidden_size),
                len(TARGET_STATE_LABELS),
            )

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            target_state_labels: Any = None,
            target_state_class_weights: Any = None,
        ) -> dict[str, Any]:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if not hasattr(encoded, "last_hidden_state"):
                raise ValueError("encoder output must expose last_hidden_state")
            pooled = self.dropout(masked_mean_pool(encoded.last_hidden_state, attention_mask))
            logits = self.target_state_head(pooled)
            result = {"target_state_logits": logits}
            if target_state_labels is not None:
                result["loss"] = compute_target_conditioned_loss(
                    target_state_logits=logits,
                    target_state_labels=target_state_labels,
                    class_weights=target_state_class_weights,
                )
            return result


else:

    class ModernBertRelevanceModel:  # pragma: no cover - dependency guard
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()


    class ModernBertTargetConditionedModel:  # pragma: no cover - dependency guard
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()


__all__ = [
    "ABSENT_TARGET_STATE_INDEX",
    "TARGET_STATE_LABELS",
    "CascadeDecodedBatch",
    "ModernBertRelevanceModel",
    "ModernBertTargetConditionedModel",
    "compute_relevance_loss",
    "compute_target_conditioned_loss",
    "decode_cascade_logits",
    "encode_relevance_semantic_label",
    "encode_target_conditioned_semantic_label",
    "expand_target_conditioned_labels",
    "render_target_conditioned_text",
]
