"""Factorised ModernBERT v2 model and masked training losses.

The v2 measurement model deliberately separates two encoders:

* a binary relevance encoder trained only on codable rows; and
* a target/stance encoder with six independent target-presence decisions and
  stance predictions for the five analytic targets.

``not_codable`` is a mask, never a learned ontology class.  Probability
calibration and decoding belong exclusively to :mod:`semantic_evaluation_v2`.
PyTorch and Transformers remain optional dependencies so the metadata-only
package can import this module without the ML environment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reddit_china_stance.modernbert_model import masked_mean_pool
from reddit_china_stance.semantic_ontology_v2 import ANALYTIC_TARGETS, STANCES, TARGETS

try:  # pragma: no cover - exercised in the optional ML environment
    import torch
    import torch.nn.functional as torch_functional
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - normal for the base environment
    torch = None
    torch_functional = None
    nn = None

STANCE_VARIANTS = ("b4", "b2")
POLARITY_BITS = ("negative", "positive")
B4_STANCE_LABELS = ("negative", "mixed", "no_directed_stance", "positive")
if frozenset(B4_STANCE_LABELS) != frozenset(STANCES):  # pragma: no cover - import invariant
    raise RuntimeError("frozen B4 stance order does not match the v2 ontology")
B2_STATE_TO_BITS: Mapping[str, tuple[int, int]] = {
    "negative": (1, 0),
    "positive": (0, 1),
    "mixed": (1, 1),
    "no_directed_stance": (0, 0),
}


@dataclass(frozen=True, slots=True)
class FactorisedHeadConfig:
    """Frozen output-head choice for one target/stance model."""

    stance_variant: str
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.stance_variant not in STANCE_VARIANTS:
            raise ValueError(f"stance_variant must be one of {STANCE_VARIANTS}")
        _validate_dropout(self.dropout)


def _require_torch() -> None:
    if torch is None or nn is None or torch_functional is None:
        raise ModuleNotFoundError(
            "factorised ModernBERT execution requires the optional ML dependencies "
            "torch and transformers"
        )


def _validate_dropout(dropout: float) -> float:
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0.0 <= dropout < 1.0
    ):
        raise ValueError("dropout must be finite and within [0, 1)")
    return float(dropout)


def _hidden_size(encoder: Any, hidden_size: int | None) -> int:
    if hidden_size is None:
        hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer or available on encoder.config")
    return hidden_size


def b2_bits_for_stance(stance: str) -> tuple[int, int]:
    """Encode one four-state stance as negative/positive evidence bits."""

    if stance not in B2_STATE_TO_BITS:
        raise ValueError(f"stance must be one of {STANCES}")
    return B2_STATE_TO_BITS[stance]


def b2_stance_from_bits(bits: Sequence[int | bool]) -> str:
    """Map exact negative/positive evidence bits to a four-state stance."""

    values = tuple(bits)
    if len(values) != len(POLARITY_BITS) or any(
        isinstance(value, float) or value not in (0, 1, False, True) for value in values
    ):
        raise ValueError("B2 state must contain exactly two binary values")
    canonical = (int(values[0]), int(values[1]))
    return next(stance for stance, encoded in B2_STATE_TO_BITS.items() if encoded == canonical)


def _validate_float_tensor(value: Any, *, shape: tuple[int, ...], name: str) -> None:
    _require_torch()
    if not hasattr(value, "shape") or tuple(value.shape) != shape:
        actual = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} must have shape {shape}, got {actual}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_boolean_mask(value: Any, *, shape: tuple[int, ...], name: str, device: Any) -> None:
    if not hasattr(value, "shape") or tuple(value.shape) != shape:
        actual = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} must have shape {shape}, got {actual}")
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must use the boolean dtype")
    if value.device != device:
        raise ValueError(f"{name} must use the logits device")


def _validate_binary_labels(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
    device: Any,
    selected_mask: Any | None = None,
) -> None:
    if not hasattr(value, "shape") or tuple(value.shape) != shape:
        actual = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} must have shape {shape}, got {actual}")
    if value.device != device:
        raise ValueError(f"{name} must use the logits device")
    selected = value if selected_mask is None else value[selected_mask]
    if selected.numel() and not bool(((selected == 0) | (selected == 1)).all().item()):
        raise ValueError(f"selected {name} must be binary")


def _differentiable_zero(logits: Any) -> Any:
    return logits.sum() * 0.0


def compute_masked_relevance_loss(
    *,
    relevance_logits: Any,
    relevance_labels: Any,
    codable_mask: Any,
) -> Any:
    """Standard unweighted binary cross entropy over codable rows only."""

    _require_torch()
    if not hasattr(relevance_logits, "shape") or relevance_logits.ndim != 1:
        raise ValueError("relevance_logits must be a non-empty rank-one tensor")
    batch_size = int(relevance_logits.shape[0])
    if batch_size <= 0:
        raise ValueError("relevance_logits must be a non-empty rank-one tensor")
    _validate_float_tensor(
        relevance_logits,
        shape=(batch_size,),
        name="relevance_logits",
    )
    _validate_boolean_mask(
        codable_mask,
        shape=(batch_size,),
        name="codable_mask",
        device=relevance_logits.device,
    )
    _validate_binary_labels(
        relevance_labels,
        shape=(batch_size,),
        name="relevance_labels",
        device=relevance_logits.device,
        selected_mask=codable_mask,
    )
    if not bool(codable_mask.any().item()):
        return _differentiable_zero(relevance_logits)
    return torch_functional.binary_cross_entropy_with_logits(
        relevance_logits[codable_mask],
        relevance_labels[codable_mask].to(dtype=relevance_logits.dtype),
        reduction="mean",
    )


def compute_masked_target_presence_loss(
    *,
    target_presence_logits: Any,
    target_presence_labels: Any,
    reference_material_mask: Any,
) -> Any:
    """Standard unweighted BCE over all six targets on reference-material rows."""

    _require_torch()
    if not hasattr(target_presence_logits, "shape") or target_presence_logits.ndim != 2:
        raise ValueError("target_presence_logits must be a non-empty rank-two tensor")
    batch_size = int(target_presence_logits.shape[0])
    if batch_size <= 0:
        raise ValueError("target_presence_logits must be a non-empty rank-two tensor")
    shape = (batch_size, len(TARGETS))
    _validate_float_tensor(target_presence_logits, shape=shape, name="target_presence_logits")
    _validate_boolean_mask(
        reference_material_mask,
        shape=(batch_size,),
        name="reference_material_mask",
        device=target_presence_logits.device,
    )
    _validate_binary_labels(
        target_presence_labels,
        shape=shape,
        name="target_presence_labels",
        device=target_presence_logits.device,
    )
    if not bool(reference_material_mask.any().item()):
        return _differentiable_zero(target_presence_logits)
    return torch_functional.binary_cross_entropy_with_logits(
        target_presence_logits[reference_material_mask],
        target_presence_labels[reference_material_mask].to(dtype=target_presence_logits.dtype),
        reduction="mean",
    )


def _effective_stance_mask(
    *,
    stance_known_mask: Any,
    target_presence_labels: Any,
    reference_material_mask: Any,
    batch_size: int,
    device: Any,
) -> Any:
    _validate_boolean_mask(
        stance_known_mask,
        shape=(batch_size, len(ANALYTIC_TARGETS)),
        name="stance_known_mask",
        device=device,
    )
    _validate_boolean_mask(
        reference_material_mask,
        shape=(batch_size,),
        name="reference_material_mask",
        device=device,
    )
    _validate_binary_labels(
        target_presence_labels,
        shape=(batch_size, len(TARGETS)),
        name="target_presence_labels",
        device=device,
    )
    analytic_present = target_presence_labels[:, : len(ANALYTIC_TARGETS)].bool()
    return stance_known_mask & analytic_present & reference_material_mask.unsqueeze(1)


def compute_masked_b4_stance_loss(
    *,
    stance_logits: Any,
    stance_labels: Any,
    stance_known_mask: Any,
    target_presence_labels: Any,
    reference_material_mask: Any,
) -> Any:
    """Standard unweighted four-way CE over known-present analytic targets."""

    _require_torch()
    if not hasattr(stance_logits, "shape") or stance_logits.ndim != 3:
        raise ValueError("B4 stance_logits must be a non-empty rank-three tensor")
    batch_size = int(stance_logits.shape[0])
    if batch_size <= 0:
        raise ValueError("B4 stance_logits must be a non-empty rank-three tensor")
    _validate_float_tensor(
        stance_logits,
        shape=(batch_size, len(ANALYTIC_TARGETS), len(B4_STANCE_LABELS)),
        name="B4 stance_logits",
    )
    effective = _effective_stance_mask(
        stance_known_mask=stance_known_mask,
        target_presence_labels=target_presence_labels,
        reference_material_mask=reference_material_mask,
        batch_size=batch_size,
        device=stance_logits.device,
    )
    if not hasattr(stance_labels, "shape") or tuple(stance_labels.shape) != (
        batch_size,
        len(ANALYTIC_TARGETS),
    ):
        actual = (
            tuple(stance_labels.shape)
            if hasattr(stance_labels, "shape")
            else type(stance_labels).__name__
        )
        raise ValueError(
            "B4 stance_labels must have shape "
            f"{(batch_size, len(ANALYTIC_TARGETS))}, got {actual}"
        )
    if stance_labels.device != stance_logits.device:
        raise ValueError("B4 stance_labels must use the logits device")
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if stance_labels.dtype not in integer_dtypes:
        raise ValueError("B4 stance_labels must use an integer dtype")
    if effective.any() and not bool(
        (
            (stance_labels[effective] >= 0)
            & (stance_labels[effective] < len(B4_STANCE_LABELS))
        )
        .all()
        .item()
    ):
        raise ValueError("selected B4 stance_labels are out of range")
    if not bool(effective.any().item()):
        return _differentiable_zero(stance_logits)
    return torch_functional.cross_entropy(
        stance_logits[effective],
        stance_labels[effective].long(),
        reduction="mean",
    )


def compute_masked_b2_stance_loss(
    *,
    stance_logits: Any,
    stance_labels: Any,
    stance_known_mask: Any,
    target_presence_labels: Any,
    reference_material_mask: Any,
) -> Any:
    """Standard unweighted BCE over polarity bits for known-present targets."""

    _require_torch()
    if not hasattr(stance_logits, "shape") or stance_logits.ndim != 3:
        raise ValueError("B2 stance_logits must be a non-empty rank-three tensor")
    batch_size = int(stance_logits.shape[0])
    if batch_size <= 0:
        raise ValueError("B2 stance_logits must be a non-empty rank-three tensor")
    shape = (batch_size, len(ANALYTIC_TARGETS), len(POLARITY_BITS))
    _validate_float_tensor(stance_logits, shape=shape, name="B2 stance_logits")
    effective = _effective_stance_mask(
        stance_known_mask=stance_known_mask,
        target_presence_labels=target_presence_labels,
        reference_material_mask=reference_material_mask,
        batch_size=batch_size,
        device=stance_logits.device,
    )
    _validate_binary_labels(
        stance_labels,
        shape=shape,
        name="B2 stance_labels",
        device=stance_logits.device,
        selected_mask=effective,
    )
    if not bool(effective.any().item()):
        return _differentiable_zero(stance_logits)
    return torch_functional.binary_cross_entropy_with_logits(
        stance_logits[effective],
        stance_labels[effective].to(dtype=stance_logits.dtype),
        reduction="mean",
    )


def compute_factorised_target_stance_losses(
    *,
    target_presence_logits: Any,
    stance_logits: Any,
    target_presence_labels: Any,
    stance_labels: Any,
    reference_material_mask: Any,
    stance_known_mask: Any,
    stance_variant: str,
) -> dict[str, Any]:
    """Compute the fixed, unweighted target plus stance objective."""

    if stance_variant not in STANCE_VARIANTS:
        raise ValueError(f"stance_variant must be one of {STANCE_VARIANTS}")
    target_loss = compute_masked_target_presence_loss(
        target_presence_logits=target_presence_logits,
        target_presence_labels=target_presence_labels,
        reference_material_mask=reference_material_mask,
    )
    stance_loss_function = (
        compute_masked_b4_stance_loss
        if stance_variant == "b4"
        else compute_masked_b2_stance_loss
    )
    stance_loss = stance_loss_function(
        stance_logits=stance_logits,
        stance_labels=stance_labels,
        stance_known_mask=stance_known_mask,
        target_presence_labels=target_presence_labels,
        reference_material_mask=reference_material_mask,
    )
    return {
        "loss": target_loss + stance_loss,
        "target_presence_loss": target_loss,
        "stance_loss": stance_loss,
    }


def assert_independent_encoders(
    relevance_model: Any,
    target_stance_model: Any,
) -> None:
    """Fail if the two models share their encoder or any parameter object."""

    relevance_encoder = getattr(relevance_model, "encoder", None)
    target_encoder = getattr(target_stance_model, "encoder", None)
    if relevance_encoder is None or target_encoder is None:
        raise ValueError("both models must expose an encoder")
    if relevance_encoder is target_encoder:
        raise ValueError("relevance and target/stance models must use independent encoders")
    if not hasattr(relevance_model, "parameters") or not hasattr(target_stance_model, "parameters"):
        raise ValueError("both models must expose parameters")
    relevance_parameters = {id(parameter) for parameter in relevance_model.parameters()}
    target_parameters = {id(parameter) for parameter in target_stance_model.parameters()}
    if relevance_parameters & target_parameters:
        raise ValueError("relevance and target/stance models share parameter objects")


if nn is not None:  # pragma: no branch - class definitions require optional Torch

    class ModernBertBinaryRelevanceModel(nn.Module):
        """Independent encoder with one binary material-relevance logit."""

        def __init__(self, encoder: Any, *, hidden_size: int | None = None, dropout: float = 0.1):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(_validate_dropout(dropout))
            self.relevance_head = nn.Linear(_hidden_size(encoder, hidden_size), 1)

        @classmethod
        def from_pretrained(
            cls,
            model_name: str,
            *,
            revision: str,
            dropout: float = 0.1,
            **encoder_kwargs: Any,
        ) -> ModernBertBinaryRelevanceModel:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name is required")
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("exact revision is required")
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
            codable_mask: Any = None,
        ) -> dict[str, Any]:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if not hasattr(encoded, "last_hidden_state"):
                raise ValueError("encoder output must expose last_hidden_state")
            pooled = self.dropout(masked_mean_pool(encoded.last_hidden_state, attention_mask))
            logits = self.relevance_head(pooled).squeeze(-1)
            result = {"relevance_logits": logits}
            supplied = (relevance_labels is not None, codable_mask is not None)
            if any(supplied) and not all(supplied):
                raise ValueError("relevance_labels and codable_mask must be supplied together")
            if all(supplied):
                result["loss"] = compute_masked_relevance_loss(
                    relevance_logits=logits,
                    relevance_labels=relevance_labels,
                    codable_mask=codable_mask,
                )
            return result


    class ModernBertFactorisedTargetStanceModel(nn.Module):
        """Independent encoder with target presence and selectable B4/B2 stance heads."""

        def __init__(
            self,
            encoder: Any,
            *,
            config: FactorisedHeadConfig,
            hidden_size: int | None = None,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.config = config
            self.dropout = nn.Dropout(_validate_dropout(config.dropout))
            width = _hidden_size(encoder, hidden_size)
            self.target_presence_head = nn.Linear(width, len(TARGETS))
            stance_width = (
                len(B4_STANCE_LABELS)
                if config.stance_variant == "b4"
                else len(POLARITY_BITS)
            )
            self.stance_head = nn.Linear(
                width,
                len(ANALYTIC_TARGETS) * stance_width,
            )

        @classmethod
        def from_pretrained(
            cls,
            model_name: str,
            *,
            revision: str,
            config: FactorisedHeadConfig,
            **encoder_kwargs: Any,
        ) -> ModernBertFactorisedTargetStanceModel:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name is required")
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("exact revision is required")
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
            return cls(encoder, config=config)

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            target_presence_labels: Any = None,
            stance_labels: Any = None,
            reference_material_mask: Any = None,
            stance_known_mask: Any = None,
        ) -> dict[str, Any]:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if not hasattr(encoded, "last_hidden_state"):
                raise ValueError("encoder output must expose last_hidden_state")
            pooled = self.dropout(masked_mean_pool(encoded.last_hidden_state, attention_mask))
            target_logits = self.target_presence_head(pooled)
            stance_width = (
                len(B4_STANCE_LABELS)
                if self.config.stance_variant == "b4"
                else len(POLARITY_BITS)
            )
            stance_logits = self.stance_head(pooled).reshape(
                pooled.shape[0],
                len(ANALYTIC_TARGETS),
                stance_width,
            )
            result = {
                "target_presence_logits": target_logits,
                "stance_logits": stance_logits,
            }
            supplied = tuple(
                value is not None
                for value in (
                    target_presence_labels,
                    stance_labels,
                    reference_material_mask,
                    stance_known_mask,
                )
            )
            if any(supplied) and not all(supplied):
                raise ValueError("all target/stance label and mask tensors are required for loss")
            if all(supplied):
                result.update(
                    compute_factorised_target_stance_losses(
                        target_presence_logits=target_logits,
                        stance_logits=stance_logits,
                        target_presence_labels=target_presence_labels,
                        stance_labels=stance_labels,
                        reference_material_mask=reference_material_mask,
                        stance_known_mask=stance_known_mask,
                        stance_variant=self.config.stance_variant,
                    )
                )
            return result


else:

    class ModernBertBinaryRelevanceModel:  # pragma: no cover - dependency guard
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

        @classmethod
        def from_pretrained(
            cls, *_args: Any, **_kwargs: Any
        ) -> ModernBertBinaryRelevanceModel:
            _require_torch()
            raise AssertionError("unreachable")


    class ModernBertFactorisedTargetStanceModel:  # pragma: no cover - dependency guard
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

        @classmethod
        def from_pretrained(
            cls, *_args: Any, **_kwargs: Any
        ) -> ModernBertFactorisedTargetStanceModel:
            _require_torch()
            raise AssertionError("unreachable")


__all__ = [
    "ANALYTIC_TARGETS",
    "B2_STATE_TO_BITS",
    "B4_STANCE_LABELS",
    "POLARITY_BITS",
    "STANCES",
    "STANCE_VARIANTS",
    "TARGETS",
    "FactorisedHeadConfig",
    "ModernBertBinaryRelevanceModel",
    "ModernBertFactorisedTargetStanceModel",
    "assert_independent_encoders",
    "b2_bits_for_stance",
    "b2_stance_from_bits",
    "compute_factorised_target_stance_losses",
    "compute_masked_b2_stance_loss",
    "compute_masked_b4_stance_loss",
    "compute_masked_relevance_loss",
    "compute_masked_target_presence_loss",
]
