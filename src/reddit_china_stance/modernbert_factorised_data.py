"""Strict data contract for the factorised ModernBERT-v2 students.

This module deliberately has no Torch or Transformers dependency.  It converts one
validated ontology-v2 label into fixed-width, collator-friendly integer arrays and
renders bounded target-first text.  ``not_codable`` is a mask, not a class; binary
relevance has no ``unclear`` value; and downstream losses are masked for non-material
rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from reddit_china_stance.semantic_ontology_v2 import (
    ANALYTIC_TARGETS,
    TARGETS,
    validate_v2_label,
)

DATA_CONTRACT_VERSION = "modernbert-factorised-data-v1"
IGNORE_INDEX = -100

RELEVANCE_CLASSES = ("not_material", "material")
TARGET_CLASSES = TARGETS
ANALYTIC_TARGET_CLASSES = ANALYTIC_TARGETS

# Match the existing ModernBERT categorical stance ordering.  The ontology's display
# order is deliberately not used as an implicit model index.
STANCE_CLASSES_B4 = ("negative", "mixed", "no_directed_stance", "positive")
STANCE_TO_B2 = {
    "negative": (1, 0),
    "positive": (0, 1),
    "mixed": (1, 1),
    "no_directed_stance": (0, 0),
}
B2_TO_STANCE = {bits: stance for stance, bits in STANCE_TO_B2.items()}


def canonical_sha256(value: Any) -> str:
    """Hash one finite JSON-compatible value under the repository convention."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("contract value must be finite JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_binary_int(value: Any) -> bool:
    return type(value) is int and value in {0, 1}


def _require_binary_vector(value: Sequence[int], *, size: int, where: str) -> None:
    if type(value) is not tuple or len(value) != size or any(
        not _is_binary_int(item) for item in value
    ):
        raise ValueError(f"{where} must contain exactly {size} binary integers")


@dataclass(frozen=True, slots=True)
class TextRenderConfig:
    """Bounded, target-first text rendering contract.

    The segment caps prevent one context field from consuming the entire input before
    tokenisation.  The total cap is a second hard bound that includes separators.
    """

    max_target_chars: int = 6_144
    max_parent_chars: int = 3_072
    max_submission_chars: int = 3_072
    max_total_chars: int = 12_320

    def __post_init__(self) -> None:
        values = (
            self.max_target_chars,
            self.max_parent_chars,
            self.max_submission_chars,
            self.max_total_chars,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("text-render bounds must be positive integers")
        if self.max_total_chars > 32_768:
            raise ValueError("max_total_chars exceeds the registered safety bound")
        if max(values[:3]) > self.max_total_chars:
            raise ValueError("a segment bound cannot exceed max_total_chars")

    def digest(self) -> str:
        return canonical_sha256(
            {"contract": DATA_CONTRACT_VERSION, "text_render": asdict(self)}
        )


@dataclass(frozen=True, slots=True)
class FactorisedLabelEncoding:
    """Fixed-width integer labels and masks consumed by the v2 trainers."""

    codability_mask: int
    relevance_label: int
    relevance_mask: int
    target_presence_labels: tuple[int, ...]
    target_presence_mask: tuple[int, ...]
    stance_b4_labels: tuple[int, ...]
    stance_b2_labels: tuple[tuple[int, int], ...]
    stance_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_binary_int(self.codability_mask):
            raise ValueError("codability_mask must be a binary integer")
        if not _is_binary_int(self.relevance_mask):
            raise ValueError("relevance_mask must be a binary integer")
        if self.relevance_mask != self.codability_mask:
            raise ValueError("relevance_mask must equal codability_mask")
        if self.relevance_mask:
            if type(self.relevance_label) is not int or not 0 <= self.relevance_label < 2:
                raise ValueError("an unmasked relevance label must be binary")
        elif self.relevance_label != IGNORE_INDEX:
            raise ValueError("a masked relevance label must use IGNORE_INDEX")

        _require_binary_vector(
            self.target_presence_labels,
            size=len(TARGET_CLASSES),
            where="target_presence_labels",
        )
        _require_binary_vector(
            self.target_presence_mask,
            size=len(TARGET_CLASSES),
            where="target_presence_mask",
        )
        _require_binary_vector(
            self.stance_mask,
            size=len(ANALYTIC_TARGET_CLASSES),
            where="stance_mask",
        )
        if (
            type(self.stance_b4_labels) is not tuple
            or len(self.stance_b4_labels) != len(ANALYTIC_TARGET_CLASSES)
        ):
            raise ValueError("stance_b4_labels has the wrong width")
        if (
            type(self.stance_b2_labels) is not tuple
            or len(self.stance_b2_labels) != len(ANALYTIC_TARGET_CLASSES)
            or any(type(bits) is not tuple for bits in self.stance_b2_labels)
        ):
            raise ValueError("stance_b2_labels has the wrong width")

        material = self.relevance_mask == 1 and self.relevance_label == 1
        expected_target_mask = (1,) * len(TARGET_CLASSES) if material else (0,) * len(
            TARGET_CLASSES
        )
        if self.target_presence_mask != expected_target_mask:
            raise ValueError("target presence must be masked exactly outside material rows")
        if not material and any(self.target_presence_labels):
            raise ValueError("a masked downstream row cannot contain positive target labels")
        if material and not any(self.target_presence_labels):
            raise ValueError("a reference material row must contain at least one target")

        expected_stance_mask = self.target_presence_labels[: len(ANALYTIC_TARGET_CLASSES)]
        if self.stance_mask != expected_stance_mask:
            raise ValueError("stance_mask must equal analytical reference-target presence")
        for index, mask in enumerate(self.stance_mask):
            b4 = self.stance_b4_labels[index]
            b2 = self.stance_b2_labels[index]
            if mask:
                if type(b4) is not int or not 0 <= b4 < len(STANCE_CLASSES_B4):
                    raise ValueError("an unmasked B4 stance label is invalid")
                if (
                    len(b2) != 2
                    or any(not _is_binary_int(bit) for bit in b2)
                    or B2_TO_STANCE.get(tuple(b2)) != STANCE_CLASSES_B4[b4]
                ):
                    raise ValueError("B2 and B4 stance labels disagree")
            elif b4 != IGNORE_INDEX or tuple(b2) != (IGNORE_INDEX, IGNORE_INDEX):
                raise ValueError("masked stance labels must use IGNORE_INDEX")

    def as_feature(self) -> dict[str, Any]:
        """Return list-backed fields suitable for a dynamic-padding collator."""

        return {
            "codability_mask": self.codability_mask,
            "relevance_labels": self.relevance_label,
            "relevance_mask": self.relevance_mask,
            "target_presence_labels": list(self.target_presence_labels),
            "target_presence_mask": list(self.target_presence_mask),
            "stance_b4_labels": list(self.stance_b4_labels),
            "stance_b2_labels": [list(bits) for bits in self.stance_b2_labels],
            "stance_mask": list(self.stance_mask),
        }


@dataclass(frozen=True, slots=True)
class FactorisedRecord:
    """Private, collator-friendly record with bounded rendered text."""

    item_id: str
    text: str
    encoding: FactorisedLabelEncoding
    text_contract_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("item_id must be a non-empty private identifier")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("text must be non-empty")
        if not isinstance(self.encoding, FactorisedLabelEncoding):
            raise ValueError("encoding must be a FactorisedLabelEncoding")
        if (
            not isinstance(self.text_contract_digest, str)
            or len(self.text_contract_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.text_contract_digest)
        ):
            raise ValueError("text_contract_digest must be a lowercase SHA-256")

    def as_feature(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "text_contract_digest": self.text_contract_digest,
            **self.encoding.as_feature(),
        }


def encode_v2_label(label: Mapping[str, Any]) -> FactorisedLabelEncoding:
    """Encode one strict ontology-v2 label without inventing an unclear class."""

    clean = validate_v2_label(label)
    codable = clean["codability"] == "codable"
    material = codable and clean["relevance"] == "material"

    target_presence = [0] * len(TARGET_CLASSES)
    stance_b4 = [IGNORE_INDEX] * len(ANALYTIC_TARGET_CLASSES)
    stance_b2 = [(IGNORE_INDEX, IGNORE_INDEX)] * len(ANALYTIC_TARGET_CLASSES)
    stance_mask = [0] * len(ANALYTIC_TARGET_CLASSES)

    for item in clean["targets"]:
        target_index = TARGET_CLASSES.index(item["target"])
        target_presence[target_index] = 1
        if target_index >= len(ANALYTIC_TARGET_CLASSES):
            continue
        stance = item["stance"]
        stance_index = STANCE_CLASSES_B4.index(stance)
        stance_b4[target_index] = stance_index
        stance_b2[target_index] = STANCE_TO_B2[stance]
        stance_mask[target_index] = 1

    return FactorisedLabelEncoding(
        codability_mask=int(codable),
        relevance_label=(RELEVANCE_CLASSES.index(clean["relevance"]) if codable else IGNORE_INDEX),
        relevance_mask=int(codable),
        target_presence_labels=tuple(target_presence),
        target_presence_mask=(
            (1,) * len(TARGET_CLASSES) if material else (0,) * len(TARGET_CLASSES)
        ),
        stance_b4_labels=tuple(stance_b4),
        stance_b2_labels=tuple(stance_b2),
        stance_mask=tuple(stance_mask),
    )


def _clean_segment(row: Mapping[str, Any], field: str, *, required: bool) -> str:
    if field not in row:
        raise ValueError(f"row is missing required text field {field}")
    value = row[field]
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string" + (" or null" if not required else ""))
    clean = value.strip()
    if required and not clean:
        raise ValueError(f"{field} must be non-empty")
    return clean


def render_factorised_text(
    row: Mapping[str, Any],
    *,
    separator: str,
    config: TextRenderConfig | None = None,
) -> str:
    """Render target, parent and submission text under exact deterministic bounds."""

    active = config or TextRenderConfig()
    if not isinstance(separator, str) or not separator or len(separator) > 64:
        raise ValueError("separator must be a non-empty string of at most 64 characters")
    delimiter = f"\n{separator}\n"
    overhead = 2 * len(delimiter)
    if active.max_total_chars <= overhead:
        raise ValueError("max_total_chars cannot fit the registered separators")

    segments = (
        _clean_segment(row, "target_text", required=True)[: active.max_target_chars],
        _clean_segment(row, "parent_context", required=False)[: active.max_parent_chars],
        _clean_segment(row, "submission_context", required=False)[
            : active.max_submission_chars
        ],
    )
    payload_budget = active.max_total_chars - overhead
    bounded: list[str] = []
    for segment in segments:
        bounded.append(segment[:payload_budget])
        payload_budget -= len(bounded[-1])
    rendered = delimiter.join(bounded)
    if not rendered or len(rendered) > active.max_total_chars:
        raise RuntimeError("bounded text renderer violated its output contract")
    return rendered


def build_factorised_record(
    *,
    item_id: str,
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    separator: str,
    config: TextRenderConfig | None = None,
) -> FactorisedRecord:
    """Build one private typed record for later tokenisation and collation."""

    active = config or TextRenderConfig()
    return FactorisedRecord(
        item_id=item_id,
        text=render_factorised_text(row, separator=separator, config=active),
        encoding=encode_v2_label(label),
        text_contract_digest=active.digest(),
    )
