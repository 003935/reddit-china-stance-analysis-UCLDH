"""Training and private-runtime primitives for factorised ModernBERT v2.

The relevance and target/stance components are instantiated independently.  Public
outputs are metadata-only receipts; private development logits remain inside the
registered local/Modal data root.  The only authoritative scorer is
``semantic_evaluation_v2``.  This module has no locked-test or corpus-inference path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from reddit_china_stance.modernbert_factorised_data import (
    ANALYTIC_TARGET_CLASSES,
    STANCE_CLASSES_B4,
    TARGET_CLASSES,
    TextRenderConfig,
    build_factorised_record,
)
from reddit_china_stance.modernbert_factorised_experiment import (
    COMPONENTS,
    MODEL_ID,
    MODEL_REVISION,
    NAMESPACE,
    TOKENIZER_REVISION,
    FactorisedExperimentContractError,
    artifact_descriptor,
    build_trial_receipt,
    canonical_sha256,
    evaluate_representation_gate,
    file_sha256,
    validate_run_manifest,
    validate_trial_receipt,
    validate_trial_spec,
)
from reddit_china_stance.modernbert_factorised_model import (
    FactorisedHeadConfig,
    ModernBertBinaryRelevanceModel,
    ModernBertFactorisedTargetStanceModel,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.semantic_evaluation_v2 import (
    NATURAL_DESIGN_DIGEST_FIELDS,
    DecodeThresholds,
    NaturalArmWeightingConfig,
    decode_factorised_probabilities,
    natural_arm_weighting_config_from_summary,
    score_factorised_predictions,
    score_natural_probability_arm,
)
from reddit_china_stance.semantic_ontology_v2 import (
    load_v2_label_validator,
    validate_v2_label,
)

Component = Literal["relevance", "target_stance_b4", "target_stance_b2"]
MAX_LENGTH = 768
PRIVATE_PREDICTIONS_KIND = "modernbert-factorised-private-development-logits-v2"
TRIAL_METRICS_KIND = "modernbert-factorised-trial-metrics-v2"
TRIAL_ATTEMPT_KIND = "modernbert-factorised-trial-attempt-v1"
EXPOSURE_REGISTER_KIND = "modernbert-factorised-exposure-register-v1"
LEGACY_OVERLAP_AUDIT_KIND = "modernbert-factorised-legacy-overlap-audit-v1"
EXPECTED_SOURCE_ROWS = 10_000
EXPOSURE_REGISTER_COUNTS = {"bridge": 480}
ACCEPTED_BRIDGE_PACKET_ID = (
    "aba1d87f00a2174160e27839c970509970863357aaf7a7438f969b2a3942452c"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PRIVATE_PREPARATION_ROOT = REPO_ROOT / "data/private-modernbert-factorised-v2"
LOCAL_PUBLIC_PREPARATION_ROOT = REPO_ROOT / "outputs/modernbert-factorised-v2"
VOLUME_PREPARATION_ROOT_NAME = "student-modernbert-factorised-v2"
SOURCE_ROW_HASH_FIELDS = (
    "sample_id",
    "thread_id",
    "target_text",
    "submission_context",
    "parent_context",
    "subreddit",
    "year",
    "content_type",
    "retrieval_mode",
)


def _component(value: str) -> Component:
    if value not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    return value  # type: ignore[return-value]


def _sha(value: Any, *, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def source_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash the exact canonical source payload used by Phase 1 training."""

    if set(row) != set(SOURCE_ROW_HASH_FIELDS):
        raise ValueError("source-row digest payload schema drifted")
    return canonical_sha256({field: row[field] for field in SOURCE_ROW_HASH_FIELDS})


def label_sha256(
    label: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
    validator: Any | None = None,
) -> str:
    """Hash one strict ontology-v2 label after canonical validation."""

    return canonical_sha256(
        validate_v2_label(label, schema=schema, validator=validator)
    )


def _prepared_frame_descriptors(
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = ("training", "development", "calibration")
    if set(descriptors) != set(expected):
        raise FactorisedExperimentContractError(
            "prepared frame descriptor inventory drifted"
        )
    return {frame: dict(descriptors[frame]) for frame in expected}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Modal-only dependency path
        raise RuntimeError("factorised ModernBERT training requires PyTorch") from exc
    return torch


def _experiment_module() -> Any:
    return importlib.import_module("reddit_china_stance.modernbert_factorised_experiment")


@dataclass(frozen=True, slots=True)
class FactorisedOptimisationConfig:
    """The sole approved optimisation family for all three components."""

    encoder_learning_rate: float = 5e-5
    head_learning_rate_multiplier: float = 5.0
    dropout: float = 0.1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.06
    gradient_clip_norm: float = 1.0
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 32
    use_bf16: bool = True

    def __post_init__(self) -> None:
        for name in (
            "encoder_learning_rate",
            "head_learning_rate_multiplier",
            "adam_epsilon",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not (0 < self.adam_beta1 < 1 and 0 < self.adam_beta2 < 1):
            raise ValueError("Adam betas must be in (0, 1)")
        for name in (
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.per_device_batch_size * self.gradient_accumulation_steps != (
            self.effective_batch_size
        ):
            raise ValueError("microbatch and accumulation must equal effective batch 32")

    def digest(self) -> str:
        return canonical_sha256(asdict(self))


def build_optimisation_config(trial_spec: Mapping[str, Any]) -> FactorisedOptimisationConfig:
    component = _component(str(trial_spec.get("component")))
    config = trial_spec.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("trial spec lacks a component config")
    expected = _experiment_module().frozen_component_config(component)
    if dict(config) != expected:
        raise FactorisedExperimentContractError("trial optimiser configuration drifted")
    return FactorisedOptimisationConfig(
        encoder_learning_rate=config["encoder_learning_rate"],
        head_learning_rate_multiplier=config["head_learning_rate_multiplier"],
        dropout=config["dropout"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_betas"][0],
        adam_beta2=config["adam_betas"][1],
        adam_epsilon=config["adam_epsilon"],
        warmup_ratio=config["warmup_ratio"],
        gradient_clip_norm=config["gradient_clip"],
        per_device_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        effective_batch_size=config["effective_batch_size"],
        use_bf16=config["precision"] == "bf16",
    )


def load_pinned_tokenizer(*, tokenizer_loader: Any | None = None) -> Any:
    if tokenizer_loader is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - Modal-only dependency path
            raise RuntimeError("factorised ModernBERT training requires Transformers") from exc
        tokenizer_loader = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_loader(MODEL_ID, revision=TOKENIZER_REVISION, use_fast=True)
    reported = (
        (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
        or getattr(tokenizer, "_commit_hash", None)
    )
    if reported is not None and reported != TOKENIZER_REVISION:
        raise RuntimeError("loaded tokenizer revision differs from the frozen revision")
    if not getattr(tokenizer, "sep_token", None):
        raise RuntimeError("pinned tokenizer does not expose a separator token")
    return tokenizer


def tokenise_factorised_record(
    tokenizer: Any,
    *,
    item_id: str,
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Create exact labels/masks and target-first tokens for one v2 row."""

    if max_length != MAX_LENGTH:
        raise ValueError(f"max_length must remain frozen at {MAX_LENGTH}")
    record = build_factorised_record(
        item_id=item_id,
        row=row,
        label=label,
        separator=tokenizer.sep_token,
        config=TextRenderConfig(),
    )
    raw = tokenizer(
        record.text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    input_ids = list(raw["input_ids"])
    attention_mask = list(raw.get("attention_mask", [1] * len(input_ids)))
    if not input_ids or len(input_ids) != len(attention_mask):
        raise RuntimeError("tokenizer returned empty or mismatched inputs")
    original_tokens = len(input_ids)
    if original_tokens > max_length:
        input_ids = [*input_ids[: max_length - 1], input_ids[-1]]
        attention_mask = [*attention_mask[: max_length - 1], attention_mask[-1]]
    return {
        "item_id": item_id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_count": original_tokens,
        "truncated_tokens": max(0, original_tokens - max_length),
        **record.encoding.as_feature(),
    }


class FactorisedDynamicPaddingCollator:
    """Pad inputs and expose only the tensors required by one component."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        component: str,
        return_tensors: str | None = "pt",
    ) -> None:
        self.tokenizer = tokenizer
        self.component = _component(component)
        self.return_tensors = return_tensors

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        batch = dict(
            self.tokenizer.pad(
                [
                    {key: row[key] for key in ("input_ids", "attention_mask")}
                    for row in features
                ],
                padding=True,
                return_tensors=self.return_tensors,
            )
        )
        ids = [row.get("item_id") for row in features]
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("every factorised batch row requires a private item_id")
        batch["item_ids"] = ids
        fields = (
            ("relevance_labels", "long"),
            ("codable_mask", "bool"),
        ) if self.component == "relevance" else (
            ("target_presence_labels", "float"),
            ("reference_material_mask", "bool"),
            ("stance_labels", "long" if self.component.endswith("b4") else "float"),
            ("stance_known_mask", "bool"),
        )
        values: dict[str, Any] = {}
        if self.component == "relevance":
            values = {
                "relevance_labels": [row["relevance_labels"] for row in features],
                "codable_mask": [bool(row["codability_mask"]) for row in features],
            }
        else:
            stance_key = "stance_b4_labels" if self.component.endswith("b4") else (
                "stance_b2_labels"
            )
            values = {
                "target_presence_labels": [row["target_presence_labels"] for row in features],
                "reference_material_mask": [
                    bool(row["target_presence_mask"][0]) for row in features
                ],
                "stance_labels": [row[stance_key] for row in features],
                "stance_known_mask": [
                    [bool(value) for value in row["stance_mask"]] for row in features
                ],
            }
        if self.return_tensors == "pt":
            torch = _require_torch()
            dtypes = {"long": torch.long, "float": torch.float32, "bool": torch.bool}
            for field, dtype in fields:
                batch[field] = torch.tensor(values[field], dtype=dtypes[dtype])
        else:
            batch.update(values)
        return batch


def create_component_model(
    *,
    component: str,
    config: FactorisedOptimisationConfig,
    encoder_loader: Any | None = None,
) -> Any:
    """Load one fresh pinned encoder; components never share trainable state."""

    component = _component(component)
    if encoder_loader is None:
        try:
            from transformers import ModernBertModel
        except ImportError as exc:  # pragma: no cover - Modal-only dependency path
            raise RuntimeError("factorised ModernBERT training requires Transformers") from exc
        encoder_loader = ModernBertModel.from_pretrained
    encoder = encoder_loader(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="sdpa",
    )
    reported = getattr(getattr(encoder, "config", None), "_commit_hash", None)
    if reported is not None and reported != MODEL_REVISION:
        raise RuntimeError("loaded ModernBERT revision differs from the frozen revision")
    encoder.config.reference_compile = False
    if component == "relevance":
        return ModernBertBinaryRelevanceModel(encoder, dropout=config.dropout)
    variant = "b4" if component.endswith("b4") else "b2"
    return ModernBertFactorisedTargetStanceModel(
        encoder,
        config=FactorisedHeadConfig(stance_variant=variant, dropout=config.dropout),
    )


def adamw_parameter_groups(
    model: Any, config: FactorisedOptimisationConfig
) -> list[dict[str, Any]]:
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
        lower = name.lower()
        no_decay = (
            name.endswith(".bias")
            or "layernorm" in lower
            or "layer_norm" in lower
            or ".norm." in lower
        )
        grouped[(scope, not no_decay)].append(parameter)
    if not seen:
        raise ValueError("model has no trainable parameters")
    result: list[dict[str, Any]] = []
    for scope in ("encoder", "heads"):
        for decay in (True, False):
            if not grouped[(scope, decay)]:
                continue
            result.append(
                {
                    "params": grouped[(scope, decay)],
                    "lr": config.encoder_learning_rate
                    * (config.head_learning_rate_multiplier if scope == "heads" else 1.0),
                    "weight_decay": config.weight_decay if decay else 0.0,
                    "group_name": f"{scope}_{'decay' if decay else 'no_decay'}",
                }
            )
    return result


def create_adamw(model: Any, config: FactorisedOptimisationConfig) -> Any:
    torch = _require_torch()
    return torch.optim.AdamW(
        adamw_parameter_groups(model, config),
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key != "item_ids"
    }


def train_epoch(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Any,
    *,
    config: FactorisedOptimisationConfig,
    device: str,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Train one component using only its exact unweighted model loss."""

    torch = _require_torch()
    resolved = torch.device(device)
    if config.use_bf16 and resolved.type != "cuda":
        raise ValueError("BF16 execution is registered only for CUDA")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = examples = optimizer_steps = 0
    loss_sum = 0.0
    for batches, batch in enumerate(dataloader, start=1):
        model_batch = _move_batch(batch, resolved)
        batch_size = int(model_batch["input_ids"].shape[0])
        examples += batch_size
        with torch.autocast(
            device_type=resolved.type,
            dtype=torch.bfloat16,
            enabled=config.use_bf16,
        ):
            output = model(**model_batch)
            scaled = output["loss"] / config.gradient_accumulation_steps
        scaled.backward()
        loss_sum += float(output["loss"].detach().float().cpu()) * batch_size
        if batches % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    if not batches:
        raise ValueError("training dataloader yielded no batches")
    if batches % config.gradient_accumulation_steps:
        remainder = batches % config.gradient_accumulation_steps
        factor = config.gradient_accumulation_steps / remainder
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    return {
        "batches": batches,
        "examples": examples,
        "optimizer_steps": optimizer_steps,
        "mean_loss": loss_sum / examples,
    }


def collect_development_logits(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    component: str,
    device: str,
    use_bf16: bool = True,
) -> list[dict[str, Any]]:
    """Collect private raw logits; decoding is deferred to semantic_evaluation_v2."""

    torch = _require_torch()
    component = _component(component)
    resolved = torch.device(device)
    if use_bf16 and resolved.type != "cuda":
        raise ValueError("BF16 execution is registered only for CUDA")
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            item_ids = batch.get("item_ids")
            if not isinstance(item_ids, list):
                raise ValueError("development batches require item_ids")
            model_batch = {
                key: value.to(resolved)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask"}
            }
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model(**model_batch)
            if component == "relevance":
                logits = output["relevance_logits"].detach().float().cpu().tolist()
                rows.extend(
                    {"item_id": item_id, "relevance_logit": logit}
                    for item_id, logit in zip(item_ids, logits, strict=True)
                )
            else:
                targets = output["target_presence_logits"].detach().float().cpu().tolist()
                stances = output["stance_logits"].detach().float().cpu().tolist()
                rows.extend(
                    {
                        "item_id": item_id,
                        "target_presence_logits": target,
                        "stance_logits": stance,
                    }
                    for item_id, target, stance in zip(item_ids, targets, stances, strict=True)
                )
    if not rows or len({row["item_id"] for row in rows}) != len(rows):
        raise RuntimeError("development logits are empty or contain duplicate IDs")
    return sorted(rows, key=lambda row: row["item_id"])


def _finite_vector(value: Any, *, size: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{where} must contain exactly {size} logits")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{where} must contain finite numeric logits")
        result.append(float(item))
    return result


def build_private_development_predictions(
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact private logit envelope for one independent component."""

    clean_trial = validate_trial_spec(trial_spec, experiment=experiment)
    component = clean_trial["component"]
    expected_rows = experiment["bindings"]["development_frame"]["row_count"]
    if len(rows) != expected_rows:
        raise FactorisedExperimentContractError("private predictions do not conserve development")
    ids: set[str] = set()
    clean_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise ValueError(f"private prediction row {index} has an invalid or duplicate ID")
        ids.add(item_id)
        if component == "relevance":
            if set(raw) != {"item_id", "relevance_logit"}:
                raise ValueError("relevance private prediction schema drifted")
            logit = raw["relevance_logit"]
            if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(
                logit
            ):
                raise ValueError("relevance logit must be finite")
            clean_rows.append({"item_id": item_id, "relevance_logit": float(logit)})
        else:
            if set(raw) != {"item_id", "target_presence_logits", "stance_logits"}:
                raise ValueError("target/stance private prediction schema drifted")
            target = _finite_vector(
                raw["target_presence_logits"], size=len(TARGET_CLASSES), where="target logits"
            )
            width = len(STANCE_CLASSES_B4) if component.endswith("b4") else 2
            stance_raw = raw["stance_logits"]
            if not isinstance(stance_raw, list) or len(stance_raw) != len(
                ANALYTIC_TARGET_CLASSES
            ):
                raise ValueError("stance logits have the wrong target width")
            stance = [
                _finite_vector(row, size=width, where="stance target logits")
                for row in stance_raw
            ]
            clean_rows.append(
                {
                    "item_id": item_id,
                    "target_presence_logits": target,
                    "stance_logits": stance,
                }
            )
    body = {
        "schema_version": "1.0.0",
        "kind": PRIVATE_PREDICTIONS_KIND,
        "experiment_run_id": experiment["experiment_run_id"],
        "trial_id": clean_trial["trial_id"],
        "component": component,
        "optimiser_seed": clean_trial["optimiser_seed"],
        "config_sha256": clean_trial["config"]["config_sha256"],
        "development_frame_sha256": clean_trial["development_frame_sha256"],
        "row_count": len(clean_rows),
        "rows": sorted(clean_rows, key=lambda row: row["item_id"]),
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def validate_private_development_predictions(
    value: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
) -> dict[str, Any]:
    clean_trial = validate_trial_spec(trial_spec, experiment=experiment)
    if value.get("trial_id") != clean_trial["trial_id"]:
        raise FactorisedExperimentContractError("private prediction trial binding drifted")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("private prediction payload lacks rows")
    expected = build_private_development_predictions(experiment, clean_trial, rows=rows)
    if dict(value) != expected:
        raise FactorisedExperimentContractError("private prediction payload binding drifted")
    return expected


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1 / (1 + factor)
    factor = math.exp(value)
    return factor / (1 + factor)


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def combine_component_predictions(
    *,
    reference: Mapping[str, Mapping[str, Any]],
    relevance_payload: Mapping[str, Any],
    target_stance_payload: Mapping[str, Any],
    representation: str,
    natural_design: Mapping[str, Mapping[str, Any]] | None = None,
    natural_weighting: NaturalArmWeightingConfig | None = None,
) -> dict[str, Any]:
    """Score one paired seed with the authoritative exact-bound evaluator."""

    if representation not in {"B4", "B2"}:
        raise ValueError("representation must be B4 or B2")
    relevance = {row["item_id"]: row for row in relevance_payload["rows"]}
    target_stance = {row["item_id"]: row for row in target_stance_payload["rows"]}
    if set(reference) != set(relevance) or set(reference) != set(target_stance):
        raise FactorisedExperimentContractError("paired logits do not conserve development IDs")
    thresholds = DecodeThresholds(stance_representation=representation)
    decoded = {}
    for item_id in reference:
        target_row = target_stance[item_id]
        raw_stance = target_row["stance_logits"]
        probabilities = (
            [_softmax(row) for row in raw_stance]
            if representation == "B4"
            else [[_sigmoid(value) for value in row] for row in raw_stance]
        )
        decoded[item_id] = decode_factorised_probabilities(
            relevance_probability=_sigmoid(relevance[item_id]["relevance_logit"]),
            target_probabilities=[
                _sigmoid(value) for value in target_row["target_presence_logits"]
            ],
            stance_probabilities=probabilities,
            thresholds=thresholds,
        )
    if (natural_design is None) is not (natural_weighting is None):
        raise ValueError("natural design and weighting must be supplied together")
    if natural_design is not None and natural_weighting is not None:
        return score_natural_probability_arm(
            reference,
            decoded,
            natural_design,
            thresholds=thresholds,
            weighting=natural_weighting,
        )
    return score_factorised_predictions(reference, decoded, thresholds=thresholds)


def gate_metric_surface(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce the aggregate scorer to the preregistered B2/B4 gate surface."""

    if metrics.get("scientific_aggregate_eligible") is not True:
        raise FactorisedExperimentContractError(
            "representation-gate primary metrics must be design-weighted"
        )
    end = metrics["design_weighted"]["end_to_end"]
    targets = end["target_presence"]
    tuples = end["target_stance_tuples"]
    tuple_f1 = tuples["micro"]["f1"] or 0.0
    brier = targets["micro"]["proper_scores"]["brier"]
    per_target = {
        target: {
            "score": tuples["per_target"][target]["f1"] or 0.0,
            "reference_support": tuples["per_target"][target]["sample_support"],
        }
        for target in ANALYTIC_TARGET_CLASSES
    }
    per_cell = {
        f"{target}:{stance}": {
            "score": tuples["reference_cells"][target][stance]["accuracy"]
            or 0.0,
            "reference_support": tuples["reference_cells"][target][stance][
                "sample_support"
            ],
        }
        for target in ANALYTIC_TARGET_CLASSES
        for stance in STANCE_CLASSES_B4
    }
    return {
        "tuple_micro_f1": tuple_f1,
        "target_presence_macro_f1": targets["macro_f1"] or 0.0,
        "calibration_error": brier if brier is not None else 1.0,
        # No selective thresholds are fit in Phase 1.  Full-coverage tuple error
        # is the registered retained-coverage risk diagnostic.
        "retained_coverage_risk": 1.0 - tuple_f1,
        "per_target": per_target,
        "per_target_stance_cell": per_cell,
        "invalid_outputs": metrics["outputs"]["missing"],
    }


def load_private_frame(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_frame: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read one exact private Parquet frame with the fixed v2 row schema."""

    if expected_frame not in {"training", "development"}:
        raise ValueError("expected_frame must be training or development")
    clean = artifact_descriptor(descriptor, where="private_frame")
    if clean.get("frame") != expected_frame:
        raise FactorisedExperimentContractError(
            "private frame descriptor identity drifted"
        )
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or file_sha256(path) != clean["sha256"]
    ):
        raise FactorisedExperimentContractError("private frame is missing or corrupt")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - Modal dependency path
        raise RuntimeError("private v2 frames require PyArrow") from exc
    table = pq.read_table(path)
    required = {
        "item_id",
        "frame",
        "target_text",
        "parent_context",
        "submission_context",
        "label_json",
        "selection_component",
        "selection_stratum",
        "inclusion_probability_numerator",
        "inclusion_probability_denominator",
        "inclusion_probability",
        "probability_scope",
    }
    if not required <= set(table.column_names):
        raise ValueError("private v2 frame omits required columns")
    rows = table.select(sorted(required)).to_pylist()
    if "row_count" in clean and len(rows) != clean["row_count"]:
        raise FactorisedExperimentContractError("private frame row count drifted")
    ids: set[str] = set()
    reference: dict[str, dict[str, Any]] = {}
    clean_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise ValueError(f"private frame row {index} has an invalid or duplicate ID")
        ids.add(item_id)
        if row.get("frame") != expected_frame:
            raise FactorisedExperimentContractError(
                f"private frame row {index} is not in the expected {expected_frame} frame"
            )
        try:
            label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("private frame contains invalid label JSON") from exc
        if not isinstance(label, dict):
            raise ValueError("private frame label must be an object")
        # Build once here to apply the authoritative ontology and masking rules.
        build_factorised_record(
            item_id=item_id,
            row=row,
            label=label,
            separator="[SEP]",
        )
        reference[item_id] = label
        clean_rows.append(dict(row))
    if "selection_component_counts" in clean:
        counts: dict[str, int] = {}
        for row in clean_rows:
            component = row.get("selection_component")
            if not isinstance(component, str) or not component:
                raise ValueError("private development row lacks a selection component")
            counts[component] = counts.get(component, 0) + 1
        if counts != clean["selection_component_counts"]:
            raise FactorisedExperimentContractError(
                "private development selection-component counts drifted"
            )
    return clean_rows, reference


def _thread_set_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_bound_input_file(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    descriptor_root: Path,
    where: str,
) -> dict[str, Any]:
    clean = artifact_descriptor(descriptor, where=where)
    try:
        relative = path.relative_to(descriptor_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{where} must remain inside descriptor_root") from exc
    if clean["relative_path"] != relative:
        raise FactorisedExperimentContractError(f"{where} path binding drifted")
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or file_sha256(path) != clean["sha256"]
    ):
        raise FactorisedExperimentContractError(f"{where} file binding drifted")
    return clean


def _validate_preparation_output_roots(
    *,
    private_split_root: Path,
    public_split_root: Path,
    frame_output_root: Path,
    descriptor_root: Path,
) -> None:
    """Require a registered private/public layout before any artefact is written."""

    descriptor = descriptor_root.resolve()
    private_split = private_split_root.resolve()
    public_split = public_split_root.resolve()
    frames = frame_output_root.resolve()
    roots = (private_split, public_split, frames)
    for root in roots:
        try:
            root.relative_to(descriptor)
        except ValueError as exc:
            raise ValueError(
                "all preparation roots must remain inside descriptor_root"
            ) from exc
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if root == other or root in other.parents or other in root.parents:
                raise ValueError(
                    "private split, public split and frames require disjoint roots"
                )

    if descriptor == REPO_ROOT.resolve():
        try:
            private_split.relative_to(LOCAL_PRIVATE_PREPARATION_ROOT)
            frames.relative_to(LOCAL_PRIVATE_PREPARATION_ROOT)
            public_split.relative_to(LOCAL_PUBLIC_PREPARATION_ROOT)
        except ValueError as exc:
            raise ValueError(
                "local preparation outputs must use the registered private/public "
                "factorised namespaces"
            ) from exc
        for private_root in (private_split, frames):
            ignored = subprocess.run(
                ["git", "check-ignore", "--quiet", "--", str(private_root)],
                cwd=REPO_ROOT,
                check=False,
            )
            if ignored.returncode != 0:
                raise ValueError(
                    "local private split/frame output is not covered by Git ignore"
                )
        return

    registered = descriptor / VOLUME_PREPARATION_ROOT_NAME
    for root in roots:
        try:
            root.relative_to(registered)
        except ValueError as exc:
            raise ValueError(
                "remote preparation outputs must use the registered Volume namespace"
            ) from exc
    if (
        private_split.parent != public_split.parent
        or private_split.parent != frames.parent
        or private_split.name != "private-split"
        or public_split.name != "public-split"
        or frames.name != "frames"
    ):
        raise ValueError(
            "remote preparation roots must be sibling private-split/public-split/frames"
        )


def _build_exposure_register(
    *,
    scope: str,
    thread_ids: Sequence[str],
    source_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if scope not in EXPOSURE_REGISTER_COUNTS:
        raise ValueError("exposure scope is not registered")
    expected = EXPOSURE_REGISTER_COUNTS[scope]
    threads = sorted(thread_ids)
    if (
        len(threads) != expected
        or len(set(threads)) != expected
        or any(not isinstance(value, str) or not value for value in threads)
    ):
        raise FactorisedExperimentContractError(
            f"{scope} exposure derivation must yield exactly {expected} unique threads"
        )
    clean_sources = {
        name: artifact_descriptor(value, where=f"source_artifacts.{name}")
        for name, value in sorted(source_artifacts.items())
    }
    if not clean_sources:
        raise ValueError("exposure register must bind its upstream source artefacts")
    body = {
        "schema_version": "1.0.0",
        "kind": EXPOSURE_REGISTER_KIND,
        "scope": scope,
        "thread_count": expected,
        "thread_set_sha256": canonical_sha256(threads),
        "source_artifacts": clean_sources,
        "thread_ids": threads,
    }
    return {**body, "register_id": canonical_sha256(body)}


def validate_private_exposure_register(
    value: Mapping[str, Any], *, scope: str
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "scope",
        "thread_count",
        "thread_set_sha256",
        "source_artifacts",
        "thread_ids",
        "register_id",
    }
    if set(value) != expected_keys or value.get("kind") != EXPOSURE_REGISTER_KIND:
        raise ValueError("private exposure-register schema drifted")
    sources = value.get("source_artifacts")
    threads = value.get("thread_ids")
    if not isinstance(sources, Mapping) or not isinstance(threads, list):
        raise ValueError("private exposure register lacks sources or thread IDs")
    expected = _build_exposure_register(
        scope=scope,
        thread_ids=threads,
        source_artifacts=sources,
    )
    if dict(value) != expected:
        raise FactorisedExperimentContractError("private exposure-register digest drifted")
    return expected


def derive_bridge_exposure_register(
    *,
    pilot_mapping_parquet_path: Path,
    pilot_mapping_descriptor: Mapping[str, Any],
    pilot_packet_manifest_path: Path,
    pilot_packet_manifest_descriptor: Mapping[str, Any],
    pilot_packet_receipt_path: Path,
    pilot_packet_receipt_descriptor: Mapping[str, Any],
    bridge_authorisation_path: Path,
    bridge_authorisation_descriptor: Mapping[str, Any],
    descriptor_root: Path,
) -> dict[str, Any]:
    """Derive the 480-thread register from the accepted pilot and bridge run."""

    clean = _validate_bound_input_file(
        pilot_mapping_parquet_path,
        pilot_mapping_descriptor,
        descriptor_root=descriptor_root,
        where="bridge pilot mapping",
    )
    if clean.get("row_count") != EXPOSURE_REGISTER_COUNTS["bridge"]:
        raise FactorisedExperimentContractError("bridge mapping row-count binding drifted")
    packet_manifest = _validate_bound_input_file(
        pilot_packet_manifest_path,
        pilot_packet_manifest_descriptor,
        descriptor_root=descriptor_root,
        where="bridge pilot packet manifest",
    )
    packet_receipt = _validate_bound_input_file(
        pilot_packet_receipt_path,
        pilot_packet_receipt_descriptor,
        descriptor_root=descriptor_root,
        where="bridge pilot packet receipt",
    )
    bridge_authorisation = _validate_bound_input_file(
        bridge_authorisation_path,
        bridge_authorisation_descriptor,
        descriptor_root=descriptor_root,
        where="accepted bridge authorisation",
    )
    manifest = _read_json(pilot_packet_manifest_path, where="bridge pilot packet manifest")
    receipt = _read_json(pilot_packet_receipt_path, where="bridge pilot packet receipt")
    if (
        manifest.get("kind") != "semantic-ontology-v2-pilot-packet-v1"
        or receipt.get("kind") != "semantic-ontology-v2-pilot-packet-receipt-v1"
        or receipt.get("status") != "complete"
        or manifest.get("packet_id") != ACCEPTED_BRIDGE_PACKET_ID
        or receipt.get("packet_id") != ACCEPTED_BRIDGE_PACKET_ID
        or manifest.get("private_mapping_sha256") != clean["sha256"]
        or receipt.get("private_mapping_sha256") != clean["sha256"]
        or receipt.get("manifest_sha256") != packet_manifest["sha256"]
        or manifest.get("target_rows") != 480
        or receipt.get("selected_rows") != 480
        or receipt.get("unique_threads") != 480
    ):
        raise FactorisedExperimentContractError(
            "bridge packet/receipt/mapping binding drifted"
        )
    assert_metadata_only(receipt, where="bridge pilot packet receipt")
    bridge_module = importlib.import_module("reddit_china_stance.sol_teacher_10k_v2")
    bridge_validator = getattr(bridge_module, "validate_accepted_bridge_receipt", None)
    if not callable(bridge_validator):
        raise RuntimeError("accepted bridge receipt validator is unavailable")
    validated_authorisation = bridge_validator(bridge_authorisation_path)
    if canonical_sha256(validated_authorisation) != canonical_sha256(
        _read_json(bridge_authorisation_path, where="accepted bridge authorisation")
    ):
        raise FactorisedExperimentContractError(
            "accepted bridge authorisation validation drifted"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - preparation dependency path
        raise RuntimeError("bridge exposure derivation requires PyArrow") from exc
    parquet = pq.ParquetFile(pilot_mapping_parquet_path)
    if not {"sample_id", "thread_id"} <= set(parquet.schema_arrow.names):
        raise ValueError("bridge pilot mapping omits canonical sample/thread IDs")
    rows = parquet.read(columns=["sample_id", "thread_id"]).to_pylist()
    sample_ids = [row.get("sample_id") for row in rows]
    thread_ids = [row.get("thread_id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)) or any(
        not isinstance(value, str) or not value for value in sample_ids
    ):
        raise ValueError("bridge mapping contains invalid or duplicate sample IDs")
    return _build_exposure_register(
        scope="bridge",
        thread_ids=thread_ids,
        source_artifacts={
            "pilot_private_mapping": clean,
            "pilot_packet_manifest": packet_manifest,
            "pilot_packet_receipt": packet_receipt,
            "bridge_authorisation": bridge_authorisation,
        },
    )


def build_legacy_overlap_audit(
    *,
    development_proxy_parquet_path: Path,
    development_proxy_descriptor: Mapping[str, Any],
    source_parquet_path: Path,
    source_parquet_descriptor: Mapping[str, Any],
    descriptor_root: Path,
) -> dict[str, Any]:
    """Prove the 452-row legacy proxy has no ID or exact-text overlap with the 10k.

    Legacy proxy IDs are not canonical 10k IDs, so they must never be treated as
    in-source exposure registers.  The audit reads only IDs, split and permitted text
    surfaces; no legacy label is returned or joined to the v2 source.
    """

    proxy = _validate_bound_input_file(
        development_proxy_parquet_path,
        development_proxy_descriptor,
        descriptor_root=descriptor_root,
        where="legacy development proxy",
    )
    source = _validate_bound_input_file(
        source_parquet_path,
        source_parquet_descriptor,
        descriptor_root=descriptor_root,
        where="canonical source Parquet",
    )
    if proxy.get("row_count") != 452 or source.get("row_count") != EXPECTED_SOURCE_ROWS:
        raise FactorisedExperimentContractError("legacy exposure source row counts drifted")
    legacy = importlib.import_module("reddit_china_stance.modernbert_training")
    validator = getattr(legacy, "validate_development_proxy_parquet", None)
    if not callable(validator):
        raise RuntimeError("legacy development-proxy validator is unavailable")
    validated = validator(development_proxy_parquet_path)
    if validated.get("sha256") != proxy["sha256"]:
        raise FactorisedExperimentContractError("legacy development proxy validation drifted")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - preparation dependency path
        raise RuntimeError("legacy exposure derivation requires PyArrow") from exc
    source_rows = pq.read_table(
        source_parquet_path,
        columns=["sample_id", "target_text", "parent_context", "submission_context"],
    ).to_pylist()
    source_ids: set[str] = set()
    source_surfaces: list[str] = []
    for row in source_rows:
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in source_ids
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
            or any(
                row.get(field) is not None and not isinstance(row[field], str)
                for field in ("parent_context", "submission_context")
            )
        ):
            raise ValueError("canonical source has invalid or duplicate surface rows")
        source_ids.add(sample_id)
        source_surfaces.append(
            canonical_sha256(
                {
                    "target_text": row["target_text"],
                    "parent_context": row["parent_context"],
                    "submission_context": row["submission_context"],
                }
            )
        )
    proxy_table = pq.read_table(
        development_proxy_parquet_path,
        columns=[
            "source_sample_id",
            "target_text",
            "parent_context",
            "submission_context",
            "split",
        ],
    )
    proxy_rows = proxy_table.to_pylist()
    schema_metadata = proxy_table.schema.metadata or {}
    schema_metadata_sha256 = canonical_sha256(
        {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in sorted(schema_metadata.items())
        }
    )
    proxy_ids: set[str] = set()
    proxy_ids_by_split: dict[str, list[str]] = {
        "development": [],
        "locked_test_candidate": [],
    }
    proxy_surfaces: dict[str, list[str]] = {
        "development": [],
        "locked_test_candidate": [],
    }
    split_counts = {"development": 0, "locked_test_candidate": 0}
    for row in proxy_rows:
        sample_id = row.get("source_sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in proxy_ids
            or row.get("split") not in split_counts
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
            or any(
                row.get(field) is not None and not isinstance(row[field], str)
                for field in ("parent_context", "submission_context")
            )
        ):
            raise ValueError(
                "legacy proxy contains an invalid or duplicate ID/text surface"
            )
        proxy_ids.add(sample_id)
        proxy_ids_by_split[row["split"]].append(sample_id)
        split_counts[row["split"]] += 1
        proxy_surfaces[row["split"]].append(
            canonical_sha256(
                {
                    "target_text": row["target_text"],
                    "parent_context": row["parent_context"],
                    "submission_context": row["submission_context"],
                }
            )
        )
    if split_counts != {"development": 222, "locked_test_candidate": 230}:
        raise FactorisedExperimentContractError("legacy proxy split counts drifted")
    id_overlap = len(source_ids & proxy_ids)
    combined_proxy_surfaces = [
        *proxy_surfaces["development"],
        *proxy_surfaces["locked_test_candidate"],
    ]
    exact_surface_overlap = len(set(source_surfaces) & set(combined_proxy_surfaces))
    if id_overlap or exact_surface_overlap:
        raise FactorisedExperimentContractError(
            "legacy proxy overlaps the canonical 10k and cannot remain audit-only"
        )
    body = {
        "schema_version": "1.0.0",
        "kind": LEGACY_OVERLAP_AUDIT_KIND,
        "source_artifacts": {
            "legacy_proxy": proxy,
            "canonical_source_parquet": source,
        },
        "legacy_proxy_file_sha256": proxy["sha256"],
        "legacy_proxy_rows": 452,
        "legacy_development_rows": 222,
        "legacy_locked_rows": 230,
        "legacy_proxy_schema_metadata_sha256": schema_metadata_sha256,
        "legacy_development_proxy_id_set_sha256": canonical_sha256(
            sorted(proxy_ids_by_split["development"])
        ),
        "legacy_locked_proxy_id_set_sha256": canonical_sha256(
            sorted(proxy_ids_by_split["locked_test_candidate"])
        ),
        "legacy_development_proxy_surface_set_sha256": canonical_sha256(
            sorted(proxy_surfaces["development"])
        ),
        "legacy_locked_proxy_surface_set_sha256": canonical_sha256(
            sorted(proxy_surfaces["locked_test_candidate"])
        ),
        "canonical_source_surface_set_sha256": canonical_sha256(
            sorted(source_surfaces)
        ),
        "legacy_id_overlap_count": id_overlap,
        "legacy_development_id_overlap_count": len(
            source_ids & set(proxy_ids_by_split["development"])
        ),
        "legacy_locked_id_overlap_count": len(
            source_ids & set(proxy_ids_by_split["locked_test_candidate"])
        ),
        "legacy_exact_surface_overlap_count": exact_surface_overlap,
    }
    result = {**body, "audit_id": canonical_sha256(body)}
    assert_metadata_only(result, where="legacy proxy overlap audit")
    return result


def validate_legacy_overlap_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    sources = value.get("source_artifacts")
    if not isinstance(sources, Mapping) or set(sources) != {
        "legacy_proxy",
        "canonical_source_parquet",
    }:
        raise ValueError("legacy overlap audit lacks its exact source bindings")
    expected_fields = {
        "schema_version",
        "kind",
        "source_artifacts",
        "legacy_proxy_file_sha256",
        "legacy_proxy_rows",
        "legacy_development_rows",
        "legacy_locked_rows",
        "legacy_proxy_schema_metadata_sha256",
        "legacy_development_proxy_id_set_sha256",
        "legacy_locked_proxy_id_set_sha256",
        "legacy_development_proxy_surface_set_sha256",
        "legacy_locked_proxy_surface_set_sha256",
        "canonical_source_surface_set_sha256",
        "legacy_id_overlap_count",
        "legacy_development_id_overlap_count",
        "legacy_locked_id_overlap_count",
        "legacy_exact_surface_overlap_count",
        "audit_id",
    }
    if set(value) != expected_fields or value.get("kind") != LEGACY_OVERLAP_AUDIT_KIND:
        raise ValueError("legacy overlap audit schema drifted")
    clean_sources = {
        name: artifact_descriptor(descriptor, where=f"source_artifacts.{name}")
        for name, descriptor in sources.items()
    }
    body = {key: value[key] for key in expected_fields - {"audit_id"}}
    body["source_artifacts"] = clean_sources
    if (
        value.get("legacy_proxy_rows") != 452
        or value.get("legacy_development_rows") != 222
        or value.get("legacy_locked_rows") != 230
        or value.get("legacy_id_overlap_count") != 0
        or value.get("legacy_development_id_overlap_count") != 0
        or value.get("legacy_locked_id_overlap_count") != 0
        or value.get("legacy_exact_surface_overlap_count") != 0
        or value.get("legacy_proxy_file_sha256") != clean_sources["legacy_proxy"]["sha256"]
        or any(
            not isinstance(value.get(field), str)
            or len(value[field]) != 64
            or any(character not in "0123456789abcdef" for character in value[field])
            for field in (
                "legacy_proxy_schema_metadata_sha256",
                "legacy_development_proxy_id_set_sha256",
                "legacy_locked_proxy_id_set_sha256",
                "legacy_development_proxy_surface_set_sha256",
                "legacy_locked_proxy_surface_set_sha256",
                "canonical_source_surface_set_sha256",
            )
        )
        or value.get("audit_id") != canonical_sha256(body)
    ):
        raise FactorisedExperimentContractError("legacy overlap audit binding drifted")
    assert_metadata_only(value, where="legacy proxy overlap audit")
    return dict(value)


def _joined_rows_from_teacher_artifacts(
    *,
    teacher_labels_parquet_path: Path,
    blinded_input_json_path: Path,
    private_mapping_parquet_path: Path,
    source_parquet_path: Path,
    expected_teacher_run_id: str,
) -> list[dict[str, Any]]:
    """Reconstruct the sole authoritative source/mapping/teacher join."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - preparation dependency path
        raise RuntimeError("factorised evidence preparation requires PyArrow") from exc

    label_columns = (
        "opaque_id",
        "codability",
        "relevance",
        "label_json",
        "quality_tier",
        "primary_training_eligible",
    )
    label_table = pq.read_table(teacher_labels_parquet_path)
    if tuple(label_table.column_names) != label_columns:
        raise ValueError("teacher labels Parquet column contract drifted")
    metadata = label_table.schema.metadata or {}
    if metadata.get(b"run_id", b"").decode("utf-8") != expected_teacher_run_id:
        raise FactorisedExperimentContractError("teacher labels run binding drifted")
    labels: dict[str, dict[str, Any]] = {}
    label_order: list[str] = []
    label_validator = load_v2_label_validator()
    for index, row in enumerate(label_table.to_pylist()):
        opaque_id = row.get("opaque_id")
        if not isinstance(opaque_id, str) or not opaque_id or opaque_id in labels:
            raise ValueError("teacher labels contain an invalid or duplicate opaque ID")
        try:
            clean_label = validate_v2_label(
                json.loads(row.get("label_json")), validator=label_validator
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"teacher label row {index} contains invalid JSON") from exc
        encoded = build_factorised_record(
            item_id=opaque_id,
            row={"target_text": "probe", "parent_context": None, "submission_context": None},
            label=clean_label,
            separator="[SEP]",
        ).encoding
        expected_codability = "codable" if encoded.codability_mask else "not_codable"
        expected_relevance = (
            None
            if not encoded.relevance_mask
            else "material"
            if encoded.relevance_label == 1
            else "not_material"
        )
        if (
            row.get("codability") != expected_codability
            or row.get("relevance") != expected_relevance
            or type(row.get("primary_training_eligible")) is not bool
            or row.get("quality_tier")
            not in {"exact_consensus", "blind_majority", "informed_adjudication"}
        ):
            raise FactorisedExperimentContractError(
                "teacher label columns disagree with the canonical label or quality contract"
            )
        labels[opaque_id] = {**row, "clean_label": clean_label}
        label_order.append(opaque_id)

    blinded = _read_json(blinded_input_json_path, where="teacher blinded input")
    if set(blinded) != {
        "schema_version",
        "kind",
        "packet_id",
        "rubric_sha256",
        "label_schema_sha256",
        "rows",
    } or blinded.get("kind") != "sol-teacher-10k-v2-blinded-input-v1":
        raise ValueError("teacher blinded-input contract drifted")
    blinded_rows = blinded.get("rows")
    if not isinstance(blinded_rows, list) or not blinded_rows:
        raise ValueError("teacher blinded input contains no rows")
    blinded_by_opaque: dict[str, dict[str, Any]] = {}
    blinded_order: list[str] = []
    for row in blinded_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "source_sample_id",
            "target_text",
            "submission_context",
            "parent_context",
        }:
            raise ValueError("teacher blinded row schema drifted")
        opaque_id = row.get("source_sample_id")
        if (
            not isinstance(opaque_id, str)
            or not opaque_id
            or opaque_id in blinded_by_opaque
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
            or any(
                row.get(field) is not None and not isinstance(row[field], str)
                for field in ("submission_context", "parent_context")
            )
        ):
            raise ValueError("teacher blinded input contains an invalid row")
        blinded_by_opaque[opaque_id] = dict(row)
        blinded_order.append(opaque_id)

    mapping_columns = ("opaque_id", "sample_id", "thread_id", "packet_order")
    mapping_table = pq.read_table(private_mapping_parquet_path)
    if tuple(mapping_table.column_names) != mapping_columns:
        raise ValueError("teacher private mapping column contract drifted")
    mapping_rows = sorted(mapping_table.to_pylist(), key=lambda row: row["packet_order"])
    if [row.get("packet_order") for row in mapping_rows] != list(range(len(mapping_rows))):
        raise FactorisedExperimentContractError("teacher private mapping order drifted")
    if [row.get("opaque_id") for row in mapping_rows] != blinded_order:
        raise FactorisedExperimentContractError("mapping and blinded-input order drifted")
    mapping_by_opaque: dict[str, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    thread_ids: set[str] = set()
    for row in mapping_rows:
        opaque_id = row.get("opaque_id")
        sample_id = row.get("sample_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(opaque_id, str)
            or not opaque_id
            or opaque_id in mapping_by_opaque
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in thread_ids
        ):
            raise ValueError("teacher private mapping violates row/thread uniqueness")
        mapping_by_opaque[opaque_id] = dict(row)
        sample_ids.add(sample_id)
        thread_ids.add(thread_id)

    source_table = pq.read_table(source_parquet_path, columns=list(SOURCE_ROW_HASH_FIELDS))
    source_by_sample: dict[str, dict[str, Any]] = {}
    source_threads: set[str] = set()
    for row in source_table.to_pylist():
        sample_id = row.get("sample_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in source_by_sample
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in source_threads
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
            or any(
                row.get(field) is not None and not isinstance(row[field], str)
                for field in ("submission_context", "parent_context")
            )
            or not isinstance(row.get("subreddit"), str)
            or not row["subreddit"]
            or type(row.get("year")) is not int
            or not isinstance(row.get("content_type"), str)
            or not row["content_type"]
            or not isinstance(row.get("retrieval_mode"), str)
            or not row["retrieval_mode"]
        ):
            raise ValueError("source Parquet contains an invalid or duplicate row/thread")
        clean_source = {field: row[field] for field in SOURCE_ROW_HASH_FIELDS}
        source_by_sample[sample_id] = clean_source
        source_threads.add(thread_id)

    if (
        len(labels) != EXPECTED_SOURCE_ROWS
        or len(blinded_rows) != EXPECTED_SOURCE_ROWS
        or len(mapping_rows) != EXPECTED_SOURCE_ROWS
        or len(source_by_sample) != EXPECTED_SOURCE_ROWS
        or set(labels) != set(blinded_by_opaque)
        or set(labels) != set(mapping_by_opaque)
        or sample_ids != set(source_by_sample)
        or label_order != blinded_order
    ):
        raise FactorisedExperimentContractError(
            "teacher/source inputs fail exact 10k row conservation"
        )

    joined: list[dict[str, Any]] = []
    for opaque_id in blinded_order:
        mapping = mapping_by_opaque[opaque_id]
        source = source_by_sample[mapping["sample_id"]]
        blinded_row = blinded_by_opaque[opaque_id]
        teacher = labels[opaque_id]
        if source["thread_id"] != mapping["thread_id"] or any(
            source[field] != blinded_row[field]
            for field in ("target_text", "submission_context", "parent_context")
        ):
            raise FactorisedExperimentContractError(
                "teacher mapping, blinded text/context and source Parquet disagree"
            )
        label_digest = label_sha256(
            teacher["clean_label"], validator=label_validator
        )
        source_digest = source_row_sha256(source)
        mapping_record = {
            "sample_id": mapping["sample_id"],
            "thread_id": mapping["thread_id"],
            "opaque_id": opaque_id,
        }
        teacher_record = {
            "opaque_id": opaque_id,
            "label_sha256": label_digest,
            "quality_tier": teacher["quality_tier"],
            "primary_training_eligible": teacher["primary_training_eligible"],
        }
        mapping_digest = canonical_sha256(mapping_record)
        teacher_digest = canonical_sha256(teacher_record)
        joined.append(
            {
                "item_id": mapping["sample_id"],
                "teacher_opaque_id": opaque_id,
                "thread_id": mapping["thread_id"],
                "year": source["year"],
                "subreddit": source["subreddit"],
                "content_type": source["content_type"],
                "retrieval_mode": source["retrieval_mode"],
                "target_text": source["target_text"],
                "submission_context": source["submission_context"],
                "parent_context": source["parent_context"],
                "label_json": teacher["label_json"],
                "label_sha256": label_digest,
                "source_row_sha256": source_digest,
                "mapping_row_sha256": mapping_digest,
                "teacher_row_sha256": teacher_digest,
                "joined_row_sha256": canonical_sha256(
                    {
                        "source_row_sha256": source_digest,
                        "mapping_row_sha256": mapping_digest,
                        "teacher_row_sha256": teacher_digest,
                    }
                ),
                "quality_tier": teacher["quality_tier"],
                "primary_training_eligible": teacher["primary_training_eligible"],
            }
        )
    return joined


def _validate_teacher_receipt_binding(
    value: Mapping[str, Any],
    *,
    expected_teacher_run_id: str,
    teacher_labels_sha256: str,
    private_mapping_sha256: str,
    source_parquet_sha256: str,
    rubric_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    """Bind the aggregate teacher receipt to every private training input."""

    required = {
        "kind": "sol-teacher-10k-v2-receipt-v1",
        "status": "complete",
        "run_id": expected_teacher_run_id,
        "row_count": EXPECTED_SOURCE_ROWS,
        "private_labels_parquet_sha256": teacher_labels_sha256,
        "private_mapping_sha256": private_mapping_sha256,
        "source_parquet_sha256": source_parquet_sha256,
        "rubric_sha256": rubric_sha256,
        "label_schema_sha256": schema_sha256,
        "evidence_boundary": "silver-model-assisted-teacher-labels-not-human-validation",
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise FactorisedExperimentContractError(
            "teacher receipt does not bind the exact fresh-v2 inputs"
        )
    _sha(value.get("packet_id"), where="teacher receipt packet_id")
    assert_metadata_only(value, where="factorised teacher receipt")
    return dict(value)


def _validate_source_provenance_binding(
    value: Mapping[str, Any],
    *,
    receipt_path: Path,
    expected_packet_id: str,
    source_parquet_sha256: str,
    private_mapping_sha256: str,
    blinded_input_sha256: str,
) -> dict[str, Any]:
    """Validate the exact teacher-packet receipt behind source/mapping inputs."""

    required = {
        "kind": "sol-teacher-10k-v2-packet-receipt-v1",
        "status": "complete",
        "packet_id": expected_packet_id,
        "source_parquet_sha256": source_parquet_sha256,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "selected_rows": EXPECTED_SOURCE_ROWS,
        "unique_threads": EXPECTED_SOURCE_ROWS,
        "blinded_input_sha256": blinded_input_sha256,
        "private_mapping_sha256": private_mapping_sha256,
        "evidence_boundary": "silver-model-assisted-teacher-labels-not-human-validation",
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise FactorisedExperimentContractError(
            "source packet receipt does not bind the exact source, mapping and blinded input"
        )
    expected_name = f"receipt-{canonical_sha256(value)}.json"
    if receipt_path.name != expected_name:
        raise FactorisedExperimentContractError(
            "source packet receipt path is not its canonical content address"
        )
    assert_metadata_only(value, where="factorised source packet receipt")
    return dict(value)


def _legacy_proxy_surface_hashes(path: Path) -> dict[str, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - preparation dependency path
        raise RuntimeError("factorised evidence preparation requires PyArrow") from exc
    rows = pq.read_table(
        path,
        columns=["target_text", "parent_context", "submission_context", "split"],
    ).to_pylist()
    result = {"development": [], "locked_test_candidate": []}
    for row in rows:
        split = row.get("split")
        if split not in result:
            raise ValueError("legacy proxy split drifted")
        result[split].append(
            canonical_sha256(
                {
                    "target_text": row["target_text"],
                    "parent_context": row["parent_context"],
                    "submission_context": row["submission_context"],
                }
            )
        )
    if len(result["development"]) != 222 or len(result["locked_test_candidate"]) != 230:
        raise FactorisedExperimentContractError("legacy proxy surface split counts drifted")
    return result


def _legacy_proxy_id_sets(path: Path) -> dict[str, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - preparation dependency path
        raise RuntimeError("factorised evidence preparation requires PyArrow") from exc
    rows = pq.read_table(path, columns=["source_sample_id", "split"]).to_pylist()
    result = {"development": [], "locked_test_candidate": []}
    for row in rows:
        split = row.get("split")
        sample_id = row.get("source_sample_id")
        if split not in result or not isinstance(sample_id, str) or not sample_id:
            raise ValueError("legacy proxy ID/split contract drifted")
        result[split].append(sample_id)
    if (
        len(result["development"]) != 222
        or len(set(result["development"])) != 222
        or len(result["locked_test_candidate"]) != 230
        or len(set(result["locked_test_candidate"])) != 230
    ):
        raise FactorisedExperimentContractError("legacy proxy ID-set counts drifted")
    return result


def prepare_evidence_frames_from_artifacts(
    *,
    teacher_receipt_path: Path,
    teacher_receipt_descriptor: Mapping[str, Any],
    teacher_labels_parquet_path: Path,
    teacher_labels_descriptor: Mapping[str, Any],
    blinded_input_json_path: Path,
    blinded_input_descriptor: Mapping[str, Any],
    private_mapping_parquet_path: Path,
    private_mapping_descriptor: Mapping[str, Any],
    source_parquet_path: Path,
    source_parquet_descriptor: Mapping[str, Any],
    source_receipt_path: Path,
    source_receipt_descriptor: Mapping[str, Any],
    source_metadata_mapping_path: Path,
    source_metadata_mapping_descriptor: Mapping[str, Any],
    bridge_receipt_path: Path,
    bridge_receipt_descriptor: Mapping[str, Any],
    bridge_exposure_register_json_path: Path,
    bridge_exposure_register_descriptor: Mapping[str, Any],
    legacy_proxy_parquet_path: Path,
    legacy_proxy_descriptor: Mapping[str, Any],
    private_split_root: Path,
    public_split_root: Path,
    frame_output_root: Path,
    descriptor_root: Path,
    expected_teacher_run_id: str,
    rubric_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    """Build production splits from exact external artefacts, then materialise frames.

    This is the sole production preparation boundary.  It never accepts externally
    joined rows or an existing membership as authority.
    """

    _validate_preparation_output_roots(
        private_split_root=private_split_root,
        public_split_root=public_split_root,
        frame_output_root=frame_output_root,
        descriptor_root=descriptor_root,
    )

    bound_inputs = {
        "teacher_receipt": (teacher_receipt_path, teacher_receipt_descriptor),
        "teacher_labels": (teacher_labels_parquet_path, teacher_labels_descriptor),
        "teacher_blinded_input": (blinded_input_json_path, blinded_input_descriptor),
        "teacher_private_mapping": (
            private_mapping_parquet_path,
            private_mapping_descriptor,
        ),
        "source_parquet": (source_parquet_path, source_parquet_descriptor),
        "source_receipt": (source_receipt_path, source_receipt_descriptor),
        "source_metadata_mapping": (
            source_metadata_mapping_path,
            source_metadata_mapping_descriptor,
        ),
        "bridge_receipt": (bridge_receipt_path, bridge_receipt_descriptor),
        "bridge_exposure_register": (
            bridge_exposure_register_json_path,
            bridge_exposure_register_descriptor,
        ),
        "legacy_proxy": (legacy_proxy_parquet_path, legacy_proxy_descriptor),
    }
    clean_descriptors = {
        name: _validate_bound_input_file(
            path,
            descriptor,
            descriptor_root=descriptor_root,
            where=name,
        )
        for name, (path, descriptor) in bound_inputs.items()
    }
    teacher_receipt = _validate_teacher_receipt_binding(
        _read_json(teacher_receipt_path, where="teacher receipt"),
        expected_teacher_run_id=expected_teacher_run_id,
        teacher_labels_sha256=clean_descriptors["teacher_labels"]["sha256"],
        private_mapping_sha256=clean_descriptors["teacher_private_mapping"]["sha256"],
        source_parquet_sha256=clean_descriptors["source_parquet"]["sha256"],
        rubric_sha256=rubric_sha256,
        schema_sha256=schema_sha256,
    )
    if source_metadata_mapping_path.resolve() != private_mapping_parquet_path.resolve():
        raise FactorisedExperimentContractError(
            "source metadata mapping must be the exact teacher private mapping file"
        )
    if (
        clean_descriptors["source_metadata_mapping"]["sha256"]
        != clean_descriptors["teacher_private_mapping"]["sha256"]
        or clean_descriptors["source_metadata_mapping"]["bytes"]
        != clean_descriptors["teacher_private_mapping"]["bytes"]
    ):
        raise FactorisedExperimentContractError(
            "source metadata mapping descriptor drifted from the teacher private mapping"
        )
    blinded_input = _read_json(blinded_input_json_path, where="teacher blinded input")
    packet_id = _sha(teacher_receipt.get("packet_id"), where="teacher receipt packet_id")
    if blinded_input.get("packet_id") != packet_id:
        raise FactorisedExperimentContractError(
            "teacher receipt and blinded input packet IDs drifted"
        )
    _validate_source_provenance_binding(
        _read_json(source_receipt_path, where="canonical source receipt"),
        receipt_path=source_receipt_path,
        expected_packet_id=packet_id,
        source_parquet_sha256=clean_descriptors["source_parquet"]["sha256"],
        private_mapping_sha256=clean_descriptors["teacher_private_mapping"]["sha256"],
        blinded_input_sha256=clean_descriptors["teacher_blinded_input"]["sha256"],
    )
    if any(
        clean_descriptors[name].get("row_count") != EXPECTED_SOURCE_ROWS
        for name in (
            "teacher_labels",
            "teacher_blinded_input",
            "teacher_private_mapping",
            "source_parquet",
        )
    ):
        raise FactorisedExperimentContractError(
            "teacher/source descriptors do not bind the exact 10k"
        )
    bridge_register = validate_private_exposure_register(
        _read_json(bridge_exposure_register_json_path, where="bridge exposure register"),
        scope="bridge",
    )
    if clean_descriptors["bridge_exposure_register"].get("row_count") != 480:
        raise FactorisedExperimentContractError(
            "bridge exposure register descriptor must bind 480 threads"
        )
    source_artifacts = bridge_register["source_artifacts"]
    if not any(
        descriptor.get("sha256") == clean_descriptors["bridge_receipt"]["sha256"]
        for descriptor in source_artifacts.values()
    ):
        raise FactorisedExperimentContractError(
            "bridge register does not bind the accepted bridge receipt"
        )

    joined_rows = _joined_rows_from_teacher_artifacts(
        teacher_labels_parquet_path=teacher_labels_parquet_path,
        blinded_input_json_path=blinded_input_json_path,
        private_mapping_parquet_path=private_mapping_parquet_path,
        source_parquet_path=source_parquet_path,
        expected_teacher_run_id=expected_teacher_run_id,
    )
    overlap_audit = build_legacy_overlap_audit(
        development_proxy_parquet_path=legacy_proxy_parquet_path,
        development_proxy_descriptor=legacy_proxy_descriptor,
        source_parquet_path=source_parquet_path,
        source_parquet_descriptor=source_parquet_descriptor,
        descriptor_root=descriptor_root,
    )
    surfaces = _legacy_proxy_surface_hashes(legacy_proxy_parquet_path)
    proxy_ids = _legacy_proxy_id_sets(legacy_proxy_parquet_path)
    bindings = {
        "source_receipt_sha256": clean_descriptors["source_receipt"]["sha256"],
        "source_metadata_mapping_sha256": clean_descriptors[
            "source_metadata_mapping"
        ]["sha256"],
        "teacher_receipt_sha256": clean_descriptors["teacher_receipt"]["sha256"],
        "teacher_final_ledger_sha256": clean_descriptors["teacher_labels"]["sha256"],
        "bridge_receipt_sha256": clean_descriptors["bridge_receipt"]["sha256"],
        "bridge_exposure_register_file_sha256": clean_descriptors[
            "bridge_exposure_register"
        ]["sha256"],
        "bridge_exposure_register_content_sha256": bridge_register[
            "thread_set_sha256"
        ],
        "legacy_development_proxy_file_sha256": clean_descriptors["legacy_proxy"][
            "sha256"
        ],
        "legacy_development_proxy_schema_metadata_sha256": overlap_audit[
            "legacy_proxy_schema_metadata_sha256"
        ],
        "legacy_development_proxy_sample_id_set_sha256": overlap_audit[
            "legacy_development_proxy_id_set_sha256"
        ],
        "legacy_development_proxy_surface_set_sha256": overlap_audit[
            "legacy_development_proxy_surface_set_sha256"
        ],
        "legacy_locked_proxy_file_sha256": clean_descriptors["legacy_proxy"]["sha256"],
        "legacy_locked_proxy_schema_metadata_sha256": overlap_audit[
            "legacy_proxy_schema_metadata_sha256"
        ],
        "legacy_locked_proxy_sample_id_set_sha256": overlap_audit[
            "legacy_locked_proxy_id_set_sha256"
        ],
        "legacy_locked_proxy_surface_set_sha256": overlap_audit[
            "legacy_locked_proxy_surface_set_sha256"
        ],
    }
    splits = importlib.import_module(
        "reddit_china_stance.modernbert_factorised_splits"
    )
    policy = splits.EvidenceFramePolicy()
    private_membership, public_manifest = splits.build_evidence_frames(
        joined_rows,
        input_bindings=bindings,
        bridge_exposed_thread_ids=bridge_register["thread_ids"],
        legacy_development_proxy_sample_ids=proxy_ids["development"],
        legacy_development_proxy_surface_hashes=surfaces["development"],
        legacy_locked_proxy_sample_ids=proxy_ids["locked_test_candidate"],
        legacy_locked_proxy_surface_hashes=surfaces["locked_test_candidate"],
        policy=policy,
    )
    splits.validate_evidence_frames(
        private_membership,
        public_manifest,
        rows=joined_rows,
        input_bindings=bindings,
        bridge_exposed_thread_ids=bridge_register["thread_ids"],
        legacy_development_proxy_sample_ids=proxy_ids["development"],
        legacy_development_proxy_surface_hashes=surfaces["development"],
        legacy_locked_proxy_sample_ids=proxy_ids["locked_test_candidate"],
        legacy_locked_proxy_surface_hashes=surfaces["locked_test_candidate"],
        policy=policy,
    )
    private_path = private_split_root / (
        f"membership-{private_membership['membership_id']}.json"
    )
    public_path = public_split_root / f"manifest-{public_manifest['manifest_id']}.json"
    _write_json_atomic(private_path, private_membership)
    _write_json_atomic(public_path, public_manifest)

    result = materialise_private_frames(
        teacher_labels_parquet_path=teacher_labels_parquet_path,
        blinded_input_json_path=blinded_input_json_path,
        private_mapping_parquet_path=private_mapping_parquet_path,
        source_parquet_path=source_parquet_path,
        membership_json_path=private_path,
        split_public_manifest_json_path=public_path,
        bridge_exposure_register_json_path=bridge_exposure_register_json_path,
        legacy_proxy_parquet_path=legacy_proxy_parquet_path,
        legacy_proxy_descriptor=legacy_proxy_descriptor,
        source_parquet_descriptor=source_parquet_descriptor,
        output_root=frame_output_root,
        descriptor_root=descriptor_root,
        expected_teacher_run_id=expected_teacher_run_id,
    )
    result["split_manifest"] = {
        **_descriptor(private_path, root=descriptor_root),
        "row_count": EXPECTED_SOURCE_ROWS,
        "manifest_id": private_membership["membership_id"],
    }
    result["split_public_manifest"] = {
        **_descriptor(public_path, root=descriptor_root),
        "row_count": EXPECTED_SOURCE_ROWS,
        "manifest_id": public_manifest["manifest_id"],
    }
    result["input_bindings_sha256"] = canonical_sha256(bindings)
    assert_metadata_only(result, where="factorised production preparation summary")
    return result


def materialise_private_frames(
    *,
    teacher_labels_parquet_path: Path,
    blinded_input_json_path: Path,
    private_mapping_parquet_path: Path,
    source_parquet_path: Path,
    membership_json_path: Path,
    split_public_manifest_json_path: Path,
    bridge_exposure_register_json_path: Path,
    legacy_proxy_parquet_path: Path,
    legacy_proxy_descriptor: Mapping[str, Any],
    source_parquet_descriptor: Mapping[str, Any],
    output_root: Path,
    descriptor_root: Path,
    expected_teacher_run_id: str,
) -> dict[str, Any]:
    """Exact-join frozen teacher inputs and freeze fresh private evidence frames.

    The labels deliberately contain no text or canonical source IDs.  This function
    therefore reconstructs the private training view from four independently bound
    artefacts: final labels, blinded inputs, the opaque-to-canonical mapping and the
    original source Parquet.  Split membership uses canonical ``sample_id`` values.
    Every input row is conserved before calibration and audit-excluded rows are
    omitted from the Phase-1 output Parquets.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - preparation dependency path
        raise RuntimeError("factorised frame preparation requires PyArrow") from exc
    input_paths = (
        teacher_labels_parquet_path,
        blinded_input_json_path,
        private_mapping_parquet_path,
        source_parquet_path,
        membership_json_path,
        split_public_manifest_json_path,
        bridge_exposure_register_json_path,
        legacy_proxy_parquet_path,
    )
    if any(not path.is_file() for path in input_paths):
        raise FileNotFoundError("all four teacher artefacts and split membership are required")
    if (
        not isinstance(expected_teacher_run_id, str)
        or len(expected_teacher_run_id) != 64
        or any(character not in "0123456789abcdef" for character in expected_teacher_run_id)
    ):
        raise ValueError("expected_teacher_run_id must be a lowercase SHA-256")
    try:
        output_root.relative_to(descriptor_root)
    except ValueError as exc:
        raise ValueError("output_root must be contained by descriptor_root") from exc
    membership = _read_json(membership_json_path, where="factorised split membership")
    expected_membership_keys = {
        "schema_version",
        "kind",
        "contract_digest",
        "source_count",
        "probability_designs",
        "rare_cell_support",
        "memberships",
        "membership_id",
    }
    if set(membership) != expected_membership_keys:
        raise ValueError("split membership top-level schema drifted")
    if membership.get("kind") != "modernbert-factorised-private-membership-v1":
        raise ValueError("split membership identity drifted")
    membership_base = {
        key: membership[key] for key in expected_membership_keys - {"membership_id"}
    }
    if membership.get("membership_id") != canonical_sha256(membership_base):
        raise FactorisedExperimentContractError("split membership digest drifted")
    raw_memberships = membership.get("memberships")
    if not isinstance(raw_memberships, list) or not raw_memberships:
        raise ValueError("split membership must contain a non-empty memberships list")

    split_public = _read_json(
        split_public_manifest_json_path, where="factorised public split manifest"
    )
    if split_public.get("kind") != "modernbert-factorised-evidence-frames-v1":
        raise ValueError("factorised public split manifest identity drifted")
    if membership.get("probability_designs") != split_public.get(
        "probability_designs"
    ):
        raise FactorisedExperimentContractError(
            "private/public probability-design summaries drifted"
        )
    public_base = {
        key: value for key, value in split_public.items() if key != "manifest_id"
    }
    if split_public.get("manifest_id") != canonical_sha256(public_base):
        raise FactorisedExperimentContractError("public split manifest digest drifted")
    if (
        split_public.get("private_membership_id") != membership["membership_id"]
        or split_public.get("private_membership_sha256")
        != file_sha256(membership_json_path)
        or split_public.get("contract_digest") != membership["contract_digest"]
        or split_public.get("source_count") != membership["source_count"]
        or split_public.get("row_conservation") is not True
        or split_public.get("group_disjointness") is not True
    ):
        raise FactorisedExperimentContractError(
            "public/private split-manifest bindings disagree"
        )
    assert_metadata_only(split_public, where="factorised public split manifest")

    bridge_register = validate_private_exposure_register(
        _read_json(bridge_exposure_register_json_path, where="bridge exposure register"),
        scope="bridge",
    )
    legacy_overlap_audit = build_legacy_overlap_audit(
        development_proxy_parquet_path=legacy_proxy_parquet_path,
        development_proxy_descriptor=legacy_proxy_descriptor,
        source_parquet_path=source_parquet_path,
        source_parquet_descriptor=source_parquet_descriptor,
        descriptor_root=descriptor_root,
    )
    split_contract = split_public.get("contract")
    if not isinstance(split_contract, Mapping):
        raise ValueError("public split manifest lacks its frozen contract")
    split_bindings = split_contract.get("input_bindings")
    if not isinstance(split_bindings, Mapping):
        raise ValueError("public split manifest lacks exposure-register bindings")
    expected_split_auxiliary_bindings = {
        "bridge_exposure_register_file_sha256": file_sha256(
            bridge_exposure_register_json_path
        ),
        "bridge_exposure_register_content_sha256": bridge_register[
            "thread_set_sha256"
        ],
        "legacy_development_proxy_file_sha256": legacy_overlap_audit[
            "legacy_proxy_file_sha256"
        ],
        "legacy_development_proxy_schema_metadata_sha256": legacy_overlap_audit[
            "legacy_proxy_schema_metadata_sha256"
        ],
        "legacy_development_proxy_sample_id_set_sha256": legacy_overlap_audit[
            "legacy_development_proxy_id_set_sha256"
        ],
        "legacy_development_proxy_surface_set_sha256": legacy_overlap_audit[
            "legacy_development_proxy_surface_set_sha256"
        ],
        "legacy_locked_proxy_file_sha256": legacy_overlap_audit[
            "legacy_proxy_file_sha256"
        ],
        "legacy_locked_proxy_schema_metadata_sha256": legacy_overlap_audit[
            "legacy_proxy_schema_metadata_sha256"
        ],
        "legacy_locked_proxy_sample_id_set_sha256": legacy_overlap_audit[
            "legacy_locked_proxy_id_set_sha256"
        ],
        "legacy_locked_proxy_surface_set_sha256": legacy_overlap_audit[
            "legacy_locked_proxy_surface_set_sha256"
        ],
    }
    if any(
        split_bindings.get(key) != value
        for key, value in expected_split_auxiliary_bindings.items()
    ):
        raise FactorisedExperimentContractError(
            "bridge/legacy overlap evidence does not match the split contract"
        )

    label_columns = (
        "opaque_id",
        "codability",
        "relevance",
        "label_json",
        "quality_tier",
        "primary_training_eligible",
    )
    label_table = pq.read_table(teacher_labels_parquet_path)
    if tuple(label_table.column_names) != label_columns:
        raise ValueError("teacher labels Parquet column contract drifted")
    metadata = label_table.schema.metadata or {}
    if metadata.get(b"run_id", b"").decode("utf-8") != expected_teacher_run_id:
        raise FactorisedExperimentContractError("teacher labels run binding drifted")
    label_rows = label_table.to_pylist()
    label_by_opaque: dict[str, dict[str, Any]] = {}
    label_hash_by_opaque: dict[str, str] = {}
    label_validator = load_v2_label_validator()
    for index, row in enumerate(label_rows):
        opaque_id = row.get("opaque_id")
        if (
            not isinstance(opaque_id, str)
            or not opaque_id
            or opaque_id in label_by_opaque
        ):
            raise ValueError("teacher labels contain an invalid or duplicate opaque ID")
        try:
            parsed_label = json.loads(row.get("label_json"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"teacher label row {index} contains invalid JSON") from exc
        if not isinstance(parsed_label, dict):
            raise ValueError("teacher label must decode to an object")
        clean_label = validate_v2_label(parsed_label, validator=label_validator)
        # Build the feature once as an integration check against the exact masks that
        # the training collator will later consume.
        encoded_probe = build_factorised_record(
            item_id=opaque_id,
            row={"target_text": "probe", "parent_context": None, "submission_context": None},
            label=clean_label,
            separator="[SEP]",
        )
        expected_codability = "codable" if encoded_probe.encoding.codability_mask else (
            "not_codable"
        )
        expected_relevance = (
            None
            if not encoded_probe.encoding.relevance_mask
            else "material"
            if encoded_probe.encoding.relevance_label == 1
            else "not_material"
        )
        if (
            row.get("codability") != expected_codability
            or row.get("relevance") != expected_relevance
            or type(row.get("primary_training_eligible")) is not bool
            or row.get("quality_tier")
            not in {"exact_consensus", "blind_majority", "informed_adjudication"}
        ):
            raise FactorisedExperimentContractError(
                "teacher label columns disagree with the canonical label or quality contract"
            )
        label_by_opaque[opaque_id] = {**row, "parsed_label": clean_label}
        label_hash_by_opaque[opaque_id] = label_sha256(
            clean_label, validator=label_validator
        )

    blinded = _read_json(blinded_input_json_path, where="teacher blinded input")
    blinded_keys = {
        "schema_version",
        "kind",
        "packet_id",
        "rubric_sha256",
        "label_schema_sha256",
        "rows",
    }
    if set(blinded) != blinded_keys or blinded.get("kind") != (
        "sol-teacher-10k-v2-blinded-input-v1"
    ):
        raise ValueError("teacher blinded-input contract drifted")
    blinded_rows = blinded.get("rows")
    if not isinstance(blinded_rows, list) or not blinded_rows:
        raise ValueError("teacher blinded input contains no rows")
    blinded_by_opaque: dict[str, dict[str, Any]] = {}
    blinded_order: list[str] = []
    for row in blinded_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "source_sample_id",
            "target_text",
            "submission_context",
            "parent_context",
        }:
            raise ValueError("teacher blinded row schema drifted")
        opaque_id = row.get("source_sample_id")
        if (
            not isinstance(opaque_id, str)
            or not opaque_id
            or opaque_id in blinded_by_opaque
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
            or any(
                row.get(field) is not None and not isinstance(row[field], str)
                for field in ("submission_context", "parent_context")
            )
        ):
            raise ValueError("teacher blinded input contains an invalid row")
        blinded_by_opaque[opaque_id] = dict(row)
        blinded_order.append(opaque_id)

    mapping_columns = ("opaque_id", "sample_id", "thread_id", "packet_order")
    mapping_table = pq.read_table(private_mapping_parquet_path)
    if tuple(mapping_table.column_names) != mapping_columns:
        raise ValueError("teacher private mapping column contract drifted")
    mapping_rows = sorted(mapping_table.to_pylist(), key=lambda row: row["packet_order"])
    if [row.get("packet_order") for row in mapping_rows] != list(range(len(mapping_rows))):
        raise FactorisedExperimentContractError("teacher private mapping order drifted")
    if [row.get("opaque_id") for row in mapping_rows] != blinded_order:
        raise FactorisedExperimentContractError("mapping and blinded-input order drifted")
    mapping_by_sample: dict[str, dict[str, Any]] = {}
    mapping_by_opaque: dict[str, dict[str, Any]] = {}
    seen_threads: set[str] = set()
    for row in mapping_rows:
        opaque_id = row.get("opaque_id")
        sample_id = row.get("sample_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(opaque_id, str)
            or not opaque_id
            or opaque_id in mapping_by_opaque
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in mapping_by_sample
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in seen_threads
        ):
            raise ValueError("teacher private mapping violates row/thread uniqueness")
        mapping_by_opaque[opaque_id] = row
        mapping_by_sample[sample_id] = row
        seen_threads.add(thread_id)

    source_columns = SOURCE_ROW_HASH_FIELDS
    source_table = pq.read_table(source_parquet_path, columns=list(source_columns))
    source_rows = source_table.to_pylist()
    source_by_sample: dict[str, dict[str, Any]] = {}
    source_hash_by_sample: dict[str, str] = {}
    source_threads: set[str] = set()
    for row in source_rows:
        sample_id = row.get("sample_id")
        thread_id = row.get("thread_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in source_by_sample
            or not isinstance(thread_id, str)
            or not thread_id
            or thread_id in source_threads
            or not isinstance(row.get("target_text"), str)
            or not row["target_text"]
            or any(
                row.get(field) is not None and not isinstance(row[field], str)
                for field in ("submission_context", "parent_context")
            )
            or not isinstance(row.get("subreddit"), str)
            or not row["subreddit"]
            or type(row.get("year")) is not int
            or not isinstance(row.get("content_type"), str)
            or not row["content_type"]
            or not isinstance(row.get("retrieval_mode"), str)
            or not row["retrieval_mode"]
        ):
            raise ValueError("source Parquet contains an invalid or duplicate row/thread")
        clean_source = {field: row[field] for field in source_columns}
        source_by_sample[sample_id] = clean_source
        source_hash_by_sample[sample_id] = source_row_sha256(clean_source)
        source_threads.add(thread_id)

    row_count = len(label_rows)
    if (
        row_count != EXPECTED_SOURCE_ROWS
        or len(blinded_rows) != row_count
        or len(mapping_rows) != row_count
        or len(source_rows) != row_count
        or int(membership.get("source_count", -1)) != row_count
        or set(label_by_opaque) != set(blinded_by_opaque)
        or set(label_by_opaque) != set(mapping_by_opaque)
        or set(mapping_by_sample) != set(source_by_sample)
    ):
        raise FactorisedExperimentContractError(
            "teacher/source/split inputs fail exact row conservation"
        )
    if [row["opaque_id"] for row in label_rows] != blinded_order:
        raise FactorisedExperimentContractError("teacher labels and packet order drifted")
    for opaque_id, mapping_row in mapping_by_opaque.items():
        source = source_by_sample[mapping_row["sample_id"]]
        blinded_row = blinded_by_opaque[opaque_id]
        if (
            source["thread_id"] != mapping_row["thread_id"]
            or any(
                source[field] != blinded_row[field]
                for field in ("target_text", "submission_context", "parent_context")
            )
        ):
            raise FactorisedExperimentContractError(
                "teacher mapping, blinded text/context and source Parquet disagree"
            )

    allowed_frames = {"training", "development", "calibration", "audit_excluded"}
    membership_by_id: dict[str, dict[str, Any]] = {}
    frame_counts = {frame: 0 for frame in allowed_frames}
    for index, row in enumerate(raw_memberships):
        if not isinstance(row, Mapping):
            raise ValueError(f"memberships[{index}] must be an object")
        item_id = row.get("item_id")
        thread_id = row.get("thread_id")
        frame = row.get("frame")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in membership_by_id
            or not isinstance(thread_id, str)
            or not thread_id
            or frame not in allowed_frames
        ):
            raise ValueError("split membership contains an invalid or duplicate member")
        membership_by_id[item_id] = dict(row)
        frame_counts[frame] += 1

    if set(membership_by_id) != set(mapping_by_sample):
        raise FactorisedExperimentContractError("split membership does not conserve source IDs")
    public_frames = split_public.get("frames")
    if not isinstance(public_frames, Mapping) or set(public_frames) != allowed_frames:
        raise ValueError("public split manifest frame inventory drifted")
    for frame in allowed_frames:
        summary = public_frames[frame]
        if (
            not isinstance(summary, Mapping)
            or summary.get("row_count") != frame_counts[frame]
            or summary.get("group_count") != frame_counts[frame]
        ):
            raise FactorisedExperimentContractError(
                "public split manifest frame counts disagree with membership"
            )
    if (
        split_public.get("input_item_set_digest")
        != canonical_sha256(sorted(membership_by_id))
        or split_public.get("input_group_set_digest")
        != canonical_sha256(sorted(row["thread_id"] for row in membership_by_id.values()))
    ):
        raise FactorisedExperimentContractError(
            "public split manifest item/thread set bindings disagree"
        )
    bridge_threads = set(bridge_register["thread_ids"])
    if not bridge_threads <= source_threads:
        raise FactorisedExperimentContractError(
            "bridge exposure register contains threads outside the canonical 10k"
        )
    for member in membership_by_id.values():
        if (
            member.get("bridge_exposed")
            is not (member["thread_id"] in bridge_threads)
            or "legacy_development_exposed" in member
            or "legacy_locked_exposed" in member
        ):
            raise FactorisedExperimentContractError(
                "split membership exposure flags disagree with the sole bridge register"
            )
        if member["frame"] in {"calibration", "development"} and member[
            "bridge_exposed"
        ]:
            raise FactorisedExperimentContractError(
                "bridge-exposed thread entered an evaluation frame"
            )
    selected_ids = {
        item_id
        for item_id, row in membership_by_id.items()
        if row["frame"] in {"training", "development", "calibration"}
    }
    if not selected_ids:
        raise FactorisedExperimentContractError("private evidence-frame membership is empty")
    frames: dict[str, list[dict[str, Any]]] = {
        "training": [],
        "development": [],
        "calibration": [],
    }
    threads: dict[str, set[str]] = {
        "training": set(),
        "development": set(),
        "calibration": set(),
    }
    for item_id, member in membership_by_id.items():
        mapping_row = mapping_by_sample[item_id]
        label_row = label_by_opaque[mapping_row["opaque_id"]]
        source = source_by_sample[item_id]
        if (
            member["thread_id"] != mapping_row["thread_id"]
            or member.get("quality_tier") != label_row["quality_tier"]
            or member.get("primary_training_eligible")
            is not label_row["primary_training_eligible"]
            or member.get("label_sha256")
            != label_hash_by_opaque[mapping_row["opaque_id"]]
            or member.get("source_row_sha256") != source_hash_by_sample[item_id]
        ):
            raise FactorisedExperimentContractError(
                "split membership row bindings disagree with teacher/source inputs"
            )
    for item_id in sorted(selected_ids):
        member = membership_by_id[item_id]
        mapping_row = mapping_by_sample[item_id]
        teacher = label_by_opaque[mapping_row["opaque_id"]]
        source = source_by_sample[item_id]
        if teacher.get("primary_training_eligible") is not True:
            raise FactorisedExperimentContractError(
                "train/development membership contains an ineligible teacher row"
            )
        frame = member["frame"]
        thread_id = member["thread_id"]
        threads[frame].add(thread_id)
        output_row = {
            "item_id": item_id,
            "frame": frame,
            "thread_id": thread_id,
            "target_text": source["target_text"],
            "parent_context": source["parent_context"],
            "submission_context": source["submission_context"],
            "label_json": teacher["label_json"],
            "selection_component": member["selection_component"],
            "selection_stratum": member["selection_stratum"],
            "inclusion_probability_numerator": member[
                "inclusion_probability_numerator"
            ],
            "inclusion_probability_denominator": member[
                "inclusion_probability_denominator"
            ],
            "inclusion_probability": member["inclusion_probability"],
            "probability_scope": member["probability_scope"],
        }
        if frame == "calibration":
            output_row["quality_tier"] = teacher["quality_tier"]
        frames[frame].append(output_row)
    if any(
        threads[left] & threads[right]
        for left, right in (
            ("training", "development"),
            ("training", "calibration"),
            ("development", "calibration"),
        )
    ):
        raise FactorisedExperimentContractError("private evidence-frame threads overlap")

    output_root.mkdir(parents=True, exist_ok=True)
    overlap_audit_path = output_root / "legacy-overlap-audit.json"
    _write_json_atomic(overlap_audit_path, legacy_overlap_audit)
    descriptors: dict[str, dict[str, Any]] = {}
    schema = pa.schema(
        [
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("frame", pa.string(), nullable=False),
            pa.field("thread_id", pa.string(), nullable=False),
            pa.field("target_text", pa.string(), nullable=False),
            pa.field("parent_context", pa.string()),
            pa.field("submission_context", pa.string()),
            pa.field("label_json", pa.string(), nullable=False),
            pa.field("selection_component", pa.string(), nullable=False),
            pa.field("selection_stratum", pa.string()),
            pa.field("inclusion_probability_numerator", pa.int64()),
            pa.field("inclusion_probability_denominator", pa.int64()),
            pa.field("inclusion_probability", pa.float64()),
            pa.field("probability_scope", pa.string()),
        ]
    )
    for frame in ("training", "development", "calibration"):
        path = output_root / f"{frame}.parquet"
        temporary = path.with_suffix(".parquet.incomplete")
        frame_schema = (
            pa.schema([*schema, pa.field("quality_tier", pa.string(), nullable=False)])
            if frame == "calibration"
            else schema
        )
        expected_table = pa.Table.from_pylist(frames[frame], schema=frame_schema)
        if path.exists():
            existing = pq.read_table(path)
            if not existing.equals(expected_table):
                raise FactorisedExperimentContractError(
                    f"existing immutable {frame} frame differs"
                )
        else:
            if temporary.exists():
                raise FactorisedExperimentContractError(
                    f"stale incomplete {frame} frame exists"
                )
            pq.write_table(expected_table, temporary, compression="zstd")
            os.replace(temporary, path)
        descriptors[frame] = {
            "relative_path": str(path.relative_to(descriptor_root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "row_count": len(frames[frame]),
            "frame": frame,
            "thread_set_sha256": _thread_set_sha256(list(threads[frame])),
        }
        if frame == "development":
            selection_counts: dict[str, int] = {}
            for row in frames[frame]:
                component = row["selection_component"]
                selection_counts[component] = selection_counts.get(component, 0) + 1
            expected_counts = {
                "development_context_available": 75,
                "development_multi_target": 75,
                "development_probability": 300,
                "development_rare_target_stance": 150,
            }
            if selection_counts != expected_counts:
                raise FactorisedExperimentContractError(
                    "development selection-component design drifted"
                )
            descriptors[frame]["primary_probability_row_count"] = 300
            descriptors[frame]["selection_component_counts"] = expected_counts
    result = {
        "teacher_labels_sha256": file_sha256(teacher_labels_parquet_path),
        "blinded_input_sha256": file_sha256(blinded_input_json_path),
        "private_mapping_sha256": file_sha256(private_mapping_parquet_path),
        "source_parquet_sha256": file_sha256(source_parquet_path),
        "membership_json_sha256": file_sha256(membership_json_path),
        "split_public_manifest_sha256": file_sha256(
            split_public_manifest_json_path
        ),
        "bridge_exposure_register_sha256": file_sha256(
            bridge_exposure_register_json_path
        ),
        "legacy_proxy_sha256": file_sha256(legacy_proxy_parquet_path),
        "membership_id": membership["membership_id"],
        "source_rows": row_count,
        "membership_frame_counts": dict(sorted(frame_counts.items())),
        "probability_designs": membership["probability_designs"],
        "selected_rows": len(selected_ids),
        **_prepared_frame_descriptors(descriptors),
        "legacy_overlap_audit": _descriptor(overlap_audit_path, root=descriptor_root),
        "thread_overlap": 0,
    }
    assert_metadata_only(result, where="factorised frame preparation summary")
    return result


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FactorisedExperimentContractError(
                "existing immutable JSON artefact differs"
            )
        return
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise FactorisedExperimentContractError("stale incomplete JSON exists")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _descriptor(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _validate_descriptor(root: Path, value: Mapping[str, Any], *, where: str) -> Path:
    clean = artifact_descriptor(value, where=where)
    path = root / clean["relative_path"]
    if (
        not path.is_file()
        or path.stat().st_size != clean["bytes"]
        or file_sha256(path) != clean["sha256"]
    ):
        raise FactorisedExperimentContractError(f"{where} artefact is missing or corrupt")
    return path


def _trial_roots(
    *, volume_root: Path, experiment_run_id: str, component: str, trial_id: str
) -> tuple[Path, Path]:
    run = volume_root / NAMESPACE / f"run={experiment_run_id}" / "phase=fixed-comparison"
    final = run / f"component={component}" / f"trial={trial_id}"
    attempts = run / ".attempts" / f"component={component}" / f"trial={trial_id}"
    return final, attempts


def _new_attempt_id() -> str:
    """Return an execution-unique identifier for preemption-safe staging."""

    return uuid.uuid4().hex


def _validate_attempt_id(value: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError("factorised trial attempt ID must be 32 lowercase hex characters")
    return value


def _attempt_binding(
    *,
    experiment_run_id: str,
    phase_run_id: str,
    run_manifest_sha256: str,
    trial: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "kind": TRIAL_ATTEMPT_KIND,
        "attempt_id": _validate_attempt_id(attempt_id),
        "experiment_run_id": experiment_run_id,
        "phase_run_id": phase_run_id,
        "run_manifest_sha256": run_manifest_sha256,
        "trial_id": trial["trial_id"],
        "trial_spec_sha256": canonical_sha256(trial),
        "component": trial["component"],
        "locked_test_rows_accessed": 0,
    }


def _inspect_attempt_evidence(
    attempts_root: Path,
    *,
    experiment_run_id: str,
    phase_run_id: str,
    run_manifest_sha256: str,
    trial: Mapping[str, Any],
) -> int:
    """Validate attempt bindings while allowing arbitrary interrupted work files."""

    if not attempts_root.exists():
        return 0
    if not attempts_root.is_dir():
        raise FactorisedExperimentContractError("factorised trial attempts root is not a directory")
    count = 0
    for path in sorted(attempts_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("attempt="):
            raise FactorisedExperimentContractError("factorised trial attempt inventory drifted")
        attempt_id = _validate_attempt_id(path.name.removeprefix("attempt="))
        marker_path = path / "attempt.json"
        # An empty directory is valid evidence of a process killed between mkdir and
        # marker publication. It is retained but never reused by a later execution.
        if marker_path.exists():
            marker = _read_json(marker_path, where="factorised trial attempt")
            expected = _attempt_binding(
                experiment_run_id=experiment_run_id,
                phase_run_id=phase_run_id,
                run_manifest_sha256=run_manifest_sha256,
                trial=trial,
                attempt_id=attempt_id,
            )
            if marker != expected:
                raise FactorisedExperimentContractError(
                    "factorised trial attempt binding drifted"
                )
        count += 1
    return count


def _record_attempt_outcome(
    attempt_root: Path, *, name: str, phase: str, error_type: str | None = None
) -> None:
    """Best-effort metadata marker which never masks the authoritative outcome."""

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "modernbert-factorised-trial-attempt-outcome-v1",
        "status": name,
        "phase": phase,
        "locked_test_rows_accessed": 0,
    }
    if error_type is not None:
        payload["error_type"] = error_type
    # The original training/publication exception is more important than an
    # auxiliary marker. The attempt directory itself remains durable evidence.
    with suppress(Exception):
        _write_json_atomic(attempt_root / f"{name}.json", payload)


def _validate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "experiment_run_id",
        "phase_run_id",
        "run_manifest_sha256",
        "experiment_contract",
        "trial_spec",
        "trial_spec_sha256",
        "locked_test_rows_accessed",
    }
    if set(job) != required or job.get("locked_test_rows_accessed") != 0:
        raise ValueError("factorised training job schema or evidence boundary drifted")
    experiment = job["experiment_contract"]
    trial = job["trial_spec"]
    if not isinstance(experiment, Mapping) or not isinstance(trial, Mapping):
        raise ValueError("factorised training job lacks contract or trial")
    clean_trial = validate_trial_spec(trial, experiment=experiment)
    if job["trial_spec_sha256"] != canonical_sha256(clean_trial):
        raise FactorisedExperimentContractError("factorised training job trial digest drifted")
    return dict(job)


def validate_trial_artifacts(
    trial_root: Path,
    receipt: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase_run_id: str,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    clean = validate_trial_receipt(
        experiment,
        trial_spec,
        receipt,
        phase_run_id=phase_run_id,
        run_manifest_sha256=run_manifest_sha256,
    )
    paths = {
        key: _validate_descriptor(trial_root, descriptor, where=key)
        for key, descriptor in clean["artifacts"].items()
    }
    predictions = _read_json(paths["private_development_predictions"], where="predictions")
    validate_private_development_predictions(
        predictions, experiment=experiment, trial_spec=trial_spec
    )
    metrics = _read_json(paths["metrics"], where="metrics")
    if metrics.get("aggregate_metrics") != clean["aggregate_metrics"]:
        raise FactorisedExperimentContractError("metrics artefact and receipt disagree")
    expected_files = {
        "receipt.json",
        *(descriptor["relative_path"] for descriptor in clean["artifacts"].values()),
    }
    observed = {
        str(path.relative_to(trial_root)) for path in trial_root.rglob("*") if path.is_file()
    }
    if observed != expected_files:
        raise FactorisedExperimentContractError("final trial artefact inventory drifted")
    return clean


def inspect_trial_output(
    *, manifest: Mapping[str, Any], trial_id: str, volume_root: Path
) -> dict[str, Any]:
    clean = validate_run_manifest(manifest)
    by_id = {trial["trial_id"]: trial for trial in clean["trials"]}
    if trial_id not in by_id:
        raise ValueError("trial_id is not registered")
    trial = by_id[trial_id]
    final, attempts = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=clean["experiment_run_id"],
        component=trial["component"],
        trial_id=trial_id,
    )
    attempt_count = _inspect_attempt_evidence(
        attempts,
        experiment_run_id=clean["experiment_run_id"],
        phase_run_id=clean["phase_run_id"],
        run_manifest_sha256=canonical_sha256(clean),
        trial=trial,
    )
    if not final.exists():
        if attempt_count:
            return {
                "trial_id": trial_id,
                "component": trial["component"],
                "status": "incomplete",
                "attempt_count": attempt_count,
            }
        return {"trial_id": trial_id, "component": trial["component"], "status": "missing"}
    receipt = _read_json(final / "receipt.json", where="factorised receipt")
    clean_receipt = validate_trial_artifacts(
        final,
        receipt,
        experiment=clean["experiment_contract"],
        trial_spec=trial,
        phase_run_id=clean["phase_run_id"],
        run_manifest_sha256=canonical_sha256(clean),
    )
    return {
        "trial_id": trial_id,
        "component": trial["component"],
        "status": "complete",
        "receipt_id": clean_receipt["receipt_id"],
        "estimated_cost_usd": clean_receipt["estimated_cost_usd"],
        "attempt_count": attempt_count,
    }


def aggregate_trial_inspections(
    *, manifest: Mapping[str, Any], inspections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    clean = validate_run_manifest(manifest)
    expected = {trial["trial_id"] for trial in clean["trials"]}
    by_id: dict[str, Mapping[str, Any]] = {}
    for inspection in inspections:
        trial_id = inspection.get("trial_id")
        if trial_id not in expected or trial_id in by_id:
            raise ValueError("inspection contains unknown or duplicate trial")
        if inspection.get("status") not in {"missing", "incomplete", "complete"}:
            raise ValueError("inspection status is invalid")
        by_id[trial_id] = inspection
    if set(by_id) != expected:
        raise ValueError("inspection does not exactly cover the trial inventory")
    complete = sorted(key for key, value in by_id.items() if value["status"] == "complete")
    incomplete = sorted(key for key, value in by_id.items() if value["status"] == "incomplete")
    missing = sorted(key for key, value in by_id.items() if value["status"] == "missing")
    cost = sum(
        (float(by_id[key].get("estimated_cost_usd", 0)) for key in complete), start=0.0
    )
    result = {
        "status": "complete" if len(complete) == len(expected) else "incomplete",
        "expected_trials": len(expected),
        "complete_trials": len(complete),
        "incomplete_trials": len(incomplete),
        "missing_trials": len(missing),
        "complete_trial_ids": complete,
        "incomplete_trial_ids": incomplete,
        "missing_trial_ids": missing,
        "estimated_cost_usd": f"{cost:.6f}",
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="factorised trial inspection")
    return result


NATURAL_DEVELOPMENT_ESTIMAND = (
    "unexposed primary-eligible factorised-v2 engineering population"
)


def _natural_design_from_development_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    design: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("frame") != "development":
            raise FactorisedExperimentContractError(
                f"natural development design row {index} has missing or drifted frame identity"
            )
        if row.get("selection_component") != "development_probability":
            continue
        missing = [field for field in NATURAL_DESIGN_DIGEST_FIELDS if field not in row]
        if missing:
            raise FactorisedExperimentContractError(
                "natural development design row omits required design fields: "
                + ", ".join(sorted(missing))
            )
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in design:
            raise FactorisedExperimentContractError(
                "natural development design contains an invalid or duplicate item ID"
            )
        design[item_id] = {
            field: row[field] for field in NATURAL_DESIGN_DIGEST_FIELDS
        }
    if len(design) != 300:
        raise FactorisedExperimentContractError(
            "natural development design must bind exactly 300 unique rows"
        )
    return design


def _load_natural_development_weighting(
    *,
    bindings: Mapping[str, Any],
    volume_root: Path,
) -> tuple[NaturalArmWeightingConfig, dict[str, Any]]:
    descriptor = bindings.get("split_public_manifest")
    if not isinstance(descriptor, Mapping):
        raise ValueError("experiment lacks the split-public-manifest binding")
    path = volume_root / str(descriptor.get("relative_path"))
    split_public = _read_json(path, where="factorised public split manifest")
    probability_designs = split_public.get("probability_designs")
    if not isinstance(probability_designs, Mapping):
        raise ValueError("split manifest lacks probability-design summaries")
    summary = probability_designs.get("development_probability")
    if not isinstance(summary, Mapping):
        raise ValueError("split manifest lacks the natural development design")
    weighting = natural_arm_weighting_config_from_summary(
        summary,
        estimand=NATURAL_DEVELOPMENT_ESTIMAND,
        include_unweighted_conditional_diagnostics=False,
    )
    registered = bindings.get("development_probability_weighting")
    expected = {
        "design_summary": dict(summary),
        "design_summary_sha256": canonical_sha256(summary),
        "weighting_config_digest": weighting.digest(),
    }
    if registered != expected:
        raise FactorisedExperimentContractError(
            "natural development weighting drifted from the frozen experiment"
        )
    return weighting, dict(summary)


def _score_checkpoint(
    *,
    component: str,
    reference: Mapping[str, Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    selection_component_by_item: Mapping[str, str],
    natural_design: Mapping[str, Mapping[str, Any]],
    natural_weighting: NaturalArmWeightingConfig,
    selected_epoch: int,
) -> dict[str, Any]:
    component = _component(component)
    if set(selection_component_by_item) != set(reference):
        raise FactorisedExperimentContractError(
            "development selection components do not conserve reference IDs"
        )
    rows_by_id = {row["item_id"]: row for row in private_rows}
    if set(rows_by_id) != set(reference):
        raise FactorisedExperimentContractError(
            "checkpoint predictions do not conserve development IDs"
        )
    component_ids: dict[str, set[str]] = {}
    for item_id, selection_component in selection_component_by_item.items():
        component_ids.setdefault(selection_component, set()).add(item_id)
    primary_ids = component_ids.get("development_probability", set())
    if len(primary_ids) != 300:
        raise FactorisedExperimentContractError(
            "checkpoint scoring requires exactly 300 probability-primary rows"
        )

    def score_subset(item_ids: set[str], *, weighted: bool) -> dict[str, Any]:
        subset_reference = {item_id: reference[item_id] for item_id in sorted(item_ids)}
        subset_rows = [rows_by_id[item_id] for item_id in sorted(item_ids)]
        if component == "relevance":
            payload = {"rows": subset_rows}
            dummy_rows = [
                {
                    "item_id": item_id,
                    "target_presence_logits": [-20.0] * len(TARGET_CLASSES),
                    "stance_logits": [[0.0] * len(STANCE_CLASSES_B4)]
                    * len(ANALYTIC_TARGET_CLASSES),
                }
                for item_id in subset_reference
            ]
            return combine_component_predictions(
                reference=subset_reference,
                relevance_payload=payload,
                target_stance_payload={"rows": dummy_rows},
                representation="B4",
                natural_design=natural_design if weighted else None,
                natural_weighting=natural_weighting if weighted else None,
            )
        gold_relevance_rows = []
        for item_id, label in subset_reference.items():
            codable = label.get("codability") == "codable"
            material = codable and label.get("relevance") == "material"
            gold_relevance_rows.append(
                {"item_id": item_id, "relevance_logit": 20.0 if material else -20.0}
            )
        return combine_component_predictions(
            reference=subset_reference,
            relevance_payload={"rows": gold_relevance_rows},
            target_stance_payload={"rows": subset_rows},
            representation="B4" if component.endswith("b4") else "B2",
            natural_design=natural_design if weighted else None,
            natural_weighting=natural_weighting if weighted else None,
        )

    if set(natural_design) != primary_ids:
        raise FactorisedExperimentContractError(
            "natural design does not match the probability-primary rows"
        )
    primary_metrics = score_subset(primary_ids, weighted=True)
    weighted = primary_metrics["design_weighted"]
    component_metrics: dict[str, Any] = {}
    for name, item_ids in sorted(component_ids.items()):
        metrics = (
            primary_metrics
            if name == "development_probability"
            else score_subset(item_ids, weighted=False)
        )
        surface = (
            metrics["design_weighted"]
            if name == "development_probability"
            else metrics
        )
        if component == "relevance":
            component_metrics[name] = {
                "row_count": len(item_ids),
                "evidence_scope": metrics["evidence_scope"],
                "relevance_macro_f1": surface["relevance"]["macro_f1"] or 0.0,
                "material_recall": surface["relevance"]["material"]["recall"] or 0.0,
            }
        else:
            component_metrics[name] = {
                "row_count": len(item_ids),
                "evidence_scope": metrics["evidence_scope"],
                "conditional_tuple_micro_f1": surface["end_to_end"][
                    "target_stance_tuples"
                ]["micro"]["f1"]
                or 0.0,
            }
    if component == "relevance":
        macro = weighted["relevance"]["macro_f1"] or 0.0
        recall = weighted["relevance"]["material"]["recall"] or 0.0
        return {
            "selected_epoch": selected_epoch,
            "development_rows": len(reference),
            "primary_probability_rows": len(primary_ids),
            "invalid_outputs": primary_metrics["outputs"]["missing"],
            "relevance_macro_f1": macro,
            "material_recall": recall,
            "checkpoint_score": 0.5 * macro + 0.5 * recall,
            "natural_arm_weighting_config_digest": natural_weighting.digest(),
            "probability_design": primary_metrics["probability_design"],
            "development_component_metrics": component_metrics,
        }
    score = weighted["end_to_end"]["target_stance_tuples"]["micro"]["f1"] or 0.0
    return {
        "selected_epoch": selected_epoch,
        "development_rows": len(reference),
        "primary_probability_rows": len(primary_ids),
        "invalid_outputs": primary_metrics["outputs"]["missing"],
        "conditional_tuple_micro_f1": score,
        "checkpoint_score": score,
        "natural_arm_weighting_config_digest": natural_weighting.digest(),
        "probability_design": primary_metrics["probability_design"],
        "development_component_metrics": component_metrics,
    }


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise FactorisedExperimentContractError("stale incomplete checkpoint exists")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def execute_registered_gpu_trial(
    job: Mapping[str, Any], volume_root: Path, attempt_work_root: Path
) -> dict[str, Any]:
    """Execute one clean, non-resumable CUDA attempt for Modal's ``train_l4``."""

    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("factorised ModernBERT training requires the requested L4")
    try:
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - Modal-only dependency path
        raise RuntimeError("factorised training runtime dependencies are incomplete") from exc
    from reddit_china_stance.modernbert_trainer import (
        build_length_bucket_batches,
        seed_everything,
    )

    clean = _validate_job(job)
    trial = clean["trial_spec"]
    component = trial["component"]
    bindings = clean["experiment_contract"]["bindings"]
    train_path = volume_root / bindings["training_frame"]["relative_path"]
    development_path = volume_root / bindings["development_frame"]["relative_path"]
    train_rows, _ = load_private_frame(
        train_path, bindings["training_frame"], expected_frame="training"
    )
    development_rows, reference = load_private_frame(
        development_path,
        bindings["development_frame"],
        expected_frame="development",
    )
    natural_weighting, _ = _load_natural_development_weighting(
        bindings=bindings, volume_root=volume_root
    )
    natural_design = _natural_design_from_development_rows(development_rows)
    if {row["item_id"] for row in train_rows} & set(reference):
        raise FactorisedExperimentContractError("training/development item IDs overlap")

    seed = trial["optimiser_seed"]
    optimisation = build_optimisation_config(trial)
    seed_everything(seed)
    tokenizer = load_pinned_tokenizer()

    def encode_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        encoded = []
        for row in rows:
            label = json.loads(row["label_json"])
            feature = tokenise_factorised_record(
                tokenizer,
                item_id=row["item_id"],
                row=row,
                label=label,
            )
            if component != "relevance" and not feature["target_presence_mask"][0]:
                continue
            encoded.append(feature)
        if not encoded:
            raise FactorisedExperimentContractError("component training frame is empty")
        return encoded

    train_encoded = encode_rows(train_rows)
    # Development rows are not filtered: every component must publish one logit row
    # per exact development item for paired end-to-end closeout.
    development_encoded = []
    for row in development_rows:
        development_encoded.append(
            tokenise_factorised_record(
                tokenizer,
                item_id=row["item_id"],
                row=row,
                label=json.loads(row["label_json"]),
            )
        )
    collator = FactorisedDynamicPaddingCollator(tokenizer, component=component)
    development_batches = build_length_bucket_batches(
        [len(row["input_ids"]) for row in development_encoded],
        batch_size=optimisation.per_device_batch_size,
        seed=0,
        epoch=0,
    )
    development_loader = DataLoader(
        development_encoded,
        batch_sampler=development_batches,
        collate_fn=collator,
        num_workers=0,
    )

    model = create_component_model(component=component, config=optimisation).to("cuda")
    optimizer = create_adamw(model, optimisation)
    max_epochs = trial["config"]["max_epochs"]
    train_lengths = [len(row["input_ids"]) for row in train_encoded]
    batches_per_epoch = math.ceil(len(train_encoded) / optimisation.per_device_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / optimisation.gradient_accumulation_steps)
    total_updates = updates_per_epoch * max_epochs
    warmup_steps = int(total_updates * optimisation.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] | None = None
    checkpoint_path = attempt_work_root / "checkpoint.pt"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        train_batches = build_length_bucket_batches(
            train_lengths,
            batch_size=optimisation.per_device_batch_size,
            seed=seed,
            epoch=epoch,
        )
        train_loader = DataLoader(
            train_encoded,
            batch_sampler=train_batches,
            collate_fn=collator,
            num_workers=0,
        )
        trained = train_epoch(
            model,
            train_loader,
            optimizer,
            config=optimisation,
            device="cuda",
            scheduler=scheduler,
        )
        private_rows = collect_development_logits(
            model,
            development_loader,
            component=component,
            device="cuda",
            use_bf16=True,
        )
        metrics = _score_checkpoint(
            component=component,
            reference=reference,
            private_rows=private_rows,
            selection_component_by_item={
                row["item_id"]: row["selection_component"]
                for row in development_rows
            },
            natural_design=natural_design,
            natural_weighting=natural_weighting,
            selected_epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mean_loss": round(float(trained["mean_loss"]), 8),
                "checkpoint_score": round(float(metrics["checkpoint_score"]), 8),
            }
        )
        if float(metrics["checkpoint_score"]) > best_score:
            best_score = float(metrics["checkpoint_score"])
            best_metrics = metrics
            best_rows = private_rows
            _save_checkpoint_atomic(
                checkpoint_path,
                {
                    "schema_version": "1.0.0",
                    "kind": "modernbert-factorised-checkpoint-v2",
                    "experiment_run_id": clean["experiment_run_id"],
                    "phase_run_id": clean["phase_run_id"],
                    "trial_id": trial["trial_id"],
                    "component": component,
                    "optimiser_seed": seed,
                    "config_sha256": trial["config"]["config_sha256"],
                    "training_frame_sha256": trial["training_frame_sha256"],
                    "development_frame_sha256": trial["development_frame_sha256"],
                    "teacher_ledger_sha256": trial["teacher_ledger_sha256"],
                    "source_bundle_sha256": bindings["source_bundle_sha256"],
                    "dependency_lock_sha256": bindings["dependency_lock_sha256"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "selected_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
            )
    if best_metrics is None or best_rows is None:
        raise RuntimeError("factorised trial did not select a checkpoint")
    elapsed = time.monotonic() - started
    return {
        "checkpoint_path": str(checkpoint_path),
        "private_prediction_rows": best_rows,
        "aggregate_metrics": best_metrics,
        "epoch_history": history,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": elapsed,
        "gpu_seconds": elapsed,
    }


def run_training_trial(
    *,
    job: Mapping[str, Any],
    volume_root: Path,
    trial_executor: Any,
) -> dict[str, Any]:
    """Atomically publish one trial executed by the injected GPU runtime.

    The executor boundary is intentional: preparation and orchestration are testable
    without importing the GPU stack, while Modal supplies the concrete training loop.
    It must return checkpoint bytes/path, private logits and aggregate metrics already
    selected on the fresh v2 development frame. Every invocation receives a fresh
    attempt work root; interrupted optimiser state is evidence, never resume input.
    """

    clean = _validate_job(job)
    trial = clean["trial_spec"]
    final, attempts = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=clean["experiment_run_id"],
        component=trial["component"],
        trial_id=trial["trial_id"],
    )
    _inspect_attempt_evidence(
        attempts,
        experiment_run_id=clean["experiment_run_id"],
        phase_run_id=clean["phase_run_id"],
        run_manifest_sha256=clean["run_manifest_sha256"],
        trial=trial,
    )
    if final.exists():
        receipt = _read_json(final / "receipt.json", where="factorised receipt")
        valid = validate_trial_artifacts(
            final,
            receipt,
            experiment=clean["experiment_contract"],
            trial_spec=trial,
            phase_run_id=clean["phase_run_id"],
            run_manifest_sha256=clean["run_manifest_sha256"],
        )
        return {
            "status": "already_complete",
            "trial_id": trial["trial_id"],
            "receipt_id": valid["receipt_id"],
        }

    attempts.mkdir(parents=True, exist_ok=True)
    attempt_id = _validate_attempt_id(_new_attempt_id())
    attempt_root = attempts / f"attempt={attempt_id}"
    attempt_root.mkdir(exist_ok=False)
    work_root = attempt_root / "work"
    publication_root = attempt_root / "publication"
    phase = "initialisation"
    started = time.monotonic()
    try:
        _write_json_atomic(
            attempt_root / "attempt.json",
            _attempt_binding(
                experiment_run_id=clean["experiment_run_id"],
                phase_run_id=clean["phase_run_id"],
                run_manifest_sha256=clean["run_manifest_sha256"],
                trial=trial,
                attempt_id=attempt_id,
            ),
        )
        work_root.mkdir()
        phase = "training"
        result = trial_executor(clean, volume_root, work_root)
        if not isinstance(result, Mapping):
            raise RuntimeError("trial executor must return an object")
        checkpoint_path = Path(result["checkpoint_path"])
        if (
            checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
            or checkpoint_path.resolve().parent != work_root.resolve()
        ):
            raise RuntimeError("trial executor checkpoint must be inside its attempt work root")

        phase = "publication"
        publication_root.mkdir()
        published_checkpoint = publication_root / "checkpoint.pt"
        # Work and publication live on the same mounted Volume. Moving avoids
        # retaining a duplicate multi-gigabyte checkpoint after successful
        # publication while still preserving every interrupted attempt in place.
        os.replace(checkpoint_path, published_checkpoint)
        predictions = build_private_development_predictions(
            clean["experiment_contract"], trial, rows=result["private_prediction_rows"]
        )
        prediction_path = publication_root / "development-predictions.json"
        _write_json_atomic(prediction_path, predictions)
        metrics = dict(result["aggregate_metrics"])
        metrics_payload = {
            "schema_version": "1.0.0",
            "kind": TRIAL_METRICS_KIND,
            "component": trial["component"],
            "aggregate_metrics": metrics,
            "epoch_history": list(result.get("epoch_history", [])),
            "peak_gpu_bytes": int(result.get("peak_gpu_bytes", 0)),
        }
        assert_metadata_only(metrics_payload, where="factorised trial metrics")
        metrics_path = publication_root / "metrics.json"
        _write_json_atomic(metrics_path, metrics_payload)
        artifacts = {
            "checkpoint": _descriptor(published_checkpoint, root=publication_root),
            "metrics": _descriptor(metrics_path, root=publication_root),
            "private_development_predictions": _descriptor(
                prediction_path, root=publication_root
            ),
        }
        wall = float(result.get("wall_seconds", time.monotonic() - started))
        gpu = float(result.get("gpu_seconds", time.monotonic() - started))
        receipt = build_trial_receipt(
            clean["experiment_contract"],
            trial,
            phase_run_id=clean["phase_run_id"],
            run_manifest_sha256=clean["run_manifest_sha256"],
            artifacts=artifacts,
            metrics=metrics,
            wall_seconds=wall,
            gpu_seconds=gpu,
        )
        _write_json_atomic(publication_root / "receipt.json", receipt)
        validate_trial_artifacts(
            publication_root,
            receipt,
            experiment=clean["experiment_contract"],
            trial_spec=trial,
            phase_run_id=clean["phase_run_id"],
            run_manifest_sha256=clean["run_manifest_sha256"],
        )
        _write_json_atomic(
            attempt_root / "ready.json",
            {
                "schema_version": "1.0.0",
                "kind": "modernbert-factorised-trial-attempt-ready-v1",
                "attempt_id": attempt_id,
                "receipt_id": receipt["receipt_id"],
                "locked_test_rows_accessed": 0,
            },
        )

        phase = "promotion"
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(publication_root, final)
        except OSError:
            # A concurrent platform retry may have won the atomic promotion race.
            # Its final output is authoritative only after exact validation; this
            # attempt remains retained and is never merged with the winner.
            if not final.is_dir():
                raise
            existing_receipt = _read_json(
                final / "receipt.json", where="factorised receipt"
            )
            existing = validate_trial_artifacts(
                final,
                existing_receipt,
                experiment=clean["experiment_contract"],
                trial_spec=trial,
                phase_run_id=clean["phase_run_id"],
                run_manifest_sha256=clean["run_manifest_sha256"],
            )
            _record_attempt_outcome(
                attempt_root, name="superseded", phase="promotion"
            )
            return {
                "status": "already_complete",
                "trial_id": trial["trial_id"],
                "receipt_id": existing["receipt_id"],
            }
        validate_trial_artifacts(
            final,
            receipt,
            experiment=clean["experiment_contract"],
            trial_spec=trial,
            phase_run_id=clean["phase_run_id"],
            run_manifest_sha256=clean["run_manifest_sha256"],
        )
        _record_attempt_outcome(attempt_root, name="promoted", phase="promotion")
        return {
            "status": "complete",
            "trial_id": trial["trial_id"],
            "component": trial["component"],
            "receipt_id": receipt["receipt_id"],
            "estimated_cost_usd": receipt["estimated_cost_usd"],
        }
    except Exception as exc:
        _record_attempt_outcome(
            attempt_root,
            name="failed",
            phase=phase,
            error_type=type(exc).__name__,
        )
        # Attempt-scoped evidence is retained, but it never blocks a clean retry.
        raise


def _load_trial_payload(
    *, manifest: Mapping[str, Any], trial: Mapping[str, Any], volume_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    final, _ = _trial_roots(
        volume_root=volume_root,
        experiment_run_id=manifest["experiment_run_id"],
        component=trial["component"],
        trial_id=trial["trial_id"],
    )
    receipt = _read_json(final / "receipt.json", where="factorised receipt")
    clean_receipt = validate_trial_artifacts(
        final,
        receipt,
        experiment=manifest["experiment_contract"],
        trial_spec=trial,
        phase_run_id=manifest["phase_run_id"],
        run_manifest_sha256=canonical_sha256(manifest),
    )
    prediction_path = final / clean_receipt["artifacts"]["private_development_predictions"][
        "relative_path"
    ]
    payload = validate_private_development_predictions(
        _read_json(prediction_path, where="factorised private logits"),
        experiment=manifest["experiment_contract"],
        trial_spec=trial,
    )
    return clean_receipt, payload


def closeout_fixed_comparison(
    *, manifest: Mapping[str, Any], volume_root: Path
) -> dict[str, Any]:
    """Pair relevance+B4/B2 by seed and publish only the aggregate gate."""

    clean = validate_run_manifest(manifest)
    development_descriptor = clean["experiment_contract"]["bindings"]["development_frame"]
    development_path = volume_root / development_descriptor["relative_path"]
    development_rows, reference = load_private_frame(
        development_path,
        development_descriptor,
        expected_frame="development",
    )
    natural_weighting, natural_summary = _load_natural_development_weighting(
        bindings=clean["experiment_contract"]["bindings"],
        volume_root=volume_root,
    )
    natural_design = _natural_design_from_development_rows(development_rows)
    component_ids: dict[str, set[str]] = {}
    for row in development_rows:
        component_ids.setdefault(row["selection_component"], set()).add(row["item_id"])
    primary_ids = component_ids.get("development_probability", set())
    enrichment_ids = set(reference) - primary_ids
    if len(primary_ids) != 300 or len(enrichment_ids) != 300:
        raise FactorisedExperimentContractError(
            "closeout requires the exact 300 probability / 300 enrichment split"
        )

    def subset_payload(payload: Mapping[str, Any], ids: set[str]) -> dict[str, Any]:
        return {
            **{key: value for key, value in payload.items() if key != "rows"},
            "rows": [row for row in payload["rows"] if row["item_id"] in ids],
        }

    def subset_reference(ids: set[str]) -> dict[str, Mapping[str, Any]]:
        return {item_id: reference[item_id] for item_id in sorted(ids)}
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    receipt_ids: list[str] = []
    for trial in clean["trials"]:
        receipt, payload = _load_trial_payload(
            manifest=clean, trial=trial, volume_root=volume_root
        )
        by_key[(trial["component"], trial["optimiser_seed"])] = payload
        receipt_ids.append(receipt["receipt_id"])
    paired = []
    diagnostics: dict[str, dict[str, list[float]]] = {
        selection: {"B4": [], "B2": []} for selection in sorted(component_ids)
    }
    for seed in _experiment_module().REGISTERED_SEEDS:
        relevance = by_key[("relevance", seed)]
        variants = {}
        for representation, component in (
            ("B4", "target_stance_b4"),
            ("B2", "target_stance_b2"),
        ):
            target_stance = by_key[(component, seed)]
            primary_metrics = combine_component_predictions(
                reference=subset_reference(primary_ids),
                relevance_payload=subset_payload(relevance, primary_ids),
                target_stance_payload=subset_payload(target_stance, primary_ids),
                representation=representation,
                natural_design=natural_design,
                natural_weighting=natural_weighting,
            )
            variants[representation] = gate_metric_surface(primary_metrics)
            for selection, ids in sorted(component_ids.items()):
                metrics = (
                    primary_metrics
                    if selection == "development_probability"
                    else combine_component_predictions(
                        reference=subset_reference(ids),
                        relevance_payload=subset_payload(relevance, ids),
                        target_stance_payload=subset_payload(target_stance, ids),
                        representation=representation,
                    )
                )
                surface = (
                    metrics["design_weighted"]
                    if selection == "development_probability"
                    else metrics
                )
                diagnostics[selection][representation].append(
                    surface["end_to_end"]["target_stance_tuples"]["micro"]["f1"]
                    or 0.0
                )
        paired.append({"optimiser_seed": seed, **variants})
    gate = evaluate_representation_gate(
        clean["experiment_contract"],
        phase_run_id=clean["phase_run_id"],
        paired_metrics=paired,
        trial_receipt_ids=receipt_ids,
    )
    closeout_root = (
        volume_root / NAMESPACE / f"run={clean['experiment_run_id']}" / "closeout"
    )
    descriptor = _experiment_module().publish_immutable_json(
        closeout_root / "representation-gate.json", gate
    )
    component_diagnostics = {
        selection: {
            representation: {
                "row_count": len(component_ids[selection]),
                "evidence_scope": (
                    "design-weighted-natural-probability-arm"
                    if selection == "development_probability"
                    else "unweighted-enrichment-diagnostic-only"
                ),
                "mean_tuple_micro_f1": round(
                    sum(values) / len(values), 6
                ),
            }
            for representation, values in sorted(variants.items())
        }
        for selection, variants in sorted(diagnostics.items())
    }
    result = {
        "status": "complete",
        "experiment_run_id": clean["experiment_run_id"],
        "phase_run_id": clean["phase_run_id"],
        "trial_count": len(clean["trials"]),
        "paired_seed_count": len(paired),
        "selected_representation": gate["selected_representation"],
        "verdict": gate["verdict"],
        "aggregate_evidence": gate["aggregate_evidence"],
        "development_design": {
            "primary_metric_component": "development_probability",
            "primary_rows": 300,
            "weighting_config_digest": natural_weighting.digest(),
            "design_summary_sha256": canonical_sha256(natural_summary),
            "probability_design_diagnostics": natural_summary,
            "enrichment_diagnostic_rows": 300,
            "selection_component_diagnostics": component_diagnostics,
        },
        "gate_receipt_id": gate["gate_receipt_id"],
        "gate_artifact": descriptor,
        "calibration_or_threshold_frozen": False,
        "locked_test_rows_accessed": 0,
        "corpus_inference_authorised": False,
    }
    assert_metadata_only(result, where="factorised fixed comparison closeout")
    return result


__all__ = [
    "MAX_LENGTH",
    "PRIVATE_PREDICTIONS_KIND",
    "Component",
    "FactorisedDynamicPaddingCollator",
    "FactorisedOptimisationConfig",
    "adamw_parameter_groups",
    "aggregate_trial_inspections",
    "build_legacy_overlap_audit",
    "build_optimisation_config",
    "build_private_development_predictions",
    "closeout_fixed_comparison",
    "collect_development_logits",
    "combine_component_predictions",
    "create_adamw",
    "create_component_model",
    "derive_bridge_exposure_register",
    "gate_metric_surface",
    "inspect_trial_output",
    "label_sha256",
    "load_pinned_tokenizer",
    "load_private_frame",
    "materialise_private_frames",
    "prepare_evidence_frames_from_artifacts",
    "run_training_trial",
    "source_row_sha256",
    "tokenise_factorised_record",
    "train_epoch",
    "validate_legacy_overlap_audit",
    "validate_private_development_predictions",
    "validate_private_exposure_register",
    "validate_trial_artifacts",
]
