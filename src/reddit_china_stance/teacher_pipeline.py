"""Privacy-preserving orchestration for the active three-stage teacher pipeline.

This module is deliberately provider-neutral.  Callers inject provider calls and
persist the returned content-addressed objects under an ignored private root.
Only :func:`build_public_receipt` returns a publication-safe aggregate.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from reddit_china_stance.annotation_contract import PROMPT_BUNDLE, SOL_ADJUDICATION_PROMPT
from reddit_china_stance.human_seeded_consensus_v1 import (
    load_label_schema,
    validate_semantic_label,
)

SCHEMA_VERSION = "1.0.0"
PACKET_KIND = "private-bounded-context-packet-v1"
MANIFEST_KIND = "three-stage-teacher-manifest-v1"
TEACHER_CHECKPOINT_KIND = "three-stage-teacher-decisions-v1"
SOL_QUEUE_KIND = "blinded-sol-adjudication-queue-v1"
SOL_CHECKPOINT_KIND = "sol-adjudication-decisions-v1"
FINAL_LABELS_KIND = "private-teacher-labels-v1"
PUBLIC_RECEIPT_KIND = "teacher-pipeline-metadata-receipt-v1"
TEACHER_JOURNAL_KIND = "three-stage-teacher-row-journal-v1"

MODEL_ROLES = ("gemma", "gpt_oss", "sol")
TEACHER_ROLES = MODEL_ROLES[:2]
STAGES = ("relevance", "targets", "stance")
MAX_TARGET_TEXT_CHARS = 16_000
MAX_CONTEXT_CHARS = 16_000
ROW_SEED_STRIDE = 100_000
TARGETS_SEED_OFFSET = 10_000
STANCE_SEED_OFFSET = 20_000
SOL_SEED_OFFSET = 50_000

_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "record_id",
        "record_ids",
        "queue_item_id",
        "queue_item_ids",
        "text",
        "target_text",
        "submission_context",
        "parent_context",
        "semantic_label",
        "label",
        "labels",
        "target",
        "targets",
        "stance",
        "stances",
        "rows",
        "items",
        "routing",
        "stage_outputs",
    }
)
_INTERNAL_PROVENANCE_TERMS = frozenset({"silver", "bronze"})
_SECRET_CONFIG_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "key",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
_SECRET_CONFIG_SUFFIXES = (
    "_credential",
    "_key",
    "_password",
    "_secret",
    "_token",
)

ProviderCall = Callable[..., Any]


class PlannedRunInterruption(RuntimeError):
    """A test/operator-requested stop after durable row publication."""


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of the canonical JSON representation of ``value``."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ACTIVE_PROMPT_SHA256 = canonical_sha256(PROMPT_BUNDLE)


def _json_clone(value: Any, *, where: str) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be finite JSON") from exc


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _required_string(value: Mapping[str, Any], key: str, *, where: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bound_label_schema(
    manifest: Mapping[str, Any], schema: Mapping[str, Any] | None
) -> dict[str, Any]:
    label_schema = dict(schema) if schema is not None else load_label_schema()
    expected = manifest.get("contract", {}).get("label_schema_sha256")
    if canonical_sha256(label_schema) != expected:
        raise RuntimeError("runtime label schema differs from the frozen manifest")
    return label_schema


def _provider_contract(manifest: Mapping[str, Any]) -> dict[str, str]:
    contract = manifest["contract"]
    return {
        "manifest_id": manifest["manifest_id"],
        "label_schema_sha256": contract["label_schema_sha256"],
        "rubric_sha256": contract["rubric_sha256"],
        "prompt_sha256": contract["prompt_sha256"],
    }


def _validate_packet_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    where = f"packet.rows[{index}]"
    _require_exact_keys(
        row,
        {"record_id", "target_text", "submission_context", "parent_context"},
        where=where,
    )
    record_id = _required_string(row, "record_id", where=where)
    target_text = _required_string(row, "target_text", where=where)
    if len(target_text) > MAX_TARGET_TEXT_CHARS:
        raise ValueError(f"{where}.target_text exceeds the bounded-context limit")
    contexts: dict[str, str | None] = {}
    for field in ("submission_context", "parent_context"):
        context = row[field]
        if context is not None and not isinstance(context, str):
            raise ValueError(f"{where}.{field} must be text or null")
        if isinstance(context, str) and len(context) > MAX_CONTEXT_CHARS:
            raise ValueError(f"{where}.{field} exceeds the bounded-context limit")
        contexts[field] = context
    return {
        "record_id": record_id,
        "target_text": target_text,
        **contexts,
    }


def freeze_packet(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and content-address a private bounded-context packet."""

    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("packet rows must be a non-empty sequence")
    clean_rows = [_validate_packet_row(row, index=index) for index, row in enumerate(rows)]
    record_ids = [row["record_id"] for row in clean_rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("packet record IDs must be unique")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": PACKET_KIND,
        "context_limits": {
            "target_text_characters": MAX_TARGET_TEXT_CHARS,
            "each_context_characters": MAX_CONTEXT_CHARS,
        },
        "row_count": len(clean_rows),
        "rows": clean_rows,
    }
    return {**body, "packet_id": canonical_sha256(body)}


def freeze_packet_from_context_records(
    records: Sequence[Mapping[str, Any]], *, english_record_ids: Sequence[str]
) -> dict[str, Any]:
    """Validate the language/context boundary and freeze the teacher packet."""

    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise ValueError("context records must be a sequence")
    if isinstance(english_record_ids, (str, bytes, bytearray)) or not isinstance(
        english_record_ids, Sequence
    ):
        raise ValueError("english_record_ids must be a sequence")
    english_ids = list(english_record_ids)
    if any(not isinstance(record_id, str) or not record_id for record_id in english_ids):
        raise ValueError("english_record_ids must contain non-empty strings")
    if len(english_ids) != len(set(english_ids)):
        raise ValueError("english_record_ids must be unique")

    packet_rows: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    allowed_empty_statuses = {"missing", "not_applicable", "same_as_target", "same_as_submission"}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"context records[{index}] must be an object")
        record_id = _required_string(record, "record_id", where=f"context records[{index}]")
        observed_ids.append(record_id)
        if record.get("target_join_status") != "present":
            raise ValueError(f"context records[{index}] has no validated target join")
        if record.get("context_role_rule") != "reference_disambiguation_only":
            raise ValueError(f"context records[{index}] has an unsupported context role")

        extracted: dict[str, str | None] = {}
        for field, status_field in (
            ("target_text", "target_join_status"),
            ("submission_context", "submission_join_status"),
            ("parent_context", "parent_join_status"),
        ):
            payload = record.get(field)
            status = record.get(status_field)
            if status == "invalid_relation":
                raise ValueError(f"context records[{index}].{status_field} is invalid")
            if payload is None:
                if field == "target_text" or status not in allowed_empty_statuses:
                    raise ValueError(f"context records[{index}].{field} is inconsistent")
                extracted[field] = None
                continue
            if status != "present" or not isinstance(payload, Mapping):
                raise ValueError(f"context records[{index}].{field} is inconsistent")
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"context records[{index}].{field}.text is invalid")
            extracted[field] = text
        packet_rows.append({"record_id": record_id, **extracted})

    if observed_ids != english_ids:
        if set(observed_ids) != set(english_ids):
            raise ValueError("context records differ from the exact English-eligible row set")
        raise ValueError("context records differ from the frozen English-row order")
    return freeze_packet(packet_rows)


def validate_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen packet and return a defensive copy."""

    _require_exact_keys(
        packet,
        {"schema_version", "kind", "context_limits", "row_count", "rows", "packet_id"},
        where="packet",
    )
    rows = packet.get("rows")
    if not isinstance(rows, list):
        raise ValueError("packet.rows must be an array")
    rebuilt = freeze_packet(rows)
    if dict(packet) != rebuilt:
        raise RuntimeError("packet content address or frozen contract drifted")
    return deepcopy(rebuilt)


def _validate_model_config(config: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError(f"models.{role} must be an object")
    clean = _json_clone(dict(config), where=f"models.{role}")

    def reject_secrets(value: Any, *, where: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalised = str(key).casefold().replace("-", "_")
                if normalised in _SECRET_CONFIG_KEYS or normalised.endswith(
                    _SECRET_CONFIG_SUFFIXES
                ):
                    raise ValueError(f"{where} must not contain credentials or secrets")
                reject_secrets(child, where=f"{where}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_secrets(child, where=f"{where}[{index}]")

    reject_secrets(clean, where=f"models.{role}")
    for field in ("provider", "model_id", "revision", "reasoning_effort"):
        _required_string(clean, field, where=f"models.{role}")
    max_output_tokens = clean.get("max_output_tokens")
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens <= 0
    ):
        raise ValueError(f"models.{role}.max_output_tokens must be a positive integer")
    return clean


def freeze_manifest(
    packet: Mapping[str, Any],
    *,
    models: Mapping[str, Mapping[str, Any]],
    label_schema_sha256: str,
    rubric_sha256: str,
    prompt_sha256: str,
    base_seed: int,
    audit_size: int,
    audit_seed: str,
) -> dict[str, Any]:
    """Freeze the exact packet, model, rubric, schema, and audit contract."""

    clean_packet = validate_packet(packet)
    if set(models) != set(MODEL_ROLES):
        raise ValueError(f"models must contain exactly {list(MODEL_ROLES)}")
    if not all(_is_sha256(value) for value in (label_schema_sha256, rubric_sha256, prompt_sha256)):
        raise ValueError("schema, rubric and prompt digests must be lowercase SHA-256 values")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if isinstance(audit_size, bool) or not isinstance(audit_size, int) or audit_size < 0:
        raise ValueError("audit_size must be a non-negative integer")
    if not isinstance(audit_seed, str) or not audit_seed:
        raise ValueError("audit_seed must be a non-empty string")
    clean_models = {
        role: _validate_model_config(models[role], role=role) for role in MODEL_ROLES
    }
    contract = {
        "packet_id": clean_packet["packet_id"],
        "ordered_record_ids_sha256": canonical_sha256(
            [row["record_id"] for row in clean_packet["rows"]]
        ),
        "row_count": clean_packet["row_count"],
        "models": clean_models,
        "stages": list(STAGES),
        "label_schema_sha256": label_schema_sha256,
        "rubric_sha256": rubric_sha256,
        "prompt_sha256": prompt_sha256,
        "base_seed": base_seed,
        "seed_derivation": {
            "row_stride": ROW_SEED_STRIDE,
            "targets_offset": TARGETS_SEED_OFFSET,
            "stance_offset": STANCE_SEED_OFFSET,
            "sol_offset": SOL_SEED_OFFSET,
        },
        "routing_policy": {
            "escalate_invalid": True,
            "escalate_disagreement": True,
            "escalate_unclear": True,
            "audit_only_exact_clear_agreements": True,
        },
        "audit": {"size": audit_size, "seed": audit_seed},
        "privacy": {
            "packet_and_row_outputs_private": True,
            "sol_items_blinded_to_teacher_outputs_and_routing": True,
            "public_receipts_metadata_only": True,
        },
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "contract": contract,
    }
    return {**body, "manifest_id": canonical_sha256(body)}


def validate_manifest(
    packet: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate packet binding and every frozen manifest field."""

    _require_exact_keys(
        manifest,
        {"schema_version", "kind", "contract", "manifest_id"},
        where="manifest",
    )
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("manifest.contract must be an object")
    audit = contract.get("audit")
    models = contract.get("models")
    if not isinstance(audit, Mapping) or not isinstance(models, Mapping):
        raise ValueError("manifest model and audit contracts must be objects")
    rebuilt = freeze_manifest(
        packet,
        models=models,
        label_schema_sha256=contract.get("label_schema_sha256", ""),
        rubric_sha256=contract.get("rubric_sha256", ""),
        prompt_sha256=contract.get("prompt_sha256", ""),
        base_seed=contract.get("base_seed"),
        audit_size=audit.get("size"),
        audit_seed=audit.get("seed"),
    )
    if dict(manifest) != rebuilt:
        raise RuntimeError("manifest content address or frozen contract drifted")
    return deepcopy(rebuilt)


def _semantic_from_stage_outputs(
    stage_outputs: Mapping[str, Any], *, schema: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(stage_outputs, Mapping):
        raise ValueError("stage_outputs must be an object")
    relevance_output = stage_outputs.get("relevance")
    if not isinstance(relevance_output, Mapping):
        raise ValueError("relevance output must be an object")
    _require_exact_keys(relevance_output, {"relevance"}, where="stage_outputs.relevance")
    relevance = relevance_output["relevance"]
    if relevance != "material":
        _require_exact_keys(stage_outputs, {"relevance"}, where="stage_outputs")
        semantic = validate_semantic_label(
            {"relevance": relevance, "target_stances": []}, schema=schema
        )
        return {"relevance": {"relevance": relevance}}, semantic

    _require_exact_keys(
        stage_outputs, {"relevance", "targets", "stances"}, where="stage_outputs"
    )
    targets_output = stage_outputs["targets"]
    if not isinstance(targets_output, Mapping):
        raise ValueError("targets output must be an object")
    _require_exact_keys(targets_output, {"targets"}, where="stage_outputs.targets")
    targets = targets_output["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("material targets output must be a non-empty array")
    if any(not isinstance(target, str) for target in targets) or len(targets) != len(set(targets)):
        raise ValueError("targets output must contain unique strings")

    stances = stage_outputs["stances"]
    if not isinstance(stances, list) or any(not isinstance(row, Mapping) for row in stances):
        raise ValueError("stances output must be an array of objects")
    clean_stances: list[dict[str, Any]] = []
    for index, row in enumerate(stances):
        _require_exact_keys(row, {"target", "stance"}, where=f"stage_outputs.stances[{index}]")
        clean_stances.append(dict(row))
    stance_targets = [row["target"] for row in clean_stances]
    if len(stance_targets) != len(set(stance_targets)) or set(stance_targets) != set(targets):
        raise ValueError("stances must cover each selected target exactly once")
    semantic = validate_semantic_label(
        {"relevance": relevance, "target_stances": clean_stances}, schema=schema
    )
    canonical_targets = [row["target"] for row in semantic["target_stances"]]
    stance_by_target = {row["target"]: row for row in semantic["target_stances"]}
    clean_outputs = {
        "relevance": {"relevance": relevance},
        "targets": {"targets": canonical_targets},
        "stances": [stance_by_target[target] for target in canonical_targets],
    }
    return clean_outputs, semantic


def _normalise_teacher_output(
    output: Mapping[str, Any], *, schema: Mapping[str, Any]
) -> dict[str, Any]:
    _require_exact_keys(output, {"record_id", "stage_outputs"}, where="teacher output")
    record_id = _required_string(output, "record_id", where="teacher output")
    raw = _json_clone(output["stage_outputs"], where="teacher output.stage_outputs")
    try:
        clean_outputs, semantic = _semantic_from_stage_outputs(raw, schema=schema)
    except ValueError as exc:
        return {
            "record_id": record_id,
            "status": "invalid",
            "stage_outputs": raw,
            "validation_error": {
                "code": "invalid_stage_output",
                "message": str(exc),
            },
        }
    return {
        "record_id": record_id,
        "status": "valid",
        "stage_outputs": clean_outputs,
        "semantic_label": semantic,
    }


def freeze_teacher_checkpoint(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    model_role: str,
    outputs: Sequence[Mapping[str, Any]],
    observed_model: str,
    observed_provider: str,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and freeze one complete teacher model output checkpoint.

    Invalid stage schemas remain explicit private evidence and are routed to Sol;
    missing or duplicate rows fail the entire checkpoint.
    """

    clean_packet = validate_packet(packet)
    clean_manifest = validate_manifest(clean_packet, manifest)
    if model_role not in TEACHER_ROLES:
        raise ValueError(f"model_role must be one of {list(TEACHER_ROLES)}")
    model_config = clean_manifest["contract"]["models"][model_role]
    if observed_model != model_config.get("observed_model_id", model_config["model_id"]):
        raise RuntimeError("observed teacher model differs from the frozen manifest")
    if observed_provider != model_config["provider"]:
        raise RuntimeError("observed teacher provider differs from the frozen manifest")
    if isinstance(outputs, (str, bytes, bytearray)) or not isinstance(outputs, Sequence):
        raise ValueError("teacher outputs must be a sequence")
    label_schema = _bound_label_schema(clean_manifest, schema)
    by_id: dict[str, dict[str, Any]] = {}
    for output in outputs:
        if not isinstance(output, Mapping):
            raise ValueError("each teacher output must be an object")
        clean = _normalise_teacher_output(output, schema=label_schema)
        record_id = clean["record_id"]
        if record_id in by_id:
            raise ValueError(f"duplicate teacher output record_id: {record_id}")
        by_id[record_id] = clean
    expected_ids = [row["record_id"] for row in clean_packet["rows"]]
    missing = set(expected_ids) - set(by_id)
    extra = set(by_id) - set(expected_ids)
    if missing or extra:
        raise ValueError(
            f"teacher output row set mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": TEACHER_CHECKPOINT_KIND,
        "manifest_id": clean_manifest["manifest_id"],
        "model_role": model_role,
        "model_config_sha256": canonical_sha256(
            model_config
        ),
        "observed_model": observed_model,
        "observed_provider": observed_provider,
        "prompt_sha256": clean_manifest["contract"]["prompt_sha256"],
        "base_seed": clean_manifest["contract"]["base_seed"],
        "row_count": len(expected_ids),
        "rows": [by_id[record_id] for record_id in expected_ids],
    }
    return {**body, "checkpoint_id": canonical_sha256(body)}


def validate_teacher_checkpoint(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    model_role: str,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a checkpoint, including truthful valid/invalid classification."""

    _require_exact_keys(
        checkpoint,
        {
            "schema_version",
            "kind",
            "manifest_id",
            "model_role",
            "model_config_sha256",
            "observed_model",
            "observed_provider",
            "prompt_sha256",
            "base_seed",
            "row_count",
            "rows",
            "checkpoint_id",
        },
        where="teacher checkpoint",
    )
    rows = checkpoint.get("rows")
    if not isinstance(rows, list):
        raise ValueError("teacher checkpoint rows must be an array")
    imported: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"teacher checkpoint rows[{index}] must be an object")
        status = row.get("status")
        if status == "valid":
            _require_exact_keys(
                row,
                {"record_id", "status", "stage_outputs", "semantic_label"},
                where=f"teacher checkpoint rows[{index}]",
            )
        elif status == "invalid":
            _require_exact_keys(
                row,
                {"record_id", "status", "stage_outputs", "validation_error"},
                where=f"teacher checkpoint rows[{index}]",
            )
        else:
            raise ValueError(f"teacher checkpoint rows[{index}].status is invalid")
        imported.append({"record_id": row.get("record_id"), "stage_outputs": row["stage_outputs"]})
    rebuilt = freeze_teacher_checkpoint(
        packet,
        manifest,
        model_role=model_role,
        outputs=imported,
        observed_model=checkpoint.get("observed_model"),
        observed_provider=checkpoint.get("observed_provider"),
        schema=schema,
    )
    if dict(checkpoint) != rebuilt:
        raise RuntimeError("teacher checkpoint content address or validation state drifted")
    return deepcopy(rebuilt)


def _provider_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_text": row["target_text"],
        "submission_context": row["submission_context"],
        "parent_context": row["parent_context"],
    }


def _journal_entry_path(
    journal_root: Path, *, manifest_id: str, model_role: str, record_id: str
) -> Path:
    record_key = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return (
        journal_root
        / f"manifest={manifest_id}"
        / f"role={model_role}"
        / f"row-{record_key}.json"
    )


def _write_atomic_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical JSON atomically without overwriting an existing entry."""

    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable journal entry differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pending",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError(f"immutable journal entry differs: {path}")
        else:
            os.replace(temporary_path, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _journal_input_sha256(
    *, manifest: Mapping[str, Any], model_role: str, row_index: int, row: Mapping[str, Any]
) -> str:
    return canonical_sha256(
        {
            "manifest_id": manifest["manifest_id"],
            "model_role": model_role,
            "row_index": row_index,
            "row": row,
            "model_config": manifest["contract"]["models"][model_role],
            "prompt_sha256": manifest["contract"]["prompt_sha256"],
        }
    )


def _load_teacher_journal_entry(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    model_role: str,
    row_index: int,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"teacher journal entry is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("teacher journal entry must be an object")
    expected_fields = {
        "schema_version",
        "kind",
        "manifest_id",
        "model_role",
        "record_id",
        "row_index",
        "input_sha256",
        "model_config_sha256",
        "prompt_sha256",
        "observed_model",
        "observed_provider",
        "stage_outputs",
        "entry_id",
    }
    _require_exact_keys(value, expected_fields, where="teacher journal entry")
    body = {key: child for key, child in value.items() if key != "entry_id"}
    expected_model_config_sha256 = canonical_sha256(
        manifest["contract"]["models"][model_role]
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != TEACHER_JOURNAL_KIND
        or value.get("manifest_id") != manifest["manifest_id"]
        or value.get("model_role") != model_role
        or value.get("record_id") != row["record_id"]
        or value.get("row_index") != row_index
        or value.get("input_sha256")
        != _journal_input_sha256(
            manifest=manifest,
            model_role=model_role,
            row_index=row_index,
            row=row,
        )
        or value.get("model_config_sha256") != expected_model_config_sha256
        or value.get("prompt_sha256") != manifest["contract"]["prompt_sha256"]
        or value.get("entry_id") != canonical_sha256(body)
    ):
        raise RuntimeError("teacher journal entry binding or content address drifted")
    stage_outputs = value.get("stage_outputs")
    if not isinstance(stage_outputs, Mapping):
        raise RuntimeError("teacher journal stage_outputs must be an object")
    return dict(value)


def run_teacher_model(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    model_role: str,
    provider_call: ProviderCall,
    journal_root: Path | None = None,
    stop_after_new_rows: int | None = None,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a journalled three-stage teacher through a no-retry provider callable."""

    clean_packet = validate_packet(packet)
    clean_manifest = validate_manifest(clean_packet, manifest)
    if model_role not in TEACHER_ROLES:
        raise ValueError(f"model_role must be one of {list(TEACHER_ROLES)}")
    model_config = deepcopy(clean_manifest["contract"]["models"][model_role])
    label_schema = _bound_label_schema(clean_manifest, schema)
    relevance_values = label_schema["properties"]["relevance"]["enum"]
    target_values_allowed = label_schema["properties"]["target_stances"]["items"][
        "properties"
    ]["target"]["enum"]
    outputs: list[dict[str, Any]] = []
    observed_model: str | None = None
    observed_provider: str | None = None
    new_rows = 0
    if stop_after_new_rows is not None:
        if journal_root is None:
            raise ValueError("stop_after_new_rows requires journal_root")
        if (
            isinstance(stop_after_new_rows, bool)
            or not isinstance(stop_after_new_rows, int)
            or stop_after_new_rows <= 0
        ):
            raise ValueError("stop_after_new_rows must be a positive integer")
    if journal_root is not None:
        journal_directory = (
            journal_root
            / f"manifest={clean_manifest['manifest_id']}"
            / f"role={model_role}"
        )
        expected_paths = {
            _journal_entry_path(
                journal_root,
                manifest_id=clean_manifest["manifest_id"],
                model_role=model_role,
                record_id=row["record_id"],
            ).resolve()
            for row in clean_packet["rows"]
        }
        extras = {
            path.resolve() for path in journal_directory.glob("row-*.json")
        } - expected_paths
        if extras:
            raise RuntimeError(
                f"teacher journal contains {len(extras)} unexpected row entries"
            )

    def accept_route(call_model: Any, call_provider: Any) -> None:
        nonlocal observed_model, observed_provider
        expected_model = model_config.get("observed_model_id", model_config["model_id"])
        if call_model != expected_model or call_provider != model_config["provider"]:
            raise RuntimeError("observed provider route differs from the frozen manifest")
        if observed_model is None:
            observed_model, observed_provider = call_model, call_provider
        elif (call_model, call_provider) != (observed_model, observed_provider):
            raise RuntimeError("observed provider route changed within the checkpoint")

    def persist_output(
        output: Mapping[str, Any], *, row_index: int, row: Mapping[str, Any]
    ) -> None:
        nonlocal new_rows
        clean_output = _json_clone(dict(output), where="teacher row output")
        if journal_root is not None:
            assert observed_model is not None and observed_provider is not None
            body = {
                "schema_version": SCHEMA_VERSION,
                "kind": TEACHER_JOURNAL_KIND,
                "manifest_id": clean_manifest["manifest_id"],
                "model_role": model_role,
                "record_id": row["record_id"],
                "row_index": row_index,
                "input_sha256": _journal_input_sha256(
                    manifest=clean_manifest,
                    model_role=model_role,
                    row_index=row_index,
                    row=row,
                ),
                "model_config_sha256": canonical_sha256(model_config),
                "prompt_sha256": clean_manifest["contract"]["prompt_sha256"],
                "observed_model": observed_model,
                "observed_provider": observed_provider,
                "stage_outputs": clean_output["stage_outputs"],
            }
            entry = {**body, "entry_id": canonical_sha256(body)}
            _write_atomic_immutable_json(
                _journal_entry_path(
                    journal_root,
                    manifest_id=clean_manifest["manifest_id"],
                    model_role=model_role,
                    record_id=row["record_id"],
                ),
                entry,
            )
            new_rows += 1
            if stop_after_new_rows is not None and new_rows >= stop_after_new_rows:
                raise PlannedRunInterruption(
                    f"planned stop after {new_rows} newly journalled row(s)"
                )
        outputs.append(clean_output)

    def invoke(*, stage: str, seed: int, **request: Any) -> Any:
        response = provider_call(
            model_role=model_role,
            model_config=deepcopy(model_config),
            stage=stage,
            seed=seed,
            contract=_provider_contract(clean_manifest),
            **request,
        )
        if not isinstance(response, Mapping):
            raise ValueError("provider response envelope must be an object")
        for field in ("value", "observed_model", "observed_provider"):
            if field not in response:
                raise ValueError(f"provider response envelope is missing {field}")
        call_model = response["observed_model"]
        call_provider = response["observed_provider"]
        accept_route(call_model, call_provider)
        return response["value"]

    for row_index, row in enumerate(clean_packet["rows"]):
        if journal_root is not None:
            entry_path = _journal_entry_path(
                journal_root,
                manifest_id=clean_manifest["manifest_id"],
                model_role=model_role,
                record_id=row["record_id"],
            )
            if entry_path.exists():
                entry = _load_teacher_journal_entry(
                    entry_path,
                    manifest=clean_manifest,
                    model_role=model_role,
                    row_index=row_index,
                    row=row,
                )
                accept_route(entry["observed_model"], entry["observed_provider"])
                outputs.append(
                    {
                        "record_id": row["record_id"],
                        "stage_outputs": entry["stage_outputs"],
                    }
                )
                continue
        row_seed = clean_manifest["contract"]["base_seed"] + row_index * ROW_SEED_STRIDE
        item = _provider_item(row)
        relevance = invoke(
            stage="relevance",
            seed=row_seed,
            item=deepcopy(item),
            prior=None,
            target=None,
        )
        stages: dict[str, Any] = {"relevance": _json_clone(relevance, where="provider output")}
        try:
            _require_exact_keys(relevance, {"relevance"}, where="provider relevance output")
            relevance_value = relevance["relevance"]
            if relevance_value not in relevance_values:
                raise ValueError("provider relevance output is outside the canonical enum")
        except (TypeError, ValueError):
            persist_output(
                {"record_id": row["record_id"], "stage_outputs": stages},
                row_index=row_index,
                row=row,
            )
            continue
        if relevance_value == "material":
            targets = invoke(
                stage="targets",
                seed=row_seed + TARGETS_SEED_OFFSET,
                item=deepcopy(item),
                prior={"relevance": relevance_value},
                target=None,
            )
            stages["targets"] = _json_clone(targets, where="provider output")
            try:
                _require_exact_keys(targets, {"targets"}, where="provider targets output")
                target_values = targets["targets"]
                if (
                    not isinstance(target_values, list)
                    or not target_values
                    or any(not isinstance(target, str) for target in target_values)
                    or len(target_values) != len(set(target_values))
                    or any(target not in target_values_allowed for target in target_values)
                ):
                    raise ValueError("provider targets output is invalid")
            except (TypeError, ValueError):
                persist_output(
                    {"record_id": row["record_id"], "stage_outputs": stages},
                    row_index=row_index,
                    row=row,
                )
                continue
            stance_outputs = []
            for target in target_values:
                stance = invoke(
                    stage="stance",
                    seed=row_seed + STANCE_SEED_OFFSET + len(stance_outputs),
                    item=deepcopy(item),
                    prior={"relevance": relevance_value, "targets": list(target_values)},
                    target=target,
                )
                clean_stance = _json_clone(stance, where="provider output")
                if isinstance(clean_stance, Mapping) and set(clean_stance) == {"stance"}:
                    stance_outputs.append({"target": target, **clean_stance})
                else:
                    stance_outputs.append(
                        {"target": target, "provider_output": clean_stance}
                    )
            stages["stances"] = stance_outputs
        persist_output(
            {"record_id": row["record_id"], "stage_outputs": stages},
            row_index=row_index,
            row=row,
        )
    if observed_model is None or observed_provider is None:
        raise RuntimeError("teacher run produced no observed provider evidence")
    return freeze_teacher_checkpoint(
        clean_packet,
        clean_manifest,
        model_role=model_role,
        outputs=outputs,
        observed_model=observed_model,
        observed_provider=observed_provider,
        schema=schema,
    )


def _label_has_unclear(label: Mapping[str, Any]) -> bool:
    return label["relevance"] == "unclear" or any(
        row["stance"] == "unclear" for row in label["target_stances"]
    )


def build_sol_queue(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    gemma_checkpoint: Mapping[str, Any],
    gpt_oss_checkpoint: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a blinded Sol queue plus a private operator-only routing map."""

    clean_packet = validate_packet(packet)
    clean_manifest = validate_manifest(clean_packet, manifest)
    checkpoints = {
        "gemma": validate_teacher_checkpoint(
            clean_packet,
            clean_manifest,
            gemma_checkpoint,
            model_role="gemma",
            schema=schema,
        ),
        "gpt_oss": validate_teacher_checkpoint(
            clean_packet,
            clean_manifest,
            gpt_oss_checkpoint,
            model_role="gpt_oss",
            schema=schema,
        ),
    }
    decisions = {
        role: {row["record_id"]: row for row in checkpoint["rows"]}
        for role, checkpoint in checkpoints.items()
    }
    routed: dict[str, list[str]] = {}
    audit_pool: list[str] = []
    for row in clean_packet["rows"]:
        record_id = row["record_id"]
        left = decisions["gemma"][record_id]
        right = decisions["gpt_oss"][record_id]
        reasons: list[str] = []
        if left["status"] == "invalid" or right["status"] == "invalid":
            reasons.append("invalid")
        else:
            if left["semantic_label"] != right["semantic_label"]:
                reasons.append("disagreement")
            if _label_has_unclear(left["semantic_label"]) or _label_has_unclear(
                right["semantic_label"]
            ):
                reasons.append("unclear")
            if not reasons:
                audit_pool.append(record_id)
        if reasons:
            routed[record_id] = reasons

    audit = clean_manifest["contract"]["audit"]
    if audit["size"] > len(audit_pool):
        raise ValueError(
            f"audit_size={audit['size']} exceeds {len(audit_pool)} eligible exact agreements"
        )
    ranked_audit = sorted(
        audit_pool,
        key=lambda record_id: canonical_sha256(
            {
                "seed": audit["seed"],
                "manifest_id": clean_manifest["manifest_id"],
                "record_id": record_id,
            }
        ),
    )
    for record_id in ranked_audit[: audit["size"]]:
        routed[record_id] = ["audit"]

    packet_by_id = {row["record_id"]: row for row in clean_packet["rows"]}
    items: list[dict[str, Any]] = []
    routing: list[dict[str, Any]] = []
    for row in clean_packet["rows"]:
        record_id = row["record_id"]
        if record_id not in routed:
            continue
        queue_item_id = canonical_sha256(
            {
                "manifest_id": clean_manifest["manifest_id"],
                "record_id": record_id,
                "purpose": "sol-adjudication",
            }
        )
        items.append({"queue_item_id": queue_item_id, **_provider_item(packet_by_id[record_id])})
        routing.append(
            {
                "queue_item_id": queue_item_id,
                "record_id": record_id,
                "reasons": routed[record_id],
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOL_QUEUE_KIND,
        "manifest_id": clean_manifest["manifest_id"],
        "teacher_checkpoint_ids": {
            role: checkpoint["checkpoint_id"] for role, checkpoint in checkpoints.items()
        },
        "blindness": {
            "items_exclude_record_ids": True,
            "items_exclude_teacher_outputs": True,
            "items_exclude_routing_reasons": True,
            "provider_receives_items_only": True,
        },
        "item_count": len(items),
        "items": items,
        "routing": routing,
    }
    return {**body, "queue_id": canonical_sha256(body)}


def validate_sol_queue(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    gemma_checkpoint: Mapping[str, Any],
    gpt_oss_checkpoint: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild and validate the deterministic blinded queue."""

    rebuilt = build_sol_queue(
        packet,
        manifest,
        gemma_checkpoint=gemma_checkpoint,
        gpt_oss_checkpoint=gpt_oss_checkpoint,
        schema=schema,
    )
    if dict(queue) != rebuilt:
        raise RuntimeError("Sol queue content address or deterministic routing drifted")
    return deepcopy(rebuilt)


def _validate_sol_queue_envelope(
    manifest: Mapping[str, Any], queue: Mapping[str, Any]
) -> None:
    """Validate the queue fields needed before any blinded provider call."""

    _require_exact_keys(
        manifest,
        {"schema_version", "kind", "contract", "manifest_id"},
        where="manifest",
    )
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_id") != canonical_sha256(manifest_body):
        raise RuntimeError("manifest is not content addressed")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("manifest.contract must be an object")
    models = contract.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_ROLES):
        raise ValueError("manifest model contract is invalid")
    _validate_model_config(models["sol"], role="sol")

    _require_exact_keys(
        queue,
        {
            "schema_version",
            "kind",
            "manifest_id",
            "teacher_checkpoint_ids",
            "blindness",
            "item_count",
            "items",
            "routing",
            "queue_id",
        },
        where="Sol queue",
    )
    queue_body = {key: value for key, value in queue.items() if key != "queue_id"}
    if queue.get("queue_id") != canonical_sha256(queue_body):
        raise RuntimeError("Sol queue is not content addressed")
    if manifest["manifest_id"] != queue.get("manifest_id"):
        raise RuntimeError("Sol queue is bound to a different manifest")
    items = queue.get("items")
    routing = queue.get("routing")
    if not isinstance(items, list) or not isinstance(routing, list):
        raise ValueError("Sol queue items and routing must be arrays")
    if queue.get("item_count") != len(items) or len(routing) != len(items):
        raise RuntimeError("Sol queue item conservation failed")
    item_ids: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"Sol queue items[{index}] must be an object")
        _require_exact_keys(
            item,
            {"queue_item_id", "target_text", "submission_context", "parent_context"},
            where=f"Sol queue items[{index}]",
        )
        item_id = _required_string(item, "queue_item_id", where=f"Sol queue items[{index}]")
        if not _is_sha256(item_id):
            raise ValueError(f"Sol queue items[{index}].queue_item_id must be a SHA-256")
        _validate_packet_row(
            {
                "record_id": item_id,
                "target_text": item["target_text"],
                "submission_context": item["submission_context"],
                "parent_context": item["parent_context"],
            },
            index=index,
        )
        item_ids.append(item_id)
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Sol queue item IDs must be unique")
    route_ids: list[str] = []
    for index, route in enumerate(routing):
        if not isinstance(route, Mapping):
            raise ValueError(f"Sol queue routing[{index}] must be an object")
        _require_exact_keys(
            route,
            {"queue_item_id", "record_id", "reasons"},
            where=f"Sol queue routing[{index}]",
        )
        route_ids.append(
            _required_string(route, "queue_item_id", where=f"Sol queue routing[{index}]")
        )
        _required_string(route, "record_id", where=f"Sol queue routing[{index}]")
        reasons = route["reasons"]
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(
                reason not in {"invalid", "disagreement", "unclear", "audit"}
                for reason in reasons
            )
        ):
            raise ValueError(f"Sol queue routing[{index}].reasons is invalid")
    if route_ids != item_ids:
        raise RuntimeError("Sol queue items and private routing differ")


def freeze_sol_checkpoint(
    manifest: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    outputs: Sequence[Mapping[str, Any]],
    observed_model: str,
    observed_provider: str,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and freeze one Sol label for every blinded queue item."""

    _validate_sol_queue_envelope(manifest, queue)
    model_config = manifest["contract"]["models"]["sol"]
    if observed_model != model_config["model_id"]:
        raise RuntimeError("observed Sol model differs from the frozen manifest")
    if observed_provider != model_config["provider"]:
        raise RuntimeError("observed Sol provider differs from the frozen manifest")
    label_schema = _bound_label_schema(manifest, schema)
    expected_ids = [item["queue_item_id"] for item in queue["items"]]
    by_id: dict[str, dict[str, Any]] = {}
    for index, output in enumerate(outputs):
        if not isinstance(output, Mapping):
            raise ValueError(f"Sol output {index} must be an object")
        _require_exact_keys(output, {"queue_item_id", "semantic_label"}, where="Sol output")
        queue_item_id = _required_string(output, "queue_item_id", where="Sol output")
        if queue_item_id in by_id:
            raise ValueError(f"duplicate Sol output queue_item_id: {queue_item_id}")
        label = output["semantic_label"]
        if not isinstance(label, Mapping):
            raise ValueError("Sol semantic_label must be an object")
        by_id[queue_item_id] = {
            "queue_item_id": queue_item_id,
            "semantic_label": validate_semantic_label(label, schema=label_schema),
        }
    missing = set(expected_ids) - set(by_id)
    extra = set(by_id) - set(expected_ids)
    if missing or extra:
        raise ValueError(f"Sol output row set mismatch: missing={len(missing)}, extra={len(extra)}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOL_CHECKPOINT_KIND,
        "manifest_id": manifest["manifest_id"],
        "queue_id": queue["queue_id"],
        "model_config_sha256": canonical_sha256(model_config),
        "observed_model": observed_model,
        "observed_provider": observed_provider,
        "prompt_sha256": manifest["contract"]["prompt_sha256"],
        "row_count": len(expected_ids),
        "rows": [by_id[queue_item_id] for queue_item_id in expected_ids],
    }
    return {**body, "checkpoint_id": canonical_sha256(body)}


def run_sol_adjudication(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    gemma_checkpoint: Mapping[str, Any],
    gpt_oss_checkpoint: Mapping[str, Any],
    provider_call: ProviderCall,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run blinded Sol adjudication via an injected, no-retry provider call."""

    clean_packet = validate_packet(packet)
    clean_manifest = validate_manifest(clean_packet, manifest)
    clean_queue = validate_sol_queue(
        clean_packet,
        clean_manifest,
        queue,
        gemma_checkpoint=gemma_checkpoint,
        gpt_oss_checkpoint=gpt_oss_checkpoint,
        schema=schema,
    )
    model_config = deepcopy(clean_manifest["contract"]["models"]["sol"])
    label_schema = _bound_label_schema(clean_manifest, schema)
    outputs = []
    observed_model: str | None = None
    observed_provider: str | None = None
    for queue_index, item in enumerate(clean_queue["items"]):
        blinded_item = _provider_item(item)
        response = provider_call(
            model_role="sol",
            model_config=deepcopy(model_config),
            stage="semantic_label",
            prompt=SOL_ADJUDICATION_PROMPT,
            seed=(
                clean_manifest["contract"]["base_seed"]
                + queue_index * ROW_SEED_STRIDE
                + SOL_SEED_OFFSET
            ),
            item=blinded_item,
            prior=None,
            target=None,
            contract=_provider_contract(clean_manifest),
        )
        if not isinstance(response, Mapping):
            raise ValueError("Sol provider response envelope must be an object")
        for field in ("value", "observed_model", "observed_provider"):
            if field not in response:
                raise ValueError(f"Sol provider response envelope is missing {field}")
        call_model = response["observed_model"]
        call_provider = response["observed_provider"]
        if call_model != model_config["model_id"] or call_provider != model_config["provider"]:
            raise RuntimeError("observed Sol provider route differs from the frozen manifest")
        if observed_model is None:
            observed_model, observed_provider = call_model, call_provider
        elif (call_model, call_provider) != (observed_model, observed_provider):
            raise RuntimeError("observed Sol provider route changed within the checkpoint")
        semantic_label = response["value"]
        if not isinstance(semantic_label, Mapping):
            raise ValueError("Sol semantic label must be an object")
        clean_label = validate_semantic_label(semantic_label, schema=label_schema)
        outputs.append(
            {"queue_item_id": item["queue_item_id"], "semantic_label": clean_label}
        )
    if observed_model is None or observed_provider is None:
        raise RuntimeError("Sol run produced no observed provider evidence")
    return freeze_sol_checkpoint(
        clean_manifest,
        clean_queue,
        outputs=outputs,
        observed_model=observed_model,
        observed_provider=observed_provider,
        schema=schema,
    )


def finalise_labels(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    gemma_checkpoint: Mapping[str, Any],
    gpt_oss_checkpoint: Mapping[str, Any],
    queue: Mapping[str, Any],
    sol_checkpoint: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalise private labels with internal silver/bronze provenance."""

    clean_packet = validate_packet(packet)
    clean_manifest = validate_manifest(clean_packet, manifest)
    clean_queue = validate_sol_queue(
        clean_packet,
        clean_manifest,
        queue,
        gemma_checkpoint=gemma_checkpoint,
        gpt_oss_checkpoint=gpt_oss_checkpoint,
        schema=schema,
    )
    teachers = {
        "gemma": validate_teacher_checkpoint(
            clean_packet,
            clean_manifest,
            gemma_checkpoint,
            model_role="gemma",
            schema=schema,
        ),
        "gpt_oss": validate_teacher_checkpoint(
            clean_packet,
            clean_manifest,
            gpt_oss_checkpoint,
            model_role="gpt_oss",
            schema=schema,
        ),
    }
    if sol_checkpoint.get("manifest_id") != clean_manifest["manifest_id"]:
        raise RuntimeError("Sol checkpoint is bound to a different manifest")
    if sol_checkpoint.get("queue_id") != clean_queue["queue_id"]:
        raise RuntimeError("Sol checkpoint is bound to a different queue")
    sol_rows = sol_checkpoint.get("rows")
    if not isinstance(sol_rows, list):
        raise ValueError("Sol checkpoint rows must be an array")
    rebuilt_sol = freeze_sol_checkpoint(
        clean_manifest,
        clean_queue,
        outputs=sol_rows,
        observed_model=sol_checkpoint.get("observed_model"),
        observed_provider=sol_checkpoint.get("observed_provider"),
        schema=schema,
    )
    if dict(sol_checkpoint) != rebuilt_sol:
        raise RuntimeError("Sol checkpoint content address or validation state drifted")

    queue_to_record = {
        row["queue_item_id"]: row["record_id"] for row in clean_queue["routing"]
    }
    sol_by_record = {
        queue_to_record[row["queue_item_id"]]: row["semantic_label"]
        for row in rebuilt_sol["rows"]
    }
    routing_by_record = {
        row["record_id"]: row["reasons"] for row in clean_queue["routing"]
    }
    teacher_by_role = {
        role: {row["record_id"]: row for row in checkpoint["rows"]}
        for role, checkpoint in teachers.items()
    }
    final_rows = []
    for packet_row in clean_packet["rows"]:
        record_id = packet_row["record_id"]
        if record_id in sol_by_record:
            label = sol_by_record[record_id]
            provenance = "silver"
            origin = (
                "sol_audited_teacher_agreement"
                if routing_by_record[record_id] == ["audit"]
                else "sol_adjudicated"
            )
        else:
            left = teacher_by_role["gemma"][record_id]
            right = teacher_by_role["gpt_oss"][record_id]
            if (
                left["status"] != "valid"
                or right["status"] != "valid"
                or left["semantic_label"] != right["semantic_label"]
                or _label_has_unclear(left["semantic_label"])
            ):
                raise RuntimeError("an unresolved row escaped the Sol queue")
            label = left["semantic_label"]
            provenance = "bronze"
            origin = "unaudited_exact_teacher_agreement"
        final_rows.append(
            {
                "record_id": record_id,
                "semantic_label": deepcopy(label),
                "internal_provenance": provenance,
                "decision_origin": origin,
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": FINAL_LABELS_KIND,
        "manifest_id": clean_manifest["manifest_id"],
        "teacher_checkpoint_ids": {
            role: checkpoint["checkpoint_id"] for role, checkpoint in teachers.items()
        },
        "sol_checkpoint_id": rebuilt_sol["checkpoint_id"],
        "row_count": len(final_rows),
        "rows": final_rows,
    }
    return {**body, "labels_id": canonical_sha256(body)}


def assert_public_metadata_only(
    value: Any, *, packet: Mapping[str, Any] | None = None, path: str = "metadata"
) -> None:
    """Reject row-level fields, internal provenance terms, and packet string leakage."""

    private_strings: set[str] = set()
    if packet is not None:
        clean_packet = validate_packet(packet)
        for row in clean_packet["rows"]:
            private_strings.add(row["record_id"])
            for key in ("target_text", "submission_context", "parent_context"):
                if row[key]:
                    private_strings.add(row[key])

    def visit(child: Any, child_path: str) -> None:
        if isinstance(child, Mapping):
            for key, grandchild in child.items():
                if str(key).casefold() in _PUBLIC_FORBIDDEN_KEYS:
                    raise ValueError(f"{child_path} contains private field {key!r}")
                visit(grandchild, f"{child_path}.{key}")
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            for index, grandchild in enumerate(child):
                visit(grandchild, f"{child_path}[{index}]")
        elif isinstance(child, str):
            if child.casefold() in _INTERNAL_PROVENANCE_TERMS:
                raise ValueError(f"{child_path} exposes internal provenance terminology")
            for private in private_strings:
                if child == private or (len(private) >= 8 and private in child):
                    raise ValueError(f"{child_path} exposes private packet content")

    visit(value, path)


def build_public_receipt(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    gemma_checkpoint: Mapping[str, Any],
    gpt_oss_checkpoint: Mapping[str, Any],
    queue: Mapping[str, Any],
    sol_checkpoint: Mapping[str, Any],
    final_labels: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metadata-only receipt using paper-facing origin names."""

    clean_packet = validate_packet(packet)
    rebuilt_labels = finalise_labels(
        clean_packet,
        manifest,
        gemma_checkpoint=gemma_checkpoint,
        gpt_oss_checkpoint=gpt_oss_checkpoint,
        queue=queue,
        sol_checkpoint=sol_checkpoint,
        schema=schema,
    )
    if dict(final_labels) != rebuilt_labels:
        raise RuntimeError("final labels content address or provenance drifted")
    final_rows = rebuilt_labels["rows"]
    origin_counts = Counter(row.get("decision_origin") for row in final_rows)
    allowed_origins = {
        "sol_adjudicated",
        "sol_audited_teacher_agreement",
        "unaudited_exact_teacher_agreement",
    }
    if set(origin_counts) - allowed_origins:
        raise RuntimeError("final labels contain an unknown decision origin")
    reason_counts = Counter(
        reason for route in queue["routing"] for reason in route["reasons"]
    )
    gemma_by_record = {row["record_id"]: row for row in gemma_checkpoint["rows"]}
    sol_by_queue_item = {
        row["queue_item_id"]: row["semantic_label"] for row in sol_checkpoint["rows"]
    }
    audited_routes = [route for route in queue["routing"] if route["reasons"] == ["audit"]]
    audit_matches = sum(
        gemma_by_record[route["record_id"]]["semantic_label"]
        == sol_by_queue_item[route["queue_item_id"]]
        for route in audited_routes
    )
    audit_mismatches = len(audited_routes) - audit_matches
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": PUBLIC_RECEIPT_KIND,
        "status": "complete",
        "manifest_sha256": manifest["manifest_id"],
        "source_packet_sha256": clean_packet["packet_id"],
        "record_count": clean_packet["row_count"],
        "teacher_models": {
            role: {
                "model_id": manifest["contract"]["models"][role]["model_id"],
                "revision": manifest["contract"]["models"][role]["revision"],
                "configured_provider": manifest["contract"]["models"][role]["provider"],
                "observed_model": checkpoint["observed_model"],
                "observed_provider": checkpoint["observed_provider"],
                "checkpoint_sha256": checkpoint["checkpoint_id"],
                "valid_count": sum(row["status"] == "valid" for row in checkpoint["rows"]),
                "invalid_count": sum(
                    row["status"] == "invalid" for row in checkpoint["rows"]
                ),
            }
            for role, checkpoint in (
                ("gemma", gemma_checkpoint),
                ("gpt_oss", gpt_oss_checkpoint),
            )
        },
        "adjudicator_model": {
            "model_id": manifest["contract"]["models"]["sol"]["model_id"],
            "revision": manifest["contract"]["models"]["sol"]["revision"],
            "configured_provider": manifest["contract"]["models"]["sol"]["provider"],
            "observed_model": sol_checkpoint["observed_model"],
            "observed_provider": sol_checkpoint["observed_provider"],
            "checkpoint_sha256": sol_checkpoint["checkpoint_id"],
        },
        "routing_summary": {
            "unique_adjudicated_count": queue["item_count"],
            "reason_counts": dict(sorted(reason_counts.items())),
            "audit_requested_count": manifest["contract"]["audit"]["size"],
            "audit_selected_count": reason_counts["audit"],
            "audit_seed_sha256": canonical_sha256(manifest["contract"]["audit"]["seed"]),
        },
        "decision_origins": {
            "sol_adjudicated_count": origin_counts["sol_adjudicated"],
            "sol_audited_teacher_agreement_count": origin_counts[
                "sol_audited_teacher_agreement"
            ],
            "unaudited_exact_teacher_agreement_count": origin_counts[
                "unaudited_exact_teacher_agreement"
            ],
        },
        "agreement_audit": {
            "selected_count": len(audited_routes),
            "sol_match_count": audit_matches,
            "sol_mismatch_count": audit_mismatches,
            "sol_match_rate": (
                None if not audited_routes else round(audit_matches / len(audited_routes), 6)
            ),
        },
        "private_outputs": {
            "content_addressed": True,
            "metadata_only_publication": True,
        },
        "final_private_labels_sha256": rebuilt_labels["labels_id"],
    }
    assert_public_metadata_only(receipt, packet=clean_packet)
    return receipt


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON once; an existing different payload fails closed."""

    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    incomplete.write_text(payload, encoding="utf-8")
    incomplete.replace(path)


__all__ = [
    "ACTIVE_PROMPT_SHA256",
    "MODEL_ROLES",
    "STAGES",
    "TEACHER_ROLES",
    "assert_public_metadata_only",
    "build_public_receipt",
    "build_sol_queue",
    "canonical_sha256",
    "finalise_labels",
    "freeze_manifest",
    "freeze_packet",
    "freeze_packet_from_context_records",
    "freeze_sol_checkpoint",
    "freeze_teacher_checkpoint",
    "run_sol_adjudication",
    "run_teacher_model",
    "validate_manifest",
    "validate_packet",
    "validate_sol_queue",
    "validate_teacher_checkpoint",
    "write_immutable_json",
]
