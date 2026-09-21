"""Run the bounded interrupted/resumed live smoke for both teacher routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any

from reddit_china_stance.human_seeded_consensus_v1 import load_label_schema
from reddit_china_stance.providers import (
    effective_prompt_sha256,
    inference_client,
    preflight_route,
    route_manifest,
    teacher_provider_call,
)
from reddit_china_stance.teacher_pipeline import (
    PlannedRunInterruption,
    canonical_sha256,
    freeze_manifest,
    freeze_packet,
    run_teacher_model,
    validate_teacher_checkpoint,
    write_immutable_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIVATE_ROOT = REPO_ROOT / "data/private-teacher-live-smoke-v1"
DEFAULT_PUBLIC_ROOT = REPO_ROOT / "outputs/teacher-live-smoke-v1"
RUBRIC_PATH = REPO_ROOT / "docs/rubrics/human-reference-semantic-v1.md"
ROW_COUNT = 3


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _packet(source_path: Path) -> dict[str, Any]:
    source = _read_object(source_path)
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) < ROW_COUNT:
        raise ValueError("smoke source has too few rows")
    return freeze_packet(
        [
            {
                "record_id": f"live-smoke-{index:03d}",
                "target_text": row["target_text"],
                "submission_context": row.get("submission_context"),
                "parent_context": row.get("parent_context"),
            }
            for index, row in enumerate(rows[:ROW_COUNT])
        ]
    )


def _manifest(packet: dict[str, Any]) -> dict[str, Any]:
    routes = route_manifest()
    return freeze_manifest(
        packet,
        models={
            "gemma": routes["gemma"],
            "gpt_oss": routes["gpt-oss"],
            "sol": {
                "provider": "codex",
                "model_id": "gpt-5.6-sol",
                "revision": "service-alias:gpt-5.6-sol",
                "reasoning_effort": "high",
                "max_output_tokens": 2048,
            },
        },
        label_schema_sha256=canonical_sha256(load_label_schema()),
        rubric_sha256=_sha256_file(RUBRIC_PATH),
        prompt_sha256=effective_prompt_sha256(),
        base_seed=20260828,
        audit_size=0,
        audit_seed="bounded-live-smoke-v1",
    )


def run(
    *,
    source_path: Path,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    public_root: Path = DEFAULT_PUBLIC_ROOT,
) -> dict[str, Any]:
    packet = _packet(source_path)
    manifest = _manifest(packet)
    run_root = private_root / f"run={manifest['manifest_id']}"
    packet_path = run_root / "packet.json"
    manifest_path = run_root / "manifest.json"
    write_immutable_json(packet_path, packet)
    write_immutable_json(manifest_path, manifest)

    preflights = {name: preflight_route(name) for name in ("gemma", "gpt-oss")}
    role_results: dict[str, Any] = {}
    with inference_client() as client:
        provider_call = partial(teacher_provider_call, client=client)
        for model_role in ("gemma", "gpt_oss"):
            journal_root = run_root / "journals"
            journal_directory = (
                journal_root
                / f"manifest={manifest['manifest_id']}"
                / f"role={model_role}"
            )
            before = len(list(journal_directory.glob("row-*.json")))
            interruption_observed = False
            if before < packet["row_count"]:
                try:
                    run_teacher_model(
                        packet,
                        manifest,
                        model_role=model_role,
                        provider_call=provider_call,
                        journal_root=journal_root,
                        stop_after_new_rows=1,
                    )
                except PlannedRunInterruption:
                    interruption_observed = True
            after_interruption = len(list(journal_directory.glob("row-*.json")))
            checkpoint = run_teacher_model(
                packet,
                manifest,
                model_role=model_role,
                provider_call=provider_call,
                journal_root=journal_root,
            )
            checkpoint = validate_teacher_checkpoint(
                packet,
                manifest,
                checkpoint,
                model_role=model_role,
            )
            checkpoint_path = run_root / f"{model_role}-checkpoint.json"
            write_immutable_json(checkpoint_path, checkpoint)
            pending_count = len(list(journal_directory.glob("*.pending")))
            journal_count = len(list(journal_directory.glob("row-*.json")))
            if journal_count != packet["row_count"] or pending_count:
                raise RuntimeError("live smoke journal did not close cleanly")
            if before == 0 and (
                not interruption_observed or after_interruption != 1
            ):
                raise RuntimeError("live smoke did not prove the planned interruption boundary")
            role_results[model_role] = {
                "row_count": checkpoint["row_count"],
                "valid_count": sum(row["status"] == "valid" for row in checkpoint["rows"]),
                "invalid_count": sum(row["status"] == "invalid" for row in checkpoint["rows"]),
                "journal_entry_count": journal_count,
                "interruption_observed": interruption_observed or before > 0,
                "resume_completed": True,
                "observed_model": checkpoint["observed_model"],
                "observed_provider": checkpoint["observed_provider"],
                "checkpoint_sha256": checkpoint["checkpoint_id"],
            }

    receipt = {
        "schema_version": "1.0.0",
        "kind": "teacher-live-smoke-metadata-receipt-v1",
        "status": "validated",
        "manifest_id": manifest["manifest_id"],
        "packet_id": packet["packet_id"],
        "row_count": packet["row_count"],
        "preflights": preflights,
        "roles": role_results,
        "provider_fallbacks": False,
        "automatic_retries": False,
        "raw_text_public": False,
    }
    public_path = public_root / f"run={manifest['manifest_id']}" / "receipt.json"
    write_immutable_json(public_path, receipt)
    return {**receipt, "receipt_path": str(public_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = run(source_path=args.source)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
