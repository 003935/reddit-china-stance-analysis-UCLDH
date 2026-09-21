"""Package and publish the retained six-checkpoint ModernBERT ensemble privately.

The bundle contains model weights and metadata-only provenance. It excludes
training rows, Reddit content, direct or opaque row identifiers, labels, logits,
predictions, prompts, and reasoning traces.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.modernbert_factorised_experiment import (
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_REVISION,
)
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256

DEFAULT_REPO_ID = "aisafteycommons/reddit-china-stance-modernbert-probability-random-ensemble-v1"
DEFAULT_SOURCE_ROOT = Path("data/private-modernbert-ensemble-source-v1")
DEFAULT_OUTPUT_DIR = Path("data/private-hf-modernbert-probability-random-ensemble-v1")
CONFIG_PATH = Path("configs/modernbert-probability-random-corpus-inference-v1.toml")
CHECKPOINT_COHORT_ID = "ad01414410538d5001ac5fb0070751ccfd251d31c5b75c5460bcd5a0dbe06f9b"
CALIBRATION_RUN_ID = "3a40abaa140214f4f4d7ac2cffac44bcdbff0da879b157f83f0c97b67aa5beec"
CALIBRATION_RECEIPT_ID = "f59dd582cb24b0f38b0e69aa56dd889e8031b6234f409acb71dbfd021d50e9af"
CALIBRATION_ID = "07b429e62653fb8513364327f67e7f620cf44496471c8443023767ad5f66526c"
BASE_MODEL_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
CALIBRATION_FILES = (
    "aggregate-validation.json",
    "calibration.json",
    "checkpoint-provenance.json",
    "receipt.json",
    "source-bundle.json",
    "throughput.json",
)
FORBIDDEN_JSON_KEYS = frozenset(
    {
        "author",
        "body",
        "comment_id",
        "label_json",
        "logits",
        "opaque_id",
        "parent_context",
        "prompt",
        "reasoning",
        "record_id",
        "sample_id",
        "submission_context",
        "submission_id",
        "target_text",
        "text",
        "thread_id",
    }
)
EVIDENCE_BOUNDARY = (
    "six-checkpoint model-assisted development ensemble; not human-validated or prevalence truth"
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _reject_private_json(value: Any, *, where: str) -> None:
    if isinstance(value, Mapping):
        overlap = FORBIDDEN_JSON_KEYS.intersection(value)
        if overlap:
            raise RuntimeError(f"{where} contains forbidden private keys: {sorted(overlap)}")
        for nested in value.values():
            _reject_private_json(nested, where=where)
    elif isinstance(value, list):
        for nested in value:
            _reject_private_json(nested, where=where)


def _descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required ensemble source file is missing: {path}")
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _directory_sizes(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _checkpoint_source_path(source_root: Path, spec: candidate.CheckpointSpec) -> Path:
    return source_root / "checkpoints" / spec.component / f"seed={spec.seed}" / "checkpoint.pt"


def _checkpoint_receipt_path(source_root: Path, spec: candidate.CheckpointSpec) -> Path:
    return source_root / "checkpoints" / spec.component / f"seed={spec.seed}" / "receipt.json"


def _validate_calibration(source_root: Path) -> dict[str, dict[str, Any]]:
    root = source_root / "provenance" / "calibration"
    values: dict[str, dict[str, Any]] = {}
    for name in CALIBRATION_FILES:
        value = _read_object(root / name)
        _reject_private_json(value, where=f"calibration {name}")
        values[name] = value
    receipt = values["receipt.json"]
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    calibration = values["calibration.json"]
    calibration_body = {key: value for key, value in calibration.items() if key != "calibration_id"}
    if (
        receipt.get("receipt_id") != canonical_sha256(receipt_body)
        or receipt.get("receipt_id") != CALIBRATION_RECEIPT_ID
        or receipt.get("calibration_run_id") != CALIBRATION_RUN_ID
        or receipt.get("calibration_id") != CALIBRATION_ID
        or receipt.get("checkpoint_cohort_id") != CHECKPOINT_COHORT_ID
        or receipt.get("checkpoint_count") != len(candidate.CHECKPOINT_SPECS)
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("corpus_rows_accessed") != 0
        or receipt.get("status") != "complete"
        or calibration.get("calibration_id") != canonical_sha256(calibration_body)
        or calibration.get("calibration_id") != CALIBRATION_ID
        or calibration.get("bindings", {}).get("checkpoint_cohort_id") != CHECKPOINT_COHORT_ID
        or calibration.get("locked_test_rows_accessed") != 0
        or calibration.get("corpus_rows_accessed") != 0
    ):
        raise RuntimeError("calibration provenance drifted")
    artifact_files = {
        "calibration_artifact": "calibration.json",
        "aggregate_validation_artifact": "aggregate-validation.json",
        "throughput_artifact": "throughput.json",
    }
    for key, name in artifact_files.items():
        expected = receipt.get(key)
        path = root / name
        if (
            not isinstance(expected, Mapping)
            or expected.get("sha256") != file_sha256(path)
            or expected.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"calibration receipt artifact drifted: {name}")
    return values


def _validate_source(source_root: Path) -> dict[str, Any]:
    cohort_path = source_root / "provenance" / "checkpoint-cohort.json"
    cohort = _read_object(cohort_path)
    _reject_private_json(cohort, where="checkpoint cohort")
    cohort_body = {key: value for key, value in cohort.items() if key != "cohort_id"}
    if (
        cohort.get("cohort_id") != canonical_sha256(cohort_body)
        or cohort.get("cohort_id") != CHECKPOINT_COHORT_ID
        or cohort.get("combination")
        != "unweighted-arithmetic-mean-aligned-raw-logits-per-component"
        or cohort.get("seed_selection")
        != "all-three-preregistered-paired-seeds-no-post-hoc-seed-selection"
        or cohort.get("locked_test_access_count") != 0
    ):
        raise RuntimeError("checkpoint cohort drifted")

    receipts: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for spec in candidate.CHECKPOINT_SPECS:
        checkpoint_path = _checkpoint_source_path(source_root, spec)
        receipt_path = _checkpoint_receipt_path(source_root, spec)
        receipt = _read_object(receipt_path)
        _reject_private_json(receipt, where=f"training receipt {spec.component}/{spec.seed}")
        receipts.append(receipt)
        descriptor = _descriptor(checkpoint_path)
        if descriptor["sha256"] != spec.checkpoint_sha256:
            raise RuntimeError(f"checkpoint hash drifted: {spec.component}/{spec.seed}")
        checkpoints.append(
            {
                "component": spec.component,
                "seed": spec.seed,
                "selected_epoch": spec.selected_epoch,
                "trial_id": spec.trial_id,
                "checkpoint": descriptor,
                "receipt": _descriptor(receipt_path),
                "receipt_id": receipt.get("receipt_id"),
            }
        )
    if candidate.freeze_checkpoint_cohort(receipts) != cohort:
        raise RuntimeError("training receipts do not reproduce the frozen checkpoint cohort")

    _validate_calibration(source_root)
    metadata_files = {
        "provenance/checkpoint-cohort.json": _descriptor(cohort_path),
        **{
            f"provenance/calibration/{name}": _descriptor(
                source_root / "provenance" / "calibration" / name
            )
            for name in CALIBRATION_FILES
        },
    }
    base_files = {
        f"base-model/{name}": _descriptor(source_root / "base-model" / name)
        for name in BASE_MODEL_FILES
    }
    return {
        "cohort": cohort,
        "checkpoints": checkpoints,
        "metadata_files": metadata_files,
        "base_model_files": base_files,
    }


def _dataset_card(*, export_id: str) -> str:
    return f"""---
library_name: transformers
pipeline_tag: text-classification
base_model: {MODEL_ID}
---

# Private probability-random ModernBERT ensemble

This is the exact retained **six-checkpoint ensemble**, not a single checkpoint. It contains three
relevance checkpoints and three B4 target/stance checkpoints for seeds 47, 61, and 89. Inference
loads one model per component checkpoint, computes aligned raw logits, takes the unweighted
three-seed mean within each component, and applies calibration `{CALIBRATION_ID}`. Export ID:
`{export_id}`; checkpoint cohort: `{CHECKPOINT_COHORT_ID}`.

The base model and tokenizer are pinned to `{MODEL_ID}` revision `{MODEL_REVISION}`. The four
mirrored JSON files under `base-model/` bind its config and tokenizer; the six `.pt` files contain
the trained model state. Use the repository's factorised model and calibrated inference code.

## Evidence and privacy boundary

This is model-assisted development evidence, not independent human validation or prevalence truth.
The private bundle contains no Reddit text, authors, direct or opaque row identifiers, row-level
labels, logits, predictions, prompts, or reasoning traces. Keep this repository private and grant
access only to authorised thesis collaborators.
"""


def _publish_directory(staging: Path, output_dir: Path) -> None:
    hashes = _directory_hashes(staging)
    if output_dir.exists():
        if not output_dir.is_dir() or _directory_hashes(output_dir) != hashes:
            raise RuntimeError(f"immutable private model export differs: {output_dir}")
        shutil.rmtree(staging)
        return
    os.rename(staging, output_dir)


def build_private_hf_model_bundle(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repo_id: str = DEFAULT_REPO_ID,
) -> dict[str, Any]:
    """Build or exact-validate the immutable private Hub-ready model bundle."""

    validated = _validate_source(source_root)
    config_descriptor = _descriptor(CONFIG_PATH)
    checkpoint_files = {
        f"checkpoints/{row['component']}/seed={row['seed']}/checkpoint.pt": row["checkpoint"]
        for row in validated["checkpoints"]
    }
    receipt_files = {
        f"provenance/training-receipts/{row['component']}/seed={row['seed']}.json": row["receipt"]
        for row in validated["checkpoints"]
    }
    contract = {
        "schema_version": "1.0.0",
        "kind": "private-modernbert-probability-random-ensemble-hf-export-contract-v1",
        "repo_id": repo_id,
        "access": "private",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "components": list(candidate.COMPONENTS),
        "seeds": list(candidate.SEEDS),
        "combination": "unweighted-arithmetic-mean-aligned-raw-logits-per-component",
        "calibration_run_id": CALIBRATION_RUN_ID,
        "calibration_receipt_id": CALIBRATION_RECEIPT_ID,
        "calibration_id": CALIBRATION_ID,
        "checkpoint_files": checkpoint_files,
        "training_receipt_files": receipt_files,
        "metadata_files": validated["metadata_files"],
        "base_model_files": validated["base_model_files"],
        "inference_config": config_descriptor,
        "privacy": {
            "contains_reddit_content": False,
            "contains_author_data": False,
            "contains_direct_reddit_ids": False,
            "contains_opaque_candidate_ids": False,
            "contains_row_level_labels": False,
            "contains_logits_or_predictions": False,
            "contains_model_weights": True,
        },
        "locked_test_rows_accessed_for_export": 0,
        "corpus_rows_accessed_for_export": 0,
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }
    export_id = canonical_sha256(contract)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for row in validated["checkpoints"]:
            component = row["component"]
            seed = row["seed"]
            source = source_root / "checkpoints" / component / f"seed={seed}" / "checkpoint.pt"
            destination = staging / "checkpoints" / component / f"seed={seed}" / "checkpoint.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            receipt_source = source.parent / "receipt.json"
            receipt_destination = (
                staging / "provenance" / "training-receipts" / component / f"seed={seed}.json"
            )
            receipt_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(receipt_source, receipt_destination)
        for relative_path in validated["metadata_files"]:
            source = source_root / relative_path
            destination = staging / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for relative_path in validated["base_model_files"]:
            source = source_root / relative_path
            destination = staging / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        config_destination = staging / "configs" / CONFIG_PATH.name
        config_destination.parent.mkdir(parents=True)
        shutil.copyfile(CONFIG_PATH, config_destination)
        readme_path = staging / "README.md"
        readme_path.write_text(_dataset_card(export_id=export_id), encoding="utf-8")
        payload_hashes = _directory_hashes(staging)
        payload_sizes = _directory_sizes(staging)
        manifest = {
            **contract,
            "kind": "private-modernbert-probability-random-ensemble-hf-export-manifest-v1",
            "status": "complete",
            "export_id": export_id,
            "payload_file_hashes": payload_hashes,
            "payload_file_sizes": payload_sizes,
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        _publish_directory(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _read_object(output_dir / "manifest.json")


def _lfs_value(lfs: Any, key: str) -> Any:
    if isinstance(lfs, Mapping):
        return lfs.get(key)
    return getattr(lfs, key, None)


def _verify_remote_files(
    *,
    api: Any,
    repo_id: str,
    revision: str,
    local_hashes: Mapping[str, str],
    local_sizes: Mapping[str, int],
) -> tuple[int, int, list[str]]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    seen: set[str] = set()
    lfs_verified = 0
    download_verified = 0
    server_managed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="hf-model-remote-verify-") as directory:
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            recursive=True,
            expand=True,
            token=True,
        ):
            if not isinstance(entry, RepoFile):
                continue
            if entry.path not in local_hashes:
                server_managed.append(entry.path)
                continue
            seen.add(entry.path)
            lfs = entry.lfs
            if lfs is not None:
                if (
                    _lfs_value(lfs, "sha256") != local_hashes[entry.path]
                    or _lfs_value(lfs, "size") != local_sizes[entry.path]
                ):
                    raise RuntimeError(f"remote LFS object drifted: {entry.path}")
                lfs_verified += 1
                continue
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="model",
                    filename=entry.path,
                    revision=revision,
                    token=True,
                    local_dir=directory,
                    force_download=True,
                )
            )
            if (
                file_sha256(path) != local_hashes[entry.path]
                or path.stat().st_size != local_sizes[entry.path]
            ):
                raise RuntimeError(f"remote file drifted: {entry.path}")
            download_verified += 1
    if seen != set(local_hashes) or sorted(server_managed) != [".gitattributes"]:
        raise RuntimeError("remote model inventory drifted")
    return lfs_verified, download_verified, sorted(server_managed)


def publish_private_hf_model_bundle(
    *, output_dir: Path, repo_id: str, expected_export_id: str
) -> dict[str, Any]:
    """Upload the immutable private model bundle and exact-verify its remote revision."""

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    manifest = _read_object(output_dir / "manifest.json")
    if (
        manifest.get("export_id") != expected_export_id
        or manifest.get("repo_id") != repo_id
        or manifest.get("access") != "private"
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("private model publication authority drifted")
    api = HfApi(token=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(repo_id=repo_id, token=True)
    if info.private is not True:
        raise RuntimeError("refusing to upload the ensemble to a non-private repository")
    try:
        existing_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="manifest.json",
            token=True,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        existing = None
    else:
        existing = _read_object(Path(existing_path))
    if existing is not None and existing.get("export_id") != expected_export_id:
        raise RuntimeError("private model repository already contains a different immutable export")
    if existing is None:
        existing_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", token=True))
        if existing_files - {".gitattributes"}:
            raise RuntimeError("private model repository has unbound pre-existing payloads")
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=output_dir,
            commit_message="model: publish exact probability-random ModernBERT ensemble",
            token=True,
        )
        revision = commit.oid
    else:
        revision = info.sha
    remote_info = api.model_info(repo_id=repo_id, revision=revision, token=True)
    if remote_info.private is not True or remote_info.sha != revision:
        raise RuntimeError("private model repository visibility or revision drifted")
    local_hashes = _directory_hashes(output_dir)
    local_sizes = _directory_sizes(output_dir)
    lfs_count, download_count, server_managed = _verify_remote_files(
        api=api,
        repo_id=repo_id,
        revision=revision,
        local_hashes=local_hashes,
        local_sizes=local_sizes,
    )
    receipt_body = {
        "schema_version": "1.0.0",
        "kind": "private-modernbert-probability-random-ensemble-hf-publication-receipt-v1",
        "repo_id": repo_id,
        "revision": revision,
        "private": True,
        "export_id": expected_export_id,
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "calibration_id": CALIBRATION_ID,
        "payload_file_count": len(local_hashes),
        "payload_file_hashes_sha256": canonical_sha256(local_hashes),
        "lfs_oid_verified_file_count": lfs_count,
        "download_verified_file_count": download_count,
        "server_managed_files": server_managed,
        "locked_test_rows_accessed_for_export": 0,
        "corpus_rows_accessed_for_export": 0,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "status": "complete",
    }
    return {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-export-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_private_hf_model_bundle(
        source_root=args.source_root,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
    )
    result: Mapping[str, Any] = manifest
    if args.publish:
        if args.expected_export_id != manifest["export_id"]:
            raise RuntimeError("--publish requires the exact built --expected-export-id")
        result = publish_private_hf_model_bundle(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            expected_export_id=args.expected_export_id,
        )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
