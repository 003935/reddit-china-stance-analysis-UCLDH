"""Package and publish exact ModernBERT corpus predictions to a private Hub dataset.

This is a separate post-inference authority. It never reads Reddit text, trains a
model, or opens the consumed locked test. Its only row-level inputs are the
exact-validated prediction shards from the completed corpus-inference run.
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

import pyarrow.parquet as pq

from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance import modernbert_probability_random_corpus_inference_v1 as corpus
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256

DEFAULT_REPO_ID = "aisafteycommons/reddit-china-stance-modernbert-corpus-v1"
DEFAULT_OUTPUT_DIR = Path("data/private-hf-modernbert-probability-random-corpus-v1")
EVIDENCE_BOUNDARY = (
    "provisional-model-assisted-corpus-predictions-not-human-validated-prevalence-truth"
)
FORBIDDEN_COLUMNS = frozenset(
    {
        "author",
        "body",
        "text",
        "target_text",
        "submission_context",
        "parent_context",
        "record_id",
        "thread_id",
        "submission_id",
        "comment_id",
        "label_json",
        "logits",
    }
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


def _validate_final_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_authority_id: str,
    expected_receipt_id: str,
) -> dict[str, Any]:
    clean = dict(receipt)
    body = {key: value for key, value in clean.items() if key != "receipt_id"}
    if (
        not expected_authority_id
        or not expected_receipt_id
        or clean.get("receipt_id") != canonical_sha256(body)
        or clean.get("receipt_id") != expected_receipt_id
        or clean.get("kind") != "modernbert-probability-random-corpus-inference-receipt-v1"
        or clean.get("authority_id") != expected_authority_id
        or clean.get("checkpoint_cohort_id")
        != "ad01414410538d5001ac5fb0070751ccfd251d31c5b75c5460bcd5a0dbe06f9b"
        or clean.get("calibration_id")
        != "07b429e62653fb8513364327f67e7f620cf44496471c8443023767ad5f66526c"
        or clean.get("corpus_rows") != corpus.CORPUS_ROWS
        or clean.get("calibration_member_rows") != candidate.CALIBRATION_ROWS
        or clean.get("label_unseen_rows") != corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS
        or clean.get("ensemble_shard_receipts") != corpus.SHARD_COUNT
        or clean.get("prediction_shard_receipts") != corpus.SHARD_COUNT
        or clean.get("locked_test_rows_accessed") != 0
        or clean.get("provider_billing_hard_cap_enforced") is not False
        or clean.get("status") != "complete"
        or clean.get("evidence_boundary")
        != (
            "provisional model-assisted predictions for the post-acquisition eligible candidate "
            "corpus; default thesis aggregates exclude the 600 calibration members"
        )
    ):
        raise RuntimeError("completed corpus-inference receipt binding drifted")
    return clean


def _validate_prediction_shards(
    *, run_root: Path, final_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    index = final_receipt.get("prediction_index")
    plan = corpus.shard_plan()
    if not isinstance(index, list) or len(index) != len(plan):
        raise RuntimeError("final receipt prediction index is incomplete")
    expected_columns = corpus.private_prediction_columns()
    if FORBIDDEN_COLUMNS.intersection(expected_columns):
        raise RuntimeError("private prediction schema contains forbidden direct data")
    seen_opaque_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    calibration_members = 0
    for shard, entry in zip(plan, index, strict=True):
        if not isinstance(entry, Mapping):
            raise RuntimeError("final receipt prediction index entry is malformed")
        descriptor = entry.get("private_predictions_artifact")
        if (
            entry.get("shard_id") != shard["shard_id"]
            or not isinstance(entry.get("receipt_id"), str)
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"relative_path", "sha256", "bytes", "row_count"}
            or descriptor.get("row_count") != shard["row_count"]
        ):
            raise RuntimeError("final receipt prediction shard descriptor drifted")
        source = (
            run_root
            / "full"
            / "prediction-shards"
            / f"shard={shard['shard_id']}"
            / "predictions.parquet"
        )
        expected_suffix = (
            f"run={final_receipt['authority_id']}/full/prediction-shards/"
            f"shard={shard['shard_id']}/predictions.parquet"
        )
        if (
            not str(descriptor["relative_path"]).endswith(expected_suffix)
            or not source.is_file()
            or source.stat().st_size != descriptor["bytes"]
            or file_sha256(source) != descriptor["sha256"]
        ):
            raise RuntimeError("downloaded prediction shard differs from final receipt")
        parquet = pq.ParquetFile(source)
        if (
            tuple(parquet.schema_arrow.names) != expected_columns
            or parquet.metadata.num_rows != shard["row_count"]
        ):
            raise RuntimeError("downloaded prediction shard schema or row count drifted")
        identity = parquet.read(
            columns=["corpus_position", "opaque_id", "calibration_member"]
        ).to_pylist()
        if [row["corpus_position"] for row in identity] != list(
            range(int(shard["start"]), int(shard["stop"]))
        ):
            raise RuntimeError("downloaded prediction shard position sequence drifted")
        opaque_ids = [row["opaque_id"] for row in identity]
        if (
            any(not isinstance(value, str) or not value for value in opaque_ids)
            or len(opaque_ids) != len(set(opaque_ids))
            or seen_opaque_ids.intersection(opaque_ids)
            or any(type(row["calibration_member"]) is not bool for row in identity)
        ):
            raise RuntimeError("downloaded prediction shard private identity contract drifted")
        seen_opaque_ids.update(opaque_ids)
        calibration_members += sum(row["calibration_member"] for row in identity)
        validated.append(
            {
                "shard_id": shard["shard_id"],
                "source": source,
                "row_count": shard["row_count"],
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "prediction_receipt_id": entry["receipt_id"],
            }
        )
    if (
        len(seen_opaque_ids) != corpus.CORPUS_ROWS
        or calibration_members != candidate.CALIBRATION_ROWS
    ):
        raise RuntimeError("downloaded prediction union does not conserve the registered corpus")
    return validated


def _dataset_card(*, export_id: str, final_receipt: Mapping[str, Any]) -> str:
    fields = "\n".join(f"- `{field}`" for field in corpus.private_prediction_columns())
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Reddit China stance: private calibrated ModernBERT corpus predictions

This private dataset contains calibrated row-level predictions for the **908,141-row
post-acquisition eligible candidate corpus**. It is not all Reddit and not the 548,963,310-row
retained source corpus. Export ID: `{export_id}`. Corpus-inference receipt:
`{final_receipt["receipt_id"]}`.

## Evidence and privacy boundary

These are provisional, model-assisted predictions, not human-validated labels, prevalence truth,
causal evidence or an independent test set. The 600 calibration members are retained with an
explicit flag; default thesis aggregates use the 907,541 rows where `calibration_member=false`.
`mixed` is outside the frozen three-class stance estimand.

The export contains no Reddit text, authors, direct record/thread/comment/submission identifiers,
teacher labels, prompts, reasoning traces or logits. `opaque_id` is the private candidate-pipeline
pseudonym and must not be published. Keep this Hub repository private and restrict access to
authorised thesis collaborators.

## Fields

{fields}

The 120 Parquet files preserve the exact immutable corpus shard order. `manifest.json` binds every
file hash to the source authority, calibration, checkpoint cohort, aggregate and final receipt.
`aggregates/aggregate.json` contains metadata-only counts for the full scored scope and the default
label-unseen scope.
"""


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _publish_directory(staging: Path, output_dir: Path) -> None:
    hashes = _directory_hashes(staging)
    if output_dir.exists():
        if not output_dir.is_dir() or _directory_hashes(output_dir) != hashes:
            raise RuntimeError(f"immutable private Hub export differs: {output_dir}")
        shutil.rmtree(staging)
        return
    os.rename(staging, output_dir)


def build_private_hf_dataset(
    *,
    run_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_authority_id: str,
    expected_final_receipt_id: str,
    repo_id: str = DEFAULT_REPO_ID,
) -> dict[str, Any]:
    """Build or exact-validate the immutable private Hub-ready directory."""

    final_receipt_path = run_root / "final" / "receipt.json"
    final_receipt = _validate_final_receipt(
        _read_object(final_receipt_path),
        expected_authority_id=expected_authority_id,
        expected_receipt_id=expected_final_receipt_id,
    )
    shards = _validate_prediction_shards(run_root=run_root, final_receipt=final_receipt)
    aggregate_descriptor = final_receipt.get("aggregate_artifact")
    aggregate_path = run_root / "final" / "aggregate.json"
    if (
        not isinstance(aggregate_descriptor, Mapping)
        or set(aggregate_descriptor) != {"relative_path", "sha256", "bytes"}
        or not aggregate_path.is_file()
        or aggregate_path.stat().st_size != aggregate_descriptor["bytes"]
        or file_sha256(aggregate_path) != aggregate_descriptor["sha256"]
    ):
        raise RuntimeError("downloaded aggregate differs from final receipt")

    output_files = {
        f"data/train-{index:05d}-of-{corpus.SHARD_COUNT:05d}.parquet": {
            "sha256": shard["sha256"],
            "bytes": shard["bytes"],
            "row_count": shard["row_count"],
            "source_shard_id": shard["shard_id"],
            "prediction_receipt_id": shard["prediction_receipt_id"],
        }
        for index, shard in enumerate(shards)
    }
    output_files["aggregates/aggregate.json"] = {
        "sha256": aggregate_descriptor["sha256"],
        "bytes": aggregate_descriptor["bytes"],
    }
    output_files["provenance/corpus-inference-receipt.json"] = {
        "sha256": file_sha256(final_receipt_path),
        "bytes": final_receipt_path.stat().st_size,
    }
    contract = {
        "schema_version": "1.0.0",
        "kind": "private-modernbert-probability-random-corpus-hf-export-contract-v1",
        "repo_id": repo_id,
        "access": "private",
        "authority_id": final_receipt["authority_id"],
        "source_bundle_id": final_receipt["source_bundle_id"],
        "plan_id": final_receipt["plan_id"],
        "final_receipt_id": final_receipt["receipt_id"],
        "final_receipt_file_sha256": file_sha256(final_receipt_path),
        "checkpoint_cohort_id": final_receipt["checkpoint_cohort_id"],
        "calibration_run_id": final_receipt["calibration_run_id"],
        "calibration_receipt_id": final_receipt["calibration_receipt_id"],
        "calibration_id": final_receipt["calibration_id"],
        "aggregate_id": final_receipt["aggregate_id"],
        "corpus_rows": corpus.CORPUS_ROWS,
        "calibration_member_rows": candidate.CALIBRATION_ROWS,
        "default_label_unseen_rows": corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS,
        "shard_count": corpus.SHARD_COUNT,
        "columns": list(corpus.private_prediction_columns()),
        "output_files": output_files,
        "privacy": {
            "contains_reddit_text": False,
            "contains_author_data": False,
            "contains_direct_reddit_ids": False,
            "contains_opaque_candidate_ids": True,
            "contains_row_level_model_predictions": True,
            "contains_teacher_labels": False,
            "contains_logits": False,
            "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        },
        "locked_test_rows_accessed": 0,
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }
    export_id = canonical_sha256(contract)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for index, shard in enumerate(shards):
            destination = staging / f"data/train-{index:05d}-of-{corpus.SHARD_COUNT:05d}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(shard["source"], destination)
            if file_sha256(destination) != shard["sha256"]:
                raise RuntimeError("copied private prediction shard hash drifted")
        aggregate_destination = staging / "aggregates/aggregate.json"
        aggregate_destination.parent.mkdir(parents=True)
        shutil.copyfile(aggregate_path, aggregate_destination)
        receipt_destination = staging / "provenance/corpus-inference-receipt.json"
        receipt_destination.parent.mkdir(parents=True)
        shutil.copyfile(final_receipt_path, receipt_destination)
        card_path = staging / "README.md"
        card_path.write_text(
            _dataset_card(export_id=export_id, final_receipt=final_receipt), encoding="utf-8"
        )
        manifest = {
            **contract,
            "kind": "private-modernbert-probability-random-corpus-hf-export-manifest-v1",
            "status": "complete",
            "export_id": export_id,
            "readme": {"sha256": file_sha256(card_path), "bytes": card_path.stat().st_size},
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        _publish_directory(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _read_object(output_dir / "manifest.json")


def _remote_file_hashes(*, api: Any, repo_id: str, revision: str) -> dict[str, str]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    with tempfile.TemporaryDirectory(prefix="hf-remote-verify-") as directory:
        hashes: dict[str, str] = {}
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
            token=True,
        ):
            if not isinstance(entry, RepoFile):
                continue
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=entry.path,
                    revision=revision,
                    token=True,
                    local_dir=directory,
                    force_download=True,
                )
            )
            hashes[entry.path] = file_sha256(path)
    return hashes


def publish_private_hf_dataset(
    *, output_dir: Path, repo_id: str, expected_export_id: str
) -> dict[str, Any]:
    """Create/update a private dataset repo and verify every remote file by download."""

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    manifest = _read_object(output_dir / "manifest.json")
    if (
        manifest.get("export_id") != expected_export_id
        or manifest.get("repo_id") != repo_id
        or manifest.get("access") != "private"
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("private Hub publication authority drifted")
    api = HfApi(token=True)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id=repo_id, token=True)
    if info.private is not True:
        raise RuntimeError("refusing to upload corpus predictions to a non-private repository")
    try:
        existing_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="manifest.json",
            token=True,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        existing = None
    else:
        existing = _read_object(Path(existing_path))
    if existing is not None and existing.get("export_id") != expected_export_id:
        raise RuntimeError("private Hub repository already contains a different immutable export")
    if existing is None:
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=output_dir,
            commit_message="data: publish exact calibrated ModernBERT corpus predictions",
            token=True,
        )
        revision = commit.oid
    else:
        revision = info.sha
    remote_info = api.dataset_info(repo_id=repo_id, revision=revision, token=True)
    if remote_info.private is not True:
        raise RuntimeError("private Hub repository visibility changed during publication")
    local_hashes = _directory_hashes(output_dir)
    remote_inventory_hashes = _remote_file_hashes(api=api, repo_id=repo_id, revision=revision)
    server_managed_paths = set(remote_inventory_hashes) - set(local_hashes)
    remote_payload_hashes = {
        path: digest for path, digest in remote_inventory_hashes.items() if path in local_hashes
    }
    if server_managed_paths != {".gitattributes"} or remote_payload_hashes != local_hashes:
        raise RuntimeError("private Hub remote file inventory or hashes differ from local export")
    receipt_body = {
        "schema_version": "1.0.0",
        "kind": "private-modernbert-probability-random-corpus-hf-publication-receipt-v1",
        "repo_id": repo_id,
        "revision": revision,
        "private": True,
        "export_id": expected_export_id,
        "payload_file_count": len(remote_payload_hashes),
        "server_managed_files": sorted(server_managed_paths),
        "remote_payload_file_hashes_sha256": canonical_sha256(remote_payload_hashes),
        "locked_test_rows_accessed": 0,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "status": "complete",
    }
    return {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--authority-id", required=True)
    parser.add_argument("--final-receipt-id", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-export-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_private_hf_dataset(
        run_root=args.run_root,
        output_dir=args.output_dir,
        expected_authority_id=args.authority_id,
        expected_final_receipt_id=args.final_receipt_id,
        repo_id=args.repo_id,
    )
    result: Mapping[str, Any] = manifest
    if args.publish:
        if args.expected_export_id != manifest["export_id"]:
            raise RuntimeError("--publish requires the exact built --expected-export-id")
        result = publish_private_hf_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            expected_export_id=args.expected_export_id,
        )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
