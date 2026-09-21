"""Build and publish the private ModernBERT corpus with canonical Reddit text.

The export is an exact child of the current source-ID/timestamp Hub revision.  Text is kept in
the private repository for authorised word-cloud and corpus analysis; it is never copied into
the Git repository or public figures site.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from reddit_china_stance import export_private_hf_modernbert_corpus_v1 as base_export
from reddit_china_stance import modernbert_corpus_text_enrichment_v1 as contract
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256

DEFAULT_BASE_DIR = Path("data/private-hf-modernbert-probability-random-corpus-source-v2")
DEFAULT_OUTPUT_DIR = Path("data/private-hf-modernbert-probability-random-corpus-text-v1")
EVIDENCE_BOUNDARY = (
    "private canonical Reddit text joined to provisional model-assisted corpus predictions; "
    "not human-validated thesis truth"
)
FORBIDDEN_COLUMNS = frozenset(
    {
        "author",
        "body",
        "target_text",
        "submission_context",
        "parent_context",
        "thread_id",
        "submission_id",
        "comment_id",
        "label_json",
        "logits",
    }
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _validate_addressed(
    value: Mapping[str, Any], *, id_field: str, expected_kind: str
) -> dict[str, Any]:
    clean = dict(value)
    body = {key: item for key, item in clean.items() if key != id_field}
    if clean.get("kind") != expected_kind or clean.get(id_field) != canonical_sha256(body):
        raise RuntimeError(f"{expected_kind} content address drifted")
    return clean


def _validate_parent_export(base_dir: Path) -> dict[str, Any]:
    manifest_path = base_dir / "manifest.json"
    manifest = _read_object(manifest_path)
    if (
        file_sha256(manifest_path) != contract.HF_PREVIOUS_MANIFEST_SHA256
        or manifest.get("export_id") != contract.HF_PREVIOUS_EXPORT_ID
        or manifest.get("repo_id") != contract.HF_REPO_ID
        or manifest.get("access") != "private"
        or manifest.get("corpus_rows") != contract.CORPUS_ROWS
        or manifest.get("shard_count") != contract.SHARD_COUNT
        or manifest.get("columns") != list(contract.parent_prediction_columns())
        or manifest.get("privacy", {}).get("contains_reddit_text") is not False
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("current private Hub parent export binding drifted")
    return manifest


def _validate_text_root_inventory(root: Path, *, descriptors: Any) -> None:
    if not isinstance(descriptors, list) or len(descriptors) != contract.SHARD_COUNT:
        raise RuntimeError("text-enrichment output inventory is incomplete")
    shard_paths = {item.get("relative_path") for item in descriptors if isinstance(item, Mapping)}
    expected = {
        *shard_paths,
        "authority.json",
        "source-bundle.json",
        "canonical-inventory.json",
        "receipt.json",
    }
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise RuntimeError("text-enrichment root contains missing or unreceipted state")


def _validate_text_receipt(
    root: Path, *, expected_authority_id: str, expected_receipt_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = _validate_addressed(
        _read_object(root / "receipt.json"),
        id_field="receipt_id",
        expected_kind="modernbert-corpus-text-enrichment-receipt-v1",
    )
    authority = _validate_addressed(
        _read_object(root / "authority.json"),
        id_field="authority_id",
        expected_kind="modernbert-corpus-text-enrichment-authority-v1",
    )
    source_bundle = _validate_addressed(
        _read_object(root / "source-bundle.json"),
        id_field="source_bundle_id",
        expected_kind="modernbert-corpus-text-enrichment-source-bundle-v1",
    )
    inventory = _validate_addressed(
        _read_object(root / "canonical-inventory.json"),
        id_field="canonical_inventory_id",
        expected_kind="canonical-source-enrichment-inventory-v2",
    )
    _validate_text_root_inventory(root, descriptors=receipt.get("output_files"))
    if (
        receipt.get("receipt_id") != expected_receipt_id
        or receipt.get("authority_id") != expected_authority_id
        or authority.get("authority_id") != expected_authority_id
        or receipt.get("source_bundle_id") != source_bundle.get("source_bundle_id")
        or authority.get("source_bundle_id") != source_bundle.get("source_bundle_id")
        or receipt.get("canonical_inventory_id") != inventory.get("canonical_inventory_id")
        or authority.get("canonical_inventory_id") != inventory.get("canonical_inventory_id")
        or receipt.get("dataset_revision") != contract.DATASET_REVISION
        or receipt.get("source_schema_version") != contract.SOURCE_SCHEMA_VERSION
        or receipt.get("canonical_rows") != contract.CANONICAL_ROWS
        or receipt.get("corpus_rows") != contract.CORPUS_ROWS
        or receipt.get("shard_count") != contract.SHARD_COUNT
        or receipt.get("calibration_member_rows") != contract.CALIBRATION_ROWS
        or receipt.get("hf_previous_revision") != contract.HF_PREVIOUS_REVISION
        or receipt.get("hf_previous_export_id") != contract.HF_PREVIOUS_EXPORT_ID
        or receipt.get("parent_enrichment_authority_id")
        != contract.PARENT_ENRICHMENT_AUTHORITY_ID
        or receipt.get("parent_enrichment_receipt_id") != contract.PARENT_ENRICHMENT_RECEIPT_ID
        or receipt.get("parent_enrichment_source_bundle_id")
        != contract.PARENT_ENRICHMENT_SOURCE_BUNDLE_ID
        or receipt.get("parent_canonical_inventory_id")
        != contract.PARENT_CANONICAL_INVENTORY_ID
        or receipt.get("added_columns") != list(contract.ADDED_COLUMNS)
        or receipt.get("preserved_columns") != list(contract.parent_prediction_columns())
        or receipt.get("matched_rows") != contract.CORPUS_ROWS
        or receipt.get("distinct_source_ids") != contract.CORPUS_ROWS
        or receipt.get("distinct_canonical_record_ids") != contract.CORPUS_ROWS
        or receipt.get("missing_text_rows") != 0
        or receipt.get("missing_text_hash_rows") != 0
        or receipt.get("text_hash_mismatch_rows") != 0
        or receipt.get("metadata_mismatch_rows") != 0
        or receipt.get("contains_reddit_text") is not True
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("status") != "complete"
        or receipt.get("evidence_boundary") != EVIDENCE_BOUNDARY
    ):
        raise RuntimeError("text-enrichment receipt binding drifted")
    return receipt, authority, source_bundle, inventory


def _validate_text_shards(
    *, root: Path, base_dir: Path, receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    descriptors = receipt.get("output_files")
    if not isinstance(descriptors, list) or len(descriptors) != contract.SHARD_COUNT:
        raise RuntimeError("text-enrichment shard descriptor count drifted")
    expected_columns = contract.enriched_prediction_columns()
    if FORBIDDEN_COLUMNS.intersection(expected_columns):
        raise RuntimeError("text-enriched schema contains a forbidden column")
    seen_ids: set[str] = set()
    validated = []
    for shard, descriptor in zip(contract.shard_plan(), descriptors, strict=True):
        if not isinstance(descriptor, Mapping):
            raise RuntimeError("text-enrichment shard descriptor is malformed")
        relative = f"shards/shard={shard['shard_id']}/predictions.parquet"
        source = root / relative
        predecessor = base_dir / (
            f"data/train-{int(shard['shard_id']):05d}-of-{contract.SHARD_COUNT:05d}.parquet"
        )
        if (
            descriptor.get("shard_id") != shard["shard_id"]
            or descriptor.get("relative_path") != relative
            or descriptor.get("row_count") != shard["row_count"]
            or descriptor.get("parent_shard_sha256") != file_sha256(predecessor)
            or not source.is_file()
            or not predecessor.is_file()
            or source.stat().st_size != descriptor.get("bytes")
            or file_sha256(source) != descriptor.get("sha256")
        ):
            raise RuntimeError("text-enrichment shard differs from its receipt or parent")
        enriched = pq.read_table(source)
        parent = pq.read_table(predecessor)
        if (
            tuple(enriched.column_names) != expected_columns
            or enriched.num_rows != shard["row_count"]
            or not enriched.select(contract.parent_prediction_columns()).equals(parent)
            or not pa.types.is_string(enriched.schema.field("text").type)
            or enriched["text"].null_count
            or bool(pc.any(pc.equal(enriched["text"], "")).as_py())
        ):
            raise RuntimeError("text-enrichment changed the parent projection or text field")
        source_ids = enriched["source_id"].to_pylist()
        if (
            enriched["opaque_id"].to_pylist() != source_ids
            or any(not isinstance(value, str) or not value for value in source_ids)
            or len(source_ids) != len(set(source_ids))
            or seen_ids.intersection(source_ids)
        ):
            raise RuntimeError("text-enrichment source IDs are missing, duplicated, or reordered")
        positions = enriched["corpus_position"].to_pylist()
        if positions != list(range(int(shard["start"]), int(shard["stop"]))):
            raise RuntimeError("text-enrichment row order drifted")
        seen_ids.update(source_ids)
        validated.append(
            {
                "shard_id": shard["shard_id"],
                "source": source,
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "row_count": descriptor["row_count"],
                "parent_shard_sha256": descriptor["parent_shard_sha256"],
            }
        )
    if len(seen_ids) != contract.CORPUS_ROWS:
        raise RuntimeError("text-enrichment shard union does not conserve source IDs")
    return validated


def _dataset_card(*, export_id: str, receipt: Mapping[str, Any]) -> str:
    fields = "\n".join(f"- `{field}`" for field in contract.enriched_prediction_columns())
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Reddit China stance: private ModernBERT corpus predictions with canonical text

This private dataset contains calibrated row-level predictions for the **908,141-row
post-acquisition eligible candidate corpus**, now joined to the canonical `text` field for
authorised word-cloud and corpus analysis. It is not all Reddit and not the 548,963,310-row
retained canonical source corpus. Export ID: `{export_id}`. Text-enrichment receipt:
`{receipt["receipt_id"]}`.

## Evidence and privacy boundary

Predictions are provisional and model-assisted, not human-validated labels, prevalence truth,
causal evidence or an independent test set. The 600 calibration members remain explicitly
flagged; default thesis aggregates use the 907,541 rows where `calibration_member=false`.
`mixed` is outside the frozen three-class stance estimand.

This repository is private and contains direct public Reddit identifiers and raw canonical Reddit
text. `opaque_id` and `source_id` are Reddit fullnames; `created_utc` is the canonical UTC
timestamp. Do not publish this dataset or its row-level text. No authors, parent/submission
context, prompts, teacher labels, reasoning traces or logits are included.

The `text` field is the exact canonical ingestion field: a comment's normalised body, or a
submission's normalised title and self-text joined with two newlines. Deleted/removed and empty
rows were excluded by the canonical ingestion contract.

## Fields

{fields}

The 120 Parquet files preserve the immutable parent row and shard order. `manifest.json` binds
every file hash to the canonical inventory, text-enrichment authority, parent export and completed
corpus-inference receipt.
"""


def build_private_hf_dataset(
    *,
    enrichment_root: Path,
    base_dir: Path = DEFAULT_BASE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_authority_id: str,
    expected_receipt_id: str,
    repo_id: str = contract.HF_REPO_ID,
) -> dict[str, Any]:
    """Build or exact-validate the immutable text-bearing private Hub directory."""

    if repo_id != contract.HF_REPO_ID:
        raise RuntimeError("text enrichment may only update the frozen private Hub repository")
    parent_manifest = _validate_parent_export(base_dir)
    receipt, _authority, source_bundle, inventory = _validate_text_receipt(
        enrichment_root,
        expected_authority_id=expected_authority_id,
        expected_receipt_id=expected_receipt_id,
    )
    shards = _validate_text_shards(root=enrichment_root, base_dir=base_dir, receipt=receipt)
    parent_files = parent_manifest.get("output_files")
    if not isinstance(parent_files, Mapping):
        raise RuntimeError("private Hub parent output inventory is absent")
    copied: dict[str, Path] = {}
    for relative in parent_files:
        if not isinstance(relative, str) or not relative.startswith(("aggregates/", "provenance/")):
            continue
        path = base_dir / relative
        descriptor = parent_files[relative]
        if (
            not isinstance(descriptor, Mapping)
            or not path.is_file()
            or descriptor.get("sha256") != file_sha256(path)
            or descriptor.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"private Hub parent payload drifted: {relative}")
        copied[relative] = path
    copied.update(
        {
            "provenance/text-enrichment-receipt.json": enrichment_root / "receipt.json",
            "provenance/text-enrichment-authority.json": enrichment_root / "authority.json",
            "provenance/text-enrichment-source-bundle.json": enrichment_root / "source-bundle.json",
            "provenance/text-enrichment-canonical-inventory.json": (
                enrichment_root / "canonical-inventory.json"
            ),
        }
    )
    output_files: dict[str, dict[str, Any]] = {}
    for index, shard in enumerate(shards):
        output_files[f"data/train-{index:05d}-of-{contract.SHARD_COUNT:05d}.parquet"] = {
            key: shard[key]
            for key in ("sha256", "bytes", "row_count", "shard_id", "parent_shard_sha256")
        }
    for relative, path in copied.items():
        output_files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    export_body = {
        "schema_version": "1.0.0",
        "kind": "private-modernbert-corpus-text-hf-export-contract-v1",
        "repo_id": repo_id,
        "access": "private",
        "authority_id": receipt["authority_id"],
        "text_enrichment_receipt_id": receipt["receipt_id"],
        "source_bundle_id": source_bundle["source_bundle_id"],
        "canonical_inventory_id": inventory["canonical_inventory_id"],
        "dataset_revision": contract.DATASET_REVISION,
        "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
        "parent_hf_revision": contract.HF_PREVIOUS_REVISION,
        "parent_export_id": parent_manifest["export_id"],
        "parent_manifest_sha256": contract.HF_PREVIOUS_MANIFEST_SHA256,
        "parent_enrichment_authority_id": contract.PARENT_ENRICHMENT_AUTHORITY_ID,
        "parent_enrichment_receipt_id": contract.PARENT_ENRICHMENT_RECEIPT_ID,
        "corpus_rows": contract.CORPUS_ROWS,
        "calibration_member_rows": contract.CALIBRATION_ROWS,
        "default_label_unseen_rows": contract.CORPUS_ROWS - contract.CALIBRATION_ROWS,
        "shard_count": contract.SHARD_COUNT,
        "columns": list(contract.enriched_prediction_columns()),
        "text_source_field": "canonical_record.text",
        "text_hash_field": "canonical_record.text_sha256",
        "text_join_key": "canonical_record.record_id = parent.source_id",
        "output_files": output_files,
        "privacy": {
            "contains_reddit_text": True,
            "text_is_canonical_source_field": True,
            "contains_author_data": False,
            "contains_direct_reddit_ids": True,
            "opaque_id_is_direct_reddit_fullname": True,
            "source_id_is_direct_reddit_fullname": True,
            "contains_created_utc": True,
            "contains_row_level_model_predictions": True,
            "contains_teacher_labels": False,
            "contains_logits": False,
            "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        },
        "locked_test_rows_accessed": 0,
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }
    export_id = canonical_sha256(export_body)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for index, shard in enumerate(shards):
            destination = staging / f"data/train-{index:05d}-of-{contract.SHARD_COUNT:05d}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(shard["source"], destination)
            if file_sha256(destination) != shard["sha256"]:
                raise RuntimeError("copied text-enriched shard hash drifted")
        for relative, source in copied.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        readme = staging / "README.md"
        readme.write_text(_dataset_card(export_id=export_id, receipt=receipt), encoding="utf-8")
        manifest = {
            **export_body,
            "kind": "private-modernbert-corpus-text-hf-export-manifest-v1",
            "status": "complete",
            "export_id": export_id,
            "readme": {"sha256": file_sha256(readme), "bytes": readme.stat().st_size},
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        base_export._publish_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _read_object(output_dir / "manifest.json")


def publish_private_hf_dataset(
    *, output_dir: Path, repo_id: str, expected_export_id: str
) -> dict[str, Any]:
    """Publish one exact child of the current private Hub parent and verify every payload hash."""

    from huggingface_hub import HfApi, hf_hub_download

    manifest = _read_object(output_dir / "manifest.json")
    if (
        repo_id != contract.HF_REPO_ID
        or manifest.get("export_id") != expected_export_id
        or manifest.get("repo_id") != repo_id
        or manifest.get("access") != "private"
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("private Hub text-enrichment publication authority drifted")
    api = HfApi(token=True)
    before = api.dataset_info(repo_id=repo_id, token=True)
    if before.private is not True or before.sha != contract.HF_PREVIOUS_REVISION:
        raise RuntimeError("private Hub parent revision or visibility drifted")
    prior_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="manifest.json",
            revision=contract.HF_PREVIOUS_REVISION,
            token=True,
            force_download=True,
        )
    )
    prior = _read_object(prior_path)
    if (
        prior.get("export_id") != contract.HF_PREVIOUS_EXPORT_ID
        or file_sha256(prior_path) != contract.HF_PREVIOUS_MANIFEST_SHA256
    ):
        raise RuntimeError("private Hub parent manifest drifted")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        commit_message="data: add canonical Reddit text for word-cloud analysis",
        parent_commit=contract.HF_PREVIOUS_REVISION,
        token=True,
    )
    revision = commit.oid
    after = api.dataset_info(repo_id=repo_id, revision=revision, token=True)
    if after.private is not True:
        raise RuntimeError("private Hub visibility changed during text publication")
    local_hashes = base_export._directory_hashes(output_dir)
    remote_hashes = base_export._remote_file_hashes(api=api, repo_id=repo_id, revision=revision)
    extras = set(remote_hashes) - set(local_hashes)
    payload = {key: value for key, value in remote_hashes.items() if key in local_hashes}
    if extras != {".gitattributes"} or payload != local_hashes:
        raise RuntimeError("private Hub text export inventory or hashes differ from local export")
    body = {
        "schema_version": "1.0.0",
        "kind": "private-modernbert-corpus-text-hf-publication-receipt-v1",
        "repo_id": repo_id,
        "previous_revision": contract.HF_PREVIOUS_REVISION,
        "revision": revision,
        "private": True,
        "export_id": expected_export_id,
        "payload_file_count": len(payload),
        "server_managed_files": sorted(extras),
        "remote_payload_file_hashes_sha256": canonical_sha256(payload),
        "contains_reddit_text": True,
        "locked_test_rows_accessed": 0,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "status": "complete",
    }
    return {**body, "receipt_id": canonical_sha256(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrichment-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--authority-id", required=True)
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument("--repo-id", default=contract.HF_REPO_ID)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-export-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_private_hf_dataset(
        enrichment_root=args.enrichment_root,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        expected_authority_id=args.authority_id,
        expected_receipt_id=args.receipt_id,
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
