"""Build and publish the source-enriched private ModernBERT corpus dataset."""

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

from reddit_china_stance import export_private_hf_modernbert_corpus_v1 as previous
from reddit_china_stance import modernbert_corpus_source_enrichment_v2 as contract
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance import modernbert_probability_random_corpus_inference_v1 as corpus
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256, file_sha256

DEFAULT_OUTPUT_DIR = Path("data/private-hf-modernbert-probability-random-corpus-source-v2")
EVIDENCE_BOUNDARY = (
    "private canonical identifiers and timestamps joined to provisional model-assisted "
    "corpus predictions; not human-validated thesis truth"
)
FORBIDDEN_COLUMNS = frozenset(
    {
        "author",
        "body",
        "text",
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


def _validate_content_addressed(
    value: Mapping[str, Any], *, id_field: str, expected_kind: str
) -> dict[str, Any]:
    clean = dict(value)
    body = {key: item for key, item in clean.items() if key != id_field}
    if clean.get("kind") != expected_kind or clean.get(id_field) != canonical_sha256(body):
        raise RuntimeError(f"{expected_kind} content address drifted")
    return clean


def _validate_enrichment_root_inventory(
    enrichment_root: Path, *, output_files: Any
) -> None:
    if not isinstance(output_files, list):
        raise RuntimeError("source-enrichment output inventory is absent")
    shard_paths = {
        item.get("relative_path") for item in output_files if isinstance(item, Mapping)
    }
    if (
        len(shard_paths) != contract.SHARD_COUNT
        or any(not isinstance(path, str) for path in shard_paths)
    ):
        raise RuntimeError("source-enrichment shard inventory drifted")
    expected_files = {
        *shard_paths,
        "authority.json",
        "source-bundle.json",
        "canonical-inventory.json",
        "receipt.json",
    }
    actual_files = {
        path.relative_to(enrichment_root).as_posix()
        for path in enrichment_root.rglob("*")
        if path.is_file()
    }
    expected_directories = {
        "shards",
        *(str(Path(path).parent) for path in shard_paths),
    }
    actual_directories = {
        path.relative_to(enrichment_root).as_posix()
        for path in enrichment_root.rglob("*")
        if path.is_dir()
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError("source-enrichment root contains missing or unreceipted state")


def _validate_enrichment_receipt(
    enrichment_root: Path, *, expected_authority_id: str, expected_receipt_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = _validate_content_addressed(
        previous._read_object(enrichment_root / "receipt.json"),
        id_field="receipt_id",
        expected_kind="modernbert-corpus-source-enrichment-receipt-v2",
    )
    authority = _validate_content_addressed(
        previous._read_object(enrichment_root / "authority.json"),
        id_field="authority_id",
        expected_kind="modernbert-corpus-source-enrichment-authority-v2",
    )
    source_bundle = _validate_content_addressed(
        previous._read_object(enrichment_root / "source-bundle.json"),
        id_field="source_bundle_id",
        expected_kind="modernbert-corpus-source-enrichment-source-bundle-v2",
    )
    inventory = _validate_content_addressed(
        previous._read_object(enrichment_root / "canonical-inventory.json"),
        id_field="canonical_inventory_id",
        expected_kind="canonical-source-enrichment-inventory-v2",
    )
    _validate_enrichment_root_inventory(
        enrichment_root, output_files=receipt.get("output_files")
    )
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
        or receipt.get("inference_authority_id") != contract.INFERENCE_AUTHORITY_ID
        or receipt.get("inference_final_receipt_id") != contract.INFERENCE_FINAL_RECEIPT_ID
        or receipt.get("inference_source_bundle_id") != contract.INFERENCE_SOURCE_BUNDLE_ID
        or receipt.get("hf_previous_revision") != contract.HF_PREVIOUS_REVISION
        or receipt.get("hf_previous_export_id") != contract.HF_PREVIOUS_EXPORT_ID
        or receipt.get("corpus_rows") != corpus.CORPUS_ROWS
        or receipt.get("shard_count") != corpus.SHARD_COUNT
        or receipt.get("added_columns") != list(contract.ADDED_COLUMNS)
        or receipt.get("timestamp_unit") != contract.TIMESTAMP_UNIT
        or receipt.get("timestamp_timezone") != contract.TIMESTAMP_TIMEZONE
        or receipt.get("matched_rows") != corpus.CORPUS_ROWS
        or receipt.get("distinct_source_ids") != corpus.CORPUS_ROWS
        or receipt.get("missing_rows") != 0
        or receipt.get("metadata_mismatch_rows") != 0
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("status") != "complete"
        or receipt.get("evidence_boundary") != EVIDENCE_BOUNDARY
    ):
        raise RuntimeError("source-enrichment receipt binding drifted")
    return receipt, authority, source_bundle, inventory


def _validate_previous_export(base_dir: Path) -> dict[str, Any]:
    manifest_path = base_dir / "manifest.json"
    manifest = previous._read_object(manifest_path)
    if (
        file_sha256(manifest_path) != contract.HF_PREVIOUS_MANIFEST_SHA256
        or manifest.get("export_id") != contract.HF_PREVIOUS_EXPORT_ID
        or manifest.get("repo_id") != contract.HF_REPO_ID
        or manifest.get("access") != "private"
        or manifest.get("corpus_rows") != corpus.CORPUS_ROWS
        or manifest.get("shard_count") != corpus.SHARD_COUNT
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("previous private Hub export binding drifted")
    return manifest


def _validate_timestamp(table: pa.Table) -> None:
    timestamp_type = pa.timestamp(contract.TIMESTAMP_UNIT, tz=contract.TIMESTAMP_TIMEZONE)
    if table.schema.field("created_utc").type != timestamp_type:
        raise RuntimeError("created_utc Arrow type drifted")
    if table["created_utc"].null_count:
        raise RuntimeError("created_utc contains nulls")
    if not bool(
        pc.all(pc.equal(pc.year(table["created_utc"]), table["year"].cast(pa.int64()))).as_py()
    ):
        raise RuntimeError("created_utc year differs from prediction year")


def _validate_enriched_shards(
    *, enrichment_root: Path, base_dir: Path, receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    descriptors = receipt.get("output_files")
    plan = corpus.shard_plan()
    if not isinstance(descriptors, list) or len(descriptors) != len(plan):
        raise RuntimeError("enrichment shard descriptor count drifted")
    expected_columns = contract.enriched_prediction_columns()
    if FORBIDDEN_COLUMNS.intersection(expected_columns):
        raise RuntimeError("enriched schema contains a forbidden column")
    seen_source_ids: set[str] = set()
    validated = []
    for index, (shard, descriptor) in enumerate(zip(plan, descriptors, strict=True)):
        if not isinstance(descriptor, Mapping):
            raise RuntimeError("enrichment shard descriptor is malformed")
        relative = f"shards/shard={shard['shard_id']}/predictions.parquet"
        source = enrichment_root / relative
        base = base_dir / f"data/train-{index:05d}-of-{corpus.SHARD_COUNT:05d}.parquet"
        if (
            descriptor.get("shard_id") != shard["shard_id"]
            or descriptor.get("relative_path") != relative
            or descriptor.get("row_count") != shard["row_count"]
            or not source.is_file()
            or not base.is_file()
            or source.stat().st_size != descriptor.get("bytes")
            or file_sha256(source) != descriptor.get("sha256")
        ):
            raise RuntimeError("enrichment shard differs from receipt")
        enriched = pq.read_table(source)
        predecessor = pq.read_table(base)
        if (
            tuple(enriched.column_names) != expected_columns
            or enriched.num_rows != shard["row_count"]
            or predecessor.num_rows != enriched.num_rows
            or not enriched.select(predecessor.column_names).equals(predecessor)
            or descriptor.get("source_prediction_sha256") != file_sha256(base)
        ):
            raise RuntimeError("enrichment changed the predecessor prediction projection")
        positions = enriched["corpus_position"].to_pylist()
        if positions != list(range(int(shard["start"]), int(shard["stop"]))):
            raise RuntimeError("enrichment row order drifted")
        opaque_ids = enriched["opaque_id"].to_pylist()
        source_ids = enriched["source_id"].to_pylist()
        if (
            opaque_ids != source_ids
            or any(not isinstance(value, str) or not value for value in source_ids)
            or len(source_ids) != len(set(source_ids))
            or seen_source_ids.intersection(source_ids)
        ):
            raise RuntimeError("canonical source IDs are missing, duplicated, or misjoined")
        _validate_timestamp(enriched)
        seen_source_ids.update(source_ids)
        validated.append(
            {
                "shard_id": shard["shard_id"],
                "source": source,
                "sha256": descriptor["sha256"],
                "bytes": descriptor["bytes"],
                "row_count": descriptor["row_count"],
                "source_prediction_sha256": descriptor["source_prediction_sha256"],
            }
        )
    if len(seen_source_ids) != corpus.CORPUS_ROWS:
        raise RuntimeError("enriched shard union does not conserve canonical source IDs")
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

# Reddit China stance: private calibrated ModernBERT corpus predictions

This private dataset contains calibrated row-level predictions for the **908,141-row
post-acquisition eligible candidate corpus**. It is not all Reddit and not the 548,963,310-row
retained canonical source corpus. Export ID: `{export_id}`. Source-enrichment receipt:
`{receipt["receipt_id"]}`.

## Evidence and identifier boundary

These are provisional model-assisted predictions, not human-validated labels, prevalence truth,
causal evidence or an independent test set. The 600 calibration members remain explicitly flagged;
default thesis aggregates use the 907,541 rows where `calibration_member=false`.

The dataset is private and contains direct public Reddit identifiers. `opaque_id` was historically
described as a pseudonym, but it is the canonical `record_id` and therefore the public Reddit
fullname. `source_id` preserves that same canonical fullname explicitly. `created_utc` is the full
canonical UTC timestamp (`timestamp[us, tz=UTC]`), with exact year agreement. Do not publish either
identifier column. No Reddit text, authors, prompts, teacher labels, reasoning traces or logits are
included.

## Fields

{fields}

The 120 Parquet files preserve the immutable predecessor row and shard order. `manifest.json` binds
every file hash to the canonical inventory, source-enrichment authority, predecessor export and
completed corpus-inference receipt.
"""


def build_private_hf_dataset(
    *,
    enrichment_root: Path,
    base_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_authority_id: str,
    expected_receipt_id: str,
    repo_id: str = contract.HF_REPO_ID,
) -> dict[str, Any]:
    """Build or exact-validate the immutable enriched private Hub directory."""

    if repo_id != contract.HF_REPO_ID:
        raise RuntimeError("source enrichment may only update the frozen private Hub repository")
    previous_manifest = _validate_previous_export(base_dir)
    receipt, _authority, source_bundle, inventory = _validate_enrichment_receipt(
        enrichment_root,
        expected_authority_id=expected_authority_id,
        expected_receipt_id=expected_receipt_id,
    )
    shards = _validate_enriched_shards(
        enrichment_root=enrichment_root, base_dir=base_dir, receipt=receipt
    )
    output_files: dict[str, dict[str, Any]] = {}
    for index, shard in enumerate(shards):
        output_files[f"data/train-{index:05d}-of-{corpus.SHARD_COUNT:05d}.parquet"] = {
            key: shard[key]
            for key in ("sha256", "bytes", "row_count", "shard_id", "source_prediction_sha256")
        }
    copied = {
        "aggregates/aggregate.json": base_dir / "aggregates/aggregate.json",
        "provenance/corpus-inference-receipt.json": (
            base_dir / "provenance/corpus-inference-receipt.json"
        ),
        "provenance/source-enrichment-receipt.json": enrichment_root / "receipt.json",
        "provenance/source-enrichment-authority.json": enrichment_root / "authority.json",
        "provenance/source-enrichment-source-bundle.json": enrichment_root / "source-bundle.json",
        "provenance/canonical-inventory.json": enrichment_root / "canonical-inventory.json",
    }
    predecessor_files = previous_manifest.get("output_files")
    if not isinstance(predecessor_files, Mapping):
        raise RuntimeError("previous private Hub output inventory is absent")
    for relative, path in copied.items():
        if not path.is_file():
            raise FileNotFoundError(f"required private provenance file is missing: {relative}")
        if relative in {
            "aggregates/aggregate.json",
            "provenance/corpus-inference-receipt.json",
        }:
            predecessor = predecessor_files.get(relative)
            if (
                not isinstance(predecessor, Mapping)
                or predecessor.get("sha256") != file_sha256(path)
                or predecessor.get("bytes") != path.stat().st_size
            ):
                raise RuntimeError(f"previous private Hub payload drifted: {relative}")
        output_files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    contract_body = {
        "schema_version": "2.0.0",
        "kind": "private-modernbert-corpus-source-hf-export-contract-v2",
        "repo_id": repo_id,
        "access": "private",
        "authority_id": receipt["authority_id"],
        "source_enrichment_receipt_id": receipt["receipt_id"],
        "source_bundle_id": source_bundle["source_bundle_id"],
        "canonical_inventory_id": inventory["canonical_inventory_id"],
        "dataset_revision": contract.DATASET_REVISION,
        "source_schema_version": contract.SOURCE_SCHEMA_VERSION,
        "inference_authority_id": contract.INFERENCE_AUTHORITY_ID,
        "inference_final_receipt_id": contract.INFERENCE_FINAL_RECEIPT_ID,
        "previous_hf_revision": contract.HF_PREVIOUS_REVISION,
        "previous_export_id": previous_manifest["export_id"],
        "previous_manifest_sha256": contract.HF_PREVIOUS_MANIFEST_SHA256,
        "corpus_rows": corpus.CORPUS_ROWS,
        "calibration_member_rows": candidate.CALIBRATION_ROWS,
        "default_label_unseen_rows": corpus.CORPUS_ROWS - candidate.CALIBRATION_ROWS,
        "shard_count": corpus.SHARD_COUNT,
        "columns": list(contract.enriched_prediction_columns()),
        "timestamp_type": "timestamp[us, tz=UTC]",
        "output_files": output_files,
        "privacy": {
            "contains_reddit_text": False,
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
    export_id = canonical_sha256(contract_body)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for index, shard in enumerate(shards):
            destination = staging / f"data/train-{index:05d}-of-{corpus.SHARD_COUNT:05d}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(shard["source"], destination)
        for relative, source in copied.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        card = staging / "README.md"
        card.write_text(_dataset_card(export_id=export_id, receipt=receipt), encoding="utf-8")
        manifest = {
            **contract_body,
            "kind": "private-modernbert-corpus-source-hf-export-manifest-v2",
            "status": "complete",
            "export_id": export_id,
            "readme": {"sha256": file_sha256(card), "bytes": card.stat().st_size},
        }
        (staging / "manifest.json").write_bytes(previous._json_bytes(manifest))
        previous._publish_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return previous._read_object(output_dir / "manifest.json")


def publish_private_hf_dataset(
    *, output_dir: Path, repo_id: str, expected_export_id: str
) -> dict[str, Any]:
    """Update the pinned private dataset HEAD and exact-verify every remote payload."""

    from huggingface_hub import HfApi, hf_hub_download

    manifest = previous._read_object(output_dir / "manifest.json")
    if (
        repo_id != contract.HF_REPO_ID
        or manifest.get("export_id") != expected_export_id
        or manifest.get("repo_id") != repo_id
        or manifest.get("access") != "private"
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("private Hub source-enrichment publication authority drifted")
    api = HfApi(token=True)
    before = api.dataset_info(repo_id=repo_id, token=True)
    if before.private is not True or before.sha != contract.HF_PREVIOUS_REVISION:
        raise RuntimeError("private Hub predecessor revision or visibility drifted")
    prior_manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="manifest.json",
            revision=contract.HF_PREVIOUS_REVISION,
            token=True,
            force_download=True,
        )
    )
    prior_manifest = previous._read_object(prior_manifest_path)
    if (
        prior_manifest.get("export_id") != contract.HF_PREVIOUS_EXPORT_ID
        or file_sha256(prior_manifest_path) != contract.HF_PREVIOUS_MANIFEST_SHA256
    ):
        raise RuntimeError("private Hub predecessor manifest drifted")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        commit_message="data: add canonical source IDs and timestamps",
        parent_commit=contract.HF_PREVIOUS_REVISION,
        token=True,
    )
    revision = commit.oid
    after = api.dataset_info(repo_id=repo_id, revision=revision, token=True)
    if after.private is not True:
        raise RuntimeError("private Hub visibility changed during publication")
    local_hashes = previous._directory_hashes(output_dir)
    remote_hashes = previous._remote_file_hashes(api=api, repo_id=repo_id, revision=revision)
    extras = set(remote_hashes) - set(local_hashes)
    payload = {key: value for key, value in remote_hashes.items() if key in local_hashes}
    if extras != {".gitattributes"} or payload != local_hashes:
        raise RuntimeError("private Hub remote inventory or hashes differ from enriched export")
    receipt_body = {
        "schema_version": "2.0.0",
        "kind": "private-modernbert-corpus-source-hf-publication-receipt-v2",
        "repo_id": repo_id,
        "previous_revision": contract.HF_PREVIOUS_REVISION,
        "revision": revision,
        "private": True,
        "export_id": expected_export_id,
        "payload_file_count": len(payload),
        "server_managed_files": sorted(extras),
        "remote_payload_file_hashes_sha256": canonical_sha256(payload),
        "locked_test_rows_accessed": 0,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "status": "complete",
    }
    return {**receipt_body, "receipt_id": canonical_sha256(receipt_body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrichment-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
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
