"""Metadata-only validation of one normalised Parquet source directory on Modal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "reddit-china-stance-parquet-validator"
VOLUME_NAME = "reddit-china-stance-data"
ENVIRONMENT_NAME = "main"
VOLUME_PATH = Path("/data")
SOURCE_SCHEMA_VERSION = "1.0.2"
CONFIRMATION = "VALIDATE_ONE_PARQUET_ARCHIVE"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow==25.0.1")


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=0.5,
    memory=1024,
    timeout=600,
    max_containers=1,
)
def validate_archive(
    *, revision: str, source_file: str, expected_code_sha256: str
) -> dict[str, Any]:
    """Validate recursive discovery, schema and row metadata without returning text."""

    import pyarrow as pa
    import pyarrow.dataset as ds

    source_stem = Path(source_file).stem
    source_dir = (
        VOLUME_PATH / "normalised" / revision / f"schema={SOURCE_SCHEMA_VERSION}" / source_stem
    )
    receipt_path = source_dir / "_receipt.json"
    volume.reload()
    if not receipt_path.exists():
        raise FileNotFoundError(f"normalised receipt does not exist: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("code_sha256") != expected_code_sha256:
        raise RuntimeError("normalised receipt code digest does not match validator input")

    dataset = ds.dataset(source_dir, format="parquet", partitioning="hive")
    if dataset.schema.field("year").type != pa.int32():
        raise RuntimeError(f"Hive year type is not int32: {dataset.schema.field('year').type}")
    if "author" in dataset.schema.names:
        raise RuntimeError("author field survived normalised Parquet")
    discovered_files = {str(Path(path).relative_to(source_dir)) for path in dataset.files}
    expected_files = {partition["relative_path"] for partition in receipt["partitions"]}
    if discovered_files != expected_files:
        missing = sorted(expected_files - discovered_files)
        unexpected = sorted(discovered_files - expected_files)
        raise RuntimeError(
            f"recursive Parquet discovery mismatch: missing={missing}, unexpected={unexpected}"
        )
    row_count = dataset.count_rows()
    if row_count != receipt["retained_rows"]:
        raise RuntimeError(f"Parquet row count mismatch: {row_count} != {receipt['retained_rows']}")
    return {
        "status": "validated",
        "source_file": source_file,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "files": len(discovered_files),
        "rows": row_count,
        "columns": dataset.schema.names,
        "year_type": str(dataset.schema.field("year").type),
        "ignored_receipt": "_receipt.json" not in discovered_files,
        "ignored_exclusions": not any(path.startswith("_exclusions/") for path in discovered_files),
    }


@app.local_entrypoint()
def main(
    source_file: str = "",
    manifest_path: str = "configs/source-files.json",
    confirm: str = "",
) -> None:
    """Validate one named source directory after conversion."""

    if confirm != CONFIRMATION:
        raise ValueError(f"refusing validation: pass --confirm {CONFIRMATION}")
    from reddit_china_stance.modal_parquet import _code_state
    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest_path = root / manifest_path
    manifest = load_source_manifest(resolved_manifest_path)
    if source_file not in {source["path"] for source in manifest["files"]}:
        raise ValueError(f"source_file is not in the pinned manifest: {source_file}")
    code_state = _code_state(root, resolved_manifest_path)
    result = validate_archive.remote(
        revision=manifest["revision"],
        source_file=source_file,
        expected_code_sha256=code_state["code_sha256"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
