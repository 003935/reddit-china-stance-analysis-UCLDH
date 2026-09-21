"""Metadata-only reconciliation of the 20 canonical Parquet receipts on Modal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance.reconciliation import SOURCE_SCHEMA_VERSION, reconcile_receipts

APP_NAME = "reddit-china-stance-reconciliation"
VOLUME_NAME = "reddit-china-stance-data"
ENVIRONMENT_NAME = "main"
VOLUME_PATH = Path("/data")
CONFIRMATION = "RECONCILE_20_PARQUET_RECEIPTS"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pydantic>=2.11.0,<3")
    .add_local_python_source("reddit_china_stance")
)


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=0.5,
    memory=512,
    timeout=600,
    max_containers=1,
)
def reconcile_on_volume(
    *, source_manifest: dict[str, Any], thesis_claims: dict[str, Any]
) -> dict[str, Any]:
    """Read only final receipt JSON files and persist a text-free aggregate report."""

    revision = str(source_manifest["revision"])
    schema_root = VOLUME_PATH / "normalised" / revision / f"schema={SOURCE_SCHEMA_VERSION}"
    volume.reload()
    receipts = []
    for source in source_manifest["files"]:
        source_file = str(source["path"])
        receipt_path = schema_root / Path(source_file).stem / "_receipt.json"
        if not receipt_path.exists():
            raise FileNotFoundError(f"normalised receipt does not exist: {receipt_path}")
        receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))

    report = reconcile_receipts(
        receipts,
        source_manifest=source_manifest,
        thesis_claims=thesis_claims,
    )
    output_dir = (
        VOLUME_PATH / "manifests" / "reconciliation" / revision / f"schema={SOURCE_SCHEMA_VERSION}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_sha256 = hashlib.sha256(report_json.encode()).hexdigest()
    output_path = output_dir / f"report-{report_sha256}.json"
    temporary_path = output_dir / f".{output_path.name}.incomplete"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != report_json:
            raise RuntimeError(f"existing reconciliation report digest mismatch: {output_path}")
    else:
        if temporary_path.exists():
            raise FileExistsError(
                f"stale reconciliation output requires explicit inspection: {temporary_path}"
            )
        temporary_path.write_text(report_json, encoding="utf-8")
        temporary_path.replace(output_path)
        volume.commit()
    return {
        "report": report,
        "report_path": str(output_path.relative_to(VOLUME_PATH)),
        "report_sha256": report_sha256,
    }


@app.local_entrypoint()
def main(
    manifest_path: str = "configs/source-files.json",
    thesis_counts_path: str = "configs/thesis-counts.toml",
    confirm: str = "",
) -> None:
    """Validate and reconcile all final canonical receipts without reading Reddit text."""

    if confirm != CONFIRMATION:
        raise ValueError(f"refusing reconciliation: pass --confirm {CONFIRMATION}")

    from reddit_china_stance.reconciliation import load_thesis_claims
    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    source_manifest = load_source_manifest(root / manifest_path)
    thesis_claims = load_thesis_claims(root / thesis_counts_path)
    report = reconcile_on_volume.remote(
        source_manifest=source_manifest,
        thesis_claims=thesis_claims,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
