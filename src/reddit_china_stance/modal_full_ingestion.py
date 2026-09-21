"""Persistent, verified source-archive caching on Modal.

This module deliberately handles only the immutable raw-cache stage. Canonical Parquet conversion
is a separate command so a parser/schema change can never trigger another remote archive download.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "reddit-china-stance-full-ingestion"
VOLUME_NAME = "reddit-china-stance-data"
ENVIRONMENT_NAME = "main"
VOLUME_PATH = Path("/data")
CACHE_CONFIRMATION = "CACHE_ALL_20_ARCHIVES"
CACHE_ONE_CONFIRMATION = "CACHE_ONE_ARCHIVE"
RESUME_CONFIRMATION = "RESUME_INCOMPLETE_ARCHIVE"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = modal.Image.debian_slim(python_version="3.12")


def _source_url(dataset_id: str, revision: str, source_file: str) -> str:
    quoted_file = urllib.parse.quote(source_file, safe="/")
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{quoted_file}"


def _receipt_matches(
    receipt: dict[str, Any], *, dataset_id: str, revision: str, source: dict[str, Any]
) -> bool:
    return (
        receipt.get("status") in {"downloaded", "already_cached"}
        and receipt.get("dataset_id") == dataset_id
        and receipt.get("revision") == revision
        and receipt.get("source_file") == source["path"]
        and receipt.get("compressed_bytes") == source["size"]
        and receipt.get("sha256") == source["sha256"]
    )


def _validate_content_range(value: str | None, *, offset: int, expected_size: int) -> None:
    expected = f"bytes {offset}-{expected_size - 1}/{expected_size}"
    if value != expected:
        raise RuntimeError(f"unexpected Content-Range: expected {expected!r}, got {value!r}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=0.5,
    memory=1024,
    timeout=86_400,
    max_containers=4,
)
def cache_archive(job: dict[str, Any]) -> dict[str, Any]:
    """Cache one immutable archive, verify it, and commit a text-free receipt."""

    required_job_keys = {"dataset_id", "revision", "source", "resume_incomplete"}
    if set(job) != required_job_keys:
        raise ValueError(f"cache job keys must be exactly {sorted(required_job_keys)}")
    dataset_id = job["dataset_id"]
    revision = job["revision"]
    source = job["source"]
    if not isinstance(source, dict) or set(source) != {"path", "size", "sha256"}:
        raise ValueError("source must contain exactly path, size, and sha256")

    source_file = str(source["path"])
    expected_size = int(source["size"])
    expected_sha256 = str(source["sha256"])
    resume_incomplete = job["resume_incomplete"]
    if not isinstance(resume_incomplete, bool):
        raise ValueError("resume_incomplete must be boolean")
    archive_path = VOLUME_PATH / "raw" / revision / source_file
    incomplete_path = archive_path.with_suffix(f"{archive_path.suffix}.incomplete")
    receipt_path = VOLUME_PATH / "manifests" / "downloads" / revision / f"{source_file}.json"

    volume.reload()
    if incomplete_path.exists() and not resume_incomplete:
        raise FileExistsError(
            f"stale incomplete download requires explicit inspection: {incomplete_path}"
        )
    if archive_path.exists() or receipt_path.exists():
        if not archive_path.exists() or not receipt_path.exists():
            raise RuntimeError(
                f"cache state is incomplete for {source_file}: archive and receipt must coexist"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        observed_size = archive_path.stat().st_size
        observed_sha256 = _sha256_path(archive_path)
        if (
            observed_size != expected_size
            or observed_sha256 != expected_sha256
            or not _receipt_matches(
                receipt,
                dataset_id=dataset_id,
                revision=revision,
                source=source,
            )
        ):
            raise RuntimeError(f"existing cache does not match manifest for {source_file}")
        receipt = dict(receipt)
        receipt["status"] = "already_cached"
        print(f"CACHE already verified {source_file} ({observed_size} bytes)", flush=True)
        return receipt

    if resume_incomplete and not incomplete_path.exists():
        raise FileNotFoundError(
            f"explicit resume requested but no incomplete archive exists: {incomplete_path}"
        )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    digest = hashlib.sha256()
    bytes_written = incomplete_path.stat().st_size if incomplete_path.exists() else 0
    if bytes_written > expected_size:
        raise RuntimeError(
            f"incomplete archive exceeds manifest size for {source_file}: "
            f"{bytes_written} > {expected_size}"
        )
    if bytes_written:
        print(f"CACHE hash partial {source_file} ({bytes_written} bytes)", flush=True)
        with incomplete_path.open("rb") as partial:
            while chunk := partial.read(DOWNLOAD_CHUNK_BYTES):
                digest.update(chunk)
    next_progress = ((bytes_written // (1024**3)) + 1) * (1024**3)
    print(
        f"CACHE {'resume' if bytes_written else 'start'} {source_file} "
        f"({bytes_written}/{expected_size} bytes)",
        flush=True,
    )
    try:
        if bytes_written < expected_size:
            headers = {"User-Agent": "reddit-china-stance/0.1 persistent-cache"}
            if bytes_written:
                headers["Range"] = f"bytes={bytes_written}-"
            request = urllib.request.Request(
                _source_url(str(dataset_id), str(revision), source_file), headers=headers
            )
            mode = "ab" if bytes_written else "xb"
            with (
                urllib.request.urlopen(request, timeout=300) as response,
                incomplete_path.open(mode) as destination,
            ):
                status = response.getcode()
                if bytes_written:
                    if status != 206:
                        raise RuntimeError(
                            f"resume request for {source_file} returned HTTP {status}, not 206"
                        )
                    _validate_content_range(
                        response.headers.get("Content-Range"),
                        offset=bytes_written,
                        expected_size=expected_size,
                    )
                elif status != 200:
                    raise RuntimeError(
                        f"fresh download for {source_file} returned HTTP {status}, not 200"
                    )
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    destination.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
                    if bytes_written >= next_progress:
                        print(
                            f"CACHE progress {source_file} {bytes_written}/{expected_size}",
                            flush=True,
                        )
                        next_progress += 1 * 1024 * 1024 * 1024
                destination.flush()
                os.fsync(destination.fileno())

        observed_sha256 = digest.hexdigest()
        if bytes_written != expected_size:
            raise RuntimeError(
                f"size mismatch for {source_file}: expected {expected_size}, got {bytes_written}"
            )
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {source_file}: expected {expected_sha256}, "
                f"got {observed_sha256}"
            )
        incomplete_path.replace(archive_path)

        receipt = {
            "schema_version": "1.0.0",
            "status": "downloaded",
            "dataset_id": dataset_id,
            "revision": revision,
            "source_file": source_file,
            "compressed_bytes": bytes_written,
            "sha256": observed_sha256,
            "volume_path": str(archive_path),
            "completed_at": datetime.now(UTC).isoformat(),
            "wall_seconds": round(time.monotonic() - started, 3),
        }
        temporary_receipt = receipt_path.with_suffix(".json.incomplete")
        temporary_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_receipt.replace(receipt_path)
        volume.commit()
        print(
            f"CACHE complete {source_file} ({bytes_written} bytes, {observed_sha256})",
            flush=True,
        )
        return receipt
    except BaseException:
        # Preserve the incomplete file for explicit diagnosis. Never silently retry or replace it.
        if incomplete_path.exists():
            volume.commit()
        raise


@app.local_entrypoint()
def main(
    manifest_path: str = "configs/source-files.json",
    source_file: str = "",
    resume_incomplete: bool = False,
    confirm: str = "",
) -> None:
    """Launch the complete four-way cache fan-out after an explicit confirmation gate."""

    from reddit_china_stance.source_manifest import load_source_manifest

    manifest = load_source_manifest(Path(manifest_path))
    selected_sources = manifest["files"]
    if source_file:
        expected_confirmation = RESUME_CONFIRMATION if resume_incomplete else CACHE_ONE_CONFIRMATION
        if confirm != expected_confirmation:
            raise ValueError(
                f"refusing single-archive launch: pass --confirm {expected_confirmation}"
            )
        selected_sources = [source for source in manifest["files"] if source["path"] == source_file]
        if len(selected_sources) != 1:
            raise ValueError(f"source_file is not in the pinned manifest: {source_file}")
    else:
        if resume_incomplete:
            raise ValueError("resume_incomplete requires one explicit source_file")
        if confirm != CACHE_CONFIRMATION:
            raise ValueError(f"refusing cache launch: pass --confirm {CACHE_CONFIRMATION}")

    jobs = [
        {
            "dataset_id": manifest["dataset_id"],
            "revision": manifest["revision"],
            "source": source,
            "resume_incomplete": resume_incomplete,
        }
        for source in sorted(selected_sources, key=lambda item: int(item["size"]))
    ]
    selected_bytes = sum(int(source["size"]) for source in selected_sources)
    print(
        f"Launching {len(jobs)} verified downloads totalling "
        f"{selected_bytes} bytes with at most 4 containers."
    )
    results = list(cache_archive.map(jobs, order_outputs=False))
    completed_paths = {result["source_file"] for result in results}
    expected_paths = {job["source"]["path"] for job in jobs}
    if completed_paths != expected_paths:
        raise RuntimeError(
            f"cache result set mismatch: missing={sorted(expected_paths - completed_paths)}, "
            f"unexpected={sorted(completed_paths - expected_paths)}"
        )
    total_bytes = sum(int(result["compressed_bytes"]) for result in results)
    if total_bytes != selected_bytes:
        raise RuntimeError(f"cache byte reconciliation failed: {total_bytes} != {selected_bytes}")
    print(
        json.dumps(
            {
                "status": "cache_selection_complete",
                "dataset_id": manifest["dataset_id"],
                "revision": manifest["revision"],
                "archives": len(results),
                "compressed_bytes": total_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
