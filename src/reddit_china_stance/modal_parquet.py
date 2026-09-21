"""Restartable Modal conversion from verified raw archives to normalised Parquet."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance.ingestion import (
    SourceRowExcluded,
    infer_content_type,
    infer_subreddit,
    normalise_fullname,
    normalise_source_row,
)

APP_NAME = "reddit-china-stance-parquet"
VOLUME_NAME = "reddit-china-stance-data"
ENVIRONMENT_NAME = "main"
VOLUME_PATH = Path("/data")
SOURCE_SCHEMA_VERSION = "1.0.2"
CONVERT_ALL_CONFIRMATION = "CONVERT_ALL_20_ARCHIVES"
CONVERT_ONE_CONFIRMATION = "CONVERT_ONE_ARCHIVE"
ROW_BATCH_SIZE = 50_000

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "pyarrow==25.0.1",
        "pydantic>=2.11.0,<3",
        "zstandard==0.25.0",
    )
    .add_local_python_source("reddit_china_stance")
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class HashingReader:
    """File-like reader that hashes compressed source bytes as they are consumed."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk

    def readable(self) -> bool:
        return True


def _record_id_for_exclusion(row: Any, source_file: str) -> str | None:
    if not isinstance(row, dict):
        return None
    prefix = "t3_" if source_file.endswith("_submissions.zst") else "t1_"
    try:
        return normalise_fullname(row.get("id"), prefix)
    except SourceRowExcluded:
        return None


def _receipt_matches(receipt: dict[str, Any], *, source: dict[str, Any], code_sha256: str) -> bool:
    return (
        receipt.get("status") in {"converted", "already_converted"}
        and receipt.get("source_file") == source["path"]
        and receipt.get("source_sha256") == source["sha256"]
        and receipt.get("source_compressed_bytes") == source["size"]
        and receipt.get("source_schema_version") == SOURCE_SCHEMA_VERSION
        and receipt.get("code_sha256") == code_sha256
    )


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=(4096, 8192),
    timeout=86_400,
    max_containers=4,
)
def convert_archive(job: dict[str, Any]) -> dict[str, Any]:
    """Convert one verified archive with bounded memory and exact count conservation."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard

    required_job_keys = {"dataset_id", "revision", "source", "code_state"}
    if set(job) != required_job_keys:
        raise ValueError(f"conversion job keys must be exactly {sorted(required_job_keys)}")
    dataset_id = str(job["dataset_id"])
    revision = str(job["revision"])
    source = job["source"]
    code_state = job["code_state"]
    source_file = str(source["path"])
    source_stem = Path(source_file).stem
    content_type = infer_content_type(source_file).value
    subreddit = infer_subreddit(source_file)

    raw_path = VOLUME_PATH / "raw" / revision / source_file
    download_receipt_path = (
        VOLUME_PATH / "manifests" / "downloads" / revision / f"{source_file}.json"
    )
    schema_root = VOLUME_PATH / "normalised" / revision / f"schema={SOURCE_SCHEMA_VERSION}"
    final_dir = schema_root / source_stem
    staging_dir = schema_root / f".{source_stem}.incomplete"
    receipt_path = final_dir / "_receipt.json"

    volume.reload()
    if staging_dir.exists():
        raise FileExistsError(
            f"stale incomplete conversion requires explicit inspection: {staging_dir}"
        )
    if not raw_path.exists() or not download_receipt_path.exists():
        raise FileNotFoundError(f"verified raw archive is not cached: {source_file}")
    download_receipt = json.loads(download_receipt_path.read_text(encoding="utf-8"))
    if (
        raw_path.stat().st_size != int(source["size"])
        or download_receipt.get("sha256") != source["sha256"]
        or download_receipt.get("compressed_bytes") != source["size"]
    ):
        raise RuntimeError(f"raw archive receipt mismatch for {source_file}")

    if final_dir.exists():
        if not receipt_path.exists():
            raise RuntimeError(f"normalised output lacks receipt: {final_dir}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not _receipt_matches(receipt, source=source, code_sha256=code_state["code_sha256"]):
            raise RuntimeError(
                f"existing normalised output does not match code/input: {source_file}"
            )
        for partition in receipt.get("partitions", []):
            partition_path = final_dir / partition["relative_path"]
            if (
                not partition_path.exists()
                or partition_path.stat().st_size != partition["bytes"]
                or _sha256_file(partition_path) != partition["sha256"]
            ):
                raise RuntimeError(f"existing partition state mismatch: {partition_path}")
        exclusion_partition = receipt.get("exclusion_partition")
        if int(receipt.get("excluded_rows", -1)) > 0:
            if not isinstance(exclusion_partition, dict):
                raise RuntimeError(f"missing exclusion-partition receipt: {source_file}")
            exclusion_path = final_dir / exclusion_partition["relative_path"]
            if (
                not exclusion_path.exists()
                or exclusion_path.stat().st_size != exclusion_partition["bytes"]
                or _sha256_file(exclusion_path) != exclusion_partition["sha256"]
            ):
                raise RuntimeError(f"existing exclusion state mismatch: {exclusion_path}")
        elif exclusion_partition is not None:
            raise RuntimeError(f"unexpected exclusion-partition receipt: {source_file}")
        receipt = dict(receipt)
        receipt["status"] = "already_converted"
        print(f"PARQUET already verified {source_file}", flush=True)
        return receipt

    source_schema = pa.schema(
        [
            ("schema_version", pa.string()),
            ("record_id", pa.string()),
            ("source_id", pa.string()),
            ("content_type", pa.string()),
            ("subreddit", pa.string()),
            ("stratum", pa.string()),
            ("created_utc", pa.timestamp("us", tz="UTC")),
            ("year", pa.int32()),
            ("month", pa.int8()),
            ("language_status", pa.string()),
            ("text", pa.string()),
            ("text_sha256", pa.string()),
            ("submission_id", pa.string()),
            ("parent_id", pa.string()),
            ("source_file", pa.string()),
        ]
    )
    exclusion_schema = pa.schema(
        [
            ("record_id", pa.string()),
            ("source_file", pa.string()),
            ("reason", pa.string()),
        ]
    )

    staging_dir.mkdir(parents=True)
    writers: dict[int, Any] = {}
    partition_paths: dict[int, Path] = {}
    batches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    exclusion_writer: Any | None = None
    exclusion_batch: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    month_counts: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    physical_lines = 0
    retained_rows = 0
    started = time.monotonic()

    def flush_year(year: int) -> None:
        rows = batches[year]
        if not rows:
            return
        if year not in writers:
            relative = (
                Path(f"year={year}")
                / f"subreddit={subreddit}"
                / f"content_type={content_type}"
                / "part-00000.parquet"
            )
            path = staging_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            partition_paths[year] = path
            writers[year] = pq.ParquetWriter(path, source_schema, compression="zstd")
        writers[year].write_table(pa.Table.from_pylist(rows, schema=source_schema))
        rows.clear()

    def flush_exclusions() -> None:
        nonlocal exclusion_writer
        if not exclusion_batch:
            return
        if exclusion_writer is None:
            path = staging_dir / "_exclusions" / "part-00000.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            exclusion_writer = pq.ParquetWriter(path, exclusion_schema, compression="zstd")
        exclusion_writer.write_table(pa.Table.from_pylist(exclusion_batch, schema=exclusion_schema))
        exclusion_batch.clear()

    print(f"PARQUET start {source_file}", flush=True)
    try:
        with raw_path.open("rb") as compressed:
            hashed_source = HashingReader(compressed)
            with (
                zstandard.ZstdDecompressor().stream_reader(
                    hashed_source, closefd=False
                ) as decompressed,
                io.BufferedReader(decompressed) as buffered,
            ):
                for raw_line in buffered:
                    physical_lines += 1
                    try:
                        row = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        exclusions["malformed_json"] += 1
                        exclusion_batch.append(
                            {
                                "record_id": None,
                                "source_file": source_file,
                                "reason": "malformed_json",
                            }
                        )
                        if len(exclusion_batch) >= ROW_BATCH_SIZE:
                            flush_exclusions()
                        continue
                    if not isinstance(row, dict):
                        exclusions["malformed_json"] += 1
                        exclusion_batch.append(
                            {
                                "record_id": None,
                                "source_file": source_file,
                                "reason": "malformed_json",
                            }
                        )
                        if len(exclusion_batch) >= ROW_BATCH_SIZE:
                            flush_exclusions()
                        continue
                    try:
                        normalised = normalise_source_row(row, source_file=source_file)
                    except SourceRowExcluded as exc:
                        reason = exc.reason.value
                        exclusions[reason] += 1
                        exclusion_batch.append(
                            {
                                "record_id": _record_id_for_exclusion(row, source_file),
                                "source_file": source_file,
                                "reason": reason,
                            }
                        )
                        if len(exclusion_batch) >= ROW_BATCH_SIZE:
                            flush_exclusions()
                        continue

                    year = int(normalised["year"])
                    batches[year].append(normalised)
                    retained_rows += 1
                    year_counts[year] += 1
                    month_counts[f"{year:04d}-{int(normalised['month']):02d}"] += 1
                    if len(batches[year]) >= ROW_BATCH_SIZE:
                        flush_year(year)
                    if physical_lines % 5_000_000 == 0:
                        print(
                            f"PARQUET progress {source_file} lines={physical_lines} "
                            f"retained={retained_rows}",
                            flush=True,
                        )
            while hashed_source.read(8 * 1024 * 1024):
                pass
            source_bytes_read = hashed_source.bytes_read
            observed_source_sha256 = hashed_source.digest.hexdigest()

        if source_bytes_read != int(source["size"]):
            raise RuntimeError(
                f"compressed-byte count mismatch for {source_file}: "
                f"{source_bytes_read} != {source['size']}"
            )
        if observed_source_sha256 != source["sha256"]:
            raise RuntimeError(
                f"source SHA-256 mismatch during conversion for {source_file}: "
                f"{observed_source_sha256} != {source['sha256']}"
            )

        for year in sorted(batches):
            flush_year(year)
        flush_exclusions()
        for writer in writers.values():
            writer.close()
        if exclusion_writer is not None:
            exclusion_writer.close()

        excluded_rows = sum(exclusions.values())
        if physical_lines != retained_rows + excluded_rows:
            raise RuntimeError(
                f"count conservation failed for {source_file}: {physical_lines} != "
                f"{retained_rows} + {excluded_rows}"
            )
        if retained_rows == 0:
            raise RuntimeError(f"conversion retained zero rows for {source_file}")

        partitions = []
        for year, path in sorted(partition_paths.items()):
            relative = path.relative_to(staging_dir)
            partitions.append(
                {
                    "year": year,
                    "relative_path": str(relative),
                    "rows": year_counts[year],
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        exclusion_path = staging_dir / "_exclusions" / "part-00000.parquet"
        exclusion_partition = None
        if exclusion_path.exists():
            exclusion_partition = {
                "relative_path": str(exclusion_path.relative_to(staging_dir)),
                "rows": excluded_rows,
                "bytes": exclusion_path.stat().st_size,
                "sha256": _sha256_file(exclusion_path),
            }

        receipt = {
            "schema_version": "1.0.0",
            "status": "converted",
            "dataset_id": dataset_id,
            "revision": revision,
            "source_file": source_file,
            "source_compressed_bytes": source["size"],
            "source_sha256": source["sha256"],
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "language_status": "unclassified",
            "physical_lines": physical_lines,
            "retained_rows": retained_rows,
            "excluded_rows": excluded_rows,
            "exclusions": dict(sorted(exclusions.items())),
            "year_counts": {str(key): value for key, value in sorted(year_counts.items())},
            "month_counts": dict(sorted(month_counts.items())),
            "partitions": partitions,
            "exclusion_partition": exclusion_partition,
            "code_sha256": code_state["code_sha256"],
            "git_head": code_state["git_head"],
            "git_dirty": code_state["git_dirty"],
            "manifest_sha256": code_state["manifest_sha256"],
            "source_schema_sha256": code_state["source_schema_sha256"],
            "python_version": platform.python_version(),
            "pyarrow_version": pa.__version__,
            "zstandard_version": zstandard.__version__,
            "completed_at": datetime.now(UTC).isoformat(),
            "wall_seconds": round(time.monotonic() - started, 3),
        }
        (staging_dir / "_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging_dir.replace(final_dir)
        volume.commit()
        print(
            f"PARQUET complete {source_file} lines={physical_lines} retained={retained_rows} "
            f"excluded={excluded_rows}",
            flush=True,
        )
        return receipt
    except BaseException:
        for writer in writers.values():
            with suppress(Exception):
                writer.close()
        if exclusion_writer is not None:
            with suppress(Exception):
                exclusion_writer.close()
        if staging_dir.exists():
            volume.commit()
        raise


def _code_state(root: Path, manifest_path: Path) -> dict[str, Any]:
    tracked_inputs = [
        Path("src/reddit_china_stance/modal_parquet.py"),
        Path("src/reddit_china_stance/ingestion.py"),
        Path("src/reddit_china_stance/models.py"),
        Path("schemas/canonical-record.schema.json"),
        Path("uv.lock"),
    ]
    digest = hashlib.sha256()
    for relative in tracked_inputs:
        payload = (root / relative).read_bytes()
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    schema_path = root / "schemas" / "canonical-record.schema.json"
    return {
        "code_sha256": digest.hexdigest(),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    }


@app.local_entrypoint()
def main(
    manifest_path: str = "configs/source-files.json",
    source_file: str = "",
    confirm: str = "",
) -> None:
    """Launch one proof conversion or the complete four-way conversion fan-out."""

    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest_path = root / manifest_path
    manifest = load_source_manifest(resolved_manifest_path)
    selected_sources = manifest["files"]
    if source_file:
        if confirm != CONVERT_ONE_CONFIRMATION:
            raise ValueError(
                f"refusing single conversion: pass --confirm {CONVERT_ONE_CONFIRMATION}"
            )
        selected_sources = [source for source in manifest["files"] if source["path"] == source_file]
        if len(selected_sources) != 1:
            raise ValueError(f"source_file is not in the pinned manifest: {source_file}")
    elif confirm != CONVERT_ALL_CONFIRMATION:
        raise ValueError(f"refusing full conversion: pass --confirm {CONVERT_ALL_CONFIRMATION}")

    code_state = _code_state(root, resolved_manifest_path)
    jobs = [
        {
            "dataset_id": manifest["dataset_id"],
            "revision": manifest["revision"],
            "source": source,
            "code_state": code_state,
        }
        for source in sorted(selected_sources, key=lambda item: int(item["size"]))
    ]
    print(
        f"Launching {len(jobs)} normalised Parquet conversions with at most 4 containers; "
        f"code_sha256={code_state['code_sha256']}."
    )
    results = list(convert_archive.map(jobs, order_outputs=False))
    completed = {result["source_file"] for result in results}
    expected = {source["path"] for source in selected_sources}
    if completed != expected:
        raise RuntimeError(
            f"conversion result mismatch: missing={sorted(expected - completed)}, "
            f"unexpected={sorted(completed - expected)}"
        )
    print(
        json.dumps(
            {
                "status": "conversion_selection_complete",
                "archives": len(results),
                "physical_lines": sum(int(result["physical_lines"]) for result in results),
                "retained_rows": sum(int(result["retained_rows"]) for result in results),
                "excluded_rows": sum(int(result["excluded_rows"]) for result in results),
                "code_sha256": code_state["code_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
