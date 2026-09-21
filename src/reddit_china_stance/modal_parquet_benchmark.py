"""Bounded Modal benchmark for the canonical Parquet conversion hot path."""

from __future__ import annotations

import io
import json
import resource
import time
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance.ingestion import SourceRowExcluded, normalise_source_row
from reddit_china_stance.modal_parquet import (
    ROW_BATCH_SIZE,
    VOLUME_PATH,
    HashingReader,
    _record_id_for_exclusion,
    image,
    volume,
)

APP_NAME = "reddit-china-stance-parquet-benchmark"
CONFIRMATION = "BENCHMARK_PARQUET_CPU"
DEFAULT_SOURCE_FILE = "AskReddit_comments.zst"
DEFAULT_LINES = 5_000_000
MIN_LINES = 1_000_000
MAX_LINES = 20_000_000
ALLOWED_CPUS = {2.0, 4.0, 8.0}

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=(4096, 8192),
    timeout=3_600,
    max_containers=1,
)
def benchmark_archive(*, revision: str, source_file: str, max_lines: int) -> dict[str, Any]:
    """Run the real parse, normalisation and Parquet-write path without persistent output."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard

    raw_path = VOLUME_PATH / "raw" / revision / source_file
    volume.reload()
    if not raw_path.exists():
        raise FileNotFoundError(f"cached benchmark archive does not exist: {source_file}")

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

    output_root = Path("/tmp/parquet-benchmark")
    output_root.mkdir(parents=True, exist_ok=True)
    writers: dict[int, Any] = {}
    batches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    exclusion_writer: Any | None = None
    exclusion_batch: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    month_counts: Counter[str] = Counter()
    physical_lines = 0
    retained_rows = 0

    def flush_year(year: int) -> None:
        rows = batches[year]
        if not rows:
            return
        if year not in writers:
            writers[year] = pq.ParquetWriter(
                output_root / f"year-{year}.parquet", source_schema, compression="zstd"
            )
        writers[year].write_table(pa.Table.from_pylist(rows, schema=source_schema))
        rows.clear()

    def flush_exclusions() -> None:
        nonlocal exclusion_writer
        if not exclusion_batch:
            return
        if exclusion_writer is None:
            exclusion_writer = pq.ParquetWriter(
                output_root / "exclusions.parquet", exclusion_schema, compression="zstd"
            )
        exclusion_writer.write_table(pa.Table.from_pylist(exclusion_batch, schema=exclusion_schema))
        exclusion_batch.clear()

    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.monotonic()
    compressed_bytes_read = 0
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
                    else:
                        if not isinstance(row, dict):
                            exclusions["malformed_json"] += 1
                            exclusion_batch.append(
                                {
                                    "record_id": None,
                                    "source_file": source_file,
                                    "reason": "malformed_json",
                                }
                            )
                        else:
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
                            else:
                                year = int(normalised["year"])
                                batches[year].append(normalised)
                                retained_rows += 1
                                year_counts[year] += 1
                                month_counts[f"{year:04d}-{int(normalised['month']):02d}"] += 1
                                if len(batches[year]) >= ROW_BATCH_SIZE:
                                    flush_year(year)
                    if len(exclusion_batch) >= ROW_BATCH_SIZE:
                        flush_exclusions()
                    if physical_lines >= max_lines:
                        break
            compressed_bytes_read = hashed_source.bytes_read

        for year in sorted(batches):
            flush_year(year)
        flush_exclusions()
    finally:
        for writer in writers.values():
            with suppress(Exception):
                writer.close()
        if exclusion_writer is not None:
            with suppress(Exception):
                exclusion_writer.close()

    wall_seconds = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (usage_after.ru_utime - usage_before.ru_utime) + (
        usage_after.ru_stime - usage_before.ru_stime
    )
    excluded_rows = sum(exclusions.values())
    if physical_lines != retained_rows + excluded_rows:
        raise RuntimeError("benchmark count conservation failed")
    return {
        "status": "benchmarked",
        "source_file": source_file,
        "max_lines": max_lines,
        "physical_lines": physical_lines,
        "retained_rows": retained_rows,
        "excluded_rows": excluded_rows,
        "compressed_bytes_read": compressed_bytes_read,
        "wall_seconds": round(wall_seconds, 3),
        "lines_per_second": round(physical_lines / wall_seconds, 1),
        "process_cpu_seconds": round(cpu_seconds, 3),
        "average_process_cores": round(cpu_seconds / wall_seconds, 3),
        "peak_rss_kib": usage_after.ru_maxrss,
        "year_counts": {str(key): value for key, value in sorted(year_counts.items())},
        "month_count_keys": len(month_counts),
    }


def _parse_cpu_sequence(value: str) -> list[float]:
    cpus = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not cpus or any(cpu not in ALLOWED_CPUS for cpu in cpus):
        raise ValueError(f"CPU sequence must use only {sorted(ALLOWED_CPUS)}")
    if set(cpus) != ALLOWED_CPUS:
        raise ValueError(f"CPU sequence must cover every allocation in {sorted(ALLOWED_CPUS)}")
    return cpus


@app.local_entrypoint()
def main(
    source_file: str = DEFAULT_SOURCE_FILE,
    lines: int = DEFAULT_LINES,
    cpu_sequence: str = "2,4,8,2",
    manifest_path: str = "configs/source-files.json",
    confirm: str = "",
) -> None:
    """Run sequential CPU allocations against the same bounded archive prefix."""

    if confirm != CONFIRMATION:
        raise ValueError(f"refusing benchmark: pass --confirm {CONFIRMATION}")
    if not MIN_LINES <= lines <= MAX_LINES:
        raise ValueError(f"lines must be between {MIN_LINES} and {MAX_LINES}")

    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    manifest = load_source_manifest(root / manifest_path)
    if source_file not in {source["path"] for source in manifest["files"]}:
        raise ValueError(f"source_file is not in the pinned manifest: {source_file}")

    results = []
    for cpu in _parse_cpu_sequence(cpu_sequence):
        print(f"BENCHMARK start cpu={cpu:g} lines={lines} source={source_file}", flush=True)
        result = benchmark_archive.with_options(cpu=cpu).remote(
            revision=manifest["revision"], source_file=source_file, max_lines=lines
        )
        result["requested_cpu"] = cpu
        results.append(result)
        print(
            f"BENCHMARK complete cpu={cpu:g} lines_per_second={result['lines_per_second']} "
            f"average_process_cores={result['average_process_cores']}",
            flush=True,
        )
    print(json.dumps({"status": "benchmark_complete", "results": results}, indent=2))
