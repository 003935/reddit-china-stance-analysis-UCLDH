"""Read-only Modal audit for retained submission deletion sentinels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import modal

APP_NAME = "reddit-china-stance-sentinel-audit"
VOLUME_NAME = "reddit-china-stance-data"
ENVIRONMENT_NAME = "main"
VOLUME_PATH = Path("/data")
CONFIRMATION = "AUDIT_ALL_10_SUBMISSION_SENTINELS"
AUDIT_SCHEMA_VERSION = "1.0.0"
SOURCE_SCHEMA_VERSION = "1.0.2"
EXPECTED_DATASET_ID = "awogies/7719723843r"
EXPECTED_REVISION = "97627c42893bc479c6a4952b1db38bbc33a60ac8"
EXPECTED_SOURCE_MANIFEST_SHA256 = "405769a29c64547e4dd8220b556b68aa7801c7dec04e9f678cb7cca36ad6455b"
YEARS = tuple(range(2020, 2026))
MARKERS = ("[deleted]", "[removed]")
BATCH_SIZE = 65_536
EXPECTED_SUBMISSION_SOURCES = (
    "AskReddit_submissions.zst",
    "China_submissions.zst",
    "ChineseLanguage_submissions.zst",
    "Sino_submissions.zst",
    "funny_submissions.zst",
    "gaming_submissions.zst",
    "geopolitics_submissions.zst",
    "news_submissions.zst",
    "todayilearned_submissions.zst",
    "worldnews_submissions.zst",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow==25.0.1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_audit_contract(manifest_json: str) -> dict[str, Any]:
    """Build the exact immutable audit contract from the pinned manifest bytes."""

    if hashlib.sha256(manifest_json.encode()).hexdigest() != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest bytes do not match the pinned audit manifest")
    manifest = json.loads(manifest_json)
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("source manifest dataset_id does not match the pinned audit dataset")
    if manifest.get("revision") != EXPECTED_REVISION:
        raise ValueError("source manifest revision does not match the pinned audit revision")
    sources = [
        source
        for source in manifest.get("files", [])
        if source["path"].endswith("_submissions.zst")
    ]
    if tuple(sorted(source["path"] for source in sources)) != EXPECTED_SUBMISSION_SOURCES:
        raise ValueError("source manifest does not contain the exact ten submission archives")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_id": EXPECTED_DATASET_ID,
        "revision": EXPECTED_REVISION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "years": list(YEARS),
        "markers": list(MARKERS),
        "sources": sorted(sources, key=lambda source: source["path"]),
    }


def _validate_contract(contract: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "dataset_id",
        "revision",
        "source_schema_version",
        "source_manifest_sha256",
        "years",
        "markers",
        "sources",
    }
    if set(contract) != expected_keys:
        raise ValueError(f"audit contract keys must be exactly {sorted(expected_keys)}")
    expected_values = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_id": EXPECTED_DATASET_ID,
        "revision": EXPECTED_REVISION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "years": list(YEARS),
        "markers": list(MARKERS),
    }
    for key, expected in expected_values.items():
        if contract.get(key) != expected:
            raise ValueError(f"audit contract does not pin the expected {key}")
    sources = contract["sources"]
    if not isinstance(sources, list):
        raise ValueError("audit contract sources must be a list")
    paths = tuple(source.get("path") for source in sources if isinstance(source, dict))
    if paths != EXPECTED_SUBMISSION_SOURCES or len(sources) != len(paths):
        raise ValueError("audit contract must contain the exact ordered submission sources")
    for source in sources:
        if set(source) != {"path", "size", "sha256"}:
            raise ValueError("audit source entries must contain exactly path, size, and sha256")
        if type(source["size"]) is not int or source["size"] <= 0:
            raise ValueError(f"invalid pinned source size for {source['path']}")
        digest = source["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid pinned source digest for {source['path']}")


def aggregate_marker_counts(
    source_results: Sequence[Mapping[str, Any]], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Strictly aggregate metadata-only per-source/year marker counts."""

    _validate_contract(contract)
    expected_sources = {source["path"] for source in contract["sources"]}
    observed_sources = [result.get("source_file") for result in source_results]
    if len(observed_sources) != len(set(observed_sources)):
        raise ValueError("sentinel audit returned duplicate source results")
    if set(observed_sources) != expected_sources:
        raise ValueError("sentinel audit did not return the exact submission source set")

    totals = {marker: 0 for marker in MARKERS}
    by_source: dict[str, Any] = {}
    for result in sorted(source_results, key=lambda row: str(row["source_file"])):
        source_file = str(result["source_file"])
        year_rows = result.get("by_year")
        if not isinstance(year_rows, Mapping) or set(year_rows) != {str(year) for year in YEARS}:
            raise ValueError(f"sentinel audit returned incomplete years for {source_file}")
        source_totals = {marker: 0 for marker in MARKERS}
        normalised_years: dict[str, dict[str, int]] = {}
        for year in YEARS:
            counts = year_rows[str(year)]
            if not isinstance(counts, Mapping) or set(counts) != set(MARKERS):
                raise ValueError(f"sentinel audit returned invalid marker counts for {source_file}")
            normalised: dict[str, int] = {}
            for marker in MARKERS:
                count = counts[marker]
                if type(count) is not int or count < 0:
                    raise ValueError(
                        f"sentinel count must be a non-negative integer: {source_file}"
                    )
                normalised[marker] = count
                source_totals[marker] += count
                totals[marker] += count
            normalised["total"] = sum(normalised.values())
            normalised_years[str(year)] = normalised
        by_source[source_file] = {
            "by_year": normalised_years,
            "totals": {**source_totals, "total": sum(source_totals.values())},
        }

    return {
        "status": "audited",
        "dataset_id": contract["dataset_id"],
        "revision": contract["revision"],
        "source_schema_version": contract["source_schema_version"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "markers": list(MARKERS),
        "by_source": by_source,
        "totals": {**totals, "total": sum(totals.values())},
    }


def _canonical_arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
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


def _audit_source(job: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if set(job) != {"confirm", "contract", "source_file"}:
        raise ValueError("sentinel audit job has unexpected fields")
    if job["confirm"] != CONFIRMATION:
        raise ValueError(f"refusing sentinel audit: pass --confirm {CONFIRMATION}")
    contract = job["contract"]
    _validate_contract(contract)
    source_file = job["source_file"]
    sources = {source["path"]: source for source in contract["sources"]}
    if source_file not in sources:
        raise ValueError("sentinel audit source is not in the exact submission manifest")
    source = sources[source_file]
    subreddit = source_file.removesuffix("_submissions.zst")
    source_dir = (
        VOLUME_PATH
        / "normalised"
        / EXPECTED_REVISION
        / f"schema={SOURCE_SCHEMA_VERSION}"
        / Path(source_file).stem
    )
    receipt_path = source_dir / "_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"canonical source receipt does not exist: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt = {
        "schema_version": "1.0.0",
        "status": "converted",
        "dataset_id": EXPECTED_DATASET_ID,
        "revision": EXPECTED_REVISION,
        "source_file": source_file,
        "source_compressed_bytes": source["size"],
        "source_sha256": source["sha256"],
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"canonical receipt mismatch for {source_file}: {key}")

    partitions = receipt.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != len(YEARS):
        raise RuntimeError(f"canonical receipt lacks six partitions for {source_file}")
    partitions_by_year = {int(partition["year"]): partition for partition in partitions}
    if set(partitions_by_year) != set(YEARS) or len(partitions_by_year) != len(partitions):
        raise RuntimeError(f"canonical receipt does not pin exact audit years for {source_file}")
    expected_year_counts = {str(year): int(partitions_by_year[year]["rows"]) for year in YEARS}
    if receipt.get("year_counts") != expected_year_counts:
        raise RuntimeError(f"canonical receipt year counts disagree for {source_file}")
    if sum(int(partition["rows"]) for partition in partitions) != receipt.get("retained_rows"):
        raise RuntimeError(f"canonical receipt row totals disagree for {source_file}")

    by_year: dict[str, dict[str, int]] = {}
    for year in YEARS:
        partition = partitions_by_year[year]
        relative_path = (
            Path(f"year={year}")
            / f"subreddit={subreddit}"
            / "content_type=submission"
            / "part-00000.parquet"
        )
        if partition.get("relative_path") != str(relative_path):
            raise RuntimeError(
                f"canonical receipt has wrong partition path for {source_file}/{year}"
            )
        path = source_dir / relative_path
        if not path.exists() or path.stat().st_size != partition.get("bytes"):
            raise RuntimeError(f"canonical partition size mismatch for {source_file}/{year}")
        if _sha256_file(path) != partition.get("sha256"):
            raise RuntimeError(f"canonical partition digest mismatch for {source_file}/{year}")

        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != _canonical_arrow_schema():
            raise RuntimeError(f"canonical Parquet schema mismatch for {source_file}/{year}")
        if parquet.metadata.num_rows != partition.get("rows"):
            raise RuntimeError(f"canonical Parquet row count mismatch for {source_file}/{year}")

        marker_filter = ds.field("text").isin(list(MARKERS))
        scanner = ds.dataset(path, format="parquet").scanner(
            columns=["text"], filter=marker_filter, batch_size=BATCH_SIZE, use_threads=True
        )
        counts = {marker: 0 for marker in MARKERS}
        for batch in scanner.to_batches():
            counted_rows = 0
            for marker in MARKERS:
                count = int(pc.sum(pc.equal(batch.column("text"), marker)).as_py() or 0)
                counts[marker] += count
                counted_rows += count
            if counted_rows != batch.num_rows:
                raise RuntimeError("filtered sentinel batch contained an unexpected value")
        by_year[str(year)] = counts
    return {"source_file": source_file, "by_year": by_year}


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=2.0,
    memory=4096,
    timeout=14_400,
    max_containers=10,
)
def audit_submission_source(job: dict[str, Any]) -> dict[str, Any]:
    """Verify and count one source, returning only aggregate sentinel counts."""

    volume.reload()
    return _audit_source(job)


@app.local_entrypoint()
def main(
    manifest_path: str = "configs/source-files.json",
    confirm: str = "",
) -> None:
    """Audit all ten pinned submission archives concurrently."""

    if confirm != CONFIRMATION:
        raise ValueError(f"refusing sentinel audit: pass --confirm {CONFIRMATION}")
    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest_path = root / manifest_path
    load_source_manifest(resolved_manifest_path)
    contract = make_audit_contract(resolved_manifest_path.read_text(encoding="utf-8"))
    jobs = [
        {"confirm": confirm, "contract": contract, "source_file": source["path"]}
        for source in contract["sources"]
    ]
    results = list(audit_submission_source.map(jobs, order_outputs=False))
    print(json.dumps(aggregate_marker_counts(results, contract=contract), indent=2, sort_keys=True))
