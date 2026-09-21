"""Durable launcher for a selected set of canonical Parquet conversions."""

from __future__ import annotations

import json
from pathlib import Path

from reddit_china_stance.modal_parquet import _code_state, app, convert_archive

CONFIRMATION = "LAUNCH_DURABLE_PARQUET_BATCH"
EXPECTED_SOURCE_FILES = {
    "AskReddit_comments.zst",
    "funny_comments.zst",
    "gaming_comments.zst",
    "news_comments.zst",
    "worldnews_comments.zst",
}


def _parse_sources(value: str) -> list[str]:
    sources = [item.strip() for item in value.split(",") if item.strip()]
    if not sources:
        raise ValueError("at least one source file is required")
    if len(sources) != len(set(sources)):
        raise ValueError("source files must be unique")
    if set(sources) != EXPECTED_SOURCE_FILES:
        missing = sorted(EXPECTED_SOURCE_FILES - set(sources))
        unexpected = sorted(set(sources) - EXPECTED_SOURCE_FILES)
        raise ValueError(
            f"durable continuation must contain exactly the five pending sources: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return sources


@app.local_entrypoint()
def durable_batch(
    source_files: str = "",
    manifest_path: str = "configs/source-files.json",
    confirm: str = "",
) -> None:
    """Submit independent durable calls in the exact requested priority order."""

    if confirm != CONFIRMATION:
        raise ValueError(f"refusing durable launch: pass --confirm {CONFIRMATION}")

    from reddit_china_stance.source_manifest import load_source_manifest

    root = Path(__file__).resolve().parents[2]
    resolved_manifest_path = root / manifest_path
    manifest = load_source_manifest(resolved_manifest_path)
    requested_sources = _parse_sources(source_files)
    sources_by_path = {source["path"]: source for source in manifest["files"]}
    unknown = sorted(set(requested_sources) - set(sources_by_path))
    if unknown:
        raise ValueError(f"source files are not in the pinned manifest: {unknown}")

    code_state = _code_state(root, resolved_manifest_path)
    jobs = []
    for source_file in requested_sources:
        jobs.append(
            {
                "dataset_id": manifest["dataset_id"],
                "revision": manifest["revision"],
                "source": sources_by_path[source_file],
                "code_state": code_state,
            }
        )
    convert_archive.spawn_map(jobs)
    print(f"DURABLE submitted batch sources={requested_sources}", flush=True)

    print(
        json.dumps(
            {
                "status": "durable_batch_submitted",
                "revision": manifest["revision"],
                "code_sha256": code_state["code_sha256"],
                "source_files": requested_sources,
            },
            indent=2,
            sort_keys=True,
        )
    )
