"""Validation helpers for the immutable Hugging Face source inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_ARCHIVE_COUNT = 20
EXPECTED_COMPRESSED_BYTES = 169_284_685_552


def load_source_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate the pinned source inventory."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    required_top_level = {
        "schema_version",
        "dataset_id",
        "revision",
        "compressed_bytes",
        "files",
    }
    if set(payload) != required_top_level:
        raise ValueError(
            f"source manifest keys must be exactly {sorted(required_top_level)}, "
            f"got {sorted(payload)}"
        )
    if payload["schema_version"] != "1.0.0":
        raise ValueError("unsupported source manifest schema_version")
    if not isinstance(payload["dataset_id"], str) or not payload["dataset_id"]:
        raise ValueError("dataset_id must be a non-empty string")
    revision = payload["revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("revision must be a lowercase hexadecimal 40-character commit ID")

    files = payload["files"]
    if not isinstance(files, list) or len(files) != EXPECTED_ARCHIVE_COUNT:
        raise ValueError(f"source manifest must contain {EXPECTED_ARCHIVE_COUNT} archives")

    seen_paths: set[str] = set()
    observed_bytes = 0
    for source in files:
        if not isinstance(source, dict) or set(source) != {"path", "size", "sha256"}:
            raise ValueError("every source entry must contain exactly path, size, and sha256")
        source_path = source["path"]
        if (
            not isinstance(source_path, str)
            or not source_path.endswith(("_comments.zst", "_submissions.zst"))
            or "/" in source_path
        ):
            raise ValueError(f"invalid source archive path: {source_path!r}")
        if source_path in seen_paths:
            raise ValueError(f"duplicate source archive path: {source_path}")
        seen_paths.add(source_path)

        size = source["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"invalid source size for {source_path}")
        observed_bytes += size

        digest = source["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid SHA-256 for {source_path}")

    if payload["compressed_bytes"] != observed_bytes:
        raise ValueError(
            "compressed_bytes does not equal the sum of source sizes: "
            f"{payload['compressed_bytes']} != {observed_bytes}"
        )
    if observed_bytes != EXPECTED_COMPRESSED_BYTES:
        raise ValueError(
            f"expected {EXPECTED_COMPRESSED_BYTES} compressed bytes, got {observed_bytes}"
        )
    return payload
