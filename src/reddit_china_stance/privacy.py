"""Shared privacy checks for public metadata and the tracked repository surface."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PRIVATE_METADATA_KEYS = frozenset(
    {
        "author",
        "author_id",
        "author_name",
        "adjudicator_label",
        "body",
        "chosen_label",
        "context",
        "decision",
        "decisions",
        "human_label",
        "final_label",
        "item_id",
        "label",
        "labels",
        "parent_context",
        "parent_id",
        "prediction",
        "predictions",
        "prompt",
        "proxy_item_id",
        "queue_id",
        "record_id",
        "record_ids",
        "reviewer_1_label",
        "reviewer_2_label",
        "row_id",
        "row_ids",
        "rows",
        "source_id",
        "source_sample_id",
        "source_sample_ids",
        "submission_context",
        "submission_id",
        "target_stances",
        "target_text",
        "text",
        "thread_id",
    }
)

FORBIDDEN_TRACKED_SUFFIXES = frozenset({".jsonl", ".parquet", ".xlsx", ".zst"})


def assert_metadata_only(value: Any, *, where: str = "public") -> None:
    """Reject fields that can expose row identity, text, prompts, or row-level labels."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in PRIVATE_METADATA_KEYS:
                raise ValueError(f"{where} contains private field {key!r}")
            assert_metadata_only(child, where=f"{where}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_metadata_only(child, where=f"{where}[{index}]")


def validate_public_metadata_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("public metadata must be a JSON object")
    assert_metadata_only(value, where=str(path))
    return value


def validate_tracked_paths(paths: Sequence[str]) -> None:
    """Reject tracked data/output artefacts; source code and synthetic tests remain allowed."""

    for raw_path in paths:
        path = PurePosixPath(raw_path)
        if not path.parts:
            continue
        if path.parts[0] == "outputs":
            raise ValueError(f"generated output must not be tracked: {raw_path}")
        if path.parts[0] == "data" and raw_path != "data/README.md":
            raise ValueError(f"research data must not be tracked: {raw_path}")
        if path.suffix.casefold() in FORBIDDEN_TRACKED_SUFFIXES:
            raise ValueError(f"row-level artefact must not be tracked: {raw_path}")
        if path.name == ".env" or path.name.startswith(".env."):
            raise ValueError(f"environment file must not be tracked: {raw_path}")


def validate_repository_privacy(repo_root: Path) -> int:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    paths = [
        value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value
        and (
            (repo_root / value.decode("utf-8")).exists()
            or (repo_root / value.decode("utf-8")).is_symlink()
        )
    ]
    validate_tracked_paths(paths)
    return len(paths)
