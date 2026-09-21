from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reddit_china_stance.source_manifest import (
    EXPECTED_ARCHIVE_COUNT,
    EXPECTED_COMPRESSED_BYTES,
    load_source_manifest,
)

MANIFEST_PATH = Path("configs/source-files.json")


def test_checked_in_source_manifest_is_complete() -> None:
    manifest = load_source_manifest(MANIFEST_PATH)

    assert len(manifest["files"]) == EXPECTED_ARCHIVE_COUNT
    assert manifest["compressed_bytes"] == EXPECTED_COMPRESSED_BYTES
    assert len({source["path"] for source in manifest["files"]}) == EXPECTED_ARCHIVE_COUNT


def test_source_manifest_rejects_size_drift(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    changed = deepcopy(manifest)
    changed["files"][0]["size"] += 1
    path = tmp_path / "source-files.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="compressed_bytes does not equal"):
        load_source_manifest(path)


def test_source_manifest_rejects_missing_digest(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    changed = deepcopy(manifest)
    changed["files"][0]["sha256"] = ""
    path = tmp_path / "source-files.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid SHA-256"):
        load_source_manifest(path)


@pytest.mark.parametrize(
    "revision",
    [
        "A" * 40,
        "g" * 40,
        "a" * 39,
        "a" * 41,
    ],
    ids=["uppercase", "nonhex", "short", "long"],
)
def test_source_manifest_rejects_invalid_revision(tmp_path: Path, revision: str) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    changed = deepcopy(manifest)
    changed["revision"] = revision
    path = tmp_path / "source-files.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase hexadecimal 40-character commit ID"):
        load_source_manifest(path)
