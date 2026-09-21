from __future__ import annotations

import subprocess

import pytest

from reddit_china_stance.privacy import (
    assert_metadata_only,
    validate_repository_privacy,
    validate_tracked_paths,
)


def test_public_metadata_rejects_identity_text_and_row_labels() -> None:
    for field in (
        "record_id",
        "source_sample_ids",
        "target_text",
        "author",
        "rows",
        "target_stances",
        "final_label",
    ):
        with pytest.raises(ValueError, match="private field"):
            assert_metadata_only({"summary": {field: "private"}})


def test_public_metadata_accepts_aggregate_receipt() -> None:
    assert_metadata_only(
        {
            "status": "complete",
            "row_count": 10_000,
            "origins": {"model_agreement": 8_000, "sol_review": 2_000},
            "manifest_sha256": "a" * 64,
        }
    )


@pytest.mark.parametrize(
    "path",
    ["outputs/run.json", "data/private.json", "sample.parquet", ".env.local"],
)
def test_tracked_path_validator_rejects_private_surfaces(path: str) -> None:
    with pytest.raises(ValueError):
        validate_tracked_paths(["README.md", path])


def test_tracked_path_validator_accepts_policy_and_code() -> None:
    validate_tracked_paths(["README.md", "data/README.md", "src/example.py", "tests/test_x.py"])


def test_repository_validator_ignores_tracked_file_deleted_from_worktree(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    private_path = tmp_path / "deleted.jsonl"
    private_path.write_text('{"record_id":"private"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "deleted.jsonl"], cwd=tmp_path, check=True)
    private_path.unlink()

    assert validate_repository_privacy(tmp_path) == 0
