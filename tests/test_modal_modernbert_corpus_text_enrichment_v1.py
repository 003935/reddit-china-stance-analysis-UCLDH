from __future__ import annotations

import pytest

from reddit_china_stance import modal_modernbert_corpus_text_enrichment_v1 as modal_job
from reddit_china_stance.semantic_ontology_v2 import canonical_sha256


def test_source_bundle_validator_rejects_path_drift() -> None:
    files = [
        {"path": path, "sha256": "a" * 64, "bytes": 1}
        for path in modal_job.REQUIRED_SOURCE_FILES
    ]
    body = {
        "schema_version": "1.0.0",
        "kind": "modernbert-corpus-text-enrichment-source-bundle-v1",
        "files": files,
    }
    bundle = {**body, "source_bundle_id": canonical_sha256(body)}
    assert modal_job._validate_source_bundle(bundle) == bundle
    bundle["files"] = list(reversed(files))
    with pytest.raises(RuntimeError, match="source bundle binding drifted"):
        modal_job._validate_source_bundle(bundle)


def test_sql_paths_escape_single_quotes() -> None:
    assert modal_job._sql_paths([modal_job.Path("/data/a'b.parquet")]) == (
        "['/data/a''b.parquet']"
    )


def test_staging_tree_rejects_unreceipted_files_and_spill(tmp_path) -> None:
    staging = tmp_path / "staging"
    (staging / "shards/shard=000").mkdir(parents=True)
    (staging / "shards/shard=000/predictions.parquet").write_bytes(b"parquet")
    for name in ("authority.json", "source-bundle.json", "canonical-inventory.json"):
        (staging / name).write_text("{}", encoding="utf-8")
    modal_job._assert_staging_files(staging, shard_ids=["000"], include_receipt=False)
    (staging / "join.duckdb").write_bytes(b"private temp state")
    with pytest.raises(RuntimeError, match="missing or unreceipted"):
        modal_job._assert_staging_files(staging, shard_ids=["000"], include_receipt=False)
    (staging / "join.duckdb").unlink()
    (staging / ".duckdb-spill").mkdir()
    with pytest.raises(RuntimeError, match="retains DuckDB spill state"):
        modal_job._assert_staging_files(staging, shard_ids=["000"], include_receipt=False)
