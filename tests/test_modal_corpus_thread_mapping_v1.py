from __future__ import annotations

from pathlib import Path

import pytest

from reddit_china_stance import modal_corpus_thread_mapping_v1 as mapping
from reddit_china_stance.modernbert_corpus_source_enrichment_v2 import canonical_sha256


def test_authority_is_content_addressed_and_declares_minimal_projection() -> None:
    authority = mapping._authority(script_sha256="a" * 64)
    body = {key: value for key, value in authority.items() if key != "authority_id"}

    assert authority["authority_id"] == canonical_sha256(body)
    assert authority["mapping_columns"] == ["corpus_position", "thread_id"]
    assert authority["corpus_rows"] == mapping.contract.CORPUS_ROWS


def test_authority_rejects_invalid_script_digest() -> None:
    with pytest.raises(ValueError, match="script SHA-256 is invalid"):
        mapping._authority(script_sha256="too-short")


def test_sql_paths_quotes_apostrophes() -> None:
    assert mapping._sql_paths([Path("/tmp/plain.parquet"), Path("/tmp/a'b.parquet")]) == (
        "['/tmp/plain.parquet','/tmp/a''b.parquet']"
    )
