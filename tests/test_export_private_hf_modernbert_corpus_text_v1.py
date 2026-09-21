from __future__ import annotations

from pathlib import Path

import pytest

from reddit_china_stance import export_private_hf_modernbert_corpus_text_v1 as exporter


def test_dataset_card_explains_canonical_text_and_private_boundary() -> None:
    card = exporter._dataset_card(export_id="e" * 64, receipt={"receipt_id": "r" * 64})
    assert "authorised word-cloud and corpus analysis" in card
    assert "a comment's normalised body" in card
    assert "submission's normalised title and self-text" in card
    assert "Do not publish this dataset or its row-level text" in card
    assert "No authors" in card


def test_text_root_inventory_rejects_unreceipted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exporter.contract, "SHARD_COUNT", 1)
    root = tmp_path / "enrichment"
    relative = "shards/shard=000/predictions.parquet"
    (root / relative).parent.mkdir(parents=True)
    (root / relative).write_bytes(b"parquet")
    for name in (
        "authority.json",
        "source-bundle.json",
        "canonical-inventory.json",
        "receipt.json",
    ):
        (root / name).write_text("{}", encoding="utf-8")
    descriptor = [{"relative_path": relative}]
    exporter._validate_text_root_inventory(root, descriptors=descriptor)
    (root / "join.duckdb").write_bytes(b"unreceipted")
    with pytest.raises(RuntimeError, match="missing or unreceipted"):
        exporter._validate_text_root_inventory(root, descriptors=descriptor)


def test_text_is_allowed_but_row_level_context_columns_are_forbidden() -> None:
    assert "text" not in exporter.FORBIDDEN_COLUMNS
    assert {"author", "body", "target_text", "logits"}.issubset(exporter.FORBIDDEN_COLUMNS)
