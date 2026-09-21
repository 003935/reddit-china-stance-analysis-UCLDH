from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from reddit_china_stance.reconciliation import (
    load_thesis_claims,
    reconcile_receipts,
)
from reddit_china_stance.source_manifest import load_source_manifest

ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = load_source_manifest(ROOT / "configs/source-files.json")
    claims = load_thesis_claims(ROOT / "configs/thesis-counts.toml")
    receipts = []
    for index, source in enumerate(manifest["files"]):
        receipts.append(
            {
                "schema_version": "1.0.0",
                "status": "converted",
                "dataset_id": manifest["dataset_id"],
                "revision": manifest["revision"],
                "source_file": source["path"],
                "source_compressed_bytes": source["size"],
                "source_sha256": source["sha256"],
                "source_schema_version": "1.0.2",
                "language_status": "unclassified",
                "physical_lines": 10,
                "retained_rows": 7,
                "excluded_rows": 3,
                "exclusions": {"outside_time_window": 2, "deleted_marker": 1},
                "year_counts": {"2020": 3, "2021": 4},
                "month_counts": {"2020-01": 3, "2021-02": 4},
                "partitions": [
                    {
                        "year": 2020,
                        "relative_path": "year=2020/part-00000.parquet",
                        "rows": 3,
                        "bytes": 100,
                        "sha256": "a" * 64,
                    },
                    {
                        "year": 2021,
                        "relative_path": "year=2021/part-00000.parquet",
                        "rows": 4,
                        "bytes": 101,
                        "sha256": "b" * 64,
                    },
                ],
                "exclusion_partition": {
                    "relative_path": "_exclusions/part-00000.parquet",
                    "rows": 3,
                    "bytes": 50,
                    "sha256": "c" * 64,
                },
                "code_sha256": ("d" if index < 15 else "e") * 64,
                "git_head": ("1" if index < 15 else "2") * 40,
                "git_dirty": index >= 15,
                "manifest_sha256": "f" * 64,
                "source_schema_sha256": "0" * 64,
                "python_version": "3.12.0",
                "pyarrow_version": "25.0.1",
                "zstandard_version": "0.25.0",
                "completed_at": "2026-08-24T00:00:00+00:00",
                "wall_seconds": 1.5,
            }
        )
    return manifest, claims, receipts


def test_reconcile_aggregates_strict_receipts_and_marks_unexplained_claims() -> None:
    manifest, claims, receipts = _inputs()

    report = reconcile_receipts(
        receipts,
        source_manifest=manifest,
        thesis_claims=claims,
    )

    assert report["status"] == "corrected_legacy_claims"
    assert report["acceptance_gate_passed"] is True
    assert report["totals"]["source_compressed_bytes"] == 169_284_685_552
    assert report["totals"]["physical_lines"] == 200
    assert report["totals"]["retained_rows"] == 140
    assert report["totals"]["excluded_rows"] == 60
    assert report["totals"]["by_content_type"]["comment"]["retained_rows"] == 70
    assert report["totals"]["by_content_type"]["submission"]["retained_rows"] == 70
    assert report["totals"]["by_subreddit"]["AskReddit"]["physical_lines"] == 20
    assert report["totals"]["by_subreddit"]["AskReddit"]["retained_rows"] == 14
    assert report["totals"]["exclusions"] == {
        "deleted_marker": 20,
        "outside_time_window": 40,
    }
    assert report["totals"]["year_counts_retained_only"] == {"2020": 60, "2021": 80}
    assert report["totals"]["month_counts_retained_only"] == {
        "2020-01": 60,
        "2021-02": 80,
    }
    assert report["provenance_values"]["code_sha256"] == ["d" * 64, "e" * 64]
    assert report["provenance_values"]["git_head"] == ["1" * 40, "2" * 40]
    assert report["provenance_values"]["git_dirty"] == [False, True]
    assert {item["status"] for item in report["thesis_claim_comparisons"]} == {
        "unexplained_mismatch"
    }
    assert report["thesis_internal_reconciliation"] == {
        "line_128_filtered": 949_624,
        "line_261_analysis_total": 5_075_317,
        "line_261_minus_line_128": 4_125_693,
        "status": "unexplained_mismatch",
    }
    assert report["resolution"]["status"] == "accepted_correction"
    assert report["resolution"]["replacement_filtered_count"] == "pending_stage_a"
    assert {item["metric"] for item in report["unavailable_reconciliations"]} == {
        "excluded_rows_by_year_or_month",
        "submission_deleted_or_removed_fraction",
    }


def test_reconcile_requires_exactly_the_pinned_source_set() -> None:
    manifest, claims, receipts = _inputs()

    with pytest.raises(ValueError, match="exactly 20 receipts"):
        reconcile_receipts(
            receipts[:-1],
            source_manifest=manifest,
            thesis_claims=claims,
        )

    unexpected = copy.deepcopy(receipts)
    unexpected[-1]["source_file"] = "unexpected_comments.zst"
    with pytest.raises(ValueError, match="receipt source set mismatch"):
        reconcile_receipts(
            unexpected,
            source_manifest=manifest,
            thesis_claims=claims,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(physical_lines=11), "count conservation failed"),
        (lambda receipt: receipt.update(source_compressed_bytes=1), "compressed_bytes mismatch"),
        (lambda receipt: receipt.update(source_sha256="9" * 64), "source_sha256 mismatch"),
        (lambda receipt: receipt.update(revision="8" * 40), "dataset identity mismatch"),
        (lambda receipt: receipt["year_counts"].update({"2021": 5}), "summed year_counts"),
        (lambda receipt: receipt["month_counts"].update({"2021-02": 5}), "summed month_counts"),
        (
            lambda receipt: receipt["exclusions"].update({"outside_time_window": 3}),
            "summed exclusions",
        ),
        (lambda receipt: receipt.update(unexpected=True), "receipt for .* keys must be exactly"),
    ],
)
def test_reconcile_rejects_nonconserving_or_non_strict_receipts(
    mutation: Any, message: str
) -> None:
    manifest, claims, receipts = _inputs()
    mutation(receipts[0])

    with pytest.raises(ValueError, match=message):
        reconcile_receipts(
            receipts,
            source_manifest=manifest,
            thesis_claims=claims,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "/tmp/exclusions.parquet",
        "_exclusions/../exclusions.parquet",
    ],
)
def test_reconcile_rejects_unsafe_exclusion_partition_paths(relative_path: str) -> None:
    manifest, claims, receipts = _inputs()
    receipts[0]["exclusion_partition"]["relative_path"] = relative_path

    with pytest.raises(
        ValueError,
        match="exclusion partition path must be relative and traversal-free",
    ):
        reconcile_receipts(
            receipts,
            source_manifest=manifest,
            thesis_claims=claims,
        )


def test_load_thesis_claims_rejects_changed_registered_count(tmp_path: Path) -> None:
    path = tmp_path / "claims.toml"
    source = (ROOT / "configs/thesis-counts.toml").read_text(encoding="utf-8")
    path.write_text(source.replace("count = 73020520", "count = 73020521"), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match the four registered"):
        load_thesis_claims(path)
