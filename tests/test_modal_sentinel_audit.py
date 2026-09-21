from __future__ import annotations

from copy import deepcopy

import pytest

from reddit_china_stance.modal_sentinel_audit import (
    EXPECTED_DATASET_ID,
    EXPECTED_REVISION,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_SUBMISSION_SOURCES,
    MARKERS,
    SOURCE_SCHEMA_VERSION,
    YEARS,
    aggregate_marker_counts,
)


def _contract() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "dataset_id": EXPECTED_DATASET_ID,
        "revision": EXPECTED_REVISION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "years": list(YEARS),
        "markers": list(MARKERS),
        "sources": [
            {"path": path, "size": index + 1, "sha256": f"{index:064x}"}
            for index, path in enumerate(EXPECTED_SUBMISSION_SOURCES, start=1)
        ],
    }


def _source_results() -> list[dict[str, object]]:
    return [
        {
            "source_file": source_file,
            "by_year": {
                str(year): {
                    "[deleted]": source_index + year - YEARS[0],
                    "[removed]": 2,
                }
                for year in YEARS
            },
        }
        for source_index, source_file in enumerate(EXPECTED_SUBMISSION_SOURCES)
    ]


def test_aggregate_marker_counts_returns_only_source_year_and_total_counts() -> None:
    result = aggregate_marker_counts(list(reversed(_source_results())), contract=_contract())

    expected_deleted = sum(
        source_index + year - YEARS[0]
        for source_index in range(len(EXPECTED_SUBMISSION_SOURCES))
        for year in YEARS
    )
    expected_removed = len(EXPECTED_SUBMISSION_SOURCES) * len(YEARS) * 2
    assert result["totals"] == {
        "[deleted]": expected_deleted,
        "[removed]": expected_removed,
        "total": expected_deleted + expected_removed,
    }
    assert list(result["by_source"]) == sorted(EXPECTED_SUBMISSION_SOURCES)
    first = result["by_source"][EXPECTED_SUBMISSION_SOURCES[0]]
    assert first["by_year"]["2020"] == {"[deleted]": 0, "[removed]": 2, "total": 2}
    assert first["totals"] == {"[deleted]": 15, "[removed]": 12, "total": 27}
    assert "record_id" not in repr(result)
    assert "text" not in result


@pytest.mark.parametrize("failure", ["missing_source", "missing_year", "negative_count"])
def test_aggregate_marker_counts_rejects_incomplete_or_invalid_results(failure: str) -> None:
    results = _source_results()
    if failure == "missing_source":
        results.pop()
    elif failure == "missing_year":
        results[0]["by_year"].pop("2025")  # type: ignore[union-attr]
    else:
        results[0]["by_year"]["2025"]["[deleted]"] = -1  # type: ignore[index]

    with pytest.raises(ValueError):
        aggregate_marker_counts(results, contract=_contract())


def test_aggregate_marker_counts_rejects_contract_drift() -> None:
    contract = deepcopy(_contract())
    contract["revision"] = "0" * 40

    with pytest.raises(ValueError, match="expected revision"):
        aggregate_marker_counts(_source_results(), contract=contract)
