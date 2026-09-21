from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance import modal_modernbert_probability_random_candidate_v1 as launcher
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate


def _write_synthetic_frame(path: Path, *, duplicate_thread: bool = False) -> None:
    rows = []
    for index in range(620):
        rows.append(
            {
                "record_id": f"opaque-{index:04d}",
                "thread_id": (
                    "thread-0020"
                    if duplicate_thread and index == 21
                    else f"thread-{index:04d}"
                ),
                "near_duplicate_cluster_id": f"cluster-{index:04d}",
                "subreddit": "China" if index % 2 else "Sino",
                "year": 2022 + index % 2,
                "content_type": "comment" if index % 3 else "submission",
                "retrieval_mode": "direct",
                "target_text": f"synthetic target {index}",
                "parent_context": None,
                "submission_context": f"synthetic submission {index}",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _acquisition_rows() -> list[dict[str, str]]:
    return [
        {
            "opaque_id": f"opaque-{index:04d}",
            "thread_id": f"thread-{index:04d}",
            "near_duplicate_cluster_id": f"cluster-{index:04d}",
        }
        for index in range(20)
    ]


def test_duckdb_sampler_excludes_all_consumed_keys_and_matches_python_tiebreak(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "eligible.parquet"
    _write_synthetic_frame(frame)
    first, quotas = launcher.select_calibration_rows(
        frame_path=frame,
        acquisition_rows=_acquisition_rows(),
        expected_remaining_rows=600,
    )
    second, _ = launcher.select_calibration_rows(
        frame_path=frame,
        acquisition_rows=list(reversed(_acquisition_rows())),
        expected_remaining_rows=600,
    )
    assert first == second
    assert len(first) == candidate.CALIBRATION_ROWS
    assert sum(row["sample_rows"] for row in quotas) == candidate.CALIBRATION_ROWS
    assert {row["opaque_id"] for row in first}.isdisjoint(
        {row["opaque_id"] for row in _acquisition_rows()}
    )
    assert all(row["inclusion_probability"] == 1.0 for row in first)


def test_duckdb_sampler_fails_closed_on_remaining_identity_collision(tmp_path: Path) -> None:
    frame = tmp_path / "eligible.parquet"
    _write_synthetic_frame(frame, duplicate_thread=True)
    with pytest.raises(RuntimeError, match="identity uniqueness drifted"):
        launcher.select_calibration_rows(
            frame_path=frame,
            acquisition_rows=_acquisition_rows(),
            expected_remaining_rows=600,
        )


def test_source_bundle_is_exactly_scoped_to_candidate_preparation() -> None:
    bundle = launcher.build_source_bundle()
    assert set(bundle["files"]) == set(launcher.REQUIRED_SOURCE_FILES)
    assert bundle["source_bundle_id"] == launcher.validate_source_bundle(bundle)[
        "source_bundle_id"
    ]
