from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

from reddit_china_stance.modernbert_splits import (
    CHRONOLOGICAL_SPLITS,
    EXPECTED_FOLD_SIZES,
    EXPECTED_ROWS,
    LADDER_SEEDS,
    build_split_manifest,
    file_sha256,
    load_split_manifest,
    validate_split_manifest,
    write_split_manifest,
)


class DeterministicTenFold:
    def __init__(self, *, n_splits: int, shuffle: bool, random_state: int) -> None:
        assert n_splits == 10
        assert shuffle is True
        self.random_state = random_state

    def split(self, x: list[int], _y: list[list[int]]):
        shuffled = list(x)
        random.Random(self.random_state).shuffle(shuffled)
        cursor = 0
        for fold in range(10):
            size = EXPECTED_FOLD_SIZES[self.random_state][fold]
            test = shuffled[cursor : cursor + size]
            cursor += size
            test_set = set(test)
            yield [index for index in x if index not in test_set], test


class GuardFeatureOmittingTenFold:
    def __init__(self, *, n_splits: int, shuffle: bool, random_state: int) -> None:
        assert (n_splits, shuffle, random_state) == (10, True, random_state)
        self.random_state = random_state

    def split(self, x: list[int], _y: list[list[int]]):
        remaining = list(range(25, 10000))
        folds: list[list[int]] = []
        cursor = 0
        for size in EXPECTED_FOLD_SIZES[self.random_state][:9]:
            folds.append(remaining[cursor : cursor + size])
            cursor += size
        final_size = EXPECTED_FOLD_SIZES[self.random_state][9]
        folds.append(list(range(25)) + remaining[cursor : cursor + final_size - 25])
        for test in folds:
            test_set = set(test)
            yield [index for index in x if index not in test_set], test


class UnevenTenFold:
    def __init__(self, *, n_splits: int, shuffle: bool, random_state: int) -> None:
        assert n_splits == 10 and shuffle and random_state in LADDER_SEEDS

    def split(self, x: list[int], _y: list[list[int]]):
        sizes = [1001, 999] + [1000] * 8
        cursor = 0
        for size in sizes:
            test = x[cursor : cursor + size]
            cursor += size
            test_set = set(test)
            yield [index for index in x if index not in test_set], test


@pytest.fixture(scope="module")
def teacher_rows() -> list[dict[str, object]]:
    year_counts: list[tuple[int, int]] = []
    for contract in CHRONOLOGICAL_SPLITS.values():
        years = contract["years"]
        count = contract["expected_rows"]
        if len(years) == 1:
            year_counts.append((years[0], count))
        else:
            base, remainder = divmod(count, len(years))
            year_counts.extend(
                (year, base + (1 if index < remainder else 0))
                for index, year in enumerate(years)
            )
    years = [year for year, count in year_counts for _ in range(count)]
    assert len(years) == EXPECTED_ROWS

    rows: list[dict[str, object]] = []
    targets = ("china_general", "government_ccp", "people_culture", "other")
    stances = ("negative", "mixed", "no_directed_stance", "positive", "unclear")
    for index in range(EXPECTED_ROWS):
        material = index % 3 != 0
        active_target = targets[index % len(targets)]
        row: dict[str, object] = {
            "sample_id": f"S{index:05d}",
            "thread_id": f"t3_{index:05d}",
            "year": years[index],
            "subreddit": f"subreddit-{index % 10}",
            "content_type": "comment" if index % 5 else "submission",
            "retrieval_mode": "rare" if index < 25 else f"mode-{index % 2}",
            "relevance": "material" if material else "not_material",
        }
        for target in targets:
            present = material and target == active_target
            row[f"has_target_{target}"] = present
            row[f"stance_{target}"] = stances[index % len(stances)] if present else None
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def manifest(teacher_rows: list[dict[str, object]]) -> dict[str, object]:
    return build_split_manifest(
        teacher_rows,
        splitter_factory=DeterministicTenFold,
        splitter_version="test-1.0",
    )


def test_builds_three_exact_nested_ladders_and_chronological_split(
    manifest: dict[str, object], teacher_rows: list[dict[str, object]]
) -> None:
    assert validate_split_manifest(manifest, source_rows=teacher_rows) == manifest
    assert set(manifest["ladders"]) == {str(seed) for seed in LADDER_SEEDS}
    for ladder in manifest["ladders"].values():
        seed = ladder["acquisition_seed"]
        sizes = EXPECTED_FOLD_SIZES[seed]
        assert {key: value["row_count"] for key, value in ladder["budgets"].items()} == {
            "2k": sum(sizes[:2]),
            "5k": sum(sizes[:5]),
            "10k": sum(sizes),
        }
    assert {key: value["row_count"] for key, value in manifest["chronological"].items()} == {
        "train": 7665,
        "calibration": 1119,
        "test": 1216,
    }
    assert len({row["thread_id"] for row in manifest["rows"]}) == EXPECTED_ROWS


def test_manifest_is_deterministic_content_addressed_and_exactly_reloadable(
    tmp_path: Path,
    teacher_rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    rebuilt = build_split_manifest(
        list(reversed(teacher_rows)),
        splitter_factory=DeterministicTenFold,
        splitter_version="test-1.0",
    )
    assert rebuilt == manifest
    path = write_split_manifest(tmp_path, manifest)
    assert path.name == f"manifest-{manifest['manifest_id']}.json"
    assert write_split_manifest(tmp_path, rebuilt) == path
    assert load_split_manifest(
        path,
        source_rows=teacher_rows,
        expected_file_sha256=file_sha256(path),
    ) == manifest


def test_rejects_tampering_and_wrong_content_addressed_filename(
    tmp_path: Path, manifest: dict[str, object]
) -> None:
    tampered = deepcopy(manifest)
    tampered["rows"][0]["folds"]["101"] = 9
    with pytest.raises(ValueError, match="manifest_id"):
        validate_split_manifest(tampered)

    wrong_path = tmp_path / "manifest-wrong.json"
    wrong_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        load_split_manifest(wrong_path)


def test_rejects_duplicate_threads(teacher_rows: list[dict[str, object]]) -> None:
    duplicate = deepcopy(teacher_rows)
    duplicate[1]["thread_id"] = duplicate[0]["thread_id"]
    with pytest.raises(ValueError, match="thread_id values must be unique"):
        build_split_manifest(
            duplicate,
            splitter_factory=DeterministicTenFold,
            splitter_version="test-1.0",
        )


def test_rejects_a_guarded_feature_missing_from_2k(
    teacher_rows: list[dict[str, object]],
) -> None:
    with pytest.raises(RuntimeError, match="support >=25"):
        build_split_manifest(
            teacher_rows,
            splitter_factory=GuardFeatureOmittingTenFold,
            splitter_version="test-1.0",
        )


def test_injected_splitter_requires_a_version(teacher_rows: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="splitter_version"):
        build_split_manifest(teacher_rows, splitter_factory=DeterministicTenFold)


def test_rejects_a_fold_vector_other_than_the_frozen_observation(
    teacher_rows: list[dict[str, object]],
) -> None:
    with pytest.raises(RuntimeError, match="drifted from the frozen vector"):
        build_split_manifest(
            teacher_rows,
            splitter_factory=UnevenTenFold,
            splitter_version="test-1.0",
        )
