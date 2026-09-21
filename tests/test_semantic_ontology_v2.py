from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reddit_china_stance.semantic_ontology_v2 import (
    BLINDED_ROW_FIELDS,
    STRATUM_QUOTAS,
    TARGETS,
    build_pilot_packet,
    load_v2_label_schema,
    load_v2_label_validator,
    select_pilot_rows,
    validate_pilot_packet,
    validate_v2_label,
)


def _legacy_label(stratum: str, index: int) -> dict[str, object]:
    stances = ("negative", "positive", "mixed", "no_directed_stance")
    stance = stances[index % len(stances)]
    targets = {
        "people_only": [{"target": "people_culture", "stance": stance}],
        "other_only": [{"target": "other", "stance": stance}],
        "both": [
            {"target": "people_culture", "stance": stance},
            {"target": "other", "stance": stances[(index + 1) % len(stances)]},
        ],
        "core_controls": [
            {
                "target": "china_general" if index % 2 == 0 else "government_ccp",
                "stance": stance,
            }
        ],
    }[stratum]
    return {"relevance": "material", "target_stances": targets}


def _source_row(stratum: str, index: int) -> dict[str, object]:
    label = _legacy_label(stratum, index)
    by_target = {item["target"]: item["stance"] for item in label["target_stances"]}
    return {
        "sample_id": f"sample-{stratum}-{index:04d}",
        "thread_id": f"thread-{stratum}-{index:04d}",
        "target_text": f"synthetic text {stratum} {index}",
        "submission_context": None if index % 2 else f"submission {index}",
        "parent_context": None if index % 3 else f"parent {index}",
        "subreddit": ("news", "worldnews", "China")[index % 3],
        "year": 2020 + index % 6,
        "content_type": "comment" if index % 2 else "submission",
        "retrieval_mode": "lexical" if index % 2 else "semantic",
        "relevance": "material",
        "label_json": json.dumps(label, sort_keys=True, separators=(",", ":")),
        **{
            f"has_target_{target}": target in by_target
            for target in ("china_general", "government_ccp", "people_culture", "other")
        },
        **{
            f"stance_{target}": by_target.get(target)
            for target in ("china_general", "government_ccp", "people_culture", "other")
        },
    }


def _source_rows(populations: dict[str, int]) -> list[dict[str, object]]:
    return [
        _source_row(stratum, index)
        for stratum, count in populations.items()
        for index in range(count)
    ]


def test_v2_schema_has_exact_targets_and_no_model_unclear_class() -> None:
    schema = load_v2_label_schema()
    target_enum = schema["properties"]["targets"]["items"]["properties"]["target"]["enum"]
    stance_schema = schema["properties"]["targets"]["items"]["properties"]["stance"]

    assert tuple(target_enum) == TARGETS
    assert "unclear" not in json.dumps(stance_schema)
    compiled_validator = load_v2_label_validator()
    assert validate_v2_label(
        {
            "codability": "codable",
            "relevance": "material",
            "targets": [
                {"target": "residual_other", "stance": None},
                {"target": "people_identity", "stance": "negative"},
            ],
        },
        validator=compiled_validator,
    ) == {
        "codability": "codable",
        "relevance": "material",
        "targets": [
            {"target": "people_identity", "stance": "negative"},
            {"target": "residual_other", "stance": None},
        ],
    }
    assert validate_v2_label(
        {"codability": "not_codable", "relevance": None, "targets": []}
    ) == {
        "codability": "not_codable",
        "relevance": None,
        "targets": [],
    }
    assert validate_v2_label(
        {"codability": "codable", "relevance": "not_material", "targets": []}
    ) == {
        "codability": "codable",
        "relevance": "not_material",
        "targets": [],
    }


@pytest.mark.parametrize(
    ("label", "match"),
    [
        (
            {
                "codability": "codable",
                "relevance": "material",
                "targets": [{"target": "residual_other", "stance": "negative"}],
            },
            "null",
        ),
        (
            {
                "codability": "codable",
                "relevance": "material",
                "targets": [{"target": "people_identity", "stance": None}],
            },
            "not one of",
        ),
        (
            {
                "codability": "codable",
                "relevance": "material",
                "targets": [
                    {"target": "culture_media", "stance": "positive"},
                    {"target": "culture_media", "stance": "negative"},
                ],
            },
            "duplicate target",
        ),
        (
            {
                "codability": "not_codable",
                "relevance": None,
                "targets": [],
                "unclear": True,
            },
            "Additional",
        ),
        ({"codability": "codable", "relevance": "material", "targets": []}, "non-empty"),
        (
            {
                "codability": "codable",
                "relevance": "not_material",
                "targets": [{"target": "china_general", "stance": "negative"}],
            },
            "empty",
        ),
        (
            {"codability": "not_codable", "relevance": "not_material", "targets": []},
            "null",
        ),
    ],
)
def test_v2_label_validation_fails_closed(label: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_v2_label(label)


def test_selection_is_deterministic_exact_and_probability_bound() -> None:
    quotas = {"people_only": 4, "other_only": 4, "both": 2, "core_controls": 2}
    rows = _source_rows(
        {"people_only": 9, "other_only": 8, "both": 5, "core_controls": 6}
    )

    first = select_pilot_rows(
        rows,
        seed="fixed-test-seed",
        quotas=quotas,
        expected_source_rows=len(rows),
    )
    second = select_pilot_rows(
        list(reversed(rows)),
        seed="fixed-test-seed",
        quotas=quotas,
        expected_source_rows=len(rows),
    )

    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert Counter(row["selection_stratum"] for row in first) == Counter(quotas)
    assert len({row["thread_id"] for row in first}) == sum(quotas.values())
    assert all(
        row["inclusion_probability"]
        == row["inclusion_probability_numerator"]
        / row["inclusion_probability_denominator"]
        for row in first
    )


def test_selection_rejects_thread_leakage_and_insufficient_strata() -> None:
    quotas = {"people_only": 1, "other_only": 1, "both": 1, "core_controls": 1}
    rows = _source_rows(
        {"people_only": 2, "other_only": 2, "both": 2, "core_controls": 2}
    )
    rows[1]["thread_id"] = rows[0]["thread_id"]
    with pytest.raises(ValueError, match="one-row-per-thread"):
        select_pilot_rows(
            rows,
            quotas=quotas,
            expected_source_rows=len(rows),
        )

    with pytest.raises(ValueError, match="cannot satisfy quota"):
        select_pilot_rows(
            _source_rows(
                {"people_only": 1, "other_only": 2, "both": 2, "core_controls": 2}
            ),
            quotas={"people_only": 2, "other_only": 1, "both": 1, "core_controls": 1},
            expected_source_rows=7,
        )


def test_packet_is_blinded_immutable_and_exactly_conserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _source_rows(
        {"people_only": 170, "other_only": 170, "both": 90, "core_controls": 90}
    )
    source_path = tmp_path / "private-10k-subset.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source_path)
    output_root = tmp_path / "packet-root"

    receipt = build_pilot_packet(
        source_parquet_path=source_path,
        output_root=output_root,
        seed="packet-test-seed",
        expected_source_rows=len(rows),
    )
    packet_root = next(output_root.glob("packet=*"))
    blinded = json.loads((packet_root / "blinded-input.json").read_text())
    mapping = pq.read_table(packet_root / "private-mapping.parquet").to_pylist()

    assert receipt == validate_pilot_packet(packet_root)
    assert receipt == build_pilot_packet(
        source_parquet_path=source_path,
        output_root=output_root,
        seed="packet-test-seed",
        expected_source_rows=len(rows),
    )
    assert len(blinded["rows"]) == 480
    assert all(set(row) == BLINDED_ROW_FIELDS for row in blinded["rows"])
    assert all("legacy" not in json.dumps(row) for row in blinded["rows"])
    assert len(mapping) == 480
    assert Counter(row["selection_stratum"] for row in mapping) == Counter(STRATUM_QUOTAS)
    assert receipt["receipt_contains_raw_text"] is False
    assert receipt["receipt_contains_row_level_labels"] is False

    drifted_rubric = tmp_path / "drifted-rubric.md"
    drifted_rubric.write_text("changed after packet freeze")
    monkeypatch.setattr(
        "reddit_china_stance.semantic_ontology_v2.RUBRIC_PATH",
        drifted_rubric,
    )
    with pytest.raises(RuntimeError, match="manifest binding"):
        validate_pilot_packet(packet_root)
