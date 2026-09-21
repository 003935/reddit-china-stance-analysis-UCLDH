from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reddit_china_stance.human_seeded_consensus_v1 import (
    ADJUDICATION_BLINDNESS_CONTRACT,
    ADJUDICATION_INPUT_KIND,
    ADJUDICATION_OUTPUT_KIND,
    BLINDNESS_CONTRACT,
    FINAL_KIND,
    REVIEW_FRAGMENT_KIND,
    REVIEW_INPUT_KIND,
    REVIEW_KIND,
    SCHEMA_VERSION,
    SOL_ADJUDICATOR_MODEL,
    _assert_public_metadata_only,
    assemble_adjudication,
    assemble_review,
    file_sha256,
    finalise,
    merge,
    merge_consensus,
    prepare_adjudication_input,
    prepare_review_inputs,
    validate,
    validate_final,
)


def _label(
    relevance: str,
    target: str | None = None,
    stance: str = "no_directed_stance",
) -> dict[str, object]:
    return {
        "relevance": relevance,
        "target_stances": [] if target is None else [{"target": target, "stance": stance}],
    }


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, object]:
    schema_path = Path("schemas/human-reference-semantic-label-v1.schema.json").resolve()
    rubric_path = tmp_path / "rubric.md"
    rubric_path.write_text("frozen rubric\n", encoding="utf-8")
    human_labels = [
        _label("not_material"),
        _label("material", "china_general", "negative"),
        _label("material", "government_ccp", "positive"),
        _label("material", "people_culture", "no_directed_stance"),
        _label("unclear"),
    ]
    reference = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synthetic-reference",
        "rows": [
            {
                "source_sample_id": f"S{index}",
                "split": "development" if index <= 2 else "locked_test_candidate",
                "target_text": f"private text {index}",
                "label": label,
            }
            for index, label in enumerate(human_labels, start=1)
        ]
        + [
            {
                "source_sample_id": "S6",
                "split": "excluded_language",
                "target_text": "private non-English text",
                "label": _label("material", "people_culture"),
            }
        ],
    }
    reference_path = _write_json(tmp_path / "reference.json", reference)

    reviewer_1_labels = [
        human_labels[0],
        human_labels[1],
        _label("material", "people_culture", "positive"),
        _label("material", "china_general", "negative"),
        _label("not_material"),
    ]
    reviewer_2_labels = [
        human_labels[0],
        _label("material", "china_general", "positive"),
        human_labels[2],
        _label("material", "other", "negative"),
        _label("not_material"),
    ]

    def review(reviewer_id: str, labels: list[dict[str, object]]) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": REVIEW_KIND,
            "reviewer_id": reviewer_id,
            "source_packet_sha256": file_sha256(reference_path),
            "rubric_sha256": file_sha256(rubric_path),
            "label_schema_sha256": file_sha256(schema_path),
            "blindness": BLINDNESS_CONTRACT,
            "rows": [
                {"source_sample_id": f"S{index}", "label": label}
                for index, label in enumerate(labels, start=1)
            ],
        }

    reviewer_one_path = _write_json(
        tmp_path / "z-reviewer.json", review("reviewer-z", reviewer_1_labels)
    )
    reviewer_two_path = _write_json(
        tmp_path / "a-reviewer.json", review("reviewer-a", reviewer_2_labels)
    )
    return {
        "schema_path": schema_path,
        "rubric_path": rubric_path,
        "reference_path": reference_path,
        "reviewer_one_path": reviewer_one_path,
        "reviewer_two_path": reviewer_two_path,
        "human_labels": human_labels,
    }


def _review_fragments(
    fixture: dict[str, object], tmp_path: Path
) -> tuple[Path, list[Path], list[dict[str, object]]]:
    review_root = tmp_path / "review-workflow"
    prepared = prepare_review_inputs(
        reference_path=fixture["reference_path"],
        shard_count=2,
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
        review_root=review_root,
    )
    human_by_id = {
        f"S{index}": label
        for index, label in enumerate(fixture["human_labels"], start=1)
    }
    paths: list[Path] = []
    fragments: list[dict[str, object]] = []
    for descriptor in prepared["shards"]:
        input_path = Path(descriptor["path"])
        shard = json.loads(input_path.read_text(encoding="utf-8"))
        fragment = {
            "schema_version": SCHEMA_VERSION,
            "kind": REVIEW_FRAGMENT_KIND,
            "reviewer_id": "blinded-reviewer",
            "source_packet_sha256": shard["source_packet_sha256"],
            "rubric_sha256": shard["rubric_sha256"],
            "label_schema_sha256": shard["label_schema_sha256"],
            "blindness": BLINDNESS_CONTRACT,
            "input_shard_sha256": file_sha256(input_path),
            "shard_index": shard["shard_index"],
            "shard_count": shard["shard_count"],
            "rows": [
                {
                    "source_sample_id": row["source_sample_id"],
                    "label": human_by_id[row["source_sample_id"]],
                }
                for row in shard["rows"]
            ],
        }
        path = _write_json(tmp_path / f"fragment-{shard['shard_index']}.json", fragment)
        paths.append(path)
        fragments.append(fragment)
    return review_root, paths, fragments


def _adjudication_output(
    fixture: dict[str, object], tmp_path: Path
) -> tuple[Path, Path, dict[str, object]]:
    review_root = tmp_path / "adjudication-workflow"
    prepared = prepare_adjudication_input(
        reference_path=fixture["reference_path"],
        reviewer_one_path=fixture["reviewer_one_path"],
        reviewer_two_path=fixture["reviewer_two_path"],
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
        review_root=review_root,
    )
    input_path = Path(prepared["input_path"])
    packet = json.loads(input_path.read_text(encoding="utf-8"))
    labels = fixture["human_labels"]
    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": ADJUDICATION_OUTPUT_KIND,
        "adjudicator_id": "sol-adjudicator",
        "adjudicator_model": SOL_ADJUDICATOR_MODEL,
        "input_sha256": file_sha256(input_path),
        "consensus_id": packet["consensus_id"],
        "source_packet_sha256": packet["source_packet_sha256"],
        "rubric_sha256": packet["rubric_sha256"],
        "label_schema_sha256": packet["label_schema_sha256"],
        "blindness": ADJUDICATION_BLINDNESS_CONTRACT,
        "rows": [
            {"source_sample_id": "S4", "label": labels[3]},
            {"source_sample_id": "S5", "label": labels[4]},
        ],
    }
    output_path = _write_json(tmp_path / "adjudicator-output.json", output)
    return review_root, output_path, output


def test_prepare_review_inputs_exposes_only_the_frozen_blinded_surface(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review_root = tmp_path / "review-inputs"
    result = prepare_review_inputs(
        reference_path=fixture["reference_path"],
        shard_count=2,
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
        review_root=review_root,
    )

    assert result["total_rows"] == 5
    assert [item["rows"] for item in result["shards"]] == [3, 2]
    observed_ids: list[str] = []
    for descriptor in result["shards"]:
        path = Path(descriptor["path"])
        shard = json.loads(path.read_text(encoding="utf-8"))
        assert shard["kind"] == REVIEW_INPUT_KIND
        assert set(shard) == {
            "schema_version",
            "kind",
            "source_packet_sha256",
            "rubric_sha256",
            "label_schema_sha256",
            "blindness",
            "shard_index",
            "shard_count",
            "rows",
        }
        assert shard["blindness"] == BLINDNESS_CONTRACT
        for row in shard["rows"]:
            assert set(row) == {
                "source_sample_id",
                "target_text",
                "submission_context",
                "parent_context",
            }
            assert "human_label" not in row
            assert "label" not in row
            assert "split" not in row
            assert "language" not in row
            assert "source_row" not in row
            assert "exposure_reason" not in row
            observed_ids.append(row["source_sample_id"])
    assert observed_ids == ["S1", "S2", "S3", "S4", "S5"]
    assert "S6" not in observed_ids


def test_assemble_review_restores_exact_order_and_full_merge_contract(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review_root, fragment_paths, _ = _review_fragments(fixture, tmp_path)
    result = assemble_review(
        reference_path=fixture["reference_path"],
        fragment_paths=list(reversed(fragment_paths)),
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
        review_root=review_root,
    )

    review_path = Path(result["review_path"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["kind"] == REVIEW_KIND
    assert [row["source_sample_id"] for row in review["rows"]] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]
    assert [row["label"] for row in review["rows"]] == fixture["human_labels"]
    assert result["review_sha256"] == file_sha256(review_path)


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "order", "digest", "reviewer", "extra_field"]
)
def test_assemble_review_fails_closed_on_fragment_conservation(
    tmp_path: Path, failure: str
) -> None:
    fixture = _fixture(tmp_path)
    review_root, fragment_paths, fragments = _review_fragments(fixture, tmp_path)
    fragment = deepcopy(fragments[1 if failure == "reviewer" else 0])
    if failure == "missing":
        fragment["rows"].pop()
    elif failure == "duplicate":
        fragment["rows"][1]["source_sample_id"] = fragment["rows"][0]["source_sample_id"]
    elif failure == "order":
        fragment["rows"][0], fragment["rows"][1] = fragment["rows"][1], fragment["rows"][0]
    elif failure == "digest":
        fragment["input_shard_sha256"] = "0" * 64
    elif failure == "reviewer":
        fragment["reviewer_id"] = "different-reviewer"
    else:
        fragment["unexpected"] = "leak"
    index = fragment["shard_index"]
    _write_json(fragment_paths[index], fragment)

    with pytest.raises((ValueError, RuntimeError)):
        assemble_review(
            reference_path=fixture["reference_path"],
            fragment_paths=fragment_paths,
            schema_path=fixture["schema_path"],
            rubric_path=fixture["rubric_path"],
            review_root=review_root,
        )


def test_merger_requires_human_supported_majority_and_reports_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    artifact, receipt = merge_consensus(
        reference_path=fixture["reference_path"],
        reviewer_one_path=fixture["reviewer_one_path"],
        reviewer_two_path=fixture["reviewer_two_path"],
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
    )

    rows = artifact["rows"]
    assert [row["status"] for row in rows] == [
        "eligible",
        "eligible",
        "eligible",
        "contentious",
        "contentious",
    ]
    assert [row["reason"] for row in rows] == [
        "human_matches_both_reviewers",
        "human_matches_reviewer_2",
        "human_matches_reviewer_1",
        "no_strict_majority",
        "reviewers_agree_against_human",
    ]
    assert [row["chosen_label"] for row in rows[:3]] == fixture["human_labels"][:3]
    assert rows[3]["chosen_label"] is None
    assert rows[4]["chosen_label"] is None
    assert receipt["eligibility"] == {
        "eligible": 3,
        "contentious": 2,
        "eligible_fraction": 0.6,
        "reason_counts": {
            "human_matches_both_reviewers": 1,
            "human_matches_reviewer_1": 1,
            "human_matches_reviewer_2": 1,
            "no_strict_majority": 1,
            "reviewers_agree_against_human": 1,
        },
        "by_split": {
            "development": {
                "total": 2,
                "eligible": 2,
                "contentious": 0,
                "eligible_fraction": 1.0,
            },
            "locked_test_candidate": {
                "total": 3,
                "eligible": 1,
                "contentious": 2,
                "eligible_fraction": 1 / 3,
            },
        },
    }
    assert receipt["pairwise_agreement"]["human__reviewer_1"]["semantic_exact"]["agree"] == 2
    assert receipt["pairwise_agreement"]["human__reviewer_2"]["semantic_exact"]["agree"] == 2
    assert "rows" not in receipt
    assert "private text" not in json.dumps(receipt)


def test_prepare_adjudication_input_contains_exactly_blinded_contentious_rows(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    review_root = tmp_path / "adjudication-inputs"
    result = prepare_adjudication_input(
        reference_path=fixture["reference_path"],
        reviewer_one_path=fixture["reviewer_one_path"],
        reviewer_two_path=fixture["reviewer_two_path"],
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
        review_root=review_root,
    )

    assert result["total_rows"] == 2
    packet = json.loads(Path(result["input_path"]).read_text(encoding="utf-8"))
    assert packet["kind"] == ADJUDICATION_INPUT_KIND
    assert packet["blindness"] == ADJUDICATION_BLINDNESS_CONTRACT
    assert [row["source_sample_id"] for row in packet["rows"]] == ["S4", "S5"]
    for row in packet["rows"]:
        assert set(row) == {
            "source_sample_id",
            "target_text",
            "submission_context",
            "parent_context",
        }
        assert not {
            "split",
            "status",
            "reason",
            "human_label",
            "reviewer_1_label",
            "reviewer_2_label",
            "chosen_label",
            "label",
        } & set(row)


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "order", "reviewer", "model", "digest", "blindness", "extra"],
)
def test_adjudication_assembly_fails_closed_on_exact_contract(
    tmp_path: Path, failure: str
) -> None:
    fixture = _fixture(tmp_path)
    review_root, output_path, original = _adjudication_output(fixture, tmp_path)
    output = deepcopy(original)
    if failure == "missing":
        output["rows"].pop()
    elif failure == "duplicate":
        output["rows"][1]["source_sample_id"] = "S4"
    elif failure == "order":
        output["rows"].reverse()
    elif failure == "reviewer":
        output["adjudicator_id"] = "reviewer-a"
    elif failure == "model":
        output["adjudicator_model"] = "gpt-5.6-luna"
    elif failure == "digest":
        output["input_sha256"] = "0" * 64
    elif failure == "blindness":
        output["blindness"]["human_labels_hidden"] = False
    else:
        output["unexpected"] = "contract drift"
    _write_json(output_path, output)

    with pytest.raises(ValueError):
        assemble_adjudication(
            reference_path=fixture["reference_path"],
            reviewer_one_path=fixture["reviewer_one_path"],
            reviewer_two_path=fixture["reviewer_two_path"],
            adjudicator_output_path=output_path,
            schema_path=fixture["schema_path"],
            rubric_path=fixture["rubric_path"],
            review_root=review_root,
        )


def test_finalise_freezes_complete_proxy_and_metadata_only_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review_root, output_path, _ = _adjudication_output(fixture, tmp_path)
    assembled = assemble_adjudication(
        reference_path=fixture["reference_path"],
        reviewer_one_path=fixture["reviewer_one_path"],
        reviewer_two_path=fixture["reviewer_two_path"],
        adjudicator_output_path=output_path,
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
        review_root=review_root,
    )
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    kwargs = {
        "reference_path": fixture["reference_path"],
        "reviewer_one_path": fixture["reviewer_one_path"],
        "reviewer_two_path": fixture["reviewer_two_path"],
        "adjudication_path": Path(assembled["adjudication_path"]),
        "schema_path": fixture["schema_path"],
        "rubric_path": fixture["rubric_path"],
        "review_root": review_root,
        "private_root": private_root,
        "public_root": public_root,
        "expected_rows": 5,
    }

    receipt = finalise(**kwargs)
    assert receipt["total_rows"] == 5
    assert receipt["evidence_status"] == (
        "model-assisted/Sol-adjudicated development evidence"
    )
    assert receipt["human_gold"] is False
    assert receipt["independent_human_evaluation"] is False
    assert receipt["final_thesis_claims_require_independent_human_evaluation"] is True
    assert receipt["resolution_counts"] == {
        "blinded_sol_adjudication": 2,
        "strict_human_supported_majority": 3,
    }
    assert validate_final(**kwargs)["status"] == "validated"

    private_path = next((private_root / "final").glob("proxy-*.json"))
    artifact = json.loads(private_path.read_text(encoding="utf-8"))
    assert artifact["kind"] == FINAL_KIND
    assert len(artifact["rows"]) == 5
    assert [row["label"] for row in artifact["rows"]] == fixture["human_labels"]
    public_text = next(public_root.glob("final=*/receipt.json")).read_text(encoding="utf-8")
    assert "private text" not in public_text
    assert "S4" not in public_text

    artifact["rows"][0]["resolution"] = "blinded_sol_adjudication"
    _write_json(private_path, artifact)
    with pytest.raises(RuntimeError, match="private final proxy artefact drifted"):
        validate_final(**kwargs)


def test_finalise_requires_the_full_expected_reference_size(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review_root, output_path, _ = _adjudication_output(fixture, tmp_path)
    assembled = assemble_adjudication(
        reference_path=fixture["reference_path"],
        reviewer_one_path=fixture["reviewer_one_path"],
        reviewer_two_path=fixture["reviewer_two_path"],
        adjudicator_output_path=output_path,
        schema_path=fixture["schema_path"],
        rubric_path=fixture["rubric_path"],
        review_root=review_root,
    )

    with pytest.raises(RuntimeError, match="requires exactly 452 rows"):
        finalise(
            reference_path=fixture["reference_path"],
            reviewer_one_path=fixture["reviewer_one_path"],
            reviewer_two_path=fixture["reviewer_two_path"],
            adjudication_path=Path(assembled["adjudication_path"]),
            schema_path=fixture["schema_path"],
            rubric_path=fixture["rubric_path"],
            review_root=review_root,
            private_root=tmp_path / "private",
            public_root=tmp_path / "public",
        )


@pytest.mark.parametrize(
    "private_value",
    [
        {"row_id": "S1"},
        {"source_sample_ids": ["S1"]},
        {"text": "private text"},
        {"label": _label("not_material")},
        {"labels": [_label("not_material")]},
    ],
)
def test_public_metadata_guard_rejects_row_ids_text_and_labels(
    private_value: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="contains private field"):
        _assert_public_metadata_only({"nested": private_value})


def test_merge_is_immutable_and_validate_recomputes_every_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    kwargs = {
        "reference_path": fixture["reference_path"],
        "reviewer_one_path": fixture["reviewer_one_path"],
        "reviewer_two_path": fixture["reviewer_two_path"],
        "schema_path": fixture["schema_path"],
        "rubric_path": fixture["rubric_path"],
        "private_root": private_root,
        "public_root": public_root,
    }

    merged = merge(**kwargs)
    assert validate(**kwargs)["status"] == "validated"
    private_path = next(private_root.glob("consensus-*.json"))
    private_value = json.loads(private_path.read_text(encoding="utf-8"))
    private_value["rows"][0]["status"] = "contentious"
    _write_json(private_path, private_value)

    with pytest.raises(RuntimeError, match="private consensus artefact drifted"):
        validate(**kwargs)
    assert merged["total_rows"] == 5


@pytest.mark.parametrize("failure", ["missing", "duplicate", "order", "blindness", "digest"])
def test_review_contract_fails_closed_on_conservation_and_blindness(
    tmp_path: Path, failure: str
) -> None:
    fixture = _fixture(tmp_path)
    review_path = fixture["reviewer_one_path"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if failure == "missing":
        review["rows"].pop()
    elif failure == "duplicate":
        review["rows"][1]["source_sample_id"] = review["rows"][0]["source_sample_id"]
    elif failure == "order":
        review["rows"][0], review["rows"][1] = review["rows"][1], review["rows"][0]
    elif failure == "blindness":
        review["blindness"]["human_labels_hidden"] = False
    else:
        review["rubric_sha256"] = "0" * 64
    _write_json(review_path, review)

    with pytest.raises(ValueError):
        merge_consensus(
            reference_path=fixture["reference_path"],
            reviewer_one_path=review_path,
            reviewer_two_path=fixture["reviewer_two_path"],
            schema_path=fixture["schema_path"],
            rubric_path=fixture["rubric_path"],
        )


@pytest.mark.parametrize(
    "bad_label",
    [
        {"relevance": "material", "target_stances": []},
        {
            "relevance": "not_material",
            "target_stances": [{"target": "china_general", "stance": "negative"}],
        },
        {
            "relevance": "material",
            "target_stances": [
                {"target": "china_general", "stance": "negative"},
                {"target": "china_general", "stance": "positive"},
            ],
        },
    ],
)
def test_malformed_semantic_labels_are_rejected(
    tmp_path: Path, bad_label: dict[str, object]
) -> None:
    fixture = _fixture(tmp_path)
    review_path = fixture["reviewer_one_path"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["rows"][0]["label"] = deepcopy(bad_label)
    _write_json(review_path, review)

    with pytest.raises(ValueError, match="invalid semantic label"):
        merge_consensus(
            reference_path=fixture["reference_path"],
            reviewer_one_path=review_path,
            reviewer_two_path=fixture["reviewer_two_path"],
            schema_path=fixture["schema_path"],
            rubric_path=fixture["rubric_path"],
        )
