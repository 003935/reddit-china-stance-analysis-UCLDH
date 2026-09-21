from __future__ import annotations

import pytest

from reddit_china_stance.modernbert_factorised_data import (
    IGNORE_INDEX,
    FactorisedLabelEncoding,
    TextRenderConfig,
    build_factorised_record,
    encode_v2_label,
    render_factorised_text,
)


def test_encoding_uses_codability_and_upstream_masks_without_unclear() -> None:
    not_codable = encode_v2_label(
        {"codability": "not_codable", "relevance": None, "targets": []}
    )
    assert not_codable.codability_mask == 0
    assert not_codable.relevance_label == IGNORE_INDEX
    assert not_codable.relevance_mask == 0
    assert not any(not_codable.target_presence_mask)
    assert not any(not_codable.stance_mask)

    not_material = encode_v2_label(
        {"codability": "codable", "relevance": "not_material", "targets": []}
    )
    assert not_material.codability_mask == 1
    assert not_material.relevance_label == 0
    assert not_material.relevance_mask == 1
    assert not any(not_material.target_presence_labels)
    assert not any(not_material.target_presence_mask)
    assert not any(not_material.stance_mask)

    material = encode_v2_label(
        {
            "codability": "codable",
            "relevance": "material",
            "targets": [
                {"target": "china_general", "stance": "negative"},
                {"target": "residual_other", "stance": None},
            ],
        }
    )
    assert material.relevance_label == 1
    assert material.target_presence_labels == (1, 0, 0, 0, 0, 1)
    assert material.target_presence_mask == (1, 1, 1, 1, 1, 1)
    assert material.stance_mask == (1, 0, 0, 0, 0)
    assert material.stance_b4_labels == (0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX)
    assert material.stance_b2_labels[0] == (1, 0)
    assert material.stance_b2_labels[1:] == ((IGNORE_INDEX, IGNORE_INDEX),) * 4

    with pytest.raises(ValueError, match="not one of"):
        encode_v2_label(
            {
                "codability": "codable",
                "relevance": "unclear",
                "targets": [],
            }
        )


def test_encoding_rejects_shape_dtype_and_logical_drift() -> None:
    with pytest.raises(ValueError, match="binary integers"):
        FactorisedLabelEncoding(
            codability_mask=1,
            relevance_label=1,
            relevance_mask=1,
            target_presence_labels=(True, 0, 0, 0, 0, 0),
            target_presence_mask=(1, 1, 1, 1, 1, 1),
            stance_b4_labels=(0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX),
            stance_b2_labels=((1, 0),) + ((IGNORE_INDEX, IGNORE_INDEX),) * 4,
            stance_mask=(1, 0, 0, 0, 0),
        )

    with pytest.raises(ValueError, match="at least one target"):
        FactorisedLabelEncoding(
            codability_mask=1,
            relevance_label=1,
            relevance_mask=1,
            target_presence_labels=(0, 0, 0, 0, 0, 0),
            target_presence_mask=(1, 1, 1, 1, 1, 1),
            stance_b4_labels=(IGNORE_INDEX,) * 5,
            stance_b2_labels=((IGNORE_INDEX, IGNORE_INDEX),) * 5,
            stance_mask=(0, 0, 0, 0, 0),
        )

    with pytest.raises(ValueError, match="B2 and B4"):
        FactorisedLabelEncoding(
            codability_mask=1,
            relevance_label=1,
            relevance_mask=1,
            target_presence_labels=(1, 0, 0, 0, 0, 0),
            target_presence_mask=(1, 1, 1, 1, 1, 1),
            stance_b4_labels=(0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX),
            stance_b2_labels=((0, 1),) + ((IGNORE_INDEX, IGNORE_INDEX),) * 4,
            stance_mask=(1, 0, 0, 0, 0),
        )


def test_text_rendering_is_deterministic_target_first_and_hard_bounded() -> None:
    row = {
        "target_text": "  target-" + "x" * 50,
        "parent_context": "parent-" + "y" * 50,
        "submission_context": "submission-" + "z" * 50,
    }
    config = TextRenderConfig(
        max_target_chars=16,
        max_parent_chars=12,
        max_submission_chars=12,
        max_total_chars=40,
    )
    first = render_factorised_text(row, separator="[SEP]", config=config)
    second = render_factorised_text(row, separator="[SEP]", config=config)

    assert first == second
    assert first.startswith("target-")
    assert len(first) <= 40
    assert first.count("[SEP]") == 2

    with pytest.raises(ValueError, match="target_text"):
        render_factorised_text(
            {"target_text": " ", "parent_context": None, "submission_context": None},
            separator="[SEP]",
        )
    with pytest.raises(ValueError, match="missing required text field"):
        render_factorised_text({"target_text": "x"}, separator="[SEP]")


def test_typed_record_exposes_collator_friendly_integer_arrays() -> None:
    config = TextRenderConfig()
    record = build_factorised_record(
        item_id="private-synthetic-id",
        row={
            "target_text": "Synthetic target",
            "parent_context": None,
            "submission_context": "Synthetic submission",
        },
        label={
            "codability": "codable",
            "relevance": "material",
            "targets": [{"target": "culture_media", "stance": "mixed"}],
        },
        separator="[SEP]",
        config=config,
    )
    feature = record.as_feature()

    assert feature["text_contract_digest"] == config.digest()
    assert feature["target_presence_labels"] == [0, 0, 0, 1, 0, 0]
    assert feature["stance_b4_labels"] == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        1,
        IGNORE_INDEX,
    ]
    assert feature["stance_b2_labels"][3] == [1, 1]
    assert all(type(value) is int for value in feature["target_presence_labels"])
