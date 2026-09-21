from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from reddit_china_stance.retrieval import (
    AnchorRelation,
    CandidateRecord,
    RetrievalChannel,
    RetrievalPolicyError,
    SelectionEvidence,
    load_retrieval_policy,
    make_candidate,
    match_direct_lexical,
    merge_candidates,
    policy_digest,
)

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "configs" / "retrieval-policy-v1.toml"
V3_POLICY_PATH = ROOT / "configs" / "retrieval-policy-v3.toml"


@pytest.fixture(scope="module")
def policy():
    return load_retrieval_policy(POLICY_PATH)


def test_v3_policy_is_frozen_and_self_consistent() -> None:
    v3_raw = tomllib.loads(V3_POLICY_PATH.read_text())
    v3 = load_retrieval_policy(V3_POLICY_PATH)

    assert v3.schema_version == "1.0.0"
    assert v3.policy_version == "stage-a-v3"
    assert v3.lexicon_version == "china-entity-alias-v3"
    assert v3.policy_digest == "5b7ced047d90407ed8849bcec8744078ed768e74eb78851624c67c009b684d46"
    assert len(v3.terms) == 114
    assert len(v3_raw["terms"]) == len({term["id"] for term in v3_raw["terms"]})
    assert len(v3_raw["terms"]) == len({term["pattern"] for term in v3_raw["terms"]})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Journey to the West", ("journey_to_the_west",)),
        ("Monkey King", ("monkey_king",)),
        ("Wu Kong", ("wu_kong",)),
        ("Havoc in Heaven", ("havoc_in_heaven",)),
        ("Ne Zha", ("ne_zha",)),
        ("Creation of the Gods", ("creation_of_the_gods",)),
        ("Enter the Dragon", ("enter_the_dragon",)),
        ("龙争虎斗", ("long_zheng_hu_dou_simplified",)),
        ("龍爭虎鬥", ("long_zheng_hu_dou_traditional",)),
        ("Shang Chi", ("shang_chi",)),
        ("Shang-Chi", ("shang_chi_hyphenated",)),
        ("Fuzhounese", ("fuzhounese",)),
        ("Longjia", ("longjia",)),
        ("Sinitic", ("sinitic",)),
        ("邓紫棋", ("deng_ziqi",)),
        ("王彩樺", ("wang_caihua",)),
        ("Xiaohongshu", ("xiaohongshu",)),
        ("Zhou Dynasty", ("zhou_dynasty",)),
        ("Upper Xiajiadian Culture", ("upper_xiajiadian_culture", "xiajiadian")),
        ("Xiajiadian", ("xiajiadian",)),
        ("West Liao River", ("west_liao_river",)),
        ("Hangzhou", ("hangzhou",)),
        ("Gubeikou", ("gubeikou",)),
        ("Simatai", ("simatai",)),
        ("Great Wall", ("great_wall",)),
        ("Huanggutun Incident", ("huanggutun", "huanggutun_incident")),
        ("Huanggutun", ("huanggutun",)),
        ("Zhang Zuolin", ("zhang_zuolin",)),
        ("Nanjing Massacre", ("nanjing_massacre",)),
        (
            "Tiananmen Square Massacre",
            ("tiananmen_square", "tiananmen_square_massacre"),
        ),
        ("Tiananmen Square", ("tiananmen_square",)),
        ("People's Armed Police", ("peoples_armed_police",)),
        ("People\u2019s Armed Police", ("peoples_armed_police_curly",)),
        (
            "Belt and Road Initiative",
            ("belt_and_road", "belt_and_road_initiative"),
        ),
        ("Belt and Road", ("belt_and_road",)),
    ],
)
def test_v3_matches_approved_bounded_aliases(text: str, expected: tuple[str, ...]) -> None:
    v3 = load_retrieval_policy(V3_POLICY_PATH)
    assert match_direct_lexical(text, v3) == tuple(sorted(expected))


@pytest.mark.parametrize(
    "text",
    [
        "PAP",
        "CPC",
        "HK",
        "Xi",
        "BRI",
        "Marxism",
        "Three Kingdoms",
        "的 人 中 了",
        "https://example.test/PAP",
    ],
)
def test_v3_rejects_ambiguous_shortcuts_and_unbounded_forms(text: str) -> None:
    v3 = load_retrieval_policy(V3_POLICY_PATH)
    assert match_direct_lexical(text, v3) == ()


@pytest.mark.parametrize("policy_path", [V3_POLICY_PATH], ids=["v3"])
def test_optimised_matcher_is_exactly_equivalent_to_individual_term_scans(
    policy_path: Path,
) -> None:
    policy = load_retrieval_policy(policy_path)
    boundary = r"A-Za-z0-9_"

    def compile_bounded(phrase: str) -> re.Pattern[str]:
        expression = r"\s+".join(re.escape(token) for token in phrase.split())
        return re.compile(rf"(?<![{boundary}]){expression}(?![{boundary}])", re.IGNORECASE)

    def naive(text: str) -> tuple[str, ...]:
        if any(
            text.strip().casefold() == value.casefold()
            for rule in policy.record_exclusion_rules
            for value in rule.values
        ):
            return ()
        suppressed: dict[str, list[tuple[int, int]]] = {}
        for rule in policy.ambiguity_rules:
            spans = [
                match.span()
                for phrase in rule.phrases
                for match in compile_bounded(phrase).finditer(text)
            ]
            for term_id in rule.term_ids:
                suppressed.setdefault(term_id, []).extend(spans)
        found = []
        for term in policy.terms:
            if any(
                not any(
                    match.start() < blocked[1] and blocked[0] < match.end()
                    for blocked in suppressed.get(term.term_id, ())
                )
                for match in compile_bounded(term.pattern).finditer(text)
            ):
                found.append(term.term_id)
        return tuple(sorted(found))

    corpus = [term.pattern for term in policy.terms] + [
        "mainland China and China CDC",
        "Chinese Communist Party and ChineseLanguage",
        "Wuhan University in Wuhan",
        "中国人与中国",
        "bone china beside fine china and China",
        "Huawei, WeChat, Tencent, TikTok, Alibaba and ByteDance",
        "hanz\u0131",
        "p\u0130nyin",
        "[deleted]",
        "unrelated CCTV social credit 996",
    ]
    assert all(match_direct_lexical(text, policy) == naive(text) for text in corpus)


def evidence(
    channel: RetrievalChannel,
    anchor_record_id: str,
    term_ids: list[str],
) -> SelectionEvidence:
    relation, depth = {
        RetrievalChannel.DIRECT_LEXICAL: (AnchorRelation.SELF, 0),
        RetrievalChannel.SUBMISSION_EXPANSION: (AnchorRelation.SUBMISSION, 1),
        RetrievalChannel.DIRECT_REPLY_EXPANSION: (AnchorRelation.DIRECT_PARENT, 1),
    }[channel]
    return SelectionEvidence(
        channel=channel,
        anchor_record_id=anchor_record_id,
        anchor_relation=relation,
        expansion_depth=depth,
        match_term_ids=sorted(term_ids),
    )


def test_checked_in_policy_loads_and_verifies_digest(policy) -> None:
    assert policy.schema_version == "1.0.0"
    assert policy.policy_version == "stage-a-v1"
    assert policy.apply_same_policy_to_every_subreddit is True
    assert policy.max_reply_expansion_depth == 1
    assert len(policy.terms) >= 20


def test_policy_digest_is_content_addressed() -> None:
    raw = tomllib.loads(POLICY_PATH.read_text())
    assert policy_digest(raw) == raw["policy_digest"]
    raw["terms"][0]["rationale"] += " Changed."
    assert policy_digest(raw) != raw["policy_digest"]


def test_policy_loader_rejects_unknown_top_level_setting(tmp_path: Path) -> None:
    path = tmp_path / "unknown.toml"
    path.write_text(
        POLICY_PATH.read_text().replace(
            'lexicon_version = "china-entity-alias-v1"',
            'lexicon_version = "china-entity-alias-v1"\nunknown_setting = true',
        )
    )
    with pytest.raises(RetrievalPolicyError, match=r"unknown=.*unknown_setting"):
        load_retrieval_policy(path)


def test_policy_loader_rejects_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.toml"
    path.write_text(POLICY_PATH.read_text().replace('pattern = "Beijing"', 'pattern = "Peking"'))
    with pytest.raises(RetrievalPolicyError, match="policy digest mismatch"):
        load_retrieval_policy(path)


def test_policy_loader_rejects_malformed_expansion_even_with_valid_digest(
    tmp_path: Path,
) -> None:
    text = POLICY_PATH.read_text().replace(
        "max_reply_expansion_depth = 1", "max_reply_expansion_depth = 2"
    )
    raw = tomllib.loads(text)
    digest = policy_digest(raw)
    text = re.sub(r'policy_digest = "[0-9a-f]{64}"', f'policy_digest = "{digest}"', text)
    path = tmp_path / "recursive.toml"
    path.write_text(text)
    with pytest.raises(RetrievalPolicyError, match="must equal 1"):
        load_retrieval_policy(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("China and the CCP", ("ccp", "china")),
        ("CHINESE policy", ("chinese",)),
        ("Xi Jinping spoke in Beijing", ("beijing", "xi_jinping")),
        ("Chinatown and china123", ()),
        ("[deleted]", ()),
        ("fine china and bone china", ()),
        ("fine china, then China policy", ("china",)),
        ("People's   Republic of China", ("china", "peoples_republic_of_china")),
    ],
)
def test_direct_matcher_has_explicit_boundaries_and_ambiguity_rules(
    policy, text: str, expected: tuple[str, ...]
) -> None:
    assert match_direct_lexical(text, policy) == expected


def test_direct_matcher_rejects_non_string(policy) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        match_direct_lexical(None, policy)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("channel", "relation", "depth"),
    [
        (RetrievalChannel.DIRECT_LEXICAL, AnchorRelation.SUBMISSION, 0),
        (RetrievalChannel.SUBMISSION_EXPANSION, AnchorRelation.SUBMISSION, 0),
        (RetrievalChannel.DIRECT_REPLY_EXPANSION, AnchorRelation.DIRECT_PARENT, 0),
    ],
)
def test_selection_evidence_rejects_invalid_channel_semantics(
    channel: RetrievalChannel, relation: AnchorRelation, depth: int
) -> None:
    with pytest.raises(ValidationError, match="requires relation/depth"):
        SelectionEvidence(
            channel=channel,
            anchor_record_id="t3_anchor",
            anchor_relation=relation,
            expansion_depth=depth,
            match_term_ids=["china"],
        )


def test_merge_preserves_distinct_paths_and_is_deterministic(policy) -> None:
    record_id = "t1_candidate"
    manifest_id = "retrieval-proof-001"
    via_submission = make_candidate(
        record_id=record_id,
        content_type="comment",
        evidence=evidence(RetrievalChannel.SUBMISSION_EXPANSION, "t3_submission", ["china"]),
        policy=policy,
        manifest_id=manifest_id,
    )
    via_parent = make_candidate(
        record_id=record_id,
        content_type="comment",
        evidence=evidence(
            RetrievalChannel.DIRECT_REPLY_EXPANSION, "t1_parent", ["beijing", "china"]
        ),
        policy=policy,
        manifest_id=manifest_id,
    )
    via_other_parent = make_candidate(
        record_id=record_id,
        content_type="comment",
        evidence=evidence(RetrievalChannel.DIRECT_REPLY_EXPANSION, "t1_other_parent", ["ccp"]),
        policy=policy,
        manifest_id=manifest_id,
    )

    merged = merge_candidates([via_other_parent, via_submission, via_parent, via_submission])

    assert merged.retrieval_channels == [
        RetrievalChannel.SUBMISSION_EXPANSION,
        RetrievalChannel.DIRECT_REPLY_EXPANSION,
    ]
    assert [(item.channel, item.anchor_record_id) for item in merged.selection_evidence] == [
        (RetrievalChannel.SUBMISSION_EXPANSION, "t3_submission"),
        (RetrievalChannel.DIRECT_REPLY_EXPANSION, "t1_other_parent"),
        (RetrievalChannel.DIRECT_REPLY_EXPANSION, "t1_parent"),
    ]


def test_merge_rejects_different_manifests(policy) -> None:
    direct = evidence(RetrievalChannel.DIRECT_LEXICAL, "t1_candidate", ["china"])
    first = make_candidate(
        record_id="t1_candidate",
        content_type="comment",
        evidence=direct,
        policy=policy,
        manifest_id="run-a",
    )
    second = make_candidate(
        record_id="t1_candidate",
        content_type="comment",
        evidence=direct,
        policy=policy,
        manifest_id="run-b",
    )
    with pytest.raises(ValueError, match="different identity or run metadata"):
        merge_candidates([first, second])


def test_direct_candidate_must_anchor_to_itself(policy) -> None:
    with pytest.raises(ValidationError, match="anchor to the candidate record"):
        make_candidate(
            record_id="t1_candidate",
            content_type="comment",
            evidence=evidence(RetrievalChannel.DIRECT_LEXICAL, "t1_other", ["china"]),
            policy=policy,
            manifest_id="run-a",
        )


def test_candidate_rejects_term_id_outside_frozen_policy(policy) -> None:
    with pytest.raises(ValueError, match="unknown policy term IDs"):
        make_candidate(
            record_id="t1_candidate",
            content_type="comment",
            evidence=evidence(RetrievalChannel.DIRECT_LEXICAL, "t1_candidate", ["not_in_policy"]),
            policy=policy,
            manifest_id="run-a",
        )


def test_candidate_schema_accepts_model_output(policy) -> None:
    row = make_candidate(
        record_id="t1_candidate",
        content_type="comment",
        evidence=evidence(RetrievalChannel.DIRECT_LEXICAL, "t1_candidate", ["china"]),
        policy=policy,
        manifest_id="run-a",
    )
    schema = json.loads((ROOT / "schemas" / "candidate-record.schema.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(row.model_dump(mode="json")))
    assert errors == []


def test_candidate_schema_rejects_unknown_field(policy) -> None:
    row = make_candidate(
        record_id="t1_candidate",
        content_type="comment",
        evidence=evidence(RetrievalChannel.DIRECT_LEXICAL, "t1_candidate", ["china"]),
        policy=policy,
        manifest_id="run-a",
    ).model_dump(mode="json")
    row["raw_text"] = "must never be part of this contract"
    schema = json.loads((ROOT / "schemas" / "candidate-record.schema.json").read_text())
    assert list(Draft202012Validator(schema).iter_errors(row))


def test_candidate_schema_rejects_channel_evidence_mismatch(policy) -> None:
    row = make_candidate(
        record_id="t1_candidate",
        content_type="comment",
        evidence=evidence(RetrievalChannel.DIRECT_LEXICAL, "t1_candidate", ["china"]),
        policy=policy,
        manifest_id="run-a",
    ).model_dump(mode="json")
    row["retrieval_channels"] = ["submission_expansion"]
    schema = json.loads((ROOT / "schemas" / "candidate-record.schema.json").read_text())
    assert list(Draft202012Validator(schema).iter_errors(row))


def test_candidate_rejects_classified_language_without_model_manifest(policy) -> None:
    row = make_candidate(
        record_id="t1_candidate",
        content_type="comment",
        evidence=evidence(RetrievalChannel.DIRECT_LEXICAL, "t1_candidate", ["china"]),
        policy=policy,
        manifest_id="run-a",
    ).model_dump(mode="json")
    row["language_decision"] = {
        "status": "accepted",
        "language_code": "en",
        "model_run_manifest_id": None,
    }
    with pytest.raises(ValidationError, match="require code and model run manifest"):
        CandidateRecord.model_validate(row)


def test_submission_candidate_rejects_expansion_evidence(policy) -> None:
    with pytest.raises(ValidationError, match="only carry direct lexical evidence"):
        make_candidate(
            record_id="t3_submission",
            content_type="submission",
            evidence=evidence(RetrievalChannel.SUBMISSION_EXPANSION, "t3_anchor", ["china"]),
            policy=policy,
            manifest_id="run-a",
        )
