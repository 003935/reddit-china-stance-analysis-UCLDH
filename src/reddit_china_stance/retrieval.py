"""Frozen Stage A retrieval policy and candidate provenance contracts."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal

import ahocorasick
from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
POLICY_SCHEMA_VERSION = "1.0.0"
CANDIDATE_SCHEMA_VERSION = "1.0.0"
_BOUNDARY_CHARS = r"A-Za-z0-9_"
_REGEX_IGNORECASE_TRANSLATION = str.maketrans({"\u0130": "i", "\u0131": "i"})


class RetrievalPolicyError(ValueError):
    """Raised when a retrieval policy is malformed or fails digest verification."""


class RetrievalChannel(StrEnum):
    DIRECT_LEXICAL = "direct_lexical"
    SUBMISSION_EXPANSION = "submission_expansion"
    DIRECT_REPLY_EXPANSION = "direct_reply_expansion"


class AnchorRelation(StrEnum):
    SELF = "self"
    SUBMISSION = "submission"
    DIRECT_PARENT = "direct_parent"


class LanguageStatus(StrEnum):
    UNCLASSIFIED = "unclassified"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class LexiconTerm:
    """One bounded, case-insensitive entity or alias expression."""

    term_id: str
    pattern: str
    category: str
    rationale: str


@dataclass(frozen=True)
class AmbiguityRule:
    """A bounded phrase whose overlapping term match must be suppressed."""

    rule_id: str
    term_ids: tuple[str, ...]
    phrases: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class RecordExclusionRule:
    """An exact, case-insensitive whole-record exclusion."""

    rule_id: str
    values: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class RetrievalPolicy:
    """Validated immutable policy used identically for every subreddit."""

    schema_version: str
    policy_version: str
    policy_digest: str
    lexicon_version: str
    apply_same_policy_to_every_subreddit: bool
    max_reply_expansion_depth: int
    terms: tuple[LexiconTerm, ...]
    ambiguity_rules: tuple[AmbiguityRule, ...]
    record_exclusion_rules: tuple[RecordExclusionRule, ...]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectionEvidence(_StrictModel):
    """One independently auditable path by which a record entered Stage A."""

    channel: RetrievalChannel
    anchor_record_id: Annotated[str, Field(min_length=1)]
    anchor_relation: AnchorRelation
    expansion_depth: Annotated[int, Field(ge=0, le=1)]
    match_term_ids: Annotated[list[str], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_channel_semantics(self) -> SelectionEvidence:
        expected = {
            RetrievalChannel.DIRECT_LEXICAL: (AnchorRelation.SELF, 0),
            RetrievalChannel.SUBMISSION_EXPANSION: (AnchorRelation.SUBMISSION, 1),
            RetrievalChannel.DIRECT_REPLY_EXPANSION: (AnchorRelation.DIRECT_PARENT, 1),
        }[self.channel]
        if (self.anchor_relation, self.expansion_depth) != expected:
            raise ValueError(f"{self.channel.value} requires relation/depth {expected}")
        if len(self.match_term_ids) != len(set(self.match_term_ids)):
            raise ValueError("match_term_ids must be unique")
        if self.match_term_ids != sorted(self.match_term_ids):
            raise ValueError("match_term_ids must use deterministic lexical order")
        return self


class LanguageDecision(_StrictModel):
    """Candidate-scoped language result; initially and explicitly unclassified."""

    status: LanguageStatus
    language_code: str | None = None
    model_run_manifest_id: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> LanguageDecision:
        if self.status is LanguageStatus.UNCLASSIFIED:
            if self.language_code is not None or self.model_run_manifest_id is not None:
                raise ValueError(
                    "unclassified language decisions cannot cite a result or model run"
                )
        elif not self.language_code or not self.model_run_manifest_id:
            raise ValueError("classified language decisions require code and model run manifest")
        return self


class CandidateRecord(_StrictModel):
    """The unique Stage A row for a record, retaining every selection path."""

    schema_version: Literal["1.0.0"] = CANDIDATE_SCHEMA_VERSION
    record_id: Annotated[str, Field(min_length=1)]
    content_type: Literal["submission", "comment"]
    retrieval_channels: Annotated[list[RetrievalChannel], Field(min_length=1)]
    selection_evidence: Annotated[list[SelectionEvidence], Field(min_length=1)]
    retrieval_policy_version: Annotated[str, Field(min_length=1)]
    retrieval_policy_digest: Annotated[str, Field(pattern=SHA256_PATTERN)]
    manifest_id: Annotated[str, Field(min_length=1)]
    language_decision: LanguageDecision = Field(
        default_factory=lambda: LanguageDecision(status=LanguageStatus.UNCLASSIFIED)
    )

    @model_validator(mode="after")
    def validate_candidate(self) -> CandidateRecord:
        expected_channels = sorted(
            {item.channel for item in self.selection_evidence}, key=_channel_sort_key
        )
        if self.retrieval_channels != expected_channels:
            raise ValueError("retrieval_channels must exactly match selection_evidence")
        evidence_keys = [_evidence_sort_key(item) for item in self.selection_evidence]
        if evidence_keys != sorted(set(evidence_keys)):
            raise ValueError("selection_evidence must be unique and deterministically sorted")
        for item in self.selection_evidence:
            if (
                item.channel is RetrievalChannel.DIRECT_LEXICAL
                and item.anchor_record_id != self.record_id
            ):
                raise ValueError("direct lexical evidence must anchor to the candidate record")
            if (
                self.content_type == "submission"
                and item.channel is not RetrievalChannel.DIRECT_LEXICAL
            ):
                raise ValueError("submission records can only carry direct lexical evidence")
        return self


def policy_digest(data: Mapping[str, Any]) -> str:
    """Hash a parsed policy after removing its self-declared digest."""

    canonical = dict(data)
    canonical.pop("policy_digest", None)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_retrieval_policy(path: Path) -> RetrievalPolicy:
    """Load a policy with strict keys, values, version, and digest verification."""

    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RetrievalPolicyError(f"cannot load retrieval policy {path}: {exc}") from exc

    _require_keys(
        raw,
        {
            "schema_version",
            "policy_version",
            "policy_digest",
            "lexicon_version",
            "scope",
            "expansion",
            "terms",
            "ambiguity_rules",
            "record_exclusion_rules",
        },
        "policy",
    )
    if raw["schema_version"] != POLICY_SCHEMA_VERSION:
        raise RetrievalPolicyError(f"unsupported policy schema_version: {raw['schema_version']!r}")
    declared_digest = _required_sha256(raw["policy_digest"], "policy.policy_digest")
    computed_digest = policy_digest(raw)
    if declared_digest != computed_digest:
        raise RetrievalPolicyError(
            f"policy digest mismatch: declared {declared_digest}, computed {computed_digest}"
        )
    policy_version = _required_string(raw["policy_version"], "policy.policy_version")
    lexicon_version = _required_string(raw["lexicon_version"], "policy.lexicon_version")

    scope = _required_mapping(raw["scope"], "policy.scope")
    _require_keys(scope, {"apply_same_policy_to_every_subreddit"}, "policy.scope")
    if scope["apply_same_policy_to_every_subreddit"] is not True:
        raise RetrievalPolicyError("policy must apply identically to every subreddit")

    expansion = _required_mapping(raw["expansion"], "policy.expansion")
    _require_keys(
        expansion,
        {
            "include_comments_in_direct_hit_submission",
            "include_direct_replies_to_direct_hit_comments",
            "max_reply_expansion_depth",
            "recursive_reply_expansion",
        },
        "policy.expansion",
    )
    if expansion["include_comments_in_direct_hit_submission"] is not True:
        raise RetrievalPolicyError("submission expansion must be enabled")
    if expansion["include_direct_replies_to_direct_hit_comments"] is not True:
        raise RetrievalPolicyError("direct-reply expansion must be enabled")
    if type(expansion["max_reply_expansion_depth"]) is not int:
        raise RetrievalPolicyError("max_reply_expansion_depth must be an integer")
    if expansion["max_reply_expansion_depth"] != 1:
        raise RetrievalPolicyError("max_reply_expansion_depth must equal 1")
    if expansion["recursive_reply_expansion"] is not False:
        raise RetrievalPolicyError("recursive reply expansion must be disabled")

    terms = _load_terms(raw["terms"])
    ambiguity_rules = _load_ambiguity_rules(raw["ambiguity_rules"], terms)
    exclusion_rules = _load_record_exclusion_rules(raw["record_exclusion_rules"])
    return RetrievalPolicy(
        schema_version=POLICY_SCHEMA_VERSION,
        policy_version=policy_version,
        policy_digest=declared_digest,
        lexicon_version=lexicon_version,
        apply_same_policy_to_every_subreddit=True,
        max_reply_expansion_depth=1,
        terms=terms,
        ambiguity_rules=ambiguity_rules,
        record_exclusion_rules=exclusion_rules,
    )


def match_direct_lexical(text: str, policy: RetrievalPolicy) -> tuple[str, ...]:
    """Return deterministic term IDs for bounded matches not suppressed by ambiguity rules."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if _record_is_excluded(text, policy.record_exclusion_rules):
        return ()

    normalised_text = _normalise_lexical_text(text)
    suppressed: dict[str, list[tuple[int, int]]] = {}
    for rule in policy.ambiguity_rules:
        spans = [
            match.span()
            for phrase in rule.phrases
            for match in _compile_bounded(phrase).finditer(normalised_text)
        ]
        for term_id in rule.term_ids:
            suppressed.setdefault(term_id, []).extend(spans)

    matched: set[str] = set()
    automaton = _compile_automaton(tuple((term.term_id, term.pattern) for term in policy.terms))
    for end_index, (pattern_length, term_ids) in automaton.iter(normalised_text):
        start_index = end_index - pattern_length + 1
        if start_index > 0 and _is_boundary_char(normalised_text[start_index - 1]):
            continue
        if end_index + 1 < len(normalised_text) and _is_boundary_char(
            normalised_text[end_index + 1]
        ):
            continue
        span = (start_index, end_index + 1)
        for term_id in term_ids:
            blocked_spans = suppressed.get(term_id, ())
            if not any(_overlaps(span, blocked) for blocked in blocked_spans):
                matched.add(term_id)
    return tuple(sorted(matched))


def make_candidate(
    *,
    record_id: str,
    content_type: Literal["submission", "comment"],
    evidence: SelectionEvidence,
    policy: RetrievalPolicy,
    manifest_id: str,
) -> CandidateRecord:
    """Construct a single-evidence candidate under a verified policy."""

    known_term_ids = {term.term_id for term in policy.terms}
    unknown_term_ids = set(evidence.match_term_ids) - known_term_ids
    if unknown_term_ids:
        raise ValueError(f"evidence contains unknown policy term IDs: {sorted(unknown_term_ids)}")
    return CandidateRecord(
        record_id=record_id,
        content_type=content_type,
        retrieval_channels=[evidence.channel],
        selection_evidence=[evidence],
        retrieval_policy_version=policy.policy_version,
        retrieval_policy_digest=policy.policy_digest,
        manifest_id=manifest_id,
    )


def merge_candidates(candidates: Iterable[CandidateRecord]) -> CandidateRecord:
    """Merge duplicate candidate rows deterministically without discarding selection paths."""

    rows = list(candidates)
    if not rows:
        raise ValueError("at least one candidate is required")
    first = rows[0]
    identity = (
        first.record_id,
        first.content_type,
        first.retrieval_policy_version,
        first.retrieval_policy_digest,
        first.manifest_id,
        first.language_decision,
    )
    for row in rows[1:]:
        other = (
            row.record_id,
            row.content_type,
            row.retrieval_policy_version,
            row.retrieval_policy_digest,
            row.manifest_id,
            row.language_decision,
        )
        if other != identity:
            raise ValueError("cannot merge candidates with different identity or run metadata")

    by_key = {
        _evidence_sort_key(evidence): evidence
        for row in rows
        for evidence in row.selection_evidence
    }
    evidence = [by_key[key] for key in sorted(by_key)]
    channels = sorted({item.channel for item in evidence}, key=_channel_sort_key)
    return CandidateRecord(
        record_id=first.record_id,
        content_type=first.content_type,
        retrieval_channels=channels,
        selection_evidence=evidence,
        retrieval_policy_version=first.retrieval_policy_version,
        retrieval_policy_digest=first.retrieval_policy_digest,
        manifest_id=first.manifest_id,
        language_decision=first.language_decision,
    )


def _load_terms(value: Any) -> tuple[LexiconTerm, ...]:
    rows = _required_list(value, "policy.terms")
    if not rows:
        raise RetrievalPolicyError("policy.terms cannot be empty")
    terms: list[LexiconTerm] = []
    for index, value in enumerate(rows):
        path = f"policy.terms[{index}]"
        row = _required_mapping(value, path)
        _require_keys(row, {"id", "pattern", "category", "rationale"}, path)
        terms.append(
            LexiconTerm(
                term_id=_required_identifier(row["id"], f"{path}.id"),
                pattern=_required_string(row["pattern"], f"{path}.pattern"),
                category=_required_identifier(row["category"], f"{path}.category"),
                rationale=_required_string(row["rationale"], f"{path}.rationale"),
            )
        )
    ids = [term.term_id for term in terms]
    if len(ids) != len(set(ids)):
        raise RetrievalPolicyError("term IDs must be unique")
    for term in terms:
        _compile_bounded(term.pattern)
    return tuple(sorted(terms, key=lambda term: term.term_id))


def _load_ambiguity_rules(value: Any, terms: tuple[LexiconTerm, ...]) -> tuple[AmbiguityRule, ...]:
    rows = _required_list(value, "policy.ambiguity_rules")
    term_ids = {term.term_id for term in terms}
    rules: list[AmbiguityRule] = []
    for index, value in enumerate(rows):
        path = f"policy.ambiguity_rules[{index}]"
        row = _required_mapping(value, path)
        _require_keys(row, {"id", "term_ids", "phrases", "rationale"}, path)
        affected = tuple(sorted(_required_string_list(row["term_ids"], f"{path}.term_ids")))
        unknown = set(affected) - term_ids
        if unknown:
            raise RetrievalPolicyError(f"{path}.term_ids contains unknown IDs: {sorted(unknown)}")
        phrases = tuple(_required_string_list(row["phrases"], f"{path}.phrases"))
        rules.append(
            AmbiguityRule(
                rule_id=_required_identifier(row["id"], f"{path}.id"),
                term_ids=affected,
                phrases=phrases,
                rationale=_required_string(row["rationale"], f"{path}.rationale"),
            )
        )
    _reject_duplicate_rule_ids([rule.rule_id for rule in rules], "ambiguity")
    return tuple(sorted(rules, key=lambda rule: rule.rule_id))


def _load_record_exclusion_rules(value: Any) -> tuple[RecordExclusionRule, ...]:
    rows = _required_list(value, "policy.record_exclusion_rules")
    rules: list[RecordExclusionRule] = []
    for index, value in enumerate(rows):
        path = f"policy.record_exclusion_rules[{index}]"
        row = _required_mapping(value, path)
        _require_keys(row, {"id", "values", "rationale"}, path)
        rules.append(
            RecordExclusionRule(
                rule_id=_required_identifier(row["id"], f"{path}.id"),
                values=tuple(_required_string_list(row["values"], f"{path}.values")),
                rationale=_required_string(row["rationale"], f"{path}.rationale"),
            )
        )
    _reject_duplicate_rule_ids([rule.rule_id for rule in rules], "record exclusion")
    return tuple(sorted(rules, key=lambda rule: rule.rule_id))


@cache
def _compile_bounded(phrase: str) -> re.Pattern[str]:
    tokens = phrase.split()
    if not tokens:
        raise RetrievalPolicyError("lexical phrases cannot be blank")
    expression = r"\s+".join(re.escape(token) for token in tokens)
    return re.compile(
        rf"(?<![{_BOUNDARY_CHARS}]){expression}(?![{_BOUNDARY_CHARS}])",
        flags=re.IGNORECASE,
    )


def _normalise_lexical_text(text: str) -> str:
    # Python's Unicode re.IGNORECASE treats dotted and dotless I as equivalent
    # to ASCII i. Translate them before casefolding so the Aho-Corasick matcher
    # remains exactly compatible with the frozen regex policy.
    return " ".join(text.translate(_REGEX_IGNORECASE_TRANSLATION).casefold().split())


def _is_boundary_char(value: str) -> bool:
    return value == "_" or "a" <= value <= "z" or "0" <= value <= "9"


@cache
def _compile_automaton(
    terms: tuple[tuple[str, str], ...],
) -> Any:
    if not terms:
        raise RetrievalPolicyError("lexical automaton cannot be empty")
    pattern_ids: dict[str, list[str]] = {}
    for term_id, phrase in terms:
        pattern = _normalise_lexical_text(phrase)
        if not pattern:
            raise RetrievalPolicyError("lexical phrases cannot be blank")
        pattern_ids.setdefault(pattern, []).append(term_id)
    automaton = ahocorasick.Automaton()
    for pattern, term_ids in sorted(pattern_ids.items()):
        automaton.add_word(pattern, (len(pattern), tuple(sorted(term_ids))))
    automaton.make_automaton()
    return automaton


def _record_is_excluded(text: str, rules: tuple[RecordExclusionRule, ...]) -> bool:
    normalised = text.strip().casefold()
    return any(normalised == value.casefold() for rule in rules for value in rule.values)


def _overlaps(first: tuple[int, int], second: tuple[int, int] | None) -> bool:
    if second is None:
        return False
    return first[0] < second[1] and second[0] < first[1]


def _channel_sort_key(channel: RetrievalChannel) -> int:
    return {
        RetrievalChannel.DIRECT_LEXICAL: 0,
        RetrievalChannel.SUBMISSION_EXPANSION: 1,
        RetrievalChannel.DIRECT_REPLY_EXPANSION: 2,
    }[channel]


def _evidence_sort_key(evidence: SelectionEvidence) -> tuple[int, str, str, int, tuple[str, ...]]:
    return (
        _channel_sort_key(evidence.channel),
        evidence.anchor_record_id,
        evidence.anchor_relation.value,
        evidence.expansion_depth,
        tuple(evidence.match_term_ids),
    )


def _require_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RetrievalPolicyError(f"{path} keys invalid; missing={missing}, unknown={unknown}")


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RetrievalPolicyError(f"{path} must be a table")
    return value


def _required_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise RetrievalPolicyError(f"{path} must be an array")
    return value


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalPolicyError(f"{path} must be a non-empty string")
    return value


def _required_identifier(value: Any, path: str) -> str:
    identifier = _required_string(value, path)
    if re.fullmatch(r"[a-z][a-z0-9_]*", identifier) is None:
        raise RetrievalPolicyError(f"{path} must be a lowercase snake-case identifier")
    return identifier


def _required_sha256(value: Any, path: str) -> str:
    digest = _required_string(value, path)
    if re.fullmatch(SHA256_PATTERN, digest) is None:
        raise RetrievalPolicyError(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _required_string_list(value: Any, path: str) -> list[str]:
    rows = _required_list(value, path)
    if not rows:
        raise RetrievalPolicyError(f"{path} cannot be empty")
    strings = [_required_string(item, f"{path}[]") for item in rows]
    if len(strings) != len(set(strings)):
        raise RetrievalPolicyError(f"{path} values must be unique")
    return strings


def _reject_duplicate_rule_ids(ids: list[str], label: str) -> None:
    if len(ids) != len(set(ids)):
        raise RetrievalPolicyError(f"{label} rule IDs must be unique")
