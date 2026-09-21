"""Strict three-stage contract for relevance, target, and stance annotation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PHASES = ("relevance", "targets", "stance")
TARGETS = ("china_general", "government_ccp", "people_culture", "other")
STANCES = ("negative", "positive", "mixed", "no_directed_stance", "unclear")
RELEVANCE = ("material", "not_material", "unclear")

PROMPT_PREAMBLE = """PURPOSE
These labels support a thesis analysing changes in English-language Reddit discourse about China.

Label TARGET_TEXT only. Measure target-specific expressed stance, not generic sentiment, factual
truth, or the surrounding thread's mood. CONTEXT may resolve a pronoun, ellipsis, quotation, or
shorthand in TARGET_TEXT. It cannot donate a claim or stance absent from TARGET_TEXT. Never infer
China relevance from the subreddit, retrieval route, or the fact that a row was sampled."""

PHASE_PROMPTS = {
    "relevance": """STAGE
Decide only whether TARGET_TEXT materially discusses a China-related referent.

- material: a substantive assertion, evaluation, question, comparison, experience, instruction,
  or report about a China-related referent, including one resolved from CONTEXT.
- not_material: false lexical match, incidental mention, or no substantive China-related claim.
- unclear: a plausible China-related reading exists but supplied text cannot resolve materiality.

Neutral text, genuine questions, and unendorsed reports may still be material.
Return only {\"relevance\": \"...\"}.""",
    "targets": """STAGE
Identify every China-related target class materially discussed in TARGET_TEXT. Do not judge stance.

- china_general: China as a country, nation, society, or broad whole.
- government_ccp: the CCP, Chinese state, policy, agency, police, military, or official acting in
  an official capacity.
- people_culture: Chinese people, diaspora, ethnicity, identity, language, customs, food, arts,
  media, or culture.
- other: a China-related company, product, app, subnational place, institution, organisation,
  event, or person not acting as the state.

A specific actor does not imply china_general. \"Chinese\" does not imply government_ccp.
Return only {\"targets\": [\"...\"]}.""",
    "stance": """STAGE
Decide only the stance expressed by the author of TARGET_TEXT toward TARGET_CLASS.

- negative: criticises, blames, opposes, distrusts, attacks, or expresses contempt.
- positive: praises, supports, defends, sympathises with, or approves.
- mixed: meaningful positive and negative evaluation of the same target class.
- no_directed_stance: material but factual, inquisitive, instructive, or unendorsed reporting.
- unclear: attribution or evaluative direction cannot be resolved.

Follow negation, rebuttal, attribution, sarcasm, and contrast. Never inherit stance from CONTEXT or
another target class. Return only {\"stance\": \"...\"}.""",
}

EFFECTIVE_PROMPTS = {
    phase: f"{PROMPT_PREAMBLE}\n\n{PHASE_PROMPTS[phase]}" for phase in PHASES
}

SOL_ADJUDICATION_PROMPT = f"""{PROMPT_PREAMBLE}

TASK
Produce the complete semantic label independently. First decide relevance. If material, identify
every applicable target class and then decide stance separately for each target using the same
definitions as the three atomic stages. If relevance is not_material or unclear, target_stances
must be empty. Do not infer from sampling, source metadata, model outputs, or routing reasons.

Return only {{"relevance":"...","target_stances":[{{"target":"...","stance":"..."}}]}}."""

PROMPT_BUNDLE = {
    "teacher_phases": EFFECTIVE_PROMPTS,
    "sol_semantic_label": SOL_ADJUDICATION_PROMPT,
}

PHASE_SCHEMAS: dict[str, dict[str, Any]] = {
    "relevance": {
        "type": "object",
        "additionalProperties": False,
        "required": ["relevance"],
        "properties": {"relevance": {"enum": list(RELEVANCE)}},
    },
    "targets": {
        "type": "object",
        "additionalProperties": False,
        "required": ["targets"],
        "properties": {
            "targets": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(TARGETS),
                "uniqueItems": True,
                "items": {"enum": list(TARGETS)},
            }
        },
    },
    "stance": {
        "type": "object",
        "additionalProperties": False,
        "required": ["stance"],
        "properties": {"stance": {"enum": list(STANCES)}},
    },
}


def validate_phase_output(phase: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise one provider phase response."""

    if phase == "relevance":
        if set(value) != {"relevance"} or value.get("relevance") not in RELEVANCE:
            raise ValueError("invalid relevance output")
        return {"relevance": str(value["relevance"])}
    if phase == "targets":
        targets = value.get("targets")
        if set(value) != {"targets"} or not isinstance(targets, list) or not targets:
            raise ValueError("invalid target output")
        if any(target not in TARGETS for target in targets) or len(targets) != len(set(targets)):
            raise ValueError("invalid target output")
        return {"targets": sorted((str(target) for target in targets), key=TARGETS.index)}
    if phase == "stance":
        if set(value) != {"stance"} or value.get("stance") not in STANCES:
            raise ValueError("invalid stance output")
        return {"stance": str(value["stance"])}
    raise ValueError(f"unknown phase: {phase}")


def model_input(packet: Mapping[str, Any], *, target: str | None = None) -> str:
    """Build a provider prompt without exposing identifiers or source metadata."""

    allowed = {"target_text", "submission_context", "parent_context"}
    if "target_text" not in packet or not set(packet).issubset(allowed):
        raise ValueError("model input may contain only target text and bounded direct context")
    target_text = packet["target_text"]
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("target_text must be a non-empty string")
    for field in ("submission_context", "parent_context"):
        value = packet.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
    sections: list[str] = []
    if target is not None:
        if target not in TARGETS:
            raise ValueError("unknown target class")
        sections.append(f"TARGET_CLASS\n{target}")
    sections.extend(
        (
            f"TARGET_TEXT\n{target_text}",
            f"SUBMISSION_CONTEXT\n{packet.get('submission_context') or '[none]'}",
            f"IMMEDIATE_PARENT_CONTEXT\n{packet.get('parent_context') or '[none]'}",
        )
    )
    return "\n\n".join(sections)


def compose_label(
    relevance: Mapping[str, Any],
    targets: Mapping[str, Any] | None = None,
    stances: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compose one final semantic label from validated atomic outputs."""

    relevance_value = validate_phase_output("relevance", relevance)["relevance"]
    if relevance_value != "material":
        if targets is not None or stances:
            raise ValueError("non-material or unclear rows cannot carry targets or stances")
        return {"relevance": relevance_value, "target_stances": []}
    if targets is None:
        raise ValueError("material rows require target output")
    target_values = validate_phase_output("targets", targets)["targets"]
    if stances is None or set(stances) != set(target_values):
        raise ValueError("material rows require exactly one stance for every selected target")
    return {
        "relevance": "material",
        "target_stances": [
            {
                "target": target,
                "stance": validate_phase_output("stance", stances[target])["stance"],
            }
            for target in target_values
        ],
    }
