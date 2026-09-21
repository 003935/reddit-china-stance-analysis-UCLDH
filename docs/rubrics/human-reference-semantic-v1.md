# Human-reference semantic rubric v1

## Purpose

Create minimal labels for a master's thesis analysing changes in English-language Reddit discourse
about China. The measurement is **target-specific expressed stance**, not generic sentiment, factual
truth, the author's private attitude, or the mood of the surrounding thread.

This rubric preserves the four classes used in the supplied human workbook so those annotations can
be evaluated without silently relabelling them. `china_general`, `government_ccp`, and
`people_culture` are the core classes. `other` is retained for compatibility but is diagnostic: it
combines companies, products, places, organisations, events, and non-state people and therefore is
not a coherent confirmatory target.

## Unit and context

- Label `TARGET_TEXT` only.
- `SUBMISSION_CONTEXT` and `IMMEDIATE_PARENT_CONTEXT` may resolve a pronoun, ellipsis, nickname,
  quotation, reply function, or missing referent.
- Context may resolve the complete proposition expressed elliptically. An unambiguous assent,
  rejection, defence, or rebuttal therefore carries the target and stance expressed by that reply.
- A mere acknowledgement, link, request, question, or ambiguous backchannel does not inherit the
  context's stance.
- Never infer China relevance from the subreddit, retrieval route, sampling process, or the fact
  that a row appears in this task.
- If supplied context contains no plausible China link, use `not_material`, not `unclear`.
- Use `unclear` only when a plausible China-related interpretation exists but missing, deleted,
  truncated, or genuinely ambiguous context prevents a responsible decision.

## Relevance

- `material`: `TARGET_TEXT`, after permitted reference resolution, makes or relays a substantive
  assertion, evaluation, question, comparison, experience, instruction, or report about a
  China-related referent. Neutral material and genuine questions are still material.
- `not_material`: false lexical match, incidental location or comparison, no China-related semantic
  referent, or no substantive proposition about that referent.
- `unclear`: a plausible China-related reading exists but the supplied text cannot resolve whether
  the referent is materially discussed.

Chinese language, writing, culture, history, sinophobia, and products whose China/Chinese relation
is substantively discussed are in scope. A bare place, company, nationality adjective, or product
used incidentally is not. Missing context alone does not make a row unclear when there is no China
signal to resolve.

For `not_material` or `unclear`, return no targets. For `material`, return every materially
discussed target once.

## Targets

- `china_general`: China as a country, nation, society, or broad whole when no narrower object
  carries the same evaluation.
- `government_ccp`: the CCP, PRC state, government policy, state agency, police, military, or an
  official acting in an official capacity. Implicit government reference is allowed only when the
  predicate unambiguously describes official or state action.
- `people_culture`: Chinese people, diaspora, ethnicity, identity, language, writing, customs,
  cuisine, arts, media, or culture.
- `other`: a materially discussed China-related company, product, app, subnational place,
  institution, organisation, event, or person not acting as the state.

A specific target does not automatically imply `china_general`. The word "Chinese" does not imply
`government_ccp`. A company or person maps to `government_ccp` only when the target text presents
the relevant conduct as state or official action.

## Stance per target

- `negative`: attributable evaluative commitment that criticises, blames, opposes, distrusts,
  attacks, or expresses contempt toward the target.
- `positive`: attributable evaluative commitment that praises, supports, defends, sympathises
  with, or approves of the target.
- `mixed`: meaningful positive and negative evaluation of the same target class in the target text.
- `no_directed_stance`: the target is material but the target text is factual, genuinely
  inquisitive, instructive, or an unendorsed quotation/report.
- `unclear`: the target is identifiable but attribution or evaluative direction cannot be resolved.

Follow negation, attribution, reported speech, contrast, sarcasm, rebuttal, and defence. Adverse or
beneficial facts, allegations, headlines, and event descriptions alone remain
`no_directed_stance`; a negative topic is not automatically negative stance. Quoted or reported
evaluation is not the author's stance unless the target text endorses, rejects, praises, or
criticises it. Determine stance separately for every target; do not copy stance between classes.

## Decision order

1. Resolve only the references needed to understand `TARGET_TEXT`.
2. Decide whether a substantive China-related proposition is present.
3. If material, identify the actual target classes without adding broader implied classes.
4. For each target, determine directed stance rather than generic sentiment.
5. Return only the required structured fields. No rationale, confidence, topic, actor extraction,
   quotation, or private reasoning is part of the label.

## Synthetic boundary examples

These examples are invented for the rubric and do not come from either benchmark workbook.

- Parent: “The new visa policy is cruel.” Reply: “Yes, absolutely.” → material,
  `government_ccp: negative` when the supplied context unambiguously establishes that the policy is
  Chinese state policy; the assent expresses the resolved proposition.
- Parent: “The new visa policy is cruel.” Reply: “When did it start?” → material,
  `government_ccp: no_directed_stance`; the question does not inherit the parent's criticism.
- “A newspaper reported that growth fell last year.” → material, target determined by the stated
  referent, `no_directed_stance`; an adverse reported fact alone is not negative stance.
- “That claim that Chinese people are dishonest is nonsense.” → material,
  `people_culture: positive`; the author rejects a hostile generalisation rather than endorsing it.
- “Fantastic, another perfectly transparent censorship order.” → material,
  `government_ccp: negative` when the irony is attributable and its direction is clear.
- “This language app has a better spaced-repetition mode.” → material, `other: positive` only when
  the supplied target text substantively identifies the app as China/Chinese-related; otherwise it
  is `not_material`.

## Known limitations

- `people_culture` merges people and culture/language, which the final thesis ontology may need to
  separate before population inference.
- `other` is not a stable analytical construct and is not promotion-gated in this benchmark.
- The supplied human workbook contains only one multi-target row and very few `mixed`/`other`
  examples. Those behaviours can be inspected but not validated from this reference.
- This version measures agreement with one human coding pass. It does not establish inter-coder
  reliability or final thesis validity.
