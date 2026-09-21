# Target and stance ontology v2 pilot rubric

## Purpose and evidence boundary

This pilot tests a candidate ontology for a master's thesis on changes in English-language Reddit
discourse about China. It measures **target-specific expressed stance** in `TARGET_TEXT`. It does
not measure generic sentiment, factual truth, topic negativity, the author's private attitude, or
the mood of the surrounding thread.

The labels produced under this rubric are model-assisted development evidence. They are not human
gold and cannot validate thesis claims. The pilot decides whether the candidate ontology is clear
enough to take into independent human annotation.

## Unit and permitted context

- Label `TARGET_TEXT` only.
- `SUBMISSION_CONTEXT` and `IMMEDIATE_PARENT_CONTEXT` may resolve a pronoun, nickname, quotation,
  ellipsis, reply function, or omitted referent.
- Context may complete an unambiguous assent, rejection, defence, or rebuttal, but may not donate a
  target or stance that the reply does not express.
- A question, acknowledgement, link, request, or ambiguous backchannel does not inherit the
  context's evaluation.
- Never use the subreddit, sampling route, source metadata, prior labels, or appearance in this
  pilot as semantic evidence.

Every selected pilot item was material under the legacy ontology. Reassess it under this rubric;
do not preserve that legacy decision by default.

## Annotation-level codability and relevance

First decide whether the item is codable, then whether China is materially discussed. These are
different decisions.

- `codable` means the supplied target text and permitted context support a responsible relevance
  decision and, when material, a complete target/stance annotation.
- `not_codable` means a responsible relevance decision itself is impossible because the text is
  deleted, truncated, internally contradictory, genuinely ambiguous, or depends on missing
  context. Return `relevance: null` and no targets. It is an annotation-level mask retained for
  exclusion or manual review and is **not** a downstream model class.
- `material` means `TARGET_TEXT` substantively asserts, asks, evaluates, reports, defends, rejects,
  or otherwise communicates a proposition about at least one China-related candidate target.
  Return at least one target.
- `not_material` means a codable item contains no substantive China-related proposition: for
  example a lexical false positive, an incidental name, or context that mentions China while the
  target text does not express or inherit a relevant proposition. Return no targets.

Do not use `not_codable` for a clear non-material item, or merely because a case is difficult,
neutral, factual, sarcastic, or multi-target. Use `no_directed_stance` for a material analytical
target without attributable evaluation. There is no `unclear` relevance, stance, or target class.

## Candidate targets

Return every materially discussed target once. Use the narrowest actual object of the proposition;
a narrower target does not automatically imply `china_general`.

- `china_general`: China as a country, nation, society, or broad whole when the proposition or
  evaluation is directed at that broad entity rather than a narrower target below.
- `government_ccp`: the CCP, PRC state, government policy, state agency, police, military, or an
  official acting in an official capacity. Use implicit government reference only when the
  predicate unambiguously describes state or official action.
- `people_identity`: Chinese people, diaspora, ethnicity, nationality, identity, or a person/group
  discussed specifically as Chinese. Do not use it for language, art, food, media, or a company.
- `culture_media`: Chinese language, writing, customs, cuisine, history-as-cultural-heritage, arts,
  entertainment, news/media, or other cultural production. Do not add `people_identity` unless
  people or identity are independently material.
- `company_tech_product`: a China-related company, brand, platform, app, product, technology, or
  corporate actor when that object is materially discussed. State action involving a firm can
  additionally support `government_ccp`, but Chinese ownership alone does not.
- `residual_other`: a materially discussed China-related referent not captured above, such as a
  subnational place, non-corporate institution, organisation, event, or non-state actor whose
  Chinese identity is not itself the proposition. This is a presence-only diagnostic bucket.

`residual_other` must have `stance: null`. Do not force evaluative direction onto this incoherent
residual category. It may coexist with analytical targets.

## Stance for analytical targets

- `negative`: attributable evaluative commitment that criticises, blames, opposes, distrusts,
  attacks, or expresses contempt toward the target.
- `positive`: attributable evaluative commitment that praises, supports, defends, sympathises
  with, or approves of the target.
- `mixed`: meaningful positive and negative evaluation of the same target in the target text.
- `no_directed_stance`: the target is material, but the target text is factual, genuinely
  inquisitive, instructive, or an unendorsed quotation/report.

Follow negation, attribution, reported speech, contrast, sarcasm, defence, and rebuttal. Adverse or
beneficial facts, allegations, headlines, and event descriptions alone remain
`no_directed_stance`. A negative topic is not automatically negative stance. Determine stance
separately for each analytical target.

## Decision order

1. Resolve only the references needed to understand `TARGET_TEXT`.
2. Decide whether a relevance decision is possible. If not, return `codability: not_codable`,
   `relevance: null`, and no targets.
3. If codable, decide materiality. For `not_material`, return no targets.
4. For `material`, identify every actual target without adding broader implied targets.
5. Assign one stance to every analytical target; assign `null` only to `residual_other`.
6. Return exactly the structured label. Do not return rationale, confidence, quotations, evidence,
   topics, actor extraction, or private reasoning.

## Synthetic boundary examples

These examples are invented and do not come from the pilot packet.

- “The CCP's censorship order is indefensible.” → `codable`, `material`,
  `government_ccp: negative`.
- “Chinese immigrants deserve support, but this custom should change.” →
  `people_identity: positive`, `culture_media: negative`.
- “Mandarin uses tones.” → `culture_media: no_directed_stance`; do not add
  `people_identity`.
- “This Chinese phone is excellent.” → `company_tech_product: positive`; do not add
  `china_general`.
- “The province reported new rainfall figures.” → `residual_other: null` if the province's
  China relation is material and no analytical target is expressed.
- “A newspaper quoted someone calling China evil.” → `china_general: no_directed_stance` unless
  `TARGET_TEXT` endorses or rejects the quotation.
- “China” appearing only as part of an unrelated username → `codable`, `not_material`, no targets.
- Deleted target text whose parent allows several incompatible China referents → `not_codable`,
  `relevance: null`, no targets.

## Pilot limitations

- `residual_other` is diagnostic and cannot support target-specific stance analysis.
- Rare `mixed` cases and annotation-level `not_codable` may have inadequate support in 480 rows.
- The legacy `people_culture` and `other` labels are used only to select evaluation strata; they are
  hidden from reviewers and do not define v2 truth.
- Passing the engineering gates warrants a human pilot, not production promotion.
