# Sol direct-teacher evaluation v1

## Decision

Decide whether one isolated `gpt-5.6-sol` composite call can replace the staged Gemma/GPT-OSS
teacher cascade for the first 10,000 silver labels. This is an operational teacher-selection test,
not evidence that Sol equals independent human coders.

## Frozen experiment

- Evaluate every one of the 230 `locked_test_candidate` rows from development proxy
  `4ba40fcd0eb33dc2c7ca1c0b9ac75b06f42dc120bed470f9da7f713194a3a66d`.
- Give the model only opaque test IDs, target text and permitted submission/parent context.
- Hide the human label, final proxy label, split, resolution, reviewer outputs and repository context.
- Use `gpt-5.6-sol` with high reasoning through fresh ephemeral `codex exec` processes, read-only
  sandboxing, no tools, no web, no memories and no project instructions.
- Use the frozen semantic rubric and only its synthetic boundary examples. No test row may be used
  as a few-shot example.
- Execute six deterministic shards with at most six concurrent processes. Do not retry a failed
  shard with a changed prompt, model, reasoning level or schema.
- Freeze predictions before loading labels for scoring.

## Hypotheses

- Favoured: a direct composite Sol judgement clears the full capability gate with zero invalid
  outputs and also clears the separate human-supported gate.
- Boring: agreement is high mainly on the 41 rows whose reference was previously Sol-adjudicated;
  the 189 human-supported rows do not clear their gate.
- Adversarial: the synthetic examples or batch layout anchor the model to over-predict materiality,
  targets or directional stance.
- Unknown: disagreement is concentrated in rare targets or stances whose support is too small for
  a stable conclusion.

## Constructs and proxies

The construct is target-specific expressed stance towards China-related referents. The observed
proxy is exact agreement with the completed model-assisted development reference. Results are
reported separately for:

1. 189 strict human-supported-majority rows, where the retained label agrees with the supplied
   human coding pass;
2. 41 blinded-Sol-adjudicated rows, which are circular evidence for this model family; and
3. all 230 rows against the original human coding pass as a noisy sensitivity analysis.

None is an independent, double-coded human evaluation.

## Frozen continuation gate

The Sol-only 10k route is selected only when all of the following pass:

1. the existing conjunctive `development-proxy-capability-v1` gate passes on all 230 rows;
2. all 230 outputs are present and schema-valid;
3. on the 189 strict human-supported-majority rows, material recall is at least 0.85, core-target
   micro-F1 is at least 0.75, fixed-reference core-target stance accuracy is at least 0.75 and
   end-to-end core target-plus-stance micro-F1 is at least 0.70; and
4. on the full proxy, no supported core target has recall below 0.60.

Whole-row accuracy, the original-human sensitivity result, rare-class results and throughput are
diagnostics, not post-hoc gates. If the gate fails, retain the staged/cascade design or revise the
measurement contract in a new version; do not tune this frozen run.

## Confounds and stop rules

- The proxy was partly produced by Sol, so high overall agreement can overstate validity.
- The human-supported subset was selected because the supplied human label matched at least one
  earlier Sol review. It is therefore not an independent human comparison, even though the retained
  label equals the supplied human label.
- The human-supported subset has only 17 material rows, including 4 `china_general`, 7
  `government_ccp` and 6 `people_culture` instances. Per-class estimates are therefore noisy.
- One human pass is not inter-coder reliability evidence.
- Stop on any schema failure, row mismatch, identity drift, unexpected output or altered digest.
- Do not use this result for thesis measurement claims; the later independent human evaluation
  remains mandatory.

## Commands

```bash
uv run python -m reddit_china_stance.sol_direct_teacher_eval --action prepare
uv run python -m reddit_china_stance.sol_direct_teacher_eval --action run
uv run python -m reddit_china_stance.sol_direct_teacher_eval --action score
```

## Result

Run `b3356b5d07bffeae351c761476a8751d8a2248d7bd68fa0604b5b8d945f13923` completed all
230 rows in six isolated shards with zero invalid outputs. Slowest-shard wall time was 122.42
seconds; aggregate shard time was 473.87 seconds. The run used 106,435 input and 23,116 output
tokens, with zero cached input tokens reported.

Against the completed proxy, relevance macro-F1 was 0.985, material recall 0.976, core-target
micro-F1 0.902, fixed-target core stance accuracy 0.854 and end-to-end core target-plus-stance
micro-F1 0.854. Whole-row accuracy was 0.948.

The pre-registered human-supported gate also passed separately: across 189 rows, material recall
was 1.000, core-target micro-F1 0.914, fixed-target core stance accuracy 0.941 and end-to-end core
target-plus-stance micro-F1 0.914. Whole-row accuracy was 0.974.

The original-human-label sensitivity over all 230 rows was materially weaker: core-target
micro-F1 0.659, fixed-target core stance accuracy 0.512 and whole-row accuracy 0.817. This gap is
concentrated in the rows whose original human label did not survive the model-assisted consensus
process. It is not evidence that either side is ground truth; it reinforces the need for later
independent double coding.

All frozen gates passed. Operational decision: use direct composite Sol labels for the 10k teacher
set instead of the staged Gemma/GPT-OSS cascade. This does not promote ANN-2 or HUMAN-1 by itself.
