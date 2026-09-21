# Reddit China stance

Code used to study target-specific expressed stance in China-related Reddit discourse,
2020–2025, across ten subreddits.

This release covers data preparation, retrieval, annotation, classifier development,
calibration, corpus inference and descriptive analysis. Data and trained weights are not bundled.

## Final workflow

1. Ingest the pinned Reddit archives and retain records from 2020–2025.
2. Retrieve China-related candidates with lexical rules and semantic expansion.
3. Filter for English, assemble bounded context, and apply thread and duplicate exclusions.
4. Label development and training samples with Sol under the target-specific annotation rubric.
5. Train ModernBERT-large. Retain the B4 model and probability-random additional training sample.
6. Ensemble three relevance checkpoints and three target/stance checkpoints.
7. Fit temperatures and thresholds on a fresh 600-record probability sample.
8. Score 908,141 eligible records.
9. Join canonical IDs, timestamps, text and submission threads as needed.
10. Analyse 907,541 records after excluding the 600 calibration records.

Stance has three classes: negative, no directed stance, and positive. Mixed cases were excluded
from stance loss, while still contributing to target-presence training.

See [PIPELINE.md](PIPELINE.md) for the files implementing each stage and the sample accounting.

## Install

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked --group cloud --group analysis
uv run reddit-china-stance --help
```

For training and model inference, use a compatible GPU environment and add the ML dependencies:

```sh
uv sync --locked --group cloud --group analysis --group ml
```

The original cloud jobs use Modal. Teacher labelling uses an authenticated Codex runtime.
The recorded model identifiers, cloud volume names and input hashes describe the original setup.
They are not credentials and do not grant access.

## Reproduce the analysis

With the original prediction export and thread mapping in the expected local paths:

```sh
uv run --group analysis python -m reddit_china_stance.corpus_analysis_v1
```

The script produces 17 CSV tables, 13 figures in PNG/PDF/SVG, a summary and a receipt.
Use `--help` to provide other input and output paths.

[REPRODUCING.md](REPRODUCING.md) lists the required inputs and explains which parts can run
without the original private artefacts. This repository does not download the full corpus or
start paid jobs during installation or tests.

## Check

```sh
uv run ruff check .
uv run pytest tests/test_ingestion.py tests/test_models.py tests/test_reconciliation.py tests/test_retrieval.py tests/test_context_assembly.py tests/test_semantic_evaluation_v2.py tests/test_modernbert_probability_random_calibration_v1.py tests/test_modernbert_corpus_source_enrichment_v2.py tests/test_modernbert_corpus_text_enrichment_v1.py tests/test_corpus_analysis_v1.py
```

The command above runs 146 portable checks using synthetic examples and temporary files.
The full historical suite also includes tests requiring a Git checkout and original private
run artefacts; it cannot all pass in this data-free folder. Cloud and teacher jobs require
explicit launch commands and configured accounts.

## Layout

- `src/reddit_china_stance/`: implementation.
- `configs/`: source manifest, retrieval policies and run settings.
- `schemas/`: record and annotation formats.
- `docs/rubrics/`: annotation instructions used by the code.
- `tests/`: unit and integration tests.
- `uv.lock`: dependency versions.

Earlier model comparisons remain in the source snapshot because the final workflow shares their
loaders, scoring and validation code. They are identified in PIPELINE.md; they are not all steps
in the final classifier. The example pipeline configuration also retains the older teacher setup
and should not be treated as a one-command launcher for the final workflow.

## Interpretation

These are model-assisted predictions of expressed stance in a selected corpus.
Independent human validation of the final classifier remains a limitation. The analysis does not
estimate public opinion or establish causal event effects.

The original analysis script includes its original generated report and study-specific plot
titles. Check each statement against the tables when incorporating it into a thesis.

Keep data, model weights, credentials and generated outputs out of GitHub. The supplied
`.gitignore` excludes those directories. Upload this folder's source and documentation.
