# Reproducing the study

## What is included

This is a source-code release, with configuration, annotation rubrics, schemas, dependency lockfile
and tests. It does not contain the Reddit corpus, row-level labels, private manifests, thread maps,
trained checkpoints or provider credentials.

The portable test selection in the README runs without those artefacts. The full historical suite
also requires a Git checkout and original retained-run inputs. Recreating the full study requires the original
inputs or a separately documented new run. Code access alone does not provide private data access.

## From classifier outputs to figures

The analysis expects:

1. Prediction Parquet shards from one corpus export.
2. Its manifest, including `corpus_rows` and `default_label_unseen_rows`.
3. The canonical inventory, which supplies source-volume counts.
4. A Parquet mapping from `corpus_position` to canonical submission `thread_id`.
5. The mapping receipt, including its content hash.

Default paths are defined at the top of `corpus_analysis_v1.py`. They refer to the original
ID/timestamp-enriched export. The later text-enriched export can also be supplied through the
command-line options; the analysis does not use raw text.

```sh
uv run --group analysis python -m reddit_china_stance.corpus_analysis_v1 --help
```

Prediction rows need timestamps, community, year, content type, retrieval route, calibration
membership, relevance probabilities/labels, and presence/stance probabilities/labels for each of
the five analytical targets. The mapping must preserve the study's one-record-per-thread scope.

The original analysis expects all 72 months of 2020–2025 and the ten study communities.
It is not a generic analyser for arbitrary corpora. The decomposition compares 2020 with 2025;
the separate 2022–2025 summaries describe recovery from the observed trough.
Use a fresh output directory per run; plot titles and axis limits are study-specific.

## Earlier stages

The source dataset and revision are recorded in `configs/dataset.toml` and
`configs/source-files.json`. Retrieval and language settings are in the corresponding policy files.

Cloud runners use named Modal volumes and expect intermediate artefacts at recorded paths.
Teacher runners require an authenticated Codex installation with the requested model available.
Model training additionally requires the ML dependency group and compatible GPU hardware.
Private Hugging Face exporters refer to the original repositories and require authorised access.

These are the actual study scripts, with explicit stage entry points and immutable input checks.
They are not a turnkey orchestration service. Inspect the stage's entry point, required artefacts
and confirmation arguments before launching it. A missing account, artefact or matching hash
should be resolved explicitly, not bypassed.

## Reading the estimates

For record i and target k:

```text
weight(i,k) = P(material) × P(target k present)
score(i,k)  = P(positive | k) − P(negative | k)
group mean = sum(weight × score) / sum(weight)
```

The probabilities for negative, no directed stance and positive are also reported separately.
Confidence intervals cluster by submission thread and condition on the fitted classifier.
They exclude model and calibration uncertainty.

Calibration temperatures are chosen using weighted log loss; classification thresholds are
selected separately. F1 is a classification diagnostic, not a probability-calibration metric.
The final calibration diagnostics use the fitting sample and model-assisted reference labels.

The descriptive analysis includes hard-label, equal-community, fixed-composition and
retrieval-route checks. It does not establish causal event effects.

## Source changes

See `RELEASE.md` for the original commit and the small portability edits in this release.
Changing a file included in a source bundle produces a new bundle hash. Historical receipts
belong to the original run; editing paths does not make them receipts for a new run.
Several downstream contracts also pin generated packet and receipt IDs. A fresh run generates new
IDs and requires regenerated downstream contracts before those stages can consume its outputs.
