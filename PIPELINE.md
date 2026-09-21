# Code map

All Python files below are in `src/reddit_china_stance/`.

| Stage | Main files |
| --- | --- |
| Archive manifest and ingestion | `source_manifest.py`, `ingestion.py`, `models.py`, `reconciliation.py` |
| Cloud ingestion and Parquet validation | `modal_full_ingestion.py`, `modal_parquet*.py`, `modal_reconcile.py` |
| Candidate retrieval | `retrieval.py`, `modal_stage_a.py`; policy: `configs/retrieval-policy-v3.toml` |
| English eligibility and context | `modal_stage_a_language.py`, `context_assembly.py` |
| Development reference and teacher selection | `human_reference_v1.py`, `human_seeded_consensus_v1.py`, `sol_adjudicator.py`, `sol_direct_teacher_eval.py` |
| Teacher packet and initial labelling | `modal_sol_teacher_packet.py`, `sol_teacher_generation.py` |
| Final v2 ontology and labelling | `semantic_ontology_v2.py`, `sol_ontology_bridge_v2.py`, `sol_teacher_10k_v2.py` |
| Training frames and B4 model | `prepare_modernbert_factorised_inputs.py`, `modernbert_factorised_*.py`, `modal_modernbert_factorised.py` |
| Additional-label sampling and comparison | `modernbert_acquisition*.py`, `modal_modernbert_acquisition*.py`, `sol_teacher_acquisition_v2.py` |
| Final ensemble and fresh calibration sample | `modernbert_probability_random_candidate_v1.py`, `modal_modernbert_probability_random_candidate_v1.py`, `modal_modernbert_probability_random_sample_v1.py` |
| Calibration | `modernbert_probability_random_calibration_v1.py`, `modal_modernbert_probability_random_calibration_v1.py` |
| Corpus inference | `modernbert_probability_random_corpus_inference_v1.py`, `modal_modernbert_probability_random_corpus_inference_v1.py` |
| Canonical ID, timestamp and text joins | `modernbert_corpus_source_enrichment_v2.py`, `modernbert_corpus_text_enrichment_v1.py` and their `modal_` runners |
| Thread mapping | `modal_corpus_thread_mapping_v1.py` |
| Dataset and model exports | `export_private_hf_*.py` |
| Descriptive statistics and plots | `corpus_analysis_v1.py` |
| Shared label scoring | `semantic_evaluation.py`, `semantic_evaluation_v2.py` |

## Sample accounting

The v2 teacher run labelled 10,000 records; 9,347 were eligible for the primary training design.
These were split into 8,147 training, 600 development and 600 reserved records.

The reserved 600 were later used to evaluate probability-random versus active acquisition.
The retained probability-random arm added 930 eligible training records from 1,000 queries.
The active arm was a comparison, not part of the retained training set.

A fresh 600-record probability sample was then used for final calibration. Those records are
flagged in the 908,141-row inference export and excluded from the 907,541-row primary analysis.

The six retained checkpoints are three relevance models and three B4 target/stance models,
using seeds 47, 61 and 89. Logits are averaged within each component before calibration.

## Supporting comparisons

The `modernbert_conditional_*` and `modernbert_cascade_*` files implement earlier alternatives.
The older `modernbert_model.py`, trainer and locked-test modules support the original
three-head learning-curve work. They remain here as supporting source, not the final architecture.

Qwen probes were separate experiments and are not included in this source snapshot.
They did not supply the final classifier. No event-regression implementation is included.

## Annotation instructions

The final v2 rubric is `docs/rubrics/target-stance-v2-pilot.md`, paired with
`schemas/target-stance-v2-pilot.schema.json`. The older rubric is retained for the
development reference and teacher-selection stages. Historical filenames are preserved because
the code and source manifests refer to them.

