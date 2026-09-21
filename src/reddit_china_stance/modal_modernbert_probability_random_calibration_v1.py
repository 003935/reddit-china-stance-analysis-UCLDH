"""Run the sole calibration and throughput job for the retained random ensemble.

The launcher validates the immutable 600-row sample, final Sol-teacher labels and
all six exact-bound checkpoints before any CUDA work.  One L4 job computes the
three-seed mean logits, weighted calibration, aggregate-only diagnostics and a
128-row full-ensemble throughput benchmark.  It has no training, retry, locked-test
or corpus-inference action.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_probability_random_calibration_v1 as calibration
from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-modernbert-probability-random-calibration-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path(candidate.NAMESPACE)
SAMPLE_RUN_ID = "247c077975b5527203e2de709e8c884dd90ca8e74976130d4b30da5a4a5c957c"
SAMPLE_RECEIPT_ID = "62c3bd62d9e6d547a441f3c05499f6e80a9f244d0830d9a271c50f804122cc32"
SAMPLE_SOURCE_BUNDLE_ID = "5e41fe4be91cd3f0fc0c6b2c69bc063ff5542acf8f31bfaa98af1cc12ef06bad"
CHECKPOINT_COHORT_ID = "ad01414410538d5001ac5fb0070751ccfd251d31c5b75c5460bcd5a0dbe06f9b"
MEMBERSHIP_SHA256 = "14329149dd8e7ec0f11bdedac44f464141b0543dace86888d80c061fc8c309a1"
MEMBERSHIP_BYTES = 197_653
TEACHER_SOURCE_SHA256 = "3f80c55eaa2b6c600fe662ddbe3ff74f742ee806f15332ff519dfd4bf3f8b3a0"
TEACHER_SOURCE_BYTES = 147_112
TEACHER_RUN_ID = "75c48c76db22be4f3eb009ec379cdf75e8ac88761b7475e6d9ce31f4f3ec0949"
TEACHER_PACKET_ID = "f7c6765830c448d08e69e62f0e20a485ded5399a3a9e842838492fef44fc56ba"
TEACHER_LABEL_SHA256 = "5ad505aa6f25ac5541ded95b00f8736238c839c729502c92295f1c43b085a4e5"
TEACHER_LABEL_BYTES = 13_703
TEACHER_MAPPING_SHA256 = "d2e163d5d026d0cb9e3992275b0a8e7c9370253ee947a789b5bf55bd9da9b555"
TEACHER_MAPPING_BYTES = 19_179
TEACHER_RECEIPT_SHA256 = "d3a7bf0ca0fe0e6162b90dc91a458251676a6ec79229e2fa915e7d5daf02f69e"
TEACHER_RECEIPT_BYTES = 6_024
THROUGHPUT_ROWS = 128
THROUGHPUT_SEED = "modernbert-post-acquisition-calibration-v1-throughput-128"
GPU_TYPE = "L4"
GPU_TIMEOUT_SECONDS = 60 * 60


def _repository_root(module_path: Path) -> Path:
    resolved = module_path.resolve()
    if (
        len(resolved.parents) >= 3
        and resolved.parent.name == "reddit_china_stance"
        and resolved.parent.parent.name == "src"
    ):
        return resolved.parents[2]
    return resolved.parent


REPO_ROOT = _repository_root(Path(__file__))
POLICY_REPO_PATH = "configs/modernbert-post-acquisition-calibration-v1.toml"
POLICY_RUNTIME_PATH = "/configs/modernbert-post-acquisition-calibration-v1.toml"
SCHEMA_REPO_PATH = "schemas/target-stance-v2-pilot.schema.json"
SCHEMA_RUNTIME_PATH = "/schemas/target-stance-v2-pilot.schema.json"
REQUIRED_SOURCE_FILES = (
    POLICY_REPO_PATH,
    SCHEMA_REPO_PATH,
    "src/reddit_china_stance/modal_modernbert_probability_random_calibration_v1.py",
    "src/reddit_china_stance/modernbert_probability_random_calibration_v1.py",
    "src/reddit_china_stance/modernbert_probability_random_candidate_v1.py",
    "src/reddit_china_stance/modernbert_factorised_data.py",
    "src/reddit_china_stance/modernbert_factorised_experiment.py",
    "src/reddit_china_stance/modernbert_factorised_model.py",
    "src/reddit_china_stance/modernbert_factorised_training.py",
    "src/reddit_china_stance/modernbert_model.py",
    "src/reddit_china_stance/modernbert_trainer.py",
    "src/reddit_china_stance/privacy.py",
    "src/reddit_china_stance/semantic_evaluation_v2.py",
    "src/reddit_china_stance/semantic_ontology_v2.py",
)
RUNTIME_DEPENDENCIES = {
    "accelerate": "1.10.1",
    "huggingface-hub": "0.36.2",
    "jsonschema": "4.26.0",
    "pyarrow": "25.0.1",
    "pydantic": "2.13.4",
    "safetensors": "0.8.0",
    "torch": "2.8.0",
    "transformers": "4.57.6",
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME,
    environment_name=ENVIRONMENT_NAME,
    create_if_missing=False,
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*(f"{name}=={version}" for name, version in RUNTIME_DEPENDENCIES.items()))
    .run_commands(
        'python -c "from huggingface_hub import snapshot_download; '
        "snapshot_download(repo_id='answerdotai/ModernBERT-large', "
        "revision='45bb4654a4d5aaff24dd11d4781fa46d39bf8c13')\""
    )
    .env(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_python_source("reddit_china_stance")
    .add_local_file(POLICY_REPO_PATH, remote_path=POLICY_RUNTIME_PATH)
    .add_local_file(SCHEMA_REPO_PATH, remote_path=SCHEMA_RUNTIME_PATH)
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"existing immutable state differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise RuntimeError(f"stale immutable staging file exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_source_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    files = value.get("files")
    if (
        set(value) != {"schema_version", "kind", "files", "code_sha256", "source_bundle_id"}
        or value.get("schema_version") != candidate.SCHEMA_VERSION
        or value.get("kind") != "listed-source-bundle-v1"
        or not isinstance(files, Mapping)
        or set(files) != set(REQUIRED_SOURCE_FILES)
        or any(
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not candidate.is_sha256(digest)
            for path, digest in files.items()
        )
    ):
        raise ValueError("calibration source bundle schema drifted")
    clean_files = dict(sorted(files.items()))
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "listed-source-bundle-v1",
        "files": clean_files,
        "code_sha256": candidate.canonical_sha256(clean_files),
    }
    expected = {**body, "source_bundle_id": candidate.canonical_sha256(body)}
    if dict(value) != expected:
        raise RuntimeError("calibration source bundle digest drifted")
    return expected


def build_source_bundle(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    files = {
        relative: _file_sha256(repo_root / relative) for relative in sorted(REQUIRED_SOURCE_FILES)
    }
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "listed-source-bundle-v1",
        "files": files,
        "code_sha256": candidate.canonical_sha256(files),
    }
    return validate_source_bundle({**body, "source_bundle_id": candidate.canonical_sha256(body)})


def _validate_runtime_policy(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    bundle = validate_source_bundle(source_bundle)
    path = Path(POLICY_RUNTIME_PATH)
    if not path.is_file() or _file_sha256(path) != bundle["files"][POLICY_REPO_PATH]:
        raise RuntimeError("runtime calibration policy differs from the source bundle")
    policy = tomllib.loads(path.read_text(encoding="utf-8"))
    phases = policy.get("phase_availability", {})
    compute = policy.get("compute", {})
    if (
        phases.get("calibration_inference") is not True
        or phases.get("throughput_benchmark") is not True
        or phases.get("training") is not False
        or phases.get("provider_execution") is not False
        or phases.get("locked_test") is not False
        or phases.get("corpus_inference") is not False
        or compute.get("allowed_gpus") != [GPU_TYPE]
        or compute.get("gpu_fallback_allowed") is not False
        or compute.get("retry_authorised") is not False
        or compute.get("maximum_locked_test_rows") != 0
        or compute.get("maximum_corpus_rows") != 0
        or compute.get("phase_cost_cap_usd") != 10
        or compute.get("shared_hard_cap_usd") != 200
    ):
        raise RuntimeError("runtime calibration authority or compute boundary drifted")
    return bundle


def _validate_file(
    path: Path,
    *,
    sha256: str,
    size: int,
    where: str,
    parquet_rows: int | None = None,
) -> None:
    if not path.is_file() or path.stat().st_size != size or _file_sha256(path) != sha256:
        raise RuntimeError(f"{where} descriptor drifted")
    if parquet_rows is not None:
        import pyarrow.parquet as pq

        if pq.ParquetFile(path).metadata.num_rows != parquet_rows:
            raise RuntimeError(f"{where} row count drifted")


def _sample_root(*, volume_root: Path) -> Path:
    return volume_root / OUTPUT_PREFIX / f"run={SAMPLE_RUN_ID}"


def _validate_inputs(*, volume_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = _sample_root(volume_root=volume_root)
    sample_receipt = _read_json(root / "receipt.json", where="calibration sample receipt")
    sample_body = {key: value for key, value in sample_receipt.items() if key != "receipt_id"}
    if (
        sample_receipt.get("receipt_id") != SAMPLE_RECEIPT_ID
        or candidate.canonical_sha256(sample_body) != SAMPLE_RECEIPT_ID
        or sample_receipt.get("run_id") != SAMPLE_RUN_ID
        or sample_receipt.get("source_bundle_sha256") != SAMPLE_SOURCE_BUNDLE_ID
        or sample_receipt.get("checkpoint_cohort_id") != CHECKPOINT_COHORT_ID
        or sample_receipt.get("sample_count") != candidate.CALIBRATION_ROWS
        or sample_receipt.get("post_acquisition_population_count")
        != candidate.REMAINING_POPULATION_ROWS
        or sample_receipt.get("locked_test_access_count") != 0
    ):
        raise RuntimeError("calibration sample receipt drifted")
    membership_path = root / "calibration-membership.parquet"
    source_path = root / "calibration-source.parquet"
    _validate_file(
        membership_path,
        sha256=MEMBERSHIP_SHA256,
        size=MEMBERSHIP_BYTES,
        where="calibration membership",
        parquet_rows=candidate.CALIBRATION_ROWS,
    )
    _validate_file(
        source_path,
        sha256=TEACHER_SOURCE_SHA256,
        size=TEACHER_SOURCE_BYTES,
        where="calibration teacher source",
        parquet_rows=candidate.CALIBRATION_ROWS,
    )
    cohort = _read_json(root / "checkpoint-cohort.json", where="checkpoint cohort")
    if (
        cohort.get("cohort_id") != CHECKPOINT_COHORT_ID
        or candidate.canonical_sha256(
            {key: value for key, value in cohort.items() if key != "cohort_id"}
        )
        != CHECKPOINT_COHORT_ID
        or cohort.get("locked_test_access_count") != 0
    ):
        raise RuntimeError("checkpoint cohort drifted")
    if [member.get("trial_id") for member in cohort.get("members", [])] != [
        spec.trial_id for spec in candidate.CHECKPOINT_SPECS
    ]:
        raise RuntimeError("checkpoint cohort membership drifted")
    for member, spec in zip(cohort["members"], candidate.CHECKPOINT_SPECS, strict=True):
        descriptor = member["checkpoint"]
        _validate_file(
            volume_root / descriptor["relative_path"],
            sha256=spec.checkpoint_sha256,
            size=descriptor["bytes"],
            where=f"{spec.component} seed {spec.seed} checkpoint",
        )

    teacher_root = root / "teacher"
    labels_path = teacher_root / "labels.parquet"
    mapping_path = teacher_root / "private-mapping.parquet"
    teacher_receipt_path = teacher_root / "receipt.json"
    _validate_file(
        labels_path,
        sha256=TEACHER_LABEL_SHA256,
        size=TEACHER_LABEL_BYTES,
        where="teacher labels",
        parquet_rows=candidate.CALIBRATION_ROWS,
    )
    _validate_file(
        mapping_path,
        sha256=TEACHER_MAPPING_SHA256,
        size=TEACHER_MAPPING_BYTES,
        where="teacher private mapping",
        parquet_rows=candidate.CALIBRATION_ROWS,
    )
    _validate_file(
        teacher_receipt_path,
        sha256=TEACHER_RECEIPT_SHA256,
        size=TEACHER_RECEIPT_BYTES,
        where="teacher aggregate receipt",
    )
    teacher_receipt = _read_json(teacher_receipt_path, where="teacher aggregate receipt")
    if (
        teacher_receipt.get("status") != "complete"
        or teacher_receipt.get("run_id") != TEACHER_RUN_ID
        or teacher_receipt.get("packet_id") != TEACHER_PACKET_ID
        or teacher_receipt.get("row_count") != candidate.CALIBRATION_ROWS
        or teacher_receipt.get("private_labels_parquet_sha256") != TEACHER_LABEL_SHA256
        or teacher_receipt.get("private_mapping_sha256") != TEACHER_MAPPING_SHA256
        or teacher_receipt.get("source_parquet_sha256") != TEACHER_SOURCE_SHA256
        or teacher_receipt.get("requested_model") != "gpt-5.6-sol"
        or teacher_receipt.get("reasoning_effort") != "high"
        or teacher_receipt.get("failed_attempt_count") != 0
        or teacher_receipt.get("automatic_retry_count") != 0
        or teacher_receipt.get("primary_training_eligible_count") != 560
        or teacher_receipt.get("primary_training_ineligible_count") != 40
        or teacher_receipt.get("relevance_counts")
        != {"material": 485, "not_material": 106, "null": 9}
    ):
        raise RuntimeError("teacher aggregate receipt contract drifted")

    membership = candidate.validate_membership_rows(pq.read_table(membership_path).to_pylist())
    source = pq.read_table(source_path).to_pylist()
    labels = pq.read_table(labels_path).to_pylist()
    mapping = pq.read_table(mapping_path).to_pylist()
    by_membership = {row["opaque_id"]: row for row in membership}
    by_source = {row["sample_id"]: row for row in source}
    by_label = {row["opaque_id"]: row for row in labels}
    mapping_by_sample = {row["sample_id"]: row for row in mapping}
    if (
        len(by_source) != candidate.CALIBRATION_ROWS
        or len(by_label) != candidate.CALIBRATION_ROWS
        or len(mapping_by_sample) != candidate.CALIBRATION_ROWS
        or set(by_membership) != set(by_source)
        or set(by_membership) != set(mapping_by_sample)
        or {row["opaque_id"] for row in mapping} != set(by_label)
        or sorted(row["packet_order"] for row in mapping) != list(range(candidate.CALIBRATION_ROWS))
    ):
        raise RuntimeError("calibration source, membership and labels are not one-to-one")
    from reddit_china_stance.semantic_ontology_v2 import validate_v2_label

    for item_id, membership_row in by_membership.items():
        source_row = by_source[item_id]
        mapping_row = mapping_by_sample[item_id]
        label_row = by_label[mapping_row["opaque_id"]]
        if any(
            source_row[source_field] != membership_row[membership_field]
            for source_field, membership_field in (
                ("thread_id", "thread_id"),
                ("target_text", "target_text"),
                ("parent_context", "parent_context"),
                ("submission_context", "submission_context"),
            )
        ):
            raise RuntimeError("teacher source payload differs from frozen membership")
        if mapping_row["thread_id"] != membership_row["thread_id"]:
            raise RuntimeError("teacher private mapping thread differs from membership")
        label = validate_v2_label(json.loads(label_row["label_json"]))
        if (
            label["codability"] != label_row["codability"]
            or label["relevance"] != label_row["relevance"]
        ):
            raise RuntimeError("teacher label projection differs from label JSON")
    return {
        "root": root,
        "membership_path": membership_path,
        "source_path": source_path,
        "labels_path": labels_path,
        "membership": membership,
        "source": source,
        "labels": labels,
        "mapping": mapping,
        "cohort": cohort,
        "sample_receipt": sample_receipt,
        "teacher_receipt": teacher_receipt,
    }


def _calibration_run_id(source_bundle_id: str) -> str:
    return candidate.canonical_sha256(
        {
            "kind": "modernbert-probability-random-calibration-execution-v1",
            "source_bundle_id": source_bundle_id,
            "sample_receipt_id": SAMPLE_RECEIPT_ID,
            "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
            "teacher_run_id": TEACHER_RUN_ID,
            "teacher_label_sha256": TEACHER_LABEL_SHA256,
            "teacher_receipt_sha256": TEACHER_RECEIPT_SHA256,
            "gpu_type": GPU_TYPE,
            "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
            "throughput_rows": THROUGHPUT_ROWS,
            "locked_test_rows": 0,
            "corpus_rows": 0,
        }
    )


def _publication_descriptor(
    staging_path: Path,
    *,
    published_path: Path,
    row_count: int | None = None,
) -> dict[str, Any]:
    result = {
        "relative_path": published_path.relative_to(VOLUME_PATH).as_posix(),
        "sha256": _file_sha256(staging_path),
        "bytes": staging_path.stat().st_size,
    }
    if row_count is not None:
        result["row_count"] = row_count
    return result


def _validate_existing_output(output_root: Path, run_id: str) -> dict[str, Any]:
    receipt = _read_json(output_root / "receipt.json", where="calibration run receipt")
    if (
        receipt.get("calibration_run_id") != run_id
        or receipt.get("receipt_id")
        != candidate.canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )
        or receipt.get("locked_test_rows_accessed") != 0
        or receipt.get("corpus_rows_accessed") != 0
    ):
        raise RuntimeError("existing calibration run receipt drifted")
    for key in (
        "calibration_artifact",
        "aggregate_validation_artifact",
        "throughput_artifact",
        "private_logits_artifact",
    ):
        descriptor = receipt.get(key)
        if not isinstance(descriptor, Mapping):
            raise RuntimeError(f"existing {key} descriptor is missing")
        _validate_file(
            VOLUME_PATH / descriptor["relative_path"],
            sha256=descriptor["sha256"],
            size=descriptor["bytes"],
            where=f"existing {key}",
            parquet_rows=descriptor.get("row_count") if key == "private_logits_artifact" else None,
        )
    assert_metadata_only(receipt, where="existing probability-random calibration receipt")
    return receipt


@app.function(image=image, cpu=2, memory=6_144, volumes={str(VOLUME_PATH): volume})
def preflight(source_bundle: dict[str, Any]) -> dict[str, Any]:
    import gc

    bundle = _validate_runtime_policy(source_bundle)
    volume.reload()
    validated = _validate_inputs(volume_root=VOLUME_PATH)
    payload_count = 0
    for spec in candidate.CHECKPOINT_SPECS:
        payload, _ = _checkpoint_payload(
            VOLUME_PATH / candidate.checkpoint_relative_path(spec),
            spec=spec,
            source_bundle_id=bundle["source_bundle_id"],
        )
        del payload
        gc.collect()
        payload_count += 1
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-calibration-preflight-v1",
        "calibration_run_id": _calibration_run_id(bundle["source_bundle_id"]),
        "source_bundle_id": bundle["source_bundle_id"],
        "sample_receipt_id": validated["sample_receipt"]["receipt_id"],
        "checkpoint_cohort_id": validated["cohort"]["cohort_id"],
        "teacher_run_id": validated["teacher_receipt"]["run_id"],
        "sample_rows": candidate.CALIBRATION_ROWS,
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "checkpoint_payloads_validated": payload_count,
        "gpu_type": GPU_TYPE,
        "gpu_fallback_allowed": False,
        "retry_authorised": False,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
        "status": "ready",
    }
    result = {**body, "preflight_id": candidate.canonical_sha256(body)}
    assert_metadata_only(result, where="probability-random calibration preflight")
    return result


def _checkpoint_payload(
    path: Path, *, spec: candidate.CheckpointSpec, source_bundle_id: str
) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "kind": "modernbert-acquisition-checkpoint-v1",
        "experiment_run_id": candidate.ACQUISITION_TRAINING_RUN_ID,
        "phase_run_id": candidate.ACQUISITION_TRAINING_PHASE_ID,
        "trial_id": spec.trial_id,
        "arm": "random",
        "component": spec.component,
        "optimiser_seed": spec.seed,
        "selected_epoch": spec.selected_epoch,
        "training_frame_sha256": candidate.RANDOM_TRAINING_FRAME_SHA256,
        "model_id": "answerdotai/ModernBERT-large",
        "model_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
    }
    mismatches = (
        list(expected)
        if not isinstance(payload, Mapping)
        else [key for key, value in expected.items() if payload.get(key) != value]
    )
    state_valid = isinstance(payload, Mapping) and isinstance(
        payload.get("model_state_dict"), Mapping
    )
    if mismatches or not state_valid:
        if not state_valid:
            mismatches.append("model_state_dict")
        raise RuntimeError(
            f"checkpoint payload drifted for {spec.component} seed {spec.seed}: "
            f"{sorted(set(mismatches))}"
        )
    provenance = {
        "component": spec.component,
        "seed": spec.seed,
        "selected_epoch": spec.selected_epoch,
        "trial_id": spec.trial_id,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "training_source_bundle_sha256": payload.get("source_bundle_sha256"),
        "calibration_source_bundle_sha256": source_bundle_id,
    }
    return payload, provenance


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=36_864,
    timeout=GPU_TIMEOUT_SECONDS,
    volumes={str(VOLUME_PATH): volume},
    max_containers=1,
)
def calibrate_and_benchmark(source_bundle: dict[str, Any]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader

    from reddit_china_stance.modernbert_factorised_training import (
        FactorisedDynamicPaddingCollator,
        FactorisedOptimisationConfig,
        collect_development_logits,
        create_component_model,
        load_pinned_tokenizer,
        tokenise_factorised_record,
    )

    bundle = _validate_runtime_policy(source_bundle)
    if not torch.cuda.is_available():
        raise RuntimeError("the registered calibration job requires the requested L4")
    volume.reload()
    inputs = _validate_inputs(volume_root=VOLUME_PATH)
    run_id = _calibration_run_id(bundle["source_bundle_id"])
    output_root = inputs["root"] / f"calibration={run_id}"
    if output_root.exists():
        return _validate_existing_output(output_root, run_id)
    staging = output_root.parent / f".publishing-calibration={run_id}"
    if staging.exists():
        raise FileExistsError("stale calibration staging output requires reconciliation")
    staging.mkdir(parents=True)
    (staging / "private").mkdir()

    labels_by_opaque_id = {row["opaque_id"]: row for row in inputs["labels"]}
    label_opaque_id_by_sample = {row["sample_id"]: row["opaque_id"] for row in inputs["mapping"]}
    joined = [
        {
            "item_id": row["opaque_id"],
            "row": row,
            "label": json.loads(
                labels_by_opaque_id[label_opaque_id_by_sample[row["opaque_id"]]]["label_json"]
            ),
            "inclusion_probability": row["inclusion_probability"],
        }
        for row in inputs["membership"]
    ]
    tokenizer = load_pinned_tokenizer()
    features = [
        tokenise_factorised_record(
            tokenizer,
            item_id=item["item_id"],
            row=item["row"],
            label=item["label"],
        )
        for item in joined
    ]
    throughput_ids = {
        item["item_id"]
        for item in sorted(
            joined,
            key=lambda item: candidate.canonical_sha256(
                {"seed": THROUGHPUT_SEED, "item_id": item["item_id"]}
            ),
        )[:THROUGHPUT_ROWS]
    }
    throughput_joined = [item for item in joined if item["item_id"] in throughput_ids]
    tokenisation_started = time.monotonic()
    throughput_features = [
        tokenise_factorised_record(
            tokenizer,
            item_id=item["item_id"],
            row=item["row"],
            label=item["label"],
        )
        for item in throughput_joined
    ]
    throughput_tokenisation_seconds = time.monotonic() - tokenisation_started

    optimisation = FactorisedOptimisationConfig()
    per_component: dict[str, list[list[dict[str, Any]]]] = {
        component: [] for component in candidate.COMPONENTS
    }
    throughput_model_seconds = 0.0
    model_load_seconds = 0.0
    checkpoint_provenance = []
    peak_gpu_bytes = 0
    for spec in candidate.CHECKPOINT_SPECS:
        checkpoint_path = VOLUME_PATH / candidate.checkpoint_relative_path(spec)
        load_started = time.monotonic()
        payload, provenance = _checkpoint_payload(
            checkpoint_path,
            spec=spec,
            source_bundle_id=bundle["source_bundle_id"],
        )
        model = create_component_model(component=spec.component, config=optimisation)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
        model.to("cuda")
        torch.cuda.synchronize()
        model_load_seconds += time.monotonic() - load_started
        collator = FactorisedDynamicPaddingCollator(tokenizer, component=spec.component)
        dataloader = DataLoader(
            features,
            batch_size=optimisation.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        logits = collect_development_logits(
            model,
            dataloader,
            component=spec.component,
            device="cuda",
            use_bf16=True,
        )
        per_component[spec.component].append(logits)
        benchmark_loader = DataLoader(
            throughput_features,
            batch_size=optimisation.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        torch.cuda.synchronize()
        benchmark_started = time.monotonic()
        benchmark_logits = collect_development_logits(
            model,
            benchmark_loader,
            component=spec.component,
            device="cuda",
            use_bf16=True,
        )
        torch.cuda.synchronize()
        throughput_model_seconds += time.monotonic() - benchmark_started
        if len(benchmark_logits) != THROUGHPUT_ROWS:
            raise RuntimeError("throughput checkpoint output row count drifted")
        peak_gpu_bytes = max(peak_gpu_bytes, int(torch.cuda.max_memory_allocated()))
        checkpoint_provenance.append(provenance)
        del benchmark_logits, dataloader, benchmark_loader, model
        torch.cuda.empty_cache()

    relevance = calibration.mean_aligned_logits(per_component["relevance"], component="relevance")
    target_stance = calibration.mean_aligned_logits(
        per_component["target_stance_b4"], component="target_stance_b4"
    )
    relevance_by_id = {row["item_id"]: row for row in relevance}
    target_by_id = {row["item_id"]: row for row in target_stance}
    calibration_rows = [
        {
            "item_id": item["item_id"],
            "label": item["label"],
            "inclusion_probability": item["inclusion_probability"],
            "relevance_logit": relevance_by_id[item["item_id"]]["relevance_logit"],
            "target_presence_logits": target_by_id[item["item_id"]]["target_presence_logits"],
            "stance_logits": target_by_id[item["item_id"]]["stance_logits"],
        }
        for item in joined
    ]
    bindings = {
        "source_bundle_id": bundle["source_bundle_id"],
        "sample_run_id": SAMPLE_RUN_ID,
        "sample_receipt_id": SAMPLE_RECEIPT_ID,
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "teacher_run_id": TEACHER_RUN_ID,
        "teacher_label_sha256": TEACHER_LABEL_SHA256,
        "teacher_receipt_sha256": TEACHER_RECEIPT_SHA256,
    }
    fitted = calibration.fit_weighted_calibration(calibration_rows, bindings=bindings)
    aggregate = calibration.build_aggregate_validation(
        calibration_rows, inputs["membership"], fitted
    )

    throughput_decode_started = time.monotonic()
    for row in calibration_rows:
        if row["item_id"] in throughput_ids:
            calibration.calibrated_probabilities(row, fitted)
    throughput_decode_seconds = time.monotonic() - throughput_decode_started
    throughput_total_seconds = (
        throughput_tokenisation_seconds + throughput_model_seconds + throughput_decode_seconds
    )
    throughput = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-throughput-benchmark-v1",
        "calibration_run_id": run_id,
        "representative_rows": THROUGHPUT_ROWS,
        "subset_seed_sha256": hashlib.sha256(THROUGHPUT_SEED.encode()).hexdigest(),
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "forward_pass_rows": THROUGHPUT_ROWS * len(candidate.CHECKPOINT_SPECS),
        "ensemble_execution": "six-checkpoint-sequential-forward-passes",
        "model_loading_in_steady_state": False,
        "tokenisation_seconds": throughput_tokenisation_seconds,
        "six_checkpoint_model_seconds": throughput_model_seconds,
        "calibration_decode_seconds": throughput_decode_seconds,
        "steady_state_end_to_end_seconds": throughput_total_seconds,
        "steady_state_rows_per_second": THROUGHPUT_ROWS / throughput_total_seconds,
        "one_time_model_load_seconds": model_load_seconds,
        "cold_end_to_end_seconds": throughput_total_seconds + model_load_seconds,
        "cold_rows_per_second": THROUGHPUT_ROWS / (throughput_total_seconds + model_load_seconds),
        "gpu_type": GPU_TYPE,
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_bytes": peak_gpu_bytes,
        "batch_size": optimisation.per_device_batch_size,
        "precision": "bf16-autocast",
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
    }
    throughput = {
        **throughput,
        "throughput_id": candidate.canonical_sha256(throughput),
    }
    assert_metadata_only(throughput, where="probability-random throughput benchmark")

    private_logits_path = staging / "private" / "ensemble-logits.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "opaque_id": row["item_id"],
                    "relevance_logit": row["relevance_logit"],
                    "target_presence_logits": row["target_presence_logits"],
                    "stance_logits": row["stance_logits"],
                }
                for row in calibration_rows
            ]
        ),
        private_logits_path,
        compression="zstd",
    )
    _write_immutable_json(staging / "calibration.json", fitted)
    _write_immutable_json(staging / "aggregate-validation.json", aggregate)
    _write_immutable_json(staging / "throughput.json", throughput)
    _write_immutable_json(staging / "source-bundle.json", bundle)
    _write_immutable_json(
        staging / "checkpoint-provenance.json", {"members": checkpoint_provenance}
    )

    final_private_logits = output_root / "private" / private_logits_path.name
    calibration_path = staging / "calibration.json"
    aggregate_path = staging / "aggregate-validation.json"
    throughput_path = staging / "throughput.json"
    receipt_body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-calibration-run-receipt-v1",
        "calibration_run_id": run_id,
        "source_bundle_id": bundle["source_bundle_id"],
        "sample_receipt_id": SAMPLE_RECEIPT_ID,
        "checkpoint_cohort_id": CHECKPOINT_COHORT_ID,
        "teacher_run_id": TEACHER_RUN_ID,
        "teacher_label_sha256": TEACHER_LABEL_SHA256,
        "checkpoint_count": len(candidate.CHECKPOINT_SPECS),
        "sample_rows": candidate.CALIBRATION_ROWS,
        "calibration_id": fitted["calibration_id"],
        "validation_id": aggregate["validation_id"],
        "throughput_id": throughput["throughput_id"],
        "calibration_artifact": None,
        "aggregate_validation_artifact": None,
        "throughput_artifact": None,
        "private_logits_artifact": None,
        "locked_test_rows_accessed": 0,
        "corpus_rows_accessed": 0,
        "status": "complete",
        "evidence_boundary": (
            "model-assisted silver in-frame calibration and aggregate diagnostics; "
            "not human-gold or independent validation"
        ),
    }
    receipt_body["calibration_artifact"] = _publication_descriptor(
        calibration_path, published_path=output_root / calibration_path.name
    )
    receipt_body["aggregate_validation_artifact"] = _publication_descriptor(
        aggregate_path, published_path=output_root / aggregate_path.name
    )
    receipt_body["throughput_artifact"] = _publication_descriptor(
        throughput_path, published_path=output_root / throughput_path.name
    )
    receipt_body["private_logits_artifact"] = _publication_descriptor(
        private_logits_path,
        published_path=final_private_logits,
        row_count=candidate.CALIBRATION_ROWS,
    )
    receipt = {
        **receipt_body,
        "receipt_id": candidate.canonical_sha256(receipt_body),
    }
    assert_metadata_only(receipt, where="probability-random calibration receipt")
    _write_immutable_json(staging / "receipt.json", receipt)
    staging.replace(output_root)
    volume.commit()
    return receipt


@app.function(image=image, cpu=1, memory=1_024, volumes={str(VOLUME_PATH): volume})
def inspect_run(source_bundle: dict[str, Any], run_id: str) -> dict[str, Any]:
    bundle = _validate_runtime_policy(source_bundle)
    if run_id != _calibration_run_id(bundle["source_bundle_id"]):
        raise RuntimeError("requested calibration run ID differs from current source bundle")
    volume.reload()
    return _validate_existing_output(
        _sample_root(volume_root=VOLUME_PATH) / f"calibration={run_id}", run_id
    )


@app.local_entrypoint()
def main(action: str = "preflight", run_id: str = "") -> None:
    bundle = build_source_bundle()
    if action == "preflight":
        result = preflight.remote(bundle)
    elif action == "run":
        result = calibrate_and_benchmark.remote(bundle)
    elif action == "inspect":
        if not run_id:
            raise ValueError("inspect requires --run-id")
        result = inspect_run.remote(bundle, run_id)
    else:
        raise ValueError("action must be preflight, run or inspect")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


__all__ = [
    "APP_NAME",
    "REQUIRED_SOURCE_FILES",
    "build_source_bundle",
    "calibrate_and_benchmark",
    "inspect_run",
    "preflight",
    "validate_source_bundle",
]
