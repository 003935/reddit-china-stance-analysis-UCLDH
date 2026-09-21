"""Prepare the fresh calibration frame for the retained probability-random ensemble.

This launcher performs one CPU-only immutable operation. It validates the exact
acquisition closeout and all six retained checkpoints, excludes both consumed
acquisition arms by row, thread and near-duplicate identity, and draws the
registered 600-row stratified probability sample. It does not train, label,
touch the locked test, or run corpus inference.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-modernbert-probability-random-candidate-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path(candidate.NAMESPACE)

ACQUISITION_PREFIX = Path("student-modernbert-acquisition-v1")
TRAINING_PREFIX = Path("student-modernbert-acquisition-training-v1")
PREPARED_RECEIPT_ID = (
    "7c3e41f74fee2701535dd87b7237d0f317cb0095ef63208e51542ff70be1ec10"
)
ACQUISITION_RECEIPT_ID = (
    "d6c22840801c243e45c3bb1d0236f3129c67ade595323485254befe11b66f27f"
)
# The persisted historical receipt ID was minted from an earlier projection.
# Bind both it and the canonical digest of the final immutable receipt body.
ACQUISITION_RECEIPT_BODY_SHA256 = (
    "e35907ba3fc230d540eef589a2dad7e23107a67764e744d91eac3f06643d6419"
)
ACQUISITION_POLICY_SHA256 = (
    "fecd53f55e6c2b4e01bccb7d7beb72cc6acef200b7b5ca3a4afd1acba76b1221"
)
PREPARED_FRAME_BYTES = 295_349_714
ACQUISITION_LEDGER_BYTES = 1_059_857
PREPARE_TIMEOUT_SECONDS = 15 * 60


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
REQUIRED_SOURCE_FILES = (
    POLICY_REPO_PATH,
    "src/reddit_china_stance/modal_modernbert_probability_random_candidate_v1.py",
    "src/reddit_china_stance/modernbert_probability_random_candidate_v1.py",
    "src/reddit_china_stance/privacy.py",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME,
    environment_name=ENVIRONMENT_NAME,
    create_if_missing=False,
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "duckdb==1.4.4",
        "pyarrow==25.0.1",
        "pydantic==2.13.4",
    )
    .add_local_python_source("reddit_china_stance")
    .add_local_file(POLICY_REPO_PATH, remote_path=POLICY_RUNTIME_PATH)
)


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    files = value.get("files")
    if (
        set(value)
        != {"schema_version", "kind", "files", "code_sha256", "source_bundle_id"}
        or value.get("schema_version") != "1.0.0"
        or value.get("kind") != "listed-source-bundle-v1"
        or not isinstance(files, Mapping)
        or not files
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not candidate.is_sha256(digest)
            for path, digest in files.items()
        )
    ):
        raise ValueError("candidate source bundle schema drifted")
    clean_files = dict(sorted(files.items()))
    body = {
        "schema_version": "1.0.0",
        "kind": "listed-source-bundle-v1",
        "files": clean_files,
        "code_sha256": candidate.canonical_sha256(clean_files),
    }
    expected = {**body, "source_bundle_id": candidate.canonical_sha256(body)}
    if dict(value) != expected:
        raise RuntimeError("candidate source bundle digest drifted")
    return expected


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
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


def _descriptor(path: Path, *, root: Path, row_count: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "row_count": row_count,
    }


def _validate_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    where: str,
    parquet_rows: int | None = None,
) -> None:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{where} is missing or has the wrong size")
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{where} SHA-256 drifted")
    if parquet_rows is not None:
        import pyarrow.parquet as pq

        if pq.ParquetFile(path).metadata.num_rows != parquet_rows:
            raise RuntimeError(f"{where} row count drifted")


def _prepared_root(*, volume_root: Path = VOLUME_PATH) -> Path:
    return (
        volume_root
        / ACQUISITION_PREFIX
        / f"run={candidate.ELIGIBLE_PREPARED_RUN_ID}"
        / "prepared"
    )


def _acquisition_root(*, volume_root: Path = VOLUME_PATH) -> Path:
    return (
        volume_root
        / ACQUISITION_PREFIX
        / f"run={candidate.ELIGIBLE_PREPARED_RUN_ID}"
        / "acquisition"
    )


def _training_root(*, volume_root: Path = VOLUME_PATH) -> Path:
    return (
        volume_root
        / TRAINING_PREFIX
        / f"run={candidate.ACQUISITION_TRAINING_RUN_ID}"
    )


def _validate_prepared_inputs(*, volume_root: Path) -> Path:
    root = _prepared_root(volume_root=volume_root)
    receipt = _read_json(root / "receipt.json", where="corrected eligible-frame receipt")
    descriptor = receipt.get("prepared_frame")
    if (
        receipt.get("receipt_id") != PREPARED_RECEIPT_ID
        or candidate.canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )
        != PREPARED_RECEIPT_ID
        or receipt.get("prepared_run_id") != candidate.ELIGIBLE_PREPARED_RUN_ID
        or receipt.get("eligible_population_rows") != candidate.ELIGIBLE_POPULATION_ROWS
        or receipt.get("eligible_frame_sha256") != candidate.ELIGIBLE_PROJECTION_SHA256
        or receipt.get("policy_sha256") != ACQUISITION_POLICY_SHA256
        or receipt.get("locked_test_rows_accessed") != 0
        or not isinstance(descriptor, Mapping)
        or descriptor.get("relative_path") != "eligible-frame.parquet"
        or descriptor.get("sha256") != candidate.ELIGIBLE_FRAME_SHA256
        or descriptor.get("bytes") != PREPARED_FRAME_BYTES
        or descriptor.get("row_count") != candidate.ELIGIBLE_POPULATION_ROWS
    ):
        raise RuntimeError("corrected eligible-frame receipt drifted")
    frame = root / "eligible-frame.parquet"
    _validate_file(
        frame,
        expected_sha256=candidate.ELIGIBLE_FRAME_SHA256,
        expected_bytes=PREPARED_FRAME_BYTES,
        parquet_rows=candidate.ELIGIBLE_POPULATION_ROWS,
        where="corrected eligible frame",
    )
    return frame


def _validate_acquisition_inputs(*, volume_root: Path) -> dict[str, Any]:
    root = _acquisition_root(volume_root=volume_root)
    receipt = _read_json(root / "receipt.json", where="acquisition receipt")
    artifact = (
        receipt.get("private_artifacts", {}).get("acquisition-ledger.json")
        if isinstance(receipt.get("private_artifacts"), Mapping)
        else None
    )
    if (
        receipt.get("receipt_id") != ACQUISITION_RECEIPT_ID
        or candidate.canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_id"}
        )
        != ACQUISITION_RECEIPT_BODY_SHA256
        or receipt.get("prepared_run_id") != candidate.ELIGIBLE_PREPARED_RUN_ID
        or receipt.get("ledger_id") != candidate.ACQUISITION_LEDGER_ID
        or receipt.get("policy_sha256") != ACQUISITION_POLICY_SHA256
        or receipt.get("eligible_population_rows") != candidate.ELIGIBLE_POPULATION_ROWS
        or receipt.get("probability_arm_rows") != 1_000
        or receipt.get("active_arm_rows") != 1_000
        or receipt.get("arms_thread_disjoint") is not True
        or receipt.get("arms_near_duplicate_disjoint") is not True
        or receipt.get("locked_test_rows_accessed") != 0
        or not isinstance(artifact, Mapping)
        or artifact.get("sha256") != candidate.ACQUISITION_LEDGER_FILE_SHA256
        or artifact.get("bytes") != ACQUISITION_LEDGER_BYTES
        or artifact.get("row_count") != candidate.ACQUISITION_ROWS
    ):
        raise RuntimeError("acquisition receipt drifted")
    ledger_path = root / "acquisition-ledger.json"
    _validate_file(
        ledger_path,
        expected_sha256=candidate.ACQUISITION_LEDGER_FILE_SHA256,
        expected_bytes=ACQUISITION_LEDGER_BYTES,
        where="private acquisition ledger",
    )
    ledger = _read_json(ledger_path, where="private acquisition ledger")
    if (
        ledger.get("kind") != "modernbert-acquisition-ledger-v1"
        or ledger.get("ledger_id") != candidate.ACQUISITION_LEDGER_ID
        or candidate.canonical_sha256(
            {key: value for key, value in ledger.items() if key != "ledger_id"}
        )
        != candidate.ACQUISITION_LEDGER_ID
        or ledger.get("eligible_population_rows") != candidate.ELIGIBLE_POPULATION_ROWS
    ):
        raise RuntimeError("private acquisition ledger contract drifted")
    arms = [ledger.get("probability_arm"), ledger.get("active_arm")]
    if any(not isinstance(arm, Mapping) for arm in arms):
        raise RuntimeError("acquisition ledger arms are missing")
    rows = [dict(row) for arm in arms for row in arm.get("rows", [])]
    if len(rows) != candidate.ACQUISITION_ROWS:
        raise RuntimeError("acquisition exclusion row count drifted")
    for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id"):
        values = [row.get(field) for row in rows]
        if (
            any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise RuntimeError(f"acquisition exclusion {field} contract drifted")
    return {"receipt": receipt, "ledger": ledger, "rows": rows}


def _validate_checkpoint_cohort(*, volume_root: Path) -> dict[str, Any]:
    training_root = _training_root(volume_root=volume_root)
    closeout = _read_json(
        training_root / "closeout" / "acquisition-gate.json",
        where="acquisition closeout",
    )
    candidate.validate_parent_closeout(closeout)
    receipts = []
    for spec in candidate.CHECKPOINT_SPECS:
        receipt_path = volume_root / candidate.checkpoint_receipt_relative_path(spec)
        receipts.append(_read_json(receipt_path, where="retained checkpoint receipt"))
    cohort = candidate.freeze_checkpoint_cohort(receipts)
    for member in cohort["members"]:
        descriptor = member["checkpoint"]
        _validate_file(
            volume_root / descriptor["relative_path"],
            expected_sha256=descriptor["sha256"],
            expected_bytes=descriptor["bytes"],
            where=f"retained {member['component']} seed {member['seed']} checkpoint",
        )
    assert_metadata_only(cohort, where="retained checkpoint cohort")
    return cohort


def _selection_sql(frame_path: Path) -> str:
    escaped = str(frame_path).replace("'", "''")
    return f"""
        SELECT
            record_id AS opaque_id,
            thread_id,
            near_duplicate_cluster_id,
            subreddit,
            CAST(year AS INTEGER) AS year,
            content_type,
            retrieval_mode,
            target_text,
            parent_context,
            submission_context
        FROM read_parquet('{escaped}') AS source
        WHERE record_id NOT IN (SELECT opaque_id FROM acquisition_exclusions)
          AND thread_id NOT IN (SELECT thread_id FROM acquisition_exclusions)
          AND near_duplicate_cluster_id NOT IN (
              SELECT near_duplicate_cluster_id FROM acquisition_exclusions
          )
    """


def select_calibration_rows(
    *,
    frame_path: Path,
    acquisition_rows: Sequence[Mapping[str, Any]],
    expected_remaining_rows: int = candidate.REMAINING_POPULATION_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the exact private sample while keeping the full corpus in DuckDB."""

    import duckdb
    import pyarrow as pa

    exclusions = [
        {
            key: row[key]
            for key in ("opaque_id", "thread_id", "near_duplicate_cluster_id")
        }
        for row in acquisition_rows
    ]
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = '12GB'")
        connection.register("acquisition_exclusions_arrow", pa.Table.from_pylist(exclusions))
        connection.execute(
            "CREATE TEMP TABLE acquisition_exclusions AS "
            "SELECT * FROM acquisition_exclusions_arrow"
        )
        connection.execute("CREATE TEMP VIEW remaining AS " + _selection_sql(frame_path))
        counts = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT opaque_id), COUNT(DISTINCT thread_id), "
            "COUNT(DISTINCT near_duplicate_cluster_id) FROM remaining"
        ).fetchone()
        if tuple(counts) != (expected_remaining_rows,) * 4:
            raise RuntimeError(
                "post-acquisition universe count or identity uniqueness drifted: "
                f"observed={tuple(counts)} expected={expected_remaining_rows}"
            )
        raw_strata = connection.execute(
            "SELECT year, subreddit, content_type, COUNT(*) AS population_rows "
            "FROM remaining GROUP BY ALL"
        ).fetchall()
        quotas = candidate.allocate_minimum_one_quotas(
            [
                {
                    "stratum": {
                        "year": int(row[0]),
                        "subreddit": row[1],
                        "content_type": row[2],
                    },
                    "population_rows": int(row[3]),
                }
                for row in raw_strata
            ]
        )
        connection.register("calibration_quotas_arrow", pa.Table.from_pylist(quotas))
        connection.execute(
            "CREATE TEMP TABLE calibration_quotas AS SELECT "
            "CAST(stratum.year AS INTEGER) AS year, stratum.subreddit AS subreddit, "
            "stratum.content_type AS content_type, population_rows, sample_rows "
            "FROM calibration_quotas_arrow"
        )
        cursor = connection.execute(
            """
            WITH ranked AS (
                SELECT
                    remaining.*,
                    quotas.population_rows,
                    quotas.sample_rows,
                    sha256(json_array(
                        ?,
                        json_object(
                            'content_type', remaining.content_type,
                            'subreddit', remaining.subreddit,
                            'year', remaining.year
                        ),
                        remaining.opaque_id
                    )) AS selection_tiebreak_sha256,
                    row_number() OVER (
                        PARTITION BY remaining.year, remaining.subreddit,
                                     remaining.content_type
                        ORDER BY selection_tiebreak_sha256
                    ) AS selection_rank
                FROM remaining
                JOIN calibration_quotas AS quotas USING (year, subreddit, content_type)
            )
            SELECT * FROM ranked
            WHERE selection_rank <= sample_rows
            ORDER BY year, subreddit, content_type, selection_rank
            """,
            [candidate.CALIBRATION_SEED],
        )
        names = [description[0] for description in cursor.description]
        selected = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()
    if len(selected) != candidate.CALIBRATION_ROWS:
        raise RuntimeError("calibration sample row count drifted")
    membership = []
    for row in selected:
        built = candidate.build_membership_row(
            row,
            population_rows=int(row["population_rows"]),
            sample_rows=int(row["sample_rows"]),
            rank=int(row["selection_rank"]),
        )
        if built["selection_tiebreak_sha256"] != row["selection_tiebreak_sha256"]:
            raise RuntimeError("DuckDB/Python calibration tiebreak disagreement")
        membership.append(built)
    clean = candidate.validate_membership_rows(membership)
    excluded = {
        field: {row[field] for row in acquisition_rows}
        for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id")
    }
    for field, values in excluded.items():
        if values & {row[field] for row in clean}:
            raise RuntimeError(f"calibration sample overlaps acquisition by {field}")
    return clean, quotas


def _validate_runtime_policy(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    bundle = validate_source_bundle(source_bundle)
    expected = bundle["files"].get(POLICY_REPO_PATH)
    path = Path(POLICY_RUNTIME_PATH)
    if (
        not isinstance(expected, str)
        or not path.is_file()
        or _file_sha256(path) != expected
    ):
        raise RuntimeError("runtime calibration policy differs from the source bundle")
    return bundle


def _run_id(*, source_bundle_id: str, cohort_id: str) -> str:
    return candidate.canonical_sha256(
        {
            "kind": "modernbert-probability-random-calibration-sample-run-v1",
            "source_bundle_id": source_bundle_id,
            "prepared_receipt_id": PREPARED_RECEIPT_ID,
            "acquisition_receipt_id": ACQUISITION_RECEIPT_ID,
            "acquisition_closeout_id": candidate.ACQUISITION_CLOSEOUT_ID,
            "checkpoint_cohort_id": cohort_id,
            "sampling_seed": candidate.CALIBRATION_SEED,
            "sample_rows": candidate.CALIBRATION_ROWS,
            "locked_test_rows": 0,
            "corpus_inference_rows": 0,
        }
    )


def _validate_existing_output(
    *, output_root: Path, run_id: str, source_bundle_id: str
) -> dict[str, Any]:
    receipt = _read_json(output_root / "receipt.json", where="calibration sample receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if (
        receipt.get("run_id") != run_id
        or receipt.get("source_bundle_sha256") != source_bundle_id
        or receipt.get("receipt_id") != candidate.canonical_sha256(body)
        or receipt.get("sample_count") != candidate.CALIBRATION_ROWS
        or receipt.get("locked_test_access_count") != 0
    ):
        raise RuntimeError("existing calibration sample publication drifted")
    for key in ("membership_artifact", "teacher_source_artifact"):
        descriptor = receipt.get(key)
        if not isinstance(descriptor, Mapping):
            raise RuntimeError("existing calibration artifact descriptor is missing")
        _validate_file(
            VOLUME_PATH / descriptor["relative_path"],
            expected_sha256=descriptor["sha256"],
            expected_bytes=descriptor["bytes"],
            parquet_rows=descriptor["row_count"],
            where=f"existing {key}",
        )
    assert_metadata_only(receipt, where="existing calibration sample receipt")
    return receipt


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    volumes={str(VOLUME_PATH): volume},
    timeout=PREPARE_TIMEOUT_SECONDS,
)
def prepare_calibration(source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Publish the sole fresh 600-row sample and frozen checkpoint cohort."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    bundle = _validate_runtime_policy(source_bundle)
    frame = _validate_prepared_inputs(volume_root=VOLUME_PATH)
    acquisition = _validate_acquisition_inputs(volume_root=VOLUME_PATH)
    cohort = _validate_checkpoint_cohort(volume_root=VOLUME_PATH)
    run_id = _run_id(
        source_bundle_id=bundle["source_bundle_id"],
        cohort_id=cohort["cohort_id"],
    )
    output_root = VOLUME_PATH / OUTPUT_PREFIX / f"run={run_id}"
    if output_root.exists():
        return _validate_existing_output(
            output_root=output_root,
            run_id=run_id,
            source_bundle_id=bundle["source_bundle_id"],
        )
    staging = output_root.parent / f".publishing-run={run_id}"
    if staging.exists():
        raise FileExistsError("stale calibration publication requires reconciliation")
    membership, _quotas = select_calibration_rows(
        frame_path=frame,
        acquisition_rows=acquisition["rows"],
    )
    staging.mkdir(parents=True)
    membership_path = staging / "calibration-membership.parquet"
    source_path = staging / "calibration-source.parquet"
    pq.write_table(pa.Table.from_pylist(membership), membership_path, compression="zstd")
    teacher_rows = [
        {
            "sample_id": row["opaque_id"],
            "thread_id": row["thread_id"],
            "target_text": row["target_text"],
            "submission_context": row["submission_context"],
            "parent_context": row["parent_context"],
        }
        for row in membership
    ]
    pq.write_table(pa.Table.from_pylist(teacher_rows), source_path, compression="zstd")
    membership_artifact = _descriptor(
        membership_path, root=VOLUME_PATH, row_count=candidate.CALIBRATION_ROWS
    )
    source_artifact = _descriptor(
        source_path, root=VOLUME_PATH, row_count=candidate.CALIBRATION_ROWS
    )
    membership_artifact["relative_path"] = (
        output_root / membership_path.name
    ).relative_to(VOLUME_PATH).as_posix()
    source_artifact["relative_path"] = (
        output_root / source_path.name
    ).relative_to(VOLUME_PATH).as_posix()
    receipt = candidate.sampling_public_receipt(
        membership_rows=membership,
        membership_artifact=membership_artifact,
        teacher_source_artifact=source_artifact,
        acquisition_ledger_sha256=candidate.ACQUISITION_LEDGER_FILE_SHA256,
        source_bundle_sha256=bundle["source_bundle_id"],
        checkpoint_cohort_id=cohort["cohort_id"],
        run_id=run_id,
    )
    _write_immutable_json(staging / "checkpoint-cohort.json", cohort)
    _write_immutable_json(staging / "source-bundle.json", bundle)
    _write_immutable_json(staging / "receipt.json", receipt)
    staging.replace(output_root)
    volume.commit()
    return receipt


@app.function(image=image, cpu=1, memory=1_024, volumes={str(VOLUME_PATH): volume})
def inspect_calibration(run_id: str, source_bundle_id: str) -> dict[str, Any]:
    return _validate_existing_output(
        output_root=VOLUME_PATH / OUTPUT_PREFIX / f"run={run_id}",
        run_id=run_id,
        source_bundle_id=source_bundle_id,
    )


def build_source_bundle(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    files = {
        relative: _file_sha256(repo_root / relative)
        for relative in sorted(REQUIRED_SOURCE_FILES)
    }
    body = {
        "schema_version": "1.0.0",
        "kind": "listed-source-bundle-v1",
        "files": files,
        "code_sha256": candidate.canonical_sha256(files),
    }
    return validate_source_bundle(
        {**body, "source_bundle_id": candidate.canonical_sha256(body)}
    )


@app.local_entrypoint()
def main(action: str = "prepare", run_id: str = "") -> None:
    bundle = build_source_bundle()
    if action == "prepare":
        result = prepare_calibration.remote(bundle)
    elif action == "inspect":
        if not run_id:
            raise ValueError("inspect requires --run-id")
        result = inspect_calibration.remote(run_id, bundle["source_bundle_id"])
    else:
        raise ValueError("action must be prepare or inspect")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


__all__ = [
    "ACQUISITION_RECEIPT_BODY_SHA256",
    "ACQUISITION_RECEIPT_ID",
    "APP_NAME",
    "PREPARED_RECEIPT_ID",
    "REQUIRED_SOURCE_FILES",
    "build_source_bundle",
    "inspect_calibration",
    "prepare_calibration",
    "select_calibration_rows",
    "validate_source_bundle",
]
