"""Modal preparation for the fresh probability-random calibration sample.

The single production action validates the corrected 910,141-row eligible
universe, excludes both completed acquisition arms by identity, thread and
near-duplicate cluster, and publishes one immutable 600-row probability sample.
It performs no model inference, teacher generation or locked-test access.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modernbert_probability_random_candidate_v1 as candidate
from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-probability-random-sample-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
OUTPUT_PREFIX = Path(candidate.NAMESPACE)

ELIGIBLE_RELATIVE_PATH = Path(
    "student-modernbert-acquisition-v1/"
    f"run={candidate.ELIGIBLE_PREPARED_RUN_ID}/prepared/eligible-frame.parquet"
)
ACQUISITION_LEDGER_RELATIVE_PATH = Path(
    "student-modernbert-acquisition-v1/"
    f"run={candidate.ELIGIBLE_PREPARED_RUN_ID}/acquisition/acquisition-ledger.json"
)
ACQUISITION_LEDGER_SHA256 = "10c2555efad8d87d7ba79a30fa6a4547c5584ad600dca79c011585aac2d7dc16"
ELIGIBLE_FRAME_BYTES = 295_349_714
RATE_CARD_CPU_USD_PER_SECOND = 0.0000131
PHASE_MAX_COST_USD = 1.0

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME,
    environment_name=ENVIRONMENT_NAME,
    create_if_missing=False,
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("duckdb==1.4.4", "pyarrow==25.0.1")
    .add_local_python_source("reddit_china_stance")
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_object(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"existing immutable output differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".new")
    if temporary.exists():
        raise RuntimeError(f"stale incomplete output exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _descriptor(path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(VOLUME_PATH).as_posix(),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "row_count": row_count,
    }


def _source_bundle() -> dict[str, Any]:
    paths = {
        "modernbert_probability_random_candidate_v1.py": Path(candidate.__file__).resolve(),
        "modal_modernbert_probability_random_sample_v1.py": Path(__file__).resolve(),
    }
    files = {name: _file_sha256(path) for name, path in sorted(paths.items())}
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-sample-source-bundle-v1",
        "files": files,
        "runtime_dependencies": {"duckdb": "1.4.4", "pyarrow": "25.0.1"},
    }
    return {**body, "source_bundle_id": candidate.canonical_sha256(body)}


def _checkpoint_receipts() -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for spec in candidate.CHECKPOINT_SPECS:
        path = VOLUME_PATH / candidate.checkpoint_receipt_relative_path(spec)
        receipt = _read_object(path, where="retained checkpoint receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        if receipt.get("receipt_id") != candidate.canonical_sha256(body):
            raise RuntimeError("retained checkpoint receipt digest drifted")
        receipts.append(receipt)
    return receipts


def _authority(cohort: Mapping[str, Any], source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": candidate.SCHEMA_VERSION,
        "kind": "modernbert-probability-random-candidate-authority-v1",
        "eligible_input": {
            "relative_path": ELIGIBLE_RELATIVE_PATH.as_posix(),
            "sha256": candidate.ELIGIBLE_FRAME_SHA256,
            "bytes": ELIGIBLE_FRAME_BYTES,
            "row_count": candidate.ELIGIBLE_POPULATION_ROWS,
            "projection_sha256": candidate.ELIGIBLE_PROJECTION_SHA256,
            "prepared_run_id": candidate.ELIGIBLE_PREPARED_RUN_ID,
        },
        "acquisition_exclusions": {
            "relative_path": ACQUISITION_LEDGER_RELATIVE_PATH.as_posix(),
            "sha256": ACQUISITION_LEDGER_SHA256,
            "row_count": candidate.ACQUISITION_ROWS,
            "exclusion_fields": [
                "opaque_id",
                "thread_id",
                "near_duplicate_cluster_id",
            ],
        },
        "checkpoint_cohort": dict(cohort),
        "calibration_design": {
            "sample_count": candidate.CALIBRATION_ROWS,
            "post_exclusion_population_count": candidate.REMAINING_POPULATION_ROWS,
            "stratum_fields": list(candidate.STRATUM_FIELDS),
            "seed_sha256": candidate.canonical_sha256(candidate.CALIBRATION_SEED),
            "allocation": "minimum-one-residual-capacity-largest-remainder-srswor-v1",
            "temperature_grid": list(candidate.TEMPERATURE_GRID),
            "relevance_abstention_margin_grid": list(candidate.ABSTENTION_MARGIN_GRID),
            "target_abstention_margin_grid": list(candidate.ABSTENTION_MARGIN_GRID),
            "stance_confidence_grid": list(candidate.STANCE_CONFIDENCE_GRID),
            "full_coverage_binary_threshold": 0.5,
            "stance_representation": "B4-four-class-including-mixed",
        },
        "source_bundle": dict(source_bundle),
        "compute": {
            "cpu": 8,
            "memory_mib": 32768,
            "timeout_seconds": 3600,
            "phase_max_cost_usd": PHASE_MAX_COST_USD,
            "shared_hard_cap_usd": 200,
            "conservative_prior_spend_and_reservations_usd": 50.780908,
        },
        "evidence_boundary": {
            "selection_set_descriptive_only": True,
            "model_assisted_labels_only": True,
            "locked_test_authorised": False,
            "locked_test_access_count": 0,
            "corpus_inference_authorised": False,
        },
    }
    return {**body, "authority_id": candidate.canonical_sha256(body)}


def _load_exclusions(path: Path) -> tuple[list[dict[str, str]], str]:
    ledger = _read_object(path, where="acquisition exclusion ledger")
    expected_ledger_id = "532062f085d6ac6ed41b603b7f987157f32f45f06d9e48e58af40e4ddcfa3f74"
    if ledger.get("ledger_id") != expected_ledger_id:
        raise RuntimeError("acquisition exclusion ledger identity drifted")
    rows: list[dict[str, str]] = []
    for arm in ("probability_arm", "active_arm"):
        value = ledger.get(arm)
        raw_rows = value.get("rows") if isinstance(value, Mapping) else None
        if not isinstance(raw_rows, list) or len(raw_rows) != 1_000:
            raise RuntimeError("acquisition exclusion arm does not contain 1,000 rows")
        for row in raw_rows:
            selected = {
                field: row.get(field)
                for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id")
            }
            if any(not isinstance(item, str) or not item for item in selected.values()):
                raise RuntimeError("acquisition exclusion row is invalid")
            rows.append(selected)
    for field in ("opaque_id", "thread_id", "near_duplicate_cluster_id"):
        values = [row[field] for row in rows]
        if len(values) != candidate.ACQUISITION_ROWS or len(set(values)) != len(values):
            raise RuntimeError(f"acquisition exclusions contain duplicate {field}")
    return rows, ledger["ledger_id"]


def _prepare_selected_rows(
    *, eligible_path: Path, exclusions: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import duckdb
    import pyarrow as pa

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 8")
        connection.execute("SET memory_limit = '24GB'")
        connection.execute("SET temp_directory = '/tmp/probability-random-calibration-duckdb'")
        connection.register("excluded", pa.Table.from_pylist(exclusions))
        connection.execute(
            "CREATE TEMP VIEW eligible AS SELECT * FROM read_parquet(?)", [str(eligible_path)]
        )
        counts = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT record_id), COUNT(DISTINCT thread_id), "
            "COUNT(DISTINCT near_duplicate_cluster_id) FROM eligible"
        ).fetchone()
        if counts != (
            candidate.ELIGIBLE_POPULATION_ROWS,
            candidate.ELIGIBLE_POPULATION_ROWS,
            candidate.ELIGIBLE_POPULATION_ROWS,
            candidate.ELIGIBLE_POPULATION_ROWS,
        ):
            raise RuntimeError("eligible calibration universe identity conservation failed")
        connection.execute(
            "CREATE TEMP VIEW remaining AS SELECT e.* FROM eligible e WHERE NOT EXISTS ("
            "SELECT 1 FROM excluded x WHERE x.opaque_id=e.record_id "
            "OR x.thread_id=e.thread_id "
            "OR x.near_duplicate_cluster_id=e.near_duplicate_cluster_id)"
        )
        remaining = connection.execute("SELECT COUNT(*) FROM remaining").fetchone()[0]
        if remaining != candidate.REMAINING_POPULATION_ROWS:
            raise RuntimeError(
                "post-acquisition population drifted: "
                f"expected {candidate.REMAINING_POPULATION_ROWS}, got {remaining}"
            )
        raw_counts = connection.execute(
            "SELECT subreddit, year, content_type, retrieval_mode, COUNT(*) "
            "FROM remaining GROUP BY ALL"
        ).fetchall()
        strata = candidate.allocate_minimum_one_quotas(
            [
                {
                    "stratum": {
                        "subreddit": row[0],
                        "year": int(row[1]),
                        "content_type": row[2],
                        "retrieval_mode": row[3],
                    },
                    "population_rows": int(row[4]),
                }
                for row in raw_counts
            ]
        )
        quota_rows = [
            {
                **row["stratum"],
                "population_rows": row["population_rows"],
                "sample_rows": row["sample_rows"],
            }
            for row in strata
        ]
        connection.register("quotas", pa.Table.from_pylist(quota_rows))

        def selection_hash(
            record_id: str,
            subreddit: str,
            year: int,
            content_type: str,
            retrieval_mode: str,
        ) -> str:
            return candidate.selection_tiebreak_sha256(
                record_id,
                stratum={
                    "subreddit": subreddit,
                    "year": int(year),
                    "content_type": content_type,
                    "retrieval_mode": retrieval_mode,
                },
            )

        connection.create_function(
            "selection_hash",
            selection_hash,
            ["VARCHAR", "VARCHAR", "INTEGER", "VARCHAR", "VARCHAR"],
            "VARCHAR",
        )
        selected = (
            connection.execute(
                """
            WITH hashed AS (
              SELECT r.*,
                     selection_hash(record_id, subreddit, year, content_type,
                                    retrieval_mode) AS selection_tiebreak_sha256
              FROM remaining r
            ), ranked AS (
              SELECT h.*, q.population_rows, q.sample_rows,
                     row_number() OVER (
                       PARTITION BY h.subreddit, h.year, h.content_type, h.retrieval_mode
                       ORDER BY h.selection_tiebreak_sha256, h.record_id
                     ) AS selection_rank
              FROM hashed h JOIN quotas q USING(subreddit, year, content_type, retrieval_mode)
            )
            SELECT record_id AS opaque_id, thread_id, near_duplicate_cluster_id,
                   subreddit, CAST(year AS INTEGER) AS year, content_type, retrieval_mode,
                   target_text, parent_context, submission_context,
                   population_rows, sample_rows, CAST(selection_rank AS INTEGER) AS selection_rank
            FROM ranked WHERE selection_rank <= sample_rows
            ORDER BY subreddit, year, content_type, retrieval_mode,
                     selection_tiebreak_sha256, record_id
            """
            )
            .fetch_arrow_table()
            .to_pylist()
        )
    finally:
        connection.close()
    if len(selected) != candidate.CALIBRATION_ROWS:
        raise RuntimeError("calibration selection did not conserve 600 rows")
    members = [
        candidate.build_membership_row(
            row,
            population_rows=row["population_rows"],
            sample_rows=row["sample_rows"],
            rank=row["selection_rank"],
        )
        for row in selected
    ]
    return candidate.validate_membership_rows(members), strata


def _sample_output_root(authority_id: str) -> Path:
    return VOLUME_PATH / OUTPUT_PREFIX / f"run={authority_id}" / "sample"


def _validate_published_sample(root: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    authority_path = root / "authority.json"
    receipt_path = root / "receipt.json"
    if not all(path.is_file() for path in (authority_path, receipt_path)):
        raise RuntimeError("calibration sample output is incomplete")
    if _read_object(authority_path, where="published authority") != dict(authority):
        raise RuntimeError("published calibration authority drifted")
    receipt = _read_object(receipt_path, where="published calibration receipt")
    membership_descriptor = receipt.get("membership_artifact")
    teacher_descriptor = receipt.get("teacher_source_artifact")
    if not isinstance(membership_descriptor, Mapping) or not isinstance(
        teacher_descriptor, Mapping
    ):
        raise RuntimeError("calibration sample receipt lacks private descriptors")
    membership_path = VOLUME_PATH / str(membership_descriptor["relative_path"])
    teacher_path = VOLUME_PATH / str(teacher_descriptor["relative_path"])
    for path, descriptor in (
        (membership_path, membership_descriptor),
        (teacher_path, teacher_descriptor),
    ):
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
            or _file_sha256(path) != descriptor.get("sha256")
            or descriptor.get("row_count") != candidate.CALIBRATION_ROWS
        ):
            raise RuntimeError("published calibration private artefact drifted")
    members = candidate.validate_membership_rows(pq.read_table(membership_path).to_pylist())
    teacher = pq.read_table(teacher_path).to_pylist()
    if (
        len(teacher) != candidate.CALIBRATION_ROWS
        or set(teacher[0])
        != {"sample_id", "thread_id", "target_text", "submission_context", "parent_context"}
        or [row["sample_id"] for row in teacher] != [row["opaque_id"] for row in members]
        or [row["thread_id"] for row in teacher] != [row["thread_id"] for row in members]
    ):
        raise RuntimeError("teacher source differs from calibration membership")
    expected = candidate.sampling_public_receipt(
        membership_rows=members,
        membership_artifact=membership_descriptor,
        teacher_source_artifact=teacher_descriptor,
        acquisition_ledger_sha256=ACQUISITION_LEDGER_SHA256,
        source_bundle_sha256=authority["source_bundle"]["source_bundle_id"],
    )
    if receipt != expected:
        raise RuntimeError("published calibration sample receipt is not reproducible")
    return receipt


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    timeout=3_600,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_sample() -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    volume.reload()
    eligible_path = VOLUME_PATH / ELIGIBLE_RELATIVE_PATH
    ledger_path = VOLUME_PATH / ACQUISITION_LEDGER_RELATIVE_PATH
    if (
        not eligible_path.is_file()
        or eligible_path.stat().st_size != ELIGIBLE_FRAME_BYTES
        or _file_sha256(eligible_path) != candidate.ELIGIBLE_FRAME_SHA256
        or not ledger_path.is_file()
        or _file_sha256(ledger_path) != ACQUISITION_LEDGER_SHA256
    ):
        raise RuntimeError("calibration sampling input is missing or corrupt")
    source_bundle = _source_bundle()
    cohort = candidate.freeze_checkpoint_cohort(_checkpoint_receipts())
    authority = _authority(cohort, source_bundle)
    root = _sample_output_root(authority["authority_id"])
    if root.exists():
        return _validate_published_sample(root, authority)
    exclusions, _ = _load_exclusions(ledger_path)
    members, _ = _prepare_selected_rows(eligible_path=eligible_path, exclusions=exclusions)
    excluded_ids = {row["opaque_id"] for row in exclusions}
    excluded_threads = {row["thread_id"] for row in exclusions}
    excluded_clusters = {row["near_duplicate_cluster_id"] for row in exclusions}
    if (
        {row["opaque_id"] for row in members} & excluded_ids
        or {row["thread_id"] for row in members} & excluded_threads
        or {row["near_duplicate_cluster_id"] for row in members} & excluded_clusters
    ):
        raise RuntimeError("fresh calibration sample overlaps an acquisition arm")
    staging = root.parent / ".sample.incomplete"
    if staging.exists():
        raise RuntimeError("stale calibration sample staging output exists")
    staging.mkdir(parents=True)
    membership_path = staging / "membership.parquet"
    teacher_path = staging / "teacher-source.parquet"
    pq.write_table(pa.Table.from_pylist(members), membership_path, compression="zstd")
    teacher_rows = [
        {
            "sample_id": row["opaque_id"],
            "thread_id": row["thread_id"],
            "target_text": row["target_text"],
            "submission_context": row["submission_context"],
            "parent_context": row["parent_context"],
        }
        for row in members
    ]
    pq.write_table(pa.Table.from_pylist(teacher_rows), teacher_path, compression="zstd")
    # Descriptors are rooted at the final immutable namespace, not staging.
    final_membership = root / membership_path.name
    final_teacher = root / teacher_path.name
    membership_descriptor = {
        **_descriptor(membership_path, row_count=len(members)),
        "relative_path": final_membership.relative_to(VOLUME_PATH).as_posix(),
    }
    teacher_descriptor = {
        **_descriptor(teacher_path, row_count=len(teacher_rows)),
        "relative_path": final_teacher.relative_to(VOLUME_PATH).as_posix(),
    }
    receipt = candidate.sampling_public_receipt(
        membership_rows=members,
        membership_artifact=membership_descriptor,
        teacher_source_artifact=teacher_descriptor,
        acquisition_ledger_sha256=ACQUISITION_LEDGER_SHA256,
        source_bundle_sha256=source_bundle["source_bundle_id"],
    )
    _write_exclusive(staging / "authority.json", authority)
    _write_exclusive(staging / "receipt.json", receipt)
    os.replace(staging, root)
    volume.commit()
    validated = _validate_published_sample(root, authority)
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "status": "complete",
                "sample_count": validated["sample_count"],
                "observed_stratum_count": validated["observed_stratum_count"],
                "wall_seconds": round(elapsed, 3),
                "estimated_cost_usd": round(elapsed * RATE_CARD_CPU_USD_PER_SECOND, 6),
                "locked_test_access_count": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return validated


@app.function(image=image, cpu=2, memory=4_096, volumes={str(VOLUME_PATH): volume})
def inspect_sample() -> dict[str, Any]:
    volume.reload()
    source_bundle = _source_bundle()
    cohort = candidate.freeze_checkpoint_cohort(_checkpoint_receipts())
    authority = _authority(cohort, source_bundle)
    root = _sample_output_root(authority["authority_id"])
    if not root.exists():
        return {
            "status": "missing",
            "authority_id": authority["authority_id"],
            "sample_count": 0,
            "locked_test_access_count": 0,
        }
    receipt = _validate_published_sample(root, authority)
    result = {
        "status": "complete",
        "authority_id": authority["authority_id"],
        "receipt_id": receipt["receipt_id"],
        "sample_count": receipt["sample_count"],
        "observed_stratum_count": receipt["observed_stratum_count"],
        "kish_effective_sample_size": receipt["kish_effective_sample_size"],
        "locked_test_access_count": 0,
    }
    assert_metadata_only(result, where="calibration sample inspection")
    return result


@app.local_entrypoint()
def main(action: str) -> None:
    if action == "prepare":
        result = prepare_sample.remote()
    elif action in {"inspect", "validate"}:
        result = inspect_sample.remote()
        if action == "validate" and result["status"] != "complete":
            raise RuntimeError("calibration sample is not complete")
    else:
        raise ValueError("action must be prepare, inspect or validate")
    print(json.dumps(result, sort_keys=True))
