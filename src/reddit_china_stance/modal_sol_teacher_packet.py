"""Build the frozen thread-disjoint 10k direct-Sol teacher packet on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance.context_assembly import (
    TRUNCATION_MARKER,
    _allocate_stratified_sample,
    deterministic_head_tail,
)

APP_NAME = "reddit-china-stance-sol-teacher-packet-v1"
ENVIRONMENT_NAME = "main"
VOLUME_NAME = "reddit-china-stance-data"
VOLUME_PATH = Path("/data")
SCHEMA_VERSION = "1.0.0"
DATASET_REVISION = "97627c42893bc479c6a4952b1db38bbc33a60ac8"
SOURCE_SCHEMA_VERSION = "1.0.2"
RETRIEVAL_POLICY_DIGEST = "9e8c9050b9512fd83b2a844d4259bd3df8e3f84dc213460fc19407238266769a"
STAGE_A_RUN_ID = "834ff757fb7931e508c782d64db4f9e8761e4273edb8b2635c9c5bff84c406b1"
LANGUAGE_POLICY_DIGEST = "83857eb8b501d6bf262461c4ea2815e6649c9581acc84a031558f5cf83483be2"
LANGUAGE_RUN_ID = "34dbfdea18a7dafe84b8d5be891d6c7771a0ef71e67f5878f0ecc7130229a7b1"
SUBREDDITS = (
    "AskReddit",
    "China",
    "ChineseLanguage",
    "Sino",
    "funny",
    "gaming",
    "geopolitics",
    "news",
    "todayilearned",
    "worldnews",
)
YEARS = tuple(range(2020, 2026))
CONTENT_TYPES = ("submission", "comment")
TARGET_ROWS = 10_000
SAMPLING_SEED = "reddit-china-stance-sol-teacher-10k-v1"
RESERVE_PER_CELL = 12
MAX_CONTEXT_CHARS = 1_536
MAX_PARENT_CHARS = 768
MAX_TARGET_CHARS = 2_048
NEAR_DUPLICATE_MIN_CHARS = 80
NEAR_DUPLICATE_MAX_HAMMING = 3
CONFIRMATION = "BUILD_THREAD_DISJOINT_SOL_TEACHER_PACKET_10K"
OUTPUT_PREFIX = Path("teacher-label-packet-v1")
CPU_HOUR_USD = Decimal("0.04730")
MEMORY_GIB_HOUR_USD = Decimal("0.00800")
PLANNING_HOURS = Decimal("2")
REQUESTED_CPUS = Decimal("16")
REQUESTED_MEMORY_GIB = Decimal("64")
ESTIMATED_COST_USD = (
    (REQUESTED_CPUS * CPU_HOUR_USD + REQUESTED_MEMORY_GIB * MEMORY_GIB_HOUR_USD) * PLANNING_HOURS
).quantize(Decimal("0.01"))
HARD_MAX_APPROVAL_USD = Decimal("20")
RUNTIME_DEPENDENCIES = {
    "duckdb": "1.4.4",
    "pyarrow": "25.0.1",
    "pydantic": "2.13.5",
}
BLINDNESS_CONTRACT = {
    "human_labels_hidden": True,
    "other_reviewer_labels_hidden": True,
    "split_identity_hidden": True,
    "canonical_record_ids_hidden": True,
    "source_metadata_hidden": True,
    "repository_context_not_provided": True,
    "web_use_prohibited": True,
    "filesystem_scope_limited_to_rubric_schema_input_and_output": True,
    "output_tool_does_not_supply_semantic_evidence": True,
    "reviewer_labels_hidden": True,
    "reviewer_identities_hidden": True,
    "consensus_decisions_hidden": True,
}

STAGE_ROOT = (
    Path("derived/stage-a")
    / DATASET_REVISION
    / f"source-schema={SOURCE_SCHEMA_VERSION}"
    / f"policy={RETRIEVAL_POLICY_DIGEST}"
    / f"run={STAGE_A_RUN_ID}"
)
LANGUAGE_ROOT = (
    STAGE_ROOT / "language" / f"policy={LANGUAGE_POLICY_DIGEST}" / f"run={LANGUAGE_RUN_ID}"
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT_NAME, create_if_missing=False
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*(f"{name}=={version}" for name, version in RUNTIME_DEPENDENCIES.items()))
    .add_local_python_source("reddit_china_stance")
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _simhash(text: str) -> int:
    normalised = _normalise_surface(text)
    tokens = normalised.split()
    features = (
        [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
        if len(tokens) >= 3
        else tokens or [normalised]
    )
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def build_reference_exclusions(
    *, reference_path: Path, source_packet_path: Path, output_path: Path
) -> dict[str, Any]:
    """Write hashes only: no Reddit text or identifiers leave the private inputs."""

    import pyarrow.parquet as pq

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rows = reference.get("rows")
    if not isinstance(rows, list):
        raise ValueError("reference exclusions require a row array")
    rows = [
        row
        for row in rows
        if row.get("split") in {"development", "locked_test_candidate"}
    ]
    if len(rows) != 452:
        raise ValueError("reference exclusions require the exact 452-row development frame")
    surfaces: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = row.get("target_text")
        if not isinstance(text, str) or not text:
            raise ValueError("development reference contains an invalid target surface")
        normalised = _normalise_surface(text)
        key = hashlib.sha256(normalised.encode()).hexdigest()
        surfaces[key] = {
            "exact_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "normalised_sha256": key,
            "normalised_chars": len(normalised),
            "simhash64": f"{_simhash(text):016x}",
        }
    table = pq.read_table(source_packet_path, columns=["thread_id"])
    thread_ids = table.column("thread_id").to_pylist()
    if len(thread_ids) != 150 or any(
        not isinstance(value, str) or not value for value in thread_ids
    ):
        raise ValueError("source packet must expose exactly 150 valid private thread IDs")
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-teacher-reference-exclusions-v1",
        "reference_sha256": _file_sha256(reference_path),
        "source_packet_sha256": _file_sha256(source_packet_path),
        "reference_rows": len(rows),
        "unique_reference_surfaces": len(surfaces),
        "known_thread_count": len(set(thread_ids)),
        "surface_hashes": sorted(surfaces.values(), key=lambda item: item["normalised_sha256"]),
        "known_thread_sha256": sorted(
            {hashlib.sha256(value.encode()).hexdigest() for value in thread_ids}
        ),
    }
    payload = _json_bytes(value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != payload:
        raise RuntimeError("reference exclusion manifest is immutable and differs")
    if not output_path.exists():
        with output_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "status": "prepared",
        "path": str(output_path),
        "sha256": _file_sha256(output_path),
        "reference_rows": len(rows),
        "unique_reference_surfaces": len(surfaces),
        "known_thread_count": len(set(thread_ids)),
    }


def _validate_exclusions(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "reference_sha256",
        "source_packet_sha256",
        "reference_rows",
        "unique_reference_surfaces",
        "known_thread_count",
        "surface_hashes",
        "known_thread_sha256",
    }
    if set(value) != required or value.get("kind") != "sol-teacher-reference-exclusions-v1":
        raise ValueError("reference exclusion manifest contract drifted")
    surfaces = value.get("surface_hashes")
    threads = value.get("known_thread_sha256")
    if not isinstance(surfaces, list) or len(surfaces) != value["unique_reference_surfaces"]:
        raise ValueError("reference surface hash inventory drifted")
    if not isinstance(threads, list) or len(threads) != value["known_thread_count"]:
        raise ValueError("known thread hash inventory drifted")
    return json.loads(json.dumps(value, sort_keys=True))


def _code_state(root: Path) -> dict[str, Any]:
    paths = (
        root / "src/reddit_china_stance/modal_sol_teacher_packet.py",
        root / "src/reddit_china_stance/context_assembly.py",
        root / "docs/rubrics/human-reference-semantic-v1.md",
        root / "schemas/human-reference-semantic-label-v1.schema.json",
        root / "uv.lock",
    )
    files = {str(path.relative_to(root)): _file_sha256(path) for path in paths}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_commit": commit,
        "files": files,
        "code_sha256": _canonical_sha256(files),
    }


def make_contract(
    *, exclusion_sha256: str, code_state: Mapping[str, Any], approved_cost_usd: Decimal
) -> dict[str, Any]:
    if approved_cost_usd < ESTIMATED_COST_USD or approved_cost_usd > HARD_MAX_APPROVAL_USD:
        raise ValueError(
            f"approved cost must cover ${ESTIMATED_COST_USD} "
            f"and not exceed ${HARD_MAX_APPROVAL_USD}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-teacher-packet-contract-v1",
        "dataset_revision": DATASET_REVISION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "retrieval_policy_digest": RETRIEVAL_POLICY_DIGEST,
        "stage_a_run_id": STAGE_A_RUN_ID,
        "language_policy_digest": LANGUAGE_POLICY_DIGEST,
        "language_run_id": LANGUAGE_RUN_ID,
        "language_eligibility": "provisional_english",
        "target_rows": TARGET_ROWS,
        "sampling_seed": SAMPLING_SEED,
        "sampling_unit": "submission_thread",
        "stratify_by": ["subreddit", "year", "content_type", "retrieval_mode"],
        "exact_surface_deduplication": True,
        "near_duplicate": {
            "algorithm": "token-trigram-simhash64",
            "minimum_characters": NEAR_DUPLICATE_MIN_CHARS,
            "maximum_hamming_distance": NEAR_DUPLICATE_MAX_HAMMING,
            "length_ratio": [0.8, 1.25],
            "reserve_per_cell": RESERVE_PER_CELL,
        },
        "context_limits": {
            "target": MAX_TARGET_CHARS,
            "submission": MAX_CONTEXT_CHARS,
            "parent": MAX_PARENT_CHARS,
        },
        "reference_exclusion_sha256": exclusion_sha256,
        "code_state": dict(code_state),
        "compute": {
            "cpus": int(REQUESTED_CPUS),
            "memory_gib": int(REQUESTED_MEMORY_GIB),
            "canonical_scans": 2,
            "runtime_dependencies": dict(RUNTIME_DEPENDENCIES),
            "estimated_cost_usd": str(ESTIMATED_COST_USD),
            "approved_cost_usd": str(approved_cost_usd),
        },
    }


def _discover_inputs() -> dict[str, list[Path]]:
    candidates = sorted(
        (VOLUME_PATH / STAGE_ROOT / "candidates").glob(
            "subreddit=*/year=*/unit=*/candidates-*.parquet"
        )
    )
    languages = sorted(
        (VOLUME_PATH / LANGUAGE_ROOT / "decisions").glob(
            "subreddit=*/year=*/unit=*/language-decisions-*.parquet"
        )
    )
    canonical = sorted(
        (VOLUME_PATH / "normalised" / DATASET_REVISION / f"schema={SOURCE_SCHEMA_VERSION}").glob(
            "*/year=*/subreddit=*/content_type=*/part-00000.parquet"
        )
    )
    if len(candidates) != 60 or len(languages) != 60 or len(canonical) != 120:
        raise RuntimeError(
            f"input inventory mismatch: candidates={len(candidates)}, "
            f"languages={len(languages)}, canonical={len(canonical)}"
        )
    if any(".incomplete" in str(path) for path in (*candidates, *languages, *canonical)):
        raise RuntimeError("input inventory includes an incomplete namespace")
    return {"candidates": candidates, "languages": languages, "canonical": canonical}


def _inventory(paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        name_digest = path.stem.rsplit("-", 1)[-1]
        if len(name_digest) != 64 and path.name != "part-00000.parquet":
            raise RuntimeError(f"non-content-addressed input path: {path}")
        result.append(
            {
                "relative_path": str(path.relative_to(VOLUME_PATH)),
                "bytes": path.stat().st_size,
                "content_address": name_digest if len(name_digest) == 64 else None,
            }
        )
    return result


def _cell_id(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row[key]) for key in ("subreddit", "year", "content_type", "retrieval_mode")
    )


def _is_near_duplicate(*, simhash: int, chars: int, comparisons: Sequence[tuple[int, int]]) -> bool:
    if chars < NEAR_DUPLICATE_MIN_CHARS:
        return False
    for other_hash, other_chars in comparisons:
        if other_chars < NEAR_DUPLICATE_MIN_CHARS:
            continue
        ratio = chars / other_chars
        if (
            0.8 <= ratio <= 1.25
            and (simhash ^ other_hash).bit_count() <= NEAR_DUPLICATE_MAX_HAMMING
        ):
            return True
    return False


def _bound_optional(text: Any, *, max_chars: int) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str) or not text:
        raise ValueError("context text must be null or non-empty")
    return deterministic_head_tail(text, max_chars=max_chars).text


def _write_packet(
    *,
    output_root: Path,
    contract: Mapping[str, Any],
    contract_id: str,
    selected: Sequence[Mapping[str, Any]],
    input_inventory: Mapping[str, Sequence[Mapping[str, Any]]],
    population: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    staging = output_root.parent / f".{output_root.name}.incomplete"
    if output_root.exists():
        return _validate_existing_packet(output_root=output_root, contract_id=contract_id)
    if len(selected) != TARGET_ROWS:
        raise RuntimeError("packet writer requires exactly 10,000 selected rows")
    if staging.exists():
        raise FileExistsError(f"stale packet staging exists: {staging}")
    staging.mkdir(parents=True)

    blinded_rows = []
    mapping_rows = []
    for index, row in enumerate(selected):
        record_id = str(row["record_id"])
        opaque_id = "L" + hashlib.sha256(f"{contract_id}\0{record_id}".encode()).hexdigest()[:24]
        blinded_rows.append(
            {
                "source_sample_id": opaque_id,
                "target_text": row["target_text"],
                "submission_context": row["submission_context"],
                "parent_context": row["parent_context"],
            }
        )
        mapping_rows.append(
            {
                "opaque_id": opaque_id,
                "record_id": record_id,
                "thread_id": row["thread_id"],
                "subreddit": row["subreddit"],
                "year": row["year"],
                "month": row["month"],
                "content_type": row["content_type"],
                "retrieval_mode": row["retrieval_mode"],
                "cell_id": row["cell_id"],
                "cell_rank": row["cell_rank"],
                "population_threads_in_cell": row["population_threads_in_cell"],
                "selected_threads_in_cell": row["selected_threads_in_cell"],
                "target_surface_sha256": hashlib.sha256(row["target_text"].encode()).hexdigest(),
                "packet_order": index,
            }
        )
    if len({row["source_sample_id"] for row in blinded_rows}) != TARGET_ROWS:
        raise RuntimeError("opaque packet IDs are not unique")
    if len({row["thread_id"] for row in mapping_rows}) != TARGET_ROWS:
        raise RuntimeError("submission thread crossed selected packet rows")

    blinded = {
        "schema_version": "1.0.0",
        "kind": "human-reference-blinded-adjudication-input-v1",
        "consensus_id": contract_id,
        "consensus_artifact_sha256": _canonical_sha256(contract),
        "source_packet_sha256": _canonical_sha256(
            [
                {"opaque_id": row["opaque_id"], "packet_order": row["packet_order"]}
                for row in mapping_rows
            ]
        ),
        "rubric_sha256": contract["code_state"]["files"][
            "docs/rubrics/human-reference-semantic-v1.md"
        ],
        "label_schema_sha256": contract["code_state"]["files"][
            "schemas/human-reference-semantic-label-v1.schema.json"
        ],
        "blindness": dict(BLINDNESS_CONTRACT),
        "rows": blinded_rows,
    }
    blinded_path = staging / "blinded-input.json"
    blinded_path.write_bytes(_json_bytes(blinded))

    mapping_path = staging / "private-mapping.parquet"
    pq.write_table(pa.Table.from_pylist(mapping_rows), mapping_path, compression="zstd")
    packet_binding = {
        "contract_id": contract_id,
        "blinded_input_sha256": _file_sha256(blinded_path),
        "private_mapping_sha256": _file_sha256(mapping_path),
        "item_count": TARGET_ROWS,
        "thread_count": TARGET_ROWS,
    }
    packet_id = _canonical_sha256(packet_binding)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-teacher-packet-manifest-v1",
        "contract_id": contract_id,
        "packet_id": packet_id,
        "contract": dict(contract),
        "packet_binding": packet_binding,
        "population": dict(population),
        "input_inventory": {key: list(value) for key, value in input_inventory.items()},
        "reference_exclusions": {
            "manifest_sha256": contract["reference_exclusion_sha256"],
            "reference_rows": exclusions["reference_rows"],
            "unique_reference_surfaces": exclusions["unique_reference_surfaces"],
            "known_thread_count": exclusions["known_thread_count"],
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    receipt_body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sol-teacher-packet-receipt-v1",
        "status": "complete",
        "contract_id": contract_id,
        "packet_id": packet_id,
        "item_count": TARGET_ROWS,
        "unique_threads": TARGET_ROWS,
        "blinded_input_sha256": packet_binding["blinded_input_sha256"],
        "private_mapping_sha256": packet_binding["private_mapping_sha256"],
        "manifest_sha256": _file_sha256(manifest_path),
        "population": dict(population),
        "wall_seconds": round(wall_seconds, 3),
        "receipt_contains_raw_text": False,
        "receipt_contains_record_ids": False,
        "receipt_contains_thread_ids": False,
        "receipt_contains_row_level_labels": False,
        "private_packet_contains_raw_text": True,
        "private_mapping_contains_record_ids": True,
    }
    receipt_payload = _json_bytes(receipt_body)
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    (staging / f"receipt-{receipt_sha}.json").write_bytes(receipt_payload)
    staging.replace(output_root)
    volume.commit()
    return receipt_body


def _validate_existing_packet(*, output_root: Path, contract_id: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    manifest_path = output_root / "manifest.json"
    blinded_path = output_root / "blinded-input.json"
    mapping_path = output_root / "private-mapping.parquet"
    receipt_paths = list(output_root.glob("receipt-*.json"))
    if (
        not manifest_path.is_file()
        or not blinded_path.is_file()
        or not mapping_path.is_file()
        or len(receipt_paths) != 1
    ):
        raise RuntimeError("existing packet root is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blinded = json.loads(blinded_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
    binding = manifest.get("packet_binding")
    rows = blinded.get("rows")
    if (
        manifest.get("kind") != "sol-teacher-packet-manifest-v1"
        or manifest.get("contract_id") != contract_id
        or manifest.get("contract") is None
        or _canonical_sha256(manifest["contract"]) != contract_id
        or not isinstance(binding, Mapping)
        or blinded.get("kind") != "human-reference-blinded-adjudication-input-v1"
        or not isinstance(rows, list)
        or len(rows) != TARGET_ROWS
        or pq.ParquetFile(mapping_path).metadata.num_rows != TARGET_ROWS
        or binding.get("item_count") != TARGET_ROWS
        or binding.get("blinded_input_sha256") != _file_sha256(blinded_path)
        or binding.get("private_mapping_sha256") != _file_sha256(mapping_path)
        or receipt.get("status") != "complete"
        or receipt.get("contract_id") != contract_id
        or receipt.get("packet_id") != manifest.get("packet_id")
        or receipt.get("item_count") != TARGET_ROWS
        or receipt.get("manifest_sha256") != _file_sha256(manifest_path)
        or receipt.get("blinded_input_sha256") != binding.get("blinded_input_sha256")
        or receipt.get("private_mapping_sha256") != binding.get("private_mapping_sha256")
    ):
        raise RuntimeError("existing packet root failed immutable validation")
    return receipt


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=16.0,
    memory=65_536,
    timeout=14_400,
    max_containers=1,
)
def prepare_packet_remote(
    *, contract: dict[str, Any], contract_id: str, exclusions: dict[str, Any]
) -> dict[str, Any]:
    import duckdb

    if _canonical_sha256(contract) != contract_id:
        raise ValueError("packet contract ID drifted")
    exclusions = _validate_exclusions(exclusions)
    started = time.monotonic()
    volume.reload()
    output_root = VOLUME_PATH / OUTPUT_PREFIX / f"contract={contract_id}"
    if output_root.exists():
        return _write_packet(
            output_root=output_root,
            contract=contract,
            contract_id=contract_id,
            selected=[],
            input_inventory={},
            population={},
            exclusions=exclusions,
            wall_seconds=0,
        )
    paths = _discover_inputs()
    inventories = {key: _inventory(value) for key, value in paths.items()}
    inventory_digest = _canonical_sha256(inventories)
    exact_hashes = sorted({row["exact_sha256"] for row in exclusions["surface_hashes"]})
    normalised_hashes = sorted({row["normalised_sha256"] for row in exclusions["surface_hashes"]})
    known_threads = list(exclusions["known_thread_sha256"])
    reference_simhashes = [
        (int(row["simhash64"], 16), int(row["normalised_chars"]))
        for row in exclusions["surface_hashes"]
    ]

    database = duckdb.connect(database=":memory:")
    database.execute("SET threads = 16")
    database.execute("SET memory_limit = '58GB'")
    database.execute("SET temp_directory = '/tmp/sol-teacher-packet-duckdb'")
    candidate_paths = [str(path) for path in paths["candidates"]]
    language_paths = [str(path) for path in paths["languages"]]
    canonical_paths = [str(path) for path in paths["canonical"]]
    marker = TRUNCATION_MARKER
    source_budget = MAX_TARGET_CHARS - len(marker)
    head_chars = (source_budget + 1) // 2
    tail_chars = source_budget - head_chars
    bounded = (
        f"CASE WHEN length(src.text) <= {MAX_TARGET_CHARS} THEN src.text "
        f"ELSE left(src.text, {head_chars}) || ? || right(src.text, {tail_chars}) END"
    )
    try:
        database.execute(
            """
            CREATE TEMP TABLE english_candidates AS
            SELECT c.record_id, c.content_type, c.retrieval_channels
            FROM read_parquet(?) c
            JOIN read_parquet(?) lang USING (record_id)
            WHERE lang.provisional_status = 'provisional_english'
            """,
            [candidate_paths, language_paths],
        )
        candidate_count, distinct_candidates = database.execute(
            "SELECT count(*), count(DISTINCT record_id) FROM english_candidates"
        ).fetchone()
        if candidate_count != distinct_candidates or candidate_count < TARGET_ROWS:
            raise RuntimeError("confident-English candidate inventory is invalid")
        database.execute(
            f"""
            CREATE TEMP TABLE eligible_base AS
            SELECT src.record_id, src.submission_id AS thread_id, src.subreddit, src.year,
                   src.month, src.content_type, ec.retrieval_channels,
                   CASE WHEN list_contains(ec.retrieval_channels, 'direct_lexical')
                        THEN 'direct' ELSE 'expanded_only' END AS retrieval_mode,
                   sha256({bounded}) AS exact_surface_sha256,
                   sha256(lower(regexp_replace(trim({bounded}), '\\s+', ' ', 'g')))
                       AS normalised_surface_sha256
            FROM read_parquet(?) src
            JOIN english_candidates ec
              ON src.record_id = ec.record_id AND src.content_type = ec.content_type
            """,
            [marker, marker, canonical_paths],
        )
        observed, distinct_observed = database.execute(
            "SELECT count(*), count(DISTINCT record_id) FROM eligible_base"
        ).fetchone()
        if observed != candidate_count or observed != distinct_observed:
            raise RuntimeError(
                "candidate-to-canonical join did not conserve confident English rows"
            )
        database.execute("CREATE TEMP TABLE exact_exclusions(value VARCHAR PRIMARY KEY)")
        database.execute("CREATE TEMP TABLE normalised_exclusions(value VARCHAR PRIMARY KEY)")
        database.execute("CREATE TEMP TABLE thread_exclusions(value VARCHAR PRIMARY KEY)")
        database.execute("INSERT INTO exact_exclusions SELECT * FROM UNNEST(?)", [exact_hashes])
        database.execute(
            "INSERT INTO normalised_exclusions SELECT * FROM UNNEST(?)", [normalised_hashes]
        )
        database.execute("INSERT INTO thread_exclusions SELECT * FROM UNNEST(?)", [known_threads])
        database.execute(
            """
            CREATE TEMP TABLE excluded_threads AS
            SELECT DISTINCT e.thread_id
            FROM eligible_base e
            LEFT JOIN exact_exclusions x ON e.exact_surface_sha256 = x.value
            LEFT JOIN normalised_exclusions n ON e.normalised_surface_sha256 = n.value
            LEFT JOIN thread_exclusions t ON sha256(e.thread_id) = t.value
            WHERE x.value IS NOT NULL OR n.value IS NOT NULL OR t.value IS NOT NULL
            """
        )
        excluded_thread_count = int(
            database.execute("SELECT count(*) FROM excluded_threads").fetchone()[0]
        )
        database.execute(
            """
            CREATE TEMP TABLE thread_representatives AS
            SELECT * EXCLUDE (thread_rank)
            FROM (
                SELECT e.*,
                       row_number() OVER (
                           PARTITION BY e.thread_id
                           ORDER BY sha256(? || chr(31) || 'thread:' || e.thread_id
                                           || chr(31) || e.record_id), e.record_id
                       ) AS thread_rank
                FROM eligible_base e
                LEFT JOIN excluded_threads x USING (thread_id)
                WHERE x.thread_id IS NULL
            )
            WHERE thread_rank = 1
            """,
            [SAMPLING_SEED],
        )
        eligible_unique_threads = int(
            database.execute("SELECT count(*) FROM thread_representatives").fetchone()[0]
        )
        database.execute(
            """
            CREATE TEMP TABLE unique_surfaces AS
            SELECT * EXCLUDE (surface_rank)
            FROM (
                SELECT r.*,
                       row_number() OVER (
                           PARTITION BY r.normalised_surface_sha256
                           ORDER BY sha256(? || chr(31) || 'surface:'
                                           || r.normalised_surface_sha256
                                           || chr(31) || r.record_id), r.record_id
                       ) AS surface_rank
                FROM thread_representatives r
            )
            WHERE surface_rank = 1
            """,
            [SAMPLING_SEED],
        )
        database.execute(
            """
            CREATE TEMP TABLE population AS
            SELECT *, subreddit || '|' || CAST(year AS VARCHAR) || '|' || content_type
                       || '|' || retrieval_mode AS cell_id
            FROM unique_surfaces
            """
        )
        sizes = {
            str(cell): int(count)
            for cell, count in database.execute(
                "SELECT cell_id, count(*) FROM population GROUP BY cell_id"
            ).fetchall()
        }
        allocation = _allocate_stratified_sample(
            sizes,
            sample_size=TARGET_ROWS,
            seed=int(hashlib.sha256(SAMPLING_SEED.encode()).hexdigest()[:16], 16),
        )
        database.execute("CREATE TEMP TABLE quotas(cell_id VARCHAR PRIMARY KEY, quota INTEGER)")
        database.executemany("INSERT INTO quotas VALUES (?, ?)", sorted(allocation.items()))
        database.execute(
            """
            CREATE TEMP TABLE reserve_pool AS
            SELECT * EXCLUDE (reserve_rank), reserve_rank AS cell_rank,
                   q.quota AS selected_threads_in_cell,
                   p.population_threads_in_cell
            FROM (
                SELECT p.*,
                       row_number() OVER (
                           PARTITION BY p.cell_id
                           ORDER BY sha256(? || chr(31) || 'within-cell:' || p.cell_id
                                           || chr(31) || p.record_id), p.record_id
                       ) AS reserve_rank,
                       count(*) OVER (PARTITION BY p.cell_id) AS population_threads_in_cell
                FROM population p
            ) p
            JOIN quotas q USING (cell_id)
            WHERE reserve_rank <= q.quota + ?
            """,
            [SAMPLING_SEED, RESERVE_PER_CELL],
        )
        reserve_count = int(database.execute("SELECT count(*) FROM reserve_pool").fetchone()[0])
        database.execute(
            """
            CREATE TEMP TABLE needed_ids AS
            SELECT record_id FROM reserve_pool
            UNION
            SELECT thread_id FROM reserve_pool WHERE content_type = 'comment'
            """
        )
        database.execute(
            """
            INSERT INTO needed_ids
            SELECT src.parent_id
            FROM read_parquet(?) src
            JOIN reserve_pool r USING (record_id)
            WHERE r.content_type = 'comment'
              AND starts_with(CAST(src.parent_id AS VARCHAR), 't1_')
              AND src.parent_id IS NOT NULL
            EXCEPT SELECT record_id FROM needed_ids
            """,
            [canonical_paths],
        )
        database.execute(
            """
            CREATE TEMP TABLE selected_text AS
            SELECT src.record_id, src.content_type, src.submission_id, src.parent_id,
                   src.text, src.text_sha256
            FROM read_parquet(?) src
            JOIN needed_ids n USING (record_id)
            """,
            [canonical_paths],
        )
        relation = database.execute(
            """
            SELECT r.record_id, r.thread_id, r.subreddit, r.year, r.month, r.content_type,
                   r.retrieval_mode, r.cell_id, r.cell_rank, r.population_threads_in_cell,
                   r.selected_threads_in_cell, target.text AS target_text,
                   submission.text AS submission_context, parent.text AS parent_context,
                   target.parent_id
            FROM reserve_pool r
            JOIN selected_text target ON target.record_id = r.record_id
            LEFT JOIN selected_text submission
              ON r.content_type = 'comment'
             AND submission.record_id = r.thread_id
             AND submission.content_type = 'submission'
            LEFT JOIN selected_text parent
              ON r.content_type = 'comment'
             AND starts_with(CAST(target.parent_id AS VARCHAR), 't1_')
             AND parent.record_id = target.parent_id
             AND parent.content_type = 'comment'
             AND parent.submission_id = r.thread_id
            ORDER BY r.cell_id, r.cell_rank, r.record_id
            """
        )
        rows = []
        for batch in relation.fetch_record_batch(rows_per_batch=10_000):
            rows.extend(batch.to_pylist())
        if len(rows) != reserve_count:
            raise RuntimeError("context join changed reserve-pool row cardinality")
    finally:
        database.close()

    by_cell: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = _bound_optional(row["target_text"], max_chars=MAX_TARGET_CHARS)
        assert target is not None
        submission = (
            _bound_optional(row["submission_context"], max_chars=MAX_CONTEXT_CHARS)
            if row["content_type"] == "comment"
            else None
        )
        parent = (
            _bound_optional(row["parent_context"], max_chars=MAX_PARENT_CHARS)
            if row["content_type"] == "comment"
            else None
        )
        normalised = _normalise_surface(target)
        by_cell[row["cell_id"]].append(
            {
                **row,
                "target_text": target,
                "submission_context": submission,
                "parent_context": parent,
                "normalised_chars": len(normalised),
                "simhash": _simhash(target),
            }
        )

    selected: list[dict[str, Any]] = []
    accepted_simhashes: list[tuple[int, int]] = []
    near_reference_exclusions = 0
    near_internal_exclusions = 0
    for cell_id in sorted(allocation):
        quota = allocation[cell_id]
        accepted = 0
        for row in sorted(
            by_cell[cell_id], key=lambda item: (item["cell_rank"], item["record_id"])
        ):
            if _is_near_duplicate(
                simhash=row["simhash"],
                chars=row["normalised_chars"],
                comparisons=reference_simhashes,
            ):
                near_reference_exclusions += 1
                continue
            if _is_near_duplicate(
                simhash=row["simhash"],
                chars=row["normalised_chars"],
                comparisons=accepted_simhashes,
            ):
                near_internal_exclusions += 1
                continue
            selected.append(row)
            accepted_simhashes.append((row["simhash"], row["normalised_chars"]))
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            raise RuntimeError(
                f"near-duplicate reserve exhausted for {cell_id}: "
                f"accepted={accepted}, quota={quota}"
            )
    selected.sort(key=lambda row: (row["cell_id"], row["cell_rank"], row["record_id"]))
    if len(selected) != TARGET_ROWS:
        raise RuntimeError("near-duplicate filtering did not conserve the 10k packet")
    population = {
        "confident_english_candidates": int(candidate_count),
        "excluded_development_threads": excluded_thread_count,
        "eligible_unique_threads_before_surface_dedup": eligible_unique_threads,
        "eligible_unique_surfaces": int(sum(sizes.values())),
        "observed_cells": len(sizes),
        "selected_rows": TARGET_ROWS,
        "selected_threads": TARGET_ROWS,
        "near_reference_exclusions_from_reserve": near_reference_exclusions,
        "near_internal_exclusions_from_reserve": near_internal_exclusions,
        "input_inventory_digest": inventory_digest,
    }
    return _write_packet(
        output_root=output_root,
        contract=contract,
        contract_id=contract_id,
        selected=selected,
        input_inventory=inventories,
        population=population,
        exclusions=exclusions,
        wall_seconds=time.monotonic() - started,
    )


@app.local_entrypoint()
def main(
    exclusion_manifest_path: str = "data/private-sol-teacher-10k-v1/reference-exclusions.json",
    approved_cost_usd: str = "5",
    confirmation: str = "",
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise RuntimeError(f"confirmation must be exactly {CONFIRMATION}")
    root = Path(__file__).resolve().parents[2]
    exclusion_path = Path(exclusion_manifest_path)
    exclusions = _validate_exclusions(json.loads(exclusion_path.read_text(encoding="utf-8")))
    exclusion_sha = _file_sha256(exclusion_path)
    approved = Decimal(approved_cost_usd)
    contract = make_contract(
        exclusion_sha256=exclusion_sha,
        code_state=_code_state(root),
        approved_cost_usd=approved,
    )
    contract_id = _canonical_sha256(contract)
    result = prepare_packet_remote.remote(
        contract=contract, contract_id=contract_id, exclusions=exclusions
    )
    return {
        "contract_id": contract_id,
        "estimated_cost_usd": str(ESTIMATED_COST_USD),
        "result": result,
    }
