"""Run and reconcile the frozen 10k direct-Sol teacher generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from reddit_china_stance.human_seeded_consensus_v1 import (
    SOL_ADJUDICATOR_MODEL,
    canonical_sha256,
    file_sha256,
    load_label_schema,
    validate_semantic_label,
)
from reddit_china_stance.privacy import assert_metadata_only
from reddit_china_stance.sol_adjudicator import (
    ADJUDICATOR_ID,
    INSTRUCTION,
    REASONING_EFFORT,
    RUBRIC_PATH,
    SCHEMA_PATH,
)
from reddit_china_stance.sol_adjudicator import (
    run as run_sol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_ROOT = REPO_ROOT / "data/private-sol-teacher-10k-v1/packet"
DEFAULT_PRIVATE_ROOT = REPO_ROOT / "data/private-sol-teacher-10k-v1/generation"
DEFAULT_PUBLIC_ROOT = REPO_ROOT / "outputs/sol-teacher-10k-v1"
SHARD_COUNT = 264
JOBS = 8
TIMEOUT_SECONDS = 1_800
EXPECTED_ROWS = 10_000
TARGETS = ("china_general", "government_ccp", "people_culture", "other")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"immutable output differs: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_immutable(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not pq.read_table(path).equals(table):
            raise RuntimeError(f"immutable Parquet differs: {path}")
        return
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not pq.read_table(path).equals(table):
                raise RuntimeError(f"immutable Parquet differs: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _code_state() -> dict[str, Any]:
    paths = (
        Path(__file__),
        REPO_ROOT / "src/reddit_china_stance/sol_adjudicator.py",
        RUBRIC_PATH,
        SCHEMA_PATH,
        REPO_ROOT / "uv.lock",
    )
    files = {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"git_commit": commit, "files": files, "code_sha256": canonical_sha256(files)}


def validate_packet(packet_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest_path = packet_root / "manifest.json"
    input_path = packet_root / "blinded-input.json"
    mapping_path = packet_root / "private-mapping.parquet"
    receipts = list(packet_root.glob("receipt-*.json"))
    if not manifest_path.is_file() or not input_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError("teacher packet is incomplete")
    if len(receipts) != 1:
        raise RuntimeError("teacher packet must contain exactly one receipt")
    manifest = _read_object(manifest_path)
    receipt = _read_object(receipts[0])
    if (
        manifest.get("kind") != "sol-teacher-packet-manifest-v1"
        or receipt.get("kind") != "sol-teacher-packet-receipt-v1"
        or receipt.get("status") != "complete"
    ):
        raise ValueError("teacher packet contract drifted")
    binding = manifest.get("packet_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("teacher packet binding is missing")
    expected = {
        "contract_id": manifest.get("contract_id"),
        "packet_id": manifest.get("packet_id"),
        "item_count": EXPECTED_ROWS,
        "blinded_input_sha256": file_sha256(input_path),
        "private_mapping_sha256": file_sha256(mapping_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    observed = {
        "contract_id": receipt.get("contract_id"),
        "packet_id": receipt.get("packet_id"),
        "item_count": receipt.get("item_count"),
        "blinded_input_sha256": receipt.get("blinded_input_sha256"),
        "private_mapping_sha256": receipt.get("private_mapping_sha256"),
        "manifest_sha256": receipt.get("manifest_sha256"),
    }
    if observed != expected:
        raise RuntimeError("teacher packet receipt binding failed")
    if (
        binding.get("item_count") != EXPECTED_ROWS
        or binding.get("blinded_input_sha256") != expected["blinded_input_sha256"]
        or binding.get("private_mapping_sha256") != expected["private_mapping_sha256"]
    ):
        raise RuntimeError("teacher packet manifest binding failed")
    return manifest, receipt, input_path, mapping_path


def make_run_contract(packet_root: Path) -> dict[str, Any]:
    manifest, _, input_path, mapping_path = validate_packet(packet_root)
    return {
        "schema_version": "1.0.0",
        "kind": "sol-teacher-generation-contract-v1",
        "packet_id": manifest["packet_id"],
        "packet_contract_id": manifest["contract_id"],
        "blinded_input_sha256": file_sha256(input_path),
        "private_mapping_sha256": file_sha256(mapping_path),
        "model": SOL_ADJUDICATOR_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "adjudicator_id": ADJUDICATOR_ID,
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        "rubric_sha256": file_sha256(RUBRIC_PATH),
        "label_schema_sha256": file_sha256(SCHEMA_PATH),
        "item_count": EXPECTED_ROWS,
        "shard_count": SHARD_COUNT,
        "jobs": JOBS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "resume_policy": "reuse-exact-valid-immutable-fragments-only",
        "code_state": _code_state(),
    }


def run_generation(*, packet_root: Path, private_root: Path) -> dict[str, Any]:
    contract = make_run_contract(packet_root)
    run_id = canonical_sha256(contract)
    run_root = private_root / f"run={run_id}"
    _write_immutable(run_root / "run-manifest.json", {**contract, "run_id": run_id})
    result = run_sol(
        input_path=packet_root / "blinded-input.json",
        private_root=run_root / "execution",
        shard_count=SHARD_COUNT,
        jobs=JOBS,
        timeout_seconds=TIMEOUT_SECONDS,
    )
    return {"run_id": run_id, **result}


def _flatten_label(label: Mapping[str, Any]) -> dict[str, Any]:
    stance_by_target = {target: None for target in TARGETS}
    for item in label["target_stances"]:
        stance_by_target[item["target"]] = item["stance"]
    return {
        "relevance": label["relevance"],
        **{f"has_target_{target}": stance_by_target[target] is not None for target in TARGETS},
        **{f"stance_{target}": stance_by_target[target] for target in TARGETS},
    }


def validate_frozen_generation(
    *, packet_root: Path, run_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    packet_manifest, _, input_path, mapping_path = validate_packet(packet_root)
    run_manifest = _read_object(run_root / "run-manifest.json")
    run_id = run_manifest.get("run_id")
    frozen_contract = {key: value for key, value in run_manifest.items() if key != "run_id"}
    if (
        not isinstance(run_id, str)
        or canonical_sha256(frozen_contract) != run_id
        or run_root.name != f"run={run_id}"
        or run_manifest.get("kind") != "sol-teacher-generation-contract-v1"
        or run_manifest.get("packet_id") != packet_manifest.get("packet_id")
        or run_manifest.get("packet_contract_id") != packet_manifest.get("contract_id")
        or run_manifest.get("blinded_input_sha256") != file_sha256(input_path)
        or run_manifest.get("private_mapping_sha256") != file_sha256(mapping_path)
        or run_manifest.get("model") != SOL_ADJUDICATOR_MODEL
        or run_manifest.get("reasoning_effort") != REASONING_EFFORT
        or run_manifest.get("adjudicator_id") != ADJUDICATOR_ID
    ):
        raise RuntimeError("frozen generation manifest binding failed")
    output_path = (
        run_root
        / "execution"
        / f"input={run_manifest['blinded_input_sha256']}"
        / "adjudicator-output.json"
    )
    if not output_path.is_file():
        raise FileNotFoundError("frozen adjudicator output is missing")
    output = _read_object(output_path)
    blinded = _read_object(input_path)
    expected_output = {
        "schema_version": "1.0.0",
        "kind": "human-reference-blinded-adjudication-output-v1",
        "adjudicator_id": run_manifest["adjudicator_id"],
        "adjudicator_model": run_manifest["model"],
        "input_sha256": run_manifest["blinded_input_sha256"],
        "consensus_id": run_manifest["packet_contract_id"],
        "source_packet_sha256": blinded["source_packet_sha256"],
        "rubric_sha256": run_manifest["rubric_sha256"],
        "label_schema_sha256": run_manifest["label_schema_sha256"],
    }
    if any(output.get(key) != value for key, value in expected_output.items()):
        raise RuntimeError("frozen adjudicator output provenance binding failed")
    decisions = output.get("rows")
    mapping = pq.read_table(mapping_path).to_pylist()
    mapping.sort(key=lambda row: row["packet_order"])
    if (
        not isinstance(decisions, list)
        or len(decisions) != run_manifest.get("item_count")
        or len(mapping) != len(decisions)
        or [row.get("source_sample_id") for row in decisions]
        != [row["opaque_id"] for row in mapping]
    ):
        raise RuntimeError("frozen generation row conservation failed")
    schema = load_label_schema(SCHEMA_PATH)
    for decision in decisions:
        label = decision.get("label")
        if not isinstance(label, Mapping):
            raise ValueError("frozen teacher label is not an object")
        validate_semantic_label(label, schema=schema)
    execution_root = output_path.parent
    events = sorted(execution_root.glob("shard-*/event.json"))
    fragments = sorted(execution_root.glob("shard-*/fragment.json"))
    if len(events) != run_manifest.get("shard_count") or len(fragments) != len(events):
        raise RuntimeError("frozen shard evidence is incomplete")
    fragment_ids: list[str] = []
    fragment_rows_by_id: dict[str, dict[str, Any]] = {}
    for expected_index, (event_path, fragment_path) in enumerate(
        zip(events, fragments, strict=True)
    ):
        event = _read_object(event_path)
        fragment = _read_object(fragment_path)
        last_message = event_path.parent / "last-message.json"
        if (
            event.get("input_sha256") != run_manifest["blinded_input_sha256"]
            or event.get("shard_index") != expected_index
            or event.get("model") != run_manifest["model"]
            or event.get("reasoning_effort") != run_manifest["reasoning_effort"]
            or not last_message.is_file()
            or event.get("output_sha256") != file_sha256(last_message)
            or fragment.get("input_sha256") != run_manifest["blinded_input_sha256"]
            or fragment.get("shard_index") != expected_index
            or fragment.get("event_sha256") != file_sha256(event_path)
            or fragment.get("item_count") != len(fragment.get("rows", []))
        ):
            raise RuntimeError("frozen shard provenance binding failed")
        last_value = _read_object(last_message)
        last_rows = last_value.get("rows")
        if set(last_value) != {"rows"} or not isinstance(last_rows, list):
            raise RuntimeError("frozen shard model output envelope is invalid")
        normalised_last_rows = []
        for row in last_rows:
            if not isinstance(row, Mapping) or set(row) != {"source_sample_id", "label"}:
                raise RuntimeError("frozen shard model output row is invalid")
            label = row.get("label")
            if not isinstance(label, Mapping):
                raise RuntimeError("frozen shard model output label is invalid")
            normalised_last_rows.append(
                {
                    "source_sample_id": row["source_sample_id"],
                    "label": validate_semantic_label(label, schema=schema),
                }
            )
        if normalised_last_rows != fragment["rows"]:
            raise RuntimeError("frozen shard fragment differs from its bound model output")
        for row in fragment["rows"]:
            item_id = row["source_sample_id"]
            fragment_ids.append(item_id)
            if item_id in fragment_rows_by_id:
                raise RuntimeError("frozen shard evidence contains duplicate row IDs")
            fragment_rows_by_id[item_id] = row
    if len(fragment_ids) != len(set(fragment_ids)) or set(fragment_ids) != {
        row["opaque_id"] for row in mapping
    }:
        raise RuntimeError("frozen shard row set does not reconcile")
    reconstructed = [fragment_rows_by_id[row["opaque_id"]] for row in mapping]
    if reconstructed != decisions:
        raise RuntimeError("combined adjudicator output differs from bound shard contents")
    return run_manifest, mapping, decisions


def rebind_corrected_packet(
    *, old_packet_root: Path, corrected_packet_root: Path, run_root: Path, public_root: Path
) -> dict[str, Any]:
    run_manifest, old_mapping, decisions = validate_frozen_generation(
        packet_root=old_packet_root, run_root=run_root
    )
    corrected_manifest, _, corrected_input_path, corrected_mapping_path = validate_packet(
        corrected_packet_root
    )
    corrected_mapping = pq.read_table(corrected_mapping_path).to_pylist()
    corrected_mapping.sort(key=lambda row: row["packet_order"])
    old_blinded = _read_object(old_packet_root / "blinded-input.json")["rows"]
    corrected_blinded = _read_object(corrected_input_path)["rows"]
    ignored_mapping_fields = {"opaque_id", "population_threads_in_cell"}
    for old_row, corrected_row, old_surface, corrected_surface in zip(
        old_mapping, corrected_mapping, old_blinded, corrected_blinded, strict=True
    ):
        if (
            {key: value for key, value in old_row.items() if key not in ignored_mapping_fields}
            != {
                key: value
                for key, value in corrected_row.items()
                if key not in ignored_mapping_fields
            }
            or {key: value for key, value in old_surface.items() if key != "source_sample_id"}
            != {
                key: value
                for key, value in corrected_surface.items()
                if key != "source_sample_id"
            }
        ):
            raise RuntimeError("corrected packet is not semantically identical to labelled packet")
    cell_population: dict[str, int] = {}
    for row in corrected_mapping:
        cell_id = row["cell_id"]
        population = int(row["population_threads_in_cell"])
        if cell_id in cell_population and cell_population[cell_id] != population:
            raise RuntimeError("corrected cell population is not internally constant")
        if population < int(row["selected_threads_in_cell"]):
            raise RuntimeError("corrected cell population is below its selected count")
        cell_population[cell_id] = population
    label_schema = load_label_schema(SCHEMA_PATH)
    private_rows = []
    for old_source, corrected_source, decision in zip(
        old_mapping, corrected_mapping, decisions, strict=True
    ):
        label = validate_semantic_label(decision["label"], schema=label_schema)
        private_rows.append(
            {
                **dict(corrected_source),
                "generation_opaque_id": old_source["opaque_id"],
                **_flatten_label(label),
                "label_json": json.dumps(label, sort_keys=True, separators=(",", ":")),
                "teacher_run_id": run_manifest["run_id"],
                "teacher_model": run_manifest["model"],
                "sampling_packet_id": corrected_manifest["packet_id"],
            }
        )
    reconciliation_id = canonical_sha256(
        {
            "kind": "sol-teacher-corrected-packet-reconciliation-v1",
            "teacher_run_id": run_manifest["run_id"],
            "old_packet_id": run_manifest["packet_id"],
            "corrected_packet_id": corrected_manifest["packet_id"],
            "corrected_mapping_sha256": file_sha256(corrected_mapping_path),
            "item_count": len(private_rows),
        }
    )
    reconciliation_root = run_root / f"reconciliation={reconciliation_id}"
    labels_path = reconciliation_root / "teacher-labels.parquet"
    table = pa.Table.from_pylist(private_rows)
    _write_parquet_immutable(labels_path, table)
    receipt = {
        "schema_version": "1.0.0",
        "kind": "sol-teacher-corrected-packet-reconciliation-receipt-v1",
        "status": "complete",
        "reconciliation_id": reconciliation_id,
        "teacher_run_id": run_manifest["run_id"],
        "source_packet_id": run_manifest["packet_id"],
        "corrected_packet_id": corrected_manifest["packet_id"],
        "item_count": len(private_rows),
        "semantic_identity_matches": len(private_rows),
        "semantic_identity_mismatches": 0,
        "cell_count": len(cell_population),
        "private_teacher_parquet_sha256": file_sha256(labels_path),
        "contains_private_fields": False,
    }
    assert_metadata_only(receipt, where="corrected-packet-reconciliation")
    output_root = (
        public_root
        / f"run={run_manifest['run_id']}"
        / f"reconciliation={reconciliation_id}"
    )
    receipt_sha = canonical_sha256(receipt)
    _write_immutable(output_root / f"receipt-{receipt_sha}.json", receipt)
    return receipt


def finalise(
    *, packet_root: Path, private_root: Path, public_root: Path
) -> dict[str, Any]:
    contract = make_run_contract(packet_root)
    run_id = canonical_sha256(contract)
    run_root = private_root / f"run={run_id}"
    manifest_path = run_root / "run-manifest.json"
    if not manifest_path.is_file() or _read_object(manifest_path) != {**contract, "run_id": run_id}:
        raise RuntimeError("generation manifest is missing or drifted")
    validated_manifest, mapping, decisions = validate_frozen_generation(
        packet_root=packet_root, run_root=run_root
    )
    if validated_manifest != {**contract, "run_id": run_id}:
        raise RuntimeError("validated generation manifest differs from current contract")
    input_sha = contract["blinded_input_sha256"]
    output_path = run_root / "execution" / f"input={input_sha}" / "adjudicator-output.json"
    if not output_path.is_file():
        raise FileNotFoundError("complete Sol adjudicator output is missing")
    if len(decisions) != EXPECTED_ROWS or len(mapping) != EXPECTED_ROWS:
        raise RuntimeError("private mapping does not conserve 10,000 rows")
    expected_ids = [row["opaque_id"] for row in mapping]
    if [row.get("source_sample_id") for row in decisions] != expected_ids:
        raise RuntimeError("Sol output order or opaque-ID conservation failed")
    label_schema = load_label_schema(SCHEMA_PATH)
    relevance_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    stance_counts: Counter[str] = Counter()
    private_rows = []
    for source, decision in zip(mapping, decisions, strict=True):
        raw_label = decision.get("label")
        if not isinstance(raw_label, Mapping):
            raise ValueError("teacher label is not an object")
        label = validate_semantic_label(raw_label, schema=label_schema)
        relevance_counts[label["relevance"]] += 1
        for item in label["target_stances"]:
            target_counts[item["target"]] += 1
            stance_counts[item["stance"]] += 1
        private_rows.append(
            {
                **dict(source),
                **_flatten_label(label),
                "label_json": json.dumps(label, sort_keys=True, separators=(",", ":")),
                "teacher_run_id": run_id,
                "teacher_model": SOL_ADJUDICATOR_MODEL,
            }
        )
    label_path = run_root / "teacher-labels.parquet"
    table = pa.Table.from_pylist(private_rows)
    _write_parquet_immutable(label_path, table)
    events = sorted((run_root / "execution" / f"input={input_sha}").glob("shard-*/event.json"))
    fragments = sorted(
        (run_root / "execution" / f"input={input_sha}").glob("shard-*/fragment.json")
    )
    if len(events) != SHARD_COUNT or len(fragments) != SHARD_COUNT:
        raise RuntimeError("generation shard evidence is incomplete")
    input_tokens = cached_input_tokens = output_tokens = 0
    shard_elapsed_seconds = 0.0
    maximum_shard_seconds = 0.0
    for path in events:
        event = _read_object(path)
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            raise RuntimeError("generation event lacks token usage")
        input_tokens += int(usage["input_tokens"])
        cached_input_tokens += int(usage["cached_input_tokens"])
        output_tokens += int(usage["output_tokens"])
        elapsed = float(event["elapsed_seconds"])
        shard_elapsed_seconds += elapsed
        maximum_shard_seconds = max(maximum_shard_seconds, elapsed)
    summary = {
        "schema_version": "1.0.0",
        "kind": "sol-teacher-generation-summary-v1",
        "status": "complete",
        "run_id": run_id,
        "packet_id": contract["packet_id"],
        "item_count": EXPECTED_ROWS,
        "valid_item_count": EXPECTED_ROWS,
        "invalid_item_count": 0,
        "unique_thread_count": EXPECTED_ROWS,
        "shard_count": SHARD_COUNT,
        "model": SOL_ADJUDICATOR_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "relevance_counts": dict(sorted(relevance_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "stance_counts": dict(sorted(stance_counts.items())),
        "usage_totals": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
        },
        "shard_elapsed_seconds_sum": round(shard_elapsed_seconds, 3),
        "maximum_shard_seconds": round(maximum_shard_seconds, 3),
        "private_teacher_parquet_sha256": file_sha256(label_path),
        "private_adjudicator_output_sha256": file_sha256(output_path),
        "contains_private_fields": False,
    }
    assert_metadata_only(summary, where="sol-teacher-generation-summary")
    summary_path = public_root / f"run={run_id}" / "summary.json"
    _write_immutable(summary_path, summary)
    receipt = {
        **summary,
        "kind": "sol-teacher-generation-receipt-v1",
        "summary_sha256": file_sha256(summary_path),
        "run_manifest_sha256": file_sha256(manifest_path),
    }
    assert_metadata_only(receipt, where="sol-teacher-generation-receipt")
    receipt_sha = canonical_sha256(receipt)
    _write_immutable(public_root / f"run={run_id}" / f"receipt-{receipt_sha}.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("prepare", "run", "finalise", "validate-frozen", "rebind"),
        required=True,
    )
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--corrected-packet-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    args = parser.parse_args(argv)
    if args.action in {"validate-frozen", "rebind"}:
        if args.run_root is None:
            parser.error("--run-root is required for frozen-run actions")
        if args.action == "validate-frozen":
            manifest, mapping, _ = validate_frozen_generation(
                packet_root=args.packet_root, run_root=args.run_root
            )
            result = {
                "status": "valid",
                "run_id": manifest["run_id"],
                "item_count": len(mapping),
                "shard_count": manifest["shard_count"],
            }
        else:
            if args.corrected_packet_root is None:
                parser.error("--corrected-packet-root is required for rebind")
            result = rebind_corrected_packet(
                old_packet_root=args.packet_root,
                corrected_packet_root=args.corrected_packet_root,
                run_root=args.run_root,
                public_root=args.public_root,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    contract = make_run_contract(args.packet_root)
    run_id = canonical_sha256(contract)
    if args.action == "prepare":
        run_root = args.private_root / f"run={run_id}"
        _write_immutable(run_root / "run-manifest.json", {**contract, "run_id": run_id})
        result = {"status": "prepared", "run_id": run_id, "item_count": EXPECTED_ROWS}
    elif args.action == "run":
        result = run_generation(packet_root=args.packet_root, private_root=args.private_root)
    else:
        result = finalise(
            packet_root=args.packet_root,
            private_root=args.private_root,
            public_root=args.public_root,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
