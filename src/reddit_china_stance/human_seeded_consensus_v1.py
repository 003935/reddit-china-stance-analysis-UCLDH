"""Build a model-assisted development proxy from a human-seeded reference.

The supplied human annotation is retained only where it exactly matches at
least one of two blinded reviewers.  Exactly the remaining contentious rows
are sent to a distinct blinded Sol adjudicator.  This produces development
evidence, not human gold or independent human evaluation.  Public artefacts
contain aggregate metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from reddit_china_stance.privacy import assert_metadata_only

SCHEMA_VERSION = "1.0.0"
KIND = "human-seeded-consensus-v1"
REVIEW_KIND = "human-reference-independent-review-v1"
REVIEW_INPUT_KIND = "human-reference-independent-review-input-v1"
REVIEW_FRAGMENT_KIND = "human-reference-independent-review-fragment-v1"
ADJUDICATION_INPUT_KIND = "human-reference-blinded-adjudication-input-v1"
ADJUDICATION_OUTPUT_KIND = "human-reference-blinded-adjudication-output-v1"
FINAL_KIND = "human-seeded-sol-adjudicated-development-proxy-v1"
SOL_ADJUDICATOR_MODEL = "gpt-5.6-sol"
EXPECTED_FINAL_ROWS = 452
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas/human-reference-semantic-label-v1.schema.json"
RUBRIC_PATH = REPO_ROOT / "docs/rubrics/human-reference-semantic-v1.md"
PRIVATE_ROOT = REPO_ROOT / "data/private-human-reference-v1/consensus"
REVIEW_ROOT = REPO_ROOT / "data/private-human-reference-v1/reviews"
PUBLIC_ROOT = REPO_ROOT / "outputs/human-seeded-consensus-v1"

ALLOWED_SPLITS = frozenset(
    {"development", "locked_test", "locked_test_candidate", "excluded_language"}
)
REVIEWABLE_SPLITS = frozenset({"development", "locked_test", "locked_test_candidate"})
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
}
ADJUDICATION_BLINDNESS_CONTRACT = {
    **BLINDNESS_CONTRACT,
    "human_labels_hidden": True,
    "reviewer_labels_hidden": True,
    "reviewer_identities_hidden": True,
    "consensus_decisions_hidden": True,
}

def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(value).decode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    incomplete = path.with_suffix(path.suffix + ".incomplete")
    incomplete.write_text(payload, encoding="utf-8")
    incomplete.replace(path)


def _assert_public_metadata_only(value: Any, *, path: str = "metadata") -> None:
    assert_metadata_only(value, where=path)


def load_label_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    schema = _read_json_object(path)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_semantic_label(
    value: Mapping[str, Any], *, schema: Mapping[str, Any]
) -> dict[str, Any]:
    clean = deepcopy(dict(value))
    errors = sorted(
        Draft202012Validator(dict(schema)).iter_errors(clean),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"invalid semantic label: {errors[0].message}")

    relevance = clean["relevance"]
    target_stances = clean["target_stances"]
    if relevance == "material" and not target_stances:
        raise ValueError("invalid semantic label: material requires at least one target")
    if relevance != "material" and target_stances:
        raise ValueError("invalid semantic label: non-material relevance requires no targets")

    targets = [row["target"] for row in target_stances]
    if len(targets) != len(set(targets)):
        raise ValueError("invalid semantic label: duplicate target entries")

    target_enum = schema["properties"]["target_stances"]["items"]["properties"]["target"][
        "enum"
    ]
    target_order = {target: index for index, target in enumerate(target_enum)}
    clean["target_stances"] = sorted(
        (dict(row) for row in target_stances),
        key=lambda row: target_order[row["target"]],
    )
    return clean


def _required_string(value: Mapping[str, Any], field: str, *, where: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{where}.{field} must be a non-empty string")
    return result


def _reference_rows(
    reference: Mapping[str, Any], *, schema: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = reference.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("reference packet must contain a non-empty rows array")

    clean_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"reference.rows[{index}] must be an object")
        sample_id = _required_string(row, "source_sample_id", where=f"reference.rows[{index}]")
        if sample_id in seen:
            raise ValueError(f"duplicate reference source_sample_id: {sample_id}")
        seen.add(sample_id)
        split = _required_string(row, "split", where=f"reference.rows[{index}]")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"unexpected reference split: {split}")
        label_keys = {key for key in ("human_label", "label") if key in row}
        if len(label_keys) != 1:
            raise ValueError(
                f"reference.rows[{index}] must contain exactly one of human_label or label"
            )
        label = row[next(iter(label_keys))]
        if not isinstance(label, Mapping):
            raise ValueError(f"reference.rows[{index}].human_label must be an object")
        clean_label = validate_semantic_label(label, schema=schema)
        if split in REVIEWABLE_SPLITS:
            clean_rows.append(
                {
                    "source_sample_id": sample_id,
                    "split": split,
                    "human_label": clean_label,
                }
            )
    return clean_rows


def _reference_surface_rows(
    reference: Mapping[str, Any], *, schema: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the exact blinded surface after validating the human reference."""

    validated = _reference_rows(reference, schema=schema)
    raw_by_id = {row["source_sample_id"]: row for row in reference["rows"]}
    surfaces: list[dict[str, Any]] = []
    for index, clean in enumerate(validated):
        raw = raw_by_id[clean["source_sample_id"]]
        target_text = raw.get("target_text")
        if not isinstance(target_text, str) or not target_text:
            raise ValueError(f"reference.rows[{index}].target_text must be non-empty text")
        contexts: dict[str, str | None] = {}
        for field in ("submission_context", "parent_context"):
            value = raw.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"reference.rows[{index}].{field} must be text or null")
            contexts[field] = value
        surfaces.append(
            {
                "source_sample_id": clean["source_sample_id"],
                "target_text": target_text,
                **contexts,
            }
        )
    return surfaces


def _balanced_contiguous_chunks(rows: Sequence[Any], count: int) -> list[list[Any]]:
    if count <= 0:
        raise ValueError("shard count must be positive")
    if count > len(rows):
        raise ValueError("shard count cannot exceed reference row count")
    minimum, remainder = divmod(len(rows), count)
    chunks: list[list[Any]] = []
    offset = 0
    for index in range(count):
        size = minimum + (index < remainder)
        chunks.append(list(rows[offset : offset + size]))
        offset += size
    if offset != len(rows):
        raise RuntimeError("internal shard row conservation failure")
    return chunks


def make_review_input_shards(
    *,
    reference_path: Path,
    shard_count: int = 2,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
) -> list[dict[str, Any]]:
    schema = load_label_schema(schema_path)
    reference = _read_json_object(reference_path)
    surfaces = _reference_surface_rows(reference, schema=schema)
    chunks = _balanced_contiguous_chunks(surfaces, shard_count)
    binding = {
        "source_packet_sha256": file_sha256(reference_path),
        "rubric_sha256": file_sha256(rubric_path),
        "label_schema_sha256": file_sha256(schema_path),
    }
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "kind": REVIEW_INPUT_KIND,
            **binding,
            "blindness": BLINDNESS_CONTRACT,
            "shard_index": index,
            "shard_count": shard_count,
            "rows": chunk,
        }
        for index, chunk in enumerate(chunks)
    ]


def _review_input_path(
    *,
    review_root: Path,
    source_packet_sha256: str,
    shard_index: int,
    shard_count: int,
) -> Path:
    return (
        review_root
        / f"source={source_packet_sha256}"
        / "inputs"
        / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
    )


def prepare_review_inputs(
    *,
    reference_path: Path,
    shard_count: int = 2,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    review_root: Path = REVIEW_ROOT,
) -> dict[str, Any]:
    shards = make_review_input_shards(
        reference_path=reference_path,
        shard_count=shard_count,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    descriptors: list[dict[str, Any]] = []
    for shard in shards:
        path = _review_input_path(
            review_root=review_root,
            source_packet_sha256=shard["source_packet_sha256"],
            shard_index=shard["shard_index"],
            shard_count=shard["shard_count"],
        )
        _write_immutable_json(path, shard)
        descriptors.append(
            {
                "shard_index": shard["shard_index"],
                "rows": len(shard["rows"]),
                "sha256": file_sha256(path),
                "path": str(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{REVIEW_INPUT_KIND}-prepare-result",
        "source_packet_sha256": shards[0]["source_packet_sha256"],
        "shard_count": shard_count,
        "total_rows": sum(item["rows"] for item in descriptors),
        "shards": descriptors,
    }


def _validate_fragment(
    fragment: Mapping[str, Any],
    *,
    expected_shard: Mapping[str, Any],
    expected_shard_sha256: str,
    schema: Mapping[str, Any],
    where: str,
) -> tuple[str, list[dict[str, Any]]]:
    expected_fields = {
        "schema_version",
        "kind",
        "reviewer_id",
        "source_packet_sha256",
        "rubric_sha256",
        "label_schema_sha256",
        "blindness",
        "input_shard_sha256",
        "shard_index",
        "shard_count",
        "rows",
    }
    if set(fragment) != expected_fields:
        raise ValueError(f"{where} top-level fields differ from the frozen contract")
    if (
        fragment.get("schema_version") != SCHEMA_VERSION
        or fragment.get("kind") != REVIEW_FRAGMENT_KIND
    ):
        raise ValueError(f"{where} has an unsupported schema_version or kind")
    reviewer_id = _required_string(fragment, "reviewer_id", where=where)
    for field in ("source_packet_sha256", "rubric_sha256", "label_schema_sha256"):
        if fragment.get(field) != expected_shard[field]:
            raise ValueError(f"{where}.{field} digest drifted")
    if fragment.get("blindness") != BLINDNESS_CONTRACT:
        raise ValueError(f"{where}.blindness does not match the frozen contract")
    if fragment.get("input_shard_sha256") != expected_shard_sha256:
        raise ValueError(f"{where}.input_shard_sha256 digest drifted")
    if fragment.get("shard_index") != expected_shard["shard_index"]:
        raise ValueError(f"{where}.shard_index differs from its input")
    if fragment.get("shard_count") != expected_shard["shard_count"]:
        raise ValueError(f"{where}.shard_count differs from its input")

    rows = fragment.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{where}.rows must be an array")
    expected_ids = [row["source_sample_id"] for row in expected_shard["rows"]]
    clean_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"source_sample_id", "label"}:
            raise ValueError(
                f"{where}.rows[{index}] must contain only source_sample_id and label"
            )
        sample_id = _required_string(row, "source_sample_id", where=f"{where}.rows[{index}]")
        label = row.get("label")
        if not isinstance(label, Mapping):
            raise ValueError(f"{where}.rows[{index}].label must be an object")
        clean_rows.append(
            {
                "source_sample_id": sample_id,
                "label": validate_semantic_label(label, schema=schema),
            }
        )
    observed_ids = [row["source_sample_id"] for row in clean_rows]
    if observed_ids != expected_ids:
        if len(observed_ids) != len(set(observed_ids)):
            raise ValueError(f"{where} contains duplicate source_sample_id values")
        if set(observed_ids) != set(expected_ids):
            raise ValueError(f"{where} has missing or unexpected rows")
        raise ValueError(f"{where} row order differs from its input shard")
    return reviewer_id, clean_rows


def assemble_review(
    *,
    reference_path: Path,
    fragment_paths: Sequence[Path],
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    review_root: Path = REVIEW_ROOT,
) -> dict[str, Any]:
    if not fragment_paths:
        raise ValueError("at least one review fragment is required")
    first = _read_json_object(fragment_paths[0])
    shard_count = first.get("shard_count")
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count <= 0:
        raise ValueError("fragment shard_count must be a positive integer")
    if len(fragment_paths) != shard_count:
        raise ValueError("review fragment count differs from shard_count")

    schema = load_label_schema(schema_path)
    expected_shards = make_review_input_shards(
        reference_path=reference_path,
        shard_count=shard_count,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    fragments_by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in fragment_paths:
        fragment = _read_json_object(path)
        index = fragment.get("shard_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < shard_count:
            raise ValueError(f"invalid fragment shard_index: {path}")
        if index in fragments_by_index:
            raise ValueError(f"duplicate fragment shard_index: {index}")
        fragments_by_index[index] = (path, fragment)
    if set(fragments_by_index) != set(range(shard_count)):
        raise ValueError("review fragments do not cover every shard exactly once")

    reviewer_ids: set[str] = set()
    assembled_rows: list[dict[str, Any]] = []
    for index, expected_shard in enumerate(expected_shards):
        input_path = _review_input_path(
            review_root=review_root,
            source_packet_sha256=expected_shard["source_packet_sha256"],
            shard_index=index,
            shard_count=shard_count,
        )
        expected_input_sha256 = hashlib.sha256(_json_bytes(expected_shard)).hexdigest()
        if (
            not input_path.is_file()
            or file_sha256(input_path) != expected_input_sha256
            or _read_json_object(input_path) != expected_shard
        ):
            raise RuntimeError(f"immutable review input shard is missing or drifted: {index}")
        _, fragment = fragments_by_index[index]
        reviewer_id, rows = _validate_fragment(
            fragment,
            expected_shard=expected_shard,
            expected_shard_sha256=expected_input_sha256,
            schema=schema,
            where=f"fragment[{index}]",
        )
        reviewer_ids.add(reviewer_id)
        assembled_rows.extend(rows)
    if len(reviewer_ids) != 1:
        raise ValueError("all review fragments must have the same reviewer_id")
    reviewer_id = reviewer_ids.pop()

    expected_ids = [row["source_sample_id"] for shard in expected_shards for row in shard["rows"]]
    observed_ids = [row["source_sample_id"] for row in assembled_rows]
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("assembled review failed exact row conservation")
    review = {
        "schema_version": SCHEMA_VERSION,
        "kind": REVIEW_KIND,
        "reviewer_id": reviewer_id,
        "source_packet_sha256": expected_shards[0]["source_packet_sha256"],
        "rubric_sha256": expected_shards[0]["rubric_sha256"],
        "label_schema_sha256": expected_shards[0]["label_schema_sha256"],
        "blindness": BLINDNESS_CONTRACT,
        "rows": assembled_rows,
    }
    review_sha = hashlib.sha256(_json_bytes(review)).hexdigest()
    reviewer_key = hashlib.sha256(reviewer_id.encode("utf-8")).hexdigest()
    output_path = review_root / f"reviewer={reviewer_key}" / f"review-{review_sha}.json"
    _write_immutable_json(output_path, review)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{REVIEW_KIND}-assemble-result",
        "reviewer_id": reviewer_id,
        "source_packet_sha256": review["source_packet_sha256"],
        "shard_count": shard_count,
        "total_rows": len(assembled_rows),
        "review_sha256": file_sha256(output_path),
        "review_path": str(output_path),
    }


def _review_rows(
    review: Mapping[str, Any],
    *,
    expected_ids: Sequence[str],
    reference_packet_sha256: str,
    rubric_sha256: str,
    label_schema_sha256: str,
    schema: Mapping[str, Any],
    where: str,
) -> tuple[str, list[dict[str, Any]]]:
    expected_top_level = {
        "schema_version",
        "kind",
        "reviewer_id",
        "source_packet_sha256",
        "rubric_sha256",
        "label_schema_sha256",
        "blindness",
        "rows",
    }
    if set(review) != expected_top_level:
        raise ValueError(f"{where} top-level fields differ from the frozen contract")
    if review.get("schema_version") != SCHEMA_VERSION or review.get("kind") != REVIEW_KIND:
        raise ValueError(f"{where} has an unsupported schema_version or kind")
    reviewer_id = _required_string(review, "reviewer_id", where=where)
    for field, expected in (
        ("source_packet_sha256", reference_packet_sha256),
        ("rubric_sha256", rubric_sha256),
        ("label_schema_sha256", label_schema_sha256),
    ):
        actual = review.get(field)
        if not _is_sha256(actual) or actual != expected:
            raise ValueError(f"{where}.{field} digest drifted")
    if review.get("blindness") != BLINDNESS_CONTRACT:
        raise ValueError(f"{where}.blindness does not match the frozen contract")

    rows = review.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{where}.rows must be an array")
    clean_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"source_sample_id", "label"}:
            raise ValueError(
                f"{where}.rows[{index}] must contain only source_sample_id and label"
            )
        sample_id = _required_string(row, "source_sample_id", where=f"{where}.rows[{index}]")
        if sample_id in seen:
            raise ValueError(f"duplicate {where} source_sample_id: {sample_id}")
        seen.add(sample_id)
        label = row.get("label")
        if not isinstance(label, Mapping):
            raise ValueError(f"{where}.rows[{index}].label must be an object")
        clean_rows.append(
            {
                "source_sample_id": sample_id,
                "label": validate_semantic_label(label, schema=schema),
            }
        )

    observed_ids = [row["source_sample_id"] for row in clean_rows]
    if observed_ids != list(expected_ids):
        if set(observed_ids) != set(expected_ids):
            raise ValueError(f"{where} has missing or unexpected rows")
        raise ValueError(f"{where} row order differs from the reference packet")
    return reviewer_id, clean_rows


def make_prepare_manifest(
    *,
    reference_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
) -> dict[str, Any]:
    schema = load_label_schema(schema_path)
    reference = _read_json_object(reference_path)
    rows = _reference_rows(reference, schema=schema)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}-prepare-contract",
        "reference_packet_sha256": file_sha256(reference_path),
        "label_schema_sha256": file_sha256(schema_path),
        "rubric_sha256": file_sha256(rubric_path),
        "expected_rows": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "review_kind": REVIEW_KIND,
        "blindness": BLINDNESS_CONTRACT,
    }
    return {"contract": contract, "manifest_id": canonical_sha256(contract)}


def _pairwise_agreement(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("pairwise agreement requires row conservation")
    semantic_exact = 0
    relevance_exact = 0
    target_set_exact = 0
    target_stance_pairs_exact = 0
    shared_stance_agree = 0
    shared_stance_total = 0
    for left_label, right_label in zip(left, right, strict=True):
        semantic_exact += left_label == right_label
        relevance_exact += left_label["relevance"] == right_label["relevance"]
        left_pairs = {
            row["target"]: row["stance"] for row in left_label["target_stances"]
        }
        right_pairs = {
            row["target"]: row["stance"] for row in right_label["target_stances"]
        }
        target_set_exact += set(left_pairs) == set(right_pairs)
        target_stance_pairs_exact += left_pairs == right_pairs
        for target in set(left_pairs) & set(right_pairs):
            shared_stance_total += 1
            shared_stance_agree += left_pairs[target] == right_pairs[target]

    rows = len(left)
    return {
        "total": rows,
        "semantic_exact": {"agree": semantic_exact, "fraction": semantic_exact / rows},
        "relevance": {"agree": relevance_exact, "fraction": relevance_exact / rows},
        "target_set": {"agree": target_set_exact, "fraction": target_set_exact / rows},
        "target_stance_pairs": {
            "agree": target_stance_pairs_exact,
            "fraction": target_stance_pairs_exact / rows,
        },
        "stance_on_shared_targets": {
            "agree": shared_stance_agree,
            "denominator": shared_stance_total,
            "fraction": (
                shared_stance_agree / shared_stance_total if shared_stance_total else None
            ),
        },
    }


def _row_decision(
    human_label: Mapping[str, Any],
    reviewer_1_label: Mapping[str, Any],
    reviewer_2_label: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any] | None]:
    if human_label == reviewer_1_label == reviewer_2_label:
        return "eligible", "human_matches_both_reviewers", dict(human_label)
    if human_label == reviewer_1_label:
        return "eligible", "human_matches_reviewer_1", dict(human_label)
    if human_label == reviewer_2_label:
        return "eligible", "human_matches_reviewer_2", dict(human_label)
    if reviewer_1_label == reviewer_2_label:
        return "contentious", "reviewers_agree_against_human", None
    return "contentious", "no_strict_majority", None


def merge_consensus(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = load_label_schema(schema_path)
    reference = _read_json_object(reference_path)
    reference_rows = _reference_rows(reference, schema=schema)
    reference_sha = file_sha256(reference_path)
    schema_sha = file_sha256(schema_path)
    rubric_sha = file_sha256(rubric_path)
    expected_ids = [row["source_sample_id"] for row in reference_rows]

    reviews_with_paths: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for where, path in (
        ("reviewer_one", reviewer_one_path),
        ("reviewer_two", reviewer_two_path),
    ):
        review = _read_json_object(path)
        reviewer_id, rows = _review_rows(
            review,
            expected_ids=expected_ids,
            reference_packet_sha256=reference_sha,
            rubric_sha256=rubric_sha,
            label_schema_sha256=schema_sha,
            schema=schema,
            where=where,
        )
        reviews_with_paths.append((reviewer_id, path, rows))
    reviews_with_paths.sort(key=lambda item: item[0])
    if reviews_with_paths[0][0] == reviews_with_paths[1][0]:
        raise ValueError("reviewer_id values must be distinct")

    (reviewer_1_id, reviewer_1_path, reviewer_1_rows), (
        reviewer_2_id,
        reviewer_2_path,
        reviewer_2_rows,
    ) = reviews_with_paths

    consensus_rows: list[dict[str, Any]] = []
    for human_row, reviewer_1_row, reviewer_2_row in zip(
        reference_rows, reviewer_1_rows, reviewer_2_rows, strict=True
    ):
        human_label = human_row["human_label"]
        reviewer_1_label = reviewer_1_row["label"]
        reviewer_2_label = reviewer_2_row["label"]
        status, reason, chosen_label = _row_decision(
            human_label, reviewer_1_label, reviewer_2_label
        )
        consensus_rows.append(
            {
                "source_sample_id": human_row["source_sample_id"],
                "split": human_row["split"],
                "status": status,
                "reason": reason,
                "human_label": human_label,
                "reviewer_1_label": reviewer_1_label,
                "reviewer_2_label": reviewer_2_label,
                "chosen_label": chosen_label,
            }
        )

    input_binding = {
        "reference_packet_sha256": reference_sha,
        "label_schema_sha256": schema_sha,
        "rubric_sha256": rubric_sha,
        "reviewers": [
            {"reviewer_id": reviewer_1_id, "file_sha256": file_sha256(reviewer_1_path)},
            {"reviewer_id": reviewer_2_id, "file_sha256": file_sha256(reviewer_2_path)},
        ],
    }
    consensus_id = canonical_sha256(input_binding)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "consensus_id": consensus_id,
        **input_binding,
        "rows": consensus_rows,
    }

    status_counts = Counter(row["status"] for row in consensus_rows)
    reason_counts = Counter(row["reason"] for row in consensus_rows)
    split_aggregate: dict[str, Any] = {}
    for split in sorted({row["split"] for row in consensus_rows}):
        split_rows = [row for row in consensus_rows if row["split"] == split]
        eligible = sum(row["status"] == "eligible" for row in split_rows)
        split_aggregate[split] = {
            "total": len(split_rows),
            "eligible": eligible,
            "contentious": len(split_rows) - eligible,
            "eligible_fraction": eligible / len(split_rows) if split_rows else None,
        }
    chosen = [row["chosen_label"] for row in consensus_rows if row["chosen_label"] is not None]
    target_support: Counter[str] = Counter()
    stance_support: Counter[str] = Counter()
    for label in chosen:
        for pair in label["target_stances"]:
            target_support[pair["target"]] += 1
            stance_support[pair["stance"]] += 1

    human_labels = [row["human_label"] for row in consensus_rows]
    reviewer_1_labels = [row["reviewer_1_label"] for row in consensus_rows]
    reviewer_2_labels = [row["reviewer_2_label"] for row in consensus_rows]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}-receipt",
        "status": "merged",
        "consensus_id": consensus_id,
        "input_binding": input_binding,
        "total_rows": len(consensus_rows),
        "eligibility": {
            "eligible": status_counts["eligible"],
            "contentious": status_counts["contentious"],
            "eligible_fraction": status_counts["eligible"] / len(consensus_rows),
            "reason_counts": dict(sorted(reason_counts.items())),
            "by_split": split_aggregate,
        },
        "eligible_support": {
            "relevance": dict(sorted(Counter(label["relevance"] for label in chosen).items())),
            "targets": dict(sorted(target_support.items())),
            "stances": dict(sorted(stance_support.items())),
        },
        "pairwise_agreement": {
            "human__reviewer_1": _pairwise_agreement(human_labels, reviewer_1_labels),
            "human__reviewer_2": _pairwise_agreement(human_labels, reviewer_2_labels),
            "reviewer_1__reviewer_2": _pairwise_agreement(
                reviewer_1_labels, reviewer_2_labels
            ),
        },
        "consensus_artifact_sha256": hashlib.sha256(_json_bytes(artifact)).hexdigest(),
        "human_seeded": True,
        "independently_double_human_coded": False,
        "population_representative": False,
    }
    _assert_public_metadata_only(receipt)
    return artifact, receipt


def make_adjudication_input(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
) -> dict[str, Any]:
    """Build the exact blinded surface for unresolved consensus rows."""

    consensus, _ = merge_consensus(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    contentious_ids = [
        row["source_sample_id"]
        for row in consensus["rows"]
        if row["status"] == "contentious"
    ]
    if not contentious_ids:
        raise ValueError("consensus has no contentious rows to adjudicate")

    schema = load_label_schema(schema_path)
    reference = _read_json_object(reference_path)
    surface_by_id = {
        row["source_sample_id"]: row
        for row in _reference_surface_rows(reference, schema=schema)
    }
    rows = [surface_by_id[sample_id] for sample_id in contentious_ids]
    if [row["source_sample_id"] for row in rows] != contentious_ids:
        raise RuntimeError("adjudication input failed exact contentious-row conservation")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ADJUDICATION_INPUT_KIND,
        "consensus_id": consensus["consensus_id"],
        "consensus_artifact_sha256": hashlib.sha256(
            _json_bytes(consensus)
        ).hexdigest(),
        "source_packet_sha256": consensus["reference_packet_sha256"],
        "rubric_sha256": consensus["rubric_sha256"],
        "label_schema_sha256": consensus["label_schema_sha256"],
        "blindness": ADJUDICATION_BLINDNESS_CONTRACT,
        "rows": rows,
    }


def _adjudication_input_path(*, review_root: Path, packet: Mapping[str, Any]) -> Path:
    return (
        review_root
        / f"source={packet['source_packet_sha256']}"
        / "adjudication"
        / f"consensus={packet['consensus_id']}"
        / "input.json"
    )


def prepare_adjudication_input(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    review_root: Path = REVIEW_ROOT,
) -> dict[str, Any]:
    packet = make_adjudication_input(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    path = _adjudication_input_path(review_root=review_root, packet=packet)
    _write_immutable_json(path, packet)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{ADJUDICATION_INPUT_KIND}-prepare-result",
        "consensus_id": packet["consensus_id"],
        "total_rows": len(packet["rows"]),
        "input_sha256": file_sha256(path),
        "input_path": str(path),
    }


def _validate_adjudication_output(
    output: Mapping[str, Any],
    *,
    expected_input: Mapping[str, Any],
    expected_input_sha256: str,
    reviewer_ids: set[str],
    schema: Mapping[str, Any],
    where: str,
) -> tuple[str, list[dict[str, Any]]]:
    expected_fields = {
        "schema_version",
        "kind",
        "adjudicator_id",
        "adjudicator_model",
        "input_sha256",
        "consensus_id",
        "source_packet_sha256",
        "rubric_sha256",
        "label_schema_sha256",
        "blindness",
        "rows",
    }
    if set(output) != expected_fields:
        raise ValueError(f"{where} top-level fields differ from the frozen contract")
    if (
        output.get("schema_version") != SCHEMA_VERSION
        or output.get("kind") != ADJUDICATION_OUTPUT_KIND
    ):
        raise ValueError(f"{where} has an unsupported schema_version or kind")
    adjudicator_id = _required_string(output, "adjudicator_id", where=where)
    if adjudicator_id in reviewer_ids:
        raise ValueError("adjudicator must be distinct from both reviewers")
    if output.get("adjudicator_model") != SOL_ADJUDICATOR_MODEL:
        raise ValueError(
            f"{where}.adjudicator_model must be the frozen Sol model "
            f"{SOL_ADJUDICATOR_MODEL!r}"
        )
    for field, expected in (
        ("input_sha256", expected_input_sha256),
        ("consensus_id", expected_input["consensus_id"]),
        ("source_packet_sha256", expected_input["source_packet_sha256"]),
        ("rubric_sha256", expected_input["rubric_sha256"]),
        ("label_schema_sha256", expected_input["label_schema_sha256"]),
    ):
        if output.get(field) != expected:
            raise ValueError(f"{where}.{field} binding drifted")
    if output.get("blindness") != ADJUDICATION_BLINDNESS_CONTRACT:
        raise ValueError(f"{where}.blindness does not match the frozen contract")

    rows = output.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{where}.rows must be an array")
    expected_ids = [row["source_sample_id"] for row in expected_input["rows"]]
    clean_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"source_sample_id", "label"}:
            raise ValueError(
                f"{where}.rows[{index}] must contain only source_sample_id and label"
            )
        sample_id = _required_string(row, "source_sample_id", where=f"{where}.rows[{index}]")
        label = row.get("label")
        if not isinstance(label, Mapping):
            raise ValueError(f"{where}.rows[{index}].label must be an object")
        clean_rows.append(
            {
                "source_sample_id": sample_id,
                "label": validate_semantic_label(label, schema=schema),
            }
        )
    observed_ids = [row["source_sample_id"] for row in clean_rows]
    if observed_ids != expected_ids:
        if len(observed_ids) != len(set(observed_ids)):
            raise ValueError(f"{where} contains duplicate source_sample_id values")
        if set(observed_ids) != set(expected_ids):
            raise ValueError(f"{where} differs from the exact contentious row set")
        raise ValueError(f"{where} row order differs from the adjudication input")
    return adjudicator_id, clean_rows


def assemble_adjudication(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    adjudicator_output_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    review_root: Path = REVIEW_ROOT,
) -> dict[str, Any]:
    expected_input = make_adjudication_input(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    input_path = _adjudication_input_path(review_root=review_root, packet=expected_input)
    expected_input_sha256 = hashlib.sha256(_json_bytes(expected_input)).hexdigest()
    if (
        not input_path.is_file()
        or file_sha256(input_path) != expected_input_sha256
        or _read_json_object(input_path) != expected_input
    ):
        raise RuntimeError("immutable adjudication input is missing or drifted")

    consensus, _ = merge_consensus(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    schema = load_label_schema(schema_path)
    output = _read_json_object(adjudicator_output_path)
    adjudicator_id, rows = _validate_adjudication_output(
        output,
        expected_input=expected_input,
        expected_input_sha256=expected_input_sha256,
        reviewer_ids={row["reviewer_id"] for row in consensus["reviewers"]},
        schema=schema,
        where="adjudicator_output",
    )
    assembled = {**output, "rows": rows}
    adjudication_sha256 = hashlib.sha256(_json_bytes(assembled)).hexdigest()
    adjudicator_key = hashlib.sha256(adjudicator_id.encode("utf-8")).hexdigest()
    output_path = (
        input_path.parent
        / f"adjudicator={adjudicator_key}"
        / f"adjudication-{adjudication_sha256}.json"
    )
    _write_immutable_json(output_path, assembled)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{ADJUDICATION_OUTPUT_KIND}-assemble-result",
        "consensus_id": expected_input["consensus_id"],
        "adjudicator_id": adjudicator_id,
        "adjudicator_model": SOL_ADJUDICATOR_MODEL,
        "total_rows": len(rows),
        "adjudication_sha256": file_sha256(output_path),
        "adjudication_path": str(output_path),
    }


def _paths(
    *, consensus_id: str, private_root: Path, public_root: Path
) -> tuple[Path, Path]:
    return (
        private_root / f"consensus-{consensus_id}.json",
        public_root / f"run={consensus_id}" / "receipt.json",
    )


def prepare(
    *,
    reference_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    private_root: Path = PRIVATE_ROOT,
    public_root: Path = PUBLIC_ROOT,
) -> dict[str, Any]:
    manifest = make_prepare_manifest(
        reference_path=reference_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    manifest_id = manifest["manifest_id"]
    _write_immutable_json(private_root / f"prepare-{manifest_id}.json", manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "manifest_id": manifest_id,
        "contract": manifest["contract"],
        "kind": f"{KIND}-prepare-receipt",
    }
    _assert_public_metadata_only(receipt)
    _write_immutable_json(public_root / f"prepare={manifest_id}" / "receipt.json", receipt)
    return receipt


def merge(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    private_root: Path = PRIVATE_ROOT,
    public_root: Path = PUBLIC_ROOT,
) -> dict[str, Any]:
    artifact, receipt = merge_consensus(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    private_path, public_path = _paths(
        consensus_id=artifact["consensus_id"],
        private_root=private_root,
        public_root=public_root,
    )
    _write_immutable_json(private_path, artifact)
    _write_immutable_json(public_path, receipt)
    return receipt


def build_final_proxy(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    adjudication_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    review_root: Path = REVIEW_ROOT,
    expected_rows: int = EXPECTED_FINAL_ROWS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows <= 0
    ):
        raise ValueError("expected_rows must be a positive integer")
    consensus, _ = merge_consensus(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    if len(consensus["rows"]) != expected_rows:
        raise RuntimeError(
            f"final proxy requires exactly {expected_rows} rows; "
            f"observed {len(consensus['rows'])}"
        )

    expected_input = make_adjudication_input(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    input_path = _adjudication_input_path(review_root=review_root, packet=expected_input)
    expected_input_sha256 = hashlib.sha256(_json_bytes(expected_input)).hexdigest()
    if (
        not input_path.is_file()
        or file_sha256(input_path) != expected_input_sha256
        or _read_json_object(input_path) != expected_input
    ):
        raise RuntimeError("immutable adjudication input is missing or drifted")

    adjudication = _read_json_object(adjudication_path)
    schema = load_label_schema(schema_path)
    adjudicator_id, adjudication_rows = _validate_adjudication_output(
        adjudication,
        expected_input=expected_input,
        expected_input_sha256=expected_input_sha256,
        reviewer_ids={row["reviewer_id"] for row in consensus["reviewers"]},
        schema=schema,
        where="adjudication",
    )
    adjudication_sha256 = hashlib.sha256(_json_bytes(adjudication)).hexdigest()
    adjudicator_key = hashlib.sha256(adjudicator_id.encode("utf-8")).hexdigest()
    expected_adjudication_path = (
        input_path.parent
        / f"adjudicator={adjudicator_key}"
        / f"adjudication-{adjudication_sha256}.json"
    )
    if adjudication_path.resolve() != expected_adjudication_path.resolve():
        raise RuntimeError("finalisation requires the assembled immutable adjudication output")
    if file_sha256(adjudication_path) != adjudication_sha256:
        raise RuntimeError("assembled adjudication output bytes drifted")

    adjudication_by_id = {
        row["source_sample_id"]: row["label"] for row in adjudication_rows
    }
    final_rows: list[dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    for row in consensus["rows"]:
        if row["status"] == "eligible":
            label = row["chosen_label"]
            resolution = "strict_human_supported_majority"
        else:
            label = adjudication_by_id[row["source_sample_id"]]
            resolution = "blinded_sol_adjudication"
        resolution_counts[resolution] += 1
        final_rows.append(
            {
                "source_sample_id": row["source_sample_id"],
                "split": row["split"],
                "label": label,
                "resolution": resolution,
                "consensus_reason": row["reason"],
            }
        )
    if len(final_rows) != expected_rows or any(row["label"] is None for row in final_rows):
        raise RuntimeError("final proxy failed complete row and label conservation")

    input_binding = {
        "reference_packet_sha256": consensus["reference_packet_sha256"],
        "label_schema_sha256": consensus["label_schema_sha256"],
        "rubric_sha256": consensus["rubric_sha256"],
        "consensus_id": consensus["consensus_id"],
        "consensus_artifact_sha256": expected_input["consensus_artifact_sha256"],
        "reviewers": consensus["reviewers"],
        "adjudicator": {
            "adjudicator_id": adjudicator_id,
            "model": SOL_ADJUDICATOR_MODEL,
            "input_sha256": expected_input_sha256,
            "output_sha256": adjudication_sha256,
        },
    }
    proxy_id = canonical_sha256(input_binding)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": FINAL_KIND,
        "proxy_id": proxy_id,
        "evidence_status": "model-assisted/Sol-adjudicated development evidence",
        "input_binding": input_binding,
        "rows": final_rows,
    }
    split_counts = Counter(row["split"] for row in final_rows)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{FINAL_KIND}-receipt",
        "status": "finalised",
        "proxy_id": proxy_id,
        "input_binding": input_binding,
        "total_rows": len(final_rows),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "private_proxy_sha256": hashlib.sha256(_json_bytes(artifact)).hexdigest(),
        "evidence_status": "model-assisted/Sol-adjudicated development evidence",
        "human_gold": False,
        "independent_human_evaluation": False,
        "final_thesis_claims_require_independent_human_evaluation": True,
        "intended_use": "development proxy only",
    }
    _assert_public_metadata_only(receipt)
    return artifact, receipt


def _final_paths(
    *, proxy_id: str, private_root: Path, public_root: Path
) -> tuple[Path, Path]:
    return (
        private_root / "final" / f"proxy-{proxy_id}.json",
        public_root / f"final={proxy_id}" / "receipt.json",
    )


def finalise(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    adjudication_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    review_root: Path = REVIEW_ROOT,
    private_root: Path = PRIVATE_ROOT,
    public_root: Path = PUBLIC_ROOT,
    expected_rows: int = EXPECTED_FINAL_ROWS,
) -> dict[str, Any]:
    artifact, receipt = build_final_proxy(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        adjudication_path=adjudication_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
        review_root=review_root,
        expected_rows=expected_rows,
    )
    private_path, public_path = _final_paths(
        proxy_id=artifact["proxy_id"],
        private_root=private_root,
        public_root=public_root,
    )
    _write_immutable_json(private_path, artifact)
    _write_immutable_json(public_path, receipt)
    return receipt


def validate_final(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    adjudication_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    review_root: Path = REVIEW_ROOT,
    private_root: Path = PRIVATE_ROOT,
    public_root: Path = PUBLIC_ROOT,
    expected_rows: int = EXPECTED_FINAL_ROWS,
) -> dict[str, Any]:
    expected_artifact, expected_receipt = build_final_proxy(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        adjudication_path=adjudication_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
        review_root=review_root,
        expected_rows=expected_rows,
    )
    private_path, public_path = _final_paths(
        proxy_id=expected_artifact["proxy_id"],
        private_root=private_root,
        public_root=public_root,
    )
    if not private_path.is_file() or not public_path.is_file():
        raise RuntimeError("expected immutable final proxy artefacts are missing")
    if _read_json_object(private_path) != expected_artifact:
        raise RuntimeError("private final proxy artefact drifted")
    receipt = _read_json_object(public_path)
    if receipt != expected_receipt:
        raise RuntimeError("public final proxy receipt drifted")
    _assert_public_metadata_only(receipt)
    return {**receipt, "status": "validated"}


def validate(
    *,
    reference_path: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    schema_path: Path = SCHEMA_PATH,
    rubric_path: Path = RUBRIC_PATH,
    private_root: Path = PRIVATE_ROOT,
    public_root: Path = PUBLIC_ROOT,
) -> dict[str, Any]:
    expected_artifact, expected_receipt = merge_consensus(
        reference_path=reference_path,
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        schema_path=schema_path,
        rubric_path=rubric_path,
    )
    private_path, public_path = _paths(
        consensus_id=expected_artifact["consensus_id"],
        private_root=private_root,
        public_root=public_root,
    )
    if not private_path.is_file() or not public_path.is_file():
        raise RuntimeError("expected immutable consensus artefacts are missing")
    if _read_json_object(private_path) != expected_artifact:
        raise RuntimeError("private consensus artefact drifted")
    receipt = _read_json_object(public_path)
    if receipt != expected_receipt:
        raise RuntimeError("public consensus receipt drifted")
    _assert_public_metadata_only(receipt)
    return {**receipt, "status": "validated"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "prepare",
            "prepare-review-inputs",
            "assemble-review",
            "merge",
            "validate",
            "prepare-adjudication-input",
            "assemble-adjudication",
            "finalise",
            "validate-final",
        ),
        required=True,
    )
    parser.add_argument("--reference-packet", type=Path, required=True)
    parser.add_argument("--reviewer-one", type=Path)
    parser.add_argument("--reviewer-two", type=Path)
    parser.add_argument("--adjudicator-output", type=Path)
    parser.add_argument("--review-fragment", action="append", type=Path, default=[])
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--schema-path", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--rubric-path", type=Path, default=RUBRIC_PATH)
    parser.add_argument("--private-root", type=Path, default=PRIVATE_ROOT)
    parser.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    parser.add_argument("--public-root", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_FINAL_ROWS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "prepare-review-inputs":
        result = prepare_review_inputs(
            reference_path=args.reference_packet,
            shard_count=args.shards,
            schema_path=args.schema_path,
            rubric_path=args.rubric_path,
            review_root=args.review_root,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.action == "assemble-review":
        result = assemble_review(
            reference_path=args.reference_packet,
            fragment_paths=args.review_fragment,
            schema_path=args.schema_path,
            rubric_path=args.rubric_path,
            review_root=args.review_root,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.action in {
        "prepare-adjudication-input",
        "assemble-adjudication",
        "finalise",
        "validate-final",
    }:
        if args.reviewer_one is None or args.reviewer_two is None:
            raise SystemExit("--reviewer-one and --reviewer-two are required")
        adjudication_common = {
            "reference_path": args.reference_packet,
            "reviewer_one_path": args.reviewer_one,
            "reviewer_two_path": args.reviewer_two,
            "schema_path": args.schema_path,
            "rubric_path": args.rubric_path,
            "review_root": args.review_root,
        }
        if args.action == "prepare-adjudication-input":
            result = prepare_adjudication_input(**adjudication_common)
        else:
            if args.adjudicator_output is None:
                raise SystemExit("--adjudicator-output is required")
            if args.action == "assemble-adjudication":
                result = assemble_adjudication(
                    **adjudication_common,
                    adjudicator_output_path=args.adjudicator_output,
                )
            else:
                final_common = {
                    **adjudication_common,
                    "adjudication_path": args.adjudicator_output,
                    "private_root": args.private_root,
                    "public_root": args.public_root,
                    "expected_rows": args.expected_rows,
                }
                result = (
                    finalise if args.action == "finalise" else validate_final
                )(**final_common)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    common = {
        "reference_path": args.reference_packet,
        "schema_path": args.schema_path,
        "rubric_path": args.rubric_path,
        "private_root": args.private_root,
        "public_root": args.public_root,
    }
    if args.action == "prepare":
        result = prepare(**common)
    else:
        if args.reviewer_one is None or args.reviewer_two is None:
            raise SystemExit("--reviewer-one and --reviewer-two are required")
        result = (merge if args.action == "merge" else validate)(
            **common,
            reviewer_one_path=args.reviewer_one,
            reviewer_two_path=args.reviewer_two,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
