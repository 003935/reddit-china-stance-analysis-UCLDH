"""Build and validate the frozen ModernBERT acquisition-ladder manifest.

The manifest is private because it contains opaque sample and thread identifiers.  It contains
no Reddit text.  Fold construction follows the preregistered ModernBERT experiment exactly:
three iterative multilabel-stratified ten-fold assignments, with folds 0--1 and 0--4 defining
the nested 2k and 5k subsets respectively.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = "1.0.0"
KIND = "modernbert-private-split-manifest-v1"
EXPECTED_ROWS = 10_000
N_SPLITS = 10
NOMINAL_FOLD_SIZE = 1_000
LADDER_SEEDS = (101, 202, 303)
OPTIMISER_SEEDS = {101: 47, 202: 61, 303: 89}
EXPECTED_FOLD_SIZES = {
    101: (1_001, 1_000, 1_000, 999, 1_000, 999, 1_001, 1_000, 999, 1_001),
    202: (1_001, 1_000, 1_002, 997, 998, 1_001, 999, 1_000, 1_001, 1_001),
    303: (1_000, 1_000, 1_001, 1_000, 1_000, 998, 1_001, 999, 1_000, 1_001),
}
BUDGET_FOLDS = {"2k": (0, 1), "5k": (0, 1, 2, 3, 4), "10k": tuple(range(10))}
TARGETS = ("china_general", "government_ccp", "people_culture", "other")
RELEVANCE = ("material", "not_material", "unclear")
STANCES = ("negative", "mixed", "no_directed_stance", "positive", "unclear")
CHRONOLOGICAL_SPLITS = {
    "train": {"years": (2020, 2021, 2022, 2023), "expected_rows": 7_665},
    "calibration": {"years": (2024,), "expected_rows": 1_119},
    "test": {"years": (2025,), "expected_rows": 1_216},
}
FROZEN_INPUT_BINDING = {
    "dataset_repo_id": "aisafteycommons/reddit-china-stance-10k-sol-v1",
    "dataset_revision": "b0fa1c3e5e3caf4dc3fb43b9024858364bc44834",
    "parquet_sha256": "f81146988ef504f81fb77b36333084e5ea10ab77908568d5be26bdba296a5bba",
    "teacher_run_id": "bc7854a60140b3584d5cdeb5513b07ff413253fd59cb84877c278de21702afa4",
}


class Splitter(Protocol):
    """Structural protocol for iterative-stratification's splitter."""

    def split(
        self, x: Sequence[int], y: Sequence[Sequence[int]]
    ) -> Any:  # pragma: no cover - protocol only
        ...


SplitterFactory = Callable[..., Splitter]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_string(row: Mapping[str, Any], field: str, *, sample: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{sample}.{field} must be a non-empty string")
    return value


def _feature_name(dimension: str, value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{dimension}={encoded}"


def _normalise_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    where = f"rows[{index}]"
    sample_id = _required_string(row, "sample_id", sample=where)
    clean: dict[str, Any] = {
        "sample_id": sample_id,
        "thread_id": _required_string(row, "thread_id", sample=where),
        "subreddit": _required_string(row, "subreddit", sample=where),
        "content_type": _required_string(row, "content_type", sample=where),
        "retrieval_mode": _required_string(row, "retrieval_mode", sample=where),
        "relevance": _required_string(row, "relevance", sample=where),
    }
    year = row.get("year")
    if not isinstance(year, int) or isinstance(year, bool):
        raise ValueError(f"{where}.year must be an integer")
    clean["year"] = year
    if clean["relevance"] not in RELEVANCE:
        raise ValueError(f"{where}.relevance is outside the frozen label set")

    present_targets = 0
    for target in TARGETS:
        present_field = f"has_target_{target}"
        present = row.get(present_field)
        if not isinstance(present, bool):
            raise ValueError(f"{where}.{present_field} must be boolean")
        stance = row.get(f"stance_{target}")
        if present:
            present_targets += 1
            if stance not in STANCES:
                raise ValueError(f"{where}.stance_{target} is missing or unsupported")
        elif stance is not None:
            raise ValueError(f"{where}.stance_{target} must be null when target is absent")
        clean[present_field] = present
        clean[f"stance_{target}"] = stance

    if clean["relevance"] == "material" and present_targets == 0:
        raise ValueError(f"{where}: material relevance requires a target")
    if clean["relevance"] != "material" and present_targets:
        raise ValueError(f"{where}: non-material relevance cannot have a target")
    return clean


def _row_features(row: Mapping[str, Any]) -> tuple[str, ...]:
    features = {
        _feature_name("year", row["year"]),
        _feature_name("subreddit", row["subreddit"]),
        _feature_name("content_type", row["content_type"]),
        _feature_name("retrieval_mode", row["retrieval_mode"]),
        _feature_name("relevance", row["relevance"]),
    }
    for target in TARGETS:
        if row[f"has_target_{target}"]:
            features.add(_feature_name("target", target))
            features.add(_feature_name("target_stance", [target, row[f"stance_{target}"]]))
    return tuple(sorted(features))


def encode_features(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...], list[list[int]], Counter[str]]:
    """Normalise rows and build the exact binary feature matrix in sample-ID order."""

    clean_rows = [_normalise_row(row, index=index) for index, row in enumerate(rows)]
    clean_rows.sort(key=lambda row: row["sample_id"])
    sample_ids = [row["sample_id"] for row in clean_rows]
    thread_ids = [row["thread_id"] for row in clean_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")
    if len(set(thread_ids)) != len(thread_ids):
        raise ValueError("thread_id values must be unique; thread-level leakage is not allowed")

    active = [_row_features(row) for row in clean_rows]
    feature_names = tuple(sorted({feature for values in active for feature in values}))
    matrix = [
        [1 if feature in values else 0 for feature in feature_names]
        for values in (set(row_features) for row_features in active)
    ]
    support = Counter[str]()
    for values in active:
        support.update(values)
    return clean_rows, feature_names, matrix, support


def _default_splitter() -> tuple[SplitterFactory, str]:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "ModernBERT split construction requires the pinned "
            "'iterative-stratification' package"
        ) from exc
    try:
        version = importlib.metadata.version("iterative-stratification")
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("cannot resolve iterative-stratification package version") from exc
    return MultilabelStratifiedKFold, version


def _fold_assignments(
    matrix: Sequence[Sequence[int]], *, seed: int, splitter_factory: SplitterFactory
) -> list[int]:
    splitter = splitter_factory(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    assignments = [-1] * len(matrix)
    split_count = 0
    for fold_id, split in enumerate(splitter.split(list(range(len(matrix))), matrix)):
        if fold_id >= N_SPLITS:
            raise RuntimeError("splitter emitted more than ten folds")
        try:
            _, test_indices = split
        except (TypeError, ValueError) as exc:
            raise RuntimeError("splitter must emit (train_indices, test_indices) pairs") from exc
        for raw_index in test_indices:
            index = int(raw_index)
            if index < 0 or index >= len(matrix):
                raise RuntimeError("splitter emitted an out-of-range test index")
            if assignments[index] != -1:
                raise RuntimeError("splitter assigned a row to multiple folds")
            assignments[index] = fold_id
        split_count += 1
    if split_count != N_SPLITS or any(fold < 0 for fold in assignments):
        raise RuntimeError("splitter did not assign every row exactly once across ten folds")
    observed = tuple(Counter(assignments)[fold] for fold in range(N_SPLITS))
    if observed != EXPECTED_FOLD_SIZES[seed]:
        raise RuntimeError(
            f"fold sizes for seed {seed} drifted from the frozen vector: {observed}"
        )
    return assignments


def _sample_digest(sample_ids: Sequence[str]) -> str:
    return canonical_sha256(sorted(sample_ids))


def _chronological_split(year: int) -> str:
    for name, contract in CHRONOLOGICAL_SPLITS.items():
        if year in contract["years"]:
            return name
    raise ValueError(f"year {year} is outside the frozen 2020--2025 chronological split")


def build_split_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    splitter_factory: SplitterFactory | None = None,
    splitter_version: str | None = None,
    input_binding: Mapping[str, Any] = FROZEN_INPUT_BINDING,
) -> dict[str, Any]:
    """Construct the content-addressed private split manifest."""

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected exactly {EXPECTED_ROWS} teacher rows, got {len(rows)}")
    if splitter_factory is None:
        splitter_factory, detected_version = _default_splitter()
        if splitter_version is not None and splitter_version != detected_version:
            raise ValueError("declared splitter version does not match installed package")
        splitter_version = detected_version
    elif not isinstance(splitter_version, str) or not splitter_version:
        raise ValueError("splitter_version is required when injecting a splitter_factory")

    binding = dict(input_binding)
    required_binding = set(FROZEN_INPUT_BINDING)
    if set(binding) != required_binding:
        raise ValueError(f"input_binding must contain exactly {sorted(required_binding)}")
    for field, value in binding.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"input_binding.{field} must be a non-empty string")
    if not _is_git_revision(binding["dataset_revision"]):
        raise ValueError("input_binding.dataset_revision must be a full Git or SHA-256 revision")
    if not _is_sha256(binding["parquet_sha256"]):
        raise ValueError("input_binding.parquet_sha256 must be SHA-256")
    if not _is_sha256(binding["teacher_run_id"]):
        raise ValueError("input_binding.teacher_run_id must be SHA-256")
    if binding != FROZEN_INPUT_BINDING:
        raise ValueError("input_binding drifted from the frozen teacher dataset")

    clean_rows, feature_names, matrix, feature_support = encode_features(rows)
    assignments = {
        seed: _fold_assignments(matrix, seed=seed, splitter_factory=splitter_factory)
        for seed in LADDER_SEEDS
    }
    manifest_rows: list[dict[str, Any]] = []
    chronological_ids: dict[str, list[str]] = {name: [] for name in CHRONOLOGICAL_SPLITS}
    for index, row in enumerate(clean_rows):
        chronological = _chronological_split(row["year"])
        chronological_ids[chronological].append(row["sample_id"])
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "thread_id": row["thread_id"],
                "year": row["year"],
                "chronological_split": chronological,
                "feature_sha256": canonical_sha256(_row_features(row)),
                "folds": {str(seed): assignments[seed][index] for seed in LADDER_SEEDS},
            }
        )

    features = [
        {"name": feature, "corpus_support": feature_support[feature]}
        for feature in feature_names
    ]
    ladders: dict[str, Any] = {}
    for seed in LADDER_SEEDS:
        budget_metadata: dict[str, Any] = {}
        previous: set[str] = set()
        for budget, folds in BUDGET_FOLDS.items():
            selected_indices = {
                index for index, fold in enumerate(assignments[seed]) if fold in folds
            }
            selected_ids = [clean_rows[index]["sample_id"] for index in sorted(selected_indices)]
            current = set(selected_ids)
            if not previous.issubset(current):
                raise RuntimeError(f"ladder {seed} is not nested at budget {budget}")
            previous = current
            selected_support = Counter[str]()
            for index in selected_indices:
                selected_support.update(_row_features(clean_rows[index]))
            missing_supported = sorted(
                feature
                for feature, count in feature_support.items()
                if count >= 25 and selected_support[feature] == 0
            )
            if budget != "10k" and missing_supported:
                raise RuntimeError(
                    f"ladder {seed} {budget} omits corpus features with support >=25: "
                    + ", ".join(missing_supported)
                )
            budget_metadata[budget] = {
                "folds": list(folds),
                "row_count": len(selected_ids),
                "sample_ids_sha256": _sample_digest(selected_ids),
                "feature_support": {
                    feature: selected_support[feature] for feature in feature_names
                },
            }
        expected_budget_sizes = {
            budget: sum(EXPECTED_FOLD_SIZES[seed][fold] for fold in folds)
            for budget, folds in BUDGET_FOLDS.items()
        }
        if {
            key: value["row_count"] for key, value in budget_metadata.items()
        } != expected_budget_sizes:
            raise RuntimeError(f"ladder {seed} does not conserve the registered budgets")
        ladders[str(seed)] = {
            "acquisition_seed": seed,
            "optimiser_seed": OPTIMISER_SEEDS[seed],
            "budgets": budget_metadata,
        }

    chronological: dict[str, Any] = {}
    for name, contract in CHRONOLOGICAL_SPLITS.items():
        sample_ids = chronological_ids[name]
        if len(sample_ids) != contract["expected_rows"]:
            raise RuntimeError(
                f"chronological {name} split expected {contract['expected_rows']} rows, "
                f"got {len(sample_ids)}"
            )
        chronological[name] = {
            "years": list(contract["years"]),
            "row_count": len(sample_ids),
            "sample_ids_sha256": _sample_digest(sample_ids),
        }

    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "private": True,
        "contains_reddit_text": False,
        "input": binding,
        "contract": {
            "row_count": EXPECTED_ROWS,
            "thread_count": EXPECTED_ROWS,
            "sort_key": "sample_id",
            "splitter": "iterstrat.ml_stratifiers.MultilabelStratifiedKFold",
            "splitter_package": "iterative-stratification",
            "splitter_version": splitter_version,
            "n_splits": N_SPLITS,
            "shuffle": True,
            "nominal_fold_size": NOMINAL_FOLD_SIZE,
            "expected_fold_sizes": {
                str(seed): list(EXPECTED_FOLD_SIZES[seed]) for seed in LADDER_SEEDS
            },
            "ladder_seeds": list(LADDER_SEEDS),
            "minimum_guarded_feature_support": 25,
        },
        "feature_encoding": {
            "names": list(feature_names),
            "corpus_support": features,
            "matrix_sha256": canonical_sha256(matrix),
        },
        "ladders": ladders,
        "chronological": chronological,
        "rows": manifest_rows,
    }
    return {**body, "manifest_id": canonical_sha256(body)}


def validate_split_manifest(
    manifest: Mapping[str, Any], *, source_rows: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Validate IDs, hashes, conservation, nestedness and optional source-row binding."""

    clean = dict(manifest)
    manifest_id = clean.pop("manifest_id", None)
    if not _is_sha256(manifest_id) or manifest_id != canonical_sha256(clean):
        raise ValueError("split manifest_id does not match its canonical body")
    if clean.get("schema_version") != SCHEMA_VERSION or clean.get("kind") != KIND:
        raise ValueError("split manifest contract drifted")
    if clean.get("private") is not True or clean.get("contains_reddit_text") is not False:
        raise ValueError("split manifest privacy declaration drifted")
    if clean.get("input") != FROZEN_INPUT_BINDING:
        raise ValueError("split manifest input binding drifted")
    contract = clean.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("split manifest contract is missing")
    expected_contract = {
        "row_count": EXPECTED_ROWS,
        "thread_count": EXPECTED_ROWS,
        "sort_key": "sample_id",
        "splitter": "iterstrat.ml_stratifiers.MultilabelStratifiedKFold",
        "splitter_package": "iterative-stratification",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "nominal_fold_size": NOMINAL_FOLD_SIZE,
        "expected_fold_sizes": {
            str(seed): list(EXPECTED_FOLD_SIZES[seed]) for seed in LADDER_SEEDS
        },
        "ladder_seeds": list(LADDER_SEEDS),
        "minimum_guarded_feature_support": 25,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"split manifest contract.{field} drifted")
    if not isinstance(contract.get("splitter_version"), str) or not contract["splitter_version"]:
        raise ValueError("split manifest splitter version is missing")

    rows = clean.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("split manifest must contain exactly 10,000 row assignments")
    sample_ids: list[str] = []
    thread_ids: list[str] = []
    assignments: dict[int, list[int]] = {seed: [] for seed in LADDER_SEEDS}
    chronological_ids: dict[str, list[str]] = {name: [] for name in CHRONOLOGICAL_SPLITS}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"split manifest rows[{index}] must be an object")
        sample_id = _required_string(row, "sample_id", sample=f"rows[{index}]")
        thread_id = _required_string(row, "thread_id", sample=f"rows[{index}]")
        sample_ids.append(sample_id)
        thread_ids.append(thread_id)
        if not _is_sha256(row.get("feature_sha256")):
            raise ValueError(f"rows[{index}].feature_sha256 is invalid")
        chronological = row.get("chronological_split")
        year = row.get("year")
        if chronological != _chronological_split(year):
            raise ValueError(f"rows[{index}] chronological assignment drifted")
        chronological_ids[chronological].append(sample_id)
        folds = row.get("folds")
        if not isinstance(folds, Mapping) or set(folds) != {str(seed) for seed in LADDER_SEEDS}:
            raise ValueError(f"rows[{index}].folds drifted")
        for seed in LADDER_SEEDS:
            fold = folds[str(seed)]
            if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(N_SPLITS):
                raise ValueError(f"rows[{index}].folds[{seed}] is invalid")
            assignments[seed].append(fold)
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != EXPECTED_ROWS:
        raise ValueError("manifest rows must have unique sample IDs in sorted order")
    if len(set(thread_ids)) != EXPECTED_ROWS:
        raise ValueError("manifest thread IDs are not unique")

    feature_encoding = clean.get("feature_encoding")
    if not isinstance(feature_encoding, Mapping):
        raise ValueError("feature encoding is missing")
    feature_names = feature_encoding.get("names")
    supports = feature_encoding.get("corpus_support")
    if not isinstance(feature_names, list) or feature_names != sorted(set(feature_names)):
        raise ValueError("feature names must be unique and sorted")
    if not isinstance(supports, list):
        raise ValueError("feature supports are missing")
    support_by_name: dict[str, int] = {}
    for item in supports:
        if not isinstance(item, Mapping) or item.get("name") not in feature_names:
            raise ValueError("invalid corpus feature support entry")
        count = item.get("corpus_support")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("invalid corpus feature support count")
        support_by_name[str(item["name"])] = count
    if set(support_by_name) != set(feature_names):
        raise ValueError("corpus feature support entries do not conserve feature names")

    ladders = clean.get("ladders")
    if not isinstance(ladders, Mapping) or set(ladders) != {str(seed) for seed in LADDER_SEEDS}:
        raise ValueError("split manifest ladders drifted")
    for seed in LADDER_SEEDS:
        fold_sizes = tuple(Counter(assignments[seed])[fold] for fold in range(N_SPLITS))
        if fold_sizes != EXPECTED_FOLD_SIZES[seed]:
            raise ValueError(f"ladder {seed} fold conservation failed")
        ladder = ladders[str(seed)]
        if not isinstance(ladder, Mapping) or ladder.get("acquisition_seed") != seed:
            raise ValueError(f"ladder {seed} metadata drifted")
        if ladder.get("optimiser_seed") != OPTIMISER_SEEDS[seed]:
            raise ValueError(f"ladder {seed} optimiser pairing drifted")
        budgets = ladder.get("budgets")
        if not isinstance(budgets, Mapping) or set(budgets) != set(BUDGET_FOLDS):
            raise ValueError(f"ladder {seed} budgets drifted")
        previous: set[str] = set()
        for budget, expected_folds in BUDGET_FOLDS.items():
            selected = {
                sample_ids[index]
                for index, fold in enumerate(assignments[seed])
                if fold in expected_folds
            }
            metadata = budgets[budget]
            if not isinstance(metadata, Mapping):
                raise ValueError(f"ladder {seed} {budget} metadata is missing")
            if metadata.get("folds") != list(expected_folds):
                raise ValueError(f"ladder {seed} {budget} fold definition drifted")
            if metadata.get("row_count") != len(selected):
                raise ValueError(f"ladder {seed} {budget} row count drifted")
            if metadata.get("sample_ids_sha256") != _sample_digest(list(selected)):
                raise ValueError(f"ladder {seed} {budget} sample digest drifted")
            if not previous.issubset(selected):
                raise ValueError(f"ladder {seed} is not nested at {budget}")
            previous = selected
            selected_support = metadata.get("feature_support")
            if not isinstance(selected_support, Mapping) or set(selected_support) != set(
                feature_names
            ):
                raise ValueError(f"ladder {seed} {budget} feature supports drifted")
            if budget != "10k":
                absent = [
                    feature
                    for feature, count in support_by_name.items()
                    if count >= 25 and selected_support.get(feature) == 0
                ]
                if absent:
                    raise ValueError(f"ladder {seed} {budget} omits guarded features: {absent}")

    chronological = clean.get("chronological")
    if not isinstance(chronological, Mapping) or set(chronological) != set(CHRONOLOGICAL_SPLITS):
        raise ValueError("chronological split metadata drifted")
    for name, chrono_contract in CHRONOLOGICAL_SPLITS.items():
        metadata = chronological[name]
        ids = chronological_ids[name]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"chronological {name} metadata is missing")
        expected = {
            "years": list(chrono_contract["years"]),
            "row_count": chrono_contract["expected_rows"],
            "sample_ids_sha256": _sample_digest(ids),
        }
        if dict(metadata) != expected:
            raise ValueError(f"chronological {name} metadata drifted")

    if source_rows is not None:
        clean_source, source_features, matrix, source_support = encode_features(source_rows)
        if [row["sample_id"] for row in clean_source] != sample_ids:
            raise ValueError("source rows do not match manifest sample IDs")
        if [row["thread_id"] for row in clean_source] != thread_ids:
            raise ValueError("source rows do not match manifest thread IDs")
        if list(source_features) != feature_names:
            raise ValueError("source feature names do not match manifest")
        if canonical_sha256(matrix) != feature_encoding.get("matrix_sha256"):
            raise ValueError("source feature matrix does not match manifest")
        if dict(source_support) != support_by_name:
            raise ValueError("source feature support does not match manifest")
        for index, source in enumerate(clean_source):
            if rows[index]["feature_sha256"] != canonical_sha256(_row_features(source)):
                raise ValueError("source row feature digest does not match manifest")
            if rows[index]["year"] != source["year"]:
                raise ValueError("source row year does not match manifest")
    return dict(manifest)


def write_split_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Publish an immutable manifest under its content-derived filename."""

    validated = validate_split_manifest(manifest)
    manifest_id = validated["manifest_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"manifest-{manifest_id}.json"
    payload = _json_bytes(validated)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable output differs: {path}")
        return path
    temporary = output_dir / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
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
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_split_manifest(
    path: Path,
    *,
    source_rows: Sequence[Mapping[str, Any]] | None = None,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Reload and exactly validate an immutable split manifest."""

    if expected_file_sha256 is not None and file_sha256(path) != expected_file_sha256:
        raise ValueError("split manifest file SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("split manifest must be a JSON object")
    validated = validate_split_manifest(value, source_rows=source_rows)
    expected_name = f"manifest-{validated['manifest_id']}.json"
    if path.name != expected_name:
        raise ValueError(f"split manifest filename must be {expected_name}")
    return validated
