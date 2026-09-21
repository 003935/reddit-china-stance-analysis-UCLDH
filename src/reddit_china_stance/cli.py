"""Small strict CLI for contract validation and compute planning."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from reddit_china_stance.human_seeded_consensus_v1 import validate_semantic_label
from reddit_china_stance.privacy import (
    validate_public_metadata_file,
    validate_repository_privacy,
)
from reddit_china_stance.runtime import estimate_runtime


def default_annotation_schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "human-reference-semantic-label-v1.schema.json"
    )


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reddit-china-stance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser(
        "estimate-runtime", help="Estimate teacher-inference wall time from explicit rates."
    )
    estimate.add_argument("--rows", required=True, type=_positive_int)
    estimate.add_argument("--prompt-tokens", required=True, type=_positive_float)
    estimate.add_argument("--output-tokens", required=True, type=_positive_float)
    estimate.add_argument("--prefill-tps", required=True, type=_positive_float)
    estimate.add_argument("--decode-tps", required=True, type=_positive_float)
    estimate.add_argument("--overhead", required=True, type=_nonnegative_float)

    validate = subparsers.add_parser(
        "validate-annotations", help="Validate every JSONL row; fail on the first error."
    )
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--schema",
        type=Path,
        default=default_annotation_schema_path(),
        help="Annotation JSON Schema (defaults to the repository contract).",
    )

    public_metadata = subparsers.add_parser(
        "validate-public-metadata",
        help="Reject row identity, text, prompts, or row-level labels in a public JSON receipt.",
    )
    public_metadata.add_argument("path", type=Path)

    repository_privacy = subparsers.add_parser(
        "validate-repository-privacy",
        help=(
            "Reject tracked research data, generated outputs, row artefacts, "
            "and environment files."
        ),
    )
    repository_privacy.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def _load_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_annotations(path: Path, schema_path: Path) -> int:
    schema = _load_schema(schema_path)
    validator = Draft202012Validator(schema)
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank lines are not valid JSONL records")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            schema_errors = sorted(
                validator.iter_errors(payload), key=lambda error: list(error.path)
            )
            if schema_errors:
                first = schema_errors[0]
                location = ".".join(str(part) for part in first.absolute_path) or "<root>"
                raise ValueError(
                    f"line {line_number}: schema violation at {location}: {first.message}"
                )
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: annotation must be a JSON object")
            validate_semantic_label(payload, schema=schema)
            count += 1
    if count == 0:
        raise ValueError("annotation file contains no records")
    print(json.dumps({"status": "valid", "records": count, "path": str(path)}))
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "estimate-runtime":
            result = estimate_runtime(
                rows=args.rows,
                prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens,
                prefill_tokens_per_second=args.prefill_tps,
                decode_tokens_per_second=args.decode_tps,
                overhead_fraction=args.overhead,
            )
            print(
                json.dumps(
                    {
                        "rows": result.rows,
                        "estimated_seconds": result.seconds,
                        "estimated_hours": result.hours,
                        "formula": "N * (P / prefill_tps + O / decode_tps) * (1 + overhead)",
                    }
                )
            )
            return 0
        if args.command == "validate-annotations":
            return validate_annotations(args.path, args.schema)
        if args.command == "validate-public-metadata":
            validate_public_metadata_file(args.path)
            print(json.dumps({"status": "valid", "path": str(args.path)}))
            return 0
        if args.command == "validate-repository-privacy":
            count = validate_repository_privacy(args.root)
            print(json.dumps({"status": "valid", "tracked_files": count}))
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise RuntimeError(f"unhandled command: {args.command}")


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
