"""Bounded throughput benchmark for the frozen acquisition scoring path.

This module is intentionally separate from ``modal_modernbert_acquisition`` so
benchmark instrumentation cannot invalidate the already prepared frame's exact
runtime-source binding.  It never writes predictions or returns row-level data.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import modal

from reddit_china_stance import modal_modernbert_acquisition as acquisition
from reddit_china_stance.privacy import assert_metadata_only

APP_NAME = "reddit-china-stance-modernbert-acquisition-benchmark-v1"
SAMPLE_ROWS = 10_000
BATCH_SIZES = (64, 128)
MAX_APPROVAL_USD = Decimal("1")
DEFAULT_INPUT_SPEC_PATH = Path("data/private-modernbert-acquisition-v1/prepared-input.json")

app = modal.App(APP_NAME)


def confirmation_token(prepared_run_id: str) -> str:
    return f"BENCHMARK_MODERNBERT_ACQUISITION_SCORING_{prepared_run_id[:12]}"


def validate_approval(value: str) -> Decimal:
    try:
        approved = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("approved cost must be a decimal") from exc
    if approved != MAX_APPROVAL_USD:
        raise RuntimeError("scoring benchmark requires exact $1 approval")
    return approved


def evenly_spaced_indices(*, population_rows: int, sample_rows: int) -> list[int]:
    if population_rows < sample_rows or sample_rows <= 0:
        raise ValueError("benchmark sample must fit inside a non-empty population")
    indices = [(offset * population_rows) // sample_rows for offset in range(sample_rows)]
    if len(indices) != len(set(indices)) or indices[-1] >= population_rows:
        raise RuntimeError("benchmark index construction did not conserve unique rows")
    return indices


@app.function(
    image=acquisition.image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(acquisition.VOLUME_PATH): acquisition.volume},
)
def benchmark(spec: dict[str, Any], approved_cost_usd: str) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    validate_approval(approved_cost_usd)
    acquisition.volume.reload()
    spec = acquisition.validate_input_spec(spec, action="cuda-preflight")
    run_id = acquisition._sha256(spec.get("prepared_run_id"), where="prepared run ID")
    prepared, frame_path = acquisition._load_prepared_contract(run_id)
    acquisition._validate_prepared_spec_binding(prepared, spec)
    checkpoints = acquisition._checkpoint_specs(spec)
    checkpoint = next(
        item
        for item in checkpoints
        if item["component"] == "target_stance_b4" and item["optimiser_seed"] == 47
    )

    load_started = time.monotonic()
    table = pq.read_table(frame_path)
    indices = evenly_spaced_indices(
        population_rows=table.num_rows,
        sample_rows=SAMPLE_ROWS,
    )
    rows = table.take(pa.array(indices, type=pa.int64())).to_pylist()
    frame_load_seconds = time.monotonic() - load_started
    if len(rows) != SAMPLE_ROWS:
        raise RuntimeError("benchmark sample did not conserve rows")

    training = acquisition._training_module()
    tokenizer = training.load_pinned_tokenizer()
    model = acquisition._load_checkpoint_model(checkpoint)

    # Warm the exact rendering/tokenisation/model path before timing each batch
    # size.  Output stays in memory and is discarded immediately.
    warmup = acquisition._collect_unlabelled_logits(
        model=model,
        tokenizer=tokenizer,
        rows=rows[:128],
        component="target_stance_b4",
        render="full",
        batch_size=8,
    )
    if len(warmup) != 128:
        raise RuntimeError("benchmark warmup did not conserve rows")

    measurements: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        output = acquisition._collect_unlabelled_logits(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            component="target_stance_b4",
            render="full",
            batch_size=batch_size,
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if len(output) != SAMPLE_ROWS:
            raise RuntimeError("benchmark inference did not conserve rows")
        measurements.append(
            {
                "batch_size": batch_size,
                "row_count": SAMPLE_ROWS,
                "wall_seconds": round(elapsed, 6),
                "rows_per_second": SAMPLE_ROWS / elapsed,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )
        del output

    fastest = max(measurements, key=lambda item: item["rows_per_second"])
    result = {
        "kind": "modernbert-acquisition-scoring-benchmark-v1",
        "prepared_run_id": run_id,
        "gpu_type": "L4",
        "component": "target_stance_b4",
        "render": "full",
        "population_rows": int(table.num_rows),
        "sample_rows": SAMPLE_ROWS,
        "frame_load_seconds": round(frame_load_seconds, 6),
        "measurements": measurements,
        "fastest_batch_size": fastest["batch_size"],
        "fastest_rows_per_second": fastest["rows_per_second"],
        "approved_cost_usd": format(MAX_APPROVAL_USD, "f"),
        "locked_test_rows_accessed": 0,
    }
    assert_metadata_only(result, where="acquisition scoring benchmark")
    return result


@app.local_entrypoint()
def main(
    input_spec_path: str = str(DEFAULT_INPUT_SPEC_PATH),
    approved_cost_usd: str = "0",
    confirm: str = "",
) -> None:
    spec = json.loads(Path(input_spec_path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("acquisition input spec must contain an object")
    clean = acquisition.validate_input_spec(spec, action="cuda-preflight")
    run_id = acquisition._sha256(clean.get("prepared_run_id"), where="prepared run ID")
    required = confirmation_token(run_id)
    if confirm != required:
        raise RuntimeError(f"refusing benchmark: pass --confirm {required}")
    result = benchmark.remote(clean, approved_cost_usd)
    print(json.dumps(result, sort_keys=True, indent=2))


__all__ = [
    "APP_NAME",
    "BATCH_SIZES",
    "SAMPLE_ROWS",
    "confirmation_token",
    "evenly_spaced_indices",
    "validate_approval",
]
