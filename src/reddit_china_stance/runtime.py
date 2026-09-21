"""Transparent wall-clock planning arithmetic for batched teacher inference."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeEstimate:
    rows: int
    seconds: float

    @property
    def hours(self) -> float:
        return self.seconds / 3600


def estimate_runtime(
    *,
    rows: int,
    prompt_tokens: float,
    output_tokens: float,
    prefill_tokens_per_second: float,
    decode_tokens_per_second: float,
    overhead_fraction: float,
) -> RuntimeEstimate:
    """Estimate runtime from aggregate token throughput.

    Formula: N * (P / prefill_rate + O / decode_rate) * (1 + overhead).
    Rates must be measured for the actual server, batching policy, and prompt mix
    before this estimate is used for scheduling.
    """

    if rows <= 0:
        raise ValueError("rows must be greater than zero")
    positive_inputs = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
    }
    for name, value in positive_inputs.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
    if not math.isfinite(overhead_fraction) or overhead_fraction < 0:
        raise ValueError("overhead_fraction must be finite and non-negative")

    seconds_per_row = (
        prompt_tokens / prefill_tokens_per_second + output_tokens / decode_tokens_per_second
    ) * (1 + overhead_fraction)
    return RuntimeEstimate(rows=rows, seconds=rows * seconds_per_row)
