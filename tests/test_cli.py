from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_china_stance.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_annotations_cli_accepts_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "valid.jsonl"
    path.write_text(
        json.dumps(
            {
                "relevance": "material",
                "target_stances": [
                    {"target": "china_general", "stance": "mixed"}
                ],
            }
        )
        + "\n"
    )
    result = _run_cli("validate-annotations", str(path))
    assert result.returncode == 0, result.stderr
    assert '"status": "valid"' in result.stdout


def test_validate_annotations_cli_exits_nonzero_on_invalid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"relevance":"direct","target_stances":[]}\n')
    result = _run_cli("validate-annotations", str(path))
    assert result.returncode != 0
    assert "error: line 1: schema violation" in result.stderr


def test_estimate_runtime_cli_emits_json() -> None:
    result = _run_cli(
        "estimate-runtime",
        "--rows",
        "10000",
        "--prompt-tokens",
        "650",
        "--output-tokens",
        "45",
        "--prefill-tps",
        "6000",
        "--decode-tps",
        "400",
        "--overhead",
        "0.25",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert round(payload["estimated_hours"], 1) == 0.8
