from __future__ import annotations

from decimal import Decimal
from inspect import signature
from pathlib import Path

import pytest

from reddit_china_stance import modal_modernbert_factorised as launcher
from reddit_china_stance import modernbert_factorised_training as training
from reddit_china_stance import prepare_modernbert_factorised_inputs as preparation


def test_prepare_helper_binds_exact_sources_and_bridge_authorisation() -> None:
    assert tuple(launcher.REQUIRED_SOURCE_FILES) == preparation.SOURCE_FILES
    assert (
        "src/reddit_china_stance/prepare_modernbert_factorised_inputs.py"
        in preparation.SOURCE_FILES
    )
    parameters = signature(training.derive_bridge_exposure_register).parameters
    assert {
        "bridge_authorisation_path",
        "bridge_authorisation_descriptor",
    } <= set(parameters)


def test_prepare_cost_accounting_stays_within_approved_envelope() -> None:
    assert sum(
        preparation.MEASURED_MODAL_COSTS_USD.values(), Decimal("0")
    ) == preparation.CUMULATIVE_MEASURED_SPEND_USD
    reserved = preparation.L4_RATE_USD_PER_GPU_SECOND * Decimal(
        3 * sum(preparation.MAX_GPU_SECONDS_BY_COMPONENT.values())
    )
    assert reserved == Decimal("19.2")
    assert reserved <= preparation.PLANNED_PHASE_UPPER_USD
    assert (
        preparation.CUMULATIVE_MEASURED_SPEND_USD
        + preparation.ACTIVE_RESERVATION_USD
        + preparation.PLANNED_PHASE_UPPER_USD
        <= preparation.HARD_COST_CAP_USD
    )


def test_relative_output_root_resolves_inside_registered_private_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = Path("data/private-modernbert-factorised-v2/prepare-relative-test")
    expected = (preparation.REPO_ROOT / relative).resolve()
    assert preparation._normalise_output_root(relative) == expected

    captured: list[Path] = []
    monkeypatch.setattr(
        preparation,
        "build_prepare_inputs",
        lambda *, output_root: captured.append(output_root)
        or {"status": "prepared-locally"},
    )
    assert preparation.main(["--output-root", str(relative)]) == 0
    assert captured == [expected]


def test_output_root_rejects_paths_outside_registered_private_namespace(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must remain inside"):
        preparation._normalise_output_root(tmp_path / "outside")
    with pytest.raises(ValueError, match="must remain inside"):
        preparation._normalise_output_root(Path("../outside"))
