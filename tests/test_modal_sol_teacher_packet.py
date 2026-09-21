from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import reddit_china_stance.modal_sol_teacher_packet as packet
from reddit_china_stance.human_seeded_consensus_v1 import ADJUDICATION_BLINDNESS_CONTRACT


def test_surface_normalisation_and_simhash_are_deterministic() -> None:
    left = "  Chinese   culture is interesting.  "
    right = "chinese culture is interesting."
    assert packet._normalise_surface(left) == packet._normalise_surface(right)
    assert packet._simhash(left) == packet._simhash(right)


def test_near_duplicate_respects_length_and_hamming_gates() -> None:
    text = "A sufficiently long synthetic sentence about a repeated semantic surface. " * 2
    value = packet._simhash(text)
    chars = len(packet._normalise_surface(text))
    assert packet._is_near_duplicate(
        simhash=value, chars=chars, comparisons=[(value ^ 0b111, chars)]
    )
    assert not packet._is_near_duplicate(
        simhash=value, chars=chars, comparisons=[(value ^ 0b1111, chars)]
    )
    assert not packet._is_near_duplicate(simhash=value, chars=10, comparisons=[(value, 10)])


def test_contract_is_content_addressable_and_cost_bounded() -> None:
    state = {"git_commit": "a" * 40, "files": {}, "code_sha256": "b" * 64}
    contract = packet.make_contract(
        exclusion_sha256="c" * 64,
        code_state=state,
        approved_cost_usd=Decimal("5"),
    )
    assert packet._canonical_sha256(contract) == packet._canonical_sha256(contract)
    assert contract["target_rows"] == 10_000
    with pytest.raises(ValueError, match="approved cost"):
        packet.make_contract(
            exclusion_sha256="c" * 64,
            code_state=state,
            approved_cost_usd=Decimal("0.01"),
        )


def test_packet_blindness_contract_matches_sol_runner() -> None:
    assert packet.BLINDNESS_CONTRACT == ADJUDICATION_BLINDNESS_CONTRACT


def test_existing_packet_requires_complete_bound_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(packet, "TARGET_ROWS", 1)
    root = tmp_path / "packet"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"contract_id": "c" * 64}))
    with pytest.raises(RuntimeError, match="incomplete"):
        packet._validate_existing_packet(output_root=root, contract_id="c" * 64)
