from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from reddit_china_stance.sol_ontology_bridge_v2 import (
    DISABLED_FEATURES,
    ENGINEERING_GATES,
    MODEL,
    REASONING_EFFORT,
    _evaluate_engineering_gates,
    _observed_provider_identity,
    _run_shard,
    build_codex_command,
    codex_cli_provenance,
    finalise_bridge,
    output_schema,
    reconcile_review_labels,
    score_bridge_labels,
)


def _label(target: str, stance: str | None) -> dict[str, object]:
    return {
        "codability": "codable",
        "relevance": "material",
        "targets": [{"target": target, "stance": stance}],
    }


def _not_codable() -> dict[str, object]:
    return {"codability": "not_codable", "relevance": None, "targets": []}


def _not_material() -> dict[str, object]:
    return {"codability": "codable", "relevance": "not_material", "targets": []}


def _row(item_id: str, label: dict[str, object]) -> dict[str, object]:
    return {"source_sample_id": item_id, "label": label}


def test_transport_schema_is_strict_but_semantics_remain_post_validated() -> None:
    schema = output_schema(["opaque-a", "opaque-b"])
    Draft202012Validator.check_schema(schema)
    text = json.dumps(schema)

    assert "unclear" not in text
    assert schema["properties"]["rows"]["minItems"] == 2
    assert schema["properties"]["rows"]["maxItems"] == 2
    with pytest.raises(ValueError, match="unique"):
        output_schema(["same", "same"])


def test_reconciliation_uses_tie_break_only_for_exact_disagreements() -> None:
    review_a = [
        _row("one", _label("people_identity", "negative")),
        _row("two", _label("culture_media", "positive")),
        _row("three", _not_codable()),
    ]
    review_b = [
        _row("one", _label("people_identity", "negative")),
        _row("two", _label("culture_media", "negative")),
        _row("three", _not_codable()),
    ]
    tie_break = [_row("two", _label("culture_media", "no_directed_stance"))]

    final, disagreements = reconcile_review_labels(
        review_a,
        review_b,
        tie_break=tie_break,
    )

    assert disagreements == ["two"]
    assert final == [
        review_a[0],
        tie_break[0],
        review_a[2],
    ]
    with pytest.raises(ValueError, match="exactly the disagreements"):
        reconcile_review_labels(review_a, review_b, tie_break=[])
    with pytest.raises(ValueError, match="exactly the disagreements"):
        reconcile_review_labels(
            review_a,
            review_b,
            tie_break=[_row("one", _label("people_identity", "negative"))],
        )


def _complete_frame() -> list[dict[str, object]]:
    targets = (
        "china_general",
        "government_ccp",
        "people_identity",
        "culture_media",
        "company_tech_product",
        "residual_other",
    )
    stances = ("negative", "positive", "mixed", "no_directed_stance")
    rows = []
    for index in range(480):
        target = targets[index % len(targets)]
        stance = None if target == "residual_other" else stances[index % len(stances)]
        rows.append(_row(f"opaque-{index:03d}", _label(target, stance)))
    return rows


def test_aggregate_scoring_passes_supported_exact_agreement() -> None:
    rows = _complete_frame()

    score = score_bridge_labels(rows, rows, rows)

    assert score["row_count"] == 480
    assert score["dual_exact_agreement_rate"] == 1.0
    assert score["dual_codability_agreement_rate"] == 1.0
    assert score["dual_relevance_agreement_rate"] == 1.0
    assert score["not_material_rate"] == 0.0
    assert score["tie_break_count"] == 0
    assert score["target_support"] == {
        "china_general": 80,
        "government_ccp": 80,
        "people_identity": 80,
        "culture_media": 80,
        "company_tech_product": 80,
        "residual_other": 80,
    }
    assert score["engineering_verdict"] == "pass"
    assert all(score["gate_results"].values())
    for target, metric in score["dual_target_presence_agreement"].items():
        assert metric["pairwise_presence_f1"] == 1.0, target
    for target, metric in score["dual_target_stance_agreement"].items():
        assert metric["exact_agreement_numerator"] == metric["exact_agreement_denominator"], target
        assert metric["exact_agreement_rate"] == 1.0, target


def test_aggregate_scoring_rejects_low_dual_agreement() -> None:
    first = _complete_frame()
    second = _complete_frame()
    for index in range(200):
        second[index] = _row(
            second[index]["source_sample_id"],
            _not_codable(),
        )

    score = score_bridge_labels(first, second, first)

    assert score["engineering_verdict"] == "fail"
    assert score["gate_results"]["exact_label_agreement"] is False
    assert score["gate_results"]["codability_agreement"] is False


def _passing_gate_inputs() -> dict[str, object]:
    presence = {
        target: {
            "pairwise_presence_f1": 1.0,
            "concordant_positive_support": minimum,
        }
        for target, minimum in ENGINEERING_GATES[
            "minimum_target_concordant_positive_support"
        ].items()
    }
    stance = {
        target: {"exact_agreement_rate": 1.0}
        for target in ENGINEERING_GATES["minimum_target_concordant_positive_support"]
    }
    return {
        "exact_rate": 1.0,
        "codability_rate": 1.0,
        "not_codable_rate": 0.0,
        "target_support": dict(ENGINEERING_GATES["minimum_target_support"]),
        "target_presence": presence,
        "target_stance": stance,
    }


@pytest.mark.parametrize(
    ("gate", "mutation"),
    [
        ("exact_label_agreement", "exact"),
        ("codability_agreement", "codability"),
        ("not_codable_rate", "not_codable"),
        ("analytic_target_support", "final_support"),
        ("dual_target_presence_f1", "presence_f1"),
        ("dual_target_concordant_positive_support", "dual_support"),
        ("dual_target_stance_exact_agreement", "stance"),
    ],
)
def test_each_engineering_gate_fails_on_its_registered_boundary_only(
    gate: str,
    mutation: str,
) -> None:
    inputs = _passing_gate_inputs()
    target = "people_identity"
    if mutation == "exact":
        inputs["exact_rate"] = 0.69
    elif mutation == "codability":
        inputs["codability_rate"] = 0.89
    elif mutation == "not_codable":
        inputs["not_codable_rate"] = 0.11
    elif mutation == "final_support":
        inputs["target_support"][target] = 19
    elif mutation == "presence_f1":
        inputs["target_presence"][target]["pairwise_presence_f1"] = 0.69
    elif mutation == "dual_support":
        inputs["target_presence"][target]["concordant_positive_support"] = 9
    else:
        inputs["target_stance"][target]["exact_agreement_rate"] = 0.69

    results = _evaluate_engineering_gates(**inputs)

    assert results[gate] is False
    assert all(value for name, value in results.items() if name != gate)


def test_scoring_rejects_row_drift_and_invalid_labels() -> None:
    rows = _complete_frame()
    with pytest.raises(ValueError, match="exactly 480"):
        score_bridge_labels(rows[:-1], rows[:-1], rows[:-1])
    invalid = deepcopy(rows)
    invalid[0]["label"] = _not_material() | {"targets": [_label("china_general", "negative")]}
    with pytest.raises(ValueError, match="invalid v2 label"):
        score_bridge_labels(invalid, rows, rows)


def test_tie_break_support_cannot_mask_target_presence_disagreement() -> None:
    first = _complete_frame()
    second = deepcopy(first)
    people_indexes = [
        index
        for index, row in enumerate(second)
        if row["label"]["targets"][0]["target"] == "people_identity"
    ]
    for index in people_indexes[:40]:
        second[index] = _row(second[index]["source_sample_id"], _label("residual_other", None))

    score = score_bridge_labels(first, second, first)

    assert score["gate_results"]["analytic_target_support"] is True
    assert score["target_support"]["people_identity"] == 80
    assert score["dual_target_presence_agreement"]["people_identity"] == {
        "pairwise_presence_f1": pytest.approx(2 / 3),
        "concordant_positive_support": 40,
        "review_a_only": 40,
        "review_b_only": 0,
    }
    assert score["gate_results"]["dual_target_presence_f1"] is False
    assert score["engineering_verdict"] == "fail"


def test_per_target_stance_gate_catches_hidden_disagreement() -> None:
    first = _complete_frame()
    second = deepcopy(first)
    people_indexes = [
        index
        for index, row in enumerate(second)
        if row["label"]["targets"][0]["target"] == "people_identity"
    ]
    for index in people_indexes[:30]:
        original = second[index]["label"]["targets"][0]["stance"]
        replacement = "positive" if original != "positive" else "negative"
        second[index] = _row(
            second[index]["source_sample_id"],
            _label("people_identity", replacement),
        )

    score = score_bridge_labels(first, second, first)
    metric = score["dual_target_stance_agreement"]["people_identity"]

    assert metric["exact_agreement_denominator"] == 80
    assert metric["exact_agreement_numerator"] == 50
    assert metric["exact_agreement_rate"] == 0.625
    assert score["gate_results"]["dual_target_stance_exact_agreement"] is False
    assert score["engineering_verdict"] == "fail"


def test_relevance_and_final_not_material_are_reported() -> None:
    rows = _complete_frame()
    rows[0] = _row(rows[0]["source_sample_id"], _not_material())

    score = score_bridge_labels(rows, rows, rows)

    assert score["dual_relevance_agreement_count"] == 480
    assert score["dual_relevance_agreement_rate"] == 1.0
    assert score["not_material_count"] == 1
    assert score["not_material_rate"] == pytest.approx(1 / 480)


def test_codex_command_is_high_reasoning_ephemeral_and_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.shutil.which",
        lambda *_args, **_kwargs: "/usr/local/bin/codex",
    )
    command = build_codex_command(
        schema_path=tmp_path / "schema.json",
        output_path=tmp_path / "output.json",
        work_dir=tmp_path,
    )

    assert command[:3] == ["/usr/local/bin/codex", "exec", "--ignore-user-config"]
    assert "--ignore-rules" in command
    assert "--ephemeral" in command
    assert command[command.index("--model") + 1] == MODEL
    assert f'model_reasoning_effort="{REASONING_EFFORT}"' in command
    assert 'web_search="disabled"' in command
    assert "read-only" in command
    for feature in DISABLED_FEATURES:
        assert feature in command


def test_codex_cli_provenance_binds_exact_binary_and_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "codex"
    binary.write_bytes(b"exact-codex-binary")
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: str(binary),
    )
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="codex-cli 1.2.3\n",
            stderr="",
        ),
    )

    provenance = codex_cli_provenance()

    assert provenance["requested_model"] == MODEL
    assert provenance["codex_cli_version"] == "codex-cli 1.2.3"
    assert len(provenance["codex_cli_binary_sha256"]) == 64


def test_observed_provider_identity_is_optional_but_model_mismatch_fails() -> None:
    assert _observed_provider_identity(
        '\n'.join(
            (
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"turn.completed","usage":{"input_tokens":1}}',
            )
        )
    ) is None
    assert _observed_provider_identity(
        '{"type":"thread.started","model":"gpt-5.6-sol","version":"2026-08"}'
    ) == {"model": ["gpt-5.6-sol"], "version": ["2026-08"]}
    with pytest.raises(RuntimeError, match="different from the requested"):
        _observed_provider_identity('{"type":"thread.started","model":"other-model"}')


def _blinded_row(item_id: str) -> dict[str, object]:
    return {
        "source_sample_id": item_id,
        "target_text": "private synthetic text",
        "submission_context": None,
        "parent_context": None,
    }


def test_shard_publication_is_transactional_and_exactly_reusable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run=exact"
    run_root.mkdir()
    provenance = {
        "run_id": "exact",
        "packet_id": "packet-exact",
        "requested_model": MODEL,
        "codex_cli_binary_sha256": "a" * 64,
        "codex_cli_version": "codex-cli test",
    }
    (run_root / "run-manifest.json").write_text(json.dumps(provenance))
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.codex_cli_provenance",
        lambda: dict(provenance),
    )
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._codex_binary",
        lambda: "/exact/codex",
    )
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(
            json.dumps({"rows": [_row("opaque-one", _label("china_general", "negative"))]})
        )
        stdout = "\n".join(
            (
                '{"type":"thread.started","thread_id":"thread-exact"}',
                '{"type":"turn.completed","usage":{"input_tokens":10,'
                '"cached_input_tokens":2,"output_tokens":3}}',
            )
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.subprocess.run",
        fake_run,
    )
    first = _run_shard(
        run_root=run_root,
        pass_name="review_a",
        rows=[_blinded_row("opaque-one")],
        shard_index=0,
    )
    shard_root = first.parent

    assert calls == 1
    assert {path.name for path in shard_root.iterdir()} == {"event.json", "fragment.json"}
    assert not list(shard_root.parent.glob(".shard-000.incomplete-*"))

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact fragment reuse must not call the provider")

    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.subprocess.run",
        forbidden_run,
    )
    reused = _run_shard(
        run_root=run_root,
        pass_name="review_a",
        rows=[_blinded_row("opaque-one")],
        shard_index=0,
    )
    assert reused == first


@pytest.mark.parametrize("state", ["orphan-final", "incomplete-staging"])
def test_shard_refuses_orphan_or_incomplete_publication(
    state: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run=exact"
    pass_root = run_root / "pass=review_a"
    pass_root.mkdir(parents=True)
    provenance = {
        "run_id": "exact",
        "packet_id": "packet-exact",
        "requested_model": MODEL,
        "codex_cli_binary_sha256": "a" * 64,
        "codex_cli_version": "codex-cli test",
    }
    (run_root / "run-manifest.json").write_text(json.dumps(provenance))
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.codex_cli_provenance",
        lambda: dict(provenance),
    )
    if state == "orphan-final":
        shard_root = pass_root / "shard-000"
        shard_root.mkdir()
        (shard_root / "event.json").write_text("{}")
        match = "incomplete or orphaned"
    else:
        (pass_root / ".shard-000.incomplete-crash").mkdir()
        match = "incomplete Sol shard staging"

    with pytest.raises(RuntimeError, match=match):
        _run_shard(
            run_root=run_root,
            pass_name="review_a",
            rows=[_blinded_row("opaque-one")],
            shard_index=0,
        )


def test_synthetic_finalisation_publishes_metadata_only_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "private" / "run=synthetic-run"
    public_root = tmp_path / "public"
    run_root.mkdir(parents=True)
    provenance = {
        "requested_model": MODEL,
        "codex_cli_binary_sha256": "b" * 64,
        "codex_cli_version": "codex-cli test",
    }
    (run_root / "run-manifest.json").write_text(json.dumps(provenance))
    rows = _complete_frame()
    for index, pass_name in enumerate(("review_a", "review_b")):
        event_root = run_root / f"pass={pass_name}" / "shard-000"
        event_root.mkdir(parents=True)
        event = {
            **provenance,
            "thread_id": f"private-thread-{index}",
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 5,
            },
            "elapsed_seconds": 1.0,
            "observed_provider_identity": None,
        }
        (event_root / "event.json").write_text(json.dumps(event))

    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._ensure_run",
        lambda *_args, **_kwargs: (
            run_root,
            [_blinded_row(row["source_sample_id"]) for row in rows],
        ),
    )
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2._dual_outputs",
        lambda *_args, **_kwargs: ({"rows": rows}, {"rows": rows}, []),
    )
    monkeypatch.setattr(
        "reddit_china_stance.sol_ontology_bridge_v2.make_run_contract",
        lambda *_args, **_kwargs: {"packet_id": "synthetic-packet", **provenance},
    )

    receipt = finalise_bridge(
        packet_root=tmp_path / "unused-packet",
        private_root=tmp_path / "private",
        public_root=public_root,
    )

    public_receipts = list(public_root.glob("run=*/receipt-*.json"))
    assert len(public_receipts) == 1
    assert json.loads(public_receipts[0].read_text()) == receipt
    serialised = json.dumps(receipt)
    assert "opaque-" not in serialised
    assert "private-thread" not in serialised
    assert "target_text" not in serialised
    assert receipt["provider_identity_observed_execution_count"] == 0
    assert receipt["observed_provider_identities"] == []
    assert receipt["provider_identity_limitation"] is not None
    assert receipt["receipt_contains_raw_text"] is False
    assert receipt["receipt_contains_row_ids"] is False
    assert receipt["receipt_contains_thread_ids"] is False
    assert receipt["receipt_contains_row_level_labels"] is False
