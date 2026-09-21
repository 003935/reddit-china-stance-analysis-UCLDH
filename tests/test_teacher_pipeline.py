from __future__ import annotations

import json
from copy import deepcopy

import pytest

from reddit_china_stance.human_seeded_consensus_v1 import load_label_schema
from reddit_china_stance.teacher_pipeline import (
    ACTIVE_PROMPT_SHA256,
    PlannedRunInterruption,
    assert_public_metadata_only,
    build_public_receipt,
    build_sol_queue,
    canonical_sha256,
    finalise_labels,
    freeze_manifest,
    freeze_packet,
    freeze_packet_from_context_records,
    freeze_sol_checkpoint,
    freeze_teacher_checkpoint,
    run_sol_adjudication,
    run_teacher_model,
    validate_manifest,
    validate_teacher_checkpoint,
    write_immutable_json,
)


def _packet() -> dict[str, object]:
    return freeze_packet(
        [
            {
                "record_id": f"private-record-{index}",
                "target_text": f"private target text sentinel {index}",
                "submission_context": f"private submission context {index}",
                "parent_context": None,
            }
            for index in range(5)
        ]
    )


def _manifest(packet: dict[str, object], *, audit_size: int = 1) -> dict[str, object]:
    return freeze_manifest(
        packet,
        models={
            "gemma": {
                "provider": "cerebras",
                "model_id": "google/gemma",
                "revision": "gemma-revision",
                "reasoning_effort": "none",
                "max_output_tokens": 512,
            },
            "gpt_oss": {
                "provider": "groq",
                "model_id": "openai/gpt-oss",
                "revision": "gpt-oss-revision",
                "reasoning_effort": "low",
                "max_output_tokens": 512,
            },
            "sol": {
                "provider": "codex",
                "model_id": "openai/sol",
                "revision": "sol-revision",
                "reasoning_effort": "high",
                "max_output_tokens": 2048,
            },
        },
        label_schema_sha256=canonical_sha256(load_label_schema()),
        rubric_sha256="b" * 64,
        prompt_sha256=ACTIVE_PROMPT_SHA256,
        base_seed=20250823,
        audit_size=audit_size,
        audit_seed="frozen-audit-seed",
    )


def _label(
    relevance: str = "material",
    *,
    target: str = "government_ccp",
    stance: str = "negative",
) -> dict[str, object]:
    return {
        "relevance": relevance,
        "target_stances": (
            [{"target": target, "stance": stance}] if relevance == "material" else []
        ),
    }


def _stages(label: dict[str, object]) -> dict[str, object]:
    if label["relevance"] != "material":
        return {"relevance": {"relevance": label["relevance"]}}
    target_stances = label["target_stances"]
    return {
        "relevance": {"relevance": "material"},
        "targets": {"targets": [row["target"] for row in target_stances]},
        "stances": deepcopy(target_stances),
    }


def _teacher_outputs(packet: dict[str, object], labels: list[dict[str, object]]) -> list[dict]:
    return [
        {"record_id": row["record_id"], "stage_outputs": _stages(label)}
        for row, label in zip(packet["rows"], labels, strict=True)
    ]


def _checkpoints() -> tuple[dict, dict, dict, dict]:
    packet = _packet()
    manifest = _manifest(packet)
    clear = _label()
    exact_other = _label(target="people_culture", stance="positive")
    gemma_labels = [
        clear,
        clear,
        _label("unclear"),
        clear,
        exact_other,
    ]
    gpt_labels = [
        clear,
        _label(stance="positive"),
        _label("unclear"),
        clear,
        exact_other,
    ]
    gemma = freeze_teacher_checkpoint(
        packet,
        manifest,
        model_role="gemma",
        outputs=_teacher_outputs(packet, gemma_labels),
        observed_model="google/gemma",
        observed_provider="cerebras",
    )
    gpt_outputs = _teacher_outputs(packet, gpt_labels)
    gpt_outputs[3]["stage_outputs"] = {
        "relevance": {"relevance": "material"},
        "targets": {"targets": ["government_ccp"]},
        "stances": [],
    }
    gpt_oss = freeze_teacher_checkpoint(
        packet,
        manifest,
        model_role="gpt_oss",
        outputs=gpt_outputs,
        observed_model="openai/gpt-oss",
        observed_provider="groq",
    )
    return packet, manifest, gemma, gpt_oss


def test_packet_and_manifest_freeze_private_inputs_without_row_ids_in_contract() -> None:
    packet = _packet()
    manifest = _manifest(packet)

    assert validate_manifest(packet, manifest) == manifest
    assert manifest["contract"]["row_count"] == 5
    assert manifest["contract"]["prompt_sha256"] == ACTIVE_PROMPT_SHA256
    assert manifest["contract"]["base_seed"] == 20250823
    assert manifest["contract"]["models"]["gemma"]["provider"] == "cerebras"
    manifest_json = json.dumps(manifest, sort_keys=True)
    assert "private-record-0" not in manifest_json
    assert manifest["contract"]["stages"] == ["relevance", "targets", "stance"]

    models_with_secret = deepcopy(manifest["contract"]["models"])
    models_with_secret["gemma"]["runtime"] = {"provider_token": "sk-private"}
    with pytest.raises(ValueError, match="credentials or secrets"):
        freeze_manifest(
            packet,
            models=models_with_secret,
            label_schema_sha256=canonical_sha256(load_label_schema()),
            rubric_sha256="b" * 64,
            prompt_sha256=ACTIVE_PROMPT_SHA256,
            base_seed=20250823,
            audit_size=1,
            audit_seed="frozen-audit-seed",
        )


def test_context_adapter_validates_english_rows_and_join_statuses() -> None:
    records = [
        {
            "record_id": "private-record-0",
            "target_join_status": "present",
            "target_text": {"text": "bounded target"},
            "submission_join_status": "present",
            "submission_context": {"text": "bounded submission"},
            "parent_join_status": "missing",
            "parent_context": None,
            "context_role_rule": "reference_disambiguation_only",
        }
    ]
    packet = freeze_packet_from_context_records(
        records, english_record_ids=["private-record-0"]
    )
    assert packet["rows"] == [
        {
            "record_id": "private-record-0",
            "target_text": "bounded target",
            "submission_context": "bounded submission",
            "parent_context": None,
        }
    ]

    drifted = deepcopy(records)
    drifted[0]["parent_join_status"] = "invalid_relation"
    with pytest.raises(ValueError, match="parent_join_status is invalid"):
        freeze_packet_from_context_records(
            drifted, english_record_ids=["private-record-0"]
        )
    with pytest.raises(ValueError, match="exact English-eligible row set"):
        freeze_packet_from_context_records(records, english_record_ids=["different-record"])


def test_duplicate_packet_ids_and_missing_or_duplicate_teacher_rows_fail_closed() -> None:
    rows = deepcopy(_packet()["rows"])
    rows[1]["record_id"] = rows[0]["record_id"]
    with pytest.raises(ValueError, match="record IDs must be unique"):
        freeze_packet(rows)

    packet = _packet()
    manifest = _manifest(packet)
    outputs = _teacher_outputs(packet, [_label()] * 5)
    with pytest.raises(ValueError, match="row set mismatch: missing=1"):
        freeze_teacher_checkpoint(
            packet,
            manifest,
            model_role="gemma",
            outputs=outputs[:-1],
            observed_model="google/gemma",
            observed_provider="cerebras",
        )
    duplicate = deepcopy(outputs)
    duplicate[-1]["record_id"] = duplicate[0]["record_id"]
    with pytest.raises(ValueError, match="duplicate teacher output"):
        freeze_teacher_checkpoint(
            packet,
            manifest,
            model_role="gemma",
            outputs=duplicate,
            observed_model="google/gemma",
            observed_provider="cerebras",
        )


def test_invalid_stage_schema_is_preserved_as_invalid_and_tampering_is_rejected() -> None:
    packet = _packet()
    manifest = _manifest(packet)
    outputs = _teacher_outputs(packet, [_label()] * 5)
    outputs[2]["stage_outputs"]["stances"] = []
    checkpoint = freeze_teacher_checkpoint(
        packet,
        manifest,
        model_role="gemma",
        outputs=outputs,
        observed_model="google/gemma",
        observed_provider="cerebras",
    )

    assert checkpoint["rows"][2]["status"] == "invalid"
    assert checkpoint["rows"][2]["validation_error"]["code"] == "invalid_stage_output"
    tampered = deepcopy(checkpoint)
    tampered["rows"][2]["status"] = "valid"
    tampered["rows"][2]["semantic_label"] = _label()
    tampered["rows"][2].pop("validation_error")
    body = {key: value for key, value in tampered.items() if key != "checkpoint_id"}
    tampered["checkpoint_id"] = canonical_sha256(body)
    with pytest.raises(RuntimeError, match="validation state drifted"):
        validate_teacher_checkpoint(
            packet, manifest, tampered, model_role="gemma"
        )
    with pytest.raises(RuntimeError, match="observed teacher provider"):
        freeze_teacher_checkpoint(
            packet,
            manifest,
            model_role="gemma",
            outputs=outputs,
            observed_model="google/gemma",
            observed_provider="wrong-provider",
        )


def test_teacher_provider_injection_runs_atomic_stages_without_identifiers_or_retries() -> None:
    packet = freeze_packet(deepcopy(_packet()["rows"][:2]))
    manifest = _manifest(packet, audit_size=0)
    calls = []

    def provider_call(**request: object) -> dict[str, object]:
        calls.append(deepcopy(request))
        assert "record_id" not in json.dumps(request["item"])
        assert request["contract"] == {
            "manifest_id": manifest["manifest_id"],
            "label_schema_sha256": manifest["contract"]["label_schema_sha256"],
            "rubric_sha256": manifest["contract"]["rubric_sha256"],
            "prompt_sha256": manifest["contract"]["prompt_sha256"],
        }
        if request["stage"] == "relevance":
            value = {
                "relevance": (
                    "material"
                    if request["item"]["target_text"].endswith("0")
                    else "not_material"
                )
            }
        elif request["stage"] == "targets":
            value = {"targets": ["government_ccp"]}
        else:
            value = {"stance": "negative"}
        return {
            "value": value,
            "observed_model": "google/gemma",
            "observed_provider": "cerebras",
        }

    checkpoint = run_teacher_model(
        packet,
        manifest,
        model_role="gemma",
        provider_call=provider_call,
    )

    assert [call["stage"] for call in calls] == [
        "relevance",
        "targets",
        "stance",
        "relevance",
    ]
    assert [call["seed"] for call in calls] == [20250823, 20260823, 20270823, 20350823]
    assert checkpoint["observed_provider"] == "cerebras"
    assert checkpoint["prompt_sha256"] == ACTIVE_PROMPT_SHA256
    assert all(row["status"] == "valid" for row in checkpoint["rows"])


def test_provider_schema_binding_and_malformed_stance_fail_as_explicit_invalid_row() -> None:
    packet = freeze_packet(deepcopy(_packet()["rows"][:1]))
    manifest = _manifest(packet, audit_size=0)

    def provider_call(**request: object) -> object:
        if request["stage"] == "relevance":
            value: object = {"relevance": "material"}
        elif request["stage"] == "targets":
            value = {"targets": ["government_ccp"]}
        else:
            value = ["not-an-object"]
        return {
            "value": value,
            "observed_model": "google/gemma",
            "observed_provider": "cerebras",
        }

    checkpoint = run_teacher_model(
        packet,
        manifest,
        model_role="gemma",
        provider_call=provider_call,
    )
    assert checkpoint["rows"][0]["status"] == "invalid"

    drifted_schema = deepcopy(load_label_schema())
    drifted_schema["title"] = "drift"
    with pytest.raises(RuntimeError, match="runtime label schema differs"):
        freeze_teacher_checkpoint(
            packet,
            manifest,
            model_role="gemma",
            outputs=_teacher_outputs(packet, [_label()]),
            observed_model="google/gemma",
            observed_provider="cerebras",
            schema=drifted_schema,
        )


def test_teacher_journal_resumes_only_missing_rows_after_planned_interruption(
    tmp_path,
) -> None:
    packet = _packet()
    manifest = _manifest(packet)
    calls: list[str] = []

    def provider_call(**request: object) -> object:
        item = request["item"]
        assert isinstance(item, dict)
        calls.append(str(item["target_text"]))
        return {
            "value": {"relevance": "not_material"},
            "observed_model": "google/gemma",
            "observed_provider": "cerebras",
        }

    journal_root = tmp_path / "journal"
    with pytest.raises(PlannedRunInterruption, match="2 newly journalled"):
        run_teacher_model(
            packet,
            manifest,
            model_role="gemma",
            provider_call=provider_call,
            journal_root=journal_root,
            stop_after_new_rows=2,
        )
    assert len(calls) == 2
    assert len(list(journal_root.rglob("row-*.json"))) == 2
    assert not list(journal_root.rglob("*.pending"))

    checkpoint = run_teacher_model(
        packet,
        manifest,
        model_role="gemma",
        provider_call=provider_call,
        journal_root=journal_root,
    )
    assert len(calls) == 5
    assert checkpoint["row_count"] == 5
    assert all(row["status"] == "valid" for row in checkpoint["rows"])
    assert len(list(journal_root.rglob("row-*.json"))) == 5
    assert not list(journal_root.rglob("*.pending"))


def test_teacher_journal_fails_closed_on_entry_or_row_set_drift(tmp_path) -> None:
    packet = freeze_packet(deepcopy(_packet()["rows"][:1]))
    manifest = _manifest(packet, audit_size=0)

    def provider_call(**request: object) -> object:
        return {
            "value": {"relevance": "not_material"},
            "observed_model": "google/gemma",
            "observed_provider": "cerebras",
        }

    journal_root = tmp_path / "journal"
    run_teacher_model(
        packet,
        manifest,
        model_role="gemma",
        provider_call=provider_call,
        journal_root=journal_root,
    )
    entry_path = next(journal_root.rglob("row-*.json"))
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["row_index"] = 9
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding or content address drifted"):
        run_teacher_model(
            packet,
            manifest,
            model_role="gemma",
            provider_call=provider_call,
            journal_root=journal_root,
        )

    entry_path.unlink()
    unexpected = entry_path.with_name("row-" + "f" * 64 + ".json")
    unexpected.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="1 unexpected row entries"):
        run_teacher_model(
            packet,
            manifest,
            model_role="gemma",
            provider_call=provider_call,
            journal_root=journal_root,
        )


def test_sol_queue_routes_invalid_disagreement_unclear_and_deterministic_exact_audit() -> None:
    packet, manifest, gemma, gpt_oss = _checkpoints()
    queue = build_sol_queue(
        packet,
        manifest,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
    )
    repeated = build_sol_queue(
        packet,
        manifest,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
    )

    assert repeated == queue
    reasons_by_record = {
        route["record_id"]: route["reasons"] for route in queue["routing"]
    }
    assert reasons_by_record["private-record-1"] == ["disagreement"]
    assert reasons_by_record["private-record-2"] == ["unclear"]
    assert reasons_by_record["private-record-3"] == ["invalid"]
    assert list(reasons_by_record.values()).count(["audit"]) == 1
    assert set(reasons_by_record) >= {
        "private-record-1",
        "private-record-2",
        "private-record-3",
    }
    assert queue["item_count"] == 4
    blinded = json.dumps(queue["items"], sort_keys=True)
    assert "private-record" not in blinded
    assert "disagreement" not in blinded
    assert "semantic_label" not in blinded


def test_final_labels_use_internal_provenance_and_public_receipt_is_metadata_only() -> None:
    packet, manifest, gemma, gpt_oss = _checkpoints()
    queue = build_sol_queue(
        packet,
        manifest,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
    )
    sol = freeze_sol_checkpoint(
        manifest,
        queue,
        outputs=[
            {"queue_item_id": item["queue_item_id"], "semantic_label": _label("not_material")}
            for item in queue["items"]
        ],
        observed_model="openai/sol",
        observed_provider="codex",
    )
    final_labels = finalise_labels(
        packet,
        manifest,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
        queue=queue,
        sol_checkpoint=sol,
    )
    receipt = build_public_receipt(
        packet,
        manifest,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
        queue=queue,
        sol_checkpoint=sol,
        final_labels=final_labels,
    )

    provenance = [row["internal_provenance"] for row in final_labels["rows"]]
    assert provenance.count("silver") == 4
    assert provenance.count("bronze") == 1
    receipt_json = json.dumps(receipt, sort_keys=True)
    assert "silver" not in receipt_json and "bronze" not in receipt_json
    assert "private-record" not in receipt_json
    assert "private target text sentinel" not in receipt_json
    assert receipt["decision_origins"] == {
        "sol_adjudicated_count": 3,
        "sol_audited_teacher_agreement_count": 1,
        "unaudited_exact_teacher_agreement_count": 1,
    }
    assert receipt["agreement_audit"] == {
        "selected_count": 1,
        "sol_match_count": 0,
        "sol_mismatch_count": 1,
        "sol_match_rate": 0.0,
    }
    assert_public_metadata_only(receipt, packet=packet)

    with pytest.raises(ValueError, match="private field"):
        assert_public_metadata_only({"target_text": "leak"}, packet=packet)
    with pytest.raises(ValueError, match="private packet content"):
        assert_public_metadata_only(
            {"note": "prefix private target text sentinel 0 suffix"}, packet=packet
        )
    with pytest.raises(ValueError, match="internal provenance"):
        assert_public_metadata_only({"origin": "silver"}, packet=packet)


def test_missing_and_duplicate_sol_outputs_fail_closed() -> None:
    packet, manifest, gemma, gpt_oss = _checkpoints()
    queue = build_sol_queue(
        packet,
        manifest,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
    )
    outputs = [
        {"queue_item_id": item["queue_item_id"], "semantic_label": _label("not_material")}
        for item in queue["items"]
    ]
    with pytest.raises(ValueError, match="row set mismatch: missing=1"):
        freeze_sol_checkpoint(
            manifest,
            queue,
            outputs=outputs[:-1],
            observed_model="openai/sol",
            observed_provider="codex",
        )
    with pytest.raises(ValueError, match="duplicate Sol output"):
        freeze_sol_checkpoint(
            manifest,
            queue,
            outputs=[*outputs, outputs[0]],
            observed_model="openai/sol",
            observed_provider="codex",
        )


def test_sol_provider_receives_only_blinded_items() -> None:
    packet, manifest, gemma, gpt_oss = _checkpoints()
    queue = build_sol_queue(
        packet,
        manifest,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
    )
    calls = []

    def provider_call(**request: object) -> dict[str, object]:
        calls.append(request)
        payload = json.dumps(request, sort_keys=True)
        assert "private-record" not in payload
        assert "disagreement" not in payload
        assert "queue_item_id" not in payload
        assert request["contract"]["prompt_sha256"] == ACTIVE_PROMPT_SHA256
        assert "Produce the complete semantic label independently" in request["prompt"]
        return {
            "value": _label("not_material"),
            "observed_model": "openai/sol",
            "observed_provider": "codex",
        }

    checkpoint = run_sol_adjudication(
        packet,
        manifest,
        queue,
        gemma_checkpoint=gemma,
        gpt_oss_checkpoint=gpt_oss,
        provider_call=provider_call,
    )

    assert len(calls) == queue["item_count"]
    assert checkpoint["row_count"] == queue["item_count"]

    invalid_calls = []

    def invalid_provider(**request: object) -> dict[str, object]:
        invalid_calls.append(request)
        return {
            "value": {"unexpected": "shape"},
            "observed_model": "openai/sol",
            "observed_provider": "codex",
        }

    with pytest.raises(ValueError, match="invalid semantic label"):
        run_sol_adjudication(
            packet,
            manifest,
            queue,
            gemma_checkpoint=gemma,
            gpt_oss_checkpoint=gpt_oss,
            provider_call=invalid_provider,
        )
    assert len(invalid_calls) == 1


def test_immutable_writer_is_idempotent_and_rejects_drift(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    write_immutable_json(path, {"value": 1})
    write_immutable_json(path, {"value": 1})
    with pytest.raises(RuntimeError, match="immutable output differs"):
        write_immutable_json(path, {"value": 2})
