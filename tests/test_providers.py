from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from reddit_china_stance.providers import (
    MODEL_ROUTES,
    call_phase,
    effective_prompt_sha256,
    preflight_route,
    route_manifest,
    run_three_stage,
    teacher_provider_call,
)
from reddit_china_stance.teacher_pipeline import ACTIVE_PROMPT_SHA256


def _packet() -> dict[str, str | None]:
    return {
        "target_text": "Synthetic target text.",
        "submission_context": None,
        "parent_context": None,
    }


def test_three_stage_short_circuits_non_material() -> None:
    phases: list[str] = []

    def call(**kwargs: Any) -> dict[str, Any]:
        phases.append(kwargs["phase"])
        return {"value": {"relevance": "not_material"}}

    result = run_three_stage(route_name="gemma", packet=_packet(), seed=1, phase_call=call)
    assert result["label"] == {"relevance": "not_material", "target_stances": []}
    assert phases == ["relevance"]


def test_three_stage_calls_stance_once_per_target() -> None:
    calls: list[tuple[str, str]] = []

    def call(**kwargs: Any) -> dict[str, Any]:
        phase = kwargs["phase"]
        content = kwargs["content"]
        calls.append((phase, content))
        if phase == "relevance":
            return {"value": {"relevance": "material"}}
        if phase == "targets":
            return {"value": {"targets": ["people_culture", "government_ccp"]}}
        stance = "negative" if "government_ccp" in content else "positive"
        return {"value": {"stance": stance}}

    result = run_three_stage(route_name="gpt-oss", packet=_packet(), seed=7, phase_call=call)
    assert result["label"]["target_stances"] == [
        {"target": "government_ccp", "stance": "negative"},
        {"target": "people_culture", "stance": "positive"},
    ]
    assert [phase for phase, _ in calls] == ["relevance", "targets", "stance", "stance"]


def test_three_stage_rejects_noncanonical_provider_output() -> None:
    def call(**kwargs: Any) -> dict[str, Any]:
        return {"value": {"relevance": "direct"}}

    with pytest.raises(ValueError, match="invalid relevance"):
        run_three_stage(route_name="gemma", packet=_packet(), seed=1, phase_call=call)


def test_preflight_requires_exact_revision_and_live_route() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload: dict[str, Any] = {
            "id": MODEL_ROUTES["gemma"].model,
            "sha": MODEL_ROUTES["gemma"].revision,
        }
        if request.url.params.get("expand"):
            payload["inferenceProviderMapping"] = {
                "cerebras": {
                    "status": "live",
                    "task": "conversational",
                    "providerId": MODEL_ROUTES["gemma"].observed_model,
                }
            }
        return httpx.Response(200, json=payload)

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://huggingface.co"
    ) as client:
        result = preflight_route("gemma", client=client)

    assert len(requests) == 2
    assert result["revision"] == MODEL_ROUTES["gemma"].revision
    assert result["provider_model"] == MODEL_ROUTES["gemma"].observed_model


def test_phase_call_sends_one_strict_request_and_validates_response() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": MODEL_ROUTES["gemma"].observed_model,
                "choices": [{"message": {"content": '{"relevance":"material"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            },
            headers={"x-inference-provider": "cerebras"},
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://router.huggingface.co/v1",
    ) as client:
        result = call_phase(
            route_name="gemma",
            phase="relevance",
            content="synthetic bounded input",
            seed=17,
            client=client,
        )

    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == f"{MODEL_ROUTES['gemma'].model}:cerebras"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "reasoning_effort" not in request
    assert result["value"] == {"relevance": "material"}
    assert result["usage"]["input_tokens"] == 12


def test_phase_call_fails_after_one_provider_error_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request)

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://router.huggingface.co/v1",
    ) as client, pytest.raises(RuntimeError, match="provider request failed"):
        call_phase(
            route_name="gpt-oss",
            phase="relevance",
            content="synthetic bounded input",
            seed=17,
            client=client,
        )

    assert calls == 1


def test_phase_call_rejects_missing_observed_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"relevance":"material"}'}}]},
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://router.huggingface.co/v1",
    ) as client, pytest.raises(RuntimeError, match="omitted observed model identity"):
        call_phase(
            route_name="gemma",
            phase="relevance",
            content="synthetic bounded input",
            seed=17,
            client=client,
        )


def test_teacher_adapter_binds_route_prompt_and_seed() -> None:
    assert effective_prompt_sha256() == ACTIVE_PROMPT_SHA256
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": MODEL_ROUTES["gemma"].observed_model,
                "provider": "cerebras",
                "choices": [{"message": {"content": '{"relevance":"material"}'}}],
            },
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://router.huggingface.co/v1",
    ) as client:
        result = teacher_provider_call(
            model_role="gemma",
            model_config=route_manifest()["gemma"],
            stage="relevance",
            seed=23,
            item=_packet(),
            prior=None,
            target=None,
            contract={"prompt_sha256": effective_prompt_sha256()},
            client=client,
        )

    assert requests[0]["seed"] == 23
    assert result["observed_model"] == MODEL_ROUTES["gemma"].observed_model

    drifted = route_manifest()["gemma"]
    drifted["revision"] = "drifted"
    with pytest.raises(RuntimeError, match="configuration differs"):
        teacher_provider_call(
            model_role="gemma",
            model_config=drifted,
            stage="relevance",
            seed=23,
            item=_packet(),
            prior=None,
            target=None,
            contract={"prompt_sha256": effective_prompt_sha256()},
        )
