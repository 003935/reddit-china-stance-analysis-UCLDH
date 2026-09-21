"""Pinned Hugging Face provider routes for the active annotation pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from reddit_china_stance.annotation_contract import (
    EFFECTIVE_PROMPTS,
    PHASE_SCHEMAS,
    PROMPT_BUNDLE,
    compose_label,
    model_input,
    validate_phase_output,
)


@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    model: str
    observed_model: str
    revision: str
    reasoning_effort: str
    max_output_tokens: int = 512

    @property
    def routed_model(self) -> str:
        return f"{self.model}:{self.provider}"


MODEL_ROUTES = {
    "gemma": ProviderRoute(
        provider="cerebras",
        model="google/gemma-4-31B-it",
        observed_model="gemma-4-31b",
        revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
        reasoning_effort="none",
    ),
    "gpt-oss": ProviderRoute(
        provider="groq",
        model="openai/gpt-oss-20b",
        observed_model="openai/gpt-oss-20b",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        reasoning_effort="low",
    ),
}


def route_manifest() -> dict[str, dict[str, Any]]:
    """Return the immutable provider configuration for manifest binding."""

    return {
        name: {
            "provider": route.provider,
            "model_id": route.model,
            "observed_model_id": route.observed_model,
            "revision": route.revision,
            "reasoning_effort": route.reasoning_effort,
            "max_output_tokens": route.max_output_tokens,
        }
        for name, route in MODEL_ROUTES.items()
    }


def effective_prompt_sha256() -> str:
    payload = json.dumps(
        PROMPT_BUNDLE,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token() -> str:
    value = os.environ.get("HF_TOKEN", "").lstrip(chr(0xFEFF)).strip()
    if not value:
        raise RuntimeError("HF_TOKEN is required")
    return value


def inference_client() -> httpx.Client:
    return httpx.Client(
        base_url="https://router.huggingface.co/v1",
        headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"},
        timeout=90.0,
    )


def hub_client() -> httpx.Client:
    return httpx.Client(
        base_url="https://huggingface.co",
        headers={"Authorization": f"Bearer {_token()}"},
        timeout=30.0,
    )


def preflight_route(name: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """Require the pinned revision and a live conversational provider route."""

    try:
        route = MODEL_ROUTES[name]
    except KeyError as exc:
        raise ValueError(f"unknown provider route: {name}") from exc
    owned_client = client is None
    client = client or hub_client()
    try:
        metadata_response = client.get(f"/api/models/{route.model}")
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        if metadata.get("id") != route.model:
            raise RuntimeError("Hugging Face model identity drifted")
        if metadata.get("sha") != route.revision:
            raise RuntimeError("Hugging Face model revision drifted")
        mapping_response = client.get(
            f"/api/models/{route.model}", params={"expand": "inferenceProviderMapping"}
        )
        mapping_response.raise_for_status()
        mapping = mapping_response.json().get("inferenceProviderMapping", {})
        selected = mapping.get(route.provider)
        if not isinstance(selected, Mapping):
            raise RuntimeError("configured provider mapping is unavailable")
        if selected.get("status") != "live" or selected.get("task") != "conversational":
            raise RuntimeError("configured provider route is not live for chat")
        if selected.get("providerId") != route.observed_model:
            raise RuntimeError("configured provider model identity drifted")
        return {
            "route": name,
            "model": route.model,
            "revision": route.revision,
            "provider": route.provider,
            "provider_model": selected.get("providerId"),
            "structured_output_required": True,
        }
    finally:
        if owned_client:
            client.close()


def _normalise_usage(value: Mapping[str, Any] | None) -> dict[str, int | float | None]:
    value = value or {}
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "reasoning_tokens": ("reasoning_tokens",),
        "total_tokens": ("total_tokens",),
        "cost_usd": ("cost", "cost_usd"),
    }
    return {
        target: next((value[field] for field in fields if value.get(field) is not None), None)
        for target, fields in aliases.items()
    }


def call_phase(
    *,
    route_name: str,
    phase: str,
    content: str,
    seed: int,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Call one strict atomic phase once; callers resume from immutable checkpoints."""

    if phase not in PHASE_SCHEMAS:
        raise ValueError(f"unknown phase: {phase}")
    try:
        route = MODEL_ROUTES[route_name]
    except KeyError as exc:
        raise ValueError(f"unknown provider route: {route_name}") from exc
    body: dict[str, Any] = {
        "model": route.routed_model,
        "messages": [
            {"role": "system", "content": EFFECTIVE_PROMPTS[phase]},
            {"role": "user", "content": content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"{phase}_label",
                "strict": True,
                "schema": PHASE_SCHEMAS[phase],
            },
        },
        "seed": seed,
        "max_tokens": route.max_output_tokens,
    }
    if route.reasoning_effort != "none":
        body["reasoning_effort"] = route.reasoning_effort

    owned_client = client is None
    client = client or inference_client()
    started = time.monotonic()
    try:
        response = client.post("/chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"provider request failed for {route_name}/{phase}") from exc
    finally:
        if owned_client:
            client.close()
    elapsed_seconds = round(time.monotonic() - started, 6)

    try:
        raw_content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("provider response did not contain valid structured JSON") from exc
    if not isinstance(parsed, Mapping):
        raise RuntimeError("provider response JSON must be an object")
    observed_model = payload.get("model")
    observed_provider = payload.get("provider") or response.headers.get(
        "x-inference-provider"
    )
    if not isinstance(observed_model, str) or not observed_model:
        raise RuntimeError("provider response omitted observed model identity")
    if not isinstance(observed_provider, str) or not observed_provider:
        raise RuntimeError("provider response omitted observed provider identity")
    value = validate_phase_output(phase, parsed)
    return {
        "value": value,
        "observed_model": observed_model,
        "observed_provider": observed_provider,
        "elapsed_seconds": elapsed_seconds,
        "usage": _normalise_usage(payload.get("usage")),
    }


PhaseCall = Callable[..., Mapping[str, Any]]


def teacher_provider_call(
    *,
    model_role: str,
    model_config: Mapping[str, Any],
    stage: str,
    seed: int,
    item: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
    target: str | None,
    contract: Mapping[str, Any],
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Adapt the frozen teacher callback contract to one Hugging Face phase call."""

    del prior
    route_name = {"gemma": "gemma", "gpt_oss": "gpt-oss"}.get(model_role)
    if route_name is None:
        raise ValueError(f"unsupported Hugging Face teacher role: {model_role}")
    if dict(model_config) != route_manifest()[route_name]:
        raise RuntimeError("teacher model configuration differs from the pinned provider route")
    if contract.get("prompt_sha256") != effective_prompt_sha256():
        raise RuntimeError("effective prompt digest differs from the frozen teacher manifest")
    return call_phase(
        route_name=route_name,
        phase=stage,
        content=model_input(item, target=target),
        seed=seed,
        client=client,
    )


def run_three_stage(
    *,
    route_name: str,
    packet: Mapping[str, Any],
    seed: int,
    phase_call: PhaseCall = call_phase,
) -> dict[str, Any]:
    """Run relevance, then targets, then one stance request per selected target."""

    phase_results: list[dict[str, Any]] = []

    def invoke(phase: str, *, target: str | None = None, offset: int = 0) -> dict[str, Any]:
        result = dict(
            phase_call(
                route_name=route_name,
                phase=phase,
                content=model_input(packet, target=target),
                seed=seed + offset,
            )
        )
        value = result.get("value")
        if not isinstance(value, Mapping):
            raise RuntimeError("provider result contains no phase value")
        result["value"] = validate_phase_output(phase, value)
        phase_results.append({"phase": phase, **({"target": target} if target else {}), **result})
        return result

    relevance = invoke("relevance")
    relevance_value = relevance["value"]
    if relevance_value["relevance"] != "material":
        label = compose_label(relevance_value)
    else:
        targets = invoke("targets", offset=10_000)["value"]
        stances = {
            target: invoke("stance", target=target, offset=20_000 + index)["value"]
            for index, target in enumerate(targets["targets"])
        }
        label = compose_label(relevance_value, targets, stances)
    return {"label": label, "phase_results": phase_results}
