"""Phase 5 LLM HTTP client. Server-independent by construction.

The production transport talks to a local `llama-server` chat-completions
endpoint. Tests inject a fake transport, so no test ever needs a live model.
The client only moves bytes: it never renders a note and never repairs a
bad response. Validation lives in `scripts/phase5_validate.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from scripts.phase5_packet import canonical_bytes

Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class SynthesisError(RuntimeError):
    """Any failure to obtain a usable structured response. Fail closed."""


def build_request(
    system_prompt: str,
    packet: dict[str, Any],
    decoding: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": canonical_bytes(packet).decode("utf-8"),
            },
        ],
        "temperature": decoding["temperature"],
        "top_k": decoding["top_k"],
        "top_p": decoding["top_p"],
        "seed": decoding["seed"],
        "n_predict": decoding["n_predict"],
        "response_format": {"type": "json_object"},
    }
    return payload


def httpx_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise SynthesisError(f"synthesis transport failed: {exc}") from exc
    if response.status_code != 200:
        raise SynthesisError(f"synthesis server returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise SynthesisError(f"synthesis server returned non-JSON body: {exc}") from exc


class SynthesisClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        decoding: dict[str, Any],
        system_prompt: str,
        timeout_s: float = 120.0,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.decoding = decoding
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.transport = transport or httpx_transport

    def complete(self, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = build_request(self.system_prompt, packet, self.decoding, self.model)
        try:
            body = self.transport(
                f"{self.base_url}/v1/chat/completions", payload, self.timeout_s
            )
        except SynthesisError:
            raise
        except Exception as exc:
            raise SynthesisError(f"synthesis transport failed: {exc}") from exc
        return payload, extract_json(body)


def extract_json(body: dict[str, Any]) -> dict[str, Any]:
    import json as _json

    try:
        choices = body["choices"]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SynthesisError(f"synthesis response has no chat content: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise SynthesisError("synthesis response content is empty")
    try:
        parsed = _json.loads(content)
    except ValueError as exc:
        raise SynthesisError(f"synthesis response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SynthesisError("synthesis response JSON is not an object")
    return parsed
