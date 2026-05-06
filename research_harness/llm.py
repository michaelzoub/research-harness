from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        self.provider = (provider or os.environ.get("RESEARCH_HARNESS_LLM_PROVIDER") or "auto").lower()
        self.model = model or os.environ.get("RESEARCH_HARNESS_LLM_MODEL") or "gpt-4.1-mini"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds

    @property
    def is_live(self) -> bool:
        if self.provider == "local":
            return False
        return _looks_like_openai_key(self.api_key) and self.provider in {"auto", "openai"}

    @property
    def model_label(self) -> str:
        if self.is_live:
            return self.model
        return "local-deterministic-fallback"

    def complete(self, system: str, user: str, *, max_output_tokens: int = 900) -> LLMResponse:
        if not self.is_live:
            return LLMResponse(
                text=self._local_response(system, user),
                model=self.model_label,
                provider="local",
                prompt_tokens=_estimate_tokens(system + "\n" + user),
                completion_tokens=80,
            )
        return self._openai_response(system, user, max_output_tokens=max_output_tokens)

    def complete_json(self, system: str, user: str, *, max_output_tokens: int = 900) -> dict[str, object]:
        response = self.complete(system, user, max_output_tokens=max_output_tokens)
        try:
            return json.loads(_extract_json(response.text))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {exc}") from exc

    def _openai_response(self, system: str, user: str, *, max_output_tokens: int) -> LLMResponse:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "max_output_tokens": max_output_tokens,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "research-harness/0.1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = _responses_text(data)
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self.model),
            provider="openai",
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
        )

    def validate(self) -> bool:
        if not self.is_live:
            return False
        payload = {
            "model": self.model,
            "input": "Return ok.",
            "max_output_tokens": 8,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "research-harness/0.1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 15.0)):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return False
            raise

    def _local_response(self, system: str, user: str) -> str:
        if "json" in system.lower():
            return json.dumps({"score": 0.5, "rationale": "Local fallback score; configure OPENAI_API_KEY for live judging."})
        return (
            "Local deterministic fallback response. Configure OPENAI_API_KEY and "
            "RESEARCH_HARNESS_LLM_PROVIDER=openai to use a live model.\n\n"
            f"Prompt excerpt: {user[:500]}"
        )


def _responses_text(data: dict[str, object]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    chunks: list[str] = []
    for item in data.get("output", []) if isinstance(data.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _looks_like_openai_key(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    cleaned = api_key.strip()
    if cleaned in {"", "...", "changeme", "your-key-here"}:
        return False
    return cleaned.startswith(("sk-", "sess-"))
