from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


class LLMError(RuntimeError):
    pass


# USD per token for known models. Used to compute real per-run costs.
# Prices sourced from public pricing pages; marked as estimates for unreleased models.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50e-6, "output": 10.00e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
    "gpt-4-turbo": {"input": 10.00e-6, "output": 30.00e-6},
    "gpt-4": {"input": 30.00e-6, "output": 60.00e-6},
    "gpt-3.5-turbo": {"input": 0.50e-6, "output": 1.50e-6},
    "o1": {"input": 15.00e-6, "output": 60.00e-6},
    "o1-mini": {"input": 3.00e-6, "output": 12.00e-6},
    "o3": {"input": 10.00e-6, "output": 40.00e-6},
    "o3-mini": {"input": 1.10e-6, "output": 4.40e-6},
    "o4-mini": {"input": 1.10e-6, "output": 4.40e-6},
    "gpt-5": {"input": 10.00e-6, "output": 30.00e-6},   # estimate
    "gpt-5.2": {"input": 10.00e-6, "output": 30.00e-6},  # estimate
}
_DEFAULT_PRICING: dict[str, float] = {"input": 10.00e-6, "output": 30.00e-6}


def _pricing_for(model: str) -> dict[str, float]:
    # Match by prefix so "gpt-4o-2024-05-13" resolves to "gpt-4o".
    for key, pricing in _MODEL_PRICING.items():
        if model.startswith(key):
            return pricing
    return _DEFAULT_PRICING


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
        self.model = model or os.environ.get("RESEARCH_HARNESS_LLM_MODEL") or "gpt-5.2"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        # Accumulated real token counts across all calls on this client instance.
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

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

    def complete(self, system: str, user: str, *, max_output_tokens: int = 900, temperature: float = 0.7) -> LLMResponse:
        if not self.is_live:
            response = LLMResponse(
                text=self._local_response(system, user),
                model=self.model_label,
                provider="local",
                prompt_tokens=_estimate_tokens(system + "\n" + user),
                completion_tokens=80,
            )
        else:
            response = self._openai_response(system, user, max_output_tokens=max_output_tokens, temperature=temperature)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        return response

    def total_cost(self) -> float:
        """Return accumulated cost in USD based on model pricing table."""
        pricing = _pricing_for(self.model)
        return (
            self.total_prompt_tokens * pricing["input"]
            + self.total_completion_tokens * pricing["output"]
        )

    def cost_breakdown(self) -> dict[str, object]:
        pricing = _pricing_for(self.model)
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "cost_usd": round(self.total_cost(), 6),
            "pricing_input_per_token": pricing["input"],
            "pricing_output_per_token": pricing["output"],
            "pricing_note": "Prices are estimates for unreleased models; verify against provider billing.",
        }

    def complete_json(self, system: str, user: str, *, max_output_tokens: int = 900, temperature: float = 0.7) -> dict[str, object]:
        response = self.complete(system, user, max_output_tokens=max_output_tokens, temperature=temperature)
        try:
            return json.loads(_extract_json(response.text))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {exc}") from exc

    def _openai_response(self, system: str, user: str, *, max_output_tokens: int, temperature: float = 0.7) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_output_tokens,
            "temperature": round(max(0.0, min(2.0, temperature)), 2),
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
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
        text = str(data["choices"][0]["message"]["content"] or "")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self.model),
            provider="openai",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def validate(self) -> bool:
        if not self.is_live:
            return False
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Return ok."}],
            "max_tokens": 8,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
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
