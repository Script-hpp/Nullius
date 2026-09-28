"""LLM-Client: OpenRouter (Prototyp) | Token Factory (Prod). Retry, Cache, Kosten-Log.

Anbieter-Sonderfaelle nur hier, nirgendwo sonst im Code.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from .config import Settings, load_settings

Role = Literal["fast", "code", "reasoning"]

# USD pro Token (Token Factory, GET /v1/models am 2026-09-28). OpenRouter meldet Kosten selbst.
TF_PRICES = {
    "nvidia/Nemotron-3_5-Lightning": (0.06e-6, 0.24e-6),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06e-6, 0.24e-6),
    "nvidia/nemotron-3-super-120b-a12b": (0.3e-6, 0.9e-6),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.0e-6, 3.0e-6),
}

MAX_RETRIES = 5
BACKOFF_BASE_S = 1.0

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "logs" / "llm_cache.jsonl"
COST_LOG_PATH = REPO_ROOT / "logs" / "costs.jsonl"


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cached: bool


class BadResponse(Exception):
    """HTTP 200, aber kein nutzbarer Inhalt (z. B. Rate-Limit-Fehler im Body). Wird wiederholt."""


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._client = httpx.Client(
            base_url=self.settings.llm.base_url,
            headers={"Authorization": f"Bearer {self.settings.llm.api_key}"},
            timeout=120.0,
        )
        self._cache: dict[str, dict] = {}
        self.total_cost_usd = 0.0   # alle Aufrufe dieser Instanz (Rollen 2/4 eingeschlossen)
        self._cost_lock = threading.Lock()
        if self.settings.llm_cache:
            self._load_cache()

    def _load_cache(self) -> None:
        if not CACHE_PATH.exists():
            return
        for line in CACHE_PATH.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            self._cache[entry["key"]] = entry["response"]

    def _cache_key(self, role: Role, messages: list[dict], temperature: float) -> str:
        model = self._model_for_role(role)
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _model_for_role(self, role: Role) -> str:
        return {
            "fast": self.settings.llm.model_fast,
            "code": self.settings.llm.model_code,
            "reasoning": self.settings.llm.model_reasoning,
        }[role]

    def complete(
        self,
        role: Role,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        model = self._model_for_role(role)
        cache_key = self._cache_key(role, messages, temperature)

        if self.settings.llm_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            response = LLMResponse(**{**cached, "cached": True})
            self._log_cost(response)
            return response

        response = self._call_with_retry(model, messages, temperature, max_tokens, timeout_s)
        with self._cost_lock:
            self.total_cost_usd += response.cost_usd
        self._log_cost(response)

        if self.settings.llm_cache:
            self._append_cache(cache_key, response)

        return response

    def _call_with_retry(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                body: dict = {"model": model, "messages": messages, "temperature": temperature}
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens

                resp = self._client.post(
                    "/chat/completions", json=body, timeout=timeout_s or httpx.USE_CLIENT_DEFAULT
                )
                resp.raise_for_status()
                data = resp.json()

                usage = data.get("usage", {})
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    content = None
                if not content:
                    raise BadResponse(f"Antwort ohne Inhalt: {str(data.get('error', data))[:200]}")

                return LLMResponse(
                    content=content,
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    cost_usd=usage.get("cost") or self._estimate_cost(model, usage),
                    cached=False,
                )
            except (httpx.HTTPStatusError, httpx.TransportError, BadResponse, ValueError) as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(BACKOFF_BASE_S * (2**attempt))

        raise RuntimeError(
            f"LLM-Aufruf fehlgeschlagen nach {MAX_RETRIES} Versuchen (Modell {model}): {last_error}"
        )

    @staticmethod
    def _estimate_cost(model: str, usage: dict) -> float:
        p_in, p_out = TF_PRICES.get(model, (0.0, 0.0))
        return usage.get("prompt_tokens", 0) * p_in + usage.get("completion_tokens", 0) * p_out

    def _log_cost(self, response: LLMResponse) -> None:
        COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": 0.0 if response.cached else response.cost_usd,
            "cache_hit": response.cached,
        }
        with COST_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _append_cache(self, key: str, response: LLMResponse) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "response": {
                "content": response.content,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cost_usd": response.cost_usd,
            },
        }
        self._cache[key] = entry["response"]
        with CACHE_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        self._client.close()
