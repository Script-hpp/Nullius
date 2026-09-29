"""LLM-Client: OpenRouter (Prototyp) | Token Factory (Prod). Retry, Cache, Kosten-Log.

Anbieter-Sonderfaelle nur hier, nirgendwo sonst im Code.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from .config import Settings, load_settings
from .costs import BudgetExceeded, budget_status, gross, logged_gross_total

# "extract": strukturierte Extraktion (Rollen 2/4). Braucht kein Ultra, nutzt das code-Modell (Super).
Role = Literal["fast", "code", "reasoning", "extract"]

# USD pro Token (Token Factory, GET /v1/models am 2026-09-28). OpenRouter meldet Kosten selbst.
TF_PRICES = {
    "nvidia/Nemotron-3_5-Lightning": (0.06e-6, 0.24e-6),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06e-6, 0.24e-6),
    "nvidia/nemotron-3-super-120b-a12b": (0.3e-6, 0.9e-6),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.0e-6, 3.0e-6),
}

# Abrechnungs-/Auth-Fehler: Statuscodes, oder Text (ausser 5xx). Kein "credit"/"balance" allein:
# OpenRouters Free-Tageslimit (429) nennt "credits" und ist kein Abrechnungsproblem.
BILLING_STATUS = (401, 402, 403)
BILLING_WORDS = ("billing", "suspend", "insufficient", "payment", "api key")

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
    finish_reason: str | None = None
    reasoning_tokens: int | None = None   # in completion_tokens enthalten (usage.completion_tokens_details)
    status: str = "ok"                    # "ok" | "bad_response" (bezahlt, aber ohne Inhalt)


def is_billing_error(status: int | None, text: str) -> bool:
    if status in BILLING_STATUS:
        return True
    if status is not None and status >= 500:
        return False
    return any(w in text.lower() for w in BILLING_WORDS)


class BillingBlocked(RuntimeError):
    """Abrechnung oder Zugang blockiert. Kein Retry, kein Reparaturversuch, ganzer Lauf bricht ab."""


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
        self._blocked: str | None = None     # gesetzt = alle weiteren Aufrufe scheitern sofort
        self._warned_budget = False
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
            "extract": self.settings.llm.model_code,
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
        if self._blocked:
            raise BillingBlocked(self._blocked)
        model = self._model_for_role(role)
        cache_key = self._cache_key(role, messages, temperature)

        if self.settings.llm_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            response = LLMResponse(**{**cached, "cached": True})
            self._log_cost(response)
            return response

        self._check_budget()
        response = self._call_with_retry(model, messages, temperature, max_tokens, timeout_s)
        self._book(response)

        if self.settings.llm_cache:
            self._append_cache(cache_key, response)

        return response

    def _book(self, response: LLMResponse) -> None:
        with self._cost_lock:
            self.total_cost_usd += response.cost_usd
        self._log_cost(response)

    def gross(self, net_usd: float) -> float:
        return gross(net_usd, self.settings.vat_rate)

    def _check_budget(self) -> None:
        """Globales Budget (brutto) gegen das Kosten-Log: ab 80 % Warnung, ab 95 % Abbruch."""
        total = self.settings.budget_total_usd
        spent = logged_gross_total(COST_LOG_PATH, self.settings.vat_rate) if total > 0 else 0.0
        status = budget_status(spent, total)
        if status == "abort":
            raise BudgetExceeded(
                f"Budget fast erschoepft: {spent:.2f} $ von {total:.2f} $ brutto laut Log "
                f"({spent / total:.0%}, Grenze 95 %). Konsole pruefen; Log ist nur eine Schaetzung."
            )
        if status == "warn" and not self._warned_budget:
            self._warned_budget = True
            print(f"WARNUNG: {spent:.2f} $ von {total:.2f} $ Budget brutto verbraucht ({spent / total:.0%}).",
                  file=sys.stderr, flush=True)

    def _block(self, status: int | None, text: str) -> None:
        provider = self.settings.llm.provider
        who = "Nebius-Abrechnung" if provider == "tokenfactory" else f"{provider}-Abrechnung/Zugang"
        self._blocked = f"{who} blockiert – Konsole prüfen (HTTP {status}: {text[:200]})"
        raise BillingBlocked(self._blocked)

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
                if resp.status_code >= 400 and is_billing_error(resp.status_code, resp.text):
                    self._block(resp.status_code, resp.text)
                resp.raise_for_status()
                data = resp.json()
                if data.get("error") and is_billing_error(resp.status_code, str(data["error"])):
                    self._block(resp.status_code, str(data["error"]))

                usage = data.get("usage") or {}
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    content = None
                finish = (data.get("choices") or [{}])[0].get("finish_reason")
                if not content:
                    if usage.get("prompt_tokens") or usage.get("completion_tokens"):
                        self._book(self._response(model, "", usage, finish, "bad_response"))   # bezahlt
                    raise BadResponse(f"Antwort ohne Inhalt: {str(data.get('error', data))[:200]}")

                return self._response(model, content, usage, finish, "ok")
            except (httpx.HTTPStatusError, httpx.TransportError, BadResponse, ValueError) as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(BACKOFF_BASE_S * (2**attempt))

        raise RuntimeError(
            f"LLM-Aufruf fehlgeschlagen nach {MAX_RETRIES} Versuchen (Modell {model}): {last_error}"
        )

    def _response(self, model: str, content: str, usage: dict, finish: str | None, status: str) -> LLMResponse:
        details = usage.get("completion_tokens_details") or {}
        return LLMResponse(
            content=content, model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),   # aus usage der API, nie geschaetzt
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=usage.get("cost") or self._estimate_cost(model, usage),
            cached=False, finish_reason=finish,
            reasoning_tokens=details.get("reasoning_tokens"), status=status,
        )

    def _estimate_cost(self, model: str, usage: dict) -> float:
        if model not in TF_PRICES and self.settings.llm.provider == "tokenfactory":
            print(f"WARNUNG: kein Preis fuer Modell {model} in TF_PRICES, Kosten als 0 geloggt.",
                  file=sys.stderr, flush=True)
        p_in, p_out = TF_PRICES.get(model, (0.0, 0.0))
        return usage.get("prompt_tokens", 0) * p_in + usage.get("completion_tokens", 0) * p_out

    def _log_cost(self, response: LLMResponse) -> None:
        COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "provider": self.settings.llm.provider,
            "base_url": self.settings.llm.base_url,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": 0.0 if response.cached else response.cost_usd,
            "cache_hit": response.cached,
            "finish_reason": response.finish_reason,
            "reasoning_tokens": response.reasoning_tokens,
            "status": "cached" if response.cached else response.status,
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
