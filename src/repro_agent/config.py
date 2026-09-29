"""Liest .env, waehlt Anbieter (openrouter | tokenfactory) und Sandbox-Backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
# Nicht-leere Shell-Variablen gewinnen (z. B. `LLM_PROVIDER=tokenfactory repro-agent ...`).
# Leere oder fehlende Variablen fuellt .env, damit ein leerer Key nichts still ueberdeckt.
for _key, _value in dotenv_values(ENV_PATH).items():
    if _value is not None and not os.environ.get(_key):
        os.environ[_key] = _value


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(f"Fehlt oder leer: {', '.join(missing)} (.env pruefen)")


def _get(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Fehlende Umgebungsvariable: {name}")
    return value


def _get_optional(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class LLMConfig:
    provider: str            # "openrouter" | "tokenfactory"
    base_url: str
    api_key: str
    model_fast: str
    model_code: str
    model_reasoning: str


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    sandbox_backend: str      # "local" | "nebius"
    tavily_api_key: str
    max_cost_per_run_usd: float
    max_attempts_per_claim: int
    llm_cache: bool
    vat_rate: float = 0.19            # MwSt.; Log speichert netto, Limits gelten brutto
    budget_total_usd: float = 0.0     # brutto, Kontostand zum Stichtag; 0 = keine Pruefung
    budget_since: str = ""            # Stichtag (ISO, UTC): nur Log-Eintraege ab dann zaehlen


def load_settings() -> Settings:
    provider = _get_optional("LLM_PROVIDER", "openrouter").lower()

    if _get_optional("SANDBOX_BACKEND", "nebius").lower() == "nebius":
        require_env("NEBIUS_API_KEY", "NEBIUS_PROJECT_ID")

    if provider == "openrouter":
        llm = LLMConfig(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=_get("OPENROUTER_API_KEY"),
            model_fast=_get("OR_MODEL_FAST"),
            model_code=_get("OR_MODEL_CODE"),
            model_reasoning=_get("OR_MODEL_REASONING"),
        )
    elif provider == "tokenfactory":
        llm = LLMConfig(
            provider="tokenfactory",
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=_get("NEBIUS_API_KEY"),
            model_fast=_get("TF_MODEL_FAST"),
            model_code=_get("TF_MODEL_CODE"),
            model_reasoning=_get("TF_MODEL_REASONING"),
        )
    else:
        raise RuntimeError(f"Unbekannter LLM_PROVIDER: {provider}")

    return Settings(
        llm=llm,
        sandbox_backend=_get_optional("SANDBOX_BACKEND", "nebius").lower(),
        tavily_api_key=_get_optional("TAVILY_API_KEY"),
        max_cost_per_run_usd=float(_get_optional("MAX_COST_PER_RUN_USD", "0.50")),
        max_attempts_per_claim=int(_get_optional("MAX_ATTEMPTS_PER_CLAIM", "4")),
        llm_cache=_get_optional("LLM_CACHE", "on").lower() == "on",
        vat_rate=float(_get_optional("VAT_RATE", "0.19")),
        budget_total_usd=float(_get_optional("BUDGET_TOTAL_USD", "29.75")),
        budget_since=_get_optional("BUDGET_SINCE", ""),
    )
