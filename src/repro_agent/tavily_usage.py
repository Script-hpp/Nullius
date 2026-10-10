"""Tavily-Credits zaehlen: ein Log-Eintrag je API-Aufruf (`logs/tavily_usage.jsonl`), gelesen von der Live-Sperre.

Das Log ist eine Schaetzung (wie `logs/costs.jsonl`), massgeblich ist das Tavily-Dashboard. Im Zweifel wird zu hoch geschaetzt.

Credit-Tabelle (Tavily-Doku "API Credits Overview", docs.tavily.com, Stand der Doku-Kenntnis; vor Abgabe gegen das Dashboard pruefen):
- Search: basic 1, advanced 2 Credits je Aufruf.
- Extract: je 5 URLs 1 Credit (basic) bzw. 2 Credits (advanced). Geschaetzt wird mit der Zahl der angefragten URLs (die Doku
  rechnet nur erfolgreiche ab, wir zaehlen konservativ alle).
- Crawl: Mapping 1 Credit je 10 Seiten (2 mit `instructions`) plus Extraktion je 5 Seiten 1 Credit (basic) bzw. 2 (advanced).
  Seitenzahl = `limit` (obere Grenze; Tavily rechnet die tatsaechlich besuchten Seiten ab).
Fehlgeschlagene Aufrufe werden wie erfolgreiche gezaehlt (konservativ), mit Feld `error`.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

from .costs import parse_since
from .costs import BudgetExceeded
from . import spend_guard
import uuid

TAVILY_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "tavily_usage.jsonl"   # Tests biegen den Pfad um
_lock = threading.Lock()

DEFAULT_CRAWL_LIMIT = 50   # Tavily-Standard, falls ein Aufruf kein `limit` setzt (der Code setzt es immer)


def estimate_credits(endpoint: str, kwargs: dict) -> int:
    if endpoint == "search":
        return 2 if kwargs.get("search_depth") == "advanced" else 1
    advanced = kwargs.get("extract_depth") == "advanced"
    if endpoint == "extract":
        urls = kwargs.get("urls") or []
        n = 1 if isinstance(urls, str) else max(1, len(urls))
        return math.ceil(n / 5) * (2 if advanced else 1)
    if endpoint == "crawl":
        pages = int(kwargs.get("limit") or DEFAULT_CRAWL_LIMIT)
        mapping = math.ceil(pages / 10) * (2 if kwargs.get("instructions") else 1)
        return mapping + math.ceil(pages / 5) * (2 if advanced else 1)
    return 1   # unbekannter Endpunkt: mindestens 1


def log_call(endpoint: str, kwargs: dict, error: str | None = None) -> int:
    credits = estimate_credits(endpoint, kwargs)
    entry = {"ts": time.time(), "endpoint": endpoint, "credits": credits}
    if endpoint == "search":
        entry["depth"] = kwargs.get("search_depth", "basic")
    if error:
        entry["error"] = error[:200]
    with _lock:
        TAVILY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TAVILY_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return credits


def used_credits(since: str = "") -> int:
    """Summe der geloggten Credits ab Stichtag `since` (ISO, UTC; leer = ganzes Log)."""
    if not TAVILY_LOG_PATH.exists():
        return 0
    t0 = parse_since(since)
    total = 0
    for line in TAVILY_LOG_PATH.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if e["ts"] >= t0:
                total += int(e["credits"])
    return total


CALL_ATTEMPTS = 3
BACKOFF_S = 2.0
_sleep = time.sleep   # Tests biegen um


def _is_transient(exc: Exception) -> bool:
    import requests
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError))


class CountedClient:
    """Umhuellt TavilyClient: jeder search/extract/crawl-Aufruf schreibt genau einen Log-Eintrag."""

    def __init__(self, client) -> None:
        self._client = client
        self._budget = spend_guard.current_budget.get()

    def _call(self, endpoint: str, *args, **kwargs):
        # Verbindungsabbrueche (RemoteDisconnected, Timeout) werden bis zu CALL_ATTEMPTS Mal mit Backoff wiederholt; jeder Versuch zaehlt
        # als Aufruf (konservativ). Alle anderen Fehler (Key, Limit, 4xx) sofort weiter.
        for attempt in range(1, CALL_ATTEMPTS + 1):
            reservation = None
            if self._budget:
                self._budget.check()
            if self._budget or os.environ.get("LLM_PROVIDER") == "tokenfactory":
                if args:
                    raise BudgetExceeded("Counted live search calls require keyword arguments.")
                with spend_guard.ledger(TAVILY_LOG_PATH) as state:
                    if self._budget:
                        self._budget.check()
                    credits = estimate_credits(endpoint, kwargs)
                    remaining = spend_guard.number("TAVILY_CREDITS_TOTAL", 1000) - used_credits(os.environ.get("TAVILY_CREDITS_SINCE", ""))
                    if remaining - sum(r["amount"] for r in state.values()) - credits < spend_guard.number("LIVE_MIN_TAVILY_CREDITS", 300):
                        raise BudgetExceeded("Live search credit floor reached.")
                    reservation = uuid.uuid4().hex
                    state[reservation] = {"amount": credits, "ts": time.time()}
            try:
                result = getattr(self._client, endpoint)(*args, **kwargs)
            except Exception as exc:
                self._record(reservation, endpoint, kwargs, error=f"{type(exc).__name__}: {exc}")
                if attempt == CALL_ATTEMPTS or not _is_transient(exc):
                    raise
                _sleep(BACKOFF_S * 2 ** (attempt - 1))
                continue
            self._record(reservation, endpoint, kwargs)
            return result

    def _record(self, reservation, endpoint, kwargs, error=None):
        if reservation:
            with spend_guard.ledger(TAVILY_LOG_PATH) as state:
                log_call(endpoint, kwargs, error)
                state.pop(reservation)
        else:
            log_call(endpoint, kwargs, error)

    def search(self, *args, **kwargs):
        return self._call("search", *args, **kwargs)

    def extract(self, *args, **kwargs):
        return self._call("extract", *args, **kwargs)

    def crawl(self, *args, **kwargs):
        return self._call("crawl", *args, **kwargs)
