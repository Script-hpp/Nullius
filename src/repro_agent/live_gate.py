"""Sperre fuer Live-Laeufe der offenen Demo: Budget, Tavily-Credits, Tageslimits, Limit je IP. Kein Zugangscode.

Alle Werte kommen aus Umgebungsvariablen und werden bei jedem Aufruf gelesen (Tests setzen sie per monkeypatch).
Die Sperre gibt nur Gruende zurueck, nie Kontostaende.
"""

from __future__ import annotations

import datetime as dt
import os

from . import costs, llm, tavily_usage

REASONS = ("budget_floor", "tavily_floor", "daily_limit", "busy")
MESSAGES = {
    "budget_floor": "Live checks are paused: the model budget for this demo is nearly used up.",
    "tavily_floor": "Live checks are paused: the search budget for this demo is nearly used up.",
    "daily_limit": "Live checks are paused: the daily limit is reached.",
    "busy": "Another check is running. Try again in a few minutes.",
}


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def budget_remaining_usd() -> float:
    """Token-Factory-Budget brutto: BUDGET_TOTAL_USD minus Bruttokosten seit BUDGET_SINCE (costs.py, nicht neu gerechnet)."""
    total = _f("BUDGET_TOTAL_USD", 29.75)
    spent = costs.logged_gross_total(llm.COST_LOG_PATH, _f("VAT_RATE", 0.19), os.environ.get("BUDGET_SINCE", ""))
    return total - spent


def tavily_remaining() -> int:
    return int(_f("TAVILY_CREDITS_TOTAL", 1000)) - tavily_usage.used_credits(os.environ.get("TAVILY_CREDITS_SINCE", ""))


def floor_reason() -> str | None:
    if budget_remaining_usd() < _f("LIVE_MIN_BUDGET_USD", 3.0):
        return "budget_floor"
    if tavily_remaining() < _f("LIVE_MIN_TAVILY_CREDITS", 300):
        return "tavily_floor"
    return None


def next_reset_iso(now: dt.datetime | None = None) -> str:
    """Tageslimits gelten je lokalem Serverdatum (datetime.date.today); Reset = naechste lokale Mitternacht, als UTC."""
    now = now or dt.datetime.now().astimezone()
    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class IpLimits:
    """Je IP und Tag ein Lauf je Art (im Speicher, geht beim Neustart verloren; das globale Tageslimit bleibt davon unberuehrt)."""

    def __init__(self) -> None:
        self.day: dt.date | None = None
        self.counts: dict[tuple[str, str], int] = {}

    def _roll(self) -> None:
        today = dt.date.today()
        if self.day != today:
            self.day, self.counts = today, {}

    def used_up(self, kind: str, ip: str) -> bool:
        self._roll()
        limit = int(_f("MAX_DAILY_REPRO_PER_IP" if kind == "reproduction" else "MAX_DAILY_DISCOVER_PER_IP", 1))
        return self.counts.get((kind, ip), 0) >= limit

    def add(self, kind: str, ip: str) -> None:
        self._roll()
        self.counts[(kind, ip)] = self.counts.get((kind, ip), 0) + 1


def client_ip(request) -> str:
    """Hinter einem Reverse-Proxy (TRUST_PROXY=1) der letzte X-Forwarded-For-Eintrag (vom Proxy angehaengt, nicht faelschbar),
    sonst die Peer-Adresse."""
    if os.environ.get("TRUST_PROXY") == "1":
        xff = request.headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"
