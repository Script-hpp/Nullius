"""Sperre fuer Live-Laeufe der offenen Demo: Budget, Tavily-Credits, Tageslimits, Limit je IP. Kein Zugangscode.

Alle Werte kommen aus Umgebungsvariablen und werden bei jedem Aufruf gelesen (Tests setzen sie per monkeypatch).
Die Sperre gibt nur Gruende zurueck, nie Kontostaende.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import json
import os
import time
from pathlib import Path

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


def daily_cost_usd() -> float:
    """Bruttokosten seit lokaler Mitternacht (Serverdatum wie die Tageslimits), aus dem Kosten-Log, ueberlebt Neustarts."""
    midnight = dt.datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return costs.gross(sum(e["cost_usd"] for e in costs.read_log(llm.COST_LOG_PATH) if e["ts"] >= midnight), _f("VAT_RATE", 0.19))


def misconfigured() -> bool:
    """Token Factory ohne gesetztes Budget: die Code-Standards (29,75 $) gelten nicht fuer dieses Konto, also zu, nicht offen."""
    return os.environ.get("LLM_PROVIDER") == "tokenfactory" and not (os.environ.get("BUDGET_TOTAL_USD") and os.environ.get("BUDGET_SINCE"))


def floor_reason() -> str | None:
    if misconfigured() or budget_remaining_usd() < _f("LIVE_MIN_BUDGET_USD", 3.0):
        return "budget_floor"
    cap = _f("MAX_DAILY_COST_USD", 0.0)   # 0 = aus; Grund bleibt "daily_limit": das Frontend zeigt Tageslimit mit Reset
    if cap > 0 and daily_cost_usd() >= cap:
        return "daily_limit"
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

    @staticmethod
    def key(ip: str) -> str:
        """Nur ein Hash der Adresse im Speicher und im Log (kein Klartext-IP-Log)."""
        return hashlib.sha256(ip.encode()).hexdigest()[:16]

    def _roll(self) -> None:
        today = dt.date.today()
        if self.day != today:
            self.day, self.counts = today, {}

    def used_up(self, kind: str, ip: str) -> bool:
        self._roll()
        limit = int(_f("MAX_DAILY_REPRO_PER_IP" if kind == "reproduction" else "MAX_DAILY_DISCOVER_PER_IP", 1))
        return self.counts.get((kind, self.key(ip)), 0) >= limit

    def add(self, kind: str, ip: str) -> None:
        self._roll()
        k = (kind, self.key(ip))
        self.counts[k] = self.counts.get(k, 0) + 1

    def restore(self, kind: str, ip_key: str) -> None:
        self._roll()
        k = (kind, ip_key)
        self.counts[k] = self.counts.get(k, 0) + 1


def _normalize_ip(raw: str) -> str:
    """IPv6 zaehlt je /64 (ein Anschluss hat Milliarden Adressen, sonst waere das Limit je IP wertlos)."""
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return raw[:64]
    if ip.version == 6:
        return str(ipaddress.ip_network(f"{ip}/64", strict=False).network_address) + "/64"
    return str(ip)


def client_ip(request) -> str:
    """Hinter einem Reverse-Proxy (TRUST_PROXY=1) der letzte X-Forwarded-For-Eintrag (vom Proxy angehaengt, nicht faelschbar),
    sonst die Peer-Adresse."""
    if os.environ.get("TRUST_PROXY") == "1":
        xff = request.headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return _normalize_ip(parts[-1])
    return _normalize_ip(request.client.host) if request.client else "unknown"


# ---- Persistente Zaehler: ein Neustart/Redeploy setzt die Tageslimits nicht zurueck ----
def jobs_log_path() -> Path:
    return Path(llm.COST_LOG_PATH).parent / "live_jobs.jsonl"


def record_start(kind: str, ip: str) -> None:
    """Start eines Live-Jobs anhaengen (Zeit, Art, IP-Hash). Fehler beim Schreiben duerfen den Start nie verhindern."""
    try:
        p = jobs_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind, "ip": IpLimits.key(ip)}) + "\n")
    except OSError:
        pass


def todays_starts() -> list[dict]:
    p = jobs_log_path()
    if not p.exists():
        return []
    midnight = dt.datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    out = []
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("ts", 0) >= midnight and e.get("kind") in ("reproduction", "discovery"):
            out.append(e)
    return out
