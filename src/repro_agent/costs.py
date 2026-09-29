"""Kosten-Log lesen, aggregieren, Budget pruefen. Das Log ist eine Schaetzung, massgeblich ist die Konsole."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

WARN_FRACTION = 0.80
ABORT_FRACTION = 0.95


class BudgetExceeded(RuntimeError):
    pass


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def entry_provider(entry: dict) -> str:
    """Altes Log (vor 2026-09-29) kennt kein Feld provider: ':free'-Modelle sind OpenRouter."""
    if entry.get("provider"):
        return entry["provider"]
    return "openrouter" if entry["model"].endswith(":free") else "tokenfactory"


def gross(net_usd: float, vat_rate: float) -> float:
    return net_usd * (1.0 + vat_rate)


def logged_gross_total(path: Path, vat_rate: float) -> float:
    """Bruttokosten aller bisher geloggten Aufrufe (Cache-Treffer kosten 0)."""
    return gross(sum(e["cost_usd"] for e in read_log(path)), vat_rate)


def budget_status(spent_gross: float, budget_total: float) -> str:
    """'off' (kein Budget gesetzt) | 'ok' | 'warn' (ab 80 %) | 'abort' (ab 95 %)."""
    if budget_total <= 0:
        return "off"
    if spent_gross >= ABORT_FRACTION * budget_total:
        return "abort"
    if spent_gross >= WARN_FRACTION * budget_total:
        return "warn"
    return "ok"


def aggregate(entries: list[dict], vat_rate: float) -> list[dict]:
    """Je (Anbieter, Modell, Tag UTC, cache_hit): Aufrufe, Tokens, Kosten netto/brutto."""
    import datetime as dt

    groups: dict[tuple, dict] = defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "net": 0.0}
    )
    for e in entries:
        day = dt.datetime.fromtimestamp(e["ts"], dt.timezone.utc).strftime("%Y-%m-%d")
        g = groups[(entry_provider(e), e["model"], day, bool(e.get("cache_hit")))]
        g["calls"] += 1
        g["prompt_tokens"] += e["prompt_tokens"]
        g["completion_tokens"] += e["completion_tokens"]
        g["net"] += e["cost_usd"]
    rows = []
    for (provider, model, day, cache), g in sorted(groups.items()):
        rows.append({
            "provider": provider, "model": model, "day": day, "cache_hit": cache, **g,
            "gross": gross(g["net"], vat_rate),
        })
    return rows
