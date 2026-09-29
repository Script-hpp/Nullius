"""Kosten-Log aggregieren und gegen die Nebius-Konsole abgleichen. Keine LLM-Aufrufe.

Nutzung: uv run python scripts/cost_report.py
Konsolenwerte unten aktualisieren (Konsole > Usage), Log ist nur eine Schaetzung.
"""

from __future__ import annotations

import os

from repro_agent import config  # noqa: F401  laedt .env
from repro_agent.costs import aggregate, gross, read_log, entry_provider
from repro_agent.llm import COST_LOG_PATH, TF_PRICES

VAT = float(os.getenv("VAT_RATE", "0.19"))
BUDGET = float(os.getenv("BUDGET_TOTAL_USD", "29.75"))

# Nebius-Konsole, Stand 2026-09-29 05:35 UTC (Mio. Tokens, netto USD; Tokens auf 0,01 Mio. gerundet).
# Lightning-Kosten stehen dort nur als "<0,01 $".
CONSOLE = {
    "nvidia/Nemotron-3-Ultra-550b-a55b": (0.12, 0.09, 0.12 + 0.27),
    "nvidia/nemotron-3-super-120b-a12b": (1.09, 0.59, 0.33 + 0.53),
    "nvidia/Nemotron-3_5-Lightning": (0.03, 0.03, None),
}
OFFICIAL = {  # USD pro Mio. Tokens netto (Nebius-Preisliste)
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
}


def pct(a: float, b: float) -> str:
    return f"{(a - b) / b:+.1%}" if b else "n/a"


def main() -> None:
    entries = read_log(COST_LOG_PATH)
    print(f"Log: {len(entries)} Eintraege, MwSt. {VAT:.0%}\n")
    print("== Aggregat (Anbieter, Modell, Tag UTC) ==")
    print(f"{'Anbieter':13} {'Modell':38} {'Tag':10} {'Cache':5} {'Aufr':>5} {'In':>10} {'Out':>9} {'netto $':>9} {'brutto $':>9}")
    for r in aggregate(entries, VAT):
        print(f"{r['provider']:13} {r['model']:38} {r['day']} {'ja' if r['cache_hit'] else 'nein':5} {r['calls']:5} "
              f"{r['prompt_tokens']:10} {r['completion_tokens']:9} {r['net']:9.4f} {r['gross']:9.4f}")

    tf = [e for e in entries if entry_provider(e) == "tokenfactory" and not e.get("cache_hit")]
    print("\n== Token Factory: Log vs. Konsole (ohne Cache-Treffer) ==")
    for model, (c_in, c_out, c_cost) in CONSOLE.items():
        rows = [e for e in tf if e["model"] == model]
        l_in = sum(e["prompt_tokens"] for e in rows) / 1e6
        l_out = sum(e["completion_tokens"] for e in rows) / 1e6
        l_cost = sum(e["cost_usd"] for e in rows)
        cost_c = f"{c_cost:.2f}" if c_cost is not None else "<0.01"
        print(f"{model}\n  Input  Mio.: log {l_in:.3f} | konsole {c_in:.2f} | {pct(l_in, c_in)}"
              f"\n  Output Mio.: log {l_out:.3f} | konsole {c_out:.2f} | {pct(l_out, c_out)}"
              f"\n  Kosten netto: log {l_cost:.4f} | konsole {cost_c}"
              + (f" | {pct(l_cost, c_cost)}" if c_cost is not None else ""))
    l_total = sum(e["cost_usd"] for e in tf)
    print(f"  Summe netto: log {l_total:.4f} | konsole 1.26 | {pct(l_total, 1.26)}; brutto log {gross(l_total, VAT):.4f} | konsole 1.49")
    print("  Hinweis: Konsolentokens auf 0,01 Mio. gerundet, Rundung ±0,005 Mio. Tokens (bei kleinen Mengen bis ±15 %).")

    print("\n== Preistabelle im Code vs. offiziell ==")
    for model, (p_in, p_out) in OFFICIAL.items():
        c_in, c_out = (x * 1e6 for x in TF_PRICES[model])
        ok = abs(c_in - p_in) < 1e-9 and abs(c_out - p_out) < 1e-9
        print(f"  {model}: code {c_in:.2f}/{c_out:.2f} | offiziell {p_in:.2f}/{p_out:.2f} | {'OK' if ok else 'ABWEICHUNG'}")

    spent = gross(sum(e["cost_usd"] for e in entries), VAT)
    print(f"\n== Budget ==\n  Brutto laut Log {spent:.2f} $ von {BUDGET:.2f} $ ({spent / BUDGET:.1%}), Rest {BUDGET - spent:.2f} $")
    old = sum(1 for e in entries if "provider" not in e)
    print(f"  {old} Alteintraege ohne Feld provider (Anbieter aus Modell-ID abgeleitet)")


if __name__ == "__main__":
    main()
