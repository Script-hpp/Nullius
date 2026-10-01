# Benchmark v4: Vorab-Festlegung (Tavily-Rolle 5, Fehlersuche)

Stand 2026-10-01. Läuft **nach** Benchmark v3, auf dem Code-Stand von v3 plus Rolle 5 (Branch `role5-error-search`, nach v3 in `main`).

## Frage

Senkt die Tavily-Fehlersuche (Rolle 5: Websuche mit der letzten Fehlerzeile, Treffer als Hinweis an die Code-Reparatur) die Ausfälle?

## Aufbau

- **Nur Arm B** (mit Tavily), 8 Einheiten × 3 Generierungen = 24 Läufe, sonst alles wie v3/v2 (`benchmark_v3.md`, `benchmark_v2.md`): Referenzurteile, Toleranzen, Seeds, Placebos, Mehrheitsregel, Rollen 2/4 aus Fixtures, LLM-Cache aus, Budgetstopp 6 $ brutto.
- Einziger Unterschied: Rolle 5 ruft Tavily **live** auf (`error_search_live=True`), höchstens 4 Suchen pro Lauf, jede Fehlerzeile einmal. Grenze also höchstens 96 Tavily-Suchen.
- Vergleich: **Arm B aus v3** (gleicher Code ohne Rolle 5, gleiche Konfiguration).

## Kennzahlen (vorab)

1. Primär: Zahl der Ausfall-Urteile ("nicht reproduzierbar (Setup)", "Agenten-Fehler vermutet") in Arm B, echte Claims und Placebos, v4 gegen v3.
2. Sekundär: Übereinstimmung nach Mehrheitsregel (wie v2/v3), Zahl der Tavily-Suchen und Anteil der Reparaturen, die nach einer Suche lauffähigen Code lieferten (aus `events.jsonl`: `error_search` gefolgt von einem gültigen Versuch).

Mit 24 Läufen ist kein Effekt statistisch belegbar; berichtet werden Zählungen mit dieser Einschränkung. Das Ergebnis wird im README berichtet, auch wenn Rolle 5 nichts ändert.

## Durchführung

`LLM_PROVIDER=tokenfactory MAX_PARALLEL_SEEDS=3 uv run python scripts/run_benchmark.py --v4 <papers>` (drei Prozesse wie v2), Store `runs/benchmark_v4.jsonl`, Auswertung `scripts/eval_benchmark_v2.py runs/benchmark_v4.jsonl > results/benchmark_v4.md` plus Zählung der `error_search`-Ereignisse.
