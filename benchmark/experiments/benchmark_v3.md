# Benchmark v3: Vorab-Festlegung (final, vor dem ersten Lauf)

Stand 2026-10-01. **Alles wie Benchmark v2** (`benchmark_v2.md`: 8 Einheiten, Referenzurteile, Toleranzen, Seeds, Placebos, Zählregel "Mehrheit über 3 Generierungen", Arme A ohne / B mit Tavily, Tavily-Modus fixture, LLM-Cache aus, Budgetstopp 6 $ brutto). Einziger Unterschied: der Code-Stand.

## Änderung gegenüber v2 (nur Produktpfad `graph_repair=True`)

Aus der Fehleranalyse der 23 Ausfälle in v2 (`results/benchmark_v2.md`): 7 × "Reparatur ohne Code-Block" (Abbruch), 4 × Timeout ohne Reparaturversuch, 7 × vier Versuche erschöpft, 5 × "Agenten-Fehler vermutet". Geändert (Commit `f6f4222`, Tests `tests/test_code_recovery.py`):
1. Code-Extraktion toleranter (andere Sprachmarke, offener Block, Code ohne Block, nur wenn `ast.parse` gelingt).
2. Fehlt nach der Reparatur weiterhin Code: Code neu aus dem Plan schreiben statt Abbruch.
3. Timeout: Reparatur mit festem Hinweis "schneller machen, Methode und Hyperparameter behalten" statt Abbruch.

Keine Änderung an Harness, Toleranzen, Referenzurteilen, Prompts für Plan und Code, Zielwerten.

## Erwartung (vorab, Hypothese)

Weniger Ausfälle "nicht reproduzierbar (Setup)" als in v2 (A 9, B 9 Urteile). Die Übereinstimmung (v2: 2 von 8 in beiden Armen) kann steigen oder gleich bleiben; v3 ersetzt v2 nicht. **Beide Ergebnisse werden im README berichtet**, v2 bleibt stehen.

## Durchführung

`LLM_PROVIDER=tokenfactory MAX_PARALLEL_SEEDS=3 uv run python scripts/run_benchmark.py --v3 <papers>` in drei Prozessen wie v2, Store `runs/benchmark_v3.jsonl`, Auswertung `uv run python scripts/eval_benchmark_v2.py runs/benchmark_v3.jsonl > results/benchmark_v3.md`. `src/` eingefroren, bis alle 48 Läufe fertig sind.
