# Benchmark v3: Vorab-Festlegung (final, vor dem ersten gezählten Lauf)

Stand 2026-10-01, 10:05. **Alles wie Benchmark v2** (`benchmark_v2.md`: 8 Einheiten, Referenzurteile, Toleranzen, Seeds, Placebos, Zählregel "Mehrheit über 3 Generierungen", Arme A ohne / B mit Tavily, Rollen 2/4 aus Fixtures, LLM-Cache aus, Budgetstopp 6 $ brutto). Unterschied: Code-Stand mit allen Fixes aus der Fehleranalyse und Tavily-Rolle 5 in Arm B.

**Vorgeschichte:** Ein erster v3-Start (nur Fixes 1-3, ohne Rolle 5, Commit `6acaa2d`) wurde nach 1 von 48 Läufen auf Entscheidung des Nutzers abgebrochen, um alle Fixes zusammen zu testen. Rohdaten aufbewahrt: `runs/benchmark_v3_aborted.jsonl`, `logs/bench_v3_aborted/`; sie zählen nicht. Die zwischenzeitlich geplante v4 (nur Rolle 5) ist in dieser v3 aufgegangen.

## Änderungen gegenüber v2 (nur Produktpfad `graph_repair=True`)

Fehleranalyse der 23 Ausfälle in v2 (`results/benchmark_v2.md`): 7 × "Reparatur ohne Code-Block" (Abbruch), 4 × Timeout ohne Reparaturversuch, 7 × vier Versuche erschöpft, 5 × "Agenten-Fehler vermutet".
1. Code-Extraktion toleranter (andere Sprachmarke, offener Block, Code ohne Block, nur wenn `ast.parse` gelingt). Commit `f6f4222`.
2. Fehlt nach der Reparatur weiterhin Code: Code neu aus dem Plan schreiben statt Abbruch (`f6f4222`).
3. Timeout: Reparatur mit festem Hinweis "schneller machen, Methode und Hyperparameter behalten" statt Abbruch (`f6f4222`).
4. **Tavily-Rolle 5 (nur Arm B, live):** Bei einem Absturz Websuche mit der letzten Fehlerzeile (Pfade und lange Zahlen entfernt), die 3 besten Treffer gehen als Hinweis an die Code-Reparatur. Höchstens 4 Suchen pro Lauf, jede Fehlerzeile einmal, kein LLM-Aufruf (`f53139c`). Grenze: höchstens 96 Tavily-Suchen.
5. Feste Zusatzhinweise bei Fehlern, an denen die Reparatur in v2 viermal scheiterte: verbotene Dataset-Bibliothek (`torch_geometric` u. a.: "nicht installiert, Schichten in reinem PyTorch") und Vorhersage nur einer Klasse (Checkliste Klassifikation aus `harness/repair_hints.py`).
6. Fehler-Check auch auf dem Mittelwert über alle Seeds: Lag Seed 0 über der trivialen Baseline, der Mittelwert aber darunter (instabiler Code, v2: NLinear), eine Reparatur und alle Seeds neu, innerhalb der Grenze `MAX_FAULT_REPAIRS`.

7. **Repo finden (Rolle 1, nur Arm B):** Ein Link im Paper ohne Hinweiswort zählt, wenn er der einzige GitHub-Link ist (v2: SGC hatte einen Fußnoten-Link, es wurde kein Repo gelesen). Hat das Paper keinen Link (Shchur), sucht Tavily auf github.com; ein Treffer zählt nur, wenn er arXiv-ID oder vollen Titel nennt. Ergebnis als Fixture (`benchmark/fixtures/repos/search_1811.05868.json`, ein Live-Aufruf am 2026-10-01: `shchur/gnn-benchmark`).
8. **Repo-Dateien:** Zusätzlich zu den vom LLM gewählten Dateien werden lokale Module gelesen, die diese importieren (eine Ebene, höchstens 6). v2: Bei GAT fehlte `utils/layers.py` (Attention-Kopf mit Dropout auf den Koeffizienten). Die Repo-Fixtures von GAT, DGI und Kipf wurden deshalb neu erzeugt (alte Fassungen in `benchmark/fixtures/repos/v2/`); neue Fixtures entstehen im ersten Lauf von Arm B und gelten dann für alle Generierungen.
9. **Setup-Extraktion (Rolle 4):** fragt ausdrücklich nach Umsetzungsdetails (wo Dropout sitzt, Normalisierung von Merkmalen und Adjazenz, Initialisierung, welche Parameter Weight Decay bekommen, Abbruchkriterium mit Geduld und Höchstzahl Epochen, welche Gewichte getestet werden) und verlangt, bei mehreren Protokollen im Text (Original vs. Neubewertung) die Zugehörigkeit im Topic zu nennen. v2: Bei Shchur mischten sich "200 Epochen" (Original) und das Protokoll der Neubewertung.

Begründung 7-9: Die häufigsten echten Abweichungen in v2 waren knapp daneben ("teilweise", Median-Abstand 0,011-0,017 bei Toleranz 0,01). Die Analyse der Codes zeigte fehlende Setup-Angaben (Shchur: Höchstzahl Epochen, Protokoll gemischt) und fehlende Umsetzungsdetails (GAT: richtige Hyperparameter, aber 0,79-0,82 statt Referenz 0,828). Diese Fixes betreffen nur Arm B (Tavily und Repo); Arm A bleibt ohne Setup-Angaben und dient als Vergleich. Kein Zielwert und keine Teststatistik gehen in Prompts.

Nicht behoben (bewusst): Logikfehler, die nach vier Versuchen bleiben, außer über Rolle 5; fehlende Stabilität zwischen Generierungen ("keine Mehrheit").

Keine Änderung an Harness, Toleranzen, Referenzurteilen, Zielwerten, Plan- und Code-Prompts.

## Kennzahlen und Erwartung (vorab)

- Primär wie v2: Übereinstimmung nach Mehrheitsregel je Arm, Placebos getrennt.
- Ausfälle: Erwartung weniger Ausfall-Urteile als in v2 (A 11, B 12 von je 48 Urteilen) in beiden Armen (Fixes 1-3); in Arm B zusätzlich Wirkung von Rolle 5.
- Rolle 5: Zahl der `error_search`-Ereignisse und Anteil, nach dem der nächste Versuch gültig lief (aus `events.jsonl`). Arm A hat keine Rolle 5; ein Unterschied der Ausfälle zwischen den Armen ist aber nicht allein Rolle 5 zuzuschreiben (Arm B hat auch Rolle 4).
- Keine Signifikanzaussagen. v3 ersetzt v2 nicht; **beide Ergebnisse stehen im README**, auch wenn v3 nicht besser ist.

## Durchführung

`LLM_PROVIDER=tokenfactory MAX_PARALLEL_SEEDS=3 uv run python scripts/run_benchmark.py --v3 <papers>` in drei Prozessen wie v2, Store `runs/benchmark_v3.jsonl`, Auswertung `uv run python scripts/eval_benchmark_v2.py runs/benchmark_v3.jsonl > results/benchmark_v3.md`. `src/` eingefroren, bis alle 48 Läufe fertig sind.
