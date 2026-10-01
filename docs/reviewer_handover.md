# Übergabe an die nächste Reviewer-Session (Stand 2026-10-01, `main` bei `d71a8af`)

Diese Datei ersetzt den Verlauf der vorherigen Reviewer-Session. Danach `CLAUDE.md`, `docs/worker_handover.md` und `docs/benchmark_plan.md` lesen.

## Rollen und Terminals (Orca)

- **Reviewer** (Opus): vergibt Aufträge, prüft Code/Suite/Screenshots, pusht, macht alle echten Läufe (Token Factory, Tavily, Nebius-Sandbox). Worker schreiben an das Reviewer-Terminal; nach einem Neustart hat der Reviewer eine **neue Terminal-ID** (`echo $ORCA_TERMINAL_HANDLE`). Dem Worker die neue ID mitteilen und `docs/worker_handover.md` anpassen (dort steht die alte `term_83e002c7-…`).
- **worker-2 (sonnet)**: Terminal `term_1f71e793-9652-4019-b162-945f40c1af17`, Checkout `/home/onuralp/Desktop/hackathon/research`. Implementiert nur mit Fakes, committet, pusht nie.
- Alter Worker `term_7431f070-…` ist abgemeldet, kann geschlossen werden.
- Nachrichten: `orca terminal send --terminal <id> --enter --wait-submit 30 --text '<einzeilig>'`. Die Warnung "no turn start was observed" kommt oft, obwohl die Nachricht ankam: mit `orca terminal read --terminal <id> | grep -c '<Stichwort>'` prüfen, sonst mit `--retry-request <id>` erneut.
- Aufträge an Worker als Datei (Scratchpad oder `docs/`), im Terminal nur Verweis.

## Laufende Prozesse (lokal)

- Vite Dev-Server `:5174` (aus `web/`), API `uvicorn repro_agent.api:app --app-dir src --port 8000` mit `LLM_PROVIDER=tokenfactory` und einem `ACCESS_CODE` (nicht im Repo; bei Neustart neuen Code setzen und dem Nutzer nennen). Port 5173 gehört einem anderen Checkout.
- API neu starten nach `src/`-Änderungen: PID per `ps -eo pid,args | grep '[u]vicorn repro_agent.api'`, `kill <pid>` (nicht `pkill -f` mit dem Muster im selben Befehl, das beendet die eigene Shell).
- Hintergrund: Referenzläufe `scripts/validate_v2_references.py gat_lcc 30` und `nlinear 5` (nach Fix `3b94d36`). Ausgaben lagen im Scratchpad der alten Session; bei Bedarf neu starten und nach `benchmark/experiments/results/v2_reference_{gat_lcc,nlinear}.txt` schreiben.

## STAND JETZT (zuerst lesen, Stand 2026-10-01 abends)

- **Benchmarks fertig:** v2 (`d71a8af`) und v3 (`4bab96e`, alle Fixes aus der Fehleranalyse + Tavily-Rollen 1 und 5) je 48 Läufe, je **2 von 8** Einheiten übereinstimmend in beiden Armen, **kein Placebo je "reproduziert"** (0 von 96). Ergebnisse `benchmark/experiments/results/benchmark_v2.md`, `benchmark_v3.md`, Fehlerprotokoll `results/failure_log_v3.md` (Skript `scripts/failure_log.py`), Muster mit Beleg `benchmark/experiments/failure_patterns_v3.md` (M1-M7). README berichtet v2 und v3. Abgebrochene v3-Starts: `runs/benchmark_v3_aborted*.jsonl` (zählen nicht).
- **Plan bis Abgabe:** `docs/roadmap_quality.md` (vom Nutzer bestätigt). Nutzer will ein sehr gutes Projekt, nicht nur ehrlich-mittelmäßig: erst **Block A** (Agent verbessern) und Benchmark v4, dann Benchmark erweitern (B) und Produkt (C).
- **Aktueller Auftrag an worker-2:** `docs/worker_brief_15_block_a.md` (A1 mehrere Code-Kandidaten mit Auswahl über **Validierungs**-Metrik unseres Harness, A3 Einzelfehler M1/M3/M7, danach A2 Abgleich mit Original-Code). Worker schickt zuerst einen Plan. Reviewer prüft, pusht, schreibt Vorab-Festlegung `benchmark/experiments/benchmark_v4.md` (wie v3, nur Code-Stand neu) und startet v4: `LLM_PROVIDER=tokenfactory MAX_PARALLEL_SEEDS=3 .venv/bin/python scripts/run_benchmark.py --v3 <papers>` braucht dafür ein neues Flag/Store (`--v4`, `runs/benchmark_v4.jsonl`) analog zu `--v3` in `scripts/run_benchmark.py`; drei Prozesse wie bisher (p1 Shchur-GCN/Kipf/SGC, p2 Shchur-GAT/GAT/DGI, p3 DLinear/NLinear), etwa 6-8 h, ca. 3-4 $ brutto (mit A1 mehr). Auswertung `scripts/eval_benchmark_v2.py runs/benchmark_v4.jsonl`, `scripts/failure_log.py`.
- **Web, heute gebaut und gepusht:** Hero-Landing (Schreibschrift "Nullius in verba" + zweifarbige Headline, Creme/Schwarz, `a8ba60a`, `ca7003b`), Discover mit "AI summary" (noch **Beispieltexte**, `example:true` in `web/public/data/discovery/summaries.json`; vor Deployment echte Nemotron-Erklärungen, Auftrag 13 Teil 2b), Ergebnis-Grafik, Schritt-Graph, Wortwahl ohne "verified". Vite läuft auf `:5174`.
- **Budget:** Token Factory bisher ca. 9 $ brutto von 59,50 $ (`BUDGET_TOTAL_USD`), Tavily sparsam (Rolle 5 im v3 29 Suchen).
- **Lehre heute:** "ändere noch nichts" vom Nutzer heißt "nicht weiterbauen", nicht "verwerfen". Nie Worker-Arbeit verwerfen lassen ohne ausdrückliche Anweisung; vorher committen lassen.

## Nächste Schritte (Reihenfolge, ursprüngliche Liste; 1-3 sind erledigt bzw. laufen)

1. **Referenzläufe prüfen** (Erwartung nach Fix, lokal gemessen: GAT-LCC ~0.81, NLinear MSE ~0.374). Werte als `reference_mean` in den YAML-Block `reference_verdicts` von `benchmark/experiments/benchmark_v2.md` eintragen, Ergebnisdateien committen. Nicht auf den Zielwert tunen; weichen sie ab, Ursache belegen lassen.
2. **Vorab-Festlegung final**: `benchmark_v2.md` Draft-Status entfernen, Commit "final", pushen. Ab dann `src/` eingefroren, bis Benchmark fertig.
3. `uv run python scripts/run_benchmark.py --v2 --dry-run` (muss "Fixtures vollständig" zeigen), dann Start: `LLM_PROVIDER=tokenfactory MAX_PARALLEL_SEEDS=3 uv run python scripts/run_benchmark.py --v2` (48 Läufe, geschätzt ~3 $ brutto, Budgetstopp 6 $, Stunden). Auswertung: `scripts/eval_benchmark_v2.py > benchmark/experiments/results/benchmark_v2.md`.
4. Parallel (ohne `src/`): worker-2 ergänzt `docs/deploy.md` um **Coolify** (Nutzer hat Coolify-Hosting und eine Subdomain; Nebius-VM nur falls Guthaben für Compute gilt). Dann Deployment mit dem Nutzer.
5. Katalog-Nachläufe mit aktuellem Code (Cache aus: `LLM_CACHE=off`): MoHETS (letzter Lauf lief vor Code-Retry-Fix `530f748` in `finish_reason length`), SL-GCN (letzter Lauf vor Fehler-Reparatur, Agenten-Fehler). Ziel: mindestens ein Katalog-Paper mit fairem Ergebnis für die Demo.
6. W4: README (englisch; Nemotron, Token Factory, Sandboxes, Tavily, Benchmark-Zahl mit Grenzen, PaperBench/CORE-Bench), Video < 3 Min., `docs/nebius_feedback.md`, Devpost. `CLAUDE.md` aktualisieren (Live-Modus ist gebaut; Zählregel v2).

## Entscheidungen des Nutzers (gültig)

- Benchmark v2 in **Produktkonfiguration** (`graph_repair=True`), v1 bleibt eingefroren.
- Kennzahl: **Zahl-Urteil (Achse 1), Mehrheit je Paper über 3 Generierungen**, README sagt "method fidelity not checked".
- **Dacrema nicht gebaut** (MovieLens-1M: NeuMF reproduziert, kein Negativfall). Negativkontrolle = Placebos.
- Live-Läufe aus der UI: LLM-Cache aus; API schreibt nie `catalog.yaml`.
- Autoren benachrichtigen: nur Idee (`docs/idea_author_notification.md`), nicht vor Abgabe.
- Offen: Jury-Filter (Empfehlung: einfachere Kandidaten, negatives "vorläufig" nur nach manueller Prüfung).

## Bewährte Prüfweise

- Jede Worker-Meldung gegen Datei:Zeile, `git show`, Suite (`--deselect tests/test_harness_baselines.py --deselect tests/test_nebius_sandbox.py`) und Screenshots prüfen, dann pushen.
- Leak-Schutz: nie Teststatistik (auch keine Baseline-Zahlen) in Prompts an den Agenten.
- Echte Läufe: Kosten aus `logs/costs.jsonl` (brutto = netto × 1.19) berichten; `LLM_CACHE=off`, wenn ein Lauf wirklich neu rechnen soll.

## Uncommittete Dateien im Hauptcheckout (vom Reviewer, bewusst)

`benchmark/catalog.yaml` (Nachläufe MoHETS/SL-GCN), geänderte Fixtures `2511.08094.json`, `2601.21866.json`, `repos/kevvzhang_StuartLandauGNN_readme.json`, `web/public/data/index.json`, `web/public/data/disc_2511_08094/20260930-221657/` (Live-Lauf, LLM aus Cache). Vor dem Committen prüfen; Katalog-Einträge der Nachläufe sind "nicht reproduzierbar (Setup)" bzw. "Agenten-Fehler vermutet".
