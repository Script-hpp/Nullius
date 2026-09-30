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

## STAND JETZT (zuerst lesen)

- **Benchmark v2 läuft** seit 2026-10-01 ca. 01:00 (Token Factory, Produktkonfiguration), gestartet mit Vorab-Festlegung `d71a8af` (final). **`src/` ist eingefroren, bis alle 48 Läufe fertig sind.** Drei Prozesse (`ps -eo args | grep '[p]ython scripts/run_benchmark'`), Logs `logs/bench_v2/p1.log` (Shchur-GCN, Kipf, Wu), `p2.log` (Shchur-GAT, GAT, DGI), `p3.log` (DLinear, NLinear). Store `runs/benchmark_v2.jsonl`, Budgetstopp 6 $ brutto. Nach Ende: `uv run python scripts/eval_benchmark_v2.py > benchmark/experiments/results/benchmark_v2.md`, prüfen, committen, Zahlen ins README.
- Referenzläufe erledigt: Shchur-GAT 0,8140 ± 0,0115, NLinear MSE 0,3739 ± 0,0013 (beide reproduziert, eingetragen).
- worker-2 schreibt den README-Entwurf (Auftrag `docs/worker_brief_10_readme.md`, uncommittet lassen): altes README wird nach `docs/research_log.md` verschoben, neues README produktzentriert, Benchmark-Zahlen `TBD`. Prüfen, dann pushen.
- Coolify-Weg dokumentiert (`docs/deploy.md`, `df24d77`); Deployment mit dem Nutzer steht aus (Subdomain vorhanden).

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
