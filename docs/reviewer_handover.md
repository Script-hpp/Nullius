# Übergabe an die nächste Reviewer-Session (Stand 2026-10-02 morgens)

Diese Datei ersetzt den Verlauf der vorherigen Reviewer-Session. Danach `CLAUDE.md`, `docs/worker_handover.md` und `docs/benchmark_plan.md` lesen.

## Rollen und Terminals (Orca)

- **Reviewer** (Opus): vergibt Aufträge, prüft Code/Suite/Screenshots, pusht, macht alle echten Läufe (Token Factory, Tavily, Nebius-Sandbox). Worker schreiben an das Reviewer-Terminal; nach einem Neustart hat der Reviewer eine **neue Terminal-ID** (`echo $ORCA_TERMINAL_HANDLE`). Dem Worker die neue ID mitteilen und `docs/worker_handover.md` anpassen (dort steht die alte `term_83e002c7-…`).
- **worker-2 (sonnet)**: Terminal `term_1f71e793-9652-4019-b162-945f40c1af17`, Checkout `<main checkout>`. Implementiert nur mit Fakes, committet, pusht nie.
- Alter Worker `term_7431f070-…` ist abgemeldet, kann geschlossen werden.
- Nachrichten: `orca terminal send --terminal <id> --enter --wait-submit 30 --text '<einzeilig>'`. Die Warnung "no turn start was observed" kommt oft, obwohl die Nachricht ankam: mit `orca terminal read --terminal <id> | grep -c '<Stichwort>'` prüfen, sonst mit `--retry-request <id>` erneut.
- Aufträge an Worker als Datei (Scratchpad oder `docs/`), im Terminal nur Verweis.

## Laufende Prozesse (lokal)

- Vite Dev-Server `:5174` (aus `web/`), API `uvicorn repro_agent.api:app --app-dir src --port 8000` mit `LLM_PROVIDER=tokenfactory` und einem `ACCESS_CODE` (nicht im Repo; bei Neustart neuen Code setzen und dem Nutzer nennen). Port 5173 gehört einem anderen Checkout.
- API neu starten nach `src/`-Änderungen: PID per `ps -eo pid,args | grep '[u]vicorn repro_agent.api'`, `kill <pid>` (nicht `pkill -f` mit dem Muster im selben Befehl, das beendet die eigene Shell).
- Hintergrund: Referenzläufe `scripts/validate_v2_references.py gat_lcc 30` und `nlinear 5` (nach Fix `3b94d36`). Ausgaben lagen im Scratchpad der alten Session; bei Bedarf neu starten und nach `benchmark/experiments/results/v2_reference_{gat_lcc,nlinear}.txt` schreiben.

## STAND JETZT (zuerst lesen, Stand 2026-10-02 morgens)

- **Benchmark v4 fertig** (48 Läufe, 7,78 $ brutto, Stopp 8 $ nicht erreicht). Ergebnis: mit Tavily 2 von 8 Einheiten (Kipf-GCN, Shchur-GCN), ohne 1 von 8 (DLinear); vorab erwartete Verbesserung gegenüber v3 **nicht** erreicht. Dafür: keine Einheit ohne Mehrheit (v3: 4), 1 Ausfall bei echten Claims (v3: 9), alle 16 Placebo-Mehrheiten "nicht reproduziert". Dateien: `results/benchmark_v4.md`, `failure_log_v4.md`, `benchmark_v4_incidents.md` (Prozess 2 starb 03:01 an `ContreeTransportError`, Rest mit unverändertem Code neu gestartet, zwei Prozesse nach Paper getrennt; die letzten 11 Zeilen tragen `code_commit` 2b13b31, `src/` identisch mit bf520c1). README hat v4-Absatz und Tabelle v2/v3/v4; `web/public/data/benchmark.json` aus `scripts/export_benchmark_json.py` (neueste Version zuerst, Landing zeigt v4).
- **Lücke im Benchmark-Skript (offen, nicht behoben):** `run_benchmark.py` liest den Store nur beim Start; parallele Prozesse sehen die Kosten der anderen nicht, der 8-$-Stopp wurde deshalb vom Reviewer-Watcher von außen durchgesetzt. Vor v5 im Skript beheben (Store vor jedem Lauf neu lesen).
- **`src/` ist wieder frei.** Gemergt nach v4: `repair-to-hash`, `plan-export-fix`, `wow-droplast`, `wow-badge` (Badge in `export_run` jetzt in try/except), `community-check` (Tavily live nur im Produktpfad, 7 Tage Cache, nur Bericht, Film-Szene "Others"). Suite 467 passed, npm test 62.
- **Web-Features heute auf main (alle nur web):** Logo-Leiste "Built with" (Nebius/Tavily aus Brand-Kits, NVIDIA als Text, Markenrecht), Logo-Zeichen `[•]` (Toleranzklammer, Header 20 px, Favicon, Apple-Icon), Reparatur-Erzählung, Paper-Kopf (Kurzbeschreibung, Autoren, Uni-Chips mit Monogramm, Zitat-geprüft, `summaries.json`), Film statt Zeitraffer (6 Szenen aus echten Lauf-Daten, KaTeX lokal, Plan-Karten mit Plan->Code-Zuordnung aus Agenten-Kommentaren).
- **API neu gestartet** (07:3x, neuer Code inkl. community-check), gleicher `ACCESS_CODE`, Log `logs/api.log`. Dev-Server `:5174` neu gestartet (nach KaTeX-Installation, `--force`).
- **Worker-2:** keine offenen Aufträge. Schickt Meldungen ans Reviewer-Terminal `term_83e002c7-…`.
- **Offen / nächste Schritte:** (1) echter Community-Lauf für die Demo (Kipf live, Tavily-Credits sparsam; bisher nur Testdaten in Screenshots); (2) B1-Erweiterung SparseTSF/RLinear (Referenzen liegen), RLinear als Vorzeige-Lauf für den drop_last-Hinweis; (3) Pipeline-Kurzbeschreibung für neue arXiv-Links (src, Schritt "What is this paper about"); (4) Deployment Coolify, Video, Devpost; (5) Skript-Lücke Budget (oben).
- **Budget:** Token Factory laut Log 18,39 $ brutto seit Stichtag 29.09. (59,50 $), Rest ca. 41 $, Konsole am 01.10. 19:55 UTC stimmte auf 0,4 % (netto). Tavily: v4 lief mit Fixtures, keine Credits.

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
