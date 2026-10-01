# Übergabe an die nächste Reviewer-Session (Stand 2026-10-02)

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

## STAND JETZT (zuerst lesen, Stand 2026-10-02 nachts)

- **Benchmark v4 läuft** (Neustart 2026-10-01 ca. 20:15, Code `bf520c1`, Vorab-Festlegung `benchmark/experiments/benchmark_v4.md` final): drei Prozesse `ps -eo pid,args | grep '[r]un_benchmark.py --v4'`, Logs `logs/bench_v4/p{1,2,3}.log`, Store `runs/benchmark_v4.jsonl`, Budgetstopp 8 $ brutto (Stand 9/48 Läufe, 1,36 $; Hochrechnung ~7 $). **`src/` auf `main` eingefroren, bis v4 fertig.** Prüfen: `grep -cE "Traceback|BillingBlocked|Budget-Grenze" logs/bench_v4/p*.log` und `grep -c abgebrochen runs/benchmark_v4.jsonl`. Erster v4-Start abgebrochen (A2-Aufruf ohne Antwort beendete Arm-B-Läufe), Rohdaten `runs/benchmark_v4_aborted.jsonl`, Fixes `1ce7097`, `bf520c1`.
- **Nach v4:** `.venv/bin/python scripts/eval_benchmark_v2.py runs/benchmark_v4.jsonl > benchmark/experiments/results/benchmark_v4.md`, `scripts/failure_log.py runs/benchmark_v4.jsonl`, README (v2/v3/v4), `web/public/data/benchmark.json` schreiben (Format in `docs/worker_brief_17_ui_jury.md`-Meldung: versions[].units[] mit reference_verdict, arms A/B majority+verdicts, placebo, summary agree_A/agree_B/placebo_passed/n_units), dann Branches mergen (unten).
- **Block A gebaut und in v4:** A1 drei Code-Kandidaten, Auswahl nur über Validierungs-Metrik unseres Harness (`loop.py` `select_candidate`, `_run_candidates`); A3 M1/M3/M7; A2 Repo-Abgleich (`repo_diff.py`, 16k Token, Fehlschlag beendet nie den Lauf). Probeläufe `logs/v4_probe/`.
- **Fertige Branches, warten auf Merge nach v4** (alle auf origin): `community-check` (`901218a`, "What others found"/"Possible explanations", Tavily live, nur Bericht; drei echte Testaufrufe Kipf, letzter brauchbar; Konflikte mit UI in `App.tsx`/`phases.ts`, worker-2 löst sie nach Merge-Wunsch), `b-research` (`9ec3d24`, B1/B2-Recherche), `b-references` (`e2913f0`, Referenzen SparseTSF 0,3607, RLinear 0,3704, GCNII verworfen, ARMA verworfen; drop_last-Beleg mit Datei:Zeile).
- **Auf main gemergt heute:** UI-Umbau Auftrag 17 (`be9fc2b`, Landing-Ästhetik überall, PageIntro/NextStep, 9-Schritte-Streifen, Why you can trust, Benchmark-Seite liest `benchmark.json`), `f2c1c11` (Live-Kopf Seeds aus run_start), echte Nemotron-Summaries Discover (`2dd6224`, `scripts/make_summaries.py`), Nebius-Feedback ergänzt (`2f9297f`, `7376ecd`).
- **Stand worker-2 beim Reset:** F1 What-if-Regler fertig und auf main gemergt (`d21b95e`). F2 `wow-droplast` nur WIP-Commit `dfcb2ca`, ungetestet: fertig harness/droplast.py, etth1_eval-Zusatzfeld, Dossier-Felder, loop `_droplast_round`, droplast_note.py (Regel); offen: Texte a/b, report.py, Web-Karte, Tests, lokale Prüfung der vier Referenzen, Suite. Auftrag 19 nicht begonnen.
- **Auftrag 18/19 (Details):** Auftrag 18 (`docs/worker_brief_18_wow.md`): F1 What-if-Regler "Try to fool Nullius" (Branch `wow-slider`, nur web, darf nach Abnahme direkt auf main), F2 drop_last-Detektor ETTh1 (Branch `wow-droplast`, Harness, Merge erst nach v4). Freigegebene Regel F2: Hinweis, wenn |Behauptung − droplast(b)| ≤ Toleranz/4 und näher als am Vollwert; b ∈ {32, 64, 128, 256}; Urteil unberührt. Danach Auftrag 19 (`docs/worker_brief_19_wow2.md`): Badge, Reparatur-Erzählung, Zeitraffer, je Plan vorab. Worker meldete WIP-Stand per WIP-Commit auf den Branches.
- **Danach (Reviewer):** B1-Erweiterung um SparseTSF und RLinear (Rolle-2-Fixtures, `papers.yaml`, Referenzurteile, Benchmark v5), RLinear als Vorzeige-Lauf für den drop_last-Hinweis; Katalog-Paper mit fairem Ergebnis für die Demo.
- **Nutzer-Prioritäten:** Wow-Features für die Jury vor Deployment/Video/Devpost (die haben Zeit, Abgabe 30.10.). Erklärungen in der UI sichtbar, nicht klein/versteckt.
- **Budget:** Token Factory ca. 11 $ brutto seit Stichtag (59,50 $). Tavily: Community-Tests ca. 10-12 Credits heute; sparsam.
- **Lehren heute:** Probeläufe vor einem Benchmark auch mit großen Repo-Dateien (Arm B Shchur) fahren. Überwachung muss auch "abgebrochen"-Zeilen im Store melden. Hintergrund-Befehle enden nach 2 h; nicht `pkill -f` mit Muster im eigenen Befehl.

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
