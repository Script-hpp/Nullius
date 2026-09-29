# Nullius als Produkt: Plan und Entscheidungen

Stand 29.09.2026. Ziel: ein vollständiges Produkterlebnis (Bewertungskriterium „Design“) und ein erkennbarer Nutzen für eine echte Zielgruppe (Doktorandinnen, Reviewer, ML-Teams, die auf Papers aufbauen), nicht nur Kommandozeile und `papers.yaml`.

## Nutzerablauf
1. **Paper eingeben:** arXiv-Link in ein Eingabefeld.
2. **Claims auswählen:** Nullius schlägt die überprüfbaren Behauptungen vor (Zielwert mit wörtlichem Zitat und Spaltenprüfung, Rolle 2), die Nutzerin wählt aus und bestätigt den Task (Datensatz und Split).
3. **Zuschauen:** Der Lauf ist live sichtbar (Tavily-Funde, Branches, Messungen), Fortschritt aus `events.jsonl`.
4. **Zeugnis erhalten:** Urteil mit zwei Achsen, Belege, Einschränkungen, als teilbare Seite (die Evidence-Ansicht).
5. **Verlauf:** frühere Prüfungen und der Benchmark als Übersicht.

## Der ehrliche Rahmen (kommt sichtbar auf die Startseite)
**Nullius prüft Papers, die auf unterstützten Benchmarks evaluieren.** Jeder Datensatz braucht einen von Hand gebauten Task (Daten, Harness, Baselines, Referenzlösung). Unterstützt sind: Cora mit festem Split (Planetoid), Cora auf der größten Komponente mit zufälligen Splits (Shchur), ETTh1 (DLinear-Protokoll, Horizont 96). GAT und DGI haben gezeigt, dass ein neues Paper auf einem vorhandenen Datensatz ohne neuen Task funktioniert. Die Startseite zeigt die Liste und erklärt, wie man einen neuen Datensatz hinzufügt (`docs/adding_a_task.md`, noch zu schreiben).

## Entscheidungen (29.09.2026)
- **Demo offen für alle Besucher** (kein Zugangscode als Pflicht), mit Warteschlange und **hartem Budget**. Grund: die Jury soll ohne Zugangsdaten testen können; die Regeln verlangen eine kostenlos testbare Demo bis mindestens 15.12.2026.
  - Ein Lauf gleichzeitig (Warteschlange), Position sichtbar.
  - Budgetgrenzen aus der Umgebung (`DEMO_MAX_RUNS_PER_DAY`, `DEMO_MAX_USD_PER_DAY`, `DEMO_MAX_USD_TOTAL`, brutto), zusätzlich Grenze je IP und je Lauf (`MAX_COST_PER_RUN_USD`, Seeds und Zeit begrenzt).
  - Ist das Budget aufgebraucht: freundliche Meldung, alle vorberechneten Läufe bleiben ansehbar und abspielbar. Notausschalter (`DEMO_ENABLED=0`).
  - Optionaler Zugangscode (`DEMO_ACCESS_CODE`), der die Grenzen für Zugangsinhaber anhebt; er ist nicht nötig, um zu testen.
  - Kosten je Lauf laut Log grob 0,02 bis 0,10 $ netto plus 2 bis 3 Tavily-Credits (live). Tavily-Credits sind knapp: eigene Tageslimits.
- **Erst lokal bauen.** Hosting wird voraussichtlich über **Coolify** laufen (Docker-basiert, auf dem eigenen Server, HTTPS): Backend und Frontend als Container, Keys nur als Umgebungsvariablen des Servers, persistente Ablage für `runs/` und `logs/`. Deshalb: Konfiguration nur über Umgebungsvariablen, ein Health-Endpoint, keine Pfade mit Nutzernamen, ein Dockerfile je Dienst. Die Entscheidung wird vor dem Funktionsstopp (23.10.2026) festgezogen.
- **GitHub-Pages-Spiegel** der Wiedergabe (statisch) als Absicherung bis 15.12.2026.

## Architektur
```
web/ (React, statisch gebaut)  ──HTTP──▶  api/ (FastAPI)  ──▶  Orchestrator (vorhanden)  ──▶ Sandbox, LLM, Tavily
                                             │
                                             ├─ Warteschlange (1 Worker) + Budgetzähler (SQLite/JSON)
                                             └─ runs/<paper>/<ts>/ (dossier.json, events.jsonl, ...)
```
Endpunkte (Entwurf): `GET /api/health`, `GET /api/supported`, `POST /api/propose` (arXiv-Link → vorgeschlagene Claims), `POST /api/runs` (Claims → Lauf in die Warteschlange), `GET /api/runs` (Verlauf), `GET /api/runs/{id}` (Dossier), `GET /api/runs/{id}/events?after=N` (Ereignisse, Polling), `GET /api/budget` (verbleibendes Demo-Budget, ohne Keys).

## Bausteine und Reihenfolge (ein bis anderthalb Wochen)
1. **Registry der unterstützten Tasks** und ein `run_paper(paper_dict)` im Orchestrator (Läufe ohne Eintrag in `papers.yaml`).
2. **Backend:** FastAPI mit `health`, `supported`, `runs` (lesend), Warteschlange und Budgetzähler, Lauf starten.
3. **Claim-Vorschläge** aus dem Paper-Text (Rolle 2 erweitert: Modell liest, Programm prüft: Zitat, Zahl, Spalte), eingeschränkt auf unterstützte Tasks.
4. **Frontend:** Startseite mit Eingabefeld und Liste der Tasks, Claim-Auswahl, Laufansicht mit Live-Ereignissen, Übersichtsseite (Verlauf + Benchmark), Name und Logo einheitlich.
5. **Deployment:** Dockerfiles, Coolify-Anleitung, Budget- und Notausschalter getestet.
6. Danach README, Video, Abgabe.

## Risiken
- **Guthaben leer:** harte Grenzen (oben); der laufende Verbrauch wird im Kosten-Log und über `/api/budget` sichtbar.
- **Missbrauch** (beliebige Papers, große Modelle): nur unterstützte Tasks; Sandbox-Isolation und statische Codeprüfung gelten unverändert; Läufe ohne unterstützten Task werden abgelehnt.
- **Falsche Claim-Vorschläge:** Jeder Zielwert braucht ein wörtliches Zitat, das im Text steht; die Spaltenprüfung läuft; die Nutzerin bestätigt.
- **Sandbox-API instabil:** Infrastruktur-Wiederholung (Commit `8661d9f`); ein abgebrochener Lauf zeigt eine klare Meldung.
- **Alles, was nicht gebaut ist, steht im README** (Tavily v2, Achse 2, echter negativer Fall).
