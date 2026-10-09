# Übergabe an Worker-Sessions (Stand 2026-09-30, `main` bei `2a05a70`+)

Diese Datei ersetzt den Verlauf der vorherigen Worker-Session. Lies sie ganz, dann `CLAUDE.md`, dann den Auftrag.

## Rollen

- **Reviewer** (Claude Opus, Orca-Terminal `term_83e002c7-886d-4807-8921-ebbbbd6c29be`): vergibt Aufträge, prüft Code, Suite und Screenshots, pusht, macht alle echten Läufe (Token Factory, Tavily, Nebius-Sandbox).
- **Worker** (du, Sonnet): implementiert mit Tests über Fakes. Keine echten Aufrufe.

## Arbeitsregeln

- Keine echten Aufrufe: kein Token Factory, OpenRouter, Tavily, Nebius-Sandbox. Tests mit Fakes (Vorlagen: `tests/test_loop_fakes.py`, `tests/test_fault_repair.py`, `tests/test_api_repro.py`).
- Suite: `.venv/bin/python -m pytest -q --deselect tests/test_harness_baselines.py --deselect tests/test_nebius_sandbox.py` (die zwei Dateien brauchen echte Sandbox). Web: `cd web && npm run build && npm run lint`.
- Thematische Commits auf `main`, **nicht pushen**. Uncommittete Dateien, die du nicht angelegt hast, nicht anfassen (gehören dem Reviewer).
- Erst Plan und Fragen, dann Code. Unklares fragen statt raten. Jede "erledigt"-Meldung mit Datei:Zeile und Testendzeile belegen.
- Benchmark-Pfade (`benchmark/papers.yaml`, Läufe ohne Flag `graph_repair`) bleiben unverändert (Experiment-Disziplin: vorab festlegen, `src/` während eines Experiments einfrieren).
- UI-Texte englisch, Design nach CLAUDE.md "Design Philosophy", Dark und Light.
- Screenshots mit Playwright: `playwright-core` im eigenen Scratchpad, `executablePath: ~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome`, gegen eigene Fake-API + eigenen Vite (z. B. :8010/:5190), Dateien außerhalb des Repos.
- Laufende Prozesse des Reviewers nicht beenden: Vite `:5174`, API `:8000`, Hintergrund-Läufe von `scripts/curate_catalog.py`.

## Rückmeldung an den Reviewer (Orca)

Einzeilig, ohne Zeilenumbruch, Präfix `worker:`:

```
orca terminal send --terminal term_83e002c7-886d-4807-8921-ebbbbd6c29be --enter --text 'worker: <Nachricht>'
```

## Was es gibt (Kurzfassung)

- **Loop** `src/repro_agent/loop.py`: Rolle 2 (Zielwert per Tavily Extract), Rolle 4 (Setup-Details aus Paper/Repo), Plan (reasoning), Code (code), Sandbox, unabhängiger Harness, Urteil zweidimensional (Achse 2 bleibt "nicht geprüft"). Mit Flag `graph_repair` (nur Katalog/API): Plan-Validator, Fehler-Reparatur (`harness/repair_hints.py`, keine Teststatistik im Prompt), Graph-Ablation mit Reparatur nach Seed 0, Gesamtgrenze 3 Runden.
- **Tasks** `src/repro_agent/tasks.py`: Cora (Planetoid), Cora-LCC (Shchur), ETTh1 (Horizont 96).
- **Discovery** `src/repro_agent/discovery.py` + `#discover`: Tavily findet neue Papers auf unseren Benchmarks.
- **Katalog** `benchmark/catalog.yaml`, `src/repro_agent/catalog.py`, `scripts/curate_catalog.py`, UI `#catalog`, Jury-Filter `jury_ready`.
- **API** `src/repro_agent/api.py`: `/api/discover`, `/api/reproduce` (SSE, Cache aus, schreibt nie catalog.yaml), `/api/jobs/current`.
- **UI** `web/`: Overview, Discover, Catalog, Live-Lauf `#/live/<job>` (Phasen-Timeline `components/run/`), Evidence `#/run/<id>` mit Replay.
- **Pläne/Doku**: `docs/next_step_catalog_pipeline.md`, `docs/idea_author_notification.md`, Roadmap unten.

## Roadmap bis Abgabe 30.10.2026, 18:00 MEZ

1. W1: Qualität der Katalog-Läufe (erledigt: Auftrag 6; echte Nachläufe durch Reviewer).
2. W2: Benchmark 8-10 Papers mit/ohne Tavily (Kennzahl fürs README), vorab festgelegt.
3. W3: Hosting/Demo-URL (bis 15.12. erreichbar), Replay-Fallback, UI-Feinschliff.
4. W4: README, Video (< 3 Min.), Nebius-Feedback, Devpost.
