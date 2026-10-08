# Code-grounded Pilot (Vorab-Festlegung)

Stand 2026-10-08, Branch `Script-hpp/cto`. Diese Datei, `scripts/review/cg_pilot.py` (Zaehlregel `classify`, Entscheidungsregel `decide`, Aufrufzaehler) und `tests/test_cg_pilot.py` werden **vor dem ersten Modell-Aufruf** committet. Danach aendern sich Einheiten, Arme, Proben, Regeln und Grenzen nicht mehr. Bauplan: `docs/recall_analysis_3.md` Abschnitt 4 Hebel 3 und 4, Abschnitt 5 (Gate G0); gebaut in Commit `a93cdd0` (Flag `code_grounded`).

**Einordnung (wichtig):** Das ist ein **In-Sample-Pilot, keine Benchmark-Zahl.** Die drei Entwicklungseinheiten wurden vier Mal von Hand analysiert, `code_grounded` wurde mit Blick auf genau diese Fehlerbilder gebaut. 16 Proben, 6 Versuche je Arm auf den Entwicklungseinheiten: das reicht fuer ein Ja/Nein-Signal ueber den Aufwand eines bezahlten Laufs, nicht fuer eine Wirkungsschaetzung. Nichts hieraus kommt ins README oder ins Video. Modus: Nachbau aus Paper **und** Autorencode, unabhaengig gemessen (Entscheidung der Nutzerin/des Nutzers vom 2026-10-08).

## Frage

Treffen Generierungen das Referenzurteil deutlich oefter, wenn der Agent den entscheidenden Autorencode (Modelldatei, aufgerufene Vorverarbeitungs-/Schedule-Funktionen, Startskript, Trainingsskript samt Early Stopping) **roh** sieht statt destillierter Zitate, und die Zahlenwerte per `CONFIG` deterministisch erzwungen werden? Gemessen wird die **Accuracy** (Klasse des Zahl-Urteils gegen Referenzurteil), nicht die Umsetzung einzelner Fakten (das war V1).

## Aufbau

- **Einheiten (4):** Entwicklungseinheiten, die in Benchmark v4 Arm B in 0 von 3 Generierungen das Referenzurteil trafen und deren entscheidende Angabe nur im Repo steht: Zeng NLinear (ETTh1, 5 Seeds), Wu SGC (Cora, 10 Seeds), Veličković GAT (Cora, 10 Seeds); dazu Kipf GCN (Cora, 10 Seeds) als Nicht-Verschlechterung (v4 Arm B: 3/3). Echter Claim je Einheit (`claims[0]`), kein Placebo. **Referenzurteil jeweils „reproduziert“** (`benchmark/experiments/benchmark_v2.md`: Referenzmittel NLinear 0,3739, SGC 0,8081, GAT 0,8280, GCN 0,8136; Toleranz 0,01).
- **Arme (je Einheit):**
  1. `heute`: Produktpfad wie Benchmark v4 (`graph_repair=True`, `setup_v2`): `format_for_prompt_v2` mit den gespeicherten belegten Setup-Details des v4-Arm-B-Laufs, Konflikthinweis, Plan und Code. Der Repo-Ausschnitt ist in diesem Arm nur ueber Rolle 4 (Zitate) im Prompt.
  2. `cg`: `code_grounded=True`: numerischer Steckbrief (S1 nur Zahlenschluessel, ein Extraktionsaufruf je Einheit), CONFIG-Pflicht mit S2-Override, belegte Paper-Angaben, dazu der **rohe** fokussierte Repo-Ausschnitt (S4, höchstens 15 000 Zeichen) im Plan-, Code- und Reparatur-Prompt; argparse-Defaults sind entfernt, wo ein Startskript das Flag setzt; Modellnamen-Regel im Prompt.
- **Eingabe, kein Tavily:** Paper-Text und belegte Setup-Details aus dem gespeicherten v4-Arm-B-Dossier, Repo-Dateien aus lokalen Klonen am Commit des Laufs (NLinear `0c11366`, SGC `2c7a272`, GAT `5af87e7`, GCN `39a4089`) ueber den echten `read_repo` (Dateiwahl des gespeicherten Laufs als Proxy der LLM-Dateiwahl, wie in V1), README aus dem Klon. Kein Tavily-, kein Nebius-, kein Token-Factory-Aufruf.
- **Proben je Einheit und Arm:** 1 Plan (Rolle `reasoning`), 2 Code-Proben bei Temperatur **0,2 und 0,8** (gleicher Plan), höchstens **1 gemeinsame Reparatur** je Probe: Anlass ist ein Fehler im Seed-0-Lauf (inkl. Formpruefung `SMOKE=1` und statischer Codepruefung) und im Arm `cg` zusaetzlich ein CONFIG-Befund (CONFIG fehlt, Schluessel ungenutzt, konkurrierendes Literal). Danach S2-Override (deterministisch). Ist der urspruengliche Code lauffaehig und der reparierte nicht, bleibt der urspruengliche. Insgesamt 8 Plaene, 16 Code-Proben, 4 S1-Aufrufe.
- **Keine** Kandidatenauswahl, **keine** Graph-Ablation oder Graph-Reparatur, keine Kanten-/Validierungslabel-Pruefung, keine Fehler-Reparatur nach allen Seeds, kein S3. Der Pilot ist damit konservativer als das Produkt.

## Modell, Kosten, Grenzen

- **Anbieter:** OpenRouter, `nvidia/nemotron-3-super-120b-a12b:free` fuer **alle** Rollen (`reasoning`, `code`, `extract`). 0 $. Kein Ultra. `LLM_PROVIDER=openrouter`, `LLM_CACHE=off`. Prototyp-Regeln aus `CLAUDE.md`: nur oeffentliche Papers und Repos, keine Schluessel oder Nutzerpfade im Prompt.
- **Aufrufgrenze:** **45 HTTP-Anfragen insgesamt** ueber alle Tage, gezaehlt auf HTTP-Ebene (auch 429 und Wiederholungen des Clients), Datei `calls.jsonl`, Drossel auf hoechstens eine Anfrage je 3,3 s (20 pro Minute). Vorhersage: 4 + 8 + 16 + bis 16 = 44 plus Reserve 1; Reparaturen laufen abwechselnd `heute`/`cg`, damit eine erreichte Grenze beide Arme gleich trifft. Meldet OpenRouter das Tageslimit (429, Text „per-day“) oder ist die Grenze erreicht, **stoppt das Skript und berichtet**; Wiederaufnahme am naechsten Tag mit demselben Befehl (fertige Teile in `out/` bleiben, keine Anfrage wird doppelt gestellt). Eine Probe, die wegen der Grenze nicht gelaufen ist, zaehlt nicht als Fehlschlag: der Pilot ist dann **unvollstaendig** (siehe Regel).
- **Messung:** jede Probe laeuft in `LocalSandbox` (Docker, `python:3.11-slim`, CPU) mit unserem Harness (Testlabels frisch aus dem Original-Datensatz) und unserer Metrik, mit der vollen Seed-Zahl der Einheit (5 oder 10). Waehrend des Agenten-Codes ist der Container vom Netz getrennt (`docker network disconnect`), `unshare -n` entfaellt dort (braucht Rechte im Container); fuer den Harness ist das Netz wieder verbunden. Agenten-Code laeuft nie auf dem Host.

## Zaehlregel

Klasse je Probe = Urteil des Orchestrators (`Orchestrator._verdict`: `judge(Mittel, Ziel, Toleranz)`, dazu Agenten-Fehler-Pruefung gegen die Baselines) ueber den Mittelwert der gueltigen Seeds. Mindestens die Haelfte der Seeds muss einen gueltigen Wert liefern, sonst gilt „nicht reproduzierbar (Setup)“. **Treffer** = Klasse gleich Referenzurteil („reproduziert“). „Agenten-Fehler vermutet“, „teilweise“, „nicht reproduziert“, Setup-Ausfall und fehlender Code sind keine Treffer. Ein Ausfall zaehlt als Fehlschlag, wenn der Pilot vollstaendig gelaufen ist.

## Entscheidungsregel (vorab)

Der Pilot **besteht**, wenn alle drei gelten:

1. `cg` trifft auf den drei Entwicklungseinheiten **mindestens 3 von 6**,
2. `cg` trifft dort **mindestens `heute` + 2**,
3. Kipf: `cg`-Treffer **≥ `heute`-Treffer − 1** (von 2).

Sonst: nicht bestanden, Stopp, Bericht „Code-Grounding hat die Accuracy im kostenlosen Pilot nicht angehoben“; v4 bleibt. Ein unvollstaendiger Pilot (nicht alle 16 Proben gemessen) ist weder bestanden noch nicht bestanden; er wird am naechsten Tag fortgesetzt. Besteht der Pilot, gibt es **keine** Aussage ueber Wirksamkeit, sondern die Empfehlung an die Nutzerin/den Nutzer, 10 $ OpenRouter-Guthaben fuer das volle Gate G0 (`docs/recall_analysis_3.md` Abschnitt 5, neun Einheiten, drei Proben) freizugeben.

**Grenzen der Regel:** 6 Versuche je Arm; „3 von 6“ gegen „1 von 6“ ist ein Signal, kein Test. Zwei Proben je Einheit und Arm teilen einen Plan, sind also nicht unabhaengig. Seeds und Modellantworten schwanken; ein Neulauf kann anders ausgehen. Der Pilot misst nichts ueber neue Papers (kein Held-out).

## Zusaetzlich berichtet (beschreibend, nicht Teil der Regel)

- Je Probe: Klasse, Mittel, Std, gueltige Seeds, Reparatur ja/nein, Anzahl S2-Overrides, kurze Fehlerbeschreibung.
- **Welche Details ankamen:** die Fakt-Checks aus `scripts/review/uptake_probe.py` (`FACTS`: NLinear Fenster 336, lr 0,005, Halbierung je Epoche, letzten Wert abziehen, keine Zerlegung; SGC und GAT Zeilen-Normalisierung; SGC ohne Spalten-Standardisierung), je Probe und Arm. Nur Heuristik (Regex/AST), ein Treffer belegt Anwesenheit, nicht richtige Verdrahtung.
- Gespeichert: Plaene, Code der Proben und Reparaturen, Steckbrief samt Ausschnitt (`out/<Einheit>/`), Zaehler `calls.jsonl`.

## Bekannte Abweichungen vom Produkt (offen genannt)

- Keine Kandidatenauswahl (Produkt: 3 Kandidaten), keine Graph-Reparatur, keine Fehler-Reparatur nach allen Seeds: Der Pilot unterschaetzt das Produkt in beiden Armen.
- Eine Reparatur statt bis zu vier Versuchen je Seed; spaetere Seeds werden nicht repariert (ein Seed ohne gueltigen Wert zaehlt als fehlend).
- Der Arm `heute` nutzt die gespeicherten Rolle-4-Details des v4-Laufs (Super bei Ultra-Plan in v4, hier alles Super), nicht eine neue Extraktion.
- Die Dateiwahl des Repos ist die des gespeicherten Laufs (Proxy). Entscheidende Angaben, die in keiner gelesenen Datei stehen (z. B. das getunte SGC-Weight-Decay in einer Pickle-Datei, `lradj`-Default in `run_longExp.py` fuer NLinear), kann auch `cg` nicht liefern; das ist Teil der Frage, nicht ein Fehler des Pilots.
- Minimaler I/O-Helfer (Hebel 4) **nicht gebaut**: Er beruehrt Task-Beschreibung und Snapshot aller Einheiten und wuerde beide Arme ungleich treffen; Hebel 4 bleibt offen.

## Reihenfolge

1. Diese Datei, `cg_pilot.py`, `test_cg_pilot.py` committen (**vor dem ersten Aufruf**).
2. Pilot laufen lassen (`scripts/review/cg_pilot.py`), Zaehler pruefen.
3. Ergebnisse in `benchmark/experiments/results/recall_cg_pilot.md`, committen, nicht pushen, nicht mergen.
