# Auftrag 24 (worker-3): Ursachenanalyse Benchmark v4

Rolle: Analyst. Du liest Daten und Code und schreibst einen Bericht. Du änderst nichts unter `src/`, `scripts/`, `web/`, `benchmark/papers.yaml`, und du committest und pushst nicht. Der Reviewer committet.

Arbeitsverzeichnis: `/home/onuralp/Desktop/hackathon/research` (Hauptcheckout, Branch `main`). Kein `git checkout`, kein `git stash`, kein Branchwechsel: dort laufen gerade Demo-Läufe des Reviewers.

Erlaubt: Dateien lesen, Python lokal zum Auswerten (`.venv/bin/python`), lokale CPU-Läufe der menschlichen Referenzlösungen (`tests/reference/*.py`, `scripts/run_reference_local.py`) mit geänderten Hyperparametern. **Nicht erlaubt:** LLM-Aufrufe (Token Factory, OpenRouter), Tavily-Aufrufe, Nebius-Sandbox. Keine Teststatistik in irgendeinen Prompt (es gibt keine Prompts in diesem Auftrag).

## Frage

Benchmark v4 (`benchmark/experiments/results/benchmark_v4.md`, Rohdaten `runs/benchmark_v4.jsonl`, Fehlerprotokoll `results/failure_log_v4.md`): Nur 2 von 8 Einheiten (mit Tavily) bzw. 1 von 8 (ohne) treffen das Referenzurteil. Fast alle Abweichungen gehen in dieselbe Richtung: der Agenten-Code ist schlechter als das Paper (Median-Abstand: Kipf −0,7 Pp, SGC −1,9, GAT −2,3, Shchur-GCN −1,8, Shchur-GAT −2,7, DGI −10 Pp, DLinear MSE +0,011, NLinear MSE +0,24). Toleranz überall 0,01. **Warum?** Belegt, nicht vermutet.

## Hypothesen (prüfen, nicht bestätigen)

- **H1 Setup-Details kommen nicht im Code an.** Arm B hat belegte Angaben aus Paper/Repo (`dossier.json`, Feld der Setup-Details aus Rolle 4, mit Zitat). Prüfe je Lauf: steht der Wert (lr, weight decay, Epochen, Dropout, Hidden, Early Stopping/Patience, Normalisierung) im gewählten Code (`runs/<paper>/<ts>/<claim>_<hash>.py`, gewählter Kandidat aus Ereignis `candidates` in `events.jsonl`)? Zähle: übernommen / abweichend / fehlt.
- **H2 Trainingsprotokoll.** Epochen zu wenig, kein Early Stopping auf Validierung, falsche Auswahl des Checkpoints, falsche Feature-Normalisierung, falscher Split. Vergleich mit der Referenzlösung `tests/reference/<paper>_reference.py`, die das Ziel trifft.
- **H3 Toleranz gegen Seed-Streuung.** Wie groß ist die Streuung der Referenzlösung und der Agenten-Läufe über Seeds und über Generierungen? Wie oft wäre "reproduziert" bei Toleranz = 2 × Seed-Std des Papers? (Nur post hoc, klar kennzeichnen; die Regel wird nicht geändert.)
- **H4 Auswahl über Validierung (A1).** Wählt die Validierungs-Metrik systematisch Kandidaten, die auf Test schlechter sind? Vergleich der Kandidaten innerhalb eines Laufs (Seed 0: Val- und Testwerte im Dossier/Events, Testwerte nur hier im Bericht verwenden).
- **H5 Einzelfälle.** DGI (−10 Pp: Pretraining, Readout, logistische Regression?), NLinear (MSE 0,8: Normalisierungsfehler?), Shchur-GAT (ein Lauf 0,706).

## Gegenprobe (stärkster Beleg)

Für die zwei, drei häufigsten Abweichungen: Referenzlösung lokal mit dem Hyperparameter des Agenten laufen lassen (z. B. Epochen 200 statt Early Stopping, Weight Decay 0). Fällt die Referenz dadurch auf den Agentenwert, ist die Ursache belegt. Jeder Lauf mit Befehl, Seeds, Ergebnis im Bericht.

## Ausgabe

`benchmark/experiments/failure_analysis_v4.md` (Deutsch, Stil wie die anderen Dateien dort):
1. Tabelle je Einheit und Arm: Median-Abstand, Hauptursache, Beleg (Datei:Zeile im Agenten-Code und im Dossier).
2. H1-Zählung (übernommen / abweichend / fehlt) je Arm und je Angabe.
3. Gegenproben mit Befehl und Ergebnis.
4. Rangliste der Ursachen nach Anzahl betroffener Läufe, mit "belegt" / "wahrscheinlich" / "offen".
5. Vorschläge für ein vorab festlegbares v5 (je Vorschlag: was sich ändert, welche Einheiten es betrifft, erwarteter Effekt, Risiko eines Leaks; keine Testwerte in Prompts, keine Toleranzänderung nachträglich).

Arbeite token-sparsam: erst ein Skript, das alle 48 Läufe tabelliert, dann gezielt einzelne Codes lesen.

## Meldung

Ein Zwischenstand nach H1 (Zählung), dann die Endmeldung. Meldungen einzeilig an das Reviewer-Terminal:
`orca terminal send --terminal term_83e002c7-886d-4807-8921-ebbbbd6c29be --enter --text '<einzeilig>'`
