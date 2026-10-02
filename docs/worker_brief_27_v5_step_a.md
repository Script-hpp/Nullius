# Auftrag 27 (worker-2): v5 Schritt A, Dossier-Treue von Rolle 4

Nach Auftrag 26. Vorab-Festlegung (final, nicht ändern): `benchmark/experiments/v5_step_a.md`. Lies sie zuerst ganz. Echte LLM-Läufe macht nur der Reviewer; du baust mit Fakes.

## Teil 1: Messwerkzeug (kein `src/`, Branch `v5-step-a-tools` von main)

`scripts/measure_setup_fidelity.py` (Vorlage: `scripts/experiment_role4.py`, `measure_role4_drops.py`):
- Läuft je Einheit nur Rolle 4 (Arm B, `tavily_mode="fixture"`, Repo-Dateien aus Fixtures, LLM-Cache aus), `--reps 3`, schreibt die rohen Setup-Angaben je Wiederholung nach `runs/v5_step_a/<label>/<unit>_<rep>.json` (`--label basis|v1|...`), dazu den Text, der in den Plan-Prompt ginge (`format_for_prompt`, Konflikte), und die Kosten. Kein Training, keine Agenten-Sandbox, kein Tavily. Wenn Rolle 4 für Repo-Dateien einen Sandbox-Klon braucht und keine Fixture hat: melden, nicht live klonen.
- `--dry-run` listet Einheiten, vorhandene Fixtures, geschätzte Aufrufe.
- Soll-Liste als `benchmark/experiments/v5_step_a_soll.yaml` (Inhalt wörtlich aus der Vorab-Festlegung) plus je Punkt feste Muster (Regex auf Thema und Wert) in derselben Datei unter `patterns`. Auswertung `--score <label>`: je Punkt und Wiederholung korrekt / fehlt / falsch, Tabelle je Einheit, Summen, die drei Ziele mit ja/nein, Liste der Zweifelsfälle (Thema getroffen, Wert unklar) für die Handprüfung. Ziel 3 (irreführende Angabe als gültig im Plan-Text) als eigene Muster.
- **Belegbarkeit:** `--check-soll` prüft je Soll-Punkt, ob der Wert in den gespeicherten Quellen (Paper-Fixture, README-Fixture, Repo-Dateien-Fixture) wörtlich vorkommt (Zahl und Stichwort in einem Fenster), Ausgabe Tabelle "belegbar in <Datei>" / "nicht gefunden". Nichts streichen; das entscheidet der Reviewer.
- Tests mit Fakes für Auswertung und Muster. Meldung mit Hash, `--dry-run`- und `--check-soll`-Ausgabe. Danach misst der Reviewer die Basis.

## Teil 2: Änderung V1 bis V3 (Branch `v5-step-a` von main, `src/`)

Wie in der Vorab-Festlegung beschrieben. Regeln:
- Nur Produktpfad? Nein: Rolle 4 läuft in beiden Pfaden; die Änderung hinter einem Schalter `setup_v2` (Orchestrator-Parameter, Default aus), damit v2 bis v4 reproduzierbar bleiben; die API und v5 schalten ihn ein.
- Kein Wert und kein Methodenwissen aus `tests/reference/` oder der Soll-Liste in Prompts oder Code (Test: Prompt-Texte enthalten keine Zahl aus der Soll-Liste). Die Pflichtliste nennt nur Themen ("feature normalisation"), nie Werte.
- Jede Angabe weiter mit wörtlichem Zitat (bestehender Check). "Nicht gefunden" je Pflichtpunkt ins Dossier.
- V3: Feld `applies_to` {model, dataset, setting} mit Zitat; Filter vor `format_for_prompt` (anderes Modell/Datensatz: nicht in den Plan, im Dossier markiert). Repo-Dateiauswahl nennt den Modellnamen des Claims.
- V2: Quellenrang als Feld `source_rank` (start script, config, argparse default, readme text, paper), Konflikte nach Rang aufgelöst und beide Werte im Plan-Prompt genannt (wie A3-M7), "optional advice" markiert.
- Tests mit Fakes, Suite. Meldung mit Hash; der Reviewer misst danach mit Teil 1.
