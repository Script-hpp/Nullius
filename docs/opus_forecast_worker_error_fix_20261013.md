# Fix: primaere Worker-Ausnahme im Forecast-Broker

Stand **2026-10-13**. Basis: `ceo`/main `8bb416d` (in `analyst` gemergt, Baum identisch). Geaendert nur
`src/repro_agent/harness/forecast_runner.py`, `src/repro_agent/harness/forecast_worker.py` und der neue Test
`tests/test_forecast_worker_errors.py`. Keine Provider-, Tavily- oder Nebius-Aufrufe; Docker-Laeufe nur mit `--network none`.

## Ursache, genau

Live-Kandidat `83f508ba9484` (FITS) scheitert im Training: 186 Ausgabeschritte (90 Rekonstruktion + 96 Forecast)
gegen 96 Ziele. Der Worker stirbt, waehrend der Broker die erste Historie (ca. 100 kB, mehr als ein Pipe-Puffer)
schreibt. Der Broker endet mit `BrokenPipeError` und eigenem Traceback **nach** dem Worker-Traceback.

Lokal mit dem alten Broker (Docker, gleiche Fehlerart, `docs/audits/opus_forecast_worker_error_docker_20261013.json`):

| | letzte Ausnahmezeile im Reparaturfenster (1500 Zeichen) | `error_query` (Rolle 5, Tavily) |
| --- | --- | --- |
| alt | `BrokenPipeError: [Errno 32] Broken pipe` | `BrokenPipeError: [Errno 32] Broken pipe` |
| neu | Zusammenfassung des Workers am Ende | `RuntimeError: The size of tensor a (N) must match the size of tensor b (96) …` |

Praezisierung zu `ceo/docs/fits_live_setup_verification_20261013.md`: Bei kurzem Worker-Traceback steht die
`RuntimeError`-Zeile im alten Fall noch **im** 1500-Zeichen-Fenster, aber nicht am Ende. Maskiert sind sicher:
die Fehlersuche (letzte Ausnahmezeile), die Ereignisanzeige (letzte 400 Zeichen), die Abbruchmeldung „letzter
Fehler“ und jedes Modell, das die letzte Ausnahme als Ursache liest. Bei laengerer Ausgabe nach dem Fehler faellt
die Ursache ganz aus dem Fenster.

## Loesung

- **Worker** (`forecast_worker.py`): Jede Ausnahme, auch `SystemExit` aus generiertem Code, wird gefangen. Der volle
  Traceback bleibt auf stderr. Zusaetzlich schreibt der Worker eine kurze JSON-Zusammenfassung (Phase, Ausnahme
  bis 600 Zeichen, bis zu drei Frames aus `train.py`, letzter Frame) auf einen eigenen Pipe-Deskriptor
  (`NULLIUS_FAILURE_FD`) und endet mit Exit-Code 1. Phasen: `loading train.py`, `training (train_forecaster)`,
  `prediction for history N (predictor)`. Formfehler nennen die gelieferte Form.
- **Broker** (`forecast_runner.py`, `WorkerChannel`): Schreibfehler (`BrokenPipeError`, `ConnectionResetError`)
  und `readline`-EOF fuehren zu `failure(stage)`: Worker einsammeln (10 s, sonst `kill`), Zusammenfassung
  nichtblockierend lesen (max. 8000 Bytes), eine begrenzte Meldung bauen (< 1200 Zeichen). Ohne Zusammenfassung
  nennt sie Exit-Code oder Signal (`SIGKILL` mit Hinweis auf das Speicherlimit). Vertragsfehler des Brokers
  (Form, JSON, Nichtdeterminismus, Exit-Code) sind ebenfalls `WorkerFailure`. Im `__main__` druckt der Broker
  nur diese Meldung als letzten Block und endet mit 1, ohne eigenen Traceback. Echte Broker-Defekte behalten
  ihren Traceback.
- **Unveraendert:** `setpriv`-Rechteabgabe (UID 65534, keine Capabilities, `--no-new-privs`), `/output` nur fuer
  den Broker, Reihenfolge Vorhersage festschreiben vor naechster Historie, Wiederholungsproben, 256000-Zeichen-
  Zeilengrenze, Soft-Timeout (`timeout -s ABRT` trifft weiter Broker und Worker; der Worker schreibt seinen Stack
  weiter direkt auf stderr). Unversandte Eingabe wird beim Aufraeumen geschlossen, damit beim Interpreterende kein
  zweiter Pipe-Fehler hinter der Meldung erscheint.

Grenze: Generierter Code kann den Zusammenfassungs-Deskriptor beschreiben oder schliessen. Das betrifft nur den
Diagnosetext fuer die eigene Reparatur, nie Messwerte oder Rechte.

## Tests

`tests/test_forecast_worker_errors.py`, echter Worker-Code in einem Kindprozess (ohne `setpriv`):

| Test | prueft |
| --- | --- |
| `test_training_error_survives_the_first_broken_history_write` | Trainingsfehler waehrend des **ersten Pipe-Writes**; Meldung beginnt mit der Ursache (`shapes (32,186,7) (32,96,7)`), Frame `train.py line 8 in mse`; voller Traceback auf stderr (> 4500 Zeichen); 1500-Zeichen-Fenster ohne `BrokenPipeError`; `error_query` liefert die Ursache |
| `test_prediction_error_after_successful_training_is_reported_at_readline_eof` | Vorhersagefehler, Broker sieht **`readline`-EOF**; Phase `prediction for history 0`, `IndexError`, Frame `lookup` |
| `test_wrong_forecast_shape_names_the_returned_shape` | Form `[186, 7]` steht in der Meldung |
| `test_exit_from_generated_training_is_a_named_failure` | `sys.exit(0)` im Training ist benannter Fehler |
| `test_worker_killed_without_python_exception_reports_the_signal` | `SIGKILL` ohne Python-Ausnahme |
| `test_valid_worker_still_serves_and_exits_cleanly` | Normalfall, Exit-Code 0 |
| `test_real_broker_in_sandbox_shows_the_torch_shape_error_to_repair` (Docker, `NULLIUS_DOCKER_TESTS=1`) | voller Produktpfad `_attempt` in `repro-snapshot:gate-etth1`, `--network none`, Rechteabgabe; Torch-Fehler `size of tensor a (186) … (96)` im Reparaturfenster, Frame `train_epoch`, kein `BrokenPipeError`, voller Torch-Traceback (`mse_loss`) in `detail` |

Derselbe Docker-Test **scheitert mit dem alten Broker** (Vorher/Nachher im JSON oben).

Ergebnisse: neue Datei 7 passed (inkl. Docker); Broker-/Vertragstests mit Docker (`test_selection_contract`,
`test_forecast_calendar`, `test_trusted_evaluation`, `test_final_admission`, neue Datei) 42 passed;
lokale Gesamtsuite 1530 passed, 20 skipped, 5 xfailed.
