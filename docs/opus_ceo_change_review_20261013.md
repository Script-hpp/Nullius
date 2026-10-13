# Review der laufenden Aenderungen im Worktree `ceo`

Stand **2026-10-13, 10:35 UTC**. Gegenstand: uncommittete Aenderungen in `ceo` auf Basis `3b3dc10` (22 geaenderte Dateien, neue Module
`forecast_runner.py`, `forecast_worker.py`, `selection_contract.py`, `data_identity.py` und vier neue Testdateien). Read-only geprueft:
Diff und neue Dateien wurden in den Scratchpad kopiert; in `ceo` wurde nichts geaendert. Spaetere Aenderungen in `ceo` sind nicht erfasst.

## Geprueft und ausgefuehrt

| Pruefung | Ergebnis |
| --- | --- |
| Neue Tests auf einer Kopie (`test_selection_contract`, `test_trusted_evaluation`, `test_final_admission`, `test_observability_and_archives`) | 24 passed, 2 skipped |
| Dieselben Docker-Integrationstests mit `NULLIUS_DOCKER_TESTS=1` (Container mit `--network none`, lokale Images) | 13 passed (Selection-Contract und Trusted-Evaluation-Datei) |
| `setpriv` im Task-Image `python:3.11-slim` | vorhanden (`/usr/bin/setpriv`) |
| Code-Lesung | `loop.py`, `tasks.py`, `sandbox/local.py`, `harness/*_val_eval.py`, neue Module |

Keine Nebius-, Provider- oder Tavily-Aufrufe.

## Was die Aenderungen erreichen

- **E01:** Bewertet wird in einem frischen Branch des Setup-Snapshots. Nur `predictions.csv` (und bei Bedarf `val_predictions.csv`) wird
  uebertragen. Die Harness-Ausgabe muss genau ein JSON-Objekt sein. Der Docker-Test bestaetigt, dass ein vom Agenten installierter
  Startup-Hook weder Test- noch Val-Wert aendert. Das schliesst die Klasse aus meinem Bericht fuer beide Backends, soweit die
  Branches unabhaengig vom Agenten-Branch erzeugt werden.
- **E02:** Fuer Forecast passt der Fit-Datensatz nur noch train und 80 % Val; die Testfenster werden einzeln und chronologisch
  gereicht, jede Vorhersage wird vor der naechsten Historie festgeschrieben. Der generierte Code laeuft unter einer eigenen UID ohne
  Capabilities und kann die private Quelldatei nicht lesen (im Test belegt).
- **E03/E04:** Auswahl auf zurueckgehaltenen 20 % (Graph: `selection_idx` mit verborgenen Labels; ETT: Fenster, deren Ziel ganz im
  zurueckgehaltenen Teil liegt). Sichtbare Val-Daten bleiben fuer Checkpoints. Damit ist die Auswahl strukturell unabhaengig.
- **E05/E06:** Kontrollbefehle mit Exit-Code ungleich 0 machen den Versuch ungueltig; `nicht messbar` schliesst im Produkt aus.
- **N03/E13:** Jede Ersatzversion durchlaeuft `_selection_check`; der finale Hash wird vor den Seeds zugelassen (`_admit_final_code`);
  Spec-Score wird fuer den finalen Hash nachgerechnet. Die Begruendung nennt Ablation nur noch, wenn sie lief.
- **N01/N02:** `LocalSandbox.run_agent` trennt das Container-Netz statt `unshare` zu brauchen; `put_file` legt Verzeichnisse an.
- **R01/R02:** Datenidentitaet zwischen Vorbereitung und Bewertung; sequentielle Kandidaten mit Rueckfall auf bereits validierte
  Kandidaten bei Budgetgrenze (Abbruch bleibt bei Billing/Cancellation).

## Befunde

Nach Schwere geordnet. Keiner blockiert die Integration, aber 1 bis 3 sollten vor einem Produktlauf geklaert sein.

1. **Nebius-Faehigkeit ist nicht belegt (offen).** Der Forecast-Vertrag braucht im Nebius-Branch Root-Rechte zum Rechte-Abgeben
   (`setpriv`) und `chown`. `_prepare_selection_contract` prueft das vorab und stoppt sonst mit Infrastrukturfehler (gut, fail-closed).
   Folge: Wenn Nebius-Sessions das nicht erlauben, laeuft **kein** ETT-Produktlauf mehr. Empfehlung: einmalige, kostenlose
   Sandbox-Vorpruefung nur mit diesem Preflight-Befehl, nach ausdruecklicher Freigabe durch den Nutzer.
2. **Snapshot-Versionierung fuer das Datenmanifest.** Unter dem Selection-Contract ist ein fehlendes `.source.json` ein harter Stopp.
   Aeltere, wiederverwendete Snapshots (z. B. `repro-snapshot:gate-etth1`) haben kein Manifest. Pruefen, dass Setup-Snapshots nach dieser
   Aenderung neu gebaut werden (Label/Version im Snapshot-Namen), sonst stoppt jeder Lauf mit „identity unavailable“.
3. **Zustandsbehafteter Predictor.** Ein Worker-Prozess bekommt nacheinander alle Val- und Testhistorien. Kausal ist das korrekt
   (nur Vergangenheit), aber ein Predictor kann zwischen Fenstern lernen (Test-Time-Adaptation). Das ist kein Zukunftsleck, aber eine
   Protokollabweichung gegenueber Papers mit festem Modell. Minimale Pruefung: am Ende einige fruehere Historien erneut senden und
   identische Vorhersagen verlangen (Toleranz ~1e-6); Abweichung = ungueltig. Kein Testscore geht dabei in eine Reparatur.
4. **Val-Baselines im Holdout abgeschaltet.** `ett_val_eval` setzt im Selection-Modus `baselines = {}`. Damit ist der Val-Fehlercheck
   (`_val_fault_check`, Vergleich mit trivialer Baseline) fuer ETT still aus. Empfehlung: Baselines (Repeat, Zero) auf denselben
   Holdout-Fenstern berechnen, statt sie zu leeren. Graph-Val-Eval entsprechend pruefen.
5. **Kleine Auswahlmenge.** ETTh1: 21 von 117 Val-Fenstern (Starts ab Zeile 2304 von 2880), Cora: 100 Knoten. Das ist methodisch
   sauber, aber verrauschter als bisher. Im Zeugnis neben der Auswahl nennen (Anzahl Fenster/Knoten).
6. **Uebertragungsgroesse.** `get_file` liest die Vorhersagedatei aus dem Agenten-Branch vollstaendig in den Orchestrator. Eine
   Obergrenze (erwartete Zeilenzahl x Zeilenlaenge, z. B. 64 MB fuer ETTh1) vor dem Lesen pruefen, damit eine fehlerhafte Ausgabe den
   Server nicht belastet.
7. **Graph-Agent laeuft weiterhin privilegiert im eigenen Branch.** Das ist nach der Trennung der Bewertung unkritisch, weil der Branch
   keine verborgenen Labels mehr enthaelt (Setup loescht Rohdaten, `graph_public` ueberschreibt die Datei). Bitte als bewusste Grenze
   dokumentieren; jede kuenftige Datei mit Selektionslabels im Agenten-Branch wuerde die Garantie aufheben.
8. **Statische Val-Pruefung fuer Forecast schliesst auch Training auf sichtbarer Val aus.** Unter dem neuen Vertrag ist das kein
   Auswahlleck mehr, aber der Prompt verlangt Fit nur auf train. Konsistent, aber strenger als noetig; kann gueltige Kandidaten kosten.
   Entscheidung dokumentieren.

## Bisherige Benchmarks

- `independent_selection` ist nur in `product_kwargs()` an; der Orchestrator-Default bleibt aus. Benchmark-Skripte (`graph_repair=False`)
  nutzen weder Kandidatenauswahl noch Zulassung.
- **Gilt trotzdem fuer alle Pfade:** Bewertung im frischen Branch, strenges Ein-Objekt-Parsing der Harness-Ausgabe, Datenidentitaet
  (ohne Selection-Contract nicht blockierend), Exit-Code der Kontrollbefehle. Fuer ehrlichen Code sollten die Zahlen identisch sein.
  Empfehlung: eine Referenzloesung (`tests/reference/gcn_cora_reference.py`) lokal und netzlos durch den alten und neuen Pfad laufen lassen
  und gleiche Accuracy verlangen. Alte Ergebnisse bleiben unveraendert gespeichert.
- **Produkt-ETT ist ein neuer Code-Vertrag** (`train_forecaster` statt Skript mit CSV). Katalogergebnisse vor dieser Aenderung
  (TSMixer, SparseTSF, SARAF-Versuche) sind damit nicht direkt vergleichbar; im Zeugnis Vertragsversion nennen.

## E16-Nachbesserung in `analyst`

Commits `f1b183b` und `3814a6b`: Jeder ausdruecklich verlinkte Ref wird ausgecheckt, auch `main` und `master`; nur `HEAD` bezeichnet den
Standard-Branch. Fehlt der Ref, scheitert das Setup sichtbar, ohne zweiten Klon. Test `test_requested_main_is_not_replaced_by_a_different_default_branch`
(Repo-Default master, angefordert main). Angepasste Erwartungen in `test_repo_resolution.py` und `test_discovery.py`. Repo-nahe Tests 295 passed,
fokussierte Suite 493 passed, 3 skipped. Repo-Suchtreffer bleiben beim Standard-Branch (keine Anforderung des Papers); Katalog und Fixtures unveraendert.
