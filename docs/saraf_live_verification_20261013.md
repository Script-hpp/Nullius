# SARAF: echte Gegenprobe nach Deployment, 2026-10-13

**Ergebnis: gestoppt, kein gemessener MSE, kein Paperurteil.** Der Quellenfix ist im echten Lauf bestaetigt; eine erfolgreiche SARAF-Reproduktion ist weiterhin nicht belegt. Lokale Regressionstests hatten diese weiteren Fehler in neu generiertem Code nicht ausgeschlossen.

## Auftrag und Lauf

Nach Nutzerfreigabe und bestaetigtem Deployment wurde genau ein neuer Live-Check gestartet:

- Job `175724ef-f713-488f-b0cd-304eb880ea46`, [Live-Ansicht](https://nullius.akca.dev/#/live/175724ef-f713-488f-b0cd-304eb880ea46).
- Kandidat `disc_2606_04135_f7b4e6db785fdb63`, SARAF, ETTh1, Horizont 96, Ziel 0.371.
- Autoren-Nachbau, drei geplante Seeds, `selection_policy=validation_only`.
- Start 08:09:31 UTC, Ende 08:23:08 UTC; Dauer 816.27 Sekunden.
- Deploytes Frontend bytegleich mit dem lokalen Produktionsbuild von `8aa81c5`; SHA-256 `7a5aa57b182e5d1676e006bf50306c1ace1cf30bb6c3658ec630ab9dcd3b1c9b`.
- Keine neue Discovery und kein zweiter Reproduktionslauf gestartet. Budgetgrenzen unveraendert; der einzelne Lauf verwendete Tavily fuer README und Fehlersuche.

Die Originalereignisse, sechs erzeugte Code-Versionen, der Plan und Browserprotokolle sind lokal unter `logs/verification/saraf-20261013-080931/` gesichert. `events.jsonl` enthaelt 56 Ereignisse; SHA-256 `f6cf3183da16fb1dd78c9f93e741c8a1541aa472a95ac0cd6f5e6ce3ee7adbbe`. Diese Laufartefakte bleiben ausserhalb von Git; der Bericht dokumentiert die entscheidenden Befunde dauerhaft.

## Bestaetigte Fixes

Die Quellen-Gegenpruefung bestaetigte `verified_value=0.371`, `configured_value=0.371`, `agrees=true`, `from_store=false`. Der Autoren-Kontext enthielt **134162 Zeichen aus 14 Dateien**, darunter SARAF, Retrieval, Trainingsschleife, Datenloader und Scheduler. Repository-Revision `7b49aa8aabb38e8fe63adbc3146b1890958bf3ca`.

Bekannte Platzhalter wurden vor einem numerischen Urteil erkannt. Kein verworfener Kandidat wurde zur Endmessung erklaert. Der Abbruch bleibt `ungeprueft`, keine Behauptung gegen das Paper. Der Server gab den Job-Platz anschliessend wieder frei.

## Tatsaechliche Fehler vor dem Budget-Stopp

| Kandidat | Beleg | Ergebnis |
| --- | --- | --- |
| 0, `8a6ec78ce127` | `DummyRetrievalTool.prepare_dataset` tut nichts; `retrieve_all` liefert konstante Nullen. | Methodenschutz weist beide Platzhalter zurueck. Keine Messung. |
| 1, zuerst `f5da3bf799d6`, nach Steckbrief-Reparatur `c99217192f34` | Vollstaendigere Retrieval-Implementierung mit Stationaritaet, MMR und Ueberlappungsmaskierung. `StaticInputDataset.__getitem__` liefert aber **fuenf** Werte. Die Inferenzschleifen erwarten **vier**. | `ValueError: too many values to unpack (expected 4)` in Zeile 702. Versuch endete nach 47.7 Sekunden ohne Messung. |
| 2, `b18f7825347c`, `ab1e99ecf73f`, `7935cc403420` | `RetrievalTool.compute_time_aware_bonus` wurde durch konstante Nullen ersetzt. | Methodenschutz weist den Platzhalter auch nach Steckbrief-Reparaturen zurueck. Keine Messung. |

Der Entpackfehler wurde lokal isoliert wiederholt: Nur die Dataset-Klasse wurde mit harmlosen Array-Attrappen geladen, ohne erzeugte Imports, Modell, Training oder Datenzugriff auszufuehren. Ihr Rueckgabetupel hat Laenge 5; die beiden Inferenzschleifen in Zeile 702/711 entpacken 4 Werte. Das fuehrt genau zum aufgezeichneten `ValueError`.

## Weitere belegte Zuverlaessigkeitsluecken

1. **Falsche interne Validierungsziele.** `StaticInputDataset` erzeugt `seq_y` als Nullen. Die Trainings-Validierung in Zeile 674 nutzt diesen selben Dataset-Typ und berechnet `criterion(outputs, batch_y[:, -pred_len:, :])`. Die Checkpoint-Auswahl bewertet damit Nulltargets. Der unabhaengige Harness wuerde spaeter echte Targets benutzen; das macht die bereits falsche Checkpoint-Auswahl nicht richtig. Selbst eine Reparatur des Entpackfehlers wuerde diese Ursache offenlassen.
2. **Inferenz-Batch mit Trainings-Batch verwechselt.** Der neue Steckbrief erzwingt 32 aufgrund des Paper-Zitats ueber einen Inferenz-Effizienzvergleich. `run.py` definiert dagegen `--batch_size` mit Default 8 fuer das Training; der passende ETTh1/Horizont-96-Aufruf ueberschreibt ihn nicht. Diese Parameterzuordnung braucht eine Pruefung des Experimentzwecks.
3. **Unpassende Startskript-Overrides.** Der Plan und erzeugte Code nutzen `time_aware_weight=0.3`. `run.py` hat Default 0.5; der passende ETTh1/96-Aufruf setzt das Flag nicht. Im gemeinsamen Skript stehen Overrides fuer andere Datensaetze/Horizonte. `drop_run_script_defaults` sammelt momentan Flags aus dem gesamten Skript und entfernt den passenden Default trotzdem. Der Live-Steckbrief zeigt `time_aware_weight` in `dropped_defaults`.
4. **Unnoetige Konfigurationsreparatur.** Die Steckbrief-Pruefung beanstandet `batch_size=1024` in `RetrievalTool.retrieve_all` als Konflikt mit Trainings-Batch 32. Die Retrieval-Vorberechnung hat einen anderen Zweck; der Autorencode verwendet dort ebenfalls 1024. Ein globaler Keywordvergleich unterscheidet diese Rollen nicht. Diese Reparatur verbrauchte Modellbudget.
5. **Darstellung nach Stopp.** Der Seed-Bereich zeigt nach Abbruch weiterhin "Code candidates are being tested in parallel", obwohl der Job beendet ist. STOPPED und Abbruchgrund sind korrekt, dieser untergeordnete Hinweis ist veraltet.

Diese Befunde sind keine Testscore-Optimierung: Sie folgen aus Rueckgabetypen, Datenfluss und den gepinnten Quellen. Es liegen keine Testmessungen vor, anhand deren ein Zielwert optimiert werden koennte.

## Budget und abschliessende Verifikation

Der letzte Fehler lautet `BudgetExceeded: Job budget cannot cover another model request.` Zustaendig ist das Geldlimit pro Reproduktionsauftrag (`MAX_COST_PER_RUN_USD`), nicht das Tageslimit der Jobanzahl. Die Admission prueft bisherige Ausgaben plus offene Reservierungen plus eine konservative Obergrenze der naechsten Anfrage. Der Fehler bedeutet daher nicht zwingend, dass das gesamte Limit bereits faktisch ausgegeben war.

Das letzte aufgezeichnete LLM-Gesamtkostenfeld betraegt **0.36413 USD netto**. Zwei Tavily-Aufrufe sind separat aufgezeichnet; dies ist keine vollstaendige Anbieterrechnung. Grosse Quellen-Prompts, Generierungswiederholungen und Reparaturen lassen weniger Spielraum fuer die naechste Anfrage. Eine Budgeterhoehung allein wuerde weder falsche Validierungsziele noch falsche Parameterzuordnung beheben.

Echte Browserpruefungen auf Desktop und Mobil, waehrend des Laufs und nach dem Stopp: Zielwert sichtbar, keine JavaScript-Seitenfehler, kein horizontaler Ueberlauf, keine zusaetzlichen POST-Auftraege. Abschliessend `/api/jobs/current` mit `busy=false`; Live-API weiterhin freigegeben.

Naechste technische Arbeit: Daten-/Validierungsvertrag vor dem Training pruefen, Hyperparameter auf den passenden Trainingsaufruf beziehen, Retrieval- und Trainingsbatches trennen und vor kostenintensiven Reparaturen belegte Platzhalter aussortieren. Danach ist eine neue begrenzte Gegenprobe erforderlich. Dieser Test aendert keine Produktimplementierung und kein historisches Ergebnis.
