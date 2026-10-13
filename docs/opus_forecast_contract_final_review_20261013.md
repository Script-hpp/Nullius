# Abschluss-Review: Forecast-Vertrag und Abgabe im Worktree `ceo`

Stand **2026-10-13, 11:15 UTC**, `ceo` HEAD `b2e471d` plus 47 uncommittete Dateien (Kopie im Scratchpad, `ceo` unveraendert).
Fokus nur auf den neuen Forecast-Vertrag (`forecast_runner.py`, `forecast_worker.py`, `nullius_io.forecast_data`, Verdrahtung in `loop.py`).
Kein neuer breiter Audit, keine Fixes, keine externen Aufrufe.

## Ausgefuehrt

| Pruefung | Ergebnis |
| --- | --- |
| `test_selection_contract`, `test_trusted_evaluation`, `test_final_admission`, `test_observability_and_archives` mit `NULLIUS_DOCKER_TESTS=1` (netzlose Container) | 31 passed |
| JSON-Pipe-Kosten je Fenster (720x7 hin, 96x7 zurueck, lokal gemessen) | 2.6 ms, ca. 7.5 s fuer 2785 Test- + 117 Val- + 8 Kontrollfenster |

## Ergebnis

**Kein Blocker fuer die lokale Integration.** Die zuvor genannten Punkte 2 bis 6 sind im Code nachvollziehbar umgesetzt:
Wiederholungspruefung von 8 zufaelligen frueheren Historien (`atol=1e-4`) nach Abschluss aller Vorhersagen, `/output` nur fuer den Broker
beschreibbar, Holdout-Baselines, begrenzte Transfers. Die Reihenfolge stimmt: Vorhersage festschreiben, dann naechste Historie senden.
Smoke-Lauf, Val-Datei und Testdatei laufen ueber denselben Broker. Fehler des Workers enden fail-closed als Versuchsfehler ohne Testzahl.

## Verbleibende Hinweise (keine Blocker)

1. **Nebius bleibt ungeprueft.** Rechteabgabe (`setpriv`, `chown`) im Nebius-Branch ist nicht belegt. Bei fehlender Faehigkeit stoppt
   jeder ETT-Produktlauf sauber. Vor dem naechsten Produktlauf mit ETT braucht es eine Freigabe fuer eine einzelne Vorabpruefung.
2. **Laufzeit fuer schwere Modelle.** Inferenz laeuft jetzt fensterweise (2910 einzelne Aufrufe) statt gebuendelt. Fuer lineare Modelle
   unkritisch; Modelle mit teurer Inferenz je Fenster (z. B. Retrieval ueber die Trainingsreihe) koennen das Zeitlimit von 600 s fuer
   Training plus Inferenz reissen. Das ist eine Produktgrenze, kein Fehler; im Zeugnis bei Timeout als „Inferenzvertrag zu langsam“ benennen.
3. **Fehlermeldung fuer Reparaturen.** Stirbt der Worker, folgt auf seinen Traceback der Traceback des Brokers. Die Reparatur sieht nur
   die letzten 1500 Zeichen. Die eigentliche Fehlerzeile des Agenten bleibt meist erhalten, obere Frames koennen fehlen. Optional: Broker
   gibt nur eine kurze Zeile statt eines eigenen Tracebacks aus.
4. **Ausgabekanal des Workers.** Der Python-`print` des Agenten wird auf stderr umgeleitet. Ausgaben, die direkt auf den urspruenglichen
   Prozess-Ausgabekanal schreiben (selten, z. B. aus nativen Erweiterungen), stoeren das Zeilenprotokoll und enden als Vertragsfehler.
   Fail-closed, also nur Robustheit. Optional: Protokoll ueber einen eigenen Dateideskriptor statt stdout.
5. **Abdeckungsmessung fuer ETT entfaellt.** Mit dem Broker laeuft keine Zeilenabdeckung mehr (`with_cov` nur fuer `python train.py`).
   Achse 2 ist ohnehin „nicht geprueft“; im Zeugnis sollte fuer ETT kein Abdeckungshinweis erwartet werden.
6. **Vergleichbarkeit.** Produkt-ETT nutzt einen neuen Code-Vertrag. Fruehere Katalogwerte bleiben gespeichert, sind aber kein Vergleich
   unter demselben Vertrag.
