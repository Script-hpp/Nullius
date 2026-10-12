# Discovery-Replay: Jury-Einstieg — 12. Oktober 2026

Nutzerbefund: Im aufgezeichneten Suchlauf stehen `Agent error suspected` und `Not reproduced` gross auf den gefundenen Papers. Dadurch entsteht der Eindruck, die gerade gezeigte Discovery funktioniere nicht.

Der Replay enthaelt eine reale Suche vom 30. September: 25 gefundene Papers, sechs gelesene Papers, zwei ausgewaehlte Claims und vier verworfene Papers. Discovery liefert Claims und Quellen, keine neuen experimentellen Messungen. Die UI mischte diese Suchergebnisse mit nachtraeglich geladenen Katalogurteilen, bisher nur ueber die arXiv-ID verknuepft.

MoHETS' gespeicherter Check vom 30. September misst MSE 2.3191 und ist schlechter als die triviale Null-Baseline (1.1099). Sein Urteil `Agenten-Fehler vermutet` ist kein Befund ueber das Paper. Beim SL-GCN-Check vom 9. Oktober wurden rund 59.51 % statt des Zielwerts 82.92 % gemessen; die Methodentreue ist ungeprueft, und die Modell-Datei fehlte laut Warnung im Repo-Ausschnitt. Diese Werte wurden nicht im Discovery-Replay gemessen.

## Aenderungen

- Hauptansicht: ausgewaehlte Claims zuerst, als `Checkable claim found`, mit Paper-Ziel, Datensatz, Quelle und den vorhandenen Spalten-/Protokollhinweisen.
- Verworfenes Suchmaterial erscheint unter `Papers outside the discovery criteria`, mit unveraenderten Gruenden. Alle Kandidaten bleiben sichtbar, unabhaengig von spaeteren Messungen.
- Gespeicherte Reproduktionschecks stehen in einem neutral beschrifteten, standardmaessig geschlossenen Bereich mit Datum. Er erklaert, dass der Suchlauf dieses separate Experiment nicht ausfuehrt oder aktualisiert. Positive und negative Urteile folgen derselben Regel; Messwerte und Evidenzlinks bleiben erreichbar.
- Ein gespeicherter Check wird nur zugeordnet, wenn arXiv-ID, Methode, Task, Metrik, Ziel, Forecast-Horizont und bekannte Eingabelaenge zusammenpassen. Genauigkeitswerte als Anteil und Prozent werden normalisiert; bei mehreren passenden Checks gilt der neueste, unabhaengig vom Urteil.
- Im Replay heissen Tavily-Marken `recorded`; im echten Live-Modus weiter `live`. Der Feed benennt Wiedergabe ohne neue API-Aufrufe.
- Der naechste gefuehrte Schritt oeffnet ein vorhandenes positives Katalogbeispiel (aktuell GLinear), ausdruecklich als separates aufgezeichnetes Beispiel mit ungepruefter Methode. Die Gesamtergebnisse bleiben im Katalog zugaenglich.

Keine Replay-Ereignisse, gespeicherten Messungen, Urteile oder Suchfilter veraendert; keine frischen kostenpflichtigen Laeufe.

## Pruefung

- 135 Frontend-Tests bestanden, darunter drei neue Regressionen mit echten Replay-/Katalogdaten fuer korrekte Zuordnung, abweichende Claims und die neueste Messung unabhaengig vom Urteil.
- Produktionsbuild und Lint erfolgreich; bestehende Warnungen bleiben.
- Browserpruefung auf Desktop und mobil, jeweils Replay und simulierte Live-Ereignisse: zwei Claims sichtbar, alte Urteile anfangs eingeklappt, beide Negativurteile und alle vier verworfenen Papers aufklappbar. Aufgezeichnete und neue API-Aufrufe eindeutig beschriftet. Weiterfuehrender Link zeigt zum gespeicherten GLinear-Beispiel. Keine JavaScript-Seitenfehler.
