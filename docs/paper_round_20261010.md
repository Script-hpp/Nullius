# Naechste Paper-Runde — Auswahl vor den Laeufen

Stand: 2026-10-10. Manuelle Quellensuche und Abgleich mit dem bestehenden Produkt. Noch keine Reproduktionslaeufe dieser Runde gestartet. Auswahl ist gezielt nach Machbarkeit; keine zufaellige Stichprobe oder allgemeine Erfolgsquote. Die Papers sind keine Einheiten des bisherigen Acht-Einheiten-Benchmarks. Ihre Namen tauchen teilweise bereits in vorhandenen Quellen auf; „bisher voellig unbekannt“ waere nicht belegt.

**Nachpruefung auf Nutzerhinweis:** CycleNet wurde bereits am 2026-10-09 in der Seeded-Discovery-Runde im Workspace `catalog-live` angefragt. Der protokollierte Durchgang um 18:41:45 UTC meldete `REJECT 2409.18479: [extract] Volltext nicht lesbar`; der Abschlussbericht bestaetigte, dass diese Ablehnung nicht gecacht wurde. Quelle: gespeicherte Orca/Claude-Session `475f2a0c-ec31-4307-ba40-cb608ff7672e`. Zuvor trat in derselben Runde auch arXiv HTTP 429 auf. Das ist ein belegter frueherer Discovery-Versuch, kein abgeschlossener Reproduktionslauf. In den geprueften Run-Verzeichnissen, Dossiers und Git-Historien wurde kein eigener CycleNet-Experimentlauf gefunden. CycleNet ist daher ein erneuter Discovery-Kandidat mit inzwischen lesbarem LaTeX-Volltext, kein erstmals entdecktes Paper.

Ausserdem wurde CycleNet im SparseTSF-Lauf vom 2026-10-09 als Community-Quelle und im gelesenen README erwaehnt. Auch das ist keine CycleNet-Reproduktion.

## Drei Kandidaten, feste Reihenfolge

| Reihenfolge | Paper / Variante | Vorab festgelegter Zielwert | Setup | Einschaetzung |
|---|---|---:|---|---|
| 1 | CycleNet / Linear | MSE 0.378 | ETTh1, Eingang 96, Horizont 96, alle sieben Kanaele | Einfaches Modell und klares Skript; mein erster Kandidat. CPU-Laufzeit noch nicht gemessen. |
| 2 | SOFTS | MSE 0.381 | ETTh1, Eingang 96, Horizont 96, alle sieben Kanaele | Offizielle Quelle und passende Zahlen; groesseres Modell, Laufzeit vorab beobachten. |
| 3 | TimeMixer, vereinheitlichtes Setup | MSE 0.375 | ETTh1, Eingang 96, Horizont 96, alle sieben Kanaele | Passendes kleines Setup, komplexere Methode. Das andere Ziel 0.361 gehoert zu einem anderen Setup und wird nicht als Ersatz verwendet. |

FreTS wurde angeschaut und fuer diese Runde zurueckgestellt: Die [Originaltabelle](https://arxiv.org/html/2311.06184v1) berichtet MAE/RMSE statt des direkt vergleichbaren MSE-Setups dieser Runde. Zahlen anderer Arbeiten werden nicht als FreTS-Originalclaim eingesetzt.

### CycleNet

[Paper v1, Appendix Table 7](https://arxiv.org/html/2409.18479v1) trennt Linear und MLP: 0.378 ist Linear, 0.375 ist MLP. Wir waehlen vorab Linear und wechseln nach einem Fehlschlag nicht zur anderen Spalte.

[Offizielles Repo](https://github.com/ACAT-SCUT/CycleNet), gelesener Commit `d807e51fc2dcd143885ee639d97965a7ab0926f4`. [Skript](https://github.com/ACAT-SCUT/CycleNet/blob/d807e51fc2dcd143885ee639d97965a7ab0926f4/scripts/CycleNet/Linear-Input-96/etth1.sh): Zyklus 24, maximal 30 Epochen, Patience 5, Batch 256, Lernrate 0.01; RevIN standardmaessig eingeschaltet. Paper berichtet fuenf Seeds, unser Produktcheck verwendet vorab drei; entsprechend beschriften.

### SOFTS

[Paper v3, Appendix C.1](https://arxiv.org/html/2404.14197v3): ETTh1 / 96, SOFTS-Spalte, MSE 0.381; der Wert 0.449 in der Zusammenfassung ist der Durchschnitt ueber Horizonte und kein Ersatz fuer diesen Claim.

[Offizielles Repo](https://github.com/Secilia-Cxy/SOFTS), Commit `f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44`. [Skript](https://github.com/Secilia-Cxy/SOFTS/blob/f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44/scripts/long_term_forecast/ETT_script/SOFTS_ETTh1.sh): eine Schicht, `d_model=d_core=d_ff=256`, Lernrate 0.0003, Cosine-Schedule, 20 Epochen, Patience 3; Batch-Default aus `run.py`: 32.

### TimeMixer

[Paper v1, vollstaendige Forecasting-Ergebnisse](https://arxiv.org/html/2405.14616v1): fuer das vereinheitlichte Setup MSE 0.375; gesuchte Hyperparameter liefern einen anderen Wert. [Offizielles Repo](https://github.com/kwuking/TimeMixer), Commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`.

[Skript](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh): zwei Schichten, drei Downsampling-Schichten, Faktor 2, `d_model=16`, `d_ff=32`, Lernrate 0.01, 10 Epochen, Patience 10, tatsaechliches Batch-Argument 128. Die Shell-Variable `batch_size=16` wird von diesem Aufruf nicht verwendet. `drop_last` im offiziellen Programm verlangt einen Abgleich mit der vollstaendigen Harness-Messung; eine ergaenzende Teilmengenmessung ersetzt nicht nachtraeglich die Hauptmetrik.

## Vor den bezahlten Laeufen

- Bestehenden ETTh1-Harness verwenden: Split 12/4/4 Monate, Skalierung mit Trainingsdaten, Hauptmetrik ueber alle Testfenster. Offiziellen Loader gegen diesen Split abgleichen. Abweichungen vor dem Lauf dokumentieren, keinen passenden Split nach Ergebnis auswaehlen.
- Modus: Paper plus Autoren-Code als Text, eigenes Agentenprogramm; Originalcode nicht ausfuehren. Offizielle Commits als Quellen festhalten.
- Je ein Produktlauf mit drei Seeds `0,1,2`, Toleranz ±0.01 zum Mittelwert. Zielwert erreicht, verfehlt oder Setup-Fehler getrennt berichten. Methodentreue bleibt ein separater Pruefpunkt.
- Planungsschwelle hoechstens bestehende 0.50 USD brutto pro Paper, insgesamt hoechstens 1.50 USD Inferenzplanung fuer drei Paper. Das ist wegen der bekannten Budget-Luecken kein nachgewiesenes hartes Kostenlimit und deckt nicht automatisch Tavily-/Sandboxkosten ab. Keine Limits erhoehen.
- Vor dem ersten bezahlten Aufruf: massgebliche Kostenlogs, aktuellen Restbetrag und Tavily-Reserve verifizieren. Der derzeit selbstreferenzierende gemeinsame `logs`-Symlink verhindert verlaessliche Standard-Logzugriffe. Nicht durch einen leeren Logordner ersetzen: dadurch koennte die bisherige Budgethistorie verschwinden.
- Jede Ausfuehrung samt Fehlschlag behalten. Erneute Generationen oder Reparaturen nach Abschluss als weitere Versuche markieren, statt nur den besten Lauf zu berichten.

## Discovery: was jetzt belegt ist

Nutzerbericht: offizielle Repositories wurden nicht erkannt; Tabellenprobleme fuehrten zur Ergaenzung von LaTeX-Quellen. Eine belastbare Vorher/Nachher-Erfolgsquote liegt nicht vor.

1. **Falscher pauschaler Repo-Link behoben und gemergt.** `find_repo_links` fuegte zuvor SegRNN in jeden Papertext ein. Die Entfernung plus Regressionstests sind in `97cd438` auf `main`.
2. **Bekannte Katalog-Papers werden jetzt wirklich erkannt.** `discover_ids.py` las eine Liste statt des realen `candidates`-Mappings. Jetzt verwendet es den vorhandenen Katalog-Lader. Reproduzierter Test vor Fix fehlgeschlagen, danach bestanden; Commit `62897ec` auf `main`. 68 fokussierte Tests bestanden.
3. **Aktueller Repo-Weg an vier echten Quellen geprueft.** Mit der aktuellen LaTeX-Extraktion und den echten READMEs erkennt `_find_repo_and_context` TSMixer (Google-Unterordner), SparseTSF, CycleNet und SOFTS als offizielle, im Paper verlinkte Repositories. Kein LLM-Aufruf; bezahlte Tavily-Suche im Diagnoseobjekt gesperrt. Das beweist diese vier heutigen Kontrollfaelle, nicht die Erfolgsquote aller Discovery-Kandidaten gestern.
4. **LaTeX-Tabellen bleiben nicht automatisch spaltenverifiziert.** Die aktuelle Abschnitts-/Tabellenauswahl liefert Tabellen. Bei direkt uebergebenen Zielwert-Zeilen aus TSMixer, SparseTSF und SOFTS meldet `check_columns` mit aktueller Normalisierung „Kopfzeile der Tabelle nicht erkannt“ / nicht pruefbar. Dieser Teiltest simuliert keine LLM-Extraktion und misst keine Verbesserung ihrer Trefferquote. Die Pipeline darf solche Faelle weiterhin nur als `unchecked` ausweisen.
5. **Lokales Logproblem reproduziert.** `/home/onuralp/Desktop/hackathon/research/logs` ist ein Symlink auf denselben Pfad; auch die verlinkten Workspaces sind betroffen. Ein vorhandener Backup-Cache im Assistant-Workspace enthaelt 22 Eintraege, davon 16 Ablehnungen: sechs ohne numerischen Claim, vier Zitate nicht gefunden, drei Prefilter, je ein Machbarkeits-, Horizont- und Protokollfall. Das Backup ist keine vollstaendige Ablaufhistorie; daraus folgt keine gesicherte Ursache jedes gestrigen Fehlers.

Naechster gezielter Discovery-Schritt: LaTeX-Tabellenstruktur mit Kopf, Modell, Metrik, Datensatz, Horizont und Eingabelaenge als Regressionfaelle aufarbeiten. Bis diese Zuordnung verifiziert ist, die manuell geprueften Kandidaten als Auswahlhilfe verwenden und ungepruefte Spalten nicht als bestaetigt anzeigen.
