# Katalog- und Produktionspruefung: Quellenbindung

Stand: 2026-10-13. Anlass war der FITS-Live-Lauf
`3fd85051-a3cf-4972-8f27-f7a79d769a2c`, der vor dem Experiment stoppte:
Katalogziel 0.372, generische Modellgegenpruefung 0.368. Beide Zahlen stehen
in verschiedenen Ergebnistabellen derselben Paperfassung; die bisherige
Gegenpruefung erhielt das ausgewaehlte Discovery-Zitat nicht als Tabellenanker.

## Belege und Umfang

Geprueft wurden alle **17** Eintraege aus `benchmark/catalog.yaml` und die
**5** oeffentlich gespeicherten Live-Kandidaten, insgesamt **19** unterschiedliche
Paper. Nach dem begrenzten erneuten Quellenabruf waren alle 19 Quellen lesbar.
Das bedeutet Quellen-/Katalogpruefung, nicht 22 neue Reproduktionen.

Die frische FITS-Quelle hat SHA-256
`9022b12b628fc1807f0ef11bfd7d1266f25b8575c1418afeb7bb90c251a00d64`;
dies ist derselbe Text-Hash wie im Produktionslauf. Mit diesem Text bindet die
neue Pruefung 0.372 an `tab:FITSett` und 0.368 separat an `tab:app_ett`.
Echte Tabellen von FITS, SparseTSF und SegRNN sind als lokale Regressionsexzerpte
gespeichert, jeweils mit Quellenangabe.

Quellen: [FITS v3](https://arxiv.org/html/2307.03756v3),
[SparseTSF](https://arxiv.org/abs/2405.00946),
[SegRNN](https://arxiv.org/abs/2308.11200),
[xCPD](https://arxiv.org/abs/2603.13702),
[Acacia](https://arxiv.org/abs/2609.30894).

Maschinenlesbare Einzelfunde: `docs/audits/catalog_source_binding_20261013.json`.
Unabhaengige Gegenpruefung: `docs/opus_fits_target_binding_catalog_review_20261013.md`
mit Erstbericht und Schlusspruefung; Opus arbeitete auf Scratch-Kopien.

## Ursachenfixes

1. **Tabellenanker weitergeben.** Bei strikten Forecast-Katalogclaims prueft
   `resolve_targets` das ausgewaehlte Zitat gegen genau eine Quelltabelle.
   Methode, Datensatz, Horizont, Metrik und Ergebniszelle muessen gemeinsam
   belegt sein. Die Evidenz enthaelt die echte Originalzeile, Tabellenlabel,
   Quellen-URL/-Version und Text-Hash. Es wird nicht nach einer Zahl gesucht,
   die den gewuenschten Zielwert trifft.
2. **Weitere reale Tabellenformen.** Der Parser unterstuetzt Datensatz- und
   Horizontspalten, `multicolumn`, `multirow`-Methodennamen und verschachtelte
   Mini-Tabellen in Kopfzellen. Explizite Mittelwert-/Streuungszellen werden
   getrennt gelesen. Bei einer Caption mit MSE und MAE ist eine passende
   Metrikkopfzeile erforderlich. Mehrdeutige oder unpassende Spalten stoppen.
3. **Quellendrift und Kosten.** Fehlt ein LaTeX-Anker, wird kein anderer Wert
   aus einer generischen Modellantwort uebernommen. Eine eindeutig geparste
   Originaltabelle kann weiterhin alte synthetisierte Zitate ersetzen; sonst
   stoppt die Pruefung. Vorkommen des Ankers werden im ganzen Quelltext
   gezaehlt, auch ausserhalb normaler `table`-Umgebungen. Ein unlesbarer
   benoetigter LaTeX-Text fuehrt nicht zu einem bezahlten Tavily-Formatwechsel.
   Der freie Download versucht transiente Fehler maximal zweimal mit 30 s
   Zeitlimit; 404 wird nicht wiederholt. Eine ermittelte arXiv-Version wird
   nun auch tatsaechlich in der E-Print-URL verwendet.
4. **Neue Discovery-Belege erhalten.** Neue Kandidaten speichern Quellenart,
   Text-Hash, URL und nach Moeglichkeit Tabellenlabel. Die Live-Discovery
   prueft den unterstuetzten Horizont auch bei transponierten Tabellen;
   der gespeicherte Kandidat besteht anschliessend dieselbe Zielpruefung.
5. **Eingabelaengen belegen.** Ein gespeichertes `input_length` und eine 720
   im Horizontkopf sind keine Setup-Evidenz. Ein eigenes Laengenzitat oder
   eine konkrete Angabe im `protocol_quote` muss in der frisch gelesenen
   Quelle vorkommen. Formatierung, `L = 512` und „for all experiments“ werden
   beruecksichtigt. FITS ist auf **360** korrigiert; SparseTSF und SegRNN
   erhalten belegte Laengenzitate. Die bekannten Angaben fuer GLinear 336,
   TSMixer 512, SparseTSF 720 und SegRNN 720 bleiben nutzbar.
6. **Unpassende gespeicherte Claims vor Start sperren.** Der Acacia-Claim
   betrifft einen auf dem Webgraph vortrainierten, eingefrorenen Kern;
   Cora-Training von Grund auf prueft ihn nicht. Beim xCPD-Claim 0.357 fehlt
   der TSMixer-Unterbau, und seine Eingabelaenge 96 passt nicht zur belegten
   512. Beide Eintraege bleiben sichtbar, sind aber mit konkretem Grund
   `NotRunnable`, bevor Tageszaehler oder Provider-Aufrufe starten. Die
   allgemeine Discovery-Vorpruefung erkennt auch die explizite Acacia-artige
   Abhaengigkeit von externem Webgraph-Training.
7. **Redeploy mit persistentem Volume.** `refresh_catalog_metadata` uebernimmt
   korrigierte Definitionen in `/data/catalog.json` beim Start. Vorhandene
   Statuswerte, Urteile, Kosten, Run-IDs, Zusammenfassungen, Index und nur live
   vorhandene Kandidaten bleiben erhalten. Sonst wuerde das Volume trotz
   korrigierter API-Definition weiter die alte FITS-Laenge anzeigen.

Historische Zielwerte, Messwerte, Urteile und Laufartefakte wurden nicht
ersetzt. Die expliziten Sperren betreffen zwei ungepruefte Katalogclaims.
Der alte Live-SARAF-Claim 0.366 bleibt bei der frischen Spaltenpruefung
blockiert; 0.371 ist separat belegt.

## Abnahme und Budget

- Backend-Suite ohne Provider-Integration: **1508 passed, 9 skipped,
  8 deselected, 5 xfailed**; 321.51 s. Die anschliessende kleine Ergaenzung
  fuer Metadatenrefresh und Acacia ist in den fokussierten Gates abgedeckt.
- Fokussierte Schlusspruefung einschliesslich Produktionsstart, Export,
  Quellenfehlern und Katalog: **146 passed**; weitere letzte Gate-Runde
  fuer Laengenbindung/Discovery und Quellenparser: **108 passed**.
- Unabhaengige Opus-Schlusspruefung: **311 passed**, keine Blocker.
- Produktions-Dockerimage gebaut. Netzloser Image-Smoke: HTTP-Health ok,
  exportierte FITS-Laenge 360, FITS startbar, xCPD und Acacia gesperrt.
- `git diff --check` bestanden.
- Zusaetzliche Modell-/Tavily-/Nebius-Aufrufe: **0**; zusaetzliche
  Provider-Kosten **0 USD**. Downloads oeffentlicher Quellen sind kostenfrei.

## Grenzen

Markdown-/PDF-Altzitate ohne erkennbare LaTeX-Herkunft behalten den bisherigen
Extraktionspfad mit nachfolgender Abweichungs-/Spaltenpruefung. Nicht jedes
Paperlayout wird deterministisch erkannt. Die kurze Eingabelaengenangabe ist
an den frischen Quelltext gebunden, nicht automatisch an jeden Tabellenkontext;
die korrigierten kuratierten Angaben wurden deshalb gegen Paper und Autorenpreset
kontrolliert. Unbekannte arXiv-Versionen werden nicht erfunden; der Text-Hash
belegt den wirklich gelesenen Inhalt.

Die lokale Abnahme beweist die reparierte Zielpruefung und den API-/Deploymentpfad.
Ein neuer autonomer FITS-Trainingslauf und die Nebius-Isolation bleiben nach dem
Deployment zu pruefen. Es gibt keinen neuen Paper-Reproduktionsbefund aus dieser
Reparaturrunde.
