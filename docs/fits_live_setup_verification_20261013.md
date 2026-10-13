# Live-Gegenprobe FITS: Quellenfix bestaetigt, Setup weiterhin falsch

Gepruefter Auftrag: `ee039f0f-5a4c-47d3-83a7-ad7388473738`,
[Live-Verlauf](https://nullius.akca.dev/#/live/ee039f0f-5a4c-47d3-83a7-ad7388473738).
Diese Untersuchung liest den vom Nutzer gestarteten Lauf; sie startet keine
zusaetzlichen Providerjobs und aendert keine Produktlogik.

## Quellenpruefung erfolgreich

Der Server liest FITS v3 frisch um `2026-10-13T16:55:28+00:00`.
Quellenhash `9022b12b628fc1807f0ef11bfd7d1266f25b8575c1418afeb7bb90c251a00d64`
passt. `target_check` bestaetigt 0.372, `target_resolved` bindet diesen Wert
an `tab:FITSett`, Spaltenpruefung `ok`, `from_store=false`.
Der vorherige Abbruch wegen fehlender LaTeX-Quelle tritt hier nicht auf.

## Neue, belegte Setupfehler

| Merkmal | Lokale Autorenreferenz, vor Scores festgelegt | Live-Plan/erster Code |
| --- | --- | --- |
| Eingabelaenge | 360 | 90 |
| cut_freq | 106 | 3, aus `seq_len // 24` |
| Batch | effektive 128, CLI 64 | 32 |
| Learning rate | 0.0005 | 0.001 |
| Epochen/Patience | 50 / 20 | 10 / 3 |

Die Referenz samt primaeren Quellen ist in
`docs/fits_independent_verification_20261013.md` dokumentiert; die
Hyperparameter wurden vor Testmessungen festgelegt. Unterschiede allein
beweisen keine ausschliessliche Ursache eines Scores. Sie beweisen aber,
dass der Live-Versuch dieses veroeffentlichte Setting nicht nachbaut.

1. **Mehrfachauswahl wird als Einzelvorgabe erzwungen.** `setup_details`
   nimmt 90 aus einer Tabellenkopfzeile mit 90/180/360/720. `spec` macht
   daraus `seq_len=90, enforce=true`, obwohl der Katalog und das zusaetzliche
   Discovery-Detail die belegte 360 enthalten. Der Plan und Codehash
   `947d5088ab2c` verwenden wirklich 90. Die Spec-Pruefung meldet trotzdem
   2/2 erfuellt: Sie prueft den falschen Steckbrief.
2. **Discovery-Beleg geht bei der Spec-Verifikation verloren.** Lokale
   deterministische Gegenprobe mit unveraenderter v3-Quelle:
   `verify_entries` weist `seq_len=360`, Zitat `look-back window of 360`,
   `source=discovery` mit `Quelle unbekannt` ab. Dasselbe Zitat mit
   `source=paper` wird angenommen. `catalog.discovery_details` erzeugt
   gerade den ersten Quellennamen; `build_spec` hat kein entsprechendes
   Dokument in seiner Quellenliste. Eine richtige Katalogangabe reicht
   deshalb noch nicht fuer einen richtigen effektiven Versuch.
3. **LaTeX-Kommentar gilt als Trainingsbeleg.** Batch 32 stammt aus
   `% \\textbf{Note that the following results are run under batch size 32 ...}`.
   Dieser Kommentar steht hinter normalem Fliesstext. Der Reader entfernt
   nur Kommentarzeilen mit `%` am Zeilenanfang. Die Spec-Verifikation nimmt
   diesen Wert und das uneindeutige 90er-Zitat deterministisch an, ohne
   einen Eintrag zu verwerfen. Das ist keine belegte Einstellung des
   ausgewaehlten Ergebnisses.
4. **Repo-Kontext passt nicht zum Forecast-Startpfad.** Die gelesenen Dateien
   enthalten `AD/FITS.py`, `AD/main.py` und `AD/solver_recon.py`, aber weder
   `models/FITS.py` noch `scripts/FITS/etth1_best.sh` und `run_longExp_F.py`.
   Die Vollstaendigkeitspruefung stellt vorhandene Abhaengigkeiten sicher,
   prueft aber damit noch nicht die Auswahl des richtigen Experimentpfads.

## Absturzdiagnose verdeckt die erste Ausnahme

Kandidat 1, Codehash `83f508ba9484`, endet im Live-Verlauf mit
`BrokenPipeError` aus `forecast_runner.py`; die Reparatur und die
Tavily-Fehlersuche bekommen dieses nachfolgende Broker-Symptom.

Der unveraenderte generierte Kandidat wurde lokal in
`repro-snapshot:gate-etth1` mit `--network none` und synthetischen
400x7-Train-/Val-Arrays im Smoke-Modus ausgefuehrt. Keine echten
Testmessungen, kein Tuning, keine Provideraufrufe. Bereits die erste
Trainingsiteration belegt die primaere Ausnahme:

```
RuntimeError: The size of tensor a (186) must match the size of tensor b (96)
at non-singleton dimension 1
```

Das Modell liefert Rekonstruktion plus Forecast (90+96 Schritte), der
Trainingscode vergleicht dieses gesamte Ergebnis mit 96 Zielschritten.
Auch Inferenz liefert im urspruenglichen Kandidaten das gesamte Ergebnis.
Erst der spaetere Code `b1d7c4d038db` schneidet in Training und Inferenz
die letzten 96 Schritte aus und wird anschliessend gemessen. Die ersten
beiden Reparaturversuche sehen im Protokoll nur den Broker-Traceback.

## Einordnung und naechste Reparatur

Erster Kandidat: Test-MSE 0.7042857621, Auswahl-Val-MSE 0.7269202064.
Reparierter Kandidat 1: Test-MSE 0.7029395600, Auswahl-Val-MSE 0.7273665563.
Diese Werte sind unter abweichendem Setup entstanden; sie duerfen nicht
als Nachweis gegen den veroeffentlichten FITS-Claim behandelt werden.
Ein korrektes Setting garantiert fuer den autonomen Nachbau trotzdem
keinen bestimmten Score.

Die naechste Reparatur muss allgemein wirken: experimentbezogene
Startskripte und Modellpfade lesen, einzelne Parameter an Setting-Belege
binden, Kommentare/Mehrfachauswahlen nicht erzwingen, belegte
Discovery-Herkunft bis zur effektiven Spec erhalten und die erste
Worker-Ausnahme in der Reparatur bewahren. Abnahme muss die komplette
Kette Quelle -> Detail -> Spec -> Plan/CONFIG pruefen, nicht allein das
Vorhandensein einer Zahl im Katalog. Keine Parameter nach Testscore waehlen.

Kuratiertes Ereignisartefakt:
`docs/audits/fits_live_setup_evidence_20261013.json`.

## Abschluss

Der Auftrag ist nach rund 812 Sekunden vollstaendig abgeschlossen, Run-ID
`disc_2307_03756--20261013-165528`. Alle drei ausgewaehlten Messungen verwenden
Codehash `947d5088ab2c`: 0.7042857621 / 0.7032250044 / 0.7032369177.
Mittel **0.7035825614**, Stichproben-Std **0.0006090188**. Produkturteil:
`nicht reproduziert`, `vorlaeufig`, Methode `nicht geprueft`.
Der Repo-Abgleich ist ausdruecklich `incomplete`, weil Dateien in diesem
gesonderten Vergleich abgeschnitten bzw. Befunde verworfen wurden.

Gemeldete Modellkosten: **0.2577 Dollar netto / 0.3067 Dollar brutto**.
Diese Kosten stammen aus dem vom Nutzer gestarteten Auftrag, nicht aus
zusaetzlichen Modellaufrufen dieser Untersuchung. Tavily-/Sandboxverbrauch
ist in dieser Zahl nicht als Dollarbetrag ausgewiesen. Die lokalen
Gegenproben brauchen keine externen Provider. Kein weiterer Live-Auftrag
gestartet, keine Produktfixes in diesem Verifikationsschritt vorgenommen.
