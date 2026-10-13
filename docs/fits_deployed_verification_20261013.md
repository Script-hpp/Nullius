# FITS: echter Deployment-Test und Nachfix

Der autorisierte neue autonome Schnelllauf nach Deployment von `be1a643`
ist abgeschlossen: **0/3 gueltige Seeds, kein Paperurteil**.
[Live-Test](https://nullius.akca.dev/#/live/35758c32-51ae-4c01-85c4-9922cde52ce3),
kuratiert in [fits_deployed_check_20261013.json](audits/fits_deployed_check_20261013.json).
Die Dauer war etwa 10:06 Minuten. Es wurde kein zweiter bezahlter Job gestartet.

Die Modellkosten betragen 0.27836 USD netto; bei der Standard-MwSt. 19% etwa
0.33125 USD brutto. Fuer diese Zahl gilt der hoechste kumulative Kostenwert im
Stream. Die Summe einzelner sichtbarer Antwortkosten ist kleiner, weil auch
fehlgeschlagene Modellantworten bezahlt wurden. Fuenf Tavily-Aufrufe sind
gesondert protokolliert, ihr Dollarpreis wird dort nicht ausgewiesen.

## Was im Deployment funktioniert

- Health und Live-Status erreichbar, FITS gestartet, Stream bis Ende beobachtet.
- Ziel 0.372 an v3/Hash9022 und `tab:FITSett` gebunden, Spalte geprueft.
- Belegte Discovery-Eingabelänge360 erreicht die Spec; kein falscher90er-Wert.
- Worker-Fehler zeigen jetzt den eigentlichen Helfer-/Trainingsfehler statt
  eines blossen nachfolgenden BrokenPipeError.
- UI zeigt den Lauf und anschliessend STOPPED/0 Seeds/kein Paperurteil.
  Ein erfolgloser Agentenaufbau wird damit nicht als Gegenbeweis zum Paper dargestellt.

## Neue Transportluecke, lokal und unabhaengig reproduziert

Der gepinnte FITS-Git-Baum `d040bb015b6299da26d879b90dd19c80fb72c160`
enthaelt **1410 Dateipfade / 83540 Bytes**. Nebius kappt stdout standardmaessig
nach **65535 Bytes**. `models/FITS.py` beginnt bei Byte82903, das finale
Startskript bei83168, `run_longExp_F.py` bei83078. Diese Dateien liegen also
hinter der Kappung. Der abgeschnittene letzte Eintrag `lo` ist sogar ein
unvollstaendiger Pfad.

Der echte Live-Lauf liest deshalb wieder AD/FITS.py und AD/main.py. Die neue
deterministische Forecast-Auswahl bekommt die entscheidenden Dateinamen nicht
zu sehen. Die Live-Spec hat nur Batch128 und Eingabelänge360, statt der neun
belegten Laufzeitwerte. Der Plan nimmt wieder LR0.001, Epochen10 und cut_freq15
als unsicher an. Die vorige lokale Referenzpruefung lieferte eine vollstaendige
Dateiliste und hatte diese Provider-Uebertragungsgrenze nicht emuliert.

Nachfix in `repo_source.verified_listing`:

1. `git ls-files -z` bzw. Unterordner `git ls-tree -r -l -z` schreiben in eine
   separate Datei. Nur die kleine Byte-/NUL-Anzahl-/SHA256-Metadatenantwort
   verwendet stdout.
2. Vor dem Transfer gilt eine explizite Grenze von1MB. Danach muessen Bytezahl,
   NUL-Anzahl, Abschluss-NUL und SHA256 exakt passen; UTF8 wird strikt gelesen.
   Unverifizierte Listen stoppen vor Dateiauswahl und vor einem Experiment.
3. Der etablierte Datei-Transfer liest die komplette Liste. Unterordnergrenzen
   rechnen nun auch die zuvor gekappten Eintraege mit. Pfade mit Leerzeichen,
   Umlaut, Quote, Tab oder Newline bleiben durch NUL-Trennung exakt.
4. `setup_details.repo.n_tracked` macht die Zahl der gelesenen Dateipfade sichtbar.

## Zweite Luecke: Agentenvertrag fuer den Datenhelfer

Acht Live-Ausfuehrungen scheitern, darunter mehrfach
`forecast_data('train')`, Tupel-Unpacking eines Dictionaries, numerischer
Dictionary-Zugriff und eine Trainings-/Zielachsenabweichung. Der Agentenvertrag
beschrieb bisher die gelieferten Arrays, aber nicht die Rueckgabeform und Keys.
Die echte Helfer-API selbst gibt bereits korrekt ein Dictionary zurueck.

Der Forecast-Agentenvertrag nennt jetzt die genaue Signatur, Integer-Laenge,
Dictionary-Keys und Arrayformen. Er enthaelt ein vollstaendiges Beispiel fuer
Train-/Val-Eingaben und echte Ziele plus optionale Kalenderarrays. Das Beispiel
wird im Test aus dem wirklich an den Agenten gesendeten Vertrag gelesen und
gegen den echten Helfer ausgefuehrt, jeweils mit und ohne Kalender. Kein
Benchmark-Zielwert und keine Methode wurden in dieses Beispiel eingebaut.

## Validierung des Nachfixes

- Acht Transportregressionen, einschliesslich derselben64KiB-Kappung und
  ungueltiger Laenge/Hash/Anzahl/Metadaten: bestanden. Vor dem Fix waren die
  ersten sieben Proben rot.
- Zwei echte Helper-Beispieltests: vorher rot, danach gruen.
- Naechste Settings-/Repo-/Vertragsgates:162 passed /3 skipped.
- Unabhaengiges Opus-Review: Ursache bestaetigt, Delta ohne Blocker.
  [Bericht](opus_repo_listing_cap_review_20261013.md). Echte Shellbefehle mit
  Kappung jedes stdout; auch voller Unterordner und Sonderzeichen getestet.
- Neues Produktionsimage `nullius:listing-20261013` gebaut. Darin netzlos:
  echte Manifest-Shellbefehle, jeder stdout auf65535 begrenzt, voller gepinnter
  Autorenbaum readonly. Die Git-Dateiliste stammt aus dem lokalen Checkout;
  kein Klon, keine Ausfuehrung von Autorenprogrammen, kein Modellaufruf.
  **1410/1410 Pfade**, kanonisches Modell/finales Startskript/Python-Einstieg,
  AD verworfen, wieder alle neun korrekten Pflichtwerte:
  L360/Batch128/LR0.0005/Epochen50/Patience20/cut106/7Kanaele/H96/train_mode1.
  [Produktionsbeleg](audits/fits_listing_production_20261013.json).

Abschliessender Backend-Gate: **1577 passed /10 skipped /8 deselected /
5 xfailed**, keine Fehler (323.76s). Neues Produktionsimage startet netzlos;
`/api/health` und `/api/status` liefern HTTP200. Kein erneutes Parameter-Tuning und keine Veraenderung alter
Messungen. Die vorherige feste menschliche FITS-Referenz bleibt erfolgreich;
der autonome Neuaufbau ist durch diesen fehlgeschlagenen Deployment-Test
ausdruecklich **nicht nachgewiesen**. Nach Deployment des Nachfixes muss ein
neuer autonomer Lauf diese offene Aussage pruefen.

Bekannte separate Grenzen: allgemeines Training-stderr kann weiterhin durch
das SDK gekappt werden; der LLM-Dateiauswahl-Prompt zeigt die ersten600 Pfade,
waehrend die deterministische Forecast-Auswahl die volle verifizierte Liste
verwendet. Beide Punkte sind im unabhaengigen Review benannt und werden hier
nicht als bereits behoben ausgegeben.
