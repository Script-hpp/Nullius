# FITS: Code-Erzeugung und Fehlertexte nach Live-Lauf 35758c32

Grundlage: autonomer Live-Lauf `35758c32-51ae-4c01-85c4-9922cde52ce3` (Deployment `be1a643`, 0/3 Seeds,
kein Paperurteil). Die beiden Hauptursachen dieses Laufs (gekappte Dateiliste, unklarer Datenhelfer-Vertrag)
behebt bereits `30c67b0`. Dieser Nachtrag behebt drei weitere, im selben Ereignisprotokoll belegte Ursachen.

## 1. Temperaturstaffel erzeugte unbrauchbaren Code

Kandidat i startete bei `0.2 + 0.3 i`, jede Wiederholung nach fehlendem Code-Block lag 0.3 hoeher.
Kandidat 2 lief damit bei 0.8 / 1.1 / 1.4. Im Lauf:

- sechs `code_missing`: Antworten brechen mitten im Skript ab, ohne `finish_reason length`
  (die Kosten der Aufrufe liegen im Bereich vollstaendiger Antworten, kein Ausgabelimit erkennbar);
- bei 1.1 ein `SyntaxError` in einem f-String, bei 1.4 Wortsalat in mehreren Sprachen;
- Kandidat 2 lieferte nie Code. Kandidat 0 bei 0.2 hatte in beiden FITS-Laeufen keinen Abbruch.

Fix: Code-Antworten sind auf 0.7 gedeckelt (Kandidaten 0.2 / 0.5 / 0.7), Wiederholungen steigen um 0.15
bis zur Grenze. Damit der LLM-Cache an der Grenze nicht dreimal dieselbe Antwort liefert, traegt der
Hinweis ab der zweiten Wiederholung die Versuchsnummer.

## 2. Ende langer Trainingsausgaben ging auf Nebius verloren

Das Sandbox-SDK kappt stdout und stderr bei 65535 Bytes und behaelt den Anfang. Traceback,
`WorkerFailure`-Zusammenfassung des Brokers und faulthandler-Stacks stehen am Ende. Smoke- und
Trainingslauf schreiben jetzt in ein frisches `mktemp -d`-Verzeichnis; danach gehen nur die letzten
30000 Bytes je Strom zurueck, Exit-Code unveraendert. Nicht-ASCII-Bytes werden ersetzt, weil das SDK
strikt UTF-8 dekodiert und ein Schnitt mitten im Zeichen sonst eine Ausnahme ausloest.

## 3. Websuche fuer Fehler im eigenen Datenhelfer

Fehler, die in `nullius_io.py` ausgeloest werden, betreffen die Nutzung unseres Helfers. Die
Tavily-Fehlersuche lieferte dafuer pycbc-, xarray- und langchain-Treffer. Sie entfaellt jetzt fuer
diese Fehler; der Helfervertrag steht ohnehin im Prompt.

## Nicht geaendert

Keine Hyperparameter, kein Zielwert, keine Methode und keine Auswahlregel. Der Formfehler 186 vs. 96
(Rekonstruktion plus Forecast) bleibt Sache des Agenten; mit `train_mode 1` ist das Ausschneiden der
letzten 96 Schritte laut `exp/exp_main_F.py` korrekt, und der Autorencode liegt nach `30c67b0` im Prompt.
Ob der autonome Nachbau jetzt gelingt, zeigt nur ein neuer Live-Lauf.

## Gegenprobe: woher kam 0.7036 im Lauf ee039f0f?

Der Autorenkern `models/FITS.py` (Commit `d040bb0`) lief lokal im selben kausalen Vertrag, netzlos,
mit genau den Werten des Live-Plans: L90, `cut_freq` 3, LR 0.001, Batch 32, 10 Epochen, Patience 3.
Seeds 0/1/2: 0.70405 / 0.70345 / 0.70321, Mittel **0.70357 +/- 0.00043**. Der Live-Lauf hatte
**0.70358 +/- 0.00061**. Mit dem veroeffentlichten Preset liefert derselbe Aufbau 0.37205.

Damit ist belegt: Harness und Vertrag messen richtig, der Agentencode von ee039f0f setzte FITS
funktional wie der Autorenkern um, und die gesamte Abweichung kommt aus dem falschen Setting
(welcher Einzelwert dominiert, ist nicht ablatiert; plausibel ist `cut_freq` 3: bei 90 Schritten liegt die Tagesperiode bei Frequenzindex 3.75 und wird
abgeschnitten). Das stuetzt die Setting-Fixes aus `be1a643`/`30c67b0`; es ist kein Nachweis eines
autonomen Erfolgs. Skript `scripts/review/fits_live_setting_probe.py`, Beleg
`docs/audits/fits_live_setting_probe_20261014.json`. Keine Provideraufrufe.
