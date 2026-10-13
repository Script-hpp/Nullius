# Lokale SARAF-Vertragspruefung nach dem allgemeinen Audit

Stand: **2026-10-13**. Nutzerfreigabe fuer lokale Arbeit. Keine Modell-, Tavily-, Nebius- oder
Produktionsauftraege. Die Gegenproben verwenden vorhandene Autorendateien und ETTh1-Daten; Autorencode
laeuft ausschliesslich in temporaeren Docker-Containern mit `--network none`, ohne Host-Mounts.

**Ergebnis:** Eine weitere fehlende Eingabe wurde belegt und behoben: Kalendermerkmale fuer SARAF-
Retrieval. Der Originalkern besteht danach einen lokalen Smoke-Test unter dem kausalen Vertrag.
Dies ist **kein vollstaendiger Reproduktionslauf und kein Nachweis fuer MSE 0.371**.

## Befund E24: Kalenderdaten fehlten im Forecast-Vertrag

Der gepinnte Autorenloader erzeugt im standardmaessigen `embed=fixed`-Pfad fuenf Merkmale:
Jahr, Monat, Tag, Wochentag und Stunde; bei ETTm zusaetzlich Minute. Das Autoren-Retrieval verwendet
sie mit `time_aware_weight=0.5` zur Gewichtung seiner Aehnlichkeiten. Der bisherige Nullius-Datensatz
enthielt ausschliesslich `train`, `val` und `test_inputs`. Der Worker bekam nur numerische Historien.
Kalendermerkmale mussten somit zusaetzlich rekonstruiert oder weggelassen werden.

Eine isolierte Probe auf den vorhandenen echten ETTh1-Daten ruft die urspruenglichen Korrelations-
und Kalenderbonusfunktionen auf. Auf drei Trainingskandidaten veraendert Weglassen der Kalenderdaten
die Zwischen-Aehnlichkeit maximal um **0.6314319372**. Das ist ein Mechanismusbeleg, keine Forecast-MSE.
Die abschliessende MMR-Aggregation wurde in dieser ersten Probe durch eine Ergebniserfassung ersetzt;
es gab weder Training noch Scoring. Ungenutzte pandas-/tqdm-Imports wurden nur im Container ersetzt.

Beleg: [Probe](audits/saraf_calendar_contract_probe_20261013.py),
[Ergebnis](audits/saraf_calendar_contract_probe_20261013.json). Das Skript verwendet explizit
den alten Produktstand `07f53b614084bd6a760545756aa1f3e4d11e952b` fuer seine Vertragsdateien.

## Fix und Kompatibilitaet

- ETT-Produktvorbereitung liest die Kalendermerkmale aus der echten CSV, auf denselben Zeilen wie
  die Messwerte. Der bisherige Benchmark-/Experimentvertrag ohne Produktflag bleibt erhalten;
  seine numerischen Arrays sind bitweise identisch.
- Der Fit-Prozess bekommt `train_calendar` und nur den sichtbaren `val_calendar`-Praefix. Die
  gesamte Liste der Testhistorien und ihrer Kalenderfenster bleibt beim privaten Broker.
- `nullius_io.forecast_data(length)` liefert zu den numerischen Eingaben passend indizierte
  Kalenderfenster. Kein Trainings-/Auswahlziel wird dadurch zugaenglich.
- Der Broker liefert die Kalenderdaten desselben vergangenen 720-Zeilen-Fensters zusammen mit der
  Historie. Methoden mit Zeitmerkmalen koennen `predictor(history, *, calendar)` definieren;
  bisherige `predictor(history)`-Funktionen funktionieren weiter. Allgemeine `**kwargs`-Wrapper
  gelten nicht automatisch als Kalender-Unterstuetzung. Fehlende erforderliche Metadaten stoppen.
- Wiederholungsproben verwenden dasselbe Paar aus Messwerten und Kalenderdaten. Geschuetzte CSVs,
  UID-Abgabe und die Trennung der Bewertung bleiben erhalten.
- Neue ETT-Zeugnisse nennen **`causal_stream_v2`** und die Kalender-Spalten. Alte Zeugnisse behalten
  ihre aufgezeichnete Version. Der veraenderte Setup-Quelltext/Flag erzeugt einen neuen Snapshot-
  Schluessel; vorhandene Snapshots werden nicht umgeschrieben.

Rueckweg: bisherige einargumentige Forecaster bleiben kompatibel, Kalenderargument ist optional.
Historische Datendateien und Laeufe wurden nicht migriert. Die bisherige Auswahlaufteilung
`withheld_validation_v1` bleibt erhalten.

## Lokaler Autorenkern-Smoke-Test

Aufruf: `uv run python scripts/review/saraf_contract_smoke.py`.
Benutzt den lokalen Cache `repro-snapshot:gate-etth1`, CPU (vier Threads), Seed 0 und die Autorendateien
`models/SARAF.py` und `layers/Retrieval.py`. Deren SHA-256 und der Datensatzhash stehen im Ergebnis.
Ein von Hand geschriebener Diagnoseadapter verwendet den Originalkern, reale Kalenderdaten,
das urspruengliche Retrieval mit Ueberlappungsmaske, einen Adam-Trainingsbatch mit acht Beispielen
und dann je acht Validierungs-/Testfenster unter dem neuen Broker. Kein LLM schreibt diesen Adapter.

| Gegenprobe | Ergebnis und Einordnung |
| --- | --- |
| Erste Diagnoseumgebung | Stopp wegen fehlendem `__spec__` im Stub des ungenutzten pandas-Imports beim PyTorch-Optimizer. Diagnose-Setupfehler, kein Produktfehler; [Artefakt](audits/saraf_contract_smoke_setup_failure_20261013.json). |
| Original-Retrieval pro Wiederholung erneut berechnet | Wiederholungspruefung stoppt korrekt. `mmr_select` verwendet `torch.multinomial`; erneutes Sampling liefert andere Werte. [Artefakt](audits/saraf_contract_smoke_uncached_20261013.json). |
| Retrieval wie im Original einmal berechnet und gespeichert | **Bestanden, 4.57 Sekunden**. [Ergebnis](audits/saraf_contract_smoke_20261013.json). |

Der Originalcode berechnet Retrieval einmal je Fenster und speichert es vor den Modellaufrufen.
Der kausale Adapter bildet dies durch einen Cache je bereits gelieferter Historie und Kalender ab:
keine kuenftigen Fenster vorab lesen, keine erneute Zufallsauswahl fuer dasselbe Fenster. Die
Wiederholungspruefung wurde weder abgeschaltet noch toleranter gestellt. Gewichte bleiben nach dem
Trainingsbatch unveraendert. Das ist eine lokale Eingangs-/Ausfuehrungspruefung; weder Trainings-
epochen noch Paper-Zielwert wurden anhand eines Testscore angepasst.

## Regressionen und Abnahme

| Pruefung | Ergebnis |
| --- | --- |
| Neue Kalender-Regressionen vor dem Fix | Vier Tests reproduzierten fehlende Felder/Projektion/Task-Aktivierung. |
| `tests/test_forecast_calendar.py` | Echte Datumsfelder fuer Stunden/15-Minuten, exakt gleiche numerische Legacy-Arrays, passende Fenster, verborgene Auswahl, optionale Callback-Kompatibilitaet, partielle Metadaten abgelehnt. |
| Kalender-/Selection-/Trusted-Evaluation mit `NULLIUS_DOCKER_TESTS=1` | 25 passed; echte Kalender-Weitergabe, Legacy-Forecaster und Privilegien-/CSV-Schutz. |
| Fokussierte ETT-/IO-/Identitaets-/Produktmodus-/Kalender-Suite | 61 passed, 1 skipped. |
| Komplette Backend-Suite ohne Provider-Integration | 1462 passed, 9 skipped, 8 deselected, 5 xfailed; 326.23 Sekunden. |
| `docker build -t nullius:calendar-20261013 .` | bestanden. |

Frontendquellen bleiben gegenueber dem zuvor geprueften Stand unveraendert (153 Tests/Build bestanden).
Historische Benchmark-/Katalogwerte bleiben unveraendert. Nebius-UID-/Capability-Preflight und ein neuer
autonomer SARAF-Lauf mit vollstaendigem Training sind weiterhin offen. Der Smoke-Test bestaetigt keine
allgemeine Methodentreue und keine Erfolgsquote der KI-Codegenerierung.

## Nachfolgend freigegebener lokaler Volltest

Der Nutzer bat danach ausdruecklich um Ausfuehrung des Volltests. Aufruf:
`uv run python scripts/review/saraf_contract_full.py`, **Seeds 0/1/2 vorher festgelegt**.
Produktstand `db5e571`. Der von Hand geschriebene Adapter um denselben unveraenderten SARAF-/Retrieval-
Kern trainiert jetzt zehn Epochen mit den belegten Einstellungen: History 720, Horizont 96, sieben
Kanaele, Batch 8, Adam lr 0.001, originaler type1-Schedule, n_period 1, topm 20 und Zeitgewicht 0.5.
Keine Dropout-Schicht hinzugefuegt. Bester unabhaengiger Checkpoint ausschliesslich aus sichtbarer Val.
Die Inferenz laeuft chronologisch mit dem beschriebenen Retrieval-Cache. Die normale Code- und
Checkpointpruefung bleibt aktiv; nichts wurde anhand von Testwerten repariert oder nachjustiert.

**Drei komplette Runs, alle innerhalb der vorherigen Toleranz von +/-0.01:**

| Seed | Test-MSE | Test-MAE | Bester Val-Checkpoint | Gesamtlaufzeit |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.3711257752 | 0.3979054375 | Epoche 4 | 92.55 s |
| 1 | 0.3685850588 | 0.3960209257 | Epoche 6 | 110.45 s |
| 2 | 0.3690020300 | 0.3956394735 | Epoche 8 | 110.92 s |

Mittel **MSE 0.3695709547**, Stichproben-Std **0.0013625588**. Paper-Ziel **0.371**.
Alle drei Runs trainierten auf 7825 Fenstern und fuehrten alle zehn Epochen aus.
Die vollstaendigen 2785 Testfenster wurden vom normalen Harness in einem frischen, vom Trainings-
Branch unabhaengigen Container bewertet. Nur sein Dateidownload wurde auf die vorhandene, mit
SHA-256 verifizierte CSV umgeleitet; Metrik-/Zielberechnung und Pruefung der Vorhersagen bleiben original.
Der Trainingsbranch erhielt keine Rohdatei fuer die Bewertung. Repeat-MSE **1.2943705948**, Zero-MSE
**1.1099284101** entsprechen den vorher belegten Dataset-Baselines.

Belege: [Driver](../scripts/review/saraf_contract_full.py),
[ausgefuehrter Adapter](audits/saraf_contract_full_20261013.adapter.py),
[drei Runs einschliesslich Epochenprotokollen](audits/saraf_contract_full_20261013.json).
Autorendateien/Dataset und Adapter haben aufgezeichnete SHA-256-Hashes.

**Was damit bewiesen ist:** Der neue lokale Daten-/Inferenz-/Harness-Pfad kann mit dem Autorenkern
den Paper-Zielwert erreichen. Die hohe alte MSE ist kein notwendiges Ergebnis des Tasks oder des
Harness. Das spricht fuer die zuvor belegten Implementierungs-/Eingabefehler als entscheidende Ursachen;
es isoliert nicht den Beitrag jedes einzelnen Fixes.

**Protokollunterschiede bleiben:** Nullius verwendet 93 sichtbare, taeglich gesampelte Val-Fenster fuer
Checkpoints und 21 verborgene Auswahlfenster; Autorenvalidierung ist dicht und umfasst die volle Val.
Seeds 0/1/2 entsprechen Nullius, das Autorenskript nennt 2021/0/42. Online-Retrieval nach dem Training
veraendert die Reihenfolge der Zufallsziehungen gegenueber kompletter Testvorberechnung vor dem Training.
Der Erfolg ist deshalb eine numerische lokale Referenz unter dem neuen Nullius-Vertrag, keine bitweise
Wiederholung der Autorenlaeufe. Der KI-generierte Nachbau und Nebius wurden hier weiterhin nicht ausgefuehrt.

Eine vorherige Volltest-Runde stoppte alle drei Seeds **vor einer Testmessung**: NumPy interpretierte
einen einelementigen Torch-Index als Skalar und entfernte die Batch-Achse im lokalen Adapter. Der Adapter
normalisiert Indizes nun ausdruecklich zu einem eindimensionalen Array. Diese Formkorrektur aendert keine
Hyperparameter und ist mit vier Batch-Regressionsfaellen abgesichert; der fuenfte Test prueft Syntax und
die bestehenden Produktguards. [Unveraenderte Fehlerartefakte](audits/saraf_contract_full_batch_failure_20261013.json)
und [damaliger Adapter](audits/saraf_contract_full_batch_failure_20261013.adapter.py) bleiben erhalten.
Vor dem Start wurde zudem ein unnoetiger dynamischer Import im Diagnoseadapter durch die normale
Codepruefung abgelehnt und entfernt; der Produktguard wurde nicht abgeschwaecht.

Abnahme des Drivers: **5 passed** (`uv run pytest -q tests/test_saraf_local_reference.py`); Produktquellen
gegenueber der oben mit 1462 Backend-Tests geprueften Kalenderkorrektur unveraendert. Kein Deployment,
kein neuer Katalog-/Benchmarkeintrag und kein automatisches Paperurteil im Produkt erzeugt.
