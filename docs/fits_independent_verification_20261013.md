# Gegenprobe auf einem anderen Paper: FITS

Stand **2026-10-13**, freigegeben nach dem lokalen SARAF-Erfolg. Die zuvor geprueften Fixes und
SARAF-Belege wurden bereits als `4e1093d` auf `origin/main` gepusht. Fuer diese Gegenprobe wurden
keine Produktquellen, Katalogdaten, Benchmarkwerte oder Hyperparameter nach einer Testmessung geaendert.

## Auswahl und vorher festgelegter Versuch

FITS verwendet einen anderen Modellkern als SARAF: komplexwertige Frequenzinterpolation statt
Retrieval. Gewaehlt wurde ein im Projekt bisher nicht als lokale Referenz umgesetztes Modell,
bevor neue Testwerte vorlagen. Das Paper blieb auch bei einem moeglichen Fehlschlag festgelegt.

Quellen: [Paper v3, Tabelle 1](https://arxiv.org/html/2307.03756v3#S4.T1),
[offizielles Repository](https://github.com/VEWOXIC/FITS/tree/d040bb015b6299da26d879b90dd19c80fb72c160).
Die README bezeichnet `*_best`-Skripte als die optimalen Parameter; Skripte ohne dieses Suffix
sind Ablationen/Gridsearch. Wir uebernehmen das veroeffentlichte ETTh1-Best-Preset, ohne selbst
nach einer Testmetrik zu suchen.

| Einstellung | Festlegung vor Messung |
| --- | --- |
| Dataset/Claim | ETTh1, Horizont 96, MSE **0.372**, Toleranz **+/-0.01** |
| Autorencode | Commit `d040bb015b6299da26d879b90dd19c80fb72c160` |
| Look-back | **360**, aus `scripts/FITS/etth1_best.sh` |
| H_order/base_T/cut_freq | **6 / 24 / 106**; `(360 // 24 + 1) * 6 + 10`, wie `run_longExp_F.py` |
| Training | `train_mode=1`, Forecast-Verlust, Adam, lr **0.0005** |
| Batch | Skriptargument **64**, tatsaechlich **128**: CLI verdoppelt ohne Augmentation |
| Epochen/Schedule | maximal **50**, Patience **20**, `type3`: Faktor **0.95** pro Update nach Epochende |
| Modell | Original `models/FITS.py`, `individual=False`; Original-Import `models/NLinear.py` mitkopiert |
| Seeds | **0, 1, 2**, alle vor Beginn festgelegt |
| Auswertung | unveraenderter Nullius-Harness, alle **2785** Testfenster, frischer Bewertungsbranch |

Konfiguration, Quellhashes und Adapter wurden vom Driver vor dem ersten Trainings-/Testresultat
geschrieben. SHA-256 des ausgefuehrten Adapters und der Autorendateien stehen im Ergebnisartefakt.
Es gab keine Modell-, Tavily-, Nebius- oder Produktionsjobs. Oeffentliche Paper-/Repo-Quellen wurden
gelesen und gepinnt; die numerischen Laeufe fanden ausschliesslich in netzlosen Docker-Containern statt.

## Ergebnis

| Seed | Test-MSE | Test-MAE | Bester Checkpoint | Epochen | Gesamtlaufzeit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3720515430 | 0.3944735398 | 48 | 50 | 36.03 s |
| 1 | 0.3719971343 | 0.3944846314 | 49 | 50 | 36.64 s |
| 2 | 0.3720890698 | 0.3945426528 | 43 | 50 | 37.39 s |

Mittlere **MSE 0.3720459157**, Stichproben-Std **0.0000462254**. Alle drei Seeds und der Mittelwert
liegen innerhalb der vorher festgelegten Toleranz zum Paperwert **0.372**.

Belege: [Driver](../scripts/review/fits_independent_reference.py),
[ausgefuehrter Adapter](audits/fits_independent_20261013.adapter.py),
[drei Runs samt Epochenprotokollen/Hashes](audits/fits_independent_20261013.json).
Die normale Code-/Checkpointpruefung, Rechteabgabe, kausale Inferenz, Wiederholungsproben,
Datenidentitaet und getrennte Bewertung blieben aktiv. Der Forecaster verwendet nur `predict(history)`;
damit ist die Kompatibilitaet fuer Modelle ohne Kalenderargument auch mit einem weiteren echten
Modellkern geprueft. Beste Checkpoints wurden ausschliesslich nach sichtbarer Val gewaehlt.

## Einordnung und Grenzen

Diese positive Gegenprobe spricht gegen einen reinen SARAF-Spezialfall des reparierten lokalen
Verfahrens. Das FITS-Modell und seine veroeffentlichten Einstellungen wurden nicht an den erhaltenen
Testscore angepasst. Es gab keinen nachtraeglichen Wechsel des Papers und keinen Testscore-Gridsearch.

Dies ist eine **lokale Referenz mit menschlichem Adapter um den Autorenkern**, kein autonomer
KI-Nachbau und keine Nebius-Abnahme. Auch das Dataset bleibt ETTh1; damit ist noch keine allgemeine
Uebertragbarkeit auf andere Datensaetze belegt. Die 20-%-Trennung der Validierung bleibt neu:
FITS-Checkpoints nutzen 2209 dichte sichtbare Val-Fenster mit dem originalen `drop_last` im Val-/Train-
Loader; die 21 verborgenen Auswahlfenster werden getrennt bewertet. Die Testauswertung verwirft
keine Fenster. Die publizierte Paperzahl entstand mit voller Validierung und anderen Seeds;
wir uebernehmen das Best-Preset, wiederholen nicht die gesamte Autoren-Gridsearch.

Die FLOP-Profilierung des Autoren-Trainers wurde nicht ausgefuehrt; deren Zufallsverbrauch kann die
Trainingsreihenfolge veraendern. Die Modelldateien selbst wurden unveraendert ausgefuehrt. Der
Dateidownload des Auswertungsharness liest ausschliesslich die vorhandene, SHA-256-gepruefte CSV
im sauberen Bewertungsbranch. Die vorherigen Projektwerte werden nicht ersetzt oder umetikettiert.

Abnahme: sechs fokussierte FITS-/SARAF-Driver-Tests bestanden; alle drei echten FITS-Laeufe erfolgreich;
`git diff --check` bestanden. Keine Aenderungen in Produktquellen oder `benchmark/`/`web/public/`.
