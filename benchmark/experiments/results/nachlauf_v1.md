# Modus „Nachlauf" v1: Ergebnis der Prüfung an vier neuen Einheiten

Vorab-Festlegung: `../nachlauf_v1.md` (final, mit Nachtrag vom 2026-10-02/03). Lauf in der Nacht zum 2026-10-03, 22:17 bis 22:30 UTC, Token Factory (Nemotron 3 Super), Nebius-Sandbox, LLM-Cache aus, kein Tavily. Code-Stand `751be6b` (`code_commit` im Store `7c6ddbf`, danach nur Dokumente). Store: `nachlauf_v1_store.jsonl`, Ausgaben je Lauf: `nachlauf_v1_{sparsetsf,fits,patchtst,nlinear}.txt`, Dossiers mit Roh-Antworten des Modells und Prompts unter `runs/nachlauf/<paper>/<zeitstempel>/` (nicht in Git).

## Ergebnis

**Erfolgsregel Punkt 1 ist verfehlt: 0 von 4 neuen Einheiten haben einen gültigen Nachlauf** (nötig: mindestens 2 von 4). In keinem Lauf lief ein vollständiges Training; es gibt keinen Messwert und kein Urteil über ein Paper.

| Einheit | Ausgang | LLM-Aufrufe | Kosten brutto | Was geschah |
|---|---|---|---|---|
| SparseTSF | nicht reproduzierbar (Setup), Training gescheitert | 5 | 0,057 $ | Dokumentierter Befehl richtig gewählt (`scripts/SparseTSF/linear/etth1.sh:15`). Das Training stürzt beim Laden der Daten ab: `TypeError: Value after * must be an iterable, not int` in `pandas` (Snapshot: pandas 3.0.6). Drei Reparaturen änderten nur Schalter des Befehls und halfen nicht. |
| FITS | nicht reproduzierbar (Setup), kein Befehl | 2 | 0,021 $ | Das Programm fand 137 dokumentierte Befehle, zeigte dem Modell 30. Die gezeigten stammen aus anderen Experimenten des Repos (Anomalie-Erkennung). Das Modell wählte keinen. |
| PatchTST/42 | nicht reproduzierbar (Setup), kein Befehl | 2 | 0,021 $ | 461 Befehle gefunden, 30 gezeigt, zuerst die Vergleichsmodelle (Autoformer, FEDformer, Informer). Das Modell wählte keinen. |
| NLinear | nicht reproduzierbar (Setup), kein Befehl | 2 | 0,022 $ | 757 Befehle gefunden, 30 gezeigt, zuerst eine Skriptdatei mit anderem Eingabefenster und Schalter `--individual`. Das Modell wählte keinen. |
| **Summe** | **0 von 4 gültig** | 11 | **0,12 $** | |

Kosten aus `logs/costs.jsonl` (11 Aufrufe im Zeitraum, 0,121 $ brutto), gleich der Summe im Store. Kostenstopp 3 $ nicht erreicht. Kein Infrastruktur-Abbruch, kein Lauf wiederholt.

## Erfolgsregel

| Punkt | Ergebnis |
|---|---|
| 1. Gültiger Nachlauf bei mindestens 2 der 4 neuen Einheiten (mit Startbefehl aus einer wörtlichen Repo-Zeile, Nachtrag) | **verfehlt (0 von 4)** |
| 2. Drei Negativkontrollen abgelehnt | erfüllt (`nachlauf_m1_m2.md`) |
| 3. Kein von Hand gefundener Verstoß in gültig gewerteten Läufen | nicht anwendbar: es gibt keinen gültigen Lauf |

**Folge nach der Vorab-Festlegung:** Der Modus bleibt als „funktioniert für die beiden Entwicklungs-Einheiten" im Produkt, und README und Zeugnis sagen das so. Er wird nicht als allgemein funktionierend dargestellt. Erlaubt ist nach einer Code-Änderung **genau ein** weiterer Lauf auf denselben vier Einheiten, der dann als „zweiter Versuch" berichtet wird; ob er stattfindet, entscheidet der Nutzer.

## Ursachen (aus Dossiers, Prompts und Roh-Antworten gelesen; keine Gegenprobe)

1. **Auswahl der Befehls-Kandidaten (3 von 4 Läufen).** Die Vorauswahl auf 30 Kandidaten ordnet in großen Repos falsch: Sie zeigt Befehle anderer Modelle oder anderer Experimente zuerst, der passende Befehl ist nicht unter den gezeigten. An den beiden Entwicklungs-Repos fiel das nicht auf (dort stand der richtige Befehl auf Platz 1). Das Modell hat sich in allen drei Fällen regelkonform verhalten: kein Kandidat passt, also `null`, kein erfundener Befehl.
2. **Umgebung (1 von 4).** Der Snapshot nutzt aktuelle Paketversionen. Der SparseTSF-Code ruft `pandas` auf eine Weise auf, die mit pandas 3 nicht mehr geht. Die Reparatur darf nur Pakete nachinstallieren, die in den Abhängigkeitsdateien des Repos stehen, und kennt keinen Weg, eine **ältere Version** eines vorhandenen Pakets zu wählen.
3. **Reparatur ändert die Bedeutung des Befehls (Beobachtung bei SparseTSF).** Nach dem Absturz stellte das Modell `--features M` auf `--features S` und `--enc_in 7` auf `1` um. Das wäre ein anderer Versuch (eine statt sieben Messreihen). P1 hätte die Abweichung gelistet, aber nicht abgelehnt. Wäre das Training damit gelaufen, hätte der Lauf nach der Regel als gültig gezählt. **Das ist eine Lücke in P1**, hier ohne Folge, weil auch dieser Befehl abstürzte.
4. Das Fehlerende von 1500 Zeichen zeigte beim pandas-Fehler nur die Bibliothek, nicht die Zeile im Repo; das erschwert die Reparatur.

## Einordnung

- Der Agenten-Teil wurde an zwei Repos entwickelt, in fünf Nachbesserungsrunden (siehe Nachtrag der Vorab-Festlegung). Die Prüfung zeigt, dass er darüber hinaus **nicht** trägt. Genau dafür war sie da.
- Es gab kein falsches Ergebnis: kein erfundener Befehl, kein Urteil ohne Messung. Die vier Läufe enden als „nicht reproduzierbar (Setup)" mit Grund.
- Über die vier Papers sagt die Prüfung nichts. Kein Autorencode wurde vollständig ausgeführt.
- Vier Einheiten, eine Generierung: eine Stichprobe.

## Entwicklungs-Einheiten (zählen nicht für die Erfolgsregel)

Mit dem eingefrorenen Stand, je fünf Seeds, Token Factory (Store `nachlauf_m3_dev_store.jsonl`, letzte zwei Zeilen):

| Einheit | Startbefehl | Autorencode (Repo-Seed) | Harness, alle 2785 Fenster | Urteil | LLM-Aufrufe | Kosten brutto |
|---|---|---|---|---|---|---|
| RLinear | `script.md:3`, nach Absturz vom Modell korrigiert (`--model Linear` → `RLinear`), als Abweichung gelistet | 0,3660 auf 2688 Fenstern (`drop_last`, Batch 128) | 0,3708 ± 0,0013 | reproduziert | 4 | 0,019 $ |
| DLinear | `scripts/EXP-LongForecasting/Linear/etth1.sh`, dokumentiert | 0,3841 | 0,3781 ± 0,0082 | reproduziert | 3 | 0,026 $ |

P2 und P3 bei allen zehn Seed-Läufen bestanden, Adapter jeweils im ersten Versuch. Die Werte sind identisch mit der Handarbeit aus dem Vorversuch. Die gesamte Entwicklung an diesen zwei Einheiten kostete 0,31 $ brutto (9 Läufe im Store, darunter ein Lauf mit erfundenem Befehl und ein gescheiterter; dazu ein Absturz vor dem ersten Store-Eintrag).

## Offene Punkte für einen zweiten Versuch (nur falls der Nutzer ihn will)

- Kandidaten nach Passung zum Claim ordnen (Modellname, Datensatz, Horizont nach Auflösung der Skript-Variablen), nach Dateien gruppieren, und dem Modell erlauben, weitere Kandidaten anzufordern.
- Versionen: Bei einem Absturz in einer Bibliothek eine ältere Version erlauben, wenn die Abhängigkeitsdatei des Repos sie nennt; sonst als Setup-Urteil mit klarem Grund enden.
- P1 verschärfen: Schalter, die Datensatz, Aufgabe oder Zahl der Kanäle ändern, dürfen in der Reparatur nicht geändert werden.
- Diese Änderungen wären durch die Fehler an den vier neuen Einheiten veranlasst. Ein zweiter Versuch ist deshalb kein Test an unberührten Einheiten mehr und muss so beschriftet werden.

---

# Zweiter Versuch (2026-10-03, 01:16 bis 01:30 UTC)

Vorab festgelegt im Nachtrag vom 2026-10-03 (`../nachlauf_v1.md`), vor jeder Code-Änderung; Code `4388a76` (Änderungen `7296b36`), im Nachtrag vor dem Lauf eingetragen. Bedingung vorher erfüllt: beide Entwicklungs-Einheiten mit dem neuen Code gültig (RLinear 0,3708 ± 0,0013, DLinear 0,3781 ± 0,0082). Store `nachlauf_v1_try2_store.jsonl`, Ausgaben `nachlauf_v1_try2_{sparsetsf,fits,patchtst,nlinear}.txt`. Unbeaufsichtigt, vom Nutzer freigegeben. **Diese vier Einheiten waren nicht mehr unberührt**: Die Änderungen sind durch ihre Fehler im ersten Versuch veranlasst.

## Ergebnis: wieder 0 von 4 gültig

| Einheit | Ausgang | LLM-Aufrufe | Kosten brutto | Was geschah |
|---|---|---|---|---|
| SparseTSF | nicht reproduzierbar (Setup), Absturz der Umgebung | 3 | 0,022 $ | Das Modell wählte diesmal den Kandidaten mit `--features S` (univariat). Training stürzte wie im ersten Versuch in `pandas` ab; die neue Repo-Zeile im Fehlertext zeigte die Ursache (`data_provider/data_loader.py`, Zeile 69, `df_stamp.date.apply(lambda row: row.month, 1)`). Das Modell verlangte `pandas==1.2.0`; die Prüfung ließ das zu (niedrigere Version, Paket im Traceback). Der Neubau des Snapshots scheiterte, weil sich pandas 1.2.0 unter Python 3.11 nicht installieren lässt. Lauf endete mit `SandboxSetupError`. |
| FITS | kein Befehl | 4 | 0,047 $ | Drei Seiten mit zusammen 90 von 137 Kandidaten gezeigt; **keiner davon enthält ETTh1**. Die neue Ordnung hat die passenden Befehle nicht nach oben gebracht. Das Modell forderte zweimal weitere an und wählte keinen. |
| PatchTST/42 | kein Befehl | 4 | 0,044 $ | 90 von 491 gezeigt; 22 enthalten ETTh1, aber alle für das Vergleichsmodell DLinear, keiner für PatchTST. Das Modell wählte keinen. |
| NLinear | kein Befehl | 4 | 0,051 $ | 90 von 1209 gezeigt (die Schleifen-Auflösung vervielfacht die Kandidaten). Oben stehen Befehle für NLinear mit Eingabelänge 104 statt 336. Der Befehl für das Experiment des Claims steht auf Platz 37, aber mit `--model DLinear`: Das Repo dokumentiert NLinear auf ETTh1-336 nicht als eigene Zeile. Das Modell wählte keinen. |
| **Summe** | **0 von 4** | 15 | **0,165 $** | |

Kein Infrastruktur-Abbruch im Sinne des Nachtrags (der SparseTSF-Abbruch folgt aus der Reparatur des Agenten). Kein Lauf wiederholt. Kosten seit dem Stichtag 2026-10-02 19:43 UTC insgesamt 0,60 $ brutto.

## Folge

- Punkt 1 der Erfolgsregel ist auch im zweiten Versuch verfehlt. **Ein dritter Versuch ist laut Vorab-Festlegung ausgeschlossen.**
- Der Modus „Nachlauf" gilt damit als: **funktioniert für die zwei Entwicklungs-Einheiten, nicht für fremde Repos.** README, Zeugnis und Video sagen das so.

## Ursachen (post hoc, aus Prompts und Antworten)

1. **Ordnung der Kandidaten trägt nicht.** Die Punktwerte heben Befehle des Claim-Modells nicht zuverlässig an (FITS: keiner mit ETTh1 unter 90; PatchTST: nur Vergleichsmodelle). Eine fehlende Eingabelänge wird nicht bestraft (NLinear: 104 vor 336). Ursache im Punktwert nicht untersucht.
2. **Was die Autoren nicht als Zeile dokumentieren, kann der Agent nicht starten.** NLinear auf ETTh1 mit Eingabelänge 336 steht im Repo nur als DLinear-Zeile mit Modell als Variable. Die Regel „Befehl aus wörtlicher Zeile" ist hier strenger als die Praxis des Repos.
3. **Versions-Rückgriff ohne Prüfung der Installierbarkeit.** Die Prüfung lässt jede niedrigere Version zu, auch eine, die es für die Python-Version des Snapshots nicht gibt.
4. **Aufgabenart beim ersten Wählen nicht geprüft.** Der Schutz der Aufgabenart gilt für Reparaturen; bei der ersten Wahl nahm das Modell einen univariaten Befehl für einen multivariaten Claim.

Das Modell selbst hat in allen vier Läufen nichts erfunden und nichts gemessen, was nicht gemessen wurde.

## Gesamtbild Modus „Nachlauf" (für README und Video)

- Gebaut und geprüft: Trennung von Training (nur Autorencode) und Vorhersage (ohne Rohdaten), Prüfungen P1 bis P4, drei Negativkontrollen abgelehnt.
- Per Agent gültig an zwei Repos (RLinear, DLinear), je fünf Seeds, Werte identisch mit der Handarbeit, rund 0,02 $ je Lauf. Dort liefert er den belegten `drop_last`-Befund (RLinear: Autorencode 0,366 auf 2688 Fenstern, alle 2785 Fenster 0,370).
- An vier fremden Repos in zwei vorab festgelegten Versuchen **0 von 4**. Der Engpass ist nicht das Training oder der Adapter, sondern das Finden des richtigen dokumentierten Befehls in großen Repos.
