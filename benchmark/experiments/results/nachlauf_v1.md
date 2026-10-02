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
