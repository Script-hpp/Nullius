# Task Shchur et al. 2018 (Cora, zufällige Splits): Festlegungen und Harness-Validierung

Vorab festgelegt (Abschnitt 1 vor dem ersten Agentenlauf); Abschnitt 2 ist ein Ergebnis ohne LLM.

## 1. Festlegungen (aus dem Paper, Fixture `benchmark/fixtures/1811.05868.json`, arXiv v2, PDF, 1 Tavily-Extract-Aufruf)
Belegt im Paper-Text (wörtlich geprüft, nicht aus dem Gedächtnis):
- Datensatz: „we treat the graphs as undirected and only consider the largest connected component“; Cora nach Tabelle 3: 2485 Knoten, 7 Klassen, 1433 Merkmale.
- Split: „we use 20 labeled nodes per class as the training set, 30 nodes per class as the validation set, and the rest as the test set“.
- Paper-Mittelung: 100 zufällige Splits × 20 Initialisierungen (2000 Läufe). **Nicht bezahlbar.**
- Tabelle 1 (Mittel ± Std in Prozent, CORA): GCN 81,5 ± 1,3; MLP 58,2 ± 2,1; LogReg 57,1 ± 2,3. Tabelle 2b (ein einzelner anderer Split): GCN 79,0 ± 0,7.
- Trainingsprotokoll (Anhang C, Tabelle 4): GCN Hidden 64, lr 0,01, Dropout 0,8, L2 0,001, Adam, Early Stopping auf Gesamt-Validierungsloss (Geduld 50), Gewichte des besten Schritts zurücksetzen, Glorot-Init, Nullbias.

Unsere Abweichungen (offen im Zeugnis zu nennen):
- **30 statt 2000 Läufe: 30 zufällige Splits × 1 Initialisierung** (`seeds: 30`; Lauf `s` = Split `s`, Initialisierung 0). Ursprünglich 10 Splits × 3 Initialisierungen, vor jedem Agentenlauf umgestellt (siehe Abschnitt 3).
  Splits sind pro Split-Nummer deterministisch (`harness/cora_lcc_data.py`, `RandomState(1000 + k)`), nicht identisch zu denen des Papers.
- Der Agent bekommt pro Lauf nur die Datei seines Splits (Testlabels fehlen); der Harness erzeugt Split und Testlabels selbst neu.
- Die Rolle 4 (Setup-Details) darf die Angaben des Papers übernehmen (Anhang C steht im extrahierten Text).

**Claim** `shchur_gcn_cora_acc`: GCN erreicht 81,5 % auf Cora (größte Komponente, zufällige 20/30/Rest-Splits), Tabelle 1.
**Toleranz ±0,01** (feste Regel für Accuracy). Begründung: Die Std 1,3 des Papers gilt für Einzelläufe; der Mittelwert über 30 Läufe
streut bei uns mit Std/√30: Die Referenz hat Std 1,18 Prozentpunkte über 30 Läufe (Abschnitt 3), also Standardfehler ≈ 0,22 Prozentpunkte. Urteil auf dem Mittelwert über alle 30 Läufe.
Nicht aufgenommen: ein MLP-Claim (58,2). Die No-Graph-Baseline des Harness liegt bei ~0,51 und die Fehlerregel „schlechter als Baseline“ könnte für ein
so schwaches Modell fälschlich „Agenten-Fehler vermutet“ auslösen. Falls später aufgenommen, vorab die Fehlerregel dafür entscheiden.
Die Aussage des Papers ist eigentlich eine Rangfolge (GCN vorn bei zufälligen Splits, GAT vorn bei Planetoid); dieser Task prüft nur die GCN-Zahl.

## 2. Harness-Validierung mit menschlicher Referenzlösung (kein LLM)
`tests/reference/gcn_lcc_reference.py` (Protokoll oben) über alle 30 Läufe durch die Pipeline (Datenvorbereitung im Branch, Harness):
`results/lcc_harness_reference.txt`. **30/30 gültig, Accuracy Mittel 0,8068, Std 0,0128; je Split-Mittel 0,786 bis 0,830.**
- Die Std über die Läufe (1,28 Prozentpunkte) entspricht der Std des Papers (1,3): die Streuung durch Splits und Initialisierung passt.
- Das Mittel (80,7) liegt 0,8 Prozentpunkte unter dem Paperwert (81,5), innerhalb der Toleranz ±1, aber knapp. Ursache offen (nicht geprüft):
  Details des Papers-Code (z. B. Trainingslänge, Vorverarbeitung), andere Splits, nur 10 Splits.
- Baselines Split 0: Mehrheitsklasse 0,138, Softmax ohne Graph 0,509 (Paper: LogReg 57,1, MLP 58,2 mit Tuning).
- Dauer: 8,5 Minuten für 30 Läufe (5 parallel).
Das ist ein Selbsttest der Pipeline, kein Reproduktionsurteil für den Agenten.

## 3. Nachtrag: Referenz mit 30 Splits (90 Läufe) und mit Anhang-C-Stoppkriterium
- Referenz angepasst auf Anhang C (Early Stopping auf Gesamt-Validierungsloss inkl. Regularisierung, max. 10000 Epochen): 10 Splits Mittel **0,8065**
  (vorher 0,8068, `results/lcc_harness_reference_v1.txt`). Das Stoppkriterium erklärt den Abstand zu 81,5 nicht.
- **30 Splits × 3 Initialisierungen (90 Läufe, `results/lcc_harness_reference_90runs.txt`): Mittel 0,8111, Std 0,0124.** Der Abstand zum Paperwert schrumpft
  von 0,85 auf 0,39 Prozentpunkte. Die ersten 10 Splits liegen also etwa 0,5 Punkte unter dem Mittel aller 30; ein Teil des Abstands ist Stichprobe der 10 Splits.
- Folge: Mit den festen 10 Splits (`seeds: 30`) erreicht selbst die Referenz nur 0,8065, Abweichung 0,0085 von 0,815 bei Toleranz 0,01. Der Spielraum für Agenten-Codes ist klein.
  Bewusst unverändert (Vorab-Festlegung); Alternative wäre mehr Splits (Kosten: 8,5 Minuten je 10 Splits).

## 4. Umstellung auf 30 Splits × 1 Initialisierung (entschieden vor jedem Agentenlauf)
Begründung (statistisch): Der Referenztest mit 90 Läufen zeigt, dass die Varianz überwiegend zwischen den Splits entsteht (die ersten 10 Splits lagen
0,5 Punkte unter dem Mittel aller 30, Abschnitt 3). Mehr Splits statt mehr Initialisierungen verbessert die Schätzung des Mittelwerts bei gleichen Kosten (30 Läufe).
Beide Arme laufen auf denselben 30 Splits (gepaarter Vergleich). Kein Agenten-Ergebnis lag zum Zeitpunkt der Entscheidung vor.
Prüflauf der neuen Konfiguration (Referenz, `results/lcc_harness_reference_30x1.txt`): 30/30 gültig, **Mittel 0,8107, Std 0,0118**, Abweichung von 0,815: 0,43 Prozentpunkte.
Das Mittel der 90 Läufe (30 Splits × 3 Init.) war 0,8111; die Initialisierungen ändern es nicht (0,0004).
