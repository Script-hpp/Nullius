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
- **30 statt 2000 Läufe: 10 zufällige Splits × 3 Initialisierungen** (`seeds: 30`; Lauf `s` = Split `s // 3`, Initialisierung `s % 3`).
  Splits sind pro Split-Nummer deterministisch (`harness/cora_lcc_data.py`, `RandomState(1000 + k)`), nicht identisch zu denen des Papers.
- Der Agent bekommt pro Lauf nur die Datei seines Splits (Testlabels fehlen); der Harness erzeugt Split und Testlabels selbst neu.
- Die Rolle 4 (Setup-Details) darf die Angaben des Papers übernehmen (Anhang C steht im extrahierten Text).

**Claim** `shchur_gcn_cora_acc`: GCN erreicht 81,5 % auf Cora (größte Komponente, zufällige 20/30/Rest-Splits), Tabelle 1.
**Toleranz ±0,01** (feste Regel für Accuracy). Begründung: Die Std 1,3 des Papers gilt für Einzelläufe; der Mittelwert über 30 Läufe
streut bei uns grob mit 1,3/√10 ≈ 0,4 Prozentpunkten (Splits dominieren, geschätzt, nicht belegt). Urteil auf dem Mittelwert über alle 30 Läufe.
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
