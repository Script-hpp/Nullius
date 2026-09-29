# Shwartz-Ziv & Armon 2021 („Tabular Data: Deep Learning is Not All You Need“): Vorklärung

Ein Tavily-Extract (Fixture `benchmark/fixtures/2106.03253.json`, arXiv v2, PDF, 50 597 Zeichen). Keine Vorab-Festlegung, nur Bestandsaufnahme; Werte aus dem Paper-Text, nichts aus dem Gedächtnis.

## Was im Paper steht
- 11 Datensätze (Tabelle 1): Gesture Phase (32 Merkmale, 5 Klassen, 9,8k Zeilen, OpenML), Gas Concentrations (129, 6, 13,9k, OpenML), Eye Movements (26, 3, 10,9k, OpenML),
  Shrutime (11, 2, 10k, Kaggle), Blastchar (20, 2, 7k, Kaggle) sowie große: Epsilon 500k, YearPrediction 515k, MSLR 964k, Rossmann 1018k, Forest Cover 580k, Higgs 800k.
- Metrik: Klassifikation Kreuzentropie **mal 100**, Regression RMSE laut Fließtext, aber Tabelle 2 sagt „MSE is presented for YearPrediction and Rossman“ (Widerspruch im Paper; Werte wie 77,98 passen zu MSE).
- Protokoll: HyperOpt (Bayes), **1000 Schritte** pro Datensatz und Modell; Splits „wie in den Originalpapern“ (dort nachzulesen, nicht in diesem Text); bei Zufallssplit 3 Wiederholungen, sonst 4 Seeds.
- Tabelle 2 (Mittel ± Standardfehler, CE×100), XGBoost: Gesture 80,64 ± 0,80; Gas 2,18 ± 0,20; Eye 56,07 ± 0,65; Shrutime 13,82 ± 0,19; Blastchar 20,39 ± 0,21.
  Tiefe Modelle auf denselben (Eye/Gesture/Gas): NODE 68,35 / 92,12 / 2,17; DNF-Net 68,38 / 86,98 / 1,44; TabNet 67,13 / 96,42 / 1,92; 1D-CNN 67,9 / 97,89 / 1,79.
- Aussage (Tabelle 3): relative Verschlechterung auf ungesehenen Datensätzen: XGBoost 3,34 %, 1D-CNN 7,56 %, TabNet 10,51 %, DNF-Net 11,96 %, NODE 14,21 %; Ensemble mit XGBoost 2,32 %.

## Was für unsere CPU-Grenze (unter 10 Minuten) in Frage kommt
Nur die kleinen: Eye Movements (10,9k), Gesture Phase (9,8k), Gas Concentrations (13,9k), Shrutime (10k), Blastchar (7k). Shrutime/Blastchar liegen bei Kaggle (Anmeldung nötig, keine öffentliche Datei geprüft);
die drei OpenML-Sätze sind öffentlich (OpenML-IDs **nicht geprüft**, der Abruf der OpenML-API schlug in einem Schnelltest fehl: Erreichbarkeit/URL klären).

## Probleme, bevor ein Task Sinn ergibt
1. **Protokoll steht nicht in diesem Paper.** Split-Art und Vorverarbeitung stehen in den Originalpapern (DNF-Net für Eye/Gesture/Gas). Ohne sie ist die Datenaufteilung geraten.
2. **1000 HyperOpt-Schritte sind auf CPU in 10 Minuten nicht machbar** (je Schritt ein XGBoost-Training). Ein Agent mit kleiner Suche landet vermutlich hinter dem Paperwert; „nicht reproduziert“ könnte dann das Tuning-Budget messen, nicht das Paper.
3. **Metrik unklar:** „Kreuzentropie × 100“ bei Shrutime (13,82) und Blastchar (20,39) wirkt für Kundenabwanderungsdaten sehr niedrig (übliche Logloss-Werte liegen deutlich höher); ob das Paper etwas anderes berechnet, ist nicht geprüft. Eye/Gesture/Gas (56/81/2,2) wirken plausibel.
4. **Vertrag des Harness:** Kreuzentropie braucht Klassenwahrscheinlichkeiten, unser Vertrag `id,prediction` liefert eine Zahl pro Zeile. Nötig: Vertrag mit Wahrscheinlichkeiten (z. B. `id = {zeile}_{klasse}`) und eine geprüfte Kreuzentropie in `metrics.py`.
5. Die Kernaussage (XGBoost schlägt tiefe Modelle) verlangt tiefe Modelle nachzubauen (TabNet, NODE, DNF-Net), das sprengt Zeit und CPU.

## Vorschlag
Engster sinnvoller Claim: **XGBoost auf Eye Movements, Kreuzentropie×100 = 56,07 ± 0,65** (Tabelle 2), ergänzt um den Vergleichswert des besten tiefen Modells (TabNet 67,13) als Kontext. Voraussetzungen vorab klären:
(a) OpenML-Zugang und Split laut DNF-Nachweis, (b) festes Tuning-Budget für den Agenten (z. B. höchstens 30 Konfigurationen) und daraus eine **vorab** festgelegte, ehrlich weitere Toleranz (ein Paperwert mit 1000 Schritten ist mit 30 nicht zu erwarten), (c) Wahrscheinlichkeits-Vertrag und Kreuzentropie im Harness.
Ohne (a) bis (c) würde der Task ein Urteil über Protokollraten liefern. Alternative: Shwartz-Ziv aus dem Benchmark lassen und durch ein Paper ersetzen, dessen Protokoll im Text steht.
