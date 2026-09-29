# Task Wu et al. 2019 (SGC, „Simplifying Graph Convolutional Networks“)

Festlegungen vor dem ersten Agentenlauf. Quelle: Fixture `benchmark/fixtures/1902.07153.json` (arXiv v2, PDF, 1 Tavily-Extract-Aufruf), Werte aus dem Text, nicht aus dem Gedächtnis.

## Belegt im Paper
- Tabelle 2 („Our experiments“, Test-Accuracy in %, Mittel über 10 Läufe, Planetoid-Split 140/500/1000): **SGC Cora 81,0 ± 0,0**, Citeseer 71,9 ± 0,1, Pubmed 78,9 ± 0,0; GCN 81,4 ± 0,4; GAT 83,3 ± 0,7.
  Aus der Literatur (Tabelle 2, obere Hälfte): GCN 81,5, GAT 83,0 ± 0,7.
- Training auf den Zitiernetzen: „we train SGC for 100 epochs using Adam with learning rate 0.2“, Weight Decay per hyperopt (60 Iterationen) auf dem Validierungsset des öffentlichen Splits; transduktiv.
- Modell: SGC hat keine gelernte Merkmalsextraktion, `S^K X` (Propagation ohne Parameter, danach logistische Regression); Abschnitt 5.1 nennt „SGC involves no learning because the feature extraction step, S^K X, has no parameter“.
- Zusatz Tabelle 10 (zufällige Trainingssplits, gleiche Validierungs-/Testmenge): SGC Cora 80,62 ± 1,21; GAT 82,29 ± 1,16; GCN 80,53 ± 1,40.

## Task
Gleicher Task wie Kipf & Welling (`Cora`, Planetoid-Split, gleicher Harness, gleiche Baselines, Formprüfung, Graph-Ablation). Nur ein Eintrag in `benchmark/papers.yaml` (`wu_sgc_2019`), kein neuer Code.
**Claim** `sgc_cora_acc`: SGC erreicht 81,0 % auf Cora (Planetoid-Split). **Toleranz ±0,01** (feste Regel für Accuracy; die Std 0,0 der Tabelle beschreibt ein fast deterministisches Modell, keine Toleranz). 10 Seeds.
Zielwert-Spalte: die Kopfzeile „Cora Citeseer Pubmed“ steht vor der Zeile; die Spaltenprüfung (`table_check.py`, Variante ohne „Method“-Kopf) bestätigt Position 1 von 3 (ohne LLM geprüft: 81,0 ok, 71,9 wäre Citeseer).

## Erwartung und Grenzen
- Erwartet: reproduziert. Die Graph-Ablation ist auch hier anwendbar (SGC ohne Graph = logistische Regression, ~0,5 Accuracy).
- SGC ist einfach genug, dass ein Agent die Methode leicht trifft; das macht den Task zu einem Test der Pipeline, nicht des Agenten (wie GCN).
- Die Einzelheit „Weight Decay per hyperopt“ ist eine Suche, die der Agent vereinfachen muss (Rolle 4 kann die Werte 100 Epochen, lr 0,2 liefern, den Wert des Decay nicht).
- Nicht aufgenommen: Citeseer/Pubmed (bräuchten neue Tasks), GAT (aufwendiger, mehr Agenten-Fehler; Kandidat, falls Zeit bleibt).
