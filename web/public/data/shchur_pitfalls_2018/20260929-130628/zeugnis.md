# Reproduktions-Zeugnis: shchur_pitfalls_2018

## Claim shchur_gcn_cora_acc: teilweise
Gemessen: 0.8018 (std 0.0127)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.7991 mit Graph, 0.5953 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0127. Mit unserem Setup gemessen (30 von 30 Seeds, Zielwert 0.815 +/- 0.01, Table 1, arXiv v2 (pdf)): Mittel 0.8018.
Zielwert 0.815 belegt: "GCN 81 .5 ± 1.3 71.9 ± 1.9 77 .8 ± 2.9 62.2 ± 0.6" (Table 1, https://arxiv.org/pdf/1811.05868v2, Version v2, abgerufen 2026-09-29T07:54:33+00:00, SHA-256 47ac4b09ea8c)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1811.05868v2

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] data_splits: 100 random train/validation/test splits — "For each dataset we use 100 random train/validation/test splits"
- [paper] random_initializations: 20 random initializations for each split — "perform 20 random initializations for each split"
- [paper] training_procedure: standardized training and hyperparameter selection procedure for all models — "We use a standardized training and hyperparameter selection procedure for all models"
- [paper] data splits: 20 labeled nodes per class as the training set, 30 nodes per class as the validation set, and the rest as the test set — "In all cases, we use 20 labeled nodes per class as the training set, 30 nodes per class as the validation set, and the rest as the test set."
- [paper] number of random splits: 100 random train/validation/test splits — "Table 1: Mean test set accuracy and standard deviation in percent averaged over 100 random train/validation/test splits with 20 random weight initializations each for all models and all datasets."
- [paper] hyperparameter search: extensive grid search for learning rate, size of the hidden layer, strength of the L2 regularization, and dropout probability — "We performed an extensive grid search for learning rate, size of the hidden layer, strength of the L2 regularization, and dropout probability (Appendix C)."
- [paper] hyperparameter selection criterion: picked the hyperparameter configuration that achieved the best average accuracy on Cora and CiteSeer datasets (averaged over 100 train/validation/test splits and 20 random initializations for each) — "For every model, we picked the hyperparameter configuration that achieved the best average accuracy on Cora and CiteSeer datasets (averaged over 100 train/validation/test splits and 20 random initializations for each)."
- [paper] evaluation protocol: Mean test set accuracy and standard deviation averaged over 100 splits and 20 random initializations each — "Results Table 1 shows mean accuracies (and their standard deviations 2) of all models for all 8 datasets averaged over 100 splits and 20 random initializations for each split."
- [paper] Evaluation protocol: averaged over multiple data splits — "the results are averaged over multiple data splits"
- [paper] Train/validation/test split sizes: same as Yang et al. 2016 — "different random split with the same train/validation/test set sizes"
- [paper] data splits: 100 random train/validation/test splits — "over 100 random train/validation/test splits"
- [paper] weight initializations: 20 random weight initializations each — "with 20 random weight initializations each."

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch