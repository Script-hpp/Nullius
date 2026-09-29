# Reproduktions-Zeugnis: wu_sgc_2019

## Claim sgc_cora_acc: teilweise
Gemessen: 0.7804 (std 0.0109)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): nicht messbar (Genauigkeit 0.7860 mit Graph, n/a ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0109. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.81 +/- 0.01, Table 2, arXiv v2 (pdf)): Mittel 0.7804.
Zielwert 0.81 belegt: "SGC 81.0 ± 0.0" (Table 2, https://arxiv.org/pdf/1902.07153v2, Version v2, abgerufen 2026-09-29T09:57:31+00:00, SHA-256 3f53d96637bf)
Quellen:
- https://arxiv.org/pdf/1902.07153v2

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] data splits: Train/Dev/Test Nodes: 140 /500 /1,000 — "Cora 2, 708 5, 429 140 /500 /1, 000"
- [paper] evaluation protocol: semi-supervised node classification performance; test accuracy reported — "We evaluate the semi-supervised node classification performance of SGC on the Cora, Citeseer, and Pubmed citation network datasets (Table 2)"
- [paper] epochs: 100 — "On the citation net-works, we train SGC for 100 epochs using Adam (Kingma & Ba, 2015) with learning rate 0.2."
- [paper] weight_decay_tuning: tuned via hyperopt for 60 iterations on public split validation set — "In addition, we use weight decay and tune this hyperparameter on each dataset using hyperopt (Bergstra et al., 2015) for 60 iterations on the public split validation set."
- [paper] training_setting: transductive — "Experiments on citation net-works are conducted transductively ."
- [paper] epoch_tuning: number of epochs tuned based on convergence behavior and validation accuracy — "On all datasets, we tune the number of epochs based on both convergence behavior and validation accuracy."
- [paper] model_architecture: SGC (Simplified Graph Convolution) with no learnable parameters in feature extraction step — "We note that SGC involves no learning because the feature extraction step, SK X, has no parameter."
- [paper] evaluation_protocol: test accuracy averaged over 10 runs — "Table 2. Test accuracy (%) averaged over 10 runs on citation net-works."

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch