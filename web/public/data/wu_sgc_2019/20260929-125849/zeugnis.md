# Reproduktions-Zeugnis: wu_sgc_2019

## Claim sgc_cora_acc: reproduziert
Gemessen: 0.8165 (std 0.0069)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8170 mit Graph, 0.5920 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0069. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.81 +/- 0.01, Table 2, arXiv v2 (pdf)): Mittel 0.8165.
Zielwert 0.81 belegt: "SGC 81.0 ± 0.0" (Table 2, https://arxiv.org/pdf/1902.07153v2, Version v2, abgerufen 2026-09-29T09:57:31+00:00, SHA-256 3f53d96637bf)
Quellen:
- https://arxiv.org/pdf/1902.07153v2

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] dataset: Cora (Sen et al., 2008) — "We evaluate the semi-supervised node classification performance of SGC on the Cora, Citeseer, and Pubmed citation network datasets (Table 2) (Sen et al., 2008)."
- [paper] data split: 140 training nodes, 500 validation nodes, 1000 test nodes — "Cora 2, 708 5, 429 140 /500 /1, 000"
- [paper] training_epochs: 100 epochs — "we train SGC for 100 epochs using Adam (Kingma & Ba, 2015) with learning rate 0.2."
- [paper] weight_decay_tuning: tuned via hyperopt for 60 iterations on public split validation set — "we use weight decay and tune this hyperparameter on each dataset using hyperopt (Bergstra et al., 2015) for 60 iterations on the public split validation set."
- [paper] epochs_tuning: tuned based on convergence behavior and validation accuracy — "On all datasets, we tune the number of epochs based on both convergence behavior and validation accuracy."

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch