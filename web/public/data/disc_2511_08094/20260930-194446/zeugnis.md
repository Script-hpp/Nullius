# Reproduktions-Zeugnis: disc_2511_08094

## Hinweise zur Belegbarkeit
- Info: Das Paper nennt Streuung +-1.39 (ueber Splits/Laeufe, nicht Toleranz des Mittelwerts); die Toleranz +-0.01 bleibt fest.

## Claim disc_2511_08094_accuracy: nicht reproduziert
Gemessen: 0.5520 (std 0.0249)
Methodentreue: nein. Gesamtaussage: nicht bewertbar (Methode nicht umgesetzt)
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph nicht genutzt (Genauigkeit 0.5541 mit Graph, 0.5541 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0249. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.8292 +/- 0.01, Table 2, arXiv ? (arxiv_html)): Mittel 0.5520.
Zielwert 0.8292 belegt: "| SL-GCN | 82.92 ± 1.39 | 73.29 ± 2.04 | 79.21 ± 1.30 |" (Table 2, https://arxiv.org/html/2511.08094, Version None, abgerufen 2026-09-30T17:45:07+00:00, SHA-256 1cd8a992d1da)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/html/2511.08094

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/kevvzhang/StuartLandauGNN @ 40944b1e0d

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] encoder/decoder depth: single-layered MLPs — "Both the encoder ENC and the decoder DEC are single-layered MLPs."
- [paper] decoder type: choice between real- or complex-valued DEC does not noticeably affect model performance — "While ENC is complex-valued, from our experience the choice between a real- or complex-valued DEC does not noticeably affect model performance."
- [paper] readout types: do not improve and may sometimes degrade model performance — "Other readout types including modulus, phases, and phase differences do not improve and may sometimes degrade model performance."
- [paper] polar-cartesian conversion overhead: negligible computational overhead and does not impact model performance — "Although this scheme requires repeated polar-cartesian conversion at each layer, empirically this introduces negligible computational overhead and does not impact model performance."
- [paper] Newton method convergence: typically converges within 2 - 5 steps, depending on the set error tolerance — "When Newton’s method is used, it typically converges within 2 - 5 steps, depending on the set error tolerance."
- [paper] hyperparameter release: best hyperparameters released along with the code — "We release the best hyperparameters along with the code (available upon publication)."
- [paper] num_random_splits: 100 — "The means and standard deviations are computed over 100 random dataset splits and initialization."
- [paper] evaluation_protocol: transductive node classification — "Transductive node classification accuracy (%) on homophilic datasets."
- [paper] data split scheme for homophilic graphs: follow shchurPitfallsGraphNeural2018 — "For homophilic graphs, we follow the data split schemes laid out by shchurPitfallsGraphNeural2018 (shchurPitfallsGraphNeural2018)."
- [paper] evaluation protocol (splits and initializations): 10 pre-defined splits with 10 random initializations per split — "The means and standard deviations are computed over 10 pre-defined splits, with 10 random initializations per split."
- [paper] task type: transductive node classification — "We evaluate all models on transductive node classification tasks on both homophilic and heterophilic datasets, including Cora (mccallumAutomatingConstructionInternet2000), Citeseer (senCollectiveClassificationNetwork2008), Pubmed (namataQuerydrivenActiveSurveying2012), WebKB (peiGeomGCNGeometricGraph2020) (Texas, Wisconsin, Cornell), WikipediaNetwork (rozemberczkiMultiscaleAttributedNode2019) (Chameleon, Squirrel), heterophilous graphs (platonovCriticalLookEvaluation2023) (Amazon-rating)."
- [paper] data splits: 10 random splits — "The mean values are computed over 10 random splits and weight initializations."
- [paper] depth tuning: seven fixed depths ranging from 2 to 128 layers — "we therefore re-tune the models for seven fixed depths ranging from 2 to 128 layers."
- [paper] evaluation protocol: hyperparameters tuned on validation set; final test performance reported using best validation configuration — "Hyperparameters are optimized by maximizing classification accuracy or minimizing regression loss on the validation set. We report the final evaluation results on the test set with best-performing configuration on the validation set."

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch