# Reproduktions-Zeugnis: disc_2511_08094

## Hinweise zur Belegbarkeit
- Info: Das Paper nennt Streuung +-1.39 (ueber Splits/Laeufe, nicht Toleranz des Mittelwerts); die Toleranz +-0.01 bleibt fest.

## Claim disc_2511_08094_accuracy: nicht reproduziert
Gemessen: 0.7830 (std 0.0176)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.7508 mit Graph, 0.5588 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0176. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.8292 +/- 0.01, Table 2, arXiv v2 (arxiv_html)): Mittel 0.7830.
Zielwert 0.8292 belegt: "SL-GCN 82.92 ± 1.39 73.29 ± 2.04 79.21 ± 1.30" (Table 2, https://arxiv.org/html/2511.08094v2, Version v2, abgerufen 2026-09-30T18:00:54+00:00, SHA-256 d2924fd2ef1b)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/html/2511.08094v2

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/kevvzhang/StuartLandauGNN @ 40944b1e0d

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] model architecture: SL-GCN equations (Stuart-Landau Oscillatory GNN) — "\displaystyle Y^{(l+1)}\displaystyle=Y^{(l)}+dt~\sigma(F^{(l)}_{\theta}(X^{(l)}))-\gamma X^{(l)}-\alpha Y^{(l)},
\displaystyle X^{(l+1)}\displaystyle=X^{(l)}+dt~Y^{(l+1)},"
- [paper] initialisation: random initialisation (unspecified) — "The means and standard deviations are computed over 100 random dataset splits and initialization."
- [paper] dataset: Cora — "we evaluate their performance on Cora"
- [paper] data splits: 10 random splits — "The mean values are computed over 10 random splits and weight initializations."
- [paper] model depth range: seven fixed depths ranging from 2 to 128 layers — "we therefore re-tune the models for seven fixed depths ranging from 2 to 128 layers."
- [paper] model architecture: OGNNs with GCN coupling — "we evaluate their performance on Cora with GCN coupling across OGNNs of different depths."
- [paper] evaluation metric: node classification accuracy (%) — "Figure 1. Node classification accuracy (%) on Cora for GNNs of varying depths."

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch