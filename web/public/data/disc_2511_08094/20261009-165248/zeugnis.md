# Reproduktions-Zeugnis: disc_2511_08094

## Hinweise zur Belegbarkeit
- Info: Das Paper nennt Streuung +-1.39 (ueber Splits/Laeufe, nicht Toleranz des Mittelwerts); die Toleranz +-0.01 bleibt fest.
- Steckbrief: Datei des Modells SL-GCN nicht im Repo-Ausschnitt

## Claim disc_2511_08094_accuracy: nicht reproduziert
Gemessen: 0.5951 (std 0.0421)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): nicht messbar (Genauigkeit 0.6197 mit Graph, n/a ohne Kanten)
Fehler-Reparaturen: 2 (Ergebnis schlechter als eine triviale Baseline, Code wurde neu geschrieben)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0421. Mit unserem Setup gemessen (5 von 5 Seeds, Zielwert 0.8292 +/- 0.01, Table 2, arXiv v2 (arxiv_html)): Mittel 0.5951. Code-Kandidaten: 3, gewaehlt Kandidat 0 (Code 49be1ef2015f) nach Validierungs-Metrik (0: accuracy 0.6524, 1: accuracy 0.3095, 2: accuracy 0.5857); Rueckfall ohne Val-Wert; Repo-Abgleich: keine geprueften Abweichungen
Zielwert 0.8292 belegt: "SL-GCN 82.92 ± 1.39 73.29 ± 2.04 79.21 ± 1.30" (Table 2, https://arxiv.org/html/2511.08094v2, Version v2, abgerufen 2026-09-30T18:00:54+00:00, SHA-256 d2924fd2ef1b)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/html/2511.08094v2

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-09T15:15:26+00:00 (live), 1 Seiten gelesen, 0 Aussagen verworfen (Zitat nicht im Text).

## Possible explanations für disc_2511_08094_accuracy (Tavily Rolle 6, ungeprüfte Hypothesen, nicht Teil des Urteils)
- The implementation loads tuned hyperparameters from best_params.py automatically, which may not be disclosed in the paper. — "Note:run.py automatically loads tuned hyperparameters from best_params.py when available." (https://github.com/kevvzhang/StuartLandauGNN)

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/kevvzhang/StuartLandauGNN @ 40944b1e0d

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:src/run.py@40944b1] learning rate: 0.001 — "parser.add_argument('--learning_rate', type=float, default=0.001)"
- [repo:src/experiment.py@40944b1] weight decay: args.weight_decay — "self.optimizer = getattr(
    torch.optim, args.optimizer
    )(
    self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )"
- [repo:src/run.py@40944b1] weight decay: 1e-5 — "parser.add_argument('--weight_decay', type=float, default=1e-5)"
- [repo:src/slmodel.py@40944b1] dropout: args.dropout — "self.dropout = args.dropout"
- [repo:src/slmodel.py@40944b1] dropout: args.input_dropout — "self.input_dropout = args.input_dropout"
- [repo:src/run.py@40944b1] dropout: 0.0 — "parser.add_argument('--dropout', type=float, default=0.0)"
- [repo:src/run.py@40944b1] input_dropout: 0.0 — "parser.add_argument('--input_dropout', type=float, default=0.0)"
- [repo:src/run.py@40944b1] hidden size: 128 — "parser.add_argument('--nhid', type=int, default=128)"
- [repo:src/run.py@40944b1] heads: 2 — "parser.add_argument('--nhead', type=int, default=2)"
- [repo:src/experiment.py@40944b1] epochs: 100 — "def train(self, max_epoch=100, disable_tqdm=False):"
- [repo:src/run.py@40944b1] epochs: 1000 — "parser.add_argument('--max_epoch', type=int, default=1000)"
- [repo:src/slmodel.py@40944b1] early stopping rule: self.tol — "if torch.max(torch.abs(update)) < self.tol:"
- [repo:src/experiment.py@40944b1] early stopping rule: self.lr_stop = self.args.learning_rate * 0.9 — "self.lr_stop = self.args.learning_rate * 0.9"
- [repo:src/experiment.py@40944b1] early stopping rule: self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=self.args.lr_decay_patience, factor=self.args.lr_decay_factor,) — "self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=self.args.lr_decay_patience, factor=self.args.lr_decay_factor,)"
- [repo:src/experiment.py@40944b1] early stopping rule: if new_lr < self.lr_stop: break — "if new_lr < self.lr_stop: break"
- [paper] other settings: 100 random dataset splits and initialization — "The means and standard deviations are computed over 100 random dataset splits and initialization."
- [repo:src/experiment.py@40944b1] other_settings: device — "self.device = args.device"
- [repo:src/experiment.py@40944b1] other_settings: scheduler mode (max) — "ReduceLROnPlateau(self.optimizer, 'max', patience=self.args.lr_decay_patience, factor=self.args.lr_decay_factor,)"
- [repo:src/run.py@40944b1] other: False — "parser.add_argument('--bias', type=bool, default=False)"
- [repo:src/run.py@40944b1] other: 80 — "parser.add_argument('--natt', type=int, default=80)"
- [repo:src/run.py@40944b1] other: 10 — "parser.add_argument('--nlayer', type=int, default=10)"
- [repo:src/run.py@40944b1] other: 0.3 — "parser.add_argument('--k', type=int, default=0.3)"
- [repo:src/run.py@40944b1] other: 2. — "parser.add_argument('--p', type=float, default=2.)"
- [repo:src/run.py@40944b1] other: 1.0 — "parser.add_argument('--dt', type=float, default=1.0)"
- [repo:src/run.py@40944b1] other: 0.0 — "parser.add_argument('--zeta_real', type=float, default=0.0)"
- [repo:src/run.py@40944b1] other: 1.0 — "parser.add_argument('--zeta_imag', type=float, default=1.0)"
- [repo:src/run.py@40944b1] other: 1.0 — "parser.add_argument('--nu_real', type=float, default=1.0)"
- [repo:src/run.py@40944b1] other: 1.0 — "parser.add_argument('--nu_imag', type=float, default=1.0)"
- [repo:src/run.py@40944b1] other: 1e-3 — "parser.add_argument('--tol', type=float, default=1e-3)"
- [repo:src/run.py@40944b1] other: 0.3 — "parser.add_argument('--leaky_relu_slope', type=float, default=0.3)"
- [repo:src/run.py@40944b1] other: NAdam — "parser.add_argument('--optimizer', type=str, default='NAdam')"
- [repo:src/run.py@40944b1] other: 0.1 — "parser.add_argument('--lr_decay_factor', type=float, default=0.1)"
- [repo:src/run.py@40944b1] other: 100 — "parser.add_argument('--lr_decay_patience', type=int, default=100)"
- [repo:src/run.py@40944b1] other: 64 — "parser.add_argument('--batch_size', type=int, default=64)"
- [repo:src/run.py@40944b1] other: cuda — "parser.add_argument('--device', type=str, default='cuda')"
- [repo:src/run.py@40944b1] other: -1 — "parser.add_argument('--seed', type=int, default=-1)"
- [repo:src/run.py@40944b1] other: -1 — "parser.add_argument('--dataset_seed', type=int, default=-1)"
- [repo:src/run.py@40944b1] other: 10 — "parser.add_argument('--test_repeats', type=int, default=10, help='Number of test repeats')"
- [repo:src/run.py@40944b1] other: 0.0 — "parser.add_argument('--alpha', type=float, default=0.0, help='Alpha for GraphhCON model')"
- [repo:src/run.py@40944b1] other: 0.0 — "parser.add_argument('--gamma', type=float, default=0.0, help='Gamma for GraphhCON model')"
- [repo:src/data.py@40944b1] other settings: default seed=12345 — "def get_dataset(dataset_name, data_dir='./data/', use_lcc=False, seed=12345, split=-1) -> InMemoryDataset:"
- [repo:src/data.py@40944b1] other settings: num_development=1500 for train/val/test split — "dataset._data = set_train_val_test_split(
            seed,
            dataset._data,
            num_development=1500)"
- [repo:src/data.py@40944b1] other settings: num_per_class=20 for per‑class training samples — "def set_train_val_test_split(
        seed,
        data: Data,
        num_development = 1500,
        num_per_class = 20) -> Data:"
- Not found in the sources: feature normalisation; adjacency normalisation

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch