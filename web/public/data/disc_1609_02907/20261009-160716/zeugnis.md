# Reproduktions-Zeugnis: disc_1609_02907

## Claim disc_1609_02907_accuracy: reproduziert
Gemessen: 0.8146 (std 0.0077)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8220 mit Graph, 0.5520 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0077. Mit unserem Setup gemessen (9 von 10 Seeds, Zielwert 0.815 +/- 0.01, Table 2, arXiv v4 (pdf)): Mittel 0.8146. Code-Kandidaten: 3, gewaehlt Kandidat 1 (Code f3fd662c7915) nach Validierungs-Metrik (0: accuracy 0.7740, 1: accuracy 0.7940, 2: accuracy 0.7840); Rueckfall ohne Val-Wert; Repo-Abgleich: keine geprueften Abweichungen; Seed 5: kein gueltiger Lauf nach 4 Versuchen (letzter Fehler: Nur eine einzige Vorhersage-Klasse ueber alle IDs -- verdaechtig)
Zielwert 0.815 belegt: "GCN (this paper) 70.3 (7s) 81.5 (4s) 79.0 (38s) 66.0 (48s)" (Table 2, https://arxiv.org/pdf/1609.02907v4, Version v4, abgerufen 2026-10-09T14:07:17+00:00, SHA-256 c919b23efc1d)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1609.02907v4

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-09T14:21:17+00:00 (live), 6 Seiten gelesen, 1 Aussagen verworfen (Zitat nicht im Text).

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/tkipf/gcn @ 39a4089fe7

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:gcn/train.py@39a4089] feature_normalisation: preprocess_features applied to feature matrix — "features = preprocess_features(features)"
- [repo:gcn/utils.py@39a4089] feature normalisation: row-normalize feature matrix (divide each row by its sum) — """"Row-normalize feature matrix and convert to tuple representation""""
- [repo:gcn/train.py@39a4089] adjacency normalisation: support = [preprocess_adj(adj)] — "support = [preprocess_adj(adj)]"
- [paper] learning rate: 0.01 — "We train all models for a maximum of 200 epochs (training iterations) using Adam (Kingma & Ba, 2015) with a learning rate of 0.01 and early stopping with a window size of 10, i.e. we stop training if the validation loss does not decrease for 10 consecutive epochs."
- [repo:gcn/train.py@39a4089] learning rate: 0.01 — "flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')"
- [paper] weight decay: 5·10−4 — "5 · 10−4 (L2 regularization)"
- [repo:gcn/train.py@39a4089] weight_decay: 5e-4 — "flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')"
- [paper] dropout: used (applied via dropout) — "Stochasticity in the training process is introduced via dropout (Srivastava et al., 2014)."
- [repo:gcn/train.py@39a4089] dropout: 0.5 — "flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')"
- [repo:gcn/models.py@39a4089] dropout: True — "dropout=True,"
- [paper] hidden size: 16 — "We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 0.5 (dropout rate), 5 · 10−4 (L2 regularization) and 16 (number of hidden units);"
- [repo:gcn/train.py@39a4089] hidden_size: 16 — "flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')"
- [repo:gcn/train.py@39a4089] epochs: 200 — "flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')"
- [paper] other: two-layer GCN — "Unless otherwise noted, we train a two-layer GCN as described in Section 3.1 and evaluate prediction accuracy on a test set of 1,000 labeled examples."
- [paper] other: 500 labeled examples validation set — "We choose the same dataset splits as in Yang et al. (2016) with an additional validation set of 500 labeled examples for hyperparameter optimization (dropout rate for all layers, L2 regularization factor for the first GCN layer and number of hidden units)."
- [paper] other: validation set labels not used for training — "We do not use the validation set labels for training."
- [paper] other: hyperparameters optimized on Cora only — "For the citation network datasets, we optimize hyperparameters on Cora only and use the same set of parameters for Citeseer and Pubmed."
- [paper] other: Glorot & Bengio weight initialization — "We initialize weights using the initialization described in Glorot & Bengio (2010) and accordingly (row-)normalize input feature vectors."
- [repo:requirements.txt@39a4089] package_version: networkx==2.2 — "networkx==2.2"
- [repo:gcn/train.py@39a4089] seed: 123 — "seed = 123"
- [repo:gcn/train.py@39a4089] dataset: 'cora' — "flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'"
- [repo:gcn/train.py@39a4089] model: 'gcn' — "flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'"
- [repo:gcn/train.py@39a4089] early_stopping: 10 — "flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')"
- [repo:gcn/train.py@39a4089] max_degree: 3 — "flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')"
- [repo:gcn/models.py@39a4089] other settings: learning_rate — "self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)"
- [repo:gcn/models.py@39a4089] other settings: weight_decay — "self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)"
- [repo:gcn/models.py@39a4089] other settings: hidden1 — "output_dim=FLAGS.hidden1"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch