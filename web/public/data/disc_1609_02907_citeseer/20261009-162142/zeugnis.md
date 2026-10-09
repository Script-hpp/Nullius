# Reproduktions-Zeugnis: disc_1609_02907_citeseer

## Claim disc_1609_02907_citeseer_accuracy: reproduziert
Gemessen: 0.7086 (std 0.0075)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.7000 mit Graph, 0.5490 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0075. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.703 +/- 0.01, Table 2, arXiv v4 (pdf)): Mittel 0.7086. Code-Kandidaten: 3, gewaehlt Kandidat 1 (Code 23afd088dad7) nach Validierungs-Metrik (0: accuracy 0.6940, 1: accuracy 0.7080, 2: accuracy 0.7020); Rueckfall ohne Val-Wert; Repo-Abgleich: keine geprueften Abweichungen
Zielwert 0.703 belegt: "GCN (this paper) 70.3 (7s) 81.5 (4s) 79.0 (38s) 66.0 (48s)" (Table 2, https://arxiv.org/pdf/1609.02907v4, Version v4, abgerufen 2026-10-09T14:21:42+00:00, SHA-256 c919b23efc1d)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1609.02907v4

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-09T14:21:17+00:00 (cache), 6 Seiten gelesen, 1 Aussagen verworfen (Zitat nicht im Text).

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/tkipf/gcn @ 39a4089fe7

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:gcn/train.py@39a4089] feature normalisation: features = preprocess_features(features) — "# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)"
- [repo:gcn/utils.py@39a4089] feature normalisation: Row-normalize feature matrix (divide each row by its sum) — "def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)"
- [repo:gcn/train.py@39a4089] adjacency normalisation: preprocess_adj — "support = [preprocess_adj(adj)]"
- [paper] learning rate: 0.01 — "We train all models for a maximum of 200 epochs (training iterations) using Adam (Kingma & Ba, 2015) with a learning rate of 0.01 and early stopping with a window size of 10, i.e. we stop training if the validation loss does not decrease for 10 consecutive epochs."
- [repo:gcn/train.py@39a4089] learning rate: 0.01 — "flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')"
- [paper] weight decay: 5 · 10−4 — "We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 0.5 (dropout rate), 5 · 10−4 (L2 regularization) and 16 (number of hidden units);"
- [repo:gcn/train.py@39a4089] weight decay: 5e-4 — "flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')"
- [repo:gcn/models.py@39a4089] weight decay: FLAGS.weight_decay — "self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)"
- [paper] dropout: 0.5 — "We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 0.5 (dropout rate), 5 · 10−4 (L2 regularization) and 16 (number of hidden units); and for NELL: 0.1 (dropout rate), 1 · 10−5 (L2 regularization) and 64 (number of hidden units)."
- [repo:gcn/train.py@39a4089] dropout: 0.5 — "flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')"
- [repo:gcn/models.py@39a4089] dropout: True — "dropout=True,"
- [paper] hidden size: 16 — "We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 0.5 (dropout rate), 5 · 10−4 (L2 regularization) and 16 (number of hidden units); and for NELL:"
- [repo:gcn/train.py@39a4089] hidden_size: 16 — "flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')"
- [repo:gcn/train.py@39a4089] epochs: 200 — "flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')"
- [paper] other settings: two-layer GCN architecture — "Unless otherwise noted, we train a two-layer GCN as described in Section 3.1 and evaluate prediction accuracy on a test set of 1,000 labeled examples."
- [paper] other settings: validation set of 500 labeled examples for hyperparameter optimization — "We choose the same dataset splits as in Yang et al. (2016) with an additional validation set of 500 labeled examples for hyperparameter optimization (dropout rate for all layers, L2 regularization factor for the first GCN layer and number of hidden units)."
- [paper] other settings: validation set labels not used for training — "We do not use the validation set labels for training."
- [paper] other settings: hyperparameters optimized on Cora only and transferred to Citeseer — "For the citation network datasets, we optimize hyperparameters on Cora only and use the same set of parameters for Citeseer and Pubmed."
- [paper] other settings: weight initialization via Glorot & Bengio (2010) and row‑normalization of input features — "We initialize weights using the initialization described in Glorot & Bengio (2010) and accordingly (row-)normalize input feature vectors."
- [paper] other settings: mean accuracy reported over 100 runs with random weight initializations — "We trained and tested our model on the same dataset splits as in Yang et al. (2016) and report mean accuracy of 100 runs with random weight initializations."
- [repo:gcn/train.py@39a4089] random_seed: 123 — "# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)"
- [repo:gcn/train.py@39a4089] dataset: 'cora' (default, can be overridden to 'citeseer') — "flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'"
- [repo:gcn/train.py@39a4089] model: 'gcn' — "flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'"
- [repo:gcn/train.py@39a4089] early_stopping: 10 — "flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')"
- [repo:gcn/utils.py@39a4089] validation set size: 500 — "idx_val = range(len(y), len(y)+500)"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch