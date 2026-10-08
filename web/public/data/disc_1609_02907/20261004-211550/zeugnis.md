# Reproduktions-Zeugnis: disc_1609_02907

## Claim disc_1609_02907_accuracy: reproduziert
Gemessen: 0.8153 (std 0.0106)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8290 mit Graph, 0.5950 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0106. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.815 +/- 0.01, Table 2, arXiv v4 (pdf)): Mittel 0.8153. Code-Kandidaten: 3, gewaehlt Kandidat 2 (Code 7d33525f97cb) nach Validierungs-Metrik (0: accuracy 0.7780, 1: accuracy 0.7900, 2: accuracy 0.8040); Repo-Abgleich: keine geprueften Abweichungen
Zielwert 0.815 belegt: "GCN (this paper) 70 .3 (7s) 81 .5 (4s) 79 .0 (38s) 66 .0 (48s)" (Table 2, https://arxiv.org/pdf/1609.02907, Version v4, abgerufen 2026-09-28T19:12:22+00:00, SHA-256 818c55c77cfd)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1609.02907

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-02T09:02:10+00:00 (cache), 7 Seiten gelesen, 2 Aussagen verworfen (Zitat nicht im Text).

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/tkipf/gcn @ 39a4089fe7

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] Data preprocessing: construct a binary, symmetric adjacency matrix A — "We treat the citation links as (undirected) edges and construct a binary, symmetric adjacency matrix A."
- [paper] Feature representation: sparse bag-of-words feature vectors for each document — "The datasets contain sparse bag-of-words feature vectors for each document and a list of citation links between documents."
- [paper] Training label usage: For training, we only use 20 labels per class, but all feature vectors. — "For training, we only use 20 labels per class, but all feature vectors."
- [paper] model_layers: two-layer GCN — "Unless otherwise noted, we train a two-layer GCN as described in Section 3.1 and evaluate pre-diction accuracy on a test set of 1,000 labeled examples."
- [paper] dataset_split_validation: validation set of 500 labeled examples for hyperparameter optimization — "We choose the same dataset splits as in Yang et al. (2016) with an additional validation set of 500 labeled examples for hyperparameter optimization"
- [paper] dataset_split_training: remaining labeled examples used for training (validation labels not used for training) — "We do not use the validation set labels for training."
- [paper] hyperparameter_dropout: 0.5 — "We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 0.5 (dropout rate), 5 · 10 −4 (L2 regularization) and 16 (number of hid-den units);"
- [paper] dropout_application: applied to all layers — "dropout rate for all layers"
- [paper] l2_regularization_layer: first GCN layer — "L2 regularization factor for the first GCN layer"
- [paper] optimizer: Adam — "We train all models for a maximum of 200 epochs (training iterations) using Adam (Kingma & Ba, 2015) with a learning rate of 0.01 and early stopping with a window size of 10 , i.e. we stop training if the validation loss does not decrease for 10 consecutive epochs."
- [paper] weight_initialization: Glorot & Bengio (2010) — "We initialize weights using the initialization described in Glorot & Bengio (2010) and accordingly (row-)normalize input feature vectors."
- [paper] evaluation protocol: mean classification accuracy for 100 repeated runs with random weight matrix initializations — "Reported numbers denote mean classification accuracy for 100 repeated runs with random weight matrix initializations."
- [paper] regularisation: L2 regularization imposed on all weight matrices of the first layer — "In case of multiple variables Θi per layer, we impose L2 regularization on all weight matrices of the first layer."
- [paper] model architecture: two-layer GCN (propagation model applied to both neural network layers) — "the propagation model of both neural network layers is replaced with the model specified under propagation model"
- [paper] data split: 5‑fold cross‑validation — "we report results on a 5‑fold cross-validation experiment on the Cora, Citeseer and Pubmed datasets (Sen et al., 2008) using all labels."
- [paper] training epochs: 400 epochs (no early stopping) — "On each cross-validation split, we train for 400 epochs (without early stopping) using the Adam optimizer"
- [paper] optimizer: Adam — "using the Adam optimizer (Kingma & Ba, 2015) with a learning rate of 0.01"
- [paper] learning rate: 0.01 — "with a learning rate of 0.01 . Other hyperparameters are chosen as follows: 0.5 (dropout rate, first and last layer), 5 · 10 −4 (L2 regularization, first layer), 16 (number of units for each hidden layer) and 0.01 (learning rate)."
- [paper] dropout: 0.5 applied to first and last layer — "0.5 (dropout rate, first and last layer)"
- [paper] L2 regularization: 5·10⁻⁴ applied to first layer — "5 · 10 −4 (L2 regularization, first layer)"
- [paper] hidden units per layer: 16 — "16 (number of units for each hidden layer)"
- [paper] early stopping: none (training without early stopping) — "we train for 400 epochs (without early stopping)"
- [repo:gcn/models.py@39a4089] loss function: masked_softmax_cross_entropy — "self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])"
- [repo:gcn/models.py@39a4089] accuracy metric: masked_accuracy — "self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])"
- [repo:gcn/models.py@39a4089] model architecture: Two-layer GraphConvolution: first layer ReLU activation, second layer linear activation — "self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                    output_dim=self.output_dim,
                                    placeholders=self.placeholders,
                                    act=lambda x: x,
                                    dropout=True,
                                    logging=self.logging))"
- [repo:gcn/models.py@39a4089] dropout application: Applied in both GraphConvolution layers (dropout=True) — "dropout=True"
- [repo:gcn/models.py@39a4089] sparse inputs handling: First GraphConvolution layer uses sparse_inputs=True — "sparse_inputs=True"
- [repo:gcn/layers.py@39a4089] weight_initialization: Glorot (Xavier) initializer — "self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))"
- [repo:gcn/layers.py@39a4089] bias_initialization: zeros initializer — "self.vars['bias'] = zeros([output_dim], name='bias')"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch