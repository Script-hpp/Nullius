# Reproduktions-Zeugnis: kipf_gcn_2017

## Claim gcn_cora_acc: reproduziert
Gemessen: 0.8151 (std 0.0107)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8290 mit Graph, 0.5950 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0107. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.815 +/- 0.01, Table 2, arXiv v4 (pdf)): Mittel 0.8151. Code-Kandidaten: 3, gewaehlt Kandidat 2 (Code 66882439fb64) nach Validierungs-Metrik (0: accuracy 0.7720, 1: no val value, 2: accuracy 0.8040); Repo-Abgleich: keine geprueften Abweichungen
Zielwert 0.815 belegt: "GCN (this paper) 70 .3 (7s) 81 .5 (4s) 79 .0 (38s) 66 .0 (48s)" (Table 2, https://arxiv.org/pdf/1609.02907, Version v4, abgerufen 2026-09-28T19:12:22+00:00, SHA-256 818c55c77cfd)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1609.02907

## Claim gcn_cora_acc_placebo: nicht reproduziert
Gemessen: 0.8151 (std 0.0107)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8290 mit Graph, 0.5950 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0107. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.88 +/- 0.01, PLACEBO: konstruierter falscher Zielwert, kein Paper-Wert): Mittel 0.8151. Code-Kandidaten: 3, gewaehlt Kandidat 1 (Code 0b1d7ea0c76d) nach Validierungs-Metrik (0: accuracy 0.7920, 1: accuracy 0.8040, 2: accuracy 0.8040); Repo-Abgleich: keine geprueften Abweichungen
Zielwert 0.88: PLACEBO: konstruierter falscher Zielwert, kein Paper-Wert

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-02T09:02:10+00:00 (cache+live), 7 Seiten gelesen, 2 Aussagen verworfen (Zitat nicht im Text).

## Possible explanations für gcn_cora_acc_placebo (Tavily Rolle 6, ungeprüfte Hypothesen, nicht Teil des Urteils)
None found.

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/tkipf/gcn @ 39a4089fe7

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] model architecture: two-layer GCN with ReLU after first layer and softmax output — "Our forward model then takes the simple form: Z = f (X, A ) = softmax ( ˆA ReLU ( ˆAXW (0) ) W (1) )."
- [paper] optimizer / training procedure: batch gradient descent using the full dataset each iteration — "The neural network weights W (0) and W (1) are trained using gradient descent. In this work, we perform batch gradient descent using the full dataset for every training iteration..."
- [paper] dropout usage: dropout introduced for stochasticity — "Stochasticity in the training process is introduced via dropout (Srivastava et al., 2014)."
- [paper] model architecture: two-layer GCN — "we train a two-layer GCN as described in Section 3.1"
- [paper] test set size: 1,000 labeled examples — "evaluate prediction accuracy on a test set of 1,000 labeled examples."
- [paper] dataset splits: same as Yang et al. (2016) with additional validation set of 500 labeled examples — "We choose the same dataset splits as in Yang et al. (2016) with an additional validation set of 500 labeled examples for hyperparameter optimization"
- [paper] validation set usage: not used for training — "We do not use the validation set labels for training."
- [paper] hyperparameter optimization scope: optimized on Cora only, same params for Citeseer and PubMed — "For the citation network datasets, we optimize hyperparameters on Cora only and use the same set of parameters for Citeseer and Pubmed."
- [paper] max epochs: 200 epochs — "We train all models for a maximum of 200 epochs (training iterations)"
- [paper] optimizer: Adam — "using Adam (Kingma & Ba, 2015)"
- [paper] learning rate: 0.01 — "with a learning rate of 0.01"
- [paper] early stopping criterion: window size 10 (stop if validation loss does not decrease for 10 consecutive epochs) — "and early stopping with a window size of 10, i.e. we stop training if the validation loss does not decrease for 10 consecutive epochs."
- [paper] weight initialization: Glorot & Bengio (2010) initialization — "We initialize weights using the initialization described in Glorot & Bengio (2010)"
- [paper] feature normalization: (row-)normalize input feature vectors — "and accordingly (row-)normalize input feature vectors."
- [paper] dropout rate: 0.5 (applied to all layers) — "0.5 (dropout rate)"
- [paper] L2 regularization factor: 5·10−4 (for first GCN layer) — "5 · 10 −4 (L2 regularization)"
- [paper] number of hidden units: 16 — "and 16 (number of hidden units)"
- [paper] evaluation protocol – repetitions: mean accuracy of 100 runs with random weight initializations — "We trained and tested our model on the same dataset splits as in Yang et al. (2016) and report mean accuracy of 100 runs with random weight initializations."
- [paper] metric: classification accuracy in percent — "Reported numbers denote classification accuracy in percent."
- [paper] cross_validation: 5-fold cross-validation experiment on Cora, Citeseer and Pubmed datasets — "In these experiments, we investigate the influence of model depth (number of layers) on classification performance. We report results on a 5-fold cross-validation experiment on the Cora, Citeseer and Pubmed datasets (Sen et al., 2008) using all labels."
- [paper] epochs: 400 epochs (without early stopping) — "On each cross-validation split, we train for 400 epochs (without early stopping) using the Adam optimizer (Kingma & Ba, 2015) with a learning rate of 0.01."
- [paper] dropout_rate: 0.5 (dropout rate, first and last layer) — "Other hyperparameters are chosen as follows: 0.5 (dropout rate, first and last layer), 5 · 10 −4 (L2 regularization, first layer), 16 (number of units for each hidden layer) and 0.01 (learning rate)."
- [repo:gcn/models.py@39a4089] Loss function: masked_softmax_cross_entropy — "self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])"
- [repo:gcn/models.py@39a4089] Accuracy metric: masked_accuracy — "self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])"
- [repo:gcn/models.py@39a4089] Model architecture - first GraphConvolution layer: input_dim=self.input_dim, output_dim=FLAGS.hidden1, act=tf.nn.relu, dropout=True, sparse_inputs=True — "self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))"
- [repo:gcn/models.py@39a4089] Model architecture - second GraphConvolution layer: input_dim=FLAGS.hidden1, output_dim=self.output_dim, act=lambda x: x, dropout=True — "self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))"
- [repo:gcn/models.py@39a4089] Prediction: tf.nn.softmax(self.outputs) — "return tf.nn.softmax(self.outputs)"
- [repo:gcn/layers.py@39a4089] weight_initialization: glorot([input_dim, output_dim]) — "self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))"
- [repo:gcn/layers.py@39a4089] bias_initialization: zeros([output_dim]) — "self.vars['bias'] = zeros([output_dim], name='bias')"
- [repo:gcn/layers.py@39a4089] dropout_application: applied to inputs: sparse_dropout if sparse_inputs else tf.nn.dropout with rate 1-self.dropout — "# dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch