# Reproduktions-Zeugnis: kipf_gcn_2017

## Claim gcn_cora_acc: reproduziert
Gemessen: 0.8211 (std 0.0072)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8280 mit Graph, 0.5700 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0072. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.815 +/- 0.01, Table 2, arXiv v4 (pdf)): Mittel 0.8211.
Zielwert 0.815 belegt: "GCN (this paper) 70 .3 (7s) 81 .5 (4s) 79 .0 (38s) 66 .0 (48s)" (Table 2, https://arxiv.org/pdf/1609.02907, Version v4, abgerufen 2026-09-28T19:12:22+00:00, SHA-256 818c55c77cfd)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1609.02907

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/tkipf/gcn @ 39a4089fe7

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] optimizer: batch gradient descent using the full dataset for every training iteration — "In this work, we perform batch gradient descent using the full dataset for every training iteration, which is a viable option as long as datasets fit in memory."
- [paper] regularisation: dropout (rate unspecified) — "Stochasticity in the training process is introduced via dropout (Srivastava et al., 2014)."
- [paper] training data proportion: 5% of labels used for training — "visualization of hidden layer activations of a two-layer GCN trained on the Cora dataset (Sen et al., 2008) using 5% of labels."
- [paper] data_preprocessing: construct binary, symmetric adjacency matrix from undirected citation links — "We treat the citation links as (undirected) edges and construct a binary, symmetric adjacency matrix A."
- [paper] data_splits: use 20 labeled nodes per class for training (all feature vectors used) — "For training, we only use 20 labels per class, but all feature vectors."
- [paper] model_architecture: single weight matrix per layer with adjacency normalization — "Our model instead uses a single weight matrix per layer and deals with varying node degrees through an appropriate normalization of the adjacency matrix (see Section 3.1)."
- [paper] model architecture: two-layer GCN — "Unless otherwise noted, we train a two-layer GCN as described in Section 3.1 and evaluate pre-diction accuracy on a test set of 1,000 labeled examples."
- [paper] data splits: test set of 1,000 labeled examples; validation set of 500 labeled examples; training set remainder (same as Yang et al. 2016) — "We choose the same dataset splits as in Yang et al. (2016) with an additional validation set of 500 labeled examples for hyperparameter optimization"
- [paper] data splits usage: validation set labels not used for training — "We do not use the validation set labels for training."
- [paper] feature preprocessing: row-normalize input feature vectors — "We initialize weights using the initialization described in Glorot & Bengio (2010) and accordingly (row-)normalize input feature vectors."
- [paper] hyperparameters (Cora): dropout rate = 0.5; L2 regularization = 5·10⁻4; hidden units = 16 — "We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 0.5 (dropout rate), 5 · 10 −4 (L2 regularization) and 16 (number of hid-den units);"
- [paper] optimizer: Adam with learning rate 0.01 — "We train all models for a maximum of 200 epochs (training iterations) using Adam (Kingma & Ba, 2015) with a learning rate of 0.01 and early stopping with a window size of 10"
- [paper] regularisation: dropout (rate 0.5) on all layers; L2 regularization (5·10⁻4) on first GCN layer — "We choose the same dataset splits as in Yang et al. (2016) with an additional validation set of 500 labeled examples for hyperparameter optimization (dropout rate for all layers, L2 regularization factor for the first GCN layer and number of hidden units)."
- [paper] training length and stopping: max 200 epochs; early stopping if validation loss does not improve for 10 consecutive epochs — "We train all models for a maximum of 200 epochs (training iterations) using Adam (Kingma & Ba, 2015) with a learning rate of 0.01 and early stopping with a window size of 10 , i.e. we stop training if the validation loss does not decrease for 10 consecutive epochs."
- [paper] evaluation protocol: mean accuracy over 100 runs with random weight initializations on the standard test set — "We trained and tested our model on the same dataset splits as in Yang et al. (2016) and report mean accuracy of 100 runs with random weight initializations."
- [paper] model architecture: two-layer GCN — "In all other cases, the propagation model of both neural network layers is replaced with the model specified under propagation model ."
- [paper] propagation model: renormalization trick (Eq. 8) ˜D−1/2 ˜A ˜D−1/2 XΘ — "Renormalization trick (Eq. 8) ˜D− 12 ˜A ˜D− 12 XΘ"
- [paper] weight initialization: random weight matrix initializations — "Reported numbers denote mean classification accuracy for 100 repeated runs with random weight matrix initializations."
- [paper] regularisation: L2 regularization on all weight matrices of the first layer (when multiple Θi per layer) — "In case of multiple variables Θi per layer, we impose L2 regularization on all weight matrices of the first layer."
- [paper] data split: 5-fold cross-validation — "we report results on a 5-fold cross-validation experiment on the Cora, Citeseer and Pubmed datasets (Sen et al., 2008) using all labels."
- [paper] dropout rate: 0.5 (first and last layer) — "Other hyperparameters are chosen as follows: 0.5 (dropout rate, first and last layer), 5 · 10 −4 (L2 regularization, first layer), 16 (number of units for each hidden layer) and 0.01 (learning rate)."
- [repo:gcn/models.py@39a4089] optimizer: tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) — "self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)"
- [repo:gcn/models.py@39a4089] weight_decay: FLAGS.weight_decay * tf.nn.l2_loss(var) — "self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)"
- [repo:gcn/models.py@39a4089] hidden_layer_size: FLAGS.hidden1 — "self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))"
- [repo:gcn/models.py@39a4089] first_layer_activation: tf.nn.relu — "act=tf.nn.relu"
- [repo:gcn/models.py@39a4089] second_layer_activation: lambda x: x — "act=lambda x: x"
- [repo:gcn/models.py@39a4089] dropout: True — "dropout=True"
- [repo:gcn/models.py@39a4089] sparse_inputs: True — "sparse_inputs=True"
- [repo:gcn/models.py@39a4089] loss_function: masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask']) — "self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])"
- [repo:gcn/models.py@39a4089] accuracy_metric: masked_accuracy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask']) — "self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])"
- [repo:gcn/models.py@39a4089] model architecture: 2 GraphConvolution layers — "self.layers.append(GraphConvolution(input_dim=self.input_dim, output_dim=FLAGS.hidden1, placeholders=self.placeholders, act=tf.nn.relu, dropout=True, sparse_inputs=True, logging=self.logging))"
- [repo:gcn/models.py@39a4089] prediction: softmax of outputs — "return tf.nn.softmax(self.outputs)"
- [repo:gcn/layers.py@39a4089] weight_initialization: glorot (Xavier) initialization — "self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))"
- [repo:gcn/layers.py@39a4089] bias_initialization: zeros — "self.vars['bias'] = zeros([output_dim], name='bias')"
- [repo:gcn/layers.py@39a4089] dropout: applied to inputs with keep probability 1-self.dropout (sparse_dropout if sparse_inputs else tf.nn.dropout) — "if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)"
- [repo:gcn/layers.py@39a4089] bias_inclusion: bias added if self.bias is True — "if self.bias:
            output += self.vars['bias']"
- [repo:gcn/layers.py@39a4089] activation_function: self.act applied after bias — "return self.act(output)"
- [repo:gcn/layers.py@39a4089] logging_variables: log variables when self.logging is True — "if self.logging:
            self._log_vars()"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch