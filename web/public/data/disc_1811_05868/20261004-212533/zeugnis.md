# Reproduktions-Zeugnis: disc_1811_05868

## Hinweise zur Belegbarkeit
- Info: Das Paper nennt Streuung +-1.3 (ueber Splits/Laeufe, nicht Toleranz des Mittelwerts); die Toleranz +-0.01 bleibt fest.

## Claim disc_1811_05868_accuracy: teilweise
Gemessen: 0.8049 (std 0.0158)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.7902 mit Graph, 0.5925 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0158. Mit unserem Setup gemessen (30 von 30 Seeds, Zielwert 0.815 +/- 0.01, Table 1, arXiv v2 (pdf)): Mittel 0.8049. Code-Kandidaten: 3, gewaehlt Kandidat 2 (Code 1c049829bfa9) nach Validierungs-Metrik (0: accuracy 0.8667, 1: accuracy 0.8571, 2: accuracy 0.8762); Repo-Abgleich: 1 Abweichungen, Reparatur nicht uebernommen (Validierung schlechter als beim alten Code)
Zielwert 0.815 belegt: "GCN 81 .5 ± 1.3 71.9 ± 1.9 77 .8 ± 2.9 62.2 ± 0.6" (Table 1, https://arxiv.org/pdf/1811.05868v2, Version v2, abgerufen 2026-10-04T19:25:35+00:00, SHA-256 47ac4b09ea8c)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1811.05868v2

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-04T19:34:15+00:00 (live), 9 Seiten gelesen, 9 Aussagen verworfen (Zitat nicht im Text).

## Possible explanations für disc_1811_05868_accuracy (Tavily Rolle 6, ungeprüfte Hypothesen, nicht Teil des Urteils)
- The paper points out that prior attacks use a validation set much larger than the training set, which is not realistic and can affect reported numbers. — "All existing label poisoning attacks (LPATTACK, LAFAK, MG) evaluate on data splits with validation (val) set size much larger than that of training (train) set." (https://openreview.net/pdf/79e770b6f8016087b03a0131c7e2c33e8dba0aae.pdf)
- The paper notes that existing attacks use default hyper‑parameters tuned for clean labels, which may not be appropriate for poisoned data and can lead to different results. — "Despite establishing this, all the label-poisoning attacks for GNNs still use default h-params to evaluate the effectiveness of the proposed attack. The default h-params were tuned to work with clean labels (unpoisoned)." (https://openreview.net/pdf/79e770b6f8016087b03a0131c7e2c33e8dba0aae.pdf)
- Treating the validation set as always clean is described as an unrealistic assumption that can distort evaluation outcomes. — "It is an unrealistic assumption for an attacker to treat train and val sets as separate entities. In other words, the current evaluation setup in literature for label-poisoning attacks training nodes while the val set is untouched and remains sanitized." (https://openreview.net/pdf/79e770b6f8016087b03a0131c7e2c33e8dba0aae.pdf)
- The paper highlights that variance in results is large, often comparable to the claimed improvement, which can hinder reproducibility. — "We see that the variance is relatively large, often of the same order as the supposed improvement by an attack." (https://openreview.net/pdf/79e770b6f8016087b03a0131c7e2c33e8dba0aae.pdf)
- Different GNN layers consider neighborhoods of different sizes, causing incomparable evaluations unless adjusted. — "The first source of inconsistency when comparing GNNs comes from the fact that different layers are designed to take into account neighborhoods of different sizes." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- A fair evaluation should ensure all models see the same neighborhoods, but this is often ignored. — "A fair evaluation should keep these differences into account and allow each GNN to look at the same neighborhoods, but at the same time, it could be argued that a layer designed to operate on larger neighborhoods is more expressive." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Variations in neighborhood size affect parameters, nonlinearity, and regularization, leading to performance differences. — "Moreover, this difference between methods may have an impact on the number of parameters, nonlinearity, and overall amount of regularization in a GNN." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Equivalent neighborhood reach may require different numbers of dropout layers, altering regularization. — "For instance, a GCN that reaches a neighborhood of order 3 may have 3 dropout layers, while the equivalent Cheby with K=3 will have only one." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- SGC can reach any neighborhood with constant parameters unlike other methods, affecting comparability. — "Another example: an SGC architecture can reach any neighborhood with a constant number of parameters, while other methods can’t." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- The literature applies regularization inconsistently, making it unclear whether gains are due to model or regularization. — "The literature is largely inconsistent with how regularization is applied across different papers, making it difficult to say whether the performance improvements reported for a model are due to the actual contribution or to a different regularization scheme." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Frequent use of high learning rates, high L2 penalty, extreme dropout, low epochs, and low early‑stopping patience introduces variability. — "High learning rates; High L2 penalty; Extremely high dropout rates on node features and adjacency matrix; Low number of training epochs; Low patience for early stopping." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Using a fixed number of epochs is task‑specific and can cause results to depend heavily on initialization. — "First, I argue that setting a fixed number of training epochs is a form of alchemy that should be avoided if possible, because it’s incredibly task-specific. Letting a model train to convergence is almost always a better approach, because it’s less dependent on the initialization of the weights." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Applying dropout to the adjacency matrix improves generalization, but papers differ on whether they use it, causing unfair comparisons. — "When comparing different models, if one is using dropout on the adjacency matrix then all the others should do the same. However, the common practice of comparing methods using the “same architecture from the original paper” means that some methods will be tested with dropout on A, and some without, as if the dropout is a particular characteristic of only some methods." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Learning rates and weight decay are often taken as‑is without tuning, yet they significantly affect performance. — "Finally, the remaining key factors in training are the learning rate and weight decay. These are often given as-is in the literature, but it is a good idea to tune them whenever possible." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Fundamental differences between head‑based models (e.g., GAT) and filter‑based models (e.g., ARMA) make head‑count comparisons unfair. — "Because of these fundamental semantic differences, it is impossible to say whether a comparison between GAT with H heads and an ARMA layer of order H is fair." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Benchmark datasets like Cora are insufficient for fair evaluation, leading to unreliable results. — "Finally, I’m going to spend a few words on datasets, because there is no chance of having a fair evaluation if the datasets on which we test our models are not good. And in truth, the benchmark datasets that we use for evaluating GNNs are not that good." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Experiments often require hundreds of repetitions to obtain significant results, indicating instability and reproducibility issues. — "However, many experiments in the literature still had to be repeated hundreds of times in order to give significant results, and that is bad for three reasons: time, money, and the environment, in no particular order." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)
- Grid search on these datasets demands excessive computation, making reliable outcomes hard to achieve. — "Especially if running a grid search of hyperparameters, it just doesn’t make sense to be using datasets that require that much computation to give reliable outcomes, more so if we consider that these are supposed to be easy datasets." (https://danielegrattarola.github.io/posts/2019-12-13/pitfalls.html)

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/shchur/gnn-benchmark @ 1e72912a08

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] train/validation/test splits: 100 random train/validation/test splits per dataset — "For each dataset we use 100 random train/validation/test splits and perform 20 random initializations for each split."
- [paper] data_split: 20 labeled nodes per class for training, 30 nodes per class for validation, rest for test — "In all cases, we use 20 labeled nodes per class as the training set, 30 nodes per class as the validation set, and the rest as the test set."
- [paper] evaluation_protocol: Mean test accuracy averaged over 100 random train/validation/test splits with 20 random weight initializations each — "Table 1: Mean test set accuracy and standard deviation in percent averaged over 100 random train/validation/test splits with 20 random weight initializations each for all models and all datasets."
- [paper] hyperparameter_search: Grid search over learning rate, hidden layer size, L2 regularization strength, dropout probability — "We performed an extensive grid search for learning rate, size of the hidden layer, strength of the L2 regularization, and dropout probability (Appendix C)."
- [paper] model_selection_criterion: Chosen hyperparameter configuration that achieved best average accuracy on Cora and CiteSeer (averaged over 100 splits and 20 initializations) — "For every model, we picked the hyperparameter configuration that achieved the best average accuracy on Cora and CiteSeer datasets (averaged over 100 train/validation/test splits and 20 random initializations for each)."
- [paper] config_usage: Best-performing configurations from Table 4 used for all subsequent experiments — "The chosen best‑performing configurations were used for all subsequent experiments and are listed in Table 4."
- [paper] GCN early stopping (original): stop optimization if the validation loss is larger than the mean of validation losses of the last 10 epochs — "Early stopping: stop optimization if the validation loss is larger than the mean of validation losses of the last 10 epochs."
- [paper] GCN training mode: Full-batch training — "Full-batch training."
- [paper] GCN max epochs (original): 200 — "Maximum number of epochs: 200."
- [paper] GCN data split: 20 per class for training; 500 nodes validation; 1000 nodes test (Planetoid split) — "Train set: 20 per class; validation set: 500 nodes; test set: 1000 (as in the Planetoid split)."
- [paper] GCN unified early stopping: stop if total validation loss does not improve for 50 epochs; reset weights to best validation loss — "we train for a maximum of 100 k epochs. However, the actual training time is considerably shorter since we use strict early stopping. Specifically, with our unified early stopping criterion training stops if the total validation loss (loss on the data plus regularization loss) does not improve for 50 epochs. Once training has stopped, we reset the state of the weights to the step with the lowest validation loss."
- [repo:config/optimized/gcn.conf.yaml@1e72912] Model name: "GCN" — "model_name: "GCN""
- [repo:config/optimized/gcn.conf.yaml@1e72912] Number of layers: 2 — "num_layers: 2"
- [repo:config/optimized/gcn.conf.yaml@1e72912] Hidden size: 64 — "hidden_size: 64"
- [repo:config/optimized/gcn.conf.yaml@1e72912] Dropout probability: 0.8 — "dropout_prob: 0.8"
- [repo:config/optimized/gcn.conf.yaml@1e72912] Learning rate: 0.01 — "learning_rate: 0.01"
- [repo:config/optimized/gcn.conf.yaml@1e72912] Weight decay: 0.001 — "weight_decay: 0.001"
- [repo:config/reference/gcn.conf.yaml@1e72912] model_name: GCN — "model_name: "GCN""
- [repo:config/reference/gcn.conf.yaml@1e72912] num_layers: 2 — "num_layers: 2"
- [repo:config/reference/gcn.conf.yaml@1e72912] hidden_size: 16 — "hidden_size: 16"
- [repo:config/reference/gcn.conf.yaml@1e72912] dropout_prob: 0.5 — "dropout_prob: 0.5"
- [repo:config/reference/gcn.conf.yaml@1e72912] learning_rate: 0.01 — "learning_rate: 0.01"
- [repo:gnnbench/data/make_dataset.py@1e72912] adjacency_preprocessing: Make graph undirected and remove self-loops when standardize is False — "else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)"
- [repo:gnnbench/data/make_dataset.py@1e72912] adjacency_preprocessing: Apply standardization (unknown transformation) when standardize is True — "if standardize:
        dataset_graph = dataset_graph.standardize()"
- [repo:gnnbench/data/make_dataset.py@1e72912] label_processing: Binarize labels — "labels = binarize_labels(labels)"
- [repo:gnnbench/data/make_dataset.py@1e72912] split_function: Function to generate train/val/test splits supporting per-class or size-based specification — "def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):"
- [repo:gnnbench/data/make_dataset.py@1e72912] split_coverage: If test size not specified, ensure all indices are used in split — "if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples"
- [repo:gnnbench/data/make_dataset.py@1e72912] split_balance_check: When using per-class train split, assert each class has equal number of training examples — "if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1"
- [repo:gnnbench/data/make_dataset.py@1e72912] class-balanced sampling: selects num_examples_per_class per class without replacement — "return np.concatenate([random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False) for class_index in range(len(sample_indices_per_class))])"
- [repo:gnnbench/data/make_dataset.py@1e72912] validation cardinality check: assert all classes have equal cardinality — "# assert all classes have equal cardinality
assert np.unique(val_sum).size == 1"
- [repo:gnnbench/data/make_dataset.py@1e72912] test cardinality check: assert all classes have equal cardinality — "# assert all classes have equal cardinality
assert np.unique(test_sum).size == 1"
- [repo:gnnbench/data/make_dataset.py@1e72912] data type casting: adjacency and features cast to float32 — "adj = adj.astype(np.float32)
features = features.tocsr()
features = features.astype(np.float32)"
- [repo:gnnbench/early_stopping.py@1e72912] Early stopping improvement tolerance: Reset patience only if validation accuracy improves by at least tolerance or validation loss improves by at least tolerance — "# But only reset patience if accuracy or loss improved by a certain degree. This avoids long-running
            # convergence processes like for the LabelPropagation algorithm.
            if val_accuracy >= self.val_accuracy_max + self.tolerance or val_loss <= self.val_loss_min - self.tolerance:
                self.patience_step = 0"
- [repo:gnnbench/early_stopping.py@1e72912] Patience increment: Otherwise increment patience step — "else:
                self.patience_step += 1"
- [repo:gnnbench/early_stopping.py@1e72912] Stopping condition: Training stops when patience_step reaches self.patience — "return self.patience_step >= self.patience"
- [repo:gnnbench/early_stopping.py@1e72912] Variable reset for best weights: Functions to extract and restore trainable variables to the state at minimum validation loss — "def get_reset_variable_ops():
    # Operations for resetting the state of trainable variables to the one at the minimum val loss step.
    with tf.name_scope('reset_variable_state_ops'):
        extract_variables_state_op = extract_variables_state()
        variables_state = [tf.placeholder(var.dtype, var.shape) for var in tf.trainable_variables()]
        set_trainable_variables_op = set_trainable_variables(variables_state)
    return extract_variables_state_op, variables_state, set_trainable_variables_op"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch