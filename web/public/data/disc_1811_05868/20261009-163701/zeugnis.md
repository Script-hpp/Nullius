# Reproduktions-Zeugnis: disc_1811_05868

## Hinweise zur Belegbarkeit
- Info: Das Paper nennt Streuung +-1.3 (ueber Splits/Laeufe, nicht Toleranz des Mittelwerts); die Toleranz +-0.01 bleibt fest.

## Claim disc_1811_05868_accuracy: teilweise
Gemessen: 0.7998 (std 0.0164)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.7794 mit Graph, 0.5452 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0164. Mit unserem Setup gemessen (30 von 30 Seeds, Zielwert 0.815 +/- 0.01, Table 1, arXiv v2 (pdf)): Mittel 0.7998. Code-Kandidaten: 3, gewaehlt Kandidat 1 (Code eedfcfdc4dc0) nach Validierungs-Metrik (0: accuracy 0.8619, 1: accuracy 0.8381, 2: accuracy 0.8524); Rueckfall ohne Val-Wert; Repo-Abgleich: 1 Abweichungen, Code eedfcfdc4dc0 durch 8f9e0d049f38 ersetzt
Zielwert 0.815 belegt: "CORA CiteSeer PubMed CORA Full GCN 81 .5 ± 1.3 71.9 ± 1.9 77 .8 ± 2.9 62.2 ± 0.6" (Table 1, https://arxiv.org/pdf/1811.05868v2, Version v2, abgerufen 2026-10-09T14:37:04+00:00, SHA-256 47ac4b09ea8c)
Quellen:
- https://arxiv.org/pdf/1811.05868v2

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
- [reproduction report] The authors were unable to replicate the reported improvement of SIGN over GCN. — "On the other hand, the authors of SIGN  report an improvement over GCN , which we were not able to replicate in our experiments." (https://arxiv.org/html/2207.05669v1)
- [protocol note] Existing attacks use a validation set much larger than the training set and assume it is clean, which is unrealistic. — "All existing label poisoning attacks (LPATTACK, LAFAK, MG) evaluate on data splits with validation (val) set size much larger than that of training (train) set. To substantiate, in the default setting as tabulated in Table 1 in the Appendix, the val set contains 500 nodes and the train set contains 20 nodes per class (e.g. 140 nodes in the Cora-ML dataset). Moreover, the labels of the validation set are assumed to be clean – not poisoned." (https://openreview.net/pdf/79e770b6f8016087b03a0131c7e2c33e8dba0aae.pdf)
- [other] All label‑poisoning attacks use default hyper‑parameters that were tuned for clean labels, ignoring the effect of poisoning. — "Despite establishing this, all the label-poisoning attacks for GNNs still use default h-params to evaluate the effectiveness of the proposed attack. The default h-params were tuned to work with clean labels (unpoisoned)." (https://openreview.net/pdf/79e770b6f8016087b03a0131c7e2c33e8dba0aae.pdf)
- [protocol note] Assuming a clean validation set separate from training data is unrealistic because the defender cannot know which labels are corrupted. — "It is an unrealistic assumption for an attacker to treat train and val sets as separate entities. In other words, the current evaluation setup in literature for label-poisoning attacks training nodes while the val set is untouched and remains sanitized." (https://openreview.net/pdf/79e770b6f8016087b03a0131c7e2c33e8dba0aae.pdf)
- [reproduction report] The repository uses a different split for Cora to avoid overfitting, raising concerns about reproducibility with the official split. — "You are correct that we use a different split for Cora here, since SplineConv overfits when using the fixed split, but it starts outperforming GCN/Cheby when increasing the label rate." (https://github.com/pyg-team/pytorch_geometric/issues/893)
- [reproduction report] The PyGCN re‑implementation is described as only a proof of concept and not intended for reproducing the official GCN results. — "Furthermore, Thomas states that his PyGCN re-implementation only serves as a proof of concept and is not intended for reproduction of the official results." (https://github.com/pyg-team/pytorch_geometric/issues/891)
- [other] Using the official split is claimed to make the implementation correct, implying other splits may yield unreliable evaluations. — "Here, we use the official split, and hence I believe that our implementation is the correct one :)" (https://github.com/pyg-team/pytorch_geometric/issues/891)
Abfrage: 2026-10-09T14:51:57+00:00 (live), 16 Seiten gelesen, 8 Aussagen verworfen (Zitat nicht im Text).

## Possible explanations für disc_1811_05868_accuracy (Tavily Rolle 6, ungeprüfte Hypothesen, nicht Teil des Urteils)
- The paper evaluates models using many random data splits and multiple initializations, so a reimplementation using a single fixed split could obtain different results. — "splitting the data in a different way results in a different performance of each method." (https://medium.com/uci-nlp/summary-pitfalls-of-graph-neural-network-evaluation-16c1e0c02ceb)
- The authors report average performance over 2,000 random settings (100 splittings × 20 initializations), which differs from evaluations that use a single train/validation/test split. — "they introduce 4 new datasets and find the average performance of each one of the methods on 2,000 random settings (100 random splittings of the datasets and for each one of them 20 random initializations of parameters)" (https://medium.com/uci-nlp/summary-pitfalls-of-graph-neural-network-evaluation-16c1e0c02ceb)
- The paper states that using the same train/validation/test splits and altering training procedures (e.g., early stopping) prevents fair comparison, and that different splits lead to dramatically different model rankings. — "We show that using the same train/validation/test splits of the same datasets, as well as making significant changes to the training procedure (e.g. early stopping criteria) precludes a fair comparison of different architectures." (https://www.alphaxiv.org/abs/1811.05868)
- The paper notes that simpler GNN architectures can outperform more sophisticated ones only if hyperparameters and training procedures are tuned fairly for all models, implying that unstated hyperparameter choices affect results. — "Even more importantly, our findings suggest that simpler GNN architectures are able to outperform the more sophisticated ones if the hyperparameters and the training procedure are tuned fairly for all models." (https://www.alphaxiv.org/abs/1811.05868)
- By showing that rankings change dramatically across different data splits, the paper indicates that reported numbers may not be reproducible when using alternative splits. — "We perform a thorough empirical evaluation of four prominent GNN models and show that considering different splits of the data leads to dramatically different rankings of models." (https://www.alphaxiv.org/abs/1811.05868)

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/shchur/gnn-benchmark @ 1e72912a08

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:gnnbench/models/gcn.py@1e72912] feature normalisation: normalize_features flag: when True, features are row‑normalized via row_normalize(); when False, no normalization is applied — "def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return to_sparse_tensor(features)"
- [paper] adjacency normalisation: symmetric normalization with added self-loops (as in the original GCN) — "We keep the model architectures as they are in the original papers / reference implementations."
- [repo:gnnbench/models/gcn.py@1e72912] adjacency normalisation: apply renormalize_adj to adjacency matrix before converting to sparse tensor — "return to_sparse_tensor(renormalize_adj(graph_adj))"
- [paper] learning rate: 0.01 — "GCN 64 0.01 0.8 0.001 92 K"
- [repo:config/optimized/gcn.conf.yaml@1e72912] learning rate: 0.01 — "learning_rate: 0.01"
- [repo:config/reference/gcn.conf.yaml@1e72912] learning rate: 0.01 — "learning_rate: 0.01"
- [repo:config/optimized/gcn.conf.yaml@1e72912] weight decay: 0.001 — "weight_decay: 0.001"
- [repo:config/reference/gcn.conf.yaml@1e72912] weight_decay: 5e-4 — "weight_decay: !!float 5e-4"
- [repo:gnnbench/models/gcn.py@1e72912] weight decay: weight_decay parameter passed to GCN constructor and stored as self.weight_decay — "self.weight_decay = weight_decay"
- [repo:gnnbench/models/gcn.py@1e72912] weight decay: L2 regularization (weight decay) applied only to the first GCN layer — "# original implementation uses L2 regularization only on first layer"
- [repo:gnnbench/models/gcn.py@1e72912] weight decay: weight_decay set to 0.0 for all layers after the first (including the final layer) — "weight_decay=self.weight_decay if i == 0 else 0.0,"
- [repo:config/optimized/gcn.conf.yaml@1e72912] dropout: 0.8 — "dropout_prob: 0.8"
- [repo:config/reference/gcn.conf.yaml@1e72912] dropout: 0.5 — "dropout_prob: 0.5"
- [repo:config/optimized/gcn.conf.yaml@1e72912] hidden size: 64 — "hidden_size: 64"
- [repo:config/reference/gcn.conf.yaml@1e72912] hidden_size: 16 — "hidden_size: 16"
- [paper] epochs: 200 — "• Maximum number of epochs: 200."
- [paper] early stopping rule: same early stopping criterion, patience and validation frequency (display step) used for all models — "...same maximum number of training epochs, early stopping criterion, patience and validation frequency (display step) for all models (Appendix C)."
- [repo:gnnbench/train.py@1e72912] early_stopping_rule: if early_stopping_criterion == 'gcn': early_stopping = GCNCriterion(early_stopping_tolerance, sess, _log) — "if early_stopping_criterion == 'gcn':
        early_stopping = GCNCriterion(early_stopping_tolerance, sess, _log)"
- [repo:gnnbench/train.py@1e72912] early_stopping_rule: if early_stopping.should_stop(epoch, val_loss, val_metrics['accuracy']): — "if early_stopping.should_stop(epoch, val_loss, val_metrics['accuracy']):"
- [paper] other_settings: 100 random train/validation/test splits — "For each dataset we use 100 random train/validation/test splits and perform 20 random initializations for each split."
- [paper] other_settings: standardized training and hyperparameter selection procedure — "We use a standardized training and hyperparameter selection procedure for all models."
- [repo:config/optimized/gcn.conf.yaml@1e72912] hyperparameter: GCN — "model_name: "GCN""
- [repo:config/optimized/gcn.conf.yaml@1e72912] hyperparameter: 2 — "num_layers: 2"
- [repo:config/reference/gcn.conf.yaml@1e72912] hyperparameter: model_name: "GCN" — "model_name: "GCN""
- [repo:config/reference/gcn.conf.yaml@1e72912] hyperparameter: num_layers: 2 — "num_layers: 2"
- [repo:gnnbench/models/gcn.py@1e72912] other_settings: Dropout is applied to the inputs conditionally based on the dropout probability placeholder — "# Apply dropout to inputs if required
        inputs = tf.cond(
            tf.cast(dropout_prob, tf.bool),
            true_fn=(lambda: dropout_supporting_sparse_tensors(inputs, 1 - dropout_prob)),
            false_fn=(lambda: inputs),
        )"
- [repo:gnnbench/models/gcn.py@1e72912] other_settings: Feature normalization is controlled by the normalize_features flag; when True, features are row-normalized — "if self.normalize_features:
            features = row_normalize(features)"
- [repo:gnnbench/models/base_model.py@1e72912] other settings: Adam optimizer used for training — "optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)"
- [repo:gnnbench/models/base_model.py@1e72912] other settings: Regularization losses are added to the total loss — "regularization_losses = tf.losses.get_regularization_losses()
            if not regularization_losses:
                return loss
            return loss + tf.add_n(regularization_losses)"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch