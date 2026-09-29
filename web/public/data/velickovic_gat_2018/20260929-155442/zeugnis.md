# Reproduktions-Zeugnis: velickovic_gat_2018

## Claim gat_cora_acc: teilweise
Gemessen: 0.8036 (std 0.0088)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8080 mit Graph, 0.5530 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0088. Mit unserem Setup gemessen (10 von 10 Seeds, Zielwert 0.83 +/- 0.01, Table 2, arXiv v3 (pdf)): Mittel 0.8036.
Zielwert 0.83 belegt: "GAT (ours) 83.0 ± 0.7% 72.5 ± 0.7% 79.0 ± 0.3%" (Table 2, https://arxiv.org/pdf/1710.10903v3, Version v3, abgerufen 2026-09-29T13:54:42+00:00, SHA-256 f7f86c91c240)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://arxiv.org/pdf/1710.10903v3

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/PetarV-/GAT @ 5af87e7fce

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] neighborhood definition: first-order neighbors of i (including i) — "In all our experiments, these will be exactly the first-order neighbors of i (including i)."
- [paper] shared linear transformation: weight matrix W ∈ R^{F'×F} applied to every node — "a shared linear transformation, parametrized by a weight matrix , W ∈ RF ′×F , is applied to every node."
- [paper] dataset: Cora — "The Cora dataset contains 2708 nodes, 5429 edges, 7 classes and 1433 features per node."
- [paper] preprocessing: bag-of-words representation — "Node features correspond to elements of a bag-of-words representation of a document."
- [paper] training_split: 20 nodes per class (total 140) — "We allow for only 20 nodes per class to be used for training—however, honoring the transductive setup, the training algorithm has access to all of the nodes’ feature vectors."
- [paper] evaluation_protocol: accuracy on test set — "The predictive power of the trained models is evaluated on 1000 test nodes, and we use 500 additional nodes for validation purposes (the same ones as used by Kipf & Welling (2017))."
- [paper] model_type: two-layer GAT model — "For the transductive learning tasks, we apply a two-layer GAT model."
- [paper] first_layer_attention_heads: K = 8 attention heads computing F' = 8 features each (total 64 features) — "The first layer consists of K = 8 attention heads computing F ′ = 8 features each (for a total of 64 features), followed by an exponential linear unit (ELU) (Clevert et al., 2016) nonlinearity."
- [paper] second_layer_attention_heads: single attention head computing C features (number of classes) — "The second layer is used for classification: a single attention head that computes C features (where C is the number of classes), followed by a softmax activation."
- [paper] L2_regularization_lambda: 0.0005 — "During training, we apply L2 regulariza-tion with λ = 0 .0005 ."
- [paper] hyperparameter_optimization_dataset: Cora — "Its architectural hyperparameters have been optimized on the Cora dataset and are then reused for Cite-seer."
- [paper] skip connections: employed skip connections across the intermediate attentional layer — "L2 regularization or dropout—we have, however, successfully employed skip connections (He et al., 2016) across the intermediate attentional layer."
- [paper] batch size: 2 graphs — "We utilize a batch size of 2 graphs during training."
- [repo:execute_cora.py@5af87e7] Early stopping criterion: patience-based early stopping — "if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)"
- [repo:execute_cora.py@5af87e7] Checkpoint saving condition: save when both validation accuracy improves and validation loss decreases — "if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)"
- [repo:execute_cora.py@5af87e7] Best validation metrics tracking: track maximum validation accuracy and minimum validation loss — "vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))"
- [repo:execute_cora.py@5af87e7] Model restoration: restore best model from checkpoint before testing — "saver.restore(sess, checkpt_file)"
- [repo:execute_cora.py@5af87e7] Test batch evaluation: evaluate test set in batches, averaging loss and accuracy — "while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)"
- [repo:execute_cora.py@5af87e7] Dropout rates during testing: attn_drop=0.0, ffd_drop=0.0 — "attn_drop: 0.0, ffd_drop: 0.0"
- [repo:execute_cora_sparse.py@5af87e7] Validation dropout rates: attn_drop: 0.0, ffd_drop: 0.0 — "attn_drop: 0.0, ffd_drop: 0.0})"
- [repo:execute_cora_sparse.py@5af87e7] Checkpoint saving: saver.save(sess, checkpt_file) — "saver.save(sess, checkpt_file)"
- [repo:execute_cora_sparse.py@5af87e7] Checkpoint restoring: saver.restore(sess, checkpt_file) — "saver.restore(sess, checkpt_file)"
- [repo:execute_cora_sparse.py@5af87e7] Validation batch iteration: while vl_step * batch_size < vl_size: — "while vl_step * batch_size < vl_size:"
- [repo:execute_cora_sparse.py@5af87e7] Test batch iteration: while ts_step * batch_size < ts_size: — "while ts_step * batch_size < ts_size:"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch