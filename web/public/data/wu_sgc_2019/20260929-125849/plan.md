**Reproduction Plan for SGC on Cora (Planetoid Split)**

1. **Model**: SGC – precompute `S^K X` where `S = D^{-1/2}(A+I)D^{-1/2}` (symmetric normalized adjacency with self-loops), then train a linear classifier (multinomial logistic regression) on the transformed features.
2. **Propagation steps**: `K = 2` (paper: “We set K=2 for all datasets.”)
3. **Features**: raw binary bag-of-words (no normalization).
4. **Optimizer**: Adam, learning rate `0.2` (paper: “train SGC for 100 epochs using Adam with learning rate 0.2.”)
5. **Weight decay**: tuned via hyperopt on validation set (exact value **uncertain**).
6. **Training epochs**: up to `100`; select epoch with highest validation accuracy (paper: “tune the number of epochs based on both convergence behavior and validation accuracy.”)
7. **Loss**: cross-entropy on `train_idx` (full-batch).
8. **Validation**: evaluate on `val_idx` after each epoch for model selection.
9. **Random seed**: read `SEED` env var; seed `numpy` and `torch`.
10. **Output**: predict classes for `test_idx`, write `/output/predictions.csv` with header `id,prediction`.