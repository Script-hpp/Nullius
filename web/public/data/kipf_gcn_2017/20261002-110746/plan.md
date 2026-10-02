**Reproduction Plan (GCN on Cora)**

1. **Model**: 2-layer GCN: `Z = softmax(Â ReLU(Â X W⁽⁰⁾) W⁽¹⁾)` with `Â = D̃⁻¹ᐟ² (A+I) D̃⁻¹ᐟ²` (symmetric normalization, self-loops added).  
2. **Layers**: (1) 1433→16, ReLU, dropout=0.5 on input (sparse dropout); (2) 16→7, linear, dropout=0.5 on hidden; output softmax.  
3. **Loss**: Masked cross-entropy on `train_idx` + L2 regularization 5·10⁻⁴ on first-layer weights only.  
4. **Optimizer**: Adam, lr=0.01, full-batch (all nodes each epoch).  
5. **Weight init**: Glorot uniform (Xavier) for weights, zeros for biases.  
6. **Feature preprocessing**: Row-normalize features (L1 norm per row) **(uncertain: L1 vs L2)**.  
7. **Adjacency**: Edges already symmetric; add self-loops, compute `Â`.  
8. **Training**: Max 200 epochs; early stopping on validation loss with patience=10 (stop if no decrease for 10 consecutive epochs).  
9. **Validation**: `val_idx` (500 nodes) for early stopping/model selection; **not used for training**.  
10. **Test**: Predict on `test_idx` (1000 nodes) after training; output class indices.  
11. **Dropout**: Applied to inputs of each layer (sparse dropout for first layer) with rate 0.5 (keep_prob=0.5).  
12. **Metric**: Masked accuracy on `test_idx`.  
13. **Hyperparameters (paper)**: hidden=16, dropout=0.5, lr=0.01, L2=5·10⁻⁴, epochs=200, early_stop=10.  
14. **Random seed**: Read `SEED` env var; seed NumPy and PyTorch. Paper reports mean over 100 runs; this is a single run **(uncertain: single-run variance)**.  
15. **Output**: Write `/output/predictions.csv` with header `id,prediction` (one row per test node).