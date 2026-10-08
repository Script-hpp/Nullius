1. Load `/data/cora_public.npz`; row‑normalize features (L1) **(uncertain: L1 vs L2)**.  
2. Build symmetric binary adjacency from edges (add self‑loops); compute symmetrically normalized \(\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}\).  
3. Model: 2‑layer GCN – Layer1: `GraphConvolution(1433→16, ReLU, dropout=0.5, sparse_inputs=True)`; Layer2: `GraphConvolution(16→7, linear, dropout=0.5)`.  
4. Weight init: Xavier (Glorot) for weights, zeros for bias.  
5. Loss: masked softmax cross‑entropy on `train_idx`.  
6. L2 regularization: \(5\cdot10^{-4}\) on first‑layer weights only.  
7. Optimizer: Adam, learning rate 0.01.  
8. Train: max 200 epochs, early stopping on validation loss (patience=10) using `val_idx`.  
9. Seed NumPy and PyTorch from `SEED` env var.  
10. Predict on `test_idx`; write `/output/predictions.csv` with header `id,prediction`.