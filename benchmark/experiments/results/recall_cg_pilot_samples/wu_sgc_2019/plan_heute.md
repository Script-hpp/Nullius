1. Set random seeds: `numpy.random.seed(SEED); torch.manual_seed(SEED)`.  
2. Load `/data/cora_public.npz`; extract `features` (float32), `edges` (int64), `labels`, `train_idx`, `val_idx`, `test_idx`.  
3. Build sparse adjacency `A` from `edges`; **add self‑loops** (A+I) *(uncertain)*.  
4. Compute symmetric normalization \(\hat{A}=D^{-1/2}(A+I)D^{-1/2}\) *(uncertain if self‑loops used)*.  
5. Propagate features: \(X' = \hat{A}^K X\) with **K = 2** *(uncertain)*. Use sparse matrix powers.  
6. Optionally row‑normalize `X'` (not in paper) – skip.  
8. Split `X'` into train/val/test using the provided indices.  
9. Train a multinomial logistic regression (softmax) with LBFGS solver, `max_iter=1000`.  
10. Tune L2 regularization strength **C** via validation: search `[0.001,0.01,0.1,1,10,100]` *(uncertain which value yields 81.0%)*.  
11. Choose C with highest validation accuracy.  
12. Retrain classifier on **train + val** (or train only) with selected C *(uncertain whether retraining on train+val)*.  
13. Predict class probabilities on test nodes, take `argmax` as prediction.  
14. Write `/output/predictions.csv` with header `id,prediction`; one row per node in `test_idx` (id = node index).  
15. Ensure total runtime < 5 minutes on CPU.