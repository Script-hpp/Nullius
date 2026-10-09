

- Read SEED env var, set `numpy.random.seed(SEED)` and `torch.manual_seed(SEED)` (CPU only).  
- Load `/data/cora_lcc_public.npz`; features ∈ ℝ^{2485×1433}, edges, labels, train/val/test indices.  
- **Model**: 2‑layer GCN, input_dim=1433, hidden_dim=16 *(uncertain)*, output_dim=7, ReLU activation, dropout p=0.5 *(uncertain)* after each hidden layer.  
- **Optimizer**: Adam(lr=0.01 *(uncertain)*, weight_decay=5e-4 *(uncertain)*) .  
- **Loss**: CrossEntropyLoss computed only on training‑labelled nodes.  
- **Training**: max_epochs=200 *(uncertain)*; early stopping on validation loss with patience=50 epochs *(uncertain)*; use `train_idx` for gradient steps, `val_idx` for early‑stopping check.  
- After training, run inference on all nodes, extract predictions for `test_idx`.  
- Write `/output/predictions.csv` with header `id,prediction`; `id` = node index from `test_idx`, `prediction` = predicted class (0‑6).  
- Ensure everything runs on CPU; total runtime should stay well under 5 minutes.  
- The env var `SPLIT` is informational only; do not recompute splits.  
- All uncertain hyper‑parameters are marked above; adjust if the paper specifies different values.