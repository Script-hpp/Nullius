1. Set random seeds: `numpy.random.seed(int(os.getenv('SEED'))); torch.manual_seed(int(os.getenv('SEED')))`.  
2. Load `/data/cora_public.npz`; extract `features`, `edges`, `labels`, `train_idx`, `val_idx`, `test_idx`.  
3. Build adjacency matrix `A` from `edges` (ensure symmetry), add self‑loops, compute \(\hat{A}=D^{-1/2}(A+I)D^{-1/2}\).  
4. Define GCN model:  
   - Layer 1: `GCNConv(1433, 16)` → ReLU → dropout `p=0.5 (uncertain)`.  
   - Layer 2: `GCNConv(16, 7)` (no activation).  
5. Loss: `CrossEntropyLoss` applied only on training nodes (`train_idx`).  
6. Optimizer: `Adam(lr=0.01 (uncertain), weight_decay=5e-4 (uncertain))`.  
7. Training loop: max `200 epochs (uncertain)`, early stop if validation loss does not improve for `20 epochs (uncertain)`.  
8. After each epoch compute validation accuracy on `val_idx` for model selection (keep best weights).  
9. Inference: run model in eval mode, obtain logits for all nodes, predict `argmax` for nodes in `test_idx`.  
10. Write `/output/predictions.csv` with header `id,prediction`; each row: node index from `test_idx`, predicted class (0‑6).  

All steps must finish well under 5 minutes on CPU.