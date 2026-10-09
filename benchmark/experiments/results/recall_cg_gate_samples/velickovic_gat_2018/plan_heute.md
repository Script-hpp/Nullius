

1. Set SEED from `os.environ['SEED']`; seed NumPy (`np.random.seed`) and PyTorch (`torch.manual_seed`).  
2. Load `/data/cora_public.npz`: features [2708,1433] (float32), edges [E,2] (int64, bidirectional), labels [2708] (‑1 for test), and `train_idx`, `val_idx`, `test_idx`.  
3. Model – 2‑layer GAT:  
   - Layer 1: Multi‑head GAT, `heads=8 (uncertain)`, `out_dim_per_head=8 (uncertain)` → hidden = 64, activation = ELU, input dropout = 0.6 (uncertain), attention dropout = 0.6 (uncertain).  
   - Layer 2: Single‑head GAT, `out_dim=7` (num_classes), no activation, input dropout = 0.6 (uncertain), attention dropout = 0.6 (uncertain).  
4. Optimizer: Adam, `lr=0.005 (uncertain)`, `weight_decay=5e-4`.  
5. Loss: `CrossEntropyLoss` computed only on `train_idx`.  
6. Training: full‑batch, `max_epochs=10000 (uncertain)`, early stopping with `patience=100 (uncertain)` on validation loss; restore best model (lowest val loss).  
7. After each epoch evaluate loss on `val_idx`; stop if no improvement for `patience` epochs.  
8. Test: run inference on `test_idx`, take `argmax` of logits → predicted class.  
9. Write `/output/predictions.csv` with header `id,prediction`; one row per node in `test_idx` (sorted by node index). No extra content.