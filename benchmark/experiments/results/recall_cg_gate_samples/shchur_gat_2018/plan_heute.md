

- Load data: `np.load('/data/cora_lcc_public.npz')` → features, edges, labels, train/val/test indices.  
- Model: 2‑layer GAT.  
  - Layer 1: 8 attention heads, each output dim 8 → concatenated 64‑dim (uncertain). Activation ELU (uncertain). Dropout 0.6 on features & attention coefficients (uncertain).  
  - Layer 2: 1 head, output dim = num_classes = 7, no concatenation (uncertain).  
- Optimizer: Adam, learning_rate = 0.005 (uncertain), weight_decay = 5e‑4 (uncertain).  
- Loss: CrossEntropy on train_idx.  
- Training: full‑graph batch, max_epochs = 10000 (uncertain), early stopping patience = 100 on validation loss (uncertain).  
- Seed: read `SEED` env var, set `numpy.random.seed(SEED)` and `torch.manual_seed(SEED)`.  
- Validation: after each epoch compute accuracy on val_idx; retain model with highest val accuracy.  
- Testing: load best model, predict `argmax` for every node in test_idx, write `/output/predictions.csv` with header `id,prediction`.  
- Runtime limit: < 5 minutes on CPU.