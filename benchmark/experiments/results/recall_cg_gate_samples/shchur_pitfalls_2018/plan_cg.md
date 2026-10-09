

- Load `/data/cora_lcc_public.npz`; use provided `train_idx`, `val_idx`, `test_idx`.  
- Preprocess adjacency: add self‑loop, symmetrically normalize \(\hat{A}=D^{-1/2}(A+I)D^{-1/2}\), convert to sparse tensor; keep features as sparse tensor (`normalize_features=False`).  
- **Model**: GCN, 2 layers. Layer 0: input→64, ReLU, dropout 0.8, **weight_decay=0.001** (applied only to this layer). Layer 1: 64→7 (Cora classes), linear, dropout 0.8, weight_decay=0.0. *(architecture from source)*  
- Optimizer: **Adam** with learning_rate = 0.01 *(uncertain – code calls `model.optimize`)*.  
- Training: full‑batch (all nodes) *(uncertain – no mini‑batch mentioned)*, up to `num_epochs=100000`.  
- Early stopping: **gnnbench** criterion, patience = 50 epochs on validation loss; reset variables to best validation state when stopped.  
- No learning‑rate decay (`lr_decay_factor=0.0`).  
- Seed NumPy and TensorFlow with env var `SEED` for weight initialization.  
- After stopping, restore best validation weights, compute logits on `test_idx`, predict `argmax`.  
- Write `/output/predictions.csv` with header `id,prediction` (id = node index from `test_idx`).  

All values taken from the paper/config except those marked *(uncertain)*.