

**Reproduction plan (≤15 lines)**  

- **Data**: Load `/data/cora_public.npz`; row‑normalize features (as in `preprocess_features`). Add self‑loop to adjacency, set Ãᵢⱼ=1 if edge exists or i=j else 0, and build bias bᵢⱼ=0 for Ãᵢⱼ>0 else −1e9 (mask for softmax). Use the provided `train_idx`, `val_idx`, `test_idx`.  
- **Model**: GAT with 2 layers.  
  - Layer 1: 8 attention heads, each outputs F′=8 features → concatenated dim = 64, activation ELU, dropout (in‑drop = ffd_drop = 0.6) on inputs and attention coef‑drop = 0.6.  
  - Layer 2 (output): 1 head, outputs C=7 logits, no activation, same dropout rates.  
  - No residual connections.  
  - (Uncertain) Number of layers inferred from `hid_units=[8]`; if more layers were intended, they would be specified similarly.  
- **Hyper‑parameters** (exact from paper/code hierarchy):  
  - Learning rate lr = 0.005  
  - Weight decay (L2) λ = 0.0005  
  - Epochs ≤ 100 000 with early stopping patience = 100 on validation loss  
  - Batch size = 2 graphs *(uncertain – code uses 1, paper says 2)*  
  - Dropout p = 0.6 (both feature and attention dropout)  
  - Hidden units per head = 8  
  - Heads = [8, 1] (first layer 8, output layer 1)  
  - Non‑linearity = ELU  
- **Optimizer**: Adam with lr = 0.005 and weight_decay = 0.0005.  
- **Training loop**: For each epoch, iterate over batches (size = batch_size) of the single graph (duplicate/shuffle as needed), compute masked softmax‑cross‑entropy loss and accuracy on training masks, update with Adam; after each epoch evaluate loss/accuracy on validation mask; stop if validation loss does not improve for patience epochs, restore best model.  
- **Inference**: Set `is_train=False`, dropout = 0, compute logits for all nodes, take argmax → predicted class.  
- **Output**: Write `/output/predictions.csv` with header `id,prediction`; rows ordered by `test_idx`.  
- **Reproducibility**: Read integer seed from env `SEED`; call `numpy.seed(seed)` and `torch.manual_seed(seed)`.  

All values above are taken directly from the paper, the start script, or the authors’ code; those marked *(uncertain)* reflect points where sources diverge or are unspecified.