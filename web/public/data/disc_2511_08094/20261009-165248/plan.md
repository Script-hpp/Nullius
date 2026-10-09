

- Model: SLGCN (Stuart‑Landau GCN) – 10 layers, hidden = 128, 2 attention heads.  
- Optimizer: NAdam, lr = 0.001, weight_decay = 1e‑5, dropout = 0.0, input_dropout = 0.0.  
- SL‑GCN specific params: ζ = 0 + 1i, ν = 1 + 1i, dt = 1.0, k = 0.3, p = 2.0, bias = False, leaky_relu_slope = 0.3.  
- Training: max_epoch = 100, batch_size = 64 (uncertain if used for node‑level), patience = 100 for LR decay, lr_decay_factor = 0.1.  
- LR scheduler: ReduceLROnPlateau(metric=val_acc, mode='max', factor=0.1, patience=100); stop when lr < lr_initial × 0.9.  
- Loss: cross‑entropy on train mask; accuracy computed on train/val/test masks each epoch.  
- Early stopping: keep best validation accuracy; record its corresponding test accuracy.  
- Randomness: seed from env SEED for NumPy & PyTorch weight init; dataset split fixed by file (SPLIT informational).  
- Preprocess: Cora LCC (2485 nodes), binary bag‑of‑words features (no normalization), undirected edges stored as both directions.  
- Split: 20 labeled nodes per class → train = 140, 30 per class → val = 210, remaining → test = 2135.  
- Output: predict class for each test node, write `/output/predictions.csv` with header `id,prediction`.  
- Runtime constraint: finish < 5 minutes on CPU.