

- Read SEED from os.environ, set `np.random.seed(seed)` and `torch.manual_seed(seed)`.  
- Load `/data/cora_public.npz`; obtain `features` (float32, shape [2708,1433]), `edges` (int64, [E,2]), `labels`, and splits `train_idx`, `val_idx`, `test_idx`.  
- Build adjacency matrix **A** (add self‑loops → **Ã = A + I**), compute degree vector **d**, and normalize: **Â = D⁻¹ᐟ² Ã D⁻¹ᐟ²** (sparse).  
- Choose propagation steps **K = 2** *(uncertain)*; compute propagated features **H = Âᵏ · features** (use sparse‑dense multiplication).  
- Initialize a single linear layer **W**: `torch.nn.Linear(H.shape[1], num_classes, bias=False)`.  
- Loss: `torch.nn.CrossEntropyLoss()`.  
- Optimizer: Adam with **lr = 0.2** *(uncertain)*, **weight_decay = 5e‑4** *(uncertain)*.  
- Training loop: max **200 epochs** *(uncertain)*; after each epoch evaluate accuracy on `val_idx`.  
- Early stopping: stop if validation accuracy does not improve for **50 epochs** *(uncertain)*; restore best weights.  
- After training, predict logits for all nodes, take `argmax` to get class labels.  
- Write `/output/predictions.csv` with header `id,prediction`; rows correspond to `test_idx` in order, `id` = node index, `prediction` = integer class.  
- No other output is produced; total runtime should stay well under 5 minutes on CPU.