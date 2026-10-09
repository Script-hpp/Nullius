

- Model: 2‑layer GCN, hidden size = 64, ReLU activation after first layer, no activation after second layer.  
- Dropout probability = 0.8 applied to inputs of each GCN layer (as in the code).  
- Weight decay = 0.001 (L2) applied **only** to the first‑layer weight matrix (second layer gets 0.0). *(uncertain whether this matches the paper’s intent)*  
- Learning rate = 0.01, optimizer = Adam.  
- Maximum epochs = 200; early stopping used with the GCNCriterion (patience and improvement tolerance not specified). *(uncertain)*  
- Validation evaluated every report_interval epochs (value not given). *(uncertain)*  
- Adjacency pre‑processed with symmetric normalization + self‑loops (Â = D̃⁻⁰·⁵ Ã D̃⁻⁰·⁵, Ã = A + I).  
- Feature pre‑processing: row‑normalisation if `normalize_features=True`; flag not provided in config. *(uncertain)*  
- Training uses full‑batch (no mini‑batch size specified). *(uncertain)*  
- Random seed for weight initialisation taken from env var `SEED`; both NumPy and PyTorch seeded with it.  
- After training, predict class labels for all nodes in `test_idx` and write `/output/predictions.csv` with columns `id,prediction`.  
- Total runtime must stay under 5 minutes on CPU.