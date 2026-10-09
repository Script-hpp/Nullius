

**Reproduction plan for Deep Graph Infomax (DGI) on Cora**

- **Model**: DGI with a single‑layer GCN encoder (`GCN(n_in, n_h, activation)`) → average readout → sigmoid → bilinear discriminator.  
  - GCN: linear + bias + PReLU activation (`nonlinearity='prelu'`). *(layers = 1 (uncertain))*  
  - Hidden dimension `n_h = 512`.  
  - No attention heads (DG​I does not use GAT; heads = 1 (uncertain)).  

- **Input preprocessing**:  
  - Features row‑normalized (`process.preprocess_features`).  
  - Adjacency: add self‑loop, symmetrically normalize (`process.normalize_adj`).  

- **Training hyper‑parameters**:  
  - Learning rate `lr = 0.001`.  
  - Weight decay `l2_coef = 0.0`.  
  - Batch size `= 1` (one graph per batch).  
  - Dropout `drop_prob = 0.0`.  
  - Epochs `nb_epochs = 10000` with early stopping (`patience = 20`) on the BCE loss.  
  - Optimiser: Adam.  
  - Loss: `BCEWithLogitsLoss` between original and shuffled feature representations.  

- **Embedding extraction**: after early stopping, obtain node embeddings via `model.embed`.  

- **Linear evaluation**:  
  - Logistic regression classifier (`LogReg`) with input size `hid_units`, output `nb_classes`.  
  - Optimiser: Adam, `lr = 0.01`, weight decay `0.0`.  
  - Train for 100 epochs on training embeddings, evaluate on test embeddings.  
  - Repeat 50 times with different random seeds; report average test accuracy.  

All values above follow the paper/official code; those marked *(uncertain)* are not explicitly stated but are standard choices for the reported experiments.