

- Model: two‑layer GCN (input → hidden = 16 → output = 7 classes); ReLU after first layer, linear after second.  
- Features: row‑normalized (each node’s feature vector divided by its sum) [certain].  
- Adjacency: renormalization trick Â = A + I, ĤÂ = Ď⁻¹ᐟ² Â Ď⁻¹ᐟ² [certain].  
- Weight initialization: Glorot (Xavier) uniform [certain].  
- Optimizer: Adam, learning_rate = 0.01 [certain]; weight_decay (L2) = 5·10⁻⁴ applied to the first GCN layer weights [(uncertain)].  
- Dropout: rate = 0.5 applied to hidden‑layer inputs (sparse dropout on features and hidden activations) [certain].  
- Epochs: maximum 200, early stopping with patience = 10 (stop if validation loss does not improve for 10 consecutive epochs) [certain].  
- Validation set: 500 nodes (val_idx) used only for early‑stopping; not used for training gradients [certain].  
- Training loss: computed only on train_idx (140 nodes); validation and test labels are masked out [certain].  
- Batch size: full‑batch (all nodes processed each epoch) [(uncertain)].  
- Number of layers: 2 (certain); no attention heads (GCN does not use heads) [certain].  
- Random seed: read from environment variable SEED, set NumPy and PyTorch seeds [certain].  
- After training, restore the model with lowest validation loss (or last epoch if no early stop) and compute logits for test_idx.  
- Output: CSV `/output/predictions.csv` with header `id,prediction`; `id` = node index from test_idx, `prediction` = argmax class.  
- Expected test accuracy ≈ 81.5 % on the Cora standard split.