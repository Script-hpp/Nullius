

- Model: 2‑layer GCN (1433 → 16 → 7) with ReLU after the first layer, linear after the second.  
- Dropout: rate = 0.5 applied to both layers during training.  
- Weight decay: L2 = 5e‑4 applied only to the first‑layer weights.  
- Optimizer: Adam, learning rate = 0.01 (constant).  
- Pre‑processing: features row‑normalized; adjacency Â = D⁻¹ᐟ²(A+I)D⁻¹ᐟ², both converted to sparse tuples.  
- Training: full‑batch (all 2708 nodes) → batch_size = 2708 (uncertain).  
- Epochs: max = 200; early stopping with patience = 10 (stop if val loss doesn’t improve for 10 consecutive epochs).  
- Validation set used for early stopping and optional model selection.  
- Test accuracy computed after training (or from the best validation model).  
- Seed: read from env var SEED; set numpy and torch seeds accordingly.  
- No learning‑rate schedule or additional regularisation beyond weight decay.  
- Implementation follows the official TensorFlow 1.x GCN code (layers, loss, etc.).  
- Unspecified in sources: batch_size, number of layers, attention heads → chosen as full batch, 2 layers, heads = 1 (uncertain).  
- All other hyper‑parameters (hidden = 16, weight_decay = 5e‑4, dropout = 0.5, lr = 0.01, epochs = 200, patience = 10) are as specified.  
- Write predictions for test_idx to /output/predictions.csv with header `id,prediction`.