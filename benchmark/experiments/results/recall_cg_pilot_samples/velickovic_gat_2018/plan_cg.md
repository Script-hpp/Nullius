1. Dataset: Cora (Planetoid split) – features [2708,1433] binary BoW, edges directed both ways, labels hidden for test nodes.  
2. Model: GAT – 2 layers; Layer 1: 8 attention heads × 8 features each → 64‑dim concat; Layer 2: 1 head → 7‑class logits (averaged).  
3. Activation: ELU after Layer 1; linear (identity) after Layer 2.  
4. Dropout: p = 0.6 applied to layer inputs and to attention coefficients (training only).  
5. Learning rate = 0.005; weight decay (L2) = 0.0005; batch size = 2 graphs (paper).  
6. Optimizer: Adam with the above lr and L2.  
7. Training epochs = 100000 (uncertain); early stopping patience = 100 (uncertain) on validation loss.  
8. Validation: compute loss & accuracy on val_idx each epoch; stop when no improvement for patience epochs.  
9. Pre‑processing: row‑normalize features (each row ÷ sum); add self‑loops; adjacency bias = normalized A + I.  
10. TF graph: placeholders of shape (batch_size, N, F); each batch feeds the same Cora graph twice (batch_size=2).  
11. Seed: read env var SEED; set numpy.random.seed(SEED) and tf.set_random_seed(SEED).  
12. After training, restore checkpoint with lowest validation loss; run inference on all nodes.  
13. Predictions: argmax over logits for nodes in test_idx; write /output/predictions.csv with header id,prediction.  
14. Target test accuracy ≈ 83.0 % (as reported in the paper).  
15. All values marked (uncertain) are epochs and patience; other hyperparameters follow spec/paper/code.