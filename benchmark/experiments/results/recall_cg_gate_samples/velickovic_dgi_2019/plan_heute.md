

1. Read SEED from os.environ, set numpy.random.seed(SEED) and torch.manual_seed(SEED).  
2. Load `/data/cora_public.npz`; obtain X (features), edge_index, y, train_idx, val_idx, test_idx.  
3. Encoder: 1‑layer GCN, in_dim=1433 → H (hidden, **uncertain**: 512) → out_dim=H, ReLU activation.  
4. Readout: s = σ( mean(Z, dim=0) ), σ = sigmoid (**uncertain**).  
5. Discriminator: score(z, s) = zᵀ W s, W∈ℝ^{H×H} learned (**uncertain** initialization Xavier).  
6. Corruption: shuffle rows of X to get Ẋ, encode to Ẑ.  
7. Loss: −log σ(score(Z,s)) − log σ(−score(Ẑ,s)) summed over nodes (JSD estimator).  
8. Optimizer: Adam(lr=LR_e (**uncertain**: 0.001), weight_decay=0.0).  
9. Train encoder for Epoch_e (**uncertain**: 200) epochs, full‑batch, re‑shuffle corruption each epoch.  
10. Freeze encoder, compute node embeddings Z for all nodes.  
11. Classifier: linear layer C = 7 → logits; optimizer Adam(lr=LR_c (**uncertain**: 0.01), weight_decay=0.0).  
12. Train classifier for Epoch_c (**uncertain**: 100) epochs, early stop on val accuracy (patience 10).  
13. Predict on test_idx: argmax of logits, produce integer class labels.  
14. Write `/output/predictions.csv` with header `id,prediction`; one row per test node (id = node index).  
15. Verify total runtime < 5 minutes on CPU.