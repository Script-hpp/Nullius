
1. Read SEED from os.environ, set numpy.random.seed(SEED) and torch.manual_seed(SEED).  
2. Load /​data/​cora_public.npz → features X (2708×1433), edge list, labels, train/val/test splits.  
3. Row‑normalize X → X̂ = D⁻¹X.  
4. Form adjacency A, add self‑loop, symmetrically normalize → Â = Ď⁻¹/² Ã Ď⁻¹/² with Ã = A+I.  
5. Convert X̂ and Â to sparse tensors (or dense equivalents).  
6. Two‑layer GCN:  
   a. Layer 0: Z₀ = ReLU(Â X̂ W₀), dropout 0.5, L2 5e‑4 on W₀.  
   b. Layer 1: Ẑ = Â Z₀ W₁, dropout 0.5, no activation, output dim = 7.  
7. Optimiser: Adam(lr=0.01) with weight‑decay 5e‑4 applied to W₀ only (as in the original code).  
8. Train full‑batch, max 200 epochs; after each epoch compute validation loss.  
9. Early stopping: halt if validation loss has not decreased for 10 consecutive epochs [(uncertain) exact stopping rule].  
10. Validation set (500 nodes) used only for early stopping / model selection; not included in gradient steps.  
11. Restore the model with lowest validation loss and evaluate on the test set (1000 nodes).  
12. Predict class = argmax softmax(Ẑ) for each test node.  
13. Write /output/predictions.csv with header `id,prediction` (id = node index from test_idx).  
14. Batch size: full‑batch (all nodes) [(uncertain) if a mini‑batch variant were used].  
15. Number of GCN layers: 2 (certain).