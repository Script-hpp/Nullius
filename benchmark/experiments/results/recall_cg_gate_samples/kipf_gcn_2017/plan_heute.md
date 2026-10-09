

- Load `/data/cora_public.npz`; build symmetric adjacency A (already bidirectional), add self‑loops → Â = D⁻¹ᐟ² (A+I) D⁻¹ᐟ²; convert to sparse torch tensor.  
- Seed numpy & torch with `int(os.getenv('SEED'))`.  
- **Model**: two‑layer GCN.  
  - Layer 0: H₀ = ReLU(Â X W₀), X∈ℝ^{2708×1433}, W₀∈ℝ^{1433×16}.  
  - Dropout(p=0.5) **(uncertain)** on H₀.  
  - Layer 1: Z = Â H₀ W₁, W₁∈ℝ^{16×7} (logits).  
- Loss: cross‑entropy on `train_idx`.  
- Optimizer: Adam(lr=0.01, weight_decay=5e-4).  
- Train ≤200 epochs; after each epoch compute val loss on `val_idx`; stop if val loss hasn’t improved for 100 epochs **(uncertain)**.  
- Inference: argmax(Z[test_idx]) → predicted class.  
- Write `/output/predictions.csv` with header `id,prediction` (id = node index from `test_idx`).  
- All operations on CPU; total runtime < 5 min.