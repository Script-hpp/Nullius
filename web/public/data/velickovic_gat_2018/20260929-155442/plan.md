**Reproduction Plan: GAT on Cora (Planetoid split)**

1. **Data**: Load `/data/cora_public.npz` → features [2708,1433] (binary BoW, no norm), edges [E,2] (directed, symmetric), labels [2708] (-1=test), splits: train_idx(140), val_idx(500), test_idx(1000).
2. **Graph**: Build adjacency with self-loops (first-order neighbors incl. self). Use sparse COO/CSR for message passing.
3. **Model**: 2-layer GAT.
   - Layer 1: Shared `W₁ ∈ ℝ^{64×1433}` → 8 heads × 8 dims. Attention: additive (LeakyReLU a=0.2) over neighbors. Concat heads → ELU. **Skip connection**: add input (projected to 64) to output (uncertain).
   - Layer 2: Shared `W₂ ∈ ℝ^{7×64}` → 1 head × 7 dims. Attention over neighbors → softmax → logits.
4. **Loss**: Cross-entropy on train_idx + L2(λ=0.0005) on all weights (uncertain: bias inclusion).
5. **Training**: Transductive (full graph each step). Optimizer: Adam (lr=0.005, uncertain). Dropout: attn_drop=0.6, ffd_drop=0.6 (uncertain; paper repo uses 0.0 only for val/test). Batch size: full graph (paper says "2 graphs" uncertain for single-graph transductive).
6. **Early stopping**: Patience=100 (uncertain). Track max val_acc & min val_loss. Save checkpoint **only if both improve**.
7. **Validation**: Every epoch (or 10 steps, uncertain). Evaluate on val_idx with dropout=0.0.
8. **Testing**: Restore best checkpoint. Evaluate on test_idx (dropout=0.0) → predictions.csv (`id,prediction`).
9. **Seed**: Read `SEED` env var → `torch.manual_seed`, `np.random.seed`.
10. **Target**: ≥83.0% test accuracy.