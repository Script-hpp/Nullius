**Reproduction Plan: GCN on Cora (LCC)**

1. **Model**: 2-layer GCN (Kipf & Welling, 2017), symmetric normalization `Â = D̂⁻⁰·⁵ Â D̂⁻⁰·⁵`, ReLU after first layer, no activation after second.
2. **Hidden dimension**: 64 (uncertain – grid search included 16, 32, 64, 128, 256).
3. **Dropout**: 0.5 (uncertain – grid search included 0.0, 0.5, 0.8).
4. **Learning rate**: 0.01 (uncertain – grid search included 0.001, 0.01, 0.1).
5. **Weight decay (L2)**: 5e-4 (uncertain – grid search included 0, 5e-4, 5e-3, 5e-2).
6. **Optimizer**: Adam.
7. **Loss**: Cross-entropy on `train_idx` only.
8. **Early stopping**: Monitor validation accuracy on `val_idx`, patience = 100 epochs, max epochs = 10,000.
9. **Weight initialization**: Seed `torch.manual_seed(SEED)`, `np.random.seed(SEED)` (env var `SEED`).
10. **Data split**: Use provided `train_idx` (140), `val_idx` (210), `test_idx` (2135) from `/data/cora_lcc_public.npz` (env var `SPLIT` informational).
11. **Features**: Use raw binary bag-of-words [2485, 1433] **without** row-normalisation (paper uses raw features).
12. **Edges**: Convert directed pairs to undirected `edge_index` (coalesce duplicates, add self-loops via `Â`).
13. **Training loop**: Single run per seed (the file already fixes one of the 100 splits); no split/initialization averaging needed.
14. **Output**: Predict class for every node in `test_idx`, write `/output/predictions.csv` with header `id,prediction`.
15. **Runtime**: < 5 min on CPU (single GCN run ≈ 10–30 s).