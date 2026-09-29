**Reproduction Plan: GCN on Cora (Planetoid split)**

1. **Model**: 2-layer GCN: `X → GCN(16, ReLU, dropout) → GCN(7, linear, dropout) → softmax`
2. **Propagation**: `Ã = A + I`; `Â = D̃^(-1/2) Ã D̃^(-1/2)` (renormalization trick, Eq. 8)
3. **Features**: Row-normalize (L1) the binary bag-of-words vectors
4. **Adjacency**: Binary, symmetric from directed edges (already bidirectional in file)
5. **Weight init**: Glorot (Xavier) for weights; zeros for biases
6. **Dropout**: 0.5 on inputs of both layers (sparse_dropout for first layer)
7. **L2 reg**: 5e-4 on first-layer weight matrix only
8. **Optimizer**: Adam, lr=0.01, full-batch gradient descent
9. **Loss**: Masked softmax cross-entropy on train_idx (labels_mask)
10. **Early stopping**: Stop if val loss fails to decrease for 10 consecutive epochs (max 200)
11. **Splits**: Train=140 (20/class), Val=500, Test=1000 (val labels NOT used for training)
12. **Seed**: Read `SEED` env var; seed numpy & torch
13. **Output**: Predict class for each test_idx node → `/output/predictions.csv` (id,prediction)

**Uncertainties**: (1) L1 vs L2 row-normalization (paper says "row-normalize") (2) Exact early stopping: strict decrease vs non-improvement (3) Whether sparse_dropout uses `keep_prob=1-dropout` or `rate=dropout` (repo shows `1-self.dropout`) (4) Whether bias included in second layer (repo suggests yes) (5) Single run vs 100-run mean (target is 81.5% mean; single run will vary)