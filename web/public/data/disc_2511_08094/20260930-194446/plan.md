**Reproduction Plan: SL-GCN on Cora (LCC)**

1. **Model**: SL-GCN with complex-valued encoder (single-layer MLP), real-valued decoder (single-layer MLP, *uncertain*), real-valued readout (modulus of final node embeddings, *uncertain*). No polar-cartesian conversion in readout.

2. **Hidden dimensions**: encoder/decoder hidden dim = 64 (*uncertain*), output dim = 7 (classes).

3. **Propagation depth**: fixed depth from {2, 4, 8, 16, 32, 64, 128} layers; select best on validation (*uncertain* which depth wins for Cora).

4. **Newton solver**: max 5 iterations, tolerance 1e-3 (*uncertain*), damping factor 0.5 (*uncertain*).

5. **Training**: Adam, lr=0.01 (*uncertain*), weight_decay=5e-4 (*uncertain*), max 500 epochs, early stopping patience=50 on val accuracy.

6. **Loss**: CrossEntropy on train_idx only.

7. **Seed**: set `numpy.random.seed(SEED)`, `torch.manual_seed(SEED)` from env var.

8. **Split**: use provided `train_idx`/`val_idx`/`test_idx` from `/data/cora_lcc_public.npz` (20/30/rest per class).

9. **Selection**: pick depth + hyperparams maximizing val accuracy; retrain on train+val? (*uncertain* — paper says "best-performing configuration on validation set" then test).

10. **Output**: predict on `test_idx`, write `/output/predictions.csv` with header `id,prediction`.