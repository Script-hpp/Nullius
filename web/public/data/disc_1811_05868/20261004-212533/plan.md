**Reproduction Plan: GCN on Cora (LCC)**

1. **Model**: 2-layer GCN (Kipf & Welling), hidden size 64 (uncertain: optimized config=64, reference=16)
2. **Hyperparameters**: lr=0.01, weight_decay=0.001 (uncertain: only in optimized config), dropout=0.8 (uncertain: optimized=0.8, reference=0.5)
3. **Input**: Features [2485,1433] binary → float32; adjacency from edges (undirected, no self-loops, float32)
4. **Labels**: One-hot (7 classes), train_idx (140), val_idx (210), test_idx (2135) from npz
5. **Training**: Full-batch, Adam, max 200 epochs (uncertain: unified protocol uses 100k)
6. **Early Stopping**: Stop if val_loss > mean(val_loss[-10:]) (GCN-specific; uncertain: unified uses 50-epoch patience)
7. **Model Selection**: Restore weights from best validation loss epoch
8. **Seeding**: Seed numpy & torch from env `SEED` before weight init
9. **Output**: Predict on test_idx → `/output/predictions.csv` (id,prediction)