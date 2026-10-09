

1. Model: GAT, 2 layers, hidden size 8, first layer 8 heads, second layer 1 head → 7‑class logits.  
2. Features: raw bag‑of‑words (float32, no row‑norm); adjacency made undirected, self‑loops added (as in `add_self_loops`).  
3. Dropout: input_dropout_prob = 0.6 **(uncertain)**, coefficient_dropout_prob = 0.3 **(uncertain)**.  
4. Optimizer: Adam (TF default) with learning_rate = 0.01, weight_decay = 0.01 (L2).  
5. Epochs: max 100000; early stopping: GAT criterion, patience = 100 epochs (no improvement in val loss **or** val accuracy).  
6. Validation: 210 nodes (30 per class); training: 140 nodes (20 per class).  
7. Test: remaining 2135 nodes (labels hidden); predict class for each test_idx node.  
8. Seed: read env `SEED`, set `np.random.seed` and `tf.random.set_seed`.  
9. Batch size: full‑batch (no mini‑batches) **(uncertain)**.  
10. Learning‑rate decay: disabled (lr_decay_factor = 0.0) **(uncertain)**.  
11. Weight initialization: Glorot uniform for all TF variables (as in the code).  
12. Early stopping restores weights to the best validation state (per `GATCriterion`).  
13. Output: write `/output/predictions.csv` with header `id,prediction`, one row per test node.  
14. Must finish within 5 minutes on CPU.  
15. All other settings follow the provided configs/code; any unspecified item follows standard TensorFlow defaults.