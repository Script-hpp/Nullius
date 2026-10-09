
1. Model: GAT with 2 layers; Layer 1 – 8 attention heads, each outputting 8 features (concatenated → 64), ELU activation; Layer 2 – 1 head, outputting nb_classes = 7 logits, linear activation, averaged over heads.  
2. Hidden units per head: **hid_units** = [8]; output units = nb_classes (derived from labels).  
3. Number of heads per layer: **n_heads** = [8, 1].  
4. Dropout: feature dropout (**ffd_drop**) = 0.6, attention‑coefficient dropout (**attn_drop**) = 0.6 (applied to inputs and attention coefficients of both layers).  
5. Activation: hidden = tf.nn.elu; output = linear.  
6. Residual connections: **residual** = False.  
7. Learning rate: **lr** = 0.005; weight decay (L2): **l2_coef** = 0.0005.  
8. Optimizer: Adam (as used in BaseGAttN.training) with the above lr and weight_decay.  
9. Batch size: **batch_size** = 1 (full‑graph per batch); max epochs: **nb_epochs** = 100000.  
10. Early stopping: monitor validation loss; stop if no improvement for **patience** = 100 consecutive epochs.  
11. Preprocessing: features row‑normalized (process.preprocess_features); adjacency matrix augmented with self‑loops and converted to bias matrix via adj_to_bias (nhood=1) **(uncertain exact form)**.  
12. Seed: read **SEED** from environment; set numpy.random.seed, torch.manual_seed, tf.random.set_seed.  
13. Training: iterate over batches of size 1 (i.e., one step per epoch) applying dropout placeholders attn_drop=0.6, ffd_drop=0.6 when is_train=True.  
14. After training, restore checkpoint with lowest validation loss; compute logits for all nodes; predict class = argmax(logits) for nodes in test_idx.  
15. Output: write `/output/predictions.csv` with header `id,prediction`; one row per test node in test_idx order.