

**Reproduction plan (GCN on Cora‑LCC)**  

1. **Data** – Load `/data/cora_lcc_public.npz`; use the given `train_idx` (140), `val_idx` (210), `test_idx` (2135).  
2. **Pre‑processing** –  
   - Add self‑loops, compute symmetric normalized adjacency Â = D̃⁻⁰·⁵ Ã D̃⁻⁰·⁵ (Ã = A + I).  
   - If `normalize_features` is false (as in the config) keep raw binary BoW features; otherwise row‑normalize.  
   - Convert feature matrix and adjacency to sparse tensors (as in `_preprocess_features` / `_preprocess_adj`).  
3. **Model** – 2‑layer GCN:  
   - Layer 0: input → hidden (64) with ReLU, dropout = 0.8, **L2 weight‑decay = 0.001 applied only to this layer** (as in the original `graph_convolution_layer`).  
   - Layer 1: hidden → output (7 classes), linear, dropout = 0.8, no weight‑decay.  
   - Weight initialisation: Glorot uniform; biases zero.  
   - All operations follow the `graph_convolution` definition Â·X·W.  
4. **Optimizer** – Adam with learning rate **lr = 0.01** (placeholder fed each step).  
5. **Training protocol** –  
   - Max epochs = 100 000; early stopping using the **gnnbench criterion** (stop when validation loss does not improve for `patience = 50` consecutive epochs, then restore best weights).  
   - Validation evaluated each epoch; training loss logged for monitoring.  
   - No learning‑rate decay (`lr_decay_factor = 0.0`).  
   - Seed numpy/TensorFlow with the integer from env var `SEED`.  
6. **Inference** – After training, run the model on `test_feed` (dropout = 0) to obtain logits; predict `argmax` per node.  
7. **Output** – Write `/output/predictions.csv` with header `id,prediction`; `id` equals the original node index from `test_idx`.  

*(Uncertain items: exact optimizer hyper‑parameters beyond lr, whether weight‑decay should also apply to the second layer, and whether TensorFlow 1.x compatibility mode is required.)*