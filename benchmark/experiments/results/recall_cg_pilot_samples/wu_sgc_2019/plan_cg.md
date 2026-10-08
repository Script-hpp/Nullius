**Reproduction plan for SGC on Cora (Planetoid split)**  

1. **Model** – SGC: a single linear layer `W ∈ ℝ^{F×C}` (F=1433, C=7) applied to k‑step propagated features; bias term included (default of `nn.Linear`) *(uncertain)*.  
2. **Feature preprocessing** – row‑normalize the raw bag‑of‑words matrix (`float32`).  
3. **Adjacency preprocessing** – compute symmetrically normalized adjacency with self‑loops (AugNormAdj) *(uncertain exact formula)*.  
4. **Graph propagation** – obtain `X̂ = (Ã)^k X` with `k = degree = 2` *(uncertain if paper used k=2)*; implemented via `sgc_precompute`.  
5. **Optimizer** – Adam, learning rate `lr = 0.2`, weight decay `= 5e‑6`.  
6. **Training** – run for `epochs = 100` on the full batch (all training nodes); no dropout (`dropout = 0`).  
7. **Validation** – after each epoch compute accuracy on `val_idx`; keep the model with highest validation accuracy (early stopping not used in the paper) *(uncertain)*.  
8. **Testing** – evaluate the selected model on `test_idx`; output class predictions.  
9. **Randomness** – read integer seed from environment variable `SEED`; set `numpy` and `torch` seeds accordingly.  
10. **Device** – CPU only (as required).  
11. **Additional notes** – No hidden layers (`hidden = 0`), no learning‑rate schedule, no patience or heads (not applicable).  

All values marked *(uncertain)* are those not explicitly stated in the paper, code excerpt, or argparse defaults.