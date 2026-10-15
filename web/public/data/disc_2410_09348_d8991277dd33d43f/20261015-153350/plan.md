

**Reproduction plan for BANGS (Cora, standard split)**  

1. **Model** – 2‑layer GCN:  
   - Input dim = 1433, hidden = 128 (ReLU), output dim = 7.  
   - Dropout = 0.3 on the hidden layer.  
   - Weight decay = 5e‑4, learning rate = 0.001, optimizer = Adam.  

2. **Adjacency preprocessing** – add self‑loop, then symmetric normalisation:  
   \(\tilde A = D^{-1/2}(A+I)D^{-1/2}\).  

3. **Influence matrix (used for IGP selection)** – Personalized PageRank on \(\tilde A\) with teleport probability \(\alpha = 0.85\) (default in `get_ppr_influence_matrix`).  
   *(Uncertain: exact α value; the paper only states \(\alpha\in(0,1]\).)*  

4. **Self‑training loop** – maximum outer iterations = 40 (as reported in the paper).  
   - Inner training: train the GCN for up to 2000 epochs with early stopping (patience = 500 epochs) on the validation set; keep the model with best validation accuracy.  
   - After each inner training, compute softmax logits → confidence, mask out already‑labelled nodes.  

5. **Node selection (IGP‑BANGS)** –  
   - Sample \(N = \text{sample\_num}\times\text{SAMPLES\_RANGE}/\text{top}\) times (code uses `sample_num=100`, `top=100`, `SAMPLES_RANGE = 2*top` if `top<100` else `top+100`).  
   - For each sampled subset compute the Banzhaf/shapley value via the entropy‑utility function (see `IGP_sample_banzhaf`).  
   - Take the top `top` (=100) nodes with highest IGP score as pseudo‑labels.  
   *(Uncertain: exact values of `sample_num` and the derived `SAMPLES_RANGE`; the paper does not specify them.)*  

6. **Pseudo‑labeling** – assign the predicted class (argmax of softmax) to the selected nodes and add them to the training set for the next iteration.  

7. **Retraining** – after each expansion, retrain the GCN from scratch (or fine‑tune if `--FT` were set; we follow the default `--FT=False`). Loss is the weighted cross‑entropy with bald = 1 and transition matrix T = I (i.e. ordinary cross‑entropy).  

8. **Early stopping for outer loop** – keep the iteration that yields the highest validation accuracy; report its test accuracy.  

9. **Reproducibility** – read integer seed from environment variable `SEED`; set `numpy.random.seed(seed)`, `torch.manual_seed(seed)`, and (if GPU were available) `torch.cuda.manual_seed(seed)`.  

10. **Output** – after training, predict the class for every node in `test_idx` and write `/output/predictions.csv` with header `id,prediction`.  

*All values not explicitly fixed by the paper or code are marked uncertain above.*