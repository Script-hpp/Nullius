**Reproduction Plan for SGC on Cora (Planetoid Split)**

1. **Model**: SGC – fixed feature extraction `S^K X` (no learnable params), then linear classifier.  
   - `S = D^{-1/2} A D^{-1/2}` (symmetric normalization) **(uncertain: could be row-normalized)**  
   - `K` propagation steps **(uncertain: paper doesn’t specify; typical default K=2)**  

2. **Classifier**: Single linear layer (logistic regression) on propagated features; 7 output classes.  

3. **Optimizer**: Adam, `lr=0.2` (fixed), `weight_decay` tuned via hyperopt (60 trials on `val_idx`) **(uncertain: final value)**.  

4. **Training**: Transductive; max 100 epochs; select best epoch by `val_idx` accuracy (early stopping patience **(uncertain)**).  

5. **Seed**: Read `SEED` env var; set `numpy.random.seed(SEED)`, `torch.manual_seed(SEED)`.  

6. **Data**: Load `/data/cora_public.npz`; use provided `train_idx` (140), `val_idx` (500), `test_idx` (1000). Features are binary bag-of-words (no further normalization).  

7. **Output**: Predict on `test_idx`; write `/output/predictions.csv` with header `id,prediction` (node index, class 0–6).  

8. **Evaluation**: Single run (paper averages 10 runs **(uncertain: we run once with given seed)**).