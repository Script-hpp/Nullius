

**Reproduction plan for TSMixer on ETTh1 (horizon 96)**  

1. **Model**: TSMixer (6 residual blocks) with reversible instance normalization (RevIN) applied per‑sample before the model and inverted after.  
2. **Input shape**: `(seq_len, C) = (512, 7)` (last L = 512 hours of each test window).  
3. **Block hyper‑parameters**:  
   - `n_block = 6` (certain)  
   - `ff_dim = hidden = 512` (certain)  
   - `dropout = 0.9` (certain)  
   - `activation = relu` (certain)  
   - `norm_type = BatchNorm` **(uncertain)** – paper only specifies RevIN; internal norm defaults to BatchNorm in the code.  
4. **Output head**: Dense layer projecting to `pred_len = 96` timesteps, producing shape `(batch, 96, 7)`.  
5. **Loss**: Mean‑squared error (MSE) on standardized values.  
6. **Optimizer**: Adam **(uncertain)** – not explicitly stated; typical default.  
7. **Learning rate**: `0.0001` (certain).  
8. **Batch size**: `32` (certain, argparse default).  
9. **Epochs**: maximum `100` with early stopping if validation loss does not improve for `patience = 5` epochs (certain).  
10. **Weight decay**: `0` **(uncertain)** – not mentioned in paper or code.  
11. **Seeding**: read `SEED` from environment, set `numpy`, `torch`, and `tensorflow` random seeds.  
12. **Training procedure**: standard mini‑batch shuffling, validation on the provided `val` split, checkpoint best model, then predict on all 2785 test windows using only their own history.  
13. **Inference**: for each window, feed the last L = 512 steps (or fewer if L < 720, but we use the full 512) through the trained model with RevIN, collect the 96‑step forecasts for all 7 channels.  
14. **Output format**: CSV `/output/predictions.csv` with rows `id="{window}_{step}_{channel}"` and prediction in standardized units (total 2785 × 96 × 7 = 1 871 520 rows).  

All values marked **(uncertain)** reflect specifications not explicitly given in the paper or the supplied code excerpts.