

**Reproduction plan for NLinear (as reported in “Are Transformers Effective for Time Series Forecasting?”)**  

1. **Model** – NLinear: a single linear layer that maps the last L time steps of each variate to the next H steps (no hidden layers, weight‑shared across channels or separate per channel – both variants give similar results; we use separate linear layers per channel *(uncertain)*).  
2. **Input / output lengths** – L = 336 (history), H = 96 (forecast horizon).  
3. **Data preparation** – From the training tensor `[8640,7]` create sliding windows:  
   - `x = train[t:t+L]` → shape `[L,7]`  
   - `y = train[t+L:t+L+H]` → shape `[H,7]`  
   for `t = 0 … len(train)-L-H`. Shuffle and batch.  
4. **Optimizer** – Adam with learning rate = 0.001 *(uncertain)*, β₁=0.9, β₂=0.999, weight decay = 0.  
5. **Loss** – Mean‑squared error (MSE) averaged over all horizons and channels.  
6. **Training schedule** – Maximum 20 epochs; early‑stop if validation MSE does not improve for 5 consecutive epochs *(uncertain)*. Batch size = 256 *(uncertain)*.  
7. **Validation** – Built from the first 20 % of the training set (or use the provided `[2880,7]` val split) to compute MSE after each epoch.  
8. **Seeding** – Read integer `SEED` from environment; call `numpy.random.seed(SEED)` and `torch.manual_seed(SEED)`.  
9. **Inference** – For each test window `i` (0…2784): take the last L steps of `test_inputs[i]` (`test_inputs[i][-L:]`) → model forward → output `[H,7]` predictions.  
10. **Output** – Write `/output/predictions.csv` with header `id,prediction`; id format `"{window}_{step}_{channel}"` (window 0‑2784, step 0‑95, channel 0‑6).  
11. **Runtime** – CPU‑only; with the above settings the whole pipeline finishes well under 5 minutes.  

All values marked *(uncertain)* are those not explicitly stated in the paper and are chosen based on typical practice for NLinear on ETTh1.