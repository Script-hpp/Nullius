

**Reproduction plan for SparseTSF (linear variant)**  

1. **Model** – `SparseTSF` with `model_type='linear'` (as implied by the few‑parameter goal).  
2. **Architecture** –  
   - `seq_len = 720` (input window)  
   - `pred_len = 96` (forecast horizon)  
   - `enc_in = 7` (channels)  
   - `period_len = 24` (down‑sampling block) → `seg_num_x = seq_len // period_len = 30`, `seg_num_y = pred_len // period_len = 4`  
   - `d_model` **(uncertain)** – hidden size for the optional MLP branch (not used in linear mode).  
   - One depth‑wise `Conv1d` with kernel `1 + 2*(period_len//2)`, padding `period_len//2`, bias = False.  
   - Linear projection `nn.Linear(seg_num_x, seg_num_y, bias=False)`.  
   - Instance normalisation per batch: subtract temporal mean, add back after projection.  

3. **Hyper‑parameters** –  
   - Learning rate `lr = 0.02` (initial).  
   - Optimiser: Adam (`betas=(0.9,0.999)`).  
   - Weight decay **(uncertain)** – not mentioned; set to 0 if needed.  
   - Dropout **(uncertain)** – not used in the model; keep 0.  
   - Batch size `= 256`.  
   - Epochs `= 30` (max) with early stopping patience `= 5` on validation MSE.  
   - Loss: MSE.  
   - Learning‑rate schedule **(uncertain)** – paper stresses a large constant LR; we will keep LR constant unless validation suggests otherwise.  

4. **Training protocol** –  
   - Load the provided `train` (`[8640,7]`) and `val` (`[2880,7]`) arrays.  
   - Create sliding windows of length `seq_len` → targets of length `pred_len`.  
   - Shuffle only within each epoch; no cross‑window leakage.  
   - After each epoch compute validation MSE; stop if no improvement for 5 consecutive epochs.  
   - Retain the best model (lowest validation MSE).  

5. **Inference** –  
   - For each test window `i` (`0 … 2784`) use the supplied `test_inputs[i]` (`[720,7]`).  
   - Optionally truncate to the last `L ≤ 720` steps; we will use the full 720.  
   - Produce forecasts of shape `[96,7]`; store as `"{i}_{step}_{channel}"`.  

All uncertain items are marked; if the original paper/code specifies them, replace the marks accordingly. This plan fits within the 5‑minute CPU budget.